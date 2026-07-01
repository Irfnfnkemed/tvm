from tvm.script import tir as T
from tvm.script import tirx as Tx

from tvm.tir.layout import TileLayout, tid_in_wg as axis_tid_in_wg
from tvm.tirx.bench.utils import CudaProfiler

from .gemm import GemmTile
from tvm.tirx.megakernel.utils.base import SmemManager
from tvm.tirx.megakernel.utils.utils import silu
from tvm.tirx.megakernel.utils.config import KernelConfig, ProfileEventType


class GateUpSiluTile(GemmTile):
    # Notes: Fused Gate/Up Weight Matrix (B), shape: [intermediate_size * 2, hidden_size].
    #        Gate, Up weights shape: [intermediate_size, hidden_size].
    #        B is formed by row-wise interleaving of 16-row blocks from g and u.
    #        B = [g[0:16, :]; u[0:16, :]; g[16:32, :]; u[16:32, :]; ...].

    need_init = False
    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.A_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.MAX_BLK_M, self.BLK_K),
            self.a_type,
            layout=self.A_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            name="A_smem",
            method="exclusive",
        )
        self.B_smem = smem_manager.alloc(
            (self.SMEM_PIPE_DEPTH, self.BLK_N, self.BLK_K),
            self.b_type,
            layout=self.B_layout,
            align=1024,
            split=self.SMEM_PIPE_DEPTH,
            name="B_smem",
            method="exclusive",
        )
        self.D_layout = T.TileLayout(
            shard=(
                (GemmTile.TMEM_PIPE_DEPTH, GemmTile.EPI_TILE, GemmTile.MMA_N // 2),
                (GemmTile.EPI_TILE * GemmTile.MMA_N // 2, GemmTile.MMA_N // 2, 1)
            )
        )
        self.output_smem = smem_manager.alloc(
            (self.TMEM_PIPE_DEPTH, self.EPI_TILE, self.MMA_N // 2),
            "float16",
            layout=self.D_layout,
            align=1024,
            name="output_smem",
            method="exclusive",
        )

    def _alloc_local(self, m_idx):
        # alloc local memory
        self.reg = T.alloc_buffer((self.TMEM_LD_SIZE,), "float32", scope="local", name="reg")
        self.reg_fp16 = T.alloc_buffer((self.TMEM_LD_SIZE // 2,), "float16", scope="local", name="reg_fp16")
        self.tmem_idx = T.local_cell("int32", name="tmem_idx")
        self.tmem_phase = T.local_cell("int32", name="tmem_phase")
        self.stage = T.local_cell("int32", name="stage")
        self.wait_complete = T.local_cell("bool", name="wait_complete")
        self.off = T.local_cell("int32", name="off")

    @T.macro
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @T.macro
    def _consumer_wg(self, m_idx, n_idx, k_idx, A, B, output, profiler: CudaProfiler):
        with T.thread():
            tid_in_wg = T.thread_id([128], parent="warpgroup")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER], parent="warpgroup")
            lane_id = T.thread_id([32], parent="warp")
            T.cuda.trap_when_assert_failed(self.tmem_addr[0] == 0)
            if warp_id == 0:
                self.smem_manager.wait_specific(lane_id, self.output_smem, 0)
            T.ptx.bar.sync(10, 128)
            self.phase[0] = 0
            self.tmem_idx = self.tile_idx % self.TMEM_PIPE_DEPTH
            self.tmem_phase = (self.tile_idx // self.TMEM_PIPE_DEPTH) & 1

            # flush previous tma
            # wait for the completion of all the mma of the same tile
            self.mma2ld_bar.wait(self.tmem_idx, self.tmem_phase)
            T.ptx.tcgen05.fence.after_thread_sync()

            SILU_HANDLE_UNIT = T.meta_var(self.TMEM_LD_SIZE // 2)
            self.off = T.if_then_else(lane_id < 16, SILU_HANDLE_UNIT, 0)
            for ko in T.unroll(self.MMA_M // self.EPI_TILE):
                self.stage = (self.tile_idx * self.MMA_M // self.EPI_TILE + ko) % self.TMEM_PIPE_DEPTH
                # wait the smem to be free
                if ko >= self.TMEM_PIPE_DEPTH:
                    if lane_id == 0 and warp_id == 0:
                        T.ptx.cp_async.bulk.wait_group(self.TMEM_PIPE_DEPTH - 1)
                    T.cuda.warpgroup_sync(10)

                # tmem -> rf (ld) -> smem
                for ki in T.unroll(self.EPI_TILE // self.TMEM_LD_SIZE):
                    with T.warpgroup():
                        reg_wg = self.reg.view(128, self.TMEM_LD_SIZE, layout=TileLayout(([128, self.TMEM_LD_SIZE], [1@axis_tid_in_wg, 1])))
                        col_st = T.meta_var(self.tmem_idx * self.M_pad_size + ko * self.EPI_TILE + ki * self.TMEM_LD_SIZE)
                        Tx.copy(reg_wg[:, :], self.tmem[:, col_st : col_st + self.TMEM_LD_SIZE])
                    if self.profiler_on:
                        profiler.start(ProfileEventType.SILU_MUL, lane_id == 0)
                    # for each warp, lane 0~15 holds the gate output, lane 16~31 holds the up output
                    for kv in T.unroll(SILU_HANDLE_UNIT):
                        self.reg[self.off + kv] = T.tvm_warp_shuffle_xor(0xffffffff, self.reg[self.off + kv], 16, 32, 32)
                    for kv in T.unroll(SILU_HANDLE_UNIT):
                        self.reg[kv] = silu(self.reg[kv]) * self.reg[SILU_HANDLE_UNIT + kv]
                    Tx.cast(self.reg_fp16[:], self.reg[0 : SILU_HANDLE_UNIT], vec=SILU_HANDLE_UNIT)
                    st = T.meta_var(ki * self.TMEM_LD_SIZE + (lane_id // 16) * SILU_HANDLE_UNIT)
                    Tx.copy(self.output_smem[self.stage, st : st + SILU_HANDLE_UNIT, warp_id * 16 + lane_id % 16], self.reg_fp16[:], vec=SILU_HANDLE_UNIT)
                    if self.profiler_on:
                        profiler.end(ProfileEventType.SILU_MUL, lane_id == 0)

                # the tmem can be overwritten
                if ko == self.MMA_M // self.EPI_TILE - 1:
                    T.ptx.tcgen05.fence.before_thread_sync()
                    self.ld2mma_bar.arrive(self.tmem_idx)

                T.ptx.fence.proxy(scope="shared")
                T.cuda.warpgroup_sync(10)
                # smem -> gmem
                with T.thread(parent="warpgroup")[tid_in_wg == 0]:
                    m_st = T.meta_var(m_idx * self.M_pad_size + ko * self.EPI_TILE)
                    n_st = T.meta_var(n_idx * self.BLK_N // 2)
                    tma_config = T.meta_var({"dispatch": "tma", "cta_group": KernelConfig.CTA_GROUP})
                    Tx.copy_async(output[m_st : m_st + self.EPI_TILE, n_st : n_st + self.BLK_N // 2],
                                    self.output_smem[self.stage, :, :], **tma_config)
                    T.ptx.cp_async.bulk.commit_group()
            if tid_in_wg == 0:
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)
            self.tile_idx += 1
            if warp_id == 0:
                self.smem_manager.arrive_specific(lane_id, self.output_smem, 0)