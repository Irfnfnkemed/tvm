from tvm.script import tir as T
from tvm.script import tirx as Tx 

from tvm.tirx.megakernel.utils.base import Tile, SmemManager
from tvm.tirx.megakernel.utils.utils import ceildiv, find_power_of_two, rsqrt
from tvm.tirx.megakernel.utils.config import KernelConfig, F16_BYTES, F32_BYTES


class AddRMSNormTile(Tile):
    vec_size = 16 // F16_BYTES
    loop_inner = 1
    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx

    # inplace add rms norm
    def __init__(self, rms_norm_eps, hidden_size):
        super().__init__()
        self.EPS = rms_norm_eps
        self.hidden_size = hidden_size
    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.x_smem = smem_manager.alloc([self.loop_inner * self.hidden_size], name="x_smem", dtype="float32")
        self.sum_sq_smem = smem_manager.alloc([self.loop_inner, self.bdy], name="sum_sq_smem", dtype="float32")

    def _alloc_local(self):
        # alloc local memory
        self.input_vec = T.alloc_local(
            [self.loop_inner, self.vec_size], "float16", name="input_vec"
        )
        self.residual_vec = T.alloc_local(
            [self.loop_inner, self.vec_size], "float16", name="residual_vec"
        )
        self.weight_vec = T.alloc_local(
            [self.loop_inner, self.vec_size], "float16", name="weight_vec"
        )
        self.input_vec_f32 = T.alloc_local(
            [self.loop_inner, self.vec_size], "float32", name="input_vec_f32"
        )
        self.residual_vec_f32 = T.alloc_local(
            [self.loop_inner, self.vec_size], "float32", name="residual_vec_f32"
        )
        self.weight_vec_f32 = T.alloc_local(
            [self.loop_inner, self.vec_size], "float32", name="weight_vec_f32"
        )
        self.x_vec = T.alloc_local([self.loop_inner, self.vec_size], "float32", name="x_vec")
        self.x_tmp = T.alloc_local([self.loop_inner, 1], "float32", name="x_tmp")
        self.sum_sq = T.alloc_local([self.loop_inner, 1], "float32", name="sum_sq")
        self.rms_norm = T.alloc_local([1], "float32", name="rms_norm")

    @T.macro
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    # fmt: off
    @T.macro
    def run(self, m_idx, n_idx, k_idx, input, residual, weight, output=None, out_residual=None):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            warp_id_in_cta = T.warp_id([self.bdy], parent="cta")
            lane_id = T.thread_id([self.bdx], parent="warp")
            output_buf = T.meta_var(input if output is None else output)
            out_residual_buf = T.meta_var(residual if out_residual is None else out_residual)
            # add & sum square
            self._alloc_local()
            with T.thread():
                self.smem_manager.wait_all("cta")
                for kl in T.unroll(self.loop_inner):
                    self.sum_sq[kl, 0] = 0.0
                self.rms_norm[0] = 0.0
                for ki in T.serial(ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS * self.loop_inner)):
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec[kl, kv] = 0.0
                            self.residual_vec[kl, kv] = 0.0
                            self.x_vec[kl, kv] = 0.0

                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size * self.loop_inner)

                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(self.input_vec[kl, :], input[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                            Tx.copy(self.residual_vec[kl, :], residual[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])

                    for kl in T.unroll(self.loop_inner):
                        Tx.cast(self.input_vec_f32[kl, :], self.input_vec[kl, :])
                        Tx.cast(self.residual_vec_f32[kl, :], self.residual_vec[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.x_tmp[kl, 0] = self.input_vec_f32[kl, kv] + self.residual_vec_f32[kl, kv]
                            self.sum_sq[kl, 0] += self.x_tmp[kl, 0] * self.x_tmp[kl, 0]
                            self.residual_vec[kl, kv] = T.cast(self.x_tmp[kl, 0], "float16")
                            self.x_vec[kl, kv] = self.x_tmp[kl, 0]
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(out_residual_buf[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.residual_vec[kl, :])
                            Tx.copy(self.x_smem[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.x_vec[kl, :])

                # warp reduce sum
                for kl in T.unroll(self.loop_inner):
                    for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                        self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + T.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                        )
                for kl in T.unroll(self.loop_inner):
                    self.sum_sq_smem[kl, warp_id_in_cta] = self.sum_sq[kl, 0]
                T.tvm_storage_sync("shared")

                # reduce sum through different warps
                if warp_id_in_cta == 0:
                    for kl in T.unroll(self.loop_inner):
                        if lane_id < self.bdy:
                            self.sum_sq[kl, 0] = self.sum_sq_smem[kl, lane_id]
                        else:
                            self.sum_sq[kl, 0] = 0.0
                    for kl in T.unroll(self.loop_inner):
                        for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + T.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                            )
                    for kl in T.unroll(self.loop_inner):
                        self.sum_sq_smem[kl, 0] = self.sum_sq[kl, 0]
                T.tvm_storage_sync("shared")

                # rms norm
                for kl in T.unroll(self.loop_inner):
                    self.rms_norm[0] += self.sum_sq_smem[kl, 0]
                self.rms_norm[0] = rsqrt(self.rms_norm[0] / self.hidden_size + self.EPS)

                # handle the weight
                for ki in T.serial(ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS)):
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec[kl, kv] = 0.0
                            self.weight_vec_f32[kl, kv] = 0.0
                            self.x_vec[kl, kv] = 0.0
                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size * self.loop_inner)
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(self.weight_vec[kl, :], weight[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                            Tx.copy(self.x_vec[kl, :], self.x_smem[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                    for kl in T.unroll(self.loop_inner):
                        Tx.cast(self.weight_vec_f32[kl, :], self.weight_vec[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec_f32[kl, kv] = self.x_vec[kl, kv] * self.rms_norm[0] * self.weight_vec_f32[kl, kv]
                    for kl in T.unroll(self.loop_inner):
                        Tx.cast(self.input_vec[kl, :], self.input_vec_f32[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(output_buf[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.input_vec[kl, :])

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on

class RMSNormTile(Tile):
    vec_size = 16 // F16_BYTES
    loop_inner = 1
    bdx = 32
    bdy = KernelConfig.NUM_THREADS // bdx

    # inplace add rms norm
    def __init__(self, rms_norm_eps, hidden_size):
        super().__init__()
        self.EPS = rms_norm_eps
        self.hidden_size = hidden_size
    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.x_smem = smem_manager.alloc([self.loop_inner * self.hidden_size], "float32").buffer
        self.sum_sq_smem = smem_manager.alloc([self.loop_inner, self.bdy], "float32").buffer

    def _alloc_local(self):
        # alloc local memory
        self.input_vec = T.alloc_local(
            [self.loop_inner, self.vec_size], "float16", name="input_vec"
        )
        self.weight_vec = T.alloc_local(
            [self.loop_inner, self.vec_size], "float16", name="weight_vec"
        )
        self.input_vec_f32 = T.alloc_local(
            [self.loop_inner, self.vec_size], "float32", name="input_vec_f32"
        )
        self.weight_vec_f32 = T.alloc_local(
            [self.loop_inner, self.vec_size], "float32", name="weight_vec_f32"
        )
        self.sum_sq = T.alloc_local([self.loop_inner, 1], "float32", name="sum_sq")
        self.rms_norm = T.alloc_local([1], "float32", name="rms_norm")

    @T.macro
    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    # fmt: off
    @T.macro
    def run(self, m_idx, n_idx, k_idx, output, input, weight):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            warp_id_in_cta = T.warp_id([self.bdy], parent="cta")
            lane_id = T.thread_id([self.bdx], parent="warp")
            # add & sum square
            self._alloc_local()
            with T.thread():
                self.smem_manager.wait_all("cta")
                for kl in T.unroll(self.loop_inner):
                    self.sum_sq[kl, 0] = 0.0
                self.rms_norm[0] = 0.0
                for ki in T.serial(ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS * self.loop_inner)):
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec_f32[kl, kv] = 0.0
                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size * self.loop_inner)
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(self.input_vec_f32[kl, :], input[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.sum_sq[kl, 0] += self.input_vec_f32[kl, kv] * self.input_vec_f32[kl, kv]
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(self.x_smem[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.input_vec_f32[kl, :])

                # warp reduce sum
                for kl in T.unroll(self.loop_inner):
                    for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                        self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + T.tvm_warp_shuffle_xor(
                            0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                        )
                for kl in T.unroll(self.loop_inner):
                    self.sum_sq_smem[kl, warp_id_in_cta] = self.sum_sq[kl, 0]
                T.tvm_storage_sync("shared")

                # reduce sum through different warps
                if warp_id_in_cta == 0:
                    for kl in T.unroll(self.loop_inner):
                        if lane_id < self.bdy:
                            self.sum_sq[kl, 0] = self.sum_sq_smem[kl, lane_id]
                        else:
                            self.sum_sq[kl, 0] = 0.0
                    for kl in T.unroll(self.loop_inner):
                        for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[kl, 0] = self.sum_sq[kl, 0] + T.tvm_warp_shuffle_xor(
                                0xFFFFFFFF, self.sum_sq[kl, 0], (self.bdx // 2) >> kr, 32, 32
                            )
                    for kl in T.unroll(self.loop_inner):
                        self.sum_sq_smem[kl, 0] = self.sum_sq[kl, 0]
                T.tvm_storage_sync("shared")

                # rms norm
                for kl in T.unroll(self.loop_inner):
                    self.rms_norm[0] += self.sum_sq_smem[kl, 0]
                self.rms_norm[0] = rsqrt(self.rms_norm[0] / self.hidden_size + self.EPS)

                # handle the weight
                for ki in T.serial(ceildiv(self.hidden_size, self.vec_size * KernelConfig.NUM_THREADS)):
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec_f32[kl, kv] = 0.0

                    st = T.meta_var((ki * KernelConfig.NUM_THREADS + tid) * self.vec_size * self.loop_inner)
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(self.weight_vec[kl, :], weight[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                            Tx.copy(self.input_vec_f32[kl, :], self.x_smem[st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size])
                    for kl in T.unroll(self.loop_inner):
                        Tx.cast(self.weight_vec_f32[kl, :], self.weight_vec[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        for kv in T.unroll(self.vec_size):
                            self.input_vec_f32[kl, kv] = self.input_vec_f32[kl, kv] * self.rms_norm[0] * self.weight_vec_f32[kl, kv]
                    for kl in T.unroll(self.loop_inner):
                        Tx.cast(self.input_vec[kl, :], self.input_vec_f32[kl, :])
                    for kl in T.unroll(self.loop_inner):
                        if st < self.hidden_size - kl * self.vec_size:
                            Tx.copy(output[m_idx, st + kl * self.vec_size : st + kl * self.vec_size + self.vec_size], self.input_vec[kl, :])

                self.smem_manager.arrive_all("cta")
                self.smem_manager.advance()
    # fmt: on
