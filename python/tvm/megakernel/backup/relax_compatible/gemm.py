from tvm.script import tir as T, tirx as Tx
from tvm.tir.layout import TileLayout, TLane, TCol

from tvm.tirx.megakernel.utils.config import KernelConfig
from tvm.tirx.megakernel.utils.base import SmemManager
from tvm.tirx.megakernel.kernels.gemm import GemmTile, BarTMA2MMA, BarMMA2TMA, BarMMA2LD, BarLD2MMA
from tvm.tirx.megakernel.kernels import GateUpSiluTile

class FuseGemmTile(GemmTile):

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager):
        # alloc shared memory
        GemmTile.tmem_addr = smem_manager.alloc([1], "uint32", name="tmem_addr", method="persistent")
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        GemmTile.tile_idx = T.alloc_cell("int32", scope="local.persistent", name="tile_idx")
        GemmTile.phase = T.alloc_buffer((1,), "int32", scope="local.persistent", name="phase")
        GemmTile.tmem = T.decl_buffer((128, 512), "float32", scope="tmem.persistent", allocated_addr=0, layout=TileLayout(([128, 512], [1@TLane, 1@TCol])), name="tmem")


    @staticmethod
    def get_func(N, K, ab_dtype, out_dtype, BLK_M, prefetch_on=False, split_k_factor=1, use_tma_reduce=False, A_dim=2):
        symbolic_tile = FuseGemmTile(N, K, ab_dtype, ab_dtype, split_k_factor, BLK_M, BLK_M, prefetch_on=prefetch_on, use_tma_reduce=use_tma_reduce, profiler_on=False)
        num_tiles = (1, symbolic_tile.N // symbolic_tile.BLK_N, split_k_factor)
        tile_size = (None, symbolic_tile.BLK_N, symbolic_tile.TILE_K)
        
        @T.macro
        def gemm_body(A, B, output, m_idx, n_idx, k_idx):
            gemm_tile = T.meta_var(FuseGemmTile(N, K, ab_dtype, ab_dtype, split_k_factor, BLK_M, BLK_M, prefetch_on=prefetch_on, use_tma_reduce=use_tma_reduce, profiler_on=False))
            gemm_tile.host_init()
            with T.cta():
                buf = T.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(gemm_tile)
                gemm_tile.init(smem_manager)
                with T.cta():
                    T.block_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirx.tile_class.persistent.init": True})
                    gemm_tile.class_init(smem_manager)
                with T.cta():
                    T.block_attr({"tirx.tile_class.prefetch": True})
                    if prefetch_on:
                        gemm_tile.prefetch(m_idx, n_idx, k_idx, A, B, output, None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    gemm_tile.run(m_idx, n_idx, k_idx, A, B, output, None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.persistent.finalize": True})
                    gemm_tile.class_finalize()    

        if split_k_factor == 1:
            assert A_dim == 2
            @T.prim_func(tirx=True, private=True)
            def gemm_func_no_split_k(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
                T.func_attr({"megakernel.device_func": "gemm"})
                m = T.int32()
                A = T.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
                B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                C = T.match_buffer(C_ptr, (m, N), out_dtype, layout="default")
                gemm_body(A, B, C, m_idx, n_idx, k_idx)
            return gemm_func_no_split_k, num_tiles, tile_size

        else:
            if A_dim == 2:
                if use_tma_reduce:
                    @T.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(A_ptr: T.handle, B_ptr: T.handle, output_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32): 
                        T.func_attr({"megakernel.device_func": "gemm"})
                        m = T.int32()
                        A = T.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
                        B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        output = T.match_buffer(output_ptr, (m, N), out_dtype, layout="default")
                        gemm_body(A, B, output, m_idx, n_idx, k_idx)
                    return gemm_func_split_k, num_tiles, tile_size
                else:            
                    @T.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(A_ptr: T.handle, B_ptr: T.handle, partial_sum_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32): 
                        T.func_attr({"megakernel.device_func": "gemm"})
                        m = T.int32()
                        A = T.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
                        B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        partial_sum = T.match_buffer(partial_sum_ptr, (split_k_factor, m, N), out_dtype, layout="default")
                        gemm_body(A, B, partial_sum, m_idx, n_idx, k_idx)
                    return gemm_func_split_k, num_tiles, tile_size
            else:
                assert A_dim == 3
                if use_tma_reduce:
                    @T.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(A_ptr: T.handle, B_ptr: T.handle, output_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32): 
                        T.func_attr({"megakernel.device_func": "gemm"})
                        m = T.int32()
                        K1 = T.int32()
                        K2 = T.int32()
                        A = T.match_buffer(A_ptr, (m, K1, K2), ab_dtype, layout="default")
                        B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        output = T.match_buffer(output_ptr, (m, N), out_dtype, layout="default")
                        gemm_body(A.view(m, -1).buffer, B, output, m_idx, n_idx, k_idx)
                    return gemm_func_split_k, num_tiles, tile_size
                else:
                    @T.prim_func(tirx=True, private=True)
                    def gemm_func_split_k(A_ptr: T.handle, B_ptr: T.handle, partial_sum_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
                        T.func_attr({"megakernel.device_func": "gemm"})
                        m = T.int32()
                        K1 = T.int32()
                        K2 = T.int32()
                        A = T.match_buffer(A_ptr, (m, K1, K2), ab_dtype, layout="default")
                        B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
                        partial_sum = T.match_buffer(partial_sum_ptr, (split_k_factor, m, N), out_dtype, layout="default")
                        gemm_body(A.view(m, -1).buffer, B, partial_sum, m_idx, n_idx, k_idx)
                    return gemm_func_split_k, num_tiles, tile_size

class FuseGateUpSiluTile(GateUpSiluTile):

    @classmethod
    def _alloc_buffer_class_member(cls, smem_manager):
        # alloc shared memory
        GemmTile.tmem_addr = smem_manager.alloc([1], "uint32", name="tmem_addr", method="persistent")
        GemmTile.tma2mma_bar = BarTMA2MMA(smem_manager, cls.SMEM_PIPE_DEPTH, True)
        GemmTile.mma2tma_bar = BarMMA2TMA(smem_manager, cls.SMEM_PIPE_DEPTH, False)
        GemmTile.mma2ld_bar = BarMMA2LD(smem_manager, cls.TMEM_PIPE_DEPTH, True)
        GemmTile.ld2mma_bar = BarLD2MMA(smem_manager, cls.TMEM_PIPE_DEPTH, False)
        # alloc local memory
        GemmTile.tile_idx = T.alloc_cell("int32", scope="local.persistent", name="tile_idx")
        GemmTile.phase = T.alloc_buffer((1,), "int32", scope="local.persistent", name="phase")
        GemmTile.tmem = T.decl_buffer((128, 512), "float32", scope="tmem.persistent", allocated_addr=0, layout=TileLayout(([128, 512], [1@TLane, 1@TCol])), name="tmem")

    @staticmethod
    def get_func(N, K, ab_dtype, out_dtype, BLK_M, prefetch_on=False):
        symbolic_tile = FuseGateUpSiluTile(N, K, ab_dtype, ab_dtype, 1, BLK_M, BLK_M, prefetch_on=prefetch_on, profiler_on=False)
        num_tiles = (1, symbolic_tile.N // symbolic_tile.BLK_N, 1)
        tile_size = (None, symbolic_tile.BLK_N, symbolic_tile.TILE_K)

        @T.prim_func(tirx=True, private=True)
        def gate_up_silu_func(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "gemm"})
            m = T.int32()
            A = T.match_buffer(A_ptr, (m, K), ab_dtype, layout="default")
            B = T.match_buffer(B_ptr, (N, K), ab_dtype, layout="default")
            output = T.match_buffer(C_ptr, (m, N // 2), out_dtype, layout="default")
            gemm_tile = T.meta_var(FuseGateUpSiluTile(N, K, ab_dtype, ab_dtype, 1, BLK_M, BLK_M, prefetch_on=prefetch_on, profiler_on=False))
            gemm_tile.host_init()
            with T.cta():
                buf = T.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(gemm_tile)
                gemm_tile.init(smem_manager)
                with T.cta():
                    T.block_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirx.tile_class.persistent.init": True})
                    gemm_tile.class_init(smem_manager)
                with T.cta():
                    T.block_attr({"tirx.tile_class.prefetch": True})
                    if prefetch_on:
                        gemm_tile.prefetch(m_idx, n_idx, k_idx, A, B, output, None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    gemm_tile.run(m_idx, n_idx, k_idx, A, B, output, None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.persistent.finalize": True})
                    gemm_tile.class_finalize()    
        return gate_up_silu_func, num_tiles, tile_size
