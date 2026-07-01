from tvm.script import tir as T, tirx as Tx

from tvm.tirx.megakernel.utils.config import KernelConfig
from tvm.tirx.megakernel.utils.base import SmemManager
from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.kernels import BatchAttnTile


class FuseBatchAttnTile(BatchAttnTile):

    @staticmethod
    def get_func(page_size, qo_heads, kv_heads, head_dim, attn_task_num, prefetch_on=False):
        num_tiles = (T.min(ceildiv(attn_task_num, KernelConfig.WG_NUMBER), KernelConfig.SM_NUMBER), 1, 1)
        tile_size = (None, None, None)

        @T.prim_func(tirx=True, private=True)
        def batch_attn_func(q_ptr: T.handle, kv_ptr: T.handle, q_indptr_ptr: T.handle, kv_indptr_ptr: T.handle, 
                            partial_indptr_ptr: T.handle, kv_indices_ptr: T.handle, q_len_ptr: T.handle, kv_len_ptr: T.handle,
                            q_start_ptr: T.handle, kv_start_ptr: T.handle, kv_end_ptr: T.handle, kv_head_idx_ptr: T.handle,
                            work_indptr_ptr: T.handle, len_kv_chunk_ptr: T.handle, o_ptr: T.handle, partial_o_ptr: T.handle,
                            partial_lse_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "batch_attn"})
            batch_size = T.int32()
            max_page_num = T.int32()
            total_page_num = T.int32()
            q = T.match_buffer(q_ptr, (batch_size, qo_heads + 2 * kv_heads, head_dim), "float16", scope="global")
            kv = T.match_buffer(kv_ptr, (max_page_num, 2, kv_heads, page_size, head_dim), "float16", scope="global")
            q_indptr = T.match_buffer(q_indptr_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            kv_indptr = T.match_buffer(kv_indptr_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            partial_indptr = T.match_buffer(partial_indptr_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            kv_indices = T.match_buffer(kv_indices_ptr, (total_page_num), "int32", scope="global", offset_factor=1)
            q_len = T.match_buffer(q_len_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            kv_len = T.match_buffer(kv_len_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            q_start = T.match_buffer(q_start_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            kv_start = T.match_buffer(kv_start_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            kv_end = T.match_buffer(kv_end_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            kv_head_idx = T.match_buffer(kv_head_idx_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            work_indptr = T.match_buffer(work_indptr_ptr, (FuseBatchAttnTile.max_total_num_workers), "int32", scope="global", offset_factor=1)
            len_kv_chunk = T.match_buffer(len_kv_chunk_ptr, (2), "int32", scope="global", offset_factor=1)
            o = T.match_buffer(o_ptr, (batch_size, qo_heads, head_dim), "float16", scope="global")
            partial_o = T.match_buffer(partial_o_ptr, (FuseBatchAttnTile.max_num_kv_splits * kv_heads * head_dim), "float32", scope="global")
            partial_lse = T.match_buffer(partial_lse_ptr, (FuseBatchAttnTile.max_num_kv_splits * kv_heads), "float32", scope="global")
            attn_tile = T.meta_var(FuseBatchAttnTile(page_size, qo_heads, kv_heads, head_dim, prefetch_on=prefetch_on, profiler_on=False))
            with T.cta():
                buf = T.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(attn_tile)
                attn_tile.init(smem_manager)
                with T.cta():
                    T.block_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirx.tile_class.prefetch": True})
                    attn_tile.prefetch(m_idx, n_idx, k_idx, q, kv, q_indptr, kv_indptr, partial_indptr, kv_indices, q_len, kv_len,
                                       q_start, kv_start, kv_end, kv_head_idx, work_indptr, len_kv_chunk, o, partial_o, partial_lse, None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    attn_tile.run(m_idx, n_idx, k_idx, q, kv, q_indptr, kv_indptr, partial_indptr, kv_indices, q_len, kv_len,
                                    q_start, kv_start, kv_end, kv_head_idx, work_indptr, len_kv_chunk, o, partial_o, partial_lse, None)
        return batch_attn_func, num_tiles, tile_size
