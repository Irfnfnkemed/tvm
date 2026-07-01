from tvm.script import tir as T, tirx as Tx

from tvm.tirx.megakernel.utils.config import KernelConfig
from tvm.tirx.megakernel.utils.base import SmemManager
from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.kernels import BatchMergeTile

class FuseBatchMergeTile(BatchMergeTile):

    @staticmethod   
    def get_func(head_dim, attn_task_num, kv_heads, qo_heads, batch_size):
        num_tiles = (T.if_then_else(attn_task_num > kv_heads * batch_size, ceildiv(batch_size * qo_heads, KernelConfig.WG_NUMBER * KernelConfig.WARP_NUMBER), 0), 1, 1)
        tile_size = (None, None, None)

        @T.prim_func(tirx=True, private=True)
        def batch_merge_func(partial_o_ptr: T.handle, partial_lse_ptr: T.handle, num_qo_len_ptr: T.handle, merge_indptr_ptr: T.handle,
                             merge_o_indices_ptr: T.handle, final_o_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "batch_merge"})
            batch_size = T.int32()
            partial_o = T.match_buffer(partial_o_ptr, (FuseBatchMergeTile.max_num_kv_splits * kv_heads * head_dim), "float32", scope="global")
            final_o = T.match_buffer(final_o_ptr, (batch_size, qo_heads, head_dim), "float16", scope="global")
            partial_lse = T.match_buffer(partial_lse_ptr, (FuseBatchMergeTile.max_num_kv_splits * kv_heads), "float32", scope="global")
            num_qo_len = T.match_buffer(num_qo_len_ptr, (1), "int32", scope="global", offset_factor=1)
            merge_indptr = T.match_buffer(merge_indptr_ptr, (FuseBatchMergeTile.max_num_kv_splits), "int32", scope="global", offset_factor=1)
            merge_o_indices = T.match_buffer(merge_o_indices_ptr, (FuseBatchMergeTile.max_num_kv_splits), "int32", scope="global", offset_factor=1)
            batch_merge_tile = T.meta_var(FuseBatchMergeTile(head_dim, kv_heads, qo_heads))
            with T.cta():
                buf = T.alloc_buffer((KernelConfig.MAX_SMEM_SIZE,), "uint8", scope="shared.dyn")
                smem_manager = T.meta_var(SmemManager(KernelConfig.MAX_SMEM_SIZE, 16384, buf.data, fusion_mode=True))
                smem_manager.set_tile(batch_merge_tile)
                batch_merge_tile.init(smem_manager)
                with T.cta():
                    T.block_attr({"tirx.megakernel.persistent.init": True})
                    smem_manager.init()
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    batch_merge_tile.run(m_idx, n_idx, k_idx, partial_o, final_o, partial_lse, num_qo_len, merge_indptr, merge_o_indices)     
        return batch_merge_func, num_tiles, tile_size
