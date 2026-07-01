from tvm.script import tir as T, tirx as Tx

from tvm.tirx.megakernel.kernels import (
    SplitKReduceRMSnormRopeQTile, 
    SplitKReduceRMSnormRopeAppendKTile,
    SplitKReduceAppendVTile,
)


class FuseSplitKReduceRMSnormRopeQTile(SplitKReduceRMSnormRopeQTile):

    @staticmethod
    def get_func(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor):
        symbolic_tile = FuseSplitKReduceRMSnormRopeQTile(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor)
        num_tiles = (symbolic_tile.m_split, qo_heads, 1)
        tile_size = (symbolic_tile.m_tile, 1, None)

        @T.prim_func(tirx=True, private=True)
        def split_k_reduce_rms_rope_func(partial_ptr: T.handle, q_weight_ptr: T.handle, rope_pos_ptr: T.handle, cos_sin_cache_ptr: T.handle,
                                         qkv_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "reduce_rms_rope"})
            batch_size = T.int32()
            cos_sin_cache_len = T.int32()
            partial = T.match_buffer(partial_ptr, [split_k_factor, batch_size, (qo_heads + 2 * kv_heads) * head_dim], "float32", scope="global")
            qkv = T.match_buffer(qkv_ptr, [batch_size, qo_heads + 2 * kv_heads, head_dim], "float16", scope="global")
            q_rms_weight = T.match_buffer(q_weight_ptr, [head_dim], "float16", scope="global")
            rope_pos = T.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            cos_sin_cache = T.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, head_dim], "float32", scope="global")
            reduce_rms_rope_tile = T.meta_var(FuseSplitKReduceRMSnormRopeQTile(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor))
            with T.cta():
                reduce_rms_rope_tile.init(None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    reduce_rms_rope_tile.run(m_idx, n_idx, k_idx, partial, qkv, q_rms_weight, rope_pos, cos_sin_cache)
        return split_k_reduce_rms_rope_func, num_tiles, tile_size

class FuseSplitKReduceRMSnormRopeAppendKTile(SplitKReduceRMSnormRopeAppendKTile):

    @staticmethod
    def get_func(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, page_size):
        symbolic_tile = FuseSplitKReduceRMSnormRopeAppendKTile(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, page_size)
        num_tiles = (symbolic_tile.m_split, kv_heads, 1)
        tile_size = (symbolic_tile.m_tile, 1, None)

        @T.prim_func(tirx=True, private=True)
        def split_k_reduce_rms_rope_append_func(partial_ptr: T.handle, k_weight_ptr: T.handle, rope_pos_ptr: T.handle, cos_sin_cache_ptr: T.handle,
                                                append_pos_ptr: T.handle, kv_cache_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "reduce_rms_rope_append"})
            max_page_num = T.int32()
            batch_size = T.int32()
            cos_sin_cache_len = T.int32()
            partial = T.match_buffer(partial_ptr, [split_k_factor, batch_size, (qo_heads + 2 * kv_heads) * head_dim], "float32", scope="global")
            k_rms_weight = T.match_buffer(k_weight_ptr, [head_dim], "float16", scope="global")
            rope_pos = T.match_buffer(rope_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            cos_sin_cache = T.match_buffer(cos_sin_cache_ptr, [cos_sin_cache_len, head_dim], "float32", scope="global")
            append_pos = T.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            kv_cache = T.match_buffer(kv_cache_ptr, [max_page_num, 2, kv_heads, page_size, head_dim], "float16", scope="global")
            reduce_rms_rope_append_tile = T.meta_var(FuseSplitKReduceRMSnormRopeAppendKTile(batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, page_size))
            with T.cta():
                reduce_rms_rope_append_tile.init(None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    reduce_rms_rope_append_tile.run(m_idx, n_idx, k_idx, partial, k_rms_weight, rope_pos, cos_sin_cache, append_pos, kv_cache)
        return split_k_reduce_rms_rope_append_func, num_tiles, tile_size


class FuseSplitKReduceAppendVTile(SplitKReduceAppendVTile):

    @staticmethod
    def get_func(batch_size, qo_heads, kv_heads, head_dim, split_k_factor, page_size):
        symbolic_tile = FuseSplitKReduceAppendVTile(batch_size, kv_heads, qo_heads, head_dim, split_k_factor, page_size)
        num_tiles = (symbolic_tile.m_split, kv_heads, 1)
        tile_size = (symbolic_tile.m_tile, 1, None)

        @T.prim_func(tirx=True, private=True)
        def split_k_reduce_append_func(partial_ptr: T.handle, kv_cache_ptr: T.handle, append_pos_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "reduce_append"})
            max_page_num = T.int32()
            batch_size = T.int32()
            partial = T.match_buffer(partial_ptr, [split_k_factor, batch_size, (qo_heads + 2 * kv_heads) * head_dim], "float32", scope="global")
            append_pos = T.match_buffer(append_pos_ptr, [batch_size], "int32", scope="global", offset_factor=1)
            kv_cache = T.match_buffer(kv_cache_ptr, [max_page_num, 2, kv_heads, page_size, head_dim], "float16", scope="global")
            reduce_rms_rope_append_tile = T.meta_var(FuseSplitKReduceAppendVTile(batch_size, kv_heads, qo_heads, head_dim, split_k_factor, page_size))
            with T.cta():
                reduce_rms_rope_append_tile.init(None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    reduce_rms_rope_append_tile.run(m_idx, n_idx, k_idx, partial, kv_cache, append_pos)
        return split_k_reduce_append_func, num_tiles, tile_size
