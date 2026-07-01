from tvm.script import tir as T, tirx as Tx

from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.kernels import SplitKReduceTile

class FuseSplitKReduceTile(SplitKReduceTile):

    @staticmethod
    def get_func(batch_size, N, dtype, split_k_factor):
        symbolic_tile = FuseSplitKReduceTile(batch_size, N, dtype, split_k_factor)
        num_tiles = (symbolic_tile.M_split, ceildiv(N, symbolic_tile.N_TILE), 1)
        tile_size = (symbolic_tile.M_TILE, symbolic_tile.N_TILE, 1)

        @T.prim_func(tirx=True, private=True)
        def split_k_reduce_func(input_ptr: T.handle, output_ptr: T.handle, m_idx: T.int32, n_idx: T.int32, k_idx: T.int32):
            T.func_attr({"megakernel.device_func": "split_k_reduce"})
            batch_size = T.int32()
            input = T.match_buffer(input_ptr, (split_k_factor, batch_size, N), "float32", layout="default")
            output = T.match_buffer(output_ptr, (batch_size, N), dtype, layout="default")
            split_k_reduce_tile = T.meta_var(FuseSplitKReduceTile(batch_size, N, dtype, split_k_factor))
            with T.cta():
                split_k_reduce_tile.init(None)
                with T.cta():
                    T.block_attr({"tirx.tile_class.run": True})
                    split_k_reduce_tile.run(m_idx, n_idx, k_idx, input, output)
        return split_k_reduce_func, num_tiles, tile_size
