from tvm.script import tir as T

from tvm.tirx.megakernel.utils.base import KernelConfig, Tile
from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.utils.config import F16_BYTES

class AppendKVTile(Tile):

    # kv_cache_tvm: [max_page_num, 2, kv_heads, page_size, head_dim]
    # qkv_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # kv_indptr_tvm: [batch_size + 1]
    # kv_indices_tvm: [total_page_num]
    # kv_last_page_len_tvm: [batch_size]
    # pos_map_tvm: [batch_size]

    loop_inner = 1
    min_bdy = 1
    h_tile = 1

    def __init__(self, batch_size, num_attention_heads, num_key_value_heads, head_dim, page_size):
        super().__init__()
        self.qo_heads = num_attention_heads
        self.kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.vec_size = max(16 // F16_BYTES, self.head_dim // 32)
        self.bdx = self.head_dim // self.vec_size
        self.bdy = KernelConfig.NUM_THREADS // self.bdx

        self.batch_size = batch_size
        self.m_split = ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads)
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)

    def _alloc_local(self):
        self.idx = T.alloc_local([1], "int32", name="idx")
        self.pos = T.alloc_local([1], "int32", name="pos")
        self.vec = T.alloc_local([self.vec_size], "float16", name="vec")


    @T.macro
    def run(self, m_idx, n_idx, k_idx, kv_cache, qkv, pos_map):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var(tid // self.bdx)
            stx = T.meta_var(tx * self.vec_size)
            self._alloc_local()

            with T.thread():
                self.idx[0] = ty

                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = T.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = T.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    if batch_idx < self.batch_size and head_idx < self.kv_heads:

                        self.pos[0] = T.cuda.ldg(
                            T.address_of(pos_map[batch_idx]), "int32"
                        )
                        page_id = T.meta_var(self.pos[0] // self.page_size)
                        offset = T.meta_var(self.pos[0] % self.page_size)
                        for vec in T.vectorized(self.vec_size):
                            self.vec[vec] = qkv[
                                batch_idx,
                                self.qo_heads + k_idx * self.kv_heads + head_idx,
                                stx + vec,
                            ]
                        for vec in T.vectorized(self.vec_size):
                            kv_cache[page_id, k_idx, head_idx, offset, stx + vec] = (
                                self.vec[vec]
                            )

                    self.idx[0] += self.bdy
