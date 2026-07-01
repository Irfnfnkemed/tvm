import tvm.script.tir as T
import tvm.script.tirx as Tx

from tvm.tirx.megakernel.utils.base import Tile
from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.utils.config import KernelConfig, F16_BYTES


class RopeTile(Tile):

    # qk_tvm: [batch_size, qo_heads + 2 * kv_heads, head_dim]
    # cos_sin_cache_tvm: [1, head_dim]

    loop_inner = 1
    min_bdy = 1
    h_tile = 1

    def __init__(self, batch_size, qo_heads, kv_heads, head_dim):
        super().__init__()
        self.qo_heads = qo_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.vec_size = max(16 // F16_BYTES, self.head_dim // 32)
        self.bdx = self.head_dim // self.vec_size
        self.bdy = KernelConfig.NUM_THREADS // self.bdx

        self.batch_size = batch_size
        self.m_split = ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads)
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)


    def _alloc_local(self):
        self.idx = T.alloc_local([1], "int32", name="idx")
        self.cos = T.alloc_local([self.vec_size], "float32", name="cos")
        self.sin = T.alloc_local([self.vec_size], "float32", name="sin")
        self.qk_vec = T.alloc_local([self.vec_size], "float16", name="qk_vec")
        self.qk_vec32 = T.alloc_local([self.vec_size], "float32", name="qk_vec32")
        self.qk_vec32_other = T.alloc_local([self.vec_size], "float32", name="qk_vec32_other")
        self.mask = T.alloc_local([1], "uint32", name="mask")


    @T.macro
    def run(self, m_idx, n_idx, k_idx, qkv, cos_sin_cache, rope_pos):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var(tid // self.bdx)
            half_dim = self.head_dim // 2

            self._alloc_local()

            with T.thread():
                self.idx[0] = ty

                while (
                    self.idx[0] < self.m_tile * self.h_tile
                    and m_idx * self.m_tile + self.idx[0] // self.h_tile < self.batch_size
                ):
                    batch_idx = T.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                    head_idx = T.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                    stx = T.meta_var(tx * self.vec_size)
                    cache_stx = T.meta_var(stx % half_dim)
                    pos = T.meta_var(rope_pos[batch_idx])

                    if batch_idx < self.batch_size and head_idx < self.qo_heads + self.kv_heads:
                        # load cache
                        for kv in T.unroll(self.vec_size):
                            self.cos[kv] = cos_sin_cache[pos, cache_stx + kv]
                        for kv in T.unroll(self.vec_size):
                            self.sin[kv] = cos_sin_cache[pos, half_dim + cache_stx + kv]

                        # load qk
                        for kv in T.unroll(self.vec_size):
                            self.qk_vec[kv] = qkv[batch_idx, head_idx, stx + kv]
                        Tx.cast(self.qk_vec32[:], self.qk_vec[:], vec_len=self.vec_size)

                        # shuffle qk value
                        if ty % 2 == 0 and (
                            batch_idx + 1 == self.batch_size
                            or self.idx[0] // self.h_tile + 1 == self.m_tile
                        ):
                            self.mask[0] = 0xFFFF
                        else:
                            self.mask[0] = 0xFFFFFFFF
                        for kv in T.serial(self.vec_size):
                            self.qk_vec32_other[kv] = T.tvm_warp_shuffle_xor(
                                self.mask[0], self.qk_vec32[kv], self.bdx // 2, 32, 32
                            )

                        # compute rope
                        if stx < half_dim:
                            for kv in T.unroll(self.vec_size):
                                self.qk_vec32[kv] = (
                                    self.qk_vec32[kv] * self.cos[kv]
                                    - self.qk_vec32_other[kv] * self.sin[kv]
                                )
                        else:
                            for kv in T.unroll(self.vec_size):
                                self.qk_vec32[kv] = (
                                    self.qk_vec32[kv] * self.cos[kv]
                                    + self.qk_vec32_other[kv] * self.sin[kv]
                                )

                        # store qk
                        Tx.cast(self.qk_vec[:], self.qk_vec32[:], vec_len=self.vec_size)
                        for kv in T.unroll(self.vec_size):
                            qkv[batch_idx, head_idx, stx + kv] = self.qk_vec[kv]

                    self.idx[0] += self.bdy
