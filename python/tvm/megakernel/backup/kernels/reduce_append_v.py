from tvm.script import tir as T
from tvm.script import tirx as Tx

from tvm.tirx.megakernel.utils.base import Tile
from tvm.tirx.megakernel.utils.utils import ceildiv
from tvm.tirx.megakernel.utils.config import KernelConfig, F16_BYTES, F32_BYTES


class SplitKReduceAppendVTile(Tile):

    VEC_SIZE_32 = 16 // F32_BYTES
    VEC_SIZE_16 = 16 // F16_BYTES

    def __init__(self, batch_size, kv_heads, qo_heads, head_dim, split_k_factor, page_size, h_tile=1):
        super().__init__()
        self.batch_size = batch_size
        self.kv_heads = kv_heads
        self.qo_heads = qo_heads
        self.head_dim = head_dim
        self.split_k_factor = split_k_factor
        self.h_tile = h_tile
        self.batch_size = batch_size
        self.m_split = ceildiv(KernelConfig.SM_NUMBER, self.qo_heads + 2 * self.kv_heads)
        self.m_tile = ceildiv(self.batch_size, self.m_split)
        self.m_split = ceildiv(self.batch_size, self.m_tile)
        self.bdx = self.head_dim // self.VEC_SIZE_16
        self.bdy = KernelConfig.NUM_THREADS // self.bdx
        self.page_size = page_size
        assert self.bdx % 2 == 0
        assert KernelConfig.NUM_THREADS % self.bdx == 0

    def _alloc_local(self):
        self.idx = T.local_cell("int32", name="idx")
        self.pos = T.local_cell("int32", name="pos")
        self.vec_32 = T.alloc_local([self.VEC_SIZE_16], "float32", name="vec_32")
        self.tmp = T.alloc_local([self.VEC_SIZE_16], "float32", name="tmp")
        self.vec_16 = T.alloc_local([self.VEC_SIZE_16], "float16", name="vec_16")


    # handle: [batch_size, h_tile * head_dim]
    # bdx, bdy = head_dim // VEC_SIZE_16, NUM_THREADS // bdx
    @T.macro
    def run(self, m_idx, n_idx, k_idx, partial, kv_cache, pos_map):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var(tid // self.bdx)
            stx = T.meta_var(tx * self.VEC_SIZE_16)
            handle_num = T.meta_var(T.min(self.m_tile, self.batch_size - m_idx * self.m_tile) * self.h_tile)
            self._alloc_local()
            with T.thread():
                batch_idx = T.meta_var(m_idx * self.m_tile + self.idx // self.h_tile)
                head_idx = T.meta_var(n_idx * self.h_tile + self.idx % self.h_tile)
                self.idx = ty
                while self.idx < handle_num:
                    # reduce
                    qkv_stx = T.meta_var((self.qo_heads + self.kv_heads + head_idx) * self.head_dim + tx * self.VEC_SIZE_16)
                    for kv in T.unroll(self.VEC_SIZE_16):
                        self.vec_32[kv] = 0.0
                    for kt in T.serial(self.split_k_factor):
                        Tx.copy(self.tmp[:], partial[kt, batch_idx, qkv_stx:qkv_stx + self.VEC_SIZE_16])
                        for kv in T.unroll(self.VEC_SIZE_16):
                            self.vec_32[kv] += self.tmp[kv]
                    Tx.cast(self.vec_16[:], self.vec_32[:])
                    # append
                    self.pos = T.cuda.ldg(T.address_of(pos_map[batch_idx]), "int32")
                    page_id = T.meta_var(self.pos // self.page_size)
                    offset = T.meta_var(self.pos % self.page_size)
                    Tx.copy(kv_cache[page_id, 1, head_idx, offset, stx:stx + self.VEC_SIZE_16], self.vec_16[:])
                    self.idx += self.bdy
