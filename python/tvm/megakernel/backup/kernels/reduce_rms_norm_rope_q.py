from tvm.script import tir as T
from tvm.script import tirx as Tx

from tvm.tirx.megakernel.utils.base import Tile
from tvm.tirx.megakernel.utils.utils import ceildiv, find_power_of_two, rsqrt
from tvm.tirx.megakernel.utils.config import KernelConfig, F16_BYTES


# TODO: pipeline
class SplitKReduceRMSnormRopeQTile(Tile):

    # weight_tvm: [num_heads]
    # qk_tvm: [batch_size, num_heads, head_dim]

    loop_inner = 1
    min_bdy = 1

    def __init__(self, batch_size, rms_norm_eps, qo_heads, kv_heads, head_dim, split_k_factor, h_tile=1, use_rms_norm=True):
        super().__init__()
        self.rms_norm_eps = rms_norm_eps
        self.qo_heads = qo_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.h_tile = h_tile
        self.use_rms_norm = use_rms_norm
        self.split_k_factor = split_k_factor
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
        self.mask = T.alloc_local([1], "uint32", name="mask")
        self.weight_vec = T.alloc_local([self.vec_size], "float16", name="weight_vec")
        self.weight_vec_f32 = T.alloc_local([self.vec_size], "float32", name="weight_vec_f32")
        self.sum_sq = T.alloc_local([1], "float32", name="sum_sq")
        self.rms_norm = T.alloc_local([1], "float32", name="rms_norm")
        self.cos = T.alloc_local([self.vec_size], "float32", name="cos")
        self.sin = T.alloc_local([self.vec_size], "float32", name="sin")
        self.q_vec = T.alloc_local([self.vec_size], "float16", name="q_vec")
        self.q_vec32 = T.alloc_local([self.vec_size], "float32", name="q_vec32")
        self.q_vec32_other = T.alloc_local([self.vec_size], "float32", name="q_vec32_other")


    @T.macro
    def run(self, m_idx, n_idx, k_idx, partial, qkv, q_weight, rope_pos, cos_sin_cache):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            tx = T.meta_var(tid % self.bdx)
            ty = T.meta_var(tid // self.bdx)
            self._alloc_local()
            with T.thread():
                batch_idx = T.meta_var(m_idx * self.m_tile + self.idx[0] // self.h_tile)
                head_idx = T.meta_var(n_idx * self.h_tile + self.idx[0] % self.h_tile)
                st = T.meta_var(tx * self.vec_size)
                half_dim = T.meta_var(self.head_dim // 2)
                group_in_warp = T.meta_var(32 // self.bdx)
                cache_stx = T.meta_var(st % half_dim)
                handle_num = T.meta_var(T.min(self.m_tile, self.batch_size - m_idx * self.m_tile) * self.h_tile)
                self.idx[0] = ty
                while self.idx[0] < handle_num:
                    self.pos[0] = rope_pos[batch_idx]
                    self.sum_sq[0] = 0.0
                    # reduce
                    qkv_stx = T.meta_var(head_idx * self.head_dim + tx * self.vec_size)
                    for kv in T.unroll(self.vec_size):
                        self.q_vec32[kv] = 0.0
                    for kt in T.serial(self.split_k_factor):
                        Tx.copy(self.q_vec32_other[:], partial[kt, batch_idx, qkv_stx:qkv_stx + self.vec_size])
                        for kv in T.unroll(self.vec_size):
                            self.q_vec32[kv] += self.q_vec32_other[kv]
                    remain = handle_num - self.idx[0] // group_in_warp * group_in_warp
                    if remain >= group_in_warp:
                        self.mask[0] = 0xFFFFFFFF
                    else:
                        self.mask[0] = (1 << (remain * self.bdx)) - 1
                    if self.use_rms_norm:
                        # sum square
                        for kv in T.unroll(self.vec_size):
                            self.sum_sq[0] += self.q_vec32[kv] * self.q_vec32[kv]
                        # warp reduce sum
                        for kr in T.unroll(find_power_of_two(self.bdx // 2) + 1):
                            self.sum_sq[0] = self.sum_sq[0] + T.tvm_warp_shuffle_xor(
                                self.mask[0], self.sum_sq[0], (self.bdx // 2) >> kr, 32, 32
                            )
                        # rms norm
                        self.rms_norm[0] = rsqrt(self.sum_sq[0] / self.head_dim + self.rms_norm_eps)
                        # handle the weight
                        Tx.copy(self.weight_vec[:], q_weight[st:st + self.vec_size])
                        Tx.cast(self.weight_vec_f32[:], self.weight_vec[:])
                        for kv in T.unroll(self.vec_size):
                            self.q_vec32[kv] = self.q_vec32[kv] * self.rms_norm[0] * self.weight_vec_f32[kv]
                    # load cache
                    Tx.copy(self.cos[:], cos_sin_cache[self.pos[0], cache_stx:cache_stx + self.vec_size])
                    Tx.copy(self.sin[:], cos_sin_cache[self.pos[0], half_dim + cache_stx:half_dim + cache_stx + self.vec_size])
                    # shuffle q value
                    for kv in T.serial(self.vec_size):
                        self.q_vec32_other[kv] = T.tvm_warp_shuffle_xor(
                            self.mask[0], self.q_vec32[kv], self.bdx // 2, 32, 32
                        )
                    # compute rope
                    if st < half_dim:
                        for kv in T.unroll(self.vec_size):
                            self.q_vec32[kv] = self.q_vec32[kv] * self.cos[kv] - self.q_vec32_other[kv] * self.sin[kv]
                    else:
                        for kv in T.unroll(self.vec_size):
                            self.q_vec32[kv] = self.q_vec32[kv] * self.cos[kv] + self.q_vec32_other[kv] * self.sin[kv]
                    # store q
                    Tx.cast(self.q_vec[:], self.q_vec32[:])
                    Tx.copy(qkv[batch_idx, head_idx, st:st + self.vec_size], self.q_vec[:])
                    self.idx[0] += self.bdy
