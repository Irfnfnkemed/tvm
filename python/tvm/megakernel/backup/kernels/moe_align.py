from tvm.script import tir as T
from tvm.script import tirx as Tx

from tvm.tirx.megakernel.utils.base import Tile, SmemManager
from tvm.tirx.megakernel.utils.utils import ceildiv, next_power_of_two
from tvm.tirx.megakernel.utils.config import KernelConfig, F16_BYTES


class MOEAlignTile(Tile):

    def __init__(self, num_experts, numel, block_size, pad_sorted_token_ids=False):
        super().__init__()
        self.num_experts = num_experts
        self.scan_size = next_power_of_two(num_experts)
        self.numel = numel  # numel = num_tokens * num_experts
        self.block_size = block_size
        self.pad_sorted_token_ids = pad_sorted_token_ids

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        # alloc shared memory
        self.shared_counts = smem_manager.alloc([self.num_experts], "int32", name="shared_counts")
        self.prefix = smem_manager.alloc([self.num_experts + 1], "int32", name="prefix")
        self.scan_buf = smem_manager.alloc([self.scan_size], "int32", name="scan_buf")
        self.warp_sums = smem_manager.alloc(
            [KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], "int32", name="warp_sums"
        )
        self.s_total_tokens_post_pad = smem_manager.alloc([1], "int32", name="s_total_tokens_post_pad")

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    @T.macro
    def warp_exclusive_scan(self, v, output, mask=0xFFFFFFFF):
        # offset = T.alloc_cell("int32", name="offset")
        # original = T.alloc_cell("int32", name="original")
        # with T.warp():
        #     lane_id = T.thread_id(32, parent="warp")
        #     offset = 1
        #     original = v
        #     while offset < 32:
        #         n = T.tvm_warp_shuffle_up(mask, v, offset)
        #         if lane_id >= offset:
        #             v += n
        #         offset = offset << 1
        #     output = v - original
        #     v = original
        output[0] = T.cuda.func_call(
            "warp_exclusive_scan",
            v,
            mask,
            source_code="""
            __device__ __forceinline__ int warp_exclusive_scan(int v, unsigned mask = 0xffffffffu) {
            int original = v;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int n = __shfl_up_sync(mask, v, offset);
                if ((threadIdx.x & (32 - 1)) >= offset) v += n;
            }
            return v - original;
            }
                                  """,
            return_type="int32",
        )

    @T.macro
    def run(
        self,
        m_idx,
        n_idx,
        k_idx,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        total_tokens_post_pad,
        cumsum_buffer,
        num_valid_tokens=None,
    ):
        idx = T.alloc_local([1], "int32", name="idx")
        pre = T.alloc_local([1], "int32", name="pre")
        sum_val = T.alloc_local([1], "int32", name="sum_val")
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            lane_id = T.thread_id([32], parent="warp")
            warp_id = T.warp_id([KernelConfig.WARP_NUMBER * KernelConfig.WG_NUMBER], parent="cta")
            self.smem_manager.wait_all("cta")
            if tid < self.num_experts:
                self.shared_counts[tid] = 0
            T.tvm_storage_sync("shared")
            idx[0] = tid
            while idx[0] < self.numel:
                expert_id = topk_ids[
                    idx[0]
                ]  # TODO: the reference expert use topk_ids[idx[0]] + 1. Is it correct?
                T.cuda.atomic_add(T.address_of(self.shared_counts[expert_id]), 1)
                idx[0] += KernelConfig.NUM_THREADS
            T.tvm_storage_sync("shared")
            if tid < self.num_experts:
                padded_count = ceildiv(self.shared_counts[tid], self.block_size) * self.block_size
                self.scan_buf[tid] = padded_count
            elif tid < self.scan_size:
                self.scan_buf[tid] = 0
            T.tvm_storage_sync("shared")
            v = T.if_then_else(tid < self.num_experts, self.scan_buf[tid], 0)
            self.warp_exclusive_scan(v, pre)
            if lane_id == 31:
                self.warp_sums[warp_id] = pre[0] + v
            T.tvm_storage_sync("shared")
            num_warps_for_scan = T.meta_var(ceildiv(self.scan_size, 32))
            if warp_id == 0:
                val = T.if_then_else(lane_id < num_warps_for_scan, self.warp_sums[lane_id], 0)
                self.warp_exclusive_scan(val, sum_val)
                if lane_id == num_warps_for_scan - 1:
                    self.prefix[self.num_experts] = sum_val[0] + val
                    self.s_total_tokens_post_pad[0] = sum_val[0] + val
                    total_tokens_post_pad[0] = self.s_total_tokens_post_pad[0]
                self.warp_sums[lane_id] = sum_val[0]
            T.tvm_storage_sync("shared")
            if tid < self.scan_size:
                self.scan_buf[tid] = pre[0] + self.warp_sums[warp_id]
            if tid < self.num_experts:
                self.prefix[tid] = self.scan_buf[tid]
            if tid <= self.num_experts:
                cumsum_buffer[tid] = self.prefix[tid]
            T.tvm_storage_sync("shared")
            num_blocks = self.s_total_tokens_post_pad[0] // self.block_size
            idx[0] = tid
            while idx[0] < num_blocks:
                block_start = idx[0] * self.block_size
                left = T.alloc_local([1], "int32", name="left")
                right = T.alloc_local([1], "int32", name="right")
                with T.thread():
                    left[0] = 0
                    right[0] = self.num_experts
                    while left[0] < right[0]:
                        mid = (left[0] + right[0]) // 2
                        if self.prefix[mid] <= block_start:
                            left[0] = mid + 1
                        else:
                            right[0] = mid
                expert_ids[idx[0]] = left[0] - 1  # TODO: the reference expert use left - 2
                if num_valid_tokens is not None:
                    if idx[0] < cumsum_buffer[left[0]] // self.block_size - 1:
                        num_valid_tokens[idx[0]] = self.block_size
                    else:
                        num_valid_tokens[idx[0]] = (
                            self.shared_counts[left[0] - 1] - 1
                        ) % self.block_size + 1

                idx[0] += KernelConfig.NUM_THREADS
            if self.pad_sorted_token_ids:
                VEC_SIZE = 4
                idx[0] = tid * VEC_SIZE
                out_ptr = sorted_token_ids.view(-1)
                fill_vec = T.alloc_buffer([4], "int32", scope="local")
                for vec in T.vectorized(VEC_SIZE):
                    fill_vec[vec] = self.numel
                while idx[0] < self.s_total_tokens_post_pad[0]:
                    for vec in T.vectorized(VEC_SIZE):
                        out_ptr[idx[0] + vec] = fill_vec[vec]
                    idx[0] += VEC_SIZE * KernelConfig.NUM_THREADS
            self.smem_manager.arrive_all("cta")
            self.smem_manager.advance()


class CountAndSortExpertTokens(Tile):

    VEC_SIZE = 16 // F16_BYTES
    PIPE_DEPTH = 8

    def __init__(self, numel, hidden_size, topk):
        super().__init__()
        self.numel = numel  # numel = num_tokens * num_experts
        self.hidden_size = hidden_size
        assert self.hidden_size <= KernelConfig.NUM_THREADS * self.VEC_SIZE
        self.topk = topk

    def _alloc_buffer(self, smem_manager: SmemManager):
        self.smem_manager = smem_manager
        self.s_rank_post_pad = smem_manager.alloc([KernelConfig.NUM_THREADS], "int64", name="s_rank_post_pad")
        self.fetched_data = smem_manager.alloc(
            [self.PIPE_DEPTH, KernelConfig.NUM_THREADS, self.VEC_SIZE], "float16", name="fetched_data"
        )

    def init(self, smem_manager: SmemManager):
        self._alloc_buffer(smem_manager)

    # fmt: off
    @T.macro
    def run(self, m_idx, n_idx, k_idx, topk_ids, sorted_token_ids, cumsum_buffer, data, reordered_data):
        idx = T.alloc_local([1], "int32", name="idx")
        cnt = T.alloc_local([1], "int32", name="cnt")
        col_idx = T.alloc_local([1], "int32", name="col_idx")
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            process_token_idx = m_idx + tid * KernelConfig.SM_NUMBER
            self.smem_manager.wait_all("cta")
            if process_token_idx < self.numel:
                expert_id = topk_ids[process_token_idx]
                rank_post_pad = T.cuda.atomic_add(T.address_of(cumsum_buffer[expert_id]), 1)
                sorted_token_ids[rank_post_pad] = process_token_idx
                self.s_rank_post_pad[tid] = rank_post_pad
            idx[0] = m_idx
            T.tvm_storage_sync("shared")
            col_idx[0] = tid * self.VEC_SIZE
            for i in T.unroll(self.PIPE_DEPTH - 1):
                with T.thread():
                    if idx[0] < self.numel and col_idx[0] < self.hidden_size:
                        Tx.copy_async(self.fetched_data[i, tid, :], data[idx[0] // self.topk, col_idx[0]:col_idx[0] + self.VEC_SIZE], dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                    T.ptx.cp_async.commit_group()
                idx[0] += KernelConfig.SM_NUMBER
            cnt[0] = 0
            while idx[0] < self.numel + (self.PIPE_DEPTH - 1) * KernelConfig.SM_NUMBER:
                with T.thread():
                    if idx[0] < self.numel and col_idx[0] < self.hidden_size:
                        cp_pipe_idx = T.meta_var((idx[0] // KernelConfig.SM_NUMBER) % self.PIPE_DEPTH)
                        Tx.copy_async(self.fetched_data[cp_pipe_idx, tid, :], data[idx[0] // self.topk, col_idx[0]:col_idx[0] + self.VEC_SIZE], dispatch="non-bulk-copy", vec_len=self.VEC_SIZE)
                    T.ptx.cp_async.commit_group()
                    T.ptx.cp_async.wait_group(self.PIPE_DEPTH - 1)
                    rank_post_pad = self.s_rank_post_pad[cnt[0]]
                    pipe_idx = T.meta_var(cnt[0] % self.PIPE_DEPTH)
                    if col_idx[0] < self.hidden_size:
                        for vec in T.vectorized(self.VEC_SIZE):
                            reordered_data[rank_post_pad, col_idx[0] + vec] = self.fetched_data[pipe_idx, tid, vec]
                    idx[0] += KernelConfig.SM_NUMBER
                    cnt[0] += 1
            self.smem_manager.arrive_all("cta")
            self.smem_manager.advance()
    # fmt: on
