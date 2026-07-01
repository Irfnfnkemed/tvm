from __future__ import annotations

import torch
import tvm

# from tvm.script import tirx as T
# from tvm.script.tirx import tile as Tx

import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

# =========================
# Shape config
# =========================

M = 1024
N = 1024

BLOCK_M = 64
BLOCK_N = 64

NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N

SM_COUNT = 148

# Reserve one CTA per row-block for stage2.
# The rest of CTAs produce stage1 partial sums.
STAGE2_WORKERS = NUM_BLOCK_M
STAGE1_WORKERS = SM_COUNT - STAGE2_WORKERS

EVENT_BASE = 1 << 16


@T.jit
def _two_stage_reduce_kernel(
    A: T.Buffer((M, N), "float32"),
    B: T.Buffer((M, NUM_BLOCK_N), "float32"),
    C: T.Buffer((M, 1), "float32"),
    row_ready: T.Buffer((NUM_BLOCK_M,), "int32"),
    *,
    M: T.constexpr,
    N: T.constexpr,
    BLOCK_M: T.constexpr,
    BLOCK_N: T.constexpr,
    NUM_BLOCK_M: T.constexpr,
    NUM_BLOCK_N: T.constexpr,
    SM_COUNT: T.constexpr,
    STAGE1_WORKERS: T.constexpr,
    STAGE2_WORKERS: T.constexpr,
    EVENT_BASE: T.constexpr,
):
    T.device_entry()
    T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})

    bx = T.cta_id([SM_COUNT])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # Same style as your low-level examples: use inline CUDA device helper
    # for global event notification.
    atomic_add_int32 = T.meta_var("""
__forceinline__ __device__ void atomic_add_int32(int32_t* addr, int32_t value) {
    asm volatile("red.async.release.global.gpu.add.s32 [%0], %1;"
                 :: "l"(addr), "r"(value)
                 : "memory");
}
""")

    # =========================
    # Shared memory pool
    # =========================

    pool = T.SMEMPool()

    A_smem = pool.alloc((BLOCK_M, BLOCK_N), "float32")
    P_smem = pool.alloc((BLOCK_M, 1), "float32")

    B_smem = pool.alloc((BLOCK_M, NUM_BLOCK_N), "float32")
    C_smem = pool.alloc((BLOCK_M, 1), "float32")

    pool.commit()

    # =========================
    # Event helpers
    # =========================

    @T.inline
    def notify_row_ready(m_idx):
        # row_ready[m] is initialized as:
        #   NUM_BLOCK_N * (EVENT_BASE + 1)
        #
        # Each stage1 tile subtracts:
        #   EVENT_BASE + 1
        #
        # Therefore row_ready[m] becomes 0 only after all NUM_BLOCK_N
        # partial sums of this row-block are produced.
        T.cuda.cta_sync()
        if (warp_id == 0) & (lane_id == 0):
            T.cuda.func_call(
                "atomic_add_int32",
                row_ready.ptr_to([m_idx]),
                -(EVENT_BASE + 1),
                source_code=atomic_add_int32,
            )

    @T.inline
    def wait_row_ready(m_idx):
        state = T.alloc_buffer([1], "int32", scope="local")

        while 1:
            T.ptx.ld_global_acquire(
                state[0],
                row_ready.access_ptr(
                    "r",
                    offset=row_ready.elem_offset_of([m_idx]),
                ),
            )
            if T.cuda.syncthreads_and(state[0] == 0):
                break
            T.cuda.nano_sleep(40)

    # =========================
    # Stage 1 tile
    # =========================

    @T.inline
    def stage1(m_idx, n_idx):
        # A tile:
        #   A[m-block, n-block] -> B[m-block, n]
        #
        # This computes row-wise partial sums over one N-block.

        Tx.copy(
            A_smem,
            A[
                m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M,
                n_idx * BLOCK_N : (n_idx + 1) * BLOCK_N,
            ],
        )

        Tx.sum(P_smem, A_smem)

        Tx.copy(
            B[m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M, n_idx],
            P_smem,
        )

        notify_row_ready(m_idx)

    # =========================
    # Stage 2 tile
    # =========================

    @T.inline
    def stage2(m_idx):
        # Wait until all partial sums of this row-block are ready.
        wait_row_ready(m_idx)

        # B tile:
        #   B[m-block, :] -> C[m-block, 0]
        #
        # This computes final row-wise reduction.

        Tx.copy(
            B_smem,
            B[m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M, :],
        )

        Tx.sum(C_smem, B_smem)

        Tx.copy(
            C[m_idx * BLOCK_M : (m_idx + 1) * BLOCK_M, 0],
            C_smem,
        )

    # =========================
    # Minimal static role split
    # =========================
    #
    # CTA [0, STAGE1_WORKERS)     : stage1 workers
    # CTA [STAGE1_WORKERS, SM)    : stage2 workers
    #
    # This is intentionally simple for the demo. Later this part can become
    # the minimal megakernel tile scheduler.

    if bx < STAGE1_WORKERS:
        linear: T.int32
        linear = bx

        while linear < NUM_BLOCK_M * NUM_BLOCK_N:
            m_idx = T.meta_var(linear // NUM_BLOCK_N)
            n_idx = T.meta_var(linear % NUM_BLOCK_N)

            stage1(m_idx, n_idx)

            linear = linear + STAGE1_WORKERS

    else:
        stage2_id = T.meta_var(bx - STAGE1_WORKERS)

        if stage2_id < NUM_BLOCK_M:
            stage2(stage2_id)


def tir_kernel():
    return _two_stage_reduce_kernel.specialize(
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_M=NUM_BLOCK_M,
        NUM_BLOCK_N=NUM_BLOCK_N,
        SM_COUNT=SM_COUNT,
        STAGE1_WORKERS=STAGE1_WORKERS,
        STAGE2_WORKERS=STAGE2_WORKERS,
        EVENT_BASE=EVENT_BASE,
    )


def run_test():
    A = torch.randn((M, N), dtype=torch.float32, device="cuda")
    B = torch.zeros((M, NUM_BLOCK_N), dtype=torch.float32, device="cuda")
    C = torch.zeros((M, 1), dtype=torch.float32, device="cuda")

    row_ready = torch.full(
        (NUM_BLOCK_M,),
        NUM_BLOCK_N * (EVENT_BASE + 1),
        dtype=torch.int32,
        device="cuda",
    )

    kernel = tir_kernel()

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        ex = tvm.compile(mod, target=target, tir_pipeline="tirx")
        ex(A, B, C, row_ready)

    C_ref = torch.sum(A, dim=1, keepdim=True)
    torch.testing.assert_close(C.cpu(), C_ref.cpu(), rtol=1e-4, atol=1e-4)

    print("two-stage TIRX reduce: pass")


if __name__ == "__main__":
    run_test()
