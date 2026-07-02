import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import KernelSpec, TileImpl


# Configuration parameters
M = 1024
N = 1024

BLOCK_M = 64
BLOCK_N = 64

NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N


# ============================================================
# Lower DSL layer: tile implementations
# ============================================================


class Stage1ReduceTile(TileImpl):
    """Reduce one A[m-block, n-block] tile into B[m-block, n]."""

    def __init__(self, A, B, *, block_m, block_n):
        super().__init__()
        self.A = A
        self.B = B
        self.block_m = block_m
        self.block_n = block_n

        self.A_smem = None
        self.P_smem = None

    def device_init(self):
        # The concrete allocation API is selected by the later lowering pass.
        self.A_smem = T.alloc_buffer((self.block_m, self.block_n), "float32", scope="shared")
        self.P_smem = T.alloc_buffer((self.block_m, 1), "float32", scope="shared")

    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(
            self.A_smem,
            self.A[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                n_idx * self.block_n : (n_idx + 1) * self.block_n,
            ],
        )
        Tx.sum(
            self.P_smem,
            self.A_smem,
        )
        Tx.copy(
            self.B[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                n_idx,
            ],
            self.P_smem,
        )


class Stage2ReduceTile(TileImpl):
    """Reduce all B n-block results for one m-block into C[m-block, 0]."""

    def __init__(self, B, C, *, block_m, num_block_n):
        super().__init__()
        self.B = B
        self.C = C
        self.block_m = block_m
        self.num_block_n = num_block_n

        self.B_smem = None
        self.C_smem = None

    def device_init(self):
        # The concrete allocation API is selected by the later lowering pass.
        self.B_smem = T.alloc_buffer((self.block_m, self.num_block_n), "float32", scope="shared")
        self.C_smem = T.alloc_buffer((self.block_m, 1), "float32", scope="shared")

    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(
            self.B_smem,
            self.B[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                :,
            ],
        )
        Tx.sum(
            self.C_smem,
            self.B_smem,
        )
        Tx.copy(
            self.C[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                0,
            ],
            self.C_smem,
        )


# ============================================================
# Upper DSL layer: megakernel spec
# ============================================================


kernel = KernelSpec(
    "two_stage_reduce",
    attrs={
        "source": "B = reduce_each_n_block(A); C = reduce_all_n_blocks(B)",
    },
)

A = kernel.input(
    "A",
    shape=(M, N),
    dtype="float32",
)

B = kernel.intermediate(
    "B",
    shape=(M, NUM_BLOCK_N),
    dtype="float32",
)

C = kernel.output(
    "C",
    shape=(M, 1),
    dtype="float32",
)

row_ready = kernel.event(
    "row_ready",
    shape=(NUM_BLOCK_M,),
    init_count=NUM_BLOCK_N,
    attrs={
        "meaning": "all n-block reductions for one m-block have written B",
    },
)

stage1 = (
    kernel.tile(
        name="stage1_partial_reduce",
        impl=Stage1ReduceTile(
            A,
            B,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
        ),
        tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
        attrs={
            "source_stage": "B = reduce_each_n_block(A)",
        },
    )
    .read(A)
    .write(B)
    .notify(row_ready, coord_map=lambda m, n, k: (m,))
)

stage2 = (
    kernel.tile(
        name="stage2_final_reduce",
        impl=Stage2ReduceTile(
            B,
            C,
            block_m=BLOCK_M,
            num_block_n=NUM_BLOCK_N,
        ),
        tile_num=(NUM_BLOCK_M, 1, 1),
        attrs={
            "source_stage": "C = reduce_all_n_blocks(B)",
        },
    )
    .read(B)
    .write(C)
    .wait(row_ready, coord_map=lambda m, n, k: (m,))
)
