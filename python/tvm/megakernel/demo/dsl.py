import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import KernelSpec, TileImpl

# Configeration parameters
M = 1024
N = 1024

BLOCK_M = 64
BLOCK_N = 64

NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N


class Stage1ReduceTile(TileImpl):

    def __init__(self, A, B, *, BLOCK_M, BLOCK_N):
        super().__init__()
        self.A = A
        self.B = B
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N

        self.A_smem = None
        self.P_smem = None

    def init(self, smem_manager):
        self.A_smem = smem_manager.alloc(
            (self.BLOCK_M, self.BLOCK_N),
            "float32",
        )
        self.P_smem = smem_manager.alloc(
            (self.BLOCK_M, 1),
            "float32",
        )

    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(
            self.A_smem,
            self.A[
                m_idx * self.BLOCK_M : (m_idx + 1) * self.BLOCK_M,
                n_idx * self.BLOCK_N : (n_idx + 1) * self.BLOCK_N,
            ],
        )
        Tx.sum(
            self.P_smem,
            self.A_smem,
        )
        Tx.copy(
            self.B[
                m_idx * self.BLOCK_M : (m_idx + 1) * self.BLOCK_M,
                n_idx,
            ],
            self.P_smem,
        )


class Stage2ReduceTile(TileImpl):

    def __init__(self, B, C, *, BLOCK_M, NUM_BLOCK_N):
        super().__init__()
        self.B = B
        self.C = C
        self.BLOCK_M = BLOCK_M
        self.NUM_BLOCK_N = NUM_BLOCK_N

        self.B_smem = None
        self.C_smem = None

    def init(self, smem_manager):
        self.B_smem = smem_manager.alloc(
            (self.BLOCK_M, self.NUM_BLOCK_N),
            "float32",
        )
        self.C_smem = smem_manager.alloc(
            (self.BLOCK_M, 1),
            "float32",
        )

    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(
            self.B_smem,
            self.B[
                m_idx * self.BLOCK_M : (m_idx + 1) * self.BLOCK_M,
                :,
            ],
        )
        Tx.sum(
            self.C_smem,
            self.B_smem,
        )
        Tx.copy(
            self.C[
                m_idx * self.BLOCK_M : (m_idx + 1) * self.BLOCK_M,
                0,
            ],
            self.C_smem,
        )


kernel = KernelSpec("two_stage_reduce")


# ============================================================
# Tensors
# ============================================================

A = kernel.tensor(
    "A",
    shape=(M, N),
    dtype="float32",
)

B = kernel.tensor(
    "B",
    shape=(M, NUM_BLOCK_N),
    dtype="float32",
)

C = kernel.tensor(
    "C",
    shape=(M, 1),
    dtype="float32",
)


# ============================================================
# Event
# ============================================================

row_ready = kernel.event(
    "row_ready",
    shape=(NUM_BLOCK_M,),
    init=NUM_BLOCK_N,
)


# ============================================================
# Tile 1: Stage1 partial row reduction
# ============================================================
#
# tile instance:
#   stage1(m_idx, n_idx, 0)
#
# computes:
#   A[m-block, n-block] -> B[m-block, n_idx]
#
# notifies:
#   row_ready[m_idx]
#

stage1 = (
    kernel.tile(
        name="stage1_partial_reduce",
        impl=Stage1ReduceTile(
            A,
            B,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        ),
        tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
    )
    .read(
        A,
    )
    .write(
        B,
    )
    .notify(
        row_ready,
        coord_map=lambda m, n, k: (m,),
    )
)


# ============================================================
# Tile 2: Stage2 final row reduction
# ============================================================
#
# tile instance:
#   stage2(m_idx, 0, 0)
#
# waits:
#   row_ready[m_idx] has received NUM_BLOCK_N notifications
#
# computes:
#   B[m-block, :] -> C[m-block, 0]
#

stage2 = (
    kernel.tile(
        name="stage2_final_reduce",
        impl=Stage2ReduceTile(
            B,
            C,
            BLOCK_M=BLOCK_M,
            NUM_BLOCK_N=NUM_BLOCK_N,
        ),
        tile_num=(NUM_BLOCK_M, 1, 1),
    )
    .read(
        B,
    )
    .write(
        C,
    )
    .wait(
        row_ready,
        coord_map=lambda m, n, k: (m,),
        expected=NUM_BLOCK_N,
    )
)


kernel.validate()
kernel.lower()
