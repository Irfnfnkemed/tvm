from tvm.script import tir as T

from tvm.tirx.megakernel.utils.base import Tile, KernelConfig


ld_reduce_8xfp16 = """
__forceinline__ __device__ void ld_reduce_8_fp16(void* src_addr, void* dst_addr) {
    int4* source = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, src_addr);
    int4* dest = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, dst_addr);
    constexpr int UNROLL = 1;
    union {
        uint32_t u4[4 * UNROLL];
        uint16_t u2[8 * UNROLL];
    };
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        asm("multimem.ld_reduce.acquire.sys.global.add.acc::f32.v8.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=h"(u2[8 * u]), "=h"(u2[8 * u + 1]), "=h"(u2[8 * u + 2]), "=h"(u2[8 * u + 3]), "=h"(u2[8 * u + 4]), "=h"(u2[8 * u + 5]), "=h"(u2[8 * u + 6]), "=h"(u2[8 * u + 7])
            : "l"(source));
    }
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
        asm("multimem.st.release.sys.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dest), "r"(u4[4 * u]), "r"(u4[4 * u + 1]), "r"(u4[4 * u + 2]), "r"(u4[4 * u + 3]): "memory");
    }
}
"""

class AllreduceTile(Tile):
    M_TILE = 16
    N_TILE = 128
    def __init__(self, world_size):
        super().__init__()
        self.world_size = world_size

    @T.macro
    def run(self, m_idx, n_idx, k_idx, input, output):
        with T.cta():
            tid = T.thread_id([KernelConfig.NUM_THREADS], parent="cta")
            rank = T.nvshmem.my_pe()
            # restrict tile size so that this will not be iterated over
            if tid < self.M_TILE * self.N_TILE // 8 and (m_idx * self.M_TILE +  (tid // (self.N_TILE // 8))) < input.shape[0]:
                m_start = T.meta_var(m_idx * self.M_TILE +  (tid // (self.N_TILE // 8)))
                n_start = T.meta_var(n_idx * self.N_TILE * self.world_size + rank * self.N_TILE + tid % (self.N_TILE // 8) * 8)
                T.cuda.func_call(
                    "ld_reduce_8_fp16",
                    input.ptr_to([m_start, n_start]),
                    output.ptr_to([m_start, n_start]),
                    source_code=ld_reduce_8xfp16,
                )
