// B-22: GEMM through MetalPerformancePrimitives tensor ops (matmul2d = Neural
// Accelerator on Apple10) for FP16/BF16 -> FP32 and INT8 -> INT32 (and INT8 x
// INT4 -> INT32), swept over the tile descriptor (TM x TN) and the execution
// scope (execution_simdgroups<SG>); simdgroup_matrix (shader ALUs) baseline.
//
// C[M x N] = A[M x K] * B[K x N], all row-major; tensors see extent(0) as the
// fastest index, so A is (K, M), B is (N, K), C is (N, M).  One threadgroup of
// 32*SG threads computes one TM x TN tile of C over the whole K.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims { int m, n, k, pad; };

template <typename TA, typename TB, typename TC, int TM, int TN, int SG>
inline void gemmTile(device TA* a, device TB* b, device TC* c, constant Dims& d, uint2 tgid) {
    auto A = tensor<device TA, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(d.k, d.m));
    auto B = tensor<device TB, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(d.n, d.k));
    auto C = tensor<device TC, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(d.n, d.m));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = A.slice(0, int(tgid.y) * TM);
    auto mB = B.slice(int(tgid.x) * TN, 0);
    auto mC = C.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(mA, mB, mC);
}

#define GEMM(NAME, TA, TB, TC, TM, TN, SG)                                                                        \
    kernel void NAME(device TA* a [[buffer(0)]], device TB* b [[buffer(1)]], device TC* c [[buffer(2)]],          \
                     constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {              \
        gemmTile<TA, TB, TC, TM, TN, SG>(a, b, c, d, tgid);                                                       \
    }

#define GEMM_TILES(PFX, TA, TB, TC)                                                                               \
    GEMM(PFX##_32x32_s1, TA, TB, TC, 32, 32, 1)  GEMM(PFX##_32x32_s2, TA, TB, TC, 32, 32, 2)                     \
    GEMM(PFX##_32x32_s4, TA, TB, TC, 32, 32, 4)  GEMM(PFX##_32x32_s8, TA, TB, TC, 32, 32, 8)                     \
    GEMM(PFX##_64x32_s1, TA, TB, TC, 64, 32, 1)  GEMM(PFX##_64x32_s2, TA, TB, TC, 64, 32, 2)                     \
    GEMM(PFX##_64x32_s4, TA, TB, TC, 64, 32, 4)  GEMM(PFX##_64x32_s8, TA, TB, TC, 64, 32, 8)                     \
    GEMM(PFX##_64x64_s1, TA, TB, TC, 64, 64, 1)  GEMM(PFX##_64x64_s2, TA, TB, TC, 64, 64, 2)                     \
    GEMM(PFX##_64x64_s4, TA, TB, TC, 64, 64, 4)  GEMM(PFX##_64x64_s8, TA, TB, TC, 64, 64, 8)                     \
    GEMM(PFX##_128x64_s1, TA, TB, TC, 128, 64, 1) GEMM(PFX##_128x64_s2, TA, TB, TC, 128, 64, 2)                  \
    GEMM(PFX##_128x64_s4, TA, TB, TC, 128, 64, 4) GEMM(PFX##_128x64_s8, TA, TB, TC, 128, 64, 8)

GEMM_TILES(f16, half, half, float)
GEMM_TILES(bf16, bfloat, bfloat, float)
GEMM_TILES(i8, int8_t, int8_t, int32_t)

// INT8 (left) x INT4 (right, two elements per byte, element 2j in the low
// nibble) -> INT32.  The tensor's data handle is the packed byte pointer.
template <int TM, int TN, int SG>
inline void gemmI4(device int8_t* a, device uchar* b, device int32_t* c, constant Dims& d, uint2 tgid) {
    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(d.k, d.m));
    auto B = tensor<device int4b_format, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(d.n, d.k));
    auto C = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(d.n, d.m));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = A.slice(0, int(tgid.y) * TM);
    auto mB = B.slice(int(tgid.x) * TN, 0);
    auto mC = C.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(mA, mB, mC);
}
#define GEMM_I4(NAME, TM, TN, SG)                                                                                 \
    kernel void NAME(device int8_t* a [[buffer(0)]], device uchar* b [[buffer(1)]], device int32_t* c [[buffer(2)]], \
                     constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {              \
        gemmI4<TM, TN, SG>(a, b, c, d, tgid);                                                                     \
    }
GEMM_I4(i4_64x32_s4, 64, 32, 4)
GEMM_I4(i4_64x64_s4, 64, 64, 4)
GEMM_I4(i4_128x64_s4, 128, 64, 4)

// Baseline on the shader ALUs: simdgroup_matrix 8x8 FP16 GEMM, each simdgroup
// computes a 32x32 block of C (4x4 matrices), 4 simdgroups per threadgroup =
// a 64x64 block (operands straight from device memory).
kernel void simd_f16(device half* a [[buffer(0)]], device half* b [[buffer(1)]], device float* c [[buffer(2)]],
                     constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    const int row0 = int(tgid.y) * 64 + int(sg / 2) * 32;
    const int col0 = int(tgid.x) * 64 + int(sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (int k = 0; k < d.k; k += 8) {
        simdgroup_half8x8 ma[4], mb[4];
        for (int i = 0; i < 4; ++i) simdgroup_load(ma[i], a + (row0 + i * 8) * d.k + k, d.k);
        for (int j = 0; j < 4; ++j) simdgroup_load(mb[j], b + k * d.n + col0 + j * 8, d.n);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], c + (row0 + i * 8) * d.n + col0 + j * 8, d.n);
}
