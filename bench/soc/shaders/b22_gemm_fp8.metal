// B-22: FP8 (e4m3) x FP8 (e4m3) -> FP32 matmul2d.  Separate file: the FP8
// element types (metal_fp8_e4m3_format) exist only from MSL 4.1 / OS 27, the
// library is compiled on its own so that an older compiler only loses this
// group (b22_gemm.metal keeps compiling as MSL 4.0).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims { int m, n, k, pad; };

template <int TM, int TN, int SG>
inline void gemmFp8(device uchar* a, device uchar* b, device float* c, constant Dims& d, uint2 tgid) {
    auto A = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(d.k, d.m));
    auto B = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(d.n, d.k));
    auto C = tensor<device float, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(d.n, d.m));
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = A.slice(0, int(tgid.y) * TM);
    auto mB = B.slice(int(tgid.x) * TN, 0);
    auto mC = C.slice(int(tgid.x) * TN, int(tgid.y) * TM);
    op.run(mA, mB, mC);
}
#define GEMM_FP8(NAME, TM, TN, SG)                                                                                \
    kernel void NAME(device uchar* a [[buffer(0)]], device uchar* b [[buffer(1)]], device float* c [[buffer(2)]], \
                     constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {              \
        gemmFp8<TM, TN, SG>(a, b, c, d, tgid);                                                                    \
    }
GEMM_FP8(fp8_64x32_s4, 64, 32, 4)
GEMM_FP8(fp8_64x64_s4, 64, 64, 4)
GEMM_FP8(fp8_128x64_s4, 128, 64, 4)
