#include <metal_stdlib>
using namespace metal;

// Float storage readback of LINEAR pre-exposure HDR. No sampler, exposure,
// gamma, tone mapping, upscaling or UI is involved. read() converts RGBA16Float
// to float4 when needed; RGBA32Float values are copied directly. Buffer0 is
// exactly host std::array<u32,2> {activeWidth,activeHeight}; buffer1 is float4.
kernel void capture_linear_hdr(constant uint2& extent [[buffer(0)]],
                               device float4* output [[buffer(1)]],
                               texture2d<float, access::read> linearHdr [[texture(0)]],
                               uint2 pixel [[thread_position_in_grid]]) {
    if (any(pixel >= extent)) return;
    output[pixel.y * extent.x + pixel.x] = linearHdr.read(pixel);
}
