// F15.1 (ticket 17/18, RT tier): temporal accumulation and edge-aware spatial prefilter of the subsurface walk signal, so that the walk meets its bound in the
// PRODUCT output (default walks per pixel, one frame) and not only as a converged estimator.
//
// Inputs of a frame: the walk's noisy term (weight x the mean radiance of the walks, added to the base colour) and its gain (the composite's ambient / GI factor
// of a walk pixel).  Per pixel: (1) spatial prefilter of the CURRENT frame's term and gain over a (2 r + 1)^2 window of pixels with the same material code, a
// depth within the tolerance and a normal within the cone of the centre (the window of the reference's own pooling: the response of a medium is smooth at the
// scale of its mean free path); (2) reprojection of the history through the engine's motion vectors (current -> previous, pixels) with rejection by depth and
// material code, and by the history counter; (3) exponential moving average with the weight max(1 / (count + 1), 1 / maxCount) (an exact running mean while the
// history is young, a bounded memory afterwards).  The history stores the accumulated term / gain in rgb and, packed in alpha, the half-precision view depth and
// the count.  A reset (camera cut, resize, scene change, a light that moved) starts a new history.  The result is base + accumulated term; the accumulated gain
// replaces the walk's gain target.
#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;
using namespace phosphor;

inline float4 saUnpack(float4 history, thread float& depth, thread uint& count) {
    const uint bits = as_type<uint>(history.w);
    count = bits & 0xFFu;
    depth = float(as_type<half>(ushort(bits >> 8)));
    return float4(history.xyz, 0.0f);
}
inline float saPack(float depth, uint count) {
    return as_type<float>((uint(as_type<ushort>(half(min(depth, 60000.0f)))) << 8) | min(count, 255u));
}

kernel void subsurface_accumulate(constant GPUSubsurfaceAccumParams& p [[buffer(0)]],
                                  texture2d<float, access::read> color [[texture(0)]], texture2d<float, access::read> term [[texture(1)]],
                                  texture2d<float, access::read> gain [[texture(2)]], texture2d<float, access::read> sss [[texture(3)]],
                                  depth2d<float, access::read> depth [[texture(4)]], texture2d<float, access::read> normalRoughness [[texture(5)]],
                                  texture2d<float, access::read> motion [[texture(6)]], texture2d<float, access::read> termHistory [[texture(7)]],
                                  texture2d<float, access::read> gainHistory [[texture(8)]], texture2d<float, access::write> result [[texture(9)]],
                                  texture2d<float, access::write> termOut [[texture(10)]], texture2d<float, access::write> gainOut [[texture(11)]],
                                  texture2d<float, access::write> gainResult [[texture(12)]], uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight) return;
    const float4 base = color.read(pixel);
    const float4 own = sss.read(pixel);
    const uint material = uint(own.w);
    if (material == 0u || pixel.x >= p.width || pixel.y >= p.height) {
        result.write(base, pixel);
        termOut.write(float4(0.0f), pixel);
        gainOut.write(float4(0.0f), pixel);
        gainResult.write(float4(0.0f), pixel);
        return;
    }
    const float d0 = depth.read(pixel);
    const float z0 = p.nearPlane / max(d0, 1e-12f);
    const float3 n0 = normalRoughness.read(pixel).xyz;
    // (1) spatial prefilter of the current frame
    float3 termSum = float3(0.0f), gainSum = float3(0.0f);
    float count = 0.0f;
    const int r = int(p.window);
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            const int2 q = int2(pixel) + int2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= int(p.width) || q.y >= int(p.height)) continue;
            const uint2 neighbour = uint2(q);
            if (uint(sss.read(neighbour).w) != material) continue;
            const float dn = depth.read(neighbour);
            if (!(dn > 0.0f) || abs(p.nearPlane / dn - z0) > p.depthTolerance * z0) continue;
            if (dot(normalRoughness.read(neighbour).xyz, n0) < p.normalCosine) continue;
            termSum += term.read(neighbour).rgb;
            gainSum += gain.read(neighbour).rgb;
            count += 1.0f;
        }
    float3 filteredTerm = count > 0.0f ? termSum / count : term.read(pixel).rgb;
    float3 filteredGain = count > 0.0f ? gainSum / count : gain.read(pixel).rgb;
    // (2) history
    uint historyCount = 0u;
    float3 accTerm = filteredTerm, accGain = filteredGain;
    if ((p.flags & SSS_ACCUM_RESET) == 0u) {
        const bool noReject = (p.flags & SSS_ACCUM_NO_REJECT) != 0u;
        const float2 previous = float2(pixel) + 0.5f + (noReject ? float2(0.0f) : motion.read(pixel).xy);
        const int2 q = int2(floor(previous));
        if (q.x >= 0 && q.y >= 0 && q.x < int(p.width) && q.y < int(p.height)) {
            float previousDepth;
            uint c;
            const float4 th = termHistory.read(uint2(q));
            const float4 t = saUnpack(th, previousDepth, c);
            if (c > 0u && (noReject || abs(previousDepth - z0) <= p.depthTolerance * z0)) {
                const float4 g = saUnpack(gainHistory.read(uint2(q)), previousDepth, c);
                historyCount = c;
                const float alpha = max(1.0f / float(historyCount + 1u), 1.0f / float(max(p.maxCount, 1u)));
                accTerm = mix(t.xyz, filteredTerm, alpha);
                accGain = mix(g.xyz, filteredGain, alpha);
            }
        }
    }
    const uint newCount = min(historyCount + 1u, min(p.maxCount, 255u));
    const float packed = saPack(z0, newCount);
    termOut.write(float4(accTerm, packed), pixel);
    gainOut.write(float4(accGain, packed), pixel);
    result.write(float4(base.rgb + accTerm, base.a), pixel);
    gainResult.write(float4(accGain, gain.read(pixel).a), pixel);
}
