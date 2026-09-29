// overlay.metal -- F4.7 debug overlays: overdraw, light count, tile cost and
// the composite that draws the heatmap over the frame.
//
// Bindings (Metal 4 argument tables):
//   scene draws (overlay_vs / overdraw_fs / lightcount_fs) use the forward
//   pass's table: buffer(0) FrameConstants, buffer(1) GPUVertex[], buffer(2)
//   GPUInstance[], buffer(4) GPULight[].
//   compute overlay_tilecost: buffer(0) OverlayConstants, texture(0) overdraw,
//   texture(1) light count, texture(2) tile costs (written).
//   composite: buffer(0) OverlayConstants, texture(0) the value texture.
//
// The palette, scales and legend geometry live in diagnostics/overlay_math.h
// (shared with the CPU reference and its tests).

#include <metal_stdlib>
#include "renderer/gpu_types.h"
#include "diagnostics/overlay_math.h"

using namespace metal;
using namespace phosphor;

struct OverlayVertexOut {
    float4 position [[position]];
    float3 worldPos;
};

static float4x4 loadMatrix(constant float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

static float4x4 loadMatrix(const device float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

// Same transform as forward_vs (positions must match it exactly).
vertex OverlayVertexOut overlay_vs(uint vertexId                       [[vertex_id]],
                                   uint instanceId                     [[instance_id]],
                                   constant FrameConstants& frame      [[buffer(0)]],
                                   const device GPUVertex* vertices    [[buffer(1)]],
                                   const device GPUInstance* instances [[buffer(2)]])
{
    const device GPUVertex& v    = vertices[vertexId];
    const device GPUInstance& gi = instances[instanceId];
    const float4 world = loadMatrix(gi.modelMatrix) * float4(v.px, v.py, v.pz, 1.0);
    OverlayVertexOut out;
    out.position = loadMatrix(frame.viewProjection) * world;
    out.worldPos = world.xyz;
    return out;
}

// Overdraw: one more fragment.  The attachment is read back through
// programmable blending ([[color(0)]], tile memory), so no blend state is
// needed; there is no depth test, so every rasterised fragment counts.
fragment half4 overdraw_fs(half4 previous [[color(0)]]) {
    return previous + half4(1.0h, 0.0h, 0.0h, 0.0h);
}

// Light count: lights that contribute at the visible surface, i.e. whose
// attenuation is > 0 by the same formulas as forward_fs (window to zero at
// `range`, spot cone).  A directional light counts as 1.
static float distanceAttenuation(float dist, float range) {
    const float r = dist / max(range, 1e-3);
    const float window = saturate(1.0 - r * r * r * r);
    return window * window / max(dist * dist, 1e-4);
}

fragment half4 lightcount_fs(OverlayVertexOut in                    [[stage_in]],
                             constant FrameConstants& frame         [[buffer(0)]],
                             const device GPULight* lights          [[buffer(4)]])
{
    uint count = 0;
    for (uint i = 0; i < frame.lightCount; ++i) {
        const device GPULight& light = lights[i];
        if (light.type == LIGHT_DIRECTIONAL) {
            ++count;
            continue;
        }
        const float3 toLight = float3(light.position[0], light.position[1], light.position[2]) - in.worldPos;
        const float dist = length(toLight);
        const float3 L = toLight / max(dist, 1e-4);
        float attenuation = distanceAttenuation(dist, light.range);
        if (light.type == LIGHT_SPOT) {
            const float3 spotDir = normalize(float3(light.direction[0], light.direction[1], light.direction[2]));
            attenuation *= smoothstep(cos(light.outerCone), cos(light.innerCone), dot(-L, spotDir));
        }
        if (attenuation > 0.0) ++count;
    }
    return half4(half(count), 0.0h, 0.0h, 0.0h);
}

// Tile cost: one 32x32 threadgroup per tile (32 SIMD groups of 32 threads, the
// Apple GPU SIMD width); average over the tile's in-bounds pixels of
// overdraw x (1 + lights).  CPU twin: overlay::tileCostReference.
kernel void overlay_tilecost(constant OverlayConstants& c                    [[buffer(0)]],
                             texture2d<float, access::read> overdraw          [[texture(0)]],
                             texture2d<float, access::read> lights            [[texture(1)]],
                             texture2d<float, access::write> tiles            [[texture(2)]],
                             uint2 tile                                       [[threadgroup_position_in_grid]],
                             uint2 local                                      [[thread_position_in_threadgroup]],
                             uint lane                                        [[thread_index_in_simdgroup]],
                             uint simdIndex                                   [[simdgroup_index_in_threadgroup]])
{
    const uint2 p = tile * overlay::TILE_SIZE + local;
    float cost  = 0.0;
    float count = 0.0;
    if (p.x < c.width && p.y < c.height) {
        cost  = overdraw.read(p).x * (1.0 + max(lights.read(p).x, 0.0));
        count = 1.0;
    }
    threadgroup float partial[64];
    const float simdCost  = simd_sum(cost);
    const float simdCount = simd_sum(count);
    if (lane == 0) {
        partial[simdIndex]      = simdCost;
        partial[32 + simdIndex] = simdCount;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdIndex == 0) {
        const float totalCost  = simd_sum(partial[lane]);
        const float totalCount = simd_sum(partial[32 + lane]);
        if (lane == 0) {
            tiles.write(float4(totalCount > 0.0 ? totalCost / totalCount : 0.0), tile);
        }
    }
}

// --- composite ------------------------------------------------------------------

vertex float4 overlay_composite_vs(uint vid [[vertex_id]]) {
    const float2 p = float2((vid << 1) & 2, vid & 2);
    return float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
}

// The palette is defined in sRGB; the target is an sRGB format (the hardware
// encodes on write), so the colour is decoded here (gamma 2.2 approximation).
static half4 heat(float t, float alpha) {
    const float3 srgb = float3(overlay::heatChannel(t, 0), overlay::heatChannel(t, 1), overlay::heatChannel(t, 2));
    return half4(half3(pow(srgb, float3(2.2))), half(alpha));
}

fragment half4 overlay_composite_fs(float4 position                       [[position]],
                                    constant OverlayConstants& c          [[buffer(0)]],
                                    texture2d<float, access::read> values [[texture(0)]])
{
    const uint2 p = uint2(position.xy);

    // Legend strip, bottom-left: the palette from 0 to the scale's maximum,
    // with a 1 pixel dark border.
    const uint legendTop = c.height - overlay::LEGEND_MARGIN - overlay::LEGEND_HEIGHT;
    if (p.x + 1 >= overlay::LEGEND_X && p.x <= overlay::LEGEND_X + overlay::LEGEND_WIDTH &&
        p.y + 1 >= legendTop && p.y <= legendTop + overlay::LEGEND_HEIGHT) {
        const bool inside = p.x >= overlay::LEGEND_X && p.x < overlay::LEGEND_X + overlay::LEGEND_WIDTH &&
                            p.y >= legendTop && p.y < legendTop + overlay::LEGEND_HEIGHT;
        if (!inside) return half4(0.0h, 0.0h, 0.0h, 1.0h);
        return heat(float(p.x - overlay::LEGEND_X) / float(overlay::LEGEND_WIDTH - 1), 1.0);
    }

    const uint2 texel = c.kind == overlay::KIND_TILE_COST ? p / overlay::TILE_SIZE : p;
    const float value = values.read(texel).x;
    if (!overlay::hasData(c.kind, value)) {
        return half4(0.0h);
    }
    return heat(overlay::normalize(c.kind, value), c.alpha);
}
