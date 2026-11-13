// F14-K2 compiles the engine's brute-force DI shading kernel itself
// (light_cluster_shade_rt) with the engine RT library, never a copy.
#include "../../../shaders/rt_scene.metal"

#include "../../../shaders/rt_intersections.metal"

#include "../../../shaders/light_cluster.metal"

// Reduction (F14-K2 with F14_KERNEL=k_cfg F14_CFG=bits): a copy of the engine
// kernel whose constructs sit behind one function-constant bit each, so the
// compiler drops the disabled code.  The default run uses the engine kernel.

static bool svSimpleVisible(GPUDISurface surface, DISample sample, instance_acceleration_structure tlas,
                            intersection_function_table<triangle_data, instancing> ift, const device GPUInstance* instances, uint slotCount) {
    const float3 origin = diVec(surface.position) + float3(0, 1e-3f, 0);
    const float3 to = sample.position - origin;
    const float distance = length(to);
    GPURtRay ray{};
    ray.ox = origin.x; ray.oy = origin.y; ray.oz = origin.z;
    const float3 d = to / distance;
    ray.dx = d.x; ray.dy = d.y; ray.dz = d.z;
    ray.tmin = 0; ray.tmax = distance * 0.999f;
    ray.mask = RT_MASK_SHADOW; ray.type = RT_PROBE_SHADOW; ray.coneWidth = 0;
    RtPayload payload{};
    const GPURtHit hit = rtTrace(ray, tlas, ift, instances, slotCount, payload);
    return hit.hit == 0u && hit.t != -2.0f;
}

// k_cfg: the engine kernel with each construct behind a function constant
// bit (F14_CFG): 1 textured sample, 2 BRDF, 4 noise, 8 atomics, 16 texture
// output, 32 engine endpoint ray, 64 any(contribution > 0) short cut, 128 pdf.
constant uint kCfg [[function_constant(0)]];
kernel void k_cfg(constant GPUDIParams& p [[buffer(0)]],
                  const device GPUDISurface* surfaces [[buffer(1)]],
                  const device GPUSampledLight* lights [[buffer(2)]],
                  const device uint* ranks [[buffer(6)]],
                  const device GPUInstance* instances [[buffer(7)]],
                  instance_acceleration_structure tlas [[buffer(8)]],
                  intersection_function_table<triangle_data, instancing> ift [[buffer(9)]],
                  device float4* outBuffer [[buffer(10)]],
                  const device GPUEmissiveSurface* emitters [[buffer(12)]],
                  const device GPUMaterial* materials [[buffer(13)]],
                  const device DITextureHandle* textures [[buffer(14)]],
                  device atomic_uint* signalErrors [[buffer(15)]],
                  texture2d<float, access::write> localDirect [[texture(0)]],
                  uint tid [[thread_position_in_grid]]) {
    if (tid >= p.width * p.height) return;
    const GPUDISurface surface = surfaces[tid];
    const uint2 pixel(tid % p.width, tid / p.width);
    float3 value(0);
    if (surface.valid) {
        for (uint i = 0; i < p.lightCount; ++i) {
            if (lights[i].type == LIGHT_DIRECTIONAL) continue;
            float2 uv(0.37f, 0.61f);
            if (kCfg & 4u) uv = float2(fract(diRandom(pixel, 40u, p, ranks) + diWhite(0, 0, p.frameIndex, i * 2u, p.stbnSeed)),
                                       fract(diRandom(pixel, 41u, p, ranks) + diWhite(0, 0, p.frameIndex, i * 2u + 1u, p.stbnSeed)));
            if ((kCfg & 8u) && !diLightFinite(lights[i])) atomic_fetch_add_explicit(signalErrors + 3, 1u, memory_order_relaxed);
            const DISample sample = (kCfg & 1u) ? diSampleTexturedLight(lights[i], i, uv, diVec(surface.position), emitters, materials, textures)
                                                : diSampleLight(lights[i], uv, diVec(surface.position));
            const float3 contribution = (kCfg & 2u) ? diBRDFWeighted(surface, sample, diEmitterSpecularWeight(p, lights[i], surface))
                                                    : (sample.valid ? float3(1.0f, 0.5f, 0.25f) : float3(0));
            if (kCfg & 8u) diRecordNonfinite(contribution, signalErrors, 0u);
            const bool shortcut = (kCfg & 64u) && !any(contribution > 0);
            if (sample.valid && sample.pdfArea > 0 && (shortcut ||
                ((kCfg & 32u) ? diEndpointVisible(surface, sample, tlas, ift, instances, p.slotCount)
                              : svSimpleVisible(surface, sample, tlas, ift, instances, p.slotCount)))) {
                value += (kCfg & 128u) ? contribution / sample.pdfArea : contribution;
                if (kCfg & 8u) diRecordNonfinite(value, signalErrors, 2u);
            }
        }
    }
    if (kCfg & 16u) localDirect.write(float4(value, 1), pixel);
    else outBuffer[tid] = float4(value, 1);
}
