#pragma once
#include "restir_common.h"
#define PHOSPHOR_RT_NO_ALPHA_FUNCTION
#include "rt_common.h"

// Shared F10.3 visibility for a selected endpoint. The caller supplies an IFT
// created from the caller's own PSO. The generic alpha intersection function
// is linked from rt_scene.metal; its LOD is fixed at 0 for shadow rays.
// F15.1 (ticket 14): the result is the RGB TRANSMITTANCE of the straight segment to the endpoint: 0 when an opaque or alpha-masked caster (or an
// invalid ray) blocks it, 1 when nothing is in the way, the product of the crossed transmissive casters' factors otherwise (ADR-0004).
// F15 ticket 24: `Payload` = RtPayloadLite in a pipeline that links rt_alpha_lite (a scene without transmissive materials: transmittance 1, exact).
template <typename Payload = RtPayload>
inline float3 diEndpointTransmittance(GPUDISurface surface, DISample sample,
                               instance_acceleration_structure tlas,
                               intersection_function_table<triangle_data, instancing> ift,
                               const device GPUInstance* instances, uint slotCount) {
    const float3 geometric = diVec(surface.geometricNormal);
    if (!sample.valid || !all(isfinite(geometric)) || !(dot(geometric, geometric) > 0)) return float3(0);
    float3 n = normalize(geometric);
    if (dot(n, sample.wi) < 0) n = -n;
    const float3 origin = rtOffsetRay(diVec(surface.position), n);
    float3 endpoint = sample.position;
    if (!sample.delta) {
        float3 emitterNormal = sample.normal;
        if (dot(emitterNormal, origin - endpoint) < 0) emitterNormal = -emitterNormal;
        endpoint = rtOffsetRay(endpoint, emitterNormal);
    }
    const float3 toLight = endpoint - origin;
    const float distance = length(toLight);
    if (!all(isfinite(origin)) || !isfinite(distance) || !(distance > 0)) return float3(0);
    const float3 direction = toLight / distance;
    GPURtRay ray{};
    ray.ox = origin.x; ray.oy = origin.y; ray.oz = origin.z;
    ray.dx = direction.x; ray.dy = direction.y; ray.dz = direction.z;
    ray.tmin = 0; ray.tmax = nextafter(distance, 0.0f);
    ray.mask = RT_MASK_SHADOW; ray.type = RT_PROBE_SHADOW_TRANSMITTED; ray.coneWidth = 0;
    Payload payload{};
    const GPURtHit hit = rtTrace(ray, tlas, ift, instances, slotCount, payload);
    return (hit.hit == 0u && hit.t != -2.0f) ? rtShadowTransmittance(payload) : float3(0);
}
