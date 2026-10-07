#include <metal_stdlib>
#include "restir_common.h"
#include "light_visibility.h"
using namespace metal;
using namespace phosphor;

inline float3 emitterWorldPoint(GPUInstance instance, float3 point) {
    return float3(instance.modelMatrix[0], instance.modelMatrix[1], instance.modelMatrix[2]) * point.x +
           float3(instance.modelMatrix[4], instance.modelMatrix[5], instance.modelMatrix[6]) * point.y +
           float3(instance.modelMatrix[8], instance.modelMatrix[9], instance.modelMatrix[10]) * point.z +
           float3(instance.modelMatrix[12], instance.modelMatrix[13], instance.modelMatrix[14]);
}

// Source lights and local full-scene emitter metadata are loading/revision data.
// Destination is a per-slot sampled-light buffer. The graph runs this AFTER
// Scene transforms and BEFORE clusters/candidates; no camera-culling input.
// params0, full-scene emitter records1, scene instances2, materials3,
// source sampled lights4, destination sampled lights5. No allocation here.
kernel void light_emissive_update(constant GPUEmissiveUpdateParams& p [[buffer(0)]],
                                  const device GPUEmissiveSurface* records [[buffer(1)]],
                                  const device GPUInstance* instances [[buffer(2)]],
                                  const device GPUMaterial* materials [[buffer(3)]],
                                  const device GPUSampledLight* source [[buffer(4)]],
                                  device GPUSampledLight* destination [[buffer(5)]],
                                  const device GPUVertex* vertices [[buffer(6)]],
                                  uint tid [[thread_position_in_grid]]) {
    if (tid >= p.lightCount) return;
    GPUSampledLight light = source[tid];
    const GPUEmissiveSurface emitter = records[tid];
    if (emitter.valid) {
        bool valid = emitter.instanceSlot < p.slotCount && emitter.materialIndex < p.materialCount;
        GPUInstance instance{};
        if (valid) {
            instance = instances[emitter.instanceSlot];
            valid = (instance.flags & INSTANCE_FLAG_VALID) && instance.generation == emitter.instanceGeneration &&
                    instance.materialIndex == emitter.materialIndex;
        }
        if(emitter.geometryValid && (emitter.vertex0>=p.pad || emitter.vertex1>=p.pad || emitter.vertex2>=p.pad))valid=false;
        if (valid) {
            const GPUMaterial material = materials[emitter.materialIndex];
            const GPUVertex v0=emitter.geometryValid?vertices[emitter.vertex0]:GPUVertex{};
            const GPUVertex v1=emitter.geometryValid?vertices[emitter.vertex1]:GPUVertex{};
            const GPUVertex v2=emitter.geometryValid?vertices[emitter.vertex2]:GPUVertex{};
            const float3 a = emitterWorldPoint(instance, emitter.geometryValid?float3(v0.px,v0.py,v0.pz):float3(emitter.p0[0],emitter.p0[1],emitter.p0[2]));
            const float3 b = emitterWorldPoint(instance, emitter.geometryValid?float3(v1.px,v1.py,v1.pz):float3(emitter.p1[0],emitter.p1[1],emitter.p1[2]));
            const float3 c = emitterWorldPoint(instance, emitter.geometryValid?float3(v2.px,v2.py,v2.pz):float3(emitter.p2[0],emitter.p2[1],emitter.p2[2]));
            const float3 u = b - a, v = c - a;
            light.position[0] = a.x; light.position[1] = a.y; light.position[2] = a.z;
            light.axisU[0] = u.x; light.axisU[1] = u.y; light.axisU[2] = u.z;
            light.axisV[0] = v.x; light.axisV[1] = v.y; light.axisV[2] = v.z;
            light.emission[0] = material.emissive[0]; light.emission[1] = material.emissive[1]; light.emission[2] = material.emissive[2];
            light.flags = DI_LIGHT_TEXTURED_EMISSION |
                          ((material.flags & MATERIAL_FLAG_DOUBLE_SIDED) ? DI_LIGHT_TWO_SIDED : 0u) |
                          ((instance.flags & INSTANCE_FLAG_MIRRORED) ? DI_LIGHT_MIRRORED : 0u);
            light.type = DI_LIGHT_TRIANGLE;
            light.range = 0; // physically unbounded emissive triangle
        } else {
            // Removed/reincarnated/mismatched emitter: black, zero-area light.
            // Keep its stable slot until host structural revision rebuilds it.
            light.axisU[0] = light.axisU[1] = light.axisU[2] = 0;
            light.axisV[0] = light.axisV[1] = light.axisV[2] = 0;
            light.emission[0] = light.emission[1] = light.emission[2] = 0;
            light.flags = 0;
        }
    }
    destination[tid] = light;
}

inline float3 clusterViewPoint(constant GPULightClusterParams& p, float3 world) {
    const constant float* m = p.view;
    return float3(m[0], m[1], m[2]) * world.x + float3(m[4], m[5], m[6]) * world.y +
           float3(m[8], m[9], m[10]) * world.z + float3(m[12], m[13], m[14]);
}
inline bool clusterLightIntersects(constant GPULightClusterParams& p, uint index, GPUSampledLight light) {
    if (!(light.range > 0) || !isfinite(light.range)) return true; // unbounded light
    float extent = 0;
    if (light.type == DI_LIGHT_RECTANGLE || light.type == DI_LIGHT_TRIANGLE)
        extent = length(diVec(light.axisU)) + length(diVec(light.axisV));
    if (light.type == DI_LIGHT_DISK) extent = light.radius * (length(diVec(light.axisU)) + length(diVec(light.axisV)));
    if (light.type == DI_LIGHT_TUBE) extent = length(diVec(light.axisU)) + light.radius;
    const float radius = light.range + max(extent, 0.0f);
    const float3 center = clusterViewPoint(p, diVec(light.position));
    if (!all(isfinite(center)) || !isfinite(radius)) return true;
    const uint x = index % p.gridX, y = (index / p.gridX) % p.gridY, z = index / (p.gridX * p.gridY);
    const float z0 = p.nearPlane * pow(p.farPlane / p.nearPlane, float(z) / float(p.gridZ));
    const float z1 = p.nearPlane * pow(p.farPlane / p.nearPlane, float(z + 1u) / float(p.gridZ));
    const float depth = -center.z;
    if (depth + radius < z0 || depth - radius > z1) return false;
    const float left = (2.0f * float(x) / p.gridX - 1.0f) * p.tanHalfFovX;
    const float right = (2.0f * float(x + 1u) / p.gridX - 1.0f) * p.tanHalfFovX;
    const float bottom = (2.0f * float(y) / p.gridY - 1.0f) * p.tanHalfFovY;
    const float top = (2.0f * float(y + 1u) / p.gridY - 1.0f) * p.tanHalfFovY;
    return center.x - left * depth >= -radius * sqrt(1.0f + left * left) &&
           right * depth - center.x >= -radius * sqrt(1.0f + right * right) &&
           center.y - bottom * depth >= -radius * sqrt(1.0f + bottom * bottom) &&
           top * depth - center.y >= -radius * sqrt(1.0f + top * top);
}

// Baseline: one thread per frustum cell loops over local lights. No atomic
// truncation or ordering race. Overflow means the shading pass loops ALL
// lights, not the first capacity entries. Host validates products/buffer sizes.
// params0, local lights1, cells2, cell-major light indices3.
kernel void light_cluster_build(constant GPULightClusterParams& p [[buffer(0)]],
                                const device GPUSampledLight* lights [[buffer(1)]],
                                device GPULightCluster* cells [[buffer(2)]],
                                device uint* indices [[buffer(3)]],
                                uint cell [[thread_position_in_grid]]) {
    if (cell >= p.gridX * p.gridY * p.gridZ) return;
    GPULightCluster out{};
    for (uint i = 0; i < p.lightCount; ++i) {
        if (lights[i].type == LIGHT_DIRECTIONAL || !clusterLightIntersects(p, cell, lights[i])) continue;
        if (out.count < p.capacity) indices[cell * p.capacity + out.count] = i;
        else out.overflow = 1;
        ++out.count;
    }
    cells[cell] = out;
}

inline uint clusterCell(constant GPULightClusterParams& p, float3 world) {
    const float3 v = clusterViewPoint(p, world);
    const float depth = -v.z;
    if (!all(isfinite(v)) || depth < p.nearPlane || depth > p.farPlane ||
        !p.gridX || !p.gridY || !p.gridZ) return ~0u;
    const float2 uv = float2(v.x / (depth * p.tanHalfFovX), v.y / (depth * p.tanHalfFovY)) * 0.5f + 0.5f;
    if (any(uv < 0) || any(uv > 1)) return ~0u;
    const uint x = min(uint(uv.x * p.gridX), p.gridX - 1u), y = min(uint(uv.y * p.gridY), p.gridY - 1u);
    const float slice = log(depth / p.nearPlane) / log(p.farPlane / p.nearPlane);
    const uint z = min(uint(slice * p.gridZ), p.gridZ - 1u);
    return (z * p.gridY + y) * p.gridX + x;
}

// Cluster fallback shades punctual lights exactly and samples EACH area light
// once, with its inverse AREA density. No energy clamp, no hidden light cap.
// params0,surfaces1,lights2,cluster params3,cells4,indices5,STBN ranks6;
// output texture0 = local direct contribution only.
kernel void light_cluster_shade(constant GPUDIParams& p [[buffer(0)]],
                                const device GPUDISurface* surfaces [[buffer(1)]],
                                const device GPUSampledLight* lights [[buffer(2)]],
                                constant GPULightClusterParams& clusters [[buffer(3)]],
                                const device GPULightCluster* cells [[buffer(4)]],
                                const device uint* indices [[buffer(5)]],
                                const device uint* ranks [[buffer(6)]],
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
        const uint cell = p.pad1 ? ~0u : clusterCell(clusters, diVec(surface.position));
        const bool full = cell == ~0u || cells[cell].overflow != 0;
        const uint count = full ? p.lightCount : min(cells[cell].count, clusters.capacity);
        for (uint i = 0; i < count; ++i) {
            const uint lightIndex = full ? i : indices[cell * clusters.capacity + i];
            if (lightIndex >= p.lightCount || lights[lightIndex].type == LIGHT_DIRECTIONAL) continue;
            // Per-light rotation prevents identical endpoints of nearby emitters
            // while maintaining a uniform marginal and repeatable A/B seeds.
            const float2 uv(fract(diRandom(pixel, 40u, p, ranks) + diWhite(0, 0, p.frameIndex, lightIndex * 2u, p.stbnSeed)),
                            fract(diRandom(pixel, 41u, p, ranks) + diWhite(0, 0, p.frameIndex, lightIndex * 2u + 1u, p.stbnSeed)));
            if(!diLightFinite(lights[lightIndex]))atomic_fetch_add_explicit(signalErrors+3,1u,memory_order_relaxed);
            const DISample sample = diSampleTexturedLight(lights[lightIndex], lightIndex, uv,
                                                          diVec(surface.position), emitters, materials, textures);
            const float3 contribution=diBRDF(surface,sample);diRecordNonfinite(contribution,signalErrors,0u);
            if (sample.valid && sample.pdfArea > 0) {value += contribution / sample.pdfArea;diRecordNonfinite(value,signalErrors,2u);}
        }
    }
    diRecordNonfinite(value,signalErrors,1u);
    localDirect.write(float4(all(isfinite(value)) ? value : float3(0), 1), pixel);
}

// Visibility-correct clustered fallback, with the same overflow behaviour.
// Every contributing light gets its own any-hit ray, intentionally expensive
// but correct. This PSO owns its IFT; it never borrows the ReSTIR shade IFT.
// Above ABI plus instances7/TLAS8/IFT9.
kernel void light_cluster_shade_rt(constant GPUDIParams& p [[buffer(0)]],
                                   const device GPUDISurface* surfaces [[buffer(1)]],
                                   const device GPUSampledLight* lights [[buffer(2)]],
                                   constant GPULightClusterParams& clusters [[buffer(3)]],
                                   const device GPULightCluster* cells [[buffer(4)]],
                                   const device uint* indices [[buffer(5)]],
                                   const device uint* ranks [[buffer(6)]],
                                   const device GPUInstance* instances [[buffer(7)]],
                                   instance_acceleration_structure tlas [[buffer(8)]],
                                   intersection_function_table<triangle_data, instancing> ift [[buffer(9)]],
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
        const uint cell = p.pad1 ? ~0u : clusterCell(clusters, diVec(surface.position));
        const bool full = cell == ~0u || cells[cell].overflow != 0;
        const uint count = full ? p.lightCount : min(cells[cell].count, clusters.capacity);
        for (uint i = 0; i < count; ++i) {
            const uint lightIndex = full ? i : indices[cell * clusters.capacity + i];
            if (lightIndex >= p.lightCount || lights[lightIndex].type == LIGHT_DIRECTIONAL) continue;
            const float2 uv(fract(diRandom(pixel, 40u, p, ranks) + diWhite(0, 0, p.frameIndex, lightIndex * 2u, p.stbnSeed)),
                            fract(diRandom(pixel, 41u, p, ranks) + diWhite(0, 0, p.frameIndex, lightIndex * 2u + 1u, p.stbnSeed)));
            if(!diLightFinite(lights[lightIndex]))atomic_fetch_add_explicit(signalErrors+3,1u,memory_order_relaxed);
            const DISample sample = diSampleTexturedLight(lights[lightIndex], lightIndex, uv,
                                                          diVec(surface.position), emitters, materials, textures);
            const float3 contribution = diBRDF(surface, sample);diRecordNonfinite(contribution,signalErrors,0u);
            if (sample.valid && sample.pdfArea > 0 && (!(p.flags & DI_ENABLE_VISIBILITY) || !any(contribution > 0) ||
                diEndpointVisible(surface, sample, tlas, ift, instances, p.slotCount))) {value += contribution / sample.pdfArea;diRecordNonfinite(value,signalErrors,2u);}
        }
    }
    diRecordNonfinite(value,signalErrors,1u);
    localDirect.write(float4(all(isfinite(value)) ? value : float3(0), 1), pixel);
}
