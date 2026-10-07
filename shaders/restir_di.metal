#include <metal_stdlib>
#include "restir_common.h"
#include "light_visibility.h"
using namespace metal;
using namespace phosphor;

// Four independent compute passes, same core buffer ABI:
// params0, current surfaces1, local lights2, alias3, current input4,
// previous temporal reservoirs5, previous surfaces6, output7, STBN ranks8.
// Candidate only needs 0/1/2/3/7/8. Temporal additionally texture0 = motion
// current->previous in INPUT PIXELS, +Y down, same F8 temporalMotion contract.
// Spatial only reads finalized temporal buffer4; never in-place output7.
// History is temporal output, not spatial output (avoid feedback correlations).
// All kernels additionally bind emissive records12/materials13/texture handles14.
kernel void restir_di_candidates(constant GPUDIParams& p [[buffer(0)]],
                                 const device GPUDISurface* surfaces [[buffer(1)]],
                                 const device GPUSampledLight* lights [[buffer(2)]],
                                 const device GPUAliasEntry* alias [[buffer(3)]],
                                 device GPUDIReservoir* output [[buffer(7)]],
                                 const device uint* ranks [[buffer(8)]],
                                 const device GPUEmissiveSurface* emitters [[buffer(12)]],
                                 const device GPUMaterial* materials [[buffer(13)]],
                                 const device DITextureHandle* textures [[buffer(14)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid >= p.width * p.height) return;
    GPUDIReservoir r = diEmpty(p);
    const GPUDISurface surface = surfaces[tid];
    const uint2 pixel(tid % p.width, tid / p.width);
    if (surface.valid && p.lightCount && p.targetFloor > 0 && isfinite(p.targetFloor)) {
        // Fixed dimension allocation: candidate i -> [5i,5i+4], max 8.
        // Host validates preset bounds and supplies at least 64 STBN dimensions.
        for (uint i = 0; i < min(p.candidateCount, 8u); ++i) {
            const uint base = 5u * i;
            const uint column = min(uint(diRandom(pixel, base, p, ranks) * p.lightCount), p.lightCount - 1u);
            const GPUAliasEntry entry = alias[column];
            const uint selected = diRandom(pixel, base + 1u, p, ranks) < entry.probability ? column : entry.alias;
            if (selected >= p.lightCount) { r.pad[0] |= DI_ERROR_ALIAS; continue; } // never OOB
            const GPUAliasEntry selectedEntry = alias[selected];
            if (selectedEntry.lightIndex >= p.lightCount || !isfinite(selectedEntry.selectionPdf) ||
                !(selectedEntry.selectionPdf > 0) || !isfinite(entry.probability) || entry.probability < 0 || entry.probability > 1) {
                r.pad[0] |= DI_ERROR_ALIAS; continue;
            }
            const GPUSampledLight light = lights[selectedEntry.lightIndex];
            const float2 uv(diRandom(pixel, base + 2u, p, ranks), diRandom(pixel, base + 3u, p, ranks));
            const DISample sample = diSampleTexturedLight(light, selectedEntry.lightIndex, uv,
                                                          diVec(surface.position), emitters, materials, textures);
            GPUDIReservoir candidate = diEmpty(p);
            candidate.lightIndex = selectedEntry.lightIndex; candidate.lightID = light.id; candidate.lightGeneration = light.generation;
            candidate.u = uv.x; candidate.v = uv.y;
            candidate.target = diTarget(surface, sample, p.targetFloor);
            const float proposal = selectedEntry.selectionPdf * sample.pdfArea;
            const float weight = sample.valid && proposal > 0 ? candidate.target / proposal : 0;
            // Degenerate emitters have black integrands: count these proposals
            // with zero weight, rather than changing the sample count silently.
            diStream(r, candidate, weight, 1u, diRandom(pixel, base + 4u, p, ranks));
        }
    }
    if (surface.valid && p.lightCount && (!isfinite(p.targetFloor) || p.targetFloor <= 0)) r.pad[0] |= DI_ERROR_TARGET;
    diFinalize(r);
    output[tid] = r;
}

kernel void restir_di_temporal(constant GPUDIParams& p [[buffer(0)]],
                               const device GPUDISurface* surfaces [[buffer(1)]],
                               const device GPUSampledLight* lights [[buffer(2)]],
                               const device GPUDIReservoir* candidates [[buffer(4)]],
                               const device GPUDIReservoir* history [[buffer(5)]],
                               const device GPUDISurface* previousSurfaces [[buffer(6)]],
                               device GPUDIReservoir* output [[buffer(7)]],
                               const device uint* ranks [[buffer(8)]],
                               const device GPUEmissiveSurface* emitters [[buffer(12)]],
                               const device GPUMaterial* materials [[buffer(13)]],
                               const device DITextureHandle* textures [[buffer(14)]],
                               texture2d<float, access::read> motion [[texture(0)]],
                               uint tid [[thread_position_in_grid]]) {
    if (tid >= p.width * p.height) return;
    const GPUDISurface surface = surfaces[tid];
    const uint2 pixel(tid % p.width, tid / p.width);
    GPUDIReservoir r = candidates[tid];
    // Validity/epoch belongs to the VIEW/SIGNAL, and resize/cut/reset must set
    // DI_RESET_HISTORY for this pass; no previous resource is read on reset.
    if (surface.valid && (p.flags & DI_ENABLE_TEMPORAL) && !(p.flags & DI_RESET_HISTORY)) {
        const float2 delta = motion.read(pixel).xy;
        const float2 previousCenter = float2(pixel) + 0.5f + delta;
        if (all(isfinite(delta)) && all(previousCenter >= 0) && all(previousCenter < float2(p.width, p.height))) {
            const uint2 previousPixel = uint2(previousCenter);
            const uint index = previousPixel.y * p.width + previousPixel.x;
            const GPUDIReservoir old = history[index];
            if (old.lightIndex < p.lightCount && diCompatible(surface, previousSurfaces[index], p, true) &&
                diReusable(old, lights[old.lightIndex], p)) {
                // Candidate reservoir is already finalized. Restore its stream
                // weightSum and merge directly, recomputing normalization once.
                diMerge(r, old, surface, lights[old.lightIndex], p, diRandom(pixel, 48u, p, ranks), true,
                         emitters, materials, textures);
            }
        }
    }
    if (!surface.valid) r = diEmpty(p);
    diFinalize(r);
    output[tid] = r;
}

kernel void restir_di_spatial(constant GPUDIParams& p [[buffer(0)]],
                              const device GPUDISurface* surfaces [[buffer(1)]],
                              const device GPUSampledLight* lights [[buffer(2)]],
                              const device GPUDIReservoir* temporal [[buffer(4)]],
                              device GPUDIReservoir* output [[buffer(7)]],
                              const device uint* ranks [[buffer(8)]],
                              const device GPUEmissiveSurface* emitters [[buffer(12)]],
                              const device GPUMaterial* materials [[buffer(13)]],
                              const device DITextureHandle* textures [[buffer(14)]],
                              uint tid [[thread_position_in_grid]]) {
    if (tid >= p.width * p.height) return;
    const GPUDISurface surface = surfaces[tid];
    const uint2 pixel(tid % p.width, tid / p.width);
    GPUDIReservoir r = temporal[tid];
    uint seen[4];
    uint seenCount = 0;
    if (surface.valid && (p.flags & DI_ENABLE_SPATIAL) && p.spatialRadius > 0) {
        const int radius = int(min(p.spatialRadius, 1024u));
        for (uint i = 0; i < min(p.spatialCount, 4u); ++i) {
            const uint dim = 49u + 3u * i;
            const int2 offset(int(diRandom(pixel, dim, p, ranks) * float(2 * radius + 1)) - radius,
                               int(diRandom(pixel, dim + 1u, p, ranks) * float(2 * radius + 1)) - radius);
            const int2 neighborPixel = int2(pixel) + offset;
            if (all(offset == 0) || any(neighborPixel < 0) || any(neighborPixel >= int2(p.width, p.height))) continue;
            const uint index = uint(neighborPixel.y) * p.width + uint(neighborPixel.x);
            bool duplicate = false;
            for (uint j = 0; j < seenCount; ++j) duplicate |= seen[j] == index;
            if (duplicate) continue;
            seen[seenCount++] = index;
            const GPUDIReservoir source = temporal[index];
            if (source.lightIndex < p.lightCount && diCompatible(surface, surfaces[index], p, false))
                diMerge(r, source, surface, lights[source.lightIndex], p, diRandom(pixel, dim + 2u, p, ranks), false,
                         emitters, materials, textures);
        }
    }
    if (!surface.valid) r = diEmpty(p);
    diFinalize(r);
    output[tid] = r;
}

inline float3 diShadeSelected(GPUDISurface surface, GPUDIReservoir reservoir,
                               const device GPUSampledLight* lights, constant GPUDIParams& p,
                               const device GPUEmissiveSurface* emitters,
                               const device GPUMaterial* materials, const device DITextureHandle* textures,
                               thread DISample& sample) {
    if (!surface.valid || reservoir.lightIndex >= p.lightCount || !diValid(reservoir, lights[reservoir.lightIndex], p)) return float3(0);
    sample = diSampleTexturedLight(lights[reservoir.lightIndex], reservoir.lightIndex,
                                   float2(reservoir.u, reservoir.v), diVec(surface.position), emitters, materials, textures);
    return diBRDF(surface, sample) * reservoir.normalization;
}

// No RT resources are bound to the explicitly unshadowed fallback PSO.
kernel void restir_di_shade(constant GPUDIParams& p [[buffer(0)]],
                            const device GPUDISurface* surfaces [[buffer(1)]],
                            const device GPUSampledLight* lights [[buffer(2)]],
                            const device GPUDIReservoir* reservoirs [[buffer(3)]],
                            const device GPUEmissiveSurface* emitters [[buffer(12)]],
                            const device GPUMaterial* materials [[buffer(13)]],
                            const device DITextureHandle* textures [[buffer(14)]],
                            texture2d<float, access::write> localDirect [[texture(0)]],
                            uint tid [[thread_position_in_grid]]) {
    if (tid >= p.width * p.height) return;
    DISample sample{};
    const float3 value = diShadeSelected(surfaces[tid], reservoirs[tid], lights, p, emitters, materials, textures, sample);
    localDirect.write(float4(all(isfinite(value)) ? value : float3(0), 1), uint2(tid % p.width, tid / p.width));
}

// F10.3: shade PSO owns ITS OWN IFT linked to rt_alpha_generic. Binding it to
// another consumer's PSO is invalid even when function names are identical.
// params0/surfaces1/lights2/reservoir3/instances4/TLAS5/IFT6, output texture0.
kernel void restir_di_shade_rt(constant GPUDIParams& p [[buffer(0)]],
                               const device GPUDISurface* surfaces [[buffer(1)]],
                               const device GPUSampledLight* lights [[buffer(2)]],
                               const device GPUDIReservoir* reservoirs [[buffer(3)]],
                               const device GPUInstance* instances [[buffer(4)]],
                               instance_acceleration_structure tlas [[buffer(5)]],
                               intersection_function_table<triangle_data, instancing> ift [[buffer(6)]],
                               const device GPUEmissiveSurface* emitters [[buffer(12)]],
                               const device GPUMaterial* materials [[buffer(13)]],
                               const device DITextureHandle* textures [[buffer(14)]],
                               texture2d<float, access::write> localDirect [[texture(0)]],
                               uint tid [[thread_position_in_grid]]) {
    if (tid >= p.width * p.height) return;
    const GPUDISurface surface = surfaces[tid];
    DISample sample{};
    float3 value = diShadeSelected(surface, reservoirs[tid], lights, p, emitters, materials, textures, sample);
    if ((p.flags & DI_ENABLE_VISIBILITY) && sample.valid && any(value > 0) &&
        !diEndpointVisible(surface, sample, tlas, ift, instances, p.slotCount)) value = float3(0);
    localDirect.write(float4(all(isfinite(value)) ? value : float3(0), 1), uint2(tid % p.width, tid / p.width));
}
