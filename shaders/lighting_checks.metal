#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;
using namespace phosphor;

// Checker counters, eight u32 words: pixels, validSurfaces, reservoirsChecked,
// invalidNumericOrNormalization, lightIdentityMismatch, reservoirErrorFlags,
// nonfiniteOutput, negativeOutput. Counters3..7 MUST be zero. No summation of
// RGB energy is treated as evidence of estimator correctness or convergence.
// Reduction is one-dimensional over width*height dense pixels; tail SIMD
// lanes are allowed, as in the existing F9 checker.
kernel void lighting_check_clear(device atomic_uint* counts [[buffer(0)]],
                                  uint tid [[thread_position_in_grid]]) {
    if (tid < 8u) atomic_store_explicit(counts + tid, 0u, memory_order_relaxed);
}

inline void lightingCheckCount(device atomic_uint* word, uint value, uint lane) {
    const uint sum = simd_sum(value);
    if (lane == 0u && sum) atomic_fetch_add_explicit(word, sum, memory_order_relaxed);
}

// params0 GPULightingCheckParams(16 B), surfaces1, optional reservoirs2,
// sampled lights3, counts4, readonly localDirect texture0. The cluster mode
// must pass restir=0 and need not initialize/read the reservoir buffer.
kernel void lighting_check_di(constant GPULightingCheckParams& p [[buffer(0)]],
                               const device GPUDISurface* surfaces [[buffer(1)]],
                               const device GPUDIReservoir* reservoirs [[buffer(2)]],
                               const device GPUSampledLight* lights [[buffer(3)]],
                               device atomic_uint* counts [[buffer(4)]],
                               texture2d<float, access::read> localDirect [[texture(0)]],
                               uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (tid >= p.width * p.height) return;
    const GPUDISurface s = surfaces[tid];
    const float3 position(s.position[0], s.position[1], s.position[2]);
    const float3 geometric(s.geometricNormal[0], s.geometricNormal[1], s.geometricNormal[2]);
    const float3 normal(s.shadingNormal[0], s.shadingNormal[1], s.shadingNormal[2]);
    const float3 albedo(s.albedo[0], s.albedo[1], s.albedo[2]);
    const float3 view(s.viewDirection[0], s.viewDirection[1], s.viewDirection[2]);
    bool numeric = s.valid && (!all(isfinite(position)) || !all(isfinite(geometric)) ||
                    !all(isfinite(normal)) || !all(isfinite(albedo)) || !all(isfinite(view)) ||
                    !isfinite(s.depth) || !(s.depth > 0) || !isfinite(s.roughness) ||
                    !isfinite(s.metallic) || !(dot(geometric, geometric) > 0 && dot(normal, normal) > 0 && dot(view, view) > 0));
    uint checked = 0, identity = 0, errors = 0;
    if (p.restir) {
        const GPUDIReservoir r = reservoirs[tid];
        checked = s.valid ? 1u : 0u;
        errors = r.pad[0] ? 1u : 0u; // independent of valid bit/output masking
        numeric |= !isfinite(r.weightSum) || !isfinite(r.target) || !isfinite(r.normalization) ||
                   r.weightSum < 0 || r.target < 0 || r.normalization < 0;
        if (r.valid) {
            numeric |= r.M == 0 || !(r.target > 0 && r.weightSum > 0 && r.normalization > 0) ||
                       !isfinite(r.u) || !isfinite(r.v) || r.u < 0 || r.u >= 1 || r.v < 0 || r.v >= 1;
            if (r.M && r.target > 0 && isfinite(r.weightSum)) {
                // Independent reconstruction, using the final receiver target.
                // Product overflow is avoided as in FP32 source normalization.
                const float expected = (r.weightSum / float(r.M)) / r.target;
                const float scale = max(max(abs(expected), abs(r.normalization)), 1e-30f);
                numeric |= !isfinite(expected) || abs(expected - r.normalization) > 4e-5f * scale;
            }
            if (r.lightIndex >= p.lightCount) identity = 1u;
            else identity = r.lightID != lights[r.lightIndex].id || r.lightGeneration != lights[r.lightIndex].generation;
            numeric |= !s.valid; // background must not retain a selected light
        } else {
            // Empty/dark/degenerate proposal domain is legitimate. A positive
            // stream sum with invalid normalization is a failed reservoir.
            numeric |= r.weightSum > 0 || r.normalization != 0;
        }
    }
    const float4 output = localDirect.read(uint2(tid % p.width, tid / p.width));
    lightingCheckCount(counts + 0u, 1u, lane);
    lightingCheckCount(counts + 1u, s.valid ? 1u : 0u, lane);
    lightingCheckCount(counts + 2u, checked, lane);
    lightingCheckCount(counts + 3u, numeric ? 1u : 0u, lane);
    lightingCheckCount(counts + 4u, identity, lane);
    lightingCheckCount(counts + 5u, errors, lane);
    lightingCheckCount(counts + 6u, all(isfinite(output)) ? 0u : 1u, lane);
    lightingCheckCount(counts + 7u, any(output.xyz < 0) ? 1u : 0u, lane);
}

// Negative controls intentionally corrupt state without setting checker flags.
// The independent checker detects the actual generation mismatch. Dispatch
// width*height AFTER shade and BEFORE check; every live selected reservoir is
// corrupted so a background first pixel cannot turn the control into a pass.
kernel void lighting_corrupt_reservoir_generation(constant GPULightingCheckParams& p [[buffer(0)]],
                                                   device GPUDIReservoir* reservoirs [[buffer(1)]],
                                                   uint tid [[thread_position_in_grid]]) {
    if (tid < p.width * p.height && reservoirs[tid].valid) reservoirs[tid].lightGeneration ^= 1u;
}

// Alias negative BEFORE candidate sampling: corrupt EVERY source PDF to zero
// so the test cannot miss the affected alias column. The candidate pass marks
// DI_ERROR_ALIAS; checker sees pad[0], even when final shading was black.
kernel void lighting_corrupt_alias(constant GPULightingCheckParams& p [[buffer(0)]],
                                    device GPUAliasEntry* aliases [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid < p.lightCount) aliases[tid].selectionPdf = 0;
}
