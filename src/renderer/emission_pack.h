#pragma once
// ---------------------------------------------------------------------------
// F13.1 -- resolve emission carried with each DI receiver (GPUDISurface::pad[1])
// so that specular fallbacks reading the resolved image (SSR source, radiance
// cache) can leave out the emission that primary DI already integrates.
// Compiled by C++ (unit tests) and by MSL from this single source.
//
// Shared-exponent RGB: three 9-bit mantissas and a 5-bit exponent with bias 12,
// value = mantissa * 2^(stored - 12 - 9). Largest value 511 * 2^10 = 523264
// (covers the 368640 wide-emission fixture); the largest channel keeps 8-9
// significant bits (relative error <= 2^-9), channels below 2^-21 of the
// range and non-finite or negative input pack to zero. Larger values clamp,
// so a subtraction never removes more than the resolve added.
// ---------------------------------------------------------------------------
#include "renderer/gpu_types.h"
#ifdef __METAL_VERSION__
#define PHOSPHOR_EMISSION_FN static inline __attribute__((unused))
#define PHOSPHOR_EMISSION_AS_UINT(x) as_type<uint>(x)
#define PHOSPHOR_EMISSION_AS_FLOAT(x) as_type<float>(x)
#define PHOSPHOR_EMISSION_FLOOR(x) metal::floor(x)
#define PHOSPHOR_EMISSION_FINITE(x) metal::isfinite(x)
#else
#include <bit>
#include <cmath>
#include <cstdint>
#define PHOSPHOR_EMISSION_FN inline
#define PHOSPHOR_EMISSION_AS_UINT(x) std::bit_cast<uint32_t>(x)
#define PHOSPHOR_EMISSION_AS_FLOAT(x) std::bit_cast<float>(x)
#define PHOSPHOR_EMISSION_FLOOR(x) std::floor(x)
#define PHOSPHOR_EMISSION_FINITE(x) std::isfinite(x)
#endif

namespace phosphor {
#ifdef __METAL_VERSION__
using emission_u32 = uint;
#else
using emission_u32 = uint32_t;
#endif
PHOSPHOR_GPU_CONSTANT int EmissionPackBias = 12;
PHOSPHOR_GPU_CONSTANT float EmissionPackMax = 523264.0f; // 511 * 2^(31 - 12 - 9)

// 2^e for -126 <= e <= 127, exact.
PHOSPHOR_EMISSION_FN float emissionPow2(int e) { return PHOSPHOR_EMISSION_AS_FLOAT(emission_u32(e + 127) << 23); }

PHOSPHOR_EMISSION_FN float emissionSanitize(float c) {
    return (PHOSPHOR_EMISSION_FINITE(c) && c > 0.0f) ? (c < EmissionPackMax ? c : EmissionPackMax) : 0.0f;
}

PHOSPHOR_EMISSION_FN emission_u32 packEmission(float red, float green, float blue) {
    const float r = emissionSanitize(red), g = emissionSanitize(green), b = emissionSanitize(blue);
    const float m = r > g ? (r > b ? r : b) : (g > b ? g : b);
    if (!(m > 0.0f)) return 0u;
    // m < 2^e: the float exponent of m plus one, never below -bias.
    int e = int((PHOSPHOR_EMISSION_AS_UINT(m) >> 23) & 255u) - 127 + 1;
    if (e < -EmissionPackBias) e = -EmissionPackBias;
    float scale = emissionPow2(e - 9);
    if (PHOSPHOR_EMISSION_FLOOR(m / scale + 0.5f) > 511.0f) { ++e; scale *= 2.0f; }
    if (e + EmissionPackBias > 31) return 0x1ffu | (0x1ffu << 9) | (0x1ffu << 18) | (31u << 27);
    const float q0 = PHOSPHOR_EMISSION_FLOOR(r / scale + 0.5f), q1 = PHOSPHOR_EMISSION_FLOOR(g / scale + 0.5f),
                q2 = PHOSPHOR_EMISSION_FLOOR(b / scale + 0.5f);
    const emission_u32 m0 = emission_u32(q0 < 511.0f ? q0 : 511.0f), m1 = emission_u32(q1 < 511.0f ? q1 : 511.0f),
                       m2 = emission_u32(q2 < 511.0f ? q2 : 511.0f);
    return m0 | (m1 << 9) | (m2 << 18) | (emission_u32(e + EmissionPackBias) << 27);
}

PHOSPHOR_EMISSION_FN float unpackEmissionChannel(emission_u32 packed, emission_u32 channel) {
    const float scale = emissionPow2(int(packed >> 27) - EmissionPackBias - 9);
    return float((packed >> (9u * channel)) & 0x1ffu) * scale;
}
#ifdef __METAL_VERSION__
PHOSPHOR_EMISSION_FN float3 unpackEmission3(emission_u32 packed) {
    return float3(unpackEmissionChannel(packed, 0u), unpackEmissionChannel(packed, 1u), unpackEmissionChannel(packed, 2u));
}
#endif
} // namespace phosphor
