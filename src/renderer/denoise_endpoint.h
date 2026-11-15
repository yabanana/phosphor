#pragma once
// ---------------------------------------------------------------------------
// F15.1 (ticket 13, temporal clip): endpoint-matched specular history of transmissive pixels, compiled by C++ (unit tests, CPU mirror) and
// MSL (reflections.metal writes the endpoint, denoise.metal matches it) from this single source; scalars only.
//
// Why: the surface motion vector of a glass pixel does not move the REFRACTED image.  Under a camera orbit or a moving emitter behind the glass
// the previous-frame pixel that saw the same refracted point is somewhere else in a window around the surface reprojection.  The transport
// therefore walks, for every transmissive pixel whatever Fresnel branch the frame drew, the DETERMINISTIC refraction path (h = n, reflection only
// under total internal reflection) to its ENDPOINT: the first non-glass hit (key = instance slot + 1, point = OBJECT-space position, so a moving
// panel and an orbiting camera keep the same point) or the environment (key 0, point = the final ray direction).  The denoiser stores the endpoint in
// its specular history and searches a window of the previous frame for the pixel whose endpoint is nearest the current one.
//
// Sample encoding (GPUSpecularSample, fields no downstream stage reads): direction[3] = point, proposalSolidAngle = the key encoded as the float
// 0x3f800000 | key (a normal float in [1, 2): never a denormal, key < 2^23), flag SPECULAR_SAMPLE_ENDPOINT.
// ---------------------------------------------------------------------------
#include "renderer/gpu_types.h"
#ifdef __METAL_VERSION__
#define PHOSPHOR_ENDPOINT_SQRT(x) metal::sqrt(x)
#else
#include <cmath>
#include <cstring>
#define PHOSPHOR_ENDPOINT_SQRT(x) std::sqrt(x)
#endif

namespace phosphor {

/// History record without an endpoint (no search is possible against it): the history stores key + 1, so a zero-initialised record (a fresh
/// buffer, a non-transmissive pixel, a value-initialised DenoiseHistory) never matches.
PHOSPHOR_GPU_CONSTANT u32 DENOISE_ENDPOINT_NONE = 0u;
inline u32 denoiseEndpointHistoryKey(u32 key) { return key + 1u; }
/// Search window radius in pixels around the surface-motion reprojection: (2R + 1)^2 candidates.
PHOSPHOR_GPU_CONSTANT int kDenoiseEndpointRadius = 4;
/// A candidate is accepted when its endpoint distance is at most kDenoiseEndpointAccept times the local footprint of the endpoint field in the
/// CURRENT frame (the largest endpoint distance to a same-key 4-neighbour: how far one pixel moves the endpoint).
PHOSPHOR_GPU_CONSTANT float kDenoiseEndpointAccept = 1.5f;

/// Key of the first non-glass hit of slot `slot`; 0 is reserved for the environment.
inline u32 denoiseEndpointKeyOfSlot(u32 slot) { return slot + 1u; }

inline float denoiseEndpointKeyEncode(u32 key) {
#ifdef __METAL_VERSION__
    return as_type<float>(0x3f800000u | key);
#else
    const u32 bits = 0x3f800000u | key;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}
inline u32 denoiseEndpointKeyDecode(float encoded) {
#ifdef __METAL_VERSION__
    return as_type<u32>(encoded) & 0x007fffffu;
#else
    u32 bits;
    std::memcpy(&bits, &encoded, sizeof(bits));
    return bits & 0x007fffffu;
#endif
}

/// Euclidean distance of two endpoint points (object-space positions or environment directions of the same key).
inline float denoiseEndpointDistance(float ax, float ay, float az, float bx, float by, float bz) {
    const float dx = ax - bx, dy = ay - by, dz = az - bz;
    return PHOSPHOR_ENDPOINT_SQRT(dx * dx + dy * dy + dz * dz);
}
/// Acceptance rule.  A footprint of 0 (no same-key 4-neighbour in the current frame: a sliver of a minified image, e.g. a pixel that sees one cell
/// of a checkerboard through a lens) has no local scale to bound the distance with; the candidate is then the nearest same-key endpoint of the
/// window, which is at most kDenoiseEndpointRadius pixels from the surface reprojection (flat emitters: same key = same radiance).
inline bool denoiseEndpointAccept(float distance, float footprint) {
    return footprint > 0.0f ? distance <= kDenoiseEndpointAccept * footprint : true;
}

} // namespace phosphor
