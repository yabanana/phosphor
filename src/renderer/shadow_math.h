#pragma once
#include "renderer/gpu_types.h"
#ifdef __METAL_VERSION__
#define SHADOW_FN static __attribute__((unused))
#define SHADOW_AS constant
#define SHADOW_TAN(x) metal::tan(x)
#define SHADOW_SQRT(x) metal::sqrt(x)
#define SHADOW_ABS(x) metal::abs(x)
#define SHADOW_EXP(x) metal::precise::exp(x)
#define SHADOW_MAX(a,b) metal::max(a,b)
#else
#include <cmath>
#include <algorithm>
#define SHADOW_FN inline
#define SHADOW_AS
#define SHADOW_TAN(x) std::tan(x)
#define SHADOW_SQRT(x) std::sqrt(x)
#define SHADOW_ABS(x) std::abs(x)
#define SHADOW_EXP(x) std::exp(x)
#define SHADOW_MAX(a,b) std::max(a,b)
#endif
namespace phosphor {
// Directional/orthographic PCSS: distances are metres, output is metres.
// A perspective blocker-depth ratio is incorrect for sunlight.
SHADOW_FN float shadowPenumbraWorld(float receiverDistance, float blockerDistance, float angularRadius) {
    return SHADOW_MAX(0.0f, receiverDistance - blockerDistance) * SHADOW_TAN(angularRadius);
}
// Reverse-Z: a blocker is nearer the light, so has GREATER stored depth.
SHADOW_FN bool shadowDepthVisible(float receiverDepth, float storedDepth, float biasDepth) {
    return receiverDepth + biasDepth >= storedDepth;
}
// A raw solar sample is Bernoulli, so an all-zero/all-one neighborhood is
// legitimate Monte Carlo noise, not a bound on the expected visibility.
// Geometry/revision checks reject stale history before this bounded mean.
SHADOW_FN float shadowTemporalMean(float previous, float observation, u32 samples) {
    return previous + (observation - previous) / float(SHADOW_MAX(samples, 1u));
}
// Sphere vs orthographic light volume. Each matrix row scales sphere radius;
// no camera-visible, distance or occlusion input is allowed in this predicate.
SHADOW_FN bool shadowCasterIntersects(const SHADOW_AS GPUShadowCascade& c,
                                     float x, float y, float z, float radius) {
    const SHADOW_AS float* m = c.viewProjection;
    const float px = m[0]*x + m[4]*y + m[8]*z + m[12];
    const float py = m[1]*x + m[5]*y + m[9]*z + m[13];
    const float pz = m[2]*x + m[6]*y + m[10]*z + m[14];
    const float rx = radius * SHADOW_SQRT(m[0]*m[0]+m[4]*m[4]+m[8]*m[8]);
    const float ry = radius * SHADOW_SQRT(m[1]*m[1]+m[5]*m[5]+m[9]*m[9]);
    const float rz = radius * SHADOW_SQRT(m[2]*m[2]+m[6]*m[6]+m[10]*m[10]);
    return SHADOW_ABS(px) <= 1.0f+rx && SHADOW_ABS(py) <= 1.0f+ry && pz >= -rz && pz <= 1.0f+rz;
}
// The same exact identity and geometric rejection contract is used by CPU
// oracle tests and the GPU. World-point rejection safely discards deforming
// surfaces when prior object transforms are unavailable.
template <typename History>
SHADOW_FN bool shadowHistoryMatches(const SHADOW_AS GPUShadowParams& p,
                                   GPUShadowSurface s, History h) {
    if (!s.valid || !h.valid || !(p.flags & SHADOW_FLAG_HISTORY_VALID) ||
        s.slot != h.slot || s.generation != h.generation || p.viewID != h.viewID ||
        p.lightID != h.lightID || p.lightRevision != h.lightRevision || p.sceneRevision != h.sceneRevision)
        return false;
    const float dx=s.position[0]-h.position[0], dy=s.position[1]-h.position[1], dz=s.position[2]-h.position[2];
    const float nd=s.geometricNormal[0]*h.geometricNormal[0]+s.geometricNormal[1]*h.geometricNormal[1]+
                   s.geometricNormal[2]*h.geometricNormal[2];
    return dx*dx+dy*dy+dz*dz <= p.temporalPositionThreshold*p.temporalPositionThreshold &&
           nd >= p.temporalNormalThreshold;
}
// ---------------------------------------------------------------------------
// F15.1 (ticket 15): the single-layer transmittance map of a CSM cascade.  Transmissive casters (KHR_materials_transmission, not alpha-masked:
// the ray-traced path treats a masked one as an opaque-masked caster too) stay out of the opaque depth maps and draw into one RGBA32Float map per
// cascade instead: the SUM over every crossed surface of the ln RGB transmittance of transmissionShadowSurfaceLn (additive blending, so the
// accumulation is order independent) and, in alpha, the NEAREST transmissive depth (max blending: reverse-Z, larger = nearer the light; 0 = none).
// A receiver farther from the light than that depth multiplies its visibility by exp(min(sum, 0)): the declared single-layer approximation (the
// nearest transmissive depth decides "behind"; every layer in the texel contributes its factor, like the ray-traced sum).
// Distances of the signed medium term are measured from the far plane of the cascade (the common origin): the entry and exit terms of a closed
// volume cancel the origin, so a receiver behind the whole volume gets exactly -sigma_t (t_exit - t_entry).
// ---------------------------------------------------------------------------
SHADOW_FN bool shadowCasterTransmissive(u32 materialFlags, u32 extensionIndex, float alphaCutoff) {
    return (materialFlags & (MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT)) != 0u &&
           extensionIndex != MATERIAL_EXTENSION_INVALID && !(alphaCutoff > 0.0f);
}
/// Distance along the light direction from the FAR plane of the cascade (reverse-Z depth 0) to a fragment of reverse-Z depth `z`: it grows
/// towards the light like the ray's own distance from a receiver below the volume (an entry surface, met first, has the smaller value).
SHADOW_FN float shadowTransmitDistance(float z, float depthRange) { return z * depthRange; }
/// True when the receiver lies behind the nearest transmissive surface of the texel (`nearestDepth` 0: no transmissive caster there).
SHADOW_FN bool shadowTransmittanceBehind(float receiverDepth, float nearestDepth, float biasDepth) {
    return nearestDepth > 0.0f && receiverDepth + biasDepth < nearestDepth;
}
/// Transmittance of one channel from the summed ln (a medium cannot amplify: T <= 1, like the ray-traced clamp).
SHADOW_FN float shadowTransmittanceOfLn(float ln) { return SHADOW_EXP(ln < 0.0f ? ln : 0.0f); }
/// Facing of a transmit-pass fragment: `frontFacing` is the rasteriser's (clockwise winding in the light view), `mirrored` the instance flag
/// (its world winding is reversed).  Returns true for an ENTRY surface (the ray from the receiver towards the light goes against the outward
/// normal: the surface faces away from the light, a back face in the light's view).
SHADOW_FN bool shadowTransmitEntry(bool frontFacing, bool mirrored) { return (frontFacing == mirrored) ? false : true; }
} // namespace phosphor
#undef SHADOW_FN
#undef SHADOW_AS
#undef SHADOW_TAN
#undef SHADOW_SQRT
#undef SHADOW_ABS
#undef SHADOW_EXP
#undef SHADOW_MAX
