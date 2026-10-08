#pragma once
#include "renderer/gpu_types.h"
#ifdef __METAL_VERSION__
#define SHADOW_FN static __attribute__((unused))
#define SHADOW_AS constant
#define SHADOW_TAN(x) metal::tan(x)
#define SHADOW_SQRT(x) metal::sqrt(x)
#define SHADOW_ABS(x) metal::abs(x)
#define SHADOW_MAX(a,b) metal::max(a,b)
#else
#include <cmath>
#include <algorithm>
#define SHADOW_FN inline
#define SHADOW_AS
#define SHADOW_TAN(x) std::tan(x)
#define SHADOW_SQRT(x) std::sqrt(x)
#define SHADOW_ABS(x) std::abs(x)
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
SHADOW_FN bool shadowHistoryMatches(const SHADOW_AS GPUShadowParams& p,
                                   GPUShadowSurface s, GPUShadowHistory h) {
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
} // namespace phosphor
#undef SHADOW_FN
#undef SHADOW_AS
#undef SHADOW_TAN
#undef SHADOW_SQRT
#undef SHADOW_ABS
#undef SHADOW_MAX
