#pragma once

// ---------------------------------------------------------------------------
// F6.2/F6.4 -- meshlet culling math, compiled by C++ (renderer/
// meshlet_cull_reference.cpp, the tests, the self-check) and by MSL (the
// object shader of shaders/meshlet.metal) from this single source, like
// renderer/cull_math.h.  Scalars only.  Every test is CONSERVATIVE: when a
// decision cannot be proven the meshlet stays visible.
//
// World sphere   centre = model * (c, 1); radius = r * cullScaleBound(model)
//                (spectral-norm bound, renderer/cull_math.h: valid for any
//                matrix, shear included).
// Frustum        culled when the sphere is fully outside one of the 5 planes
//                (dist + r < 0, as F5).
// Normal cone    meshoptimizer v1.3 (meshletutils.cpp): every triangle of the
//                meshlet faces away from a camera at `cam` when
//                  dot(normalize(apex - cam), axis) >= cutoff.
//                Mesh-space cones are moved to world space only for
//                SIMILARITY transforms (rotation, uniform scale, mirror,
//                translation; tested with a relative tolerance
//                MESHLET_SIMILARITY_TOL on the Gram matrix of the columns):
//                angles are preserved, apex -> M (apex, 1), axis -> M3 axis
//                (for M = sR, the outward normal M^-T n is parallel to R n =
//                M3 n / s, mirrored or not: a mirrored instance is drawn with
//                front-face culling, i.e. the faces whose transformed outward
//                normal faces away are the culled ones -- the same test).  Any
//                other matrix, a zero axis (degenerate cluster), cutoff >= 1
//                (cone wider than ~168 degrees), the camera at the apex: no
//                rejection.  The threshold is raised by MESHLET_CONE_EPS (cos
//                space) to absorb the tolerance and rounding.  Double-sided
//                materials (cull class None) never use the cone (caller).
// Hi-Z footprint The 8 corners of the world sphere's AABB are projected with a
//                view-projection matrix (current frame, or the history's):
//                any corner with clip w <= near * (1 + 1e-6) (the AABB crosses
//                the near plane or lies behind the camera) or a non-finite
//                value -> not testable (visible).  Screen rectangle = min/max
//                of the projected corners, padded by HIZ_RECT_PAD pixels
//                (raster snapping), clamped to the viewport; empty -> not
//                testable.  Nearest depth = near / min_w (reverse-Z infinite:
//                NDC z = near / w, larger = nearer; the AABB contains the
//                sphere, so no point of the meshlet is nearer).  Level = the
//                finest pyramid level whose texels covering the rectangle span
//                < HIZ_TEST_SPAN per axis (level L texel x covers pixels
//                [x * 2^(L+1), (x + 1) * 2^(L+1))); the caller reads ALL those
//                texels and takes their minimum.
// Occlusion      occluded iff nearestDepth * (1 + HIZ_DEPTH_EPS) < minimum of
//                the texels (strictly behind the farthest occluder depth of the
//                whole covered region).  Background (clear depth 0) can never
//                occlude: nothing is < 0.
// Size (approx.) MESHLET_CULL_SIZE: the projected rectangle (unpadded) has an
//                area below minPixels^2.  Never enabled in the exact preset.
// ---------------------------------------------------------------------------

#include "renderer/cull_math.h"
#include "renderer/meshlet_layout.h"

#ifdef __METAL_VERSION__
#define PHOSPHOR_MC_FN static __attribute__((unused))
#define PHOSPHOR_MC_DEV device
#define PHOSPHOR_MC_CONST constant
#define PHOSPHOR_MC_STRICT _Pragma("METAL fp contract(off)") _Pragma("clang fp reassociate(off)")
#define PHOSPHOR_MC_SQRT(x) metal::precise::sqrt(x)
#define PHOSPHOR_MC_MIN(a, b) metal::min((a), (b))
#define PHOSPHOR_MC_MAX(a, b) metal::max((a), (b))
#define PHOSPHOR_MC_ABS(a) metal::fabs(a)
#define PHOSPHOR_MC_FLOOR(a) metal::floor(a)
#define PHOSPHOR_MC_FINITE(a) metal::isfinite(a)
#else
#include <algorithm>
#include <cmath>
#define PHOSPHOR_MC_FN inline
#define PHOSPHOR_MC_DEV
#define PHOSPHOR_MC_CONST
#define PHOSPHOR_MC_STRICT _Pragma("clang fp contract(off)")
#define PHOSPHOR_MC_SQRT(x) std::sqrt(x)
#define PHOSPHOR_MC_MIN(a, b) std::min((a), (b))
#define PHOSPHOR_MC_MAX(a, b) std::max((a), (b))
#define PHOSPHOR_MC_ABS(a) std::fabs(a)
#define PHOSPHOR_MC_FLOOR(a) std::floor(a)
#define PHOSPHOR_MC_FINITE(a) std::isfinite(a)
#endif

namespace phosphor {

PHOSPHOR_GPU_CONSTANT float MESHLET_CONE_EPS       = 1.0e-3f; // cos-space margin of the cone test
PHOSPHOR_GPU_CONSTANT float MESHLET_SIMILARITY_TOL = 1.0e-4f; // relative tolerance of the similarity check
PHOSPHOR_GPU_CONSTANT float HIZ_DEPTH_EPS          = 1.0e-5f; // relative depth margin of the occlusion test
PHOSPHOR_GPU_CONSTANT float HIZ_RECT_PAD           = 1.0f;    // pixels added on every side of the rectangle
PHOSPHOR_GPU_CONSTANT float HIZ_NEAR_EPS           = 1.0e-6f; // clip w must exceed near * (1 + eps)

/// Power-of-two level-0 size of the pyramid of a `viewport` (>= ceil(v / 2),
/// at least 1) and its level count (down to 1x1).
PHOSPHOR_MC_FN u32 hizLevel0Size(u32 viewport) {
    const u32 halfSize = (viewport + 1u) / 2u;
    u32 s = 1u;
    while (s < halfSize && s < (1u << (HIZ_MAX_LEVELS - 1u))) s <<= 1u;
    return s;
}
PHOSPHOR_MC_FN u32 hizLevelCount(u32 w0, u32 h0) {
    const u32 m = PHOSPHOR_MC_MAX(w0, h0);
    u32 levels = 1u;
    while ((1u << (levels - 1u)) < m && levels < HIZ_MAX_LEVELS) ++levels;
    return levels;
}
/// Size of level `level` (Metal's rule max(1, s >> level); exact halving for
/// power-of-two level 0).
PHOSPHOR_MC_FN u32 hizLevelSize(u32 size0, u32 level) { return PHOSPHOR_MC_MAX(size0 >> level, 1u); }

/// World sphere of a meshlet under the column-major model matrix `m`.
PHOSPHOR_MC_FN GPUWorldSphere meshletWorldSphere(const PHOSPHOR_MC_DEV float* m, const PHOSPHOR_MC_DEV GPUMeshletBounds& b) {
    PHOSPHOR_MC_STRICT
    GPUWorldSphere w;
    w.x = m[0] * b.center[0] + m[4] * b.center[1] + m[8] * b.center[2] + m[12];
    w.y = m[1] * b.center[0] + m[5] * b.center[1] + m[9] * b.center[2] + m[13];
    w.z = m[2] * b.center[0] + m[6] * b.center[1] + m[10] * b.center[2] + m[14];
    w.r = b.radius * cullScaleBound(m);
    return w;
}

/// True when the world sphere is fully outside one of the 5 frustum planes.
PHOSPHOR_MC_FN bool meshletFrustumCulled(const PHOSPHOR_MC_CONST GPUMeshletCullParams& p, GPUWorldSphere w) {
    PHOSPHOR_MC_STRICT
    for (u32 i = 0; i < 5u; ++i) {
        const float dist = p.planes[i * 4u + 0u] * w.x + p.planes[i * 4u + 1u] * w.y + p.planes[i * 4u + 2u] * w.z +
                           p.planes[i * 4u + 3u];
        if (dist + w.r < 0.0f) return true;
    }
    return false;
}

/// True when the 3x3 part of `m` is a similarity (rotation x uniform scale,
/// mirror allowed) within MESHLET_SIMILARITY_TOL.
PHOSPHOR_MC_FN bool meshletIsSimilarity(const PHOSPHOR_MC_DEV float* m) {
    PHOSPHOR_MC_STRICT
    const float s0   = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
    const float s1   = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
    const float s2   = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];
    const float g01  = m[0] * m[4] + m[1] * m[5] + m[2] * m[6];
    const float g02  = m[0] * m[8] + m[1] * m[9] + m[2] * m[10];
    const float g12  = m[4] * m[8] + m[5] * m[9] + m[6] * m[10];
    const float smax = PHOSPHOR_MC_MAX(PHOSPHOR_MC_MAX(s0, s1), s2);
    const float tol  = MESHLET_SIMILARITY_TOL * smax;
    if (!(smax > 0.0f) || !PHOSPHOR_MC_FINITE(smax)) return false;
    return PHOSPHOR_MC_ABS(s0 - s1) <= tol && PHOSPHOR_MC_ABS(s0 - s2) <= tol && PHOSPHOR_MC_ABS(s1 - s2) <= tol &&
           PHOSPHOR_MC_ABS(g01) <= tol && PHOSPHOR_MC_ABS(g02) <= tol && PHOSPHOR_MC_ABS(g12) <= tol;
}

/// True when every triangle of the meshlet faces away from the camera
/// (normal cone, see the header comment).  The caller skips double-sided
/// materials.
PHOSPHOR_MC_FN bool meshletConeCulled(const PHOSPHOR_MC_DEV float* m, const PHOSPHOR_MC_DEV GPUMeshletBounds& b,
                                      const PHOSPHOR_MC_CONST float* cam) {
    PHOSPHOR_MC_STRICT
    const float threshold = b.coneCutoff + MESHLET_CONE_EPS;
    if (!(threshold < 1.0f)) return false; // no usable cone (cutoff 1 = cone wider than ~168 degrees)
    if (b.coneAxis[0] == 0.0f && b.coneAxis[1] == 0.0f && b.coneAxis[2] == 0.0f) return false; // degenerate cluster
    if (!meshletIsSimilarity(m)) return false;
    const float ax = m[0] * b.coneAxis[0] + m[4] * b.coneAxis[1] + m[8] * b.coneAxis[2];
    const float ay = m[1] * b.coneAxis[0] + m[5] * b.coneAxis[1] + m[9] * b.coneAxis[2];
    const float az = m[2] * b.coneAxis[0] + m[6] * b.coneAxis[1] + m[10] * b.coneAxis[2];
    const float dx = m[0] * b.coneApex[0] + m[4] * b.coneApex[1] + m[8] * b.coneApex[2] + m[12] - cam[0];
    const float dy = m[1] * b.coneApex[0] + m[5] * b.coneApex[1] + m[9] * b.coneApex[2] + m[13] - cam[1];
    const float dz = m[2] * b.coneApex[0] + m[6] * b.coneApex[1] + m[10] * b.coneApex[2] + m[14] - cam[2];
    const float dd = dx * dx + dy * dy + dz * dz;
    const float aa = ax * ax + ay * ay + az * az;
    if (!(dd > 0.0f) || !(aa > 0.0f) || !PHOSPHOR_MC_FINITE(dd) || !PHOSPHOR_MC_FINITE(aa)) return false;
    const float lhs = dx * ax + dy * ay + dz * az;
    return lhs >= threshold * PHOSPHOR_MC_SQRT(dd) * PHOSPHOR_MC_SQRT(aa);
}

/// Texels of a pyramid level an occlusion test must read: [x0, x1] x [y0, y1]
/// of `level` (inclusive, each span < HIZ_TEST_SPAN), and the nearest depth of
/// the bound.  usable == 0: no occlusion decision is possible (visible).
struct HiZFootprint {
    u32   x0, y0, x1, y1;
    u32   level;
    u32   usable;
    float nearestDepth;
    float area; // pixels of the unpadded rectangle (size cull)
};

/// Footprint of the world sphere `w` under `viewProj` (16 floats, column-major)
/// in a viewport/pyramid described by `p` (see the header comment).
PHOSPHOR_MC_FN HiZFootprint hizFootprint(const PHOSPHOR_MC_CONST float* vp, GPUWorldSphere w,
                                         const PHOSPHOR_MC_CONST GPUMeshletCullParams& p) {
    PHOSPHOR_MC_STRICT
    HiZFootprint f;
    f.x0 = f.y0 = f.x1 = f.y1 = 0u;
    f.level = 0u;
    f.usable = 0u;
    f.nearestDepth = 1.0f;
    f.area = 0.0f;
    const float wLimit = p.nearPlane * (1.0f + HIZ_NEAR_EPS);
    float minX = 3.0e38f, minY = 3.0e38f, maxX = -3.0e38f, maxY = -3.0e38f, minW = 3.0e38f;
    for (u32 c = 0; c < 8u; ++c) {
        const float x  = (c & 1u) != 0u ? w.x + w.r : w.x - w.r;
        const float y  = (c & 2u) != 0u ? w.y + w.r : w.y - w.r;
        const float z  = (c & 4u) != 0u ? w.z + w.r : w.z - w.r;
        const float cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
        const float cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
        const float cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        if (!(cw > wLimit)) return f; // crosses the near plane / behind the camera (NaN too)
        const float sx = (cx / cw * 0.5f + 0.5f) * p.viewport[0];
        const float sy = (0.5f - cy / cw * 0.5f) * p.viewport[1];
        minX = PHOSPHOR_MC_MIN(minX, sx);
        maxX = PHOSPHOR_MC_MAX(maxX, sx);
        minY = PHOSPHOR_MC_MIN(minY, sy);
        maxY = PHOSPHOR_MC_MAX(maxY, sy);
        minW = PHOSPHOR_MC_MIN(minW, cw);
    }
    if (!PHOSPHOR_MC_FINITE(minX) || !PHOSPHOR_MC_FINITE(maxX) || !PHOSPHOR_MC_FINITE(minY) ||
        !PHOSPHOR_MC_FINITE(maxY)) {
        return f;
    }
    f.area = (maxX - minX) * (maxY - minY);
    const float vw = p.viewport[0], vh = p.viewport[1];
    const float x0 = PHOSPHOR_MC_FLOOR(minX - HIZ_RECT_PAD), x1 = PHOSPHOR_MC_FLOOR(maxX + HIZ_RECT_PAD);
    const float y0 = PHOSPHOR_MC_FLOOR(minY - HIZ_RECT_PAD), y1 = PHOSPHOR_MC_FLOOR(maxY + HIZ_RECT_PAD);
    if (x1 < 0.0f || y1 < 0.0f || x0 > vw - 1.0f || y0 > vh - 1.0f) return f; // off screen: no decision
    const u32 px0 = static_cast<u32>(PHOSPHOR_MC_MAX(x0, 0.0f));
    const u32 py0 = static_cast<u32>(PHOSPHOR_MC_MAX(y0, 0.0f));
    const u32 px1 = static_cast<u32>(PHOSPHOR_MC_MIN(x1, vw - 1.0f));
    const u32 py1 = static_cast<u32>(PHOSPHOR_MC_MIN(y1, vh - 1.0f));
    u32 level = 0u;
    for (; level + 1u < p.hizLevels; ++level) {
        const u32 s = level + 1u;
        if ((px1 >> s) - (px0 >> s) < HIZ_TEST_SPAN && (py1 >> s) - (py0 >> s) < HIZ_TEST_SPAN) break;
    }
    const u32 s = level + 1u;
    f.x0 = px0 >> s;
    f.y0 = py0 >> s;
    f.x1 = px1 >> s;
    f.y1 = py1 >> s;
    f.level = level;
    f.nearestDepth = p.nearPlane / minW;
    f.usable = 1u;
    return f;
}

/// The occlusion decision of a footprint against the minimum of its texels.
PHOSPHOR_MC_FN bool hizOccluded(float nearestDepth, float minTexel) {
    PHOSPHOR_MC_STRICT
    return nearestDepth * (1.0f + HIZ_DEPTH_EPS) < minTexel;
}

} // namespace phosphor

#undef PHOSPHOR_MC_FN
#undef PHOSPHOR_MC_DEV
#undef PHOSPHOR_MC_CONST
