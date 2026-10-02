#pragma once

// ---------------------------------------------------------------------------
// F5.5 -- instance culling math, compiled by C++ (renderer/cull_reference.cpp,
// the unit tests) and by MSL (shaders/gpu_scene.metal, kernel scene_cull_flags)
// from this single source, like gpu_types.h.  Scalars only, no glm.
//
// World bounding sphere: centre = model * (c, 1), radius = r * an upper bound
// of the spectral norm of the model's 3x3 part (F6 fix: the longest basis
// column, used by F5, is NOT an upper bound when the matrix has shear, e.g.
// a child rotated under a non-uniformly scaled parent: [[1,1],[0,1]] has
// columns of length 1 and 1.414 but stretches by 1.618).  sigma_max^2 is the
// largest eigenvalue of the Gram matrix G = M^T M (G_ij = c_i . c_j), and
// Gershgorin bounds it by max_i (G_ii + sum_{j != i} |G_ij|): exact (the
// longest column) when the columns are orthogonal (rotation x scale), never
// below sigma_max otherwise.  The result is inflated by 2^-16 against the
// rounding of the float evaluation.
//
// cullSphere() tests, in this order and each only when its CULL_FLAG_* is set:
//   1 CULL_REASON_FRUSTUM : any of the 5 planes (left, right, bottom, top,
//                           near; the reverse-Z infinite projection has no far
//                           plane) with dot(n, c) + d + r < 0.  A sphere that
//                           intersects the frustum is VISIBLE (conservative).
//   2 CULL_REASON_DISTANCE: |c - camera| - r > maxDistance.
//   3 CULL_REASON_SIZE    : projected diameter in pixels below minPixels, i.e.
//                           2 r pixelScale / depth < minPixels with
//                             depth = max(dot(forward, c - camera), nearPlane)
//                           and forward = the normal of the near plane
//                           (planes[16..18]: the camera's forward axis, no extra
//                           field needed).  Evaluated multiplied out
//                           (2 r pixelScale < minPixels depth): no division, so
//                           fast-math cannot turn it into a reciprocal.
// The first failing test names the reason; 0 = visible.
//
// Bit-identity C++ / MSL: every function switches floating-point contraction
// off (no FMA) and uses the same expressions in the same order; the only
// operation that is not an IEEE-754 basic one is sqrt, which the MSL side
// takes from metal::precise (correctly rounded, like std::sqrt).  The GPU
// build may still be compiled with fast math (reassociation of the sums), so
// the host never assumes bit equality at a decision boundary: cullSphereEval()
// returns the signed margin to the nearest decision boundary and the tests
// and the GPU check accept a flip only inside |margin| < CULL_BAND.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"

#ifdef __METAL_VERSION__
#define PHOSPHOR_CULL_FN static __attribute__((unused))
#define PHOSPHOR_DEV device
#define PHOSPHOR_CONSTANT_AS constant
#define PHOSPHOR_FP_STRICT _Pragma("METAL fp contract(off)") _Pragma("clang fp reassociate(off)")
#define PHOSPHOR_SQRT(x) metal::precise::sqrt(x)
#define PHOSPHOR_MAX(a, b) metal::max((a), (b))
#define PHOSPHOR_ABS(a) metal::fabs(a)
#else
#include <algorithm>
#include <cfloat>
#include <cmath>
#define PHOSPHOR_CULL_FN inline
#define PHOSPHOR_DEV
#define PHOSPHOR_CONSTANT_AS
#define PHOSPHOR_FP_STRICT _Pragma("clang fp contract(off)")
#define PHOSPHOR_SQRT(x) std::sqrt(x)
#define PHOSPHOR_MAX(a, b) std::max((a), (b))
#define PHOSPHOR_ABS(a) std::fabs(a)
#endif

namespace phosphor {

// Cull reasons (the value of cullSphere(); also the index + 1 of the
// per-reason counters in GPUSceneCounters).
PHOSPHOR_GPU_CONSTANT u32 CULL_REASON_VISIBLE  = 0;
PHOSPHOR_GPU_CONSTANT u32 CULL_REASON_FRUSTUM  = 1;
PHOSPHOR_GPU_CONSTANT u32 CULL_REASON_DISTANCE = 2;
PHOSPHOR_GPU_CONSTANT u32 CULL_REASON_SIZE     = 3;

// scene_cull_flags writes one u32 per slot: 0 = invalid slot (not tested),
// 1 = visible, 2 * reason = culled for that reason (2, 4, 6).  Consumers test
// bit 0; the per-reason value lets the host check every decision.
PHOSPHOR_GPU_CONSTANT u32 CULL_FLAG_OUT_VISIBLE = 1;

// Word index of each counter in GPUSceneCounters (device atomics are
// indexed, a struct of atomics is not expressible in MSL).
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_TESTED          = 0;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_VISIBLE         = 1;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_CULLED_FRUSTUM  = 2;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_CULLED_DISTANCE = 3;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_CULLED_SIZE     = 4;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_DRAW_COMMANDS   = 5;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_NODES_UPDATED   = 6;
PHOSPHOR_GPU_CONSTANT u32 SCENE_COUNTER_QUEUE_OVERFLOW  = 7;

/// Relative width of the band around a decision boundary inside which the GPU
/// (fast math) and the CPU reference may disagree (see cullSphereEval).
PHOSPHOR_GPU_CONSTANT float CULL_BAND = 1.0e-4f;

struct GPUWorldSphere {
    float x, y, z, r;
};

/// Upper bound of the spectral norm (largest stretch) of the 3x3 part of the
/// column-major matrix `m`: Gershgorin on the Gram matrix of the columns (see
/// the header comment), inflated by 2^-16.
PHOSPHOR_CULL_FN float cullScaleBound(const PHOSPHOR_DEV float* m) {
    PHOSPHOR_FP_STRICT
    const float s0  = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
    const float s1  = m[4] * m[4] + m[5] * m[5] + m[6] * m[6];
    const float s2  = m[8] * m[8] + m[9] * m[9] + m[10] * m[10];
    const float g01 = PHOSPHOR_ABS(m[0] * m[4] + m[1] * m[5] + m[2] * m[6]);
    const float g02 = PHOSPHOR_ABS(m[0] * m[8] + m[1] * m[9] + m[2] * m[10]);
    const float g12 = PHOSPHOR_ABS(m[4] * m[8] + m[5] * m[9] + m[6] * m[10]);
    const float b0  = s0 + g01 + g02;
    const float b1  = s1 + g01 + g12;
    const float b2  = s2 + g02 + g12;
    return PHOSPHOR_SQRT(PHOSPHOR_MAX(PHOSPHOR_MAX(b0, b1), b2)) * (1.0f + 1.0f / 65536.0f);
}

/// World sphere of `sphere` (centre xyz + radius, mesh space) under the
/// column-major model matrix `m` (16 floats).
PHOSPHOR_CULL_FN GPUWorldSphere cullWorldSphere(const PHOSPHOR_DEV float* m, const PHOSPHOR_DEV float* sphere) {
    PHOSPHOR_FP_STRICT
    const float cx = sphere[0], cy = sphere[1], cz = sphere[2];
    GPUWorldSphere w;
    w.x = m[0] * cx + m[4] * cy + m[8] * cz + m[12];
    w.y = m[1] * cx + m[5] * cy + m[9] * cz + m[13];
    w.z = m[2] * cx + m[6] * cy + m[10] * cz + m[14];
    w.r = sphere[3] * cullScaleBound(m);
    return w;
}

/// 0 visible, else CULL_REASON_* (see the header comment).  The sphere is in
/// world space.
PHOSPHOR_CULL_FN u32 cullSphere(const PHOSPHOR_CONSTANT_AS GPUCullParams& p, float cx, float cy, float cz, float r) {
    PHOSPHOR_FP_STRICT
    if ((p.flags & CULL_FLAG_FRUSTUM) != 0u) {
        for (u32 i = 0; i < 5u; ++i) {
            const float dist = p.planes[i * 4u + 0u] * cx + p.planes[i * 4u + 1u] * cy + p.planes[i * 4u + 2u] * cz + p.planes[i * 4u + 3u];
            if (dist + r < 0.0f) return CULL_REASON_FRUSTUM;
        }
    }
    if ((p.flags & (CULL_FLAG_DISTANCE | CULL_FLAG_SIZE)) != 0u) {
        const float dx = cx - p.cameraPosition[0];
        const float dy = cy - p.cameraPosition[1];
        const float dz = cz - p.cameraPosition[2];
        if ((p.flags & CULL_FLAG_DISTANCE) != 0u) {
            const float len = PHOSPHOR_SQRT(dx * dx + dy * dy + dz * dz);
            if (len - r > p.maxDistance) return CULL_REASON_DISTANCE;
        }
        if ((p.flags & CULL_FLAG_SIZE) != 0u) {
            const float viewDepth = dx * p.planes[16] + dy * p.planes[17] + dz * p.planes[18];
            const float depth     = PHOSPHOR_MAX(viewDepth, p.nearPlane);
            const float lhs       = 2.0f * r * p.pixelScale;
            const float rhs       = p.minPixels * depth;
            if (lhs < rhs) return CULL_REASON_SIZE;
        }
    }
    return CULL_REASON_VISIBLE;
}

#ifndef __METAL_VERSION__

/// cullSphere() + the signed margin to the nearest decision boundary that
/// matters for the outcome.  Each test has a relative margin (>= 0 passes):
///   plane    (dist + r) / (|dist| + r + 1)
///   distance (maxDistance - (len - r)) / (|len - r| + |maxDistance|)
///   size     (lhs - rhs) / (lhs + rhs)
/// The frustum test's margin is the minimum over its planes.  Visible: margin
/// = the smallest margin of the enabled tests (> 0 or 0 exactly on a
/// boundary).  Culled with first failing test k: -min(|margin_k|, margins of
/// the tests before k): a result is robust against rounding when
/// |margin| >= CULL_BAND.  No test enabled: +FLT_MAX.
struct CullEval {
    u32 reason;
    float margin;
};

inline CullEval cullSphereEval(const GPUCullParams& p, float cx, float cy, float cz, float r) {
    PHOSPHOR_FP_STRICT
    float group[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    if ((p.flags & CULL_FLAG_FRUSTUM) != 0u) {
        float m = FLT_MAX;
        for (u32 i = 0; i < 5u; ++i) {
            const float dist = p.planes[i * 4u + 0u] * cx + p.planes[i * 4u + 1u] * cy + p.planes[i * 4u + 2u] * cz + p.planes[i * 4u + 3u];
            m = std::min(m, (dist + r) / (std::fabs(dist) + r + 1.0f));
        }
        group[0] = m;
    }
    if ((p.flags & (CULL_FLAG_DISTANCE | CULL_FLAG_SIZE)) != 0u) {
        const float dx = cx - p.cameraPosition[0];
        const float dy = cy - p.cameraPosition[1];
        const float dz = cz - p.cameraPosition[2];
        if ((p.flags & CULL_FLAG_DISTANCE) != 0u) {
            const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float den = std::fabs(len - r) + std::fabs(p.maxDistance);
            group[1]        = den > 0.0f ? (p.maxDistance - (len - r)) / den : 0.0f;
        }
        if ((p.flags & CULL_FLAG_SIZE) != 0u) {
            const float viewDepth = dx * p.planes[16] + dy * p.planes[17] + dz * p.planes[18];
            const float depth     = std::max(viewDepth, p.nearPlane);
            const float lhs       = 2.0f * r * p.pixelScale;
            const float rhs       = p.minPixels * depth;
            const float den       = lhs + rhs;
            group[2]              = den > 0.0f ? (lhs - rhs) / den : 0.0f;
        }
    }
    CullEval e{CULL_REASON_VISIBLE, FLT_MAX};
    for (u32 k = 0; k < 3u; ++k) {
        if (group[k] < 0.0f) {
            e.reason = k + 1u;
            e.margin = -std::fabs(group[k]);
            for (u32 j = 0; j < k; ++j) e.margin = -std::min(-e.margin, group[j]);
            return e;
        }
        e.margin = std::min(e.margin, group[k]);
    }
    return e;
}

#endif // !__METAL_VERSION__

} // namespace phosphor

#undef PHOSPHOR_CULL_FN
#undef PHOSPHOR_DEV
#undef PHOSPHOR_CONSTANT_AS
