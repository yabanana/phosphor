#pragma once

// ---------------------------------------------------------------------------
// F15.4 (ticket 23) -- layer stack blend rules, compiled by C++ (tests/test_layer_blend.cpp, the terrain fixture's
// reference) and by MSL (shaders/material_shading.h, shadeSurface) from this single source, like cull_math.h.  Scalars
// only, no glm.  Decision 05 / research layered-materials.md section 6:
//
//   weights     the RGBA mask weights of the stack's layers (one channel each), clamped to >= 0 (NaN counts as 0), unused
//               layers forced to 0 and the rest normalised to sum 1.  An all-zero mask resolves to layer 0 (weight 1):
//               never a division by zero, never a NaN (the "no coverage" pixel shows the first layer, a declared rule).
//   height      OPTIONAL endpoint-exact contrast (a derivation of this project, not a published formula): with layer
//               heights h_i in metres, w'_i = n_i exp((h_i - hmax) / S) / sum_j n_j exp((h_j - hmax) / S) where n are the
//               normalised weights, S = LAYER_HEIGHT_SOFTNESS_M and hmax the largest height among the layers with
//               n_i > 0 (so every exponent is <= 0 and the denominator is >= n_best > 0).  n_i = 0 gives w'_i = 0 exactly
//               (a layer that is absent cannot win by height, unlike the max(h + a) - depth formula of Mishkinis 2013),
//               a single n_j = 1 gives w'_j = 1 exactly, equal heights give w' = n, and for fixed heights w'_i is
//               monotone in n_i.
//   parameters  base colour, metallic, roughness, occlusion blend linearly with the final weights (parameter space, one
//               BSDF evaluation afterwards).  The blend is a sum in layer order; at a weight vector e_j the result is the
//               layer's own parameter bit for bit (0 * p adds an exact zero for finite p).
//   features    (F15.4, ticket 23) a layer is an ordinary material, so its feature blocks blend too, in parameter space with ONE BSDF
//               evaluation afterwards.  Per feature F and layer i the source is the layer's own block, else the stack's own block (the
//               "finish" a layer without F inherits), else the neutral value (coat factor 0, sheen colour 0, anisotropy strength 0,
//               iridescence strength 0, specular ior 1.5 / factor 1 / colour 1).  Presence-like scalars (coat factor, sheen colour,
//               anisotropy strength, iridescence strength, thin subsurface weight, specular factor and colour) blend linearly with the
//               final weights; the secondary parameters of a lobe (coat and sheen roughness, the film ior and thickness, the anisotropy
//               direction, the subsurface colour) are weighted by w_i x presence_i so that a layer without the lobe does not dilute them
//               (layerBlendPresence).  At a weight vector e_j every blend returns the layer's own value bit for bit (exact select).
//   normals     surface-gradient blending (Mikkelsen 2020): a layer's tangent-space normal t = (tx, ty, tz) (tz > 0) is
//               the gradient g = (tx / tz, ty / tz) with n ~ N + g.x T + g.y B; gradients add with the final weights, the
//               result is returned as the tangent-space vector (sum w g, 1) that the caller maps with its own frame:
//               normalize(T a + B b + N c).  At a weight vector e_j the layer's raw t is returned (the exact path: the
//               caller's arithmetic is the one of the single material), a flat layer (0, 0, 1) flattens its share of the
//               relief (that is the specified behaviour of gradient blending).
//
// Weights, heights and normals use the LAYER_MAX = 4 arrays below; `count` bounds the used layers.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"

#ifdef __METAL_VERSION__
#define PHOSPHOR_LAYER_FN static inline __attribute__((unused))
#define PHOSPHOR_LAYER_EXP(x) metal::exp(x)
#else
#include <cmath>
#define PHOSPHOR_LAYER_FN inline
#define PHOSPHOR_LAYER_EXP(x) std::exp(x)
#endif

namespace phosphor {

/// Layers per stack (mirrors MATERIAL_LAYER_MAX of material_description.h and GPUMaterialExt3).
PHOSPHOR_GPU_CONSTANT u32 LAYER_STACK_MAX = 4;
/// Height softness S in metres: a height difference of S weighs a layer e times more (the contrast slope).
PHOSPHOR_GPU_CONSTANT float LAYER_HEIGHT_SOFTNESS_M = 0.005f;

struct LayerWeights {
    float w[4];
};
struct LayerHeights {
    float h[4];
};
struct LayerNormals {
    float tx[4], ty[4], tz[4]; // tangent-space unit-ish vectors (tz > 0 for a valid texel)
};
struct LayerNormalResult {
    float a, b, c; // map with normalize(T a + B b + N c)
};

/// Clamp (NaN -> 0), force the unused layers to 0 and normalise to sum 1; all zero -> layer 0.
PHOSPHOR_LAYER_FN LayerWeights layerNormalise(LayerWeights in, u32 count) {
    LayerWeights out{{0.0f, 0.0f, 0.0f, 0.0f}};
    float sum = 0.0f;
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i) {
        const float v = (i < count && in.w[i] > 0.0f) ? in.w[i] : 0.0f; // NaN fails the comparison
        out.w[i] = v;
        sum += v;
    }
    if (!(sum > 0.0f) || sum > 3.0e38f) { // all zero, or an infinite input
        LayerWeights first{{1.0f, 0.0f, 0.0f, 0.0f}};
        return first;
    }
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i) out.w[i] = out.w[i] / sum;
    return out;
}

/// The layer's height in metres: contrast * (texel - 0.5); a negative contrast means "no height" (0).
PHOSPHOR_LAYER_FN float layerHeight(float contrast, float texel) { return contrast >= 0.0f ? contrast * (texel - 0.5f) : 0.0f; }

/// Height contrast on NORMALISED weights.  Layers beyond `count` or with weight 0 stay at 0.
PHOSPHOR_LAYER_FN LayerWeights layerHeightBlend(LayerWeights n, LayerHeights heights, u32 count) {
    float hmax = -3.0e38f;
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i)
        if (i < count && n.w[i] > 0.0f && heights.h[i] > hmax) hmax = heights.h[i];
    LayerWeights e{{0.0f, 0.0f, 0.0f, 0.0f}};
    float sum = 0.0f;
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count && n.w[i] > 0.0f) {
            e.w[i] = n.w[i] * PHOSPHOR_LAYER_EXP((heights.h[i] - hmax) / LAYER_HEIGHT_SOFTNESS_M);
            sum += e.w[i];
        }
    }
    if (!(sum > 0.0f)) return n; // cannot happen for normalised weights; keep the input rather than a NaN
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i) e.w[i] = e.w[i] / sum;
    return e;
}

/// Parameter-space blend of one scalar: the sum in layer order (exact at a weight vector e_j).
PHOSPHOR_LAYER_FN float layerBlendScalar(LayerWeights w, float p0, float p1, float p2, float p3) {
    return ((w.w[0] * p0 + w.w[1] * p1) + w.w[2] * p2) + w.w[3] * p3;
}

/// The index of the layer whose weight is exactly 1 (the endpoint of a transition), else LAYER_STACK_MAX.
PHOSPHOR_LAYER_FN u32 layerSole(LayerWeights w) {
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i)
        if (w.w[i] == 1.0f) return i;
    return LAYER_STACK_MAX;
}

/// Presence-weighted blend of a lobe's secondary parameter p_i (coat roughness, film thickness, one component of a direction ...):
/// sum_i w_i c_i p_i / sum_i w_i c_i with c_i the presence (coat factor, strength ...) of the lobe in layer i; `fallback` when the lobe is
/// absent everywhere (sum w c = 0); the sole layer's own value at an endpoint (exact).
PHOSPHOR_LAYER_FN float layerBlendPresence(LayerWeights w, float c0, float c1, float c2, float c3, float p0, float p1, float p2, float p3,
                                           float fallback) {
    const u32 sole = layerSole(w);
    if (sole < LAYER_STACK_MAX) {
        const float cs[4] = {c0, c1, c2, c3};
        const float ps[4] = {p0, p1, p2, p3};
        return cs[sole] > 0.0f ? ps[sole] : fallback;
    }
    const float d0 = w.w[0] * c0, d1 = w.w[1] * c1, d2 = w.w[2] * c2, d3 = w.w[3] * c3;
    const float sum = ((d0 + d1) + d2) + d3;
    if (!(sum > 0.0f)) return fallback;
    return (((d0 * p0 + d1 * p1) + d2 * p2) + d3 * p3) / sum;
}

/// Surface-gradient blend of the layers' tangent-space normals (see the header comment).  `w` are final weights.
PHOSPHOR_LAYER_FN LayerNormalResult layerNormalBlend(LayerWeights w, LayerNormals t) {
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i)
        if (w.w[i] == 1.0f) {
            LayerNormalResult exact{t.tx[i], t.ty[i], t.tz[i]};
            return exact;
        }
    float gx = 0.0f, gy = 0.0f;
    for (u32 i = 0; i < LAYER_STACK_MAX; ++i) {
        if (!(w.w[i] > 0.0f)) continue;
        const float tz = t.tz[i] > 1.0e-3f ? t.tz[i] : 1.0e-3f; // a degenerate texel keeps a bounded gradient
        gx += w.w[i] * (t.tx[i] / tz);
        gy += w.w[i] * (t.ty[i] / tz);
    }
    LayerNormalResult out{gx, gy, 1.0f};
    return out;
}

} // namespace phosphor
