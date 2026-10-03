#pragma once

#include "renderer/gpu_types.h"

namespace phosphor {

PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_TRIANGLE_BITS = 7;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_TRIANGLE_MASK = 127;
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_BACKGROUND = 0u;
// Bias packed IDs by one so integer render targets can clear to zero.
// Reserve the last cluster entirely to keep the bias from overflowing.
PHOSPHOR_GPU_CONSTANT u32 VISIBILITY_CLUSTER_LIMIT = (1u << 25) - 1;

inline u32 visibilityPack(u32 cluster, u32 triangle) {
    return cluster < VISIBILITY_CLUSTER_LIMIT && triangle <= VISIBILITY_TRIANGLE_MASK
               ? ((cluster << VISIBILITY_TRIANGLE_BITS) | triangle) + 1u
               : VISIBILITY_BACKGROUND;
}
inline u32 visibilityCluster(u32 id) {
    return (id - 1u) >> VISIBILITY_TRIANGLE_BITS;
}
inline u32 visibilityTriangle(u32 id) {
    return (id - 1u) & VISIBILITY_TRIANGLE_MASK;
}

struct VisibilityBarycentrics {
    float value[3];
    float dx[3];
    float dy[3];
    u32 valid;
};

// Invert the homogeneous screen triangle, never dividing an individual
// vertex by w. This remains defined for a triangle crossing the near plane
// and even for a vertex with w == 0. Pixel coordinates have a top-left origin.
// The rational quotient gives perspective-correct weights AND derivatives.
inline VisibilityBarycentrics visibilityBarycentrics(float ax, float ay, float aw, float bx, float by, float bw,
                                                     float cx, float cy, float cw, float px, float py, float width,
                                                     float height) {
    VisibilityBarycentrics out{};
    if (!(width > 0.0f && height > 0.0f))
        return out;
    const float rows[9] = {by * cw - bw * cy, bw * cx - bx * cw, bx * cy - by * cx,
                           cy * aw - cw * ay, cw * ax - cx * aw, cx * ay - cy * ax,
                           ay * bw - aw * by, aw * bx - ax * bw, ax * by - ay * bx};
    const float det = ax * rows[0] + ay * rows[1] + aw * rows[2];
    if (!(det > 1e-20f || det < -1e-20f))
        return out;
    const float x = px * (2.0f / width) - 1.0f;
    const float y = 1.0f - py * (2.0f / height);
    float q[3], gx[3], gy[3];
    float sum = 0.0f, sx = 0.0f, sy = 0.0f;
    for (u32 i = 0; i < 3; ++i) {
        q[i] = rows[i * 3] * x + rows[i * 3 + 1] * y + rows[i * 3 + 2];
        gx[i] = rows[i * 3] * (2.0f / width);
        gy[i] = rows[i * 3 + 1] * (-2.0f / height);
        sum += q[i];
        sx += gx[i];
        sy += gy[i];
    }
    if (!(sum > 1e-20f || sum < -1e-20f))
        return out;
    const float reciprocal = 1.0f / sum;
    for (u32 i = 0; i < 3; ++i) {
        out.value[i] = q[i] * reciprocal;
        out.dx[i] = (gx[i] - out.value[i] * sx) * reciprocal;
        out.dy[i] = (gy[i] - out.value[i] * sy) * reciprocal;
    }
    out.valid = 1;
    return out;
}

} // namespace phosphor
