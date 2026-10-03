#pragma once
#include "renderer/normal_transform.h"
static float3 surfaceNormal(float4x4 m, float3 n) {
    const auto result = phosphor::transformSurfaceNormal(m[0].x, m[0].y, m[0].z, m[1].x, m[1].y, m[1].z, m[2].x, m[2].y,
                                                         m[2].z, n.x, n.y, n.z);
    return float3(result.x, result.y, result.z);
}
