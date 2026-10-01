#include "renderer/cull_reference.h"

#include "renderer/gpu_scene_layout.h"

#include <cassert>
#include <cmath>

namespace phosphor {

std::array<glm::vec4, 5> extractFrustumPlanesReverseZ(const glm::mat4& viewProj) {
    // m[col][row].  Metal clip volume: -w <= x <= w, -w <= y <= w, 0 <= z <= w.
    // Reverse-Z: ndc z = near / depth, so the NEAR plane is z <= w (row3 - row2)
    // and z >= 0 holds at infinity (no far plane).  The OpenGL extraction
    // (near = row3 + row2) is z >= -w: for this projection it is z <= +near, a
    // plane parallel to the near plane on the camera's side: it accepts what
    // is behind the near plane up to the camera and beyond (spike S4).
    const glm::mat4& m = viewProj;
    const auto row     = [&](int r) { return glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]); };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
    std::array<glm::vec4, 5> planes = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2};
    for (glm::vec4& p : planes) {
        const float len = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if (len > 0.0f) p /= len;
    }
    return planes;
}

GPUCullParams makeCullParams(const glm::mat4& viewProj, float proj11, u32 viewportHeight, glm::vec3 cameraPos,
                             glm::vec3 /*cameraForward*/, float nearPlane, u32 flags, float maxDistance, float minPixels,
                             u32 slotCount) {
    GPUCullParams p{};
    const auto planes = extractFrustumPlanesReverseZ(viewProj);
    for (u32 i = 0; i < 5; ++i)
        for (u32 k = 0; k < 4; ++k) p.planes[i * 4 + k] = planes[i][static_cast<int>(k)];
    p.cameraPosition[0] = cameraPos.x;
    p.cameraPosition[1] = cameraPos.y;
    p.cameraPosition[2] = cameraPos.z;
    p.maxDistance       = maxDistance;
    p.minPixels         = minPixels;
    p.pixelScale        = proj11 * static_cast<float>(viewportHeight) / 2.0f;
    p.nearPlane         = nearPlane;
    p.slotCount         = slotCount;
    p.groupCount        = (slotCount + SCENE_CULL_GROUP - 1) / SCENE_CULL_GROUP;
    p.flags             = flags;
    return p;
}

void cullReference(std::span<const GPUInstance> instances, std::span<const GPUMeshInfo> meshes, const GPUCullParams& params,
                   std::vector<u8>& result, std::vector<float>* margins) {
    const u32 n = params.slotCount;
    assert(instances.size() >= n);
    result.assign(n, CULL_RESULT_INVALID);
    if (margins) margins->assign(n, 0.0f);
    for (u32 i = 0; i < n; ++i) {
        const GPUInstance& in = instances[i];
        if ((in.flags & INSTANCE_FLAG_VALID) == 0u) continue;
        assert(in.meshIndex < meshes.size());
        const GPUWorldSphere w = cullWorldSphere(in.modelMatrix, meshes[in.meshIndex].boundingSphere);
        const CullEval e       = cullSphereEval(params, w.x, w.y, w.z, w.r);
        result[i]              = static_cast<u8>(e.reason);
        if (margins) (*margins)[i] = e.margin;
    }
}

void compactReference(std::span<const u8> result, std::vector<u32>& visible, std::vector<u32>& prefix) {
    visible.clear();
    prefix.assign(result.size() + 1, 0);
    u32 run = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        prefix[i] = run;
        if (result[i] == 0) {
            visible.push_back(static_cast<u32>(i));
            ++run;
        }
    }
    prefix[result.size()] = run;
}

void drawArgsReference(std::span<const GPUDrawBucket> buckets, std::span<const u32> commandBuckets, std::span<const u32> prefix,
                       std::vector<u32>& args) {
    args.assign(commandBuckets.size() * 2, 0);
    for (size_t c = 0; c < commandBuckets.size(); ++c) {
        const u32 b = commandBuckets[c];
        if (b == DRAW_COMMAND_SENTINEL) continue;
        assert(b < buckets.size());
        const GPUDrawBucket& k = buckets[b];
        assert(size_t(k.firstSlot) + k.capacity < prefix.size());
        const u32 count = prefix[k.firstSlot + k.capacity] - prefix[k.firstSlot];
        if (count == 0) continue;
        args[c * 2 + 0] = count;
        args[c * 2 + 1] = prefix[k.firstSlot];
    }
}

} // namespace phosphor
