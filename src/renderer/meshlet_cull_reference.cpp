#include "renderer/meshlet_cull_reference.h"

#include <algorithm>

namespace phosphor {

HiZPyramid buildHiZReference(const float* depth, u32 width, u32 height) {
    HiZPyramid p;
    p.width0  = hizLevel0Size(width);
    p.height0 = hizLevel0Size(height);
    p.levels  = hizLevelCount(p.width0, p.height0);
    p.level.resize(p.levels);

    p.level[0].assign(static_cast<size_t>(p.width0) * p.height0, 1.0f);
    for (u32 y = 0; y < p.height0; ++y) {
        for (u32 x = 0; x < p.width0; ++x) {
            float m = 1.0f;
            for (u32 dy = 0; dy < 2; ++dy) {
                for (u32 dx = 0; dx < 2; ++dx) {
                    const u32 px = 2 * x + dx, py = 2 * y + dy;
                    if (px < width && py < height) m = std::min(m, depth[static_cast<size_t>(py) * width + px]);
                }
            }
            p.level[0][static_cast<size_t>(y) * p.width0 + x] = m;
        }
    }
    for (u32 l = 1; l < p.levels; ++l) {
        const u32 w = hizLevelSize(p.width0, l), h = hizLevelSize(p.height0, l);
        const u32 pw = hizLevelSize(p.width0, l - 1), ph = hizLevelSize(p.height0, l - 1);
        p.level[l].assign(static_cast<size_t>(w) * h, 1.0f);
        for (u32 y = 0; y < h; ++y) {
            for (u32 x = 0; x < w; ++x) {
                float m = 1.0f;
                for (u32 dy = 0; dy < 2; ++dy) {
                    for (u32 dx = 0; dx < 2; ++dx) {
                        const u32 sx = 2 * x + dx, sy = 2 * y + dy;
                        if (sx < pw && sy < ph) m = std::min(m, p.level[l - 1][static_cast<size_t>(sy) * pw + sx]);
                    }
                }
                p.level[l][static_cast<size_t>(y) * w + x] = m;
            }
        }
    }
    return p;
}

float hizMinOverFootprint(const HiZPyramid& pyramid, const HiZFootprint& f) {
    float m = 1.0f;
    for (u32 y = f.y0; y <= f.y1; ++y)
        for (u32 x = f.x0; x <= f.x1; ++x) m = std::min(m, pyramid.at(f.level, x, y));
    return m;
}

float hizOcclusionMargin(float nearestDepth, float minTexel) {
    return (minTexel - nearestDepth * (1.0f + HIZ_DEPTH_EPS)) / std::max(minTexel, 1e-30f);
}

u32 meshletDecisionPhaseA(const GPUMeshletCullParams& p, const MeshletDecisionInput& in, const HiZPyramid* history) {
    const float* model     = in.instance->modelMatrix;
    const GPUWorldSphere w = meshletWorldSphere(model, *in.bounds);
    if ((p.flags & MESHLET_CULL_FRUSTUM) != 0u && meshletFrustumCulled(p, w)) return MESHLET_DECISION_FRUSTUM;
    if ((p.flags & MESHLET_CULL_CONE) != 0u && in.cullClass != 2u && meshletConeCulled(model, *in.bounds, p.cameraPosition))
        return MESHLET_DECISION_CONE;
    if ((p.flags & MESHLET_CULL_SIZE) != 0u) {
        const HiZFootprint f = hizFootprint(p.viewProj, w, p);
        if (f.usable != 0u && f.area < p.minPixels * p.minPixels) return MESHLET_DECISION_SIZE;
    }
    if ((p.flags & MESHLET_CULL_OCCLUSION) != 0u && (p.flags & MESHLET_CULL_HISTORY_VALID) != 0u && history != nullptr) {
        const HiZFootprint f = hizFootprint(p.prevViewProj, w, p);
        if (f.usable != 0u && hizOccluded(f.nearestDepth, hizMinOverFootprint(*history, f))) return MESHLET_DECISION_HISTORY;
    }
    return MESHLET_DECISION_DRAWN_A;
}

u32 meshletDecisionPhaseB(const GPUMeshletCullParams& p, const MeshletDecisionInput& in, const HiZPyramid& current) {
    const GPUWorldSphere w = meshletWorldSphere(in.instance->modelMatrix, *in.bounds);
    const HiZFootprint f   = hizFootprint(p.viewProj, w, p);
    if (f.usable != 0u && hizOccluded(f.nearestDepth, hizMinOverFootprint(current, f))) return MESHLET_DECISION_OCCLUDED;
    return MESHLET_DECISION_DRAWN_B;
}

MeshletListRef buildCandidatesReference(std::span<const GPUInstance> instances, std::span<const GPUMeshInfo> meshes,
                                        std::span<const GPUMaterial> materials, std::span<const u32> sceneFlags,
                                        u32 slotCount) {
    MeshletListRef r;
    for (u32 c = 0; c < 3; ++c) {
        const u32 first = static_cast<u32>(r.list.size());
        for (u32 s = 0; s < slotCount; ++s) {
            if ((sceneFlags[s] & 1u) == 0u || (instances[s].flags & INSTANCE_FLAG_VALID) == 0u) continue;
            const bool doubleSided = (materials[instances[s].materialIndex].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u;
            const u32 cls = doubleSided ? 2u : ((instances[s].flags & INSTANCE_FLAG_MIRRORED) != 0u ? 1u : 0u);
            if (cls != c) continue;
            const GPUMeshInfo& mesh = meshes[instances[s].meshIndex];
            for (u32 j = 0; j < mesh.meshletCount; ++j) r.list.push_back({s, mesh.meshletOffset + j});
        }
        r.ranges[c] = {first, static_cast<u32>(r.list.size()) - first, c, MESHLET_PHASE_A};
    }
    r.total = static_cast<u32>(r.list.size());
    return r;
}

MeshletListRef buildBListReference(const MeshletListRef& candidates, std::span<const u32> bFlags) {
    MeshletListRef r;
    for (u32 c = 0; c < 3; ++c) {
        const u32 first                = static_cast<u32>(r.list.size());
        const GPUMeshletDrawRange& src = candidates.ranges[c];
        for (u32 i = src.first; i < src.first + src.count; ++i)
            if (bFlags[i] != 0u) r.list.push_back(candidates.list[i]);
        r.ranges[c] = {first, static_cast<u32>(r.list.size()) - first, c, MESHLET_PHASE_B};
    }
    r.total = static_cast<u32>(r.list.size());
    return r;
}

u64 meshletCandidateCapacity(std::span<const SceneBucket> buckets, std::span<const GPUMeshInfo> meshes) {
    u64 total = 0;
    for (const SceneBucket& b : buckets) total += static_cast<u64>(b.capacity) * meshes[b.mesh].meshletCount;
    return total;
}

} // namespace phosphor
