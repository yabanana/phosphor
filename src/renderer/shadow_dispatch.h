#pragma once
#include "renderer/gpu_types.h"

namespace phosphor {
// Draws use zero baseInstance. casterSlot is the explicit first scene slot;
// meshletFirst/count address only this draw's chunk of its bucket's mesh.
inline u32 shadowDrawSlot(u32 first, u32 local, u32 totalSlots) {
    return first < totalSlots && local < totalSlots - first ? first + local : ~0u;
}
inline u32 shadowDrawMeshlet(u32 first, u32 local, u32 count) {
    return local < count && first <= ~0u - local ? first + local : ~0u;
}
} // namespace phosphor

#ifndef __METAL_VERSION__
#include "renderer/scene_store.h"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace phosphor {
struct ShadowCasterDraw {
    u32 mesh = 0;
    u32 firstSlot = 0, slotCount = 0;
    u32 meshletFirst = 0, meshletCount = 0;
    CullClass cull = CullClass::Back;
    bool meshShader = false;
};

// SceneStore owns disjoint bucket ranges. `used` is the high-water prefix of
// LIVE + HOLE slots; `count` is only the live population and cannot bound a
// draw after removals. GPU flags still reject holes/noncasters/per-cascade
// misses. No camera-visible list is consulted. Output capacity is reused.
inline void planShadowCasterDraws(std::span<const SceneBucket> buckets, std::span<const GPUMeshInfo> meshes,
                                  u32 totalSlots, bool allowMesh, std::vector<ShadowCasterDraw>& out,
                                  u32 meshletAxisLimit = 65535, u32 instanceAxisLimit = 65535) {
    if (!meshletAxisLimit || !instanceAxisLimit || meshletAxisLimit > 65535 || instanceAxisLimit > 65535)
        throw std::invalid_argument("shadow draw axis limits must be 1..65535");
    out.clear();
    for (const auto& bucket : buckets) {
        if (bucket.count > bucket.used || bucket.used > bucket.capacity ||
            u64(bucket.firstSlot) + bucket.capacity > totalSlots || bucket.mesh >= meshes.size())
            throw std::invalid_argument("shadow bucket exceeds its scene-owned range");
        if (!bucket.count || !bucket.used) continue;
        const auto& mesh = meshes[bucket.mesh];
        if (mesh.indexCount % 3 || u64(mesh.meshletOffset) + mesh.meshletCount > u64(~0u))
            throw std::invalid_argument("shadow mesh metadata is invalid");
        if (!mesh.indexCount) continue;
        const bool meshShader = allowMesh && mesh.meshletCount != 0;
        for (u32 firstSlot = 0; firstSlot < bucket.used;) {
            const u32 slots = std::min(instanceAxisLimit, bucket.used - firstSlot);
            if (meshShader) {
                for (u32 firstMeshlet = 0; firstMeshlet < mesh.meshletCount;) {
                    const u32 meshlets = std::min(meshletAxisLimit, mesh.meshletCount - firstMeshlet);
                    out.push_back({bucket.mesh, bucket.firstSlot + firstSlot, slots,
                                   mesh.meshletOffset + firstMeshlet, meshlets, bucket.cull, true});
                    firstMeshlet += meshlets;
                }
            } else {
                // Indexed fallback draws the complete mesh exactly once per
                // slot chunk, never once per meshlet chunk.
                out.push_back({bucket.mesh, bucket.firstSlot + firstSlot, slots,
                               mesh.meshletOffset, mesh.meshletCount, bucket.cull, false});
            }
            firstSlot += slots;
        }
    }
}
} // namespace phosphor
#endif
