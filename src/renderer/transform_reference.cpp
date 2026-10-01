#include "renderer/transform_reference.h"

#include "renderer/transform_math.h"

#include <cstring>

namespace phosphor {

void referenceWorlds(const HierarchyView& v, const float* sinCos, std::vector<float>& world) {
    const u32 slots = static_cast<u32>(v.instances.size());
    world.assign(size_t(slots) * 16, 0.0f);
    for (u32 s = 0; s < slots; ++s) std::memcpy(&world[size_t(s) * 16], v.instances[s].modelMatrix, 16 * sizeof(float));

    for (const u32 s : v.motionSlots) {
        const GPUMotion& m = v.motions[s];
        const u32 k = m.speedClass < SCENE_MOTION_CLASSES ? m.speedClass : SCENE_MOTION_CLASSES - 1;
        motionWorld(m, sinCos[2 * k], sinCos[2 * k + 1], &world[size_t(s) * 16]);
    }

    if (v.childOffsets.size() < size_t(slots) + 1) return; // no hierarchy
    // Breadth-first from the roots that have children: a parent's world is
    // final before its children are processed (depth order, like the GPU levels).
    std::vector<u8> isChild(slots, 0);
    for (const u32 c : v.childSlots) isChild[c] = 1;
    std::vector<u32> level, next;
    for (u32 s = 0; s < slots; ++s)
        if (!isChild[s] && v.childOffsets[s + 1] > v.childOffsets[s]) level.push_back(s);
    float tmp[16];
    for (u32 depth = 0; !level.empty() && depth < SCENE_MAX_LEVELS; ++depth) {
        next.clear();
        for (const u32 p : level) {
            for (u32 i = v.childOffsets[p]; i < v.childOffsets[p + 1]; ++i) {
                const u32 c = v.childSlots[i];
                mat4Mul(&world[size_t(p) * 16], v.nodes[c].local, tmp);
                std::memcpy(&world[size_t(c) * 16], tmp, sizeof(tmp));
                if (v.childOffsets[c + 1] > v.childOffsets[c]) next.push_back(c);
            }
        }
        level.swap(next);
    }
}

} // namespace phosphor
