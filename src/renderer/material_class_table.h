#pragma once

// F15.2 (ticket 21): the per-scene class table of the per-class tile binning (visibility resolve).
//
// A class is the feature mask of one material (materialFeatureBits).  The table holds up to VISIBILITY_FEATURE_CLASSES
// (= the generated active-class budget) classes, one bin each, ordered by descending material count and then
// ascending mask (deterministic); a scene with more classes keeps the most used ones and its other materials fall in the
// overflow bin, which runs the all-lobes pipeline (logged by the engine).  Portable: header only, unit tested.

#include "renderer/gpu_types.h"
#include "renderer/material_extension_pack.h"
#include "renderer/visibility_layout.h"

#include <algorithm>
#include <array>
#include <map>
#include <span>
#include <vector>

namespace phosphor {

struct MaterialClassTable {
    u32 count = 0;                                            // classes with a bin
    std::array<u32, VISIBILITY_FEATURE_CLASSES> masks{};      // feature mask of each bin (the classification key)
    std::array<u32, VISIBILITY_FEATURE_CLASSES> materials{};  // live materials in each bin
    u32 overflowClasses = 0;                                  // distinct classes beyond the budget (served by the overflow bin)
    u32 overflowMaterials = 0;                                // materials of those classes
    [[nodiscard]] bool overflow() const { return overflowClasses > 0; }
    /// Bin of a material feature mask, VISIBILITY_FEATURE_CLASSES (the overflow bin) when the table has no such class.
    [[nodiscard]] u32 binOf(u32 mask) const {
        for (u32 i = 0; i < count; ++i)
            if (masks[i] == mask) return i;
        return VISIBILITY_FEATURE_CLASSES;
    }
    bool operator==(const MaterialClassTable&) const = default;
};

[[nodiscard]] inline MaterialClassTable buildMaterialClassTable(std::span<const GPUMaterial> materials, u32 budget = VISIBILITY_FEATURE_CLASSES) {
    std::map<u32, u32> perMask;
    for (const GPUMaterial& m : materials) ++perMask[materialFeatureBits(m)];
    std::vector<std::pair<u32, u32>> ranked(perMask.begin(), perMask.end()); // (mask, materials), ascending mask
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    MaterialClassTable t;
    const u32 limit = std::min<u32>(budget, VISIBILITY_FEATURE_CLASSES);
    for (const auto& [mask, count] : ranked) {
        if (t.count < limit) {
            t.masks[t.count] = mask;
            t.materials[t.count] = count;
            ++t.count;
        } else {
            ++t.overflowClasses;
            t.overflowMaterials += count;
        }
    }
    return t;
}

} // namespace phosphor
