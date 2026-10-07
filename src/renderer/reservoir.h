#pragma once

#include "renderer/gpu_types.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace phosphor::di {

inline u32 historyPixel(u32 x, u32 y, float motionX, float motionY, u32 width, u32 height) {
    if (x >= width || y >= height || !std::isfinite(motionX) || !std::isfinite(motionY) ||
        u64(width) * height > std::numeric_limits<u32>::max()) return ~0u;
    const double px = double(x) + 0.5 + motionX, py = double(y) + 0.5 + motionY;
    if (px < 0 || py < 0 || px >= width || py >= height) return ~0u;
    return u32(py) * width + u32(px);
}

// A reservoir integrates over the union of discrete lights and area endpoints.
// pHat is positive on the complete proposal support (see F11-CONTRACT.md).
// M counts proposals, INCLUDING proposals whose final contribution is black.
// A source reservoir carries normalized W, never a cached visibility value.
inline bool stream(GPUDIReservoir& r, const GPUDIReservoir& sample, double weight,
                   u32 multiplicity, double random) {
    if (!std::isfinite(weight) || weight < 0.0 || !std::isfinite(random) ||
        random < 0.0 || random >= 1.0 || multiplicity == 0 ||
        multiplicity > std::numeric_limits<u32>::max() - r.M) {
        r.pad[0] |= DI_ERROR_WEIGHT;
        return false;
    }
    const double total = double(r.weightSum) + weight;
    if (!std::isfinite(total) || total > std::numeric_limits<float>::max()) { r.pad[0] |= DI_ERROR_WEIGHT; return false; }
    r.M += multiplicity;
    r.pad[1]|=DI_PROPOSAL_VALID;
    if (weight > 0.0 && random * total < weight) {
        r.lightIndex = sample.lightIndex;
        r.lightID = sample.lightID;
        r.lightGeneration = sample.lightGeneration;
        r.lightRevision = sample.lightRevision;
        r.u = sample.u;
        r.v = sample.v;
        r.target = sample.target;
        r.age = sample.age;
        r.valid = 1;
    }
    r.weightSum = float(total);
    r.normalization = 0.0f; // only finalize makes the reservoir consumable
    return true;
}

inline bool finalize(GPUDIReservoir& r) {
    if(!std::isfinite(r.weightSum)||!std::isfinite(r.target)||r.weightSum<0||r.target<0){
        r.pad[0]|=DI_ERROR_WEIGHT;r.pad[1]&=~DI_PROPOSAL_VALID;r.valid=0;r.normalization=0;return false;
    }
    const double denominator = double(r.M) * double(r.target);
    const double w = denominator > 0 ? double(r.weightSum) / denominator : 0;
    if (!r.valid || r.pad[0] != 0 || !std::isfinite(w) || w <= 0 || w > std::numeric_limits<float>::max()) {
        r.normalization = 0;
        r.valid = 0;
        return false;
    }
    r.normalization = float(w);
    return true;
}

inline bool reusable(const GPUDIReservoir& source, const GPUDIParams& p,
                     const GPUSampledLight& light) {
    const bool zero=(source.pad[1]&DI_PROPOSAL_VALID) && !source.valid && source.weightSum==0 && source.normalization==0 && std::isfinite(source.target);
    return (zero || source.valid) && source.pad[0] == 0 && source.M > 0 && source.age < p.maxHistoryAge &&
           source.viewID == p.viewID && source.historyEpoch == p.historyEpoch &&
           source.lightRevision == p.lightRevision && (zero || (source.lightID == light.id &&
           source.lightGeneration == light.generation && source.target > 0 &&
           std::isfinite(source.target) && source.normalization > 0 &&
           std::isfinite(source.normalization) && std::isfinite(source.u) && std::isfinite(source.v) &&
           source.u >= 0 && source.u < 1 && source.v >= 0 && source.v < 1));
}

inline bool merge(GPUDIReservoir& destination, GPUDIReservoir source,
                  float currentTarget, u32 maxM, double random, bool advanceAge = true) {
    if((source.pad[1]&DI_PROPOSAL_VALID) && !source.valid && source.pad[0]==0 && source.M &&
       source.weightSum==0 && source.normalization==0 && std::isfinite(source.target)) {
        return stream(destination,source,0,std::min(source.M,maxM),random);
    }
    if (!source.valid || !std::isfinite(currentTarget) || currentTarget <= 0 ||
        !std::isfinite(source.normalization) || source.normalization <= 0)
        return false;
    const u32 m = std::min(source.M, maxM);
    source.target = currentTarget;
    if (advanceAge && source.age != std::numeric_limits<u32>::max()) ++source.age;
    return stream(destination, source, double(currentTarget) * source.normalization * m, m, random);
}

inline bool compatible(const GPUDISurface& a, const GPUDISurface& b,
                       float depthRelativeThreshold, float normalThreshold, bool temporal) {
    if (!a.valid || !b.valid || !std::isfinite(a.depth) || !std::isfinite(b.depth) ||
        a.depth <= 0 || b.depth <= 0 || !std::isfinite(depthRelativeThreshold) ||
        depthRelativeThreshold < 0 || !std::isfinite(normalThreshold) || normalThreshold < -1 || normalThreshold > 1)
        return false;
    if (temporal && (a.instanceSlot != b.instanceSlot || a.instanceGeneration != b.instanceGeneration ||
                     a.materialRevision != b.materialRevision)) return false;
    double dot = 0, aa = 0, bb = 0, plane = 0;
    for (u32 i = 0; i < 3; ++i) {
        if (!std::isfinite(a.geometricNormal[i]) || !std::isfinite(b.geometricNormal[i]) ||
            !std::isfinite(a.position[i]) || !std::isfinite(b.position[i])) return false;
        dot += double(a.geometricNormal[i]) * b.geometricNormal[i];
        aa += double(a.geometricNormal[i]) * a.geometricNormal[i];
        bb += double(b.geometricNormal[i]) * b.geometricNormal[i];
        plane += (double(b.position[i]) - a.position[i]) * a.geometricNormal[i];
    }
    if (!(aa > 0 && bb > 0) || dot / std::sqrt(aa * bb) < normalThreshold) return false;
    const double tolerance = double(depthRelativeThreshold) * std::max(a.depth, b.depth);
    return std::abs(double(a.depth) - b.depth) <= tolerance && std::abs(plane) / std::sqrt(aa) <= tolerance;
}

} // namespace phosphor::di
