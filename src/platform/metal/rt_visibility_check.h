#pragma once

// Shared host/MSL diagnostic ABI. This header intentionally exposes only its
// scalar layouts to MSL; the backend class is guarded below.
#include "renderer/gpu_types.h"

namespace phosphor {
struct GPURtVisibilityParams {
    float viewProjection[16];
    float inverseViewProjection[16];
    u32 width, height, candidateCapacity, slotCount;
    u32 materialCount, rayCount, frameLo, frameHi;
    float relativeDistanceTolerance;
    u32 pad[3];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtVisibilityParams) == 176, "RT visibility params ABI");

struct GPURtVisibilityCategory {
    u32 compared, mismatches, hitMiss, slot, depth;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtVisibilityCategory) == 20, "RT visibility category ABI");

struct GPURtVisibilityCounters {
    u32 compared, mismatches, hitMiss, slot;
    u32 depth, bothBackground, bothHit, rasterHits;
    u32 rtHits, edgePixels, maskPixels, maskNeighborPixels;
    u32 invalidVisibility, invalidRt, invalidDepth, depthWithoutVisibility;
    u32 depthBitDifferent, maxDepthUlp, maxDepthErrorBits, maxRelativeDistanceErrorBits;
    // Disjoint bins for valid both-hit raw depth ULP distance:
    // exactly 0, 1, 2..4, 5..16, 17..64, 65..256, >256.
    u32 depthUlpHistogram[7];
    // 0 interior opaque, 1 edge opaque, 2 interior MASK, 3 edge MASK.
    // All pixels enter exactly one category; none are excluded from totals.
    GPURtVisibilityCategory category[4];
    u32 frameLo, frameHi, width, height;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtVisibilityCounters) == 204, "RT visibility counters ABI");
} // namespace phosphor

#ifndef __METAL_VERSION__
#include "pipeline/pipeline_registry.h"
#include "platform/metal/metal_context.h"
#include "rendergraph/render_graph.h"
#include <array>

namespace phosphor {
class PipelineCache;
class SceneRenderer;
class MeshRenderer;
class AccelerationStructures;
class MetalGraphExecutor;

// Full-resolution primary RT versus the SAME FRAME's V-buffer and depth.
// Ray t is compared with the raster surface reconstructed via inverse(VP):
// |tRT-tRaster| > tolerance * max(1, |tRaster|). Raw float bits are diagnostic,
// never by themselves a geometric mismatch. Edge/MASK categorization explains
// populations; it neither excuses nor silently drops a mismatch.
class RtVisibilityChecker {
public:
    static constexpr float RelativeDistanceTolerance = 2e-4f; // F9-S0 reference contract
    struct Result {
        bool valid = false;
        u64 frame = 0;
        u32 width = 0, height = 0;
        u32 compared = 0, mismatches = 0, hitMiss = 0, slot = 0, depth = 0;
        u32 bothBackground = 0, bothHit = 0, rasterHits = 0, rtHits = 0;
        u32 edgePixels = 0, maskPixels = 0, maskNeighborPixels = 0;
        u32 invalidVisibility = 0, invalidRt = 0, invalidDepth = 0, depthWithoutVisibility = 0;
        u32 depthBitDifferent = 0, maxDepthUlp = 0;
        float maxDepthError = 0, maxRelativeDistanceError = 0;
        float relativeDistanceTolerance = RelativeDistanceTolerance;
        std::array<u32, 7> depthUlpHistogram{};
        std::array<GPURtVisibilityCategory, 4> category{};
    };

    RtVisibilityChecker(MetalContext&, PipelineCache&, SceneRenderer&, MeshRenderer&, AccelerationStructures&);
    ~RtVisibilityChecker();
    RtVisibilityChecker(const RtVisibilityChecker&) = delete;
    RtVisibilityChecker& operator=(const RtVisibilityChecker&) = delete;
    // Call after scene/mesh/AS prepareFrame, with the raster's exact (possibly
    // jittered) camera. Root must select full pixel rays (pad=tid), never the
    // sampled --debug-rt ray set or a secondary probe's output.
    void prepareFrame(u32 slot, u32 width, u32 height, const FrameConstants&);
    void addToGraph(rg::RenderGraph&, rg::TextureRef visibility, rg::TextureRef depth);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] bool ready() const;
    // Read only AFTER this slot's GPU completion and BEFORE its next prepare.
    // An unavailable/incomplete counter buffer returns valid=false, not PASS.
    [[nodiscard]] Result result(u32 slot) const;

private:
    MetalContext& context_;
    PipelineCache& pipelines_;
    SceneRenderer& scene_;
    MeshRenderer& mesh_;
    AccelerationStructures& rt_;
    struct Frame {
        MTL::Buffer* counters = nullptr;
        MTL::GPUAddress params = 0;
        u64 index = ~u64{0};
        u32 width = 0, height = 0;
        bool encoded = false;
    };
    std::array<Frame, METAL_FRAMES_IN_FLIGHT> frames_{};
    std::array<MTL4::ArgumentTable*, 2> tables_{};
    pipe::PipelineHandle clear_, compare_;
    u32 slot_ = 0;
    rg::BufferRef countersRef_;
    rg::TextureRef visibilityRef_, depthRef_;
};
} // namespace phosphor
#endif
