#pragma once

#include "platform/metal/metal_context.h"
#include "platform/metal/pipeline_cache.h"
#include "rendergraph/render_graph.h"

#include <array>
#include <string>
#include <vector>

namespace phosphor {
class MetalGraphExecutor;

// F12 reference comparison readback. The source MUST be linear pre-exposure
// HDR, before tonemapping/upscaling/UI. GPU float4 readback preserves values
// from RGBA16Float or RGBA32Float; PFM writes only RGB, as linear float32.
// Explicit scalar captures accept R16Float/R32Float and replicate R to RGB.
// Scalar values retain their own units (e.g. dimensionless AO), never radiance.
class LinearCapture {
  public:
    struct Config {
        std::string path;
        std::string sequence;
        u64 frame = 0;
        u32 every = 1;
        bool scalar = false;
    };
    LinearCapture(MetalContext& context, PipelineCache& pipelines, Config config);
    ~LinearCapture();
    LinearCapture(const LinearCapture&) = delete;
    LinearCapture& operator=(const LinearCapture&) = delete;

    // Call after the slot is reusable and consume(slot) drained its old tag.
    // Dimensions are the active linear source extent, not the output drawable.
    void prepareFrame(u32 slot, u64 index, u32 width, u32 height);
    void addToGraph(rg::RenderGraph& graph, rg::TextureRef linearHdr);
    void bindFrame(MetalGraphExecutor& executor);
    // Call only after frameEvent reached captured index+1, BEFORE scene reload
    // or slot reuse. Returns false with no pending tag. Early reads, a missed
    // consume, malformed input and output failures throw; none are hidden.
    bool consume(u32 slot);
    // Include in the graph key. Changes when active toggles or active extent
    // changes. The capture pass is absent on nonmatching frames.
    [[nodiscard]] u64 version() const { return version_; }

  private:
    struct Slot {
        MTL::Buffer* readback = nullptr;
        MTL4::ArgumentTable* table = nullptr;
        u64 capacityPixels = 0, index = 0;
        u32 width = 0, height = 0;
        bool pending = false, single = false, sequence = false;
        std::vector<float> rgb;
    };
    MetalContext& context_;
    PipelineCache& pipelines_;
    Config config_;
    std::array<Slot, METAL_FRAMES_IN_FLIGHT> slots_{};
    pipe::PipelineHandle pipeline_ = pipe::INVALID_PIPELINE;
    rg::BufferRef readbackRef_{};
    u64 version_ = 1;
    u32 currentSlot_ = 0, currentWidth_ = 0, currentHeight_ = 0;
    MTL::GPUAddress extentAddress_ = 0;
    bool active_ = false, prepared_ = false, singleWritten_ = false;
};
} // namespace phosphor
