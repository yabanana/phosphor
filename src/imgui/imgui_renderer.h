#pragma once

#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/metal_context.h"


struct ImDrawData;

namespace phosphor {

class PipelineCache;

// ---------------------------------------------------------------------------
// ImGuiRenderer -- Dear ImGui renderer backend on Metal 4.
//
// Replaces imgui_impl_metal (Metal 3, second queue, manual retain/release).
// The overlay is appended to an already open render pass on the drawable, so
// it shares the frame's single MTL4 command buffer and never leaves tile
// memory.  Vertices, indices and the projection are written into the frame
// upload ring, like the scene's per-frame data.
// ---------------------------------------------------------------------------

class ImGuiRenderer {
public:
    /// Requires a current ImGui context; uploads the font atlas.
    ImGuiRenderer(MetalContext& context, PipelineCache& pipelines);
    ~ImGuiRenderer();

    ImGuiRenderer(const ImGuiRenderer&) = delete;
    ImGuiRenderer& operator=(const ImGuiRenderer&) = delete;

    /// Encode `drawData` into `encoder`, an open pass on the frame's drawable.
    void render(MTL4::RenderCommandEncoder* encoder, const ImDrawData* drawData);

private:
    void buildPipeline();
    void createFontTexture();
    void setupRenderState(MTL4::RenderCommandEncoder* encoder, MTL::GPUAddress vertices, MTL::GPUAddress uniforms);

    MetalContext& context_;
    PipelineCache& pipelines_;

    pipe::PipelineHandle      pipeline_    = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle pipelineEDR_ = pipe::INVALID_PIPELINE;
    MTL::DepthStencilState*   depthState_  = nullptr;
    MTL4::ArgumentTable*      arguments_   = nullptr;
    MTL::Texture*             fontTexture_ = nullptr;
};

} // namespace phosphor
