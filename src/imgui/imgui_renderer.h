#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"

#include <array>

struct ImDrawData;

namespace phosphor {

// ---------------------------------------------------------------------------
// ImGuiRenderer -- Dear ImGui renderer backend on Metal 4.
//
// Replaces imgui_impl_metal (Metal 3, second queue, manual retain/release).
// The overlay is appended to an already open render pass on the drawable, so
// it shares the frame's single MTL4 command buffer and never leaves tile
// memory.  Vertices, indices and the projection are written into a per-slot
// shared buffer, like the scene's per-frame data.
// ---------------------------------------------------------------------------

class ImGuiRenderer {
public:
    /// Requires a current ImGui context; uploads the font atlas.
    explicit ImGuiRenderer(MetalContext& context);
    ~ImGuiRenderer();

    ImGuiRenderer(const ImGuiRenderer&) = delete;
    ImGuiRenderer& operator=(const ImGuiRenderer&) = delete;

    /// Encode `drawData` into `encoder`, an open pass on the frame's drawable.
    void render(const MetalContext::Frame& frame, MTL4::RenderCommandEncoder* encoder, const ImDrawData* drawData);

private:
    struct UploadBuffer {
        MTL::Buffer* buffer   = nullptr;
        size_t       capacity = 0;
    };

    void buildPipeline();
    void createFontTexture();
    void setupRenderState(MTL4::RenderCommandEncoder* encoder, MTL::GPUAddress vertices, MTL::GPUAddress uniforms);

    MetalContext& context_;

    MTL::RenderPipelineState* pipeline_    = nullptr;
    MTL::DepthStencilState*   depthState_  = nullptr;
    MTL4::ArgumentTable*      arguments_   = nullptr;
    MTL::Texture*             fontTexture_ = nullptr;

    std::array<UploadBuffer, METAL_FRAMES_IN_FLIGHT> uploads_{};
};

} // namespace phosphor
