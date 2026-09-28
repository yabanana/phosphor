#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"
#include "renderer/gpu_types.h"


namespace phosphor {

class GpuScene;
struct FrameScene;

// ---------------------------------------------------------------------------
// SceneRenderer -- F0 forward pass on Metal 4.
//
// Static geometry lives in private buffers rebuilt when the GpuScene's
// geometry version changes.  Per-frame data (constants, instances,
// materials, lights) is written into the context's frame upload ring, so the
// CPU never touches memory the GPU may still be reading.  Depth is a
// memoryless attachment: on a TBDR GPU it never leaves tile memory.
//
// render() leaves the pass open so the overlay (ImGui) is drawn into the same
// pass: the drawable stays in tile memory instead of a store + reload.
// ---------------------------------------------------------------------------

class SceneRenderer {
public:
    explicit SceneRenderer(MetalContext& context);
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    /// Upload vertex/index data if the scene geometry changed (blocking).
    void syncGeometry(const GpuScene& scene);

    /// Begin the forward pass on the frame's drawable and encode the scene.
    /// Returns the still-open encoder; the caller appends the overlay and
    /// calls endEncoding().
    [[nodiscard]] MTL4::RenderCommandEncoder* render(MetalContext::Frame& frame, const GpuScene& scene, const FrameScene& frameScene,
                const FrameConstants& constants, MTL::GPUAddress textureTable);

    [[nodiscard]] u32 lastTriangleCount() const { return lastTriangles_; }
    [[nodiscard]] static MTL::PixelFormat depthFormat() { return MTL::PixelFormatDepth32Float; }

private:
    void buildPipeline();
    void ensureDepthTarget(u32 width, u32 height);
    MTL::Buffer* createPrivateBuffer(const void* data, size_t size, const char* label);
    void releaseGeometry();

    MetalContext& context_;

    MTL::RenderPipelineState* pipeline_   = nullptr;
    MTL::DepthStencilState*   depthState_ = nullptr;
    MTL4::ArgumentTable*      arguments_  = nullptr;
    MTL4::RenderPassDescriptor* passDesc_ = nullptr;
    NS::String*               passLabel_  = nullptr;
    MTL::Texture*             depth_      = nullptr;

    MTL::Buffer* vertexBuffer_ = nullptr;
    MTL::Buffer* indexBuffer_  = nullptr;
    u64          geometryVersion_ = ~u64{0};

    u32 lastTriangles_ = 0;
};

} // namespace phosphor
