#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"
#include "renderer/gpu_types.h"

#include <array>
#include <string>

namespace phosphor {

class GpuScene;
struct FrameScene;

// ---------------------------------------------------------------------------
// SceneRenderer -- F0 forward pass on Metal 4.
//
// Static geometry lives in private buffers rebuilt when the GpuScene's
// geometry version changes.  Per-frame data (constants, instances,
// materials, lights) is written into a shared buffer owned by the frame slot,
// so the CPU never touches memory the GPU may still be reading.  Depth is a
// memoryless attachment: on a TBDR GPU it never leaves tile memory.
// ---------------------------------------------------------------------------

class SceneRenderer {
public:
    SceneRenderer(MetalContext& context, const std::string& libraryPath);
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    /// Upload vertex/index data if the scene geometry changed (blocking).
    void syncGeometry(const GpuScene& scene);

    /// Encode the forward pass into the frame's command buffer, rendering to
    /// the frame's drawable.
    void render(MetalContext::Frame& frame, const GpuScene& scene, const FrameScene& frameScene,
                const FrameConstants& constants, MTL::GPUAddress textureTable);

    [[nodiscard]] u32 lastTriangleCount() const { return lastTriangles_; }

private:
    struct UploadBuffer {
        MTL::Buffer* buffer = nullptr;
        size_t       capacity = 0;
    };

    void buildPipeline();
    void ensureDepthTarget(u32 width, u32 height);
    MTL::Buffer* createPrivateBuffer(const void* data, size_t size, const char* label);
    void releaseGeometry();

    MetalContext& context_;

    MTL::Library*             library_    = nullptr;
    MTL::RenderPipelineState* pipeline_   = nullptr;
    MTL::DepthStencilState*   depthState_ = nullptr;
    MTL4::ArgumentTable*      arguments_  = nullptr;
    MTL::Texture*             depth_      = nullptr;

    MTL::Buffer* vertexBuffer_ = nullptr;
    MTL::Buffer* indexBuffer_  = nullptr;
    u64          geometryVersion_ = ~u64{0};

    std::array<UploadBuffer, METAL_FRAMES_IN_FLIGHT> uploads_{};
    u32 lastTriangles_ = 0;
};

} // namespace phosphor
