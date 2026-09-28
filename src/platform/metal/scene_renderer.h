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
// CPU never touches memory the GPU may still be reading.
//
// Since F2.7 the pass itself (targets, load/store, memoryless depth, fusion
// with the ImGui overlay) belongs to the render graph: prepareFrame() writes
// the frame's data, encode() records the draws into the graph's encoder.
// Back faces are culled; mirrored instances and double-sided materials come
// in their own batches (DrawBatch::cull).
// ---------------------------------------------------------------------------

class SceneRenderer {
public:
    explicit SceneRenderer(MetalContext& context);
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    /// Upload vertex/index data if the scene geometry changed (blocking).
    void syncGeometry(const GpuScene& scene);

    /// Upload the frame's constants, instances, materials and lights and bind
    /// them; `scene`/`frameScene` must stay alive until encode() is done.
    void prepareFrame(const GpuScene& scene, const FrameScene& frameScene, const FrameConstants& constants,
                      MTL::GPUAddress textureTable, u32 width, u32 height);

    /// Encode draw batches [chunk * n / chunks, (chunk + 1) * n / chunks) into
    /// `encoder`, an open render pass with the color and depth targets.
    /// Chunks may be encoded concurrently on different encoders (F2.5).
    void encode(MTL4::RenderCommandEncoder* encoder, u32 chunk = 0, u32 chunks = 1) const;

    [[nodiscard]] u32 lastTriangleCount() const { return lastTriangles_; }
    [[nodiscard]] static MTL::PixelFormat depthFormat() { return MTL::PixelFormatDepth32Float; }

private:
    void buildPipeline();
    MTL::Buffer* createPrivateBuffer(const void* data, size_t size, const char* label);
    void releaseGeometry();

    MetalContext& context_;

    MTL::RenderPipelineState* pipeline_   = nullptr;
    MTL::DepthStencilState*   depthState_ = nullptr;
    MTL4::ArgumentTable*      arguments_  = nullptr;

    MTL::Buffer* vertexBuffer_ = nullptr;
    MTL::Buffer* indexBuffer_  = nullptr;
    u64          geometryVersion_ = ~u64{0};

    // Frame state from prepareFrame(), read by encode().
    const GpuScene*   scene_      = nullptr;
    const FrameScene* frameScene_ = nullptr;
    u32 width_  = 0;
    u32 height_ = 0;
    u32 lastTriangles_ = 0;
};

} // namespace phosphor
