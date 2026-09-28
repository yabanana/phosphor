#include "platform/metal/scene_renderer.h"
#include "platform/metal/gpu_memory.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "core/log.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

constexpr size_t kAlign = 256;

size_t alignUp(size_t v) { return (v + kAlign - 1) & ~(kAlign - 1); }

// Argument table slots; must match shaders/forward.metal.
enum Binding : NS::UInteger {
    BindFrame     = 0,
    BindVertices  = 1,
    BindInstances = 2,
    BindMaterials = 3,
    BindLights    = 4,
    BindTextures  = 5,
    BindCount     = 6,
};

} // namespace

SceneRenderer::SceneRenderer(MetalContext& context)
    : context_(context) {
    buildPipeline();

    MTL::DepthStencilDescriptor* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionGreater); // reverse-Z
    dsDesc->setDepthWriteEnabled(true);
    depthState_ = context_.device()->newDepthStencilState(dsDesc);
    dsDesc->release();

    passDesc_ = MTL4::RenderPassDescriptor::alloc()->init();
    MTL::RenderPassColorAttachmentDescriptor* color = passDesc_->colorAttachments()->object(0);
    color->setLoadAction(MTL::LoadActionClear);
    color->setStoreAction(MTL::StoreActionStore);
    color->setClearColor(MTL::ClearColor::Make(0.02, 0.025, 0.035, 1.0));
    MTL::RenderPassDepthAttachmentDescriptor* depth = passDesc_->depthAttachment();
    depth->setLoadAction(MTL::LoadActionClear);
    depth->setStoreAction(MTL::StoreActionDontCare);
    depth->setClearDepth(0.0); // reverse-Z: far = 0
    passLabel_ = str("Forward")->retain();

    NS::Error* error = nullptr;
    MTL4::ArgumentTableDescriptor* atDesc = MTL4::ArgumentTableDescriptor::alloc()->init();
    atDesc->setMaxBufferBindCount(BindCount);
    atDesc->setLabel(str("Forward arguments"));
    arguments_ = context_.device()->newArgumentTable(atDesc, &error);
    atDesc->release();
    if (!arguments_) {
        throw std::runtime_error("Failed to create argument table");
    }
}

SceneRenderer::~SceneRenderer() {
    context_.waitIdle();
    releaseGeometry();
    context_.memory().release(depth_, MemoryCategory::RenderTargets);
    passLabel_->release();
    passDesc_->release();
    arguments_->release();
    depthState_->release();
    pipeline_->release();
}

void SceneRenderer::buildPipeline() {
    MTL4::LibraryFunctionDescriptor* vs = MTL4::LibraryFunctionDescriptor::alloc()->init();
    vs->setLibrary(context_.library());
    vs->setName(str("forward_vs"));
    MTL4::LibraryFunctionDescriptor* fs = MTL4::LibraryFunctionDescriptor::alloc()->init();
    fs->setLibrary(context_.library());
    fs->setName(str("forward_fs"));

    MTL4::RenderPipelineDescriptor* desc = MTL4::RenderPipelineDescriptor::alloc()->init();
    desc->setLabel(str("Forward"));
    desc->setVertexFunctionDescriptor(vs);
    desc->setFragmentFunctionDescriptor(fs);
    desc->colorAttachments()->object(0)->setPixelFormat(context_.colorFormat());
    // MTL4 render pipelines take no depth format: it is inferred from the pass.

    NS::Error* error = nullptr;
    pipeline_ = context_.compiler()->newRenderPipelineState(desc, nullptr, &error);
    desc->release();
    fs->release();
    vs->release();
    if (!pipeline_) {
        const char* reason = error ? error->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error(std::string("Failed to build forward pipeline: ") + reason);
    }
}

MTL::Buffer* SceneRenderer::createPrivateBuffer(const void* data, size_t size, const char* label) {
    MTL::Buffer* buffer =
        context_.memory().newBuffer(size, MTL::ResourceStorageModePrivate, MemoryCategory::Geometry, label);
    const UploadRing::Slice staging = context_.stagingAllocate(size);
    std::memcpy(staging.cpu, data, size);
    context_.enqueueUpload([staging, buffer, size](MTL4::ComputeCommandEncoder* enc) {
        enc->copyFromBuffer(staging.buffer, staging.offset, buffer, 0, size);
    });
    return buffer;
}

void SceneRenderer::releaseGeometry() {
    for (MTL::Buffer** b : {&vertexBuffer_, &indexBuffer_}) {
        context_.memory().release(*b, MemoryCategory::Geometry);
        *b = nullptr;
    }
}

void SceneRenderer::syncGeometry(const GpuScene& scene) {
    if (scene.geometryVersion() == geometryVersion_) return;
    geometryVersion_ = scene.geometryVersion();

    // Geometry changes only on bench switches, so a full idle is acceptable.
    context_.waitIdle();
    releaseGeometry();
    if (scene.vertices().empty() || scene.indices().empty()) return;

    vertexBuffer_ = createPrivateBuffer(scene.vertices().data(),
                                        scene.vertices().size() * sizeof(GPUVertex), "Vertices");
    indexBuffer_  = createPrivateBuffer(scene.indices().data(),
                                        scene.indices().size() * sizeof(u32), "Indices");
    context_.flushUploads();
    LOG_INFO("Geometry uploaded: %zu vertices, %zu indices",
             scene.vertices().size(), scene.indices().size());
}

void SceneRenderer::ensureDepthTarget(u32 width, u32 height) {
    if (depth_ && depth_->width() == width && depth_->height() == height) return;
    context_.memory().release(depth_, MemoryCategory::RenderTargets);

    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(
        depthFormat(), width, height, false);
    desc->setUsage(MTL::TextureUsageRenderTarget);
    // Depth is consumed within the pass: keep it in tile memory only.
    desc->setStorageMode(MTL::StorageModeMemoryless);
    depth_ = context_.memory().newTexture(desc, MemoryCategory::RenderTargets, "Depth (memoryless)");
}

MTL4::RenderCommandEncoder* SceneRenderer::render(MetalContext::Frame& frame, const GpuScene& scene, const FrameScene& fs,
                           const FrameConstants& constants, MTL::GPUAddress textureTable) {
    MTL::Texture* target = frame.drawable->texture();
    const u32 width  = static_cast<u32>(target->width());
    const u32 height = static_cast<u32>(target->height());
    ensureDepthTarget(width, height);

    // --- Per-frame upload -------------------------------------------------
    const size_t constantsOffset = 0;
    const size_t instancesOffset = alignUp(sizeof(FrameConstants));
    const size_t materialsOffset = instancesOffset + alignUp(fs.instances.size() * sizeof(GPUInstance));
    const size_t lightsOffset    = materialsOffset + alignUp(fs.materials.size() * sizeof(GPUMaterial));
    const size_t totalSize       = lightsOffset + alignUp(std::max<size_t>(fs.lights.size(), 1) * sizeof(GPULight));

    const UploadRing::Slice upload = context_.frameUploads().allocate(totalSize);
    u8* base = upload.cpu;
    std::memcpy(base + constantsOffset, &constants, sizeof(constants));
    if (!fs.instances.empty())
        std::memcpy(base + instancesOffset, fs.instances.data(), fs.instances.size() * sizeof(GPUInstance));
    if (!fs.materials.empty())
        std::memcpy(base + materialsOffset, fs.materials.data(), fs.materials.size() * sizeof(GPUMaterial));
    if (!fs.lights.empty())
        std::memcpy(base + lightsOffset, fs.lights.data(), fs.lights.size() * sizeof(GPULight));

    // --- Render pass ------------------------------------------------------
    // The descriptor is reused (O7): only the per-frame textures change.
    passDesc_->colorAttachments()->object(0)->setTexture(target);
    passDesc_->depthAttachment()->setTexture(depth_);

    MTL4::RenderCommandEncoder* enc = frame.commandBuffer->renderCommandEncoder(passDesc_);
    // The encoder has copied the descriptor: drop its reference to the
    // drawable so the layer can recycle it as soon as it is presented.
    passDesc_->colorAttachments()->object(0)->setTexture(nullptr);
    enc->setLabel(passLabel_);

    lastTriangles_ = 0;
    if (vertexBuffer_ && indexBuffer_ && !fs.batches.empty()) {
        const MTL::GPUAddress frameBase = upload.gpu;
        arguments_->setAddress(frameBase + constantsOffset, BindFrame);
        arguments_->setAddress(vertexBuffer_->gpuAddress(), BindVertices);
        arguments_->setAddress(frameBase + instancesOffset, BindInstances);
        arguments_->setAddress(frameBase + materialsOffset, BindMaterials);
        arguments_->setAddress(frameBase + lightsOffset, BindLights);
        arguments_->setAddress(textureTable, BindTextures);

        enc->setRenderPipelineState(pipeline_);
        enc->setDepthStencilState(depthState_);
        enc->setArgumentTable(arguments_, MTL::RenderStageVertex | MTL::RenderStageFragment);
        // glTF convention: counter-clockwise front faces (Metal defaults to
        // clockwise).  The shader flips N on back faces, so this decides lighting
        // even without culling.
        enc->setFrontFacingWinding(MTL::WindingCounterClockwise);
        // Two-sided until winding is validated for every asset path (F2.7).
        // The ImGui overlay appended to this pass also relies on CullModeNone.
        enc->setCullMode(MTL::CullModeNone);
        enc->setViewport(MTL::Viewport{0.0, 0.0, static_cast<double>(width), static_cast<double>(height), 0.0, 1.0});

        const MTL::GPUAddress indexBase = indexBuffer_->gpuAddress();
        const auto& infos = scene.meshInfos();
        for (const DrawBatch& batch : fs.batches) {
            const GPUMeshInfo& info = infos[batch.meshIndex];
            if (info.indexCount == 0) continue;
            enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, info.indexCount, MTL::IndexTypeUInt32,
                                       indexBase + static_cast<MTL::GPUAddress>(info.indexOffset) * sizeof(u32),
                                       static_cast<NS::UInteger>(info.indexCount) * sizeof(u32),
                                       batch.instanceCount, info.vertexOffset, batch.firstInstance);
            lastTriangles_ += info.indexCount / 3 * batch.instanceCount;
        }
    }
    return enc;
}

} // namespace phosphor
