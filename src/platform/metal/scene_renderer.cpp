#include "platform/metal/scene_renderer.h"
#include "core/profile.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "pipeline/forward_variants.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "core/log.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

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

SceneRenderer::SceneRenderer(MetalContext& context, PipelineCache& pipelines, u32 salt, bool genericOnly,
                             std::optional<u32> forceVariant)
    : context_(context), pipelines_(pipelines), salt_(salt), genericOnly_(genericOnly), forceVariant_(forceVariant) {
    if (forceVariant_ && *forceVariant_ >= pipe::forward::variantCount()) {
        throw std::runtime_error("--force-variant: expected 0.." + std::to_string(pipe::forward::variantCount() - 1));
    }
    // The generic pipeline serves every frame until its variant is ready.
    generic_ = pipelines_.request(pipe::forward::genericDesc(rg::Format::BGRA8Srgb));
    variants_.assign(pipe::forward::variantCount(), pipe::INVALID_PIPELINE);

    MTL::DepthStencilDescriptor* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionGreater); // reverse-Z
    dsDesc->setDepthWriteEnabled(true);
    depthState_ = context_.device()->newDepthStencilState(dsDesc);
    dsDesc->release();

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
    arguments_->release();
    depthState_->release();
}

void SceneRenderer::requestAllVariants() {
    for (u32 i = 0; i < variants_.size(); ++i) {
        if (variants_[i] == pipe::INVALID_PIPELINE) {
            variants_[i] = pipelines_.request(
                pipe::forward::pipelineDesc(pipe::forward::variantAt(i), rg::Format::BGRA8Srgb, salt_));
        }
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

void SceneRenderer::prepareFrame(const GpuScene& scene, const FrameScene& fs, const FrameConstants& constants,
                                 MTL::GPUAddress textureTable, u32 width, u32 height) {
    PH_ZONE("Scene prepare");
    scene_      = &scene;
    frameScene_ = &fs;
    width_      = width;
    height_     = height;

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

    const MTL::GPUAddress frameBase = upload.gpu;
    arguments_->setAddress(frameBase + constantsOffset, BindFrame);
    if (vertexBuffer_) arguments_->setAddress(vertexBuffer_->gpuAddress(), BindVertices);
    arguments_->setAddress(frameBase + instancesOffset, BindInstances);
    arguments_->setAddress(frameBase + materialsOffset, BindMaterials);
    arguments_->setAddress(frameBase + lightsOffset, BindLights);
    arguments_->setAddress(textureTable, BindTextures);

    // F3.3: the variant for this scene and debug mode; requested on first use
    // (compiled in the background), generic pipeline meanwhile.
    const pipe::forward::Variant variant =
        forceVariant_ ? pipe::forward::variantAt(*forceVariant_) : pipe::forward::sceneVariant(fs, constants.debugMode);
    pipe::PipelineHandle& handle = genericOnly_ ? generic_ : variants_[pipe::forward::variantIndex(variant)];
    if (handle == pipe::INVALID_PIPELINE) {
        LOG_INFO("Forward variant %u requested: light types 0x%x, emissive %d, debug mode %u",
                 pipe::forward::variantIndex(variant), variant.lightTypes, variant.emissive ? 1 : 0, variant.debugMode);
        handle = pipelines_.request(pipe::forward::pipelineDesc(variant, rg::Format::BGRA8Srgb, salt_));
    }
    usingFallback_ = genericOnly_ || !pipelines_.isFinal(handle);
    pipeline_ = pipelines_.render(handle);
    if (!pipeline_) pipeline_ = pipelines_.render(generic_);
    if (usingFallback_) pipelines_.noteFallbackUse();

    lastTriangles_ = 0;
    const auto& infos = scene.meshInfos();
    for (const DrawBatch& batch : fs.batches) {
        lastTriangles_ += infos[batch.meshIndex].indexCount / 3 * batch.instanceCount;
    }
}

void SceneRenderer::encode(MTL4::RenderCommandEncoder* enc, u32 chunk, u32 chunks) const {
    PH_ZONE("Forward encode");
    if (!scene_ || !frameScene_ || !vertexBuffer_ || !indexBuffer_ || frameScene_->batches.empty() || !pipeline_) return;

    enc->setRenderPipelineState(pipeline_);
    enc->setDepthStencilState(depthState_);
    enc->setArgumentTable(arguments_, MTL::RenderStageVertex | MTL::RenderStageFragment);
    enc->setViewport(MTL::Viewport{0.0, 0.0, static_cast<double>(width_), static_cast<double>(height_), 0.0, 1.0});

    const auto& batches = frameScene_->batches;
    const size_t count = batches.size();
    const size_t first = count * chunk / chunks;
    const size_t last  = count * (chunk + 1) / chunks;

    const MTL::GPUAddress indexBase = indexBuffer_->gpuAddress();
    const auto& infos = scene_->meshInfos();
    // glTF convention: counter-clockwise front faces (Metal defaults to
    // clockwise).  A mirrored model matrix reverses the on-screen winding, so
    // mirrored batches cull FRONT faces instead: the same triangles as
    // flipping the winding, while [[front_facing]] keeps the meaning the
    // shader expects (it inverts it itself for INSTANCE_FLAG_MIRRORED).
    // Double-sided materials are not culled (the shader lights their back
    // faces).  State is tracked from Metal's defaults (clockwise, no culling)
    // because the validation layer rejects redundant state changes.
    MTL::Winding  winding = MTL::WindingClockwise;
    MTL::CullMode cull    = MTL::CullModeNone;
    const auto setState = [&](MTL::Winding w, MTL::CullMode c) {
        if (w != winding) enc->setFrontFacingWinding(winding = w);
        if (c != cull) enc->setCullMode(cull = c);
    };
    for (size_t i = first; i < last; ++i) {
        const DrawBatch& batch = batches[i];
        const GPUMeshInfo& info = infos[batch.meshIndex];
        if (info.indexCount == 0) continue;
        switch (batch.cull) {
        case CullClass::Back:         setState(MTL::WindingCounterClockwise, MTL::CullModeBack); break;
        case CullClass::BackMirrored: setState(MTL::WindingCounterClockwise, MTL::CullModeFront); break;
        case CullClass::None:         setState(MTL::WindingCounterClockwise, MTL::CullModeNone); break;
        }
        enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, info.indexCount, MTL::IndexTypeUInt32,
                                   indexBase + static_cast<MTL::GPUAddress>(info.indexOffset) * sizeof(u32),
                                   static_cast<NS::UInteger>(info.indexCount) * sizeof(u32),
                                   batch.instanceCount, info.vertexOffset, batch.firstInstance);
    }
    // Later passes fused into this render encoder (the ImGui overlay) start
    // from the default state.
    setState(MTL::WindingClockwise, MTL::CullModeNone);
}

} // namespace phosphor
