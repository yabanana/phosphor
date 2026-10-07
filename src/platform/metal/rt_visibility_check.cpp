#include "platform/metal/rt_visibility_check.h"

#include "platform/metal/acceleration_structures.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/upload_ring.h"
#include "rendergraph/pass_context.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace phosphor {
namespace {
pipe::PipelineDesc visibilityKernel(const char* name) {
    pipe::PipelineDesc d;
    d.kind = pipe::PipelineKind::Compute;
    d.label = name;
    d.functions = {name, "", ""};
    return d;
}
} // namespace

RtVisibilityChecker::RtVisibilityChecker(MetalContext& context, PipelineCache& pipelines, SceneRenderer& scene,
                                         MeshRenderer& mesh, AccelerationStructures& rt)
    : context_(context), pipelines_(pipelines), scene_(scene), mesh_(mesh), rt_(rt) {
    clear_ = pipelines_.request(visibilityKernel("rt_visibility_clear"));
    compare_ = pipelines_.request(visibilityKernel("rt_visibility_compare"));
    // Allocate once at checker creation, before measured frames. One compact
    // shared readback per frame slot; no image readback or per-pixel CPU work.
    try {
        for (auto& frame : frames_) {
            frame.counters = context_.memory().newBuffer(sizeof(GPURtVisibilityCounters), MTL::ResourceStorageModeShared,
                                                         MemoryCategory::RayTracing, "RT V-buffer comparison counters");
            if (!frame.counters) throw std::runtime_error("RT visibility counter allocation failed");
        }
        for (auto*& table : tables_) {
            auto* descriptor = MTL4::ArgumentTableDescriptor::alloc()->init();
            descriptor->setMaxBufferBindCount(8);
            descriptor->setMaxTextureBindCount(2);
            NS::Error* error = nullptr;
            table = context_.device()->newArgumentTable(descriptor, &error);
            descriptor->release();
            if (!table) throw std::runtime_error("RT visibility argument table creation failed");
        }
    } catch (...) {
        for (auto& frame : frames_) context_.memory().release(frame.counters, MemoryCategory::RayTracing);
        for (auto* table : tables_) if (table) table->release();
        throw;
    }
}

RtVisibilityChecker::~RtVisibilityChecker() {
    context_.waitIdle();
    for (auto& frame : frames_) context_.memory().release(frame.counters, MemoryCategory::RayTracing);
    for (auto* table : tables_) if (table) table->release();
}

bool RtVisibilityChecker::ready() const {
    return pipelines_.compute(clear_) && pipelines_.compute(compare_);
}

void RtVisibilityChecker::prepareFrame(u32 slot, u32 width, u32 height, const FrameConstants& constants) {
    if (slot >= frames_.size() || width == 0 || height == 0 || u64(width) * height > std::numeric_limits<u32>::max())
        throw std::invalid_argument("RT visibility frame dimensions/slot are invalid");
    if (mesh_.capacity() > std::numeric_limits<u32>::max() / 2u)
        throw std::invalid_argument("RT visibility candidate capacity exceeds ID decoding range");
    const auto* rayBuffer = rt_.primaryRayBuffer(slot);
    const auto* hitBuffer = rt_.primaryHitBuffer(slot);
    const u64 pixels = u64(width) * height;
    if (!rayBuffer || !hitBuffer || rayBuffer->length() < pixels * sizeof(GPURtRay) ||
        hitBuffer->length() < pixels * sizeof(GPURtHit))
        throw std::invalid_argument("RT visibility requires full-resolution primary ray/hit buffers");
    if (!scene_.buffers().instances() || !scene_.buffers().materials())
        throw std::invalid_argument("RT visibility requires GPU scene buffers");
    const glm::mat4 inverse = glm::inverse(glm::make_mat4(constants.viewProjection));
    const float* inverseData = glm::value_ptr(inverse);
    for (u32 i = 0; i < 16; ++i)
        if (!std::isfinite(inverseData[i]) || !std::isfinite(constants.viewProjection[i]))
            throw std::invalid_argument("RT visibility camera matrix is singular/nonfinite");
    slot_ = slot;
    Frame& frame = frames_[slot];
    frame.index = context_.frameIndex();
    frame.width = width;
    frame.height = height;
    frame.encoded = false;
    GPURtVisibilityParams params{};
    std::memcpy(params.viewProjection, constants.viewProjection, sizeof params.viewProjection);
    std::memcpy(params.inverseViewProjection, inverseData, sizeof params.inverseViewProjection);
    params.width = width;
    params.height = height;
    params.candidateCapacity = static_cast<u32>(mesh_.capacity());
    params.slotCount = scene_.buffers().slotCapacity();
    params.materialCount = static_cast<u32>(scene_.buffers().materials()->length() / sizeof(GPUMaterial));
    params.rayCount = static_cast<u32>(pixels);
    params.frameLo = static_cast<u32>(frame.index);
    params.frameHi = static_cast<u32>(frame.index >> 32);
    params.relativeDistanceTolerance = RelativeDistanceTolerance;
    const auto upload = context_.frameUploads().allocate(sizeof params);
    std::memcpy(upload.cpu, &params, sizeof params);
    frame.params = upload.gpu;
}

void RtVisibilityChecker::addToGraph(rg::RenderGraph& graph, rg::TextureRef visibility, rg::TextureRef depth) {
    using namespace rg;
    if (!ready()) throw std::logic_error("RT visibility pipelines must be ready before declaring diagnostic passes");
    if (!visibility.valid() || !depth.valid() || !rt_.primaryRayRef().valid() || !rt_.primaryHitRef().valid())
        throw std::invalid_argument("RT visibility graph inputs are missing");
    visibilityRef_ = visibility;
    depthRef_ = depth;
    countersRef_ = graph.importBuffer("RT V-buffer comparison counters", {sizeof(GPURtVisibilityCounters)},
                                      ImportPerFrame | ImportOutput);
    graph.addPass("RT V-buffer clear", PassType::Compute,
        [&](PassBuilder& b) {
            countersRef_ = b.write(countersRef_, Usage::ShaderWrite, StageDispatch);
            b.setProfileShaders("rt_visibility_clear");
        },
        [this](PassContext& ctx) {
            auto* encoder = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
            auto* buffer = static_cast<MTL::Buffer*>(ctx.buffer(countersRef_));
            tables_[0]->setAddress(buffer->gpuAddress(), 0);
            encoder->setComputePipelineState(pipelines_.compute(clear_));
            encoder->setArgumentTable(tables_[0]);
            encoder->dispatchThreads(MTL::Size::Make(sizeof(GPURtVisibilityCounters) / sizeof(u32), 1, 1),
                                     MTL::Size::Make(64, 1, 1));
            scene_.countCommands(3);
        });
    graph.addPass("RT V-buffer compare", PassType::Compute,
        [&](PassBuilder& b) {
            b.read(countersRef_, Usage::ShaderRead, StageDispatch);
            countersRef_ = b.write(countersRef_, Usage::ShaderWrite, StageDispatch);
            b.read(visibilityRef_, Usage::ShaderRead, StageDispatch);
            b.read(depthRef_, Usage::ShaderRead, StageDispatch);
            b.read(rt_.primaryRayRef(), Usage::ShaderRead, StageDispatch);
            b.read(rt_.primaryHitRef(), Usage::ShaderRead, StageDispatch);
            b.read(scene_.dataRef(), Usage::ShaderRead, StageDispatch);
            b.read(mesh_.frameListsRef(), Usage::ShaderRead, StageDispatch);
            b.setProfileShaders("rt_visibility_compare");
            b.setSideEffect(); // CPU reads this slot after GPU completion
        },
        [this](PassContext& ctx) {
            Frame& frame = frames_[slot_];
            auto* encoder = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
            auto* table = tables_[1];
            auto* visibilityTexture = static_cast<MTL::Texture*>(ctx.texture(visibilityRef_));
            auto* depthTexture = static_cast<MTL::Texture*>(ctx.texture(depthRef_));
            if (visibilityTexture->width() < frame.width || visibilityTexture->height() < frame.height ||
                depthTexture->width() < frame.width || depthTexture->height() < frame.height)
                throw std::logic_error("RT visibility comparison exceeds the raster backing extent");
            table->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(countersRef_))->gpuAddress(), 0);
            table->setAddress(frame.params, 1);
            table->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(rt_.primaryRayRef()))->gpuAddress(), 2);
            table->setAddress(static_cast<MTL::Buffer*>(ctx.buffer(rt_.primaryHitRef()))->gpuAddress(), 3);
            table->setAddress(mesh_.frame(slot_).candidates->gpuAddress(), 4);
            table->setAddress(mesh_.frame(slot_).bList->gpuAddress(), 5);
            table->setAddress(scene_.buffers().instances()->gpuAddress(), 6);
            table->setAddress(scene_.buffers().materials()->gpuAddress(), 7);
            table->setTexture(visibilityTexture->gpuResourceID(), 0);
            table->setTexture(depthTexture->gpuResourceID(), 1);
            encoder->setComputePipelineState(pipelines_.compute(compare_));
            encoder->setArgumentTable(table);
            encoder->dispatchThreads(MTL::Size::Make(u64(frame.width) * frame.height, 1, 1), MTL::Size::Make(64, 1, 1));
            scene_.countCommands(3);
            frame.encoded = true;
        });
}

void RtVisibilityChecker::bindFrame(MetalGraphExecutor& executor) {
    if (countersRef_.valid()) executor.bindBuffer(countersRef_, frames_[slot_].counters);
}

RtVisibilityChecker::Result RtVisibilityChecker::result(u32 slot) const {
    Result out;
    if (slot >= frames_.size()) return out;
    const Frame& frame = frames_[slot];
    if (!frame.encoded || !frame.counters) return out;
    GPURtVisibilityCounters c;
    std::memcpy(&c, frame.counters->contents(), sizeof c);
    const u64 gpuFrame = (u64(c.frameHi) << 32) | c.frameLo;
    if (gpuFrame != frame.index || c.width != frame.width || c.height != frame.height ||
        c.compared != u64(frame.width) * frame.height) return out;
    u64 categorized = 0, categorizedMismatches = 0;
    for (const auto& category : c.category) {
        categorized += category.compared;
        categorizedMismatches += category.mismatches;
    }
    if (categorized != c.compared || categorizedMismatches != c.mismatches || c.mismatches > c.compared ||
        u64(c.bothBackground) + c.bothHit + c.hitMiss != c.compared ||
        u64(c.rasterHits) + c.rtHits != 2u * u64(c.bothHit) + c.hitMiss) return out;
    out.valid = true;
    out.frame = gpuFrame;
    out.width = c.width; out.height = c.height;
    out.compared = c.compared; out.mismatches = c.mismatches;
    out.hitMiss = c.hitMiss; out.slot = c.slot; out.depth = c.depth;
    out.bothBackground = c.bothBackground; out.bothHit = c.bothHit;
    out.rasterHits = c.rasterHits; out.rtHits = c.rtHits;
    out.edgePixels = c.edgePixels; out.maskPixels = c.maskPixels; out.maskNeighborPixels = c.maskNeighborPixels;
    out.invalidVisibility = c.invalidVisibility; out.invalidRt = c.invalidRt;
    out.invalidDepth = c.invalidDepth; out.depthWithoutVisibility = c.depthWithoutVisibility;
    out.depthBitDifferent = c.depthBitDifferent; out.maxDepthUlp = c.maxDepthUlp;
    out.maxDepthError = std::bit_cast<float>(c.maxDepthErrorBits);
    out.maxRelativeDistanceError = std::bit_cast<float>(c.maxRelativeDistanceErrorBits);
    for (u32 i = 0; i < out.depthUlpHistogram.size(); ++i) out.depthUlpHistogram[i] = c.depthUlpHistogram[i];
    for (u32 i = 0; i < out.category.size(); ++i) out.category[i] = c.category[i];
    return out;
}
} // namespace phosphor
