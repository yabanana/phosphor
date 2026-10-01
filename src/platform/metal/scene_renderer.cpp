#include "platform/metal/scene_renderer.h"
#include "core/log.h"
#include "core/profile.h"
#include "pipeline/forward_variants.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/pipeline_cache.h"
#include "renderer/gpu_queue.h"
#include "renderer/gpu_scene.h"
#include "renderer/gpu_scene_layout.h"
#include "renderer/scene_store.h"
#include "rendergraph/pass_context.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

NS::String* str(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

// Forward argument table slots; must match shaders/forward.metal and overlay.metal.
enum Binding : NS::UInteger {
    BindFrame     = 0,
    BindVertices  = 1,
    BindInstances = 2,
    BindMaterials = 3,
    BindLights    = 4,
    BindTextures  = 5,
    BindVisible   = FORWARD_BIND_VISIBLE,
    BindCount     = 7,
};

constexpr u32 kQueues = SCENE_MAX_LEVELS - 1;

MTL4::ArgumentTable* newTable(MTL::Device* device, u32 bindings, const char* label) {
    NS::Error* error = nullptr;
    MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(bindings);
    d->setLabel(str(label));
    MTL4::ArgumentTable* t = device->newArgumentTable(d, &error);
    d->release();
    if (!t) throw std::runtime_error(std::string("Failed to create argument table ") + label);
    return t;
}

pipe::PipelineDesc kernelDesc(const char* function) {
    pipe::PipelineDesc d;
    d.kind      = pipe::PipelineKind::Compute;
    d.label     = function;
    d.functions = {function, ""};
    return d;
}

/// Encoder barrier between two dispatches of one scene pass.
void dispatchBarrier(MTL4::ComputeCommandEncoder* enc) {
    enc->barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit, MTL::StageDispatch | MTL::StageBlit,
                                   MTL4::VisibilityOptionDevice);
}

MTL::Size threads1D(u32 n) { return MTL::Size::Make(std::max(n, 1u), 1, 1); }

} // namespace

SceneRenderer::SceneRenderer(MetalContext& context, PipelineCache& pipelines, u32 salt, bool genericOnly,
                             std::optional<u32> forceVariant)
    : context_(context), pipelines_(pipelines), salt_(salt), genericOnly_(genericOnly), forceVariant_(forceVariant),
      buffers_(context) {
    if (forceVariant_ && *forceVariant_ >= pipe::forward::variantCount()) {
        throw std::runtime_error("--force-variant: expected 0.." + std::to_string(pipe::forward::variantCount() - 1));
    }
    // The generic pipeline serves every frame until its variant is ready.
    generic_ = pipelines_.request(pipe::forward::genericDesc(rg::Format::BGRA8Srgb));
    variants_.assign(pipe::forward::variantCount(), pipe::INVALID_PIPELINE);

    kClear_     = pipelines_.request(kernelDesc(KERNEL_QUEUE_CLEAR));
    kScatter_   = pipelines_.request(kernelDesc(KERNEL_SCATTER));
    kMotion_    = pipelines_.request(kernelDesc(KERNEL_MOTION));
    kQueueArgs_ = pipelines_.request(kernelDesc(KERNEL_QUEUE_ARGS));
    kHier_      = pipelines_.request(kernelDesc(KERNEL_HIER_LEVEL));
    kCullFlags_ = pipelines_.request(kernelDesc(KERNEL_CULL_FLAGS));
    kCullScan_  = pipelines_.request(kernelDesc(KERNEL_CULL_SCAN));
    kCullWrite_ = pipelines_.request(kernelDesc(KERNEL_CULL_WRITE));
    kDrawBuild_ = pipelines_.request(kernelDesc(KERNEL_DRAW_BUILD));

    MTL::DepthStencilDescriptor* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionGreater); // reverse-Z
    dsDesc->setDepthWriteEnabled(true);
    depthState_ = context_.device()->newDepthStencilState(dsDesc);
    dsDesc->release();

    MTL::Device* device = context_.device();
    arguments_      = newTable(device, BindCount, "Forward arguments");
    updateTable_    = newTable(device, SCENE_BIND_COUNT, "Scene update arguments");
    transformTable_ = newTable(device, SCENE_BIND_COUNT, "Scene transforms arguments");
    cullTable_      = newTable(device, SCENE_BIND_COUNT, "Instance cull arguments");
    drawTable_      = newTable(device, SCENE_BIND_COUNT, "Draw build arguments");
}

SceneRenderer::~SceneRenderer() {
    context_.waitIdle();
    releaseGeometry();
    for (MTL4::ArgumentTable* t : {arguments_, updateTable_, transformTable_, cullTable_, drawTable_}) t->release();
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

MTL::ComputePipelineState* SceneRenderer::kernel(pipe::PipelineHandle h) const { return pipelines_.compute(h); }

bool SceneRenderer::kernelsReady() const {
    for (pipe::PipelineHandle h : {kClear_, kScatter_, kMotion_, kQueueArgs_, kHier_, kCullFlags_, kCullScan_,
                                   kCullWrite_, kDrawBuild_}) {
        if (!kernel(h)) return false;
    }
    return true;
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
    for (MTL::Buffer** b : {&vertexBuffer_, &indexBuffer_, &meshBuffer_}) {
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
    meshBuffer_   = createPrivateBuffer(scene.meshInfos().data(),
                                        scene.meshInfos().size() * sizeof(GPUMeshInfo), "Mesh infos");
    context_.flushUploads();
    LOG_INFO("Geometry uploaded: %zu vertices, %zu indices, %zu meshes",
             scene.vertices().size(), scene.indices().size(), scene.meshInfos().size());
}

void SceneRenderer::loadScene(const SceneStore& store) {
    context_.waitIdle();
    buffers_.clear();
    buffers_.loadAll(store);
}

u64 SceneRenderer::prepareFrame(const SceneStore& store, std::span<const GPULight> lights,
                                const FrameConstants& constants, MTL::GPUAddress textureTable,
                                const FrameParams& params) {
    PH_ZONE("Scene prepare");
    store_ = &store;
    frame_ = params;
    for (auto& c : cpuCommands_) c.store(0, std::memory_order_relaxed);

    u64 bytes = buffers_.stageFrame(store);
    UploadRing& ring = context_.frameUploads();
    const auto put = [&](const void* data, u64 size) {
        const UploadRing::Slice s = ring.allocate(std::max<u64>(size, 16));
        if (size) std::memcpy(s.cpu, data, size);
        bytes += size;
        return s.gpu;
    };

    slotCount_    = store.slotCapacity();
    cullGroups_   = buffers_.cullGroups(std::max(slotCount_, 1u));
    motionCount_  = static_cast<u32>(store.motionSlots().size());
    commandCount_ = store.commandCount();

    const MTL::GPUAddress frameConstants = put(&constants, sizeof(constants));
    GPULight noLight{};
    const MTL::GPUAddress lightsAddress =
        lights.empty() ? put(&noLight, sizeof(noLight)) : put(lights.data(), lights.size_bytes());

    // Scene update.
    GPUQueueClearParams clear{};
    clear.queues      = kQueues;
    clear.strideBytes = buffers_.queueStride();
    clearParams_ = put(&clear, sizeof(clear));
    std::array<GPUScatterParams, 4> scatter{};
    const auto& scatters = buffers_.scatters();
    for (u32 i = 0; i < 4; ++i) {
        scatter[i].count = scatters[i].count;
        scatter[i].words = 20;
    }
    MTL::GPUAddress instanceRecords = scatters[0].records;
    if (params.corrupt == SceneCorruption::Delta && store.instanceCount() > 0) {
        // Negative control of the self-check: a wrong record for the first
        // slot of the first bucket; the readback must report it.
        std::vector<GPUDeltaRecord> records(store.instanceDeltas().begin(), store.instanceDeltas().end());
        const u32 slot = store.buckets().empty() ? 0 : store.buckets()[0].firstSlot;
        // One record per slot (the scatter kernel needs distinct slots).
        auto it = std::find_if(records.begin(), records.end(), [&](const GPUDeltaRecord& r) { return r.slot == slot; });
        if (it == records.end()) {
            GPUDeltaRecord bad{};
            bad.slot = slot;
            std::memcpy(bad.payload, &store.instances()[slot], sizeof(bad.payload));
            records.push_back(bad);
            it = records.end() - 1;
        }
        // GPUInstance::pad (word 19): never overwritten by the GPU passes and
        // never read by a shader, so only the self-check can notice it.
        it->payload[19] ^= 0x5A5A5A5Au;
        instanceRecords = put(records.data(), records.size() * sizeof(GPUDeltaRecord));
        scatter[0].count = static_cast<u32>(records.size());
    }
    scatterParams_ = put(scatter.data(), sizeof(scatter));
    scatterRecords_ = {instanceRecords, scatters[1].records, scatters[2].records, scatters[3].records};
    for (u32 i = 0; i < 4; ++i) scatterCounts_[i] = scatter[i].count;

    // Scene transforms: motion table, hierarchy levels, queue 0 (the dirty roots).
    GPUMotionFrame motion{};
    motion.motionCount = motionCount_;
    if (params.motionSinCos) std::memcpy(motion.sinCos, params.motionSinCos, sizeof(motion.sinCos));
    motionFrame_ = put(&motion, sizeof(motion));
    std::array<GPUHierParams, SCENE_MAX_LEVELS> hier{};
    for (u32 l = 0; l < SCENE_MAX_LEVELS; ++l) hier[l].level = l;
    hierParams_ = put(hier.data(), sizeof(hier));
    {
        const std::span<const u32> roots = store.dirtyRoots();
        GPUQueueHeader h{};
        h.count     = static_cast<u32>(roots.size());
        h.capacity  = h.count;
        h.groups[0] = gpuQueueGroups(h.count, h.capacity, SCENE_HIER_GROUP);
        h.groups[1] = 1;
        h.groups[2] = 1;
        const u64 size = sizeof(h) + roots.size_bytes();
        const UploadRing::Slice s = ring.allocate(size);
        std::memcpy(s.cpu, &h, sizeof(h));
        if (!roots.empty()) std::memcpy(s.cpu + sizeof(h), roots.data(), roots.size_bytes());
        queue0_ = s.gpu;
        bytes += size;
    }

    // Instance cull + draw build (gpu-driven on).
    GPUCullParams cull = params.cull;
    cull.slotCount  = slotCount_;
    cull.groupCount = cullGroups_;
    if (params.corrupt == SceneCorruption::Plane) {
        for (u32 k = 0; k < 4; ++k) cull.planes[k] = -cull.planes[k]; // left plane inverted
    }
    cullParams_ = put(&cull, sizeof(cull));
    GPUDrawParams draw{};
    draw.commandCount = commandCount_;
    draw.bucketCount  = static_cast<u32>(store.gpuBuckets().size());
    draw.slotCount    = slotCount_;
    if (params.corrupt == SceneCorruption::Command && commandCount_ > 0) {
        // Negative control: the last command is not written; its self-check
        // arguments keep this marker.
        draw.commandCount = commandCount_ - 1;
        std::memset(buffers_.frame(params.slot).drawArgs->contents(), 0xFF, u64(commandCount_) * 2 * sizeof(u32));
    }
    drawParams_ = put(&draw, sizeof(draw));

    // Forward bindings.
    const GpuSceneBuffers::FrameSet& fs = buffers_.frame(params.slot);
    arguments_->setAddress(frameConstants, BindFrame);
    if (vertexBuffer_) arguments_->setAddress(vertexBuffer_->gpuAddress(), BindVertices);
    arguments_->setAddress(buffers_.instances()->gpuAddress(), BindInstances);
    arguments_->setAddress(buffers_.materials()->gpuAddress(), BindMaterials);
    arguments_->setAddress(lightsAddress, BindLights);
    arguments_->setAddress(textureTable, BindTextures);
    arguments_->setAddress(params.mode == GpuDrivenMode::On ? fs.visible->gpuAddress() : buffers_.identity()->gpuAddress(),
                           BindVisible);

    // F3.3: the variant for this scene and debug mode; requested on first use
    // (compiled in the background), generic pipeline meanwhile.
    const pipe::forward::Variant variant = forceVariant_ ? pipe::forward::variantAt(*forceVariant_)
                                                         : pipe::forward::sceneVariant(lights, store.hasEmissive(),
                                                                                       constants.debugMode);
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

    // Triangles of every live instance (off draws all of them; on draws the
    // visible subset, reported from the GPU counters).
    lastTriangles_ = 0;
    const auto gpuBuckets = store.gpuBuckets();
    for (size_t i = 0; i < gpuBuckets.size(); ++i) lastTriangles_ += gpuBuckets[i].indexCount / 3 * store.buckets()[i].count;
    return bytes;
}

void SceneRenderer::addPassesToGraph(rg::RenderGraph& graph, GpuDrivenMode mode) {
    using namespace rg;
    graphMode_ = mode;
    // Sizes are estimates for the bandwidth model only (physical buffers are
    // bound by the passes themselves).
    const u64 slots = buffers_.slotCapacity();
    graphData_ = graph.importBuffer("Scene data", {slots * (sizeof(GPUInstance) + sizeof(GPUTransformNode) + 4)},
                                    ImportContentsDefined | ImportOutput);
    graphFrame_ = graph.importBuffer("Scene frame lists", {slots * 16 + buffers_.commandCapacity() * 64},
                                     ImportPerFrame | ImportOutput);
    graph.addPass(
        PASS_SCENE_UPDATE, PassType::Compute,
        [&](PassBuilder& b) {
            b.read(graphData_, Usage::ShaderRead, StageDispatch | StageBlit);
            graphData_  = b.write(graphData_, Usage::ShaderWrite, StageDispatch | StageBlit);
            graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageDispatch);
            b.setProfileShaders("scene_queue_clear,scene_scatter");
        },
        [this](PassContext& ctx) { encodeUpdate(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder())); });
    graph.addPass(
        PASS_SCENE_TRANSFORMS, PassType::Compute,
        [&](PassBuilder& b) {
            b.read(graphData_, Usage::ShaderRead, StageDispatch);
            graphData_ = b.write(graphData_, Usage::ShaderWrite, StageDispatch);
            b.read(graphFrame_, Usage::ShaderRead, StageDispatch);
            graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageDispatch);
            b.setProfileShaders("scene_motion,scene_queue_args,scene_hier_level");
        },
        [this](PassContext& ctx) { encodeTransforms(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder())); });
    if (mode != GpuDrivenMode::On) return;
    graph.addPass(
        PASS_INSTANCE_CULL, PassType::Compute,
        [&](PassBuilder& b) {
            b.read(graphData_, Usage::ShaderRead, StageDispatch);
            b.read(graphFrame_, Usage::ShaderRead, StageDispatch);
            graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageDispatch);
            b.setProfileShaders("scene_cull_flags,scene_cull_scan,scene_cull_write");
        },
        [this](PassContext& ctx) { encodeCull(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder())); });
    graph.addPass(
        PASS_DRAW_BUILD, PassType::Compute,
        [&](PassBuilder& b) {
            b.read(graphData_, Usage::ShaderRead, StageDispatch);
            b.read(graphFrame_, Usage::ShaderRead, StageDispatch);
            graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageDispatch | StageBlit);
            b.setProfileShaders("scene_draw_build");
        },
        [this](PassContext& ctx) { encodeDrawBuild(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder())); });
}

void SceneRenderer::declareDrawReads(rg::PassBuilder& b) const {
    using namespace rg;
    if (!graphData_.valid()) return;
    b.read(graphData_, Usage::ShaderRead, StageVertex | StageFragment);
    // The ICB commands and the visible list are fetched at the Vertex stage
    // (S5: Fragment / Object|Mesh on the consumer side do not synchronise).
    if (graphMode_ == GpuDrivenMode::On) b.read(graphFrame_, Usage::IndirectArgs, StageVertex);
}

void SceneRenderer::encodeUpdate(MTL4::ComputeCommandEncoder* enc) const {
    PH_ZONE("Scene update encode");
    if (!store_ || buffers_.empty() || !kernelsReady()) return;
    u32 n = 0;
    MTL4::ArgumentTable* t = updateTable_;
    const GpuSceneBuffers::FrameSet& fs = buffers_.frame(frame_.slot);
    // Full copies (structure buffers, more than 1/8 of a buffer changed).
    for (const GpuSceneBuffers::Copy& c : buffers_.copies()) {
        enc->copyFromBuffer(c.src, c.srcOffset, c.dst, 0, c.size);
        ++n;
    }
    // Queue headers and counters of this frame (independent of the copies).
    enc->setComputePipelineState(kernel(kClear_));
    t->setAddress(clearParams_, 0);
    t->setAddress(buffers_.queues()->gpuAddress(), SB_CLEAR_QUEUES);
    t->setAddress(fs.counters->gpuAddress(), SB_CLEAR_COUNTERS);
    enc->setArgumentTable(t);
    enc->dispatchThreads(MTL::Size::Make(32, 1, 1), MTL::Size::Make(32, 1, 1));
    n += 3;
    // Delta scatters: four destinations, always encoded (count 0 returns).
    enc->setComputePipelineState(kernel(kScatter_));
    ++n;
    const auto& scatters = buffers_.scatters();
    for (u32 i = 0; i < 4; ++i) {
        t->setAddress(scatterParams_ + i * sizeof(GPUScatterParams), 0);
        t->setAddress(scatterRecords_[i] ? scatterRecords_[i] : scatterParams_, SB_SCATTER_RECORDS);
        t->setAddress(scatters[i].dst->gpuAddress(), SB_SCATTER_DST);
        enc->setArgumentTable(t);
        enc->dispatchThreads(threads1D(scatterCounts_[i]), MTL::Size::Make(SCENE_SCATTER_GROUP, 1, 1));
        n += 2;
    }
    count(0, n);
}

void SceneRenderer::encodeTransforms(MTL4::ComputeCommandEncoder* enc) const {
    PH_ZONE("Scene transforms encode");
    if (!store_ || buffers_.empty() || !kernelsReady()) return;
    u32 n = 0;
    MTL4::ArgumentTable* t = transformTable_;
    const GpuSceneBuffers::FrameSet& fs = buffers_.frame(frame_.slot);
    // Procedural motion of the roots (F5.6).
    enc->setComputePipelineState(kernel(kMotion_));
    t->setAddress(motionFrame_, 0);
    t->setAddress(buffers_.motionSlots()->gpuAddress(), SB_MOTION_SLOTS);
    t->setAddress(buffers_.motions()->gpuAddress(), SB_MOTION_RECORDS);
    t->setAddress(buffers_.instances()->gpuAddress(), SB_MOTION_INSTANCES);
    enc->setArgumentTable(t);
    enc->dispatchThreads(threads1D(motionCount_), MTL::Size::Make(SCENE_MOTION_GROUP, 1, 1));
    dispatchBarrier(enc);
    n += 4;
    // Hierarchy (F5.2) as a chain of GPU queues (F5.4): step s consumes queue
    // s with an indirect dispatch whose arguments the previous step's args
    // kernel wrote; every step is encoded whatever the scene (constant CPU work).
    const MTL::GPUAddress queues = buffers_.queues()->gpuAddress();
    const u64 stride = buffers_.queueStride();
    t->setAddress(buffers_.nodes()->gpuAddress(), SB_HIER_NODES);
    t->setAddress(buffers_.instances()->gpuAddress(), SB_HIER_INSTANCES);
    t->setAddress(buffers_.childOffsets()->gpuAddress(), SB_HIER_CHILD_OFFSETS);
    t->setAddress(buffers_.childSlots()->gpuAddress(), SB_HIER_CHILD_SLOTS);
    t->setAddress(fs.counters->gpuAddress(), SB_HIER_COUNTERS);
    for (u32 s = 0; s < SCENE_MAX_LEVELS; ++s) {
        const MTL::GPUAddress in  = s == 0 ? queue0_ : queues + (s - 1) * stride;
        const MTL::GPUAddress out = queues + std::min(s, kQueues - 1) * stride; // the last step appends nothing
        t->setAddress(hierParams_ + s * sizeof(GPUHierParams), 0);
        t->setAddress(in, SB_HIER_QUEUE_IN);
        t->setAddress(out, SB_HIER_QUEUE_OUT);
        enc->setComputePipelineState(kernel(kQueueArgs_));
        enc->setArgumentTable(t);
        enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
        dispatchBarrier(enc);
        enc->setComputePipelineState(kernel(kHier_));
        enc->setArgumentTable(t);
        enc->dispatchThreadgroups(in + GPU_QUEUE_ARGS_OFFSET, MTL::Size::Make(SCENE_HIER_GROUP, 1, 1));
        dispatchBarrier(enc);
        n += 6;
    }
    count(0, n);
}

void SceneRenderer::encodeCull(MTL4::ComputeCommandEncoder* enc) const {
    PH_ZONE("Instance cull encode");
    if (!store_ || buffers_.empty() || !kernelsReady() || !meshBuffer_) return;
    MTL4::ArgumentTable* t = cullTable_;
    const GpuSceneBuffers::FrameSet& fs = buffers_.frame(frame_.slot);
    t->setAddress(cullParams_, 0);
    t->setAddress(buffers_.instances()->gpuAddress(), SB_CULL_INSTANCES);
    t->setAddress(meshBuffer_->gpuAddress(), SB_CULL_MESHES);
    t->setAddress(fs.flags->gpuAddress(), SB_CULL_FLAGS);
    t->setAddress(fs.groups->gpuAddress(), SB_CULL_GROUPS);
    t->setAddress(fs.counters->gpuAddress(), SB_CULL_COUNTERS);
    t->setAddress(fs.visible->gpuAddress(), SB_CULL_VISIBLE);
    t->setAddress(fs.prefix->gpuAddress(), SB_CULL_PREFIX);
    const MTL::Size group = MTL::Size::Make(SCENE_CULL_GROUP, 1, 1);
    enc->setComputePipelineState(kernel(kCullFlags_));
    enc->setArgumentTable(t);
    enc->dispatchThreadgroups(MTL::Size::Make(cullGroups_, 1, 1), group);
    dispatchBarrier(enc);
    enc->setComputePipelineState(kernel(kCullScan_));
    enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), group);
    dispatchBarrier(enc);
    enc->setComputePipelineState(kernel(kCullWrite_));
    enc->dispatchThreadgroups(MTL::Size::Make(cullGroups_, 1, 1), group);
    count(0, 9);
}

void SceneRenderer::encodeDrawBuild(MTL4::ComputeCommandEncoder* enc) const {
    PH_ZONE("Draw build encode");
    if (!store_ || buffers_.empty() || !kernelsReady() || !indexBuffer_) return;
    MTL4::ArgumentTable* t = drawTable_;
    const GpuSceneBuffers::FrameSet& fs = buffers_.frame(frame_.slot);
    // Reset on the GPU timeline (never the CPU reset() of an ICB that a frame
    // in flight may execute), then one thread per command (D2).
    enc->resetCommandsInBuffer(fs.icb, NS::Range::Make(0, buffers_.commandCapacity()));
    dispatchBarrier(enc);
    t->setAddress(drawParams_, 0);
    t->setAddress(buffers_.buckets()->gpuAddress(), SB_DRAW_BUCKETS);
    t->setAddress(buffers_.commandBuckets()->gpuAddress(), SB_DRAW_COMMANDS);
    t->setAddress(fs.prefix->gpuAddress(), SB_DRAW_PREFIX);
    t->setAddress(fs.container->gpuAddress(), SB_DRAW_ICB);
    t->setAddress(indexBuffer_->gpuAddress(), SB_DRAW_INDICES);
    t->setAddress(fs.drawArgs->gpuAddress(), SB_DRAW_ARGS);
    t->setAddress(fs.counters->gpuAddress(), SB_DRAW_COUNTERS);
    enc->setComputePipelineState(kernel(kDrawBuild_));
    enc->setArgumentTable(t);
    enc->dispatchThreads(threads1D(commandCount_), MTL::Size::Make(SCENE_DRAW_GROUP, 1, 1));
    count(0, 5);
}

void SceneRenderer::encode(MTL4::RenderCommandEncoder* enc, u32 chunk, u32 chunks) const {
    PH_ZONE("Forward encode");
    encodeForward(enc, pipeline_, depthState_, chunk, chunks, 1 + std::min(chunk, kMaxChunks - 1));
}

void SceneRenderer::encodeOverlay(MTL4::RenderCommandEncoder* enc, pipe::PipelineHandle pipeline, bool depthTest) const {
    encodeForward(enc, pipelines_.render(pipeline), depthTest ? depthState_ : nullptr, 0, 1, 0);
}

void SceneRenderer::encodeForward(MTL4::RenderCommandEncoder* enc, MTL::RenderPipelineState* pipeline,
                                  MTL::DepthStencilState* depthState, u32 chunk, u32 chunks, u32 counterIndex) const {
    if (!store_ || buffers_.empty() || !vertexBuffer_ || !indexBuffer_ || !pipeline) return;
    u32 n = 0;
    enc->setRenderPipelineState(pipeline);
    // Null: the encoder's default (no depth test, no write); overlay passes
    // without a depth attachment must not set a redundant state.
    if (depthState) {
        enc->setDepthStencilState(depthState);
        ++n;
    }
    enc->setArgumentTable(arguments_, MTL::RenderStageVertex | MTL::RenderStageFragment);
    enc->setViewport(MTL::Viewport{0.0, 0.0, static_cast<double>(frame_.width), static_cast<double>(frame_.height),
                                   0.0, 1.0});
    n += 3;

    // glTF convention: counter-clockwise front faces (Metal defaults to
    // clockwise).  A mirrored model matrix reverses the on-screen winding, so
    // mirrored instances cull FRONT faces instead: the same triangles as
    // flipping the winding, while [[front_facing]] keeps the meaning the
    // shader expects (it inverts it itself for INSTANCE_FLAG_MIRRORED).
    // Double-sided materials are not culled.  State is tracked from Metal's
    // defaults (clockwise, no culling) because the validation layer rejects
    // redundant state changes; ICB commands inherit it (D2: one fixed command
    // range per class).
    MTL::Winding  winding = MTL::WindingClockwise;
    MTL::CullMode cullMode = MTL::CullModeNone;
    const auto setState = [&](MTL::Winding w, MTL::CullMode c) {
        if (w != winding) {
            enc->setFrontFacingWinding(winding = w);
            ++n;
        }
        if (c != cullMode) {
            enc->setCullMode(cullMode = c);
            ++n;
        }
    };
    const auto classState = [&](CullClass cls) {
        switch (cls) {
        case CullClass::Back:         setState(MTL::WindingCounterClockwise, MTL::CullModeBack); break;
        case CullClass::BackMirrored: setState(MTL::WindingCounterClockwise, MTL::CullModeFront); break;
        case CullClass::None:         setState(MTL::WindingCounterClockwise, MTL::CullModeNone); break;
        }
    };

    if (frame_.mode == GpuDrivenMode::On) {
        const GpuSceneBuffers::FrameSet& fs = buffers_.frame(frame_.slot);
        const auto ranges = store_->classRanges();
        for (u32 c = 0; c < SCENE_CULL_CLASSES; ++c) {
            const SceneStore::ClassRange& r = ranges[c];
            const u32 first = r.firstCommand + r.commandCount * chunk / chunks;
            const u32 last  = r.firstCommand + r.commandCount * (chunk + 1) / chunks;
            if (last <= first) continue;
            classState(static_cast<CullClass>(c));
            enc->executeCommandsInBuffer(fs.icb, NS::Range::Make(first, last - first));
            ++n;
        }
    } else {
        // gpu-driven off: one direct draw per bucket, every live instance
        // (the visible list is the identity, so instance_id = slot).
        const MTL::GPUAddress indexBase = indexBuffer_->gpuAddress();
        const auto buckets = store_->buckets();
        const auto gpu     = store_->gpuBuckets();
        const size_t count = buckets.size();
        const size_t first = count * chunk / chunks;
        const size_t last  = count * (chunk + 1) / chunks;
        for (size_t i = first; i < last; ++i) {
            const SceneBucket& b = buckets[i];
            if (b.count == 0 || gpu[i].indexCount == 0) continue;
            classState(b.cull);
            enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, gpu[i].indexCount, MTL::IndexTypeUInt32,
                                       indexBase + static_cast<MTL::GPUAddress>(gpu[i].indexOffset) * sizeof(u32),
                                       static_cast<NS::UInteger>(gpu[i].indexCount) * sizeof(u32), b.count,
                                       gpu[i].vertexOffset, b.firstSlot);
            ++n;
        }
    }
    // Later passes fused into this render encoder (the ImGui overlay) start
    // from the default state.
    setState(MTL::WindingClockwise, MTL::CullModeNone);
    count(counterIndex, n);
}

u32 SceneRenderer::cpuCommands() const {
    u32 total = 0;
    for (const auto& c : cpuCommands_) total += c.load(std::memory_order_relaxed);
    return total;
}

GPUSceneCounters SceneRenderer::counters(u32 slot) const {
    GPUSceneCounters c{};
    const GpuSceneBuffers::FrameSet& fs = buffers_.frame(slot);
    if (fs.counters) std::memcpy(&c, fs.counters->contents(), sizeof(c));
    return c;
}

} // namespace phosphor
