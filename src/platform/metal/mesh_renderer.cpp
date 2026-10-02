#include "platform/metal/mesh_renderer.h"

#include "core/log.h"
#include "core/profile.h"
#include "pipeline/forward_variants.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scene_renderer.h"
#include "renderer/gpu_scene.h"
#include "renderer/meshlet_cull_math.h"
#include "renderer/meshlet_cull_reference.h"
#include "renderer/meshlet_layout.h"
#include "renderer/scene_store.h"
#include "rendergraph/pass_context.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

MTL4::ArgumentTable* newTable(MTL::Device* device, u32 buffers, u32 textures, const char* label) {
    NS::Error* error = nullptr;
    MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(buffers);
    if (textures) d->setMaxTextureBindCount(textures);
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
    d.functions = {function, "", ""};
    return d;
}

/// Mesh pipeline of the mesh path: the forward desc's constants (variant) and
/// output, object + mesh + fragment functions and the meshlet limits.
bool gObjectStage  = true;  // spike S2 variant (one MeshRenderer per process)
bool gTriangleCull = false; // F6.3 option

pipe::PipelineDesc meshDesc(const pipe::PipelineDesc& forward, const char* label) {
    pipe::PipelineDesc d = forward;
    d.kind                   = pipe::PipelineKind::Mesh;
    d.label                  = label;
    d.functions              = {MESHLET_OBJECT_FN, MESHLET_MESH_FN, "forward_fs"};
    d.indirectCommandBuffers = false;
    d.mesh = {MESHLET_OBJECT_GROUP, MESHLET_MESH_GROUP, MESHLET_PAYLOAD_BYTES, MESHLET_OBJECT_GROUP};
    if (gTriangleCull) d.functions[1] = MESHLET_MESH_TRICULL_FN;
    if (!gObjectStage) {
        d.functions = {"", MESHLET_MESH_DIRECT_FN, "forward_fs"};
        d.mesh      = {0, MESHLET_MESH_GROUP, 0, 0};
    }
    return d;
}

void dispatchBarrier(MTL4::ComputeCommandEncoder* enc) {
    enc->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}

u64 grow(u64 have, u64 need) { return need <= have ? have : std::max<u64>(need + need / 2, 1024); }

} // namespace

MeshRenderer::MeshRenderer(MetalContext& context, PipelineCache& pipelines, SceneRenderer& scene, const Options& options)
    : context_(context), pipelines_(pipelines), scene_(scene), options_(options) {
    gObjectStage  = options_.objectStage;
    gTriangleCull = options_.triangleCull;
    if (!options_.objectStage && options_.cull != MeshletCull::Off) {
        throw std::runtime_error("--meshlet-object off (spike S2) needs --meshlet-cull off");
    }
    if (options_.forceVariant && *options_.forceVariant >= pipe::forward::variantCount()) {
        throw std::runtime_error("--force-variant: expected 0.." + std::to_string(pipe::forward::variantCount() - 1));
    }
    generic_ = pipelines_.request(meshDesc(pipe::forward::genericDesc(rg::Format::BGRA8Srgb), "Meshlet forward (generic)"));
    variants_.assign(pipe::forward::variantCount(), pipe::INVALID_PIPELINE);
    {
        pipe::PipelineDesc d = meshDesc(pipe::forward::genericDesc(rg::Format::BGRA8Srgb), "Meshlet debug");
        d.functions[1] = MESHLET_MESH_DEBUG_FN;
        d.functions[2] = MESHLET_DEBUG_FS;
        debug_ = pipelines_.request(d);
    }
    kCandCount_ = pipelines_.request(kernelDesc(KERNEL_MESHLET_CAND_COUNT));
    kCandScan_  = pipelines_.request(kernelDesc(KERNEL_MESHLET_CAND_SCAN));
    kCandWrite_ = pipelines_.request(kernelDesc(KERNEL_MESHLET_CAND_WRITE));
    kBCount_    = pipelines_.request(kernelDesc(KERNEL_MESHLET_B_COUNT));
    kBScan_     = pipelines_.request(kernelDesc(KERNEL_MESHLET_B_SCAN));
    kBWrite_    = pipelines_.request(kernelDesc(KERNEL_MESHLET_B_WRITE));
    MTL::Device* device = context_.device();
    {
        pipe::PipelineDesc d;
        d.kind      = pipe::PipelineKind::Render;
        d.label     = "Hi-Z view";
        d.functions = {"hiz_view_vs", "hiz_view_fs", ""};
        d.output(0, rg::Format::BGRA8Srgb);
        hizView_      = pipelines_.request(d);
        hizViewTable_ = newTable(device, 1, 1, "Hi-Z view arguments");
        hizViewParams_ = context_.memory().newBuffer(sizeof(GPUHiZParams), MTL::ResourceStorageModeShared,
                                                     MemoryCategory::Other, "Hi-Z view parameters");
    }
    candTable_     = newTable(device, MB_BIND_COUNT, 0, "Meshlet candidates arguments");
    bTable_        = newTable(device, MB_BIND_COUNT, 0, "Meshlet B arguments");
    drawTables_[0] = newTable(device, MR_BIND_COUNT, 1, "Meshlet phase A arguments");
    drawTables_[1] = newTable(device, MR_BIND_COUNT, 1, "Meshlet phase B arguments");
    // The pyramid exists in every mesh-path mode (frustum/off bind it without
    // reading it, so the object shader's texture slot is always valid).
    hiz_ = std::make_unique<HiZBuilder>(context_, pipelines_, options_.hiz);
    LOG_INFO("Mesh path: meshlet cull %s, Hi-Z %s, debug view %s", meshletCullName(options_.cull), hiz_->backendName(),
             meshletDebugViewName(options_.debugView));
}

MeshRenderer::~MeshRenderer() {
    context_.waitIdle();
    releaseFrames();
    releaseGeometry();
    for (MTL4::ArgumentTable* t : {candTable_, bTable_, drawTables_[0], drawTables_[1], hizViewTable_}) t->release();
    context_.memory().release(hizViewParams_, MemoryCategory::Other);
}

void MeshRenderer::requestAllVariants() {
    for (u32 i = 0; i < variants_.size(); ++i) {
        if (variants_[i] == pipe::INVALID_PIPELINE) {
            variants_[i] = pipelines_.request(meshDesc(
                pipe::forward::pipelineDesc(pipe::forward::variantAt(i), rg::Format::BGRA8Srgb, options_.salt), "Meshlet forward"));
        }
    }
}

MTL::ComputePipelineState* MeshRenderer::kernel(pipe::PipelineHandle h) const { return pipelines_.compute(h); }

bool MeshRenderer::ready() const {
    for (pipe::PipelineHandle h : {kCandCount_, kCandScan_, kCandWrite_, kBCount_, kBScan_, kBWrite_}) {
        if (!kernel(h)) return false;
    }
    return hiz_->ready() && meshlets_ && frames_[0].candidates;
}

MTL::Buffer* MeshRenderer::privateBuffer(const void* data, size_t size, const char* label) {
    size = std::max<size_t>(size, 16);
    MTL::Buffer* buffer =
        context_.memory().newBuffer(size, MTL::ResourceStorageModePrivate, MemoryCategory::Geometry, label);
    const UploadRing::Slice staging = context_.stagingAllocate(size);
    std::memset(staging.cpu, 0, size);
    if (data) std::memcpy(staging.cpu, data, size);
    context_.enqueueUpload([staging, buffer, size](MTL4::ComputeCommandEncoder* enc) {
        enc->copyFromBuffer(staging.buffer, staging.offset, buffer, 0, size);
    });
    return buffer;
}

MTL::Buffer* MeshRenderer::sharedBuffer(u64 size, const char* label) {
    MTL::Buffer* b = context_.memory().newBuffer(std::max<u64>(size, 16), MTL::ResourceStorageModeShared,
                                                 MemoryCategory::Scene, label);
    std::memset(b->contents(), 0, b->length());
    return b;
}

void MeshRenderer::releaseGeometry() {
    for (MTL::Buffer** b : {&meshlets_, &meshletVertices_, &meshletTriangles_, &bounds_}) {
        if (*b) context_.memory().release(*b, MemoryCategory::Geometry);
        *b = nullptr;
    }
}

void MeshRenderer::releaseFrames() {
    GpuMemory& m = context_.memory();
    for (FrameSet& f : frames_) {
        for (MTL::Buffer** b : {&f.candidates, &f.bFlags, &f.bList, &f.groupSums, &f.bSums, &f.ranges, &f.args, &f.counters,
                                &f.gate, &f.cullParams, &f.decisions}) {
            if (*b) m.release(*b, MemoryCategory::Scene);
            *b = nullptr;
        }
    }
    for (MTL::Buffer** b : {&check_.history, &check_.current, &check_.next, &check_.depth}) {
        if (*b) m.release(*b, MemoryCategory::Other);
        *b = nullptr;
    }
    capacity_ = 0;
    slotCap_  = 0;
}

void MeshRenderer::syncGeometry(const GpuScene& scene) {
    if (scene.geometryVersion() == geometryVersion_) return;
    geometryVersion_ = scene.geometryVersion();
    context_.waitIdle();
    releaseGeometry();
    meshletCount_ = static_cast<u32>(scene.meshlets().size());
    if (scene.meshlets().empty()) return;
    meshlets_         = privateBuffer(scene.meshlets().data(), scene.meshlets().size() * sizeof(GPUMeshlet), "Meshlets");
    meshletVertices_  = privateBuffer(scene.meshletVertices().data(), scene.meshletVertices().size() * sizeof(u32),
                                      "Meshlet vertices");
    meshletTriangles_ = privateBuffer(scene.meshletTriangles().data(), scene.meshletTriangles().size(), "Meshlet triangles");
    bounds_           = privateBuffer(scene.meshletBounds().data(), scene.meshletBounds().size() * sizeof(GPUMeshletBounds),
                                      "Meshlet bounds");
    context_.flushUploads();
    LOG_INFO("Meshlets uploaded: %u meshlets (%s), %zu vertex refs, %zu triangle bytes", meshletCount_,
             meshletOptionsName(scene.meshletOptions()).c_str(), scene.meshletVertices().size(),
             scene.meshletTriangles().size());
}

void MeshRenderer::loadScene(const SceneStore& store, const GpuScene& scene) {
    releaseFrames(); // checks re-created with the next drawable size; released by the switch's collection
    width_ = height_ = 0;
    structureVersion_ = store.structureVersion();
    ensureCapacity(meshletCandidateCapacity(store.buckets(), scene.meshInfos()), store.slotCapacity());
}

void MeshRenderer::ensureCapacity(u64 capacity, u32 slotCount) {
    const u64 cap  = grow(capacity_, std::max<u64>(capacity, 1));
    const u32 slot = static_cast<u32>(grow(slotCap_, std::max<u32>(slotCount, 1)));
    if (cap == capacity_ && slot == slotCap_ && frames_[0].candidates) return;
    // Structure change (never expected in measured frames): new lists, old
    // ones released once the frames in flight are done.
    GpuMemory& m = context_.memory();
    const u64 groups     = (slot + MESHLET_SCAN_GROUP - 1) / MESHLET_SCAN_GROUP;
    const u64 candGroups = (cap + MESHLET_SCAN_GROUP - 1) / MESHLET_SCAN_GROUP;
    for (FrameSet& f : frames_) {
        auto swap = [&](MTL::Buffer*& b, u64 size, const char* label) {
            if (b) m.release(b, MemoryCategory::Scene);
            b = sharedBuffer(size, label);
        };
        if (cap != capacity_ || !f.candidates) {
            swap(f.candidates, cap * sizeof(GPUMeshletCandidate), "Meshlet candidates");
            swap(f.bFlags, cap * sizeof(u32), "Meshlet B flags");
            swap(f.bList, cap * sizeof(GPUMeshletCandidate), "Meshlet B list");
            swap(f.bSums, candGroups * 3 * sizeof(u32), "Meshlet B sums");
            swap(f.decisions, cap * 2 * sizeof(u32), "Meshlet decisions (self-check)");
        }
        if (slot != slotCap_ || !f.groupSums) swap(f.groupSums, groups * 3 * sizeof(u32), "Meshlet group sums");
        if (!f.ranges) {
            f.ranges     = sharedBuffer(MESHLET_DRAWS * sizeof(GPUMeshletDrawRange), "Meshlet draw ranges");
            f.args       = sharedBuffer(MESHLET_ARGS_WORDS * sizeof(u32), "Meshlet indirect arguments");
            f.counters   = sharedBuffer(sizeof(GPUMeshletCounters), "Meshlet counters");
            f.gate       = sharedBuffer(16, "Meshlet overflow gate");
            f.cullParams = sharedBuffer(sizeof(GPUMeshletCullParams), "Meshlet cull parameters (copy)");
        }
    }
    capacity_ = cap;
    slotCap_  = slot;
    ++version_;
    LOG_INFO("Meshlet lists: capacity %llu candidates, %u slots (%.2f MiB per frame slot)",
             static_cast<unsigned long long>(capacity_), slotCap_,
             static_cast<double>(capacity_ * (2 * sizeof(GPUMeshletCandidate) + sizeof(u32))) / (1 << 20));
}

void MeshRenderer::prepareFrame(const SceneStore& store, const GpuScene& scene, const FrameParams& params) {
    PH_ZONE("Meshlet prepare");
    frame_ = params;
    if (store.structureVersion() != structureVersion_ || !frames_[0].candidates) {
        structureVersion_ = store.structureVersion();
        ensureCapacity(meshletCandidateCapacity(store.buckets(), scene.meshInfos()), store.slotCapacity());
    }
    if (hiz_->resize(params.width, params.height)) frame_.historyValid = false;
    if (params.width != width_ || params.height != height_ || !check_.depth) {
        GpuMemory& m = context_.memory();
        for (MTL::Buffer** b : {&check_.history, &check_.current, &check_.next, &check_.depth}) {
            if (*b) m.release(*b, MemoryCategory::Other);
        }
        const u64 pyramid = std::max<u64>(hiz_->readbackBytes(), 16);
        check_.history = m.newBuffer(pyramid, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Meshlet check history");
        check_.current = m.newBuffer(pyramid, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Meshlet check current");
        check_.next    = m.newBuffer(pyramid, MTL::ResourceStorageModeShared, MemoryCategory::Other, "Meshlet check next");
        check_.depth   = m.newBuffer(u64(params.width) * params.height * 4, MTL::ResourceStorageModeShared,
                                     MemoryCategory::Other, "Meshlet check depth");
        width_  = params.width;
        height_ = params.height;
    }
    slotCount_       = store.slotCapacity();
    groups_          = std::max(1u, (slotCount_ + MESHLET_SCAN_GROUP - 1) / MESHLET_SCAN_GROUP);
    candidateGroups_ = static_cast<u32>(std::max<u64>(1, (capacity_ + MESHLET_SCAN_GROUP - 1) / MESHLET_SCAN_GROUP));

    GPUMeshletCullParams p = params.cull;
    std::memcpy(p.prevViewProj, params.prevViewProj, sizeof(p.prevViewProj));
    p.viewport[0]       = static_cast<float>(params.width);
    p.viewport[1]       = static_cast<float>(params.height);
    p.hizSize[0]        = hiz_->width0();
    p.hizSize[1]        = hiz_->height0();
    p.hizLevels         = hiz_->levels();
    p.slotCount         = slotCount_;
    p.groupCount        = groups_;
    p.candidateCapacity = static_cast<u32>(std::min<u64>(capacity_, 0xFFFFFFFFull));
    p.candidateGroups   = candidateGroups_;
    p.flags &= ~(MESHLET_CULL_FRUSTUM | MESHLET_CULL_CONE | MESHLET_CULL_OCCLUSION | MESHLET_CULL_HISTORY_VALID |
                 MESHLET_CULL_DEBUG_ALL | MESHLET_CULL_SIZE | MESHLET_CULL_RECORD);
    if (options_.cull != MeshletCull::Off) p.flags |= MESHLET_CULL_FRUSTUM | MESHLET_CULL_CONE;
    if (options_.cull != MeshletCull::Off && options_.minPixels > 0.0f) {
        p.flags |= MESHLET_CULL_SIZE;
        p.minPixels = options_.minPixels;
    }
    if (options_.cull == MeshletCull::TwoPhase) {
        p.flags |= MESHLET_CULL_OCCLUSION;
        if (frame_.historyValid) p.flags |= MESHLET_CULL_HISTORY_VALID;
    }
    if (options_.debugView == MeshletDebugView::Cull) p.flags |= MESHLET_CULL_DEBUG_ALL;
    if (readback_) p.flags |= MESHLET_CULL_RECORD;
    p.corruptId = params.corrupt == MeshletCorruption::Id ? 1u : 0u;
    // Count corruption: a capacity one short of this frame's need forces the
    // overflow path (the indexed fallback must still draw the whole frame).
    if (params.corrupt == MeshletCorruption::Count) p.candidateCapacity = 0;
    corruptDepth_ = params.corrupt == MeshletCorruption::Depth;
    {
        const UploadRing::Slice s = context_.frameUploads().allocate(sizeof(p));
        std::memcpy(s.cpu, &p, sizeof(p));
        params_ = s.gpu;
        std::memcpy(frames_[params.slot].cullParams->contents(), &p, sizeof(p));
    }
    // Draw gate: the F5 ICB draws only on overflow.
    scene_.setDrawGate(frames_[params.slot].gate->gpuAddress());

    // Pipelines: the forward variant of the frame (SceneRenderer chose it),
    // generic meanwhile; debug views use the debug pipeline.
    const u32 vi = options_.forceVariant ? *options_.forceVariant : scene_.forwardVariantIndex();
    pipe::PipelineHandle& handle = options_.genericOnly ? generic_ : variants_[vi];
    if (handle == pipe::INVALID_PIPELINE) {
        handle = pipelines_.request(
            meshDesc(pipe::forward::pipelineDesc(pipe::forward::variantAt(vi), rg::Format::BGRA8Srgb, options_.salt),
                     "Meshlet forward"));
    }
    const bool debug = options_.debugView == MeshletDebugView::Meshlets || options_.debugView == MeshletDebugView::Cull;
    usingFallback_ = !debug && (options_.genericOnly || !pipelines_.isFinal(handle));
    pipeline_      = pipelines_.render(debug ? debug_ : handle);
    if (!pipeline_) pipeline_ = pipelines_.render(generic_);
    if (usingFallback_) pipelines_.noteFallbackUse();

    // Draw tables (both phases): forward bindings + meshlet buffers.
    for (u32 ph = 0; ph < 2; ++ph) {
        MTL4::ArgumentTable* t = drawTables_[ph];
        const FrameSet& f      = frames_[params.slot];
        t->setAddress(scene_.frameConstantsAddress(), MR_FRAME);
        if (scene_.vertexBuffer()) t->setAddress(scene_.vertexBuffer()->gpuAddress(), MR_VERTICES);
        t->setAddress(scene_.buffers().instances()->gpuAddress(), MR_INSTANCES);
        t->setAddress(scene_.buffers().materials()->gpuAddress(), MR_MATERIALS);
        t->setAddress(scene_.lightsAddress(), MR_LIGHTS);
        t->setAddress(scene_.textureTableAddress(), MR_TEXTURES);
        if (meshlets_) {
            t->setAddress(meshlets_->gpuAddress(), MR_MESHLETS);
            t->setAddress(meshletVertices_->gpuAddress(), MR_MESHLET_VERTICES);
            t->setAddress(meshletTriangles_->gpuAddress(), MR_MESHLET_TRIANGLES);
            t->setAddress(bounds_->gpuAddress(), MR_BOUNDS);
        }
        t->setAddress(params_, MR_PARAMS);
        t->setAddress((ph == 0 ? f.candidates : f.bList)->gpuAddress(), MR_LIST);
        t->setAddress(f.bFlags->gpuAddress(), MR_B_FLAGS);
        t->setAddress(f.counters->gpuAddress(), MR_COUNTERS);
        t->setAddress(f.decisions->gpuAddress(), MR_DECISIONS);
        // Phase A tests the history (the previous frame's final pyramid);
        // phase B the current pyramid.
        MTL::Texture* tex = ph == 0 ? hiz_->history() : hiz_->current();
        if (tex) t->setTexture(tex->gpuResourceID(), MR_TEX_HIZ);
    }
}

GPUMeshletCounters MeshRenderer::counters(u32 slot) const {
    GPUMeshletCounters c{};
    if (frames_[slot].counters) std::memcpy(&c, frames_[slot].counters->contents(), sizeof(c));
    return c;
}

// ---- graph ---------------------------------------------------------------------------

rg::BufferRef MeshRenderer::addCandidatePass(rg::RenderGraph& graph) {
    using namespace rg;
    graphFrame_ = graph.importBuffer("Meshlet frame lists", {capacity_ * 20 + 4096}, ImportPerFrame | ImportOutput);
    graph.addPass(
        PASS_MESHLET_CANDIDATES, PassType::Compute,
        [&](PassBuilder& b) {
            b.read(scene_.dataRef(), Usage::ShaderRead, StageDispatch);
            b.read(scene_.frameListsRef(), Usage::ShaderRead, StageDispatch);
            graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageDispatch);
            b.setProfileShaders("meshlet_cand_count,meshlet_cand_scan,meshlet_cand_write");
        },
        [this](PassContext& ctx) { encodeCandidates(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder())); });
    return graphFrame_;
}

rg::TextureRef MeshRenderer::addRasterPasses(rg::RenderGraph& graph, rg::TextureRef& color, u32 width, u32 height) {
    using namespace rg;
    const bool two = twoPhase();
    // Pyramids: persistent imports (the passes use HiZBuilder's textures
    // directly).  The history is read by phase A and rewritten by Hi-Z final:
    // the graph orders the write after every read, and frame n+1's first
    // access after frame n's (persistent import).
    const TextureDesc hizDesc{Format::R32Float, std::max(hiz_->width0(), 1u), std::max(hiz_->height0(), 1u), 1,
                              std::max(hiz_->levels(), 1u)};
    hizHistoryRef_ = graph.importTexture("Hi-Z history", hizDesc, ImportContentsDefined | (two ? ImportOutput : 0u));
    if (two) hizCurrentRef_ = graph.importTexture("Hi-Z current", hizDesc, ImportOutput);
    readbackRef_ = graph.importBuffer("Meshlet check readback", {u64(width) * height * 4 + 3 * hiz_->readbackBytes()},
                                      ImportOutput);
    // Self-check: the history phase A will test against (check frames only).
    if (options_.checks) graph.addPass(
        "Meshlet check history", PassType::Blit,
        [&](PassBuilder& b) {
            b.read(hizHistoryRef_, Usage::CopySrc, StageBlit);
            readbackRef_ = b.write(readbackRef_, Usage::CopyDst, StageBlit);
            b.setSideEffect();
        },
        [this](PassContext& ctx) {
            if (readback_) hiz_->encodeReadback(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()), hiz_->history(), check_.history);
        });
    graph.addPass(
        "Forward", PassType::Raster,
        [&](PassBuilder& b) {
            ClearValue clear;
            clear.color[0] = 0.02f;
            clear.color[1] = 0.025f;
            clear.color[2] = 0.035f;
            clear.color[3] = 1.0f;
            clear.depth    = 0.0f; // reverse-Z: far = 0
            depthRef_      = b.createTexture("Depth", {Format::Depth32Float, width, height});
            color          = b.writeColor(color, 0, LoadIntent::Clear, clear);
            depthRef_      = b.writeDepth(depthRef_, LoadIntent::Clear, clear);
            b.read(scene_.dataRef(), Usage::ShaderRead, StageVertex | StageFragment | StageObject | StageMesh);
            // F5 ICB (overflow fallback) at the Vertex stage (F5-S5); the mesh
            // draws' arguments and lists at Object|Mesh (F6-S4: Vertex does
            // not order compute-written mesh arguments).
            b.read(scene_.frameListsRef(), Usage::IndirectArgs, StageVertex);
            b.read(graphFrame_, Usage::IndirectArgs, StageObject | StageMesh);
            b.read(hizHistoryRef_, Usage::ShaderRead, StageObject);
            if (two) graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageObject); // B flags
            b.setHints(HintGeometryHeavy);
            b.setProfileShaders("meshlet_object,meshlet_mesh,forward_fs");
        },
        [this](PassContext& ctx) { encodeRaster(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), MESHLET_PHASE_A); });
    if (two) {
        graph.addPass(
            PASS_HIZ_A, PassType::Compute,
            [&](PassBuilder& b) {
                b.read(depthRef_, Usage::ShaderRead, StageDispatch);
                hizCurrentRef_ = b.write(hizCurrentRef_, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("hiz_level0,hiz_reduce_simd,hiz_reduce_sampler");
            },
            [this](PassContext& ctx) {
                hiz_->encode(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),
                             static_cast<MTL::Texture*>(ctx.texture(depthRef_)), hiz_->current(), 0);
                scene_.countCommands(hiz_->commandCount());
            });
        graph.addPass(
            PASS_MESHLET_B, PassType::Compute,
            [&](PassBuilder& b) {
                b.read(graphFrame_, Usage::ShaderRead, StageDispatch);
                graphFrame_ = b.write(graphFrame_, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("meshlet_b_count,meshlet_b_scan,meshlet_b_write");
            },
            [this](PassContext& ctx) { encodePhaseB(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder())); });
        graph.addPass(
            PASS_FORWARD_B, PassType::Raster,
            [&](PassBuilder& b) {
                color     = b.writeColor(color, 0, LoadIntent::Preserve);
                depthRef_ = b.writeDepth(depthRef_, LoadIntent::Preserve);
                b.read(scene_.dataRef(), Usage::ShaderRead, StageFragment | StageObject | StageMesh);
                b.read(graphFrame_, Usage::IndirectArgs, StageObject | StageMesh);
                b.read(hizCurrentRef_, Usage::ShaderRead, StageObject);
                b.setHints(HintGeometryHeavy);
                b.setProfileShaders("meshlet_object,meshlet_mesh,forward_fs");
            },
            [this](PassContext& ctx) { encodeRaster(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), MESHLET_PHASE_B); });
        graph.addPass(
            PASS_HIZ_FINAL, PassType::Compute,
            [&](PassBuilder& b) {
                b.read(depthRef_, Usage::ShaderRead, StageDispatch);
                hizHistoryRef_ = b.write(hizHistoryRef_, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("hiz_level0,hiz_reduce_simd,hiz_reduce_sampler");
            },
            [this](PassContext& ctx) {
                hiz_->encode(static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder()),
                             static_cast<MTL::Texture*>(ctx.texture(depthRef_)), hiz_->history(), 1, corruptDepth_);
                scene_.countCommands(hiz_->commandCount());
            });
    }
    // F6.7 --debug-view hiz: the new history pyramid (final depth), one level.
    if (two && options_.debugView == MeshletDebugView::HiZ) {
        graph.addPass(
            "Hi-Z view", PassType::Raster,
            [&](PassBuilder& b) {
                color = b.writeColor(color, 0, LoadIntent::Preserve);
                b.read(hizHistoryRef_, Usage::ShaderRead, StageFragment);
                b.setProfileShaders("hiz_view_vs,hiz_view_fs");
            },
            [this](PassContext& ctx) {
                auto* enc = static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());
                MTL::RenderPipelineState* pso = pipelines_.render(hizView_);
                if (!pso || !hiz_->history()) return;
                GPUHiZParams p{{width_, height_}, {hiz_->width0(), hiz_->height0()}, options_.debugHiZLevel, hiz_->levels(), {0, 0}};
                std::memcpy(hizViewParams_->contents(), &p, sizeof(p));
                hizViewTable_->setAddress(hizViewParams_->gpuAddress(), 0);
                hizViewTable_->setTexture(hiz_->history()->gpuResourceID(), 0);
                enc->setRenderPipelineState(pso);
                enc->setArgumentTable(hizViewTable_, MTL::RenderStageFragment);
                enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
                scene_.countCommands(3);
            });
    } else if (options_.debugView == MeshletDebugView::HiZ) {
        LOG_WARN("--debug-view hiz needs --meshlet-cull two-phase (no pyramid otherwise)");
    }
    // Self-check: the final depth and the pyramids, copied only on check frames.
    if (options_.checks) graph.addPass(
        "Meshlet check readback", PassType::Blit,
        [&](PassBuilder& b) {
            b.read(depthRef_, Usage::CopySrc, StageBlit);
            if (two) {
                b.read(hizCurrentRef_, Usage::CopySrc, StageBlit);
                b.read(hizHistoryRef_, Usage::CopySrc, StageBlit);
            }
            readbackRef_ = b.write(readbackRef_, Usage::CopyDst, StageBlit);
            b.setSideEffect();
        },
        [this, two](PassContext& ctx) {
            if (!readback_ || !check_.depth) return;
            auto* enc = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
            enc->copyFromTexture(static_cast<MTL::Texture*>(ctx.texture(depthRef_)), 0, 0, MTL::Origin::Make(0, 0, 0),
                                 MTL::Size::Make(width_, height_, 1), check_.depth, 0, u64(width_) * 4,
                                 u64(width_) * height_ * 4);
            if (two) {
                hiz_->encodeReadback(enc, hiz_->current(), check_.current);
                hiz_->encodeReadback(enc, hiz_->history(), check_.next);
            }
        });
    return depthRef_;
}

void MeshRenderer::bindFrame(MetalGraphExecutor&, u32) const {}

// ---- encoding --------------------------------------------------------------------------

void MeshRenderer::encodeCandidates(MTL4::ComputeCommandEncoder* enc) const {
    PH_ZONE("Meshlet candidates encode");
    if (!ready() || !scene_.kernelsReady()) return;
    MTL4::ArgumentTable* t = candTable_;
    const FrameSet& f      = frames_[frame_.slot];
    const auto& sf         = scene_.buffers().frame(frame_.slot);
    t->setAddress(params_, MB_PARAMS);
    t->setAddress(scene_.buffers().instances()->gpuAddress(), MB_INSTANCES);
    t->setAddress(scene_.meshBuffer() ? scene_.meshBuffer()->gpuAddress() : params_, MB_MESHES);
    t->setAddress(scene_.buffers().materials()->gpuAddress(), MB_MATERIALS);
    t->setAddress(sf.flags->gpuAddress(), MB_SCENE_FLAGS);
    t->setAddress(f.groupSums->gpuAddress(), MB_GROUP_SUMS);
    t->setAddress(f.ranges->gpuAddress(), MB_RANGES);
    t->setAddress(f.args->gpuAddress(), MB_ARGS);
    t->setAddress(f.counters->gpuAddress(), MB_COUNTERS);
    t->setAddress(f.gate->gpuAddress(), MB_GATE);
    t->setAddress(f.candidates->gpuAddress(), MB_CANDIDATES);
    t->setAddress(f.bFlags->gpuAddress(), MB_B_FLAGS);
    t->setAddress(f.bSums->gpuAddress(), MB_B_SUMS);
    t->setAddress(f.bList->gpuAddress(), MB_B_LIST);
    const MTL::Size group = MTL::Size::Make(MESHLET_SCAN_GROUP, 1, 1);
    // The counters are cleared here (the frame's first meshlet pass): a fill
    // in the same encoder, ordered before the kernels by the barrier.
    enc->fillBuffer(f.counters, NS::Range::Make(0, sizeof(GPUMeshletCounters)), 0);
    enc->barrierAfterEncoderStages(MTL::StageBlit, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    enc->setComputePipelineState(kernel(kCandCount_));
    enc->setArgumentTable(t);
    enc->dispatchThreadgroups(MTL::Size::Make(groups_, 1, 1), group);
    dispatchBarrier(enc);
    enc->setComputePipelineState(kernel(kCandScan_));
    enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), group);
    dispatchBarrier(enc);
    enc->setComputePipelineState(kernel(kCandWrite_));
    enc->dispatchThreadgroups(MTL::Size::Make(groups_, 1, 1), group);
    scene_.countCommands(12);
}

void MeshRenderer::encodePhaseB(MTL4::ComputeCommandEncoder* enc) const {
    PH_ZONE("Meshlet B encode");
    if (!ready()) return;
    MTL4::ArgumentTable* t = bTable_;
    const FrameSet& f      = frames_[frame_.slot];
    t->setAddress(params_, MB_PARAMS);
    t->setAddress(f.ranges->gpuAddress(), MB_RANGES);
    t->setAddress(f.args->gpuAddress(), MB_ARGS);
    t->setAddress(f.candidates->gpuAddress(), MB_CANDIDATES);
    t->setAddress(f.bFlags->gpuAddress(), MB_B_FLAGS);
    t->setAddress(f.bSums->gpuAddress(), MB_B_SUMS);
    t->setAddress(f.bList->gpuAddress(), MB_B_LIST);
    const MTL::Size group = MTL::Size::Make(MESHLET_SCAN_GROUP, 1, 1);
    enc->setComputePipelineState(kernel(kBCount_));
    enc->setArgumentTable(t);
    enc->dispatchThreadgroups(MTL::Size::Make(candidateGroups_, 1, 1), group);
    dispatchBarrier(enc);
    enc->setComputePipelineState(kernel(kBScan_));
    enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), group);
    dispatchBarrier(enc);
    enc->setComputePipelineState(kernel(kBWrite_));
    enc->dispatchThreadgroups(MTL::Size::Make(candidateGroups_, 1, 1), group);
    scene_.countCommands(10);
}

void MeshRenderer::encodeRaster(MTL4::RenderCommandEncoder* enc, u32 phase) const {
    PH_ZONE("Meshlet raster encode");
    if (!ready() || !pipeline_) return;
    const FrameSet& f      = frames_[frame_.slot];
    MTL4::ArgumentTable* t = drawTables_[phase];
    u32 n = 0;
    enc->setDepthStencilState(scene_.depthState());
    enc->setViewport(MTL::Viewport{0.0, 0.0, static_cast<double>(width_), static_cast<double>(height_), 0.0, 1.0});
    n += 2;
    // Cull state per class, tracked from Metal's defaults (clockwise, none):
    // validation rejects redundant changes (as SceneRenderer::encodeForward).
    MTL::Winding winding   = MTL::WindingClockwise;
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
    const MTL::Size objectGroup = MTL::Size::Make(MESHLET_OBJECT_GROUP, 1, 1);
    const MTL::Size meshGroup   = MTL::Size::Make(MESHLET_MESH_GROUP, 1, 1);
    bool meshState = false; // the mesh pipeline + table are bound (the fallback rebinds the forward ones)
    for (u32 c = 0; c < SCENE_CULL_CLASSES; ++c) {
        switch (static_cast<CullClass>(c)) {
        case CullClass::Back:         setState(MTL::WindingCounterClockwise, MTL::CullModeBack); break;
        case CullClass::BackMirrored: setState(MTL::WindingCounterClockwise, MTL::CullModeFront); break;
        case CullClass::None:         setState(MTL::WindingCounterClockwise, MTL::CullModeNone); break;
        }
        const u32 draw = phase * SCENE_CULL_CLASSES + c;
        if (!meshState) {
            enc->setRenderPipelineState(pipeline_);
            ++n;
        }
        t->setAddress(f.ranges->gpuAddress() + draw * sizeof(GPUMeshletDrawRange), MR_RANGE);
        enc->setArgumentTable(t, MTL::RenderStageObject | MTL::RenderStageMesh | MTL::RenderStageFragment);
        if (options_.objectStage) {
            enc->drawMeshThreadgroups(f.args->gpuAddress() + draw * 3 * sizeof(u32), objectGroup, meshGroup);
        } else {
            enc->drawMeshThreadgroups(f.args->gpuAddress() + (MESHLET_ARGS_DIRECT + c * 3) * sizeof(u32),
                                      MTL::Size::Make(0, 0, 0), meshGroup);
        }
        n += 2;
        meshState = true;
        // Phase A: the class's F5 ICB range (empty unless the candidates overflowed).
        if (phase == MESHLET_PHASE_A) {
            const u32 fallback = scene_.encodeFallbackClass(enc, c);
            n += fallback;
            if (fallback) meshState = false;
        }
    }
    // Later passes fused into this render encoder start from the default state.
    setState(MTL::WindingClockwise, MTL::CullModeNone);
    scene_.countCommands(n);
}

} // namespace phosphor
