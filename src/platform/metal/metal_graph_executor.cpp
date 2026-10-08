#include "platform/metal/pipeline_cache.h"
#include "platform/metal/metal_graph_executor.h"
#include <cstdlib>
#include <cstring>
#include "platform/metal/gpu_memory.h"
#include "platform/metal/gpu_timestamps.h"
#include "rendergraph/timing_plan.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"

#include <algorithm>

namespace phosphor {

namespace {

constexpr u32 kNone = ~0u;

NS::String* retainedString(const std::string& s) {
    return NS::String::string(s.c_str(), NS::UTF8StringEncoding)->retain();
}

bool hasStencil(rg::Format format) { return format == rg::Format::Depth32FloatStencil8; }

MTL::LoadAction toMetal(rg::LoadAction action) {
    switch (action) {
    case rg::LoadAction::DontCare: return MTL::LoadActionDontCare;
    case rg::LoadAction::Load:     return MTL::LoadActionLoad;
    case rg::LoadAction::Clear:    return MTL::LoadActionClear;
    }
    return MTL::LoadActionDontCare;
}

MTL::StoreAction toMetal(rg::StoreAction action) {
    return action == rg::StoreAction::Store ? MTL::StoreActionStore : MTL::StoreActionDontCare;
}

MTL::TextureDescriptor* textureDescriptor(const rg::TextureDesc& desc, MTL::TextureUsage usage,
                                          MTL::StorageMode storage) {
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::alloc()->init();
    if (desc.sampleCount > 1) {
        d->setTextureType(desc.depth > 1 ? MTL::TextureType2DMultisampleArray : MTL::TextureType2DMultisample);
    } else {
        d->setTextureType(desc.depth > 1 ? MTL::TextureType2DArray : MTL::TextureType2D);
    }
    d->setPixelFormat(toMetalFormat(desc.format));
    d->setWidth(desc.width);
    d->setHeight(desc.height);
    d->setArrayLength(std::max(desc.depth, 1u));
    d->setMipmapLevelCount(std::max(desc.mipLevels, 1u));
    d->setSampleCount(std::max(desc.sampleCount, 1u));
    d->setUsage(usage);
    d->setStorageMode(storage);
    return d; // caller releases
}

} // namespace

MTL::PixelFormat toMetalFormat(rg::Format format) {
    switch (format) {
    case rg::Format::Unknown:              return MTL::PixelFormatInvalid;
    case rg::Format::R8Unorm:              return MTL::PixelFormatR8Unorm;
    case rg::Format::RG8Unorm:             return MTL::PixelFormatRG8Unorm;
    case rg::Format::RGBA8Unorm:           return MTL::PixelFormatRGBA8Unorm;
    case rg::Format::RGBA8Srgb:            return MTL::PixelFormatRGBA8Unorm_sRGB;
    case rg::Format::BGRA8Unorm:           return MTL::PixelFormatBGRA8Unorm;
    case rg::Format::BGRA8Srgb:            return MTL::PixelFormatBGRA8Unorm_sRGB;
    case rg::Format::R16Float:             return MTL::PixelFormatR16Float;
    case rg::Format::RG16Float:            return MTL::PixelFormatRG16Float;
    case rg::Format::RGBA16Float:          return MTL::PixelFormatRGBA16Float;
    case rg::Format::R32Float:             return MTL::PixelFormatR32Float;
    case rg::Format::RG32Float:            return MTL::PixelFormatRG32Float;
    case rg::Format::RGBA32Float:          return MTL::PixelFormatRGBA32Float;
    case rg::Format::R32Uint:              return MTL::PixelFormatR32Uint;
    case rg::Format::RGBA32Uint:           return MTL::PixelFormatRGBA32Uint;
    case rg::Format::RG11B10Float:         return MTL::PixelFormatRG11B10Float;
    case rg::Format::RGB10A2Unorm:         return MTL::PixelFormatRGB10A2Unorm;
    case rg::Format::Depth16Unorm:         return MTL::PixelFormatDepth16Unorm;
    case rg::Format::Depth32Float:         return MTL::PixelFormatDepth32Float;
    case rg::Format::Depth32FloatStencil8: return MTL::PixelFormatDepth32Float_Stencil8;
    }
    return MTL::PixelFormatInvalid;
}

MTL::Stages toMetalStages(rg::Stages stages) {
    MTL::Stages out = 0;
    if (stages & rg::StageVertex)   out |= MTL::StageVertex;
    if (stages & rg::StageFragment) out |= MTL::StageFragment;
    if (stages & rg::StageTile)     out |= MTL::StageTile;
    if (stages & rg::StageObject)   out |= MTL::StageObject;
    if (stages & rg::StageMesh)     out |= MTL::StageMesh;
    if (stages & rg::StageDispatch) out |= MTL::StageDispatch;
    if (stages & rg::StageBlit)     out |= MTL::StageBlit;
    if (stages & rg::StageAccelerationStructure) out |= MTL::StageAccelerationStructure;
    if (stages & rg::StageMachineLearning)
        out |= MTL::StageMachineLearning;
    return out;
}

// --- Heap footprints for the aliasing plan ------------------------------------

class MetalGraphExecutor::Sizer final : public rg::ResourceSizer {
public:
    Sizer(const TransientHeap& heap, const std::vector<MTL::TextureUsage>& usage) : heap_(heap), usage_(usage) {}

    rg::SizeAlign textureSize(u32 resource, const rg::TextureDesc& desc) const override {
        MTL::TextureDescriptor* d = textureDescriptor(desc, usage_[resource], MTL::StorageModePrivate);
        const MTL::SizeAndAlign sa = heap_.sizeAndAlign(d);
        d->release();
        return {sa.size, sa.align};
    }
    rg::SizeAlign bufferSize(u32, const rg::BufferDesc& desc) const override {
        const MTL::SizeAndAlign sa = heap_.sizeAndAlign(desc.size);
        return {sa.size, sa.align};
    }

private:
    const TransientHeap& heap_;
    const std::vector<MTL::TextureUsage>& usage_;
};

// --- What a pass callback sees --------------------------------------------------

class MetalGraphExecutor::Context final : public rg::PassContext {
public:
  Context(const MetalGraphExecutor &executor, MTL4::CommandEncoder *encoder, u64 frame, u32 chunk, u32 chunks,
          MTL4::CommandBuffer *command = nullptr)
      : executor_(executor), encoder_(encoder), frame_(frame), chunk_(chunk), chunks_(chunks), command_(command) {}

  void *encoder() const override { return encoder_; }
  void *commandBuffer() const override { return command_; }
  void *externalFence() const override { return executor_.externalFence_; }
  void externalDependency(const ExternalDependency &d) override {
      if (!command_ || dependency.resume || !d.inputReady || !d.outputReady || !d.value || !d.resume)
          throw std::logic_error("Invalid external submission dependency");
      dependency = d;
  }
  ExternalDependency dependency;
  void *texture(rg::TextureRef t) const override { return executor_.textures_[t.resource]; }
  void *buffer(rg::BufferRef b) const override { return executor_.buffers_[b.resource]; }
  void *accelerationStructure(rg::AccelerationStructureRef a) const override {
      return executor_.accelerationStructures_[a.resource];
  }
  u32 chunk() const override { return chunk_; }
  u32 chunkCount() const override { return chunks_; }
  u64 frameIndex() const override { return frame_; }

private:
    const MetalGraphExecutor& executor_;
    MTL4::CommandEncoder*     encoder_;
    u64 frame_;
    u32 chunk_;
    u32 chunks_;
    MTL4::CommandBuffer *command_;
};

// --- Executor ---------------------------------------------------------------------

MetalGraphExecutor::MetalGraphExecutor(MetalContext &context, PipelineCache *pipelines)
    : pipelines_(pipelines), context_(context), heap_(context) {
    if (pipelines_) {
        pipe::PipelineDesc d;
        d.kind = pipe::PipelineKind::Compute;
        d.label = "External encoder boundary";
        d.functions = {"timestamp_anchor", "", ""};
        externalAnchor_ = pipelines_->request(d);
        externalFence_ = context_.device()->newFence();
    }
}

MetalGraphExecutor::~MetalGraphExecutor() {
    context_.waitIdle();
    if (splitFence_) splitFence_->release();
    if (externalFence_)
        externalFence_->release();
    releaseResources();
    workers_.reset();
    for (ExtraCommandBuffer& e : extra_) {
        for (u32 slot = 0; slot < METAL_FRAMES_IN_FLIGHT; ++slot) {
            e.buffers[slot]->release();
            e.allocators[slot]->release();
        }
    }
}

void MetalGraphExecutor::ensureParallelResources(u32 maxChunks) {
    if (maxChunks <= 1) return;
    // Worst case: every chunk and the tail of every split group (bounded by
    // the frame's extra command buffer slots).
    while (extra_.size() < MetalContext::MAX_FRAME_COMMAND_BUFFERS - 1) {
        ExtraCommandBuffer& e = extra_.emplace_back();
        for (u32 slot = 0; slot < METAL_FRAMES_IN_FLIGHT; ++slot) {
            e.allocators[slot] = context_.device()->newCommandAllocator();
            e.buffers[slot]    = context_.device()->newCommandBuffer();
            e.residencyGenerations[slot] = context_.residencyGeneration();
        }
    }
    if (!workers_ || workers_->workerCount() < maxChunks - 1) {
        workers_ = std::make_unique<WorkerPool>(maxChunks - 1);
    }
}

void MetalGraphExecutor::releaseResources() {
    for (size_t r = 0; r < owned_.size(); ++r) {
        if (!owned_[r]) continue;
        if (textures_[r]) {
            if (textures_[r]->storageMode() == MTL::StorageModeMemoryless) {
                context_.memory().release(textures_[r], MemoryCategory::RenderTargets);
            } else {
                heap_.release(textures_[r], /*evict: no-op if it was never registered*/ true);
            }
        }
        if (buffers_[r]) heap_.release(buffers_[r]);
    }
    for (MTL4::RenderPassDescriptor* d : passDescriptors_) d->release();
    for (auto* labels : {&groupLabels_, &passLabels_, &encoderLabels_}) {
        for (NS::String* s : *labels) {
            if (s) s->release();
        }
        labels->clear();
    }
    passDescriptors_.clear();
    textures_.clear();
    buffers_.clear();
    accelerationStructures_.clear();
    owned_.clear();
}

bool MetalGraphExecutor::compile(const rg::RenderGraph& graph, const rg::CompileOptions& options) {
    const auto& resources = graph.resources();

    // Texture usage flags from the accesses: they decide the heap footprint.
    usage_.assign(resources.size(), MTL::TextureUsageUnknown);
    for (const rg::PassNode& pass : graph.passes()) {
        for (const auto* list : {&pass.reads, &pass.writes}) {
            for (const rg::Access& a : *list) {
                switch (a.usage) {
                case rg::Usage::ColorAttachment:
                case rg::Usage::DepthAttachment:
                case rg::Usage::DepthRead:   usage_[a.resource] |= MTL::TextureUsageRenderTarget; break;
                case rg::Usage::ShaderRead:  usage_[a.resource] |= MTL::TextureUsageShaderRead; break;
                case rg::Usage::ShaderWrite: usage_[a.resource] |= MTL::TextureUsageShaderWrite; break;
                default: break;
                }
            }
        }
    }

    Sizer sizer(heap_, usage_);
    rg::CompileOptions opts = options;
    opts.sizer = &sizer;
    rg::CompiledGraph compiled = rg::compile(graph, opts);
    if (!compiled.ok) {
        for (const std::string& e : compiled.errors) LOG_ERROR("Render graph: %s", e.c_str());
        return false;
    }

    // The previous resources may still be used by frames in flight: their
    // release is deferred.
    releaseResources();
    graph_    = &graph;
    compiled_ = std::move(compiled);
    if (!createResources()) {
        graph_ = nullptr;
        return false;
    }
    buildPassDescriptors();

    // F2.5: raster passes to encode on several threads (one per group).
    splitPosition_.assign(compiled_.renderGroups.size(), kNone);
    hasSplit_ = false;
    u32 maxChunks = 1, extraNeeded = 0;
    for (u32 g = 0; g < compiled_.renderGroups.size(); ++g) {
        const rg::RenderGroup& group = compiled_.renderGroups[g];
        for (u32 pos = group.firstPosition; pos <= group.lastPosition; ++pos) {
            const u32 chunks = graph.passes()[compiled_.order[pos]].parallelChunks;
            if (chunks <= 1) continue;
            if (splitPosition_[g] != kNone) {
                LOG_WARN("Render graph: one split pass per render pass; '%s' is encoded on one thread",
                         graph.passes()[compiled_.order[pos]].name.c_str());
                continue;
            }
            splitPosition_[g] = pos;
            maxChunks = std::max(maxChunks, chunks);
            LOG_INFO("Render graph: '%s' encoded by %u threads (render pass suspended/resumed across %u command "
                     "buffers)", graph.passes()[compiled_.order[pos]].name.c_str(), chunks, chunks + 2);
            extraNeeded += chunks + 3; // head + chunks + tail + continuation
            hasSplit_ = true;
        }
    }
    if (extraNeeded > MetalContext::MAX_FRAME_COMMAND_BUFFERS - 1) {
        LOG_ERROR("Render graph: parallel encoding needs %u extra command buffers (max %u)", extraNeeded,
                  MetalContext::MAX_FRAME_COMMAND_BUFFERS - 1);
        graph_ = nullptr;
        return false;
    }
    ensureParallelResources(maxChunks);
    if (hasSplit_ && !splitFence_) splitFence_ = context_.device()->newSharedEvent();

    barrierIndex_.assign(compiled_.order.size(), kNone);
    for (u32 i = 0; i < compiled_.barriers.size(); ++i) barrierIndex_[compiled_.barriers[i].position] = i;
    // F2.6: submission boundaries at the queue sync points.
    waitBefore_.assign(compiled_.order.size(), 0);
    signalAfter_.assign(compiled_.order.size(), 0);
    segmented_ = false;
    for (const rg::EncoderPlan &e : compiled_.encoders)
        segmented_ |= e.queue == rg::Queue::AsyncCompute || e.type == rg::PassType::External;
    for (const rg::QueueSync& q : compiled_.queueSyncs) {
        waitBefore_[q.waitBeforePosition]   = std::max(waitBefore_[q.waitBeforePosition], q.value);
        signalAfter_[q.signalAfterPosition] = std::max(signalAfter_[q.signalAfterPosition], q.value);
    }
    if (compiled_.queueSyncs.size() + 1 >= MetalContext::TIMELINE_STRIDE ||
        2 * compiled_.queueSyncs.size() + 2 > MetalContext::MAX_FRAME_SUBMISSIONS) {
        LOG_ERROR("Render graph: too many queue syncs (%zu)", compiled_.queueSyncs.size());
        graph_ = nullptr;
        return false;
    }
    if (segmented_) ensureParallelResources(2); // extra command buffers for the submissions
    if (segmented_) {
        LOG_INFO("Render graph: segmented submissions, %zu cross-queue syncs", compiled_.queueSyncs.size());
    }

    // F4.1: timed units; a position maps to the unit that ENDS there.
    unitOfPosition_.assign(compiled_.order.size(), kNone);
    if (timestamps_) {
        const rg::TimingPlan plan = rg::buildTimingPlan(graph, compiled_, MetalContext::MAX_FRAME_SUBMISSIONS);
        for (u32 u = 0; u < plan.units.size(); ++u) unitOfPosition_[plan.units[u].lastPosition] = u;
        timestamps_->configure(plan);
    }

    ++compileCount_;
    u32 culled = 0;
    for (const bool c : compiled_.culled) culled += c ? 1 : 0;
    u32 memoryless = 0;
    for (const bool m : compiled_.memoryless) memoryless += m ? 1 : 0;
    LOG_INFO("Render graph compiled: %zu passes (%u culled), %zu encoders, %zu render passes, %zu barrier points, "
             "%u memoryless, transient heap %.2f MiB (%.2f MiB without aliasing)",
             graph.passes().size(), culled, compiled_.encoders.size(), compiled_.renderGroups.size(),
             compiled_.barriers.size(), memoryless, static_cast<double>(compiled_.aliasing.heapSize) / (1 << 20),
             static_cast<double>(compiled_.aliasing.unaliasedSize) / (1 << 20));
    return true;
}

bool MetalGraphExecutor::createResources() {
    const auto& resources = graph_->resources();
    textures_.assign(resources.size(), nullptr);
    buffers_.assign(resources.size(), nullptr);
    accelerationStructures_.assign(resources.size(), nullptr);
    owned_.assign(resources.size(), false);

    if (compiled_.aliasing.heapSize > 0 && !heap_.reserve(compiled_.aliasing.heapSize)) {
        LOG_ERROR("Render graph: cannot reserve a %llu-byte transient heap",
                  static_cast<unsigned long long>(compiled_.aliasing.heapSize));
        return false;
    }
    for (const rg::Placement& p : compiled_.aliasing.placements) {
        const rg::ResourceNode& node = resources[p.resource];
        if (node.kind == rg::ResourceKind::Texture) {
            MTL::TextureDescriptor* d = textureDescriptor(node.texture, usage_[p.resource], MTL::StorageModePrivate);
            d->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
            if(const char* diagnostic=std::getenv("PHOSPHOR_DIAGNOSTIC_MOTION_READBACK");diagnostic&&std::strcmp(diagnostic,"1")==0){
                const auto actual=heap_.sizeAndAlign(d); // Exact descriptor passed to newTexture below.
                LOG_INFO("GRAPH_FOOTPRINT resource %u '%s' offset %llu planned %llu actual %llu align %llu usage %u hazard %u",
                    p.resource,node.name.c_str(),static_cast<unsigned long long>(p.offset),static_cast<unsigned long long>(p.size),
                    static_cast<unsigned long long>(actual.size),static_cast<unsigned long long>(actual.align),unsigned(d->usage()),unsigned(d->hazardTrackingMode()));
                if(actual.size>p.size||!actual.align||p.offset%actual.align){LOG_ERROR("Final Untracked texture does not fit its planned placement");d->release();return false;}
            }
            textures_[p.resource] = heap_.createTexture(d, p.offset, node.name.c_str());
            // The heap makes its textures resident, but the Metal 4 validation
            // layer rejects a heap-backed render target that is later bound
            // through an argument table in a render pass ("attachment texture
            // ... is not added to any residency set", F4.7 overlays: light
            // count / overdraw sampled by the composite) unless the texture is
            // in a residency set itself.  Evicted again in releaseResources().
            if (textures_[p.resource] && (usage_[p.resource] & MTL::TextureUsageRenderTarget)) {
                context_.makeResident(textures_[p.resource], ResidencyClass::Static);
            }
            d->release();
        } else {
            buffers_[p.resource] = heap_.createBuffer(node.buffer.size, p.offset, node.name.c_str());
        }
        if (!textures_[p.resource] && !buffers_[p.resource]) return false;
        owned_[p.resource] = true;
    }
    for (u32 r = 0; r < resources.size(); ++r) {
        if (!compiled_.memoryless[r]) continue;
        MTL::TextureDescriptor* d =
            textureDescriptor(resources[r].texture, MTL::TextureUsageRenderTarget, MTL::StorageModeMemoryless);
        textures_[r] = context_.memory().newTexture(d, MemoryCategory::RenderTargets,
                                                    (resources[r].name + " (memoryless)").c_str());
        d->release();
        if (!textures_[r]) return false;
        owned_[r] = true;
    }
    return true;
}

void MetalGraphExecutor::buildPassDescriptors() {
    const auto& resources = graph_->resources();
    for (const rg::RenderGroup& group : compiled_.renderGroups) {
        MTL4::RenderPassDescriptor* desc = MTL4::RenderPassDescriptor::alloc()->init();
        if (group.tileWidth) {
            desc->setTileWidth(group.tileWidth);
            desc->setTileHeight(group.tileHeight);
        }
        for (const rg::AttachmentPlan& a : group.attachments) {
            MTL::Texture* texture = resources[a.resource].imported ? nullptr : textures_[a.resource];
            if (a.depth) {
                MTL::RenderPassDepthAttachmentDescriptor* d = desc->depthAttachment();
                d->setTexture(texture);
                d->setLoadAction(toMetal(a.load));
                d->setStoreAction(toMetal(a.store));
                d->setClearDepth(a.clear.depth);
                if (hasStencil(resources[a.resource].texture.format)) {
                    MTL::RenderPassStencilAttachmentDescriptor* s = desc->stencilAttachment();
                    s->setTexture(texture);
                    s->setLoadAction(toMetal(a.load));
                    s->setStoreAction(toMetal(a.store));
                    s->setClearStencil(a.clear.stencil);
                }
            } else {
                MTL::RenderPassColorAttachmentDescriptor* c = desc->colorAttachments()->object(a.slot);
                c->setTexture(texture);
                c->setLoadAction(toMetal(a.load));
                c->setStoreAction(toMetal(a.store));
                c->setClearColor(MTL::ClearColor::Make(a.clear.color[0], a.clear.color[1], a.clear.color[2],
                                                       a.clear.color[3]));
            }
        }
        passDescriptors_.push_back(desc);
        std::string label;
        for (u32 pos = group.firstPosition; pos <= group.lastPosition; ++pos) {
            label += (label.empty() ? "" : " + ") + graph_->passes()[compiled_.order[pos]].name;
        }
        groupLabels_.push_back(retainedString(label));
    }
    for (const rg::PassNode& pass : graph_->passes()) passLabels_.push_back(retainedString(pass.name));
    for (const rg::EncoderPlan& e : compiled_.encoders) {
        if (e.type == rg::PassType::Raster) {
            encoderLabels_.push_back(nullptr);
            continue;
        }
        std::string label;
        for (u32 pos = e.firstPosition; pos <= e.lastPosition; ++pos) {
            label += (label.empty() ? "" : " + ") + graph_->passes()[compiled_.order[pos]].name;
        }
        encoderLabels_.push_back(retainedString(label));
    }
}

void MetalGraphExecutor::bindTexture(rg::TextureRef texture, MTL::Texture* physical) {
    if (texture.resource < textures_.size()) textures_[texture.resource] = physical;
}

void MetalGraphExecutor::bindBuffer(rg::BufferRef buffer, MTL::Buffer* physical) {
    if (buffer.resource < buffers_.size()) buffers_[buffer.resource] = physical;
}

void MetalGraphExecutor::bindAccelerationStructure(rg::AccelerationStructureRef structure,
                                                   MTL::AccelerationStructure* physical) {
    if (structure.resource < accelerationStructures_.size()) accelerationStructures_[structure.resource] = physical;
}

void MetalGraphExecutor::encodeBarriers(MTL4::CommandEncoder* encoder, u32 position) const {
    const u32 index = barrierIndex_[position];
    if (index == kNone) return;
    for (const rg::Barrier& b : compiled_.barriers[index].barriers) {
        const MTL4::VisibilityOptions visibility =
            MTL4::VisibilityOptionDevice | (b.aliasing ? MTL4::VisibilityOptionResourceAlias : 0);
        const MTL::Stages after  = toMetalStages(b.afterStages);
        const MTL::Stages before = toMetalStages(b.beforeStages);
        if (b.scope == rg::BarrierScope::Encoder) {
            encoder->barrierAfterEncoderStages(after, before, visibility);
        } else {
            encoder->barrierAfterQueueStages(after, before, visibility);
        }
    }
}

void MetalGraphExecutor::runPass(MTL4::CommandEncoder* encoder, u32 position, const MetalContext::Frame& frame,
                                 u32 chunk, u32 chunks) const {
    const u32 pass = compiled_.order[position];
    const rg::PassNode& node = graph_->passes()[pass];
    encoder->pushDebugGroup(passLabels_[pass]);
    if (node.execute) {
        Context ctx(*this, encoder, frame.index, chunk, chunks);
        node.execute(ctx);
    }
    encoder->popDebugGroup();
}

void MetalGraphExecutor::setImportedAttachments(u32 group, bool bind) {
    const auto& resources = graph_->resources();
    MTL4::RenderPassDescriptor* desc = passDescriptors_[group];
    for (const rg::AttachmentPlan& a : compiled_.renderGroups[group].attachments) {
        if (!resources[a.resource].imported) continue;
        MTL::Texture* texture = bind ? textures_[a.resource] : nullptr;
        if (a.depth) {
            desc->depthAttachment()->setTexture(texture);
        } else {
            desc->colorAttachments()->object(a.slot)->setTexture(texture);
        }
    }
}

MTL4::CommandBuffer* MetalGraphExecutor::beginExtraCommandBuffer(MetalContext::Frame& frame, u32 index, bool async) {
    ExtraCommandBuffer& e = extra_[index];
    context_.refreshCommandBuffer(e.buffers[frame.slot], e.residencyGenerations[frame.slot]);
    e.allocators[frame.slot]->reset();
    MTL4::CommandBuffer* cmd = e.buffers[frame.slot];
    cmd->beginCommandBuffer(e.allocators[frame.slot]);
    if (segmented_) {
        addBuffer(frame, async, cmd);
    } else {
        frame.buffers[frame.bufferCount++] = cmd;
    }
    return cmd;
}

u32 MetalGraphExecutor::openSubmission(MetalContext::Frame& frame, MetalContext::SubmitQueue queue, u64 waitValue,
                                       u64 waitFrame) {
    const u32 index = frame.submissionCount++;
    frame.submissions[index] = MetalContext::Submission{queue, frame.bufferCount, 0, waitValue, 0, waitFrame};
    return index;
}

void MetalGraphExecutor::addBuffer(MetalContext::Frame& frame, bool async, MTL4::CommandBuffer* cmd) {
    u32& sub = async ? asyncSub_ : graphicsSub_;
    // A submission is a contiguous range of Frame::buffers: if the other
    // queue appended in between, continue in a new (unsynchronised) commit.
    const MetalContext::Submission& s = frame.submissions[sub];
    if (s.bufferCount > 0 && s.firstBuffer + s.bufferCount != frame.bufferCount) {
        sub = openSubmission(frame, s.queue, 0, 0);
    }
    frame.buffers[frame.bufferCount++] = cmd;
    ++frame.submissions[sub].bufferCount;
}

void MetalGraphExecutor::cutSubmissionForSplit(MetalContext::Frame& frame) {
    MetalContext::Submission& current = frame.submissions[segmented_ ? graphicsSub_ : frame.submissionCount - 1];
    if (!segmented_) current.bufferCount = frame.bufferCount - current.firstBuffer;
    current.fenceEvent  = splitFence_;
    current.fenceSignal = ++splitFenceValue_;
    const u32 next = openSubmission(frame, MetalContext::SubmitQueue::Graphics, 0, 0);
    frame.submissions[next].fenceEvent = splitFence_;
    frame.submissions[next].fenceWait  = splitFenceValue_;
    if (segmented_) graphicsSub_ = next;
}

void MetalGraphExecutor::encodeChunkJob(void* user, u32 chunk) {
    const ChunkJob& job = *static_cast<const ChunkJob*>(user);
    job.executor->runPass(job.encoders[chunk], job.position, *job.frame, chunk, job.chunks);
    job.encoders[chunk]->endEncoding();
}

MTL4::CommandBuffer* MetalGraphExecutor::encodeSplitGroup(MetalContext::Frame& frame, MTL4::CommandBuffer* cmd,
                                                          u32 group, u32 splitPosition) {
    const rg::RenderGroup& plan = compiled_.renderGroups[group];
    MTL4::RenderPassDescriptor* desc = passDescriptors_[group];
    const u32 chunks = graph_->passes()[compiled_.order[splitPosition]].parallelChunks;

    // Measured in F5 (M5 Max, macOS 27.2): no barrier orders the pieces of a
    // render pass resumed in other command buffers after earlier work of the
    // same commit -- neither the group's queue barriers (at the head or at
    // every piece) nor a producer barrierAfterStages at the end of the
    // previous compute encoder: chunks intermittently read scene data the
    // compute passes had not written yet (bench 6/8 frames = the previous
    // frame).  So the split pass starts a new commit that waits on an event
    // signalled after everything before it, and the next frame's first
    // commit waits for this one (the same hazard in reverse): debug mode only.
    {
        MTL4::CommandBuffer* previous = cmd;
        cutSubmissionForSplit(frame);
        cmd = beginExtraCommandBuffer(frame, frame.bufferCount - 1);
        if (previous != frame.commandBuffer) previous->endCommandBuffer();
    }

    // Every piece of the render pass is created here, on this thread, from
    // the same descriptor; only the recording of the chunks is parallel.
    setImportedAttachments(group, true);
    MTL4::RenderCommandEncoder* head = cmd->renderCommandEncoder(desc, MTL4::RenderEncoderOptionSuspending);
    chunkJob_.executor = this;
    chunkJob_.frame    = &frame;
    chunkJob_.position = splitPosition;
    chunkJob_.chunks   = chunks;
    for (u32 c = 0; c < chunks; ++c) {
        MTL4::CommandBuffer* chunkCmd = beginExtraCommandBuffer(frame, frame.bufferCount - 1);
        chunkJob_.encoders[c] = chunkCmd->renderCommandEncoder(
            desc, MTL4::RenderEncoderOptionResuming | MTL4::RenderEncoderOptionSuspending);
        chunkJob_.encoders[c]->setLabel(passLabels_[compiled_.order[splitPosition]]);
    }
    MTL4::CommandBuffer* tailCmd = beginExtraCommandBuffer(frame, frame.bufferCount - 1);
    MTL4::RenderCommandEncoder* tail = tailCmd->renderCommandEncoder(desc, MTL4::RenderEncoderOptionResuming);
    setImportedAttachments(group, false);

    head->setLabel(groupLabels_[group]);
    encodeBarriers(head, plan.firstPosition);

    for (u32 pos = plan.firstPosition; pos < splitPosition; ++pos) runPass(head, pos, frame);
    head->endEncoding();
    if (cmd != frame.commandBuffer) cmd->endCommandBuffer(); // tail of an earlier split

    workers_->run(chunks, &MetalGraphExecutor::encodeChunkJob, &chunkJob_);
    for (u32 c = 0; c < chunks; ++c) {
        frame.buffers[frame.bufferCount - chunks - 1 + c]->endCommandBuffer();
    }

    tail->setLabel(groupLabels_[group]);
    for (u32 pos = splitPosition + 1; pos <= plan.lastPosition; ++pos) runPass(tail, pos, frame);
    // The whole render pass is one timed unit; its timestamp lands at the end
    // of the pass (measured), so it goes into the resuming tail encoder.
    if (timestamps_ && unitOfPosition_[plan.lastPosition] != kNone) {
        timestamps_->endUnit(tail, unitOfPosition_[plan.lastPosition]);
    }
    tail->endEncoding();
    // Measured: another encoder after the resumed pass in the same command
    // buffer makes the whole commit fail (MTL4CommandQueueErrorDomain 1), so
    // later encoders go into a fresh command buffer.
    tailCmd->endCommandBuffer();
    return beginExtraCommandBuffer(frame, frame.bufferCount - 1);
}

void MetalGraphExecutor::execute(MetalContext::Frame& frame) {
    if (!valid()) return;
    using SubmitQueue = MetalContext::SubmitQueue;
    const u64 base = MetalContext::timelineBase(frame.index);

    MTL4::CommandBuffer* cmd      = frame.commandBuffer; // graphics
    MTL4::CommandBuffer* asyncCmd = nullptr;
    if (timestamps_) timestamps_->commitStart(cmd, rg::Queue::Graphics);
    bool graphicsClosed = false, asyncClosed = true, firstAsync = true;
    if (segmented_) {
        // Graphics frame N may reuse what async frame N-1 still uses (an async
        // pass need not feed a later graphics pass): wait for its end.
        graphicsSub_ = openSubmission(frame, SubmitQueue::Graphics, context_.lastAsyncDoneValue(), 0);
        frame.submissions[graphicsSub_].firstBuffer = 0; // frame.commandBuffer
        frame.submissions[graphicsSub_].bufferCount = 1;
        asyncSub_ = ~0u;
    } else if (hasSplit_) {
        // F2.5 split pass: the frame is several graphics commits (see
        // encodeSplitGroup); the first one covers frame.commandBuffer.
        openSubmission(frame, SubmitQueue::Graphics, 0, 0);
        frame.submissions[0].firstBuffer = 0;
    }
    if (hasSplit_) {
        // The previous frame's split pass reads what this frame writes first.
        frame.submissions[segmented_ ? graphicsSub_ : 0].fenceEvent = splitFence_;
        frame.submissions[segmented_ ? graphicsSub_ : 0].fenceWait  = splitFenceValue_;
    }

    for (size_t e = 0; e < compiled_.encoders.size(); ++e) {
        const rg::EncoderPlan& plan = compiled_.encoders[e];
        const bool async = plan.queue == rg::Queue::AsyncCompute;
        if (segmented_) {
            // Waits and signals sit between commits (encoder boundaries).
            const u32 wait = waitBefore_[plan.firstPosition];
            if (async && (asyncClosed || wait)) {
                if (asyncCmd) asyncCmd->endCommandBuffer();
                // Frame N's async work may reuse what graphics frame N-1 used:
                // its first commit waits for frame N-1 (frame event value N).
                const u64 waitFrame = firstAsync ? frame.index : 0;
                firstAsync = false;
                asyncSub_ = openSubmission(frame, SubmitQueue::Async, wait ? base + wait : 0, waitFrame);
                asyncCmd = beginExtraCommandBuffer(frame, frame.bufferCount - 1, true);
                if (timestamps_) timestamps_->commitStart(asyncCmd, rg::Queue::AsyncCompute);
                asyncClosed = false;
            } else if (!async && (graphicsClosed || wait)) {
                if (cmd != frame.commandBuffer) cmd->endCommandBuffer();
                graphicsSub_ = openSubmission(frame, SubmitQueue::Graphics, wait ? base + wait : 0, 0);
                cmd = beginExtraCommandBuffer(frame, frame.bufferCount - 1, false);
                if (timestamps_) timestamps_->commitStart(cmd, rg::Queue::Graphics);
                graphicsClosed = false;
            }
        }
        MTL4::CommandBuffer* target = async ? asyncCmd : cmd;

        if (plan.type == rg::PassType::Raster) {
            const u32 groupIndex = plan.renderGroup;
            if (splitPosition_[groupIndex] != kNone) {
                cmd = encodeSplitGroup(frame, cmd, groupIndex, splitPosition_[groupIndex]);
            } else {
                const rg::RenderGroup& group = compiled_.renderGroups[groupIndex];
                // Imported attachments change every frame (drawable); the
                // encoder copies the descriptor, which then drops them again
                // so that it does not keep the drawable alive.
                setImportedAttachments(groupIndex, true);
                MTL4::RenderCommandEncoder* enc = target->renderCommandEncoder(passDescriptors_[groupIndex]);
                setImportedAttachments(groupIndex, false);
                enc->setLabel(groupLabels_[groupIndex]);
                // Barriers of every member were hoisted to the group start.
                encodeBarriers(enc, group.firstPosition);
                for (u32 pos = plan.firstPosition; pos <= plan.lastPosition; ++pos) runPass(enc, pos, frame);
                if (timestamps_ && unitOfPosition_[plan.lastPosition] != kNone) {
                    timestamps_->endUnit(enc, unitOfPosition_[plan.lastPosition]);
                }
                enc->endEncoding();
            }
        } else if (plan.type == rg::PassType::External) {
            if (!pipelines_ || !externalFence_ || plan.firstPosition != plan.lastPosition)
                throw std::runtime_error("External graph pass requires a compiler and an isolated encoder boundary");
            auto *anchor = pipelines_->compute(externalAnchor_);
            if (!anchor)
                throw std::runtime_error("External boundary pipeline is not ready");
            // Framework-owned encoders must observe prior graph producers.
            // A real dispatch keeps the boundary encoder alive even with timing off.
            auto *before = target->computeCommandEncoder();
            before->setLabel(NS::String::string("Before external pass", NS::UTF8StringEncoding));
            encodeBarriers(before, plan.firstPosition);
            before->setComputePipelineState(anchor);
            before->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
            before->updateFence(externalFence_, MTL::StageDispatch);
            before->endEncoding();
            const auto &node = graph_->passes()[compiled_.order[plan.firstPosition]];
            Context ctx(*this, nullptr, frame.index, 0, 1, target);
            if (node.execute)
                node.execute(ctx);
            if (ctx.dependency.resume) {
                if (async)
                    throw std::logic_error("Isolated external work requires the graphics queue");
                auto &producer = frame.submissions[graphicsSub_];
                producer.externalInputReady = static_cast<MTL::SharedEvent *>(ctx.dependency.inputReady);
                producer.externalSignalValue = ctx.dependency.value;
                producer.externalSubmitted = ctx.dependency.submitted;
                producer.externalSubmissionUser = ctx.dependency.submissionUser;
                if (target != frame.commandBuffer)
                    target->endCommandBuffer();
                if (frame.submissionCount >= MetalContext::MAX_FRAME_SUBMISSIONS ||
                    frame.bufferCount >= MetalContext::MAX_FRAME_COMMAND_BUFFERS)
                    throw std::runtime_error("External submission capacity exhausted");
                graphicsSub_ = openSubmission(frame, SubmitQueue::Graphics, 0, 0);
                auto &consumer = frame.submissions[graphicsSub_];
                consumer.externalOutputReady = static_cast<MTL::SharedEvent *>(ctx.dependency.outputReady);
                consumer.externalWaitValue = ctx.dependency.value;
                cmd = target = beginExtraCommandBuffer(frame, frame.bufferCount - 1, false);
                Context continuation(*this, nullptr, frame.index, 0, 1, target);
                ctx.dependency.resume(continuation, ctx.dependency.user);
            }
            auto *after = target->computeCommandEncoder();
            after->setLabel(encoderLabels_[e]);
            after->waitForFence(externalFence_, MTL::StageDispatch);
            after->setComputePipelineState(anchor);
            after->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
            if (timestamps_ && unitOfPosition_[plan.lastPosition] != kNone)
                timestamps_->endUnit(after, unitOfPosition_[plan.lastPosition]);
            after->endEncoding();
        } else {
            MTL4::ComputeCommandEncoder* enc = target->computeCommandEncoder();
            enc->setLabel(encoderLabels_[e]);
            for (u32 pos = plan.firstPosition; pos <= plan.lastPosition; ++pos) {
                encodeBarriers(enc, pos);
                // Non-raster passes with chunks run them in sequence.
                const u32 chunks = graph_->passes()[compiled_.order[pos]].parallelChunks;
                for (u32 c = 0; c < chunks; ++c) runPass(enc, pos, frame, c, chunks);
                if (timestamps_ && unitOfPosition_[pos] != kNone) timestamps_->endUnit(enc, unitOfPosition_[pos]);
            }
            enc->endEncoding();
        }

        if (segmented_) {
            if (const u32 signal = signalAfter_[plan.lastPosition]) {
                const u32 sub = async ? asyncSub_ : graphicsSub_;
                frame.submissions[sub].signalValue = base + signal;
                (async ? asyncClosed : graphicsClosed) = true;
            }
        }
    }
    // The frame's own command buffer is ended by MetalContext::submitFrame.
    if (cmd != frame.commandBuffer) cmd->endCommandBuffer();
    if (hasSplit_ && frame.submissionCount > 0) {
        u32 last = 0;
        for (u32 i = 0; i < frame.submissionCount; ++i) {
            if (frame.submissions[i].queue == SubmitQueue::Graphics) last = i;
        }
        MetalContext::Submission& sub = frame.submissions[last];
        if (!segmented_) sub.bufferCount = frame.bufferCount - sub.firstBuffer;
        sub.fenceEvent  = splitFence_;
        sub.fenceSignal = ++splitFenceValue_; // waited for by the next frame's first commit
    }
    if (asyncCmd) {
        asyncCmd->endCommandBuffer();
        // The last async commit tells when the frame's async work is done.
        MetalContext::Submission& last = frame.submissions[asyncSub_];
        if (last.signalValue == 0) last.signalValue = base + MetalContext::TIMELINE_STRIDE - 1;
        frame.asyncDoneValue = last.signalValue;
    }
}

} // namespace phosphor
