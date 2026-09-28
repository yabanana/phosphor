#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/gpu_memory.h"
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
    Context(const MetalGraphExecutor& executor, MTL4::CommandEncoder* encoder, u64 frame, u32 chunk, u32 chunks)
        : executor_(executor), encoder_(encoder), frame_(frame), chunk_(chunk), chunks_(chunks) {}

    void* encoder() const override { return encoder_; }
    void* texture(rg::TextureRef t) const override { return executor_.textures_[t.resource]; }
    void* buffer(rg::BufferRef b) const override { return executor_.buffers_[b.resource]; }
    u32   chunk() const override { return chunk_; }
    u32   chunkCount() const override { return chunks_; }
    u64   frameIndex() const override { return frame_; }

private:
    const MetalGraphExecutor& executor_;
    MTL4::CommandEncoder*     encoder_;
    u64 frame_;
    u32 chunk_;
    u32 chunks_;
};

// --- Executor ---------------------------------------------------------------------

MetalGraphExecutor::MetalGraphExecutor(MetalContext& context) : context_(context), heap_(context) {}

MetalGraphExecutor::~MetalGraphExecutor() {
    context_.waitIdle();
    releaseResources();
}

void MetalGraphExecutor::releaseResources() {
    for (size_t r = 0; r < owned_.size(); ++r) {
        if (!owned_[r]) continue;
        if (textures_[r]) {
            if (textures_[r]->storageMode() == MTL::StorageModeMemoryless) {
                context_.memory().release(textures_[r], MemoryCategory::RenderTargets);
            } else {
                heap_.release(textures_[r]);
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

    barrierIndex_.assign(compiled_.order.size(), kNone);
    for (u32 i = 0; i < compiled_.barriers.size(); ++i) barrierIndex_[compiled_.barriers[i].position] = i;
    for (const rg::EncoderPlan& e : compiled_.encoders) {
        if (e.queue != rg::Queue::Graphics) {
            LOG_ERROR("Render graph: the async compute queue is not supported by this executor yet");
            graph_ = nullptr;
            return false;
        }
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
            textures_[p.resource] = heap_.createTexture(d, p.offset, node.name.c_str());
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

void MetalGraphExecutor::runPass(MTL4::CommandEncoder* encoder, u32 position, const MetalContext::Frame& frame) const {
    const u32 pass = compiled_.order[position];
    const rg::PassNode& node = graph_->passes()[pass];
    encoder->pushDebugGroup(passLabels_[pass]);
    if (node.execute) {
        Context ctx(*this, encoder, frame.index, 0, 1);
        node.execute(ctx);
    }
    encoder->popDebugGroup();
}

void MetalGraphExecutor::execute(MetalContext::Frame& frame) {
    if (!valid()) return;
    const auto& resources = graph_->resources();
    for (size_t e = 0; e < compiled_.encoders.size(); ++e) {
        const rg::EncoderPlan& plan = compiled_.encoders[e];
        if (plan.type == rg::PassType::Raster) {
            const rg::RenderGroup& group = compiled_.renderGroups[plan.renderGroup];
            MTL4::RenderPassDescriptor* desc = passDescriptors_[plan.renderGroup];
            // Imported attachments change every frame (drawable).
            for (const rg::AttachmentPlan& a : group.attachments) {
                if (!resources[a.resource].imported) continue;
                if (a.depth) {
                    desc->depthAttachment()->setTexture(textures_[a.resource]);
                } else {
                    desc->colorAttachments()->object(a.slot)->setTexture(textures_[a.resource]);
                }
            }
            MTL4::RenderCommandEncoder* enc = frame.commandBuffer->renderCommandEncoder(desc);
            // The encoder copied the descriptor: do not keep the drawable alive.
            for (const rg::AttachmentPlan& a : group.attachments) {
                if (!resources[a.resource].imported) continue;
                if (a.depth) {
                    desc->depthAttachment()->setTexture(nullptr);
                } else {
                    desc->colorAttachments()->object(a.slot)->setTexture(nullptr);
                }
            }
            enc->setLabel(groupLabels_[plan.renderGroup]);
            // Barriers of every member were hoisted to the group start.
            encodeBarriers(enc, group.firstPosition);
            for (u32 pos = plan.firstPosition; pos <= plan.lastPosition; ++pos) runPass(enc, pos, frame);
            enc->endEncoding();
        } else {
            MTL4::ComputeCommandEncoder* enc = frame.commandBuffer->computeCommandEncoder();
            enc->setLabel(encoderLabels_[e]);
            for (u32 pos = plan.firstPosition; pos <= plan.lastPosition; ++pos) {
                encodeBarriers(enc, pos);
                runPass(enc, pos, frame);
            }
            enc->endEncoding();
        }
    }
}

} // namespace phosphor
