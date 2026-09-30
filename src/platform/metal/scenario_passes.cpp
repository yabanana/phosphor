#include "platform/metal/scenario_passes.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "platform/metal/pipeline_cache.h"
#include "renderer/gpu_types.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace phosphor {

namespace {

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

// Argument table slots; must match shaders/scenario.metal.
constexpr NS::UInteger kBindArgs       = 0;
constexpr NS::UInteger kBindColorIn    = 0;  // .. +SYNTH_COLOR_INPUTS
constexpr NS::UInteger kBindDepthIn    = 6;  // .. +SYNTH_DEPTH_INPUTS
constexpr NS::UInteger kBindStorageOut = 11; // .. +SYNTH_STORAGE_OUTPUTS
constexpr NS::UInteger kTextureSlots   = 13;
static_assert(kBindDepthIn == kBindColorIn + SYNTH_COLOR_INPUTS);
static_assert(kBindStorageOut == kBindDepthIn + SYNTH_DEPTH_INPUTS);
static_assert(kTextureSlots == kBindStorageOut + SYNTH_STORAGE_OUTPUTS);

MTL4::ArgumentTable* makeTable(MetalContext& context, NS::UInteger buffers, NS::UInteger textures, const char* label) {
    MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
    d->setMaxBufferBindCount(buffers);
    d->setMaxTextureBindCount(textures);
    d->setLabel(str(label));
    NS::Error* error = nullptr;
    MTL4::ArgumentTable* table = context.device()->newArgumentTable(d, &error);
    d->release();
    if (!table) throw std::runtime_error(std::string("Failed to create argument table ") + label);
    return table;
}

MTL::DepthStencilState* depthState(MetalContext& context, MTL::CompareFunction compare, bool write) {
    MTL::DepthStencilDescriptor* d = MTL::DepthStencilDescriptor::alloc()->init();
    d->setDepthCompareFunction(compare);
    d->setDepthWriteEnabled(write);
    MTL::DepthStencilState* state = context.device()->newDepthStencilState(d);
    d->release();
    if (!state) throw std::runtime_error("Failed to create a scenario depth-stencil state");
    return state;
}

u32 channelCount(rg::Format f) {
    switch (f) {
    case rg::Format::R8Unorm:
    case rg::Format::R16Float:
    case rg::Format::R32Float:
    case rg::Format::R32Uint:      return 1;
    case rg::Format::RG8Unorm:
    case rg::Format::RG16Float:
    case rg::Format::RG32Float:    return 2;
    case rg::Format::RG11B10Float: return 3;
    default:                       return 4;
    }
}

} // namespace

ScenarioPasses::ScenarioPasses(MetalContext& context, PipelineCache& pipelines, u32 index,
                               const rg::ScenarioParams& params)
    : context_(context), pipelines_(pipelines), index_(index), params_(params), defaults_(params) {
    if (index_ >= rg::scenarioCount()) {
        throw std::runtime_error("--graph-scenario: expected 0.." + std::to_string(rg::scenarioCount() - 1));
    }
    pipe::PipelineDesc desc;
    desc.kind         = pipe::PipelineKind::Compute;
    desc.label        = "synth_cs";
    desc.functions[0] = "synth_cs";
    compute_ = pipelines_.request(desc);

    rasterArgs_  = makeTable(context_, 1, kTextureSlots, "Scenario raster arguments");
    computeArgs_ = makeTable(context_, 1, kTextureSlots, "Scenario compute arguments");
    depthWrite_  = depthState(context_, MTL::CompareFunctionGreater, true); // reverse-Z
    depthTest_   = depthState(context_, MTL::CompareFunctionGreaterEqual, false);
    depthOff_    = depthState(context_, MTL::CompareFunctionAlways, false);

    dummyColor_ = newTexture({rg::Format::RGBA16Float, 1, 1}, MemoryCategory::Other, "Scenario dummy color");
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, 1, 1, false);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageRenderTarget);
    d->setStorageMode(MTL::StorageModePrivate);
    dummyDepth_ = context_.memory().newTexture(d, MemoryCategory::Other, "Scenario dummy depth");
    if (!dummyDepth_) throw std::runtime_error("Failed to create the scenario dummy depth texture");
    LOG_INFO("Graph scenario %u '%s': %ux%u, shadows %u, work %.2f%s%s", index_, rg::scenarioName(index_),
             params_.width, params_.height, params_.shadowSize, static_cast<double>(params_.work),
             params_.wideHdr ? ", wide HDR (RGBA32Float)" : "", params_.async && params_.async->empty() ? ", async off" : "");
}

ScenarioPasses::~ScenarioPasses() {
    context_.waitIdle();
    for (MTL::Texture* t : history_) {
        if (t) context_.memory().release(t, MemoryCategory::RenderTargets);
    }
    for (MTL::Texture* t : statics_) {
        if (t) context_.memory().release(t, MemoryCategory::RenderTargets);
    }
    context_.memory().release(dummyColor_, MemoryCategory::Other);
    context_.memory().release(dummyDepth_, MemoryCategory::Other);
    depthWrite_->release();
    depthTest_->release();
    depthOff_->release();
    rasterArgs_->release();
    computeArgs_->release();
}

MTL::Texture* ScenarioPasses::newTexture(const rg::TextureDesc& desc, MemoryCategory category, const char* label) {
    MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(toMetalFormat(desc.format), desc.width,
                                                                            desc.height, false);
    d->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    d->setStorageMode(MTL::StorageModePrivate);
    MTL::Texture* t = context_.memory().newTexture(d, category, label);
    if (!t) throw std::runtime_error(std::string("Failed to create ") + label);
    return t;
}

void ScenarioPasses::fill(MTL::Texture* texture, u32 seed) {
    // Loading time: one synth_cs dispatch without inputs writes the value
    // function of `seed` (deterministic initial contents).
    pipelines_.waitReady(compute_);
    GPUSynthArgs args{};
    args.outWidth     = static_cast<u32>(texture->width());
    args.outHeight    = static_cast<u32>(texture->height());
    args.seed         = seed;
    args.storageCount = 1;
    args.signalDepthInput = ~0u;
    const UploadRing::Slice slice = context_.stagingAllocate(sizeof(args));
    std::memcpy(slice.cpu, &args, sizeof(args));
    context_.enqueueUpload([this, texture, slice](MTL4::ComputeCommandEncoder* enc) {
        computeArgs_->setAddress(slice.gpu, kBindArgs);
        for (NS::UInteger i = 0; i < SYNTH_COLOR_INPUTS; ++i) computeArgs_->setTexture(dummyColor_->gpuResourceID(), kBindColorIn + i);
        for (NS::UInteger i = 0; i < SYNTH_DEPTH_INPUTS; ++i) computeArgs_->setTexture(dummyDepth_->gpuResourceID(), kBindDepthIn + i);
        computeArgs_->setTexture(texture->gpuResourceID(), kBindStorageOut);
        computeArgs_->setTexture(dummyColor_->gpuResourceID(), kBindStorageOut + 1);
        enc->setComputePipelineState(pipelines_.compute(compute_));
        enc->setArgumentTable(computeArgs_);
        enc->dispatchThreads(MTL::Size::Make(texture->width(), texture->height(), 1), MTL::Size::Make(8, 8, 1));
    });
    context_.flushUploads();
}

void ScenarioPasses::createPersistent() {
    if (history_[0] || !statics_.empty()) return; // created once, reused by every rebuild
    u32 seed = 0xC0FFEEu;
    for (const rg::ScenarioImport& imp : scenario_.imports) {
        if (imp.role == rg::ScenarioImport::Role::Static) {
            MTL::Texture* t = newTexture(imp.desc, MemoryCategory::RenderTargets, "Scenario persistent input");
            fill(t, seed++);
            statics_.push_back(t);
            continue;
        }
        const u32 i = imp.role == rg::ScenarioImport::Role::HistoryRead ? 0 : 1;
        history_[i] = newTexture(imp.desc, MemoryCategory::RenderTargets, i ? "Scenario history B" : "Scenario history A");
        fill(history_[i], seed++);
    }
}

void ScenarioPasses::setBuildChoices(const std::vector<std::string>& remat, const std::vector<std::string>& async) {
    params_       = defaults_;
    params_.remat = remat;
    params_.async = async;
}

void ScenarioPasses::build(rg::RenderGraph& graph, rg::TextureRef drawable) {
    std::string error;
    const auto factory = [this](u32 pass) -> rg::ExecuteFn {
        return [this, pass](rg::PassContext& ctx) { execute(pass, ctx); };
    };
    if (!rg::buildScenario(index_, params_, graph, drawable, scenario_, factory, &error)) {
        throw std::runtime_error("Graph scenario: " + error);
    }
    createPersistent();
}

void ScenarioPasses::onCompiled(const rg::RenderGraph& graph, const rg::CompiledGraph& compiled) {
    const auto& resources = graph.resources();
    passes_.assign(scenario_.synth.size(), PassState{});
    for (u32 p = 0; p < scenario_.synth.size(); ++p) {
        const rg::SynthPass& s = scenario_.synth[p];
        const u32 pos = compiled.position(p);
        if (s.kind == rg::SynthKind::None || pos == ~0u) continue;
        PassState& st = passes_[p];

        // Arguments (static for this compilation; copied into the frame ring).
        GPUSynthArgs args{};
        args.outWidth   = s.width;
        args.outHeight  = s.height;
        args.seed       = s.seed;
        u32 remats = 0;
        for (const rg::SynthInput& in : s.inputs) remats += in.remat ? 1 : 0;
        args.iterations = s.aluIterations + s.rematIterations * remats;
        args.zero       = 0;
        args.inputCount = static_cast<u32>(s.inputs.size());
        args.geometry   = s.kind == rg::SynthKind::Geometry ? 1u : 0u;
        args.geometrySeed     = s.geometrySeed;
        args.vertexIterations = 0;
        args.signalDepthInput = s.signalDepthInput;
        args.storageCount     = static_cast<u32>(s.storage.size());
        for (const rg::SynthColor& c : s.colors) args.depthOnlyMask |= c.depthOnly ? (1u << c.slot) : 0u;
        for (const rg::SynthColor& c : s.storage) args.depthOnlyMask |= c.depthOnly ? (1u << c.slot) : 0u;
        u32 colorBind = 0, depthBind = 0;
        if (s.inputs.size() > SYNTH_MAX_INPUTS) throw std::runtime_error("Graph scenario: too many inputs");
        for (u32 i = 0; i < s.inputs.size(); ++i) {
            const rg::SynthInput& in = s.inputs[i];
            GPUSynthInput& g = args.inputs[i];
            if (in.remat) {
                const rg::TextureDesc& depth = resources[s.inputs[in.rematDepth].texture.resource].texture;
                g.width      = depth.width;
                g.height     = depth.height;
                g.kind       = SYNTH_INPUT_REMAT;
                g.channels   = channelCount(in.rematFormat);
                g.rematDepth = in.rematDepth;
                g.rematSeed  = in.rematSeed;
                g.rematSlot  = in.rematSlot;
                continue;
            }
            const rg::TextureDesc& t = resources[in.texture.resource].texture;
            g.width  = t.width;
            g.height = t.height;
            g.kind   = in.depth ? SYNTH_INPUT_DEPTH : SYNTH_INPUT_COLOR;
            g.bind   = in.depth ? depthBind++ : colorBind++;
        }
        if (colorBind > SYNTH_COLOR_INPUTS || depthBind > SYNTH_DEPTH_INPUTS) {
            throw std::runtime_error("Graph scenario: too many texture inputs in a pass");
        }
        if (s.kind == rg::SynthKind::Geometry) {
            const double cells = std::max(1.0, s.triangles / 2.0);
            const u32 gridW = std::max(1u, static_cast<u32>(std::lround(std::sqrt(cells * s.width / std::max(s.height, 1u)))));
            const u32 gridH = std::max(1u, static_cast<u32>(cells / gridW));
            args.gridW     = gridW;
            args.gridH     = gridH;
            st.vertexCount = gridW * gridH * 6;
        }
        st.args.resize(sizeof(args));
        std::memcpy(st.args.data(), &args, sizeof(args));

        // Pipeline.
        if (s.kind == rg::SynthKind::Compute) {
            st.pipeline = compute_;
            continue;
        }
        const u32 g = compiled.groupOfPosition[pos];
        const rg::RenderGroup& group = compiled.renderGroups[g];
        pipe::PipelineDesc desc;
        desc.label        = graph.passes()[p].name;
        desc.functions[0] = s.kind == rg::SynthKind::Geometry ? "synth_geometry_vs" : "synth_fullscreen_vs";
        const bool colorWork = !s.colors.empty() || !s.fetched.empty();
        desc.functions[1] = colorWork ? "synth_fs" : "synth_depth_fs";
        u32 fetchMask = 0, outMask = 0;
        for (const u32 slot : s.fetched) fetchMask |= 1u << slot;
        for (const rg::SynthColor& c : s.colors) outMask |= 1u << c.slot;
        if (colorWork) {
            desc.constant(0, pipe::ConstantType::UInt, fetchMask);
            desc.constant(1, pipe::ConstantType::UInt, outMask);
        }
        for (const rg::AttachmentPlan& a : group.attachments) {
            if (a.depth) continue;
            desc.output(a.slot, resources[a.resource].texture.format);
            if (!(outMask & (1u << a.slot))) desc.color[a.slot].writeMask = 0;
        }
        st.pipeline = pipelines_.request(desc, /*loading*/ true);
    }
    // Graph compilation is loading time: every pass draws with its final pipeline.
    pipelines_.waitAllFinal();
}

void ScenarioPasses::bind(MetalGraphExecutor& executor, u64 frameIndex) {
    encoder_        = nullptr;
    bound_          = nullptr;
    computeEncoder_ = nullptr;
    boundCompute_   = nullptr;
    const u32 parity = static_cast<u32>(frameIndex & 1u);
    u32 s = 0;
    for (const rg::ScenarioImport& imp : scenario_.imports) {
        MTL::Texture* t = nullptr;
        switch (imp.role) {
        case rg::ScenarioImport::Role::Static:       t = statics_[s++]; break;
        case rg::ScenarioImport::Role::HistoryRead:  t = history_[parity]; break;
        case rg::ScenarioImport::Role::HistoryWrite: t = history_[parity ^ 1u]; break;
        }
        executor.bindTexture({imp.resource, 0}, t);
    }
}

void ScenarioPasses::execute(u32 pass, rg::PassContext& ctx) {
    const rg::SynthPass& s = scenario_.synth[pass];
    const PassState& st = passes_[pass];
    if (st.pipeline == pipe::INVALID_PIPELINE || st.args.empty()) return;

    const UploadRing::Slice slice = context_.frameUploads().allocate(st.args.size());
    std::memcpy(slice.cpu, st.args.data(), st.args.size());

    const bool compute = s.kind == rg::SynthKind::Compute;
    MTL4::ArgumentTable* table = compute ? computeArgs_ : rasterArgs_;
    table->setAddress(slice.gpu, kBindArgs);
    NS::UInteger colorBind = 0, depthBind = 0;
    for (const rg::SynthInput& in : s.inputs) {
        if (in.remat) continue;
        MTL::Texture* t = static_cast<MTL::Texture*>(ctx.texture(in.texture));
        if (in.depth) {
            table->setTexture(t->gpuResourceID(), kBindDepthIn + depthBind++);
        } else {
            table->setTexture(t->gpuResourceID(), kBindColorIn + colorBind++);
        }
    }
    for (; colorBind < SYNTH_COLOR_INPUTS; ++colorBind) table->setTexture(dummyColor_->gpuResourceID(), kBindColorIn + colorBind);
    for (; depthBind < SYNTH_DEPTH_INPUTS; ++depthBind) table->setTexture(dummyDepth_->gpuResourceID(), kBindDepthIn + depthBind);

    if (compute) {
        for (NS::UInteger i = 0; i < SYNTH_STORAGE_OUTPUTS; ++i) {
            MTL::Texture* t = i < s.storage.size() ? static_cast<MTL::Texture*>(ctx.texture(s.storage[i].texture))
                                                   : dummyColor_;
            table->setTexture(t->gpuResourceID(), kBindStorageOut + i);
        }
        auto* enc = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
        MTL::ComputePipelineState* pipeline = pipelines_.compute(st.pipeline);
        if (enc != computeEncoder_ || pipeline != boundCompute_) {
            enc->setComputePipelineState(pipeline);
            computeEncoder_ = enc;
            boundCompute_   = pipeline;
        }
        enc->setArgumentTable(table);
        enc->dispatchThreads(MTL::Size::Make(s.width, s.height, 1), MTL::Size::Make(8, 8, 1));
        return;
    }
    for (NS::UInteger i = 0; i < SYNTH_STORAGE_OUTPUTS; ++i) {
        table->setTexture(dummyColor_->gpuResourceID(), kBindStorageOut + i);
    }

    auto* enc = static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());
    if (enc != encoder_) {
        encoder_ = enc; // a new render encoder starts from Metal's default state
        bound_   = nullptr;
    }
    MTL::DepthStencilState* want = nullptr;
    if (s.kind == rg::SynthKind::Geometry) {
        want = s.depthWrite ? depthWrite_ : s.depthTest ? depthTest_ : depthOff_;
    } else if (bound_) {
        want = depthOff_; // a fused Geometry pass left its state
    }
    if (want && want != bound_) {
        enc->setDepthStencilState(want);
        bound_ = want;
    }
    enc->setRenderPipelineState(pipelines_.render(st.pipeline));
    enc->setArgumentTable(table, MTL::RenderStageVertex | MTL::RenderStageFragment);
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0),
                        NS::UInteger(s.kind == rg::SynthKind::Geometry ? st.vertexCount : 3));
}

} // namespace phosphor
