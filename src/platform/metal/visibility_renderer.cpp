#include "platform/metal/visibility_renderer.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/upload_ring.h"
#include "pipeline/forward_variants.h"
#include "pipeline/material_classes.generated.h"
#include "core/log.h"
#include "renderer/scene_store.h"
#include "rendergraph/pass_context.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace phosphor {
namespace {
pipe::PipelineDesc compute(const char *name) {
    pipe::PipelineDesc d;
    d.kind = pipe::PipelineKind::Compute;
    d.label = name;
    d.functions = {name, "", ""};
    return d;
}
constexpr std::array<rg::Format, 6> kFormats = {rg::Format::RGBA16Float, rg::Format::RGBA16Float,
                                                rg::Format::RGBA16Float, rg::Format::RGBA16Float,
                                                rg::Format::RG16Float,   rg::Format::R8Unorm};
constexpr rg::Format resolveFormat(u32 index,bool lighting) {
    // F10-F14 residual contains physical sun/emission before external DI/GI.
    // Preserve its range before F13/F14 composition; F7/F8 keeps its HALF ABI.
    return index==0&&lighting?rg::Format::RGBA32Float:kFormats[index];
}
constexpr u16 kSssConstant = 23; // reserved SSS_OUTPUT (shaders/variants.def)
// F15.2: the generated class table (shaders/material_classes.def) and the shared layout agree.
static_assert(pipe::matclass::gen::kActiveClassBudget == VISIBILITY_FEATURE_CLASSES, "active-class budget of material_classes.def");
static_assert(pipe::matclass::gen::kBinClassFlag == VISIBILITY_BIN_CLASS && pipe::matclass::gen::kBinOverflowValue == VISIBILITY_BIN_OVERFLOW,
              "RESOLVE_BIN values");
static_assert(pipe::matclass::gen::kSssOutputConstantIndex == kSssConstant, "SSS_OUTPUT constant index");
constexpr std::array<const char *, 6> kNames = {
    "Linear HDR", "Signed normal + roughness", "Diffuse albedo", "Specular albedo", "Motion pixels", "Reactive mask"};
} // namespace
VisibilityRenderer::VisibilityRenderer(MetalContext &c, PipelineCache &p, SceneRenderer &s, MeshRenderer &m,
                                       bool binning, bool checks, bool tileResolve, bool adaptive, bool lighting)
    : context_(c), pipelines_(p), scene_(s), mesh_(m), binning_(binning && !tileResolve && !adaptive),
      tileResolve_(tileResolve), adaptive_(adaptive), checks_(checks), lighting_(lighting) {
    if (tileResolve && adaptive)
        throw std::invalid_argument("Tile and adaptive shading are separate experiments");
    if (lighting_) {
        auto* d = MTL::DepthStencilDescriptor::alloc()->init();d->setDepthCompareFunction(MTL::CompareFunctionGreaterEqual);
        d->setDepthWriteEnabled(false);lightingFallbackDepth_=c.device()->newDepthStencilState(d);d->release();
        if (!lightingFallbackDepth_) throw std::runtime_error("Lighting fallback depth state failed");
    }
    clear_ = p.request(compute("visibility_clear"));
    classify_ = p.request(compute("visibility_classify"));
    generic_ = p.request(compute(lighting_ ? "visibility_lit_resolve" : "visibility_resolve"));
    if (adaptive_ || p.harvesting()) {
        adaptivePipeline_ = p.request(compute("visibility_adaptive"));
        saveHistory_ = p.request(compute("visibility_history"));
    }
    if (tileResolve_ || p.harvesting()) {
        auto d = compute("visibility_tile");
        d.kind = pipe::PipelineKind::Tile;
        d.output(0, rg::Format::R32Uint);
        tile_ = p.request(d);
    }
    for (u32 i = 0; i < VISIBILITY_CLASSES; ++i) {
        auto d = compute(lighting_ ? "visibility_lit_resolve" : "visibility_resolve");
        d.constant(21, pipe::ConstantType::UInt, i);
        resolve_[i] = p.request(d);
    }
    pipe::PipelineDesc d;
    d.kind = pipe::PipelineKind::Render;
    d.label = "Visibility present";
    d.functions = {"visibility_present_vs", "visibility_present_fs", ""};
    d.output(0, rg::Format::BGRA8Srgb);
    present_ = p.request(d);
    fallback_ = p.request(fallbackDesc());
    // Ticket 17: the base of the fallback while the specialised one compiles, for a scene whose pass has the diffuse-irradiance attachment
    // (SSS_OUTPUT without the feature union: it writes the target as 0).  Requested at startup like every base pipeline (archive entry).
    fallbackSss_ = p.request(fallbackDesc(true));
    sssClear_ = p.request(compute("subsurface_clear"));
    sssDiffuse_ = p.request(compute("subsurface_diffuse"));
    for (auto &table : tables_) {
        auto *desc = MTL4::ArgumentTableDescriptor::alloc()->init();
        desc->setMaxBufferBindCount(18);
        desc->setMaxTextureBindCount(11);
        NS::Error *error = nullptr;
        table = c.device()->newArgumentTable(desc, &error);
        desc->release();
        if (!table)
            throw std::runtime_error("Visibility argument table creation failed");
    }
}
pipe::PipelineHandle VisibilityRenderer::binPipeline(u32 mask, bool overflow) {
    // Bin pipelines are keyed by what they compile, not by the scene: the class table of the generator names them.
    const u64 key = u64(mask) | (u64(overflow) << 16) | (u64(subsurface_) << 17) | (u64(textureTransform_) << 18) | (u64(dropLobes_) << 20);
    auto it = binHandles_.find(key);
    if (it == binHandles_.end()) {
        const auto describe = [&](const char* kernel) {
            auto d = compute(kernel);
            const u32 compiled = overflow ? MATERIAL_FEATURES_ALL : (mask & ~dropLobes_); // dropLobes_: negative control
            if (compiled != 0u) d.constant(pipe::matclass::gen::kMaterialFeaturesConstantIndex, pipe::ConstantType::UInt, compiled);
            if (subsurface_) d.constant(kSssConstant, pipe::ConstantType::UInt, 1u);
            if (textureTransform_) d = pipe::forward::withTextureTransform(d); // F15.1 (ticket 22)
            d.constant(pipe::matclass::gen::kResolveBinConstantIndex, pipe::ConstantType::UInt,
                       overflow ? VISIBILITY_BIN_OVERFLOW : (VISIBILITY_BIN_CLASS | mask));
            return d;
        };
        // A harvest (--harvest-pipelines) records both resolve kernels of every class (the plan names both), whichever lighting
        // path the harvested run uses.
        if (pipelines_.harvesting()) (void)pipelines_.request(describe(lighting_ ? "visibility_resolve" : "visibility_lit_resolve"));
        it = binHandles_.emplace(key, pipelines_.request(describe(lighting_ ? "visibility_lit_resolve" : "visibility_resolve"))).first;
    }
    return it->second;
}
pipe::PipelineDesc VisibilityRenderer::fallbackDesc(bool sss) const {
    pipe::PipelineDesc d = pipe::forward::genericDesc(resolveFormat(0, lighting_));
    d.label = "Visibility indexed overflow fallback";
    d.functions[0] = "forward_surface_vs";
    d.functions[1] = lighting_ ? "forward_surface_lit_fs" : "forward_surface_fs";
    for (u32 i = 0; i < kFormats.size(); ++i)
        d.output(i, resolveFormat(i, lighting_));
    if (sss) { // F15.1 (ticket 17): the diffuse-irradiance attachment of volumetric subsurface
        d.output(kFormats.size(), rg::Format::RGBA32Float);
        d.constant(kSssConstant, pipe::ConstantType::UInt, 1u);
    }
    return d;
}
VisibilityRenderer::~VisibilityRenderer() {
    context_.waitIdle();
    if (lightingFallbackDepth_) lightingFallbackDepth_->release();
    for (auto &f : frames_) {
        context_.memory().release(f.tiles, MemoryCategory::Other);
        context_.memory().release(f.args, MemoryCategory::Other);
    }
    for (auto *buffer : readbacks_)
        context_.memory().release(buffer, MemoryCategory::Other);
    context_.memory().release(currentReadback_, MemoryCategory::Other);
    context_.memory().release(previousReadback_, MemoryCategory::Other);
    for (auto *buffer : previousInstances_)
        context_.memory().release(buffer, MemoryCategory::Other);
    for (auto *buffer : shadingHistory_)
        context_.memory().release(buffer, MemoryCategory::Other);
    for (auto *table : tables_)
        if (table)
            table->release();
}
void VisibilityRenderer::prepareFrame(u32 slot, u32 width, u32 height, u32 outputWidth, u32 outputHeight,
                                      const SceneStore &store, std::array<u32, 3> defaultTextures, float exposure,
                                      u32 debugMode, const GPUTemporalParams &temporal,
                                      const FrameConstants &constants) {
    temporal_ = temporal;
    constants_ = constants;
    defaultTextures_ = defaultTextures;
    if (checks_)
        checkMaterials_.assign(store.materials().begin(), store.materials().end());
    slot_ = slot;
    view_ = temporal.viewIndex;
    if (view_ >= previousInstances_.size())
        throw std::invalid_argument("Temporal view index outside registry");
    if (adaptive_) {
        const u64 bytes = u64(outputWidth) * outputHeight * sizeof(GPUShadingHistory);
        if (shadingHistorySize_[view_] != bytes) {
            context_.memory().release(shadingHistory_[view_], MemoryCategory::Other);
            shadingHistory_[view_] = context_.memory().newBuffer(
                bytes, MTL::ResourceStorageModePrivate, MemoryCategory::Other, "Adaptive shading history per view");
            shadingHistorySize_[view_] = bytes;
            shadingHistoryValid_[view_] = false;
        }
    }
    const u64 poseBytes = std::max<u64>(sizeof(GPUInstance), store.instances().size_bytes());
    if (poseBytes > poseCapacity_) {
        for (auto *&buffer : previousInstances_) {
            context_.memory().release(buffer, MemoryCategory::Other);
            buffer = nullptr;
        }
        poseCapacity_ = poseBytes;
    }
    if (!previousInstances_[view_])
        previousInstances_[view_] = context_.memory().newBuffer(
            poseCapacity_, MTL::ResourceStorageModePrivate, MemoryCategory::Other, "Previous instance poses per view");
    const auto temporalUpload = context_.frameUploads().allocate(sizeof(temporal));
    std::memcpy(temporalUpload.cpu, &temporal, sizeof(temporal));
    temporalAddress_ = temporalUpload.gpu;
    scene_.setTemporalInputs(previousInstances_[view_]->gpuAddress(), temporalAddress_);
    mesh_.setTemporalInputs(temporalAddress_);
    // F15.1: with material features every shading pipeline is the one specialised by the scene's
    // feature union (the all-lobes reference while it compiles); union 0 keeps today's handles.
    // Ticket 17: a scene with volumetric subsurface also sets SSS_OUTPUT (constant 23) on the compute resolve and the overflow fallback
    // (its own cache slots: the same union without subsurface keeps its pipelines).
    {
        // --resolve-features all: the all-lobes pipeline resolves the whole scene (reference of the class pipelines).
        const u32 mask = forceAllLobes_ && store.featureUnion() != 0u ? MATERIAL_FEATURES_ALL : store.featureUnion();
        // F15.1 (ticket 22): per-slot texture transforms: the marker rides on the mask the shading pipelines are served with (the
        // lobe-only `mask` keeps deciding binning, subsurface and the tile path).
        textureTransform_ = store.textureTransformInUse();
        const u32 shadeMask = mask | (textureTransform_ ? FEATURE_UNION_TEXTURE_TRANSFORM : 0u);
        // F15.2: a scene with features bins per feature class (the legacy normal/emission classes only serve scenes without).
        const bool featureBinsWanted = binning_ && !forceAllLobes_ && mask != 0u;
        subsurface_ = subsurfaceAllowed_ && !tileResolve_ && mask != 0u && store.volumetricSubsurface();
        const u32 so = subsurface_ ? 50u : 0u;
        const auto sssConstant = [&](pipe::PipelineDesc d) {
            if (subsurface_) d.constant(kSssConstant, pipe::ConstantType::UInt, 1u);
            return d;
        };
        const char *kernel = lighting_ ? "visibility_lit_resolve" : "visibility_resolve";
        for (u32 i = 0; i < VISIBILITY_CLASSES; ++i) {
            if (featureBinsWanted) {
                servedResolve_[i] = resolve_[i]; // not used: the class bins below serve this scene
                continue;
            }
            servedResolve_[i] = features_.serve(pipelines_, 100 + so + i, resolve_[i], shadeMask, [&] {
                auto d = compute(kernel);
                d.constant(21, pipe::ConstantType::UInt, i);
                return sssConstant(d);
            });
        }
        servedGeneric_ = features_.serve(pipelines_, 110 + so, adaptive_ ? adaptivePipeline_ : generic_, shadeMask,
                                         [&] { return sssConstant(compute(adaptive_ ? "visibility_adaptive" : kernel)); });
        servedTile_ = tile_ == pipe::INVALID_PIPELINE ? tile_ : features_.serve(pipelines_, 111, tile_, mask, [&] {
            auto d = compute("visibility_tile");
            d.kind = pipe::PipelineKind::Tile;
            d.output(0, rg::Format::R32Uint);
            return d;
        });
        servedFallback_ = features_.serve(pipelines_, 112 + so, subsurface_ ? fallbackSss_ : fallback_, shadeMask,
                                          [&] { return fallbackDesc(subsurface_); });
    }
    featureBins_ = binning_ && !forceAllLobes_ && store.featureUnion() != 0u;
    if (featureBins_) {
        // F15.2 (ticket 21): one pipeline per class of the scene's class table (the archived class pipelines of the
        // generator) plus the all-lobes overflow bin when the scene has more classes than the budget; while one is not
        // final the frame uses the union-served generic dispatch (a transient after a load, never a fallback pipeline).
        classTable_ = buildMaterialClassTable(store.materials(), classBudget_);
        for (u32 i = 0; i < classTable_.count; ++i) servedBins_[i] = binPipeline(classTable_.masks[i], false);
        servedOverflow_ = classTable_.overflow() ? binPipeline(MATERIAL_FEATURES_ALL, true) : pipe::INVALID_PIPELINE;
        if (pipelines_.harvesting()) (void)binPipeline(MATERIAL_FEATURES_ALL, true); // the archive holds the overflow bin of every scene
        if (classTable_.overflowClasses != loggedOverflowClasses_) {
            loggedOverflowClasses_ = classTable_.overflowClasses;
            if (classTable_.overflow())
                LOG_WARN("Material classes: %u classes exceed the %u-class bin budget; their %u material(s) run the all-lobes pipeline",
                         classTable_.overflowClasses, VISIBILITY_FEATURE_CLASSES, classTable_.overflowMaterials);
        }
        useBinning_ = true;
        for (u32 i = 0; i < classTable_.count; ++i) useBinning_ = useBinning_ && pipelines_.isFinal(servedBins_[i]);
        if (classTable_.overflow()) useBinning_ = useBinning_ && pipelines_.isFinal(servedOverflow_);
    } else {
        useBinning_ = binning_ && std::all_of(servedResolve_.begin(), servedResolve_.end(),
                                              [this](auto h) { return pipelines_.isFinal(h); });
    }
    params_ = {width,
               height,
               (width + 15) / 16,
               (height + 15) / 16,
               static_cast<u32>(mesh_.capacity()),
               static_cast<u32>(store.materials().size()),
               defaultTextures[1],
               debugMode,
               exposure,
               temporal.mipBias,
               useBinning_ ? (featureBins_ ? 2u : 1u) : 0u,
               store.slotCapacity(),
               outputWidth,
               outputHeight,
               adaptive_ ? (shadingHistoryValid_[view_] ? 3u : 1u) : 0u,
               featureBins_ && useBinning_ ? classTable_.count : 0u,
               {background_[0], background_[1], background_[2]},
               0,
               {}};
    if (featureBins_ && useBinning_)
        for (u32 i = 0; i < classTable_.count; ++i) params_.classMasks[i] = classTable_.masks[i];
    const u64 tiles = u64((outputWidth + 15) / 16) * ((outputHeight + 15) / 16);
    if (tiles > tileCapacity_) {
        tileCapacity_ = tiles;
        for (auto &f : frames_) {
            context_.memory().release(f.tiles, MemoryCategory::Other);
            context_.memory().release(f.args, MemoryCategory::Other);
            f.tiles =
                context_.memory().newBuffer(tiles * VISIBILITY_MAX_BINS * sizeof(u32), MTL::ResourceStorageModePrivate,
                                            MemoryCategory::Other, "Material tile bins");
            f.args = context_.memory().newBuffer(256, MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                                 "Material indirect arguments");
        }
    }
    if (checks_ && (readWidth_ != outputWidth || readHeight_ != outputHeight || readPoseCapacity_ < poseCapacity_)) {
        // Keep raw resolve and actual Post input in independent readbacks.
        // Both can be RGBA32; guide formats remain unchanged.
        constexpr u32 bytes[] = {4, 4, 16, 8, 8, 8, 4, 1, 16};
        readWidth_ = outputWidth;
        readHeight_ = outputHeight;
        readPoseCapacity_ = poseCapacity_;
        for (u32 i = 0; i < readbacks_.size(); ++i) {
            context_.memory().release(readbacks_[i], MemoryCategory::Other);
            pitches_[i] = (u64(outputWidth) * bytes[i] + 255) & ~u64(255);
            readbacks_[i] = context_.memory().newBuffer(pitches_[i] * outputHeight, MTL::ResourceStorageModeShared,
                                                        MemoryCategory::Other, "Visibility check texture");
        }
        for (auto **b : {&currentReadback_, &previousReadback_}) {
            context_.memory().release(*b, MemoryCategory::Other);
            *b = context_.memory().newBuffer(poseCapacity_, MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                             "Visibility check poses");
        }
    }
    const auto upload = context_.frameUploads().allocate(sizeof(params_));
    std::memcpy(upload.cpu, &params_, sizeof(params_));
    paramsAddress_ = upload.gpu;
    if (subsurface_) {
        const GPUSubsurfaceParams sp{width, height, outputWidth, outputHeight, sssNear_, sssTanX_, sssTanY_, 0};
        const auto su = context_.frameUploads().allocate(sizeof(sp));
        std::memcpy(su.cpu, &sp, sizeof(sp));
        sssParamsAddress_ = su.gpu;
    }
}
void VisibilityRenderer::prepareLighting(u32 flags, u32 sunIndex, std::array<float, 3> environment, float sunAngularRadius) {
    const GPUResolveLightingParams p{flags, sunIndex, params_.width, params_.height, {environment[0], environment[1], environment[2]}, sunAngularRadius};
    auto u = context_.frameUploads().allocate(sizeof p); std::memcpy(u.cpu, &p, sizeof p); lightingAddress_ = u.gpu;
}
void VisibilityRenderer::setLightingTextures(rg::TextureRef sun, rg::TextureRef direct, rg::TextureRef gi) {
    sun_ = sun; direct_ = direct; indirect_ = gi;
}
void VisibilityRenderer::bind(rg::PassContext &ctx, MTL4::ArgumentTable *table, bool withSss) {
    table->setAddress(scene_.frameConstantsAddress(), 0);
    table->setAddress(scene_.vertexBuffer()->gpuAddress(), 1);
    table->setAddress(scene_.buffers().instances()->gpuAddress(), 2);
    table->setAddress(scene_.buffers().materials()->gpuAddress(), 3);
    table->setAddress(scene_.lightsAddress(), 4);
    table->setAddress(scene_.textureTableAddress(), 5);
    table->setAddress(mesh_.meshletBuffer()->gpuAddress(), 6);
    table->setAddress(mesh_.meshletVertexBuffer()->gpuAddress(), 7);
    table->setAddress(mesh_.meshletTriangleBuffer()->gpuAddress(), 8);
    table->setAddress(mesh_.frame(slot_).candidates->gpuAddress(), 9);
    table->setAddress(mesh_.frame(slot_).bList->gpuAddress(), 10);
    table->setAddress(previousInstances_[view_]->gpuAddress(), 14);
    table->setAddress(temporalAddress_, 15);
    table->setAddress(paramsAddress_, 11);
    if (lighting_) {
        table->setAddress(lightingAddress_, 17);
        table->setTexture(static_cast<MTL::Texture*>(ctx.texture(sun_))->gpuResourceID(), 7);
        table->setTexture(static_cast<MTL::Texture*>(ctx.texture(direct_))->gpuResourceID(), 8);
        table->setTexture(static_cast<MTL::Texture*>(ctx.texture(indirect_))->gpuResourceID(), 9);
    }
    if (adaptive_)
        table->setAddress(shadingHistory_[view_]->gpuAddress(), 16);
    table->setAddress(frames_[slot_].tiles->gpuAddress(), 12);
    table->setAddress(frames_[slot_].args->gpuAddress(), 13);
    table->setTexture(static_cast<MTL::Texture *>(ctx.texture(visibility_))->gpuResourceID(), 0);
    for (u32 i = 0; i < outputs_.size(); ++i)
        table->setTexture(static_cast<MTL::Texture *>(ctx.texture(outputs_[i]))->gpuResourceID(), i + 1);
    if (withSss) table->setTexture(static_cast<MTL::Texture *>(ctx.texture(sss_))->gpuResourceID(), 10);
}
rg::BufferRef VisibilityRenderer::importPoseHistory(rg::RenderGraph& graph) {
    // The caller imports before receiver guides; addResolve reuses this exact
    // ref, rather than creating a second logical resource for the same memory.
    if (!poses_.valid() || poses_.resource >= graph.resources().size() ||
        graph.resources()[poses_.resource].name != "Previous instance poses")
        poses_ = graph.importBuffer("Previous instance poses", {poseCapacity_}, rg::ImportContentsDefined | rg::ImportOutput);
    return poses_;
}
rg::TextureRef VisibilityRenderer::addResolve(rg::RenderGraph &graph, rg::TextureRef visibility, rg::TextureRef depth) {
    using namespace rg;
    composedColor_ = {};
    visibility_ = visibility;
    depth_ = depth;
    importPoseHistory(graph);
    if (temporal_.debugFlags & 8u)
        graph.addPass(
            "Negative control: current poses as history", PassType::Blit,
            [&](PassBuilder &b) {
                b.read(scene_.dataRef(), Usage::CopySrc, StageBlit);
                poses_ = b.write(poses_, Usage::CopyDst, StageBlit);
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                enc->copyFromBuffer(scene_.buffers().instances(), 0, previousInstances_[view_], 0,
                                    std::min<u64>(poseCapacity_, scene_.buffers().instances()->length()));
            });
    if (adaptive_)
        shadingHistoryRef_ = graph.importBuffer("Adaptive shading history per view", {shadingHistorySize_[view_]},
                                                ImportContentsDefined | ImportOutput);
    bins_ = graph.importBuffer("Material bins and arguments", {tileCapacity_ * VISIBILITY_MAX_BINS * sizeof(u32) + 256},
                               ImportPerFrame);
    if (tileResolve_) {
        graph.addPass(
            "On-tile material resolve", PassType::Raster,
            [&](PassBuilder &b) {
                b.readColor(visibility_, 0, StageTile);
                b.readDepth(depth_);
                b.setTileSize(VISIBILITY_TILE, VISIBILITY_TILE);
                b.read(scene_.dataRef(), Usage::ShaderRead, StageTile);
                b.read(mesh_.frameListsRef(), Usage::ShaderRead, StageTile);
                b.read(poses_, Usage::ShaderRead, StageTile);
                for (u32 i = 0; i < outputs_.size(); ++i) {
                    outputs_[i] = b.createTexture(kNames[i], {resolveFormat(i,lighting_), params_.outputWidth, params_.outputHeight});
                    outputs_[i] = b.write(outputs_[i], Usage::ShaderWrite, StageTile);
                }
                b.setProfileShaders("visibility_tile");
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder());
                bind(ctx, tables_[2]);
                enc->setRenderPipelineState(pipelines_.render(servedTile_));
                enc->setArgumentTable(tables_[2], MTL::RenderStageTile);
                enc->dispatchThreadsPerTile(MTL::Size::Make(VISIBILITY_TILE, VISIBILITY_TILE, 1));
                scene_.countCommands(3);
            });
    } else {
        graph.addPass(
            "Visibility clear", PassType::Compute,
            [&](PassBuilder &b) {
                bins_ = b.write(bins_, Usage::ShaderWrite, StageDispatch);
                for (u32 i = 0; i < outputs_.size(); ++i) {
                    outputs_[i] = b.createTexture(kNames[i], {resolveFormat(i,lighting_), params_.outputWidth, params_.outputHeight});
                    outputs_[i] = b.write(outputs_[i], Usage::ShaderWrite, StageDispatch);
                }
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                bind(ctx, tables_[0]);
                enc->setComputePipelineState(pipelines_.compute(clear_));
                enc->setArgumentTable(tables_[0]);
                enc->dispatchThreadgroups(
                    MTL::Size::Make((params_.outputWidth + 15) / 16, (params_.outputHeight + 15) / 16, 1),
                    MTL::Size::Make(16, 16, 1));
                scene_.countCommands(3);
            });
        if (binning_)
            graph.addPass(
                "Material classification", PassType::Compute,
                [&](PassBuilder &b) {
                    b.read(visibility_, Usage::ShaderRead, StageDispatch);
                    b.read(scene_.dataRef(), Usage::ShaderRead, StageDispatch);
                    b.read(mesh_.frameListsRef(), Usage::ShaderRead, StageDispatch);
                    b.read(bins_, Usage::ShaderRead, StageDispatch);
                    bins_ = b.write(bins_, Usage::ShaderWrite, StageDispatch);
                },
                [this](PassContext &ctx) {
                    auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                    bind(ctx, tables_[1]);
                    enc->setComputePipelineState(pipelines_.compute(classify_));
                    enc->setArgumentTable(tables_[1]);
                    enc->dispatchThreadgroups(MTL::Size::Make(params_.tilesX, params_.tilesY, 1),
                                              MTL::Size::Make(16, 16, 1));
                    scene_.countCommands(3);
                });
        if (subsurface_) {
            // F15.1 (ticket 17): the diffuse-irradiance target, cleared every frame (the resolve writes only volumetric pixels).
            graph.addPass(
                "Subsurface clear", PassType::Compute,
                [&](PassBuilder &b) {
                    sss_ = b.createTexture("Subsurface diffuse irradiance",
                                           {Format::RGBA32Float, params_.outputWidth, params_.outputHeight});
                    sss_ = b.write(sss_, Usage::ShaderWrite, StageDispatch);
                },
                [this](PassContext &ctx) {
                    auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                    auto *table = tables_[5];
                    table->setAddress(sssParamsAddress_, 0);
                    table->setTexture(static_cast<MTL::Texture *>(ctx.texture(sss_))->gpuResourceID(), 0);
                    enc->setComputePipelineState(pipelines_.compute(sssClear_));
                    enc->setArgumentTable(table);
                    enc->dispatchThreadgroups(
                        MTL::Size::Make((params_.outputWidth + 15) / 16, (params_.outputHeight + 15) / 16, 1),
                        MTL::Size::Make(16, 16, 1));
                    scene_.countCommands(3);
                });
        }
        graph.addPass(
            "Material resolve", PassType::Compute,
            [&](PassBuilder &b) {
                b.read(visibility_, Usage::ShaderRead, StageDispatch);
                b.read(depth_, Usage::ShaderRead, StageDispatch);
                b.read(scene_.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(mesh_.frameListsRef(), Usage::ShaderRead, StageDispatch);
                b.read(poses_, Usage::ShaderRead, StageDispatch);
                if (adaptive_)
                    b.read(shadingHistoryRef_, Usage::ShaderRead, StageDispatch);
                b.read(bins_, Usage::IndirectArgs, StageDispatch);
                for (auto &output : outputs_) {
                    // Preserve cleared background and pixels owned by other classes.
                    b.read(output, Usage::ShaderRead, StageDispatch);
                    output = b.write(output, Usage::ShaderWrite, StageDispatch);
                }
                if (lighting_) for (auto ref : {sun_, direct_, indirect_}) b.read(ref, Usage::ShaderRead, StageDispatch);
                if (subsurface_) {
                    b.read(sss_, Usage::ShaderRead, StageDispatch);
                    sss_ = b.write(sss_, Usage::ShaderWrite, StageDispatch);
                }
                b.setProfileShaders(lighting_ ? "visibility_lit_resolve" : "visibility_resolve");
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                bind(ctx, tables_[2], subsurface_);
                enc->setArgumentTable(tables_[2]);
                if (useBinning_ && featureBins_) {
                    // F15.2: one dispatch per class bin (indirect arguments of the bin) plus the overflow bin when used.
                    ++binnedFrames_;
                    ++featureBinnedFrames_;
                    for (u32 i = 0; i < classTable_.count; ++i) {
                        enc->setComputePipelineState(pipelines_.compute(servedBins_[i]));
                        enc->dispatchThreadgroups(frames_[slot_].args->gpuAddress() + i * 12, MTL::Size::Make(16, 16, 1));
                    }
                    u32 dispatches = classTable_.count;
                    if (classTable_.overflow()) {
                        enc->setComputePipelineState(pipelines_.compute(servedOverflow_));
                        enc->dispatchThreadgroups(frames_[slot_].args->gpuAddress() + VISIBILITY_FEATURE_CLASSES * 12, MTL::Size::Make(16, 16, 1));
                        ++dispatches;
                    }
                    scene_.countCommands(1 + 2 * dispatches);
                } else if (useBinning_) {
                    ++binnedFrames_;
                    for (u32 i = 0; i < VISIBILITY_CLASSES; ++i) {
                        enc->setComputePipelineState(pipelines_.compute(servedResolve_[i]));
                        enc->dispatchThreadgroups(frames_[slot_].args->gpuAddress() + i * 12,
                                                  MTL::Size::Make(16, 16, 1));
                    }
                    scene_.countCommands(1 + 2 * VISIBILITY_CLASSES);
                } else {
                    ++genericFrames_;
                    enc->setComputePipelineState(pipelines_.compute(servedGeneric_));
                    enc->dispatchThreadgroups(MTL::Size::Make(params_.tilesX, params_.tilesY, 1),
                                              MTL::Size::Make(adaptive_ ? 8 : 16, adaptive_ ? 8 : 16, 1));
                    scene_.countCommands(3);
                }
            });
    }
    graph.addPass(
        "Visibility overflow fallback", PassType::Raster,
        [&](PassBuilder &b) {
            for (u32 i = 0; i < outputs_.size(); ++i)
                outputs_[i] = b.writeColor(outputs_[i], i, LoadIntent::Preserve);
            if (subsurface_) sss_ = b.writeColor(sss_, u32(outputs_.size()), LoadIntent::Preserve);
            depth_ = b.writeDepth(depth_, LoadIntent::Preserve);
            scene_.declareDrawReads(b);
            b.read(poses_, Usage::ShaderRead, StageVertex);
            if (lighting_) for (auto ref : {sun_, direct_, indirect_}) b.read(ref, Usage::ShaderRead, StageFragment);
            b.setProfileShaders(lighting_ ? "forward_surface_vs,forward_surface_lit_fs" : "forward_vs,forward_surface_fs");
        },
        [this](PassContext &ctx) {
            if (lighting_) scene_.setLightingInputs(lightingAddress_, static_cast<MTL::Texture*>(ctx.texture(sun_)),
                                                     static_cast<MTL::Texture*>(ctx.texture(direct_)),
                                                     static_cast<MTL::Texture*>(ctx.texture(indirect_)));
            scene_.encodeOverlay(static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder()), servedFallback_, true, lightingFallbackDepth_);
        });
    if (subsurface_) {
        // F15.1 (ticket 17): screen-space Burley diffusion of the volumetric pixels' diffuse irradiance, added to the resolved colour.
        // It is a new colour version: every later consumer reads color() (reflections, atmosphere, post, capture).
        const TextureRef resolved = outputs_[0];
        const TextureRef sss = sss_;
        const TextureRef depth = depth_;
        const TextureRef normal = outputs_[1];
        graph.addPass(
            "Subsurface diffusion", PassType::Compute,
            [&](PassBuilder &b) {
                b.read(resolved, Usage::ShaderRead, StageDispatch);
                b.read(sss, Usage::ShaderRead, StageDispatch);
                b.read(depth, Usage::ShaderRead, StageDispatch);
                b.read(normal, Usage::ShaderRead, StageDispatch);
                b.read(scene_.dataRef(), Usage::ShaderRead, StageDispatch);
                TextureRef out = b.createTexture("Subsurface linear HDR",
                                                 {resolveFormat(0, lighting_), params_.outputWidth, params_.outputHeight});
                sssResult_ = b.write(out, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("subsurface_diffuse");
            },
            [this, resolved, sss, depth, normal](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                auto *table = tables_[6];
                table->setAddress(sssParamsAddress_, 0);
                table->setAddress(scene_.frameConstantsAddress(), 1);
                table->setAddress(scene_.buffers().materials()->gpuAddress(), 2);
                table->setTexture(static_cast<MTL::Texture *>(ctx.texture(resolved))->gpuResourceID(), 0);
                table->setTexture(static_cast<MTL::Texture *>(ctx.texture(sss))->gpuResourceID(), 1);
                table->setTexture(static_cast<MTL::Texture *>(ctx.texture(depth))->gpuResourceID(), 2);
                table->setTexture(static_cast<MTL::Texture *>(ctx.texture(normal))->gpuResourceID(), 3);
                table->setTexture(static_cast<MTL::Texture *>(ctx.texture(sssResult_))->gpuResourceID(), 4);
                enc->setComputePipelineState(pipelines_.compute(sssDiffuse_));
                enc->setArgumentTable(table);
                enc->dispatchThreadgroups(
                    MTL::Size::Make((params_.outputWidth + 15) / 16, (params_.outputHeight + 15) / 16, 1),
                    MTL::Size::Make(16, 16, 1));
                scene_.countCommands(3);
            });
        composedColor_ = sssResult_;
    }
    return outputs_[0];
}
void VisibilityRenderer::addPoseSnapshot(rg::RenderGraph &graph) {
    using namespace rg;
    if (adaptive_)
        graph.addPass(
            "Adaptive shading history", PassType::Compute,
            [&](PassBuilder &b) {
                for (auto t : {visibility_, outputs_[0], outputs_[1]})
                    b.read(t, Usage::ShaderRead, StageDispatch);
                b.read(scene_.dataRef(), Usage::ShaderRead, StageDispatch);
                b.read(mesh_.frameListsRef(), Usage::ShaderRead, StageDispatch);
                shadingHistoryRef_ = b.write(shadingHistoryRef_, Usage::ShaderWrite, StageDispatch);
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                bind(ctx, tables_[4]);
                enc->setComputePipelineState(pipelines_.compute(saveHistory_));
                enc->setArgumentTable(tables_[4]);
                enc->dispatchThreadgroups(
                    MTL::Size::Make((params_.outputWidth + 15) / 16, (params_.outputHeight + 15) / 16, 1),
                    MTL::Size::Make(16, 16, 1));
                shadingHistoryValid_[view_] = true;
                scene_.countCommands(3);
            });
    graph.addPass(
        "Previous pose snapshot", PassType::Blit,
        [&](PassBuilder &b) {
            b.read(scene_.dataRef(), Usage::CopySrc, StageBlit);
            poses_ = b.write(poses_, Usage::CopyDst, StageBlit);
        },
        [this](PassContext &ctx) {
            auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
            enc->copyFromBuffer(scene_.buffers().instances(), 0, previousInstances_[view_], 0,
                                std::min<u64>(poseCapacity_, scene_.buffers().instances()->length()));
            scene_.countCommands(1);
        });
}
std::array<u32, 2> VisibilityRenderer::adaptiveStats() const {
    if (!adaptive_)
        return {};
    const auto *args = static_cast<const u32 *>(frames_[slot_].args->contents());
    return {args[12], args[13]};
}
rg::TextureRef VisibilityRenderer::addPresent(rg::RenderGraph &graph, rg::TextureRef drawable) {
    using namespace rg;
    const TextureRef presentedColor = color();
    graph.addPass(
        "Visibility present", PassType::Raster,
        [&](PassBuilder &b) {
            b.read(presentedColor, Usage::ShaderRead, StageFragment);
            for (u32 i = 1; i < 4; ++i)
                b.read(outputs_[i], Usage::ShaderRead, StageFragment);
            drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
            b.setProfileShaders("visibility_present_vs,visibility_present_fs");
        },
        [this,presentedColor](PassContext &ctx) {
            auto *enc = static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder());
            bind(ctx, tables_[3]);
            tables_[3]->setTexture(static_cast<MTL::Texture *>(ctx.texture(presentedColor))->gpuResourceID(), 1);
            enc->setRenderPipelineState(pipelines_.render(present_));
            enc->setArgumentTable(tables_[3], MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
            scene_.countCommands(5);
        });
    return drawable;
}
} // namespace phosphor
