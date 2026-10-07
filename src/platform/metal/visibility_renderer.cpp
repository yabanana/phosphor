#include "platform/metal/visibility_renderer.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/upload_ring.h"
#include "pipeline/forward_variants.h"
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
constexpr std::array<const char *, 6> kNames = {
    "Linear HDR", "Signed normal + roughness", "Diffuse albedo", "Specular albedo", "Motion pixels", "Reactive mask"};
} // namespace
VisibilityRenderer::VisibilityRenderer(MetalContext &c, PipelineCache &p, SceneRenderer &s, MeshRenderer &m,
                                       bool binning, bool checks, bool tileResolve, bool adaptive, bool lighting)
    : context_(c), pipelines_(p), scene_(s), mesh_(m), binning_(binning && !tileResolve && !adaptive),
      tileResolve_(tileResolve), adaptive_(adaptive), checks_(checks), lighting_(lighting) {
    if (tileResolve && adaptive)
        throw std::invalid_argument("Tile and adaptive shading are separate experiments");
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
    d = pipe::forward::genericDesc(rg::Format::RGBA16Float);
    d.label = "Visibility indexed overflow fallback";
    d.functions[0] = "forward_surface_vs";
    d.functions[1] = lighting_ ? "forward_surface_lit_fs" : "forward_surface_fs";
    for (u32 i = 0; i < kFormats.size(); ++i)
        d.output(i, kFormats[i]);
    fallback_ = p.request(d);
    for (auto &table : tables_) {
        auto *desc = MTL4::ArgumentTableDescriptor::alloc()->init();
        desc->setMaxBufferBindCount(18);
        desc->setMaxTextureBindCount(10);
        NS::Error *error = nullptr;
        table = c.device()->newArgumentTable(desc, &error);
        desc->release();
        if (!table)
            throw std::runtime_error("Visibility argument table creation failed");
    }
}
VisibilityRenderer::~VisibilityRenderer() {
    context_.waitIdle();
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
    useBinning_ =
        binning_ && std::all_of(resolve_.begin(), resolve_.end(), [this](auto h) { return pipelines_.isFinal(h); });
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
               useBinning_ ? 1u : 0u,
               store.slotCapacity(),
               outputWidth,
               outputHeight,
               adaptive_ ? (shadingHistoryValid_[view_] ? 3u : 1u) : 0u,
               0};
    const u64 tiles = u64((outputWidth + 15) / 16) * ((outputHeight + 15) / 16);
    if (tiles > tileCapacity_) {
        tileCapacity_ = tiles;
        for (auto &f : frames_) {
            context_.memory().release(f.tiles, MemoryCategory::Other);
            context_.memory().release(f.args, MemoryCategory::Other);
            f.tiles =
                context_.memory().newBuffer(tiles * VISIBILITY_CLASSES * sizeof(u32), MTL::ResourceStorageModePrivate,
                                            MemoryCategory::Other, "Material tile bins");
            f.args = context_.memory().newBuffer(64, MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                                 "Material indirect arguments");
        }
    }
    if (checks_ && (readWidth_ != outputWidth || readHeight_ != outputHeight || readPoseCapacity_ < poseCapacity_)) {
        constexpr u32 bytes[] = {4, 4, 8, 8, 8, 8, 4, 1};
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
}
void VisibilityRenderer::prepareLighting(u32 flags, u32 sunIndex) {
    const GPUResolveLightingParams p{flags, sunIndex, params_.width, params_.height};
    auto u = context_.frameUploads().allocate(sizeof p); std::memcpy(u.cpu, &p, sizeof p); lightingAddress_ = u.gpu;
}
void VisibilityRenderer::setLightingTextures(rg::TextureRef sun, rg::TextureRef direct, rg::TextureRef gi) {
    sun_ = sun; direct_ = direct; indirect_ = gi;
}
void VisibilityRenderer::bind(rg::PassContext &ctx, MTL4::ArgumentTable *table) {
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
    bins_ = graph.importBuffer("Material bins and arguments", {tileCapacity_ * VISIBILITY_CLASSES * sizeof(u32) + 64},
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
                    outputs_[i] = b.createTexture(kNames[i], {kFormats[i], params_.outputWidth, params_.outputHeight});
                    outputs_[i] = b.write(outputs_[i], Usage::ShaderWrite, StageTile);
                }
                b.setProfileShaders("visibility_tile");
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder());
                bind(ctx, tables_[2]);
                enc->setRenderPipelineState(pipelines_.render(tile_));
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
                    outputs_[i] = b.createTexture(kNames[i], {kFormats[i], params_.outputWidth, params_.outputHeight});
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
                b.setProfileShaders(lighting_ ? "visibility_lit_resolve" : "visibility_resolve");
            },
            [this](PassContext &ctx) {
                auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
                bind(ctx, tables_[2]);
                enc->setArgumentTable(tables_[2]);
                if (useBinning_) {
                    ++binnedFrames_;
                    for (u32 i = 0; i < VISIBILITY_CLASSES; ++i) {
                        enc->setComputePipelineState(pipelines_.compute(resolve_[i]));
                        enc->dispatchThreadgroups(frames_[slot_].args->gpuAddress() + i * 12,
                                                  MTL::Size::Make(16, 16, 1));
                    }
                    scene_.countCommands(1 + 2 * VISIBILITY_CLASSES);
                } else {
                    ++genericFrames_;
                    enc->setComputePipelineState(pipelines_.compute(adaptive_ ? adaptivePipeline_ : generic_));
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
            scene_.encodeOverlay(static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder()), fallback_, true);
        });
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
    graph.addPass(
        "Visibility present", PassType::Raster,
        [&](PassBuilder &b) {
            for (u32 i = 0; i < 4; ++i)
                b.read(outputs_[i], Usage::ShaderRead, StageFragment);
            drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
            b.setProfileShaders("visibility_present_vs,visibility_present_fs");
        },
        [this](PassContext &ctx) {
            auto *enc = static_cast<MTL4::RenderCommandEncoder *>(ctx.encoder());
            bind(ctx, tables_[3]);
            enc->setRenderPipelineState(pipelines_.render(present_));
            enc->setArgumentTable(tables_[3], MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
            scene_.countCommands(5);
        });
    return drawable;
}
} // namespace phosphor
