#include "platform/metal/debug_overlays.h"
#include "diagnostics/overlay_math.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/scene_renderer.h"
#include "rendergraph/pass_context.h"

#include <stdexcept>
#include <string>

namespace phosphor {

namespace {

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

// Argument table slots; must match shaders/overlay.metal.
constexpr NS::UInteger kBindConstants = 0;

MTL4::ArgumentTable* makeTable(MetalContext& context, NS::UInteger buffers, NS::UInteger textures,
                               const char* label) {
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

MTL::ResourceID textureId(rg::PassContext& ctx, rg::TextureRef ref) {
    return static_cast<MTL::Texture*>(ctx.texture(ref))->gpuResourceID();
}

} // namespace

struct DebugOverlays::Impl {
    MetalContext&  context;
    PipelineCache& pipelines;

    pipe::PipelineHandle overdraw  = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle lights    = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle tileCost  = pipe::INVALID_PIPELINE;
    pipe::PipelineHandle composite = pipe::INVALID_PIPELINE;
    MTL4::ArgumentTable* compositeArgs = nullptr;
    MTL4::ArgumentTable* computeArgs   = nullptr;

    // Set while the graph is built, read when its passes execute.
    rg::TextureRef overdrawRef, lightsRef, tilesRef;
    // The frame's OverlayConstants (prepareFrame).
    MTL::GPUAddress constants = 0;

    Impl(MetalContext& c, PipelineCache& p) : context(c), pipelines(p) {}
};

DebugOverlays::DebugOverlays(MetalContext& context, PipelineCache& pipelines) : impl_(new Impl(context, pipelines)) {
    Impl& d = *impl_;
    // Requested before the first frame (the engine waits for every pipeline),
    // so switching the overlay at run time never compiles on the render thread.
    pipe::PipelineDesc desc;
    desc.label        = "Overlay overdraw";
    desc.functions    = {"overlay_vs", "overdraw_fs"};
    desc.output(0, rg::Format::R16Float);
    desc.indirectCommandBuffers = true; // redraws the scene's draws (F5.3 ICB)
    d.overdraw = pipelines.request(desc);

    desc = {};
    desc.label     = "Overlay light count";
    desc.functions = {"overlay_vs", "lightcount_fs"};
    desc.output(0, rg::Format::R16Float);
    desc.indirectCommandBuffers = true;
    d.lights = pipelines.request(desc);

    desc = {};
    desc.label     = "Overlay composite";
    desc.functions = {"overlay_composite_vs", "overlay_composite_fs"};
    desc.output(0, rg::Format::BGRA8Srgb, pipe::ColorOutput::Blend::AlphaOver);
    d.composite = pipelines.request(desc);

    desc = {};
    desc.kind         = pipe::PipelineKind::Compute;
    desc.label        = "overlay_tilecost";
    desc.functions[0] = "overlay_tilecost";
    d.tileCost = pipelines.request(desc);

    d.compositeArgs = makeTable(context, 1, 1, "Overlay composite arguments");
    d.computeArgs   = makeTable(context, 1, 3, "Overlay tile cost arguments");
}

DebugOverlays::~DebugOverlays() {
    impl_->context.waitIdle();
    impl_->computeArgs->release();
    impl_->compositeArgs->release();
    delete impl_;
}

DebugOverlays::Legend DebugOverlays::legend(OverlayMode mode) {
    switch (mode) {
    case OverlayMode::Overdraw:
        return {"fragments per pixel", overlay::OVERDRAW_MAX,
                "every rasterised fragment counts (no depth test, alpha test ignored): an upper bound of what the "
                "TBDR hidden-surface removal of the forward pass actually shades",
                false};
    case OverlayMode::LightCount:
        return {"lights reaching the visible surface", overlay::LIGHTS_MAX,
                "lights with attenuation > 0 at the nearest surface (range window, spot cone; directional = 1); "
                "logarithmic scale, 0 lights is the coldest colour",
                true};
    case OverlayMode::TileCost:
        return {"shading cost per pixel", overlay::TILECOST_MAX,
                "per 32x32 tile, mean of overdraw x (1 + lights): a model of fragment work, not a measured time "
                "(Apple GPUs expose no per-tile time); logarithmic scale",
                true};
    case OverlayMode::Timings:
        return {"pass timings", 0.0f, "drawn by the UI from the GPU timestamps; no GPU pass", false};
    case OverlayMode::None: break;
    }
    return {};
}

void DebugOverlays::prepareFrame(OverlayMode mode, u32, u32 width, u32 height) {
    if (mode == OverlayMode::None || mode == OverlayMode::Timings) return;
    OverlayConstants c{};
    c.kind   = static_cast<u32>(mode);
    c.width  = width;
    c.height = height;
    c.tilesX = overlay::tileCount(width);
    c.tilesY = overlay::tileCount(height);
    c.alpha  = overlay::OVERLAY_ALPHA;
    const UploadRing::Slice slice = impl_->context.frameUploads().allocate(sizeof(c));
    *reinterpret_cast<OverlayConstants*>(slice.cpu) = c;
    impl_->constants = slice.gpu;
}

rg::TextureRef DebugOverlays::addToGraph(rg::RenderGraph& graph, rg::TextureRef color, u32 width, u32 height,
                                         OverlayMode mode, const SceneRenderer& scene) {
    using namespace rg;
    if (mode == OverlayMode::None || mode == OverlayMode::Timings) return color;
    Impl& d = *impl_;
    const SceneRenderer* renderer = &scene;

    const bool wantOverdraw = mode == OverlayMode::Overdraw || mode == OverlayMode::TileCost;
    const bool wantLights   = mode == OverlayMode::LightCount || mode == OverlayMode::TileCost;
    const u32  tilesX       = overlay::tileCount(width);
    const u32  tilesY       = overlay::tileCount(height);

    if (wantOverdraw) {
        graph.addPass(
            "Overlay overdraw", PassType::Raster,
            [&](PassBuilder& b) {
                d.overdrawRef = b.createTexture("Overdraw count", {Format::R16Float, width, height});
                d.overdrawRef = b.writeColor(d.overdrawRef, 0, LoadIntent::Clear); // count 0
                renderer->declareDrawReads(b);
                b.setHints(HintGeometryHeavy);
                b.setProfileShaders("overlay_vs,overdraw_fs");
            },
            [this, renderer](PassContext& ctx) {
                renderer->encodeOverlay(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), impl_->overdraw,
                                        /*depthTest*/ false);
            });
    }

    if (wantLights) {
        graph.addPass(
            "Overlay light count", PassType::Raster,
            [&](PassBuilder& b) {
                ClearValue clear;
                clear.color[0] = -1.0f; // no geometry
                clear.depth    = 0.0f;  // reverse-Z: far = 0
                d.lightsRef = b.createTexture("Light count", {Format::R16Float, width, height});
                d.lightsRef = b.writeColor(d.lightsRef, 0, LoadIntent::Clear, clear);
                const TextureRef depth = b.createTexture("Overlay depth", {Format::Depth32Float, width, height});
                b.writeDepth(depth, LoadIntent::Clear, clear); // memoryless
                renderer->declareDrawReads(b);
                b.setHints(HintFragmentHeavy);
                b.setProfileShaders("overlay_vs,lightcount_fs");
            },
            [this, renderer](PassContext& ctx) {
                renderer->encodeOverlay(static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder()), impl_->lights,
                                        /*depthTest*/ true);
            });
    }

    TextureRef values = wantOverdraw && !wantLights ? d.overdrawRef : d.lightsRef;
    if (mode == OverlayMode::TileCost) {
        graph.addPass(
            "Overlay tile cost", PassType::Compute,
            [&](PassBuilder& b) {
                b.read(d.overdrawRef, Usage::ShaderRead, StageDispatch);
                b.read(d.lightsRef, Usage::ShaderRead, StageDispatch);
                d.tilesRef = b.createTexture("Tile cost", {Format::R32Float, tilesX, tilesY});
                d.tilesRef = b.write(d.tilesRef, Usage::ShaderWrite, StageDispatch);
                b.setProfileShaders("overlay_tilecost");
            },
            [this, tilesX, tilesY](PassContext& ctx) {
                auto* enc = static_cast<MTL4::ComputeCommandEncoder*>(ctx.encoder());
                impl_->computeArgs->setAddress(impl_->constants, kBindConstants);
                impl_->computeArgs->setTexture(textureId(ctx, impl_->overdrawRef), 0);
                impl_->computeArgs->setTexture(textureId(ctx, impl_->lightsRef), 1);
                impl_->computeArgs->setTexture(textureId(ctx, impl_->tilesRef), 2);
                enc->setComputePipelineState(impl_->pipelines.compute(impl_->tileCost));
                enc->setArgumentTable(impl_->computeArgs);
                enc->dispatchThreadgroups(MTL::Size::Make(tilesX, tilesY, 1),
                                          MTL::Size::Make(overlay::TILE_SIZE, overlay::TILE_SIZE, 1));
            });
        values = d.tilesRef;
    }

    const TextureRef valueTexture = values;
    graph.addPass(
        "Overlay composite", PassType::Raster,
        [&](PassBuilder& b) {
            b.read(valueTexture, Usage::ShaderRead, StageFragment);
            color = b.writeColor(color, 0, LoadIntent::Preserve);
            b.setProfileShaders("overlay_composite_vs,overlay_composite_fs");
        },
        [this, valueTexture](PassContext& ctx) {
            auto* enc = static_cast<MTL4::RenderCommandEncoder*>(ctx.encoder());
            impl_->compositeArgs->setAddress(impl_->constants, kBindConstants);
            impl_->compositeArgs->setTexture(textureId(ctx, valueTexture), 0);
            enc->setRenderPipelineState(impl_->pipelines.render(impl_->composite));
            enc->setArgumentTable(impl_->compositeArgs, MTL::RenderStageFragment);
            enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
        });
    return color;
}

} // namespace phosphor
