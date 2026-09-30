// B-13: hidden-surface removal efficiency vs overdraw and draw order.
// Serves S-TBDR-2 of docs/APPLE_SOC_PLAYBOOK.md.
//
// L (1,2,4,8,16) full-screen opaque layers at distinct depths with an
// expensive fragment shader, depth test Less, in ONE render pass (1920x1080
// RGBA8 + Depth32F, colour stored, depth dontCare):
//   hsr.front_to_back   nearest layer first: the later layers are hidden
//   hsr.back_to_front   farthest first: every layer passes the depth test, only
//                       tile-based HSR can avoid shading the hidden ones
//   blend.back_to_front alpha blending (alpha = 1 at run time): HSR cannot be used
//   blend.front_to_back blending + depth test: early depth test still culls
//   discard.*           opaque but the shader contains a (never taken) discard
// Time = CommandTimer span of the pass (minus the empty span).
//
// Verification: read back a 64x64 corner and the last 64 pixels; the visible
// layer must be layer 0 (nearest) for every depth-tested variant, and a
// blend + depth-Always pass in order 0..L-1 must show layer L-1 (proves later
// layers are really executed).  Colours are hashes of (x, y, layer) that the
// CPU recomputes.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace soc {
namespace {

constexpr u32 kW = 1920, kH = 1080;
constexpr u32 kMaxLayers = 16;
constexpr size_t kSlot = 256;

struct LayerParams {
    float z;
    u32 layer, iters, zero;
};

u32 hash(u32 x, u32 y, u32 c) {
    u32 h = x * 73856093u ^ y * 19349663u ^ (c * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

enum class Kind { Opaque, Blend, Discard };
enum class Order { FrontToBack, BackToFront, Ascending }; // Ascending = 0..L-1 regardless of depth

struct Rig {
    MTL::RenderPipelineState* pso[3] = {};
    MTL::DepthStencilState* less   = nullptr;
    MTL::DepthStencilState* always = nullptr;
    MTL::Texture* color = nullptr;
    MTL::Texture* depth = nullptr;
    MTL::Buffer* params = nullptr;
};

// Layer l of L sits at depth (l + 1) / (L + 1): layer 0 is the nearest (Less).
void writeParams(const Rig& r, u32 layers, u32 iters) {
    for (u32 l = 0; l < layers; ++l) {
        const LayerParams p{float(l + 1) / float(layers + 1), l, iters, 0};
        std::memcpy(static_cast<u8*>(r.params->contents()) + l * kSlot, &p, sizeof(p));
    }
}

void encode(Context& ctx, const Rig& r, MTL4::CommandBuffer* cmd, Kind kind, Order order, u32 layers, bool depthAlways) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(r.color);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionStore);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    auto* d = MTL::RenderPassDepthAttachmentDescriptor::alloc()->init();
    d->setTexture(r.depth);
    d->setLoadAction(MTL::LoadActionClear);
    d->setStoreAction(MTL::StoreActionDontCare);
    d->setClearDepth(1.0);
    pd->setDepthAttachment(d);
    d->release();
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setRenderPipelineState(r.pso[int(kind)]);
    re->setDepthStencilState(depthAlways ? r.always : r.less);
    for (u32 i = 0; i < layers; ++i) {
        u32 l = i; // FrontToBack and Ascending: 0..L-1
        if (order == Order::BackToFront) l = layers - 1 - i;
        ctx.table()->setAddress(r.params->gpuAddress() + l * kSlot, 0);
        re->setArgumentTable(ctx.table(), MTL::RenderStageVertex | MTL::RenderStageFragment);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    }
    re->endEncoding();
}

double timeRun(Context& ctx, const Rig& r, Kind kind, Order order, u32 layers) {
    CommandTimer t(ctx);
    encode(ctx, r, t.begin(), kind, order, layers, false);
    return t.finish() - ctx.emptySpanMs();
}

// Returns wrong components in the read-back of the pass; `expectLayer` = the visible layer.
u32 verify(Context& ctx, const Rig& r, Kind kind, Order order, u32 layers, bool depthAlways, u32 expectLayer) {
    MTL::Buffer* rb = ctx.buffer(64 * 64 * 4 + 64 * 4);
    std::memset(rb->contents(), 0, rb->length());
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    encode(ctx, r, cmd, kind, order, layers, depthAlways);
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    ce->copyFromTexture(r.color, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(64, 64, 1), rb, 0, 64 * 4, 64 * 64 * 4);
    ce->copyFromTexture(r.color, 0, 0, MTL::Origin::Make(kW - 64, kH - 1, 0), MTL::Size::Make(64, 1, 1), rb, 64 * 64 * 4,
                        64 * 4, 64 * 4);
    ce->endEncoding();
    ctx.submit();
    const u8* b = static_cast<const u8*>(rb->contents());
    u32 wrong = 0;
    auto check = [&](const u8* px, u32 x, u32 y) {
        for (u32 c = 0; c < 3; ++c)
            if (px[c] != (hash(x, y, expectLayer * 4 + c) & 255u)) ++wrong;
        if (px[3] != 255) ++wrong;
    };
    for (u32 y = 0; y < 64; ++y)
        for (u32 x = 0; x < 64; ++x) check(b + (y * 64 + x) * 4, x, y);
    for (u32 x = 0; x < 64; ++x) check(b + 64 * 64 * 4 + x * 4, kW - 64 + x, kH - 1);
    return wrong;
}

std::string num(double v, size_t n = 6) { return std::to_string(v).substr(0, n); }

void benchHsr(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b13_hsr.metal");
    Rig r;
    for (int k = 0; k < 3; ++k) {
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(ctx.function(lib, "b13_vertex"));
        d->setFragmentFunctionDescriptor(ctx.function(lib, k == 2 ? "b13_frag_discard" : "b13_frag"));
        auto* ca = d->colorAttachments()->object(0);
        ca->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
        if (k == 1) {
            ca->setBlendingState(MTL4::BlendStateEnabled);
            ca->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
            ca->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
            ca->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
            ca->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        }
        r.pso[k] = ctx.render(d);
        d->release();
    }
    for (int k = 0; k < 2; ++k) {
        MTL::DepthStencilDescriptor* dd = MTL::DepthStencilDescriptor::alloc()->init();
        dd->setDepthCompareFunction(k == 0 ? MTL::CompareFunctionLess : MTL::CompareFunctionAlways);
        dd->setDepthWriteEnabled(k == 0);
        MTL::DepthStencilState* s = ctx.device()->newDepthStencilState(dd);
        dd->release();
        ctx.keep(s);
        (k == 0 ? r.less : r.always) = s;
    }
    {
        MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kW, kH, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(MTL::StorageModePrivate);
        r.color = ctx.texture(td);
        td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, kW, kH, false);
        td->setUsage(MTL::TextureUsageRenderTarget);
        td->setStorageMode(MTL::StorageModePrivate);
        r.depth = ctx.texture(td);
    }
    r.params = ctx.buffer(kMaxLayers * kSlot);

    // Calibrate the shader so that one blended layer costs ~0.07 ms (a pass of 16 stays < 1.5 ms).
    u32 iters = 64;
    {
        writeParams(r, 8, iters);
        ctx.keepWarm(50);
        const double t8 = std::max(1e-3, ctx.measure([&] { return timeRun(ctx, r, Kind::Blend, Order::BackToFront, 8); }, 5).median);
        writeParams(r, 1, iters);
        const double t1 = std::max(1e-3, ctx.measure([&] { return timeRun(ctx, r, Kind::Blend, Order::BackToFront, 1); }, 5).median);
        // Reference layer count changes z but not the cost; use the difference of 8 and 1 layers.
        const double perLayer = std::max(1e-3, (t8 - t1) / 7.0);
        iters = std::clamp<u32>(u32(double(iters) * 0.07 / perLayer), 4, 4096);
    }
    ctx.log("B-13 calibrated: %u iterations per fragment", iters);

    const u32 layerList[] = {1, 2, 4, 8, 16};
    const u32 reps = ctx.quick() ? 9 : 21;
    struct V {
        const char* name;
        Kind kind;
        Order order;
    };
    const V variants[] = {{"hsr.front_to_back", Kind::Opaque, Order::FrontToBack},
                          {"hsr.back_to_front", Kind::Opaque, Order::BackToFront},
                          {"blend.back_to_front", Kind::Blend, Order::BackToFront},
                          {"blend.front_to_back", Kind::Blend, Order::FrontToBack},
                          {"discard.front_to_back", Kind::Discard, Order::FrontToBack},
                          {"discard.back_to_front", Kind::Discard, Order::BackToFront}};
    std::map<std::string, double> t; // "variant.L" -> ms
    for (u32 L : layerList) {
        writeParams(r, L, iters);
        for (const V& v : variants) {
            ctx.keepWarm(15);
            const Stats s = ctx.measure([&] { return timeRun(ctx, r, v.kind, v.order, L); }, reps);
            t[std::string(v.name) + "." + std::to_string(L)] = s.median;
            rep.metric(std::string(v.name) + ".layers_" + std::to_string(L), "ms", s,
                       {{"layers", double(L)}, {"iters", double(iters)}, {"w", double(kW)}, {"h", double(kH)}}, false);
        }
        ctx.log("B-13 L=%2u f2b %.3f b2f %.3f blend %.3f blend_f2b %.3f discard f2b %.3f b2f %.3f ms", L,
                t["hsr.front_to_back." + std::to_string(L)], t["hsr.back_to_front." + std::to_string(L)],
                t["blend.back_to_front." + std::to_string(L)], t["blend.front_to_back." + std::to_string(L)],
                t["discard.front_to_back." + std::to_string(L)], t["discard.back_to_front." + std::to_string(L)]);
    }
    auto T = [&](const char* n, u32 L) { return t[std::string(n) + "." + std::to_string(L)]; };
    // Derived: cost of the 16th layer relative to the first (HSR efficiency), per variant.
    for (const V& v : variants)
        rep.value(std::string(v.name) + ".scaling_16_over_1", "ratio", T(v.name, 16) / std::max(1e-6, T(v.name, 1)), {}, false);
    // Time added per extra layer over the 1 -> 16 range.
    for (const V& v : variants)
        rep.value(std::string(v.name) + ".ms_per_extra_layer", "ms", (T(v.name, 16) - T(v.name, 1)) / 15.0, {}, false);

    // --- Correctness ------------------------------------------------------------------
    u32 wrong = 0;
    writeParams(r, 8, iters);
    wrong += verify(ctx, r, Kind::Opaque, Order::FrontToBack, 8, false, 0);
    wrong += verify(ctx, r, Kind::Opaque, Order::BackToFront, 8, false, 0);
    wrong += verify(ctx, r, Kind::Blend, Order::BackToFront, 8, false, 0);
    wrong += verify(ctx, r, Kind::Discard, Order::BackToFront, 8, false, 0);
    wrong += verify(ctx, r, Kind::Blend, Order::Ascending, 8, true, 7); // every layer executed, last wins

    // --- Negative controls --------------------------------------------------------------
    // Blended time scales ~linearly with the layer count: (t16 - t2) / (t8 - t2) = 14/6 ideally.
    const double bl = (T("blend.back_to_front", 16) - T("blend.back_to_front", 2)) /
                      std::max(1e-6, T("blend.back_to_front", 8) - T("blend.back_to_front", 2));
    const double blendAdd = T("blend.back_to_front", 16) - T("blend.back_to_front", 1);
    const double f2bAdd   = T("hsr.front_to_back", 16) - T("hsr.front_to_back", 1);
    const double b2fAdd   = T("hsr.back_to_front", 16) - T("hsr.back_to_front", 1);
    rep.negative(bl > 1.6 && bl < 3.2, "blend.back_to_front (t16-t2)/(t8-t2) = " + num(bl, 4) + " (ideal 2.33, accepted 1.6..3.2)");
    // Opaque front-to-back must NOT scale: early depth test culls the hidden layers.
    rep.negative(f2bAdd < 0.3 * blendAdd, "hsr.front_to_back added time 1->16 layers " + num(f2bAdd, 5) + " ms < 0.3 x blend " +
                                              num(blendAdd, 5) + " ms");
    rep.negative(wrong == 0, wrong == 0 ? "read-back: nearest layer visible in opaque/discard variants, last layer in blend+depth-Always (all 8 layers executed)"
                                        : std::to_string(wrong) + " wrong components in the read-back");
    if (wrong) rep.status(Status::Failed, "read-back differs from the CPU model");
    if (b2fAdd > 0.3 * blendAdd)
        rep.note("back-to-front opaque scales with the layer count (" + num(b2fAdd / std::max(1e-6, blendAdd), 4) +
                 " of the blended slope): HSR does NOT remove the hidden layers of this order on this GPU");
    else
        rep.note("back-to-front opaque does not scale (" + num(b2fAdd / std::max(1e-6, blendAdd), 4) + " of the blended slope): tile-based HSR removes hidden layers regardless of order");
    rep.note("1 pass 1920x1080 RGBA8+Depth32F, fragment = " + std::to_string(iters) + " dependent FMA pairs calibrated to ~0.07 ms per blended layer; span minus empty span");
}

} // namespace

SOC_BENCH("B-13", "hsr", "HSR efficiency: opaque front-to-back / back-to-front vs blending vs discard, 1..16 layers", benchHsr);

} // namespace soc
