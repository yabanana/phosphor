#include "rendergraph/render_graph.h"
#include "rendergraph/tbdr_passes.h"

#include <doctest/doctest.h>

#include <string>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

TextureDesc colorDesc(u32 w = 1280, u32 h = 720, Format f = Format::BGRA8Unorm) {
    TextureDesc d;
    d.format = f;
    d.width  = w;
    d.height = h;
    return d;
}

TextureDesc depthDesc(u32 w = 1280, u32 h = 720) { return colorDesc(w, h, Format::Depth32Float); }

ClearValue clearColor(float r) {
    ClearValue c;
    c.color[0] = r;
    return c;
}

CompiledGraph build(const RenderGraph& graph, bool fuse = true) {
    CompiledGraph c = compileOrder(graph);
    REQUIRE(c.ok);
    buildRenderGroups(graph, c, fuse);
    return c;
}

const AttachmentPlan* findAttachment(const RenderGroup& g, u32 resource) {
    for (const AttachmentPlan& a : g.attachments) {
        if (a.resource == resource) return &a;
    }
    return nullptr;
}

} // namespace

TEST_CASE("tbdr: names of load and store actions") {
    CHECK(std::string(loadActionName(LoadAction::DontCare)) == "DontCare");
    CHECK(std::string(loadActionName(LoadAction::Load)) == "Load");
    CHECK(std::string(loadActionName(LoadAction::Clear)) == "Clear");
    CHECK(std::string(storeActionName(StoreAction::DontCare)) == "DontCare");
    CHECK(std::string(storeActionName(StoreAction::Store)) == "Store");
}

// The real frame: Forward (drawable Clear + transient depth Clear) then ImGui
// (drawable Preserve).
static void buildEngineFrame(RenderGraph& g, TextureRef& drawable, TextureRef& depthOut) {
    drawable = g.importTexture("drawable", colorDesc(), ImportOutput);
    TextureRef color = drawable;
    g.addPass("Forward", PassType::Raster, [&](PassBuilder& b) {
        color = b.writeColor(color, 0, LoadIntent::Clear, clearColor(0.1f));
        TextureRef d = b.createTexture("depth", depthDesc());
        ClearValue cv;
        cv.depth = 0.0f;
        depthOut = b.writeDepth(d, LoadIntent::Clear, cv);
    }, nullptr);
    g.addPass("ImGui", PassType::Raster, [&](PassBuilder& b) {
        color = b.writeColor(color, 0, LoadIntent::Preserve);
    }, nullptr);
}

TEST_CASE("tbdr: engine frame fuses Forward and ImGui into one render pass") {
    RenderGraph g;
    TextureRef drawable, depth;
    buildEngineFrame(g, drawable, depth);
    const CompiledGraph c = build(g);
    REQUIRE(c.ok);
    REQUIRE(c.renderGroups.size() == 1);
    const RenderGroup& grp = c.renderGroups[0];
    CHECK(grp.firstPosition == 0);
    CHECK(grp.lastPosition == 1);
    CHECK(grp.width == 1280);
    CHECK(grp.height == 720);
    CHECK(grp.sampleCount == 1);
    REQUIRE(grp.attachments.size() == 2);

    const AttachmentPlan* col = findAttachment(grp, drawable.resource);
    REQUIRE(col);
    CHECK_FALSE(col->depth);
    CHECK(col->slot == 0);
    CHECK(col->load == LoadAction::Clear);
    CHECK(col->store == StoreAction::Store);
    CHECK(col->clear.color[0] == doctest::Approx(0.1f));

    const AttachmentPlan* dep = findAttachment(grp, depth.resource);
    REQUIRE(dep);
    CHECK(dep->depth);
    CHECK(dep->load == LoadAction::Clear);
    CHECK(dep->store == StoreAction::DontCare);
    CHECK_FALSE(dep->readOnly);

    CHECK(c.memoryless[depth.resource]);
    CHECK_FALSE(c.memoryless[drawable.resource]);

    CHECK(c.groupOfPosition == std::vector<u32>{0, 0});
    REQUIRE(c.encoders.size() == 1);
    CHECK(c.encoders[0].type == PassType::Raster);
    CHECK(c.encoders[0].renderGroup == 0);
    CHECK(c.encoders[0].firstPosition == 0);
    CHECK(c.encoders[0].lastPosition == 1);
    CHECK(c.encoderOfPosition == std::vector<u32>{0, 0});
}

TEST_CASE("tbdr: fuse=false keeps two groups, depth stays memoryless") {
    RenderGraph g;
    TextureRef drawable, depth;
    buildEngineFrame(g, drawable, depth);
    const CompiledGraph c = build(g, false);
    REQUIRE(c.renderGroups.size() == 2);
    const AttachmentPlan* fwd = findAttachment(c.renderGroups[0], drawable.resource);
    REQUIRE(fwd);
    CHECK(fwd->load == LoadAction::Clear);
    CHECK(fwd->store == StoreAction::Store); // read by the overlay's load
    const AttachmentPlan* ov = findAttachment(c.renderGroups[1], drawable.resource);
    REQUIRE(ov);
    CHECK(ov->load == LoadAction::Load);
    CHECK(ov->store == StoreAction::Store); // ImportOutput
    CHECK(c.renderGroups[1].attachments.size() == 1);
    const AttachmentPlan* dep = findAttachment(c.renderGroups[0], depth.resource);
    REQUIRE(dep);
    CHECK(dep->store == StoreAction::DontCare);
    CHECK(c.memoryless[depth.resource]);
    CHECK(c.encoders.size() == 2);
    CHECK(c.encoderOfPosition == std::vector<u32>{0, 1});
}

TEST_CASE("tbdr: sampling an attachment of the previous pass breaks fusion") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    TextureRef mid;
    g.addPass("GBuffer", PassType::Raster, [&](PassBuilder& b) {
        mid = b.createTexture("mid", colorDesc());
        mid = b.writeColor(mid, 0, LoadIntent::Clear);
    }, nullptr);
    g.addPass("Lighting", PassType::Raster, [&](PassBuilder& b) {
        b.read(mid, Usage::ShaderRead, StageFragment);
        out = b.writeColor(out, 0, LoadIntent::Discard);
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 2);
    const AttachmentPlan* a = findAttachment(c.renderGroups[0], mid.resource);
    REQUIRE(a);
    CHECK(a->store == StoreAction::Store);
    CHECK(a->load == LoadAction::Clear);
    CHECK_FALSE(c.memoryless[mid.resource]);
    const AttachmentPlan* o = findAttachment(c.renderGroups[1], out.resource);
    REQUIRE(o);
    CHECK(o->load == LoadAction::DontCare); // Discard
    CHECK(o->store == StoreAction::Store);
}

TEST_CASE("tbdr: a write of something a group member samples breaks fusion") {
    RenderGraph g;
    TextureRef src = g.importTexture("src", colorDesc(), ImportContentsDefined);
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    g.addPass("A", PassType::Raster, [&](PassBuilder& b) {
        b.read(src, Usage::ShaderRead, StageFragment);
        out = b.writeColor(out, 0, LoadIntent::Clear);
    }, nullptr);
    g.addPass("B", PassType::Raster, [&](PassBuilder& b) {
        src = b.write(src, Usage::ShaderWrite, StageFragment);
        out = b.writeColor(out, 0, LoadIntent::Preserve);
    }, nullptr);
    const CompiledGraph c = build(g);
    CHECK(c.renderGroups.size() == 2);
}

TEST_CASE("tbdr: mismatched sizes give two groups") {
    RenderGraph g;
    TextureRef a = g.importTexture("a", colorDesc(1280, 720), ImportOutput);
    TextureRef b2 = g.importTexture("b", colorDesc(640, 360), ImportOutput);
    g.addPass("Full", PassType::Raster, [&](PassBuilder& b) { a = b.writeColor(a, 0, LoadIntent::Clear); }, nullptr);
    g.addPass("Half", PassType::Raster, [&](PassBuilder& b) { b2 = b.writeColor(b2, 0, LoadIntent::Clear); },
              nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 2);
    CHECK(c.renderGroups[0].width == 1280);
    CHECK(c.renderGroups[1].width == 640);
    CHECK(c.renderGroups[1].height == 360);
}

TEST_CASE("tbdr: sample count mismatch gives two groups") {
    RenderGraph g;
    TextureDesc msaa = colorDesc();
    msaa.sampleCount = 4;
    TextureRef a = g.importTexture("a", colorDesc(), ImportOutput);
    TextureRef m = g.importTexture("m", msaa, ImportOutput);
    g.addPass("P1", PassType::Raster, [&](PassBuilder& b) { a = b.writeColor(a, 0, LoadIntent::Clear); }, nullptr);
    g.addPass("P2", PassType::Raster, [&](PassBuilder& b) { m = b.writeColor(m, 0, LoadIntent::Clear); }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 2);
    CHECK(c.renderGroups[1].sampleCount == 4);
}

TEST_CASE("tbdr: slot conflict gives two groups") {
    RenderGraph g;
    TextureRef a = g.importTexture("a", colorDesc(), ImportOutput);
    TextureRef b2 = g.importTexture("b", colorDesc(), ImportOutput);
    g.addPass("P1", PassType::Raster, [&](PassBuilder& b) { a = b.writeColor(a, 0, LoadIntent::Clear); }, nullptr);
    g.addPass("P2", PassType::Raster, [&](PassBuilder& b) { b2 = b.writeColor(b2, 0, LoadIntent::Clear); },
              nullptr);
    CHECK(build(g).renderGroups.size() == 2);
}

TEST_CASE("tbdr: distinct slots fuse into one group with several attachments") {
    RenderGraph g;
    TextureRef a = g.importTexture("a", colorDesc(), ImportOutput);
    TextureRef b2 = g.importTexture("b", colorDesc(), ImportOutput);
    g.addPass("P1", PassType::Raster, [&](PassBuilder& b) { a = b.writeColor(a, 0, LoadIntent::Clear); }, nullptr);
    g.addPass("P2", PassType::Raster, [&](PassBuilder& b) { b2 = b.writeColor(b2, 1, LoadIntent::Clear); },
              nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 1);
    REQUIRE(c.renderGroups[0].attachments.size() == 2);
    CHECK(c.renderGroups[0].attachments[0].slot == 0);
    CHECK(c.renderGroups[0].attachments[1].slot == 1);
}

TEST_CASE("tbdr: same resource in a different slot gives two groups") {
    RenderGraph g;
    TextureRef a = g.importTexture("a", colorDesc(), ImportOutput);
    g.addPass("P1", PassType::Raster, [&](PassBuilder& b) { a = b.writeColor(a, 0, LoadIntent::Clear); }, nullptr);
    g.addPass("P2", PassType::Raster, [&](PassBuilder& b) { a = b.writeColor(a, 1, LoadIntent::Preserve); },
              nullptr);
    CHECK(build(g).renderGroups.size() == 2);
}

TEST_CASE("tbdr: a compute pass between raster passes separates the groups") {
    RenderGraph g;
    TextureRef drawable = g.importTexture("drawable", colorDesc(), ImportOutput);
    TextureRef buf;
    g.addPass("R1", PassType::Raster, [&](PassBuilder& b) {
        drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
    }, nullptr);
    g.addPass("C", PassType::Compute, [&](PassBuilder& b) {
        buf = b.createTexture("scratch", colorDesc(64, 64));
        buf = b.write(buf, Usage::ShaderWrite, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    g.addPass("R2", PassType::Raster, [&](PassBuilder& b) {
        drawable = b.writeColor(drawable, 0, LoadIntent::Preserve);
        b.read(buf, Usage::ShaderRead, StageFragment);
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 2);
    REQUIRE(c.order.size() == 3);
    CHECK(c.groupOfPosition[0] == 0);
    CHECK(c.groupOfPosition[1] == ~0u);
    CHECK(c.groupOfPosition[2] == 1);
    REQUIRE(c.encoders.size() == 3);
    CHECK(c.encoders[0].type == PassType::Raster);
    CHECK(c.encoders[1].type == PassType::Compute);
    CHECK(c.encoders[2].type == PassType::Raster);
    CHECK(c.encoders[2].renderGroup == 1);
    CHECK(c.encoderOfPosition == std::vector<u32>{0, 1, 2});
}

TEST_CASE("tbdr: compute and blit runs share one compute encoder") {
    RenderGraph g;
    BufferRef buf;
    g.addPass("C1", PassType::Compute, [&](PassBuilder& b) {
        buf = b.createBuffer("buf", BufferDesc{256});
        buf = b.write(buf, Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    g.addPass("Blit", PassType::Blit, [&](PassBuilder& b) {
        b.read(buf, Usage::CopySrc, StageBlit);
        BufferRef dst = b.createBuffer("dst", BufferDesc{256});
        dst = b.write(dst, Usage::CopyDst, StageBlit);
        b.setSideEffect();
    }, nullptr);
    g.addPass("C2", PassType::Compute, [&](PassBuilder& b) {
        b.read(buf, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    const CompiledGraph c = build(g);
    CHECK(c.renderGroups.empty());
    REQUIRE(c.encoders.size() == 1);
    CHECK(c.encoders[0].type == PassType::Compute);
    CHECK(c.encoders[0].queue == Queue::Graphics);
    CHECK(c.encoders[0].firstPosition == 0);
    CHECK(c.encoders[0].lastPosition == 2);
    CHECK(c.encoderOfPosition == std::vector<u32>{0, 0, 0});
    CHECK(c.groupOfPosition == std::vector<u32>{~0u, ~0u, ~0u});
}

TEST_CASE("tbdr: an async compute pass splits a compute run") {
    RenderGraph g;
    g.addPass("G1", PassType::Compute, [&](PassBuilder& b) { b.setSideEffect(); }, nullptr);
    g.addPass("A", PassType::Compute, Queue::AsyncCompute, [&](PassBuilder& b) { b.setSideEffect(); }, nullptr);
    g.addPass("G2", PassType::Compute, [&](PassBuilder& b) { b.setSideEffect(); }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.order.size() == 3);
    REQUIRE(c.encoders.size() == 3);
    CHECK(c.encoders[0].queue == Queue::Graphics);
    CHECK(c.encoders[1].queue == Queue::AsyncCompute);
    CHECK(c.encoders[2].queue == Queue::Graphics);
    CHECK(c.encoderOfPosition == std::vector<u32>{0, 1, 2});
}

TEST_CASE("tbdr: Discard loads as DontCare") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    g.addPass("Full", PassType::Raster, [&](PassBuilder& b) {
        out = b.writeColor(out, 0, LoadIntent::Discard);
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 1);
    CHECK(c.renderGroups[0].attachments[0].load == LoadAction::DontCare);
    CHECK(c.renderGroups[0].attachments[0].store == StoreAction::Store);
}

TEST_CASE("tbdr: a DepthRead-only group has a read-only depth loaded and stored when needed later") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    TextureRef depth;
    g.addPass("Depth prepass", PassType::Raster, [&](PassBuilder& b) {
        depth = b.createTexture("depth", depthDesc());
        depth = b.writeDepth(depth, LoadIntent::Clear);
    }, nullptr);
    g.addPass("Sample depth", PassType::Compute, [&](PassBuilder& b) {
        b.read(depth, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    g.addPass("Shade", PassType::Raster, [&](PassBuilder& b) {
        out = b.writeColor(out, 0, LoadIntent::Clear);
        b.readDepth(depth);
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 2);
    const RenderGroup& shade = c.renderGroups[1];
    const AttachmentPlan* d = findAttachment(shade, depth.resource);
    REQUIRE(d);
    CHECK(d->depth);
    CHECK(d->readOnly);
    CHECK(d->load == LoadAction::Load);
    CHECK(d->store == StoreAction::DontCare); // nobody reads it after the group
    CHECK_FALSE(c.memoryless[depth.resource]);
    // The prepass stores it: the compute pass and the shading pass read it.
    const AttachmentPlan* pre = findAttachment(c.renderGroups[0], depth.resource);
    REQUIRE(pre);
    CHECK(pre->store == StoreAction::Store);
    CHECK_FALSE(pre->readOnly);
}

TEST_CASE("tbdr: a read-only depth needed by a later pass is stored") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    TextureRef depth;
    g.addPass("Depth prepass", PassType::Raster, [&](PassBuilder& b) {
        depth = b.createTexture("depth", depthDesc());
        depth = b.writeDepth(depth, LoadIntent::Clear);
    }, nullptr);
    g.addPass("Shade", PassType::Raster, [&](PassBuilder& b) {
        out = b.writeColor(out, 0, LoadIntent::Clear);
        b.readDepth(depth);
    }, nullptr);
    g.addPass("Post", PassType::Compute, [&](PassBuilder& b) {
        b.read(depth, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 1);
    REQUIRE(c.renderGroups[0].attachments.size() == 2);
    const AttachmentPlan* d = findAttachment(c.renderGroups[0], depth.resource);
    REQUIRE(d);
    CHECK_FALSE(d->readOnly); // written by the prepass member of the same group
    CHECK(d->store == StoreAction::Store);
    CHECK_FALSE(c.memoryless[depth.resource]);
}

TEST_CASE("tbdr: a pure DepthRead group is readOnly") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    TextureRef depth = g.importTexture("depth", depthDesc(), ImportContentsDefined);
    g.addPass("Shade", PassType::Raster, [&](PassBuilder& b) {
        out = b.writeColor(out, 0, LoadIntent::Clear);
        b.readDepth(depth);
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 1);
    const AttachmentPlan* d = findAttachment(c.renderGroups[0], depth.resource);
    REQUIRE(d);
    CHECK(d->readOnly);
    CHECK(d->depth);
    CHECK(d->load == LoadAction::Load);
    CHECK(d->store == StoreAction::DontCare);
    CHECK_FALSE(c.memoryless[depth.resource]); // imported
}

TEST_CASE("tbdr: a raster pass without attachments is an error") {
    RenderGraph g;
    TextureRef src = g.importTexture("src", colorDesc(), ImportContentsDefined);
    g.addPass("Bad", PassType::Raster, [&](PassBuilder& b) {
        b.read(src, Usage::ShaderRead, StageFragment);
        b.setSideEffect();
    }, nullptr);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    buildRenderGroups(g, c, true);
    CHECK_FALSE(c.ok);
    REQUIRE_FALSE(c.errors.empty());
    CHECK(c.errors[0].find("Bad") != std::string::npos);
}

TEST_CASE("tbdr: attachments of different sizes in one pass are an error") {
    RenderGraph g;
    TextureRef a = g.importTexture("a", colorDesc(1280, 720), ImportOutput);
    TextureRef b2 = g.importTexture("b", colorDesc(640, 360), ImportOutput);
    g.addPass("Mixed", PassType::Raster, [&](PassBuilder& b) {
        a = b.writeColor(a, 0, LoadIntent::Clear);
        b2 = b.writeColor(b2, 1, LoadIntent::Clear);
    }, nullptr);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    buildRenderGroups(g, c, true);
    CHECK_FALSE(c.ok);
    CHECK_FALSE(c.errors.empty());
}

TEST_CASE("tbdr: a texture written by a group and read by a later compute pass is stored, not memoryless") {
    RenderGraph g;
    TextureRef t;
    g.addPass("Render", PassType::Raster, [&](PassBuilder& b) {
        t = b.createTexture("t", colorDesc());
        t = b.writeColor(t, 0, LoadIntent::Clear);
    }, nullptr);
    g.addPass("Compute", PassType::Compute, [&](PassBuilder& b) {
        b.read(t, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.renderGroups.size() == 1);
    CHECK(c.renderGroups[0].attachments[0].store == StoreAction::Store);
    CHECK_FALSE(c.memoryless[t.resource]);
}

TEST_CASE("tbdr: a transient used as attachment in two groups is not memoryless") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    TextureRef t;
    g.addPass("R1", PassType::Raster, [&](PassBuilder& b) {
        t = b.createTexture("t", colorDesc());
        t = b.writeColor(t, 0, LoadIntent::Clear);
    }, nullptr);
    g.addPass("C", PassType::Compute, [&](PassBuilder& b) {
        BufferRef x = b.createBuffer("x", BufferDesc{16});
        x = b.write(x, Usage::ShaderWrite, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    g.addPass("R2", PassType::Raster, [&](PassBuilder& b) {
        t = b.writeColor(t, 0, LoadIntent::Preserve);
        out = b.writeColor(out, 1, LoadIntent::Clear);
    }, nullptr);
    g.addPass("Use", PassType::Compute, [&](PassBuilder& b) {
        b.read(t, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, nullptr);
    const CompiledGraph c = build(g);
    CHECK_FALSE(c.memoryless[t.resource]);
    REQUIRE(c.renderGroups.size() == 2);
    CHECK(c.renderGroups[0].attachments[0].store == StoreAction::Store);
    CHECK(c.renderGroups[1].attachments[0].load == LoadAction::Load);
}

TEST_CASE("tbdr: culled passes are ignored") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    g.addPass("Dead", PassType::Raster, [&](PassBuilder& b) {
        TextureRef t = b.createTexture("dead", colorDesc(64, 64));
        t = b.writeColor(t, 0, LoadIntent::Clear);
    }, nullptr);
    g.addPass("Live", PassType::Raster, [&](PassBuilder& b) { out = b.writeColor(out, 0, LoadIntent::Clear); },
              nullptr);
    const CompiledGraph c = build(g);
    REQUIRE(c.order.size() == 1);
    REQUIRE(c.renderGroups.size() == 1);
    CHECK(c.renderGroups[0].width == 1280);
    CHECK(c.groupOfPosition.size() == 1);
    CHECK(c.encoders.size() == 1);
    CHECK_FALSE(c.memoryless[1]); // the culled texture is not used by any live pass
}

TEST_CASE("tbdr: a mid-group Clear of an already attached resource breaks fusion") {
    RenderGraph g;
    TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
    g.addPass("P1", PassType::Raster, [&](PassBuilder& b) { out = b.writeColor(out, 0, LoadIntent::Clear); b.setSideEffect(); },
              nullptr);
    g.addPass("P2", PassType::Raster, [&](PassBuilder& b) { out = b.writeColor(out, 0, LoadIntent::Clear); },
              nullptr);
    const CompiledGraph c = build(g);
    CHECK(c.renderGroups.size() == 2);
}

TEST_CASE("tbdr: tile shading keeps IDs in the raster imageblock and orders device outputs") {
    RenderGraph g;
    TextureRef id;
    auto out = g.importTexture("hdr", colorDesc(320, 180, Format::RGBA16Float), ImportOutput);
    g.addPass(
        "IDs", PassType::Raster,
        [&](PassBuilder &b) {
            id = b.createTexture("id", colorDesc(320, 180, Format::R32Uint));
            id = b.writeColor(id, 0, LoadIntent::Clear);
        },
        nullptr);
    g.addPass(
        "Tile shading", PassType::Raster,
        [&](PassBuilder &b) {
            b.setTileSize(16, 16);
            b.readColor(id, 0, StageTile);
            out = b.write(out, Usage::ShaderWrite, StageTile);
        },
        nullptr);
    const auto compiled = build(g);
    REQUIRE(compiled.ok);
    REQUIRE(compiled.renderGroups.size() == 1);
    CHECK(compiled.renderGroups[0].tileWidth == 16);
    CHECK(compiled.renderGroups[0].tileHeight == 16);
    CHECK(compiled.memoryless[id.resource]);
    CHECK(compiled.renderGroups[0].attachments[0].store == StoreAction::DontCare);
}
TEST_CASE("tbdr: incompatible tile sizes split render groups") {
    RenderGraph g;
    auto target = g.importTexture("out", colorDesc(), ImportOutput);
    g.addPass(
        "16x16", PassType::Raster,
        [&](PassBuilder &b) {
            b.setTileSize(16, 16);
            target = b.writeColor(target, 0, LoadIntent::Clear);
        },
        nullptr);
    g.addPass(
        "32x32", PassType::Raster,
        [&](PassBuilder &b) {
            b.setTileSize(32, 32);
            target = b.writeColor(target, 0, LoadIntent::Preserve);
        },
        nullptr);
    const auto compiled = build(g);
    REQUIRE(compiled.ok);
    CHECK(compiled.renderGroups.size() == 2);
}
