#include "rendergraph/barrier_plan.h"
#include "rendergraph/optimizer/optimizer.h"
#include "rendergraph/render_graph.h"
#include "rendergraph/scenario.h"

#include <doctest/doctest.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

constexpr Stages kGeometry = StageVertex | StageObject | StageMesh;

struct EncSpec {
    PassType type;
    u32      first;
    u32      last;
};

// Hand-built F2.4 output: one encoder per spec; raster encoders get one
// render group each.
void assignEncoders(const RenderGraph& graph, CompiledGraph& c, const std::vector<EncSpec>& specs) {
    const size_t n = c.order.size();
    c.encoders.clear();
    c.renderGroups.clear();
    c.encoderOfPosition.assign(n, ~0u);
    c.groupOfPosition.assign(n, ~0u);
    for (const EncSpec& s : specs) {
        EncoderPlan e;
        e.type          = s.type;
        e.queue         = graph.passes()[c.order[s.first]].queue;
        e.firstPosition = s.first;
        e.lastPosition  = s.last;
        if (s.type == PassType::Raster) {
            RenderGroup g;
            g.firstPosition = s.first;
            g.lastPosition  = s.last;
            e.renderGroup   = static_cast<u32>(c.renderGroups.size());
            c.renderGroups.push_back(g);
        }
        const u32 index = static_cast<u32>(c.encoders.size());
        c.encoders.push_back(e);
        for (u32 p = s.first; p <= s.last; ++p) {
            c.encoderOfPosition[p] = index;
            if (s.type == PassType::Raster) c.groupOfPosition[p] = e.renderGroup;
        }
    }
}

CompiledGraph plan(const RenderGraph& graph, const std::vector<EncSpec>& specs, const BarrierRules& rules = defaultBarrierRules()) {
    CompiledGraph c = compileOrder(graph);
    REQUIRE(c.ok);
    assignEncoders(graph, c, specs);
    buildBarrierPlan(graph, c, rules);
    return c;
}

const PassBarriers* at(const CompiledGraph& c, u32 position) {
    for (const PassBarriers& pb : c.barriers) if (pb.position == position) return &pb;
    return nullptr;
}

TextureDesc colorDesc() { return {Format::RGBA8Unorm, 64, 64, 1, 1, 1}; }
TextureDesc depthDesc() { return {Format::Depth32Float, 64, 64, 1, 1, 1}; }
BufferDesc  bufDesc()   { return {1024}; }

} // namespace

TEST_CASE("barriers: stagesName") {
    CHECK(stagesName(StageNone) == "none");
    CHECK(stagesName(StageVertex) == "vertex");
    CHECK(stagesName(StageVertex | StageFragment) == "vertex|fragment");
    CHECK(stagesName(StageAccelerationStructure | StageBlit | StageDispatch | StageMesh | StageObject | StageTile |
                     StageFragment | StageVertex) ==
          "vertex|fragment|tile|object|mesh|dispatch|blit|accel");
    CHECK(stagesName(StageGeometry) == "vertex|object|mesh");
}

TEST_CASE("barriers: default rules") {
    const BarrierRules r = defaultBarrierRules();
    // Measured by the F2.3 spike: Fragment synchronises on the consumer side,
    // Tile is accepted but synchronises nothing.
    CHECK(r.rasterUnsupportedBefore == StageTile);
    CHECK(r.rasterPromoteTo == StageGeometry);
    CHECK(r.unsupportedAfter == StageTile);
    CHECK(r.afterPromoteTo == StageFragment);
    CHECK(r.rasterForbiddenEncoderAfter == (StageFragment | StageTile));
    CHECK(r.computeEncoderStages == (StageDispatch | StageBlit | StageAccelerationStructure));
}

TEST_CASE("barriers: compute to compute in one encoder is an encoder-scope barrier") {
    RenderGraph g;
    BufferRef buf;
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        buf = b.write(b.createBuffer("buf", bufDesc()), Usage::ShaderWrite, StageDispatch);
        b.setSideEffect();
    }, {});
    g.addPass("b", PassType::Compute, [&](PassBuilder& b) {
        b.read(buf, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 1}});
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].position == 1);
    REQUIRE(c.barriers[0].barriers.size() == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Encoder);
    CHECK(b.afterStages == StageDispatch);
    CHECK(b.beforeStages == StageDispatch);
    CHECK_FALSE(b.aliasing);
    CHECK(b.resources == std::vector<u32>{0});
}

TEST_CASE("barriers: compute to raster fragment read waits in fragment at group start") {
    RenderGraph g;
    TextureRef tex;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("gen", PassType::Compute, [&](PassBuilder& b) {
        tex = b.write(b.createTexture("tex", colorDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("draw", PassType::Raster, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageFragment);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1}});
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].position == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == StageDispatch);
    CHECK(b.beforeStages == StageFragment);
}

TEST_CASE("barriers: raster to compute waits after fragment") {
    RenderGraph g;
    TextureRef tex;
    g.addPass("draw", PassType::Raster, [&](PassBuilder& b) {
        tex = b.writeColor(b.createTexture("tex", colorDesc()), 0, LoadIntent::Clear);
    }, {});
    g.addPass("post", PassType::Compute, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Raster, 0, 0}, {PassType::Compute, 1, 1}});
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].position == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == StageFragment);
    CHECK(b.beforeStages == StageDispatch);
}

TEST_CASE("barriers: render to render shadow map sampled in fragment") {
    RenderGraph g;
    TextureRef shadow;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("shadow", PassType::Raster, [&](PassBuilder& b) {
        shadow = b.writeDepth(b.createTexture("shadow", depthDesc()), LoadIntent::Clear);
    }, {});
    g.addPass("lit", PassType::Raster, [&](PassBuilder& b) {
        b.read(shadow, Usage::ShaderRead, StageFragment);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Raster, 0, 0}, {PassType::Raster, 1, 1}});
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].position == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == StageFragment);
    CHECK(b.beforeStages == StageFragment);
    CHECK(c.ok);
}

TEST_CASE("barriers: custom rules keep the consumer stages") {
    RenderGraph g;
    TextureRef shadow;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("shadow", PassType::Raster, [&](PassBuilder& b) {
        shadow = b.writeDepth(b.createTexture("shadow", depthDesc()), LoadIntent::Clear);
    }, {});
    g.addPass("lit", PassType::Raster, [&](PassBuilder& b) {
        b.read(shadow, Usage::ShaderRead, StageFragment);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    BarrierRules rules;
    rules.rasterUnsupportedBefore = StageNone;
    const CompiledGraph c = plan(g, {{PassType::Raster, 0, 0}, {PassType::Raster, 1, 1}}, rules);
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].barriers[0].beforeStages == StageFragment);
}

TEST_CASE("barriers: blit upload then fragment sampling") {
    RenderGraph g;
    TextureRef tex;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("upload", PassType::Blit, [&](PassBuilder& b) {
        tex = b.write(b.createTexture("tex", colorDesc()), Usage::CopyDst, StageBlit);
    }, {});
    g.addPass("draw", PassType::Raster, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageFragment);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1}});
    REQUIRE(c.barriers.size() == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == StageBlit);
    CHECK(b.beforeStages == StageFragment);
}

TEST_CASE("barriers: WAR uses the reader's stages on the after side") {
    RenderGraph g;
    BufferRef buf = g.importBuffer("buf", bufDesc(), ImportContentsDefined | ImportPerFrame);
    g.addPass("reader", PassType::Raster, [&](PassBuilder& b) {
        b.read(buf, Usage::ShaderRead, StageVertex);
        b.setSideEffect();
    }, {});
    g.addPass("writer", PassType::Compute, [&](PassBuilder& b) {
        b.write(buf, Usage::ShaderWrite, StageDispatch);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Raster, 0, 0}, {PassType::Compute, 1, 1}});
    REQUIRE(c.dependencies.size() == 1);
    CHECK(c.dependencies[0].kind == DepKind::WAR);
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].position == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == StageVertex);
    CHECK(b.beforeStages == StageDispatch);
}

TEST_CASE("barriers: WAW uses the writer's stages") {
    RenderGraph g;
    BufferRef buf;
    g.addPass("w1", PassType::Compute, [&](PassBuilder& b) {
        buf = b.write(b.createBuffer("buf", bufDesc()), Usage::ShaderWrite, StageDispatch);
        b.setSideEffect(); // keep w1 live: nothing reads its version
    }, {});
    g.addPass("w2", PassType::Compute, [&](PassBuilder& b) {
        b.write(buf, Usage::ShaderWrite, StageDispatch);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 1}});
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].barriers[0].scope == BarrierScope::Encoder);
    CHECK(c.barriers[0].barriers[0].afterStages == StageDispatch);
}

TEST_CASE("barriers: passes fused in one render group need no barrier") {
    RenderGraph g;
    TextureRef tex;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("gbuf", PassType::Raster, [&](PassBuilder& b) {
        tex = b.writeColor(b.createTexture("tex", colorDesc()), 0, LoadIntent::Clear);
    }, {});
    g.addPass("light", PassType::Raster, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageFragment);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Raster, 0, 1}});
    CHECK(c.dependencies.size() == 1);
    CHECK(c.barriers.empty());
    CHECK(c.ok);
}

TEST_CASE("barriers: dependencies at one position merge into one barrier") {
    RenderGraph g;
    BufferRef x, y;
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        x = b.write(b.createBuffer("x", bufDesc()), Usage::ShaderWrite, StageDispatch);
        y = b.write(b.createBuffer("y", bufDesc()), Usage::ShaderWrite, StageBlit);
    }, {});
    g.addPass("b", PassType::Compute, [&](PassBuilder& b) {
        b.read(y, Usage::ShaderRead, StageDispatch);
        b.read(x, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Compute, 1, 1}});
    REQUIRE(c.barriers.size() == 1);
    REQUIRE(c.barriers[0].barriers.size() == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == (StageDispatch | StageBlit));
    CHECK(b.beforeStages == StageDispatch);
    CHECK(b.resources == std::vector<u32>{0, 1});
}

TEST_CASE("barriers: consumers inside a group are hoisted to its start and merged") {
    RenderGraph g;
    BufferRef x, y;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        x = b.write(b.createBuffer("x", bufDesc()), Usage::ShaderWrite, StageDispatch);
        y = b.write(b.createBuffer("y", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("r1", PassType::Raster, [&](PassBuilder& b) {
        b.read(x, Usage::ShaderRead, StageVertex);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    g.addPass("r2", PassType::Raster, [&](PassBuilder& b) {
        b.read(y, Usage::ShaderRead, StageFragment);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 2}});
    REQUIRE(c.barriers.size() == 1);
    CHECK(c.barriers[0].position == 1);
    REQUIRE(c.barriers[0].barriers.size() == 1);
    CHECK(c.barriers[0].barriers[0].beforeStages == (StageVertex | StageFragment));
    CHECK(c.barriers[0].barriers[0].resources == std::vector<u32>{1, 2}); // target = 0
}

TEST_CASE("barriers: aliased first use gets an aliasing barrier with the union of stages") {
    RenderGraph g;
    BufferRef t1;
    TextureRef t2;
    BufferRef out = g.importBuffer("out", bufDesc(), ImportOutput | ImportPerFrame);
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        t1 = b.write(b.createBuffer("t1", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("b", PassType::Compute, [&](PassBuilder& b) {
        b.read(t1, Usage::ShaderRead, StageDispatch);
        b.write(out, Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("c", PassType::Raster, [&](PassBuilder& b) {
        t2 = b.writeColor(b.createTexture("t2", colorDesc()), 0, LoadIntent::Clear);
        b.setSideEffect();
    }, {});
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    assignEncoders(g, c, {{PassType::Compute, 0, 1}, {PassType::Raster, 2, 2}});
    // resource ids: out = 0, t1 = 1, t2 = 2
    c.aliasing.placements = {{1, 0, 1024, false}, {2, 0, 1024, true}};
    buildBarrierPlan(g, c, defaultBarrierRules());

    // Position 2: aliasing barrier before the render encoder.
    const PassBarriers* pb = at(c, 2);
    REQUIRE(pb);
    REQUIRE(pb->barriers.size() == 1);
    const Barrier& b = pb->barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.aliasing);
    CHECK(b.afterStages == (StageDispatch | StageFragment));
    CHECK(b.beforeStages == StageFragment);
    CHECK(b.resources == std::vector<u32>{2});

    // Position 0: t1's own first use, not aliased.
    const PassBarriers* p0 = at(c, 0);
    REQUIRE(p0);
    REQUIRE(p0->barriers.size() == 1);
    CHECK_FALSE(p0->barriers[0].aliasing);
    CHECK(p0->barriers[0].scope == BarrierScope::Queue);
    CHECK(p0->barriers[0].beforeStages == StageDispatch);
}

TEST_CASE("barriers: non-aliased transient first use still gets a queue barrier") {
    RenderGraph g;
    TextureRef tex;
    g.addPass("draw", PassType::Raster, [&](PassBuilder& b) {
        tex = b.writeColor(b.createTexture("tex", colorDesc()), 0, LoadIntent::Clear);
    }, {});
    g.addPass("post", PassType::Compute, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    assignEncoders(g, c, {{PassType::Raster, 0, 0}, {PassType::Compute, 1, 1}});
    c.aliasing.placements = {{0, 4096, 1024, false}};
    buildBarrierPlan(g, c, defaultBarrierRules());
    const PassBarriers* pb = at(c, 0);
    REQUIRE(pb);
    REQUIRE(pb->barriers.size() == 1);
    const Barrier& b = pb->barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK_FALSE(b.aliasing);
    CHECK(b.afterStages == (StageFragment | StageDispatch)); // previous frame's use of the same memory
    CHECK(b.beforeStages == StageFragment);
    CHECK(b.resources == std::vector<u32>{0});
    // Plus the dependency barrier at position 1.
    REQUIRE(at(c, 1));
    CHECK(c.barriers.size() == 2);
}

TEST_CASE("barriers: aliasing inside a compute encoder adds an encoder-scope barrier first") {
    RenderGraph g;
    BufferRef t1, t3;
    BufferRef out = g.importBuffer("out", bufDesc(), ImportOutput | ImportPerFrame);
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        t1 = b.write(b.createBuffer("t1", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("b", PassType::Compute, [&](PassBuilder& b) {
        b.read(t1, Usage::ShaderRead, StageDispatch);
        b.write(out, Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("c", PassType::Compute, [&](PassBuilder& b) {
        t3 = b.write(b.createBuffer("t3", bufDesc()), Usage::ShaderWrite, StageBlit);
        b.setSideEffect();
    }, {});
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    assignEncoders(g, c, {{PassType::Compute, 0, 2}});
    // resource ids: out = 0, t1 = 1, t3 = 2
    c.aliasing.placements = {{1, 0, 1024, false}, {2, 0, 1024, true}};
    buildBarrierPlan(g, c, defaultBarrierRules());
    const PassBarriers* pb = at(c, 2);
    REQUIRE(pb);
    REQUIRE(pb->barriers.size() == 2);
    CHECK(pb->barriers[0].scope == BarrierScope::Encoder);
    CHECK(pb->barriers[0].aliasing);
    CHECK(pb->barriers[0].afterStages == StageDispatch); // only earlier accesses in this encoder
    CHECK(pb->barriers[0].beforeStages == StageBlit);
    CHECK(pb->barriers[1].scope == BarrierScope::Queue);
    CHECK(pb->barriers[1].aliasing);
    CHECK(pb->barriers[1].afterStages == (StageDispatch | StageBlit));
    CHECK(c.barriers.front().position <= c.barriers.back().position);
}

TEST_CASE("barriers: encoder-scope barrier after fragment inside a render encoder is an error") {
    RenderGraph g;
    TextureRef tex;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("gbuf", PassType::Raster, [&](PassBuilder& b) {
        tex = b.writeColor(b.createTexture("tex", colorDesc()), 0, LoadIntent::Clear);
    }, {});
    g.addPass("light", PassType::Raster, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageFragment);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    // One render encoder that (incorrectly) holds two render groups.
    c.renderGroups = {RenderGroup{0, 0, 64, 64, 1, {}}, RenderGroup{1, 1, 64, 64, 1, {}}};
    c.groupOfPosition = {0, 1};
    c.encoders = {EncoderPlan{PassType::Raster, Queue::Graphics, 0, 1, 0}};
    c.encoderOfPosition = {0, 0};
    buildBarrierPlan(g, c, defaultBarrierRules());
    CHECK_FALSE(c.ok);
    REQUIRE_FALSE(c.errors.empty());
    CHECK(c.errors[0].find("fragment") != std::string::npos);
    CHECK(c.barriers.empty());
}

TEST_CASE("barriers: cross-queue dependencies become queue syncs and no barriers") {
    RenderGraph g;
    BufferRef x, y;
    g.addPass("g0", PassType::Compute, Queue::Graphics, [&](PassBuilder& b) {
        x = b.write(b.createBuffer("x", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("async", PassType::Compute, Queue::AsyncCompute, [&](PassBuilder& b) {
        b.read(x, Usage::ShaderRead, StageDispatch);
        y = b.write(b.createBuffer("y", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("g2", PassType::Compute, Queue::Graphics, [&](PassBuilder& b) {
        b.read(y, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Compute, 1, 1}, {PassType::Compute, 2, 2}});
    CHECK(c.barriers.empty());
    buildQueueSyncs(g, c);
    REQUIRE(c.queueSyncs.size() == 2);
    CHECK(c.queueSyncs[0].signalAfterPosition == 0);
    CHECK(c.queueSyncs[0].waitBeforePosition == 1);
    CHECK(c.queueSyncs[0].value == 1);
    CHECK(c.queueSyncs[1].signalAfterPosition == 1);
    CHECK(c.queueSyncs[1].waitBeforePosition == 2);
    CHECK(c.queueSyncs[1].value == 2);
}

TEST_CASE("barriers: a consumer waits only for the highest value of the other queue") {
    RenderGraph g;
    BufferRef x, y, z;
    g.addPass("g0", PassType::Compute, Queue::Graphics, [&](PassBuilder& b) {
        x = b.write(b.createBuffer("x", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("a1", PassType::Compute, Queue::AsyncCompute, [&](PassBuilder& b) {
        b.read(x, Usage::ShaderRead, StageDispatch);
        y = b.write(b.createBuffer("y", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("a2", PassType::Compute, Queue::AsyncCompute, [&](PassBuilder& b) {
        b.read(x, Usage::ShaderRead, StageDispatch);
        z = b.write(b.createBuffer("z", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("g3", PassType::Compute, Queue::Graphics, [&](PassBuilder& b) {
        b.read(y, Usage::ShaderRead, StageDispatch);
        b.read(z, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Compute, 1, 2}, {PassType::Compute, 3, 3}});
    buildQueueSyncs(g, c);
    REQUIRE(c.queueSyncs.size() == 3);
    CHECK(c.queueSyncs[0].waitBeforePosition == 1);
    CHECK(c.queueSyncs[0].value == 1);
    CHECK(c.queueSyncs[1].waitBeforePosition == 2);
    CHECK(c.queueSyncs[1].value == 1);
    CHECK(c.queueSyncs[2].signalAfterPosition == 2);
    CHECK(c.queueSyncs[2].waitBeforePosition == 3);
    CHECK(c.queueSyncs[2].value == 3);
}

TEST_CASE("barriers: same-queue dependencies produce no queue sync") {
    RenderGraph g;
    BufferRef x;
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        x = b.write(b.createBuffer("x", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("b", PassType::Compute, [&](PassBuilder& b) {
        b.read(x, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    CompiledGraph c = plan(g, {{PassType::Compute, 0, 1}});
    buildQueueSyncs(g, c);
    CHECK(c.queueSyncs.empty());
}

TEST_CASE("barriers: tile stages are promoted (accepted by Metal but ineffective)") {
    RenderGraph g;
    TextureRef tex;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("gen", PassType::Raster, [&](PassBuilder& b) {
        tex = b.createTexture("tex", colorDesc());
        tex = b.write(tex, Usage::ShaderWrite, StageTile); // e.g. a tile shader store
        b.writeColor(b.createTexture("scratch", colorDesc()), 0, LoadIntent::Clear);
    }, {});
    g.addPass("use", PassType::Raster, [&](PassBuilder& b) {
        b.read(tex, Usage::ShaderRead, StageTile);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Raster, 0, 0}, {PassType::Raster, 1, 1}});
    REQUIRE(c.ok);
    REQUIRE(c.barriers.size() == 1);
    const Barrier& b = c.barriers[0].barriers[0];
    CHECK(b.scope == BarrierScope::Queue);
    CHECK(b.afterStages == StageFragment);  // Tile -> Fragment
    CHECK(b.beforeStages == kGeometry);     // Tile -> Vertex|Object|Mesh
}

TEST_CASE("barriers: encoder-scope barrier with raster stages inside a compute encoder is an error") {
    RenderGraph g;
    BufferRef buf;
    g.addPass("a", PassType::Compute, [&](PassBuilder& b) {
        buf = b.write(b.createBuffer("buf", {1024}), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("b", PassType::Compute, [&](PassBuilder& b) {
        b.read(buf, Usage::ShaderRead, StageFragment); // nonsensical for a compute pass
        b.setSideEffect();
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 1}});
    CHECK_FALSE(c.ok);
    REQUIRE_FALSE(c.errors.empty());
    CHECK(c.errors.back().find("compute encoder") != std::string::npos);
}

TEST_CASE("barriers: compute encoders are split at queue sync points (full compile)") {
    // Graphics: seed (0) and an independent pass (1) share a compute run;
    // async reduce (2) needs seed; graphics consume (3) needs reduce.
    RenderGraph g;
    BufferRef seed, reduced;
    g.addPass("seed", PassType::Compute, [&](PassBuilder& b) {
        seed = b.write(b.createBuffer("S", {4096}), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("independent", PassType::Compute, [&](PassBuilder& b) {
        b.write(b.createBuffer("X", {4096}), Usage::ShaderWrite, StageDispatch);
        b.setSideEffect();
    }, {});
    g.addPass("reduce", PassType::Compute, Queue::AsyncCompute, [&](PassBuilder& b) {
        b.read(seed, Usage::ShaderRead, StageDispatch);
        reduced = b.write(b.createBuffer("R", {256}), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("consume", PassType::Compute, [&](PassBuilder& b) {
        b.read(reduced, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, {});
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{0, 1, 2, 3});
    REQUIRE(c.queueSyncs.size() == 2);
    CHECK(c.queueSyncs[0].signalAfterPosition == 0);
    CHECK(c.queueSyncs[0].waitBeforePosition == 2);
    CHECK(c.queueSyncs[1].signalAfterPosition == 2);
    CHECK(c.queueSyncs[1].waitBeforePosition == 3);
    // seed signals after itself: the graphics run [seed, independent] is cut.
    REQUIRE(c.encoders.size() == 4);
    for (u32 i = 0; i < 4; ++i) {
        CHECK(c.encoders[i].firstPosition == i);
        CHECK(c.encoders[i].lastPosition == i);
        CHECK(c.encoderOfPosition[i] == i);
    }
    CHECK(c.encoders[2].queue == Queue::AsyncCompute);
    // Cross-queue dependencies are events, not barriers.
    CHECK(c.barriers.empty());
}

TEST_CASE("barriers: splitEncodersAtQueueSyncs cuts before waits and after signals only") {
    CompiledGraph c;
    c.order = {0, 1, 2, 3, 4};
    c.encoders = {EncoderPlan{PassType::Compute, Queue::Graphics, 0, 4, ~0u}};
    c.encoderOfPosition.assign(5, 0);
    c.queueSyncs = {QueueSync{1, 3, 1}}; // signal after 1, wait before 3 (other queue elsewhere)
    splitEncodersAtQueueSyncs(c);
    REQUIRE(c.encoders.size() == 3);
    CHECK(c.encoders[0].firstPosition == 0);
    CHECK(c.encoders[0].lastPosition == 1);
    CHECK(c.encoders[1].firstPosition == 2);
    CHECK(c.encoders[1].lastPosition == 2);
    CHECK(c.encoders[2].firstPosition == 3);
    CHECK(c.encoders[2].lastPosition == 4);
    CHECK(c.encoderOfPosition == std::vector<u32>{0, 0, 1, 2, 2});
}

TEST_CASE("barriers: a persistent import written every frame waits for the previous frame") {
    // update: compute writes the persistent buffer; draw reads it in vertex.
    // Frame N+1's update must wait for frame N's draw (WAR) and update (WAW).
    RenderGraph g;
    BufferRef persistent = g.importBuffer("persistent", bufDesc(), ImportContentsDefined | ImportOutput);
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("update", PassType::Compute, [&](PassBuilder& b) {
        b.read(persistent, Usage::ShaderRead, StageDispatch);
        persistent = b.write(persistent, Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("draw", PassType::Raster, [&](PassBuilder& b) {
        b.read(persistent, Usage::ShaderRead, StageVertex);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1}});
    REQUIRE(c.barriers.size() == 2);
    // Position 0: previous frame's accesses (dispatch write, vertex read).
    CHECK(c.barriers[0].position == 0);
    REQUIRE(c.barriers[0].barriers.size() == 1);
    CHECK(c.barriers[0].barriers[0].scope == BarrierScope::Queue);
    CHECK(c.barriers[0].barriers[0].afterStages == (StageDispatch | StageVertex));
    CHECK(c.barriers[0].barriers[0].beforeStages == StageDispatch);
    CHECK(c.barriers[0].barriers[0].resources == std::vector<u32>{0});
    // Position 1: the ordinary RAW dependency.
    CHECK(c.barriers[1].position == 1);
    CHECK(c.barriers[1].barriers[0].afterStages == StageDispatch);
    CHECK(c.barriers[1].barriers[0].beforeStages == StageVertex);
}

TEST_CASE("barriers: per-frame and read-only imports need no barrier between frames") {
    RenderGraph g;
    BufferRef perFrame = g.importBuffer("per-frame", bufDesc(), ImportOutput | ImportPerFrame);
    BufferRef readOnly = g.importBuffer("read-only", bufDesc(), ImportContentsDefined);
    g.addPass("pass", PassType::Compute, [&](PassBuilder& b) {
        b.read(readOnly, Usage::ShaderRead, StageDispatch);
        b.write(perFrame, Usage::ShaderWrite, StageDispatch);
    }, {});
    const CompiledGraph c = plan(g, {{PassType::Compute, 0, 0}});
    CHECK(c.barriers.empty());
}

// ---------------------------------------------------------------------------
// OPT-1.4: BarrierPolicy::Minimal
// ---------------------------------------------------------------------------

namespace {

const Barrier* findBarrier(const CompiledGraph& c, u32 position, BarrierScope scope, bool aliasing) {
    const PassBarriers* pb = at(c, position);
    if (!pb) return nullptr;
    for (const Barrier& b : pb->barriers) if (b.scope == scope && b.aliasing == aliasing) return &b;
    return nullptr;
}

struct Pair {
    CompiledGraph conservative, minimal;
};

// Plans `g` with the given encoders and aliasing placements under both policies.
Pair planBoth(const RenderGraph& g, const std::vector<EncSpec>& specs, const std::vector<Placement>& placements) {
    Pair r;
    r.conservative = compileOrder(g);
    REQUIRE(r.conservative.ok);
    assignEncoders(g, r.conservative, specs);
    r.conservative.aliasing.placements = placements;
    r.minimal = r.conservative;
    buildBarrierPlan(g, r.conservative, defaultBarrierRules(), BarrierPolicy::Conservative);
    buildBarrierPlan(g, r.minimal, defaultBarrierRules(), BarrierPolicy::Minimal);
    return r;
}

// Same barriers at the same places; Minimal waits for a subset of the stages.
// Returns the number of barriers whose afterStages changed.
u32 compareBarriers(const CompiledGraph& cons, const CompiledGraph& mini, std::vector<std::string>* changes = nullptr,
                    const RenderGraph* g = nullptr) {
    REQUIRE(cons.barriers.size() == mini.barriers.size());
    u32 changed = 0;
    for (size_t i = 0; i < cons.barriers.size(); ++i) {
        const PassBarriers& a = cons.barriers[i];
        const PassBarriers& b = mini.barriers[i];
        REQUIRE(a.position == b.position);
        REQUIRE(a.barriers.size() == b.barriers.size());
        for (size_t k = 0; k < a.barriers.size(); ++k) {
            const Barrier& x = a.barriers[k];
            const Barrier& y = b.barriers[k];
            CHECK(x.scope == y.scope);
            CHECK(x.aliasing == y.aliasing);
            CHECK(x.beforeStages == y.beforeStages);
            CHECK(x.resources == y.resources);
            CHECK((y.afterStages & ~x.afterStages) == 0);
            if (y.afterStages != x.afterStages) {
                ++changed;
                if (changes) {
                    std::string name = "pos " + std::to_string(a.position);
                    if (g && a.position < cons.order.size()) name += " (" + g->passes()[cons.order[a.position]].name + ")";
                    changes->push_back(name + (x.scope == BarrierScope::Queue ? " queue " : " encoder ") +
                                       (x.aliasing ? "alias " : "") + stagesName(x.afterStages) + " -> " +
                                       stagesName(y.afterStages));
                }
            }
        }
    }
    return changed;
}

} // namespace

TEST_CASE("barriers minimal: compute to raster alias waits only for the ordered-last raster read") {
    // A (compute) writes t1; B (raster) reads t1 in fragment; C (raster) first
    // uses t2 placed over t1.  A is ordered before B by B's RAW barrier, so C
    // needs no dispatch stage.
    RenderGraph g;
    BufferRef t1;
    TextureRef t2;
    BufferRef out = g.importBuffer("out", bufDesc(), ImportOutput | ImportPerFrame);
    g.addPass("A", PassType::Compute, [&](PassBuilder& b) {
        t1 = b.write(b.createBuffer("t1", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("B", PassType::Raster, [&](PassBuilder& b) {
        b.read(t1, Usage::ShaderRead, StageFragment);
        b.write(out, Usage::ShaderWrite, StageFragment);
    }, {});
    g.addPass("C", PassType::Raster, [&](PassBuilder& b) {
        t2 = b.writeColor(b.createTexture("t2", colorDesc()), 0, LoadIntent::Clear);
        b.setSideEffect();
    }, {});
    const std::vector<EncSpec> specs = {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1}, {PassType::Raster, 2, 2}};
    // ids: out = 0, t1 = 1, t2 = 2
    const Pair r = planBoth(g, specs, {{1, 0, 1024, false}, {2, 0, 1024, true}});
    const Barrier* c = findBarrier(r.conservative, 2, BarrierScope::Queue, true);
    const Barrier* m = findBarrier(r.minimal, 2, BarrierScope::Queue, true);
    REQUIRE(c);
    REQUIRE(m);
    CHECK(c->afterStages == (StageDispatch | StageFragment));
    CHECK(m->afterStages == StageFragment);
    CHECK(m->beforeStages == c->beforeStages);
    CHECK(m->resources == c->resources);
    // t1's own first use: the previous frame's maximal access is B's read.
    const Barrier* m0 = findBarrier(r.minimal, 0, BarrierScope::Queue, false);
    REQUIRE(m0);
    CHECK(m0->afterStages == StageFragment);
    CHECK(compareBarriers(r.conservative, r.minimal) >= 2);
}

TEST_CASE("barriers minimal: unordered accesses of the same memory stay wide") {
    // Two independent readers of t1 (fragment and dispatch) are not ordered
    // with each other: the next occupant waits for both.
    RenderGraph g;
    BufferRef t1;
    TextureRef t2;
    BufferRef o1 = g.importBuffer("o1", bufDesc(), ImportOutput | ImportPerFrame);
    BufferRef o2 = g.importBuffer("o2", bufDesc(), ImportOutput | ImportPerFrame);
    g.addPass("A", PassType::Compute, [&](PassBuilder& b) {
        t1 = b.write(b.createBuffer("t1", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("B", PassType::Raster, [&](PassBuilder& b) {
        b.read(t1, Usage::ShaderRead, StageFragment);
        b.write(o1, Usage::ShaderWrite, StageFragment);
    }, {});
    g.addPass("C", PassType::Compute, [&](PassBuilder& b) {
        b.read(t1, Usage::ShaderRead, StageDispatch);
        b.write(o2, Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("D", PassType::Raster, [&](PassBuilder& b) {
        t2 = b.writeColor(b.createTexture("t2", colorDesc()), 0, LoadIntent::Clear);
        b.setSideEffect();
    }, {});
    const std::vector<EncSpec> specs = {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1},
                                        {PassType::Compute, 2, 2}, {PassType::Raster, 3, 3}};
    // ids: o1 = 0, o2 = 1, t1 = 2, t2 = 3
    const Pair r = planBoth(g, specs, {{2, 0, 1024, false}, {3, 0, 1024, true}});
    const Barrier* m = findBarrier(r.minimal, 3, BarrierScope::Queue, true);
    const Barrier* c = findBarrier(r.conservative, 3, BarrierScope::Queue, true);
    REQUIRE(m);
    REQUIRE(c);
    CHECK(m->afterStages == (StageFragment | StageDispatch));
    CHECK(m->afterStages == c->afterStages);
}

TEST_CASE("barriers minimal: persistent import waits for the maximal accesses of the previous frame") {
    RenderGraph g;
    BufferRef persistent = g.importBuffer("persistent", bufDesc(), ImportContentsDefined | ImportOutput);
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput | ImportPerFrame);
    g.addPass("update", PassType::Compute, [&](PassBuilder& b) {
        b.read(persistent, Usage::ShaderRead, StageDispatch);
        persistent = b.write(persistent, Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("draw", PassType::Raster, [&](PassBuilder& b) {
        b.read(persistent, Usage::ShaderRead, StageVertex);
        b.writeColor(target, 0, LoadIntent::Clear);
    }, {});
    const Pair r = planBoth(g, {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1}}, {});
    const Barrier* c = findBarrier(r.conservative, 0, BarrierScope::Queue, false);
    const Barrier* m = findBarrier(r.minimal, 0, BarrierScope::Queue, false);
    REQUIRE(c);
    REQUIRE(m);
    CHECK(c->afterStages == (StageDispatch | StageVertex));
    CHECK(m->afterStages == StageVertex);
    // The dependency barrier is untouched.
    const Barrier* d = findBarrier(r.minimal, 1, BarrierScope::Queue, false);
    REQUIRE(d);
    CHECK(d->afterStages == StageDispatch);
    CHECK(d->beforeStages == StageVertex);
    CHECK(compareBarriers(r.conservative, r.minimal) == 1);
}

TEST_CASE("barriers minimal: memory shared with another queue keeps the conservative stages") {
    RenderGraph g;
    BufferRef t1;
    TextureRef t2;
    BufferRef out = g.importBuffer("out", bufDesc(), ImportOutput | ImportPerFrame);
    g.addPass("A", PassType::Compute, [&](PassBuilder& b) {
        t1 = b.write(b.createBuffer("t1", bufDesc()), Usage::ShaderWrite, StageDispatch);
    }, {});
    g.addPass("B", PassType::Raster, [&](PassBuilder& b) {
        b.read(t1, Usage::ShaderRead, StageFragment);
        b.write(out, Usage::ShaderWrite, StageFragment);
    }, {});
    g.addPass("C", PassType::Raster, [&](PassBuilder& b) {
        t2 = b.writeColor(b.createTexture("t2", colorDesc()), 0, LoadIntent::Clear);
        b.setSideEffect();
    }, {});
    g.passes()[0].queue = Queue::AsyncCompute;
    const Pair r = planBoth(g, {{PassType::Compute, 0, 0}, {PassType::Raster, 1, 1}, {PassType::Raster, 2, 2}},
                            {{1, 0, 1024, false}, {2, 0, 1024, true}});
    CHECK(compareBarriers(r.conservative, r.minimal) == 0);
}

namespace {

struct Lcg {
    u64 state;
    u32 next() { state = state * 6364136223846793005ull + 1442695040888963407ull; return static_cast<u32>(state >> 33); }
    u32 below(u32 n) { return next() % n; }
};

// Deterministic random graphs: raster / compute / blit passes creating
// transients and reading or rewriting earlier ones.
void buildRandomGraph(RenderGraph& g, u64 seed, u32 passCount) {
    Lcg rng{seed};
    std::vector<BufferRef> bufs;
    std::vector<TextureRef> texs;
    BufferRef persistent = g.importBuffer("persistent", bufDesc(), ImportContentsDefined | ImportOutput);
    for (u32 i = 0; i < passCount; ++i) {
        const u32 kind = rng.below(3);
        const PassType type = kind == 0 ? PassType::Raster : kind == 1 ? PassType::Compute : PassType::Blit;
        const Stages readStage = type == PassType::Raster ? (rng.below(2) ? StageFragment : StageVertex)
                                 : type == PassType::Compute ? StageDispatch : StageBlit;
        const u32 reads = rng.below(3);
        const u32 pickB = bufs.empty() ? 0 : rng.below(static_cast<u32>(bufs.size()));
        const u32 pickT = texs.empty() ? 0 : rng.below(static_cast<u32>(texs.size()));
        const bool rewrite = !bufs.empty() && type != PassType::Raster && rng.below(4) == 0;
        const bool usePersistent = rng.below(5) == 0;
        const bool big = rng.below(3) == 0;
        g.addPass("p" + std::to_string(i), type, [&](PassBuilder& b) {
            for (u32 k = 0; k < reads; ++k) {
                if (k == 0 && !bufs.empty() && !(rewrite)) b.read(bufs[pickB], Usage::ShaderRead, readStage);
                else if (k == 1 && !texs.empty() && type != PassType::Blit) {
                    b.read(texs[pickT], Usage::ShaderRead, type == PassType::Compute ? StageDispatch : StageFragment);
                }
            }
            if (usePersistent && type == PassType::Compute) {
                persistent = b.write(persistent, Usage::ShaderWrite, StageDispatch);
            } else if (usePersistent) {
                b.read(persistent, Usage::ShaderRead, readStage);
            }
            if (rewrite) {
                bufs[pickB] = b.write(bufs[pickB], Usage::ShaderWrite, type == PassType::Compute ? StageDispatch : StageBlit);
            }
            if (type == PassType::Raster) {
                TextureDesc d = colorDesc();
                if (big) d.width = 256;
                texs.push_back(b.writeColor(b.createTexture("t" + std::to_string(i), d), 0, LoadIntent::Clear));
            } else {
                BufferDesc d = bufDesc();
                if (big) d.size = 8192;
                bufs.push_back(b.write(b.createBuffer("b" + std::to_string(i), d), Usage::ShaderWrite,
                                       type == PassType::Compute ? StageDispatch : StageBlit));
            }
            b.setSideEffect();
        }, {});
    }
}

} // namespace

TEST_CASE("barriers minimal: random graphs, stages are a subset of conservative") {
    EstimatedSizer sizer;
    u32 graphs = 0, changed = 0, total = 0;
    for (u64 seed = 1; seed <= 300; ++seed) {
        RenderGraph g;
        buildRandomGraph(g, seed * 7919, 4 + static_cast<u32>(seed % 14));
        CompileOptions a;
        a.sizer = &sizer;
        CompileOptions m = a;
        m.barrierPolicy = BarrierPolicy::Minimal;
        const CompiledGraph ca = compile(g, a);
        const CompiledGraph cm = compile(g, m);
        if (!ca.ok) continue;
        REQUIRE(cm.ok);
        ++graphs;
        for (const PassBarriers& pb : ca.barriers) total += static_cast<u32>(pb.barriers.size());
        changed += compareBarriers(ca, cm);
    }
    CHECK(graphs > 150);
    std::printf("  random graphs: %u graphs, %u barriers, %u narrowed\n", graphs, total, changed);
    CHECK(changed > 0);
}

TEST_CASE("barriers minimal: OPT-1 scenarios, 1 to 3 views") {
    EstimatedSizer sizer;
    for (u32 index = 0; index < scenarioCount(); ++index) {
        for (u32 views = 1; views <= 3; ++views) {
            RenderGraph g;
            const TextureRef drawable = g.importTexture("Drawable", {Format::BGRA8Srgb, 1920, 1080}, ImportOutput | ImportPerFrame);
            Scenario s;
            ScenarioParams params;
            params.views = views;
            std::string error;
            REQUIRE_MESSAGE(buildScenario(index, params, g, drawable, s, {}, &error), error);
            CompileOptions a;
            a.sizer = &sizer;
            CompileOptions m = a;
            m.barrierPolicy = BarrierPolicy::Minimal;
            const CompiledGraph ca = compile(g, a);
            const CompiledGraph cm = compile(g, m);
            REQUIRE(ca.ok);
            REQUIRE(cm.ok);
            std::vector<std::string> changes;
            u32 total = 0;
            for (const PassBarriers& pb : ca.barriers) total += static_cast<u32>(pb.barriers.size());
            const u32 changed = compareBarriers(ca, cm, &changes, &g);
            std::printf("  scenario %u (%s) views %u: %u of %u barriers narrowed\n", index, scenarioName(index), views,
                        changed, total);
            if (views == 1) for (const std::string& c : changes) std::printf("      %s\n", c.c_str());
        }
    }
}
