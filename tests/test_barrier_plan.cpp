#include "rendergraph/barrier_plan.h"
#include "rendergraph/render_graph.h"

#include <doctest/doctest.h>

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
    CHECK(r.rasterUnsupportedBefore == (StageFragment | StageTile));
    CHECK(r.rasterPromoteTo == StageGeometry);
    CHECK(r.rasterForbiddenEncoderAfter == (StageFragment | StageTile));
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

TEST_CASE("barriers: compute to raster fragment read is promoted at group start") {
    RenderGraph g;
    TextureRef tex;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
    CHECK(b.beforeStages == kGeometry);
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
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
    CHECK(b.beforeStages == kGeometry);
    CHECK(c.ok);
}

TEST_CASE("barriers: custom rules keep the consumer stages") {
    RenderGraph g;
    TextureRef shadow;
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
    CHECK(b.beforeStages == kGeometry);
}

TEST_CASE("barriers: WAR uses the reader's stages on the after side") {
    RenderGraph g;
    BufferRef buf = g.importBuffer("buf", bufDesc(), ImportContentsDefined);
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
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
    CHECK(c.barriers[0].barriers[0].beforeStages == kGeometry);
    CHECK(c.barriers[0].barriers[0].resources == std::vector<u32>{1, 2}); // target = 0
}

TEST_CASE("barriers: aliased first use gets an aliasing barrier with the union of stages") {
    RenderGraph g;
    BufferRef t1;
    TextureRef t2;
    BufferRef out = g.importBuffer("out", bufDesc(), ImportOutput);
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
    CHECK(b.beforeStages == kGeometry);
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
    CHECK(b.beforeStages == kGeometry);
    CHECK(b.resources == std::vector<u32>{0});
    // Plus the dependency barrier at position 1.
    REQUIRE(at(c, 1));
    CHECK(c.barriers.size() == 2);
}

TEST_CASE("barriers: aliasing inside a compute encoder adds an encoder-scope barrier first") {
    RenderGraph g;
    BufferRef t1, t3;
    BufferRef out = g.importBuffer("out", bufDesc(), ImportOutput);
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
    TextureRef target = g.importTexture("target", colorDesc(), ImportOutput);
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
