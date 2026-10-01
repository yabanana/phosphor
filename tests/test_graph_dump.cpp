#include "rendergraph/graph_dump.h"
#include "rendergraph/render_graph.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <string>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

constexpr u32 kW = 1280;
constexpr u32 kH = 720;
constexpr u64 kFrameBytes = u64(kW) * kH * 4;

// The engine frame: forward clears drawable + depth, ImGui preserves the
// drawable.  Resource 0 = drawable, 1 = depth.
struct EngineFrame {
    RenderGraph graph;
    TextureRef  drawable;
    TextureRef  depth;
    u32         forward = 0;
    u32         imgui   = 0;

    EngineFrame() {
        drawable = graph.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
        TextureRef d = drawable;
        forward = graph.addPass("forward", PassType::Raster,
                                [&](PassBuilder& b) {
                                    depth = b.createTexture("depth", {Format::Depth32Float, kW, kH});
                                    d = b.writeColor(d, 0, LoadIntent::Clear);
                                    depth = b.writeDepth(depth, LoadIntent::Clear);
                                },
                                nullptr);
        imgui = graph.addPass("imgui", PassType::Raster,
                              [&](PassBuilder& b) {
                                  d = b.writeColor(d, 0, LoadIntent::Preserve);
                                  b.setSideEffect();
                              },
                              nullptr);
    }
};

AttachmentPlan colorPlan(u32 resource, LoadAction load, StoreAction store) {
    AttachmentPlan a;
    a.resource = resource;
    a.load     = load;
    a.store    = store;
    return a;
}

AttachmentPlan depthPlan(u32 resource) {
    AttachmentPlan a;
    a.resource = resource;
    a.depth    = true;
    a.load     = LoadAction::Clear;
    a.store    = StoreAction::DontCare;
    return a;
}

RenderGroup makeGroup(u32 first, u32 last, std::vector<AttachmentPlan> attachments) {
    RenderGroup g;
    g.firstPosition = first;
    g.lastPosition  = last;
    g.width         = kW;
    g.height        = kH;
    g.attachments   = std::move(attachments);
    return g;
}

CompiledGraph fuseEngineFrame(const EngineFrame& f) {
    CompiledGraph c = compileOrder(f.graph);
    REQUIRE(c.ok);
    REQUIRE(c.order.size() == 2);
    c.renderGroups     = {makeGroup(0, 1,
                                    {colorPlan(f.drawable.resource, LoadAction::Clear, StoreAction::Store),
                                     depthPlan(f.depth.resource)})};
    c.groupOfPosition  = {0, 0};
    c.memoryless.assign(f.graph.resources().size(), false);
    c.memoryless[f.depth.resource] = true;
    return c;
}

size_t count(const std::string& s, char ch) { return size_t(std::count(s.begin(), s.end(), ch)); }

bool contains(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

} // namespace

TEST_CASE("dump: engine frame bandwidth is one drawable write, memoryless depth is free") {
    EngineFrame f;
    const CompiledGraph c = fuseEngineFrame(f);
    const BandwidthReport r = estimateBandwidth(f.graph, c);
    CHECK(r.totalReadBytes == 0);
    CHECK(r.totalWriteBytes == kFrameBytes);
    CHECK(r.totalBytes() == kFrameBytes);
    REQUIRE(r.resources.size() == 2);
    CHECK(r.resources[f.drawable.resource].writeBytes == kFrameBytes);
    CHECK(r.resources[f.depth.resource].readBytes == 0);
    CHECK(r.resources[f.depth.resource].writeBytes == 0);
}

TEST_CASE("dump: without fusion the drawable is written twice and read once") {
    EngineFrame f;
    CompiledGraph c = compileOrder(f.graph);
    REQUIRE(c.ok);
    c.renderGroups = {makeGroup(0, 0,
                                {colorPlan(f.drawable.resource, LoadAction::Clear, StoreAction::Store),
                                 depthPlan(f.depth.resource)}),
                      makeGroup(1, 1, {colorPlan(f.drawable.resource, LoadAction::Load, StoreAction::Store)})};
    c.groupOfPosition = {0, 1};
    c.memoryless.assign(f.graph.resources().size(), false);
    c.memoryless[f.depth.resource] = true;
    const BandwidthReport r = estimateBandwidth(f.graph, c);
    CHECK(r.totalReadBytes == kFrameBytes);
    CHECK(r.totalWriteBytes == 2 * kFrameBytes);
}

TEST_CASE("dump: raster pass without group information falls back to per-pass counting") {
    EngineFrame f;
    CompiledGraph c = compileOrder(f.graph);
    REQUIRE(c.ok);
    c.memoryless.assign(f.graph.resources().size(), false);
    c.memoryless[f.depth.resource] = true;
    // forward: drawable write; imgui: Preserve read + write.  Depth is memoryless.
    const BandwidthReport r = estimateBandwidth(f.graph, c);
    CHECK(r.totalReadBytes == kFrameBytes);
    CHECK(r.totalWriteBytes == 2 * kFrameBytes);
    CHECK(r.resources[f.depth.resource].writeBytes == 0);
}

TEST_CASE("dump: ShaderRead of a 1 MiB buffer counts 1 MiB read, ShaderWrite counts a write") {
    constexpr u64 kMiB = 1024 * 1024;
    RenderGraph g;
    BufferRef buf = g.importBuffer("data", {kMiB}, ImportContentsDefined);
    BufferRef out;
    g.addPass("compute", PassType::Compute,
              [&](PassBuilder& b) {
                  b.read(buf, Usage::ShaderRead, StageDispatch);
                  out = b.createBuffer("out", {kMiB});
                  out = b.write(out, Usage::ShaderWrite, StageDispatch);
                  b.setSideEffect();
              },
              nullptr);
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    const BandwidthReport r = estimateBandwidth(g, c);
    CHECK(r.totalReadBytes == kMiB);
    CHECK(r.totalWriteBytes == kMiB);
    CHECK(r.resources[buf.resource].readBytes == kMiB);
}

TEST_CASE("dump: culled pass contributes nothing") {
    constexpr u64 kMiB = 1024 * 1024;
    RenderGraph g;
    BufferRef in = g.importBuffer("in", {kMiB}, ImportContentsDefined);
    g.addPass("dead", PassType::Compute,
              [&](PassBuilder& b) {
                  b.read(in, Usage::ShaderRead, StageDispatch);
                  BufferRef t = b.createBuffer("unused", {kMiB});
                  b.write(t, Usage::ShaderWrite, StageDispatch);
              },
              nullptr);
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    REQUIRE(c.culled.size() == 1);
    REQUIRE(c.culled[0]);
    const BandwidthReport r = estimateBandwidth(g, c);
    CHECK(r.totalBytes() == 0);
    const std::string dot = dumpGraphviz(g, c);
    CHECK(contains(dot, "culled"));
    CHECK(contains(dot, "dead"));
}

TEST_CASE("dump: graphviz of the engine frame has clusters, flags and totals") {
    EngineFrame f;
    CompiledGraph c = fuseEngineFrame(f);
    Barrier barrier;
    barrier.scope        = BarrierScope::Queue;
    barrier.afterStages  = StageFragment;
    barrier.beforeStages = StageVertex;
    barrier.aliasing     = true;
    c.barriers           = {PassBarriers{0, {barrier}}};
    Placement pl;
    pl.resource = f.depth.resource;
    pl.offset   = 4096;
    pl.size     = 1024;
    pl.aliased  = true;
    c.aliasing.placements    = {pl};
    c.aliasing.heapSize      = 1024 * 1024;
    c.aliasing.unaliasedSize = 2 * 1024 * 1024;

    const std::string dot = dumpGraphviz(f.graph, c);
    CHECK(dot.rfind("digraph RenderGraph {", 0) == 0);
    CHECK(contains(dot, "rankdir=LR"));
    CHECK(contains(dot, "forward"));
    CHECK(contains(dot, "imgui"));
    CHECK(contains(dot, "cluster_group0"));
    CHECK(contains(dot, "render pass 0 1280x720"));
    CHECK_FALSE(contains(dot, "cluster_group1"));
    CHECK(contains(dot, "[memoryless]"));
    CHECK(contains(dot, "[imported]"));
    CHECK(contains(dot, "[aliased @4096]"));
    CHECK(contains(dot, "BGRA8Unorm"));
    CHECK(contains(dot, "load=Clear store=Store"));
    CHECK(contains(dot, "barrier queue fragment->vertex alias"));
    CHECK(contains(dot, "3.516 MiB write"));   // 1280*720*4 bytes
    CHECK(contains(dot, "1.000 MiB aliased"));
    CHECK(contains(dot, "2.000 MiB unaliased"));
    CHECK(contains(dot, "T2 M5 Max 569.0 GB/s [measured]"));      // OPT-1.5 tier lines
    CHECK(contains(dot, "M3 base 100.0 GB/s [NOT measured, external]"));
    CHECK(contains(dot, "lint: no findings"));
    CHECK(contains(dot, "DRAM R 0.000 / W 3.516 MiB"));          // bytes on the pass box
    CHECK(count(dot, '{') == count(dot, '}'));
    CHECK(dot == dumpGraphviz(f.graph, c)); // deterministic
}

TEST_CASE("dump: names are escaped and culled passes are dashed") {
    RenderGraph g;
    BufferRef out;
    g.addPass("say \"hi\" \\ there", PassType::Compute,
              [&](PassBuilder& b) {
                  out = b.createBuffer("buf\"x", {64});
                  out = b.write(out, Usage::ShaderWrite, StageDispatch);
                  b.setSideEffect();
              },
              nullptr);
    g.addPass("orphan", PassType::Compute,
              [&](PassBuilder& b) {
                  BufferRef t = b.createBuffer("t", {64});
                  b.write(t, Usage::ShaderWrite, StageDispatch);
              },
              nullptr);
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    const std::string dot = dumpGraphviz(g, c);
    CHECK(contains(dot, "say \\\"hi\\\" \\\\ there"));
    CHECK(contains(dot, "buf\\\"x@1"));
    CHECK_FALSE(contains(dot, "say \"hi\""));
    CHECK(contains(dot, "style=dashed"));
    CHECK(contains(dot, "culled"));
    CHECK(count(dot, '{') == count(dot, '}'));
}

TEST_CASE("dump: cross-queue syncs are drawn as event edges") {
    RenderGraph g;
    BufferRef s, r;
    const BufferRef out = g.importBuffer("out", {256}, ImportOutput | ImportPerFrame);
    g.addPass("seed", PassType::Compute, [&](PassBuilder& b) {
        s = b.write(b.createBuffer("S", {4096}), Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    g.addPass("reduce", PassType::Compute, Queue::AsyncCompute, [&](PassBuilder& b) {
        b.read(s, Usage::ShaderRead, StageDispatch);
        r = b.write(b.createBuffer("R", {256}), Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    g.addPass("consume", PassType::Compute, [&](PassBuilder& b) {
        b.read(r, Usage::ShaderRead, StageDispatch);
        b.write(out, Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    REQUIRE(c.queueSyncs.size() == 2);
    const std::string dot = dumpGraphviz(g, c);
    CHECK(dot.find("p0 -> p1 [label=\"event 1\"") != std::string::npos);
    CHECK(dot.find("p1 -> p2 [label=\"event 2\"") != std::string::npos);
    CHECK(dot.find("async") != std::string::npos);
    CHECK(dot.find("[imported, per-frame]") != std::string::npos);
}
