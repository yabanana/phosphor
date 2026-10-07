#include "rendergraph/async_probe_reference.h"
#include "rendergraph/graph_dump.h"
#include "rendergraph/render_graph.h"
#include "rendergraph/timing_plan.h"

#include <doctest/doctest.h>

#include <numeric>
#include <string>
#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

constexpr u32 kW = 640;
constexpr u32 kH = 360;

u64 sumTraffic(const std::vector<PassTraffic>& t) {
    u64 total = 0;
    for (const PassTraffic& p : t) total += p.total();
    return total;
}

void checkSumMatches(const RenderGraph& g, const CompiledGraph& c) {
    REQUIRE(c.ok);
    const std::vector<PassTraffic> per = estimatePassBandwidth(g, c);
    REQUIRE(per.size() == g.passes().size());
    const BandwidthReport whole = estimateBandwidth(g, c);
    CHECK(sumTraffic(per) == whole.totalBytes());
    u64 reads = 0, writes = 0;
    for (const PassTraffic& p : per) {
        reads += p.readBytes;
        writes += p.writeBytes;
    }
    CHECK(reads == whole.totalReadBytes);
    CHECK(writes == whole.totalWriteBytes);
}

// Engine-like frame: forward (clear drawable + depth), overlay (preserve
// drawable), optional capture blit of the drawable into a readback buffer.
struct EngineLike {
    RenderGraph graph;
    TextureRef  drawable, depth;
    u32         forward = 0, overlay = 0, capture = ~0u;

    explicit EngineLike(bool withCapture) {
        drawable = graph.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
        forward = graph.addPass("Forward", PassType::Raster,
                                [&](PassBuilder& b) {
                                    depth    = b.createTexture("depth", {Format::Depth32Float, kW, kH});
                                    drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
                                    depth    = b.writeDepth(depth, LoadIntent::Clear);
                                },
                                nullptr);
        overlay = graph.addPass("Overlay", PassType::Raster,
                                [&](PassBuilder& b) {
                                    drawable = b.writeColor(drawable, 0, LoadIntent::Preserve);
                                    b.setSideEffect();
                                },
                                nullptr);
        if (withCapture) {
            BufferRef readback = graph.importBuffer("readback", {u64(kW) * kH * 4}, ImportOutput | ImportPerFrame);
            capture = graph.addPass("Capture", PassType::Blit,
                                    [&](PassBuilder& b) {
                                        b.read(drawable, Usage::CopySrc, StageBlit);
                                        readback = b.write(readback, Usage::CopyDst, StageBlit);
                                        b.setSideEffect();
                                    },
                                    nullptr);
        }
    }
};

} // namespace

TEST_CASE("timing plan: empty graph") {
    RenderGraph g;
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    const TimingPlan plan = buildTimingPlan(g, c, 4);
    CHECK(plan.units.empty());
    CHECK(plan.unitOfPosition.empty());
    CHECK(plan.commitStartQueries == 4);
    CHECK(plan.queriesPerFrame() == 4);
    CHECK(estimatePassBandwidth(g, c).empty());
}

TEST_CASE("timing plan: fused engine frame is one render unit") {
    EngineLike f(false);
    const CompiledGraph c = compile(f.graph);
    REQUIRE(c.ok);
    REQUIRE(c.encoders.size() == 1);
    const TimingPlan plan = buildTimingPlan(f.graph, c, 8);
    REQUIRE(plan.units.size() == 1);
    const TimedUnit& u = plan.units[0];
    CHECK(u.name == "Forward + Overlay");
    CHECK(u.fused);
    CHECK(u.kind == TimestampKind::RenderEnd);
    CHECK(u.queue == Queue::Graphics);
    CHECK(u.encoder == 0);
    CHECK(u.firstPosition == 0);
    CHECK(u.lastPosition == 1);
    CHECK(u.passes == std::vector<u32>{f.forward, f.overlay});
    CHECK(plan.unitOfPosition == std::vector<u32>{0, 0});
    CHECK(plan.queriesPerFrame() == 1 + 8);
    CHECK(u.dramBytes == estimateBandwidth(f.graph, c).totalBytes());
    CHECK(u.dramBytes > 0);
}

TEST_CASE("timing plan: unfused engine frame has one unit per pass") {
    EngineLike f(false);
    CompileOptions o;
    o.fuseRasterPasses = false;
    const CompiledGraph c = compile(f.graph, o);
    REQUIRE(c.ok);
    const TimingPlan plan = buildTimingPlan(f.graph, c, 2);
    REQUIRE(plan.units.size() == 2);
    CHECK(plan.units[0].name == "Forward");
    CHECK(plan.units[1].name == "Overlay");
    CHECK_FALSE(plan.units[0].fused);
    CHECK_FALSE(plan.units[1].fused);
    CHECK(plan.units[0].kind == TimestampKind::RenderEnd);
    CHECK(plan.unitOfPosition == std::vector<u32>{0, 1});
    // Unfused, the drawable round-trips DRAM: more traffic than fused.
    const CompiledGraph fused = compile(f.graph);
    CHECK(plan.units[0].dramBytes + plan.units[1].dramBytes > estimateBandwidth(f.graph, fused).totalBytes());
}

TEST_CASE("timing plan: capture blit is a ComputeEnd unit after the render unit") {
    EngineLike f(true);
    const CompiledGraph c = compile(f.graph);
    REQUIRE(c.ok);
    REQUIRE(c.encoders.size() == 2);
    const TimingPlan plan = buildTimingPlan(f.graph, c, 4);
    REQUIRE(plan.units.size() == 2);
    CHECK(plan.units[0].kind == TimestampKind::RenderEnd);
    CHECK(plan.units[1].kind == TimestampKind::ComputeEnd);
    CHECK(plan.units[1].name == "Capture");
    CHECK(plan.units[1].encoder == 1);
    CHECK_FALSE(plan.units[1].fused);
    CHECK(plan.units[1].dramBytes == u64(kW) * kH * 4 * 2); // read the drawable, write the readback
    CHECK(plan.unitOfPosition == std::vector<u32>{0, 0, 1});
    u64 total = 0;
    for (const TimedUnit& u : plan.units) total += u.dramBytes;
    CHECK(total == estimateBandwidth(f.graph, c).totalBytes());
}

TEST_CASE("timing plan: compute encoder with several passes has one unit per pass") {
    RenderGraph g;
    BufferRef a = g.importBuffer("in", {4096}, ImportContentsDefined);
    BufferRef out = g.importBuffer("out", {4096}, ImportOutput);
    BufferRef mid;
    const u32 p0 = g.addPass("K0", PassType::Compute,
                             [&](PassBuilder& b) {
                                 b.read(a, Usage::ShaderRead, StageDispatch);
                                 mid = b.createBuffer("mid", {4096});
                                 mid = b.write(mid, Usage::ShaderWrite, StageDispatch);
                             },
                             nullptr);
    const u32 p1 = g.addPass("K1", PassType::Compute,
                             [&](PassBuilder& b) {
                                 b.read(mid, Usage::ShaderRead, StageDispatch);
                                 out = b.write(out, Usage::ShaderWrite, StageDispatch);
                             },
                             nullptr);
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    REQUIRE(c.encoders.size() == 1);
    const TimingPlan plan = buildTimingPlan(g, c, 1);
    REQUIRE(plan.units.size() == 2);
    CHECK(plan.units[0].kind == TimestampKind::ComputePassEnd);
    CHECK(plan.units[1].kind == TimestampKind::ComputePassEnd);
    CHECK(plan.units[0].name == "K0");
    CHECK(plan.units[1].name == "K1");
    CHECK(plan.units[0].passes == std::vector<u32>{p0});
    CHECK(plan.units[1].passes == std::vector<u32>{p1});
    CHECK(plan.units[0].encoder == 0);
    CHECK(plan.units[1].encoder == 0);
    CHECK_FALSE(plan.units[0].fused);
    CHECK(plan.unitOfPosition == std::vector<u32>{0, 1});
    CHECK(plan.units[0].dramBytes == 8192);
    CHECK(plan.units[1].dramBytes == 8192);
    checkSumMatches(g, c);
}

TEST_CASE("timing plan: culled passes are not covered and cost nothing") {
    RenderGraph g;
    TextureRef drawable = g.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
    const u32 dead = g.addPass("Dead", PassType::Compute,
                               [&](PassBuilder& b) {
                                   BufferRef t = b.createBuffer("unused", {1024});
                                   b.write(t, Usage::ShaderWrite, StageDispatch);
                               },
                               nullptr);
    const u32 live = g.addPass("Live", PassType::Raster,
                               [&](PassBuilder& b) { drawable = b.writeColor(drawable, 0, LoadIntent::Clear); },
                               nullptr);
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    REQUIRE(c.culled[dead]);
    const TimingPlan plan = buildTimingPlan(g, c, 1);
    REQUIRE(plan.units.size() == 1);
    CHECK(plan.units[0].name == "Live");
    CHECK(plan.units[0].passes == std::vector<u32>{live});
    CHECK_FALSE(plan.units[0].fused); // a single live pass is not fused
    CHECK(plan.unitOfPosition == std::vector<u32>{0});
    const std::vector<PassTraffic> per = estimatePassBandwidth(g, c);
    CHECK(per[dead].total() == 0);
    CHECK(per[live].writeBytes == u64(kW) * kH * 4);
    checkSumMatches(g, c);
}

TEST_CASE("timing plan: async probe graph alternates queues") {
    RenderGraph g;
    AsyncProbeRefs refs;
    TextureRef drawable = g.importTexture("Drawable", {Format::BGRA8Srgb, kW, kH}, ImportOutput);
    addAsyncProbeChain(g, refs, AsyncProbeExec{});
    g.addPass("Forward", PassType::Raster,
              [&](PassBuilder& b) {
                  TextureRef depth = b.createTexture("Depth", {Format::Depth32Float, kW, kH});
                  drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
                  b.writeDepth(depth, LoadIntent::Clear);
              },
              nullptr);
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    const TimingPlan plan = buildTimingPlan(g, c, 4);
    REQUIRE(plan.units.size() == c.order.size()); // every encoder holds one pass
    REQUIRE(plan.units.size() == c.encoders.size());
    bool sawAsync = false;
    for (size_t i = 0; i < plan.units.size(); ++i) {
        const TimedUnit& u = plan.units[i];
        CHECK(u.queue == c.encoders[i].queue);
        CHECK(u.passes.size() == 1);
        CHECK(plan.unitOfPosition[i] == i);
        CHECK(u.kind == (c.encoders[i].type == PassType::Raster ? TimestampKind::RenderEnd : TimestampKind::ComputeEnd));
        if (u.name == "Async reduce") {
            CHECK(u.queue == Queue::AsyncCompute);
            sawAsync = true;
        }
    }
    CHECK(sawAsync);
    checkSumMatches(g, c);
}

TEST_CASE("timing plan: split parallelChunks pass stays one unit") {
    RenderGraph g;
    TextureRef drawable = g.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
    const u32 split = g.addPass("Split", PassType::Raster,
                                [&](PassBuilder& b) {
                                    drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
                                    b.setParallelChunks(4);
                                },
                                nullptr);
    const CompiledGraph c = compile(g);
    REQUIRE(c.ok);
    const TimingPlan plan = buildTimingPlan(g, c, 1);
    REQUIRE(plan.units.size() == 1);
    CHECK(plan.units[0].passes == std::vector<u32>{split});
    CHECK(plan.units[0].kind == TimestampKind::RenderEnd);
    CHECK_FALSE(plan.units[0].fused);
    checkSumMatches(g, c);
}

TEST_CASE("per-pass bandwidth: sum equals estimateBandwidth on several graphs") {
    SUBCASE("engine frame, fused") {
        EngineLike f(false);
        checkSumMatches(f.graph, compile(f.graph));
    }
    SUBCASE("engine frame, unfused") {
        EngineLike f(false);
        CompileOptions o;
        o.fuseRasterPasses = false;
        checkSumMatches(f.graph, compile(f.graph, o));
    }
    SUBCASE("engine frame with capture") {
        EngineLike f(true);
        checkSumMatches(f.graph, compile(f.graph));
    }
    SUBCASE("raster chain sampling an intermediate target") {
        RenderGraph g;
        TextureRef drawable = g.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
        TextureRef scene;
        g.addPass("Scene", PassType::Raster,
                  [&](PassBuilder& b) {
                      scene = b.createTexture("scene", {Format::RGBA16Float, kW, kH});
                      scene = b.writeColor(scene, 0, LoadIntent::Clear);
                  },
                  nullptr);
        g.addPass("Post", PassType::Raster,
                  [&](PassBuilder& b) {
                      b.read(scene, Usage::ShaderRead, StageFragment);
                      drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
                  },
                  nullptr);
        const CompiledGraph c = compile(g);
        checkSumMatches(g, c);
        const std::vector<PassTraffic> per = estimatePassBandwidth(g, c);
        CHECK(per[0].writeBytes == u64(kW) * kH * 8);              // scene stored
        CHECK(per[1].readBytes == u64(kW) * kH * 8);               // scene sampled
        CHECK(per[1].writeBytes == u64(kW) * kH * 4);              // drawable stored
    }
    SUBCASE("without group information (compileOrder only)") {
        EngineLike f(true);
        const CompiledGraph c = compileOrder(f.graph);
        checkSumMatches(f.graph, c);
    }
}

TEST_CASE("per-pass bandwidth: fused group charges load to first toucher and store to last") {
    EngineLike f(false);
    const CompiledGraph c = compile(f.graph);
    REQUIRE(c.ok);
    const std::vector<PassTraffic> per = estimatePassBandwidth(f.graph, c);
    // Drawable cleared in forward, preserved in tile memory by the overlay,
    // stored once at the end of the group: the store belongs to the overlay.
    CHECK(per[f.forward].total() == 0);
    CHECK(per[f.overlay].writeBytes == u64(kW) * kH * 4);
    CHECK(per[f.overlay].readBytes == 0);
}

TEST_CASE("timing stages preserve AS build and refit without reclassifying compute units") {
    RenderGraph graph;
    auto structure = graph.importAccelerationStructure("typed AS", 4096, ImportContentsDefined | ImportOutput);
    const auto vertices = graph.importBuffer("vertices", {4096}, ImportContentsDefined);
    const auto control = graph.importBuffer("control", {4}, ImportContentsDefined);
    auto result = graph.importBuffer("result", {128}, ImportOutput);
    const u32 build = graph.addPass("not a stage hint", PassType::Compute, [&](PassBuilder& b) {
        b.read(control, Usage::ShaderRead, StageDispatch);
        b.read(vertices, Usage::ShaderRead, StageAccelerationStructure);
        structure = b.write(structure, Usage::ShaderWrite, StageAccelerationStructure);
    }, nullptr);
    const u32 refit = graph.addPass("another arbitrary name", PassType::Compute, [&](PassBuilder& b) {
        b.read(vertices, Usage::ShaderRead, StageAccelerationStructure);
        b.read(structure, Usage::ShaderRead, StageAccelerationStructure);
        structure = b.write(structure, Usage::ShaderWrite, StageAccelerationStructure);
    }, nullptr);
    const u32 trace = graph.addPass("RT TLAS misleading consumer name", PassType::Compute, [&](PassBuilder& b) {
        b.read(structure, Usage::ShaderRead, StageDispatch);
        result = b.write(result, Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    const auto compiled = compile(graph);
    REQUIRE(compiled.ok);
    const auto plan = buildTimingPlan(graph, compiled, 1);
    REQUIRE(plan.units.size() == 3);
    CHECK(plan.units[0].passes == std::vector<u32>{build});
    CHECK(plan.units[0].stages == (StageDispatch | StageAccelerationStructure));
    CHECK(plan.units[0].needsAccelerationStructureAnchor);
    CHECK(plan.units[1].passes == std::vector<u32>{refit});
    CHECK(plan.units[1].stages == StageAccelerationStructure);
    CHECK(plan.units[1].needsAccelerationStructureAnchor);
    CHECK(plan.units[2].passes == std::vector<u32>{trace});
    CHECK(plan.units[2].stages == StageDispatch); // Typed AS does not imply AS-stage work.
    CHECK_FALSE(plan.units[2].needsAccelerationStructureAnchor);
    for (const auto& unit : plan.units) {
        CHECK(unit.kind == TimestampKind::ComputePassEnd);
        CHECK(unit.encoder == 0);
    }
}

TEST_CASE("timing stages include reads of an imported AS in a single compute unit") {
    RenderGraph graph;
    const auto source = graph.importAccelerationStructure("source", 4096, ImportContentsDefined);
    auto destination = graph.importAccelerationStructure("destination", 2048, ImportOutput);
    graph.addPass("Copy compact", PassType::Compute, [&](PassBuilder& b) {
        b.read(source, Usage::ShaderRead, StageAccelerationStructure);
        destination = b.write(destination, Usage::ShaderWrite, StageAccelerationStructure);
    }, nullptr);
    const auto compiled = compile(graph);
    REQUIRE(compiled.ok);
    const auto plan = buildTimingPlan(graph, compiled, 1);
    REQUIRE(plan.units.size() == 1);
    CHECK(plan.units[0].kind == TimestampKind::ComputeEnd);
    CHECK(plan.units[0].stages == StageAccelerationStructure);
    CHECK(plan.units[0].needsAccelerationStructureAnchor);
}

TEST_CASE("timing stages unite every fused raster member and exclude culled AS work") {
    RenderGraph graph;
    auto target = graph.importTexture("target", {Format::BGRA8Unorm,kW,kH}, ImportOutput);
    const auto vertices = graph.importBuffer("vertices", {128}, ImportContentsDefined);
    const auto meshlets = graph.importBuffer("meshlets", {128}, ImportContentsDefined);
    auto unused = graph.importAccelerationStructure("unused AS", 2048, ImportContentsDefined);
    const auto dead = graph.addPass("discarded AS", PassType::Compute, [&](PassBuilder& b) {
        unused = b.write(unused, Usage::ShaderWrite, StageAccelerationStructure);
    }, nullptr);
    const auto forward = graph.addPass("Vertex draw", PassType::Raster, [&](PassBuilder& b) {
        b.read(vertices, Usage::ShaderRead, StageVertex);
        target = b.writeColor(target, 0, LoadIntent::Clear);
    }, nullptr);
    const auto overlay = graph.addPass("Mesh overlay", PassType::Raster, [&](PassBuilder& b) {
        b.read(meshlets, Usage::ShaderRead, StageObject | StageMesh);
        target = b.writeColor(target, 0, LoadIntent::Preserve);
    }, nullptr);
    const auto compiled = compile(graph);
    REQUIRE(compiled.ok);
    REQUIRE(compiled.culled[dead]);
    const auto plan = buildTimingPlan(graph, compiled, 1);
    REQUIRE(plan.units.size() == 1);
    const auto& unit = plan.units[0];
    CHECK(unit.fused);
    CHECK(unit.passes == std::vector<u32>{forward,overlay});
    CHECK(unit.kind == TimestampKind::RenderEnd);
    CHECK(unit.stages == (StageVertex | StageObject | StageMesh | StageFragment));
    CHECK((unit.stages & StageAccelerationStructure) == 0);
    CHECK_FALSE(unit.needsAccelerationStructureAnchor);
}

TEST_CASE("timing stages do not infer GPU work from an encoder kind or name") {
    RenderGraph graph;
    graph.addPass("RT BLAS maintenance", PassType::Compute, [](PassBuilder& b) { b.setSideEffect(); }, nullptr);
    const auto compiled = compile(graph);
    REQUIRE(compiled.ok);
    const auto plan = buildTimingPlan(graph, compiled, 1);
    REQUIRE(plan.units.size() == 1);
    CHECK(plan.units[0].kind == TimestampKind::ComputeEnd);
    CHECK(plan.units[0].stages == StageNone);
    CHECK_FALSE(plan.units[0].needsAccelerationStructureAnchor);
}


TEST_CASE("timing stages exclude external framework work from compute AS anchors") {
    RenderGraph graph;
    const auto input = graph.importTexture("framework input", {Format::RGBA16Float,kW,kH}, ImportContentsDefined);
    auto output = graph.importTexture("framework output", {Format::RGBA16Float,kW,kH}, ImportOutput);
    graph.addPass("MetalFX-style framework", PassType::External, [&](PassBuilder& b) {
        b.read(input, Usage::ShaderRead, StageExternal);
        output = b.write(output, Usage::ShaderWrite, StageExternal);
    }, nullptr);
    const auto compiled = compile(graph);
    REQUIRE(compiled.ok);
    const auto plan = buildTimingPlan(graph, compiled, 1);
    REQUIRE(plan.units.size() == 1);
    CHECK(plan.units[0].kind == TimestampKind::ComputeEnd);
    CHECK(plan.units[0].stages == StageExternal);
    CHECK((plan.units[0].stages & StageAccelerationStructure) != 0);
    CHECK_FALSE(plan.units[0].needsAccelerationStructureAnchor);
}
