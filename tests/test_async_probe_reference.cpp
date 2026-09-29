#include "rendergraph/async_probe_reference.h"
#include "rendergraph/render_graph.h"

#include <doctest/doctest.h>

#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

u64 roundUp(u64 v, u64 a) { return (v + a - 1) / a * a; }

struct FakeSizer final : ResourceSizer {
    SizeAlign textureSize(u32, const TextureDesc& d) const override {
        return {roundUp(d.estimatedBytes(), 16384), 16384};
    }
    SizeAlign bufferSize(u32, const BufferDesc& d) const override { return {roundUp(d.size, 256), 256}; }
};

// The probe as Engine::buildFrameGraph adds it: before a stand-in for the
// Forward pass (raster, imported drawable + a transient depth).
struct ProbeGraph {
    RenderGraph    graph;
    AsyncProbeRefs refs;
    u32            forward = 0;

    ProbeGraph() {
        TextureRef drawable = graph.importTexture("Drawable", {Format::BGRA8Srgb, 640, 480}, ImportOutput);
        addAsyncProbeChain(graph, refs, AsyncProbeExec{});
        forward = graph.addPass(
            "Forward", PassType::Raster,
            [&](PassBuilder& b) {
                TextureRef depth = b.createTexture("Depth", {Format::Depth32Float, 640, 480});
                drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
                b.writeDepth(depth, LoadIntent::Clear);
            },
            nullptr);
    }
};

u32 passIndex(const RenderGraph& g, const char* name) {
    for (u32 i = 0; i < g.passes().size(); ++i) {
        if (g.passes()[i].name == name) return i;
    }
    FAIL("no pass named ", name);
    return ~0u;
}

bool sharesMemory(const CompiledGraph& c, u32 resource) {
    for (const Placement& a : c.aliasing.placements) {
        if (a.resource != resource) continue;
        for (const Placement& b : c.aliasing.placements) {
            if (b.resource == resource) continue;
            if (a.offset < b.offset + b.size && b.offset < a.offset + a.size) return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("async probe chain: two cross-queue syncs, no cull, whole-frame async resources") {
    ProbeGraph p;
    for (const std::string& e : p.graph.errors()) MESSAGE(e);
    REQUIRE(p.graph.errors().empty());

    FakeSizer sizer;
    CompileOptions options;
    options.sizer = &sizer;
    const CompiledGraph c = compile(p.graph, options);
    for (const std::string& e : c.errors) MESSAGE(e);
    REQUIRE(c.ok);
    for (const bool culled : c.culled) CHECK_FALSE(culled);
    REQUIRE(c.order.size() == 4);

    const u32 seed = passIndex(p.graph, "Async seed");
    const u32 reduce = passIndex(p.graph, "Async reduce");
    const u32 consume = passIndex(p.graph, "Async consume");
    CHECK(p.graph.passes()[seed].queue == Queue::Graphics);
    CHECK(p.graph.passes()[reduce].queue == Queue::AsyncCompute);
    CHECK(p.graph.passes()[consume].queue == Queue::Graphics);

    // Declaration order is execution order: seed, reduce, consume, Forward.
    const u32 pSeed = c.position(seed), pReduce = c.position(reduce), pConsume = c.position(consume);
    CHECK(pSeed == 0);
    CHECK(pReduce == 1);
    CHECK(pConsume == 2);
    CHECK(c.position(p.forward) == 3);

    // seed -> reduce, reduce -> consume: one event each.
    REQUIRE(c.queueSyncs.size() == 2);
    CHECK(c.queueSyncs[0].signalAfterPosition == pSeed);
    CHECK(c.queueSyncs[0].waitBeforePosition == pReduce);
    CHECK(c.queueSyncs[0].value == 1);
    CHECK(c.queueSyncs[1].signalAfterPosition == pReduce);
    CHECK(c.queueSyncs[1].waitBeforePosition == pConsume);
    CHECK(c.queueSyncs[1].value == 2);

    // Cross-queue dependencies are events, never same-queue barriers: no
    // barrier is planned before the reduce pass for the S dependency.
    for (const PassBarriers& pb : c.barriers) {
        if (pb.position != pReduce) continue;
        for (const Barrier& b : pb.barriers) {
            for (const u32 r : b.resources) CHECK(r != p.refs.s.resource);
        }
    }

    // The async pass sits in its own compute encoder on the async queue.
    bool asyncEncoder = false;
    for (const EncoderPlan& e : c.encoders) {
        if (e.queue == Queue::AsyncCompute) {
            asyncEncoder = true;
            CHECK(e.type == PassType::Compute);
            CHECK(e.firstPosition == pReduce);
            CHECK(e.lastPosition == pReduce);
        }
    }
    CHECK(asyncEncoder);

    // Async resources live for the whole frame: never aliased with anything.
    bool seenS = false, seenR = false;
    for (const Placement& pl : c.aliasing.placements) {
        seenS = seenS || pl.resource == p.refs.s.resource;
        seenR = seenR || pl.resource == p.refs.r.resource;
        if (pl.resource == p.refs.s.resource || pl.resource == p.refs.r.resource) CHECK_FALSE(pl.aliased);
    }
    CHECK(seenS);
    CHECK(seenR);
    CHECK_FALSE(sharesMemory(c, p.refs.s.resource));
    CHECK_FALSE(sharesMemory(c, p.refs.r.resource));
}

TEST_CASE("async probe reference: exact and sensitive to the frame and to single values") {
    std::vector<u32> expected;
    expectedAsyncReadback(7, expected);
    REQUIRE(expected.size() == kAsyncResultCount);
    CHECK(countAsyncMismatches(7, expected.data()) == 0);
    CHECK(countAsyncMismatches(8, expected.data()) > 0); // another frame's data is detected

    std::vector<u32> stale(kAsyncResultCount, 0); // a consumer that ran before the reduction
    CHECK(countAsyncMismatches(7, stale.data()) > 0);

    expected[200] ^= 1u; // one flipped bit
    CHECK(countAsyncMismatches(7, expected.data()) == 1);

    // Each group of the seed buffer feeds exactly one result.
    std::vector<u32> s(kAsyncGroupSize, 1);
    const u32 base = reduceGroup(s.data());
    s[255] = 2;
    CHECK(reduceGroup(s.data()) != base);
    CHECK(seedValue(1, 2) != seedValue(1, 3));
    CHECK(consumeValue(1, 2, 3) != consumeValue(2, 2, 3));
}
