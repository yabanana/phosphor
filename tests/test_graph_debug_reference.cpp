#include "rendergraph/graph_debug_reference.h"
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

} // namespace

TEST_CASE("graph debug chain: the greedy plan shares memory and nothing is culled") {
    RenderGraph g;
    DebugChainRefs refs;
    addDebugChain(g, refs, DebugChainExec{});
    for (const std::string& e : g.errors()) MESSAGE(e);
    REQUIRE(g.errors().empty());

    FakeSizer sizer;
    CompileOptions options;
    options.sizer = &sizer;
    const CompiledGraph c = compile(g, options);
    REQUIRE(c.ok);
    for (const bool culled : c.culled) CHECK_FALSE(culled);
    CHECK(c.order.size() == 5);

    // Compute, compute, compute, raster, compute: three encoders' worth of
    // crossings (compute run, render pass, compute).
    CHECK(c.encoders.size() == 3);
    CHECK(c.renderGroups.size() == 1);

    // D is read by a later pass, so it is a real (non-memoryless) transient.
    CHECK_FALSE(c.memoryless[refs.d.resource]);

    const DebugAliasSummary s = summarizeAliasing(c);
    CHECK(s.aliasedFlags >= 1);
    CHECK_FALSE(s.sharedPairs.empty());
    CHECK(s.heapSize < s.unaliasedSize);

    // A is dead when C is written: they must be able to share.
    bool acShare = false;
    for (const auto& [a, b] : s.sharedPairs) {
        if ((a == refs.a.resource && b == refs.c.resource) || (a == refs.c.resource && b == refs.a.resource)) {
            acShare = true;
        }
    }
    CHECK(acShare);
}

TEST_CASE("graph debug reference: exact and sensitive to the frame and to single values") {
    std::vector<u32> expected;
    expectedReadback(7, expected);
    REQUIRE(expected.size() == kDebugSize * kDebugSize);
    CHECK(countMismatches(7, expected.data()) == 0);
    CHECK(countMismatches(8, expected.data()) > 0); // another frame's data is detected

    expected[12345] ^= 1u; // one flipped bit
    CHECK(countMismatches(7, expected.data()) == 1);

    // Row sums propagate: changing one input value changes the whole row.
    CHECK(expandValue(1, 3, 4) != expandValue(2, 3, 4));
    CHECK(fillValue(1, 2, 3) != fillValue(1, 2, 4));
}
