#include "rendergraph/aliasing.h"
#include "rendergraph/render_graph.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <random>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

u64 roundUp(u64 v, u64 a) { return (v + a - 1) / a * a; }

// size = bytes rounded up to the alignment; textures and buffers configurable.
struct FakeSizer final : ResourceSizer {
    u64 texAlign = 4096;
    u64 bufAlign = 4096;
    SizeAlign textureSize(const TextureDesc& d) const override {
        return {roundUp(d.estimatedBytes(), texAlign), texAlign};
    }
    SizeAlign bufferSize(const BufferDesc& d) const override { return {roundUp(d.size, bufAlign), bufAlign}; }
};

const Placement* find(const AliasingPlan& plan, u32 resource) {
    for (const Placement& p : plan.placements)
        if (p.resource == resource) return &p;
    return nullptr;
}

BufferRef produce(RenderGraph& g, const char* name, u64 size, Queue q = Queue::Graphics) {
    BufferRef out;
    g.addPass(name, PassType::Compute, q, [&](PassBuilder& b) {
        out = b.write(b.createBuffer(name, {size}), Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    return out;
}

void consume(RenderGraph& g, const char* name, BufferRef r, Queue q = Queue::Graphics) {
    g.addPass(name, PassType::Compute, q, [&](PassBuilder& b) {
        b.read(r, Usage::ShaderRead, StageDispatch);
        b.setSideEffect();
    }, nullptr);
}

} // namespace

TEST_CASE("aliasing: disjoint lifetimes share offset 0") {
    RenderGraph g;
    BufferRef a = produce(g, "A", 10000);
    consume(g, "readA", a);
    BufferRef b = produce(g, "B", 8000);
    consume(g, "readB", b);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    REQUIRE_FALSE(c.lifetimes[a.resource].overlaps(c.lifetimes[b.resource]));

    FakeSizer sizer;
    AliasingPlan plan = planAliasing(g, c, sizer);
    REQUIRE(plan.placements.size() == 2);
    const Placement* pa = find(plan, a.resource);
    const Placement* pb = find(plan, b.resource);
    REQUIRE(pa);
    REQUIRE(pb);
    CHECK(pa->offset == 0);
    CHECK(pb->offset == 0);
    CHECK(pa->aliased);
    CHECK(pb->aliased);
    CHECK(plan.heapSize == 12288);
    CHECK(plan.unaliasedSize == 12288 + 8192);
}

TEST_CASE("aliasing: overlapping lifetimes do not intersect") {
    RenderGraph g;
    BufferRef a = produce(g, "A", 5000);
    BufferRef b = produce(g, "B", 9000);
    g.addPass("use", PassType::Compute, [&](PassBuilder& p) {
        p.read(a, Usage::ShaderRead, StageDispatch);
        p.read(b, Usage::ShaderRead, StageDispatch);
        p.setSideEffect();
    }, nullptr);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    FakeSizer sizer;
    AliasingPlan plan = planAliasing(g, c, sizer);
    REQUIRE(plan.placements.size() == 2);
    CHECK_FALSE(rangesIntersect(plan.placements[0], plan.placements[1]));
    CHECK_FALSE(plan.placements[0].aliased);
    CHECK_FALSE(plan.placements[1].aliased);
    CHECK(plan.heapSize == plan.unaliasedSize);
}

TEST_CASE("aliasing: imported, memoryless and unused resources are not placed") {
    RenderGraph g;
    BufferRef imp = g.importBuffer("imp", {4096}, ImportContentsDefined | ImportOutput);
    BufferRef a = produce(g, "A", 4096);
    BufferRef m = produce(g, "M", 4096);
    // Culled chain: nobody consumes it.
    BufferRef dead = produce(g, "dead", 4096);
    g.addPass("use", PassType::Compute, [&](PassBuilder& p) {
        p.read(a, Usage::ShaderRead, StageDispatch);
        p.read(m, Usage::ShaderRead, StageDispatch);
        p.write(imp, Usage::ShaderWrite, StageDispatch);
    }, nullptr);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    c.memoryless.assign(g.resources().size(), false);
    c.memoryless[m.resource] = true;

    FakeSizer sizer;
    AliasingPlan plan = planAliasing(g, c, sizer);
    REQUIRE(plan.placements.size() == 1);
    CHECK(plan.placements[0].resource == a.resource);
    CHECK_FALSE(find(plan, imp.resource));
    CHECK_FALSE(find(plan, m.resource));
    CHECK_FALSE(find(plan, dead.resource));
}

TEST_CASE("aliasing: empty graph gives an empty plan") {
    RenderGraph g;
    CompiledGraph c = compileOrder(g);
    FakeSizer sizer;
    AliasingPlan plan = planAliasing(g, c, sizer);
    CHECK(plan.placements.empty());
    CHECK(plan.heapSize == 0);
    CHECK(plan.unaliasedSize == 0);
}

TEST_CASE("aliasing: mixed alignments are respected") {
    RenderGraph g;
    FakeSizer sizer;
    sizer.bufAlign = 4096;
    sizer.texAlign = 65536;
    BufferRef small = produce(g, "small", 4096 + 1);
    TextureRef tex;
    g.addPass("tex", PassType::Compute, [&](PassBuilder& p) {
        tex = p.write(p.createTexture("tex", {Format::RGBA8Unorm, 100, 100, 1, 1, 1}), Usage::ShaderWrite,
                      StageDispatch);
    }, nullptr);
    BufferRef tail = produce(g, "tail", 100);
    g.addPass("use", PassType::Compute, [&](PassBuilder& p) {
        p.read(small, Usage::ShaderRead, StageDispatch);
        p.read(tex, Usage::ShaderRead, StageDispatch);
        p.read(tail, Usage::ShaderRead, StageDispatch);
        p.setSideEffect();
    }, nullptr);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    AliasingPlan plan = planAliasing(g, c, sizer);
    REQUIRE(plan.placements.size() == 3);
    CHECK(find(plan, tex.resource)->offset % 65536 == 0);
    CHECK(find(plan, small.resource)->offset % 4096 == 0);
    CHECK(find(plan, tail.resource)->offset % 4096 == 0);
    for (size_t i = 0; i < plan.placements.size(); ++i)
        for (size_t j = i + 1; j < plan.placements.size(); ++j)
            CHECK_FALSE(rangesIntersect(plan.placements[i], plan.placements[j]));
    CHECK(plan.heapSize <= plan.unaliasedSize);
}

TEST_CASE("aliasing: alias=false lays resources back to back without aliased flags") {
    RenderGraph g;
    BufferRef a = produce(g, "A", 10000);
    consume(g, "readA", a);
    BufferRef b = produce(g, "B", 8000);
    consume(g, "readB", b);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    FakeSizer sizer;
    AliasingPlan plan = planAliasing(g, c, sizer, false);
    REQUIRE(plan.placements.size() == 2);
    CHECK(plan.heapSize == plan.unaliasedSize);
    CHECK(plan.placements[0].offset == 0);
    CHECK(plan.placements[1].offset == 12288);
    for (const Placement& p : plan.placements) CHECK_FALSE(p.aliased);
}

TEST_CASE("aliasing: async compute resources share memory with nothing") {
    RenderGraph g;
    BufferRef a = produce(g, "A", 4096, Queue::AsyncCompute);
    consume(g, "readA", a, Queue::AsyncCompute);
    BufferRef b = produce(g, "B", 4096);
    consume(g, "readB", b);
    BufferRef d = produce(g, "D", 4096);
    consume(g, "readD", d);
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    FakeSizer sizer;
    AliasingPlan plan = planAliasing(g, c, sizer);
    REQUIRE(plan.placements.size() == 3);
    const Placement* pa = find(plan, a.resource);
    REQUIRE(pa);
    CHECK_FALSE(pa->aliased);
    CHECK_FALSE(rangesIntersect(*pa, *find(plan, b.resource)));
    CHECK_FALSE(rangesIntersect(*pa, *find(plan, d.resource)));
    // B and D remain free to share with each other.
    CHECK(find(plan, b.resource)->offset == find(plan, d.resource)->offset);
    CHECK(plan.heapSize == 8192);
}

TEST_CASE("aliasing: deterministic") {
    RenderGraph g;
    for (int i = 0; i < 6; ++i) {
        BufferRef r = produce(g, "R", 4096 * (1 + i % 3));
        consume(g, "C", r);
    }
    CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    FakeSizer sizer;
    AliasingPlan p1 = planAliasing(g, c, sizer);
    AliasingPlan p2 = planAliasing(g, c, sizer);
    REQUIRE(p1.placements.size() == p2.placements.size());
    for (size_t i = 0; i < p1.placements.size(); ++i) {
        CHECK(p1.placements[i].resource == p2.placements[i].resource);
        CHECK(p1.placements[i].offset == p2.placements[i].offset);
        CHECK(p1.placements[i].size == p2.placements[i].size);
        CHECK(p1.placements[i].aliased == p2.placements[i].aliased);
    }
    CHECK(p1.heapSize == p2.heapSize);
    // Ordered by resource index.
    for (size_t i = 1; i < p1.placements.size(); ++i)
        CHECK(p1.placements[i - 1].resource < p1.placements[i].resource);
}

TEST_CASE("aliasing: randomized graphs keep the placement invariants") {
    std::mt19937 rng(12345);
    auto rnd = [&](u32 lo, u32 hi) { return std::uniform_int_distribution<u32>(lo, hi)(rng); };
    const u64 aligns[] = {256, 4096, 16384, 65536};

    for (int iter = 0; iter < 200; ++iter) {
        RenderGraph g;
        std::vector<BufferRef>  bufs;
        std::vector<TextureRef> texs;
        const u32 passCount = rnd(2, 14);
        for (u32 p = 0; p < passCount; ++p) {
            const bool async = rnd(0, 9) == 0;
            g.addPass("p", PassType::Compute, async ? Queue::AsyncCompute : Queue::Graphics,
                      [&](PassBuilder& b) {
                          // Reads of already-written resources.
                          for (u32 k = rnd(0, 2); k > 0; --k) {
                              if (!bufs.empty() && rnd(0, 1)) {
                                  b.read(bufs[rnd(0, u32(bufs.size() - 1))], Usage::ShaderRead, StageDispatch);
                              } else if (!texs.empty()) {
                                  b.read(texs[rnd(0, u32(texs.size() - 1))], Usage::ShaderRead, StageDispatch);
                              }
                          }
                          // New resources.
                          for (u32 k = rnd(0, 2); k > 0; --k) {
                              if (rnd(0, 1)) {
                                  u32 sz = rnd(1, 300000);
                                  bufs.push_back(b.write(b.createBuffer("b", {sz}), Usage::ShaderWrite,
                                                         StageDispatch));
                              } else {
                                  TextureDesc d{Format::RGBA8Unorm, rnd(1, 300), rnd(1, 300), 1, 1, 1};
                                  texs.push_back(b.write(b.createTexture("t", d), Usage::ShaderWrite,
                                                         StageDispatch));
                              }
                          }
                          b.setSideEffect();
                      }, nullptr);
        }
        CompiledGraph c = compileOrder(g);
        REQUIRE(c.ok);

        FakeSizer sizer;
        sizer.bufAlign = aligns[rnd(0, 3)];
        sizer.texAlign = aligns[rnd(0, 3)];
        AliasingPlan plan = planAliasing(g, c, sizer);

        // Lifetimes as widened by the async rule.
        std::vector<Lifetime> life = c.lifetimes;
        for (u32 pi = 0; pi < g.passes().size(); ++pi) {
            if (g.passes()[pi].queue != Queue::AsyncCompute || c.culled[pi]) continue;
            for (const Access& a : g.passes()[pi].reads) life[a.resource] = {0, u32(c.order.size() - 1)};
            for (const Access& a : g.passes()[pi].writes) life[a.resource] = {0, u32(c.order.size() - 1)};
        }

        u64 maxEnd = 0;
        for (const Placement& p : plan.placements) {
            const ResourceNode& n = g.resources()[p.resource];
            const u64 align = n.kind == ResourceKind::Texture ? sizer.texAlign : sizer.bufAlign;
            CHECK(p.offset % align == 0);
            maxEnd = std::max(maxEnd, p.offset + p.size);
        }
        CHECK(plan.heapSize >= maxEnd);
        CHECK(plan.heapSize <= plan.unaliasedSize);
        for (size_t i = 0; i < plan.placements.size(); ++i) {
            bool intersectsOther = false;
            for (size_t j = 0; j < plan.placements.size(); ++j) {
                if (i == j) continue;
                const bool ranges = rangesIntersect(plan.placements[i], plan.placements[j]);
                if (life[plan.placements[i].resource].overlaps(life[plan.placements[j].resource]))
                    CHECK_FALSE(ranges);
                intersectsOther = intersectsOther || ranges;
            }
            CHECK(plan.placements[i].aliased == intersectsOther);
        }
    }
}
