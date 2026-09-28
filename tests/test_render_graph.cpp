#include "rendergraph/render_graph.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

constexpr u32 kCulled = ~0u;

// Compute-style accesses on buffers: attachments are irrelevant for most tests.
void readBuf(PassBuilder& b, BufferRef r) { b.read(r, Usage::ShaderRead, StageDispatch); }
BufferRef writeBuf(PassBuilder& b, BufferRef r) { return b.write(r, Usage::ShaderWrite, StageDispatch); }

u32 addCompute(RenderGraph& g, const std::string& name, const RenderGraph::SetupFn& setup) {
    return g.addPass(name, PassType::Compute, setup, nullptr);
}

bool hasDep(const CompiledGraph& c, u32 from, u32 to, u32 resource, DepKind kind) {
    return std::any_of(c.dependencies.begin(), c.dependencies.end(), [&](const Dependency& d) {
        return d.from == from && d.to == to && d.resource == resource && d.kind == kind;
    });
}

bool hasAnyDep(const CompiledGraph& c, u32 from, u32 to) {
    return std::any_of(c.dependencies.begin(), c.dependencies.end(),
                       [&](const Dependency& d) { return d.from == from && d.to == to; });
}

bool errorMentions(const std::vector<std::string>& errors, const std::string& text) {
    return std::any_of(errors.begin(), errors.end(),
                       [&](const std::string& e) { return e.find(text) != std::string::npos; });
}

TextureDesc colorDesc() {
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width  = 64;
    d.height = 64;
    return d;
}

TextureDesc depthDesc() {
    TextureDesc d = colorDesc();
    d.format = Format::Depth32Float;
    return d;
}

constexpr BufferDesc kBuf{256};

void expectSetupError(const RenderGraph& g, const char* text) {
    CHECK_FALSE(g.errors().empty());
    CHECK(errorMentions(g.errors(), text));
    const CompiledGraph c = compileOrder(g);
    CHECK_FALSE(c.ok);
    CHECK(c.order.empty());
    CHECK_FALSE(c.errors.empty());
}

} // namespace

TEST_CASE("render graph: linear chain orders passes with RAW dependencies") {
    RenderGraph g;
    BufferRef a, b;
    addCompute(g, "A", [&](PassBuilder& p) { a = writeBuf(p, p.createBuffer("a", kBuf)); });
    addCompute(g, "B", [&](PassBuilder& p) {
        readBuf(p, a);
        b = writeBuf(p, p.createBuffer("b", kBuf));
    });
    addCompute(g, "C", [&](PassBuilder& p) {
        readBuf(p, b);
        p.setSideEffect();
    });
    CHECK(g.errors().empty());
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{0, 1, 2});
    REQUIRE(c.dependencies.size() == 2);
    CHECK(hasDep(c, 0, 1, a.resource, DepKind::RAW));
    CHECK(hasDep(c, 1, 2, b.resource, DepKind::RAW));
    CHECK(c.position(2) == 2);
}

TEST_CASE("render graph: diamond keeps declaration order and exact dependencies") {
    RenderGraph g;
    BufferRef x, y, z;
    addCompute(g, "P0", [&](PassBuilder& p) { x = writeBuf(p, p.createBuffer("x", kBuf)); });
    addCompute(g, "P1", [&](PassBuilder& p) {
        readBuf(p, x);
        y = writeBuf(p, p.createBuffer("y", kBuf));
    });
    addCompute(g, "P2", [&](PassBuilder& p) {
        readBuf(p, x);
        z = writeBuf(p, p.createBuffer("z", kBuf));
    });
    addCompute(g, "P3", [&](PassBuilder& p) {
        readBuf(p, y);
        readBuf(p, z);
        p.setSideEffect();
    });
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{0, 1, 2, 3});
    REQUIRE(c.dependencies.size() == 4);
    CHECK(hasDep(c, 0, 1, x.resource, DepKind::RAW));
    CHECK(hasDep(c, 0, 2, x.resource, DepKind::RAW));
    CHECK(hasDep(c, 1, 3, y.resource, DepKind::RAW));
    CHECK(hasDep(c, 2, 3, z.resource, DepKind::RAW));
}

TEST_CASE("render graph: WAR orders the overwrite after earlier readers, not the reverse") {
    RenderGraph g;
    BufferRef a1, a2;
    addCompute(g, "P0", [&](PassBuilder& p) { a1 = writeBuf(p, p.createBuffer("A", kBuf)); });
    addCompute(g, "P1", [&](PassBuilder& p) {
        readBuf(p, a1);
        p.setSideEffect();
    });
    addCompute(g, "P2", [&](PassBuilder& p) { a2 = writeBuf(p, a1); });
    addCompute(g, "P3", [&](PassBuilder& p) {
        readBuf(p, a2);
        p.setSideEffect();
    });
    REQUIRE(g.errors().empty());
    CHECK(a2.version == 2);
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{0, 1, 2, 3});
    CHECK(hasDep(c, 0, 1, a1.resource, DepKind::RAW));
    CHECK(hasDep(c, 1, 2, a1.resource, DepKind::WAR));
    CHECK(hasDep(c, 0, 2, a1.resource, DepKind::WAW));
    CHECK(hasDep(c, 2, 3, a1.resource, DepKind::RAW));
    // Legacy defect: a reader depended on the last writer of the frame.
    CHECK_FALSE(hasAnyDep(c, 2, 1));
    CHECK_FALSE(hasAnyDep(c, 3, 1));
}

TEST_CASE("render graph: a stale reader declared after a later writer is ordered before it (WAR)") {
    RenderGraph g;
    const BufferRef a0 = g.importBuffer("A", kBuf, ImportContentsDefined);
    addCompute(g, "writer", [&](PassBuilder& p) {
        writeBuf(p, a0);
        p.setSideEffect();
    });
    addCompute(g, "reader", [&](PassBuilder& p) {
        readBuf(p, a0); // still the initial contents
        p.setSideEffect();
    });
    REQUIRE(g.errors().empty());
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{1, 0});
    CHECK(hasDep(c, 1, 0, a0.resource, DepKind::WAR));
    CHECK_FALSE(hasAnyDep(c, 0, 1));
}

TEST_CASE("render graph: WAW orders two writers of the same resource") {
    RenderGraph g;
    BufferRef a1;
    addCompute(g, "first", [&](PassBuilder& p) {
        a1 = writeBuf(p, p.createBuffer("A", kBuf));
        p.setSideEffect();
    });
    addCompute(g, "second", [&](PassBuilder& p) {
        writeBuf(p, a1);
        p.setSideEffect();
    });
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{0, 1});
    CHECK(hasDep(c, 0, 1, a1.resource, DepKind::WAW));

    // An overwritten version nobody reads and without side effects is culled.
    RenderGraph h;
    BufferRef b1;
    addCompute(h, "first", [&](PassBuilder& p) { b1 = writeBuf(p, p.createBuffer("B", kBuf)); });
    addCompute(h, "second", [&](PassBuilder& p) {
        writeBuf(p, b1);
        p.setSideEffect();
    });
    const CompiledGraph d = compileOrder(h);
    REQUIRE(d.ok);
    CHECK(d.culled[0]);
    CHECK(d.order == std::vector<u32>{1});
}

TEST_CASE("render graph: cycle from stale read plus fresh read is a compile error") {
    RenderGraph g;
    const BufferRef a0 = g.importBuffer("A", kBuf, ImportContentsDefined);
    const BufferRef b0 = g.importBuffer("B", kBuf, ImportContentsDefined);
    BufferRef b1;
    addCompute(g, "P1", [&](PassBuilder& p) {
        writeBuf(p, a0);
        b1 = writeBuf(p, b0);
    });
    addCompute(g, "P2", [&](PassBuilder& p) {
        readBuf(p, a0); // stale: must run before P1
        readBuf(p, b1); // fresh: must run after P1
        p.setSideEffect();
    });
    REQUIRE(g.errors().empty());
    const CompiledGraph c = compileOrder(g);
    CHECK_FALSE(c.ok);
    CHECK(c.order.empty());
    CHECK(errorMentions(c.errors, "cycle"));
}

TEST_CASE("render graph: culling keeps side effects and needed writers, drops dead ones") {
    SUBCASE("unread output is culled, side-effect pass kept") {
        RenderGraph g;
        BufferRef dead, live;
        addCompute(g, "dead", [&](PassBuilder& p) { dead = writeBuf(p, p.createBuffer("dead", kBuf)); });
        addCompute(g, "producer", [&](PassBuilder& p) { live = writeBuf(p, p.createBuffer("live", kBuf)); });
        addCompute(g, "sink", [&](PassBuilder& p) {
            readBuf(p, live);
            p.setSideEffect();
        });
        addCompute(g, "lonely", [&](PassBuilder& p) { p.setSideEffect(); });
        const CompiledGraph c = compileOrder(g);
        REQUIRE(c.ok);
        CHECK(c.culled[0]);
        CHECK(c.position(0) == kCulled);
        CHECK_FALSE(c.culled[1]);
        CHECK_FALSE(c.culled[2]);
        CHECK_FALSE(c.culled[3]);
        CHECK(c.order == std::vector<u32>{1, 2, 3});
        CHECK_FALSE(c.lifetimes[dead.resource].used());
    }
    SUBCASE("ImportOutput: overwritten Clear version is culled, final writer kept") {
        RenderGraph g;
        const TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
        TextureRef o1;
        addCompute(g, "unused", [](PassBuilder&) {});
        g.addPass("first", PassType::Raster,
                  [&](PassBuilder& p) { o1 = p.writeColor(out, 0, LoadIntent::Clear); }, nullptr);
        g.addPass("second", PassType::Raster,
                  [&](PassBuilder& p) { p.writeColor(o1, 0, LoadIntent::Discard); }, nullptr);
        REQUIRE(g.errors().empty());
        const CompiledGraph c = compileOrder(g);
        REQUIRE(c.ok);
        CHECK(c.culled[0]);
        CHECK(c.culled[1]);
        CHECK_FALSE(c.culled[2]);
        CHECK(c.order == std::vector<u32>{2});
    }
    SUBCASE("writers needed transitively by a live reader are kept") {
        RenderGraph g;
        const TextureRef out = g.importTexture("out", colorDesc(), ImportOutput);
        BufferRef buf;
        TextureRef o1;
        addCompute(g, "gen", [&](PassBuilder& p) { buf = writeBuf(p, p.createBuffer("buf", kBuf)); });
        g.addPass("draw", PassType::Raster,
                  [&](PassBuilder& p) {
                      p.read(buf, Usage::ShaderRead, StageVertex);
                      o1 = p.writeColor(out, 0, LoadIntent::Clear);
                  },
                  nullptr);
        g.addPass("overlay", PassType::Raster,
                  [&](PassBuilder& p) { p.writeColor(o1, 0, LoadIntent::Preserve); }, nullptr);
        REQUIRE(g.errors().empty());
        const CompiledGraph c = compileOrder(g);
        REQUIRE(c.ok);
        CHECK(c.order == std::vector<u32>{0, 1, 2});
        CHECK(c.culled == std::vector<bool>{false, false, false});
    }
}

TEST_CASE("render graph: WAR is kept through a culled intermediate writer") {
    RenderGraph g;
    const BufferRef a0 = g.importBuffer("A", kBuf, ImportOutput);
    BufferRef a1, a2;
    addCompute(g, "P0", [&](PassBuilder& p) { a1 = writeBuf(p, a0); });
    addCompute(g, "P1", [&](PassBuilder& p) {
        readBuf(p, a1);
        p.setSideEffect();
    });
    addCompute(g, "P2", [&](PassBuilder& p) { a2 = writeBuf(p, a1); }); // nobody reads A2
    addCompute(g, "P3", [&](PassBuilder& p) { writeBuf(p, a2); });
    REQUIRE(g.errors().empty());
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.culled[2]);
    CHECK_FALSE(c.culled[0]);
    CHECK_FALSE(c.culled[1]);
    CHECK_FALSE(c.culled[3]);
    CHECK(hasDep(c, 1, 3, a0.resource, DepKind::WAR));
    CHECK(hasDep(c, 0, 3, a0.resource, DepKind::WAW));
    CHECK(c.position(1) < c.position(3));
}

TEST_CASE("render graph: order is stable for independent passes and across compilations") {
    RenderGraph g;
    for (int i = 0; i < 6; ++i) {
        addCompute(g, "p" + std::to_string(i), [](PassBuilder& p) { p.setSideEffect(); });
    }
    const CompiledGraph c1 = compileOrder(g);
    const CompiledGraph c2 = compileOrder(g);
    REQUIRE(c1.ok);
    REQUIRE(c2.ok);
    CHECK(c1.order == std::vector<u32>{0, 1, 2, 3, 4, 5});
    CHECK(c1.order == c2.order);
    CHECK(c1.dependencies.empty());

    // A dependent pass declared first only moves as far as needed.
    RenderGraph h;
    const BufferRef imp = h.importBuffer("i", kBuf, ImportContentsDefined);
    addCompute(h, "late", [&](PassBuilder& p) {
        writeBuf(p, imp);
        p.setSideEffect();
    });
    addCompute(h, "early", [&](PassBuilder& p) {
        readBuf(p, imp);
        p.setSideEffect();
    });
    addCompute(h, "free", [](PassBuilder& p) { p.setSideEffect(); });
    const CompiledGraph d = compileOrder(h);
    REQUIRE(d.ok);
    CHECK(d.order == std::vector<u32>{1, 0, 2});
}

TEST_CASE("render graph: S-TBDR-6 prefers a geometry-heavy pass after a fragment-heavy one") {
    RenderGraph g;
    addCompute(g, "frag", [](PassBuilder& p) {
        p.setSideEffect();
        p.setHints(HintFragmentHeavy);
    });
    addCompute(g, "plain", [](PassBuilder& p) { p.setSideEffect(); });
    addCompute(g, "geom", [](PassBuilder& p) {
        p.setSideEffect();
        p.setHints(HintGeometryHeavy);
    });
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.order == std::vector<u32>{0, 2, 1});

    // Without a preceding fragment-heavy pass the declaration order wins.
    RenderGraph h;
    addCompute(h, "plain", [](PassBuilder& p) { p.setSideEffect(); });
    addCompute(h, "geom", [](PassBuilder& p) {
        p.setSideEffect();
        p.setHints(HintGeometryHeavy);
    });
    const CompiledGraph d = compileOrder(h);
    REQUIRE(d.ok);
    CHECK(d.order == std::vector<u32>{0, 1});
}

TEST_CASE("render graph: setup error, transient read before any write") {
    RenderGraph g;
    addCompute(g, "P", [&](PassBuilder& p) {
        const BufferRef t = p.createBuffer("t", kBuf);
        readBuf(p, t);
        p.setSideEffect();
    });
    expectSetupError(g, "before anything wrote");
}

TEST_CASE("render graph: setup error, import version 0 read without ImportContentsDefined") {
    RenderGraph g;
    const BufferRef imp = g.importBuffer("imp", kBuf, ImportNone);
    addCompute(g, "P", [&](PassBuilder& p) {
        readBuf(p, imp);
        p.setSideEffect();
    });
    expectSetupError(g, "before anything wrote");
}

TEST_CASE("render graph: setup error, stale write through an old handle") {
    RenderGraph g;
    BufferRef b0;
    addCompute(g, "P0", [&](PassBuilder& p) {
        b0 = p.createBuffer("b", kBuf);
        writeBuf(p, b0);
    });
    addCompute(g, "P1", [&](PassBuilder& p) {
        writeBuf(p, b0); // v0 handle, but v1 exists
        p.setSideEffect();
    });
    expectSetupError(g, "stale handle");
}

TEST_CASE("render graph: setup error, pass writes the same resource twice") {
    RenderGraph g;
    addCompute(g, "P", [&](PassBuilder& p) {
        const BufferRef b1 = writeBuf(p, p.createBuffer("b", kBuf));
        writeBuf(p, b1);
        p.setSideEffect();
    });
    expectSetupError(g, "twice");
}

TEST_CASE("render graph: setup error, color/depth attachment format mismatch") {
    SUBCASE("writeColor on a depth format") {
        RenderGraph g;
        g.addPass("P", PassType::Raster,
                  [&](PassBuilder& p) {
                      p.writeColor(p.createTexture("d", depthDesc()), 0, LoadIntent::Clear);
                      p.setSideEffect();
                  },
                  nullptr);
        expectSetupError(g, "not a color texture");
    }
    SUBCASE("writeDepth on a color format") {
        RenderGraph g;
        g.addPass("P", PassType::Raster,
                  [&](PassBuilder& p) {
                      p.writeDepth(p.createTexture("c", colorDesc()), LoadIntent::Clear);
                      p.setSideEffect();
                  },
                  nullptr);
        expectSetupError(g, "not a depth texture");
    }
}

TEST_CASE("render graph: setup error, attachment in a compute pass") {
    RenderGraph g;
    addCompute(g, "P", [&](PassBuilder& p) {
        p.writeColor(p.createTexture("c", colorDesc()), 0, LoadIntent::Clear);
        p.setSideEffect();
    });
    expectSetupError(g, "raster pass");
}

TEST_CASE("render graph: setup error, raster pass on the async compute queue") {
    RenderGraph g;
    g.addPass("P", PassType::Raster, Queue::AsyncCompute, [](PassBuilder& p) { p.setSideEffect(); }, nullptr);
    expectSetupError(g, "graphics queue");
}

TEST_CASE("render graph: Preserve attachment write reads the previous version, Clear does not") {
    SUBCASE("color") {
        RenderGraph g;
        TextureRef t1, t2;
        g.addPass("clear", PassType::Raster,
                  [&](PassBuilder& p) { t1 = p.writeColor(p.createTexture("t", colorDesc()), 0, LoadIntent::Clear); },
                  nullptr);
        g.addPass("blend", PassType::Raster,
                  [&](PassBuilder& p) {
                      t2 = p.writeColor(t1, 0, LoadIntent::Preserve);
                      p.setSideEffect();
                  },
                  nullptr);
        g.addPass("clear2", PassType::Raster,
                  [&](PassBuilder& p) {
                      p.writeColor(t2, 0, LoadIntent::Clear);
                      p.setSideEffect();
                  },
                  nullptr);
        REQUIRE(g.errors().empty());
        REQUIRE(g.passes()[0].reads.empty());
        REQUIRE(g.passes()[1].reads.size() == 1);
        CHECK(g.passes()[1].reads[0].usage == Usage::ColorAttachment);
        CHECK(g.passes()[1].reads[0].version == 1);
        CHECK(g.passes()[1].reads[0].resource == t1.resource);
        CHECK(g.passes()[2].reads.empty());
        const CompiledGraph c = compileOrder(g);
        REQUIRE(c.ok);
        CHECK(hasDep(c, 0, 1, t1.resource, DepKind::RAW));
        CHECK_FALSE(hasDep(c, 1, 2, t1.resource, DepKind::RAW));
        CHECK(hasDep(c, 1, 2, t1.resource, DepKind::WAW)); // still ordered
    }
    SUBCASE("depth") {
        RenderGraph g;
        TextureRef d1;
        g.addPass("clear", PassType::Raster,
                  [&](PassBuilder& p) { d1 = p.writeDepth(p.createTexture("d", depthDesc()), LoadIntent::Clear); },
                  nullptr);
        g.addPass("more", PassType::Raster,
                  [&](PassBuilder& p) {
                      p.writeDepth(d1, LoadIntent::Preserve);
                      p.setSideEffect();
                  },
                  nullptr);
        REQUIRE(g.errors().empty());
        REQUIRE(g.passes()[1].reads.size() == 1);
        CHECK(g.passes()[1].reads[0].usage == Usage::DepthRead);
        CHECK(g.passes()[1].writes[0].usage == Usage::DepthAttachment);
        const CompiledGraph c = compileOrder(g);
        REQUIRE(c.ok);
        CHECK(hasDep(c, 0, 1, d1.resource, DepKind::RAW));
    }
}

TEST_CASE("render graph: lifetimes follow execution positions") {
    RenderGraph g;
    BufferRef a, b;
    addCompute(g, "A", [&](PassBuilder& p) { a = writeBuf(p, p.createBuffer("a", kBuf)); });
    addCompute(g, "B", [&](PassBuilder& p) {
        readBuf(p, a);
        b = writeBuf(p, p.createBuffer("b", kBuf));
    });
    addCompute(g, "C", [&](PassBuilder& p) {
        readBuf(p, b);
        p.setSideEffect();
    });
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    REQUIRE(c.lifetimes.size() == 2);
    CHECK(c.lifetimes[a.resource].first == 0);
    CHECK(c.lifetimes[a.resource].last == 1);
    CHECK(c.lifetimes[b.resource].first == 1);
    CHECK(c.lifetimes[b.resource].last == 2);
    CHECK(c.lifetimes[a.resource].overlaps(c.lifetimes[b.resource]));

    // Resource only used by a culled pass, and a never-used import.
    RenderGraph h;
    BufferRef dead;
    const BufferRef unused = h.importBuffer("unused", kBuf, ImportNone);
    addCompute(h, "dead", [&](PassBuilder& p) { dead = writeBuf(p, p.createBuffer("dead", kBuf)); });
    addCompute(h, "live", [](PassBuilder& p) { p.setSideEffect(); });
    const CompiledGraph d = compileOrder(h);
    REQUIRE(d.ok);
    CHECK_FALSE(d.lifetimes[dead.resource].used());
    CHECK_FALSE(d.lifetimes[unused.resource].used());
    CHECK_FALSE(d.lifetimes[dead.resource].overlaps(d.lifetimes[unused.resource]));
}

TEST_CASE("render graph: Lifetime::overlaps") {
    const Lifetime a{0, 2};
    const Lifetime b{2, 4};
    const Lifetime c{3, 5};
    const Lifetime unused;
    CHECK(a.used());
    CHECK_FALSE(unused.used());
    CHECK(a.overlaps(b)); // inclusive ends
    CHECK(b.overlaps(a));
    CHECK_FALSE(a.overlaps(c));
    CHECK(b.overlaps(c));
    CHECK_FALSE(a.overlaps(unused));
    CHECK_FALSE(unused.overlaps(unused));
}

TEST_CASE("render graph: reading version 0 of a contents-defined import creates no dependency") {
    RenderGraph g;
    const BufferRef imp = g.importBuffer("persistent", kBuf, ImportContentsDefined);
    addCompute(g, "reader", [&](PassBuilder& p) {
        readBuf(p, imp);
        p.setSideEffect();
    });
    REQUIRE(g.errors().empty());
    const CompiledGraph c = compileOrder(g);
    REQUIRE(c.ok);
    CHECK(c.dependencies.empty());
    CHECK(c.order == std::vector<u32>{0});
}

TEST_CASE("render graph: TextureDesc::estimatedBytes and format helpers") {
    TextureDesc d;
    d.format = Format::RGBA8Unorm;
    d.width  = 256;
    d.height = 128;
    CHECK(d.estimatedBytes() == 256u * 128u * 4u);

    d.mipLevels = 3; // 256x128 + 128x64 + 64x32
    CHECK(d.estimatedBytes() == (256u * 128u + 128u * 64u + 64u * 32u) * 4u);

    d.sampleCount = 4;
    d.depth       = 2;
    CHECK(d.estimatedBytes() == (256u * 128u + 128u * 64u + 64u * 32u) * 4u * 2u * 4u);

    TextureDesc small;
    small.format    = Format::RGBA16Float;
    small.width     = 2;
    small.height    = 1;
    small.mipLevels = 4; // dimensions clamp at 1
    CHECK(small.estimatedBytes() == (2u + 1u + 1u + 1u) * 8u);

    CHECK(bytesPerPixel(Format::R8Unorm) == 1);
    CHECK(bytesPerPixel(Format::RG8Unorm) == 2);
    CHECK(bytesPerPixel(Format::RGBA8Srgb) == 4);
    CHECK(bytesPerPixel(Format::Depth32Float) == 4);
    CHECK(bytesPerPixel(Format::RGBA16Float) == 8);
    CHECK(bytesPerPixel(Format::RGBA32Float) == 16);
    CHECK(bytesPerPixel(Format::Unknown) == 0);
    CHECK(isDepthFormat(Format::Depth16Unorm));
    CHECK(isDepthFormat(Format::Depth32Float));
    CHECK(isDepthFormat(Format::Depth32FloatStencil8));
    CHECK_FALSE(isDepthFormat(Format::RGBA8Unorm));
    CHECK_FALSE(isDepthFormat(Format::R32Float));
}
