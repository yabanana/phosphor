#include "rendergraph/graph_lint.h"
#include "rendergraph/render_graph.h"
#include "rendergraph/scenario.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

constexpr u32 kW = 640;
constexpr u32 kH = 360;

bool contains(const std::string& s, const std::string& what) { return s.find(what) != std::string::npos; }

bool anyFinding(const std::vector<LintFinding>& f, bool error, const std::string& what) {
    return std::any_of(f.begin(), f.end(),
                       [&](const LintFinding& x) { return x.error == error && contains(x.message, what); });
}

size_t errorCount(const std::vector<LintFinding>& f) {
    return size_t(std::count_if(f.begin(), f.end(), [](const LintFinding& x) { return x.error; }));
}

// gbuffer (albedo + depth) -> light (drawable, reads albedo per pixel), one
// render group: albedo and depth are memoryless.  With `sampleAlbedo` a
// compute pass samples albedo afterwards: it must be stored.
struct Deferred {
    RenderGraph graph;
    TextureRef  drawable, albedo, depth;
    explicit Deferred(bool sampleAlbedo) {
        drawable = graph.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
        graph.addPass("gbuffer", PassType::Raster,
                      [&](PassBuilder& b) {
                          albedo = b.createTexture("albedo", {Format::RGBA8Unorm, kW, kH});
                          depth  = b.createTexture("depth", {Format::Depth32Float, kW, kH});
                          albedo = b.writeColor(albedo, 0, LoadIntent::Clear);
                          depth  = b.writeDepth(depth, LoadIntent::Clear);
                      },
                      nullptr);
        graph.addPass("light", PassType::Raster,
                      [&](PassBuilder& b) {
                          b.readColor(albedo, 0);
                          drawable = b.writeColor(drawable, 1, LoadIntent::Clear);
                      },
                      nullptr);
        if (sampleAlbedo) {
            graph.addPass("blur", PassType::Compute,
                          [&](PassBuilder& b) {
                              b.read(albedo, Usage::ShaderRead, StageDispatch);
                              b.setSideEffect();
                          },
                          nullptr);
        }
    }
};

// prepass (depth) -> hist (samples depth) -> forward (drawable, read-only
// depth) [-> post (samples depth again)].
struct SplitFrame {
    RenderGraph graph;
    TextureRef  drawable, depth;
    explicit SplitFrame(bool post) {
        drawable = graph.importTexture("drawable", {Format::BGRA8Unorm, kW, kH}, ImportOutput);
        graph.addPass("prepass", PassType::Raster,
                      [&](PassBuilder& b) {
                          depth = b.createTexture("depth", {Format::Depth32Float, kW, kH});
                          depth = b.writeDepth(depth, LoadIntent::Clear);
                      },
                      nullptr);
        graph.addPass("hist", PassType::Compute,
                      [&](PassBuilder& b) {
                          b.read(depth, Usage::ShaderRead, StageDispatch);
                          b.setSideEffect();
                      },
                      nullptr);
        graph.addPass("forward", PassType::Raster,
                      [&](PassBuilder& b) {
                          drawable = b.writeColor(drawable, 0, LoadIntent::Clear);
                          b.readDepth(depth);
                      },
                      nullptr);
        if (post) {
            graph.addPass("post", PassType::Compute,
                          [&](PassBuilder& b) {
                              b.read(depth, Usage::ShaderRead, StageDispatch);
                              b.setSideEffect();
                          },
                          nullptr);
        }
    }
};

} // namespace

TEST_CASE("lint: fused deferred frame is clean, memoryless attachments give no finding") {
    Deferred d(false);
    const CompiledGraph c = compile(d.graph);
    REQUIRE(c.ok);
    REQUIRE(c.renderGroups.size() == 1);
    CHECK(c.memoryless[d.albedo.resource]);
    CHECK(c.memoryless[d.depth.resource]);
    CHECK(lintGraph(d.graph, c).empty());
}

TEST_CASE("lint: a stored transient explains why it is not memoryless") {
    Deferred d(true);
    const CompiledGraph c = compile(d.graph);
    REQUIRE(c.ok);
    const auto findings = lintGraph(d.graph, c);
    CHECK(errorCount(findings) == 0);
    REQUIRE(anyFinding(findings, false, "transient 'albedo'"));
    CHECK(anyFinding(findings, false, "read later as sampled/shader read by pass 'blur'"));
    CHECK_FALSE(anyFinding(findings, false, "transient 'depth'"));
}

TEST_CASE("lint: a compute pass between two raster passes splits the group and is named") {
    SplitFrame f(false);
    const CompiledGraph c = compile(f.graph);
    REQUIRE(c.ok);
    REQUIRE(c.renderGroups.size() == 2);
    const auto findings = lintGraph(f.graph, c);
    CHECK(errorCount(findings) == 0);
    REQUIRE(anyFinding(findings, false, "transient 'depth'"));
    CHECK(anyFinding(findings, false, "spans render groups 0/1 because pass 'hist' sits between them"));
    CHECK(anyFinding(findings, false, "loaded in group 1 (read-only attachment use)"));
    CHECK(anyFinding(findings, false, "stored by group 0 because read later as sampled/shader read by pass 'hist'"));
}

TEST_CASE("lint: a read-only attachment stored again is a conservative note, not an error") {
    SplitFrame f(true);
    const CompiledGraph c = compile(f.graph);
    REQUIRE(c.ok);
    REQUIRE(c.renderGroups.size() == 2);
    const AttachmentPlan* a = nullptr;
    for (const AttachmentPlan& p : c.renderGroups[1].attachments) {
        if (p.resource == f.depth.resource) a = &p;
    }
    REQUIRE(a);
    CHECK(a->readOnly);
    CHECK(a->store == StoreAction::Store);
    const auto findings = lintGraph(f.graph, c);
    CHECK(errorCount(findings) == 0);
    CHECK(anyFinding(findings, false, "stores read-only attachment 'depth'"));
    CHECK(anyFinding(findings, false, "pass 'post' reads it later"));
}

TEST_CASE("lint: an unjustified Store is an error (hand-built wrong plan)") {
    Deferred d(false);
    CompiledGraph c = compile(d.graph);
    REQUIRE(c.ok);
    REQUIRE(lintGraph(d.graph, c).empty());
    // Wrongly store the albedo at the end of the group: nothing reads it.
    c.memoryless[d.albedo.resource] = false;
    bool patched = false;
    for (AttachmentPlan& a : c.renderGroups[0].attachments) {
        if (a.resource == d.albedo.resource) {
            a.store = StoreAction::Store;
            patched = true;
        }
    }
    REQUIRE(patched);
    const auto findings = lintGraph(d.graph, c);
    CHECK(anyFinding(findings, true, "stores 'albedo'"));
    CHECK(anyFinding(findings, true, "no later pass reads that version"));
    // Real plans never trigger it: LintMode::Error compiles.
    CompileOptions o;
    o.lint = LintMode::Error;
    CHECK(compile(d.graph, o).ok);
}

TEST_CASE("lint: a missing Store and a memoryless attachment that loads are errors") {
    Deferred d(true);
    CompiledGraph c = compile(d.graph);
    REQUIRE(c.ok);
    for (AttachmentPlan& a : c.renderGroups[0].attachments) {
        if (a.resource == d.albedo.resource) a.store = StoreAction::DontCare;
    }
    CHECK(anyFinding(lintGraph(d.graph, c), true, "drops 'albedo'"));

    Deferred e(false);
    CompiledGraph c2 = compile(e.graph);
    REQUIRE(c2.ok);
    for (AttachmentPlan& a : c2.renderGroups[0].attachments) {
        if (a.resource == e.depth.resource) a.load = LoadAction::Load;
    }
    CHECK(anyFinding(lintGraph(e.graph, c2), true, "'depth' is memoryless but"));
}

TEST_CASE("lint: the four scenarios have no error and every note names its reason") {
    for (u32 s = 0; s < scenarioCount(); ++s) {
        CAPTURE(s);
        RenderGraph g;
        Scenario sc;
        const TextureRef drawable =
            g.importTexture("Drawable", {Format::BGRA8Srgb, 1920, 1080}, ImportOutput | ImportPerFrame);
        std::string error;
        REQUIRE_MESSAGE(buildScenario(s, {}, g, drawable, sc, {}, &error), error);
        const CompiledGraph c = compile(g);
        REQUIRE(c.ok);
        const auto findings = lintGraph(g, c);
        CHECK(errorCount(findings) == 0);
        for (const LintFinding& f : findings) {
            CAPTURE(f.message);
            CHECK_FALSE(f.message.empty());
            CHECK(f.resource != ~0u);
            if (contains(f.message, "is not memoryless")) {
                CHECK((contains(f.message, "loaded") || contains(f.message, "stored") ||
                       contains(f.message, "spans") || contains(f.message, "also used")));
            }
        }
        CompileOptions o;
        o.lint = LintMode::Error;
        CHECK(compile(g, o).ok);
    }
}
