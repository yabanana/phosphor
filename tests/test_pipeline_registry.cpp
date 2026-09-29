#include "pipeline/pipeline_registry.h"

#include <doctest/doctest.h>

#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace phosphor;
using namespace phosphor::pipe;

namespace {

// Fake GPU objects are addresses of ints; the releaser counts drops per object.
struct Fixture {
    int                 objs[64] = {};
    std::map<void*, int> released;
    u32                 releaseCalls = 0;
    PipelineRegistry    reg{16};

    Fixture() {
        reg.setReleaser(
            [](void* ctx, void* obj) {
                auto* f = static_cast<Fixture*>(ctx);
                ++f->released[obj];
                ++f->releaseCalls;
            },
            this);
    }
    void* obj(int i) { return &objs[i]; }
    int   releasedCount(int i) { return released.count(obj(i)) ? released[obj(i)] : 0; }

    PipelineHandle addEntry(PipelineKey key) {
        PipelineDesc d;
        d.functions[0] = "vs";
        return reg.add(key, d);
    }
    void post(PipelineHandle h, void* o, bool fallback, u32 gen = 0,
              ArchiveOutcome a = ArchiveOutcome::NotTried, u32 calls = 0, float ms = 0.0f) {
        Completion c;
        c.handle        = h;
        c.generation    = gen;
        c.object        = o;
        c.fallback      = fallback;
        c.archive       = a;
        c.compilerCalls = calls;
        c.compileMs     = ms;
        reg.post(c);
    }
};

} // namespace

TEST_CASE("registry: add and find") {
    Fixture f;
    CHECK(f.reg.find(5) == INVALID_PIPELINE);
    const PipelineHandle h = f.addEntry(5);
    CHECK(f.reg.find(5) == h);
    CHECK(f.reg.size() == 1);
    CHECK(f.reg.key(h) == 5);
    CHECK(f.reg.desc(h).functions[0] == "vs");
    CHECK(f.reg.state(h) == PipelineState::Pending);
    CHECK(f.reg.get(h) == nullptr);
    CHECK_FALSE(f.reg.isFinal(h));
    CHECK(f.reg.stats().requests == 1);
    const PipelineHandle h2 = f.addEntry(6);
    CHECK(h2 != h);
    CHECK(f.reg.stats().requests == 2);
}

TEST_CASE("registry: completions are invisible until drain") {
    Fixture f;
    const PipelineHandle h = f.addEntry(1);
    f.post(h, f.obj(0), false);
    CHECK(f.reg.get(h) == nullptr);
    CHECK(f.reg.drain() == 1);
    CHECK(f.reg.get(h) == f.obj(0));
    CHECK(f.reg.drain() == 0);
}

TEST_CASE("registry: fallback then final") {
    Fixture f;
    const PipelineHandle h = f.addEntry(1);
    f.post(h, f.obj(0), true, 0, ArchiveOutcome::NotTried, 1, 2.0f);
    CHECK(f.reg.drain() == 1);
    CHECK(f.reg.get(h) == f.obj(0));
    CHECK(f.reg.state(h) == PipelineState::Fallback);
    CHECK_FALSE(f.reg.isFinal(h));
    CHECK(f.reg.stats().fallbacksServed == 1);
    CHECK(f.reg.stats().archiveHits + f.reg.stats().archiveMisses + f.reg.stats().archiveUnavailable == 0);

    f.post(h, f.obj(1), false, 0, ArchiveOutcome::Miss, 1, 5.0f);
    CHECK(f.reg.drain() == 1);
    CHECK(f.reg.get(h) == f.obj(1));
    CHECK(f.reg.state(h) == PipelineState::Ready);
    CHECK(f.reg.isFinal(h));
    CHECK(f.releasedCount(0) == 1); // fallback dropped
    CHECK(f.releasedCount(1) == 0);
    CHECK(f.reg.stats().swaps == 2);
    CHECK(f.reg.stats().archiveMisses == 1);
    CHECK(f.reg.stats().compilerCalls == 2);
    CHECK(f.reg.stats().compileMs == doctest::Approx(7.0));
    CHECK(f.reg.stats().compileMsMax == doctest::Approx(5.0f));
}

TEST_CASE("registry: final then late fallback is released") {
    Fixture f;
    const PipelineHandle h = f.addEntry(1);
    f.post(h, f.obj(1), false, 0, ArchiveOutcome::Hit);
    f.post(h, f.obj(0), true);
    CHECK(f.reg.drain() == 1);
    CHECK(f.reg.get(h) == f.obj(1));
    CHECK(f.releasedCount(0) == 1);
    CHECK(f.reg.stats().fallbacksServed == 0);
    CHECK(f.reg.stats().archiveHits == 1);

    // Also across drains.
    f.post(h, f.obj(2), true);
    CHECK(f.reg.drain() == 0);
    CHECK(f.releasedCount(2) == 1);
    CHECK(f.reg.get(h) == f.obj(1));
}

TEST_CASE("registry: failed final keeps the fallback") {
    Fixture f;
    const PipelineHandle h = f.addEntry(1);
    f.post(h, f.obj(0), true);
    f.post(h, nullptr, false, 0, ArchiveOutcome::Unavailable);
    f.reg.drain();
    CHECK(f.reg.state(h) == PipelineState::Failed);
    CHECK(f.reg.get(h) == f.obj(0));
    CHECK_FALSE(f.reg.isFinal(h));
    CHECK(f.reg.stats().failures == 1);
    CHECK(f.reg.stats().archiveUnavailable == 1);
    CHECK(f.releaseCalls == 0);
}

TEST_CASE("registry: stale generation completions are released") {
    Fixture f;
    const PipelineHandle h = f.addEntry(1);
    f.post(h, f.obj(0), false, 7); // no such generation
    f.reg.drain();
    CHECK(f.reg.get(h) == nullptr);
    CHECK(f.releasedCount(0) == 1);
    // Invalid handle.
    f.post(99, f.obj(1), false);
    f.reg.drain();
    CHECK(f.releasedCount(1) == 1);
}

TEST_CASE("registry: hot reload commits atomically") {
    Fixture f;
    const PipelineHandle a = f.addEntry(1);
    const PipelineHandle b = f.addEntry(2);
    f.post(a, f.obj(0), false);
    f.post(b, f.obj(1), false);
    f.reg.drain();

    const u32 g = f.reg.beginGeneration();
    CHECK(g == 1);
    CHECK(f.reg.reloadPending());
    CHECK(f.reg.generation() == 0);

    f.post(a, f.obj(2), false, g, ArchiveOutcome::NotTried, 1, 1.0f);
    CHECK(f.reg.drain() == 0);
    CHECK(f.reg.get(a) == f.obj(0)); // still the old one: b is not ready
    CHECK(f.reg.reloadPending());

    f.post(b, f.obj(3), false, g);
    CHECK(f.reg.drain() == 2);
    CHECK_FALSE(f.reg.reloadPending());
    CHECK(f.reg.generation() == g);
    CHECK(f.reg.get(a) == f.obj(2));
    CHECK(f.reg.get(b) == f.obj(3));
    CHECK(f.releasedCount(0) == 1);
    CHECK(f.releasedCount(1) == 1);
    CHECK(f.reg.stats().reloads == 1);
    CHECK(f.reg.stats().reloadFailures == 0);

    // A completion of the old generation now is stale.
    f.post(a, f.obj(4), false, 0);
    f.reg.drain();
    CHECK(f.releasedCount(4) == 1);
    CHECK(f.reg.get(a) == f.obj(2));
}

TEST_CASE("registry: hot reload abandoned on failure") {
    Fixture f;
    const PipelineHandle a = f.addEntry(1);
    const PipelineHandle b = f.addEntry(2);
    f.post(a, f.obj(0), false);
    f.post(b, f.obj(1), false);
    f.reg.drain();

    const u32 g = f.reg.beginGeneration();
    f.post(a, f.obj(2), false, g);
    f.post(b, nullptr, false, g);
    f.reg.drain();
    CHECK_FALSE(f.reg.reloadPending());
    CHECK(f.reg.generation() == 0);
    CHECK(f.reg.get(a) == f.obj(0));
    CHECK(f.reg.get(b) == f.obj(1));
    CHECK(f.releasedCount(2) == 1); // staged object released
    CHECK(f.releasedCount(0) == 0);
    CHECK(f.reg.stats().reloadFailures == 1);
    CHECK(f.reg.stats().reloads == 0);
    CHECK(f.reg.stats().failures == 1);

    // The abandoned generation's late results are dropped; a new reload works.
    f.post(b, f.obj(3), false, g);
    f.reg.drain();
    CHECK(f.releasedCount(3) == 1);
    const u32 g2 = f.reg.beginGeneration();
    CHECK(g2 > g);
    f.post(a, f.obj(4), false, g2);
    f.post(b, f.obj(5), false, g2);
    f.reg.drain();
    CHECK(f.reg.generation() == g2);
    CHECK(f.reg.get(a) == f.obj(4));
    CHECK(f.reg.stats().reloads == 1);
}

TEST_CASE("registry: entry added during a pending reload") {
    Fixture f;
    const PipelineHandle a = f.addEntry(1);
    f.post(a, f.obj(0), false);
    f.reg.drain();

    const u32 g = f.reg.beginGeneration();
    const PipelineHandle n = f.addEntry(2); // belongs to the pending generation
    f.post(n, f.obj(1), false, g);
    CHECK(f.reg.drain() == 1);              // applied at once
    CHECK(f.reg.get(n) == f.obj(1));
    CHECK(f.reg.reloadPending());           // a is still not reloaded
    CHECK(f.reg.get(a) == f.obj(0));

    f.post(a, f.obj(2), false, g);
    f.reg.drain();
    CHECK_FALSE(f.reg.reloadPending());
    CHECK(f.reg.get(a) == f.obj(2));
    CHECK(f.reg.get(n) == f.obj(1)); // not swapped again
    CHECK(f.releasedCount(1) == 0);
    CHECK(f.reg.stats().reloads == 1);
    // Its later completions carry the served generation.
    f.post(n, f.obj(3), false, g);
    f.reg.drain();
    CHECK(f.reg.get(n) == f.obj(3));
    CHECK(f.releasedCount(1) == 1);
}

TEST_CASE("registry: reload with nothing to reload commits at next drain") {
    Fixture f;
    const u32 g = f.reg.beginGeneration();
    f.reg.drain();
    CHECK(f.reg.generation() == g);
    CHECK(f.reg.stats().reloads == 1);
}

TEST_CASE("registry: every dropped object is released exactly once") {
    Fixture f;
    {
        PipelineRegistry r(4);
        r.setReleaser(
            [](void* ctx, void* obj) { ++static_cast<Fixture*>(ctx)->released[obj]; }, &f);
        PipelineDesc d;
        const PipelineHandle a = r.add(1, d);
        const PipelineHandle b = r.add(2, d);
        auto post = [&](PipelineHandle h, int o, bool fallback, u32 gen) {
            Completion c;
            c.handle     = h;
            c.object     = f.obj(o);
            c.fallback   = fallback;
            c.generation = gen;
            r.post(c);
        };
        post(a, 0, true, 0);
        post(a, 1, true, 0);  // replaces fallback 0
        post(a, 2, false, 0); // final; drops fallback 1
        post(a, 3, true, 0);  // late fallback
        post(b, 4, false, 0);
        post(a, 5, false, 9); // stale
        r.drain();
        const u32 g = r.beginGeneration();
        post(a, 6, false, g);
        post(b, 7, false, g);
        r.drain();            // commits: drops 2 and 4
        post(a, 8, false, g); // replaces 6
        post(b, 9, true, g);  // late fallback
        r.drain();
        const u32 g3 = r.beginGeneration();
        post(a, 10, false, g3);
        r.drain();             // staged, not committed (b missing)
        post(a, 11, false, g3); // replaces staged 10
        r.drain();
        post(b, 12, false, g); // still queued at destruction
        // Destructor drops: staged 11, finals 8 and 7, queued 12.
    }
    for (int i = 0; i <= 12; ++i) {
        INFO("object " << i);
        CHECK(f.releasedCount(i) == 1);
    }
}

TEST_CASE("registry: concurrent post with drain on the render thread") {
    constexpr int kThreads   = 4;
    constexpr int kPerThread = 200;
    // One entry per completion so every final is distinct.
    std::vector<PipelineHandle> handles;
    std::vector<int>            storage(kThreads * kPerThread);
    PipelineRegistry            reg(kThreads * kPerThread);
    for (int i = 0; i < kThreads * kPerThread; ++i) {
        PipelineDesc d;
        handles.push_back(reg.add(static_cast<PipelineKey>(i) + 1, d));
    }
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int i = 0; i < kPerThread; ++i) {
                const int idx = t * kPerThread + i;
                Completion c;
                c.handle        = handles[static_cast<size_t>(idx)];
                c.object        = &storage[static_cast<size_t>(idx)];
                c.archive       = ArchiveOutcome::Miss;
                c.compilerCalls = 1;
                reg.post(c);
            }
        });
    }
    u32 changed = 0;
    while (changed < static_cast<u32>(kThreads * kPerThread)) changed += reg.drain();
    for (auto& th : threads) th.join();
    changed += reg.drain();
    CHECK(changed == static_cast<u32>(kThreads * kPerThread));
    for (int i = 0; i < kThreads * kPerThread; ++i) {
        CHECK(reg.get(handles[static_cast<size_t>(i)]) == &storage[static_cast<size_t>(i)]);
    }
    CHECK(reg.stats().compilerCalls == static_cast<u32>(kThreads * kPerThread));
    CHECK(reg.stats().archiveMisses == static_cast<u32>(kThreads * kPerThread));
}

TEST_CASE("pipeline stats: miss rate, format and JSON") {
    PipelineStats s;
    CHECK(s.archiveMissRate() == 0.0f);
    s.archiveHits        = 6;
    s.archiveMisses      = 1;
    s.archiveUnavailable = 1;
    CHECK(s.archiveMissRate() == doctest::Approx(0.25f));

    s.requests              = 8;
    s.compilerCalls         = 3;
    s.compileMs             = 12.5;
    s.compileMsMax          = 7.5f;
    s.renderThreadCompiles  = 1;
    s.renderThreadCompileMs = 2.0;
    s.fallbacksServed       = 2;
    s.fallbackDraws         = 40;
    s.failures              = 1;
    s.reloads               = 2;
    s.reloadFailures        = 1;
    const std::string line = formatPipelineStats(s);
    CHECK(line ==
          "PIPELINES requests 8 | archive hits 6, misses 1, unavailable 1 (miss rate 25.0%) | "
          "compiler calls 3 (12.5 ms, max 7.5 ms) | render-thread compiles 1 (2.0 ms) | "
          "fallbacks 2 (draws 40) | failures 1 | reloads 2 (failed 1)");
    CHECK(line.find('\n') == std::string::npos);

    const std::string json = pipelineStatsJson(s);
    CHECK(json.front() == '{');
    CHECK(json.back() == '}');
    for (const char* field : {"requests", "archiveHits", "archiveMisses", "archiveUnavailable",
                              "archiveMissRate", "compilerCalls", "compileMs", "compileMsMax",
                              "fallbacksServed", "fallbackDraws", "renderThreadCompiles",
                              "renderThreadCompileMs", "failures", "swaps", "reloads",
                              "reloadFailures"}) {
        CHECK(json.find(std::string("\"") + field + "\":") != std::string::npos);
    }
    CHECK(json.find("\"fallbackDraws\":40") != std::string::npos);
    CHECK(json.find("\"archiveMissRate\":0.2500") != std::string::npos);
}

TEST_CASE("registry: stats counters and render-thread compile") {
    Fixture f;
    const PipelineHandle h = f.addEntry(1);
    f.reg.noteFallbackUse();
    f.reg.noteFallbackUse();
    f.reg.recordRenderThreadCompile(3.5f);
    CHECK(f.reg.stats().fallbackDraws == 2);
    CHECK(f.reg.stats().renderThreadCompiles == 1);
    CHECK(f.reg.stats().renderThreadCompileMs == doctest::Approx(3.5));
    f.post(h, nullptr, true); // failed fallback
    f.reg.drain();
    CHECK(f.reg.stats().failures == 1);
    CHECK(f.reg.state(h) == PipelineState::Pending);
}
