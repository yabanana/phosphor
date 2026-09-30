// OPT-1 spike 2 -- render graph order optimisation: greedy vs exact dynamic
// programming over downsets vs simulated annealing vs a MILP (HiGHS), on the
// graph scenarios of rendergraph/scenario.h with 1..6 views (12..150 passes).
//
// Not engine code: a measurement probe (build commands in README.md).
//
// Every order is judged by the SAME black-box evaluator: the real graph
// compiler (rg::compile with CompileOptions::order: fusion, load/store,
// memoryless, aliasing with a fake sizer, barriers) plus a cost model:
//   T  = sum over timed units of max(DRAM bytes / BW, ALU + geometry time) + fixed
//   J  = T + gamma * transient heap
// The searches optimise their own models (DP: group costs + peak live
// bytes; MILP: pairwise fusion savings + peak live bytes); their orders are
// then evaluated by the black box, so the comparison is fair.
//
// Model numbers (M5 Max, docs/soc-model.md and docs/opt-log.md "OPT-1"):
//   BW 569 GB/s (B-08), IMAD 4.06 Top/s (B-01 i32 mul: one LCG step is an
//   IMAD), 8 Gtri/s (B-16 raster 7.9-9.5), render pass 11 us chained (B-14),
//   dependent compute pass ~7 us (OPT-1 spike 1: tiny chained dispatches).

#include "rendergraph/aliasing.h"
#include "rendergraph/graph_dump.h"
#include "rendergraph/render_graph.h"
#include "rendergraph/scenario.h"
#include "rendergraph/timing_plan.h"

#ifdef OPT1_WITH_HIGHS
#include "Highs.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace phosphor;
using namespace phosphor::rg;

namespace {

constexpr double kBw        = 569e9;   // B/s
constexpr double kImad      = 4.06e12; // IMAD/s
constexpr double kTris      = 8e9;     // triangles/s
constexpr double kRenderFix = 11e-6;   // s per render pass
constexpr double kCompFix   = 7e-6;    // s per compute pass
constexpr double kGammaMsPerGiB = 0.1; // memory weight in J
// Base shader cost per invocation (hashing), IMAD-equivalents: measured
// Present (5.76 Mpx, 0 ALU steps, 1 input) = 0.147 ms -> ~100 ops/px.
constexpr double kBaseOps   = 100.0;

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class FakeSizer final : public ResourceSizer {
public:
    SizeAlign textureSize(u32, const TextureDesc& d) const override {
        const u64 a = 64 * 1024; // heap alignment of large private textures (typical)
        return {(d.estimatedBytes() + a - 1) / a * a, a};
    }
    SizeAlign bufferSize(u32, const BufferDesc& d) const override { return {(d.size + 255) / 256 * 256, 256}; }
};

struct Problem {
    RenderGraph graph;
    Scenario    scenario;
    std::string name;
    std::vector<double> passSeconds; // ALU + geometry time of each pass (order-independent)
};

struct Eval {
    bool   ok = false;
    double bytes = 0, heap = 0, timeMs = 0, J = 0;
    double maxLive = 0; // max over positions of placed bytes alive (lower bound of any packing)
    u32    groups = 0, memoryless = 0, encoders = 0;
};

Eval evaluate(const Problem& pb, const std::vector<u32>& order) {
    FakeSizer sizer;
    CompileOptions opt;
    opt.sizer = &sizer;
    opt.order = order;
    const CompiledGraph c = compile(pb.graph, opt);
    Eval e;
    if (!c.ok) return e;
    e.ok = true;
    const std::vector<PassTraffic> traffic = estimatePassBandwidth(pb.graph, c);
    const TimingPlan plan = buildTimingPlan(pb.graph, c, 16);
    double t = 0;
    for (const TimedUnit& u : plan.units) {
        double alu = 0;
        for (const u32 p : u.passes) alu += pb.passSeconds[p];
        const double mem = static_cast<double>(u.dramBytes) / kBw;
        t += std::max(mem, alu) + (u.kind == TimestampKind::RenderEnd ? kRenderFix : kCompFix);
    }
    for (const PassTraffic& tr : traffic) e.bytes += static_cast<double>(tr.total());
    e.heap       = static_cast<double>(c.aliasing.heapSize);
    for (u32 pos = 0; pos < c.order.size(); ++pos) {
        double live = 0;
        for (const Placement& pl : c.aliasing.placements) {
            Lifetime l = c.lifetimes[pl.resource];
            bool async = false;
            for (const PassNode& p : pb.graph.passes()) {
                if (p.queue != Queue::AsyncCompute) continue;
                for (const Access& a : p.reads) async |= a.resource == pl.resource;
                for (const Access& a : p.writes) async |= a.resource == pl.resource;
            }
            if (async || (l.first <= pos && pos <= l.last)) live += static_cast<double>(pl.size);
        }
        e.maxLive = std::max(e.maxLive, live);
    }
    e.timeMs     = t * 1e3;
    e.J          = e.timeMs + kGammaMsPerGiB * e.heap / (1024.0 * 1024 * 1024);
    e.groups     = static_cast<u32>(c.renderGroups.size());
    e.encoders   = static_cast<u32>(c.encoders.size());
    for (const bool m : c.memoryless) e.memoryless += m ? 1 : 0;
    return e;
}

bool buildProblem(Problem& pb, u32 scenario, u32 views) {
    ScenarioParams params;
    params.views = views;
    const TextureRef drawable = pb.graph.importTexture("Drawable", {Format::BGRA8Srgb, 3200, 1800},
                                                       ImportOutput | ImportPerFrame);
    std::string error;
    if (!buildScenario(scenario, params, pb.graph, drawable, pb.scenario, {}, &error)) {
        std::fprintf(stderr, "scenario: %s\n", error.c_str());
        return false;
    }
    pb.name = std::string(scenarioName(scenario)) + " x" + std::to_string(views);
    pb.passSeconds.assign(pb.graph.passes().size(), 0.0);
    for (u32 p = 0; p < pb.scenario.synth.size(); ++p) {
        const SynthPass& s = pb.scenario.synth[p];
        const SynthWork w = synthWork(s);
        const double ops = w.intOps; // IMADs incl. the per-invocation base (synthWork)
        (void)kBaseOps;
        pb.passSeconds[p] = ops / kImad + w.triangles / kTris;
    }
    return true;
}

// --- Graph structure for the searches ---------------------------------------

struct Dag {
    u32 n = 0;
    std::vector<u32> live;                 // live pass indices
    std::vector<u32> index;                // pass -> index in live (or ~0u)
    std::vector<std::vector<u32>> preds;   // over live indices
    std::vector<std::vector<u32>> succs;
    std::vector<u32> greedy;               // default compile order (live indices)
};

Dag buildDag(const Problem& pb) {
    Dag d;
    const CompiledGraph c = compileOrder(pb.graph);
    d.index.assign(pb.graph.passes().size(), ~0u);
    for (const u32 p : c.order) {
        d.index[p] = static_cast<u32>(d.live.size());
        d.live.push_back(p);
    }
    d.n = static_cast<u32>(d.live.size());
    d.preds.resize(d.n);
    d.succs.resize(d.n);
    for (const Dependency& dep : c.dependencies) {
        const u32 a = d.index[dep.from], b = d.index[dep.to];
        if (std::find(d.preds[b].begin(), d.preds[b].end(), a) == d.preds[b].end()) {
            d.preds[b].push_back(a);
            d.succs[a].push_back(b);
        }
    }
    for (u32 i = 0; i < d.n; ++i) d.greedy.push_back(i);
    return d;
}

std::vector<u32> toPasses(const Dag& d, const std::vector<u32>& idx) {
    std::vector<u32> out;
    for (const u32 i : idx) out.push_back(d.live[i]);
    return out;
}

// --- Simulated annealing over topological orders (black-box J) --------------

std::vector<u32> anneal(const Problem& pb, const Dag& d, std::vector<u32> order, u32 iterations, u32 seed,
                        double& bestJ, u32& evals) {
    std::mt19937 rng(seed);
    std::vector<u32> pos(d.n);
    auto reindex = [&] { for (u32 k = 0; k < d.n; ++k) pos[order[k]] = k; };
    reindex();
    double cur = evaluate(pb, toPasses(d, order)).J;
    bestJ = cur;
    std::vector<u32> best = order;
    evals = 1;
    const double t0 = 0.02, t1 = 0.0002; // ms
    for (u32 it = 0; it < iterations; ++it) {
        const double temp = t0 * std::pow(t1 / t0, static_cast<double>(it) / iterations);
        const u32 v = rng() % d.n;
        u32 lo = 0, hi = d.n - 1;
        for (const u32 p : d.preds[v]) lo = std::max(lo, pos[p] + 1);
        for (const u32 s : d.succs[v]) hi = std::min(hi, pos[s] - 1);
        if (hi <= lo) continue;
        u32 target = lo + rng() % (hi - lo + 1);
        if (target == pos[v]) continue;
        std::vector<u32> cand = order;
        cand.erase(cand.begin() + pos[v]);
        cand.insert(cand.begin() + target, v);
        const Eval e = evaluate(pb, toPasses(d, cand));
        ++evals;
        if (!e.ok) continue;
        const double delta = e.J - cur;
        if (delta <= 0 || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp(-delta / temp)) {
            order = std::move(cand);
            cur = e.J;
            reindex();
            if (cur < bestJ) {
                bestJ = cur;
                best = order;
            }
        }
    }
    return best;
}

// --- Exact DP over downsets (incremental group model) ------------------------

struct Bits {
    u64 w[3] = {0, 0, 0};
    void set(u32 i) { w[i >> 6] |= 1ull << (i & 63); }
    [[nodiscard]] bool test(u32 i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    bool operator==(const Bits& o) const { return w[0] == o.w[0] && w[1] == o.w[1] && w[2] == o.w[2]; }
};
struct BitsHash {
    size_t operator()(const Bits& b) const { return b.w[0] * 0x9E3779B97F4A7C15ull ^ b.w[1] * 0xC2B2AE3D27D4EB4Full ^ b.w[2]; }
};

// Group cost (seconds): attachment loads/stores of a render group made of
// `members` (live indices, in order) + the fixed render pass cost.  Mirrors
// tbdr_passes.cpp (loads: first access Load; stores: final version read
// outside the group or ImportOutput).
struct GroupModel {
    const Problem& pb;
    const Dag&     d;
    double cost(const std::vector<u32>& members, double* memorylessBytes = nullptr) const {
        const auto& passes = pb.graph.passes();
        const auto& res    = pb.graph.resources();
        struct A { u32 r; bool load; u32 finalVersion; bool onlyAttachment; };
        std::vector<A> atts;
        auto find = [&](u32 r) -> A* { for (A& a : atts) if (a.r == r) return &a; return nullptr; };
        for (const u32 m : members) {
            const PassNode& p = passes[d.live[m]];
            for (const Access& w : p.writes) {
                if (!isAttachment(w.usage)) continue;
                A* a = find(w.resource);
                if (!a) atts.push_back({w.resource, w.load == LoadIntent::Preserve, w.version, true});
                else a->finalVersion = std::max(a->finalVersion, w.version);
            }
            for (const Access& r : p.reads) {
                if (r.usage != Usage::DepthRead && r.usage != Usage::ColorAttachment) continue;
                A* a = find(r.resource);
                if (!a) atts.push_back({r.resource, true, r.version, true});
                else a->finalVersion = std::max(a->finalVersion, r.version);
            }
        }
        double bytes = 0, ml = 0;
        for (A& a : atts) {
            const ResourceNode& node = res[a.r];
            const double size = static_cast<double>(node.texture.estimatedBytes());
            bool store = node.imported && (node.importFlags & ImportOutput);
            bool outside = false; // any access outside the group (memoryless check)
            for (u32 q = 0; q < d.n && !store; ++q) {
                if (std::find(members.begin(), members.end(), q) != members.end()) continue;
                const PassNode& p = passes[d.live[q]];
                for (const Access& r : p.reads) {
                    if (r.resource != a.r) continue;
                    outside = true;
                    if (r.version == a.finalVersion) store = true;
                }
                for (const Access& w : p.writes) outside |= w.resource == a.r;
            }
            if (a.load) bytes += size;
            if (store) bytes += size;
            if (!a.load && !store && !outside && !node.imported) ml += size;
        }
        if (memorylessBytes) *memorylessBytes = ml;
        return bytes / kBw + kRenderFix;
    }
};

// Can raster pass `p` join the open group `members`?  Same rules as
// tbdr_passes.cpp canJoin (size, slots, no non-attachment use of anything the
// group wrote, no non-attachment write of what it read, no mid-group clear).
bool canJoin(const Problem& pb, const Dag& d, const std::vector<u32>& members, u32 q) {
    if (members.empty()) return false;
    const auto& passes = pb.graph.passes();
    const auto& res    = pb.graph.resources();
    const PassNode& pass = passes[d.live[q]];
    if (pass.type != PassType::Raster || pass.queue != Queue::Graphics) return false;
    struct B { u32 r; bool depth; u32 slot; };
    auto bindings = [&](const PassNode& p) {
        std::vector<B> out;
        auto add = [&](u32 r, bool depth, u32 slot) {
            for (const B& b : out) if (b.r == r) return;
            out.push_back({r, depth, slot});
        };
        for (const Access& w : p.writes) if (isAttachment(w.usage)) add(w.resource, w.usage == Usage::DepthAttachment, w.slot);
        for (const Access& r : p.reads) {
            if (r.usage == Usage::DepthRead) add(r.resource, true, 0);
            if (r.usage == Usage::ColorAttachment) add(r.resource, false, r.slot);
        }
        return out;
    };
    const std::vector<B> mine = bindings(pass);
    if (mine.empty()) return false;
    const TextureDesc& t0 = res[mine[0].r].texture;
    std::vector<u32> written, writtenNA, readNA;
    std::vector<B> group;
    for (const u32 m : members) {
        const PassNode& p = passes[d.live[m]];
        for (const B& b : bindings(p)) {
            const TextureDesc& t = res[b.r].texture;
            if (t.width != t0.width || t.height != t0.height || std::max(t.sampleCount, 1u) != std::max(t0.sampleCount, 1u)) return false;
            bool known = false;
            for (const B& g : group) known |= g.r == b.r;
            if (!known) group.push_back(b);
        }
        for (const Access& r : p.reads) if (!isAttachment(r.usage)) readNA.push_back(r.resource);
        for (const Access& w : p.writes) {
            written.push_back(w.resource);
            if (!isAttachment(w.usage)) writtenNA.push_back(w.resource);
        }
    }
    auto has = [](const std::vector<u32>& v, u32 x) { return std::find(v.begin(), v.end(), x) != v.end(); };
    for (const B& a : mine) {
        for (const B& b : group) {
            if (a.r == b.r) {
                if (a.depth != b.depth || (!a.depth && a.slot != b.slot)) return false;
            } else if (a.depth && b.depth) {
                return false;
            } else if (!a.depth && !b.depth && a.slot == b.slot) {
                return false;
            }
        }
    }
    for (const Access& r : pass.reads) {
        if (isAttachment(r.usage)) {
            if (has(writtenNA, r.resource)) return false;
        } else if (has(written, r.resource)) {
            return false;
        }
    }
    for (const Access& w : pass.writes) {
        if (has(readNA, w.resource)) return false;
        if (isAttachment(w.usage) && w.load == LoadIntent::Clear) {
            for (const B& b : group) if (b.r == w.resource) return false;
        }
    }
    return true;
}

// Transient bytes live while pass q runs after `done` (conservative: counts
// resources that end up memoryless).
struct LiveModel {
    std::vector<double> size;                  // per resource (transients only)
    std::vector<u32>    producer;              // live index or ~0u
    std::vector<std::vector<u32>> consumers;   // live indices
    double liveDuring(const Bits& done, u32 q) const {
        double sum = 0;
        for (u32 r = 0; r < size.size(); ++r) {
            if (size[r] == 0 || producer[r] == ~0u) continue;
            const bool produced = producer[r] == q || done.test(producer[r]);
            if (!produced) continue;
            bool needed = producer[r] == q;
            for (const u32 c : consumers[r]) needed |= c == q || !done.test(c);
            if (needed) sum += size[r];
        }
        return sum;
    }
};

LiveModel buildLive(const Problem& pb, const Dag& d) {
    LiveModel m;
    const auto& res = pb.graph.resources();
    m.size.assign(res.size(), 0);
    m.producer.assign(res.size(), ~0u);
    m.consumers.resize(res.size());
    for (u32 r = 0; r < res.size(); ++r) {
        if (!res[r].imported) m.size[r] = static_cast<double>(res[r].kind == ResourceKind::Texture ? res[r].texture.estimatedBytes() : res[r].buffer.size);
    }
    for (u32 i = 0; i < d.n; ++i) {
        const PassNode& p = pb.graph.passes()[d.live[i]];
        for (const Access& w : p.writes) if (m.producer[w.resource] == ~0u) m.producer[w.resource] = i;
        for (const Access& r : p.reads) m.consumers[r.resource].push_back(i);
        for (const Access& w : p.writes) m.consumers[w.resource].push_back(i);
    }
    return m;
}

struct DpResult {
    std::vector<u32> order;
    bool exact = true;
    size_t states = 0;
};

// DP layered by |done|: state = (done set, open group members); value =
// (additive group cost, peak live bytes) combined as J; the open group is
// part of the key so that fusion follows the compiler exactly.  `cap`:
// maximum states kept per layer (beam when exceeded: result not exact).
DpResult dpSearch(const Problem& pb, const Dag& d, size_t cap) {
    const GroupModel gm{pb, d};
    const LiveModel lm = buildLive(pb, d);
    struct State {
        Bits done;
        std::vector<u32> group;  // open raster group (live indices, in order)
        double add  = 0;         // closed group costs (s)
        double peak = 0;         // bytes
        u32 parent = ~0u, pass = ~0u;
        [[nodiscard]] double J() const { return add * 1e3 + kGammaMsPerGiB * peak / (1024.0 * 1024 * 1024); }
    };
    struct KeyHash {
        size_t operator()(const std::pair<Bits, std::vector<u32>>& k) const {
            size_t h = BitsHash{}(k.first);
            for (const u32 g : k.second) h = h * 31 + g;
            return h;
        }
    };
    std::vector<std::vector<State>> layers(d.n + 1);
    layers[0].push_back(State{});
    DpResult out;
    for (u32 depth = 0; depth < d.n; ++depth) {
        std::unordered_map<std::pair<Bits, std::vector<u32>>, u32, KeyHash> seen;
        std::vector<State>& next = layers[depth + 1];
        for (u32 si = 0; si < layers[depth].size(); ++si) {
            const State& s = layers[depth][si];
            for (u32 q = 0; q < d.n; ++q) {
                if (s.done.test(q)) continue;
                bool ready = true;
                for (const u32 p : d.preds[q]) ready &= s.done.test(p);
                if (!ready) continue;
                State t;
                t.done = s.done;
                t.done.set(q);
                t.add  = s.add;
                t.parent = si;
                t.pass = q;
                const bool raster = pb.graph.passes()[d.live[q]].type == PassType::Raster;
                if (raster && canJoin(pb, d, s.group, q)) {
                    t.group = s.group;
                    t.group.push_back(q);
                } else {
                    if (!s.group.empty()) t.add += gm.cost(s.group);
                    if (raster) t.group = {q};
                }
                if (!raster) t.add += 0; // compute costs are order-independent
                t.peak = std::max(s.peak, lm.liveDuring(s.done, q));
                if (depth + 1 == d.n && !t.group.empty()) {
                    t.add += gm.cost(t.group);
                    t.group.clear();
                }
                auto key = std::make_pair(t.done, t.group);
                auto it = seen.find(key);
                if (it == seen.end()) {
                    seen.emplace(std::move(key), static_cast<u32>(next.size()));
                    next.push_back(std::move(t));
                } else if (t.J() < next[it->second].J()) {
                    next[it->second] = std::move(t);
                }
            }
        }
        out.states += next.size();
        if (next.size() > cap) {
            out.exact = false;
            std::nth_element(next.begin(), next.begin() + cap, next.end(),
                             [](const State& a, const State& b) { return a.J() < b.J(); });
            next.resize(cap);
        }
    }
    // Best final state, then walk back.
    const std::vector<State>& last = layers[d.n];
    u32 best = 0;
    for (u32 i = 1; i < last.size(); ++i) if (last[i].J() < last[best].J()) best = i;
    std::vector<u32> rev;
    u32 idx = best;
    for (u32 depth = d.n; depth > 0; --depth) {
        const State& s = layers[depth][idx];
        rev.push_back(s.pass);
        idx = s.parent;
    }
    out.order.assign(rev.rbegin(), rev.rend());
    return out;
}

#ifdef OPT1_WITH_HIGHS
// --- MILP (HiGHS): assignment x[p][k]; fusion pairs y[a][b] (b right after a)
// rewarded with their standalone saving; peak >= live bytes at every k.
std::vector<u32> milpSearch(const Problem& pb, const Dag& d, double timeLimit, double& gap, bool& optimal,
                            u32& rows, u32& cols) {
    const u32 n = d.n;
    const GroupModel gm{pb, d};
    const LiveModel lm = buildLive(pb, d);
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("time_limit", timeLimit);
    h.setOptionValue("threads", 1);
    HighsModel model;
    HighsLp& lp = model.lp_;
    auto addCol = [&](double cost, double lo, double hi, bool integer) {
        lp.col_cost_.push_back(cost);
        lp.col_lower_.push_back(lo);
        lp.col_upper_.push_back(hi);
        lp.integrality_.push_back(integer ? HighsVarType::kInteger : HighsVarType::kContinuous);
        return static_cast<int>(lp.col_cost_.size() - 1);
    };
    std::vector<std::vector<int>> x(n, std::vector<int>(n));
    for (u32 p = 0; p < n; ++p) for (u32 k = 0; k < n; ++k) x[p][k] = addCol(0, 0, 1, true);
    struct Row { std::vector<int> idx; std::vector<double> val; double lo, hi; };
    std::vector<Row> rowsV;
    // Assignment.
    for (u32 p = 0; p < n; ++p) {
        Row r{{}, {}, 1, 1};
        for (u32 k = 0; k < n; ++k) { r.idx.push_back(x[p][k]); r.val.push_back(1); }
        rowsV.push_back(r);
    }
    for (u32 k = 0; k < n; ++k) {
        Row r{{}, {}, 1, 1};
        for (u32 p = 0; p < n; ++p) { r.idx.push_back(x[p][k]); r.val.push_back(1); }
        rowsV.push_back(r);
    }
    // Precedence: q at <= k only if p at < k.
    for (u32 q = 0; q < n; ++q) {
        for (const u32 p : d.preds[q]) {
            for (u32 k = 0; k < n; ++k) {
                Row r{{}, {}, -kHighsInf, 0};
                for (u32 j = 0; j <= k; ++j) { r.idx.push_back(x[q][j]); r.val.push_back(1); }
                for (u32 j = 0; j < k; ++j) { r.idx.push_back(x[p][j]); r.val.push_back(-1); }
                rowsV.push_back(r);
            }
        }
    }
    // Fusion pairs: saving = cost(a) + cost(b) - cost(a+b) when they can fuse.
    const double scale = 1e3; // objective in ms
    for (u32 a = 0; a < n; ++a) {
        if (pb.graph.passes()[d.live[a]].type != PassType::Raster) continue;
        for (u32 b = 0; b < n; ++b) {
            if (a == b || pb.graph.passes()[d.live[b]].type != PassType::Raster) continue;
            if (!canJoin(pb, d, {a}, b)) continue;
            const double saving = gm.cost({a}) + gm.cost({b}) - gm.cost({a, b});
            if (saving <= 0) continue;
            const int y = addCol(-saving * scale, 0, 1, true);
            // y <= sum_k z_k, z_k <= x[a][k], z_k <= x[b][k+1]
            Row link{{y}, {1}, -kHighsInf, 0};
            for (u32 k = 0; k + 1 < n; ++k) {
                const int z = addCol(0, 0, 1, false);
                link.idx.push_back(z);
                link.val.push_back(-1);
                rowsV.push_back({{z, x[a][k]}, {1, -1}, -kHighsInf, 0});
                rowsV.push_back({{z, x[b][k + 1]}, {1, -1}, -kHighsInf, 0});
            }
            rowsV.push_back(link);
        }
    }
    // Peak live bytes (GiB units): live[r][k] >= A_rk + B_rk - 1 with
    // A_rk = [producer at <= k], B_rk >= [some consumer at >= k].
    const int peak = addCol(kGammaMsPerGiB, 0, kHighsInf, false);
    const double gib = 1024.0 * 1024 * 1024;
    std::vector<std::vector<int>> live(lm.size.size());
    for (u32 r = 0; r < lm.size.size(); ++r) {
        if (lm.size[r] == 0 || lm.producer[r] == ~0u || lm.consumers[r].empty()) continue;
        live[r].resize(n);
        for (u32 k = 0; k < n; ++k) {
            const int B = addCol(0, 0, 1, false);
            for (const u32 c : lm.consumers[r]) {
                Row rb{{B}, {1}, 0, kHighsInf};
                for (u32 j = k; j < n; ++j) { rb.idx.push_back(x[c][j]); rb.val.push_back(-1); }
                rowsV.push_back(rb);
            }
            const int L = addCol(0, 0, 1, false);
            Row rl{{L, B}, {1, -1}, -1, kHighsInf};
            for (u32 j = 0; j <= k; ++j) { rl.idx.push_back(x[lm.producer[r]][j]); rl.val.push_back(-1); }
            rowsV.push_back(rl);
            live[r][k] = L;
        }
    }
    for (u32 k = 0; k < n; ++k) {
        Row rp{{peak}, {1}, 0, kHighsInf};
        for (u32 r = 0; r < live.size(); ++r) {
            if (live[r].empty()) continue;
            rp.idx.push_back(live[r][k]);
            rp.val.push_back(-lm.size[r] / gib);
        }
        rowsV.push_back(rp);
    }
    if (std::getenv("MILP_CHECK")) {
        // Debug: the greedy order must satisfy every row.
        std::vector<double> v(lp.col_cost_.size(), 0.0);
        for (u32 p = 0; p < n; ++p) v[x[p][p]] = 1;
        // B/L/peak: iterate rows to raise lower-bounded auxiliaries (few passes suffice).
        for (int pass = 0; pass < 4; ++pass) {
            for (const Row& r : rowsV) {
                double a = 0;
                for (size_t i = 0; i < r.idx.size(); ++i) a += r.val[i] * v[r.idx[i]];
                if (a < r.lo - 1e-9 && r.val[0] > 0) v[r.idx[0]] += (r.lo - a) / r.val[0];
            }
        }
        for (size_t ri = 0; ri < rowsV.size(); ++ri) {
            const Row& r = rowsV[ri];
            double a = 0;
            for (size_t i = 0; i < r.idx.size(); ++i) a += r.val[i] * v[r.idx[i]];
            if (a < r.lo - 1e-6 || a > r.hi + 1e-6) {
                std::fprintf(stderr, "  [milp check] row %zu violated: %g not in [%g, %g] (first col %d)\n", ri, a, r.lo, r.hi, r.idx.empty() ? -1 : r.idx[0]);
                break;
            }
        }
        for (size_t c = 0; c < v.size(); ++c) {
            if (v[c] > lp.col_upper_[c] + 1e-9) { std::fprintf(stderr, "  [milp check] col %zu = %g above upper %g\n", c, v[c], lp.col_upper_[c]); break; }
        }
    }
    const HighsInt numCol = static_cast<HighsInt>(lp.col_cost_.size());
    h.addCols(numCol, lp.col_cost_.data(), lp.col_lower_.data(), lp.col_upper_.data(), 0, nullptr, nullptr, nullptr);
    std::vector<HighsInt> integerCols;
    for (HighsInt c = 0; c < numCol; ++c) if (lp.integrality_[c] == HighsVarType::kInteger) integerCols.push_back(c);
    std::vector<HighsVarType> kinds(integerCols.size(), HighsVarType::kInteger);
    h.changeColsIntegrality(static_cast<HighsInt>(integerCols.size()), integerCols.data(), kinds.data());
    std::vector<double> lower, upper, values;
    std::vector<HighsInt> starts, indices;
    for (const Row& r : rowsV) {
        starts.push_back(static_cast<HighsInt>(indices.size()));
        for (size_t i = 0; i < r.idx.size(); ++i) {
            indices.push_back(r.idx[i]);
            values.push_back(r.val[i]);
        }
        lower.push_back(r.lo);
        upper.push_back(r.hi);
    }
    h.addRows(static_cast<HighsInt>(rowsV.size()), lower.data(), upper.data(), static_cast<HighsInt>(indices.size()),
              starts.data(), indices.data(), values.data());
    rows = static_cast<u32>(rowsV.size());
    cols = static_cast<u32>(numCol);
    const HighsStatus passed = HighsStatus::kOk;
    const HighsStatus ran = h.run();
    std::fprintf(stderr, "  [milp] pass %d run %d model status '%s' primal status %d\n", static_cast<int>(passed),
                 static_cast<int>(ran), h.modelStatusToString(h.getModelStatus()).c_str(),
                 h.getInfo().primal_solution_status);
    const HighsInfo& info = h.getInfo();
    gap = info.mip_gap;
    optimal = h.getModelStatus() == HighsModelStatus::kOptimal;
    std::vector<u32> order(n, ~0u);
    if (info.primal_solution_status != 2) return {}; // no feasible solution
    const std::vector<double>& sol = h.getSolution().col_value;
    for (u32 p = 0; p < n; ++p) {
        for (u32 k = 0; k < n; ++k) {
            if (sol[x[p][k]] > 0.5) order[k] = p;
        }
    }
    for (const u32 o : order) if (o == ~0u) return {};
    return order;
}
#endif

void report(const char* method, const Problem& pb, const Dag& d, const std::vector<u32>& order, double seconds,
            const char* note, const Eval& ref) {
    const Eval e = evaluate(pb, toPasses(d, order));
    std::printf("  %-10s %s J %8.4f  T %7.4f ms  DRAM %7.1f MiB (%+5.1f%%)  heap %6.1f MiB (%+5.1f%%, live max %6.1f)  groups %2u  memoryless %2u  %8.3f s  %s\n",
                method, e.ok ? "ok " : "BAD", e.J, e.timeMs, e.bytes / (1 << 20), (e.bytes / ref.bytes - 1) * 100,
                e.heap / (1 << 20), (e.heap / ref.heap - 1) * 100, e.maxLive / (1 << 20), e.groups, e.memoryless,
                seconds, note);
}

} // namespace

int main(int argc, char** argv) {
    const u32 scenarioArg = argc > 1 ? static_cast<u32>(std::atoi(argv[1])) : 0;
    const u32 views       = argc > 2 ? static_cast<u32>(std::atoi(argv[2])) : 1;
    const size_t cap      = argc > 3 ? static_cast<size_t>(std::atoll(argv[3])) : 200000;
    const u32 annealIt    = argc > 4 ? static_cast<u32>(std::atoi(argv[4])) : 20000;
    const double milpLimit = argc > 5 ? std::atof(argv[5]) : 60.0;

    Problem pb;
    if (!buildProblem(pb, scenarioArg, views)) return 2;
    const Dag d = buildDag(pb);
    std::printf("== %s: %u live passes, %zu resources\n", pb.name.c_str(), d.n, pb.graph.resources().size());
    const Eval greedy = evaluate(pb, toPasses(d, d.greedy));
    report("greedy", pb, d, d.greedy, 0, "declaration-order Kahn (F2)", greedy);

    double t = now();
    const DpResult dp = dpSearch(pb, d, cap);
    char note[128];
    std::snprintf(note, sizeof note, "%s, %zu states", dp.exact ? "exact" : "beam (cap hit)", dp.states);
    report("dp", pb, d, dp.order, now() - t, note, greedy);

    t = now();
    double bestJ = 0;
    u32 evals = 0;
    const std::vector<u32> sa = anneal(pb, d, d.greedy, annealIt, 1234, bestJ, evals);
    std::snprintf(note, sizeof note, "%u evaluations", evals);
    report("anneal", pb, d, sa, now() - t, note, greedy);

    t = now();
    const std::vector<u32> sa2 = anneal(pb, d, dp.order, annealIt, 99, bestJ, evals);
    std::snprintf(note, sizeof note, "from dp, %u evaluations", evals);
    report("dp+anneal", pb, d, sa2, now() - t, note, greedy);

#ifdef OPT1_WITH_HIGHS
    t = now();
    double gap = 0;
    bool optimal = false;
    u32 rows = 0, cols = 0;
    const std::vector<u32> milp = milpSearch(pb, d, milpLimit, gap, optimal, rows, cols);
    std::snprintf(note, sizeof note, "%s, gap %.3g, %u rows x %u cols", optimal ? "optimal" : "time limit", gap, rows, cols);
    if (milp.empty()) {
        std::printf("  %-10s no solution in %.1f s (%s)\n", "milp", now() - t, note);
    } else {
        report("milp", pb, d, milp, now() - t, note, greedy);
    }
#endif
    return 0;
}
