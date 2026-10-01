#include "rendergraph/optimizer/optimizer.h"
#include "rendergraph/tbdr_passes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <unordered_map>

namespace phosphor::rg {

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

u64 alignUp(u64 v, u64 a) { return (v + a - 1) / a * a; }

std::vector<std::string> namesOf(const RenderGraph& graph, const std::vector<u32>& order) {
    std::vector<std::string> names;
    for (const u32 p : order) names.push_back(graph.passes()[p].name);
    return names;
}

// --- Live passes and their dependencies -------------------------------------

struct Dag {
    u32 n = 0;
    std::vector<u32> live;                // pass index of live index i (greedy order)
    std::vector<u32> index;               // pass -> live index or ~0u
    std::vector<std::vector<u32>> preds;  // live indices
    std::vector<std::vector<u32>> succs;
};

Dag buildDag(const RenderGraph& g) {
    Dag d;
    const CompiledGraph c = compileOrder(g);
    d.index.assign(g.passes().size(), ~0u);
    for (const u32 p : c.order) {
        d.index[p] = static_cast<u32>(d.live.size());
        d.live.push_back(p);
    }
    d.n = static_cast<u32>(d.live.size());
    d.preds.resize(d.n);
    d.succs.resize(d.n);
    for (const Dependency& dep : c.dependencies) {
        const u32 a = d.index[dep.from], b = d.index[dep.to];
        if (a == ~0u || b == ~0u) continue;
        if (std::find(d.preds[b].begin(), d.preds[b].end(), a) == d.preds[b].end()) {
            d.preds[b].push_back(a);
            d.succs[a].push_back(b);
        }
    }
    return d;
}

std::vector<u32> toPasses(const Dag& d, const std::vector<u32>& idx) {
    std::vector<u32> out;
    out.reserve(idx.size());
    for (const u32 i : idx) out.push_back(d.live[i]);
    return out;
}

// --- DP over downsets ------------------------------------------------------------

constexpr u32 kWords = 4; // up to 256 live passes
struct Bits {
    std::array<u64, kWords> w{};
    void set(u32 i) { w[i >> 6] |= 1ull << (i & 63); }
    [[nodiscard]] bool test(u32 i) const { return (w[i >> 6] >> (i & 63)) & 1; }
    bool operator==(const Bits&) const = default;
};

// Attachment load/store bytes of a render group (same rules as
// tbdr_passes.cpp: load if the first access loads; store if the version the
// group leaves is read outside it or the resource is an output) + the fixed
// cost of a render pass, in ms; `memorylessBytes` of textures that never
// leave the tile.
struct GroupModel {
    const RenderGraph& graph;
    const GraphCostParams& params;

    [[nodiscard]] double cost(const std::vector<u32>& members) const {
        const auto& passes = graph.passes();
        const auto& res    = graph.resources();
        struct A { u32 r; bool load; u32 finalVersion; };
        std::vector<A> atts;
        auto find = [&](u32 r) -> A* {
            for (A& a : atts) if (a.r == r) return &a;
            return nullptr;
        };
        for (const u32 m : members) {
            const PassNode& p = passes[m];
            for (const Access& w : p.writes) {
                if (!isAttachment(w.usage)) continue;
                if (A* a = find(w.resource)) a->finalVersion = std::max(a->finalVersion, w.version);
                else atts.push_back({w.resource, w.load == LoadIntent::Preserve, w.version});
            }
            for (const Access& r : p.reads) {
                if (r.usage != Usage::DepthRead && r.usage != Usage::ColorAttachment) continue;
                if (A* a = find(r.resource)) a->finalVersion = std::max(a->finalVersion, r.version);
                else atts.push_back({r.resource, true, r.version});
            }
        }
        double bytes = 0;
        for (const A& a : atts) {
            const ResourceNode& node = res[a.r];
            const double size = static_cast<double>(node.texture.estimatedBytes());
            bool store = node.imported && (node.importFlags & ImportOutput);
            for (u32 q = 0; q < passes.size() && !store; ++q) {
                if (std::find(members.begin(), members.end(), q) != members.end()) continue;
                for (const Access& r : passes[q].reads) {
                    if (r.resource == a.r && r.version == a.finalVersion) store = true;
                }
            }
            if (a.load) bytes += size;
            if (store) bytes += size;
        }
        return bytes / (params.dramGBs * 1e9) * 1e3 + params.renderPassUs * 1e-3;
    }
};

// Transient bytes alive while live pass q runs after `done` (a proxy of the
// heap: memoryless textures are counted too).
struct LiveModel {
    std::vector<double> size;
    std::vector<u32>    producer;
    std::vector<std::vector<u32>> users;

    LiveModel(const RenderGraph& g, const Dag& d) {
        const auto& res = g.resources();
        size.assign(res.size(), 0);
        producer.assign(res.size(), ~0u);
        users.resize(res.size());
        for (u32 r = 0; r < res.size(); ++r) {
            if (!res[r].imported) {
                size[r] = static_cast<double>(res[r].kind == ResourceKind::Texture ? res[r].texture.estimatedBytes()
                                                                                   : res[r].buffer.size);
            }
        }
        for (u32 i = 0; i < d.n; ++i) {
            const PassNode& p = g.passes()[d.live[i]];
            for (const Access& w : p.writes) {
                if (producer[w.resource] == ~0u) producer[w.resource] = i;
                users[w.resource].push_back(i);
            }
            for (const Access& a : p.reads) users[a.resource].push_back(i);
        }
    }

    [[nodiscard]] double during(const Bits& done, u32 q) const {
        double sum = 0;
        for (u32 r = 0; r < size.size(); ++r) {
            if (size[r] == 0 || producer[r] == ~0u) continue;
            if (producer[r] != q && !done.test(producer[r])) continue;
            bool needed = producer[r] == q;
            for (const u32 u : users[r]) needed |= u == q || !done.test(u);
            if (needed) sum += size[r];
        }
        return sum;
    }
};

struct DpOutcome {
    std::vector<u32> order; // live indices
    bool   exact  = true;
    size_t states = 0;
};

// Layered DP: state = (done set, open render group); per state a Pareto front
// of (closed group cost ms, peak live bytes) with parent links.  Exact on the
// model while every layer stays under `cap` states (else the best `cap` by J
// are kept: a beam).
DpOutcome dpSearch(const RenderGraph& g, const Dag& d, const GraphCostParams& P, size_t cap) {
    const GroupModel gm{g, P};
    const LiveModel lm(g, d);
    constexpr size_t kFront = 6;
    struct Entry {
        double add = 0, peak = 0;
        u32 parentState = ~0u, parentEntry = ~0u, pass = ~0u;
        [[nodiscard]] double J(double gamma) const { return add + gamma * peak / kGiB; }
    };
    struct State {
        Bits done;
        std::vector<u32> group; // pass indices of the open raster group
        std::vector<Entry> front;
    };
    struct KeyHash {
        size_t operator()(const std::pair<Bits, std::vector<u32>>& k) const {
            size_t h = 1469598103934665603ull;
            for (const u64 w : k.first.w) h = (h ^ w) * 1099511628211ull;
            for (const u32 x : k.second) h = (h ^ x) * 1099511628211ull;
            return h;
        }
    };
    const double gamma = P.gammaMsPerGiB;
    auto insert = [&](std::vector<Entry>& front, const Entry& e) {
        for (const Entry& f : front) {
            if (f.add <= e.add && f.peak <= e.peak) return; // dominated
        }
        front.erase(std::remove_if(front.begin(), front.end(),
                                   [&](const Entry& f) { return e.add <= f.add && e.peak <= f.peak; }),
                    front.end());
        front.push_back(e);
        if (front.size() > kFront) {
            std::sort(front.begin(), front.end(), [&](const Entry& a, const Entry& b) { return a.J(gamma) < b.J(gamma); });
            front.resize(kFront);
        }
    };

    std::vector<std::vector<State>> layers(d.n + 1);
    layers[0].push_back(State{{}, {}, {Entry{}}});
    DpOutcome out;
    for (u32 depth = 0; depth < d.n; ++depth) {
        std::unordered_map<std::pair<Bits, std::vector<u32>>, u32, KeyHash> seen;
        std::vector<State>& next = layers[depth + 1];
        for (u32 si = 0; si < layers[depth].size(); ++si) {
            const State& s = layers[depth][si];
            for (u32 q = 0; q < d.n; ++q) {
                if (s.done.test(q)) continue;
                bool ready = true;
                for (const u32 p : d.preds[q]) ready = ready && s.done.test(p);
                if (!ready) continue;
                const u32 pass = d.live[q];
                const bool raster = g.passes()[pass].type == PassType::Raster;
                std::vector<u32> group;
                double closed = 0;
                if (raster && canJoinGroup(g, s.group, pass)) {
                    group = s.group;
                    group.push_back(pass);
                } else {
                    if (!s.group.empty()) closed = gm.cost(s.group);
                    if (raster) group = {pass};
                }
                Bits done = s.done;
                done.set(q);
                if (depth + 1 == d.n && !group.empty()) {
                    closed += gm.cost(group);
                    group.clear();
                }
                const double live = lm.during(s.done, q);
                auto key = std::make_pair(done, group);
                auto it = seen.find(key);
                u32 target;
                if (it == seen.end()) {
                    target = static_cast<u32>(next.size());
                    seen.emplace(std::move(key), target);
                    next.push_back(State{done, group, {}});
                } else {
                    target = it->second;
                }
                for (u32 ei = 0; ei < s.front.size(); ++ei) {
                    const Entry& e = s.front[ei];
                    insert(next[target].front, Entry{e.add + closed, std::max(e.peak, live), si, ei, q});
                }
            }
        }
        out.states += next.size();
        if (next.size() > cap) {
            out.exact = false;
            auto best = [&](const State& st) {
                double j = 1e300;
                for (const Entry& e : st.front) j = std::min(j, e.J(gamma));
                return j;
            };
            std::nth_element(next.begin(), next.begin() + static_cast<std::ptrdiff_t>(cap), next.end(),
                             [&](const State& a, const State& b) { return best(a) < best(b); });
            next.resize(cap);
        }
    }
    const std::vector<State>& last = layers[d.n];
    u32 bs = 0, be = 0;
    double bj = 1e300;
    for (u32 si = 0; si < last.size(); ++si) {
        for (u32 ei = 0; ei < last[si].front.size(); ++ei) {
            if (last[si].front[ei].J(gamma) < bj) {
                bj = last[si].front[ei].J(gamma);
                bs = si;
                be = ei;
            }
        }
    }
    std::vector<u32> rev;
    for (u32 depth = d.n; depth > 0; --depth) {
        const Entry& e = layers[depth][bs].front[be];
        rev.push_back(e.pass);
        bs = e.parentState;
        be = e.parentEntry;
    }
    out.order.assign(rev.rbegin(), rev.rend());
    return out;
}

// --- Simulated annealing over topological orders (real compiler + model) --

std::vector<u32> anneal(const RenderGraph& g, const Dag& d, std::vector<u32> order, AliasPolicy alias,
                        BarrierPolicy barriers, const GraphCostParams& P, u32 iterations, u32 seed) {
    std::mt19937 rng(seed);
    std::vector<u32> pos(d.n);
    auto reindex = [&] { for (u32 k = 0; k < d.n; ++k) pos[order[k]] = k; };
    reindex();
    const Evaluation first = evaluateOrder(g, toPasses(d, order), alias, barriers, P);
    if (!first.ok) return order;
    double cur = first.cost.J, bestJ = cur;
    std::vector<u32> best = order;
    const double t0 = 0.02, t1 = 0.0002; // ms
    for (u32 it = 0; it < iterations && d.n > 1; ++it) {
        const double temp = t0 * std::pow(t1 / t0, static_cast<double>(it) / iterations);
        const u32 v = static_cast<u32>(rng() % d.n);
        u32 lo = 0, hi = d.n - 1;
        for (const u32 p : d.preds[v]) lo = std::max(lo, pos[p] + 1);
        for (const u32 s : d.succs[v]) hi = std::min(hi, pos[s] - 1);
        if (hi <= lo) continue;
        const u32 target = lo + static_cast<u32>(rng() % (hi - lo + 1));
        if (target == pos[v]) continue;
        std::vector<u32> cand = order;
        cand.erase(cand.begin() + pos[v]);
        cand.insert(cand.begin() + target, v);
        const Evaluation e = evaluateOrder(g, toPasses(d, cand), alias, barriers, P);
        if (!e.ok) continue;
        const double delta = e.cost.J - cur;
        if (delta <= 0 || std::uniform_real_distribution<double>(0, 1)(rng) < std::exp(-delta / temp)) {
            order = std::move(cand);
            cur   = e.cost.J;
            reindex();
            if (cur < bestJ) {
                bestJ = cur;
                best  = order;
            }
        }
    }
    return best;
}

// All subsets of `items` (small lists: 2^k).
std::vector<std::vector<std::string>> subsets(const std::vector<std::string>& items) {
    std::vector<std::vector<std::string>> out;
    const size_t n = std::min<size_t>(items.size(), 10);
    for (size_t mask = 0; mask < (size_t{1} << n); ++mask) {
        std::vector<std::string> s;
        for (size_t i = 0; i < n; ++i) {
            if (mask & (size_t{1} << i)) s.push_back(items[i]);
        }
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace

SizeAlign EstimatedSizer::textureSize(u32, const TextureDesc& desc) const {
    // OPT-1 spike 3: compressible private textures carry +1/128 of metadata
    // and a 2048-byte alignment (heapTextureSizeAndAlign, M5 Max).
    const u64 bytes = desc.estimatedBytes();
    return {alignUp(bytes + bytes / 128, 2048), 2048};
}

SizeAlign EstimatedSizer::bufferSize(u32, const BufferDesc& desc) const { return {alignUp(desc.size, 256), 256}; }

Evaluation evaluateOrder(const RenderGraph& graph, const std::vector<u32>& order, AliasPolicy alias,
                         BarrierPolicy barriers, const GraphCostParams& params) {
    static const EstimatedSizer sizer;
    CompileOptions opt;
    opt.sizer         = &sizer;
    opt.order         = order;
    opt.aliasPolicy   = alias;
    opt.barrierPolicy = barriers;
    Evaluation e;
    e.compiled = compile(graph, opt);
    e.ok       = e.compiled.ok;
    if (e.ok) e.cost = evaluateGraph(graph, e.compiled, params);
    return e;
}

OptimizeResult optimize(const std::string& family, const GraphBuilder& builder, const OptimizeOptions& options) {
    OptimizeResult result;
    struct Policy { AliasPolicy alias; BarrierPolicy barriers; };
    std::vector<Policy> policies = {{AliasPolicy::Greedy, BarrierPolicy::Conservative}};
    if (options.tryPolicies) {
        policies.push_back({AliasPolicy::ColoringStageClass, BarrierPolicy::Minimal});
        policies.push_back({AliasPolicy::Coloring, BarrierPolicy::Minimal}); // annealing uses the last
    }

    double bestJ = 1e300;
    auto consider = [&](const RenderGraph& g, const BuildChoices& ch, const std::string& method,
                        const std::vector<u32>& order, const Policy& pol) {
        const Evaluation e = evaluateOrder(g, order, pol.alias, pol.barriers, options.cost);
        if (!e.ok) return;
        PlanCandidate cand;
        cand.choices       = ch;
        cand.method        = method;
        cand.aliasPolicy   = pol.alias;
        cand.barrierPolicy = pol.barriers;
        cand.cost          = e.cost;
        cand.order         = namesOf(g, e.compiled.order);
        if (e.cost.J < bestJ - 1e-12) {
            bestJ = e.cost.J;
            GraphPlan& plan    = result.plan;
            plan.family        = family;
            plan.order         = cand.order;
            plan.remat         = ch.remat;
            plan.async         = ch.async;
            plan.aliasPolicy   = pol.alias;
            plan.barrierPolicy = pol.barriers;
            plan.method        = method;
            plan.key           = graphKey(g, plan.order);
            plan.predicted     = {e.cost.dramBytes, e.cost.heapBytes, e.cost.maxLiveBytes, e.cost.frameMs};
        }
        result.candidates.push_back(std::move(cand));
    };

    // Baseline: the end-of-F4 compiler on the scenario's default choices.
    {
        RenderGraph g;
        if (!builder(options.baselineChoices, g, &result.error)) return result;
        const Evaluation base = evaluateOrder(g, {}, AliasPolicy::Greedy, BarrierPolicy::Conservative, options.cost);
        if (!base.ok) {
            result.error = base.compiled.errors.empty() ? "baseline does not compile" : base.compiled.errors.front();
            return result;
        }
        result.baseline = base.cost;
    }

    for (const std::vector<std::string>& remat : subsets(options.rematCandidates)) {
        for (const std::vector<std::string>& async : subsets(options.asyncCandidates)) {
            BuildChoices ch{remat, async};
            if (options.asyncCandidates.empty()) ch.async = options.baselineChoices.async;
            RenderGraph g;
            std::string error;
            if (!builder(ch, g, &error)) continue;
            const Dag d = buildDag(g);
            if (d.n == 0) continue;
            std::vector<u32> greedy(d.n);
            for (u32 i = 0; i < d.n; ++i) greedy[i] = i;
            const DpOutcome dp = dpSearch(g, d, options.cost, options.dpStateCap);
            for (const Policy& pol : policies) {
                consider(g, ch, "greedy", toPasses(d, greedy), pol);
                consider(g, ch, dp.exact ? "dp" : "dp-beam", toPasses(d, dp.order), pol);
            }
            if (options.annealIterations > 0) {
                const Policy& pol = policies.back();
                const std::vector<u32> a1 = anneal(g, d, greedy, pol.alias, pol.barriers, options.cost,
                                                   options.annealIterations, options.seed);
                consider(g, ch, "anneal", toPasses(d, a1), pol);
                const std::vector<u32> a2 = anneal(g, d, dp.order, pol.alias, pol.barriers, options.cost,
                                                   options.annealIterations, options.seed + 1);
                consider(g, ch, "dp+anneal", toPasses(d, a2), pol);
            }
            if (options.asyncCandidates.empty()) break;
        }
    }
    if (result.candidates.empty()) {
        result.error = "no candidate compiled";
        return result;
    }
    result.plan.baseline = {result.baseline.dramBytes, result.baseline.heapBytes, result.baseline.maxLiveBytes,
                            result.baseline.frameMs};
    result.ok = true;
    return result;
}

} // namespace phosphor::rg
