#include "rendergraph/optimizer/plan.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <cstring>

namespace phosphor::rg {

using json = nlohmann::json;

namespace {

struct Fnv {
    u64 h = 14695981039346656037ull;
    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
    }
    void u(u64 v) { bytes(&v, sizeof v); } // little-endian hosts only (Apple silicon, x86-64/arm64 Linux)
    void s(const std::string& v) {
        u(v.size());
        bytes(v.data(), v.size());
    }
};

void hashAccess(Fnv& f, const RenderGraph& g, const Access& a) {
    const ResourceNode& r = g.resources()[a.resource];
    f.s(r.name);
    f.u(a.version);
    f.u(static_cast<u64>(a.usage));
    f.u(a.stages);
    f.u(a.slot);
    f.u(static_cast<u64>(a.load));
}

void hashResource(Fnv& f, const ResourceNode& r) {
    f.s(r.name);
    f.u(static_cast<u64>(r.kind));
    f.u(r.imported ? 1 : 0);
    f.u(r.importFlags);
    if (r.kind == ResourceKind::Texture) {
        f.u(static_cast<u64>(r.texture.format));
        // A per-frame import (the drawable) follows the window: its size does
        // not change what a plan orders.
        const bool perFrame = r.imported && (r.importFlags & ImportPerFrame);
        f.u(perFrame ? 0 : r.texture.width);
        f.u(perFrame ? 0 : r.texture.height);
        f.u(r.texture.depth);
        f.u(r.texture.mipLevels);
        f.u(r.texture.sampleCount);
    } else {
        f.u(r.buffer.size);
    }
}

} // namespace

const char* aliasPolicyName(AliasPolicy p) {
    switch (p) {
        case AliasPolicy::Coloring: return "coloring";
        case AliasPolicy::ColoringStageClass: return "coloring-stage";
        default: return "greedy";
    }
}
const char* barrierPolicyName(BarrierPolicy p) { return p == BarrierPolicy::Minimal ? "minimal" : "conservative"; }

u64 graphKey(const RenderGraph& graph, const std::vector<std::string>& names) {
    const auto& passes = graph.passes();
    std::vector<u32> selected;
    if (names.empty()) {
        for (u32 p = 0; p < passes.size(); ++p) selected.push_back(p);
    } else {
        for (const std::string& n : names) {
            for (u32 p = 0; p < passes.size(); ++p) {
                if (passes[p].name == n) selected.push_back(p);
            }
        }
    }
    // Planned passes in name order: the key does not depend on declaration
    // order, only on what the passes do.
    std::sort(selected.begin(), selected.end(), [&](u32 a, u32 b) { return passes[a].name < passes[b].name; });
    Fnv f;
    std::vector<u32> touched;
    for (const u32 p : selected) {
        const PassNode& n = passes[p];
        f.s(n.name);
        f.u(static_cast<u64>(n.type));
        f.u(static_cast<u64>(n.queue));
        f.u(n.hints);
        f.u(n.sideEffect ? 1 : 0);
        f.u(n.reads.size());
        for (const Access& a : n.reads) {
            hashAccess(f, graph, a);
            touched.push_back(a.resource);
        }
        f.u(n.writes.size());
        for (const Access& a : n.writes) {
            hashAccess(f, graph, a);
            touched.push_back(a.resource);
        }
    }
    std::sort(touched.begin(), touched.end(), [&](u32 a, u32 b) {
        return graph.resources()[a].name != graph.resources()[b].name ? graph.resources()[a].name < graph.resources()[b].name
                                                                       : a < b;
    });
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    for (const u32 r : touched) hashResource(f, graph.resources()[r]);
    return f.h;
}

std::vector<u32> planOrder(const RenderGraph& graph, const GraphPlan& plan, std::string* error) {
    const auto& passes = graph.passes();
    std::vector<u32> order;
    std::vector<bool> used(passes.size(), false);
    for (const std::string& n : plan.order) {
        u32 found = ~0u;
        for (u32 p = 0; p < passes.size(); ++p) {
            if (passes[p].name != n) continue;
            if (found != ~0u) {
                if (error) *error = "plan '" + plan.family + "': pass name '" + n + "' is not unique in the graph";
                return {};
            }
            found = p;
        }
        if (found == ~0u) {
            if (error) *error = "plan '" + plan.family + "': no pass named '" + n + "'";
            return {};
        }
        if (used[found]) {
            if (error) *error = "plan '" + plan.family + "': '" + n + "' listed twice";
            return {};
        }
        used[found] = true;
        order.push_back(found);
    }
    const u64 key = graphKey(graph, plan.order);
    if (key != plan.key) {
        if (error) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "key %016llx does not match the graph's %016llx",
                          static_cast<unsigned long long>(plan.key), static_cast<unsigned long long>(key));
            *error = "plan '" + plan.family + "': " + buf;
        }
        return {};
    }
    // Unplanned passes follow in declaration order, except the ones the graph
    // culls (a forced order may only list live passes).
    const CompiledGraph live = compileOrder(graph);
    for (u32 p = 0; p < passes.size(); ++p) {
        const bool culled = p < live.culled.size() && live.culled[p];
        if (!used[p] && !culled) order.push_back(p);
    }
    return order;
}

namespace {

json metricsJson(const PlanMetrics& m) {
    return {{"dram_bytes", m.dramBytes}, {"heap_bytes", m.heapBytes}, {"max_live_bytes", m.maxLive}, {"time_ms", m.timeMs}};
}

PlanMetrics metricsFrom(const json& j) {
    PlanMetrics m;
    if (!j.is_object()) return m;
    m.dramBytes = j.value("dram_bytes", 0.0);
    m.heapBytes = j.value("heap_bytes", 0.0);
    m.maxLive   = j.value("max_live_bytes", 0.0);
    m.timeMs    = j.value("time_ms", 0.0);
    return m;
}

} // namespace

std::string toJson(const std::vector<GraphPlan>& plans) {
    json list = json::array();
    for (const GraphPlan& p : plans) {
        char key[20];
        std::snprintf(key, sizeof key, "%016llx", static_cast<unsigned long long>(p.key));
        list.push_back({{"family", p.family},
                        {"key", key},
                        {"method", p.method},
                        {"order", p.order},
                        {"remat", p.remat},
                        {"async", p.async},
                        {"alias", aliasPolicyName(p.aliasPolicy)},
                        {"barriers", barrierPolicyName(p.barrierPolicy)},
                        {"predicted", metricsJson(p.predicted)},
                        {"baseline", metricsJson(p.baseline)}});
    }
    return json{{"schema", kGraphPlanSchema}, {"plans", list}}.dump(2) + "\n";
}

bool fromJson(const std::string& text, std::vector<GraphPlan>& out, std::string* error) {
    out.clear();
    try {
        const json j = json::parse(text);
        if (j.value("schema", 0u) != kGraphPlanSchema) {
            if (error) *error = "graph plans: unsupported schema";
            return false;
        }
        for (const json& e : j.at("plans")) {
            GraphPlan p;
            p.family = e.at("family").get<std::string>();
            const std::string key = e.at("key").get<std::string>();
            char* end = nullptr;
            p.key = std::strtoull(key.c_str(), &end, 16);
            if (key.empty() || *end != '\0') {
                if (error) *error = "graph plans: bad key '" + key + "'";
                return false;
            }
            p.method = e.value("method", std::string());
            p.order  = e.at("order").get<std::vector<std::string>>();
            p.remat  = e.value("remat", std::vector<std::string>{});
            p.async  = e.value("async", std::vector<std::string>{});
            const std::string alias = e.value("alias", std::string("greedy"));
            const std::string barriers = e.value("barriers", std::string("conservative"));
            if (alias == "greedy") p.aliasPolicy = AliasPolicy::Greedy;
            else if (alias == "coloring") p.aliasPolicy = AliasPolicy::Coloring;
            else if (alias == "coloring-stage") p.aliasPolicy = AliasPolicy::ColoringStageClass;
            else {
                if (error) *error = "graph plans: unknown alias policy '" + alias + "'";
                return false;
            }
            if (barriers == "conservative") p.barrierPolicy = BarrierPolicy::Conservative;
            else if (barriers == "minimal") p.barrierPolicy = BarrierPolicy::Minimal;
            else {
                if (error) *error = "graph plans: unknown barrier policy '" + barriers + "'";
                return false;
            }
            p.predicted = metricsFrom(e.value("predicted", json::object()));
            p.baseline  = metricsFrom(e.value("baseline", json::object()));
            out.push_back(std::move(p));
        }
    } catch (const std::exception& ex) {
        if (error) *error = std::string("graph plans: ") + ex.what();
        out.clear();
        return false;
    }
    return true;
}

const GraphPlan* findPlan(const std::vector<GraphPlan>& plans, const std::string& family) {
    for (const GraphPlan& p : plans) {
        if (p.family == family) return &p;
    }
    return nullptr;
}

} // namespace phosphor::rg
