#pragma once
// Plan-file merge used by graph_opt --merge (portable, unit tested).
#include "rendergraph/optimizer/plan.h"

#include <algorithm>
#include <string>
#include <vector>

namespace phosphor::rg {

/// `existing` with every plan of a family present in `updates` replaced by the
/// update of that family (first occurrence keeps its position, stale
/// duplicates are dropped); plans of other families are kept in order; new
/// families are appended.  Among `updates` the last plan of a family wins.
[[nodiscard]] inline std::vector<GraphPlan> mergePlans(const std::vector<GraphPlan>& existing,
                                                       const std::vector<GraphPlan>& updates) {
    std::vector<GraphPlan> out;
    auto lastUpdate = [&](const std::string& family) -> const GraphPlan* {
        const GraphPlan* r = nullptr;
        for (const GraphPlan& u : updates)
            if (u.family == family) r = &u;
        return r;
    };
    std::vector<std::string> used;
    auto isUsed = [&](const std::string& f) { return std::find(used.begin(), used.end(), f) != used.end(); };
    for (const GraphPlan& e : existing) {
        if (const GraphPlan* u = lastUpdate(e.family)) {
            if (!isUsed(e.family)) {
                out.push_back(*u);
                used.push_back(e.family);
            }
        } else {
            out.push_back(e);
        }
    }
    for (const GraphPlan& u : updates) {
        if (isUsed(u.family)) continue;
        out.push_back(*lastUpdate(u.family));
        used.push_back(u.family);
    }
    return out;
}

} // namespace phosphor::rg
