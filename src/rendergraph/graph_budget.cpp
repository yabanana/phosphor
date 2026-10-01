#include "rendergraph/graph_budget.h"

#include "rendergraph/timing_plan.h"

#include <cstdio>

namespace phosphor::rg {

namespace {

std::string mib(u64 bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

u64 resourceBytes(const ResourceNode& node) {
    return node.kind == ResourceKind::Buffer ? node.buffer.size : node.texture.estimatedBytes();
}

} // namespace

const std::vector<BandwidthTier>& bandwidthTiers() {
    static const std::vector<BandwidthTier> tiers = {
        {"T0 M5 base", 153.6, false, "external: Apple specifications"},
        {"T1 M5 Pro", 307.0, false, "external: Apple specifications"},
        {"T2 M5 Max", 569.0, true, "measured, B-08"},
        {"M3 base", 100.0, false, "external: Apple specifications"},
    };
    return tiers;
}

TierBudget makeTierBudget(const BandwidthTier& tier, u64 frameBytes, double fps, double fraction) {
    TierBudget b;
    b.tier        = tier;
    b.fps         = fps;
    b.fraction    = fraction;
    b.budgetBytes = fps > 0 ? tier.gbPerSecond * 1e9 / fps * fraction : 0.0;
    b.share       = b.budgetBytes > 0 ? static_cast<double>(frameBytes) / b.budgetBytes : 0.0;
    b.over        = b.share > 1.0;
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "%s %.1f GB/s [%s]: budget %.2f MiB/frame at %.0f fps x %.2f, frame %.2f MiB = %.2f%%%s",
                  tier.name.c_str(), tier.gbPerSecond, tier.measured ? "measured" : "NOT measured, external",
                  b.budgetBytes / (1024.0 * 1024.0), fps, fraction, static_cast<double>(frameBytes) / (1024.0 * 1024.0),
                  b.share * 100.0, b.over ? " OVER BUDGET" : "");
    b.line = buf;
    return b;
}

GraphBudget analyzeBudget(const RenderGraph& graph, const CompiledGraph& compiled, const BudgetOptions& options) {
    GraphBudget out;
    const std::vector<PassTraffic> traffic = estimatePassBandwidth(graph, compiled);
    for (u32 pos = 0; pos < compiled.order.size(); ++pos) {
        const u32 p = compiled.order[pos];
        PassBytes pb;
        pb.pass       = p;
        pb.position   = pos;
        pb.readBytes  = p < traffic.size() ? traffic[p].readBytes : 0;
        pb.writeBytes = p < traffic.size() ? traffic[p].writeBytes : 0;
        out.totalReadBytes += pb.readBytes;
        out.totalWriteBytes += pb.writeBytes;
        out.passes.push_back(pb);
    }
    for (const BandwidthTier& t : bandwidthTiers()) {
        out.tiers.push_back(makeTierBudget(t, out.totalBytes(), options.fps, options.fraction));
    }
    return out;
}

std::vector<std::string> budgetSummaryLines(const RenderGraph& graph, const GraphBudget& budget,
                                            const BudgetOptions& options) {
    (void)graph;
    (void)options;
    std::vector<std::string> lines;
    lines.push_back("DRAM estimate per frame: " + mib(budget.totalReadBytes) + " read + " +
                    mib(budget.totalWriteBytes) + " write = " + mib(budget.totalBytes()) +
                    " (graph estimate, not a hardware counter)");
    for (const TierBudget& t : budget.tiers) lines.push_back(t.line);
    return lines;
}

} // namespace phosphor::rg
