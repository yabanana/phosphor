#include "rendergraph/graph_budget.h"

#include "rendergraph/timing_plan.h"

#include <algorithm>
#include <cstdio>

namespace phosphor::rg {

namespace {

std::string mib(u64 bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

u64 resourceBytes(const ResourceNode& node) {
    return node.kind == ResourceKind::Texture ? node.texture.estimatedBytes() : node.buffer.size;
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
    // OPT-1.7: working set of every compute pass (traffic read + written).
    const auto& passes = graph.passes();
    for (const PassBytes& pb : out.passes) {
        if (passes[pb.pass].type != PassType::Compute) continue;
        WorkingSet w;
        w.pass       = pb.pass;
        w.position   = pb.position;
        w.readBytes  = pb.readBytes;
        w.writeBytes = pb.writeBytes;
        w.aboveSlc   = w.total() > options.slcBytes;
        out.workingSets.push_back(w);
    }
    // Reuse candidates: the pass at position p+1 reads a version written by
    // the work that ends at position p (the pass itself, or any member of the
    // render group ending there: attachments reach DRAM at the group's end),
    // same queue, not fused with the consumer in one render group (tile memory
    // would make it free anyway), resource not memoryless and within the SLC.
    for (u32 pos = 0; pos + 1 < compiled.order.size(); ++pos) {
        const u32 cons = compiled.order[pos + 1];
        const u32 gp = pos < compiled.groupOfPosition.size() ? compiled.groupOfPosition[pos] : ~0u;
        const u32 gc = pos + 1 < compiled.groupOfPosition.size() ? compiled.groupOfPosition[pos + 1] : ~0u;
        if (gp != ~0u && gp == gc) continue;
        u32 first = pos;
        if (gp != ~0u && gp < compiled.renderGroups.size()) first = compiled.renderGroups[gp].firstPosition;
        std::vector<u32> seen;
        for (u32 q = pos + 1; q-- > first;) { // newest producer first
            const u32 prod = compiled.order[q];
            if (passes[prod].queue != passes[cons].queue) continue;
            for (const Access& w : passes[prod].writes) {
                if (w.resource >= graph.resources().size()) continue;
                if (w.resource < compiled.memoryless.size() && compiled.memoryless[w.resource]) continue;
                for (const Access& r : passes[cons].reads) {
                    if (r.resource != w.resource || r.version != w.version) continue;
                    if (std::find(seen.begin(), seen.end(), w.resource) != seen.end()) continue;
                    const u64 bytes = resourceBytes(graph.resources()[w.resource]);
                    if (bytes == 0 || bytes > options.slcBytes) continue;
                    seen.push_back(w.resource);
                    ReuseCandidate rc;
                    rc.producer = prod;
                    rc.consumer = cons;
                    rc.resource = w.resource;
                    rc.bytes    = bytes;
                    rc.consumerFits = out.passes[pos + 1].total() <= options.slcBytes;
                    out.reuse.push_back(rc);
                }
            }
        }
    }
    for (const BandwidthTier& t : bandwidthTiers()) {
        out.tiers.push_back(makeTierBudget(t, out.totalBytes(), options.fps, options.fraction));
    }
    return out;
}

std::vector<std::string> budgetSummaryLines(const RenderGraph& graph, const GraphBudget& budget,
                                            const BudgetOptions& options) {
    std::vector<std::string> lines;
    lines.push_back("DRAM estimate per frame: " + mib(budget.totalReadBytes) + " read + " +
                    mib(budget.totalWriteBytes) + " write = " + mib(budget.totalBytes()) +
                    " (graph estimate, not a hardware counter)");
    for (const TierBudget& t : budget.tiers) lines.push_back(t.line);
    const auto& passes = graph.passes();
    for (const WorkingSet& w : budget.workingSets) {
        if (!w.aboveSlc) continue;
        lines.push_back("SLC: compute pass '" + passes[w.pass].name + "' working set " + mib(w.total()) +
                        " is above the SLC estimate " + mib(options.slcBytes) + " (ESTIMATE from a fit, not documented)");
    }
    for (const ReuseCandidate& r : budget.reuse) {
        lines.push_back("SLC reuse candidate: '" + passes[r.producer].name + "' -> '" + passes[r.consumer].name +
                        "' via '" + graph.resources()[r.resource].name + "' (" + mib(r.bytes) + " <= SLC estimate " +
                        mib(options.slcBytes) + (r.consumerFits ? ", consumer working set fits" : ", consumer working set does NOT fit") +
                        ")");
    }
    return lines;
}

} // namespace phosphor::rg
