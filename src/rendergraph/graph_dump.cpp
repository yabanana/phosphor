#include "rendergraph/graph_dump.h"

#include "rendergraph/graph_lint.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <utility>

namespace phosphor::rg {

namespace {

u64 resourceBytes(const ResourceNode& node) {
    return node.kind == ResourceKind::Buffer ? node.buffer.size : node.texture.estimatedBytes();
}

bool isMemoryless(const CompiledGraph& c, u32 resource) {
    return resource < c.memoryless.size() && c.memoryless[resource];
}

// Render group of a live pass, or nullptr if the group stage did not run.
const RenderGroup* groupOf(const CompiledGraph& c, u32 pass) {
    const u32 pos = c.position(pass);
    if (pos == ~0u || pos >= c.groupOfPosition.size()) {
        return nullptr;
    }
    const u32 g = c.groupOfPosition[pos];
    return g < c.renderGroups.size() ? &c.renderGroups[g] : nullptr;
}

const char* passTypeName(PassType t) {
    switch (t) {
    case PassType::Raster:  return "raster";
    case PassType::Compute: return "compute";
    case PassType::Blit:    return "blit";
    }
    return "?";
}

const char* queueName(Queue q) { return q == Queue::AsyncCompute ? "async compute" : "graphics"; }

const char* usageName(Usage u) {
    switch (u) {
    case Usage::ColorAttachment: return "ColorAttachment";
    case Usage::DepthAttachment: return "DepthAttachment";
    case Usage::DepthRead:       return "DepthRead";
    case Usage::ShaderRead:      return "ShaderRead";
    case Usage::ShaderWrite:     return "ShaderWrite";
    case Usage::CopySrc:         return "CopySrc";
    case Usage::CopyDst:         return "CopyDst";
    case Usage::IndirectArgs:    return "IndirectArgs";
    }
    return "?";
}

const char* intentName(LoadIntent l) {
    switch (l) {
    case LoadIntent::Clear:    return "Clear";
    case LoadIntent::Preserve: return "Preserve";
    case LoadIntent::Discard:  return "Discard";
    }
    return "?";
}

const char* loadName(LoadAction a) {
    switch (a) {
    case LoadAction::DontCare: return "DontCare";
    case LoadAction::Load:     return "Load";
    case LoadAction::Clear:    return "Clear";
    }
    return "?";
}

const char* storeName(StoreAction a) { return a == StoreAction::Store ? "Store" : "DontCare"; }

std::string stageList(Stages s) {
    static const struct {
        Stages bit;
        const char* name;
    } kNames[] = {
        {StageVertex, "vertex"}, {StageFragment, "fragment"}, {StageTile, "tile"},
        {StageObject, "object"}, {StageMesh, "mesh"},         {StageDispatch, "dispatch"},
        {StageBlit, "blit"},     {StageAccelerationStructure, "accel"},
    };
    std::string out;
    for (const auto& n : kNames) {
        if (s & n.bit) {
            if (!out.empty()) {
                out += '|';
            }
            out += n.name;
        }
    }
    return out.empty() ? "none" : out;
}

// Escapes for a double-quoted dot string.  Real newlines become the two
// characters "\n" (a dot line break).
std::string esc(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char ch : s) {
        switch (ch) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        default:   out += ch;     break;
        }
    }
    return out;
}

std::string mebibytes(u64 bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

const AttachmentPlan* findAttachment(const RenderGroup& g, u32 resource) {
    for (const AttachmentPlan& a : g.attachments) {
        if (a.resource == resource) {
            return &a;
        }
    }
    return nullptr;
}

} // namespace

BandwidthReport estimateBandwidth(const RenderGraph& graph, const CompiledGraph& compiled) {
    BandwidthReport report;
    const auto& resources = graph.resources();
    report.resources.resize(resources.size());
    for (u32 i = 0; i < resources.size(); ++i) {
        report.resources[i].resource = i;
    }

    auto addRead = [&](u32 r, u64 bytes) { report.resources[r].readBytes += bytes; };
    auto addWrite = [&](u32 r, u64 bytes) { report.resources[r].writeBytes += bytes; };

    // Attachment traffic, once per render group.
    for (const RenderGroup& g : compiled.renderGroups) {
        for (const AttachmentPlan& a : g.attachments) {
            if (a.resource >= resources.size() || isMemoryless(compiled, a.resource)) {
                continue;
            }
            const u64 bytes = resourceBytes(resources[a.resource]);
            if (a.load == LoadAction::Load) {
                addRead(a.resource, bytes);
            }
            if (a.store == StoreAction::Store) {
                addWrite(a.resource, bytes);
            }
        }
    }

    const auto& passes = graph.passes();
    for (u32 p = 0; p < passes.size(); ++p) {
        if ((p < compiled.culled.size() && compiled.culled[p]) || compiled.position(p) == ~0u) {
            continue; // culled passes contribute nothing
        }
        const PassNode& pass = passes[p];
        // Without group information a raster pass is counted on its own:
        // every attachment write is one write copy, a Preserve load one read.
        const bool fallback = pass.type == PassType::Raster && groupOf(compiled, p) == nullptr;

        for (const Access& a : pass.reads) {
            if (a.resource >= resources.size()) {
                continue;
            }
            const u64 bytes = resourceBytes(resources[a.resource]);
            if (isAttachment(a.usage)) {
                if (fallback && !isMemoryless(compiled, a.resource)) {
                    addRead(a.resource, bytes);
                }
                continue;
            }
            addRead(a.resource, bytes); // ShaderRead, CopySrc, IndirectArgs
        }
        for (const Access& a : pass.writes) {
            if (a.resource >= resources.size()) {
                continue;
            }
            const u64 bytes = resourceBytes(resources[a.resource]);
            if (isAttachment(a.usage)) {
                if (fallback && !isMemoryless(compiled, a.resource)) {
                    addWrite(a.resource, bytes);
                }
                continue;
            }
            addWrite(a.resource, bytes); // ShaderWrite, CopyDst
        }
    }

    for (const ResourceTraffic& t : report.resources) {
        report.totalReadBytes += t.readBytes;
        report.totalWriteBytes += t.writeBytes;
    }
    return report;
}

std::string dumpGraphviz(const RenderGraph& graph, const CompiledGraph& compiled, const BudgetOptions& budgetOptions) {
    const auto& passes = graph.passes();
    const auto& resources = graph.resources();
    const BandwidthReport bw = estimateBandwidth(graph, compiled);
    const GraphBudget budget = analyzeBudget(graph, compiled, budgetOptions);
    const std::vector<LintFinding> findings = lintGraph(graph, compiled);

    auto isCulled = [&](u32 p) {
        return (p < compiled.culled.size() && compiled.culled[p]) || compiled.position(p) == ~0u;
    };
    auto placementOf = [&](u32 resource) -> const Placement* {
        for (const Placement& pl : compiled.aliasing.placements) {
            if (pl.resource == resource) {
                return &pl;
            }
        }
        return nullptr;
    };
    auto barriersOf = [&](u32 position) -> const PassBarriers* {
        for (const PassBarriers& pb : compiled.barriers) {
            if (pb.position == position) {
                return &pb;
            }
        }
        return nullptr;
    };

    std::string out;
    out += "digraph RenderGraph {\n";
    out += "  rankdir=LR;\n";
    std::string graphLabel = "DRAM per frame: " + mebibytes(bw.totalReadBytes) + " MiB read, " +
                             mebibytes(bw.totalWriteBytes) + " MiB write\\ntransient heap: " +
                             mebibytes(compiled.aliasing.heapSize) + " MiB aliased, " +
                             mebibytes(compiled.aliasing.unaliasedSize) + " MiB unaliased";
    for (const std::string& line : budgetSummaryLines(graph, budget, budgetOptions)) {
        graphLabel += "\\n" + esc(line);
    }
    if (findings.empty()) {
        graphLabel += "\\nlint: no findings";
    } else {
        graphLabel += "\\nlint: " + std::to_string(findings.size()) + " finding(s)";
        for (const LintFinding& f : findings) graphLabel += "\\n" + esc(f.message);
    }
    out += "  label=\"" + graphLabel + "\";\n";
    out += "  labelloc=t;\n";
    out += "  node [fontname=\"Helvetica\"];\n";
    out += "  edge [fontname=\"Helvetica\", fontsize=10];\n";

    auto passNode = [&](u32 p, const char* indent) {
        const PassNode& pass = passes[p];
        std::string label = esc(pass.name) + "\\n" + passTypeName(pass.type) + ", " + queueName(pass.queue);
        std::string style;
        if (isCulled(p)) {
            label += "\\nculled";
            style = ", style=dashed, color=gray, fontcolor=gray";
        } else {
            const u32 pos = compiled.position(p);
            label += "\\n@" + std::to_string(pos);
            for (const PassBytes& pb : budget.passes) {
                if (pb.pass == p) {
                    label += "\\nDRAM R " + mebibytes(pb.readBytes) + " / W " + mebibytes(pb.writeBytes) + " MiB";
                }
            }
            if (const PassBarriers* pb = barriersOf(pos)) {
                for (const Barrier& b : pb->barriers) {
                    label += std::string("\\nbarrier ") + (b.scope == BarrierScope::Encoder ? "encoder " : "queue ") +
                             stageList(b.afterStages) + "->" + stageList(b.beforeStages);
                    if (b.aliasing) {
                        label += " alias";
                    }
                }
            }
        }
        return std::string(indent) + "p" + std::to_string(p) + " [shape=box, label=\"" + label + "\"" + style +
               "];\n";
    };

    // Passes: fused ones in a cluster per render group.
    std::vector<bool> emitted(passes.size(), false);
    for (u32 g = 0; g < compiled.renderGroups.size(); ++g) {
        const RenderGroup& grp = compiled.renderGroups[g];
        out += "  subgraph cluster_group" + std::to_string(g) + " {\n";
        out += "    label=\"render pass " + std::to_string(g) + " " + std::to_string(grp.width) + "x" +
               std::to_string(grp.height) + "\";\n";
        out += "    style=rounded;\n";
        for (u32 pos = grp.firstPosition; pos <= grp.lastPosition && pos < compiled.order.size(); ++pos) {
            const u32 p = compiled.order[pos];
            if (p < passes.size() && !emitted[p]) {
                emitted[p] = true;
                out += passNode(p, "    ");
            }
        }
        out += "  }\n";
    }
    for (u32 p = 0; p < passes.size(); ++p) {
        if (!emitted[p]) {
            out += passNode(p, "  ");
        }
    }

    // Resource version nodes (only the versions some pass touches).
    std::set<std::pair<u32, u32>> versions;
    for (const PassNode& pass : passes) {
        for (const Access& a : pass.reads) {
            versions.insert({a.resource, a.version});
        }
        for (const Access& a : pass.writes) {
            versions.insert({a.resource, a.version});
        }
    }
    for (const auto& [r, v] : versions) {
        if (r >= resources.size()) {
            continue;
        }
        const ResourceNode& res = resources[r];
        std::string label = esc(res.name) + "@" + std::to_string(v) + "\\n";
        if (res.kind == ResourceKind::Buffer) {
            label += "buffer " + std::to_string(res.buffer.size) + " B";
        } else {
            label += std::string(formatName(res.texture.format)) + " " + std::to_string(res.texture.width) + "x" +
                     std::to_string(res.texture.height) + ", " + mebibytes(res.texture.estimatedBytes()) + " MiB";
        }
        std::string flags;
        if (res.imported) {
            flags += (res.importFlags & ImportPerFrame) ? " [imported, per-frame]" : " [imported]";
        }
        if (isMemoryless(compiled, r)) {
            flags += " [memoryless]";
        }
        if (const Placement* pl = placementOf(r); pl && pl->aliased) {
            flags += " [aliased @" + std::to_string(pl->offset) + "]";
        }
        if (!flags.empty()) {
            label += "\\n" + flags.substr(1);
        }
        if (r < bw.resources.size() && v + 1 == res.versions &&
            (bw.resources[r].readBytes || bw.resources[r].writeBytes)) {
            label += "\\nDRAM R " + mebibytes(bw.resources[r].readBytes) + " / W " +
                     mebibytes(bw.resources[r].writeBytes) + " MiB";
        }
        out += "  r" + std::to_string(r) + "_v" + std::to_string(v) + " [shape=ellipse, label=\"" + label + "\"];\n";
    }

    // Edges.
    for (u32 p = 0; p < passes.size(); ++p) {
        const PassNode& pass = passes[p];
        const RenderGroup* grp = groupOf(compiled, p);
        const std::string pn = "p" + std::to_string(p);
        const char* culledStyle = isCulled(p) ? ", style=dashed, color=gray, fontcolor=gray" : "";

        auto label = [&](const Access& a, bool write) {
            std::string l = usageName(a.usage);
            if (isAttachment(a.usage)) {
                const AttachmentPlan* plan = grp ? findAttachment(*grp, a.resource) : nullptr;
                if (plan) {
                    l += std::string(" load=") + loadName(plan->load) + " store=" + storeName(plan->store);
                } else if (write) {
                    l += std::string(" load=") + intentName(a.load);
                }
            }
            return l;
        };
        for (const Access& a : pass.reads) {
            out += "  r" + std::to_string(a.resource) + "_v" + std::to_string(a.version) + " -> " + pn +
                   " [label=\"" + label(a, false) + "\"" + culledStyle + "];\n";
        }
        for (const Access& a : pass.writes) {
            out += "  " + pn + " -> r" + std::to_string(a.resource) + "_v" + std::to_string(a.version) +
                   " [label=\"" + label(a, true) + "\"" + culledStyle + "];\n";
        }
    }

    // Cross-queue events (F2.6): the producer's queue signals after the
    // signalling pass, the consumer's queue waits before the waiting pass.
    for (const QueueSync& q : compiled.queueSyncs) {
        if (q.signalAfterPosition >= compiled.order.size() || q.waitBeforePosition >= compiled.order.size()) {
            continue;
        }
        out += "  p" + std::to_string(compiled.order[q.signalAfterPosition]) + " -> p" +
               std::to_string(compiled.order[q.waitBeforePosition]) + " [label=\"event " + std::to_string(q.value) +
               "\", style=dashed, color=blue, fontcolor=blue, constraint=false];\n";
    }

    out += "}\n";
    return out;
}

} // namespace phosphor::rg
