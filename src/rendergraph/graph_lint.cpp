#include "rendergraph/graph_lint.h"

#include <algorithm>
#include <cstdio>

namespace phosphor::rg {

namespace {

constexpr u32 kNone = ~0u;

const char* usageLabel(Usage u) {
    switch (u) {
    case Usage::ColorAttachment: return "color attachment";
    case Usage::DepthAttachment: return "depth attachment";
    case Usage::DepthRead:       return "read-only depth attachment";
    case Usage::ShaderRead:      return "sampled/shader read";
    case Usage::ShaderWrite:     return "shader write";
    case Usage::CopySrc:         return "copy source";
    case Usage::CopyDst:         return "copy destination";
    case Usage::IndirectArgs:    return "indirect arguments";
    }
    return "?";
}

std::string mib(u64 bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f MiB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

u64 resourceBytes(const ResourceNode& node) {
    return node.kind == ResourceKind::Buffer ? node.buffer.size : node.texture.estimatedBytes();
}

// The last version of `resource` written or attachment-read by a member of the
// group (the version the group leaves).  Re-derived here, not copied from
// buildRenderGroups, so the lint is an independent check of its store rule.
u32 finalVersionInGroup(const RenderGraph& graph, const CompiledGraph& c, const RenderGroup& g, u32 resource) {
    u32 version = 0;
    for (u32 pos = g.firstPosition; pos <= g.lastPosition && pos < c.order.size(); ++pos) {
        const PassNode& pass = graph.passes()[c.order[pos]];
        for (const auto* list : {&pass.reads, &pass.writes}) {
            for (const Access& a : *list) {
                if (a.resource == resource && isAttachment(a.usage)) version = std::max(version, a.version);
            }
        }
    }
    return version;
}

struct Reader {
    u32   pass  = kNone;
    Usage usage = Usage::ShaderRead;
};

// First live pass after position `after` reading (resource, version).
Reader laterReader(const RenderGraph& graph, const CompiledGraph& c, u32 after, u32 resource, u32 version) {
    for (u32 pos = after + 1; pos < c.order.size(); ++pos) {
        for (const Access& r : graph.passes()[c.order[pos]].reads) {
            if (r.resource == resource && r.version == version) return {c.order[pos], r.usage};
        }
    }
    return {};
}

// Last pass of the group touching the attachment (the one charged its store).
u32 lastToucher(const RenderGraph& graph, const CompiledGraph& c, const RenderGroup& g, u32 resource) {
    u32 found = kNone;
    for (u32 pos = g.firstPosition; pos <= g.lastPosition && pos < c.order.size(); ++pos) {
        const PassNode& pass = graph.passes()[c.order[pos]];
        for (const auto* list : {&pass.reads, &pass.writes}) {
            for (const Access& a : *list) {
                if (a.resource == resource && isAttachment(a.usage)) found = c.order[pos];
            }
        }
    }
    return found;
}

std::string groupName(const RenderGraph& graph, const CompiledGraph& c, u32 g) {
    const RenderGroup& grp = c.renderGroups[g];
    std::string names;
    for (u32 pos = grp.firstPosition; pos <= grp.lastPosition && pos < c.order.size(); ++pos) {
        if (!names.empty()) names += " + ";
        names += graph.passes()[c.order[pos]].name;
    }
    return "render group " + std::to_string(g) + " [" + names + "]";
}

} // namespace

std::vector<LintFinding> lintGraph(const RenderGraph& graph, const CompiledGraph& compiled) {
    std::vector<LintFinding> out;
    const auto& resources = graph.resources();
    const auto& passes    = graph.passes();
    if (!compiled.ok) return out;

    const auto add = [&](bool error, u32 resource, u32 pass, std::string message) {
        LintFinding f;
        f.error    = error;
        f.resource = resource;
        f.pass     = pass;
        f.message  = std::move(message);
        out.push_back(std::move(f));
    };

    // --- (a) store justification (and its converse), (c) conservative stores.
    for (u32 g = 0; g < compiled.renderGroups.size(); ++g) {
        const RenderGroup& grp = compiled.renderGroups[g];
        for (const AttachmentPlan& a : grp.attachments) {
            if (a.resource >= resources.size()) continue;
            const ResourceNode& node = resources[a.resource];
            const u32 version = finalVersionInGroup(graph, compiled, grp, a.resource);
            const bool isOutput = node.imported && (node.importFlags & ImportOutput);
            const Reader reader = laterReader(graph, compiled, grp.lastPosition, a.resource, version);
            const u32 toucher = lastToucher(graph, compiled, grp, a.resource);
            const bool needed = isOutput || reader.pass != kNone;
            if (a.store == StoreAction::Store && !needed) {
                add(true, a.resource, toucher,
                    "error: " + groupName(graph, compiled, g) + " stores '" + node.name + "' (version " +
                        std::to_string(version) + ") but no later pass reads that version and it is not a graph "
                        "output (wasted " + mib(resourceBytes(node)) + " of DRAM writes)");
            } else if (a.store == StoreAction::DontCare && needed) {
                add(true, a.resource, toucher,
                    "error: " + groupName(graph, compiled, g) + " drops '" + node.name + "' (version " +
                        std::to_string(version) + ") but " +
                        (isOutput ? std::string("it is a graph output")
                                  : "pass '" + passes[reader.pass].name + "' reads it later"));
            }
            if (a.store == StoreAction::Store && a.readOnly && needed) {
                add(false, a.resource, toucher,
                    "note: " + groupName(graph, compiled, g) + " stores read-only attachment '" + node.name +
                        "' again (" + mib(resourceBytes(node)) + ", contents unchanged) because " +
                        (reader.pass != kNone ? "pass '" + passes[reader.pass].name + "' reads it later as " +
                                                    usageLabel(reader.usage)
                                              : std::string("it is a graph output")) +
                        "; conservative: a load-only attachment is not assumed to still be in memory");
            }
            if (a.resource < compiled.memoryless.size() && compiled.memoryless[a.resource] &&
                (a.load == LoadAction::Load || a.store == StoreAction::Store)) {
                add(true, a.resource, toucher,
                    "error: '" + node.name + "' is memoryless but " + groupName(graph, compiled, g) + " " +
                        (a.load == LoadAction::Load ? "loads" : "stores") + " it");
            }
        }
    }

    // --- (b) transient textures used as attachments that are not memoryless.
    for (u32 r = 0; r < resources.size(); ++r) {
        const ResourceNode& node = resources[r];
        if (node.imported || node.kind != ResourceKind::Texture) continue;
        if (r < compiled.memoryless.size() && compiled.memoryless[r]) continue;

        std::vector<u32> groups;        // groups (in order) using it as an attachment
        std::vector<std::string> other; // non-attachment accesses
        bool attachment = false;
        for (u32 pos = 0; pos < compiled.order.size(); ++pos) {
            const PassNode& pass = passes[compiled.order[pos]];
            for (const auto* list : {&pass.reads, &pass.writes}) {
                for (const Access& a : *list) {
                    if (a.resource != r) continue;
                    if (isAttachment(a.usage)) {
                        attachment = true;
                        const u32 g = pos < compiled.groupOfPosition.size() ? compiled.groupOfPosition[pos] : kNone;
                        if (g != kNone && std::find(groups.begin(), groups.end(), g) == groups.end()) {
                            groups.push_back(g);
                        }
                    } else {
                        std::string s = std::string(usageLabel(a.usage)) + " by pass '" + pass.name + "'";
                        if (std::find(other.begin(), other.end(), s) == other.end()) other.push_back(s);
                    }
                }
            }
        }
        if (!attachment) continue;

        std::vector<std::string> reasons;
        bool storeExplained = false;
        u32 reportPass = kNone;
        for (const u32 gi : groups) {
            const RenderGroup& grp = compiled.renderGroups[gi];
            const AttachmentPlan* plan = nullptr;
            for (const AttachmentPlan& a : grp.attachments) {
                if (a.resource == r) plan = &a;
            }
            if (!plan) continue;
            if (reportPass == kNone) reportPass = lastToucher(graph, compiled, grp, r);
            const std::string gname = "group " + std::to_string(gi);
            if (plan->load == LoadAction::Load) {
                reasons.push_back("loaded in " + gname + (plan->readOnly ? " (read-only attachment use)" : " (Preserve)"));
            }
            if (plan->store == StoreAction::Store) {
                const u32 version = finalVersionInGroup(graph, compiled, grp, r);
                const Reader rd = laterReader(graph, compiled, grp.lastPosition, r, version);
                if (rd.pass != kNone) {
                    reasons.push_back("stored by " + gname + " because read later as " + usageLabel(rd.usage) +
                                      " by pass '" + passes[rd.pass].name + "'");
                    storeExplained = true;
                } else {
                    reasons.push_back("stored by " + gname);
                }
            }
        }
        for (size_t i = 1; i < groups.size(); ++i) {
            const RenderGroup& a = compiled.renderGroups[groups[i - 1]];
            const RenderGroup& b = compiled.renderGroups[groups[i]];
            const std::string span = "spans render groups " + std::to_string(groups[i - 1]) + "/" +
                                     std::to_string(groups[i]);
            if (b.firstPosition > a.lastPosition + 1) {
                reasons.push_back(span + " because pass '" + passes[compiled.order[a.lastPosition + 1]].name +
                                  "' sits between them");
            } else {
                reasons.push_back(span + " (pass '" + passes[compiled.order[b.firstPosition]].name +
                                  "' could not fuse with the previous group)");
            }
        }
        // Non-attachment accesses: a sampled read after a store is that
        // store's reason; everything else is listed.
        for (const std::string& s : other) {
            const bool isRead = s.rfind("sampled", 0) == 0 || s.rfind("copy source", 0) == 0 ||
                                s.rfind("indirect", 0) == 0;
            if (!(storeExplained && isRead)) reasons.push_back("also used as " + s);
        }
        if (reasons.empty()) reasons.push_back("no attachment-only use found");
        std::string msg = "note: transient '" + node.name + "' (" + formatName(node.texture.format) + " " +
                          std::to_string(node.texture.width) + "x" + std::to_string(node.texture.height) + ", " +
                          mib(node.texture.estimatedBytes()) + ") is not memoryless: ";
        for (size_t i = 0; i < reasons.size(); ++i) msg += (i ? "; " : "") + reasons[i];
        add(false, r, reportPass, std::move(msg));
    }
    return out;
}

} // namespace phosphor::rg
