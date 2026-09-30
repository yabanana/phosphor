#include "rendergraph/tbdr_passes.h"

#include <algorithm>
#include <string>

namespace phosphor::rg {

const char* loadActionName(LoadAction action) {
    switch (action) {
    case LoadAction::DontCare: return "DontCare";
    case LoadAction::Load:     return "Load";
    case LoadAction::Clear:    return "Clear";
    }
    return "?";
}

const char* storeActionName(StoreAction action) {
    switch (action) {
    case StoreAction::DontCare: return "DontCare";
    case StoreAction::Store:    return "Store";
    }
    return "?";
}

namespace {

constexpr u32 kNone = ~0u;

// One attachment binding of a pass (deduplicated per resource).
struct PassAttachment {
    u32  resource = 0;
    bool depth    = false;
    u32  slot     = 0;
};

struct PassAttachments {
    std::vector<PassAttachment> list;
    u32  width = 0, height = 0, sampleCount = 1;
    std::string error; // empty when consistent
};

bool contains(const std::vector<u32>& v, u32 value) { return std::find(v.begin(), v.end(), value) != v.end(); }

void addUnique(std::vector<u32>& v, u32 value) {
    if (!contains(v, value)) v.push_back(value);
}

PassAttachments collectAttachments(const RenderGraph& graph, const PassNode& pass) {
    PassAttachments out;
    const auto& resources = graph.resources();
    const auto add = [&](u32 resource, bool depth, u32 slot) {
        for (const PassAttachment& a : out.list) {
            if (a.resource == resource) return; // Preserve write + its paired read
        }
        out.list.push_back({resource, depth, slot});
    };
    for (const Access& w : pass.writes) {
        if (isAttachment(w.usage)) add(w.resource, w.usage == Usage::DepthAttachment, w.slot);
    }
    for (const Access& r : pass.reads) {
        if (r.usage == Usage::DepthRead) add(r.resource, true, 0);
        // Per-pixel read of a color attachment the pass does not write
        // (PassBuilder::readColor; a Preserve write was added above).
        if (r.usage == Usage::ColorAttachment) add(r.resource, false, r.slot);
    }
    if (out.list.empty()) {
        out.error = "pass '" + pass.name + "': raster pass without attachments";
        return out;
    }
    const TextureDesc& first = resources[out.list.front().resource].texture;
    out.width       = first.width;
    out.height      = first.height;
    out.sampleCount = std::max(first.sampleCount, 1u);
    u32 depthCount = 0;
    for (size_t i = 0; i < out.list.size(); ++i) {
        const PassAttachment& a = out.list[i];
        const TextureDesc& t = resources[a.resource].texture;
        if (t.width != out.width || t.height != out.height || std::max(t.sampleCount, 1u) != out.sampleCount) {
            out.error = "pass '" + pass.name + "': attachments differ in size or sample count ('" +
                        resources[out.list.front().resource].name + "' vs '" + resources[a.resource].name + "')";
            return out;
        }
        if (a.depth) {
            ++depthCount;
        } else {
            for (size_t j = 0; j < i; ++j) {
                if (!out.list[j].depth && out.list[j].slot == a.slot) {
                    out.error = "pass '" + pass.name + "': color slot " + std::to_string(a.slot) +
                                " bound to two resources";
                    return out;
                }
            }
        }
    }
    if (depthCount > 1) out.error = "pass '" + pass.name + "': more than one depth attachment";
    return out;
}

// Incremental state of the group being built, used to decide whether the next
// raster pass can join it.
struct GroupState {
    bool open = false;
    u32  width = 0, height = 0, sampleCount = 1;
    std::vector<PassAttachment> attachments;  // union over members
    std::vector<u32> written;                 // resources written by any member (any usage)
    std::vector<u32> writtenNonAttachment;    // ... via ShaderWrite/CopyDst
    std::vector<u32> readNonAttachment;       // resources read outside an attachment

    void reset() { *this = GroupState{}; }
};

bool canJoin(const GroupState& g, const PassNode& pass, const PassAttachments& pa) {
    if (pa.width != g.width || pa.height != g.height || pa.sampleCount != g.sampleCount) return false;
    for (const PassAttachment& a : pa.list) {
        for (const PassAttachment& b : g.attachments) {
            if (a.resource == b.resource) {
                // The same resource must keep its binding.
                if (a.depth != b.depth || (!a.depth && a.slot != b.slot)) return false;
            } else if (a.depth && b.depth) {
                return false; // a render pass has one depth attachment
            } else if (!a.depth && !b.depth && a.slot == b.slot) {
                return false; // slot bound to a different resource
            }
        }
    }
    for (const Access& r : pass.reads) {
        if (isAttachment(r.usage)) {
            // Per-pixel load of something the group produced: fine, unless it
            // was produced outside the tile (shader write).
            if (contains(g.writtenNonAttachment, r.resource)) return false;
        } else if (contains(g.written, r.resource)) {
            return false; // sampling/copying/indirect use needs a fragment barrier
        }
    }
    for (const Access& w : pass.writes) {
        if (contains(g.readNonAttachment, w.resource)) return false;
        // A Clear in the middle of a render pass cannot be honoured: clears
        // are load actions of the whole group.
        if (isAttachment(w.usage) && w.load == LoadIntent::Clear) {
            for (const PassAttachment& b : g.attachments) {
                if (b.resource == w.resource) return false;
            }
        }
    }
    return true;
}

void absorb(GroupState& g, const PassNode& pass, const PassAttachments& pa) {
    if (!g.open) {
        g.open        = true;
        g.width       = pa.width;
        g.height      = pa.height;
        g.sampleCount = pa.sampleCount;
    }
    for (const PassAttachment& a : pa.list) {
        const bool known = std::any_of(g.attachments.begin(), g.attachments.end(),
                                       [&](const PassAttachment& b) { return b.resource == a.resource; });
        if (!known) g.attachments.push_back(a);
    }
    for (const Access& r : pass.reads) {
        if (!isAttachment(r.usage)) addUnique(g.readNonAttachment, r.resource);
    }
    for (const Access& w : pass.writes) {
        addUnique(g.written, w.resource);
        if (!isAttachment(w.usage)) addUnique(g.writtenNonAttachment, w.resource);
    }
}

// Load/store plan of a finished group covering positions [first, last].
RenderGroup buildGroup(const RenderGraph& graph, const CompiledGraph& c, u32 first, u32 last, const GroupState& state) {
    const auto& passes    = graph.passes();
    const auto& resources = graph.resources();
    RenderGroup group;
    group.firstPosition = first;
    group.lastPosition  = last;
    group.width         = state.width;
    group.height        = state.height;
    group.sampleCount   = state.sampleCount;

    std::vector<u32>  finalVersion; // parallel to group.attachments
    std::vector<bool> hasClear;
    const auto find = [&](u32 resource) -> size_t {
        for (size_t i = 0; i < group.attachments.size(); ++i) {
            if (group.attachments[i].resource == resource) return i;
        }
        return group.attachments.size();
    };

    for (u32 pos = first; pos <= last; ++pos) {
        const PassNode& pass = passes[c.order[pos]];
        for (const Access& w : pass.writes) {
            if (!isAttachment(w.usage)) continue;
            const size_t i = find(w.resource);
            if (i == group.attachments.size()) {
                AttachmentPlan a;
                a.resource = w.resource;
                a.depth    = w.usage == Usage::DepthAttachment;
                a.slot     = a.depth ? 0 : w.slot;
                switch (w.load) {
                case LoadIntent::Clear:    a.load = LoadAction::Clear; a.clear = w.clear; break;
                case LoadIntent::Preserve: a.load = LoadAction::Load; break;
                case LoadIntent::Discard:  a.load = LoadAction::DontCare; break;
                }
                group.attachments.push_back(a);
                finalVersion.push_back(w.version);
                hasClear.push_back(w.load == LoadIntent::Clear);
            } else {
                AttachmentPlan& a = group.attachments[i];
                a.readOnly = false;
                if (!hasClear[i] && w.load == LoadIntent::Clear) {
                    a.clear = w.clear;
                    hasClear[i] = true;
                }
                finalVersion[i] = std::max(finalVersion[i], w.version);
            }
        }
        for (const Access& r : pass.reads) {
            if (r.usage != Usage::DepthRead && r.usage != Usage::ColorAttachment) continue;
            const size_t i = find(r.resource);
            if (i == group.attachments.size()) {
                // First use is a read-only depth test or a per-pixel color
                // read: becomes a write only if a later member writes it.
                AttachmentPlan a;
                a.resource = r.resource;
                a.depth    = r.usage == Usage::DepthRead;
                a.slot     = a.depth ? 0 : r.slot;
                a.load     = LoadAction::Load;
                a.readOnly = true;
                group.attachments.push_back(a);
                finalVersion.push_back(r.version);
                hasClear.push_back(false);
            } else {
                finalVersion[i] = std::max(finalVersion[i], r.version);
            }
        }
    }

    // A DepthRead entry whose first access was the paired read of a
    // DepthAttachment write in the same pass is not read-only: the write loop
    // above runs first for each pass, so it already cleared readOnly (the
    // entry is created by the write with readOnly = false).

    // Store: the version the group leaves is read by a later live pass (any
    // usage, attachment loads included) or is the output of the graph.  A
    // read-only depth whose contents are needed later is stored too: the
    // contents are unchanged, but a load-only attachment cannot be assumed to
    // still be in memory, so this is the conservative choice.
    for (size_t i = 0; i < group.attachments.size(); ++i) {
        AttachmentPlan& a = group.attachments[i];
        const ResourceNode& node = resources[a.resource];
        bool store = node.imported && (node.importFlags & ImportOutput);
        for (u32 pos = last + 1; pos < c.order.size() && !store; ++pos) {
            for (const Access& r : passes[c.order[pos]].reads) {
                if (r.resource == a.resource && r.version == finalVersion[i]) {
                    store = true;
                    break;
                }
            }
        }
        a.store = store ? StoreAction::Store : StoreAction::DontCare;
    }
    return group;
}

} // namespace

bool canJoinGroup(const RenderGraph& graph, const std::vector<u32>& members, u32 pass) {
    const auto& passes = graph.passes();
    if (members.empty() || pass >= passes.size()) return false;
    const PassNode& node = passes[pass];
    if (node.type != PassType::Raster || node.queue != Queue::Graphics) return false;
    GroupState state;
    for (const u32 m : members) {
        if (m >= passes.size() || passes[m].type != PassType::Raster) return false;
        const PassAttachments pa = collectAttachments(graph, passes[m]);
        if (!pa.error.empty()) return false;
        absorb(state, passes[m], pa);
    }
    const PassAttachments pa = collectAttachments(graph, node);
    return pa.error.empty() && canJoin(state, node, pa);
}

void buildRenderGroups(const RenderGraph& graph, CompiledGraph& compiled, bool fuse) {
    compiled.renderGroups.clear();
    compiled.encoders.clear();
    compiled.groupOfPosition.assign(compiled.order.size(), kNone);
    compiled.encoderOfPosition.assign(compiled.order.size(), kNone);
    compiled.memoryless.assign(graph.resources().size(), false);
    if (!compiled.ok) return;

    const auto& passes = graph.passes();
    const u32 count = static_cast<u32>(compiled.order.size());

    // --- Render groups.
    GroupState state;
    u32 groupFirst = 0;
    const auto closeGroup = [&](u32 lastPosition) {
        if (!state.open) return;
        compiled.renderGroups.push_back(buildGroup(graph, compiled, groupFirst, lastPosition, state));
        state.reset();
    };
    for (u32 pos = 0; pos < count; ++pos) {
        const PassNode& pass = passes[compiled.order[pos]];
        if (pass.type != PassType::Raster) {
            closeGroup(pos - 1);
            continue;
        }
        const PassAttachments pa = collectAttachments(graph, pass);
        if (!pa.error.empty()) {
            compiled.errors.push_back(pa.error);
            compiled.ok = false;
            closeGroup(pos - 1);
            continue; // the graph is invalid anyway: no group for this pass
        }
        if (state.open && !(fuse && pass.queue == Queue::Graphics && canJoin(state, pass, pa))) {
            closeGroup(pos - 1);
        }
        if (!state.open) groupFirst = pos;
        absorb(state, pass, pa);
        compiled.groupOfPosition[pos] = static_cast<u32>(compiled.renderGroups.size());
    }
    if (count > 0) closeGroup(count - 1);
    if (!compiled.ok) return;

    // --- Memoryless: transient textures only touched as attachments of one
    // group that neither loads nor stores them.
    const u32 resourceCount = static_cast<u32>(graph.resources().size());
    std::vector<u32>  groupOf(resourceCount, kNone);
    std::vector<bool> eligible(resourceCount, true);
    for (u32 pos = 0; pos < count; ++pos) {
        const PassNode& pass = passes[compiled.order[pos]];
        for (const auto* list : {&pass.reads, &pass.writes}) {
            for (const Access& a : *list) {
                if (!isAttachment(a.usage)) {
                    eligible[a.resource] = false;
                } else if (groupOf[a.resource] == kNone) {
                    groupOf[a.resource] = compiled.groupOfPosition[pos];
                } else if (groupOf[a.resource] != compiled.groupOfPosition[pos]) {
                    eligible[a.resource] = false;
                }
            }
        }
    }
    for (const RenderGroup& group : compiled.renderGroups) {
        for (const AttachmentPlan& a : group.attachments) {
            const ResourceNode& node = graph.resources()[a.resource];
            if (!eligible[a.resource] || node.imported || node.kind != ResourceKind::Texture) continue;
            if (a.load != LoadAction::Load && a.store == StoreAction::DontCare) compiled.memoryless[a.resource] = true;
        }
    }

    // --- Encoders: a render group is one Raster encoder; consecutive
    // Compute/Blit positions on the same queue share a Compute encoder.
    u32 groupsSeen = 0;
    for (u32 pos = 0; pos < count; ++pos) {
        const PassNode& pass = passes[compiled.order[pos]];
        if (pass.type == PassType::Raster) {
            const u32 g = compiled.groupOfPosition[pos];
            if (g == groupsSeen) {
                EncoderPlan e;
                e.type          = PassType::Raster;
                e.queue         = Queue::Graphics;
                e.firstPosition = compiled.renderGroups[g].firstPosition;
                e.lastPosition  = compiled.renderGroups[g].lastPosition;
                e.renderGroup   = g;
                compiled.encoders.push_back(e);
                ++groupsSeen;
            }
        } else {
            const bool extend = !compiled.encoders.empty() && compiled.encoders.back().type == PassType::Compute &&
                                compiled.encoders.back().queue == pass.queue &&
                                compiled.encoders.back().lastPosition + 1 == pos;
            if (extend) {
                compiled.encoders.back().lastPosition = pos;
            } else {
                EncoderPlan e;
                e.type          = PassType::Compute;
                e.queue         = pass.queue;
                e.firstPosition = pos;
                e.lastPosition  = pos;
                compiled.encoders.push_back(e);
            }
        }
        compiled.encoderOfPosition[pos] = static_cast<u32>(compiled.encoders.size() - 1);
    }
}

} // namespace phosphor::rg
