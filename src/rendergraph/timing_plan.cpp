#include "rendergraph/timing_plan.h"

namespace phosphor::rg {

namespace {

u64 resourceBytes(const ResourceNode& node) {
    return node.kind == ResourceKind::Buffer ? node.buffer.size : node.texture.estimatedBytes();
}

bool isMemoryless(const CompiledGraph& c, u32 resource) {
    return resource < c.memoryless.size() && c.memoryless[resource];
}

bool isLive(const CompiledGraph& c, u32 pass) {
    return !(pass < c.culled.size() && c.culled[pass]) && c.position(pass) != ~0u;
}

bool touchesAttachment(const PassNode& pass, u32 resource) {
    for (const Access& a : pass.reads) {
        if (a.resource == resource && isAttachment(a.usage)) return true;
    }
    for (const Access& a : pass.writes) {
        if (a.resource == resource && isAttachment(a.usage)) return true;
    }
    return false;
}

} // namespace

std::vector<PassTraffic> estimatePassBandwidth(const RenderGraph& graph, const CompiledGraph& compiled) {
    const auto& passes    = graph.passes();
    const auto& resources = graph.resources();
    std::vector<PassTraffic> out(passes.size());

    // Attachment traffic of a render group: loads to the first member that
    // touches the attachment, stores to the last one.
    for (const RenderGroup& g : compiled.renderGroups) {
        for (const AttachmentPlan& a : g.attachments) {
            if (a.resource >= resources.size() || isMemoryless(compiled, a.resource)) continue;
            const u64 bytes = resourceBytes(resources[a.resource]);
            u32 first = ~0u, last = ~0u;
            for (u32 pos = g.firstPosition; pos <= g.lastPosition && pos < compiled.order.size(); ++pos) {
                const u32 p = compiled.order[pos];
                if (!touchesAttachment(passes[p], a.resource)) continue;
                if (first == ~0u) first = p;
                last = p;
            }
            if (first == ~0u) { // not declared by any member: charge the group's first pass
                if (g.firstPosition >= compiled.order.size()) continue;
                first = last = compiled.order[g.firstPosition];
            }
            if (a.load == LoadAction::Load) out[first].readBytes += bytes;
            if (a.store == StoreAction::Store) out[last].writeBytes += bytes;
        }
    }

    auto hasGroup = [&](u32 pass) {
        const u32 pos = compiled.position(pass);
        return pos < compiled.groupOfPosition.size() && compiled.groupOfPosition[pos] < compiled.renderGroups.size();
    };

    for (u32 p = 0; p < passes.size(); ++p) {
        if (!isLive(compiled, p)) continue;
        const PassNode& pass = passes[p];
        // Same fallback as estimateBandwidth(): a raster pass without group
        // information counts its own attachment accesses.
        const bool fallback = pass.type == PassType::Raster && !hasGroup(p);
        for (const Access& a : pass.reads) {
            if (a.resource >= resources.size()) continue;
            const u64 bytes = resourceBytes(resources[a.resource]);
            if (isAttachment(a.usage)) {
                if (fallback && !isMemoryless(compiled, a.resource)) out[p].readBytes += bytes;
                continue;
            }
            out[p].readBytes += bytes;
        }
        for (const Access& a : pass.writes) {
            if (a.resource >= resources.size()) continue;
            const u64 bytes = resourceBytes(resources[a.resource]);
            if (isAttachment(a.usage)) {
                if (fallback && !isMemoryless(compiled, a.resource)) out[p].writeBytes += bytes;
                continue;
            }
            out[p].writeBytes += bytes;
        }
    }
    return out;
}

TimingPlan buildTimingPlan(const RenderGraph& graph, const CompiledGraph& compiled, u32 maxCommits) {
    TimingPlan plan;
    plan.commitStartQueries = maxCommits;
    plan.unitOfPosition.assign(compiled.order.size(), ~0u);
    const auto& passes = graph.passes();
    const std::vector<PassTraffic> traffic = estimatePassBandwidth(graph, compiled);

    auto addUnit = [&](u32 encoderIndex, const EncoderPlan& e, u32 first, u32 last, TimestampKind kind) {
        TimedUnit unit;
        unit.queue         = e.queue;
        unit.kind          = kind;
        unit.encoder       = encoderIndex;
        unit.firstPosition = first;
        unit.lastPosition  = last;
        unit.fused         = last > first && kind == TimestampKind::RenderEnd;
        for (u32 pos = first; pos <= last && pos < compiled.order.size(); ++pos) {
            const u32 p = compiled.order[pos];
            unit.name += (unit.name.empty() ? "" : " + ") + passes[p].name;
            unit.passes.push_back(p);
            if (p < traffic.size()) unit.dramBytes += traffic[p].total();
            plan.unitOfPosition[pos] = static_cast<u32>(plan.units.size());
        }
        plan.units.push_back(std::move(unit));
    };

    for (u32 i = 0; i < compiled.encoders.size(); ++i) {
        const EncoderPlan& e = compiled.encoders[i];
        if (e.lastPosition < e.firstPosition || e.firstPosition >= compiled.order.size()) continue;
        if (e.type == PassType::Raster) {
            addUnit(i, e, e.firstPosition, e.lastPosition, TimestampKind::RenderEnd);
        } else if (e.firstPosition == e.lastPosition) {
            addUnit(i, e, e.firstPosition, e.lastPosition, TimestampKind::ComputeEnd);
        } else {
            for (u32 pos = e.firstPosition; pos <= e.lastPosition; ++pos) {
                addUnit(i, e, pos, pos, TimestampKind::ComputePassEnd);
            }
        }
    }
    return plan;
}

} // namespace phosphor::rg
