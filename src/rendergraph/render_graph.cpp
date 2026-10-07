#include "rendergraph/render_graph.h"
#include "rendergraph/aliasing.h"
#include "rendergraph/barrier_plan.h"
#include "rendergraph/graph_lint.h"
#include "rendergraph/tbdr_passes.h"

#include <algorithm>
#include <string>
#include <tuple>

namespace phosphor::rg {

// --- Formats ----------------------------------------------------------------

u32 bytesPerPixel(Format format) {
    switch (format) {
    case Format::Unknown:              return 0;
    case Format::R8Unorm:              return 1;
    case Format::RG8Unorm:             return 2;
    case Format::R16Float:             return 2;
    case Format::Depth16Unorm:         return 2;
    case Format::RGBA8Unorm:
    case Format::RGBA8Srgb:
    case Format::BGRA8Unorm:
    case Format::BGRA8Srgb:
    case Format::RG16Float:
    case Format::R32Float:
    case Format::R32Uint:
    case Format::RG11B10Float:
    case Format::RGB10A2Unorm:
    case Format::Depth32Float:         return 4;
    case Format::Depth32FloatStencil8: return 5;
    case Format::RGBA16Float:
    case Format::RG32Float:            return 8;
    case Format::RGBA32Float:          return 16;
    }
    return 0;
}

bool isDepthFormat(Format format) {
    return format == Format::Depth16Unorm || format == Format::Depth32Float ||
           format == Format::Depth32FloatStencil8;
}

const char* formatName(Format format) {
    switch (format) {
    case Format::Unknown:              return "unknown";
    case Format::R8Unorm:              return "R8Unorm";
    case Format::RG8Unorm:             return "RG8Unorm";
    case Format::RGBA8Unorm:           return "RGBA8Unorm";
    case Format::RGBA8Srgb:            return "RGBA8Srgb";
    case Format::BGRA8Unorm:           return "BGRA8Unorm";
    case Format::BGRA8Srgb:            return "BGRA8Srgb";
    case Format::R16Float:             return "R16Float";
    case Format::RG16Float:            return "RG16Float";
    case Format::RGBA16Float:          return "RGBA16Float";
    case Format::R32Float:             return "R32Float";
    case Format::RG32Float:            return "RG32Float";
    case Format::RGBA32Float:          return "RGBA32Float";
    case Format::R32Uint:              return "R32Uint";
    case Format::RG11B10Float:         return "RG11B10Float";
    case Format::RGB10A2Unorm:         return "RGB10A2Unorm";
    case Format::Depth16Unorm:         return "Depth16Unorm";
    case Format::Depth32Float:         return "Depth32Float";
    case Format::Depth32FloatStencil8: return "Depth32FloatStencil8";
    }
    return "unknown";
}

u64 TextureDesc::estimatedBytes() const {
    u64 bytes = 0;
    u64 w = width, h = height;
    for (u32 mip = 0; mip < std::max(mipLevels, 1u); ++mip) {
        bytes += w * h;
        w = std::max<u64>(w / 2, 1);
        h = std::max<u64>(h / 2, 1);
    }
    return bytes * bytesPerPixel(format) * std::max(depth, 1u) * std::max(sampleCount, 1u);
}

bool isWrite(Usage usage) {
    switch (usage) {
    case Usage::ColorAttachment:
    case Usage::DepthAttachment:
    case Usage::ShaderWrite:
    case Usage::CopyDst:
        return true;
    default:
        return false;
    }
}

bool isAttachment(Usage usage) {
    return usage == Usage::ColorAttachment || usage == Usage::DepthAttachment || usage == Usage::DepthRead;
}

// --- Setup ------------------------------------------------------------------

TextureRef PassBuilder::createTexture(const std::string& name, const TextureDesc& desc) {
    if (desc.format == Format::Unknown || desc.width == 0 || desc.height == 0) {
        graph_.error("texture '" + name + "': format and size are required");
    }
    return {graph_.addResource(name, ResourceKind::Texture, desc, {}, false, ImportNone), 0};
}

BufferRef PassBuilder::createBuffer(const std::string& name, const BufferDesc& desc) {
    if (desc.size == 0) graph_.error("buffer '" + name + "': size is required");
    return {graph_.addResource(name, ResourceKind::Buffer, {}, desc, false, ImportNone), 0};
}

TextureRef PassBuilder::writeColor(TextureRef target, u32 slot, LoadIntent load, const ClearValue& clear) {
    const u32 v = graph_.addWrite(pass_, target.resource, target.version, Usage::ColorAttachment, StageFragment,
                                  slot, load, clear);
    return {target.resource, v};
}

TextureRef PassBuilder::writeDepth(TextureRef target, LoadIntent load, const ClearValue& clear) {
    const u32 v = graph_.addWrite(pass_, target.resource, target.version, Usage::DepthAttachment, StageFragment,
                                  0, load, clear);
    return {target.resource, v};
}

void PassBuilder::readDepth(TextureRef target) {
    graph_.addRead(pass_, target.resource, target.version, Usage::DepthRead, StageFragment);
}

void PassBuilder::setTileSize(u32 width, u32 height) {
    auto &p = graph_.passes_[pass_];
    if (p.type != PassType::Raster ||
        !((width == 16 && height == 16) || (width == 32 && (height == 16 || height == 32)))) {
        graph_.error("Tile dispatch requires a raster pass and supported 16x16, 32x16 or 32x32 tile");
        return;
    }
    p.tileWidth = width;
    p.tileHeight = height;
}

void PassBuilder::readColor(TextureRef target, u32 slot, Stages stages) {
    RenderGraph& g = graph_;
    PassNode& p = g.passes_[pass_];
    if (target.resource >= g.resources_.size()) {
        g.error("pass '" + p.name + "': read of an invalid resource handle");
        return;
    }
    const ResourceNode& r = g.resources_[target.resource];
    if (p.type != PassType::Raster) {
        g.error("pass '" + p.name + "': attachments need a raster pass");
        return;
    }
    if (r.kind != ResourceKind::Texture || isDepthFormat(r.texture.format)) {
        g.error("pass '" + p.name + "': '" + r.name + "' is not a color texture");
        return;
    }
    // Recorded like the paired read of a Preserve write: added as a shader
    // read (validation of the version), then given the attachment usage.
    const size_t before = p.reads.size();
    g.addRead(pass_, target.resource, target.version, Usage::ShaderRead, stages);
    if (p.reads.size() > before) {
        p.reads.back().usage = Usage::ColorAttachment;
        p.reads.back().slot  = slot;
    }
}

void PassBuilder::read(TextureRef texture, Usage usage, Stages stages) {
    graph_.addRead(pass_, texture.resource, texture.version, usage, stages);
}

void PassBuilder::read(BufferRef buffer, Usage usage, Stages stages) {
    graph_.addRead(pass_, buffer.resource, buffer.version, usage, stages);
}

void PassBuilder::read(AccelerationStructureRef structure, Usage usage, Stages stages) {
    if (structure.resource < graph_.resources().size() &&
        graph_.resources()[structure.resource].kind != ResourceKind::AccelerationStructure)
        graph_.error("acceleration-structure access requires an acceleration-structure import");
    graph_.addRead(pass_, structure.resource, structure.version, usage, stages);
}

AccelerationStructureRef PassBuilder::write(AccelerationStructureRef structure, Usage usage, Stages stages) {
    if (structure.resource < graph_.resources().size() &&
        graph_.resources()[structure.resource].kind != ResourceKind::AccelerationStructure)
        graph_.error("acceleration-structure access requires an acceleration-structure import");
    if (!(stages & StageAccelerationStructure))
        graph_.error("acceleration-structure writes require the acceleration-structure stage");
    return {structure.resource,
            graph_.addWrite(pass_, structure.resource, structure.version, usage, stages, 0, LoadIntent::Discard, {})};
}

TextureRef PassBuilder::write(TextureRef texture, Usage usage, Stages stages) {
    return {texture.resource,
            graph_.addWrite(pass_, texture.resource, texture.version, usage, stages, 0, LoadIntent::Discard, {})};
}

BufferRef PassBuilder::write(BufferRef buffer, Usage usage, Stages stages) {
    return {buffer.resource,
            graph_.addWrite(pass_, buffer.resource, buffer.version, usage, stages, 0, LoadIntent::Discard, {})};
}

void PassBuilder::setSideEffect() { graph_.passes_[pass_].sideEffect = true; }
void PassBuilder::setHints(u32 hints) { graph_.passes_[pass_].hints = hints; }
void PassBuilder::setParallelChunks(u32 chunks) {
    if (graph_.passes_[pass_].type == PassType::External && chunks > 1)
        graph_.error("external passes cannot split framework-owned encoders");
    graph_.passes_[pass_].parallelChunks = std::max(chunks, 1u);
}

void PassBuilder::setCost(const PassCost& cost) { graph_.passes_[pass_].cost = cost; }

void PassBuilder::setProfileShaders(const std::string& functions) {
    std::vector<std::string>& out = graph_.passes_[pass_].profileShaders;
    out.clear();
    size_t start = 0;
    while (start <= functions.size()) {
        const size_t comma = functions.find(',', start);
        const size_t end   = comma == std::string::npos ? functions.size() : comma;
        if (end > start) out.push_back(functions.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
}

TextureRef RenderGraph::importTexture(const std::string& name, const TextureDesc& desc, u32 flags) {
    if (desc.format == Format::Unknown || desc.width == 0 || desc.height == 0) {
        error("imported texture '" + name + "': format and size are required");
    }
    return {addResource(name, ResourceKind::Texture, desc, {}, true, flags), 0};
}

BufferRef RenderGraph::importBuffer(const std::string& name, const BufferDesc& desc, u32 flags) {
    return {addResource(name, ResourceKind::Buffer, {}, desc, true, flags), 0};
}

AccelerationStructureRef RenderGraph::importAccelerationStructure(const std::string& name, u64 bytes, u32 flags) {
    return {addResource(name, ResourceKind::AccelerationStructure, {}, {bytes}, true, flags), 0};
}

u32 RenderGraph::addPass(const std::string& name, PassType type, Queue queue, const SetupFn& setup,
                         ExecuteFn execute) {
    const u32 index = static_cast<u32>(passes_.size());
    PassNode& node = passes_.emplace_back();
    node.name    = name;
    node.type    = type;
    node.queue   = queue;
    node.execute = std::move(execute);
    if (type == PassType::Raster && queue != Queue::Graphics) {
        error("pass '" + name + "': raster passes run on the graphics queue");
    }
    if (type == PassType::External && queue != Queue::Graphics)
        error("external passes require the graphics queue");
    PassBuilder builder(*this, index);
    if (setup) setup(builder);
    return index;
}

void RenderGraph::reset() {
    passes_.clear();
    resources_.clear();
    errors_.clear();
}

u32 RenderGraph::addResource(const std::string& name, ResourceKind kind, const TextureDesc& t, const BufferDesc& b,
                             bool imported, u32 flags) {
    ResourceNode& node = resources_.emplace_back();
    node.name        = name;
    node.kind        = kind;
    node.texture     = t;
    node.buffer      = b;
    node.imported    = imported;
    node.importFlags = flags;
    return static_cast<u32>(resources_.size() - 1);
}

void RenderGraph::error(const std::string& message) { errors_.push_back(message); }

void RenderGraph::addRead(u32 pass, u32 resource, u32 version, Usage usage, Stages stages) {
    PassNode& p = passes_[pass];
    if (resource >= resources_.size()) {
        error("pass '" + p.name + "': read of an invalid resource handle");
        return;
    }
    const ResourceNode& r = resources_[resource];
    if (isWrite(usage)) {
        error("pass '" + p.name + "': '" + r.name + "' read with a write usage");
        return;
    }
    if (version >= r.versions) {
        error("pass '" + p.name + "': '" + r.name + "' read of a version that does not exist");
        return;
    }
    if (version == 0 && !(r.imported && (r.importFlags & ImportContentsDefined))) {
        error("pass '" + p.name + "': '" + r.name + "' read before anything wrote it");
        return;
    }
    if (usage == Usage::DepthRead && (r.kind != ResourceKind::Texture || !isDepthFormat(r.texture.format))) {
        error("pass '" + p.name + "': '" + r.name + "' is not a depth texture");
        return;
    }
    if (usage == Usage::DepthRead && p.type != PassType::Raster) {
        error("pass '" + p.name + "': depth attachments need a raster pass");
        return;
    }
    Access a;
    a.resource = resource;
    a.version  = version;
    a.usage    = usage;
    a.stages   = stages;
    p.reads.push_back(a);
}

u32 RenderGraph::addWrite(u32 pass, u32 resource, u32 version, Usage usage, Stages stages, u32 slot, LoadIntent load,
                          const ClearValue& clear) {
    PassNode& p = passes_[pass];
    if (resource >= resources_.size()) {
        error("pass '" + p.name + "': write of an invalid resource handle");
        return version;
    }
    ResourceNode& r = resources_[resource];
    if (!isWrite(usage)) {
        error("pass '" + p.name + "': '" + r.name + "' written with a read usage");
        return version;
    }
    if (version + 1 != r.versions) {
        error("pass '" + p.name + "': '" + r.name +
              "' written through a stale handle (another pass already wrote a newer version)");
        return version;
    }
    for (const Access& w : p.writes) {
        if (w.resource == resource) {
            error("pass '" + p.name + "': '" + r.name + "' written twice by the same pass");
            return version;
        }
    }
    if (isAttachment(usage)) {
        if (p.type != PassType::Raster) {
            error("pass '" + p.name + "': attachments need a raster pass");
            return version;
        }
        const bool depth = usage == Usage::DepthAttachment;
        if (r.kind != ResourceKind::Texture || isDepthFormat(r.texture.format) != depth) {
            error("pass '" + p.name + "': '" + r.name + (depth ? "' is not a depth texture" : "' is not a color texture"));
            return version;
        }
        if (load == LoadIntent::Preserve) {
            // Blending/loading reads the previous contents: recorded as a read
            // with the attachment usage (an attachment load, see PassNode).
            const size_t before = p.reads.size();
            addRead(pass, resource, version, depth ? Usage::DepthRead : Usage::ShaderRead, stages);
            if (p.reads.size() > before) {
                p.reads.back().usage = depth ? Usage::DepthRead : Usage::ColorAttachment;
                p.reads.back().slot  = slot;
            }
        }
    }
    Access a;
    a.resource = resource;
    a.version  = version + 1;
    a.usage    = usage;
    a.stages   = stages;
    a.slot     = slot;
    a.load     = load;
    a.clear    = clear;
    p.writes.push_back(a);
    ++r.versions;
    return version + 1;
}

// --- Compilation: validation, culling, order, lifetimes (F2.1) ---------------

namespace {

constexpr u32 kNone = ~0u;

struct VersionInfo {
    std::vector<std::vector<u32>> writer;  // [resource][version] -> pass (kNone for 0)
    std::vector<std::vector<std::vector<u32>>> readers; // [resource][version] -> passes
};

VersionInfo indexVersions(const RenderGraph& graph) {
    VersionInfo info;
    const auto& res = graph.resources();
    info.writer.resize(res.size());
    info.readers.resize(res.size());
    for (size_t r = 0; r < res.size(); ++r) {
        info.writer[r].assign(res[r].versions, kNone);
        info.readers[r].resize(res[r].versions);
    }
    const auto& passes = graph.passes();
    for (u32 p = 0; p < passes.size(); ++p) {
        for (const Access& a : passes[p].writes) info.writer[a.resource][a.version] = p;
        for (const Access& a : passes[p].reads) {
            auto& list = info.readers[a.resource][a.version];
            if (std::find(list.begin(), list.end(), p) == list.end()) list.push_back(p);
        }
    }
    return info;
}

} // namespace

CompiledGraph compileOrder(const RenderGraph& graph, const std::vector<u32>& forcedOrder) {
    CompiledGraph c;
    const auto& passes    = graph.passes();
    const auto& resources = graph.resources();
    const u32 passCount = static_cast<u32>(passes.size());

    if (!graph.errors().empty()) {
        c.errors = graph.errors();
        return c;
    }
    const VersionInfo v = indexVersions(graph);

    // --- Culling: live = side effects, final versions of outputs, and the
    // writers of every version a live pass reads.
    std::vector<bool> live(passCount, false);
    std::vector<u32> stack;
    const auto markLive = [&](u32 p) {
        if (p != kNone && !live[p]) {
            live[p] = true;
            stack.push_back(p);
        }
    };
    for (u32 p = 0; p < passCount; ++p) {
        if (passes[p].sideEffect) markLive(p);
    }
    for (u32 r = 0; r < resources.size(); ++r) {
        const ResourceNode& node = resources[r];
        if (node.imported && (node.importFlags & ImportOutput) && node.versions > 1) {
            markLive(v.writer[r][node.versions - 1]);
        }
    }
    while (!stack.empty()) {
        const u32 p = stack.back();
        stack.pop_back();
        for (const Access& a : passes[p].reads) {
            if (a.version > 0) markLive(v.writer[a.resource][a.version]);
        }
    }

    // --- Dependencies between live passes.
    std::vector<Dependency> deps;
    const auto addDep = [&](u32 from, u32 to, u32 resource, DepKind kind) {
        if (from == to) return;
        deps.push_back({from, to, resource, kind});
    };
    for (u32 p = 0; p < passCount; ++p) {
        if (!live[p]) continue;
        for (const Access& a : passes[p].reads) {
            if (a.version > 0) addDep(v.writer[a.resource][a.version], p, a.resource, DepKind::RAW);
        }
        for (const Access& a : passes[p].writes) {
            // Walk back over versions whose writers were culled: the live
            // readers of those versions still use the same memory.
            for (u32 u = a.version; u-- > 0;) {
                for (const u32 reader : v.readers[a.resource][u]) {
                    if (live[reader]) addDep(reader, p, a.resource, DepKind::WAR);
                }
                if (u == 0) break;
                const u32 w = v.writer[a.resource][u];
                if (live[w]) {
                    addDep(w, p, a.resource, DepKind::WAW);
                    break;
                }
            }
        }
    }
    std::sort(deps.begin(), deps.end(), [](const Dependency& a, const Dependency& b) {
        return std::tie(a.from, a.to, a.resource, a.kind) < std::tie(b.from, b.to, b.resource, b.kind);
    });
    deps.erase(std::unique(deps.begin(), deps.end(),
                           [](const Dependency& a, const Dependency& b) {
                               return a.from == b.from && a.to == b.to && a.resource == b.resource &&
                                      a.kind == b.kind;
                           }),
               deps.end());

    // --- Stable Kahn: among ready passes, declaration order wins, except that
    // a geometry-heavy pass is preferred right after a fragment-heavy one
    // (S-TBDR-6: its vertex work overlaps the previous fragment work).
    std::vector<std::vector<u32>> successors(passCount);
    std::vector<u32> indegree(passCount, 0);
    for (size_t i = 0; i < deps.size(); ++i) {
        if (i > 0 && deps[i].from == deps[i - 1].from && deps[i].to == deps[i - 1].to) continue;
        successors[deps[i].from].push_back(deps[i].to);
        ++indegree[deps[i].to];
    }
    std::vector<u32> ready;
    u32 liveCount = 0;
    for (u32 p = 0; p < passCount; ++p) {
        if (!live[p]) continue;
        ++liveCount;
        if (indegree[p] == 0) ready.push_back(p);
    }
    c.positionOfPass.assign(passCount, kNone);
    if (!forcedOrder.empty()) {
        // OPT-1.1: an order chosen by a plan.  Accepted only if it is a
        // topological order of exactly the live passes.
        for (const u32 p : forcedOrder) {
            if (p >= passCount || !live[p] || c.positionOfPass[p] != kNone) {
                c.errors.push_back(p >= passCount ? "forced order: pass index out of range"
                                   : !live[p]  ? "forced order: '" + passes[p].name + "' is culled"
                                               : "forced order: '" + passes[p].name + "' listed twice");
                c.order.clear();
                return c;
            }
            c.positionOfPass[p] = static_cast<u32>(c.order.size());
            c.order.push_back(p);
        }
        if (c.order.size() != liveCount) {
            c.errors.push_back("forced order: " + std::to_string(liveCount - c.order.size()) +
                               " live pass(es) missing");
            c.order.clear();
            return c;
        }
        for (const Dependency& d : deps) {
            if (c.positionOfPass[d.from] > c.positionOfPass[d.to]) {
                c.errors.push_back("forced order: '" + passes[d.to].name + "' runs before '" +
                                   passes[d.from].name + "', which it depends on");
                c.order.clear();
                return c;
            }
        }
        ready.clear();
    }
    while (!ready.empty() && forcedOrder.empty()) {
        auto pick = std::min_element(ready.begin(), ready.end());
        if (!c.order.empty() && (passes[c.order.back()].hints & HintFragmentHeavy)) {
            for (auto it = ready.begin(); it != ready.end(); ++it) {
                if ((passes[*it].hints & HintGeometryHeavy) &&
                    (!(passes[*pick].hints & HintGeometryHeavy) || *it < *pick)) {
                    pick = it;
                }
            }
        }
        const u32 p = *pick;
        ready.erase(pick);
        c.positionOfPass[p] = static_cast<u32>(c.order.size());
        c.order.push_back(p);
        for (const u32 s : successors[p]) {
            if (--indegree[s] == 0) ready.push_back(s);
        }
    }
    if (c.order.size() != liveCount) {
        std::string cycle = "dependency cycle between passes:";
        for (u32 p = 0; p < passCount; ++p) {
            if (live[p] && c.positionOfPass[p] == kNone) cycle += " '" + passes[p].name + "'";
        }
        c.errors.push_back(cycle);
        c.order.clear();
        return c;
    }

    c.culled.resize(passCount);
    for (u32 p = 0; p < passCount; ++p) c.culled[p] = !live[p];
    c.dependencies = std::move(deps);

    // --- Lifetimes over positions.
    c.lifetimes.assign(resources.size(), Lifetime{});
    for (u32 pos = 0; pos < c.order.size(); ++pos) {
        const PassNode& node = passes[c.order[pos]];
        for (const auto* list : {&node.reads, &node.writes}) {
            for (const Access& a : *list) {
                Lifetime& l = c.lifetimes[a.resource];
                l.first = std::min(l.first, pos);
                l.last  = std::max(l.last, pos);
            }
        }
    }
    c.memoryless.assign(resources.size(), false);
    c.ok = true;
    return c;
}

CompiledGraph compile(const RenderGraph& graph, const CompileOptions& options) {
    CompiledGraph c = compileOrder(graph, options.order);
    if (!c.ok) return c;
    buildRenderGroups(graph, c, options.fuseRasterPasses);
    if (!c.ok) return c;
    if (options.sizer) c.aliasing = planAliasing(graph, c, *options.sizer, options.alias, options.aliasPolicy);
    // Queue syncs first: their positions become encoder boundaries, which
    // decide the scope of the barriers.
    buildQueueSyncs(graph, c);
    splitEncodersAtQueueSyncs(c);
    buildBarrierPlan(graph, c, defaultBarrierRules(), options.barrierPolicy);
    if (options.lint != LintMode::Off) {
        for (const LintFinding& f : lintGraph(graph, c)) {
            c.lint.push_back(f.message);
            if (options.lint == LintMode::Error && f.error) c.errors.push_back("lint: " + f.message);
        }
    }
    c.ok = c.errors.empty();
    return c;
}

} // namespace phosphor::rg
