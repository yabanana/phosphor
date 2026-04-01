#include "render_graph/render_graph.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <unordered_map>

namespace phosphor {

// -----------------------------------------------------------------------
//  PassBuilder
// -----------------------------------------------------------------------

RenderGraph::PassBuilder::PassBuilder(RenderGraph& graph, u32 passIndex)
    : graph_(graph)
    , passIndex_(passIndex)
{
}

void RenderGraph::PassBuilder::read(ResourceHandle        handle,
                                    VkPipelineStageFlags2 stage,
                                    VkAccessFlags2        access,
                                    VkImageLayout         layout)
{
    assert(handle != INVALID_RESOURCE && "reading INVALID_RESOURCE");
    graph_.passes_[passIndex_].reads.push_back({ handle, stage, access, layout });
}

void RenderGraph::PassBuilder::write(ResourceHandle        handle,
                                     VkPipelineStageFlags2 stage,
                                     VkAccessFlags2        access,
                                     VkImageLayout         layout)
{
    assert(handle != INVALID_RESOURCE && "writing INVALID_RESOURCE");
    graph_.passes_[passIndex_].writes.push_back({ handle, stage, access, layout });
}

ResourceHandle RenderGraph::PassBuilder::createTransientImage(const ImageDesc& desc)
{
    return graph_.registry_.createTransientImage(desc);
}

ResourceHandle RenderGraph::PassBuilder::createTransientBuffer(const BufferDesc& desc)
{
    return graph_.registry_.createTransientBuffer(desc);
}

void RenderGraph::PassBuilder::setSideEffect()
{
    graph_.passes_[passIndex_].hasSideEffects = true;
}

// -----------------------------------------------------------------------
//  RenderGraph
// -----------------------------------------------------------------------

RenderGraph::RenderGraph(ResourceRegistry& registry)
    : registry_(registry)
{
}

void RenderGraph::addPass(const std::string&                  name,
                          PassType                             type,
                          std::function<void(PassBuilder&)>    setup,
                          std::function<void(VkCommandBuffer)> execute)
{
    assert(!compiled_ && "cannot add passes after compile()");

    u32 index = static_cast<u32>(passes_.size());
    passes_.push_back({});
    PassNode& node = passes_.back();
    node.name    = name;
    node.type    = type;
    node.execute = std::move(execute);

    PassBuilder builder(*this, index);
    setup(builder);
}

// -----------------------------------------------------------------------
//  compile()
// -----------------------------------------------------------------------

void RenderGraph::compile()
{
    assert(!compiled_ && "compile() already called");

    if (passes_.empty()) {
        compiled_ = true;
        return;
    }

    buildEdges();
    topologicalSort();
    computeBarriers();

    compiled_ = true;

    LOG_DEBUG("RenderGraph: compiled %u passes into %u sorted nodes",
              static_cast<u32>(passes_.size()),
              static_cast<u32>(sortedOrder_.size()));
}

// Step 1 -- For every resource read, add an edge from its last writer.
void RenderGraph::buildEdges()
{
    // Map resource handle -> index of the pass that last wrote it.
    // A resource may be written by multiple passes (e.g. ping-pong); we
    // take the latest addPass() call order as tie-breaker.
    std::unordered_map<ResourceHandle, u32> lastWriter;

    // First pass: record all writers.
    for (u32 i = 0; i < static_cast<u32>(passes_.size()); ++i) {
        for (const auto& w : passes_[i].writes) {
            lastWriter[w.handle] = i;
        }
    }

    // Second pass: for every reader, add dependency on the writer.
    for (u32 i = 0; i < static_cast<u32>(passes_.size()); ++i) {
        for (const auto& r : passes_[i].reads) {
            auto it = lastWriter.find(r.handle);
            if (it != lastWriter.end() && it->second != i) {
                passes_[i].dependsOn.push_back(it->second);
            }
        }

        // Also add write-after-write dependencies: if two passes write
        // the same resource, the second depends on the first.
        for (const auto& w : passes_[i].writes) {
            auto it = lastWriter.find(w.handle);
            if (it != lastWriter.end() && it->second != i) {
                // Only add if the writer recorded is *before* us in
                // registration order and is a different pass.
                // (lastWriter currently points to the very last writer,
                //  which could be us.)  We need to check all passes
                //  before us that also write this resource.
            }
        }
    }

    // WAW: linear scan for same-resource writes in registration order.
    for (u32 i = 0; i < static_cast<u32>(passes_.size()); ++i) {
        for (const auto& w : passes_[i].writes) {
            for (u32 j = 0; j < i; ++j) {
                for (const auto& w2 : passes_[j].writes) {
                    if (w2.handle == w.handle) {
                        passes_[i].dependsOn.push_back(j);
                    }
                }
            }
        }
    }

    // Deduplicate dependency edges per pass.
    for (auto& pass : passes_) {
        std::sort(pass.dependsOn.begin(), pass.dependsOn.end());
        pass.dependsOn.erase(
            std::unique(pass.dependsOn.begin(), pass.dependsOn.end()),
            pass.dependsOn.end());
    }
}

// Step 2 -- Kahn's algorithm.
void RenderGraph::topologicalSort()
{
    const u32 n = static_cast<u32>(passes_.size());

    // In-degree for each pass.
    std::vector<u32> inDegree(n, 0);
    // Adjacency list (forward edges): adj[producer] = { consumers... }
    std::vector<std::vector<u32>> adj(n);

    for (u32 i = 0; i < n; ++i) {
        for (u32 dep : passes_[i].dependsOn) {
            adj[dep].push_back(i);
            ++inDegree[i];
        }
    }

    std::queue<u32> ready;
    for (u32 i = 0; i < n; ++i) {
        if (inDegree[i] == 0) {
            ready.push(i);
        }
    }

    sortedOrder_.clear();
    sortedOrder_.reserve(n);

    while (!ready.empty()) {
        u32 cur = ready.front();
        ready.pop();
        passes_[cur].sortOrder = static_cast<u32>(sortedOrder_.size());
        sortedOrder_.push_back(cur);

        for (u32 next : adj[cur]) {
            --inDegree[next];
            if (inDegree[next] == 0) {
                ready.push(next);
            }
        }
    }

    if (sortedOrder_.size() != n) {
        LOG_ERROR("RenderGraph: cycle detected in pass dependencies! "
                  "Sorted %u of %u passes.",
                  static_cast<u32>(sortedOrder_.size()), n);
        assert(false && "cycle in render graph");
    }
}

// Step 3 -- Walk sorted order and figure out which barriers are needed
// between passes that share resources.
void RenderGraph::computeBarriers()
{
    // Live resource state: tracks the most recent write for each resource.
    std::unordered_map<ResourceHandle, ResourceState> resourceStates;

    barriersBefore_.resize(sortedOrder_.size());

    for (u32 sortIdx = 0; sortIdx < static_cast<u32>(sortedOrder_.size()); ++sortIdx) {
        const PassNode& pass = passes_[sortedOrder_[sortIdx]];

        // Barriers for reads -- transition from last write to this read.
        for (const auto& r : pass.reads) {
            auto it = resourceStates.find(r.handle);
            if (it == resourceStates.end()) {
                // First access.  If a layout is requested that differs
                // from UNDEFINED we still need a layout transition.
                if (r.layout != VK_IMAGE_LAYOUT_UNDEFINED &&
                    registry_.isImage(r.handle)) {
                    PendingBarrier pb{};
                    pb.resource  = r.handle;
                    pb.srcStage  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    pb.srcAccess = VK_ACCESS_2_NONE;
                    pb.dstStage  = r.stage;
                    pb.dstAccess = r.access;
                    pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    pb.newLayout = r.layout;
                    barriersBefore_[sortIdx].push_back(pb);
                }
                continue;
            }

            const ResourceState& state = it->second;
            bool needBarrier = true;

            // Skip barrier if stage/access/layout are identical (no hazard).
            if (registry_.isImage(r.handle)) {
                if (state.lastWriteStage  == r.stage  &&
                    state.lastWriteAccess == r.access &&
                    state.currentLayout   == r.layout) {
                    needBarrier = false;
                }
            } else {
                if (state.lastWriteStage  == r.stage &&
                    state.lastWriteAccess == r.access) {
                    needBarrier = false;
                }
            }

            if (needBarrier) {
                PendingBarrier pb{};
                pb.resource  = r.handle;
                pb.srcStage  = state.lastWriteStage;
                pb.srcAccess = state.lastWriteAccess;
                pb.dstStage  = r.stage;
                pb.dstAccess = r.access;
                pb.oldLayout = state.currentLayout;
                pb.newLayout = (r.layout != VK_IMAGE_LAYOUT_UNDEFINED)
                                   ? r.layout
                                   : state.currentLayout;
                barriersBefore_[sortIdx].push_back(pb);
            }
        }

        // Barriers for writes -- WAW hazard from previous write, or
        // layout transition for first write.
        for (const auto& w : pass.writes) {
            auto it = resourceStates.find(w.handle);
            if (it == resourceStates.end()) {
                // First write.  If a layout transition from UNDEFINED is
                // needed, record it.
                if (w.layout != VK_IMAGE_LAYOUT_UNDEFINED &&
                    registry_.isImage(w.handle)) {
                    PendingBarrier pb{};
                    pb.resource  = w.handle;
                    pb.srcStage  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                    pb.srcAccess = VK_ACCESS_2_NONE;
                    pb.dstStage  = w.stage;
                    pb.dstAccess = w.access;
                    pb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    pb.newLayout = w.layout;
                    barriersBefore_[sortIdx].push_back(pb);
                }
            } else {
                const ResourceState& state = it->second;
                PendingBarrier pb{};
                pb.resource  = w.handle;
                pb.srcStage  = state.lastWriteStage;
                pb.srcAccess = state.lastWriteAccess;
                pb.dstStage  = w.stage;
                pb.dstAccess = w.access;
                pb.oldLayout = state.currentLayout;
                pb.newLayout = (w.layout != VK_IMAGE_LAYOUT_UNDEFINED)
                                   ? w.layout
                                   : state.currentLayout;
                barriersBefore_[sortIdx].push_back(pb);
            }

            // Update resource state to reflect this write.
            ResourceState& rs  = resourceStates[w.handle];
            rs.lastWriter      = sortedOrder_[sortIdx];
            rs.lastWriteStage  = w.stage;
            rs.lastWriteAccess = w.access;
            if (w.layout != VK_IMAGE_LAYOUT_UNDEFINED) {
                rs.currentLayout = w.layout;
            }
        }
    }
}

// -----------------------------------------------------------------------
//  execute()
// -----------------------------------------------------------------------

void RenderGraph::execute(VkCommandBuffer cmd)
{
    assert(compiled_ && "must call compile() before execute()");

    BarrierBuilder barrierBuilder;

    for (u32 sortIdx = 0; sortIdx < static_cast<u32>(sortedOrder_.size()); ++sortIdx) {
        const PassNode& pass = passes_[sortedOrder_[sortIdx]];

        // --- Emit barriers accumulated for this pass ---
        for (const auto& pb : barriersBefore_[sortIdx]) {
            if (registry_.isImage(pb.resource)) {
                VkImage image = registry_.getImage(pb.resource);
                const ImageDesc& desc = registry_.getImageDesc(pb.resource);

                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
                if (desc.format == VK_FORMAT_D32_SFLOAT ||
                    desc.format == VK_FORMAT_D16_UNORM) {
                    aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                } else if (desc.format == VK_FORMAT_D24_UNORM_S8_UINT ||
                           desc.format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
                    aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
                }

                VkImageSubresourceRange range{};
                range.aspectMask     = aspect;
                range.baseMipLevel   = 0;
                range.levelCount     = desc.mipLevels;
                range.baseArrayLayer = 0;
                range.layerCount     = desc.layers;

                barrierBuilder.addImageBarrier(
                    image,
                    pb.oldLayout, pb.newLayout,
                    pb.srcStage,  pb.srcAccess,
                    pb.dstStage,  pb.dstAccess,
                    range);
            } else if (registry_.isBuffer(pb.resource)) {
                VkBuffer buffer = registry_.getBuffer(pb.resource);

                barrierBuilder.addBufferBarrier(
                    buffer,
                    pb.srcStage, pb.srcAccess,
                    pb.dstStage, pb.dstAccess);
            }
        }

        barrierBuilder.flush(cmd);

        // --- Debug label + execute ---
        {
            ScopedDebugLabel label(cmd, pass.name.c_str());
            pass.execute(cmd);
        }
    }
}

// -----------------------------------------------------------------------
//  reset()
// -----------------------------------------------------------------------

void RenderGraph::reset()
{
    passes_.clear();
    sortedOrder_.clear();
    barriersBefore_.clear();
    compiled_ = false;
}

} // namespace phosphor
