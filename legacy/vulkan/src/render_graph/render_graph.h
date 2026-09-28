#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "render_graph/render_pass.h"
#include "render_graph/barrier_builder.h"
#include "render_graph/resource_registry.h"

#include <string>
#include <vector>
#include <functional>

namespace phosphor {

/// A Granite-style frame graph that records render/compute/transfer passes,
/// builds a dependency DAG from declared resource accesses, topologically sorts
/// the passes, and automatically inserts pipeline barriers between them.
///
/// Typical per-frame usage:
///   graph.reset();
///   graph.addPass("Shadow", PassType::Graphics, setup, exec);
///   graph.addPass("GBuffer", PassType::Graphics, setup, exec);
///   graph.addPass("Lighting", PassType::Compute, setup, exec);
///   graph.addPass("Tonemap", PassType::Graphics, setup, exec);
///   graph.compile();
///   graph.execute(cmd);
class RenderGraph {
public:
    explicit RenderGraph(ResourceRegistry& registry);
    ~RenderGraph() = default;

    RenderGraph(const RenderGraph&)            = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;
    RenderGraph(RenderGraph&&)                 = delete;
    RenderGraph& operator=(RenderGraph&&)      = delete;

    // ------------------------------------------------------------------
    //  PassBuilder -- handed to the setup lambda of addPass()
    // ------------------------------------------------------------------
    class PassBuilder {
    public:
        /// Declare that this pass reads a resource.
        void read(ResourceHandle       handle,
                  VkPipelineStageFlags2 stage,
                  VkAccessFlags2        access,
                  VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED);

        /// Declare that this pass writes a resource.
        void write(ResourceHandle       handle,
                   VkPipelineStageFlags2 stage,
                   VkAccessFlags2        access,
                   VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED);

        /// Convenience: create a transient image via the registry and return
        /// its handle so the caller can pass it to read()/write().
        ResourceHandle createTransientImage(const ImageDesc& desc);

        /// Convenience: create a transient buffer via the registry.
        ResourceHandle createTransientBuffer(const BufferDesc& desc);

        /// Mark this pass as having a side effect (e.g. present, blit to
        /// swapchain) so that dead-code elimination does not cull it.
        void setSideEffect();

    private:
        friend class RenderGraph;
        explicit PassBuilder(RenderGraph& graph, u32 passIndex);

        RenderGraph& graph_;
        u32          passIndex_;
    };

    /// Register a new pass.
    ///   @param name    Debug name (shown in GPU labels and logs).
    ///   @param type    The kind of work the pass performs.
    ///   @param setup   Lambda that declares resource reads/writes through
    ///                  the provided PassBuilder.
    ///   @param execute Lambda called at execution time with the command
    ///                  buffer to record into.
    void addPass(const std::string&                     name,
                 PassType                                type,
                 std::function<void(PassBuilder&)>       setup,
                 std::function<void(VkCommandBuffer)>    execute);

    /// Build the dependency DAG and topologically sort passes.
    /// Must be called after all addPass() calls and before execute().
    void compile();

    /// Walk the sorted pass list, emit barriers, record debug labels,
    /// and invoke each pass's execute lambda.
    void execute(VkCommandBuffer cmd);

    /// Clear all state so the graph can be rebuilt for the next frame.
    void reset();

private:
    struct PassNode {
        std::string                          name;
        PassType                             type          = PassType::Graphics;
        std::vector<ResourceAccess>          reads;
        std::vector<ResourceAccess>          writes;
        std::function<void(VkCommandBuffer)> execute;
        bool                                 hasSideEffects = false;
        u32                                  sortOrder      = 0;
        std::vector<u32>                     dependsOn; // indices into passes_
    };

    /// Per-resource tracking used during compile().
    struct ResourceState {
        u32                   lastWriter     = ~0u;         // pass index
        VkPipelineStageFlags2 lastWriteStage = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2        lastWriteAccess = VK_ACCESS_2_NONE;
        VkImageLayout         currentLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    /// Information for a barrier to be emitted between two passes.
    struct PendingBarrier {
        ResourceHandle        resource;
        VkPipelineStageFlags2 srcStage;
        VkAccessFlags2        srcAccess;
        VkPipelineStageFlags2 dstStage;
        VkAccessFlags2        dstAccess;
        VkImageLayout         oldLayout;
        VkImageLayout         newLayout;
    };

    void buildEdges();
    void topologicalSort();
    void computeBarriers();

    ResourceRegistry&              registry_;
    std::vector<PassNode>          passes_;
    std::vector<u32>               sortedOrder_;  // indices into passes_

    // Barriers grouped by the consuming pass index (after sort).
    // barriersBefore_[i] are emitted before sortedOrder_[i] executes.
    std::vector<std::vector<PendingBarrier>> barriersBefore_;

    bool compiled_ = false;
};

} // namespace phosphor
