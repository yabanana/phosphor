#pragma once

#include "core/launch_options.h"
#include "diagnostics/bench_report.h"
#include "renderer/gpu_types.h"
#include "renderer/rt_proxy.h"
#include "rendergraph/render_graph.h"

#include <memory>
#include <span>

namespace MTL { class Buffer; }

namespace phosphor {
class MetalContext;
class PipelineCache;
class SceneRenderer;
class SceneStore;
class GpuScene;
class MetalTextureManager;
class MetalGraphExecutor;

// F9 graphics-queue RT resources. GpuMemory owns allocation/residency;
// RtScene owns publication/retirement, and the graph owns GPU dependencies.
// prepareFrame runs AFTER SceneRenderer::prepareFrame and after the frame slot
// has completed. Consume that slot's readback BEFORE prepareFrame overwrites it.
class AccelerationStructures {
public:
    struct FrameParams {
        u32 slot = 0;
        u64 index = 0;
        u32 width = 0, height = 0, allocationWidth = 0, allocationHeight = 0;
        FrameConstants constants{};
        bool check = false;
    };
    struct Readback {
        bool valid = false;
        u64 frame = 0;
        u32 width = 0, height = 0, probeType = RT_PROBE_PRIMARY;
        std::span<const GPURtRay> rays;
        std::span<const GPURtHit> hits;
        std::span<const GPUInstance> instances;
        std::span<const GPUMaterial> materials;
        std::span<const GPUVertex> vertices; // same-frame geometry, even across refit
        u64 geometryRevision = 0;
    };

    AccelerationStructures(MetalContext&, PipelineCache&, SceneRenderer&, const LaunchOptions&);
    ~AccelerationStructures();
    AccelerationStructures(const AccelerationStructures&) = delete;
    AccelerationStructures& operator=(const AccelerationStructures&) = delete;

    // Geometry and scene renderer uploads must have completed first. Blocks at
    // loading time only; initial BLAS build uses independent scratch ranges.
    void loadScene(const GpuScene&, const SceneStore&, MetalTextureManager&);
    void clear(); // Caller is between frames; waits idle before dropping scene.
    void prepareFrame(const SceneStore&, const FrameParams&);
    void addPassesToGraph(rg::RenderGraph&);
    rg::TextureRef addDebugPresent(rg::RenderGraph&, rg::TextureRef target, rg::Format format);
    void bindResources(MetalGraphExecutor&);

    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool active() const;
    [[nodiscard]] bool hasTrace() const;
    [[nodiscard]] bool maintenancePending() const;
    [[nodiscard]] rg::AccelerationStructureRef tlasRef() const;
    [[nodiscard]] rg::BufferRef geometryBufferRef() const;
    [[nodiscard]] Readback readback(u32 slot) const;
    [[nodiscard]] GPURtCounters counters(u32 slot) const;
    [[nodiscard]] RtReport report() const;
    [[nodiscard]] MTL::Buffer* rayBuffer(u32 slot) const;
    [[nodiscard]] MTL::Buffer* hitBuffer(u32 slot) const;
    [[nodiscard]] MTL::Buffer* primaryRayBuffer(u32 slot) const;
    [[nodiscard]] MTL::Buffer* primaryHitBuffer(u32 slot) const;
    [[nodiscard]] u32 rayCount(u32 slot) const;
    [[nodiscard]] rg::BufferRef rayRef() const;
    [[nodiscard]] rg::BufferRef hitRef() const;
    [[nodiscard]] rg::BufferRef primaryRayRef() const;
    [[nodiscard]] rg::BufferRef primaryHitRef() const;
    [[nodiscard]] std::span<const GPURtMesh> meshes() const;
    [[nodiscard]] const RtProxyGeometry& geometry() const;
    [[nodiscard]] std::span<const GPUVertex> vertices() const;
    [[nodiscard]] u64 geometryRevision() const;

    // Existing GPU producers may request refit after updating vertices. Their
    // producer must be represented by the graph; CPU tests use updateVertices.
    void requestRefit(u32 mesh, u64 vertexRevision);
    void requestRebuild(u32 mesh);
    // Replaces exactly this mesh's original vertex range and stages a graph
    // upload before BLAS maintenance. Caller owns raster/meshlet bound updates;
    // F9 diagnostics use RT view with meshlet culling disabled. Proxy meshes
    // cannot deform without a new error measurement: non-Full is rejected.
    void updateVertices(u32 mesh, std::span<const GPUVertex> vertices, u64 revision, bool forceRebuild = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
