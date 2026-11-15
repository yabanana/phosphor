#pragma once
#include "core/launch_options.h"
#include "platform/metal/shadow_passes.h"
#include "renderer/gpu_types.h"
#include "rendergraph/render_graph.h"
#include <memory>
namespace phosphor {
class DirectLightingPasses {
public:
    DirectLightingPasses(MetalContext&,PipelineCache&,SceneRenderer&,MeshRenderer&,VisibilityRenderer&,AccelerationStructures*,
                         ShadowPasses&,const LaunchOptions&);
    ~DirectLightingPasses();
    void loadScene(const GpuScene&,const SceneStore&);
    void prepareFrame(const GpuScene&,const SceneStore&,std::span<const GPULight>,const ShadowPasses::Frame&);
    void addToGraph(rg::RenderGraph&,rg::TextureRef visibility,rg::TextureRef depth);
    // Diagnostic only, enabled by PHOSPHOR_DIAGNOSTIC_MOTION_READBACK=1.
    // Stage0 after compute,1 after indexed MRT,2 immediately before F13 input.
    void addMotionReadback(rg::RenderGraph&,u32 stage);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] rg::TextureRef direct() const;
    [[nodiscard]] rg::TextureRef motion() const;
    [[nodiscard]] rg::BufferRef surfaceRef() const;
    [[nodiscard]] rg::BufferRef lightsRef() const;
    [[nodiscard]] rg::BufferRef emittersRef() const;
    [[nodiscard]] MTL::Buffer* lightsBuffer() const;
    [[nodiscard]] MTL::Buffer* emittersBuffer() const;
    [[nodiscard]] u32 lightCount() const;
    [[nodiscard]] bool emptyLocalLighting() const;
    [[nodiscard]] u32 lightRevision() const;
    [[nodiscard]] u64 version() const;
    [[nodiscard]] bool check(u32 slot) const;
    /// F15.1 (ticket 06): the per-pixel lobe guide (invalid ref when the scene union is 0: nothing is allocated).  The
    /// ref is the final version of the frame (after the guide kernel, the indexed receivers and the pack pass).
    [[nodiscard]] rg::BufferRef lobeGuideRef() const;
    /// True when this frame's guide was written by the producers specialised for the scene's feature union (not by a
    /// base-pipeline fallback while they compile): mirrored by GPUDIParams::flags DI_LOBE_GUIDE_VALID.
    [[nodiscard]] bool lobeGuideValid() const;
    /// F15.1 (ticket 17, RT tier): before prepareFrame, whether the volumetric-subsurface walk is effective (DI then leaves the diffuse of the weight share of
    /// volumetric pixels, for the delta lights, to the walk: GPUDIParams::flags DI_SSS_WALK).
    void setSubsurfaceWalk(bool active);
    /// --debug-lobe-guide: reads back the slot's guide and DI surfaces (frame complete) and compares them with the CPU
    /// reconstruction; prints the LOBE-GUIDE line; false = FAIL.
    [[nodiscard]] bool checkLobeGuide(u32 slot, const SceneStore& store) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
