#pragma once
#include "platform/metal/shadow_passes.h"
#include "renderer/gi_lighting_epoch.h"
#include <array>
#include <memory>
#include <glm/glm.hpp>
namespace phosphor {
class DirectLightingPasses;
class GiPasses;
// SOURCE WRITING ONLY; no acceptance/measurement is implied by this adapter.
// Physical composition returns RGBA32Float before exposure/upscaling/UI.
class AtmospherePasses {
  public:
    struct ReadResources {
        MTL::Texture *transmittance=nullptr,*multiscattering=nullptr,*sky=nullptr;
        rg::TextureRef transmittanceRef{},multiscatteringRef{},skyRef{},output{};
        GPUAtmosphereParams atmosphere{};
        GPUFogParams fog{};
        GPUCloudParams clouds{};
    }; // borrowed, read-only: caller declares graph readback and completion
    AtmospherePasses(MetalContext&,PipelineCache&,SceneRenderer&,DirectLightingPasses*,AccelerationStructures*,
                     GiPasses*,ShadowPasses*,const LaunchOptions&);
    ~AtmospherePasses();
    AtmospherePasses(const AtmospherePasses&)=delete;
    AtmospherePasses& operator=(const AtmospherePasses&)=delete;
    // Exactly once BEFORE scene/light/RT/DI/GI preparation. Returns sun then
    // moon at a fixed ground reference; camera does not alter environment epoch.
    std::array<GPULight,2> prepareLighting(double clockSeconds,glm::dvec3 cameraPosition);
    // AFTER all source producers prepared their current frame; epochs compare
    // full tuples (no XOR/hash cancellation), not buffer-ring slot identity.
    void prepareFrame(const ShadowPasses::Frame&,u64 geometryEpoch,u64 materialEpoch);
    rg::TextureRef addToGraph(rg::RenderGraph&,rg::TextureRef linearHdr,rg::TextureRef depth);
    void bindFrame(MetalGraphExecutor&);
    [[nodiscard]] u64 version()const;
    [[nodiscard]] const GiEnvironment& environment()const;
    [[nodiscard]] u64 clockEpoch()const;
    [[nodiscard]] bool clockReset()const;
    [[nodiscard]] float exposureEv100()const; // hint only; physical shaders never expose
    [[nodiscard]] ReadResources readResources()const;
    // Reads only a slot whose frameEvent reached index+1. Not-recorded startup
    // slots have no check work; in-flight reads are rejected explicitly.
    [[nodiscard]] bool check(u32 slot)const;
    bool consumeDiagnostics(u32 completedSlot);
  private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace phosphor
