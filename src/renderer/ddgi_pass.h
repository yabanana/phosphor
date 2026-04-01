#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"

#include <glm/glm.hpp>

namespace phosphor {

class VulkanDevice;
class GpuAllocator;
class PipelineManager;
class BindlessDescriptorManager;
class AccelerationStructureManager;

// ---------------------------------------------------------------------------
// DDGIConfig — tuneable parameters for the DDGI probe grid
// ---------------------------------------------------------------------------

struct DDGIConfig {
    glm::ivec3 probeGridDims{8, 4, 8};
    float      probeSpacing        = 2.0f;
    glm::vec3  probeGridOrigin{-8.0f, 0.0f, -8.0f};
    float      maxRayDistance      = 50.0f;
    u32        raysPerProbe        = 256;
    float      hysteresis          = 0.97f;
    float      irradianceGamma     = 5.0f;
};

// ---------------------------------------------------------------------------
// GPU-side uniform struct — must match the GLSL DDGIUniforms layout exactly
// ---------------------------------------------------------------------------

struct DDGIUniformsGPU {
    glm::ivec3 probeGridDims;    // 12 bytes
    float      probeSpacing;     //  4 bytes
    glm::vec3  probeGridOrigin;  // 12 bytes
    float      maxRayDistance;   //  4 bytes
    u32        raysPerProbe;     //  4 bytes
    float      hysteresis;       //  4 bytes
    float      irradianceGamma;  //  4 bytes
    float      pad;              //  4 bytes
};                               // total: 48 bytes

// ---------------------------------------------------------------------------
// DDGIPass — manages the full DDGI pipeline:
//   1. Trace probe rays (RT pipeline)
//   2. Update irradiance atlas (compute)
//   3. Update visibility atlas (compute)
//
// The irradiance and visibility atlas textures are registered in the
// bindless descriptor table for consumption by the lighting pass.
// ---------------------------------------------------------------------------

class DDGIPass {
public:
    DDGIPass(VulkanDevice& device, GpuAllocator& allocator,
             PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
             AccelerationStructureManager& rtManager, DDGIConfig config);
    ~DDGIPass();

    DDGIPass(const DDGIPass&)            = delete;
    DDGIPass& operator=(const DDGIPass&) = delete;
    DDGIPass(DDGIPass&&)                 = delete;
    DDGIPass& operator=(DDGIPass&&)      = delete;

    /// Record the ray-trace dispatch for all probes into @p cmd.
    void traceProbeRays(VkCommandBuffer cmd);

    /// Record the irradiance atlas update compute dispatch into @p cmd.
    void updateIrradiance(VkCommandBuffer cmd);

    /// Record the visibility atlas update compute dispatch into @p cmd.
    void updateVisibility(VkCommandBuffer cmd);

    /// Update the scene globals BDA pointer used by the RT shaders.
    void setSceneGlobalsAddress(VkDeviceAddress addr);

    VkImageView        getIrradianceView() const;
    VkImageView        getVisibilityView() const;
    u32                getIrradianceBindlessIdx() const;
    u32                getVisibilityBindlessIdx() const;
    const DDGIConfig&  getConfig() const;

private:
    DDGIConfig config_;
    u32        totalProbes_ = 0;

    // --- Irradiance atlas: R11G11B10F, 8x8 per probe (+ 1-texel border) ---
    AllocatedImage irradianceAtlas_{};
    VkImageView    irradianceView_ = VK_NULL_HANDLE;
    u32            irradianceBindlessIdx_ = 0;

    // --- Visibility atlas: RG16F, 16x16 per probe (+ 1-texel border) ------
    AllocatedImage visibilityAtlas_{};
    VkImageView    visibilityView_ = VK_NULL_HANDLE;
    u32            visibilityBindlessIdx_ = 0;

    // --- Ray data buffer: vec4(radiance, hitDist) per ray per probe --------
    AllocatedBuffer rayDataBuffer_{};

    // --- DDGI uniforms GPU buffer ------------------------------------------
    AllocatedBuffer uniformsBuffer_{};

    // --- Scene globals BDA (written by renderer each frame) ----------------
    AllocatedBuffer sceneGlobalsRefBuffer_{};
    VkDeviceAddress sceneGlobalsAddress_ = 0;

    // --- RT pipeline and Shader Binding Table ------------------------------
    VkPipeline      rtPipeline_ = VK_NULL_HANDLE;
    AllocatedBuffer sbtBuffer_{};
    VkStridedDeviceAddressRegionKHR rgenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callRegion_{};

    // --- Compute pipelines -------------------------------------------------
    VkPipeline irradianceUpdatePipeline_ = VK_NULL_HANDLE;
    VkPipeline visibilityUpdatePipeline_ = VK_NULL_HANDLE;

    // --- Per-pass descriptor set (set = 1) ---------------------------------
    VkDescriptorSetLayout passLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool      passPool_   = VK_NULL_HANDLE;
    VkDescriptorSet       passSet_    = VK_NULL_HANDLE;

    // --- References --------------------------------------------------------
    VulkanDevice&                 device_;
    GpuAllocator&                 allocator_;
    PipelineManager&              pipelines_;
    BindlessDescriptorManager&    descriptors_;
    AccelerationStructureManager& rtManager_;

    // --- RT extension function pointer -------------------------------------
    PFN_vkCmdTraceRaysKHR vkCmdTraceRays_ = nullptr;

    // --- Internal helpers --------------------------------------------------
    void createAtlases();
    void createBuffers();
    void createDescriptors();
    void createPipelines();
    void createSBT();
    void uploadUniforms();
};

} // namespace phosphor
