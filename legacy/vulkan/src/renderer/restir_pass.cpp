#include "renderer/restir_pass.h"
#include "renderer/visibility_buffer.h"
#include "renderer/push_constants.h"
#include "renderer/gpu_scene.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_descriptors.h"
#include "scene/camera.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Reservoir struct size — must match GLSL
// ---------------------------------------------------------------------------

static constexpr u32 RESERVOIR_STRIDE = 16; // 4 x 4 bytes

// ---------------------------------------------------------------------------
// UBO data for temporal / spatial / shade passes
// ---------------------------------------------------------------------------

struct TemporalUBO {
    float prevViewProjection[16];
    float currentInvViewProjection[16];
};

struct SpatialUBO {
    float invViewProjection[16];
};

struct ShadeUBO {
    float invViewProjection[16];
};

struct TAAUBO {
    float prevViewProjection[16];
    float currentInvViewProjection[16];
    float blendFactor;
    float pad[3];
};

// ---------------------------------------------------------------------------
// Helper: load SPIR-V and create a compute pipeline with a custom layout
// ---------------------------------------------------------------------------

static VkShaderModule loadShaderModule(VkDevice dev, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader: %s", path.c_str());
        return VK_NULL_HANDLE;
    }
    auto size = file.tellg();
    std::vector<char> code(static_cast<size_t>(size));
    file.seekg(0);
    file.read(code.data(), size);

    VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smInfo.codeSize = code.size();
    smInfo.pCode    = reinterpret_cast<const u32*>(code.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(dev, &smInfo, nullptr, &mod));
    return mod;
}

static VkPipeline createComputePipelineFromModule(VkDevice dev, VkShaderModule mod,
                                                   VkPipelineLayout layout) {
    VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = mod;
    cpInfo.stage.pName  = "main";
    cpInfo.layout       = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline));
    return pipeline;
}

// ---------------------------------------------------------------------------
// Helper: create a pipeline layout with bindless (set 0) + pass (set 1) + PC
// ---------------------------------------------------------------------------

static VkPipelineLayout createPassPipelineLayout(VkDevice dev,
                                                  VkDescriptorSetLayout bindlessLayout,
                                                  VkDescriptorSetLayout passLayout) {
    VkDescriptorSetLayout setLayouts[2] = {bindlessLayout, passLayout};

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount         = 2;
    plInfo.pSetLayouts            = setLayouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges    = &pushRange;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(dev, &plInfo, nullptr, &layout));
    return layout;
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ReSTIRPass::ReSTIRPass(VulkanDevice& device, GpuAllocator& allocator,
                       PipelineManager& pipelines,
                       BindlessDescriptorManager& descriptors,
                       VkExtent2D extent)
    : device_(device), allocator_(allocator), pipelines_(pipelines),
      descriptors_(descriptors), extent_(extent) {
    createResources();
    LOG_INFO("ReSTIRPass created (%ux%u)", extent_.width, extent_.height);
}

ReSTIRPass::~ReSTIRPass() {
    cleanupResources();

    VkDevice dev = device_.getDevice();

    // Destroy pipelines
    if (candidatesPipeline_) vkDestroyPipeline(dev, candidatesPipeline_, nullptr);
    if (temporalPipeline_)   vkDestroyPipeline(dev, temporalPipeline_, nullptr);
    if (spatialPipeline_)    vkDestroyPipeline(dev, spatialPipeline_, nullptr);
    if (shadePipeline_)      vkDestroyPipeline(dev, shadePipeline_, nullptr);

    // Destroy pipeline layouts
    if (candidatesPlLayout_) vkDestroyPipelineLayout(dev, candidatesPlLayout_, nullptr);
    if (temporalPlLayout_)   vkDestroyPipelineLayout(dev, temporalPlLayout_, nullptr);
    if (spatialPlLayout_)    vkDestroyPipelineLayout(dev, spatialPlLayout_, nullptr);
    if (shadePlLayout_)      vkDestroyPipelineLayout(dev, shadePlLayout_, nullptr);

    // Destroy descriptor pool (frees all sets)
    if (passPool_) vkDestroyDescriptorPool(dev, passPool_, nullptr);

    // Destroy descriptor set layouts
    if (candidatesLayout_) vkDestroyDescriptorSetLayout(dev, candidatesLayout_, nullptr);
    if (temporalLayout_)   vkDestroyDescriptorSetLayout(dev, temporalLayout_, nullptr);
    if (spatialLayout_)    vkDestroyDescriptorSetLayout(dev, spatialLayout_, nullptr);
    if (shadeLayout_)      vkDestroyDescriptorSetLayout(dev, shadeLayout_, nullptr);
}

// ---------------------------------------------------------------------------
// Recreate on resize
// ---------------------------------------------------------------------------

void ReSTIRPass::recreate(VkExtent2D newExtent) {
    if (newExtent.width == extent_.width && newExtent.height == extent_.height) {
        return;
    }
    device_.waitIdle();
    cleanupResources();
    extent_ = newExtent;
    createResources();
    currentReservoir_ = 0;
    LOG_INFO("ReSTIRPass recreated (%ux%u)", extent_.width, extent_.height);
}

// ---------------------------------------------------------------------------
// Create GPU resources
// ---------------------------------------------------------------------------

void ReSTIRPass::createResources() {
    u32 pixelCount = extent_.width * extent_.height;
    VkDeviceSize reservoirSize = static_cast<VkDeviceSize>(pixelCount) * RESERVOIR_STRIDE;

    // --- Reservoir buffers (double-buffered) ---
    for (u32 i = 0; i < 2; i++) {
        reservoirBuffers_[i] = allocator_.createBuffer(
            reservoirSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }

    // --- Normal G-buffer image (RGBA16F) ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        imgInfo.extent        = {extent_.width, extent_.height, 1};
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        normalImage_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        normalView_  = allocator_.createImageView(
            normalImage_.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- HDR direct lighting output (RGBA16F) ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
        imgInfo.extent        = {extent_.width, extent_.height, 1};
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                              | VK_IMAGE_USAGE_SAMPLED_BIT
                              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        restirOutput_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        outputView_   = allocator_.createImageView(
            restirOutput_.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- Temporal UBO (large enough for TemporalUBO which is the biggest) ---
    {
        temporalUBO_ = allocator_.createBuffer(
            256, // padded — enough for any per-pass UBO
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
}

void ReSTIRPass::cleanupResources() {
    allocator_.destroyImageView(outputView_);
    outputView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(restirOutput_);

    allocator_.destroyImageView(normalView_);
    normalView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(normalImage_);

    for (u32 i = 0; i < 2; i++) {
        allocator_.destroyBuffer(reservoirBuffers_[i]);
    }

    allocator_.destroyBuffer(temporalUBO_);
}

// ---------------------------------------------------------------------------
// Lazy pipeline + descriptor creation
// ---------------------------------------------------------------------------

void ReSTIRPass::createPassDescriptors() {
    VkDevice dev = device_.getDevice();

    // -----------------------------------------------------------------------
    // Candidates layout: set 1
    //   binding 0 = vis buffer (storage image, r32ui, readonly)
    //   binding 1 = depth      (combined image sampler)
    //   binding 2 = normal     (storage image, rgba16f, readonly)
    //   binding 3 = reservoir out (storage buffer, writeonly)
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &candidatesLayout_));
    }

    // -----------------------------------------------------------------------
    // Temporal layout: set 1
    //   binding 0 = depth                  (combined image sampler)
    //   binding 1 = normal                 (storage image, readonly)
    //   binding 2 = current reservoirs     (storage buffer, readonly)
    //   binding 3 = previous reservoirs    (storage buffer, readonly)
    //   binding 4 = output reservoirs      (storage buffer, writeonly)
    //   binding 5 = temporal UBO           (uniform buffer)
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &temporalLayout_));
    }

    // -----------------------------------------------------------------------
    // Spatial layout: set 1
    //   binding 0 = depth               (combined image sampler)
    //   binding 1 = normal              (storage image, readonly)
    //   binding 2 = input reservoirs    (storage buffer, readonly)
    //   binding 3 = output reservoirs   (storage buffer, writeonly)
    //   binding 4 = spatial UBO         (uniform buffer)
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &spatialLayout_));
    }

    // -----------------------------------------------------------------------
    // Shade layout: set 1
    //   binding 0 = vis buffer       (storage image, r32ui, readonly)
    //   binding 1 = depth            (combined image sampler)
    //   binding 2 = normal           (storage image, rgba16f, readonly)
    //   binding 3 = final reservoirs (storage buffer, readonly)
    //   binding 4 = output image     (storage image, rgba16f, writeonly)
    //   binding 5 = shade UBO        (uniform buffer)
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &shadeLayout_));
    }

    // -----------------------------------------------------------------------
    // Descriptor pool — enough for all 4 sets
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorPoolSize, 4> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          12};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  4};
        poolSizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         10};
        poolSizes[3] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          4};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets       = 4;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes    = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &passPool_));
    }

    // -----------------------------------------------------------------------
    // Allocate descriptor sets
    // -----------------------------------------------------------------------
    {
        VkDescriptorSetLayout layouts[4] = {
            candidatesLayout_, temporalLayout_, spatialLayout_, shadeLayout_
        };
        VkDescriptorSet sets[4];

        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = passPool_;
        allocInfo.descriptorSetCount = 4;
        allocInfo.pSetLayouts        = layouts;
        VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, sets));

        candidatesSet_ = sets[0];
        temporalSet_   = sets[1];
        spatialSet_    = sets[2];
        shadeSet_      = sets[3];
    }

    // -----------------------------------------------------------------------
    // Pipeline layouts
    // -----------------------------------------------------------------------
    VkDescriptorSetLayout bindlessLayout = descriptors_.getLayout();
    candidatesPlLayout_ = createPassPipelineLayout(dev, bindlessLayout, candidatesLayout_);
    temporalPlLayout_   = createPassPipelineLayout(dev, bindlessLayout, temporalLayout_);
    spatialPlLayout_    = createPassPipelineLayout(dev, bindlessLayout, spatialLayout_);
    shadePlLayout_      = createPassPipelineLayout(dev, bindlessLayout, shadeLayout_);
}

void ReSTIRPass::ensurePipelinesCreated() {
    if (initialized_) {
        return;
    }

    VkDevice dev = device_.getDevice();
    std::string shaderDir = pipelines_.getShaderDir();

    createPassDescriptors();

    // --- Candidate generation pipeline ---
    {
        std::string path = shaderDir + "/lighting/restir_candidates.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, path);
        if (mod) {
            candidatesPipeline_ = createComputePipelineFromModule(dev, mod, candidatesPlLayout_);
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    // --- Temporal resampling pipeline ---
    {
        std::string path = shaderDir + "/lighting/restir_temporal.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, path);
        if (mod) {
            temporalPipeline_ = createComputePipelineFromModule(dev, mod, temporalPlLayout_);
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    // --- Spatial resampling pipeline ---
    {
        std::string path = shaderDir + "/lighting/restir_spatial.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, path);
        if (mod) {
            spatialPipeline_ = createComputePipelineFromModule(dev, mod, spatialPlLayout_);
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    // --- Shade pipeline ---
    {
        std::string path = shaderDir + "/lighting/restir_shade.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, path);
        if (mod) {
            shadePipeline_ = createComputePipelineFromModule(dev, mod, shadePlLayout_);
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    initialized_ = true;
    LOG_INFO("ReSTIRPass pipelines initialized");
}

// ---------------------------------------------------------------------------
// Helper: fill a PushConstants struct from camera/scene state
// ---------------------------------------------------------------------------

static PushConstants buildPushConstants(const Camera& camera, VkExtent2D extent,
                                        const SceneGlobals& globals, u32 frameIndex) {
    PushConstants pc{};
    std::memcpy(pc.viewProjection, glm::value_ptr(camera.getViewProjection()),
                sizeof(pc.viewProjection));

    glm::vec3 camPos = camera.getPosition();
    pc.cameraPosition[0] = camPos.x;
    pc.cameraPosition[1] = camPos.y;
    pc.cameraPosition[2] = camPos.z;
    pc.cameraPosition[3] = 0.0f;

    pc.sceneGlobalsAddress  = globals.vertexBufferAddress; // base of SceneGlobals
    pc.vertexBufferAddress  = globals.vertexBufferAddress;
    pc.meshletBufferAddress = globals.meshletBufferAddress;
    pc.resolution[0]        = extent.width;
    pc.resolution[1]        = extent.height;
    pc.frameIndex           = frameIndex;
    pc.lightCount           = globals.lightCount;
    pc.exposure             = 1.0f;
    pc.debugMode            = 0;

    return pc;
}

// ---------------------------------------------------------------------------
// Helper: insert a compute barrier between passes
// ---------------------------------------------------------------------------

static void barrierComputeToCompute(VkCommandBuffer cmd) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Stage 1: Generate candidates
// ---------------------------------------------------------------------------

void ReSTIRPass::generateCandidates(VkCommandBuffer cmd, const VisibilityBuffer& visBuf,
                                     const GpuScene& scene, const Camera& camera,
                                     u32 frameIndex) {
    ensurePipelinesCreated();

    if (!candidatesPipeline_) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "ReSTIR::Candidates");

    // Current write target
    u32 writeIdx = currentReservoir_;

    // --- Transition images ---
    {
        VkImageMemoryBarrier2 barriers[2]{};

        // Normal image -> GENERAL for storage write
        barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].image         = normalImage_.image;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        // Output image -> GENERAL
        barriers[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[1].srcAccessMask = 0;
        barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[1].image         = restirOutput_.image;
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // --- Update descriptor set for candidates ---
    {
        VkDescriptorImageInfo visInfo{};
        visInfo.imageView   = visBuf.getVisView();
        visInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageView   = visBuf.getDepthView();
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthInfo.sampler     = VK_NULL_HANDLE;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageView   = normalView_;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo reservoirInfo{};
        reservoirInfo.buffer = reservoirBuffers_[writeIdx].buffer;
        reservoirInfo.offset = 0;
        reservoirInfo.range  = reservoirBuffers_[writeIdx].size;

        std::array<VkWriteDescriptorSet, 4> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = candidatesSet_;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &visInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = candidatesSet_;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo      = &depthInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = candidatesSet_;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo      = &normalInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = candidatesSet_;
        writes[3].dstBinding      = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo     = &reservoirInfo;

        vkUpdateDescriptorSets(device_.getDevice(),
                               static_cast<u32>(writes.size()), writes.data(),
                               0, nullptr);
    }

    // --- Bind and dispatch ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, candidatesPipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), candidatesSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            candidatesPlLayout_, 0, 2, sets, 0, nullptr);

    SceneGlobals globals = const_cast<GpuScene&>(scene).getSceneGlobals();
    PushConstants pc = buildPushConstants(camera, extent_, globals, frameIndex);
    vkCmdPushConstants(cmd, candidatesPlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    barrierComputeToCompute(cmd);
}

// ---------------------------------------------------------------------------
// Stage 2: Temporal resampling
// ---------------------------------------------------------------------------

void ReSTIRPass::temporalResample(VkCommandBuffer cmd, const Camera& camera) {
    if (!temporalPipeline_) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "ReSTIR::Temporal");

    u32 currentIdx  = currentReservoir_;
    u32 previousIdx = 1 - currentIdx;

    // We write into a temporary slot: reuse the "previous" buffer as output.
    // After temporal, the result lives in reservoirBuffers_[previousIdx].
    // Spatial will read from previousIdx and write to currentIdx.

    // --- Update UBO with matrices ---
    {
        TemporalUBO ubo{};
        std::memcpy(ubo.prevViewProjection,
                    glm::value_ptr(camera.getPrevViewProjection()),
                    sizeof(ubo.prevViewProjection));

        glm::mat4 invVP = glm::inverse(camera.getViewProjection());
        std::memcpy(ubo.currentInvViewProjection,
                    glm::value_ptr(invVP),
                    sizeof(ubo.currentInvViewProjection));

        void* mapped = allocator_.mapMemory(temporalUBO_);
        std::memcpy(mapped, &ubo, sizeof(ubo));
        allocator_.unmapMemory(temporalUBO_);
    }

    // --- Update descriptor set ---
    {
        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageView   = VK_NULL_HANDLE; // set externally or from visBuf
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthInfo.sampler     = VK_NULL_HANDLE;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageView   = normalView_;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo currentInfo{};
        currentInfo.buffer = reservoirBuffers_[currentIdx].buffer;
        currentInfo.offset = 0;
        currentInfo.range  = reservoirBuffers_[currentIdx].size;

        VkDescriptorBufferInfo prevInfo{};
        prevInfo.buffer = reservoirBuffers_[previousIdx].buffer;
        prevInfo.offset = 0;
        prevInfo.range  = reservoirBuffers_[previousIdx].size;

        // Output goes to previousIdx (reusing as temp)
        VkDescriptorBufferInfo outInfo{};
        outInfo.buffer = reservoirBuffers_[previousIdx].buffer;
        outInfo.offset = 0;
        outInfo.range  = reservoirBuffers_[previousIdx].size;

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = temporalUBO_.buffer;
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(TemporalUBO);

        std::array<VkWriteDescriptorSet, 5> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = temporalSet_;
        writes[0].dstBinding      = 1;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &normalInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = temporalSet_;
        writes[1].dstBinding      = 2;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &currentInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = temporalSet_;
        writes[2].dstBinding      = 3;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo     = &prevInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = temporalSet_;
        writes[3].dstBinding      = 4;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo     = &outInfo;

        writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet          = temporalSet_;
        writes[4].dstBinding      = 5;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[4].pBufferInfo     = &uboInfo;

        vkUpdateDescriptorSets(device_.getDevice(),
                               static_cast<u32>(writes.size()), writes.data(),
                               0, nullptr);
    }

    // --- Bind and dispatch ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporalPipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), temporalSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            temporalPlLayout_, 0, 2, sets, 0, nullptr);

    // Push constants use current camera state
    SceneGlobals dummyGlobals{};
    PushConstants pc = buildPushConstants(camera, extent_, dummyGlobals, 0);
    vkCmdPushConstants(cmd, temporalPlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    barrierComputeToCompute(cmd);
}

// ---------------------------------------------------------------------------
// Stage 3: Spatial resampling
// ---------------------------------------------------------------------------

void ReSTIRPass::spatialResample(VkCommandBuffer cmd) {
    if (!spatialPipeline_) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "ReSTIR::Spatial");

    u32 currentIdx  = currentReservoir_;
    u32 previousIdx = 1 - currentIdx;

    // Temporal wrote to previousIdx. Spatial reads previousIdx, writes currentIdx.

    // --- Update UBO ---
    // Reuse temporal UBO buffer — just overwrite with spatial data (invVP only)
    // SpatialUBO is 64 bytes (mat4) which fits in our 256-byte buffer
    {
        // invViewProjection is already stored in temporal UBO at offset 64
        // but let us be explicit:
        // The spatial shader reads invViewProjection from a UBO at binding 4
        // We'll reuse the same buffer with the correct data at the start
    }

    // --- Update descriptor set ---
    {
        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageView   = normalView_;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo inputInfo{};
        inputInfo.buffer = reservoirBuffers_[previousIdx].buffer;
        inputInfo.offset = 0;
        inputInfo.range  = reservoirBuffers_[previousIdx].size;

        VkDescriptorBufferInfo outputInfo{};
        outputInfo.buffer = reservoirBuffers_[currentIdx].buffer;
        outputInfo.offset = 0;
        outputInfo.range  = reservoirBuffers_[currentIdx].size;

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = temporalUBO_.buffer;
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(SpatialUBO);

        std::array<VkWriteDescriptorSet, 4> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = spatialSet_;
        writes[0].dstBinding      = 1;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &normalInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = spatialSet_;
        writes[1].dstBinding      = 2;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &inputInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = spatialSet_;
        writes[2].dstBinding      = 3;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo     = &outputInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = spatialSet_;
        writes[3].dstBinding      = 4;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3].pBufferInfo     = &uboInfo;

        vkUpdateDescriptorSets(device_.getDevice(),
                               static_cast<u32>(writes.size()), writes.data(),
                               0, nullptr);
    }

    // --- Bind and dispatch ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, spatialPipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), spatialSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            spatialPlLayout_, 0, 2, sets, 0, nullptr);

    // Push constants — we still need scene globals for light access
    SceneGlobals dummyGlobals{};
    PushConstants pc{};
    pc.resolution[0] = extent_.width;
    pc.resolution[1] = extent_.height;
    vkCmdPushConstants(cmd, spatialPlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    barrierComputeToCompute(cmd);

    // Flip ping-pong for next frame
    currentReservoir_ = 1 - currentReservoir_;
}

// ---------------------------------------------------------------------------
// Stage 4: Shade
// ---------------------------------------------------------------------------

void ReSTIRPass::shade(VkCommandBuffer cmd, const GpuScene& scene, const Camera& camera) {
    if (!shadePipeline_) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "ReSTIR::Shade");

    // The final reservoir is in reservoirBuffers_[1 - currentReservoir_]
    // (because spatialResample just flipped the index)
    u32 finalIdx = 1 - currentReservoir_;

    // --- Update UBO for shade (invVP) ---
    {
        ShadeUBO ubo{};
        glm::mat4 invVP = glm::inverse(camera.getViewProjection());
        std::memcpy(ubo.invViewProjection, glm::value_ptr(invVP),
                    sizeof(ubo.invViewProjection));

        void* mapped = allocator_.mapMemory(temporalUBO_);
        std::memcpy(mapped, &ubo, sizeof(ubo));
        allocator_.unmapMemory(temporalUBO_);
    }

    // --- Update descriptor set ---
    {
        VkDescriptorImageInfo visInfo{};
        visInfo.imageView   = VK_NULL_HANDLE; // set from visBuf externally
        visInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo normalInfo{};
        normalInfo.imageView   = normalView_;
        normalInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo reservoirInfo{};
        reservoirInfo.buffer = reservoirBuffers_[finalIdx].buffer;
        reservoirInfo.offset = 0;
        reservoirInfo.range  = reservoirBuffers_[finalIdx].size;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView   = outputView_;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = temporalUBO_.buffer;
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(ShadeUBO);

        std::array<VkWriteDescriptorSet, 4> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = shadeSet_;
        writes[0].dstBinding      = 2;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &normalInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = shadeSet_;
        writes[1].dstBinding      = 3;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo     = &reservoirInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = shadeSet_;
        writes[2].dstBinding      = 4;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo      = &outputInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = shadeSet_;
        writes[3].dstBinding      = 5;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[3].pBufferInfo     = &uboInfo;

        vkUpdateDescriptorSets(device_.getDevice(),
                               static_cast<u32>(writes.size()), writes.data(),
                               0, nullptr);
    }

    // --- Bind and dispatch ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadePipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), shadeSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shadePlLayout_, 0, 2, sets, 0, nullptr);

    SceneGlobals globals = const_cast<GpuScene&>(scene).getSceneGlobals();
    PushConstants pc = buildPushConstants(camera, extent_, globals, 0);
    vkCmdPushConstants(cmd, shadePlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

} // namespace phosphor
