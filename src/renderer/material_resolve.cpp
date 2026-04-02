#include "renderer/material_resolve.h"
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
// Construction / Destruction
// ---------------------------------------------------------------------------

MaterialResolve::MaterialResolve(VulkanDevice& device, GpuAllocator& allocator,
                                 PipelineManager& pipelines,
                                 BindlessDescriptorManager& descriptors,
                                 VkExtent2D extent)
    : device_(device), allocator_(allocator), pipelines_(pipelines),
      descriptors_(descriptors), extent_(extent) {
    createResources();
    LOG_INFO("MaterialResolve created (%ux%u)", extent_.width, extent_.height);
}

MaterialResolve::~MaterialResolve() {
    cleanupResources();

    VkDevice dev = device_.getDevice();
    if (resolvePipeline_) vkDestroyPipeline(dev, resolvePipeline_, nullptr);
    if (passPlLayout_)    vkDestroyPipelineLayout(dev, passPlLayout_, nullptr);
    if (passPool_)        vkDestroyDescriptorPool(dev, passPool_, nullptr);
    if (passLayout_)      vkDestroyDescriptorSetLayout(dev, passLayout_, nullptr);
}

// ---------------------------------------------------------------------------
// Recreate on resize
// ---------------------------------------------------------------------------

void MaterialResolve::recreate(VkExtent2D newExtent) {
    if (newExtent.width == extent_.width && newExtent.height == extent_.height) {
        return;
    }
    device_.waitIdle();
    cleanupResources();
    extent_ = newExtent;
    createResources();
    LOG_INFO("MaterialResolve recreated (%ux%u)", extent_.width, extent_.height);
}

// ---------------------------------------------------------------------------
// Internal: create HDR target + per-pass descriptors
// ---------------------------------------------------------------------------

void MaterialResolve::createResources() {
    // --- HDR colour target: RGBA16F ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType   = VK_IMAGE_TYPE_2D;
        imgInfo.format      = VK_FORMAT_R16G16B16A16_SFLOAT;
        imgInfo.extent      = {extent_.width, extent_.height, 1};
        imgInfo.mipLevels   = 1;
        imgInfo.arrayLayers = 1;
        imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage       = VK_IMAGE_USAGE_STORAGE_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        hdrColor_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        hdrView_  = allocator_.createImageView(
            hdrColor_.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void MaterialResolve::cleanupResources() {
    allocator_.destroyImageView(hdrView_);
    hdrView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(hdrColor_);
}

// ---------------------------------------------------------------------------
// Lazy pipeline + descriptor set creation
// ---------------------------------------------------------------------------

void MaterialResolve::ensurePipelineCreated() {
    if (initialized_) {
        return;
    }

    VkDevice dev = device_.getDevice();

    // --- Descriptor set layout: 3 storage-image bindings ---
    //   binding 0 = vis buffer  (readonly,  r32ui)
    //   binding 1 = depth       (readonly,  sampled image)
    //   binding 2 = HDR output  (writeonly, rgba16f)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &passLayout_));

    // --- Descriptor pool ---
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &passPool_));

    // --- Allocate descriptor set ---
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = passPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &passLayout_;
    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &passSet_));

    // --- Pipeline layout: set 0 = bindless, set 1 = per-pass, push constants ---
    VkDescriptorSetLayout setLayouts[2] = {
        descriptors_.getLayout(),
        passLayout_
    };

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount         = 2;
    plInfo.pSetLayouts            = setLayouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges    = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(dev, &plInfo, nullptr, &passPlLayout_));

    // --- Compute pipeline ---
    // We create the compute pipeline manually because PipelineManager
    // uses its own pipeline layout (bindless-only).  We need set 1.
    {
        std::string spvPath = pipelines_.getShaderDir() + "/visibility/material_resolve.comp.spv";

        std::vector<char> code;
        {
            std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open shader: %s", spvPath.c_str());
                initialized_ = true;
                return;
            }
            auto size = file.tellg();
            code.resize(static_cast<size_t>(size));
            file.seekg(0);
            file.read(code.data(), size);
        }

        VkShaderModuleCreateInfo smInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smInfo.codeSize = code.size();
        smInfo.pCode    = reinterpret_cast<const u32*>(code.data());

        VkShaderModule mod = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(dev, &smInfo, nullptr, &mod));

        VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cpInfo.stage.module = mod;
        cpInfo.stage.pName  = "main";
        cpInfo.layout       = passPlLayout_;

        VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &resolvePipeline_));
        vkDestroyShaderModule(dev, mod, nullptr);
    }
    initialized_     = true;
    LOG_INFO("MaterialResolve pipeline initialized");
}

// ---------------------------------------------------------------------------
// Update descriptor set with current visibility / HDR views
// ---------------------------------------------------------------------------

void MaterialResolve::updateDescriptorSet(const VisibilityBuffer& visBuf) {
    VkDescriptorImageInfo visInfo{};
    visInfo.imageView   = visBuf.getVisView();
    visInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo depthInfo{};
    depthInfo.imageView   = visBuf.getDepthView();
    depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthInfo.sampler     = VK_NULL_HANDLE;  // combined image sampler: we create an inline sampler below

    VkDescriptorImageInfo hdrInfo{};
    hdrInfo.imageView   = hdrView_;
    hdrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 3> writes{};

    // binding 0: vis buffer (storage image)
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = passSet_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo      = &visInfo;

    // binding 1: depth (sampled image, shader uses texelFetch — no sampler needed)
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = passSet_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[1].pImageInfo      = &depthInfo;

    // binding 2: HDR output (storage image)
    writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet          = passSet_;
    writes[2].dstBinding      = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo      = &hdrInfo;

    vkUpdateDescriptorSets(device_.getDevice(),
                           static_cast<u32>(writes.size()), writes.data(),
                           0, nullptr);
}

// ---------------------------------------------------------------------------
// Dispatch resolve compute shader
// ---------------------------------------------------------------------------

void MaterialResolve::resolve(VkCommandBuffer cmd, const VisibilityBuffer& visBuf,
                              GpuScene& scene, const Camera& camera,
                              u32 frameIndex, float exposure) {
    ensurePipelineCreated();

    PHOSPHOR_GPU_LABEL(cmd, "MaterialResolve");

    VkExtent2D ext = visBuf.getExtent();

    // --- Transition images for the compute pass ---
    {
        VkImageMemoryBarrier2 barriers[3]{};

        // Vis buffer: COLOR_ATTACHMENT_OPTIMAL -> GENERAL (compute read as storage)
        barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].image         = visBuf.getVisImage();
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        // Depth: DEPTH_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        barriers[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                                  | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barriers[1].oldLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barriers[1].newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[1].image         = visBuf.getDepthImage();
        barriers[1].subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

        // HDR output: UNDEFINED -> GENERAL (compute write)
        barriers[2].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[2].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[2].srcAccessMask = 0;
        barriers[2].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[2].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barriers[2].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[2].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[2].image         = hdrColor_.image;
        barriers[2].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 3;
        dep.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // --- Update per-pass descriptors (only once, on first call or after resize) ---
    if (!descriptorsWritten_) {
        updateDescriptorSet(visBuf);
        descriptorsWritten_ = true;
    }

    // --- Bind pipeline and descriptor sets ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resolvePipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), passSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            passPlLayout_, 0, 2, sets, 0, nullptr);

    // --- Push constants ---
    SceneGlobals globals = scene.getSceneGlobals();

    PushConstants pc{};
    std::memcpy(pc.viewProjection, glm::value_ptr(camera.getViewProjection()),
                sizeof(pc.viewProjection));

    glm::vec3 camPos = camera.getPosition();
    pc.cameraPosition[0] = camPos.x;
    pc.cameraPosition[1] = camPos.y;
    pc.cameraPosition[2] = camPos.z;
    pc.cameraPosition[3] = 0.0f;

    pc.sceneGlobalsAddress  = globals.vertexBufferAddress;
    pc.vertexBufferAddress  = globals.vertexBufferAddress;
    pc.meshletBufferAddress = globals.meshletBufferAddress;
    pc.resolution[0]        = ext.width;
    pc.resolution[1]        = ext.height;
    pc.frameIndex           = frameIndex;
    pc.lightCount           = globals.lightCount;
    pc.exposure             = exposure;
    pc.debugMode            = 0;

    vkCmdPushConstants(cmd, passPlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    // --- Dispatch: 8x8 workgroups ---
    u32 groupsX = (ext.width  + 7) / 8;
    u32 groupsY = (ext.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

} // namespace phosphor
