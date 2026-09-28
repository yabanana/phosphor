#include "renderer/tonemap_pass.h"
#include "renderer/push_constants.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_descriptors.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <array>
#include <fstream>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Per-pass push constants for the tonemap shader (small, bound separately)
// ---------------------------------------------------------------------------

struct TonemapPushConstants {
    float exposure;
    u32   width;
    u32   height;
    u32   pad;
};
static_assert(sizeof(TonemapPushConstants) == 16);

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

TonemapPass::TonemapPass(VulkanDevice& device, GpuAllocator& allocator,
                         PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
                         VkExtent2D extent)
    : device_(device), allocator_(allocator), pipelines_(pipelines), descriptors_(descriptors), extent_(extent) {
    createResources();
    LOG_INFO("TonemapPass created (%ux%u)", extent_.width, extent_.height);
}

TonemapPass::~TonemapPass() {
    cleanupResources();

    VkDevice dev = device_.getDevice();
    if (pipeline_)     vkDestroyPipeline(dev, pipeline_, nullptr);
    if (passPlLayout_) vkDestroyPipelineLayout(dev, passPlLayout_, nullptr);
    if (passPool_)     vkDestroyDescriptorPool(dev, passPool_, nullptr);
    if (passLayout_)   vkDestroyDescriptorSetLayout(dev, passLayout_, nullptr);
}

// ---------------------------------------------------------------------------
// Recreate on resize
// ---------------------------------------------------------------------------

void TonemapPass::recreate(VkExtent2D newExtent) {
    if (newExtent.width == extent_.width && newExtent.height == extent_.height) {
        return;
    }
    device_.waitIdle();
    cleanupResources();
    extent_ = newExtent;
    createResources();
    LOG_INFO("TonemapPass recreated (%ux%u)", extent_.width, extent_.height);
}

// ---------------------------------------------------------------------------
// Internal: create LDR target
// ---------------------------------------------------------------------------

void TonemapPass::createResources() {
    VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imgInfo.imageType   = VK_IMAGE_TYPE_2D;
    imgInfo.format      = VK_FORMAT_R8G8B8A8_UNORM;
    imgInfo.extent      = {extent_.width, extent_.height, 1};
    imgInfo.mipLevels   = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage       = VK_IMAGE_USAGE_STORAGE_BIT
                        | VK_IMAGE_USAGE_SAMPLED_BIT
                        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    ldrImage_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    ldrView_  = allocator_.createImageView(
        ldrImage_.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
}

void TonemapPass::cleanupResources() {
    allocator_.destroyImageView(ldrView_);
    ldrView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(ldrImage_);
}

// ---------------------------------------------------------------------------
// Lazy pipeline creation
// ---------------------------------------------------------------------------

void TonemapPass::ensurePipelineCreated() {
    if (initialized_) {
        return;
    }

    VkDevice dev = device_.getDevice();

    // --- Descriptor set layout: 2 storage-image bindings ---
    //   binding 0 = HDR input   (readonly,  rgba16f)
    //   binding 1 = LDR output  (writeonly, rgba8)
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<u32>(bindings.size());
    layoutInfo.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &passLayout_));

    // --- Descriptor pool ---
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &passPool_));

    // --- Allocate descriptor set ---
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = passPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &passLayout_;
    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &passSet_));

    // --- Pipeline layout: set 0 = bindless global, set 1 = per-pass, 128-byte push constants ---
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushRange.offset     = 0;
    pushRange.size       = 128; // match types.glsl PushConstants

    VkDescriptorSetLayout setLayouts[2] = {descriptors_.getLayout(), passLayout_};
    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount         = 2;
    plInfo.pSetLayouts            = setLayouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges    = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(dev, &plInfo, nullptr, &passPlLayout_));

    // --- Compute pipeline ---
    {
        std::string spvPath = pipelines_.getShaderDir() + "/post/tonemap.comp.spv";

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

        VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &pipeline_));
        vkDestroyShaderModule(dev, mod, nullptr);
    }

    initialized_ = true;
    LOG_INFO("TonemapPass pipeline initialized");
}

// ---------------------------------------------------------------------------
// Update descriptor set with current HDR input + LDR output
// ---------------------------------------------------------------------------

void TonemapPass::updateDescriptorSet(VkImageView hdrInput) {
    VkDescriptorImageInfo hdrInfo{};
    hdrInfo.imageView   = hdrInput;
    hdrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo ldrInfo{};
    ldrInfo.imageView   = ldrView_;
    ldrInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 2> writes{};

    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = passSet_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo      = &hdrInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = passSet_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo      = &ldrInfo;

    vkUpdateDescriptorSets(device_.getDevice(),
                           static_cast<u32>(writes.size()), writes.data(),
                           0, nullptr);
}

// ---------------------------------------------------------------------------
// Apply ACES tonemap
// ---------------------------------------------------------------------------

void TonemapPass::apply(VkCommandBuffer cmd, VkImageView hdrInput, float exposure) {
    ensurePipelineCreated();

    if (pipeline_ == VK_NULL_HANDLE) {
        return; // shader not available
    }

    PHOSPHOR_GPU_LABEL(cmd, "TonemapPass");

    // --- Transition LDR image: UNDEFINED -> GENERAL ---
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image         = ldrImage_.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // --- Update descriptors ---
    if (!descriptorsWritten_) {
        updateDescriptorSet(hdrInput);
        descriptorsWritten_ = true;
    }

    // --- Bind pipeline and descriptors ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    // Set 0 = bindless global, set 1 = per-pass (tonemap storage images)
    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), passSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            passPlLayout_, 0, 2, sets, 0, nullptr);

    // --- Push constants (full 128 bytes to match shader's types.glsl) ---
    PushConstants pc{};
    pc.exposure    = exposure;
    pc.resolution[0] = extent_.width;
    pc.resolution[1] = extent_.height;

    vkCmdPushConstants(cmd, passPlLayout_,
                       VK_SHADER_STAGE_ALL, 0,
                       sizeof(PushConstants), &pc);

    // --- Dispatch: 8x8 workgroups ---
    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

} // namespace phosphor
