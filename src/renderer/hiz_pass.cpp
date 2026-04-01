#include "renderer/hiz_pass.h"
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
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// Hi-Z build pass push constants (matches hiz_build.comp.glsl)
// ---------------------------------------------------------------------------

struct HiZBuildPC {
    u32 dstWidth;
    u32 dstHeight;
    u32 srcWidth;
    u32 srcHeight;
};
static_assert(sizeof(HiZBuildPC) == 16);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static u32 computeMipCount(u32 width, u32 height) {
    u32 maxDim = std::max(width, height);
    return static_cast<u32>(std::floor(std::log2(static_cast<f32>(maxDim)))) + 1;
}

static u32 nextPowerOfTwo(u32 v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

static VkShaderModule loadShaderModule(VkDevice dev, const std::string& path) {
    std::vector<char> code;
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open shader: %s", path.c_str());
            return VK_NULL_HANDLE;
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
    return mod;
}

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

HiZPass::HiZPass(VulkanDevice& device, GpuAllocator& allocator,
                  PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
                  VkExtent2D extent)
    : device_(device), allocator_(allocator), pipelines_(pipelines),
      descriptors_(descriptors), extent_(extent) {
    createHiZResources();
    LOG_INFO("HiZPass created (%ux%u, %u mip levels)", hizExtent_.width, hizExtent_.height, mipCount_);
}

HiZPass::~HiZPass() {
    device_.waitIdle();
    destroyHiZResources();

    VkDevice dev = device_.getDevice();

    // Hi-Z build pipeline resources
    if (hizBuildPipeline_)  vkDestroyPipeline(dev, hizBuildPipeline_, nullptr);
    if (hizBuildPlLayout_)  vkDestroyPipelineLayout(dev, hizBuildPlLayout_, nullptr);
    if (hizBuildPool_)      vkDestroyDescriptorPool(dev, hizBuildPool_, nullptr);
    if (hizBuildDsLayout_)  vkDestroyDescriptorSetLayout(dev, hizBuildDsLayout_, nullptr);
    if (nearestSampler_)    vkDestroySampler(dev, nearestSampler_, nullptr);

    // Culling pipeline resources
    if (cullPhase1Pipeline_) vkDestroyPipeline(dev, cullPhase1Pipeline_, nullptr);
    if (cullPhase2Pipeline_) vkDestroyPipeline(dev, cullPhase2Pipeline_, nullptr);
    if (cullPlLayout_)       vkDestroyPipelineLayout(dev, cullPlLayout_, nullptr);
    if (cullPool_)           vkDestroyDescriptorPool(dev, cullPool_, nullptr);
    if (cullDsLayout_)       vkDestroyDescriptorSetLayout(dev, cullDsLayout_, nullptr);
}

// ---------------------------------------------------------------------------
// Recreate on resize
// ---------------------------------------------------------------------------

void HiZPass::recreate(VkExtent2D newExtent) {
    if (newExtent.width == extent_.width && newExtent.height == extent_.height) {
        return;
    }
    device_.waitIdle();
    destroyHiZResources();
    extent_ = newExtent;
    createHiZResources();

    // Mark pipelines as needing descriptor re-creation
    // (the descriptor sets reference the old views)
    initialized_ = false;

    LOG_INFO("HiZPass recreated (%ux%u, %u mip levels)", hizExtent_.width, hizExtent_.height, mipCount_);
}

// ---------------------------------------------------------------------------
// Create a single-mip image view
// ---------------------------------------------------------------------------

VkImageView HiZPass::createSingleMipView(VkImage image, VkFormat format,
                                           VkImageAspectFlags aspect, u32 mipLevel) {
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image    = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = format;
    viewInfo.subresourceRange.aspectMask     = aspect;
    viewInfo.subresourceRange.baseMipLevel   = mipLevel;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(device_.getDevice(), &viewInfo, nullptr, &view));
    return view;
}

// ---------------------------------------------------------------------------
// Create Hi-Z resources
// ---------------------------------------------------------------------------

void HiZPass::createHiZResources() {
    // Compute power-of-two Hi-Z dimensions
    hizExtent_.width  = nextPowerOfTwo(extent_.width);
    hizExtent_.height = nextPowerOfTwo(extent_.height);

    // Ensure minimum 1x1
    hizExtent_.width  = std::max(hizExtent_.width, 1u);
    hizExtent_.height = std::max(hizExtent_.height, 1u);

    mipCount_ = computeMipCount(hizExtent_.width, hizExtent_.height);

    // --- Hi-Z pyramid image: R32_SFLOAT with full mip chain ---
    {
        VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imgInfo.imageType   = VK_IMAGE_TYPE_2D;
        imgInfo.format      = VK_FORMAT_R32_SFLOAT;
        imgInfo.extent      = {hizExtent_.width, hizExtent_.height, 1};
        imgInfo.mipLevels   = mipCount_;
        imgInfo.arrayLayers = 1;
        imgInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling      = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage       = VK_IMAGE_USAGE_STORAGE_BIT
                            | VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        hizImage_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    }

    // Full-mip view for sampling during culling
    hizFullView_ = allocator_.createImageView(
        hizImage_.image, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, mipCount_);

    // Per-mip views for storage writes during pyramid construction
    mipViews_.resize(mipCount_);
    for (u32 mip = 0; mip < mipCount_; ++mip) {
        mipViews_[mip] = createSingleMipView(
            hizImage_.image, VK_FORMAT_R32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, mip);
    }

    // --- Culling buffers ---
    // Maximum instance count: we allocate for a generous upper bound.
    // The buffer stores [u32 count, u32 indices[...]].
    static constexpr u32 MAX_INSTANCES = 65536;
    const VkDeviceSize cullBufSize = (1 + MAX_INSTANCES) * sizeof(u32);

    visibleBuffer_ = allocator_.createBuffer(
        cullBufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    culledBuffer_ = allocator_.createBuffer(
        cullBufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

    // Small host-visible buffer with a zero u32 for resetting atomic counters
    counterResetBuffer_ = allocator_.createBuffer(
        sizeof(u32),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

    {
        void* mapped = allocator_.mapMemory(counterResetBuffer_);
        std::memset(mapped, 0, sizeof(u32));
        allocator_.unmapMemory(counterResetBuffer_);
    }
}

// ---------------------------------------------------------------------------
// Destroy Hi-Z resources
// ---------------------------------------------------------------------------

void HiZPass::destroyHiZResources() {
    VkDevice dev = device_.getDevice();

    for (auto view : mipViews_) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, view, nullptr);
        }
    }
    mipViews_.clear();

    allocator_.destroyImageView(hizFullView_);
    hizFullView_ = VK_NULL_HANDLE;

    allocator_.destroyImage(hizImage_);

    allocator_.destroyBuffer(visibleBuffer_);
    allocator_.destroyBuffer(culledBuffer_);
    allocator_.destroyBuffer(counterResetBuffer_);
}

// ---------------------------------------------------------------------------
// Lazy pipeline and descriptor setup
// ---------------------------------------------------------------------------

void HiZPass::ensurePipelinesCreated() {
    if (initialized_) {
        return;
    }

    VkDevice dev = device_.getDevice();
    std::string shaderDir = pipelines_.getShaderDir();

    // -----------------------------------------------------------------------
    // Nearest sampler for Hi-Z build (no filtering — we use texelFetch anyway,
    // but the descriptor type requires a sampler for combined image sampler)
    // -----------------------------------------------------------------------
    if (nearestSampler_ == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter    = VK_FILTER_NEAREST;
        samplerInfo.minFilter    = VK_FILTER_NEAREST;
        samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.maxLod       = static_cast<float>(mipCount_);
        VK_CHECK(vkCreateSampler(dev, &samplerInfo, nullptr, &nearestSampler_));
    }

    // ===================================================================
    // Hi-Z BUILD pipeline
    // ===================================================================
    if (hizBuildPipeline_ == VK_NULL_HANDLE) {
        // --- Descriptor set layout ---
        //   binding 0 = source mip (combined image sampler)
        //   binding 1 = dest mip (storage image, r32f)
        if (hizBuildDsLayout_ == VK_NULL_HANDLE) {
            std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layoutInfo.bindingCount = static_cast<u32>(bindings.size());
            layoutInfo.pBindings    = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &hizBuildDsLayout_));
        }

        // --- Descriptor pool (one set per mip level transition) ---
        if (hizBuildPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(dev, hizBuildPool_, nullptr);
        }

        u32 maxSets = std::max(mipCount_, 1u);
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxSets};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, maxSets};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets       = maxSets;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes    = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &hizBuildPool_));

        // --- Allocate descriptor sets (one per mip transition: mip0->1, mip1->2, ...) ---
        hizBuildSets_.resize(mipCount_ > 1 ? mipCount_ - 1 : 0);
        for (u32 i = 0; i < hizBuildSets_.size(); ++i) {
            VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocInfo.descriptorPool     = hizBuildPool_;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts        = &hizBuildDsLayout_;
            VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &hizBuildSets_[i]));

            // Update: source = mipViews_[i], dest = mipViews_[i+1]
            VkDescriptorImageInfo srcInfo{};
            srcInfo.sampler     = nearestSampler_;
            srcInfo.imageView   = mipViews_[i];
            srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo dstInfo{};
            dstInfo.imageView   = mipViews_[i + 1];
            dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            std::array<VkWriteDescriptorSet, 2> writes{};
            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = hizBuildSets_[i];
            writes[0].dstBinding      = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo      = &srcInfo;

            writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet          = hizBuildSets_[i];
            writes[1].dstBinding      = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo      = &dstInfo;

            vkUpdateDescriptorSets(dev, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
        }

        // --- Pipeline layout ---
        if (hizBuildPlLayout_ == VK_NULL_HANDLE) {
            VkPushConstantRange pushRange{};
            pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushRange.offset     = 0;
            pushRange.size       = sizeof(HiZBuildPC);

            VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plInfo.setLayoutCount         = 1;
            plInfo.pSetLayouts            = &hizBuildDsLayout_;
            plInfo.pushConstantRangeCount = 1;
            plInfo.pPushConstantRanges    = &pushRange;
            VK_CHECK(vkCreatePipelineLayout(dev, &plInfo, nullptr, &hizBuildPlLayout_));
        }

        // --- Compute pipeline ---
        {
            std::string spvPath = shaderDir + "/culling/hiz_build.comp.spv";
            VkShaderModule mod = loadShaderModule(dev, spvPath);
            if (mod == VK_NULL_HANDLE) {
                LOG_ERROR("HiZPass: failed to load hiz_build.comp.spv");
                initialized_ = true;
                return;
            }

            VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
            cpInfo.stage.module = mod;
            cpInfo.stage.pName  = "main";
            cpInfo.layout       = hizBuildPlLayout_;

            VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &hizBuildPipeline_));
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    // ===================================================================
    // CULLING pipelines (phase 1 + phase 2 via specialization constant)
    // ===================================================================
    if (cullPhase1Pipeline_ == VK_NULL_HANDLE) {
        // --- Descriptor set layout ---
        //   binding 0 = hiz pyramid (combined image sampler)
        //   binding 1 = visible output buffer (storage buffer)
        //   binding 2 = culled input buffer (storage buffer) — phase 2 only
        //   binding 3 = culled output buffer (storage buffer) — phase 1 only
        if (cullDsLayout_ == VK_NULL_HANDLE) {
            std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
            bindings[0].binding         = 0;
            bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[1].binding         = 1;
            bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[2].binding         = 2;
            bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            bindings[3].binding         = 3;
            bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[3].descriptorCount = 1;
            bindings[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layoutInfo.bindingCount = static_cast<u32>(bindings.size());
            layoutInfo.pBindings    = bindings.data();
            VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &cullDsLayout_));
        }

        // --- Descriptor pool ---
        if (cullPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(dev, cullPool_, nullptr);
        }
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets       = 1;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes    = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &cullPool_));

        // --- Allocate and update descriptor set ---
        {
            VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocInfo.descriptorPool     = cullPool_;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts        = &cullDsLayout_;
            VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &cullSet_));
        }

        // Update descriptors
        {
            VkDescriptorImageInfo hizInfo{};
            hizInfo.sampler     = nearestSampler_;
            hizInfo.imageView   = hizFullView_;
            hizInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorBufferInfo visibleInfo{};
            visibleInfo.buffer = visibleBuffer_.buffer;
            visibleInfo.offset = 0;
            visibleInfo.range  = visibleBuffer_.size;

            VkDescriptorBufferInfo culledInInfo{};
            culledInInfo.buffer = culledBuffer_.buffer;
            culledInInfo.offset = 0;
            culledInInfo.range  = culledBuffer_.size;

            VkDescriptorBufferInfo culledOutInfo{};
            culledOutInfo.buffer = culledBuffer_.buffer;
            culledOutInfo.offset = 0;
            culledOutInfo.range  = culledBuffer_.size;

            std::array<VkWriteDescriptorSet, 4> writes{};

            writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet          = cullSet_;
            writes[0].dstBinding      = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo      = &hizInfo;

            writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet          = cullSet_;
            writes[1].dstBinding      = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo     = &visibleInfo;

            writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[2].dstSet          = cullSet_;
            writes[2].dstBinding      = 2;
            writes[2].descriptorCount = 1;
            writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].pBufferInfo     = &culledInInfo;

            writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet          = cullSet_;
            writes[3].dstBinding      = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[3].pBufferInfo     = &culledOutInfo;

            vkUpdateDescriptorSets(dev, static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
        }

        // --- Pipeline layout: set 0 = bindless, set 1 = cull-pass, push constants ---
        if (cullPlLayout_ == VK_NULL_HANDLE) {
            VkDescriptorSetLayout setLayouts[2] = {
                descriptors_.getLayout(),
                cullDsLayout_
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
            VK_CHECK(vkCreatePipelineLayout(dev, &plInfo, nullptr, &cullPlLayout_));
        }

        // --- Load shader module ---
        std::string spvPath = shaderDir + "/culling/hiz_cull.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, spvPath);
        if (mod == VK_NULL_HANDLE) {
            LOG_ERROR("HiZPass: failed to load hiz_cull.comp.spv");
            initialized_ = true;
            return;
        }

        // --- Phase 1 pipeline (specialization constant CULL_PHASE = 0) ---
        {
            u32 phase = 0;
            VkSpecializationMapEntry specEntry{};
            specEntry.constantID = 0;
            specEntry.offset     = 0;
            specEntry.size       = sizeof(u32);

            VkSpecializationInfo specInfo{};
            specInfo.mapEntryCount = 1;
            specInfo.pMapEntries   = &specEntry;
            specInfo.dataSize      = sizeof(u32);
            specInfo.pData         = &phase;

            VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpInfo.stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpInfo.stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
            cpInfo.stage.module              = mod;
            cpInfo.stage.pName               = "main";
            cpInfo.stage.pSpecializationInfo = &specInfo;
            cpInfo.layout                    = cullPlLayout_;

            VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &cullPhase1Pipeline_));
        }

        // --- Phase 2 pipeline (specialization constant CULL_PHASE = 1) ---
        {
            u32 phase = 1;
            VkSpecializationMapEntry specEntry{};
            specEntry.constantID = 0;
            specEntry.offset     = 0;
            specEntry.size       = sizeof(u32);

            VkSpecializationInfo specInfo{};
            specInfo.mapEntryCount = 1;
            specInfo.pMapEntries   = &specEntry;
            specInfo.dataSize      = sizeof(u32);
            specInfo.pData         = &phase;

            VkComputePipelineCreateInfo cpInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpInfo.stage.sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpInfo.stage.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
            cpInfo.stage.module              = mod;
            cpInfo.stage.pName               = "main";
            cpInfo.stage.pSpecializationInfo = &specInfo;
            cpInfo.layout                    = cullPlLayout_;

            VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &cullPhase2Pipeline_));
        }

        vkDestroyShaderModule(dev, mod, nullptr);
    }

    initialized_ = true;
    LOG_INFO("HiZPass pipelines initialized");
}

// ---------------------------------------------------------------------------
// Reset atomic counters in culling buffers
// ---------------------------------------------------------------------------

void HiZPass::resetCounters(VkCommandBuffer cmd) {
    // Copy a zero u32 to the first 4 bytes of each culling buffer
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size      = sizeof(u32);

    vkCmdCopyBuffer(cmd, counterResetBuffer_.buffer, visibleBuffer_.buffer, 1, &region);
    vkCmdCopyBuffer(cmd, counterResetBuffer_.buffer, culledBuffer_.buffer, 1, &region);

    // Barrier to make the writes visible to compute shaders
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Build Hi-Z pyramid
// ---------------------------------------------------------------------------

void HiZPass::buildPyramid(VkCommandBuffer cmd, VkImageView depthView) {
    ensurePipelinesCreated();

    if (hizBuildPipeline_ == VK_NULL_HANDLE || mipCount_ < 2) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "HiZ Build");

    // --- Copy depth buffer into mip 0 of the Hi-Z pyramid ---
    // Transition Hi-Z mip 0 to TRANSFER_DST
    // The depth view is already in SHADER_READ_ONLY_OPTIMAL from MaterialResolve.
    // We can't directly copy a D32_SFLOAT to an R32_SFLOAT with vkCmdCopyImage
    // (format incompatibility), so we transition mip 0 to GENERAL and use a
    // shader-based copy. For simplicity, we treat mip 0 as already populated
    // by treating the depth view as the source for the first mip build step.
    //
    // Strategy: we update the descriptor set for the first transition to use
    // the actual depth view as the source (combined image sampler) and
    // mipViews_[0] as the destination.  Then subsequent steps use mipViews_[i]
    // as source and mipViews_[i+1] as destination.

    // Transition entire Hi-Z image to GENERAL for the build pass
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barrier.image         = hizImage_.image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipCount_, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // We need an additional descriptor set for the depth -> mip0 step.
    // To avoid managing yet another set, we rewrite hizBuildSets_[0] temporarily
    // to point at the depth view as source, mip0 as dest.
    // Then restore it for subsequent frames.

    // Step 0: depth -> mip0
    // Update set 0 to use depthView as source
    {
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = nearestSampler_;
        srcInfo.imageView   = depthView;
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = mipViews_[0];
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = hizBuildSets_.empty() ? VK_NULL_HANDLE : hizBuildSets_[0];
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo      = &srcInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = hizBuildSets_.empty() ? VK_NULL_HANDLE : hizBuildSets_[0];
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &dstInfo;

        if (!hizBuildSets_.empty()) {
            vkUpdateDescriptorSets(device_.getDevice(), static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizBuildPipeline_);

    // Dispatch: depth -> mip 0 (copy depth values into the R32F pyramid)
    // The depth image has extent_ dimensions, mip 0 has hizExtent_ dimensions.
    // We dispatch for hizExtent_ and the shader reads from the depth view.
    {
        HiZBuildPC pc{};
        pc.dstWidth  = hizExtent_.width;
        pc.dstHeight = hizExtent_.height;
        pc.srcWidth  = extent_.width;
        pc.srcHeight = extent_.height;

        vkCmdPushConstants(cmd, hizBuildPlLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(HiZBuildPC), &pc);

        if (!hizBuildSets_.empty()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    hizBuildPlLayout_, 0, 1, &hizBuildSets_[0], 0, nullptr);
        }

        u32 groupsX = (hizExtent_.width  + 15) / 16;
        u32 groupsY = (hizExtent_.height + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    // Restore set 0 to use mipViews_[0] as source (for future frames / subsequent steps)
    if (!hizBuildSets_.empty()) {
        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = nearestSampler_;
        srcInfo.imageView   = mipViews_[0];
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet          = hizBuildSets_[0];
        write.dstBinding      = 0;
        write.descriptorCount = 1;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo      = &srcInfo;

        vkUpdateDescriptorSets(device_.getDevice(), 1, &write, 0, nullptr);
    }

    // Subsequent mip levels: mip[i] -> mip[i+1]
    for (u32 mip = 0; mip + 1 < mipCount_; ++mip) {
        // Barrier: previous mip write must complete before we read it
        {
            VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            barrier.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.image         = hizImage_.image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};

            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        }

        u32 dstW = std::max(hizExtent_.width  >> (mip + 1), 1u);
        u32 dstH = std::max(hizExtent_.height >> (mip + 1), 1u);
        u32 srcW = std::max(hizExtent_.width  >> mip, 1u);
        u32 srcH = std::max(hizExtent_.height >> mip, 1u);

        HiZBuildPC pc{};
        pc.dstWidth  = dstW;
        pc.dstHeight = dstH;
        pc.srcWidth  = srcW;
        pc.srcHeight = srcH;

        vkCmdPushConstants(cmd, hizBuildPlLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(HiZBuildPC), &pc);

        if (mip < hizBuildSets_.size()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    hizBuildPlLayout_, 0, 1, &hizBuildSets_[mip], 0, nullptr);
        }

        u32 groupsX = (dstW + 15) / 16;
        u32 groupsY = (dstH + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    // Final transition: entire Hi-Z to SHADER_READ_ONLY_OPTIMAL for culling sampling
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image         = hizImage_.image;
        // Transition the last mip (all others already in READ_ONLY from the loop above)
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mipCount_ - 1, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

// ---------------------------------------------------------------------------
// Cull phase helper (shared between phase 1 and phase 2)
// ---------------------------------------------------------------------------

static void fillPushConstants(PushConstants& pc, GpuScene& scene, const Camera& camera,
                               VkExtent2D extent, u32 frameIndex, float exposure) {
    SceneGlobals globals = scene.getSceneGlobals();

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
    pc.resolution[0]        = extent.width;
    pc.resolution[1]        = extent.height;
    pc.frameIndex           = frameIndex;
    pc.lightCount           = globals.lightCount;
    pc.exposure             = exposure;
    pc.debugMode            = 0;
}

// ---------------------------------------------------------------------------
// Phase 1 culling
// ---------------------------------------------------------------------------

void HiZPass::cullPhase1(VkCommandBuffer cmd, GpuScene& scene, const Camera& camera,
                          u32 frameIndex, float exposure) {
    ensurePipelinesCreated();

    if (cullPhase1Pipeline_ == VK_NULL_HANDLE) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "HiZ Cull Phase 1");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPhase1Pipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), cullSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            cullPlLayout_, 0, 2, sets, 0, nullptr);

    PushConstants pc{};
    fillPushConstants(pc, scene, camera, extent_, frameIndex, exposure);
    vkCmdPushConstants(cmd, cullPlLayout_, VK_SHADER_STAGE_ALL,
                       0, sizeof(PushConstants), &pc);

    SceneGlobals globals = scene.getSceneGlobals();
    u32 instanceCount = globals.instanceCount;
    u32 groupCount = (instanceCount + 63) / 64;
    if (groupCount > 0) {
        vkCmdDispatch(cmd, groupCount, 1, 1);
    }

    // Barrier: culling output must be visible before mesh shaders read it
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                          | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
                          | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Phase 2 culling
// ---------------------------------------------------------------------------

void HiZPass::cullPhase2(VkCommandBuffer cmd, GpuScene& scene, const Camera& camera,
                          u32 frameIndex, float exposure) {
    ensurePipelinesCreated();

    if (cullPhase2Pipeline_ == VK_NULL_HANDLE) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "HiZ Cull Phase 2");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPhase2Pipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), cullSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            cullPlLayout_, 0, 2, sets, 0, nullptr);

    PushConstants pc{};
    fillPushConstants(pc, scene, camera, extent_, frameIndex, exposure);
    vkCmdPushConstants(cmd, cullPlLayout_, VK_SHADER_STAGE_ALL,
                       0, sizeof(PushConstants), &pc);

    // Phase 2 dispatches over the culled count from phase 1.
    // We use a conservative upper bound since we can't read back the exact
    // count without a readback (the shader bounds-checks internally).
    SceneGlobals globals = scene.getSceneGlobals();
    u32 instanceCount = globals.instanceCount;
    u32 groupCount = (instanceCount + 63) / 64;
    if (groupCount > 0) {
        vkCmdDispatch(cmd, groupCount, 1, 1);
    }

    // Barrier: phase 2 output must be visible before mesh shaders read it
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT
                          | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace phosphor
