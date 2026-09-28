#include "renderer/aa_pass.h"
#include "renderer/push_constants.h"
#include "rhi/vk_device.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_descriptors.h"
#include "diagnostics/debug_utils.h"
#include "core/log.h"

#include <array>
#include <cstring>
#include <fstream>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// TAA UBO layout — must match taa.comp.glsl TAAData
// ---------------------------------------------------------------------------

struct TAAUBOData {
    float prevViewProjection[16];
    float currentInvViewProjection[16];
    float blendFactor;
    float pad[3];
};

// ---------------------------------------------------------------------------
// Helper: load SPIR-V shader module
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

static VkPipeline createComputePipeline(VkDevice dev, VkShaderModule mod,
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

AAPass::AAPass(VulkanDevice& device, GpuAllocator& allocator,
               PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
               VkExtent2D extent)
    : device_(device), allocator_(allocator), pipelines_(pipelines),
      descriptors_(descriptors), extent_(extent) {
    createResources();
    LOG_INFO("AAPass created (%ux%u)", extent_.width, extent_.height);
}

AAPass::~AAPass() {
    cleanupResources();

    VkDevice dev = device_.getDevice();

    if (fxaaPipeline_) vkDestroyPipeline(dev, fxaaPipeline_, nullptr);
    if (taaPipeline_)  vkDestroyPipeline(dev, taaPipeline_, nullptr);

    if (fxaaPlLayout_) vkDestroyPipelineLayout(dev, fxaaPlLayout_, nullptr);
    if (taaPlLayout_)  vkDestroyPipelineLayout(dev, taaPlLayout_, nullptr);

    if (passPool_)   vkDestroyDescriptorPool(dev, passPool_, nullptr);
    if (fxaaLayout_) vkDestroyDescriptorSetLayout(dev, fxaaLayout_, nullptr);
    if (taaLayout_)  vkDestroyDescriptorSetLayout(dev, taaLayout_, nullptr);
}

// ---------------------------------------------------------------------------
// Recreate on resize
// ---------------------------------------------------------------------------

void AAPass::recreate(VkExtent2D newExtent) {
    if (newExtent.width == extent_.width && newExtent.height == extent_.height) {
        return;
    }
    device_.waitIdle();
    cleanupResources();
    extent_ = newExtent;
    createResources();
    LOG_INFO("AAPass recreated (%ux%u)", extent_.width, extent_.height);
}

// ---------------------------------------------------------------------------
// Create GPU resources
// ---------------------------------------------------------------------------

void AAPass::createResources() {
    // --- Output image (RGBA16F — works for both FXAA and TAA) ---
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

        outputImage_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        outputView_  = allocator_.createImageView(
            outputImage_.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- TAA history buffer (RGBA16F) ---
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
                              | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        historyBuffer_ = allocator_.createImage(imgInfo, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        historyView_   = allocator_.createImageView(
            historyBuffer_.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // --- TAA UBO ---
    {
        taaUBO_ = allocator_.createBuffer(
            256,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
}

void AAPass::cleanupResources() {
    allocator_.destroyImageView(outputView_);
    outputView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(outputImage_);

    allocator_.destroyImageView(historyView_);
    historyView_ = VK_NULL_HANDLE;
    allocator_.destroyImage(historyBuffer_);

    allocator_.destroyBuffer(taaUBO_);
}

// ---------------------------------------------------------------------------
// Descriptor and pipeline creation
// ---------------------------------------------------------------------------

void AAPass::createPassDescriptors() {
    VkDevice dev = device_.getDevice();

    // -----------------------------------------------------------------------
    // FXAA layout: set 1
    //   binding 0 = input image  (storage image, rgba8, readonly)
    //   binding 1 = output image (storage image, rgba8, writeonly)
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &fxaaLayout_));
    }

    // -----------------------------------------------------------------------
    // TAA layout: set 1
    //   binding 0 = current color    (storage image, rgba16f, readonly)
    //   binding 1 = depth buffer     (combined image sampler)
    //   binding 2 = history buffer   (combined image sampler — bilinear)
    //   binding 3 = output image     (storage image, rgba16f, writeonly)
    //   binding 4 = history output   (storage image, rgba16f, writeonly)
    //   binding 5 = TAA UBO          (uniform buffer)
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = static_cast<u32>(bindings.size());
        layoutInfo.pBindings    = bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &taaLayout_));
    }

    // -----------------------------------------------------------------------
    // Descriptor pool
    // -----------------------------------------------------------------------
    {
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          6};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2};
        poolSizes[2] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets       = 2;
        poolInfo.poolSizeCount = static_cast<u32>(poolSizes.size());
        poolInfo.pPoolSizes    = poolSizes.data();
        VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &passPool_));
    }

    // -----------------------------------------------------------------------
    // Allocate descriptor sets
    // -----------------------------------------------------------------------
    {
        VkDescriptorSetLayout layouts[2] = {fxaaLayout_, taaLayout_};
        VkDescriptorSet sets[2];

        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = passPool_;
        allocInfo.descriptorSetCount = 2;
        allocInfo.pSetLayouts        = layouts;
        VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, sets));

        fxaaSet_ = sets[0];
        taaSet_  = sets[1];
    }

    // -----------------------------------------------------------------------
    // Pipeline layouts
    // -----------------------------------------------------------------------
    VkDescriptorSetLayout bindlessLayout = descriptors_.getLayout();
    fxaaPlLayout_ = createPassPipelineLayout(dev, bindlessLayout, fxaaLayout_);
    taaPlLayout_  = createPassPipelineLayout(dev, bindlessLayout, taaLayout_);
}

void AAPass::ensurePipelinesCreated() {
    if (initialized_) {
        return;
    }

    VkDevice dev = device_.getDevice();
    std::string shaderDir = pipelines_.getShaderDir();

    createPassDescriptors();

    // --- FXAA pipeline ---
    {
        std::string path = shaderDir + "/post/fxaa.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, path);
        if (mod) {
            fxaaPipeline_ = createComputePipeline(dev, mod, fxaaPlLayout_);
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    // --- TAA pipeline ---
    {
        std::string path = shaderDir + "/post/taa.comp.spv";
        VkShaderModule mod = loadShaderModule(dev, path);
        if (mod) {
            taaPipeline_ = createComputePipeline(dev, mod, taaPlLayout_);
            vkDestroyShaderModule(dev, mod, nullptr);
        }
    }

    initialized_ = true;
    LOG_INFO("AAPass pipelines initialized");
}

// ---------------------------------------------------------------------------
// Set TAA matrices (called each frame before apply())
// ---------------------------------------------------------------------------

void AAPass::setTAAMatrices(const float* prevVP, const float* currentInvVP) {
    TAAUBOData ubo{};
    std::memcpy(ubo.prevViewProjection, prevVP, 64);
    std::memcpy(ubo.currentInvViewProjection, currentInvVP, 64);
    ubo.blendFactor = 0.05f;
    ubo.pad[0] = 0.0f;
    ubo.pad[1] = 0.0f;
    ubo.pad[2] = 0.0f;

    void* mapped = allocator_.mapMemory(taaUBO_);
    std::memcpy(mapped, &ubo, sizeof(ubo));
    allocator_.unmapMemory(taaUBO_);
}

// ---------------------------------------------------------------------------
// Apply AA
// ---------------------------------------------------------------------------

void AAPass::apply(VkCommandBuffer cmd, VkImageView input, VkImageView depthView,
                   AAMode mode) {
    if (mode == AAMode::None) {
        return;
    }

    ensurePipelinesCreated();

    // --- Transition output image to GENERAL ---
    {
        VkImageMemoryBarrier2 barriers[2]{};
        u32 barrierCount = 1;

        barriers[0].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barriers[0].oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].image         = outputImage_.image;
        barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        if (mode == AAMode::TAA) {
            // History image also needs to be in GENERAL for both read and write
            barriers[1].sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barriers[1].srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[1].dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT
                                      | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[1].oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
            barriers[1].newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            barriers[1].image         = historyBuffer_.image;
            barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrierCount = 2;
        }

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = barrierCount;
        dep.pImageMemoryBarriers    = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    if (mode == AAMode::FXAA) {
        applyFXAA(cmd, input);
    } else if (mode == AAMode::TAA) {
        applyTAA(cmd, input, depthView);
    }
}

// ---------------------------------------------------------------------------
// Private: FXAA dispatch
// ---------------------------------------------------------------------------

void AAPass::applyFXAA(VkCommandBuffer cmd, VkImageView input) {
    if (!fxaaPipeline_) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "FXAA");

    // --- Update descriptors ---
    {
        VkDescriptorImageInfo inputInfo{};
        inputInfo.imageView   = input;
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView   = outputView_;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = fxaaSet_;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &inputInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = fxaaSet_;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo      = &outputInfo;

        vkUpdateDescriptorSets(device_.getDevice(),
                               static_cast<u32>(writes.size()), writes.data(),
                               0, nullptr);
    }

    // --- Bind and dispatch ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fxaaPipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), fxaaSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            fxaaPlLayout_, 0, 2, sets, 0, nullptr);

    PushConstants pc{};
    pc.resolution[0] = extent_.width;
    pc.resolution[1] = extent_.height;
    vkCmdPushConstants(cmd, fxaaPlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

// ---------------------------------------------------------------------------
// Private: TAA dispatch
// ---------------------------------------------------------------------------

void AAPass::applyTAA(VkCommandBuffer cmd, VkImageView input, VkImageView depthView) {
    if (!taaPipeline_) {
        return;
    }

    PHOSPHOR_GPU_LABEL(cmd, "TAA");

    // --- Update descriptors ---
    {
        VkDescriptorImageInfo currentInfo{};
        currentInfo.imageView   = input;
        currentInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo depthInfo{};
        depthInfo.imageView   = depthView;
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthInfo.sampler     = VK_NULL_HANDLE;

        VkDescriptorImageInfo historyInfo{};
        historyInfo.imageView   = historyView_;
        historyInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        historyInfo.sampler     = VK_NULL_HANDLE;

        VkDescriptorImageInfo outputInfo{};
        outputInfo.imageView   = outputView_;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo historyOutInfo{};
        historyOutInfo.imageView   = historyView_;
        historyOutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = taaUBO_.buffer;
        uboInfo.offset = 0;
        uboInfo.range  = sizeof(TAAUBOData);

        std::array<VkWriteDescriptorSet, 6> writes{};

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = taaSet_;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo      = &currentInfo;

        writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet          = taaSet_;
        writes[1].dstBinding      = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo      = &depthInfo;

        writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet          = taaSet_;
        writes[2].dstBinding      = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo      = &historyInfo;

        writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet          = taaSet_;
        writes[3].dstBinding      = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[3].pImageInfo      = &outputInfo;

        writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet          = taaSet_;
        writes[4].dstBinding      = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[4].pImageInfo      = &historyOutInfo;

        writes[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[5].dstSet          = taaSet_;
        writes[5].dstBinding      = 5;
        writes[5].descriptorCount = 1;
        writes[5].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[5].pBufferInfo     = &uboInfo;

        vkUpdateDescriptorSets(device_.getDevice(),
                               static_cast<u32>(writes.size()), writes.data(),
                               0, nullptr);
    }

    // --- Bind and dispatch ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, taaPipeline_);

    VkDescriptorSet sets[2] = {descriptors_.getDescriptorSet(), taaSet_};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            taaPlLayout_, 0, 2, sets, 0, nullptr);

    PushConstants pc{};
    pc.resolution[0] = extent_.width;
    pc.resolution[1] = extent_.height;
    vkCmdPushConstants(cmd, taaPlLayout_,
                       VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pc);

    u32 groupsX = (extent_.width  + 7) / 8;
    u32 groupsY = (extent_.height + 7) / 8;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

} // namespace phosphor
