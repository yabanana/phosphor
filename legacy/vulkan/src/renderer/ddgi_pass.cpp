#include "renderer/ddgi_pass.h"
#include "rhi/vk_device.h"
#include "rhi/vk_allocator.h"
#include "rhi/vk_pipeline.h"
#include "rhi/vk_descriptors.h"
#include "rhi/vk_rt.h"
#include "core/log.h"

#include <cstring>
#include <algorithm>

namespace phosphor {

// ---------------------------------------------------------------------------
// Atlas layout constants (must match the GLSL shaders)
// ---------------------------------------------------------------------------

static constexpr u32 IRRADIANCE_PROBE_SIZE   = 8;
static constexpr u32 IRRADIANCE_WITH_BORDER  = IRRADIANCE_PROBE_SIZE + 2;
static constexpr u32 VISIBILITY_PROBE_SIZE   = 16;
static constexpr u32 VISIBILITY_WITH_BORDER  = VISIBILITY_PROBE_SIZE + 2;

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DDGIPass::DDGIPass(VulkanDevice& device, GpuAllocator& allocator,
                   PipelineManager& pipelines, BindlessDescriptorManager& descriptors,
                   AccelerationStructureManager& rtManager, DDGIConfig config)
    : config_(config)
    , device_(device)
    , allocator_(allocator)
    , pipelines_(pipelines)
    , descriptors_(descriptors)
    , rtManager_(rtManager)
{
    totalProbes_ = static_cast<u32>(config_.probeGridDims.x) *
                   static_cast<u32>(config_.probeGridDims.y) *
                   static_cast<u32>(config_.probeGridDims.z);

    // Load vkCmdTraceRaysKHR
    vkCmdTraceRays_ = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
        vkGetDeviceProcAddr(device_.getDevice(), "vkCmdTraceRaysKHR"));
    if (!vkCmdTraceRays_) {
        LOG_ERROR("Failed to load vkCmdTraceRaysKHR — VK_KHR_ray_tracing_pipeline not available");
        std::abort();
    }

    createAtlases();
    createBuffers();
    uploadUniforms();
    createDescriptors();
    createPipelines();
    createSBT();

    LOG_INFO("DDGIPass initialised: %dx%dx%d grid (%u probes), %u rays/probe",
             config_.probeGridDims.x, config_.probeGridDims.y, config_.probeGridDims.z,
             totalProbes_, config_.raysPerProbe);
}

// ---------------------------------------------------------------------------
// Destruction
// ---------------------------------------------------------------------------

DDGIPass::~DDGIPass()
{
    VkDevice dev = device_.getDevice();

    // Unregister bindless entries
    descriptors_.unregisterTexture(irradianceBindlessIdx_);
    descriptors_.unregisterTexture(visibilityBindlessIdx_);

    // Destroy image views
    if (irradianceView_ != VK_NULL_HANDLE) {
        allocator_.destroyImageView(irradianceView_);
    }
    if (visibilityView_ != VK_NULL_HANDLE) {
        allocator_.destroyImageView(visibilityView_);
    }

    // Destroy images
    allocator_.destroyImage(irradianceAtlas_);
    allocator_.destroyImage(visibilityAtlas_);

    // Destroy buffers
    allocator_.destroyBuffer(rayDataBuffer_);
    allocator_.destroyBuffer(uniformsBuffer_);
    allocator_.destroyBuffer(sceneGlobalsRefBuffer_);
    allocator_.destroyBuffer(sbtBuffer_);

    // Destroy descriptor resources
    if (passPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, passPool_, nullptr);
    }
    if (passLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, passLayout_, nullptr);
    }

    LOG_INFO("DDGIPass destroyed");
}

// ---------------------------------------------------------------------------
// createAtlases
// ---------------------------------------------------------------------------

void DDGIPass::createAtlases()
{
    // --- Irradiance atlas ---
    // Tiling: columns = probeGridDims.x * probeGridDims.z, rows = probeGridDims.y
    u32 irrCols = static_cast<u32>(config_.probeGridDims.x * config_.probeGridDims.z);
    u32 irrRows = static_cast<u32>(config_.probeGridDims.y);
    u32 irrWidth  = irrCols * IRRADIANCE_WITH_BORDER;
    u32 irrHeight = irrRows * IRRADIANCE_WITH_BORDER;

    VkImageCreateInfo irrInfo{};
    irrInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    irrInfo.imageType     = VK_IMAGE_TYPE_2D;
    irrInfo.format        = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    irrInfo.extent        = {irrWidth, irrHeight, 1};
    irrInfo.mipLevels     = 1;
    irrInfo.arrayLayers   = 1;
    irrInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    irrInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    irrInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    irrInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    irradianceAtlas_ = allocator_.createImage(irrInfo, VMA_MEMORY_USAGE_GPU_ONLY);

    irradianceView_ = allocator_.createImageView(
        irradianceAtlas_.image,
        VK_FORMAT_B10G11R11_UFLOAT_PACK32,
        VK_IMAGE_ASPECT_COLOR_BIT);

    irradianceBindlessIdx_ = descriptors_.registerTexture(
        irradianceView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // --- Visibility atlas ---
    u32 visCols = static_cast<u32>(config_.probeGridDims.x * config_.probeGridDims.z);
    u32 visRows = static_cast<u32>(config_.probeGridDims.y);
    u32 visWidth  = visCols * VISIBILITY_WITH_BORDER;
    u32 visHeight = visRows * VISIBILITY_WITH_BORDER;

    VkImageCreateInfo visInfo{};
    visInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    visInfo.imageType     = VK_IMAGE_TYPE_2D;
    visInfo.format        = VK_FORMAT_R16G16_SFLOAT;
    visInfo.extent        = {visWidth, visHeight, 1};
    visInfo.mipLevels     = 1;
    visInfo.arrayLayers   = 1;
    visInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    visInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    visInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT;
    visInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    visibilityAtlas_ = allocator_.createImage(visInfo, VMA_MEMORY_USAGE_GPU_ONLY);

    visibilityView_ = allocator_.createImageView(
        visibilityAtlas_.image,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT);

    visibilityBindlessIdx_ = descriptors_.registerTexture(
        visibilityView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    LOG_DEBUG("DDGI irradiance atlas: %ux%u (R11G11B10F)", irrWidth, irrHeight);
    LOG_DEBUG("DDGI visibility atlas:  %ux%u (RG16F)", visWidth, visHeight);
}

// ---------------------------------------------------------------------------
// createBuffers
// ---------------------------------------------------------------------------

void DDGIPass::createBuffers()
{
    // Ray data: vec4 (radiance.xyz + hitDistance) per ray per probe = 16 bytes each
    VkDeviceSize rayDataSize = static_cast<VkDeviceSize>(totalProbes_) *
                               config_.raysPerProbe * 16;

    rayDataBuffer_ = allocator_.createBuffer(
        rayDataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY);

    // Uniforms buffer (CPU-visible for easy updates)
    uniformsBuffer_ = allocator_.createBuffer(
        sizeof(DDGIUniformsGPU),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Scene globals reference buffer (CPU-visible, holds a single BDA pointer
    // that the RT shaders read to reach SceneGlobals).
    // We store a copy of the SceneGlobals struct here for the RT shaders.
    // Enough for the full SceneGlobals (104 bytes) — round up to 128.
    sceneGlobalsRefBuffer_ = allocator_.createBuffer(
        128,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    LOG_DEBUG("DDGI ray data buffer: %llu bytes (%u probes x %u rays)",
              static_cast<unsigned long long>(rayDataSize),
              totalProbes_, config_.raysPerProbe);
}

// ---------------------------------------------------------------------------
// uploadUniforms
// ---------------------------------------------------------------------------

void DDGIPass::uploadUniforms()
{
    DDGIUniformsGPU gpu{};
    gpu.probeGridDims    = config_.probeGridDims;
    gpu.probeSpacing     = config_.probeSpacing;
    gpu.probeGridOrigin  = config_.probeGridOrigin;
    gpu.maxRayDistance    = config_.maxRayDistance;
    gpu.raysPerProbe     = config_.raysPerProbe;
    gpu.hysteresis       = config_.hysteresis;
    gpu.irradianceGamma  = config_.irradianceGamma;
    gpu.pad              = 0.0f;

    void* mapped = allocator_.mapMemory(uniformsBuffer_);
    std::memcpy(mapped, &gpu, sizeof(gpu));
    allocator_.unmapMemory(uniformsBuffer_);
}

// ---------------------------------------------------------------------------
// createDescriptors — per-pass descriptor set (set = 1)
// ---------------------------------------------------------------------------

void DDGIPass::createDescriptors()
{
    VkDevice dev = device_.getDevice();

    // Bindings:
    //   0 = DDGIUniforms             (STORAGE_BUFFER)
    //   1 = RayData                  (STORAGE_BUFFER)
    //   2 = TLAS                     (ACCELERATION_STRUCTURE)
    //   3 = SceneGlobals ref         (STORAGE_BUFFER)
    //   4 = Irradiance atlas         (STORAGE_IMAGE)
    //   5 = Visibility atlas         (STORAGE_IMAGE)

    VkDescriptorSetLayoutBinding bindings[6] = {};

    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                  VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    bindings[4].binding         = 4;
    bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[5].binding         = 5;
    bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 6;
    layoutInfo.pBindings    = bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &passLayout_));

    // Pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,              3},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,  1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,               2},
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes    = poolSizes;

    VK_CHECK(vkCreateDescriptorPool(dev, &poolInfo, nullptr, &passPool_));

    // Allocate set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = passPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &passLayout_;

    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &passSet_));

    // Write descriptors
    // -- Binding 0: DDGI Uniforms --
    VkDescriptorBufferInfo uniformsInfo{};
    uniformsInfo.buffer = uniformsBuffer_.buffer;
    uniformsInfo.offset = 0;
    uniformsInfo.range  = sizeof(DDGIUniformsGPU);

    // -- Binding 1: Ray data --
    VkDescriptorBufferInfo rayDataInfo{};
    rayDataInfo.buffer = rayDataBuffer_.buffer;
    rayDataInfo.offset = 0;
    rayDataInfo.range  = VK_WHOLE_SIZE;

    // -- Binding 2: TLAS --
    VkAccelerationStructureKHR tlas = rtManager_.getTLAS();

    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures    = &tlas;

    // -- Binding 3: Scene globals ref --
    VkDescriptorBufferInfo sceneGlobalsInfo{};
    sceneGlobalsInfo.buffer = sceneGlobalsRefBuffer_.buffer;
    sceneGlobalsInfo.offset = 0;
    sceneGlobalsInfo.range  = VK_WHOLE_SIZE;

    // -- Binding 4: Irradiance atlas (storage image) --
    VkDescriptorImageInfo irrImageInfo{};
    irrImageInfo.imageView   = irradianceView_;
    irrImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // -- Binding 5: Visibility atlas (storage image) --
    VkDescriptorImageInfo visImageInfo{};
    visImageInfo.imageView   = visibilityView_;
    visImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[6] = {};

    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = passSet_;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &uniformsInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = passSet_;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo     = &rayDataInfo;

    writes[2].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].pNext           = &asWrite;
    writes[2].dstSet          = passSet_;
    writes[2].dstBinding      = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet          = passSet_;
    writes[3].dstBinding      = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].pBufferInfo     = &sceneGlobalsInfo;

    writes[4].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet          = passSet_;
    writes[4].dstBinding      = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].pImageInfo      = &irrImageInfo;

    writes[5].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet          = passSet_;
    writes[5].dstBinding      = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo      = &visImageInfo;

    vkUpdateDescriptorSets(dev, 6, writes, 0, nullptr);
}

// ---------------------------------------------------------------------------
// createPipelines
// ---------------------------------------------------------------------------

void DDGIPass::createPipelines()
{
    // RT pipeline
    RayTracingPipelineDesc rtDesc{};
    rtDesc.rgenPath          = "gi/ddgi_probe_trace.rgen.spv";
    rtDesc.rmissPath         = "gi/ddgi_probe_trace.rmiss.spv";
    rtDesc.rchitPath         = "gi/ddgi_probe_trace.rchit.spv";
    rtDesc.maxRecursionDepth = 1;

    rtPipeline_ = pipelines_.createRayTracingPipeline(rtDesc);

    // Compute pipelines
    ComputePipelineDesc irrDesc{};
    irrDesc.shaderPath = "gi/ddgi_probe_update_irradiance.comp.spv";
    irradianceUpdatePipeline_ = pipelines_.createComputePipeline(irrDesc);

    ComputePipelineDesc visDesc{};
    visDesc.shaderPath = "gi/ddgi_probe_update_visibility.comp.spv";
    visibilityUpdatePipeline_ = pipelines_.createComputePipeline(visDesc);
}

// ---------------------------------------------------------------------------
// createSBT — Shader Binding Table
// ---------------------------------------------------------------------------

void DDGIPass::createSBT()
{
    VkDevice dev = device_.getDevice();

    // Query RT pipeline properties for handle size and alignment
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(device_.getPhysicalDevice(), &props2);

    const u32 handleSize      = rtProps.shaderGroupHandleSize;
    const u32 handleAlignment = rtProps.shaderGroupHandleAlignment;
    const u32 baseAlignment   = rtProps.shaderGroupBaseAlignment;

    // Aligned handle size (each entry in the SBT must be aligned)
    auto alignUp = [](u32 size, u32 alignment) -> u32 {
        return (size + alignment - 1) & ~(alignment - 1);
    };

    const u32 handleSizeAligned = alignUp(handleSize, handleAlignment);

    // We have 3 shader groups: rgen, miss, closest-hit
    const u32 groupCount = 3;

    // Each region must start at baseAlignment
    const u32 rgenSize = alignUp(handleSizeAligned, baseAlignment);
    const u32 missSize = alignUp(handleSizeAligned, baseAlignment);
    const u32 hitSize  = alignUp(handleSizeAligned, baseAlignment);
    const u32 sbtTotalSize = rgenSize + missSize + hitSize;

    // Get shader group handles
    std::vector<u8> handleData(static_cast<size_t>(groupCount) * handleSize);

    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetHandles =
        reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(dev, "vkGetRayTracingShaderGroupHandlesKHR"));
    if (!vkGetHandles) {
        LOG_ERROR("Failed to load vkGetRayTracingShaderGroupHandlesKHR");
        std::abort();
    }

    VK_CHECK(vkGetHandles(dev, rtPipeline_, 0, groupCount,
                           handleData.size(), handleData.data()));

    // Allocate SBT buffer
    sbtBuffer_ = allocator_.createBuffer(
        sbtTotalSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Map and copy handles at aligned offsets
    u8* mapped = static_cast<u8*>(allocator_.mapMemory(sbtBuffer_));
    std::memset(mapped, 0, sbtTotalSize);

    // Group 0: rgen
    std::memcpy(mapped, handleData.data(), handleSize);
    // Group 1: miss
    std::memcpy(mapped + rgenSize, handleData.data() + handleSize, handleSize);
    // Group 2: closest hit
    std::memcpy(mapped + rgenSize + missSize, handleData.data() + 2 * handleSize, handleSize);

    allocator_.unmapMemory(sbtBuffer_);

    // Fill in the strided device address regions
    VkDeviceAddress sbtBaseAddr = sbtBuffer_.deviceAddress;

    rgenRegion_.deviceAddress = sbtBaseAddr;
    rgenRegion_.stride        = handleSizeAligned;
    rgenRegion_.size          = rgenSize;

    missRegion_.deviceAddress = sbtBaseAddr + rgenSize;
    missRegion_.stride        = handleSizeAligned;
    missRegion_.size          = missSize;

    hitRegion_.deviceAddress  = sbtBaseAddr + rgenSize + missSize;
    hitRegion_.stride         = handleSizeAligned;
    hitRegion_.size           = hitSize;

    // Callable region — unused
    callRegion_ = {};

    LOG_DEBUG("DDGI SBT created: handleSize=%u, aligned=%u, total=%u bytes",
              handleSize, handleSizeAligned, sbtTotalSize);
}

// ---------------------------------------------------------------------------
// setSceneGlobalsAddress — update the BDA pointer for the RT shaders
// ---------------------------------------------------------------------------

void DDGIPass::setSceneGlobalsAddress(VkDeviceAddress addr)
{
    sceneGlobalsAddress_ = addr;
    // The scene globals ref buffer is used directly as a storage buffer
    // by the RT shaders (binding 3), not as a BDA indirection.
    // We do not need to re-upload here; the buffer content will be
    // populated by the renderer before dispatch.
}

// ---------------------------------------------------------------------------
// traceProbeRays
// ---------------------------------------------------------------------------

void DDGIPass::traceProbeRays(VkCommandBuffer cmd)
{
    // Transition atlases to GENERAL for storage image access
    VkImageMemoryBarrier2 barriers[2] = {};

    barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[0].dstStageMask        = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barriers[0].dstAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[0].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].image               = irradianceAtlas_.image;
    barriers[0].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1] = barriers[0];
    barriers[1].image               = visibilityAtlas_.image;

    VkDependencyInfo depInfo{};
    depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.imageMemoryBarrierCount  = 2;
    depInfo.pImageMemoryBarriers     = barriers;

    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Bind RT pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline_);

    // Bind descriptor sets: set 0 = bindless, set 1 = pass
    VkDescriptorSet sets[] = {
        descriptors_.getDescriptorSet(),
        passSet_
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            pipelines_.getPipelineLayout(), 0, 2, sets, 0, nullptr);

    // Dispatch rays: width = raysPerProbe, height = totalProbes, depth = 1
    vkCmdTraceRays_(cmd,
                    &rgenRegion_,
                    &missRegion_,
                    &hitRegion_,
                    &callRegion_,
                    config_.raysPerProbe,
                    totalProbes_,
                    1);

    // Barrier: ray data writes must complete before compute reads
    VkMemoryBarrier2 memBarrier{};
    memBarrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memBarrier.srcStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    memBarrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo memDep{};
    memDep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    memDep.memoryBarrierCount = 1;
    memDep.pMemoryBarriers    = &memBarrier;

    vkCmdPipelineBarrier2(cmd, &memDep);
}

// ---------------------------------------------------------------------------
// updateIrradiance
// ---------------------------------------------------------------------------

void DDGIPass::updateIrradiance(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, irradianceUpdatePipeline_);

    VkDescriptorSet sets[] = {
        descriptors_.getDescriptorSet(),
        passSet_
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_.getPipelineLayout(), 0, 2, sets, 0, nullptr);

    // Dispatch: (1, totalProbes, 1) workgroups, each 8x8x1 threads
    vkCmdDispatch(cmd, 1, totalProbes_, 1);

    // Barrier: irradiance writes must complete before visibility reads or
    // subsequent frame reads
    VkMemoryBarrier2 barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

    VkDependencyInfo dep{};
    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// updateVisibility
// ---------------------------------------------------------------------------

void DDGIPass::updateVisibility(VkCommandBuffer cmd)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, visibilityUpdatePipeline_);

    VkDescriptorSet sets[] = {
        descriptors_.getDescriptorSet(),
        passSet_
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelines_.getPipelineLayout(), 0, 2, sets, 0, nullptr);

    // Dispatch: (1, totalProbes, 1) workgroups, each 16x16x1 threads
    vkCmdDispatch(cmd, 1, totalProbes_, 1);

    // Transition atlases to SHADER_READ_ONLY for sampling in the lighting pass
    VkImageMemoryBarrier2 barriers[2] = {};

    barriers[0].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT;
    barriers[0].dstStageMask        = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT;
    barriers[0].oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[0].image               = irradianceAtlas_.image;
    barriers[0].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    barriers[1] = barriers[0];
    barriers[1].image               = visibilityAtlas_.image;

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers    = barriers;

    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

VkImageView DDGIPass::getIrradianceView() const
{
    return irradianceView_;
}

VkImageView DDGIPass::getVisibilityView() const
{
    return visibilityView_;
}

u32 DDGIPass::getIrradianceBindlessIdx() const
{
    return irradianceBindlessIdx_;
}

u32 DDGIPass::getVisibilityBindlessIdx() const
{
    return visibilityBindlessIdx_;
}

const DDGIConfig& DDGIPass::getConfig() const
{
    return config_;
}

} // namespace phosphor
