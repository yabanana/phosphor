#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"

namespace phosphor {

enum class PassType : u8 {
    Graphics,
    Compute,
    Transfer,
    RayTracing,
};

using ResourceHandle = u32;
constexpr ResourceHandle INVALID_RESOURCE = ~0u;

struct ResourceAccess {
    ResourceHandle       handle = INVALID_RESOURCE;
    VkPipelineStageFlags2 stage  = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        access = VK_ACCESS_2_NONE;
    VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct ImageDesc {
    VkFormat              format  = VK_FORMAT_UNDEFINED;
    u32                   width   = 0;
    u32                   height  = 0;
    u32                   mipLevels = 1;
    u32                   layers    = 1;
    VkImageUsageFlags     usage   = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

struct BufferDesc {
    VkDeviceSize       size  = 0;
    VkBufferUsageFlags usage = 0;
};

} // namespace phosphor
