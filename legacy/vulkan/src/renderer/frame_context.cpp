#include "renderer/frame_context.h"
#include "rhi/vk_device.h"
#include "rhi/vk_commands.h"
#include "rhi/vk_sync.h"

namespace phosphor {

FrameContext::FrameContext(VulkanDevice& device, GpuAllocator& allocator,
                           CommandManager& commands, SyncManager& sync, u32 frameIndex)
    : device_(&device), allocator_(&allocator), commands_(&commands), sync_(&sync), frameIndex_(frameIndex) {
    // Create staging buffer (host-visible, coherent, for per-frame uploads)
    stagingBuffer_ = allocator_->createBuffer(
        STAGING_BUFFER_SIZE,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
    );
    stagingMapped_ = allocator_->mapMemory(stagingBuffer_);
    cmdBuffer_ = commands_->getCommandBuffer(frameIndex);
}

FrameContext::~FrameContext() {
    if (allocator_ && stagingBuffer_.buffer) {
        allocator_->unmapMemory(stagingBuffer_);
        allocator_->destroyBuffer(stagingBuffer_);
    }
}

FrameContext::FrameContext(FrameContext&& o) noexcept
    : device_(o.device_), allocator_(o.allocator_), commands_(o.commands_), sync_(o.sync_),
      frameIndex_(o.frameIndex_), stagingBuffer_(o.stagingBuffer_),
      stagingMapped_(o.stagingMapped_), stagingOffset_(o.stagingOffset_), cmdBuffer_(o.cmdBuffer_) {
    o.stagingBuffer_ = {};
    o.stagingMapped_ = nullptr;
}

FrameContext& FrameContext::operator=(FrameContext&& o) noexcept {
    if (this != &o) {
        if (allocator_ && stagingBuffer_.buffer) {
            allocator_->unmapMemory(stagingBuffer_);
            allocator_->destroyBuffer(stagingBuffer_);
        }
        device_ = o.device_;
        allocator_ = o.allocator_;
        commands_ = o.commands_;
        sync_ = o.sync_;
        frameIndex_ = o.frameIndex_;
        stagingBuffer_ = o.stagingBuffer_;
        stagingMapped_ = o.stagingMapped_;
        stagingOffset_ = o.stagingOffset_;
        cmdBuffer_ = o.cmdBuffer_;
        o.stagingBuffer_ = {};
        o.stagingMapped_ = nullptr;
    }
    return *this;
}

void FrameContext::begin() {
    sync_->waitForFrame(frameIndex_);
    commands_->resetPools(frameIndex_);
    stagingOffset_ = 0;

    cmdBuffer_ = commands_->getCommandBuffer(frameIndex_);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmdBuffer_, &beginInfo));
}

void FrameContext::end() {
    VK_CHECK(vkEndCommandBuffer(cmdBuffer_));
}

StagingAllocation FrameContext::allocateStaging(VkDeviceSize size, VkDeviceSize alignment) {
    // Align offset
    VkDeviceSize aligned = (stagingOffset_ + alignment - 1) & ~(alignment - 1);

    if (aligned + size > STAGING_BUFFER_SIZE) {
        LOG_ERROR("Staging buffer exhausted (requested %llu, offset %llu, capacity %llu)",
                  (unsigned long long)size, (unsigned long long)aligned, (unsigned long long)STAGING_BUFFER_SIZE);
        return {};
    }

    StagingAllocation alloc{};
    alloc.cpuPtr = static_cast<u8*>(stagingMapped_) + aligned;
    alloc.offset = aligned;
    alloc.buffer = stagingBuffer_.buffer;

    stagingOffset_ = aligned + size;
    return alloc;
}

} // namespace phosphor
