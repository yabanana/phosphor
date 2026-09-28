#include "render_graph/resource_registry.h"
#include "core/log.h"

#include <cassert>

namespace phosphor {

ResourceRegistry::ResourceRegistry(GpuAllocator& allocator)
    : allocator_(allocator)
{
}

ResourceRegistry::~ResourceRegistry()
{
    reset();
}

ResourceHandle ResourceRegistry::createTransientImage(const ImageDesc& desc)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = desc.format;
    imageInfo.extent        = { desc.width, desc.height, 1 };
    imageInfo.mipLevels     = desc.mipLevels;
    imageInfo.arrayLayers   = desc.layers;
    imageInfo.samples       = desc.samples;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = desc.usage;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    AllocatedImage allocated = allocator_.createImage(imageInfo, VMA_MEMORY_USAGE_GPU_ONLY);

    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    if (desc.format == VK_FORMAT_D32_SFLOAT ||
        desc.format == VK_FORMAT_D16_UNORM) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    } else if (desc.format == VK_FORMAT_D24_UNORM_S8_UINT ||
               desc.format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    VkImageView view = allocator_.createImageView(
        allocated.image, desc.format, aspect, desc.mipLevels,
        VK_IMAGE_VIEW_TYPE_2D, desc.layers);

    u32 imageIdx = static_cast<u32>(images_.size());
    images_.push_back({ allocated, view, desc, /*external=*/false });

    ResourceHandle handle = static_cast<ResourceHandle>(slots_.size());
    slots_.push_back({ ResourceKind::Image, imageIdx });

    LOG_DEBUG("ResourceRegistry: created transient image %u (%ux%u, fmt %d)",
              handle, desc.width, desc.height, static_cast<int>(desc.format));

    return handle;
}

ResourceHandle ResourceRegistry::createTransientBuffer(const BufferDesc& desc)
{
    AllocatedBuffer allocated = allocator_.createBuffer(
        desc.size, desc.usage, VMA_MEMORY_USAGE_GPU_ONLY);

    u32 bufIdx = static_cast<u32>(buffers_.size());
    buffers_.push_back({ allocated, desc });

    ResourceHandle handle = static_cast<ResourceHandle>(slots_.size());
    slots_.push_back({ ResourceKind::Buffer, bufIdx });

    LOG_DEBUG("ResourceRegistry: created transient buffer %u (size %llu)",
              handle, static_cast<unsigned long long>(desc.size));

    return handle;
}

ResourceHandle ResourceRegistry::importExternalImage(VkImage image,
                                                     VkImageView view,
                                                     const ImageDesc& desc)
{
    AllocatedImage allocated{};
    allocated.image     = image;
    allocated.format    = desc.format;
    allocated.extent    = { desc.width, desc.height, 1 };
    allocated.mipLevels = desc.mipLevels;
    // allocation left null -- we do not own this memory

    u32 imageIdx = static_cast<u32>(images_.size());
    images_.push_back({ allocated, view, desc, /*external=*/true });

    ResourceHandle handle = static_cast<ResourceHandle>(slots_.size());
    slots_.push_back({ ResourceKind::Image, imageIdx });

    LOG_DEBUG("ResourceRegistry: imported external image %u", handle);

    return handle;
}

VkImage ResourceRegistry::getImage(ResourceHandle handle) const
{
    assert(handle < slots_.size() && "invalid resource handle");
    const Slot& slot = slots_[handle];
    assert(slot.kind == ResourceKind::Image && "handle is not an image");
    return images_[slot.index].allocated.image;
}

VkImageView ResourceRegistry::getImageView(ResourceHandle handle) const
{
    assert(handle < slots_.size() && "invalid resource handle");
    const Slot& slot = slots_[handle];
    assert(slot.kind == ResourceKind::Image && "handle is not an image");
    return images_[slot.index].view;
}

VkBuffer ResourceRegistry::getBuffer(ResourceHandle handle) const
{
    assert(handle < slots_.size() && "invalid resource handle");
    const Slot& slot = slots_[handle];
    assert(slot.kind == ResourceKind::Buffer && "handle is not a buffer");
    return buffers_[slot.index].allocated.buffer;
}

const ImageDesc& ResourceRegistry::getImageDesc(ResourceHandle handle) const
{
    assert(handle < slots_.size() && "invalid resource handle");
    const Slot& slot = slots_[handle];
    assert(slot.kind == ResourceKind::Image && "handle is not an image");
    return images_[slot.index].desc;
}

const BufferDesc& ResourceRegistry::getBufferDesc(ResourceHandle handle) const
{
    assert(handle < slots_.size() && "invalid resource handle");
    const Slot& slot = slots_[handle];
    assert(slot.kind == ResourceKind::Buffer && "handle is not a buffer");
    return buffers_[slot.index].desc;
}

bool ResourceRegistry::isImage(ResourceHandle handle) const
{
    if (handle >= slots_.size()) return false;
    return slots_[handle].kind == ResourceKind::Image;
}

bool ResourceRegistry::isBuffer(ResourceHandle handle) const
{
    if (handle >= slots_.size()) return false;
    return slots_[handle].kind == ResourceKind::Buffer;
}

void ResourceRegistry::reset()
{
    for (auto& img : images_) {
        if (!img.external) {
            if (img.view != VK_NULL_HANDLE) {
                allocator_.destroyImageView(img.view);
            }
            if (img.allocated.image != VK_NULL_HANDLE) {
                allocator_.destroyImage(img.allocated);
            }
        }
    }

    for (auto& buf : buffers_) {
        if (buf.allocated.buffer != VK_NULL_HANDLE) {
            allocator_.destroyBuffer(buf.allocated);
        }
    }

    images_.clear();
    buffers_.clear();
    slots_.clear();
}

} // namespace phosphor
