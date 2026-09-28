#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include "rhi/vk_allocator.h"
#include "render_graph/render_pass.h"
#include <vector>

namespace phosphor {

/// Manages transient (per-frame) and external (imported) resources for the
/// render graph.  Handles are plain indices into internal vectors.
class ResourceRegistry {
public:
    explicit ResourceRegistry(GpuAllocator& allocator);
    ~ResourceRegistry();

    ResourceRegistry(const ResourceRegistry&)            = delete;
    ResourceRegistry& operator=(const ResourceRegistry&) = delete;
    ResourceRegistry(ResourceRegistry&&)                 = delete;
    ResourceRegistry& operator=(ResourceRegistry&&)      = delete;

    /// Allocate a transient image that lives for one frame.
    ResourceHandle createTransientImage(const ImageDesc& desc);

    /// Allocate a transient buffer that lives for one frame.
    ResourceHandle createTransientBuffer(const BufferDesc& desc);

    /// Import a pre-existing image (e.g. swapchain) into the graph.
    ResourceHandle importExternalImage(VkImage image, VkImageView view,
                                       const ImageDesc& desc);

    VkImage           getImage(ResourceHandle handle) const;
    VkImageView       getImageView(ResourceHandle handle) const;
    VkBuffer          getBuffer(ResourceHandle handle) const;
    const ImageDesc&  getImageDesc(ResourceHandle handle) const;
    const BufferDesc& getBufferDesc(ResourceHandle handle) const;

    bool isImage(ResourceHandle handle) const;
    bool isBuffer(ResourceHandle handle) const;

    /// Destroy all transient resources.  External resources are released
    /// (not destroyed) since the caller owns them.  Called each frame or
    /// on resize.
    void reset();

private:
    enum class ResourceKind : u8 { Image, Buffer };

    struct ImageEntry {
        AllocatedImage  allocated{};
        VkImageView     view      = VK_NULL_HANDLE;
        ImageDesc       desc{};
        bool            external  = false;
    };

    struct BufferEntry {
        AllocatedBuffer allocated{};
        BufferDesc      desc{};
    };

    /// Unified handle slot -- discriminated by kind.
    struct Slot {
        ResourceKind kind;
        u32          index; // index into images_ or buffers_
    };

    GpuAllocator&            allocator_;
    std::vector<Slot>        slots_;
    std::vector<ImageEntry>  images_;
    std::vector<BufferEntry> buffers_;
};

} // namespace phosphor
