#pragma once

#include "core/memory/memory_budget.h"
#include "core/types.h"

#include <Metal/Metal.hpp>

#include <array>

namespace phosphor {

class MetalContext;

// ---------------------------------------------------------------------------
// GpuMemory -- the only place where the backend creates GPU buffers and
// textures.
//
// Every allocation is labelled, made resident (except memoryless textures)
// and accounted per MemoryCategory.  allocationCount() is monotonic: the
// engine samples it around a benchmark run to prove that no GPU allocation
// happens once the frame loop is warm (rule O7).
//
// Releases are deferred by default: the resource stays resident and alive
// until every frame in flight at release time has completed.
// ---------------------------------------------------------------------------

class GpuMemory {
public:
    struct CategoryStats {
        u64 bytes = 0;
        u32 count = 0;
    };

    explicit GpuMemory(MetalContext& context);
    ~GpuMemory();

    GpuMemory(const GpuMemory&) = delete;
    GpuMemory& operator=(const GpuMemory&) = delete;

    [[nodiscard]] MTL::Buffer* newBuffer(u64 length, MTL::ResourceOptions options, MemoryCategory category,
                                         const char* label);
    [[nodiscard]] MTL::Texture* newTexture(const MTL::TextureDescriptor* descriptor, MemoryCategory category,
                                           const char* label);

    /// Release after the frames currently in flight complete (safe default).
    void release(MTL::Resource* resource, MemoryCategory category);
    /// Release now; the caller guarantees the GPU no longer uses it.
    void releaseNow(MTL::Resource* resource, MemoryCategory category);

    /// Allocations made since the GpuMemory was created (monotonic).
    [[nodiscard]] u64 allocationCount() const { return allocationCount_; }
    [[nodiscard]] const CategoryStats& stats(MemoryCategory category) const {
        return stats_[static_cast<u32>(category)];
    }
    [[nodiscard]] u64 totalBytes() const;

private:
    void account(MTL::Resource* resource, MemoryCategory category);
    void unaccount(MTL::Resource* resource, MemoryCategory category);

    MetalContext& context_;
    std::array<CategoryStats, MEMORY_CATEGORY_COUNT> stats_{};
    u64 allocationCount_ = 0;
};

} // namespace phosphor
