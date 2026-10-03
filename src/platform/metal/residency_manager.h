#pragma once

#include "core/types.h"

#include <Metal/Metal.hpp>

#include <array>
#include <unordered_map>

namespace phosphor {

// Which residency set an allocation belongs to.
enum class ResidencyClass : u8 {
    Static,    // engine lifetime: upload rings, tables, render targets
    Streaming, // level content: geometry and material textures (F22 streams these)
    COUNT
};

// ---------------------------------------------------------------------------
// ResidencyManager -- the queue's residency sets (Metal 4 command buffers do
// not make resources resident on their own).
//
// One set per ResidencyClass, all attached to the MTL4 queue (well below the
// limit of 32 sets per queue, S-MEM-5).  add()/remove() only record changes;
// commit() -- called by MetalContext right before each command-buffer commit
// -- commits the sets that changed, so a frame costs at most one commit per
// set and none when nothing changed.  Heaps count as one allocation each.
// ---------------------------------------------------------------------------

class ResidencyManager {
public:
    struct Stats {
        u32 allocations = 0;
        u64 bytes       = 0;
        u32 commits     = 0;
    };

    /// The sets are attached to `queue` and, if given, to `asyncQueue` (F2.6).
    ResidencyManager(MTL::Device* device, MTL4::CommandQueue* queue, MTL4::CommandQueue* asyncQueue = nullptr);
    ~ResidencyManager();

    ResidencyManager(const ResidencyManager&) = delete;
    ResidencyManager& operator=(const ResidencyManager&) = delete;

    void add(const MTL::Allocation* allocation, ResidencyClass cls);
    /// Remove from whichever set holds it (no-op if it is not resident).
    bool remove(const MTL::Allocation *allocation);
    /// Commit the sets with pending changes.
    void commit();

    [[nodiscard]] Stats stats(ResidencyClass cls) const;

private:
    static constexpr u32 COUNT = static_cast<u32>(ResidencyClass::COUNT);

    MTL4::CommandQueue* queue_      = nullptr;
    MTL4::CommandQueue* asyncQueue_ = nullptr;
    std::array<MTL::ResidencySet*, COUNT> sets_{};
    std::array<bool, COUNT> dirty_{};
    std::array<u32, COUNT> commits_{};
    std::unordered_map<const MTL::Allocation*, ResidencyClass> owner_;
};

} // namespace phosphor
