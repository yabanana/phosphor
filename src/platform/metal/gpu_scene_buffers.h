#pragma once

#include "core/types.h"
#include "platform/metal/metal_context.h"
#include "renderer/gpu_scene_layout.h"

#include <array>
#include <vector>

namespace phosphor {

class SceneStore;

// ---------------------------------------------------------------------------
// GpuSceneBuffers -- GPU memory of the persistent scene (F5.1).
//
// Persistent buffers (private, MemoryCategory::Scene, one copy: the
// cross-frame wait on them costs 0.2% per spike S2) mirror SceneStore:
// instances, materials, nodes, motion records by slot, the identity list
// (gpu-driven off), bucket table, command -> bucket table, CSR, motion slot
// list, GPU queues 1..7.  Per frame slot (METAL_FRAMES_IN_FLIGHT copies:
// the forward pass of frame n may still read them while frame n+1 culls):
// cull flags, group counts, visible list, prefix, the ICB and its container,
// draw arguments and counters (shared: read back by the CPU).
//
// Capacities only grow (x1.5), at structure changes: an allocation is an
// event, never expected in measured frames (O7), and a (re)allocated
// persistent buffer is re-sent in full.
//
// Data flow (D1): loadAll() copies the whole mirror through the staging ring
// (bench switch, blocking).  stageFrame() writes the frame's delta records,
// full copies and structure buffers into the frame upload ring and lists
// what the "Scene update" pass must encode (scatters and copies).
// ---------------------------------------------------------------------------

class GpuSceneBuffers {
public:
    struct Scatter {
        MTL::GPUAddress records = 0; // GPUDeltaRecord[] in the frame ring (0: none)
        u32             count   = 0;
        MTL::Buffer*    dst     = nullptr;
    };
    struct Copy {
        MTL::Buffer* src       = nullptr;
        u64          srcOffset = 0;
        MTL::Buffer* dst       = nullptr;
        u64          size      = 0;
    };
    /// Per-slot buffers of one frame.
    struct FrameSet {
        MTL::Buffer*                flags     = nullptr;
        MTL::Buffer*                groups    = nullptr;
        MTL::Buffer*                visible   = nullptr;
        MTL::Buffer*                prefix    = nullptr;
        MTL::Buffer*                drawArgs  = nullptr; // shared (self-check readback)
        MTL::Buffer*                counters  = nullptr; // shared GPUSceneCounters
        MTL::IndirectCommandBuffer* icb       = nullptr;
        MTL::Buffer*                container = nullptr; // shared, the ICB's resource ID
    };

    explicit GpuSceneBuffers(MetalContext& context);
    ~GpuSceneBuffers();
    GpuSceneBuffers(const GpuSceneBuffers&) = delete;
    GpuSceneBuffers& operator=(const GpuSceneBuffers&) = delete;

    /// Drop everything (bench switch; GPU idle).
    void clear();
    /// Size every buffer for `store` and upload the whole mirror through the
    /// staging ring (blocking; load time and structure changes outside a frame).
    void loadAll(const SceneStore& store);
    /// Frame: (re)size if the structure grew, then write the frame's deltas,
    /// full copies and structure buffers into the frame upload ring.  Fills
    /// scatters() and copies(); returns the bytes written by the CPU.
    u64 stageFrame(const SceneStore& store);

    [[nodiscard]] const std::array<Scatter, 4>& scatters() const { return scatters_; }
    [[nodiscard]] const std::vector<Copy>& copies() const { return copies_; }

    // --- Persistent ----------------------------------------------------------------
    [[nodiscard]] MTL::Buffer* instances() const { return instances_; }
    [[nodiscard]] MTL::Buffer* materials() const { return materials_; }
    [[nodiscard]] MTL::Buffer* nodes() const { return nodes_; }
    [[nodiscard]] MTL::Buffer* motions() const { return motions_; }
    [[nodiscard]] MTL::Buffer* identity() const { return identity_; }
    [[nodiscard]] MTL::Buffer* buckets() const { return buckets_; }
    [[nodiscard]] MTL::Buffer* commandBuckets() const { return commandBuckets_; }
    [[nodiscard]] MTL::Buffer* childOffsets() const { return childOffsets_; }
    [[nodiscard]] MTL::Buffer* childSlots() const { return childSlots_; }
    [[nodiscard]] MTL::Buffer* motionSlots() const { return motionSlots_; }
    [[nodiscard]] MTL::Buffer* queues() const { return queues_; }
    [[nodiscard]] u32 queueStride() const { return queueStride_; }
    [[nodiscard]] const FrameSet& frame(u32 slot) const { return frames_[slot]; }

    [[nodiscard]] u32 slotCapacity() const { return slotCap_; }
    [[nodiscard]] u32 commandCapacity() const { return commandCap_; }
    [[nodiscard]] u32 cullGroups(u32 slots) const { return (slots + SCENE_CULL_GROUP - 1) / SCENE_CULL_GROUP; }
    /// Incremented when a buffer is (re)allocated (render graph key, bindings).
    [[nodiscard]] u64 version() const { return version_; }
    [[nodiscard]] bool empty() const { return instances_ == nullptr; }

private:
    /// Grow every buffer to fit `store`; true if anything was (re)allocated.
    bool reserve(const SceneStore& store);
    void releaseAll();
    MTL::Buffer* privateBuffer(u64 size, const char* label);
    MTL::Buffer* sharedBuffer(u64 size, const char* label);
    void stageFull(MTL::Buffer* dst, const void* data, u64 size);

    MetalContext& context_;
    MTL::Buffer* instances_      = nullptr;
    MTL::Buffer* materials_      = nullptr;
    MTL::Buffer* nodes_          = nullptr;
    MTL::Buffer* motions_        = nullptr;
    MTL::Buffer* identity_       = nullptr;
    MTL::Buffer* buckets_        = nullptr;
    MTL::Buffer* commandBuckets_ = nullptr;
    MTL::Buffer* childOffsets_   = nullptr;
    MTL::Buffer* childSlots_     = nullptr;
    MTL::Buffer* motionSlots_    = nullptr;
    MTL::Buffer* queues_         = nullptr;
    std::array<FrameSet, METAL_FRAMES_IN_FLIGHT> frames_{};

    u32 slotCap_     = 0;
    u32 materialCap_ = 0;
    u32 bucketCap_   = 0;
    u32 commandCap_  = 0;
    u32 childCap_    = 0;
    u32 motionCap_   = 0;
    u32 queueStride_ = 0;
    u64 version_     = 0;
    bool forceFull_  = false; // a persistent buffer was reallocated: next stageFrame() re-sends everything

    std::array<Scatter, 4> scatters_{};
    std::vector<Copy>      copies_;
};

} // namespace phosphor
