#pragma once

#include "core/types.h"

#include <optional>
#include <span>
#include <vector>

namespace phosphor {

enum class RtBlasState : u8 { Pending, Built, CompactionQueued, Compacted };
enum class RtWorkKind : u8 { Build, Refit, Compact };
enum class RtTlasAction : u8 { None, Build, Refit };

struct RtWorkBudget {
    u32 builds = ~0u, refits = ~0u, compactions = ~0u;
};
struct RtWork {
    u32 mesh = 0;
    RtWorkKind kind = RtWorkKind::Build;
    u64 version = 0, topologyRevision = 0, vertexRevision = 0;
};
struct RtBlasRecord {
    RtBlasState state = RtBlasState::Pending;
    bool requested = false, forceRebuild = false;
    // version changes for every build/refit/compact publication; resourceID
    // changes only for a replacement AS. Capture version with async queries.
    u64 version = 0, resourceID = 0, bytes = 0, compactBytes = 0;
    u64 topologyRevision = 0, vertexRevision = 0;
    u64 requestedTopology = 0, requestedVertices = 0;
    u64 lastReader = 0;
};
struct RtRetiredBlas {
    u32 mesh = 0;
    u64 version = 0, resourceID = 0, bytes = 0, lastReader = 0;
};

// Portable ownership ledger; the backend owns the actual GPU resources.
// All calls are on the rendering thread. Publication happens only after an
// AS build/copy has been scheduled; frame is its submission frame. Calls that
// modify an AS in place must be ordered by the render graph after its readers.
// Resource IDs must be unique among active AND retired resources. The backend
// releases an old resource only after collectRetired returns it, then still
// uses GpuMemory's deferred release. completeFrame means ALL relevant queues
// completed that frame; never just that CPU encoding ended.
class RtScene {
public:
    explicit RtScene(u32 frameSlots = 3);
    void request(u32 mesh, u64 topologyRevision, u64 vertexRevision);
    void remove(u32 mesh, u64 frame);
    void requestRebuild(u32 mesh);
    void plan(const RtWorkBudget& budget, std::vector<RtWork>& out) const;
    [[nodiscard]] std::vector<RtWork> plan(const RtWorkBudget& budget = {}) const;
    [[nodiscard]] bool hasWork() const;

    void markBuilt(u32 mesh, u64 resourceID, u64 bytes, u64 frame);
    void markRefitted(u32 mesh, u64 frame);
    void queueCompaction(u32 mesh, u64 compactBytes);
    // For an asynchronous size query: false if its content version is stale.
    bool queueCompaction(u32 mesh, u64 compactBytes, u64 expectedVersion);
    // Returns false when an asynchronous result belongs to a replaced version.
    // In that case the caller retains ownership of its unpublished destination.
    bool markCompacted(u32 mesh, u64 expectedVersion, u64 resourceID, u64 bytes, u64 frame);

    [[nodiscard]] RtTlasAction tlasAction(u32 slot, u32 capacity, u64 frame, u32 rebuildEvery = 0) const;
    // Records a successfully encoded TLAS snapshot of the CURRENT BLAS table.
    // A busy slot cannot be overwritten. Capacity zero clears a retired slot.
    void commitSnapshot(u32 slot, u32 capacity, u64 frame, RtTlasAction action);
    void commitSnapshot(u32 slot, u32 capacity, u64 frame);
    void clearSnapshot(u32 slot);
    void completeFrame(u64 frame);
    void collectRetired(std::vector<RtRetiredBlas>& out);
    [[nodiscard]] std::vector<RtRetiredBlas> collectRetired();

    // BLAS resource-set changes only; in-place refit leaves this unchanged.
    [[nodiscard]] u64 tableGeneration() const { return tableGeneration_; }
    [[nodiscard]] std::span<const RtBlasRecord> meshes() const { return meshes_; }
    [[nodiscard]] const RtBlasRecord& mesh(u32 index) const { return meshes_.at(index); }
    [[nodiscard]] size_t retiredCount() const { return retired_.size(); }
    [[nodiscard]] u32 frameSlots() const { return static_cast<u32>(slots_.size()); }

private:
    struct Snapshot {
        bool valid = false;
        u32 capacity = 0;
        u64 frame = 0, lastBuild = 0, tableGeneration = 0;
        std::vector<u64> versions, resources;
    };
    void retire(u32 mesh, u64 frame);
    void validateDestination(u64 resourceID, u64 bytes) const;
    void ensureSlotIdle(const Snapshot& slot) const;
    std::vector<RtBlasRecord> meshes_;
    std::vector<RtRetiredBlas> retired_;
    std::vector<Snapshot> slots_;
    std::optional<u64> completedFrame_;
    u64 nextVersion_ = 1, tableGeneration_ = 0;
};

} // namespace phosphor
