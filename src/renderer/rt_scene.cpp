#include "renderer/rt_scene.h"

#include <algorithm>
#include <stdexcept>

namespace phosphor {

RtScene::RtScene(u32 frameSlots) : slots_(frameSlots) {
    if (!frameSlots) throw std::invalid_argument("RT scene needs at least one frame slot");
}

void RtScene::request(u32 index, u64 topology, u64 vertices) {
    if (index >= meshes_.size()) meshes_.resize(size_t(index) + 1);
    auto& m = meshes_[index];
    if (m.state == RtBlasState::CompactionQueued &&
        (m.requestedTopology != topology || m.requestedVertices != vertices)) {
        m.state = RtBlasState::Built;
        m.compactBytes = 0; // The queried size no longer describes this content.
    }
    m.requested = true;
    m.requestedTopology = topology;
    m.requestedVertices = vertices;
}

void RtScene::retire(u32 index, u64 frame) {
    auto& m = meshes_.at(index);
    if (m.resourceID)
        retired_.push_back({index, m.version, m.resourceID, m.bytes, std::max(frame, m.lastReader)});
}

void RtScene::remove(u32 index, u64 frame) {
    auto& m = meshes_.at(index);
    if (!m.requested && !m.resourceID) return;
    retire(index, frame);
    m = {};
    ++tableGeneration_;
}

void RtScene::requestRebuild(u32 index) {
    auto& m = meshes_.at(index);
    if (!m.requested) throw std::logic_error("RT rebuild was not requested for this mesh");
    m.forceRebuild = true;
    if (m.state == RtBlasState::CompactionQueued) m.state = RtBlasState::Built;
    m.compactBytes = 0;
}

void RtScene::plan(const RtWorkBudget& budget, std::vector<RtWork>& out) const {
    out.clear();
    u32 builds = 0, refits = 0, compactions = 0;
    for (u32 i = 0; i < meshes_.size(); ++i) {
        const auto& m = meshes_[i];
        if (!m.requested) continue;
        const bool build = m.forceRebuild || !m.resourceID || m.topologyRevision != m.requestedTopology;
        const bool refit = !build && m.vertexRevision != m.requestedVertices;
        if (build && builds < budget.builds) {
            out.push_back({i, RtWorkKind::Build, m.version, m.requestedTopology, m.requestedVertices});
            ++builds;
        } else if (refit && refits < budget.refits) {
            out.push_back({i, RtWorkKind::Refit, m.version, m.requestedTopology, m.requestedVertices});
            ++refits;
        } else if (!build && !refit && m.state == RtBlasState::CompactionQueued && compactions < budget.compactions) {
            out.push_back({i, RtWorkKind::Compact, m.version, m.requestedTopology, m.requestedVertices});
            ++compactions;
        }
    }
}

std::vector<RtWork> RtScene::plan(const RtWorkBudget& budget) const {
    std::vector<RtWork> out;
    plan(budget, out);
    return out;
}

bool RtScene::hasWork() const {
    return std::any_of(meshes_.begin(), meshes_.end(), [](const auto& m) {
        return m.requested && (m.forceRebuild || !m.resourceID || m.topologyRevision != m.requestedTopology ||
                              m.vertexRevision != m.requestedVertices || m.state == RtBlasState::CompactionQueued);
    });
}

void RtScene::validateDestination(u64 id, u64 bytes) const {
    if (!id || !bytes) throw std::invalid_argument("RT publication requires a resource and nonzero size");
    if (std::any_of(meshes_.begin(), meshes_.end(), [&](const auto& m) { return m.resourceID == id; }) ||
        std::any_of(retired_.begin(), retired_.end(), [&](const auto& m) { return m.resourceID == id; }))
        throw std::invalid_argument("RT resource ID already owned");
}

void RtScene::markBuilt(u32 index, u64 id, u64 bytes, u64 frame) {
    auto& m = meshes_.at(index);
    if (!m.requested) throw std::logic_error("RT build was not requested");
    validateDestination(id, bytes);
    retire(index, frame);
    m.state = RtBlasState::Built;
    m.forceRebuild = false;
    m.version = nextVersion_++;
    m.resourceID = id;
    m.bytes = bytes;
    m.compactBytes = 0;
    m.topologyRevision = m.requestedTopology;
    m.vertexRevision = m.requestedVertices;
    m.lastReader = frame;
    ++tableGeneration_;
}

void RtScene::markRefitted(u32 index, u64 frame) {
    auto& m = meshes_.at(index);
    if (!m.requested || !m.resourceID || m.forceRebuild || m.topologyRevision != m.requestedTopology)
        throw std::logic_error("RT refit requires a built AS with unchanged topology");
    m.vertexRevision = m.requestedVertices;
    m.version = nextVersion_++; // Invalidate size/copy results for older contents.
    m.lastReader = std::max(frame, m.lastReader);
    if (m.state == RtBlasState::CompactionQueued) m.state = RtBlasState::Built;
    m.compactBytes = 0;
    // Resource/table generation stays unchanged: TLAS refit is sufficient.
}

void RtScene::queueCompaction(u32 index, u64 bytes) {
    auto& m = meshes_.at(index);
    if (!m.resourceID || m.forceRebuild || m.state != RtBlasState::Built || !bytes || bytes > m.bytes ||
        m.topologyRevision != m.requestedTopology || m.vertexRevision != m.requestedVertices)
        throw std::logic_error("RT compaction requires a current built AS and valid compact size");
    m.compactBytes = bytes;
    m.state = RtBlasState::CompactionQueued;
}

bool RtScene::queueCompaction(u32 index, u64 bytes, u64 expectedVersion) {
    const auto& m = meshes_.at(index);
    // A completed query can arrive after a new update was requested but
    // before that update publishes a new version. It is obsolete already.
    if (m.version != expectedVersion || !m.requested || m.forceRebuild ||
        m.topologyRevision != m.requestedTopology || m.vertexRevision != m.requestedVertices)
        return false;
    queueCompaction(index, bytes);
    return true;
}

bool RtScene::markCompacted(u32 index, u64 expected, u64 id, u64 bytes, u64 frame) {
    auto& m = meshes_.at(index);
    if (!m.requested || m.version != expected || m.state != RtBlasState::CompactionQueued)
        return false;
    if (bytes != m.compactBytes) throw std::invalid_argument("RT compact size differs from queued query");
    validateDestination(id, bytes);
    retire(index, frame); // This frame's copy is itself a reader of the old AS.
    m.state = RtBlasState::Compacted;
    m.version = nextVersion_++;
    m.resourceID = id;
    m.bytes = bytes;
    m.compactBytes = 0;
    m.lastReader = frame;
    ++tableGeneration_;
    return true;
}

void RtScene::ensureSlotIdle(const Snapshot& s) const {
    if (s.valid && (!completedFrame_ || *completedFrame_ < s.frame))
        throw std::logic_error("RT snapshot slot still has GPU readers");
}

RtTlasAction RtScene::tlasAction(u32 slot, u32 capacity, u64 frame, u32 rebuildEvery) const {
    const auto& s = slots_.at(slot);
    if (!capacity) return RtTlasAction::None;
    if (!s.valid || s.capacity != capacity || s.tableGeneration != tableGeneration_ ||
        (rebuildEvery && frame >= s.lastBuild && frame - s.lastBuild >= rebuildEvery))
        return RtTlasAction::Build;
    return RtTlasAction::Refit;
}

void RtScene::commitSnapshot(u32 slot, u32 capacity, u64 frame, RtTlasAction action) {
    auto& s = slots_.at(slot);
    ensureSlotIdle(s);
    if (!capacity) {
        if (action != RtTlasAction::None) throw std::logic_error("Empty RT snapshot cannot build/refit");
        clearSnapshot(slot);
        return;
    }
    if (action == RtTlasAction::None || (action == RtTlasAction::Refit &&
        tlasAction(slot, capacity, frame) == RtTlasAction::Build))
        throw std::logic_error("RT snapshot action incompatible with BLAS table/capacity");
    s.versions.resize(meshes_.size());
    s.resources.resize(meshes_.size());
    for (size_t i = 0; i < meshes_.size(); ++i) {
        auto& m = meshes_[i];
        s.versions[i] = m.version;
        s.resources[i] = m.resourceID;
        if (m.resourceID) m.lastReader = std::max(frame, m.lastReader);
    }
    s.valid = true;
    s.capacity = capacity;
    s.frame = frame;
    s.tableGeneration = tableGeneration_;
    if (action == RtTlasAction::Build) s.lastBuild = frame;
}

void RtScene::commitSnapshot(u32 slot, u32 capacity, u64 frame) {
    commitSnapshot(slot, capacity, frame, tlasAction(slot, capacity, frame));
}

void RtScene::clearSnapshot(u32 slot) {
    auto& s = slots_.at(slot);
    ensureSlotIdle(s);
    s.valid = false;
    s.versions.clear(); // Keep capacity for subsequent snapshots.
    s.resources.clear();
}

void RtScene::completeFrame(u64 frame) {
    if (completedFrame_ && frame < *completedFrame_)
        throw std::invalid_argument("RT completed frame cannot go backwards");
    completedFrame_ = frame;
}

void RtScene::collectRetired(std::vector<RtRetiredBlas>& out) {
    out.clear();
    if (!completedFrame_) return;
    std::erase_if(retired_, [&](const auto& old) {
        if (old.lastReader > *completedFrame_) return false;
        for (const auto& s : slots_)
            if (s.valid && old.mesh < s.resources.size() && s.resources[old.mesh] == old.resourceID)
                return false;
        out.push_back(old);
        return true;
    });
}

std::vector<RtRetiredBlas> RtScene::collectRetired() {
    std::vector<RtRetiredBlas> out;
    collectRetired(out);
    return out;
}

} // namespace phosphor
