#include "pipeline/pipeline_registry.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <utility>

namespace phosphor::pipe {

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

float PipelineStats::archiveMissRate() const {
    const u64 total = static_cast<u64>(archiveHits) + archiveMisses + archiveUnavailable;
    if (total == 0) return 0.0f;
    return static_cast<float>(static_cast<double>(archiveMisses + archiveUnavailable) /
                              static_cast<double>(total));
}

std::string formatPipelineStats(const PipelineStats& s) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "PIPELINES requests %u | archive hits %u, misses %u, unavailable %u (miss rate %.1f%%) | "
                  "compiler calls %u (%.1f ms, max %.1f ms) | render-thread compiles %u (%.1f ms) | "
                  "fallbacks %u (draws %llu) | failures %u | reloads %u (failed %u)",
                  s.requests, s.archiveHits, s.archiveMisses, s.archiveUnavailable,
                  static_cast<double>(s.archiveMissRate()) * 100.0, s.compilerCalls, s.compileMs,
                  static_cast<double>(s.compileMsMax), s.renderThreadCompiles, s.renderThreadCompileMs,
                  s.fallbacksServed, static_cast<unsigned long long>(s.fallbackDraws), s.failures,
                  s.reloads, s.reloadFailures);
    return buf;
}

std::string pipelineStatsJson(const PipelineStats& s) {
    char buf[768];
    std::snprintf(buf, sizeof(buf),
                  "{\"requests\":%u,\"archiveHits\":%u,\"archiveMisses\":%u,\"archiveUnavailable\":%u,"
                  "\"archiveMissRate\":%.4f,\"compilerCalls\":%u,\"compileMs\":%.3f,\"compileMsMax\":%.3f,"
                  "\"fallbacksServed\":%u,\"fallbackDraws\":%llu,\"renderThreadCompiles\":%u,"
                  "\"renderThreadCompileMs\":%.3f,\"failures\":%u,\"swaps\":%u,\"reloads\":%u,"
                  "\"reloadFailures\":%u}",
                  s.requests, s.archiveHits, s.archiveMisses, s.archiveUnavailable,
                  static_cast<double>(s.archiveMissRate()), s.compilerCalls, s.compileMs,
                  static_cast<double>(s.compileMsMax), s.fallbacksServed,
                  static_cast<unsigned long long>(s.fallbackDraws), s.renderThreadCompiles,
                  s.renderThreadCompileMs, s.failures, s.swaps, s.reloads, s.reloadFailures);
    return buf;
}

// ---------------------------------------------------------------------------
// PipelineRegistry
// ---------------------------------------------------------------------------

PipelineRegistry::PipelineRegistry(u32 capacity) {
    entries_.reserve(capacity);
    byKey_.reserve(capacity);
    const size_t queueCap = static_cast<size_t>(capacity) * 4 + 16;
    queue_.reserve(queueCap);
    drained_.reserve(queueCap);
}

PipelineRegistry::~PipelineRegistry() {
    for (Entry& e : entries_) {
        release(e.final);
        release(e.fallback);
        release(e.staged);
        e.final = e.fallback = e.staged = nullptr;
    }
    for (const Completion& c : queue_) release(c.object);
    for (const Completion& c : drained_) release(c.object);
}

void PipelineRegistry::setReleaser(ReleaseFn release, void* context) {
    releaseFn_  = release;
    releaseCtx_ = context;
}

void PipelineRegistry::release(void* object) const {
    if (object && releaseFn_) releaseFn_(releaseCtx_, object);
}

PipelineHandle PipelineRegistry::find(PipelineKey key) const {
    const auto it = byKey_.find(key);
    return it == byKey_.end() ? INVALID_PIPELINE : it->second;
}

PipelineHandle PipelineRegistry::add(PipelineKey key, const PipelineDesc& desc) {
    assert(byKey_.find(key) == byKey_.end() && "pipeline key already registered");
    const PipelineHandle h = static_cast<PipelineHandle>(entries_.size());
    Entry e;
    e.key         = key;
    e.desc        = desc;
    e.bornPending = reloadPending();
    e.generation  = e.bornPending ? pendingGeneration_ : generation_;
    entries_.push_back(std::move(e));
    byKey_[key] = h;
    ++stats_.requests;
    return h;
}

const PipelineDesc& PipelineRegistry::desc(PipelineHandle h) const { return entries_[h].desc; }
PipelineKey PipelineRegistry::key(PipelineHandle h) const { return entries_[h].key; }
PipelineState PipelineRegistry::state(PipelineHandle h) const { return entries_[h].state; }

void* PipelineRegistry::get(PipelineHandle h) const {
    if (h >= entries_.size()) return nullptr;
    const Entry& e = entries_[h];
    return e.final ? e.final : e.fallback;
}

bool PipelineRegistry::isFinal(PipelineHandle h) const {
    return h < entries_.size() && entries_[h].final != nullptr;
}

void PipelineRegistry::recordRenderThreadCompile(float ms) {
    ++stats_.renderThreadCompiles;
    stats_.renderThreadCompileMs += static_cast<double>(ms);
}

void PipelineRegistry::post(const Completion& completion) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(completion);
}

void PipelineRegistry::applyCurrent(Entry& e, const Completion& c, u32& changed) {
    void* const before = e.final ? e.final : e.fallback;
    if (c.fallback) {
        if (!c.object) return; // failed fallback: nothing served changes
        if (e.final) {         // final already there: the fallback is obsolete
            release(c.object);
            return;
        }
        if (e.fallback != c.object) release(e.fallback);
        e.fallback = c.object;
        if (!e.servedFallback) {
            e.servedFallback = true;
            ++stats_.fallbacksServed;
        }
        if (e.state != PipelineState::Ready) e.state = PipelineState::Fallback;
    } else {
        if (!c.object) {
            if (!e.final) e.state = PipelineState::Failed;
            if (e.bornPending) reloadFailed_ = true;
            return;
        }
        if (e.final != c.object) release(e.final);
        e.final = c.object;
        if (e.fallback && e.fallback != c.object) release(e.fallback);
        e.fallback = nullptr;
        e.state    = PipelineState::Ready;
    }
    void* const after = e.final ? e.final : e.fallback;
    if (after != before) {
        ++stats_.swaps;
        ++changed;
    }
}

u32 PipelineRegistry::drain() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::swap(queue_, drained_); // drained_ is empty here; both keep their capacity
    }
    u32 changed = 0;
    for (const Completion& c : drained_) {
        // Statistics: every completion, whatever happens to its object.
        stats_.compilerCalls += c.compilerCalls;
        stats_.compileMs += static_cast<double>(c.compileMs);
        stats_.compileMsMax = std::max(stats_.compileMsMax, c.compileMs);
        if (!c.object) ++stats_.failures;
        if (!c.fallback) {
            switch (c.archive) {
                case ArchiveOutcome::Hit:         ++stats_.archiveHits; break;
                case ArchiveOutcome::Miss:        ++stats_.archiveMisses; break;
                case ArchiveOutcome::Unavailable: ++stats_.archiveUnavailable; break;
                case ArchiveOutcome::NotTried:    break;
            }
        }

        if (c.handle >= entries_.size()) {
            release(c.object);
            continue;
        }
        Entry& e = entries_[c.handle];
        if (c.generation == e.generation) {
            applyCurrent(e, c, changed);
        } else if (reloadPending() && c.generation == pendingGeneration_ && !c.fallback) {
            // Staged final object of the pending generation.
            if (!c.object) {
                reloadFailed_ = true;
            } else {
                if (e.staged != c.object) release(e.staged);
                e.staged     = c.object;
                e.stagedDone = true;
            }
        } else {
            release(c.object); // stale generation (or a fallback for a pending one)
        }
    }
    drained_.clear();

    if (reloadPending()) commitOrAbandonReload(changed);
    return changed;
}

u32 PipelineRegistry::beginGeneration() {
    if (reloadPending()) abandonReload(); // superseded, not counted as a failure
    lastIssued_        = std::max(lastIssued_, generation_) + 1;
    pendingGeneration_ = lastIssued_;
    reloadFailed_      = false;
    return pendingGeneration_;
}

void PipelineRegistry::abandonReload() {
    for (Entry& e : entries_) {
        release(e.staged);
        e.staged      = nullptr;
        e.stagedDone  = false;
        e.bornPending = false; // keeps its own generation tag, so its completions still apply
    }
    pendingGeneration_ = generation_;
    reloadFailed_      = false;
}

void PipelineRegistry::commitOrAbandonReload(u32& changed) {
    if (reloadFailed_) {
        abandonReload();
        ++stats_.reloadFailures;
        return;
    }
    for (const Entry& e : entries_) {
        const bool ready = e.bornPending ? e.final != nullptr : e.stagedDone;
        if (!ready) return;
    }
    for (Entry& e : entries_) {
        if (e.bornPending) {
            e.bornPending = false;
            continue;
        }
        release(e.final);
        release(e.fallback);
        e.final      = e.staged;
        e.fallback   = nullptr;
        e.staged     = nullptr;
        e.stagedDone = false;
        e.state      = PipelineState::Ready;
        e.generation = pendingGeneration_;
        ++stats_.swaps;
        ++changed;
    }
    generation_   = pendingGeneration_;
    reloadFailed_ = false;
    ++stats_.reloads;
}

} // namespace phosphor::pipe
