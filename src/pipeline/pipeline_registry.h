#pragma once

#include "pipeline/pipeline_desc.h"
#include "pipeline/pipeline_key.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace phosphor::pipe {

// ---------------------------------------------------------------------------
// PipelineRegistry -- portable bookkeeping of the pipeline cache (F3.2, F3.5).
//
// Threading model (the backend relies on it):
//   * add / find / get / drain / beginGeneration run on the RENDER thread only.
//   * post() may be called from any thread (compile workers).  Results become
//     visible only in the next drain(), called at the start of a frame, so a
//     frame never sees an object change while it is being encoded (encoding
//     threads only call get()).
//   * Objects are opaque (the backend's retained PSOs).  Every object the
//     registry drops -- a replaced fallback, a stale result, the previous
//     generation after a reload -- goes to the release callback, which the
//     Metal backend maps to MetalContext::deferRelease (frames in flight).
//
// Per entry: an optional FALLBACK object (the flexible pipeline specialised
// to the entry's output state, usable at once) and a FINAL object (archive hit
// or full compile).  get() returns final if present, else fallback, else null.
//
// Generations (hot reload, F3.6): beginGeneration() starts generation g+1.
// Every entry keeps serving its current objects; completions tagged g+1 are
// STAGED.  When every entry has a staged final object, drain() swaps all of
// them at once and releases the old ones.  If any g+1 completion failed, the
// whole generation is abandoned (staged objects released, old ones kept,
// reloadFailures++).  Completions tagged with an older generation than the
// current one are released and ignored.  An entry added while a reload is
// pending belongs to the pending generation: its completion is applied at
// once (it has nothing older to serve) and counts as staged.
// ---------------------------------------------------------------------------

using PipelineHandle = u32;
constexpr PipelineHandle INVALID_PIPELINE = ~0u;

enum class PipelineState : u8 {
    Pending,   // requested, nothing usable yet
    Fallback,  // serving the fallback object, final still compiling
    Ready,     // final object available
    Failed,    // final compile failed (fallback, if any, keeps serving)
};

/// How a FINAL object was obtained with respect to the archive (F3.5).
enum class ArchiveOutcome : u8 {
    NotTried,     // fallback results, or lookups skipped (--pipeline-sync, reload)
    Hit,          // served by the MTL4Archive: no compilation
    Miss,         // archive open, descriptor not found (or incompatible entry)
    Unavailable,  // no usable archive (absent, --no-pipeline-archive, rejected at open)
};

struct Completion {
    PipelineHandle handle     = INVALID_PIPELINE;
    u32            generation = 0;
    void*          object     = nullptr; // null = failure
    bool           fallback   = false;   // flexible specialisation (temporary)
    ArchiveOutcome archive    = ArchiveOutcome::NotTried;
    u32            compilerCalls = 0;    // MTL4Compiler invocations spent on this result
    float          compileMs  = 0.0f;    // wall time inside those invocations
};

struct PipelineStats {
    u32    requests           = 0; // entries added
    u32    archiveHits        = 0;
    u32    archiveMisses      = 0;
    u32    archiveUnavailable = 0;
    u32    compilerCalls      = 0; // every compiler invocation (full, flexible, specialisation)
    double compileMs          = 0.0;
    float  compileMsMax       = 0.0f;
    u32    fallbacksServed    = 0; // entries that served a fallback before their final object
    u64    fallbackDraws      = 0; // encodes that used a fallback object (noteFallbackUse)
    u32    renderThreadCompiles  = 0; // compiles the render thread waited for after startup
    double renderThreadCompileMs = 0.0;
    u32    failures           = 0; // failed completions (any generation)
    u32    swaps              = 0; // times an entry's served object changed
    u32    reloads            = 0; // generations committed after startup
    u32    reloadFailures     = 0; // generations abandoned

    /// Share of final results NOT served by the archive:
    /// (misses + unavailable) / (hits + misses + unavailable); 0 with no results.
    [[nodiscard]] float archiveMissRate() const;
};

/// "PIPELINES requests N | archive hits H, misses M, unavailable U (miss rate X%) |
///  compiler calls C (T ms, max X ms) | render-thread compiles R (T ms) |
///  fallbacks F (draws D) | failures E | reloads L (failed K)"
[[nodiscard]] std::string formatPipelineStats(const PipelineStats& stats);
/// JSON object with every field (embedded in the benchmark report).
[[nodiscard]] std::string pipelineStatsJson(const PipelineStats& stats);

class PipelineRegistry {
public:
    using ReleaseFn = void (*)(void* context, void* object);

    /// `capacity` entries are reserved up front (no reallocation below it).
    explicit PipelineRegistry(u32 capacity = 256);
    ~PipelineRegistry(); // releases every object still held (and queued completions)

    PipelineRegistry(const PipelineRegistry&) = delete;
    PipelineRegistry& operator=(const PipelineRegistry&) = delete;

    void setReleaser(ReleaseFn release, void* context);

    // --- render thread -------------------------------------------------------
    [[nodiscard]] PipelineHandle find(PipelineKey key) const;
    /// New Pending entry (key must not exist); counts one request.  It belongs
    /// to the current generation, or to the pending one during a reload.
    PipelineHandle add(PipelineKey key, const PipelineDesc& desc);
    [[nodiscard]] u32                 size() const { return static_cast<u32>(entries_.size()); }
    [[nodiscard]] const PipelineDesc& desc(PipelineHandle h) const;
    [[nodiscard]] PipelineKey         key(PipelineHandle h) const;
    [[nodiscard]] PipelineState       state(PipelineHandle h) const;
    /// Final object, else fallback, else nullptr.  Safe from encoding threads
    /// between two drain() calls.
    [[nodiscard]] void*               get(PipelineHandle h) const;
    [[nodiscard]] bool                isFinal(PipelineHandle h) const;

    /// Apply the queued completions (see the class comment).  Returns how many
    /// entries changed the object they serve.
    u32 drain();

    /// Start generation g+1 (hot reload) and return it.
    u32 beginGeneration();
    [[nodiscard]] u32  generation() const { return generation_; }        // served
    [[nodiscard]] bool reloadPending() const { return pendingGeneration_ != generation_; }
    [[nodiscard]] u32  pendingGeneration() const { return pendingGeneration_; }

    /// Count an encode that used a fallback object (stats only).
    void noteFallbackUse() { ++stats_.fallbackDraws; }
    void recordRenderThreadCompile(float ms);

    [[nodiscard]] const PipelineStats& stats() const { return stats_; }

    // --- any thread ------------------------------------------------------------
    void post(const Completion& completion);

private:
    struct Entry {
        PipelineKey  key = 0;
        PipelineDesc desc;
        PipelineState state = PipelineState::Pending;
        u32   generation = 0;       // generation of `final`/`fallback`
        void* fallback   = nullptr;
        void* final      = nullptr;
        void* staged     = nullptr; // pending-generation final object
        bool  stagedDone = false;
        bool  servedFallback = false;
        bool  bornPending    = false; // added while a reload was pending
    };

    void release(void* object) const;
    void applyCurrent(Entry& e, const Completion& c, u32& changed);
    void abandonReload();
    void commitOrAbandonReload(u32& changed);

    std::vector<Entry> entries_;
    std::unordered_map<PipelineKey, PipelineHandle> byKey_;
    u32 generation_        = 0;
    u32 pendingGeneration_ = 0;
    u32 lastIssued_        = 0;      // highest generation number handed out
    bool reloadFailed_     = false;
    PipelineStats stats_;

    ReleaseFn releaseFn_  = nullptr;
    void*     releaseCtx_ = nullptr;

    mutable std::mutex      mutex_;   // guards queue_
    std::vector<Completion> queue_;   // reserved; swapped with drained_ in drain()
    std::vector<Completion> drained_;
};

} // namespace phosphor::pipe
