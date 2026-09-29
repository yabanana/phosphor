#pragma once

#include "core/types.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace phosphor::pipe {

// ---------------------------------------------------------------------------
// CompileQueue -- dedicated pipeline-compilation threads (F3.1).
//
// Apple's guidance for MTL4Compiler: use maximumConcurrentCompilationTaskCount
// threads whose QoS is LOWER than the render thread's (the compiler inherits
// the caller's QoS).  The queue is portable: the backend passes a
// `threadInit` callback that each worker runs first (on Apple: name the
// thread and set QOS_CLASS_UTILITY or, as a negative control,
// USER_INTERACTIVE).
//
// Jobs run in priority order (Urgent < Specialize < Prewarm), FIFO within a
// priority.  Each job carries a generation: cancelBefore(g) drops every
// queued job of an older generation (a hot reload supersedes them); a job
// already running finishes normally (its result is discarded by the
// registry).  Submission allocates (std::function, deque node): it happens
// on pipeline requests, never in the steady-state frame loop.
// ---------------------------------------------------------------------------

enum class CompilePriority : u8 {
    Urgent     = 0, // something is waiting for it (fallback for a new request)
    Specialize = 1, // final object replacing a fallback
    Prewarm    = 2, // speculative (whole variant table, harvest)
};

class CompileQueue {
public:
    using Job        = std::function<void()>;
    using ThreadInit = std::function<void(u32 worker)>;

    /// Start `workers` threads (>= 1).  `threadInit(i)` runs first on worker i.
    explicit CompileQueue(u32 workers, ThreadInit threadInit = {});
    /// Drops the jobs not started yet, waits for the running ones, joins.
    ~CompileQueue();

    CompileQueue(const CompileQueue&) = delete;
    CompileQueue& operator=(const CompileQueue&) = delete;

    void submit(CompilePriority priority, u32 generation, Job job);
    /// Drop queued jobs whose generation is < `generation`; returns how many.
    u32 cancelBefore(u32 generation);
    /// Block until an instant with no job queued and none running is observed
    /// (jobs submitted concurrently by other threads afterwards are not
    /// waited for; a job submitting a job keeps the queue busy, so it is).
    /// A job must NOT call waitIdle (it would wait for itself).  Jobs may
    /// throw: the exception is swallowed and the worker keeps running.
    void waitIdle();

    [[nodiscard]] u32    workerCount() const { return static_cast<u32>(threads_.size()); }
    /// Jobs queued (not started) + running.
    [[nodiscard]] size_t outstanding() const;
    /// Jobs completed since construction.
    [[nodiscard]] u64    completed() const;

private:
    struct Item {
        u32 generation = 0;
        Job job;
    };
    static constexpr u32 PRIORITY_COUNT = 3;

    void workerMain(u32 id);

    std::vector<std::thread> threads_;
    ThreadInit               threadInit_;

    mutable std::mutex      mutex_;
    std::condition_variable workCv_;
    std::condition_variable idleCv_;
    std::deque<Item>        queues_[PRIORITY_COUNT];
    u32                     running_   = 0;
    u64                     completed_ = 0;
    bool                    stop_      = false;
};

} // namespace phosphor::pipe
