#pragma once

#include "core/types.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace phosphor {

// WorkerPool -- persistent threads that run the pieces of one parallel job
// at a time (F2.5: encoding a render pass on several threads).  run() does
// not allocate: the job is a plain function pointer plus a user pointer.
class WorkerPool {
public:
    using Job = void (*)(void* user, u32 index);

    /// `workers` background threads (0 is valid: run() executes everything
    /// on the caller).  Threads are named "phosphor-worker-<i>".
    explicit WorkerPool(u32 workers);
    ~WorkerPool(); // joins the threads
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    /// Run job(user, i) for every i in [0, count) across the workers and the
    /// calling thread; returns when all have finished.  Each index runs
    /// exactly once.  Not reentrant: one run() at a time, from one thread.
    void run(u32 count, Job job, void* user);

    [[nodiscard]] u32 workerCount() const { return static_cast<u32>(threads_.size()); }

private:
    void workerMain(u32 id);
    /// Claim and execute indices of generation `gen` until exhausted; returns
    /// how many this thread ran.
    u32 participate(u32 gen, u32 count, Job job, void* user);

    std::vector<std::thread> threads_;

    std::mutex              mutex_;
    std::condition_variable workCv_;
    std::condition_variable doneCv_;

    // Guarded by mutex_.
    u32   generation_ = 0;
    u32   count_      = 0;
    Job   job_        = nullptr;
    void* user_       = nullptr;
    bool  stop_       = false;

    // Claim cursor: generation in the high 32 bits, next index in the low 32.
    // Claims are a CAS that checks the generation, so a worker that wakes up
    // late (after run() already returned) can never steal an index of a later
    // job.  (Generation wrap would need 2^32 runs while one worker sleeps.)
    std::atomic<u64> cursor_{0};
    std::atomic<u32> done_{0}; // indices finished in the current job
};

} // namespace phosphor
