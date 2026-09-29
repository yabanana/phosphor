#include "core/worker_pool.h"

#include <cstdio>

#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#elif defined(__linux__)
#include <pthread.h>
#endif

namespace phosphor {

namespace {

void configureCurrentThread(u32 id) {
    char name[32];
#if defined(__APPLE__)
    std::snprintf(name, sizeof(name), "phosphor-worker-%u", id);
    pthread_setname_np(name);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters: "phosphor-worker-" would not fit.
    std::snprintf(name, sizeof(name), "phos-worker-%u", id % 1000);
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
    (void)id;
#endif
}

} // namespace

WorkerPool::WorkerPool(u32 workers) {
    threads_.reserve(workers);
    for (u32 i = 0; i < workers; ++i) {
        threads_.emplace_back([this, i] { workerMain(i); });
    }
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    workCv_.notify_all();
    for (std::thread& t : threads_) t.join();
}

u32 WorkerPool::participate(u32 gen, u32 count, Job job, void* user) {
    u32 ran = 0;
    u64 cur = cursor_.load(std::memory_order_acquire);
    for (;;) {
        if (static_cast<u32>(cur >> 32) != gen) break;
        const u32 index = static_cast<u32>(cur);
        if (index >= count) break;
        if (cursor_.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel,
                                          std::memory_order_acquire)) {
            job(user, index);
            ++ran;
            cur = cursor_.load(std::memory_order_acquire);
        }
    }
    return ran;
}

void WorkerPool::workerMain(u32 id) {
    configureCurrentThread(id);
    u32 seen = 0;
    for (;;) {
        u32   gen, count;
        Job   job;
        void* user;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            workCv_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen  = generation_;
            gen   = generation_;
            count = count_;
            job   = job_;
            user  = user_;
        }
        const u32 ran = participate(gen, count, job, user);
        if (ran > 0 && done_.fetch_add(ran, std::memory_order_acq_rel) + ran == count) {
            // Lock so the caller cannot miss the wake-up between its predicate
            // check and its wait.
            std::lock_guard<std::mutex> lock(mutex_);
            doneCv_.notify_one();
        }
    }
}

void WorkerPool::run(u32 count, Job job, void* user) {
    if (count == 0) return;
    if (threads_.empty()) {
        for (u32 i = 0; i < count; ++i) job(user, i);
        return;
    }

    u32 gen;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        gen    = ++generation_;
        count_ = count;
        job_   = job;
        user_  = user;
        done_.store(0, std::memory_order_relaxed);
        cursor_.store(static_cast<u64>(gen) << 32, std::memory_order_release);
    }
    workCv_.notify_all();

    const u32 ran = participate(gen, count, job, user);
    if (ran > 0) done_.fetch_add(ran, std::memory_order_acq_rel);

    // Spin briefly (jobs are short), then sleep.
    for (u32 spin = 0; spin < 2000; ++spin) {
        if (done_.load(std::memory_order_acquire) == count) return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    doneCv_.wait(lock, [&] { return done_.load(std::memory_order_acquire) == count; });
}

} // namespace phosphor
