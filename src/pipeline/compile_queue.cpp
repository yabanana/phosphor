#include "pipeline/compile_queue.h"
#include "core/profile.h"

#include <cstdio>
#include <exception>
#include <utility>

#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif

namespace phosphor::pipe {

namespace {

void nameCurrentThread(u32 index) {
#if defined(__APPLE__) || defined(__linux__)
    char name[32];
#if defined(__APPLE__)
    std::snprintf(name, sizeof(name), "phosphor-compile-%u", index);
#else
    // Linux limit: 15 chars + NUL, so a shorter prefix.
    std::snprintf(name, sizeof(name), "phosphor-cmp-%u", index);
#endif
#endif
#if defined(__APPLE__)
    pthread_setname_np(name); // macOS: names the calling thread only
    PH_THREAD_NAME(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
    PH_THREAD_NAME(name);
#else
    (void)index;
#endif
}

} // namespace

CompileQueue::CompileQueue(u32 workers, ThreadInit threadInit) : threadInit_(std::move(threadInit)) {
    if (workers == 0) workers = 1;
    threads_.reserve(workers);
    for (u32 i = 0; i < workers; ++i) threads_.emplace_back([this, i] { workerMain(i); });
}

CompileQueue::~CompileQueue() {
    std::vector<Item> dropped; // destroyed outside the lock
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
        for (auto& q : queues_) {
            for (auto& item : q) dropped.push_back(std::move(item));
            q.clear(); // queued jobs never run
        }
    }
    workCv_.notify_all();
    idleCv_.notify_all();
    // Cancel queued captures/promises before waiting for in-flight work.
    // Destroy outside mutex_: cancellation can release user-owned resources.
    dropped.clear();
    for (auto& t : threads_) t.join();
}

void CompileQueue::submit(CompilePriority priority, u32 generation, Job job) {
    if (!job) return;
    {
        std::lock_guard lock(mutex_);
        if (stop_) return; // destructor running: a running job submitting more work
        queues_[static_cast<u32>(priority)].push_back(Item{generation, std::move(job)});
    }
    workCv_.notify_one();
}

u32 CompileQueue::cancelBefore(u32 generation) {
    std::vector<Item> dropped; // destroyed outside the lock
    bool              idle = false;
    {
        std::lock_guard lock(mutex_);
        for (auto& q : queues_) {
            for (auto it = q.begin(); it != q.end();) {
                if (it->generation < generation) {
                    dropped.push_back(std::move(*it));
                    it = q.erase(it);
                } else {
                    ++it;
                }
            }
        }
        idle = running_ == 0 && queues_[0].empty() && queues_[1].empty() && queues_[2].empty();
    }
    if (idle && !dropped.empty()) idleCv_.notify_all();
    return static_cast<u32>(dropped.size());
}

void CompileQueue::waitIdle() {
    std::unique_lock lock(mutex_);
    idleCv_.wait(lock, [this] {
        return running_ == 0 && queues_[0].empty() && queues_[1].empty() && queues_[2].empty();
    });
}

size_t CompileQueue::outstanding() const {
    std::lock_guard lock(mutex_);
    return queues_[0].size() + queues_[1].size() + queues_[2].size() + running_;
}

u64 CompileQueue::completed() const {
    std::lock_guard lock(mutex_);
    return completed_;
}

void CompileQueue::workerMain(u32 id) {
    nameCurrentThread(id);
    if (threadInit_) threadInit_(id);

    std::unique_lock lock(mutex_);
    for (;;) {
        workCv_.wait(lock, [this] {
            return stop_ || !queues_[0].empty() || !queues_[1].empty() || !queues_[2].empty();
        });
        if (stop_) return;

        Job job;
        for (auto& q : queues_) {
            if (!q.empty()) {
                job = std::move(q.front().job);
                q.pop_front();
                break;
            }
        }
        ++running_;
        lock.unlock();

        try {
            PH_ZONE("Compile job");
            job();
        } catch (...) {
            // A throwing job must never kill the worker; the exception is dropped.
        }
        job = nullptr; // destroy captures before the job counts as finished

        lock.lock();
        --running_;
        ++completed_;
        if (running_ == 0 && queues_[0].empty() && queues_[1].empty() && queues_[2].empty()) {
            idleCv_.notify_all();
        }
    }
}

} // namespace phosphor::pipe
