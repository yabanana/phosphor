#include "pipeline/compile_queue.h"

#include <doctest/doctest.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace phosphor;
using namespace phosphor::pipe;

namespace {

// One-shot gate: wait() blocks until open().
struct Gate {
    std::mutex              m;
    std::condition_variable cv;
    bool                    opened = false;
    void open() {
        {
            std::lock_guard l(m);
            opened = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock l(m);
        cv.wait(l, [&] { return opened; });
    }
};

// Counts arrivals; waitFor(n) blocks until n threads arrived.
struct Started {
    std::mutex              m;
    std::condition_variable cv;
    u32                     count = 0;
    void arrive() {
        {
            std::lock_guard l(m);
            ++count;
        }
        cv.notify_all();
    }
    void waitFor(u32 n) {
        std::unique_lock l(m);
        cv.wait(l, [&] { return count >= n; });
    }
};

struct Recorder {
    std::mutex       m;
    std::vector<int> order;
    void add(int v) {
        std::lock_guard l(m);
        order.push_back(v);
    }
};

} // namespace

TEST_CASE("compile queue: priority order with one worker") {
    CompileQueue q(1);
    CHECK(q.workerCount() == 1);
    Gate     gate;
    Started  started;
    Recorder rec;
    q.submit(CompilePriority::Urgent, 0, [&] {
        started.arrive();
        gate.wait();
    });
    started.waitFor(1); // worker is now blocked; everything below stays queued
    q.submit(CompilePriority::Prewarm, 0, [&] { rec.add(30); });
    q.submit(CompilePriority::Specialize, 0, [&] { rec.add(20); });
    q.submit(CompilePriority::Urgent, 0, [&] { rec.add(10); });
    q.submit(CompilePriority::Prewarm, 0, [&] { rec.add(31); });
    q.submit(CompilePriority::Urgent, 0, [&] { rec.add(11); });
    q.submit(CompilePriority::Specialize, 0, [&] { rec.add(21); });
    CHECK(q.outstanding() == 7);
    gate.open();
    q.waitIdle();
    CHECK(rec.order == std::vector<int>{10, 11, 20, 21, 30, 31});
}

TEST_CASE("compile queue: FIFO within a priority") {
    CompileQueue q(1);
    Gate         gate;
    Started      started;
    Recorder     rec;
    q.submit(CompilePriority::Prewarm, 0, [&] {
        started.arrive();
        gate.wait();
    });
    started.waitFor(1);
    for (int i = 0; i < 50; ++i) q.submit(CompilePriority::Specialize, 0, [&rec, i] { rec.add(i); });
    gate.open();
    q.waitIdle();
    REQUIRE(rec.order.size() == 50);
    for (int i = 0; i < 50; ++i) CHECK(rec.order[i] == i);
}

TEST_CASE("compile queue: N workers run N jobs concurrently") {
    const u32    n = 4;
    CompileQueue q(n);
    CHECK(q.workerCount() == n);
    Gate             gate;
    Started          started;
    std::atomic<u32> active{0};
    std::atomic<u32> peak{0};
    for (u32 i = 0; i < n; ++i) {
        q.submit(CompilePriority::Urgent, 0, [&] {
            u32 a = active.fetch_add(1) + 1;
            u32 p = peak.load();
            while (a > p && !peak.compare_exchange_weak(p, a)) {}
            started.arrive();
            gate.wait(); // only returns once the test saw all N inside
            active.fetch_sub(1);
        });
    }
    started.waitFor(n); // hangs if fewer than N jobs can run at once
    CHECK(q.outstanding() == n);
    gate.open();
    q.waitIdle();
    CHECK(peak.load() == n);
    CHECK(q.completed() == n);
}

TEST_CASE("compile queue: zero workers is treated as one") {
    CompileQueue q(0);
    CHECK(q.workerCount() == 1);
    std::atomic<int> ran{0};
    q.submit(CompilePriority::Urgent, 0, [&] { ran++; });
    q.waitIdle();
    CHECK(ran.load() == 1);
}

TEST_CASE("compile queue: cancelBefore drops only older queued jobs") {
    CompileQueue q(1);
    Gate         gate;
    Started      started;
    Recorder     rec;
    // Running job of generation 0: must survive the cancel.
    q.submit(CompilePriority::Urgent, 0, [&] {
        started.arrive();
        gate.wait();
        rec.add(-1);
    });
    started.waitFor(1);
    q.submit(CompilePriority::Prewarm, 1, [&] { rec.add(1); });
    q.submit(CompilePriority::Urgent, 2, [&] { rec.add(2); });
    q.submit(CompilePriority::Specialize, 3, [&] { rec.add(3); });
    q.submit(CompilePriority::Urgent, 0, [&] { rec.add(100); });
    CHECK(q.cancelBefore(0) == 0);
    CHECK(q.cancelBefore(2) == 2); // generations 1 and 0
    CHECK(q.outstanding() == 3);   // running + gen 2 + gen 3
    gate.open();
    q.waitIdle();
    CHECK(rec.order == std::vector<int>{-1, 2, 3});
    CHECK(q.completed() == 3);
}

TEST_CASE("compile queue: cancelBefore wakes waitIdle when the queue drains") {
    CompileQueue q(1);
    Gate         gate;
    Started      started;
    q.submit(CompilePriority::Urgent, 5, [&] {
        started.arrive();
        gate.wait();
    });
    started.waitFor(1);
    q.submit(CompilePriority::Prewarm, 1, [] {});
    CHECK(q.cancelBefore(10) == 1); // the queued one; the running one stays
    std::atomic<bool> idle{false};
    std::thread       waiter([&] {
        q.waitIdle();
        idle = true;
    });
    gate.open();
    waiter.join(); // must not hang
    CHECK(idle.load());
    CHECK(q.outstanding() == 0);
    CHECK(q.completed() == 1);
}

TEST_CASE("compile queue: cancelBefore alone makes the queue idle") {
    // Nothing running (worker is not started on the jobs): cancel every queued job of a queue whose
    // only worker is parked in a gate job that already finished.
    CompileQueue q(1);
    Gate         gate;
    Started      started;
    q.submit(CompilePriority::Urgent, 0, [&] {
        started.arrive();
        gate.wait();
    });
    started.waitFor(1);
    gate.open();
    q.waitIdle();
    CHECK(q.cancelBefore(100) == 0);
    CHECK(q.outstanding() == 0);
}

TEST_CASE("compile queue: waitIdle after many jobs") {
    CompileQueue     q(4);
    std::atomic<u32> ran{0};
    const u32        total = 2000;
    for (u32 i = 0; i < total; ++i) {
        q.submit(static_cast<CompilePriority>(i % 3), i / 100, [&] { ran.fetch_add(1); });
    }
    q.waitIdle();
    CHECK(ran.load() == total);
    CHECK(q.completed() == total);
    CHECK(q.outstanding() == 0);
    q.waitIdle(); // idle queue returns immediately
}

TEST_CASE("compile queue: waitIdle under concurrent submitters") {
    CompileQueue             q(3);
    std::atomic<u32>         ran{0};
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&] {
            for (int i = 0; i < 200; ++i) {
                q.submit(CompilePriority::Specialize, 0, [&] { ran.fetch_add(1); });
                if (i % 50 == 0) q.waitIdle();
            }
        });
    }
    for (auto& p : producers) p.join();
    q.waitIdle();
    CHECK(ran.load() == 800);
    CHECK(q.completed() == 800);
}

TEST_CASE("compile queue: counts") {
    CompileQueue q(1);
    Gate         gate;
    Started      started;
    CHECK(q.outstanding() == 0);
    CHECK(q.completed() == 0);
    q.submit(CompilePriority::Urgent, 0, [&] {
        started.arrive();
        gate.wait();
    });
    started.waitFor(1);
    q.submit(CompilePriority::Urgent, 0, [] {});
    q.submit(CompilePriority::Urgent, 0, [] {});
    CHECK(q.outstanding() == 3);
    CHECK(q.completed() == 0);
    gate.open();
    q.waitIdle();
    CHECK(q.outstanding() == 0);
    CHECK(q.completed() == 3);
}

TEST_CASE("compile queue: threadInit runs once per worker on the worker thread") {
    const u32                    n = 4;
    std::mutex                   m;
    std::vector<u32>             indices;
    std::vector<std::thread::id> initThreads;
    CompileQueue                 q(n, [&](u32 i) {
        std::lock_guard l(m);
        indices.push_back(i);
        initThreads.push_back(std::this_thread::get_id());
    });
    // One blocking job per worker: proves every worker passed threadInit and
    // lets us compare thread ids.
    Gate                      gate;
    Started                   started;
    std::mutex                jm;
    std::set<std::thread::id> jobThreads;
    for (u32 i = 0; i < n; ++i) {
        q.submit(CompilePriority::Urgent, 0, [&] {
            {
                std::lock_guard l(jm);
                jobThreads.insert(std::this_thread::get_id());
            }
            started.arrive();
            gate.wait();
        });
    }
    started.waitFor(n);
    gate.open();
    q.waitIdle();
    std::lock_guard l(m);
    CHECK(indices.size() == n);
    CHECK(std::set<u32>(indices.begin(), indices.end()) == std::set<u32>{0, 1, 2, 3});
    std::set<std::thread::id> initSet(initThreads.begin(), initThreads.end());
    CHECK(initSet.size() == n);
    CHECK(initSet == jobThreads);
    CHECK(initSet.count(std::this_thread::get_id()) == 0);
}

TEST_CASE("compile queue: destructor drops queued jobs and does not hang") {
    std::atomic<u32> ran{0};
    Gate             gate;
    Started          started;
    std::thread      releaser;
    {
        CompileQueue q(1);
        q.submit(CompilePriority::Urgent, 0, [&] {
            started.arrive();
            gate.wait();
            ran.fetch_add(1);
        });
        started.waitFor(1);
        for (int i = 0; i < 10; ++i) q.submit(CompilePriority::Urgent, 0, [&] { ran.fetch_add(100); });
        // The destructor must wait for the running job; release it from another thread.
        releaser = std::thread([&] { gate.open(); });
    }
    releaser.join();
    CHECK(ran.load() == 1); // running job finished, queued ones never started
}

TEST_CASE("compile queue: a job may submit another job") {
    CompileQueue     q(1);
    std::atomic<u32> ran{0};
    q.submit(CompilePriority::Urgent, 0, [&] {
        ran.fetch_add(1);
        q.submit(CompilePriority::Prewarm, 0, [&] {
            ran.fetch_add(1);
            q.submit(CompilePriority::Prewarm, 0, [&] { ran.fetch_add(1); });
        });
    });
    q.waitIdle(); // includes the chained jobs: the queue is never idle in between
    CHECK(ran.load() == 3);
    CHECK(q.completed() == 3);
}

TEST_CASE("compile queue: destructor with a job that keeps submitting does not hang") {
    Started started;
    {
        CompileQueue q(1);
        q.submit(CompilePriority::Urgent, 0, [&] {
            started.arrive();
            // Submitted while the destructor may already be running: ignored safely.
            for (int i = 0; i < 100; ++i) q.submit(CompilePriority::Prewarm, 0, [] {});
        });
        started.waitFor(1);
    }
    CHECK(true);
}

TEST_CASE("compile queue: a throwing job does not kill the worker") {
    CompileQueue     q(1);
    std::atomic<u32> ran{0};
    q.submit(CompilePriority::Urgent, 0, [] { throw std::runtime_error("boom"); });
    q.submit(CompilePriority::Urgent, 0, [] { throw 42; });
    q.submit(CompilePriority::Urgent, 0, [&] { ran.fetch_add(1); });
    q.waitIdle();
    CHECK(ran.load() == 1);
    CHECK(q.completed() == 3);
    q.submit(CompilePriority::Urgent, 0, [&] { ran.fetch_add(1); });
    q.waitIdle();
    CHECK(ran.load() == 2);
}
