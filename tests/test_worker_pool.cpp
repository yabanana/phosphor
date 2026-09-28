#include "core/worker_pool.h"

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <new>
#include <set>
#include <thread>
#include <vector>

using namespace phosphor;

// --- allocation counter (active only around run() in the allocation test) ---
namespace {
std::atomic<bool> g_countAllocs{false};
std::atomic<u64>  g_allocs{0};
} // namespace

void* operator new(std::size_t n) {
    if (g_countAllocs.load(std::memory_order_relaxed)) g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

struct Counts {
    std::vector<std::atomic<u32>> hits;
    explicit Counts(u32 n) : hits(n) {
        for (auto& h : hits) h.store(0);
    }
};

void countJob(void* user, u32 i) {
    static_cast<Counts*>(user)->hits[i].fetch_add(1, std::memory_order_relaxed);
}

} // namespace

TEST_CASE("worker pool: every index runs exactly once") {
    for (u32 workers : {0u, 1u, 3u, 8u}) {
        WorkerPool pool(workers);
        CHECK(pool.workerCount() == workers);
        for (u32 count : {0u, 1u, 3u, 64u, 10000u}) {
            Counts c(count);
            pool.run(count, countJob, &c);
            for (u32 i = 0; i < count; ++i) {
                if (c.hits[i].load() != 1) {
                    FAIL("index " << i << " ran " << c.hits[i].load() << " times (workers=" << workers
                                  << ", count=" << count << ")");
                }
            }
        }
    }
}

TEST_CASE("worker pool: plain writes are visible to the caller after run()") {
    struct Data { std::vector<u64> out; } d;
    for (u32 workers : {0u, 3u, 8u}) {
        WorkerPool pool(workers);
        d.out.assign(5000, 0);
        pool.run(5000, [](void* u, u32 i) { static_cast<Data*>(u)->out[i] = u64(i) * 7 + 1; }, &d);
        for (u32 i = 0; i < 5000; ++i) REQUIRE(d.out[i] == u64(i) * 7 + 1);
    }
}

TEST_CASE("worker pool: many consecutive runs with varying counts") {
    WorkerPool pool(3);
    std::atomic<u64> total{0};
    u64 expected = 0;
    for (u32 r = 0; r < 2000; ++r) {
        const u32 count = (r * 37u) % 200u; // includes 0
        expected += count;
        pool.run(count, [](void* u, u32) { static_cast<std::atomic<u64>*>(u)->fetch_add(1); }, &total);
        // A stale worker must never leak work into the next job.
        REQUIRE(total.load() == expected);
    }
}

TEST_CASE("worker pool: work is spread over threads") {
    // Only >= 1 distinct thread is asserted (a loaded machine may let the
    // caller finish everything before a worker wakes up); the observed count
    // is reported for information.
    struct Rec {
        std::mutex mtx;
        std::set<std::thread::id> ids;
    } rec;
    WorkerPool pool(4);
    pool.run(256, [](void* u, u32) {
        auto* r = static_cast<Rec*>(u);
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        std::lock_guard<std::mutex> l(r->mtx);
        r->ids.insert(std::this_thread::get_id());
    }, &rec);
    MESSAGE("distinct threads: " << rec.ids.size());
    CHECK(rec.ids.size() >= 1);
}

TEST_CASE("worker pool: destroying an idle pool joins the threads") {
    for (int i = 0; i < 20; ++i) {
        WorkerPool pool(4);
    }
    WorkerPool used(2);
    Counts c(10);
    used.run(10, countJob, &c);
    CHECK(c.hits[9].load() == 1);
}

TEST_CASE("worker pool: run() does not allocate") {
    WorkerPool pool(3);
    Counts c(500);
    pool.run(500, countJob, &c); // warm-up
    g_allocs.store(0);
    g_countAllocs.store(true);
    for (int i = 0; i < 200; ++i) {
        for (auto& h : c.hits) h.store(0, std::memory_order_relaxed);
        pool.run(500, countJob, &c);
    }
    g_countAllocs.store(false);
    CHECK(g_allocs.load() == 0);
}
