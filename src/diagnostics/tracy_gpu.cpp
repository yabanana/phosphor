#include "diagnostics/tracy_gpu.h"

// With PHOSPHOR_TRACY the Tracy implementation lives here; without it the
// header's inline no-ops are used and this file is empty.
#ifdef PHOSPHOR_TRACY

#include "rendergraph/timing_plan.h"

#include <tracy/TracyC.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace phosphor {

namespace {

constexpr u8 CONTEXT_GRAPHICS = 0;
constexpr u8 CONTEXT_ASYNC    = 1;
constexpr u8 TRACY_GPU_TYPE_METAL = 6; // tracy::GpuContextType::Metal

} // namespace

// Source-location lifetime: Tracy keeps the pointer given to
// zone_begin_serial and asks for its contents (name, file, line) whenever
// the server first sees it, possibly long after emission.  A location must
// therefore stay valid AND unchanged for the rest of the process.  Every
// configure() appends a new generation (names + locations, never resized
// afterwards) and old generations are kept alive: growth is bounded by one
// small generation (units x ~100 bytes) per graph compile.
struct TracyGpuZones::Impl {
    struct Generation {
        std::vector<std::string>                      names;
        std::vector<___tracy_source_location_data>    locations;
        std::vector<u8>                               contexts; // per unit
    };

    bool                   contextsCreated = false;
    std::deque<Generation> generations;  // deque: elements never move
    Generation*            current = nullptr;
    u16                    query   = 0;  // wraps: zones are emitted begin+end at once
};

TracyGpuZones::TracyGpuZones() : impl_(std::make_unique<Impl>()) {}
TracyGpuZones::~TracyGpuZones() = default;

void TracyGpuZones::configure(const rg::TimingPlan& plan, double tickNs, u64 nowTicks) {
    Impl& s = *impl_;
    if (!s.contextsCreated) {
        static const char graphicsName[] = "MTL4 graphics";
        static const char asyncName[]    = "MTL4 async compute";
        const u8 ids[2] = {CONTEXT_GRAPHICS, CONTEXT_ASYNC};
        const char* names[2] = {graphicsName, asyncName};
        const u16 lens[2] = {sizeof(graphicsName) - 1, sizeof(asyncName) - 1};
        for (u32 i = 0; i < 2; ++i) {
            // Serial variants like the zones: a context emitted on the per-thread queue
            // can reach the server after serial zones queued before the connection,
            // and tracy-capture then dereferences a null context (measured: crash).
            ___tracy_emit_gpu_new_context_serial({static_cast<int64_t>(nowTicks), static_cast<float>(tickNs), ids[i], 0,
                                           TRACY_GPU_TYPE_METAL});
            ___tracy_emit_gpu_context_name_serial({ids[i], names[i], lens[i]});
        }
        s.contextsCreated = true;
    }

    s.generations.emplace_back();
    Impl::Generation& g = s.generations.back();
    const size_t n = plan.units.size();
    g.names.reserve(n);
    g.locations.reserve(n);
    g.contexts.reserve(n);
    for (const rg::TimedUnit& unit : plan.units) {
        g.names.push_back(unit.name);
        g.contexts.push_back(unit.queue == rg::Queue::AsyncCompute ? CONTEXT_ASYNC : CONTEXT_GRAPHICS);
    }
    // Names are stable now (no further push_back), so c_str() pointers stay valid.
    for (const std::string& name : g.names) {
        g.locations.push_back({name.c_str(), "GPU pass", "gpu", 0, 0});
    }
    s.current = &g;
}

void TracyGpuZones::emitFrame(const u64* startTicks, const u64* endTicks, u32 unitCount) {
    Impl& s = *impl_;
    if (!s.current) return;
    const u32 count = unitCount < s.current->locations.size() ? unitCount
                                                              : static_cast<u32>(s.current->locations.size());
    for (u32 i = 0; i < count; ++i) {
        if (startTicks[i] == 0 || endTicks[i] < startTicks[i]) continue;
        const u8  ctx   = s.current->contexts[i];
        const u16 begin = s.query++;
        const u16 end   = s.query++;
        ___tracy_emit_gpu_zone_begin_serial({reinterpret_cast<uint64_t>(&s.current->locations[i]), begin, ctx});
        ___tracy_emit_gpu_time_serial({static_cast<int64_t>(startTicks[i]), begin, ctx});
        ___tracy_emit_gpu_zone_end_serial({end, ctx});
        ___tracy_emit_gpu_time_serial({static_cast<int64_t>(endTicks[i]), end, ctx});
    }
}

} // namespace phosphor

#endif
