#pragma once

// F5 spikes (docs/opt-log.md, "F5 — Spike"): shared helpers on top of the
// bench/soc harness.  A MEASUREMENT TOOL, not engine code (same rules as
// bench/soc/harness.h: objects only through soc::Context, output only through
// ctx.log(), every benchmark sets a negative control).
//
// Benchmarks register with SOC_BENCH("F5-S2", ...) in their own file
// (bench/f5_spike/sN_*.cpp); their MSL lives in bench/f5_spike/shaders and is
// loaded with f5Library().  The runner is bench/soc/soc_bench.cpp (same CLI:
// --list, --only F5-S2, --runs N, --validate, --force-family apple9, --out).

#include "harness.h"

#include <string>

namespace f5 {

using soc::u32;
using soc::u64;

/// Library compiled from bench/f5_spike/shaders/<file> (resolved relative to
/// the harness shader directory, so do not set SOC_SHADER_DIR for f5_spike).
MTL::Library* f5Library(soc::Context& ctx, const std::string& file, bool fastMath = true);

/// Pipelined "frames" with `inFlight` command buffers, one allocator each and
/// a shared event (value n+1 = frame n done), like the engine's frame pacing:
///   FrameRing ring(ctx, 3);
///   for (...) { MTL4::CommandBuffer* cb = ring.begin(); ...encode...; ring.commit(); }
///   ring.drain();
/// begin() blocks until frame (n - inFlight) has completed, then resets that
/// slot's allocator.  extra() gives more command buffers for the current
/// frame (same allocator not allowed: each has its own allocator per slot);
/// commit() commits [primary, extras...] in one call on `queue` (default: the
/// harness queue) and signals the event.
class FrameRing {
public:
    FrameRing(soc::Context& ctx, u32 inFlight = 3, u32 extrasPerFrame = 0);
    MTL4::CommandBuffer* begin();
    /// Extra command buffer i (< extrasPerFrame) of the current frame, begun.
    MTL4::CommandBuffer* extra(u32 i);
    void commit(MTL4::CommandQueue* queue = nullptr);
    /// Wait until every committed frame has completed.
    void drain();
    [[nodiscard]] u64 frame() const { return frame_; } // index of the frame being recorded
    [[nodiscard]] MTL::SharedEvent* event() const { return event_; }
    [[nodiscard]] u32 slot() const { return static_cast<u32>(frame_ % inFlight_); }

private:
    soc::Context& ctx_;
    u32 inFlight_;
    u32 extras_;
    u64 frame_ = 0;
    bool open_ = false;
    std::vector<MTL4::CommandBuffer*> cmds_;      // [slot * (1 + extras) + k]
    std::vector<MTL4::CommandAllocator*> allocs_; // same indexing
    std::vector<bool> extraOpen_;
    MTL::SharedEvent* event_ = nullptr;
};

} // namespace f5
