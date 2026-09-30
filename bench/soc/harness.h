#pragma once

// OPT-0.1 SoC characterisation suite -- shared harness (a MEASUREMENT TOOL,
// not engine code: it creates resources with the device and compiles MSL at
// run time, both forbidden in the engine).
//
// Measurement protocol (docs/opt-log.md, "OPT-0 — Spike di
// caratterizzazione", point 1):
//   * GPU time = Precise timestamp after the work - timestamp of a 1-thread
//     "anchor" dispatch in the same encoder (ComputeTimer), or anchor
//     compute encoder -> your encoders -> tail compute encoder
//     (CommandTimer, for render/blit/mixed work);
//   * keep single measured intervals between ~0.1 and ~2 ms (longer ones are
//     interrupted by other GPU clients; shorter ones hit the 41.7 ns tick);
//   * warmUp() once, then keep the GPU busy between groups (keepWarm()):
//     with idle gaps >= 16 ms the GPU drops to a low P-state (x4.8 slower);
//   * a metric = median of >= 15 repetitions (Context::measure), min and
//     p10/p90 kept; 3 runs -> CV between runs (the suite does this);
//   * every kernel STORES a result the CPU checks (the compiler deletes work
//     whose result is unused), inputs are random (zeros compress);
//   * every benchmark sets a negative control (a variant that must scale or
//     get worse) and its detail.
//
// Rules for benchmark files (bench/soc/bNN_*.cpp):
//   * register with SOC_BENCH(id, name, title, function);
//   * create GPU objects only through Context (buffer/texture/pipelines are
//     made resident and released when the benchmark ends);
//   * honour ctx.quick() (the whole suite must finish in <= 2 minutes) and
//     ctx.apple10() (false on Apple9 or with --force-family apple9: take the
//     fallback path and say so in the notes);
//   * print nothing on stdout/stderr except through ctx.log() (--validate
//     counts every other line as a validation message);
//   * never call device->newBuffer/newTexture/newLibrary directly.

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "diagnostics/soc_results.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace soc {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using phosphor::soc::Control;
using phosphor::soc::Status;
using Stats = phosphor::soc::Stats;

struct Options {
    bool        quick       = false;
    bool        forceApple9 = false;
    bool        window      = false; // benchmarks that open a window (B-26) run only with --window
    double      soakMinutes = 0;     // B-27 soak, 0 = skip
    u32         repetitions = 15;    // default repetitions per measurement (--quick: 5)
};

/// GPU clock state from IOReport ("GPU Stats"/"GPU Performance States" and
/// the "GPU Energy" channel), no root needed.  begin()/end() bracket a window.
class GpuState {
public:
    GpuState();
    ~GpuState();
    GpuState(const GpuState&) = delete;
    GpuState& operator=(const GpuState&) = delete;

    [[nodiscard]] bool available() const { return sub_ != nullptr; }
    void begin();
    [[nodiscard]] phosphor::soc::GpuWindow end();
    /// MHz of P1..Pn (ioreg pmgr "voltage-states9"; empty if not found).
    [[nodiscard]] const std::vector<double>& pstateMHz() const { return mhz_; }

private:
    void* sub_    = nullptr; // IOReportSubscriptionRef
    void* subbed_ = nullptr; // CFMutableDictionaryRef
    void* start_  = nullptr; // CFDictionaryRef
    double startTime_ = 0;
    std::vector<double> mhz_;
};

class Context;

/// One compute encoder: anchor dispatch + Precise timestamp, then your
/// dispatches; lap() writes a Precise timestamp after them and a
/// Dispatch->Dispatch barrier so the next dispatch starts after it.
///   ComputeTimer t(ctx);
///   auto* ce = t.begin();
///   ... dispatch A ...; t.lap();
///   ... dispatch B ...; t.lap();
///   std::vector<double> ms = t.finish();   // {A ms, B ms}
class ComputeTimer {
public:
    explicit ComputeTimer(Context& ctx) : ctx_(ctx) {}
    MTL4::ComputeCommandEncoder* begin();
    void lap();
    /// Ends the encoder, commits, waits; lap durations in ms.
    std::vector<double> finish();

private:
    Context& ctx_;
    MTL4::ComputeCommandEncoder* enc_ = nullptr;
    u32 laps_ = 0;
};

/// Any encoders: an anchor compute encoder, then yours (on cmd()), then a
/// tail compute encoder that waits for every stage of the queue before its
/// timestamp.  Measures the whole span; the fixed cost of the empty span is
/// Context::emptySpanMs() (~2 us, subtract it for tiny workloads).
class CommandTimer {
public:
    explicit CommandTimer(Context& ctx) : ctx_(ctx) {}
    MTL4::CommandBuffer* begin();
    [[nodiscard]] MTL4::CommandBuffer* cmd() const { return cmd_; }
    /// Tail encoder, commit, wait; span in ms.
    double finish();

private:
    Context& ctx_;
    MTL4::CommandBuffer* cmd_ = nullptr;
};

class Context {
public:
    explicit Context(const Options& options);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] const Options& options() const { return options_; }
    [[nodiscard]] bool quick() const { return options_.quick; }
    [[nodiscard]] u32 reps() const { return options_.repetitions; }
    /// Apple10 features allowed (false on Apple9 or with --force-family apple9).
    [[nodiscard]] bool apple10() const;
    [[nodiscard]] MTL::Device* device() const { return device_; }
    [[nodiscard]] MTL4::CommandQueue* queue() const { return queue_; }
    [[nodiscard]] MTL4::Compiler* compiler() const { return compiler_; }
    [[nodiscard]] MTL4::CounterHeap* counterHeap() const { return heap_; }
    [[nodiscard]] GpuState& gpuState() { return gpu_; }
    [[nodiscard]] double tickNs() const { return tickNs_; }

    // --- Shaders ------------------------------------------------------------
    /// Library compiled from bench/soc/shaders/<file> (MSL 4.0), cached for
    /// the whole process.  Aborts the benchmark (throws) on a compile error.
    /// fastMath = false: MathModeSafe (no reassociation: use it when the op
    /// count of a chain must survive the compiler).
    MTL::Library* library(const std::string& file, bool fastMath = true);
    /// Compute pipeline for `function` of `library`; `constants` optional
    /// (function constants by index).  Cached per (library, name, constants).
    MTL::ComputePipelineState* compute(MTL::Library* library, const std::string& function,
                                       const MTL::FunctionConstantValues* constants = nullptr);
    /// Render pipeline built by the caller's descriptor (the harness only
    /// compiles it and releases it with the benchmark).
    MTL::RenderPipelineState* render(const MTL4::RenderPipelineDescriptor* desc);
    /// Function descriptor helper for pipeline descriptors (released with the
    /// benchmark).
    MTL4::LibraryFunctionDescriptor* function(MTL::Library* library, const std::string& name);

    // --- Resources (resident, released when the benchmark ends) -------------
    MTL::Buffer* buffer(size_t bytes, MTL::ResourceOptions options = MTL::ResourceStorageModeShared);
    /// Shared buffer filled with xorshift64 bytes from `seed` (incompressible).
    MTL::Buffer* randomBuffer(size_t bytes, u64 seed = 0x9E3779B97F4A7C15ull);
    MTL::Texture* texture(const MTL::TextureDescriptor* desc);
    MTL::Heap* heap(const MTL::HeapDescriptor* desc);
    /// Any other resource/allocation the benchmark made itself (e.g. from a
    /// heap, an acceleration structure): made resident and released at the end.
    void adopt(MTL::Allocation* allocation);
    /// Any other object to release at the end of the benchmark.
    void keep(NS::Object* object);
    /// Re-commit the residency set (after adopt()/buffer() if the benchmark
    /// wants to encode before the next measure; buffer()/texture() do it).
    void commitResidency();

    // --- Recording ------------------------------------------------------------
    /// Reset the allocator and begin the (single, reused) command buffer.
    MTL4::CommandBuffer* beginCommands();
    /// End, commit, wait (60 s timeout).  Returns the feedback GPU time
    /// (GPUEndTime - GPUStartTime) in ms.  Throws on a GPU error.
    double submit();
    /// End and commit on `queue` (e.g. a second queue) without waiting; wait
    /// with waitIdle().
    void submitAsync(MTL4::CommandQueue* queue, MTL4::CommandBuffer* cmd);
    void waitIdle();
    /// The shared argument table (8 buffer bindings, 8 textures, 4 samplers).
    [[nodiscard]] MTL4::ArgumentTable* table() const { return table_; }
    /// Extra command buffer / allocator / queue for benchmarks that need more
    /// than one (released at the end of the benchmark).
    MTL4::CommandBuffer* newCommandBuffer();
    MTL4::CommandAllocator* newAllocator();
    MTL4::CommandQueue* newQueue();
    /// Dispatch the 1-thread anchor kernel (used by the timers).
    void anchorDispatch(MTL4::ComputeCommandEncoder* enc);
    /// Timestamps [first, first+count) of the counter heap, raw ticks.
    std::vector<u64> readTimestamps(u32 first, u32 count);
    [[nodiscard]] double ticksToMs(u64 a, u64 b) const;
    /// Cost of an empty CommandTimer span (calibrated once, ms).
    [[nodiscard]] double emptySpanMs();

    // --- Measurement ------------------------------------------------------------
    /// Continuous GPU load until the top P-state holds >= 95% of busy time
    /// (IOReport) or `maxSeconds`; returns seconds taken (negative: not reached).
    double warmUp(double maxSeconds = 10.0);
    /// Keep the GPU busy for ~`ms` (call between groups of measurements).
    void keepWarm(double ms = 50.0);
    /// Run `once` (returns one sample, e.g. ms) `reps` times (0: options.repetitions).
    Stats measure(const std::function<double()>& once, u32 reps = 0);

    // --- Output -----------------------------------------------------------------
    /// Progress line on stderr, prefixed "[soc]" (never counted by --validate).
    void log(const char* fmt, ...) __attribute__((format(printf, 2, 3)));

    // Internal (suite runner).
    void beginBenchmark();
    void endBenchmark();

private:
    struct Impl;
    Options options_;
    MTL::Device* device_ = nullptr;
    MTL4::CommandQueue* queue_ = nullptr;
    MTL4::Compiler* compiler_ = nullptr;
    MTL4::CommandAllocator* allocator_ = nullptr;
    MTL4::CommandBuffer* cmd_ = nullptr;
    MTL::SharedEvent* event_ = nullptr;
    MTL::ResidencySet* residency_ = nullptr;
    MTL4::CounterHeap* heap_ = nullptr;
    MTL4::ArgumentTable* table_ = nullptr;
    MTL::ComputePipelineState* anchor_ = nullptr;
    MTL::ComputePipelineState* busy_ = nullptr;
    MTL::Buffer* scratch_ = nullptr;
    MTL::Buffer* busyParams_ = nullptr;
    u64 eventValue_ = 0;
    double tickNs_ = 41.667;
    double emptySpanMs_ = -1;
    GpuState gpu_;
    std::unique_ptr<Impl> impl_;
    friend class ComputeTimer;
    friend class CommandTimer;
};

/// What a benchmark reports.  Metric names are unique per benchmark and
/// encode their parameters ("stream_bw.ws_64MiB"); params repeat them.
class Report {
public:
    explicit Report(phosphor::soc::Benchmark& b) : b_(b) {}
    /// Add a metric from the stats of its repetitions (value = median).
    phosphor::soc::Metric& metric(const std::string& name, const std::string& unit, const Stats& stats,
                                  std::map<std::string, double> params = {}, bool higherIsBetter = true);
    /// Add a metric with a single derived value (e.g. a ratio, a threshold).
    phosphor::soc::Metric& value(const std::string& name, const std::string& unit, double v,
                                 std::map<std::string, double> params = {}, bool higherIsBetter = true);
    void status(Status s, const std::string& why = "");
    void note(const std::string& text);
    void negative(bool pass, const std::string& detail);
    [[nodiscard]] phosphor::soc::Benchmark& raw() { return b_; }

private:
    phosphor::soc::Benchmark& b_;
};

using BenchFn = void (*)(Context&, Report&);

struct BenchInfo {
    const char* id;
    const char* name;
    const char* title;
    BenchFn     fn;
};

bool registerBench(const BenchInfo& info);
const std::vector<BenchInfo>& registry(); // sorted by id

#define SOC_CONCAT2(a, b) a##b
#define SOC_CONCAT(a, b) SOC_CONCAT2(a, b)
#define SOC_BENCH(id, name, title, fn) \
    static const bool SOC_CONCAT(soc_registered_, __LINE__) = ::soc::registerBench({id, name, title, fn})

/// Thrown by the harness (and by benchmarks) to abort one benchmark: it is
/// reported as failed with the message; the suite continues.
struct BenchError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// Machine description for the results file (IOKit, sysctl, IOReport).
phosphor::soc::Machine describeMachine(Context& ctx);
std::string thermalStateName();

/// Helpers.
u64 xorshift64(u64& state);
double nowMs(); // mach_absolute_time in ms

} // namespace soc
