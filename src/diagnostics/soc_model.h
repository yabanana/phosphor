#pragma once

#include "core/types.h"
#include "diagnostics/soc_results.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace phosphor::soc {

// ---------------------------------------------------------------------------
// SocCostModel (OPT-0.3) -- roofline-style lower-bound cost model built from
// the measured characterisation results (soc_results.h).
//
// Units: TFLOPS / Top/s for compute rates, GB/s (1e9 B/s) for bandwidths,
// ns / us for latencies, MiB for the SLC estimate.  FLOPs are counted as in
// B-01: a fused multiply-add is 2 FLOP, add / mul are 1.  A field absent from
// the results is NaN; test it with has().
//
// predictPass() returns a LOWER BOUND: max(memory roofs, ALU time) with no
// fixed overheads (dispatch, barrier, commit...), so it can only be exceeded
// by a real measurement (which includes them).  Rates that were not measured
// contribute nothing (never an invented number), which can only lower the
// bound.
// ---------------------------------------------------------------------------

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// One extracted number, with its provenance.
struct Field {
    double      value = kNaN;
    double      runCv = kNaN;  // CV between runs (NaN if unknown)
    std::string unit;
    std::string source;        // "B-01/f32.fma.indep"

    [[nodiscard]] bool has() const { return !std::isnan(value); }
};

/// One B-14 `store.<fmt>.<res>` / `load.<fmt>.<res>` metric.
struct TbdrEntry {
    std::string kind;    // "store" or "load"
    std::string format;  // "rgba8", "rgba16f", ...
    std::string res;     // "1080p", ...
    double      gbps  = 0;
    double      runCv = kNaN;
};

struct MachineInfo {
    std::string chip, slug, gpuFamily, os, powerSource;
    u32         gpuCores = 0;
    double      topPStateMHz = kNaN; // highest GPU P-state (pmgr table)
};

/// Work of one timed unit (GPU-side totals), input of predictPass().
struct PassWork {
    double flops           = 0; // FMA = 2
    double transcendentals = 0;
    double divides         = 0;
    double intOps          = 0;
    double samples         = 0; // texture samples (informational: no rate in the roofs)
    double dramBytes       = 0;
    double onchipBytes     = 0;
    u32    dispatches      = 0; // informational: overheads are not in the bound
    u32    draws           = 0;
    u32    barriers        = 0;
};

struct PassPrediction {
    double      ms = 0;            // lower bound = max(dramMs, onchipMs, aluMs)
    double      dramMs = 0, onchipMs = 0, aluMs = 0;
    std::string bound = "none";    // "dram" | "onchip" | "alu" | "none" (no work or no rates)
    double      arithmeticIntensity = kNaN; // flops / dramBytes (NaN without DRAM traffic)
};

class SocCostModel {
public:
    MachineInfo machine;

    // ALU (B-01, B-03)
    Field f32Fma, f16Fma;                          // TFLOPS
    Field f32Add, i32Add, i32Mul;                  // Top/s
    Field f32Transcendental, f32Div;               // Top/s
    // Memory (B-08)
    Field dramBw, onchipBw;                        // GB/s
    Field latencyL1, latencyDram;                  // ns
    Field slcSizeMiB;                              // MiB
    Field writeBwDram, copyBwDram;                 // GB/s
    // TBDR (B-14)
    Field emptyPassUs;                             // us
    std::vector<TbdrEntry> tbdr;
    // Overheads (B-17, B-18, B-28), us
    Field dispatchEmptyUs, dispatchIndirectUs;
    Field barrierEncoderUs, barrierQueueUs;
    Field commitGpuUs, commitToCpuUs;
    // Ray tracing (B-20), Grays/s
    Field raysCoherent, raysIncoherent;
    // ML (B-22), Top/s
    Field gemmF16, gemmBf16, gemmI8, gemmSimdF16;

    /// Extract every field present in `r` (absent -> NaN).
    [[nodiscard]] static SocCostModel fromResults(const Results& r);

    [[nodiscard]] bool hasFp32() const { return f32Fma.has(); }
    [[nodiscard]] bool hasDramBw() const { return dramBw.has(); }
    [[nodiscard]] bool hasOnchipBw() const { return onchipBw.has(); }
    /// FP32 roof and DRAM roof both known: predictions and ridgePoint() are meaningful.
    [[nodiscard]] bool hasRoofs() const { return hasFp32() && hasDramBw(); }

    /// Lower bound of a pass (see file comment).  Compute time is the sum of
    /// flops / FMA rate + transcendentals / rate + divides / rate + intOps /
    /// (fastest measured integer rate), because they share the ALUs.
    [[nodiscard]] PassPrediction predictPass(const PassWork& w) const;

    /// FLOP/byte where the FP32 roof meets the DRAM roof (NaN if unknown).
    [[nodiscard]] double ridgePoint() const;
    /// Same for the on-chip bandwidth roof.
    [[nodiscard]] double ridgePointOnchip() const;
};

[[nodiscard]] std::string toJson(const SocCostModel& m);
/// false (and *error) on malformed JSON or another schema.
[[nodiscard]] bool fromJson(const std::string& text, SocCostModel& out, std::string* error = nullptr);

// ---------------------------------------------------------------------------
// Shader op counts (tools/air_ops.py -> bench/results/shader_ops.json) and
// the per-invocation cost used to turn shaders + work into a PassWork.
// ---------------------------------------------------------------------------

struct OpCounts {
    double flops = 0, transcendentals = 0, divides = 0, intOps = 0, samples = 0, loads = 0, stores = 0;
    OpCounts& operator+=(const OpCounts& o);
    [[nodiscard]] OpCounts scaled(double k) const;
};

struct ShaderOps {
    std::string name;
    std::string kind;              // "vertex" | "fragment" | "kernel" | ""
    OpCounts    statics;           // every static instruction once (loop bodies once)
    std::vector<OpCounts> loops;   // per-iteration cost of each detected loop
    // Cheapest CFG path (one side of every branch; tools/air_ops.py
    // static_min / per_iteration_min): what a lower bound may count.
    bool        hasMin = false;
    OpCounts    staticsMin;
    std::vector<OpCounts> loopsMin;
};

/// Ops per invocation: statics + (trips - 1) * sum(loop bodies), clamped at 0.
/// Only outermost loops are listed (tools/air_ops.py); nested loops stay inside
/// the outer body and their trip counts are not modelled.
/// lowerBound = true: the cheapest-path counts (if the JSON has them), the
/// only ones a lower bound may use; false: every block (an estimate).
[[nodiscard]] OpCounts opsPerInvocation(const ShaderOps& s, double loopTrips, bool lowerBound = false);

/// Parse the JSON of tools/air_ops.py: {"functions": {name: {kind, static, loops:[{per_iteration}]}}}.
[[nodiscard]] bool parseShaderOps(const std::string& text, std::vector<ShaderOps>& out, std::string* error = nullptr);

/// Exact name match, else the longest function name that is a prefix of `shader`.
[[nodiscard]] const ShaderOps* findShader(const std::vector<ShaderOps>& all, const std::string& shader);

} // namespace phosphor::soc
