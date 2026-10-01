#pragma once

#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// OPT-1.5 / OPT-1.7 -- DRAM bytes, tier budget and SLC working set of a
// compiled graph (O1, S-MEM-1, S-MEM-2).
//
// IMPORTANT: every byte count here is an ESTIMATE OF THE GRAPH, derived from
// the declared accesses and the load/store plan (estimatePassBandwidth): no
// hardware counter reads it (none exists headless, F4.4).  OPT-1 spike 1
// checked the model against the GPU: measured pass time >= bytes / bandwidth
// for 69/69 timed units, i.e. the estimate does not under-count enough to
// beat the DRAM bound.  It is not a measurement of the traffic.
//
// Bandwidth tiers.  Only T2 is measured on this machine (B-08, docs/soc-model.md:
// 569 GB/s).  The others come from Apple's public specifications, were NOT
// measured here, and carry `measured = false` in the struct and in every
// printed line.  GB = 1e9 bytes.
//
// The SLC size (71.4 MiB) is an ESTIMATE from a fit of the B-xx latency/
// bandwidth curve (docs/soc-model.md), not a documented hardware number.
// ---------------------------------------------------------------------------

struct BandwidthTier {
    std::string name;
    double gbPerSecond = 0;   // decimal GB/s
    bool   measured    = false;
    std::string source;       // "measured, B-08" or "external: Apple specifications"
};

/// T2 M5 Max (measured), T0 M5 base, T1 M5 Pro, M3 base (external).
[[nodiscard]] const std::vector<BandwidthTier>& bandwidthTiers();

/// SLC size estimate (71.4 MiB, from a fit: an ESTIMATE).
inline constexpr u64 kSlcEstimateBytes = 74868326ull; // 71.4 * 1024 * 1024

struct BudgetOptions {
    double fps      = 60.0;
    /// Fraction of the DRAM bandwidth the frame may use (1.0 = all of it).
    double fraction = 1.0;
    u64    slcBytes = kSlcEstimateBytes;
};

struct PassBytes {
    u32 pass     = 0;      // RenderGraph pass index
    u32 position = ~0u;    // execution position
    u64 readBytes  = 0;
    u64 writeBytes = 0;
    [[nodiscard]] u64 total() const { return readBytes + writeBytes; }
};

struct TierBudget {
    BandwidthTier tier;
    double fps         = 60.0;
    double fraction    = 1.0;
    double budgetBytes = 0;   // tier bandwidth / fps * fraction
    double share       = 0;   // frameBytes / budgetBytes
    bool   over        = false;
    /// One readable line, always ending with the measured / external flag.
    std::string line;
};

/// A compute pass and the data it touches (traffic read + written).
struct WorkingSet {
    u32  pass       = 0;
    u32  position   = 0;
    u64  readBytes  = 0;
    u64  writeBytes = 0;
    bool aboveSlc   = false; // readBytes + writeBytes > SLC estimate
    [[nodiscard]] u64 total() const { return readBytes + writeBytes; }
};

/// Producer -> consumer of the same resource version, adjacent in execution
/// order on the same queue, in different encoders' DRAM domain (not fused in
/// one render group), whose resource fits in the SLC: the consumer can hit
/// the SLC instead of DRAM (OPT-1 spike 7 measured -7% on the consumer pass).
struct ReuseCandidate {
    u32  producer = 0;       // pass indices
    u32  consumer = 0;
    u32  resource = 0;
    u64  bytes    = 0;       // resource footprint
    bool consumerFits = false; // the consumer's whole working set also fits
};

struct GraphBudget {
    std::vector<PassBytes>     passes;       // live passes, execution order
    u64 totalReadBytes  = 0;
    u64 totalWriteBytes = 0;
    [[nodiscard]] u64 totalBytes() const { return totalReadBytes + totalWriteBytes; }
    std::vector<TierBudget>    tiers;        // one per bandwidthTiers()
    std::vector<WorkingSet>    workingSets;  // compute passes, execution order
    std::vector<ReuseCandidate> reuse;
};

/// Budget arithmetic of one tier: bytes per frame the tier can move at `fps`
/// times `fraction`, and the share of it `frameBytes` takes.
[[nodiscard]] TierBudget makeTierBudget(const BandwidthTier& tier, u64 frameBytes, double fps, double fraction);

[[nodiscard]] GraphBudget analyzeBudget(const RenderGraph& graph, const CompiledGraph& compiled,
                                        const BudgetOptions& options = {});

/// Printable lines (for logs and the dump): the frame total, one line per
/// tier, the working sets above the SLC estimate and the reuse candidates.
[[nodiscard]] std::vector<std::string> budgetSummaryLines(const RenderGraph& graph, const GraphBudget& budget,
                                                          const BudgetOptions& options = {});

} // namespace phosphor::rg
