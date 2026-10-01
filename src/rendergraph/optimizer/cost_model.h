#pragma once

#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor::soc {
class SocCostModel;
}

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// OPT-1.1 / OPT-1.8 -- cost model of a compiled graph, used to rank plans.
//
// Per timed unit (TimingPlan: a render group or a compute pass):
//   t = max(DRAM bytes / BW, ALU + geometry) + fixed
// with ALU = PassCost::intOps / IMAD rate + flops / FMA rate, geometry =
// triangles / raster rate, fixed = one render pass in a chain or one
// dependent compute pass.  The frame is simulated on the two queues:
//   * units are issued in order on their queue: a unit starts no earlier than
//     the unit before it started and after the end of every earlier unit of
//     its queue whose stages one of its barriers waits for (barriers are
//     stage-wide); time spent beside units still running pays a sharing
//     penalty: 1 - overlapSame of it for the same bound (B-19: two render
//     passes = 0.84 of the sum), 1 - overlapDifferent otherwise (raster
//     geometry beside an ALU compute: ~full overlap, OPT-1 spike 5);
//   * async units run on their own timeline; a cross-queue wait starts the
//     consumer at max(own time, producer end + event latency) (B-19 ~0.07 ms);
//     overlapping async/graphics pairs pay a sharing penalty: as on one
//     queue for the same bound, else the B-19 cross-queue share (0.78 of the
//     sum for bandwidth, 0.98 for ALU).
// J = frame ms + gamma * transient heap GiB + beta * DRAM GiB (memory and
// bytes as secondary goals; raise them for memory/bandwidth-oriented plans).
//
// A model, not a measurement: it ranks candidate plans; a plan is adopted only
// after the measured comparison (tools/graph_select.py).
// ---------------------------------------------------------------------------

struct GraphCostParams {
    // Measured on M5 Max (docs/soc-model.md; OPT-1 spikes in docs/opt-log.md).
    double dramGBs            = 569.0;  // B-08 read
    double imadTops           = 4.06;   // B-01 i32 mul (one IMAD per LCG step, spike 1)
    double fmaTflops          = 15.1;   // B-01 f32 fma
    double trianglesPerSecond = 8.0e9;  // B-16 raster 7.9-9.5 G/s
    double renderPassUs       = 11.3;   // B-14 empty pass in a chain
    double computePassUs      = 7.0;    // OPT-1 spike 1: tiny dependent dispatch
    double eventLatencyMs     = 0.07;   // B-19 cross-queue event
    double overlapSame        = 0.16;   // B-19: two render passes = 0.84 of the sum
    double overlapDifferent   = 0.9;    // OPT-1 spike 5: shadows under an ALU compute
    double crossQueueBwShare  = 0.22;   // B-19: 0.78 of the sum when bandwidth bound
    double crossQueueAluShare = 0.02;   // B-19: 0.98 of the sum when ALU bound
    double gammaMsPerGiB      = 0.1;    // weight of the transient heap in J
    double betaMsPerGiB       = 0.0;    // weight of the DRAM bytes in J (energy proxy; 0 = time only)

    /// Rates from a characterisation (fields absent keep the defaults above).
    [[nodiscard]] static GraphCostParams fromSoc(const soc::SocCostModel& model);
};

enum class Bound : u8 { Fixed, Memory, Alu, Geometry };

struct UnitCost {
    std::string name;
    Queue  queue = Queue::Graphics;
    double bytes = 0;
    double memMs = 0, aluMs = 0, geoMs = 0, fixedMs = 0;
    double ms    = 0;         // standalone: max(mem, alu + geo) + fixed
    double overlapMs = 0;     // time run beside earlier units still running
    Bound  bound = Bound::Fixed;
};

struct GraphCost {
    std::vector<UnitCost> units;  // encode order (TimingPlan units)
    double frameMs      = 0;      // simulated
    double sumMs        = 0;      // sum of standalone unit times
    double dramBytes    = 0;
    double heapBytes    = 0;
    double maxLiveBytes = 0;
    u32    renderPasses = 0;
    u32    memoryless   = 0;
    u32    barriers     = 0;      // barriers encoded (after merging)
    double J            = 0;
};

/// Cost of a successfully compiled graph (aliasing plan needed for the heap).
[[nodiscard]] GraphCost evaluateGraph(const RenderGraph& graph, const CompiledGraph& compiled,
                                      const GraphCostParams& params = {});

} // namespace phosphor::rg
