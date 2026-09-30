#pragma once

#include "rendergraph/render_graph.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// OPT-1 graph scenarios -- realistic frame graphs made of synthetic passes
// whose work and DRAM bytes are known, executed on the GPU by the real
// executor (--graph-scenario N).  The engine's own frame graph has 1-2
// passes; OPT-1 decisions (order, fusion, store/memoryless, aliasing,
// rematerialisation, queues) are measured on these graphs.
//
// Data model (shaders/scenario.metal, keep in sync): every texel a synthetic
// pass writes is a deterministic function of its inputs, the pixel and the
// pass seed; channel values are multiples of 1/255 (exactly representable in
// every scenario format; readers convert back to integers), depth values
// are 0.1 + 0.8 * k / 1023 (k recovered by rounding).  Any correct schedule
// -- order, fusion, aliasing, queues, rematerialisation -- produces the same
// image bit for bit; a missing barrier or a wrong alias changes it.
//
// Work: `aluIterations` steps of 4 independent 32-bit LCG chains per pixel or
// thread (8 integer ops per step), folded into the output through a runtime
// zero so the compiler keeps them without changing any value.
// ---------------------------------------------------------------------------

enum class SynthKind : u8 {
    None,       // not a synthetic pass (added by the engine: UI, capture)
    Fullscreen, // raster: one fullscreen triangle (reads, ALU, attachment writes)
    Geometry,   // raster: a grid of `triangles` triangles covering the target,
                // one flat depth per cell (depth write, or test only)
    Compute,    // compute: one thread per texel of the first storage output
};

/// A texture read by a synthetic pass.  The reader visits the texel
/// footprint of its output pixel (in/out size ratio per axis, at most 4x4):
/// a downsample reads its whole input once.
struct SynthInput {
    TextureRef texture;
    bool depth = false;      // depth texture (depth2d)
    /// OPT-1.2: the input is not read but recomputed from `rematDepth` (an
    /// index into SynthPass::inputs, a depth input of the same size) with
    /// the producer's value function (`rematSeed`, `rematSlot`).
    bool remat      = false;
    u32  rematDepth = 0;
    u32  rematSeed  = 0;
    u32  rematSlot  = 0;
    Format rematFormat = Format::Unknown; // format the signal would be stored in
};

/// An output of a synthetic pass: color attachment `slot` (raster) or
/// storage texture `slot` (compute).
struct SynthColor {
    u32        slot = 0;
    TextureRef texture;      // version written
    /// OPT-1.2 signal: the value depends only on the pixel and the depth at
    /// that pixel (the fragment's own depth in a Geometry pass, else input
    /// SynthPass::signalDepthInput), so consumers can rematerialise it.
    bool       depthOnly = false;
};

struct SynthPass {
    SynthKind kind = SynthKind::None;
    u32 seed          = 0;   // value function seed (unique per pass)
    u32 aluIterations = 0;   // LCG steps per pixel/thread (x 8 integer ops)
    u32 rematIterations = 0; // extra steps per rematerialised input (its recompute cost)
    static constexpr u32 kRematIterations = 16; // recompute cost of one signal (reprojection-like)
    // Geometry
    u32  triangles  = 0;
    u32  geometrySeed = 0;   // grid depths; equal seeds draw identical geometry
    bool depthWrite = false; // compare Greater + write (reverse-Z)
    bool depthTest  = false; // compare GreaterEqual, no write (after a prepass)
    // Reads
    std::vector<SynthInput> inputs;   // sampled (texture reads) and rematerialised
    std::vector<u32>        fetched;  // color slots read per pixel (framebuffer fetch)
    // Writes
    std::vector<SynthColor> colors;   // color attachments written (raster)
    TextureRef              depthTarget; // Geometry: depth attachment (valid() if any)
    std::vector<SynthColor> storage;  // Compute: storage textures written (<= 2)
    u32 signalDepthInput = ~0u;       // input giving the depth of depthOnly outputs (non-Geometry)
    u32 width = 0, height = 0;        // raster target / dispatch size
};

struct ScenarioParams {
    u32   width  = 2560;   // internal render resolution
    u32   height = 1440;
    u32   shadowSize = 2048;
    float work = 1.0f;     // scales ALU iterations and triangle counts
    /// 2x-bytes control: every RGBA16Float intermediate becomes RGBA32Float
    /// (same work, twice the bytes).
    bool  wideHdr = false;
    /// Passes (by name, without view suffix) that run on the async compute
    /// queue; nullopt = the scenario's default (every candidate of
    /// scenarioAsyncCandidates() on the async queue).  A build choice of the
    /// OPT-1 plans (OPT-1.9).
    std::optional<std::vector<std::string>> async;
    /// Independent copies of the scenario (split screen, extra cameras)
    /// composited by the present pass: 1..6 (graphs of 20-80+ passes).
    u32   views = 1;
    /// OPT-1.2: resources recomputed by their consumers instead of stored
    /// (names; see scenarioRematCandidates()).
    std::vector<std::string> remat;
    /// ALU steps a consumer spends recomputing one signal (break-even sweeps).
    u32   rematIterations = SynthPass::kRematIterations;
};

/// Persistent textures the backend owns, fills once with deterministic
/// contents and binds every frame.
struct ScenarioImport {
    enum class Role : u8 {
        Static,       // read-only input (probe atlas)
        HistoryRead,  // ping-pong history: the texture written last frame
        HistoryWrite, // ping-pong history: the texture written this frame
    };
    u32         resource = ~0u;
    Role        role     = Role::Static;
    TextureDesc desc;
};

struct Scenario {
    std::string name;
    std::vector<SynthPass>      synth;    // per graph pass (kind None if not synthetic)
    std::vector<ScenarioImport> imports;  // bound by the backend every frame
    TextureRef                  output;   // final LDR image of the last view (before the present pass)
};

/// Execute callback factory: the backend turns pass index -> callback.
using SynthExecFactory = std::function<ExecuteFn(u32 pass)>;

[[nodiscard]] u32 scenarioCount();
[[nodiscard]] const char* scenarioName(u32 index);
/// Resources of scenario `index` that can be rematerialised (OPT-1.2).
[[nodiscard]] std::vector<std::string> scenarioRematCandidates(u32 index);
/// Compute passes of scenario `index` that may run on the async queue (OPT-1.9).
[[nodiscard]] std::vector<std::string> scenarioAsyncCandidates(u32 index);
/// Identity of the scenario graph before its build choices (remat, async):
/// the family of its OPT-1 plans, e.g. "scenario:deferred:2560x1440:s2048:w1:v1:hdr16".
[[nodiscard]] std::string scenarioFamily(u32 index, const ScenarioParams& params);

/// Append scenario `index` to `graph` (normally empty, or holding only the
/// caller's imports).  `drawable` is the imported target of the final present
/// pass (a synthetic Fullscreen pass writing it); invalid: no present pass
/// (tests).  Returns false for an unknown index or remat name, or when the
/// graph reports a setup error.
bool buildScenario(u32 index, const ScenarioParams& params, RenderGraph& graph, TextureRef drawable,
                   Scenario& out, const SynthExecFactory& factory, std::string* error = nullptr);

/// Base cost of one synthetic invocation (hashing of inputs and outputs), in
/// IMAD-equivalents: calibrated on the Present pass of OPT-1 spike 1
/// (5.76 Mpx, no ALU steps, 0.147 ms).
inline constexpr double kSynthBaseOps = 100.0;

/// Work of a synthetic pass for the cost model: IMADs (one per LCG step and
/// chain: 4 per step, plus kSynthBaseOps per invocation), triangles drawn.
struct SynthWork {
    double intOps    = 0;
    double triangles = 0;
    double invocations = 0;
};
[[nodiscard]] SynthWork synthWork(const SynthPass& pass);

} // namespace phosphor::rg
