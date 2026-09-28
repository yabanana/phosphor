#pragma once

#include "rendergraph/render_graph.h"

#include <string>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// F2.3 -- abstract barrier plan (Metal 4 tracks no hazards, S-MEM-5).
//
// For every dependency between live passes on the same queue (RAW, WAR and
// WAW alike: Metal 4 needs an execution dependency for all three):
//   * same render group      -> nothing (per-pixel order in tile memory);
//   * same compute encoder   -> Encoder-scope barrier before the consumer;
//   * different encoders     -> Queue-scope barrier before the consumer; for
//     a raster consumer it is hoisted to the first position of its group
//     (encoded at render-encoder start).
// afterStages = the producer's stages for that resource (for WAR: the
// reader's stages), beforeStages = the consumer's stages.  Dependencies
// across queues become QueueSync events instead (buildQueueSyncs).
//
// The first use of every placed transient (AliasingPlan) gets a Queue-scope
// barrier whose afterStages are all stages that access any resource sharing
// its memory (including itself: the previous frame used it too), with
// `aliasing` set when the memory is shared with another resource.
//
// Barriers at one position with the same scope and aliasing flag are merged
// (stages OR-ed, resources concatenated): one barrier per kind (S-SYNC-1).
//
// Legalisation (BarrierRules): stages a consumer encoder cannot wait in are
// replaced; the rules come from the F2.3 spike, see the table below.
//
// F2.3 spike results (M5 Max, macOS 27.2, MTL_DEBUG_LAYER=1 +
// MTL_SHADER_VALIDATION=1) -- TO BE FILLED BY THE SPIKE.
// ---------------------------------------------------------------------------

struct BarrierRules {
    /// Consumer-side stages that a render encoder cannot name in a
    /// Queue-scope barrier; replaced by `rasterPromoteTo`.
    Stages rasterUnsupportedBefore = StageNone;
    Stages rasterPromoteTo         = StageGeometry;
    /// Stages that may never appear on the producer side of an Encoder-scope
    /// barrier inside a render encoder (fragment/tile, S-TBDR-5).  The
    /// fusion rules make such barriers impossible; a violation is an error.
    Stages rasterForbiddenEncoderAfter = StageFragment | StageTile;
};

/// Rules measured by the F2.3 spike on Apple9/Apple10.
[[nodiscard]] BarrierRules defaultBarrierRules();

/// Fills compiled.barriers (sorted by position, only positions with at least
/// one barrier).  Needs renderGroups/encoders (buildRenderGroups) and, for
/// aliasing barriers, compiled.aliasing.  Adds errors on forbidden barriers.
void buildBarrierPlan(const RenderGraph& graph, CompiledGraph& compiled, const BarrierRules& rules);

/// Fills compiled.queueSyncs for dependencies whose passes run on different
/// queues: one signal per producer position (values increasing with
/// position, starting at 1), and per consumer position one wait for the
/// highest value it needs from each other queue.
void buildQueueSyncs(const RenderGraph& graph, CompiledGraph& compiled);

/// "vertex|fragment", "none" for 0.
[[nodiscard]] std::string stagesName(Stages stages);

} // namespace phosphor::rg
