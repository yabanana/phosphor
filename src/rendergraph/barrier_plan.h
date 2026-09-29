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
// Legalisation (BarrierRules): stages that Metal accepts but that do not
// synchronise, and illegal encoder-scope combinations, from the spike below.
//
// F2.3 spike (bench/barrier_spike, README has the full 110-row table):
// Apple M5 Max (Apple10), macOS 27.2 (26B5091g), 2026-09-29, Debug build,
// MTL_DEBUG_LAYER=1 + MTL_SHADER_VALIDATION=1 and without validation; 20
// racing repetitions per variant, ~9 ms producers; "wrong" = repetitions
// with stale data; two full runs, identical verdicts.
//
//   producer -> consumer        barrier (q = barrierAfterQueueStages)     wrong  validation
//   compute  -> vertex read     none                                      20/20  -
//                               q Dispatch -> Vertex                          0  -
//                               q Dispatch -> Fragment                    20/20  -  (does not gate vertex)
//   compute  -> fragment read   q Dispatch -> Fragment                        0  -
//                               q Dispatch -> Vertex                          0  -
//                               q Dispatch -> Tile                        20/20  -  (accepted, useless)
//   render   -> compute         q Fragment -> Dispatch                        0  -
//                               q Vertex / Tile -> Dispatch               20/20  -
//   render   -> render sampled  q Fragment -> Fragment                        0  -
//   (depth stored, then         q Fragment -> Vertex                          0  -
//    sampled in fragment)       q Tile -> Fragment ... / Fragment -> Tile 20/20  -
//   blit     -> fragment        none                                   17-18/20  -
//                               q Blit -> Fragment / Vertex                   0  -
//   in one render encoder       barrierAfterEncoderStages(Fragment|Tile, *)       ABORT: "afterEncoderStages must
//                               (without validation: ignored, 20/20 wrong)        be ... Vertex|Object|Mesh"
//                               barrierAfterEncoderStages(Vertex, Fragment)   0  -
//   in one compute encoder      none                                   19-20/20  -
//                               barrierAfterEncoderStages(Dispatch, Dispatch) 0  -  (only Dispatch|Blit|
//                                                                                 AccelerationStructure legal)
//   aliased placement memory    none                                      20/20  -
//                               q with Device / ResourceAlias / both / None   0  -  (option not observable here)
//
// Costs (GPU, medians): encoder barrier ~1 us, queue barrier ~6 us; for
// render->render, waiting in Fragment is ~16% cheaper than in Vertex
// (10.0 vs 11.9 ms over 200 pairs).  Conclusions used by the rules:
//   * Fragment IS legal and effective on the consumer side of a queue
//     barrier into a render encoder (contrary to the reading of the Feature
//     Set Tables in S-TBDR-5), and cheaper: consumer stages are kept exact.
//   * Tile is accepted everywhere but synchronises nothing: before -> the
//     geometry stages (Vertex covers everything after it), after -> Fragment.
//     Tile-shader data flow itself was not measured.
//   * Inside a render encoder the producer side may only be geometry
//     stages; fusion never creates such barriers, a violation is an error.
//   * Inside a compute encoder only Dispatch/Blit/AccelerationStructure.
//   * Aliasing barriers keep VisibilityOptionResourceAlias (no observable
//     effect on M5 Max; required by the API contract on other GPUs).
// ---------------------------------------------------------------------------

struct BarrierRules {
    /// Consumer-side stages of a render encoder that do not synchronise;
    /// replaced by `rasterPromoteTo`.
    Stages rasterUnsupportedBefore = StageNone;
    Stages rasterPromoteTo         = StageGeometry;
    /// Producer-side stages that do not synchronise; replaced by `afterPromoteTo`.
    Stages unsupportedAfter        = StageNone;
    Stages afterPromoteTo          = StageFragment;
    /// Stages allowed in an Encoder-scope barrier of a compute encoder.
    Stages computeEncoderStages    = ~Stages{0};
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

/// Metal 4 synchronises queues only between commits: split every Compute
/// encoder at the positions where a QueueSync waits (before) or signals
/// (after), so that each sync point is an encoder boundary the executor can
/// turn into a submission boundary.  Render groups are already bounded by
/// their syncs (hoisted to group start/end).  Rebuilds encoders and
/// encoderOfPosition; call after buildQueueSyncs, before buildBarrierPlan.
void splitEncodersAtQueueSyncs(CompiledGraph& compiled);

/// "vertex|fragment", "none" for 0.
[[nodiscard]] std::string stagesName(Stages stages);

} // namespace phosphor::rg
