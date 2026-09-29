#pragma once

#include "rendergraph/render_graph.h"

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// F2.4 -- TBDR pass fusion and attachment actions (O2, S-TBDR-4).
//
// Consecutive raster passes (in execution order) are fused into one render
// group -- one render encoder, attachments kept in tile memory -- when:
//   * both are Raster on the Graphics queue and `fuse` is set;
//   * all their attachments have the same width, height and sample count;
//   * no slot is bound to two different resources (and no resource to two
//     slots) across the group;
//   * the later pass depends on the group only through attachments
//     (per-pixel): a ShaderRead/CopySrc/IndirectArgs of anything written in
//     the group, or any write that a group member reads outside an
//     attachment, breaks the group (it would need a fragment barrier,
//     S-TBDR-5).
//
// Per attachment of a group (in the group's first-use order):
//   load  = Clear if the first access clears, Load if it preserves (or is
//           DepthRead), DontCare if it discards;
//   store = Store if the version left by the group is read by a live pass
//           after the group or the resource is ImportOutput, else DontCare.
// A transient texture whose every access is an attachment access inside a
// single group, with no Load and no Store, is memoryless.
//
// Also builds the encoder plan: every render group is one Raster encoder;
// runs of consecutive Compute/Blit passes on the same queue share one
// Compute encoder.
// ---------------------------------------------------------------------------

/// Fills renderGroups, groupOfPosition, memoryless, encoders and
/// encoderOfPosition of `compiled` (output of compileOrder()).  Raster
/// passes without attachments or with mismatched sizes add errors and clear
/// `compiled.ok`.
void buildRenderGroups(const RenderGraph& graph, CompiledGraph& compiled, bool fuse);

[[nodiscard]] const char* loadActionName(LoadAction action);
[[nodiscard]] const char* storeActionName(StoreAction action);

} // namespace phosphor::rg
