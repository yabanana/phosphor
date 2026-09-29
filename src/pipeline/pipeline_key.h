#pragma once

#include "pipeline/pipeline_desc.h"

#include <string>

namespace phosphor::pipe {

// ---------------------------------------------------------------------------
// PipelineKey -- 64-bit hash of a PipelineDesc (F3.2).
//
// The hash covers exactly the fields that select a GPU binary (kind,
// functions, constants in declaration order, colour outputs) and nothing else
// (not the label).  It is deterministic across runs, builds and platforms
// (fixed byte serialisation + FNV-1a 64 with a final avalanche), so it can be
// logged, stored and compared between the harvest and the runtime.  `salt`
// (--pipeline-salt) is mixed in only for specialised variants, to defeat the
// OS shader cache in cold-compile measurements.
// ---------------------------------------------------------------------------

using PipelineKey = u64;

/// Canonical, human-readable serialisation used for the hash, e.g.
/// "R|forward_vs|forward_fs|c0:u=7,c1:b=1|o0=BGRA8Srgb/none/F".  Stable format.
[[nodiscard]] std::string canonicalString(const PipelineDesc& desc);

/// Hash of canonicalString(desc) (+ salt for non-generic descriptors).
[[nodiscard]] PipelineKey pipelineKey(const PipelineDesc& desc, u32 salt = 0);

/// Low-level hash used by pipelineKey (exposed for the golden tests).
[[nodiscard]] u64 hashBytes(const void* data, size_t size, u64 seed = 0);

} // namespace phosphor::pipe
