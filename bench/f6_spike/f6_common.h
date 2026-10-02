#pragma once

// F6 spikes (docs/opt-log.md, "F6 — Spike"): shared helpers on top of the
// bench/soc harness.  A MEASUREMENT TOOL, not engine code (same rules as
// bench/soc/harness.h: objects only through soc::Context, output only through
// ctx.log(), every benchmark sets a negative control).  Runner:
// bench/soc/soc_bench.cpp (--list, --only F6-S3, --runs N, --validate,
// --force-family apple9, --out).

#include "harness.h"

#include <string>

namespace f6 {

using soc::u32;
using soc::u64;

/// Library compiled from bench/f6_spike/shaders/<file>.
MTL::Library* f6Library(soc::Context& ctx, const std::string& file, bool fastMath = true);

/// Library of an ENGINE shader (shaders/<file>) compiled from source: every
/// `#include "..."` is inlined from src/ (or build/generated/), once.
MTL::Library* engineLibrary(soc::Context& ctx, const std::string& file, bool fastMath = true);

} // namespace f6
