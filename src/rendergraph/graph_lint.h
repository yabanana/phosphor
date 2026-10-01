#pragma once

#include "rendergraph/render_graph.h"

#include <string>
#include <vector>

namespace phosphor::rg {

// ---------------------------------------------------------------------------
// OPT-1.6 -- store / memoryless lint of a compiled graph (S-TBDR-4, O2).
// (Implemented in graph_lint.cpp.)
//   * every attachment Store must be justified by a later reader of the
//     version it leaves or by an output import (error when not);
//   * every transient texture used only as an attachment is memoryless, or
//     the finding says why not (a load, a store, a second render group, a
//     non-attachment access);
//   * conservative stores (read-only attachments stored again) are noted.
// Findings are text for logs and the dump; `error` findings fail
// compilation with LintMode::Error.
// ---------------------------------------------------------------------------

struct LintFinding {
    bool        error = false;
    u32         resource = ~0u; // or ~0u
    u32         pass     = ~0u; // or ~0u
    std::string message;
};

[[nodiscard]] std::vector<LintFinding> lintGraph(const RenderGraph& graph, const CompiledGraph& compiled);

} // namespace phosphor::rg
