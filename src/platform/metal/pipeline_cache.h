#pragma once

#include "pipeline/compile_queue.h"
#include "pipeline/pipeline_registry.h"

namespace phosphor {

class MetalContext;

// PipelineCache -- Metal 4 side of F3 (compile threads, archive, flexible
// pipelines, harvest, hot reload).  Contract stub: filled on phase/f3.
class PipelineCache {
public:
    explicit PipelineCache(MetalContext& context) : context_(context) {}

private:
    MetalContext& context_;
};

} // namespace phosphor
