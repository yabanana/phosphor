#include "pipeline/pipeline_key.h"

namespace phosphor::pipe {

// F3 contract stub: implemented by the F3.2 work package.
std::string canonicalString(const PipelineDesc&) { return {}; }
PipelineKey pipelineKey(const PipelineDesc&, u32) { return 0; }
u64 hashBytes(const void*, size_t, u64) { return 0; }

} // namespace phosphor::pipe
