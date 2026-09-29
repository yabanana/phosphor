#include "pipeline/pipeline_desc.h"

namespace phosphor::pipe {

// F3 contract stub: implemented by the F3.2 work package.
PipelineDesc& PipelineDesc::constant(u16, ConstantType, u32) { return *this; }
PipelineDesc& PipelineDesc::output(u32, rg::Format, ColorOutput::Blend) { return *this; }
bool PipelineDesc::isFlexible() const { return false; }
PipelineDesc PipelineDesc::flexible() const { return *this; }
PipelineDesc PipelineDesc::generic() const { return *this; }

} // namespace phosphor::pipe
