#include "pipeline/pipeline_desc.h"

#include <cassert>

namespace phosphor::pipe {

namespace {
// An attachment slot is "used" when it has a format or was made flexible.
bool isUsed(const ColorOutput& o) { return o.unspecialized || o.format != rg::Format::Unknown; }
} // namespace

PipelineDesc& PipelineDesc::constant(u16 index, ConstantType type, u32 bits) {
    assert(constantCount < MAX_FUNCTION_CONSTANTS && "too many function constants");
    if (constantCount >= MAX_FUNCTION_CONSTANTS) return *this;
    constants[constantCount++] = FunctionConstant{index, type, bits};
    return *this;
}

PipelineDesc& PipelineDesc::output(u32 index, rg::Format format, ColorOutput::Blend blend) {
    assert(index < MAX_COLOR_ATTACHMENTS && "colour attachment index out of range");
    if (index >= MAX_COLOR_ATTACHMENTS) return *this;
    ColorOutput out;
    out.format        = format;
    out.unspecialized = false;
    out.blend         = blend;
    color[index]      = out;
    if (index + 1 > colorCount) colorCount = index + 1;
    return *this;
}

bool PipelineDesc::isFlexible() const {
    for (u32 i = 0; i < colorCount; ++i) {
        if (color[i].unspecialized) return true;
    }
    return false;
}

PipelineDesc PipelineDesc::flexible() const {
    PipelineDesc copy = *this;
    for (u32 i = 0; i < copy.colorCount; ++i) {
        if (isUsed(copy.color[i])) copy.color[i].unspecialized = true;
    }
    return copy;
}

PipelineDesc PipelineDesc::generic() const {
    PipelineDesc copy = *this;
    copy.constants     = {};
    copy.constantCount = 0;
    return copy;
}

} // namespace phosphor::pipe
