#include "pipeline/forward_variants.h"

namespace phosphor::pipe::forward {

// F3 contract stub: implemented by the F3.3 work package (generated table).
u32 variantCount() { return 1; }
u32 variantIndex(const Variant&) { return 0; }
Variant variantAt(u32) { return {}; }
Variant sceneVariant(const FrameScene&, u32 debugMode) {
    Variant v;
    v.debugMode = debugMode;
    return v;
}
u16 saltConstantIndex() { return 31; }
PipelineDesc pipelineDesc(const Variant&, rg::Format color, u32) { return genericDesc(color); }
PipelineDesc genericDesc(rg::Format) { return {}; }

} // namespace phosphor::pipe::forward
