#include "platform/metal/debug_overlays.h"

namespace phosphor {

// Contract stub (F4 contract commit): replaced by the F4.7 implementation.
struct DebugOverlays::Impl {};

DebugOverlays::DebugOverlays(MetalContext&, PipelineCache&) : impl_(new Impl) {}
DebugOverlays::~DebugOverlays() { delete impl_; }

rg::TextureRef DebugOverlays::addToGraph(rg::RenderGraph&, rg::TextureRef color, u32, u32, OverlayMode,
                                         const SceneRenderer&) {
    return color;
}

void DebugOverlays::prepareFrame(OverlayMode, u32, u32, u32) {}

DebugOverlays::Legend DebugOverlays::legend(OverlayMode) { return {}; }

} // namespace phosphor
