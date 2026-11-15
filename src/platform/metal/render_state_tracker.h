#pragma once

// F15 closure (tickets 20 / 22): depth-stencil state and viewport of the fused render encoders.
//
// Several graph passes can share one render encoder (render groups).  Each of them used to set the depth-stencil state and the viewport it
// needs, and the Metal API validation layer reports a repeat of the same value on the same encoder as "Redundant call to setDepthStencilState /
// setViewport" (96 of each per capture).  The graph executor registers the encoder of every plain (not split) render group with begin() and
// forgets it with end(); the passes set the two states through setDepthStencil() / setViewport(), which skip a value the registered encoder
// already holds.  An encoder that is not registered (standalone encoders, split groups, probe capture, ImGui callbacks of other contexts)
// always gets the call: the tracker never assumes anything about it.  Header only; state is per thread (encoding of a plain group is serial).

#include "platform/metal/metal_context.h"

#include <cstring>

namespace phosphor::metal_state {

struct Tracker {
    MTL4::RenderCommandEncoder* encoder = nullptr;
    MTL::DepthStencilState*     depth = nullptr;
    bool                        viewportSet = false;
    MTL::Viewport               viewport{};
    unsigned                    skippedDepth = 0, skippedViewport = 0; // per encoder, for tests and diagnostics
};

inline Tracker& tracker() {
    thread_local Tracker t;
    return t;
}

/// A new render encoder starts: nothing was set on it yet (Metal's defaults are never recorded as "set").
inline void begin(MTL4::RenderCommandEncoder* encoder) {
    tracker() = Tracker{};
    tracker().encoder = encoder;
}

/// The registered encoder ended.
inline void end(MTL4::RenderCommandEncoder* encoder) {
    if (tracker().encoder == encoder) tracker() = Tracker{};
}

/// setDepthStencilState unless the registered encoder already holds `state`.  Returns true when the call was made.
inline bool setDepthStencil(MTL4::RenderCommandEncoder* encoder, MTL::DepthStencilState* state) {
    Tracker& t = tracker();
    if (t.encoder == encoder) {
        if (t.depth == state) {
            ++t.skippedDepth;
            return false;
        }
        t.depth = state;
    }
    encoder->setDepthStencilState(state);
    return true;
}

/// setViewport unless the registered encoder already holds the same viewport.  Returns true when the call was made.
inline bool setViewport(MTL4::RenderCommandEncoder* encoder, const MTL::Viewport& viewport) {
    Tracker& t = tracker();
    if (t.encoder == encoder) {
        if (t.viewportSet && std::memcmp(&t.viewport, &viewport, sizeof(MTL::Viewport)) == 0) {
            ++t.skippedViewport;
            return false;
        }
        t.viewport = viewport;
        t.viewportSet = true;
    }
    encoder->setViewport(viewport);
    return true;
}

} // namespace phosphor::metal_state
