#pragma once

#include "core/types.h"

namespace phosphor {

// ---------------------------------------------------------------------------
// GPU memory categories, tracked by the backend's single allocation point
// (GpuMemory) and budgeted per hardware tier (F1.4).
// ---------------------------------------------------------------------------

enum class MemoryCategory : u8 {
    Geometry,      // vertex / index / meshlet data
    Textures,      // material textures and texture tables
    Upload,        // CPU-written rings (per-frame data, staging)
    Transient,     // render-graph transient heap (F2.2)
    RenderTargets, // persistent attachments (depth, history buffers)
    Other,         // diagnostics, capture readback, ...
    COUNT
};

constexpr u32 MEMORY_CATEGORY_COUNT = static_cast<u32>(MemoryCategory::COUNT);

[[nodiscard]] const char* memoryCategoryName(MemoryCategory category);

} // namespace phosphor
