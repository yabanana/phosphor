#include "core/memory/memory_budget.h"

namespace phosphor {

const char* memoryCategoryName(MemoryCategory category) {
    switch (category) {
    case MemoryCategory::Geometry:      return "Geometry";
    case MemoryCategory::Textures:      return "Textures";
    case MemoryCategory::Upload:        return "Upload";
    case MemoryCategory::Transient:     return "Transient";
    case MemoryCategory::RenderTargets: return "Render targets";
    case MemoryCategory::Other:         return "Other";
    case MemoryCategory::COUNT:         break;
    }
    return "?";
}

} // namespace phosphor
