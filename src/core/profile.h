#pragma once

// ---------------------------------------------------------------------------
// CPU profiling zones (F4.2).  With the CMake option PHOSPHOR_TRACY (OFF by
// default) they map to Tracy; otherwise every macro expands to nothing, so
// the default build carries no profiling code.
//
//   PH_ZONE("name")              scoped CPU zone (static name)
//   PH_ZONE_TEXT(text, size)     attach text to the enclosing PH_ZONE
//   PH_FRAME_MARK                end of a presented frame
//   PH_THREAD_NAME("name")       name the calling thread
//   PH_ALLOC(ptr, size, pool)    GPU/CPU allocation in named pool `pool`
//   PH_FREE(ptr, pool)           (static string, one per memory category)
//   PH_PLOT("name", value)       numeric plot
//
// GPU zones come from the resolved timestamps (diagnostics/tracy_gpu.h).
// ---------------------------------------------------------------------------

#ifdef PHOSPHOR_TRACY
#include <tracy/Tracy.hpp>
#define PH_ZONE(name)              ZoneScopedN(name)
#define PH_ZONE_TEXT(text, size)   ZoneText(text, size)
#define PH_FRAME_MARK              FrameMark
#define PH_THREAD_NAME(name)       tracy::SetThreadName(name)
#define PH_ALLOC(ptr, size, pool)  TracyAllocN(ptr, size, pool)
#define PH_FREE(ptr, pool)         TracyFreeN(ptr, pool)
#define PH_PLOT(name, value)       TracyPlot(name, value)
#else
#define PH_ZONE(name)
#define PH_ZONE_TEXT(text, size)
#define PH_FRAME_MARK
#define PH_THREAD_NAME(name)
#define PH_ALLOC(ptr, size, pool)
#define PH_FREE(ptr, pool)
#define PH_PLOT(name, value)
#endif
