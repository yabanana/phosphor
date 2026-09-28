#pragma once

#include "core/types.h"

#include <dispatch/dispatch.h>

namespace phosphor {

// ---------------------------------------------------------------------------
// MemoryPressureMonitor -- macOS memory-pressure notifications (F1.5).
//
// A DISPATCH_SOURCE_TYPE_MEMORYPRESSURE source on the main queue, which SDL
// drains while polling events, so the handler runs on the main thread between
// frames.  The engine polls it once per frame and reacts to rising levels by
// giving memory back (deferred releases, empty heaps).
// ---------------------------------------------------------------------------

class MemoryPressureMonitor {
public:
    enum class Level : u8 { Normal, Warning, Critical };

    MemoryPressureMonitor();
    ~MemoryPressureMonitor();

    MemoryPressureMonitor(const MemoryPressureMonitor&) = delete;
    MemoryPressureMonitor& operator=(const MemoryPressureMonitor&) = delete;

    /// True once per notification received since the previous call; `level`
    /// is the latest level.
    bool poll(Level& level);

    /// Inject a notification as if it came from the system (tests, --simulate-pressure).
    void simulate(Level level);

    [[nodiscard]] Level level() const { return level_; }
    [[nodiscard]] u32 eventCount() const { return events_; }

    [[nodiscard]] static const char* name(Level level);

private:
    static void onEvent(void* context);

    dispatch_source_t source_ = nullptr;
    Level level_  = Level::Normal;
    u32   events_ = 0; // written and read on the main thread only
    u32   seen_   = 0;
};

} // namespace phosphor
