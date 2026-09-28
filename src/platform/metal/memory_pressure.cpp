#include "platform/metal/memory_pressure.h"
#include "core/log.h"

namespace phosphor {

MemoryPressureMonitor::MemoryPressureMonitor() {
    source_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                     DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
                                         DISPATCH_MEMORYPRESSURE_CRITICAL,
                                     dispatch_get_main_queue());
    if (!source_) {
        LOG_WARN("Memory pressure notifications unavailable");
        return;
    }
    dispatch_set_context(source_, this);
    dispatch_source_set_event_handler_f(source_, &MemoryPressureMonitor::onEvent);
    dispatch_resume(source_);
}

MemoryPressureMonitor::~MemoryPressureMonitor() {
    if (!source_) return;
    // Cancelled handlers never run again; the main queue is not drained after
    // this point, so no pending invocation can reach a destroyed monitor.
    dispatch_source_cancel(source_);
    dispatch_release(source_);
}

void MemoryPressureMonitor::onEvent(void* context) {
    auto* self = static_cast<MemoryPressureMonitor*>(context);
    const uintptr_t flags = dispatch_source_get_data(self->source_);
    if (flags & DISPATCH_MEMORYPRESSURE_CRITICAL) {
        self->simulate(Level::Critical);
    } else if (flags & DISPATCH_MEMORYPRESSURE_WARN) {
        self->simulate(Level::Warning);
    } else {
        self->simulate(Level::Normal);
    }
}

void MemoryPressureMonitor::simulate(Level level) {
    level_ = level;
    ++events_;
}

bool MemoryPressureMonitor::poll(Level& level) {
    level = level_;
    if (events_ == seen_) return false;
    seen_ = events_;
    return true;
}

const char* MemoryPressureMonitor::name(Level level) {
    switch (level) {
    case Level::Normal:   return "normal";
    case Level::Warning:  return "warning";
    case Level::Critical: return "critical";
    }
    return "?";
}

} // namespace phosphor
