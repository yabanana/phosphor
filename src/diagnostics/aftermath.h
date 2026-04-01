#pragma once

#include "core/types.h"

namespace phosphor {

/// NVIDIA Aftermath GPU crash diagnostics.
///
/// When PHOSPHOR_ENABLE_AFTERMATH is defined at build time and the Aftermath
/// SDK is available, this class initialises the Aftermath library and can
/// report crash dump information after a VK_ERROR_DEVICE_LOST.
///
/// When the macro is not defined, every method is a no-op stub so that the
/// rest of the engine compiles and runs without the SDK present.
class AftermathTracker {
public:
    AftermathTracker();
    ~AftermathTracker();

    AftermathTracker(const AftermathTracker&) = delete;
    AftermathTracker& operator=(const AftermathTracker&) = delete;

    /// Returns true if the Aftermath library was loaded and initialised.
    bool isAvailable() const;
};

} // namespace phosphor
