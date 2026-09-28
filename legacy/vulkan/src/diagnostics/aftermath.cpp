#include "diagnostics/aftermath.h"
#include "core/log.h"

// ---------------------------------------------------------------------------
// When PHOSPHOR_ENABLE_AFTERMATH is defined, use the real Aftermath SDK.
// Otherwise, provide harmless no-op stubs.
// ---------------------------------------------------------------------------

#ifdef PHOSPHOR_ENABLE_AFTERMATH

#include <GFSDK_Aftermath.h>
#include <GFSDK_Aftermath_GpuCrashDump.h>
#include <GFSDK_Aftermath_GpuCrashDumpDecoding.h>

namespace phosphor {

static bool g_aftermathInitialized = false;

// Callbacks required by Aftermath
static void crashDumpCallback(const void* /*pGpuCrashDump*/,
                               const uint32_t /*gpuCrashDumpSize*/,
                               void* /*pUserData*/) {
    LOG_ERROR("Aftermath: GPU crash dump received");
}

static void shaderDebugInfoCallback(const void* /*pShaderDebugInfo*/,
                                     const uint32_t /*shaderDebugInfoSize*/,
                                     void* /*pUserData*/) {
    LOG_INFO("Aftermath: shader debug info received");
}

static void crashDumpDescriptionCallback(
    PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription,
    void* /*pUserData*/) {
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
                   "phosphor");
    addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion,
                   "1.0");
}

AftermathTracker::AftermathTracker() {
    GFSDK_Aftermath_Result result = GFSDK_Aftermath_EnableGpuCrashDumps(
        GFSDK_Aftermath_Version_API,
        GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
        GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
        crashDumpCallback,
        shaderDebugInfoCallback,
        crashDumpDescriptionCallback,
        nullptr);

    if (result == GFSDK_Aftermath_Result_Success) {
        g_aftermathInitialized = true;
        LOG_INFO("NVIDIA Aftermath initialized");
    } else {
        LOG_WARN("NVIDIA Aftermath initialization failed (result=%d)", static_cast<int>(result));
    }
}

AftermathTracker::~AftermathTracker() {
    if (g_aftermathInitialized) {
        GFSDK_Aftermath_DisableGpuCrashDumps();
        g_aftermathInitialized = false;
    }
}

bool AftermathTracker::isAvailable() const {
    return g_aftermathInitialized;
}

} // namespace phosphor

#else // PHOSPHOR_ENABLE_AFTERMATH not defined -- stubs

namespace phosphor {

AftermathTracker::AftermathTracker() {
    LOG_DEBUG("AftermathTracker: compiled without PHOSPHOR_ENABLE_AFTERMATH (no-op)");
}

AftermathTracker::~AftermathTracker() = default;

bool AftermathTracker::isAvailable() const {
    return false;
}

} // namespace phosphor

#endif // PHOSPHOR_ENABLE_AFTERMATH
