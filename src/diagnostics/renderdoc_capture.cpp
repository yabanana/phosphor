#include "diagnostics/renderdoc_capture.h"
#include "core/log.h"

// Include the RenderDoc application API header from third_party
#include "renderdoc/renderdoc_app.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

RenderDocCapture::RenderDocCapture() {
    tryLoadAPI();
}

// ---------------------------------------------------------------------------
// API loading
// ---------------------------------------------------------------------------

void RenderDocCapture::tryLoadAPI() {
#ifdef __linux__
    // RTLD_NOLOAD: only succeed if the library is already in the process
    // (i.e. we were launched from within RenderDoc).  This avoids loading
    // the library when RenderDoc is not being used.
    void* mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (mod) {
        auto getAPI = reinterpret_cast<pRENDERDOC_GetAPI>(
            dlsym(mod, "RENDERDOC_GetAPI"));
        if (getAPI) {
            int result = getAPI(eRENDERDOC_API_Version_1_6_0,
                                reinterpret_cast<void**>(&api_));
            if (result != 1) {
                api_ = nullptr;
            }
        }
    }
#elif defined(_WIN32)
    HMODULE mod = GetModuleHandleA("renderdoc.dll");
    if (mod) {
        auto getAPI = reinterpret_cast<pRENDERDOC_GetAPI>(
            GetProcAddress(mod, "RENDERDOC_GetAPI"));
        if (getAPI) {
            int result = getAPI(eRENDERDOC_API_Version_1_6_0,
                                reinterpret_cast<void**>(&api_));
            if (result != 1) {
                api_ = nullptr;
            }
        }
    }
#endif

    if (api_) {
        LOG_INFO("RenderDoc API loaded successfully (v%d.%d.%d)",
                 api_->GetAPIVersion ? 1 : 0, 6, 0);
    } else {
        LOG_DEBUG("RenderDoc not detected (not launched from RenderDoc)");
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

bool RenderDocCapture::isAvailable() const {
    return api_ != nullptr;
}

void RenderDocCapture::triggerCapture() {
    if (!api_) {
        return;
    }
    captureRequested_ = true;
    LOG_INFO("RenderDoc capture requested");
}

void RenderDocCapture::beginFrame() {
    if (!captureRequested_ || !api_) {
        return;
    }
    api_->StartFrameCapture(nullptr, nullptr);
    capturing_ = true;
}

void RenderDocCapture::endFrame() {
    if (!capturing_ || !api_) {
        return;
    }
    api_->EndFrameCapture(nullptr, nullptr);
    capturing_ = false;
    captureRequested_ = false;
    captureCount_++;
    LOG_INFO("RenderDoc capture #%u completed", captureCount_);
}

} // namespace phosphor
