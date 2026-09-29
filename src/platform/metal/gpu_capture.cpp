#include "platform/metal/gpu_capture.h"
#include "core/log.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace phosphor {

void GpuCapture::enableLayer() { setenv("MTL_CAPTURE_ENABLED", "1", 1); }

GpuCapture::GpuCapture(MetalContext& context, std::string directory, u32 maxCaptures)
    : context_(context), directory_(std::move(directory)), maxCaptures_(maxCaptures) {
    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (!manager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument)) {
        LOG_ERROR("GPU capture: .gputrace documents are not supported (capture layer missing: was "
                  "MTL_CAPTURE_ENABLED set before the device was created?)");
    } else {
        LOG_INFO("GPU capture: layer active, up to %u capture(s) into %s", maxCaptures_, directory_.c_str());
    }
}

bool GpuCapture::request(const char* reason) {
    if (!available()) return false;
    armed_  = true;
    reason_ = reason;
    return true;
}

void GpuCapture::beginFrame(u64 frameIndex) {
    if (!armed_) return;
    armed_ = false;
    std::error_code ec;
    std::filesystem::create_directories(directory_, ec);
    char name[96];
    std::snprintf(name, sizeof(name), "phosphor-frame%llu-%s.gputrace", static_cast<unsigned long long>(frameIndex),
                  reason_);
    const std::string path = (std::filesystem::path(directory_) / name).string();
    std::filesystem::remove_all(path, ec); // a document of the same name would make the capture fail

    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    MTL::CaptureDescriptor* d = MTL::CaptureDescriptor::alloc()->init();
    // Measured: "Capturing Metal 4 Device is not supported": the capture
    // object is the graphics queue (async-queue work is not in the document).
    d->setCaptureObject(context_.queue());
    d->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    d->setOutputURL(NS::URL::fileURLWithPath(NS::String::string(path.c_str(), NS::UTF8StringEncoding)));
    NS::Error* error = nullptr;
    capturing_ = manager->startCapture(d, &error);
    d->release();
    if (!capturing_) {
        ++failures_;
        LOG_ERROR("GPU capture of frame %llu failed to start: %s", static_cast<unsigned long long>(frameIndex),
                  error ? error->localizedDescription()->utf8String() : "unknown error");
        return;
    }
    lastPath_ = path;
    LOG_INFO("GPU capture: recording frame %llu (%s) into %s", static_cast<unsigned long long>(frameIndex), reason_,
             path.c_str());
}

void GpuCapture::endFrame() {
    if (!capturing_) return;
    MTL::CaptureManager::sharedCaptureManager()->stopCapture();
    capturing_ = false;
    ++captures_;
}

} // namespace phosphor
