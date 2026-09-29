#include "platform/metal/frame_capture.h"
#include "platform/metal/gpu_memory.h"
#include "core/log.h"

#include <stb_image_write.h>

#include <vector>

namespace phosphor {

FrameCapture::~FrameCapture() {
    context_.memory().release(readback_, MemoryCategory::Other);
}

void FrameCapture::prepare(u32 width, u32 height) {
    if (readback_ && width == width_ && height == height_) return;
    width_  = width;
    height_ = height;
    context_.memory().release(readback_, MemoryCategory::Other);
    readback_ = context_.memory().newBuffer(readbackSize(), MTL::ResourceStorageModeShared, MemoryCategory::Other,
                                            "Frame capture");
}

void FrameCapture::encode(MTL4::ComputeCommandEncoder* enc, MTL::Texture* source) {
    const size_t rowBytes = static_cast<size_t>(width_) * 4;
    enc->copyFromTexture(source, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(width_, height_, 1), readback_, 0,
                         rowBytes, rowBytes * height_);
}

bool FrameCapture::writePng(const std::string& path) {
    if (!readback_) {
        LOG_ERROR("No frame captured");
        return false;
    }
    // The drawable is BGRA8 (sRGB-encoded); PNG wants RGBA.
    const auto* src = static_cast<const u8*>(readback_->contents());
    std::vector<u8> rgba(static_cast<size_t>(width_) * height_ * 4);
    for (size_t i = 0; i < rgba.size(); i += 4) {
        rgba[i + 0] = src[i + 2];
        rgba[i + 1] = src[i + 1];
        rgba[i + 2] = src[i + 0];
        rgba[i + 3] = 255;
    }
    if (!stbi_write_png(path.c_str(), static_cast<int>(width_), static_cast<int>(height_), 4, rgba.data(),
                        static_cast<int>(width_) * 4)) {
        LOG_ERROR("Failed to write %s", path.c_str());
        return false;
    }
    LOG_INFO("Captured %ux%u frame to %s", width_, height_, path.c_str());
    return true;
}

} // namespace phosphor
