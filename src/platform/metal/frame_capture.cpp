#include "platform/metal/frame_capture.h"
#include "platform/metal/gpu_memory.h"
#include "core/log.h"

#include <stb_image_write.h>

#include <vector>

namespace phosphor {

FrameCapture::~FrameCapture() {
    context_.memory().release(readback_, MemoryCategory::Other);
}

void FrameCapture::encode(MetalContext::Frame& frame) {
    MTL::Texture* target = frame.drawable->texture();
    width_  = static_cast<u32>(target->width());
    height_ = static_cast<u32>(target->height());
    const size_t rowBytes = static_cast<size_t>(width_) * 4;

    context_.memory().release(readback_, MemoryCategory::Other);
    readback_ = context_.memory().newBuffer(rowBytes * height_, MTL::ResourceStorageModeShared,
                                            MemoryCategory::Other, "Frame capture");

    MTL4::ComputeCommandEncoder* enc = frame.commandBuffer->computeCommandEncoder();
    enc->setLabel(NS::String::string("Frame capture", NS::UTF8StringEncoding));
    // No hazard tracking in Metal 4: wait for the render passes that wrote the drawable.
    enc->barrierAfterQueueStages(MTL::StageFragment, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    enc->copyFromTexture(target, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(width_, height_, 1),
                         readback_, 0, rowBytes, rowBytes * height_);
    enc->endEncoding();
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
