#include "platform/metal/linear_capture.h"
#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_graph_executor.h"
#include "renderer/offline_reference.h"
#include "rendergraph/pass_context.h"
#include "core/log.h"

#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace phosphor {
namespace {
MTL4::ArgumentTable* captureTable(MetalContext& context) {
    auto* descriptor = MTL4::ArgumentTableDescriptor::alloc()->init();
    descriptor->setMaxBufferBindCount(2);
    descriptor->setMaxTextureBindCount(1);
    NS::Error* error = nullptr;
    auto* table = context.device()->newArgumentTable(descriptor, &error);
    descriptor->release();
    if (!table) throw std::runtime_error("Linear capture argument table allocation failed");
    return table;
}
void writeCapture(const std::filesystem::path& path, u32 width, u32 height, std::span<const float> rgb, u64 index) {
    std::error_code errorCode;
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, errorCode);
        if (errorCode) throw std::runtime_error("Cannot create linear capture directory: " + errorCode.message());
    }
    // Do not present a partially written float image as a complete reference.
    const std::filesystem::path staging(path.string() + ".partial");
    std::string error;
    if (!writeLinearPfm(staging, width, height, rgb, error))
        throw std::runtime_error("Linear capture failed for " + path.string() + ": " + error);
    std::filesystem::rename(staging, path, errorCode);
    if (errorCode) throw std::runtime_error("Cannot publish linear capture " + path.string() + ": " + errorCode.message());
    LOG_INFO("Linear HDR capture %ux%u frame %llu: %s", width, height,
             static_cast<unsigned long long>(index), path.string().c_str());
}
}

LinearCapture::LinearCapture(MetalContext& context, PipelineCache& pipelines, Config config)
    : context_(context), pipelines_(pipelines), config_(std::move(config)) {
    if (!config_.every) throw std::invalid_argument("Linear capture sequence interval must be positive");
    if (config_.frame == std::numeric_limits<u64>::max())
        throw std::invalid_argument("Linear capture frame cannot overflow the completion timeline");
    if (config_.path.empty() && config_.sequence.empty()) return;
    pipe::PipelineDesc descriptor;
    descriptor.kind = pipe::PipelineKind::Compute;
    descriptor.label = "Linear pre-exposure HDR float32 readback";
    descriptor.functions = {"capture_linear_hdr", "", ""};
    pipeline_ = pipelines_.request(descriptor);
    try {
        for (auto& slot : slots_) slot.table = captureTable(context_);
    } catch (...) {
        for (auto& slot : slots_) if (slot.table) slot.table->release();
        throw;
    }
}
LinearCapture::~LinearCapture() {
    // Argument tables are per slot and cannot retire while a recorded GPU
    // reader still names them. GPU buffers retain GpuMemory's deferred release.
    context_.waitIdle();
    for (auto& slot : slots_) {
        if (slot.table) slot.table->release();
        context_.memory().release(slot.readback, MemoryCategory::Other);
    }
}
void LinearCapture::prepareFrame(u32 index, u64 frameIndex, u32 width, u32 height) {
    auto& slot = slots_.at(index);
    if (slot.pending) throw std::logic_error("Consume completed linear capture before reusing its frame slot");
    if (frameIndex == std::numeric_limits<u64>::max()) throw std::overflow_error("Linear capture completion overflow");
    const bool single = !config_.path.empty() && !singleWritten_ && frameIndex == config_.frame;
    const bool sequence = !config_.sequence.empty() && frameIndex % config_.every == 0;
    const bool enabled = single || sequence;
    if (enabled && (!width || !height)) throw std::invalid_argument("Linear capture source extent is empty");
    if (active_ != enabled || (enabled && (currentWidth_ != width || currentHeight_ != height))) ++version_;
    active_ = enabled;
    prepared_ = true;
    currentSlot_ = index;
    currentWidth_ = width;
    currentHeight_ = height;
    if (!active_) return;
    const u64 pixels = u64(width) * height;
    if (pixels > std::numeric_limits<size_t>::max() / (4 * sizeof(float)) || pixels > std::numeric_limits<u32>::max())
        throw std::overflow_error("Linear capture buffer size overflow");
    if (pixels > slot.capacityPixels) {
        // Only this drained slot grows. Other slots may still own immutable
        // pending frame tags/readbacks; retain their buffers until consumed.
        auto* buffer = context_.memory().newBuffer(pixels * 4 * sizeof(float), MTL::ResourceStorageModeShared,
                                                   MemoryCategory::Other, "Linear HDR RGBA32Float readback");
        if (!buffer) throw std::runtime_error("Linear capture readback allocation failed");
        context_.memory().release(slot.readback, MemoryCategory::Other);
        slot.readback = buffer;
        slot.capacityPixels = pixels;
    }
    slot.index = frameIndex;
    slot.width = width;
    slot.height = height;
    slot.single = single;
    slot.sequence = sequence;
    // Exactly two scalar u32 words consumed by constant uint2 at shader slot0.
    const std::array<u32, 2> extent{width, height};
    auto upload = context_.frameUploads().allocate(sizeof(extent));
    std::memcpy(upload.cpu, extent.data(), sizeof(extent));
    extentAddress_ = upload.gpu;
}
void LinearCapture::addToGraph(rg::RenderGraph& graph, rg::TextureRef linearHdr) {
    if (!prepared_) throw std::logic_error("Prepare linear capture before declaring its graph pass");
    readbackRef_ = {};
    if (!active_) return;
    if (!linearHdr.valid()) throw std::invalid_argument("Linear capture has no pre-exposure HDR source");
    readbackRef_ = graph.importBuffer("Completed linear HDR readback", {u64(currentWidth_) * currentHeight_ * 4 * sizeof(float)},
                                      rg::ImportPerFrame | rg::ImportOutput);
    graph.addPass("Linear pre-exposure HDR capture", rg::PassType::Compute,
        [&](rg::PassBuilder& builder) {
            builder.read(linearHdr, rg::Usage::ShaderRead, rg::StageDispatch);
            readbackRef_ = builder.write(readbackRef_, rg::Usage::ShaderWrite, rg::StageDispatch);
            builder.setProfileShaders("capture_linear_hdr");
        },
        [this, linearHdr](rg::PassContext& context) {
            auto& slot = slots_[currentSlot_];
            if (!active_ || slot.pending) throw std::logic_error("Linear capture encoding slot is not reusable");
            auto* source = static_cast<MTL::Texture*>(context.texture(linearHdr));
            if (!source || source->width() < slot.width || source->height() < slot.height ||
                (config_.scalar ? (source->pixelFormat() != MTL::PixelFormatR16Float && source->pixelFormat() != MTL::PixelFormatR32Float)
                                : (source->pixelFormat() != MTL::PixelFormatRGBA16Float && source->pixelFormat() != MTL::PixelFormatRGBA32Float)))
                throw std::invalid_argument("Linear capture format does not match the declared RGB/scalar source");
            auto* pipeline = pipelines_.compute(pipeline_);
            if (!pipeline) throw std::runtime_error("Linear capture pipeline is not ready");
            slot.table->setAddress(extentAddress_, 0);
            slot.table->setAddress(slot.readback->gpuAddress(), 1);
            slot.table->setTexture(source->gpuResourceID(), 0);
            auto* encoder = static_cast<MTL4::ComputeCommandEncoder*>(context.encoder());
            encoder->setComputePipelineState(pipeline);
            encoder->setArgumentTable(slot.table);
            encoder->dispatchThreads(MTL::Size::Make(slot.width, slot.height, 1), MTL::Size::Make(8, 8, 1));
            slot.pending = true; // identifies scheduled work, NOT GPU completion
        });
}
void LinearCapture::bindFrame(MetalGraphExecutor& executor) {
    if (active_) {
        if (!readbackRef_.valid()) throw std::logic_error("Linear capture graph output is missing");
        executor.bindBuffer(readbackRef_, slots_[currentSlot_].readback);
    }
}
bool LinearCapture::consume(u32 index) {
    auto& slot = slots_.at(index);
    if (!slot.pending) return false;
    if (context_.frameEvent()->signaledValue() < slot.index + 1)
        throw std::logic_error("Linear capture consumed before its GPU frame completed");
    const size_t pixels = size_t(slot.width) * slot.height;
    const auto* rgba = static_cast<const float*>(slot.readback->contents());
    if (!rgba) throw std::runtime_error("Linear capture shared buffer has no CPU contents");
    slot.rgb.resize(pixels * 3);
    for (size_t pixel = 0; pixel < pixels; ++pixel)
        for (u32 channel = 0; channel < 3; ++channel) slot.rgb[pixel * 3 + channel] = rgba[pixel * 4 + (config_.scalar ? 0 : channel)];
    if (slot.single) writeCapture(config_.path, slot.width, slot.height, slot.rgb, slot.index);
    if (slot.sequence) {
        std::ostringstream name;
        name << "frame-" << std::setfill('0') << std::setw(6) << slot.index << ".pfm";
        writeCapture(std::filesystem::path(config_.sequence) / name.str(), slot.width, slot.height, slot.rgb, slot.index);
    }
    if (slot.single) singleWritten_ = true;
    slot.pending = false;
    return true;
}
} // namespace phosphor
