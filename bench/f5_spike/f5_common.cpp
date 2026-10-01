#include "f5_common.h"

#include <vector>

namespace f5 {

MTL::Library* f5Library(soc::Context& ctx, const std::string& file, bool fastMath) {
    // Context::library() reads SOC_SHADER_DIR + "/" + file; both directories
    // are in the same checkout (bench/soc/shaders, bench/f5_spike/shaders).
    return ctx.library("../../f5_spike/shaders/" + file, fastMath);
}

FrameRing::FrameRing(soc::Context& ctx, u32 inFlight, u32 extrasPerFrame)
    : ctx_(ctx), inFlight_(inFlight), extras_(extrasPerFrame) {
    const u32 per = 1 + extras_;
    for (u32 i = 0; i < inFlight_ * per; ++i) {
        cmds_.push_back(ctx_.newCommandBuffer());
        allocs_.push_back(ctx_.newAllocator());
    }
    extraOpen_.assign(extras_, false);
    event_ = ctx_.device()->newSharedEvent();
    event_->setSignaledValue(0);
    ctx_.keep(event_);
}

MTL4::CommandBuffer* FrameRing::begin() {
    if (open_) throw soc::BenchError("FrameRing::begin: previous frame not committed");
    if (frame_ >= inFlight_ && !event_->waitUntilSignaledValue(frame_ - inFlight_ + 1, 60000)) {
        throw soc::BenchError("FrameRing: GPU timeout (60 s)");
    }
    const u32 base = slot() * (1 + extras_);
    allocs_[base]->reset();
    cmds_[base]->beginCommandBuffer(allocs_[base]);
    for (u32 k = 0; k < extras_; ++k) extraOpen_[k] = false;
    open_ = true;
    return cmds_[base];
}

MTL4::CommandBuffer* FrameRing::extra(u32 i) {
    if (!open_ || i >= extras_) throw soc::BenchError("FrameRing::extra: bad index or no open frame");
    const u32 idx = slot() * (1 + extras_) + 1 + i;
    if (!extraOpen_[i]) {
        allocs_[idx]->reset();
        cmds_[idx]->beginCommandBuffer(allocs_[idx]);
        extraOpen_[i] = true;
    }
    return cmds_[idx];
}

void FrameRing::commit(MTL4::CommandQueue* queue) {
    if (!open_) throw soc::BenchError("FrameRing::commit: no open frame");
    MTL4::CommandQueue* q = queue ? queue : ctx_.queue();
    const u32 base = slot() * (1 + extras_);
    std::vector<const MTL4::CommandBuffer*> bufs;
    cmds_[base]->endCommandBuffer();
    bufs.push_back(cmds_[base]);
    for (u32 k = 0; k < extras_; ++k) {
        if (!extraOpen_[k]) continue;
        cmds_[base + 1 + k]->endCommandBuffer();
        bufs.push_back(cmds_[base + 1 + k]);
    }
    q->commit(bufs.data(), bufs.size());
    q->signalEvent(event_, frame_ + 1);
    ++frame_;
    open_ = false;
}

void FrameRing::drain() {
    if (frame_ > 0 && !event_->waitUntilSignaledValue(frame_, 60000)) throw soc::BenchError("FrameRing: GPU timeout (60 s)");
}

} // namespace f5
