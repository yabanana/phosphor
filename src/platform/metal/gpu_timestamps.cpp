#include "platform/metal/gpu_timestamps.h"
#include "diagnostics/pass_timings.h"
#include "platform/metal/pipeline_cache.h"
#include "core/log.h"

#include <algorithm>
#include <cstring>

namespace phosphor {

GpuTimestamps::GpuTimestamps(MetalContext& context, PipelineCache& pipelines)
    : context_(context),
      pipelines_(pipelines),
      commitLabel_(NS::String::string("Timestamp: commit start", NS::UTF8StringEncoding)->retain()) {
    pipe::PipelineDesc desc;
    desc.kind         = pipe::PipelineKind::Compute;
    desc.label        = "timestamp_anchor";
    desc.functions[0] = "timestamp_anchor";
    anchor_ = pipelines_.request(desc);
    const u64 frequency = context_.device()->queryTimestampFrequency();
    if (frequency > 0) tickNs_ = 1e9 / static_cast<double>(frequency);
}

GpuTimestamps::~GpuTimestamps() {
    // The owner waits for the GPU before destroying the executor.
    if (heap_) heap_->release();
    commitLabel_->release();
}

void GpuTimestamps::configure(const rg::TimingPlan& plan) {
    plan_      = plan;
    unitCount_ = static_cast<u32>(plan.units.size());
    kinds_.resize(unitCount_);
    unitQueue_.resize(unitCount_);
    for (u32 u = 0; u < unitCount_; ++u) {
        kinds_[u]     = plan.units[u].kind;
        unitQueue_[u] = static_cast<u8>(queueIndex(plan.units[u].queue));
    }
    const u32 needed = plan.queriesPerFrame();
    if (needed > stride_) {
        // Frames in flight may still write into the old heap: release it
        // after them.  Round the stride up so small graph changes reuse it.
        if (heap_) context_.deferRelease(heap_);
        stride_ = std::max(64u, (needed + 63u) & ~63u);
        MTL4::CounterHeapDescriptor* d = MTL4::CounterHeapDescriptor::alloc()->init();
        d->setType(MTL4::CounterHeapTypeTimestamp);
        d->setCount(static_cast<NS::UInteger>(stride_) * METAL_FRAMES_IN_FLIGHT);
        NS::Error* error = nullptr;
        heap_ = context_.device()->newCounterHeap(d, &error);
        d->release();
        if (!heap_) {
            LOG_ERROR("GPU timing disabled: newCounterHeap failed: %s",
                      error ? error->localizedDescription()->utf8String() : "unknown error");
            stride_ = 0;
            return;
        }
        heap_->setLabel(NS::String::string("Pass timestamps", NS::UTF8StringEncoding));
        heap_->invalidateCounterRange(NS::Range::Make(0, static_cast<NS::UInteger>(stride_) * METAL_FRAMES_IN_FLIGHT));
        LOG_INFO("GPU timing: counter heap of %u timestamps (%u per frame slot)", stride_ * METAL_FRAMES_IN_FLIGHT,
                 stride_);
    }
    // Frames in flight used the previous layout: never resolve them.
    for (Slot& s : slots_) {
        s.frame = ~0ull;
        s.startQuery.assign(unitCount_, ~0u);
        s.commitStarts = 0;
        s.firstGraphicsStart = ~0u;
    }
    ticks_.assign(stride_, 0);
    endQuery_.resize(unitCount_);
    for (u32 u = 0; u < unitCount_; ++u) endQuery_[u] = u;
    startTicks_.assign(unitCount_, 0);
    endTicks_.assign(unitCount_, 0);
    ms_.assign(unitCount_, 0.0f);
    valid_ = std::make_unique<bool[]>(std::max(unitCount_, 1u));
    recording_ = ~0u;
}

GpuTimestamps::Resolved GpuTimestamps::resolve(u32 slot) {
    Slot& s = slots_[slot];
    Resolved r;
    if (s.frame == ~0ull || !heap_) return r;
    r.frame = s.frame;
    s.frame = ~0ull;

    const u32 count = unitCount_ + s.commitStarts;
    const NS::Range range = NS::Range::Make(base(slot), count);
    NS::Data* data = heap_->resolveCounterRange(range);  // autoreleased
    std::fill(ticks_.begin(), ticks_.end(), 0);
    if (data) std::memcpy(ticks_.data(), data->bytes(), std::min<size_t>(data->length(), count * sizeof(u64)));

    // Clamp every commit start to the previous frame's last end on its queue.
    for (u32 c = 0; c < s.commitStarts; ++c) {
        u64& t = ticks_[unitCount_ + c];
        const u64 floor = lastEndTick_[s.commitQueue[c]];
        if (t != 0 && t < floor) t = floor;
    }

    computeUnitTimes(ticks_.data(), count, s.startQuery.data(), endQuery_.data(), unitCount_, tickNs_, ms_.data(),
                     valid_.get());
    // Stale entries (not written by this frame) are older than the frame's
    // first commit start on their queue.
    std::array<u64, 2> firstStart{~0ull, ~0ull};
    for (u32 c = 0; c < s.commitStarts; ++c) {
        const u64 t = ticks_[unitCount_ + c];
        if (t != 0) firstStart[s.commitQueue[c]] = std::min(firstStart[s.commitQueue[c]], t);
    }
    for (u32 u = 0; u < unitCount_; ++u) {
        if (valid_[u] && (firstStart[unitQueue_[u]] == ~0ull || ticks_[u] < firstStart[unitQueue_[u]])) {
            valid_[u] = false;
            ms_[u]    = 0.0f;
        }
    }
    u64 lastGraphicsEnd = 0;
    for (u32 u = 0; u < unitCount_; ++u) {
        const u32 q = s.startQuery[u];
        startTicks_[u] = q < count ? ticks_[q] : 0;
        endTicks_[u]   = ticks_[u];
        if (valid_[u]) {
            r.sumMs += ms_[u];
            if (unitQueue_[u] == 0) lastGraphicsEnd = std::max(lastGraphicsEnd, ticks_[u]);
        }
        if (valid_[u]) lastEndTick_[unitQueue_[u]] = std::max(lastEndTick_[unitQueue_[u]], ticks_[u]);
    }
    if (s.firstGraphicsStart < count && ticks_[s.firstGraphicsStart] != 0 &&
        lastGraphicsEnd >= ticks_[s.firstGraphicsStart]) {
        r.spanMs = static_cast<float>(static_cast<double>(lastGraphicsEnd - ticks_[s.firstGraphicsStart]) * tickNs_ *
                                      1e-6);
    }
    r.valid      = true;
    r.units      = unitCount_;
    r.startTicks = startTicks_.data();
    r.endTicks   = endTicks_.data();
    r.ms         = ms_.data();
    r.unitValid  = valid_.get();
    return r;
}

GpuTimestamps::Resolved GpuTimestamps::beginFrame(u32 slot, u64 frameIndex) {
    const Resolved previous = resolve(slot);
    Slot& s = slots_[slot];
    recording_ = enabled() ? slot : ~0u;
    if (recording_ != ~0u) {
        s.frame = frameIndex;
        std::fill(s.startQuery.begin(), s.startQuery.end(), ~0u);
        s.commitStarts = 0;
        s.firstGraphicsStart = ~0u;
        lastQuery_ = {~0u, ~0u};
    }
    return previous;
}

GpuTimestamps::Resolved GpuTimestamps::drain(u32 slot) { return resolve(slot); }

void GpuTimestamps::commitStart(MTL4::CommandBuffer* cmd, rg::Queue queue) {
    if (recording_ == ~0u) return;
    Slot& s = slots_[recording_];
    const u32 query = unitCount_ + s.commitStarts;
    if (query >= stride_) return; // cannot happen: commitStartQueries bounds the submissions
    s.commitQueue[s.commitStarts] = static_cast<u8>(queueIndex(queue));
    ++s.commitStarts;
    // Not cmd->writeTimestampIntoHeap: measured, every call appends to a
    // vector inside the driver's command buffer object that is never cleared
    // when the command buffer is reused (+1.4 MB over 30000 frames, 3 command
    // buffers growing forever).  That call opens a compute context anyway; an
    // compute encoder does the same; it needs a dispatch, or the driver drops
    // it and the timestamp is never written (measured).
    MTL4::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
    encoder->setLabel(commitLabel_);
    encoder->setComputePipelineState(pipelines_.compute(anchor_));
    encoder->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
    encoder->writeTimestamp(MTL4::TimestampGranularityRelaxed, heap_, base(recording_) + query);
    // Later encoders wait for the anchor: without it the GPU may schedule the
    // 1-thread dispatch behind the frame's fragment work and the commit start
    // lands after the unit ends (Many Lights: 25 of 200 frames invalid).
    encoder->barrierAfterStages(MTL::StageDispatch,
                                MTL::StageVertex | MTL::StageObject | MTL::StageMesh | MTL::StageFragment |
                                    MTL::StageDispatch | MTL::StageBlit,
                                MTL4::VisibilityOptionNone);
    encoder->endEncoding();
    const u32 q = queueIndex(queue);
    lastQuery_[q] = query;
    if (q == 0 && s.firstGraphicsStart == ~0u) s.firstGraphicsStart = query;
}

void GpuTimestamps::endUnit(MTL4::RenderCommandEncoder* encoder, u32 unit) {
    if (recording_ == ~0u || unit >= unitCount_) return;
    const u32 q = unitQueue_[unit];
    slots_[recording_].startQuery[unit] = lastQuery_[q];
    encoder->writeTimestamp(MTL4::TimestampGranularityRelaxed, MTL::RenderStageFragment, heap_,
                            base(recording_) + unit);
    lastQuery_[q] = unit;
}

void GpuTimestamps::endUnit(MTL4::ComputeCommandEncoder* encoder, u32 unit) {
    if (recording_ == ~0u || unit >= unitCount_) return;
    const u32 q = unitQueue_[unit];
    slots_[recording_].startQuery[unit] = lastQuery_[q];
    const MTL4::TimestampGranularity granularity = kinds_[unit] == rg::TimestampKind::ComputePassEnd
                                                       ? MTL4::TimestampGranularityPrecise
                                                       : MTL4::TimestampGranularityRelaxed;
    encoder->writeTimestamp(granularity, heap_, base(recording_) + unit);
    lastQuery_[q] = unit;
}

} // namespace phosphor
