#pragma once
#include "platform/metal/metal_context.h"
#include "pipeline/pipeline_registry.h"
#include <array>
namespace phosphor {
class AccelerationStructures;
class PipelineCache;
// A table belongs to a resolved consumer PSO and a completed frame slot.
// It may never be borrowed from a diagnostic or another trace kernel.
class RtConsumer {
public:
    RtConsumer(MetalContext&, PipelineCache&, AccelerationStructures&);
    ~RtConsumer();
    RtConsumer(const RtConsumer&) = delete;
    RtConsumer& operator=(const RtConsumer&) = delete;
    void prepare(u32 slot, pipe::PipelineHandle pipeline);
    void bind(MTL4::ArgumentTable*, u32 tlasBinding, u32 iftBinding) const;
private:
    struct Slot {
        MTL::IntersectionFunctionTable* table = nullptr;
        MTL::ComputePipelineState* pipeline = nullptr; // identity only
        u32 generation = ~u32{0};
    };
    MetalContext& context_;
    PipelineCache& pipelines_;
    AccelerationStructures& rt_;
    std::array<Slot, METAL_FRAMES_IN_FLIGHT> slots_{};
    u32 slot_ = 0;
};
} // namespace phosphor
