#include "platform/metal/rt_consumer.h"
#include "platform/metal/acceleration_structures.h"
#include "platform/metal/pipeline_cache.h"
#include "platform/metal/gpu_memory.h"
#include <stdexcept>
namespace phosphor {
RtConsumer::RtConsumer(MetalContext& c, PipelineCache& p, AccelerationStructures& rt)
    : context_(c), pipelines_(p), rt_(rt) {}
RtConsumer::~RtConsumer() {
    context_.waitIdle();
    for (auto& s : slots_) context_.memory().release(s.table, MemoryCategory::RayTracing);
}
void RtConsumer::prepare(u32 slot, pipe::PipelineHandle handle) {
    slot_ = slot;
    auto& s = slots_.at(slot);
    auto* pipeline = pipelines_.compute(handle);
    if (!pipeline) throw std::runtime_error("RT consumer pipeline is not ready");
    if (s.pipeline != pipeline || s.generation != pipelines_.generation()) {
        context_.memory().release(s.table, MemoryCategory::RayTracing);
        s.table = nullptr;
        auto* function = pipeline->functionHandle(NS::String::string("rt_alpha_generic", NS::UTF8StringEncoding));
        if (!function) throw std::runtime_error("RT consumer PSO is missing its linked alpha function");
        s.table = context_.memory().newIntersectionFunctionTable(pipeline, 1, MemoryCategory::RayTracing,
                                                                 "Lighting consumer per-slot alpha IFT");
        if (!s.table) throw std::runtime_error("RT consumer IFT allocation failed");
        s.table->setFunction(function, 0);
        s.pipeline = pipeline;
        s.generation = pipelines_.generation();
    }
    const auto r = rt_.traceResources(slot);
    if (!r.tlas || !r.params) throw std::logic_error("RT consumer requires prepared AS resources");
    s.table->setBuffer(r.materials, 0, 0);
    s.table->setBuffer(r.textures, 0, 1);
    s.table->setBuffer(r.vertices, 0, 2);
    s.table->setBuffer(r.indices, 0, 3);
    s.table->setBuffer(r.instances, 0, 4);
    s.table->setBuffer(r.meshes, 0, 5);
    s.table->setBuffer(r.params, 0, 6);
}
void RtConsumer::bind(MTL4::ArgumentTable* table, u32 asBinding, u32 iftBinding) const {
    const auto r = rt_.traceResources(slot_);
    if (!r.tlas || !slots_[slot_].table) throw std::logic_error("Unprepared RT consumer");
    table->setResource(r.tlas->gpuResourceID(), asBinding);
    table->setResource(slots_[slot_].table->gpuResourceID(), iftBinding);
}
} // namespace phosphor
