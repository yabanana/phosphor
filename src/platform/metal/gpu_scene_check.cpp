#include "platform/metal/gpu_scene_check.h"

#include "platform/metal/gpu_memory.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/scene_renderer.h"
#include "renderer/scene_store.h"

namespace phosphor {

GpuSceneChecker::~GpuSceneChecker() {
    for (MTL::Buffer* b : {instances_, materials_, prefix_, visible_}) context_.memory().release(b, MemoryCategory::Other);
}

MTL::Buffer* GpuSceneChecker::readback(MTL::Buffer*& cache, u64 size, const char* label) {
    size = std::max<u64>(size, 16);
    if (!cache || cache->length() < size) {
        context_.memory().release(cache, MemoryCategory::Other);
        cache = context_.memory().newBuffer(size, MTL::ResourceStorageModeShared, MemoryCategory::Other, label);
    }
    return cache;
}

SceneCheckResult GpuSceneChecker::check(const SceneRenderer& renderer, const SceneStore& store, const GpuScene& scene,
                                        const ECS& ecs, const GPUCullParams& cull, const float* motionSinCos, u32 slot,
                                        GpuDrivenMode mode, bool drawGateOpen) {
    const GpuSceneBuffers& b = renderer.buffers();
    const GpuSceneBuffers::FrameSet& fs = b.frame(slot);
    const u64 slots = store.slotCapacity();
    const u64 instanceBytes = slots * sizeof(GPUInstance);
    const u64 materialBytes = store.materials().size() * sizeof(GPUMaterial);
    const u64 prefixBytes   = (slots + 1) * sizeof(u32);
    MTL::Buffer* instances = readback(instances_, instanceBytes, "Scene check instances");
    MTL::Buffer* materials = readback(materials_, materialBytes, "Scene check materials");
    MTL::Buffer* prefix    = readback(prefix_, prefixBytes, "Scene check prefix");
    MTL::Buffer* visible   = readback(visible_, slots * sizeof(u32), "Scene check visible");
    context_.commitResidency();
    const bool on = mode == GpuDrivenMode::On;
    context_.enqueueUpload([&](MTL4::ComputeCommandEncoder* enc) {
        if (instanceBytes) enc->copyFromBuffer(b.instances(), 0, instances, 0, instanceBytes);
        if (materialBytes) enc->copyFromBuffer(b.materials(), 0, materials, 0, materialBytes);
        if (on) {
            enc->copyFromBuffer(fs.prefix, 0, prefix, 0, prefixBytes);
            if (slots) enc->copyFromBuffer(fs.visible, 0, visible, 0, slots * sizeof(u32));
        }
    });
    context_.flushUploads();

    SceneReadback gpu;
    gpu.instances = {static_cast<const GPUInstance*>(instances->contents()), static_cast<size_t>(slots)};
    gpu.materials = {static_cast<const GPUMaterial*>(materials->contents()), store.materials().size()};
    if (on) {
        gpu.prefix   = {static_cast<const u32*>(prefix->contents()), static_cast<size_t>(slots + 1)};
        gpu.visible  = {static_cast<const u32*>(visible->contents()), static_cast<size_t>(slots)};
        gpu.drawArgs = {static_cast<const u32*>(fs.drawArgs->contents()), static_cast<size_t>(store.commandCount()) * 2};
    }
    gpu.counters     = renderer.counters(slot);
    gpu.drawGateOpen = drawGateOpen;
    return compareScene(store, scene, ecs, cull, motionSinCos, gpu, on);
}

} // namespace phosphor
