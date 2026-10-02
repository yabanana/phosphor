#include "platform/metal/meshlet_check.h"

#include "platform/metal/gpu_memory.h"
#include "platform/metal/mesh_renderer.h"
#include "platform/metal/metal_context.h"
#include "platform/metal/scene_renderer.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"

#include <algorithm>
#include <cstring>

namespace phosphor {

MeshletChecker::~MeshletChecker() {
    for (MTL::Buffer* b : {instances_, materials_, flags_}) {
        if (b) context_.memory().release(b, MemoryCategory::Other);
    }
}

MTL::Buffer* MeshletChecker::readback(MTL::Buffer*& cache, u64 size, const char* label) {
    size = std::max<u64>(size, 16);
    if (!cache || cache->length() < size) {
        if (cache) context_.memory().release(cache, MemoryCategory::Other);
        cache = context_.memory().newBuffer(size, MTL::ResourceStorageModeShared, MemoryCategory::Other, label);
    }
    return cache;
}

MeshletCheckResult MeshletChecker::check(const MeshRenderer& mesh, const SceneRenderer& scene, const SceneStore& store,
                                         const GpuScene& gpuScene, u32 slot) {
    const GpuSceneBuffers& b = scene.buffers();
    const u64 slots          = store.slotCapacity();
    const u64 instanceBytes  = slots * sizeof(GPUInstance);
    const u64 materialBytes  = store.materials().size() * sizeof(GPUMaterial);
    MTL::Buffer* instances = readback(instances_, instanceBytes, "Meshlet check instances");
    MTL::Buffer* materials = readback(materials_, materialBytes, "Meshlet check materials");
    MTL::Buffer* flags     = readback(flags_, slots * sizeof(u32), "Meshlet check scene flags");
    context_.commitResidency();
    context_.enqueueUpload([&](MTL4::ComputeCommandEncoder* enc) {
        if (instanceBytes) enc->copyFromBuffer(b.instances(), 0, instances, 0, instanceBytes);
        if (materialBytes) enc->copyFromBuffer(b.materials(), 0, materials, 0, materialBytes);
        if (slots) enc->copyFromBuffer(b.frame(slot).flags, 0, flags, 0, slots * sizeof(u32));
    });
    context_.flushUploads();

    const MeshRenderer::FrameSet& f        = mesh.frame(slot);
    const MeshRenderer::CheckReadback& rb  = mesh.checkReadback();
    MeshletCheckInput in;
    std::memcpy(&in.params, f.cullParams->contents(), sizeof(in.params));
    in.twoPhase  = mesh.twoPhase();
    in.capacity  = mesh.capacity();
    in.slotCount = static_cast<u32>(slots);
    in.instances = {static_cast<const GPUInstance*>(instances->contents()), static_cast<size_t>(slots)};
    in.materials = {static_cast<const GPUMaterial*>(materials->contents()), store.materials().size()};
    in.sceneFlags = {static_cast<const u32*>(flags->contents()), static_cast<size_t>(slots)};
    in.meshes    = gpuScene.meshInfos();
    in.bounds    = gpuScene.meshletBounds();
    in.meshlets  = gpuScene.meshlets();
    const size_t cap = static_cast<size_t>(mesh.capacity());
    in.candidates = {static_cast<const GPUMeshletCandidate*>(f.candidates->contents()), cap};
    in.bList      = {static_cast<const GPUMeshletCandidate*>(f.bList->contents()), cap};
    in.bFlags     = {static_cast<const u32*>(f.bFlags->contents()), cap};
    in.decisions  = {static_cast<const u32*>(f.decisions->contents()), cap * 2};
    in.ranges     = static_cast<const GPUMeshletDrawRange*>(f.ranges->contents());
    in.args       = static_cast<const u32*>(f.args->contents());
    in.counters   = mesh.counters(slot);
    in.gate       = *static_cast<const u32*>(f.gate->contents());
    auto floats = [](const MTL::Buffer* buf) {
        return std::span<const float>(static_cast<const float*>(const_cast<MTL::Buffer*>(buf)->contents()), buf->length() / 4);
    };
    in.history = floats(rb.history);
    in.current = floats(rb.current);
    in.next    = floats(rb.next);
    in.depth   = floats(rb.depth);
    in.width   = mesh.depthWidth();
    in.height  = mesh.depthHeight();
    return checkMeshletFrame(in);
}

} // namespace phosphor
