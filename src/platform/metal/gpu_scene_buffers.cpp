#include "platform/metal/gpu_scene_buffers.h"

#include "core/log.h"
#include "core/profile.h"
#include "platform/metal/gpu_memory.h"
#include "renderer/gpu_queue.h"
#include "renderer/scene_store.h"

#include <algorithm>
#include <cstring>

namespace phosphor {

namespace {

constexpr u32 kQueues = SCENE_MAX_LEVELS - 1; // queues 1..7 (queue 0 is CPU-written)

u32 grow(u32 current, u32 needed) {
    if (needed <= current) return current;
    return std::max(needed, current + current / 2);
}

template <typename T>
u64 bytesOf(std::span<const T> s) {
    return static_cast<u64>(s.size()) * sizeof(T);
}

} // namespace

GpuSceneBuffers::GpuSceneBuffers(MetalContext& context) : context_(context) {}

GpuSceneBuffers::~GpuSceneBuffers() { releaseAll(); }

MTL::Buffer* GpuSceneBuffers::privateBuffer(u64 size, const char* label) {
    return context_.memory().newBuffer(std::max<u64>(size, 16), MTL::ResourceStorageModePrivate, MemoryCategory::Scene,
                                       label);
}

MTL::Buffer* GpuSceneBuffers::sharedBuffer(u64 size, const char* label) {
    return context_.memory().newBuffer(std::max<u64>(size, 16), MTL::ResourceStorageModeShared, MemoryCategory::Scene,
                                       label);
}

void GpuSceneBuffers::releaseAll() {
    GpuMemory& m = context_.memory();
    for (MTL::Buffer** b : {&instances_, &materials_, &nodes_, &motions_, &identity_, &buckets_, &commandBuckets_,
                            &childOffsets_, &childSlots_, &motionSlots_, &queues_, &motionParents_}) {
        m.release(*b, MemoryCategory::Scene);
        *b = nullptr;
    }
    for (FrameSet& f : frames_) {
        for (MTL::Buffer** b : {&f.flags, &f.groups, &f.visible, &f.prefix, &f.drawArgs, &f.counters, &f.container}) {
            m.release(*b, MemoryCategory::Scene);
            *b = nullptr;
        }
        m.release(f.icb, MemoryCategory::Scene);
        f.icb = nullptr;
    }
    slotCap_ = materialCap_ = bucketCap_ = commandCap_ = childCap_ = motionCap_ = queueStride_ = motionParentCap_ = 0;
}

void GpuSceneBuffers::clear() {
    releaseAll();
    ++version_;
    forceFull_ = true;
}

bool GpuSceneBuffers::reserve(const SceneStore& store) {
    GpuMemory& m = context_.memory();
    bool changed = false;
    const auto swap = [&](MTL::Buffer*& b, MTL::Buffer* fresh) {
        m.release(b, MemoryCategory::Scene);
        b = fresh;
        changed = true;
    };

    const u32 slots = grow(slotCap_, std::max(store.slotCapacity(), 1u));
    if (slots != slotCap_) {
        slotCap_ = slots;
        swap(instances_, privateBuffer(u64(slots) * sizeof(GPUInstance), "Scene instances"));
        swap(nodes_, privateBuffer(u64(slots) * sizeof(GPUTransformNode), "Scene nodes"));
        swap(motions_, privateBuffer(u64(slots) * sizeof(GPUMotion), "Scene motion"));
        swap(identity_, privateBuffer(u64(slots) * sizeof(u32), "Scene identity list"));
        swap(childOffsets_, privateBuffer(u64(slots + 1) * sizeof(u32), "Scene child offsets"));
        queueStride_ = static_cast<u32>((gpuQueueBytes(slots) + 255) & ~u64{255});
        swap(queues_, privateBuffer(u64(queueStride_) * kQueues, "Scene queues"));
        for (u32 s = 0; s < METAL_FRAMES_IN_FLIGHT; ++s) {
            FrameSet& f = frames_[s];
            swap(f.flags, privateBuffer(u64(slots) * sizeof(u32), "Scene cull flags"));
            swap(f.groups, privateBuffer(u64(cullGroups(slots)) * sizeof(u32), "Scene cull groups"));
            swap(f.visible, privateBuffer(u64(slots) * sizeof(u32), "Scene visible list"));
            swap(f.prefix, privateBuffer(u64(slots + 1) * sizeof(u32), "Scene visible prefix"));
            if (!f.counters) swap(f.counters, sharedBuffer(sizeof(GPUSceneCounters), "Scene counters"));
        }
    }
    const u32 materials = grow(materialCap_, std::max<u32>(static_cast<u32>(store.materials().size()), 1u));
    if (materials != materialCap_) {
        materialCap_ = materials;
        swap(materials_, privateBuffer(u64(materials) * sizeof(GPUMaterial), "Scene materials"));
    }
    const u32 buckets = grow(bucketCap_, std::max<u32>(static_cast<u32>(store.gpuBuckets().size()), 1u));
    if (buckets != bucketCap_) {
        bucketCap_ = buckets;
        swap(buckets_, privateBuffer(u64(buckets) * sizeof(GPUDrawBucket), "Scene buckets"));
    }
    const u32 commands = grow(commandCap_, std::max(store.commandCount(), SCENE_CULL_CLASSES));
    if (commands != commandCap_) {
        commandCap_ = commands;
        swap(commandBuckets_, privateBuffer(u64(commands) * sizeof(u32), "Scene command buckets"));
        MTL::IndirectCommandBufferDescriptor* d = MTL::IndirectCommandBufferDescriptor::alloc()->init();
        d->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
        // Everything comes from the render encoder (D2): pipeline, argument
        // table, depth state and the cull state of the class range.
        d->setInheritPipelineState(true);
        d->setInheritBuffers(true);
        d->setInheritDepthStencilState(true);
        d->setInheritCullMode(true);
        d->setInheritFrontFacingWinding(true);
        d->setInheritDepthBias(true);
        d->setInheritDepthClipMode(true);
        d->setInheritTriangleFillMode(true);
        d->setMaxVertexBufferBindCount(0);
        d->setMaxFragmentBufferBindCount(0);
        for (u32 s = 0; s < METAL_FRAMES_IN_FLIGHT; ++s) {
            FrameSet& f = frames_[s];
            swap(f.drawArgs, sharedBuffer(u64(commands) * 2 * sizeof(u32), "Scene draw arguments"));
            m.release(f.icb, MemoryCategory::Scene);
            f.icb = m.newIndirectCommandBuffer(d, commands, MTL::ResourceStorageModePrivate, MemoryCategory::Scene,
                                               "Scene ICB");
            if (!f.container) f.container = sharedBuffer(sizeof(MTL::ResourceID), "Scene ICB container");
            const MTL::ResourceID id = f.icb->gpuResourceID();
            std::memcpy(f.container->contents(), &id, sizeof(id));
        }
        d->release();
    }
    const u32 children = grow(childCap_, std::max<u32>(static_cast<u32>(store.childSlots().size()), 1u));
    if (children != childCap_) {
        childCap_ = children;
        swap(childSlots_, privateBuffer(u64(children) * sizeof(u32), "Scene child slots"));
    }
    const u32 motion = grow(motionCap_, std::max<u32>(static_cast<u32>(store.motionSlots().size()), 1u));
    if (motion != motionCap_) {
        motionCap_ = motion;
        swap(motionSlots_, privateBuffer(u64(motion) * sizeof(u32), "Scene motion slots"));
    }
    const u32 parents = grow(motionParentCap_, std::max<u32>(static_cast<u32>(store.motionParentSlots().size()), 1u));
    if (parents != motionParentCap_) {
        motionParentCap_ = parents;
        swap(motionParents_, privateBuffer(gpuQueueBytes(parents), "Scene motion parent queue"));
    }
    if (changed) {
        ++version_;
        forceFull_ = true;
        LOG_INFO("GPU scene buffers: %u slots, %u materials, %u buckets, %u commands, %u children, %u motion slots",
                 slotCap_, materialCap_, bucketCap_, commandCap_, childCap_, motionCap_);
    }
    return changed;
}

void GpuSceneBuffers::loadAll(const SceneStore& store) {
    PH_ZONE("Scene load");
    reserve(store);
    const auto upload = [&](MTL::Buffer* dst, const void* data, u64 size) {
        if (size == 0) return;
        const UploadRing::Slice staging = context_.stagingAllocate(size);
        std::memcpy(staging.cpu, data, size);
        context_.enqueueUpload([staging, dst, size](MTL4::ComputeCommandEncoder* enc) {
            enc->copyFromBuffer(staging.buffer, staging.offset, dst, 0, size);
        });
    };
    upload(instances_, store.instances().data(), bytesOf(store.instances()));
    upload(materials_, store.materials().data(), bytesOf(store.materials()));
    upload(nodes_, store.nodes().data(), bytesOf(store.nodes()));
    upload(motions_, store.motions().data(), bytesOf(store.motions()));
    upload(buckets_, store.gpuBuckets().data(), bytesOf(store.gpuBuckets()));
    upload(commandBuckets_, store.commandBuckets().data(), bytesOf(store.commandBuckets()));
    upload(childOffsets_, store.childOffsets().data(), bytesOf(store.childOffsets()));
    upload(childSlots_, store.childSlots().data(), bytesOf(store.childSlots()));
    upload(motionSlots_, store.motionSlots().data(), bytesOf(store.motionSlots()));
    {
        const std::vector<u8>& q = motionParentQueue(store);
        upload(motionParents_, q.data(), q.size());
    }
    {
        std::vector<u32> identity(slotCap_);
        for (u32 i = 0; i < slotCap_; ++i) identity[i] = i;
        upload(identity_, identity.data(), u64(slotCap_) * sizeof(u32));
        // Queue headers: capacity set once, counts cleared every frame on the GPU.
        std::vector<u8> headers(u64(queueStride_) * kQueues, 0);
        for (u32 q = 0; q < kQueues; ++q) {
            GPUQueueHeader h{};
            h.capacity = slotCap_;
            std::memcpy(headers.data() + u64(q) * queueStride_, &h, sizeof(h));
        }
        upload(queues_, headers.data(), headers.size());
    }
    context_.flushUploads();
    forceFull_ = false;
}

const std::vector<u8>& GpuSceneBuffers::motionParentQueue(const SceneStore& store) {
    const std::span<const u32> slots = store.motionParentSlots();
    queueScratch_.resize(gpuQueueBytes(static_cast<u32>(slots.size())));
    GPUQueueHeader h{};
    h.count     = static_cast<u32>(slots.size());
    h.capacity  = h.count;
    h.groups[0] = gpuQueueGroups(h.count, h.capacity, SCENE_HIER_GROUP);
    h.groups[1] = 1;
    h.groups[2] = 1;
    std::memcpy(queueScratch_.data(), &h, sizeof(h));
    if (!slots.empty()) std::memcpy(queueScratch_.data() + sizeof(h), slots.data(), slots.size_bytes());
    return queueScratch_;
}

void GpuSceneBuffers::stageFull(MTL::Buffer* dst, const void* data, u64 size) {
    if (size == 0) return;
    const UploadRing::Slice slice = context_.frameUploads().allocate(size);
    std::memcpy(slice.cpu, data, size);
    copies_.push_back({slice.buffer, slice.offset, dst, size});
}

u64 GpuSceneBuffers::stageFrame(const SceneStore& store) {
    PH_ZONE("Scene stage");
    copies_.clear();
    scatters_ = {};
    const bool reallocated = reserve(store);
    if (reallocated) {
        // A buffer was replaced mid-run (structure growth): send the whole
        // mirror; identity and queue headers are rebuilt by loadAll's path.
        // Rare event (churn beyond the slack), outside measured frames.
        LOG_WARN("GPU scene buffers reallocated during the frame loop: full re-upload");
        context_.waitIdle();
        loadAll(store);
    }
    const SceneSyncStats& st = store.stats();
    u64 bytes = 0;
    const auto stage = [&](u32 index, std::span<const GPUDeltaRecord> records, bool full, MTL::Buffer* dst,
                           const void* data, u64 size) {
        if (full) {
            stageFull(dst, data, size);
            bytes += size;
            return;
        }
        Scatter& s = scatters_[index];
        s.dst   = dst;
        s.count = static_cast<u32>(records.size());
        if (s.count == 0) return;
        const u64 recordBytes = bytesOf(records);
        const UploadRing::Slice slice = context_.frameUploads().allocate(recordBytes);
        std::memcpy(slice.cpu, records.data(), recordBytes);
        s.records = slice.gpu;
        bytes += recordBytes;
    };
    stage(0, store.instanceDeltas(), st.fullInstances, instances_, store.instances().data(), bytesOf(store.instances()));
    stage(1, store.materialDeltas(), st.fullMaterials, materials_, store.materials().data(), bytesOf(store.materials()));
    stage(2, store.nodeDeltas(), st.fullNodes, nodes_, store.nodes().data(), bytesOf(store.nodes()));
    stage(3, store.motionDeltas(), st.fullMotions, motions_, store.motions().data(), bytesOf(store.motions()));
    for (u32 i = 0; i < 4; ++i) {
        if (!scatters_[i].dst) scatters_[i].dst = i == 0 ? instances_ : i == 1 ? materials_ : i == 2 ? nodes_ : motions_;
    }
    if (st.motionParentsChanged && !reallocated) {
        const std::vector<u8>& q = motionParentQueue(store);
        stageFull(motionParents_, q.data(), q.size());
        bytes += q.size();
    }
    if (st.structure && !reallocated) {
        stageFull(buckets_, store.gpuBuckets().data(), bytesOf(store.gpuBuckets()));
        stageFull(commandBuckets_, store.commandBuckets().data(), bytesOf(store.commandBuckets()));
        stageFull(childOffsets_, store.childOffsets().data(), bytesOf(store.childOffsets()));
        stageFull(childSlots_, store.childSlots().data(), bytesOf(store.childSlots()));
        stageFull(motionSlots_, store.motionSlots().data(), bytesOf(store.motionSlots()));
        bytes += bytesOf(store.gpuBuckets()) + bytesOf(store.commandBuckets()) + bytesOf(store.childOffsets()) +
                 bytesOf(store.childSlots()) + bytesOf(store.motionSlots());
    }
    return bytes;
}

} // namespace phosphor
