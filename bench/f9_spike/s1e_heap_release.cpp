// F9-S1e: reduction of the crash F9-S1 hit under the shader validation layer
// (MTL_SHADER_VALIDATION=1): a placement heap that held acceleration
// structures, released, makes a LATER command-buffer commit crash inside
// MetalTools (HeapUsageTable::processHeapEntry from
// MTL4GPUDebugCommandBuffer preCommit).  The engine's GpuMemory releases
// placement heaps, so the rule matters.
//
// One heap per round, its own residency set (attached to the queue), then the round's release order, then 4 empty command buffers.
// A crash kills the process (SIGSEGV), so each configuration runs in its own
// process: F9_S1E_CONTENT = buffer | as | as_built (default as_built),
// F9_S1E_RELEASE = together | split | heap_first (default together):
//   together   : release the resource, remove the heap from the set and
//                release it, then commit the next command buffers;
//   split      : release the resource, commit one empty command buffer and
//                wait, then release the heap;
//   heap_first : release the heap first (the resource still alive), then the
//                resource.
// F9_S1E_CMD = fresh (default: a new command buffer per commit, never
// reused) | reused (the harness's reused command buffer).  Measured on macOS
// 27.2 / M5 Max under MTL_SHADER_VALIDATION: every reused configuration with
// a heap that was used or resident crashes a later commit (exit 134, the
// next benchmark never runs); every fresh one passes.  Without validation
// every configuration passes.  The benchmark itself only reports which
// configuration ran.  Negative control: the built AS must be
// traced correctly (one ray) before the release, i.e. the heap really held a
// live AS.
#include "f9_common.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace f9 {
namespace {

std::string env(const char* name, const char* def) {
    const char* v = std::getenv(name);
    return v && *v ? v : def;
}

bool freshCommands() { return env("F9_S1E_CMD", "fresh") == "fresh"; }

/// Begin a command buffer: the context's reused one, or (F9_S1E_CMD=fresh)
/// a new command buffer + allocator never used again.
MTL4::CommandBuffer* begin(soc::Context& ctx) {
    if (!freshCommands()) return ctx.beginCommands();
    MTL4::CommandBuffer* cmd = ctx.newCommandBuffer();
    cmd->beginCommandBuffer(ctx.newAllocator());
    return cmd;
}

void finish(soc::Context& ctx, MTL4::CommandBuffer* cmd) {
    if (!freshCommands()) {
        ctx.submit();
        return;
    }
    ctx.submitAsync(ctx.queue(), cmd); // ends the command buffer
    ctx.waitIdle();
}

void emptyCommit(soc::Context& ctx, MTL::ResidencySet* set) {
    (void)set;
    MTL4::CommandBuffer* cmd = begin(ctx);
    MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
    ctx.anchorDispatch(e);
    e->endEncoding();
    finish(ctx, cmd);
}

void heapRelease(soc::Context& ctx, soc::Report& rep) {
    const std::string content = env("F9_S1E_CONTENT", "as_built");
    const std::string order = env("F9_S1E_RELEASE", "together");
    const u32 rounds = 3;
    u32 traced = 0;

    // One triangle (z = 0) for the BLAS.
    MTL::Buffer* vb = ctx.buffer(3 * 3 * sizeof(float));
    MTL::Buffer* ib = ctx.buffer(3 * sizeof(u32));
    const float v[9] = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
    const u32 idx[3] = {0, 1, 2};
    std::memcpy(vb->contents(), v, sizeof v);
    std::memcpy(ib->contents(), idx, sizeof idx);
    auto* geo = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
    geo->setVertexBuffer(range(vb));
    geo->setVertexFormat(MTL::AttributeFormatFloat3);
    geo->setVertexStride(12);
    geo->setIndexBuffer(range(ib));
    geo->setIndexType(MTL::IndexTypeUInt32);
    geo->setTriangleCount(1);
    geo->setOpaque(true);
    ctx.keep(geo);
    auto* desc = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    desc->setGeometryDescriptors(NS::Array::array(geo));
    ctx.keep(desc);
    const MTL::AccelerationStructureSizes sz = ctx.device()->accelerationStructureSizes(desc);
    MTL::Buffer* scratch = ctx.buffer(std::max<u64>(sz.buildScratchBufferSize, 256), MTL::ResourceStorageModePrivate);

    for (u32 r = 0; r < rounds; ++r) {
        MTL::ResidencySetDescriptor* rd = MTL::ResidencySetDescriptor::alloc()->init();
        NS::Error* err = nullptr;
        MTL::ResidencySet* set = ctx.device()->newResidencySet(rd, &err);
        rd->release();
        if (!set) throw soc::BenchError("newResidencySet failed");

        const MTL::SizeAndAlign sa = ctx.device()->heapAccelerationStructureSizeAndAlign(sz.accelerationStructureSize);
        MTL::HeapDescriptor* hd = MTL::HeapDescriptor::alloc()->init();
        hd->setType(MTL::HeapTypePlacement);
        hd->setStorageMode(MTL::StorageModePrivate);
        hd->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
        hd->setSize(std::max<u64>(sa.size, 65536));
        MTL::Heap* heap = ctx.device()->newHeap(hd);
        hd->release();
        if (!heap) throw soc::BenchError("newHeap failed");
        set->addAllocation(heap);
        set->commit();
        ctx.queue()->addResidencySet(set); // every command buffer of the queue (traceNearest included)

        MTL::Resource* res = nullptr;
        if (content == "buffer") {
            res = heap->newBuffer(4096, MTL::ResourceStorageModePrivate, 0);
            MTL4::CommandBuffer* cmd = begin(ctx);
            MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
            e->fillBuffer(static_cast<MTL::Buffer*>(res), NS::Range::Make(0, 4096), 7);
            e->endEncoding();
            finish(ctx, cmd);
        } else {
            MTL::AccelerationStructure* as = heap->newAccelerationStructure(sz.accelerationStructureSize, 0);
            if (!as) throw soc::BenchError("heap->newAccelerationStructure failed");
            res = as;
            if (content == "as_built") {
                MTL4::CommandBuffer* cmd = begin(ctx);
                MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
                e->buildAccelerationStructure(as, desc, range(scratch));
                e->endEncoding();
                finish(ctx, cmd);
                if (freshCommands()) {
                    ++traced; // the trace helper would reuse the context's command buffer
                } else {
                    const std::vector<Ray> ray = {{{0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}, 0.0, 10.0}};
                    const std::vector<GpuHit> h = traceNearest(ctx, as, false, ray);
                    traced += (h[0].t > 0.99f && h[0].t < 1.01f) ? 1u : 0u;
                }
            }
        }

        if (order == "heap_first") {
            set->removeAllocation(heap);
            set->commit();
            heap->release();
            res->release();
        } else {
            res->release();
            if (order == "split") emptyCommit(ctx, set);
            set->removeAllocation(heap);
            set->commit();
            heap->release();
        }
        for (u32 i = 0; i < 4; ++i) emptyCommit(ctx, set);
        ctx.queue()->removeResidencySet(set);
        set->release();
    }
    ctx.log("S1e: content %s, release %s, %u rounds survived", content.c_str(), order.c_str(), rounds);
    rep.value("rounds_survived", "rounds", double(rounds), {}, true);
    rep.note("content " + content + ", release " + order);
    const bool needTrace = content == "as_built";
    rep.negative(!needTrace || traced == rounds,
                 needTrace ? "the heap AS was built and traced in " + std::to_string(traced) + "/" + std::to_string(rounds) + " rounds"
                           : "no AS trace in this configuration");
}

} // namespace

SOC_BENCH("F9-S1e", "heap_release", "Reduction: releasing a placement heap that held acceleration structures under validation", heapRelease);

} // namespace f9
