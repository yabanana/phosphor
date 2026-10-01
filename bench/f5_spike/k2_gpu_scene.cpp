// F5-K2: correctness check of the production GPU scene kernels
// (shaders/gpu_scene.metal, F5.3 / F5.5): scene_queue_clear, scene_scatter,
// scene_cull_flags / scan / write (stable compaction) and scene_draw_build,
// against the portable references in src/renderer/cull_reference.* -- on
// random scenes, several bucket layouts and slot counts up to 4M.  A
// CORRECTNESS tool: it measures nothing.  Not engine code (bench rules apply).
//
// The harness compiles MSL from source without include paths, so the kernel
// source is expanded here (every `#include "renderer/..."` inlined from
// src/, once) into a temp file that Context::library() loads through a
// relative path; the render side (instances[visible[instance_id]]) is
// bench/f5_spike/shaders/k2_render.metal.
//
// What is compared:
//   * flags: per slot, the GPU reason against cullReference() (reason +
//     invalid slots); a slot may differ only when |margin| < CULL_BAND (the
//     GPU is compiled with fast math; the count inside the band is reported;
//     with MathModeSafe too);
//   * counters (tested, visible, culled per reason): exactly the counts of the
//     GPU's own flags; drawCommands: exactly the non-empty commands;
//   * visible list + prefix (slotCount + 1): EXACTLY compactReference() of the
//     GPU's own flags (so cull rounding cannot hide a compaction bug);
//   * draw args: EXACTLY drawArgsReference() of the GPU's own prefix;
//   * scatter, queue clear: exact;
//   * one layout renders: the encoded ICB (executed in the same command
//     buffer, class ranges with CPU state) against direct draws built from
//     the args: identical drawn[] flags (= the visible set) and image.
// Negative controls (all must be DETECTED): a flipped frustum plane, a dropped
// scatter record, a dropped ICB command, a corrupted visible entry.
//
// Layouts: 1 slot, 1025 slots (2 groups, empty buckets), 4099 slots with
// invalid holes, 100k slots / 1000 buckets (draw build over many groups, all
// flag combinations), 1,250,000 slots and 4,194,304 slots (4096 groups: the
// scan loops over 4 chunks of 1024), render layout of 200k slots.

#include "f5_common.h"

#include "renderer/cull_math.h"
#include "renderer/cull_reference.h"
#include "renderer/gpu_queue.h"
#include "renderer/gpu_scene_layout.h"
#include "renderer/gpu_types.h"
#include "scene/camera.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <tuple>
#include <string>
#include <unistd.h>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

namespace f5k2 {

namespace ph = phosphor;
using soc::BenchError;
using soc::Context;
using soc::Report;
using soc::Status;
using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace {

struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 1) { for (int i = 0; i < 8; ++i) next(); }
    u64 next64() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    u32 next() { return static_cast<u32>(next64() >> 32); }
    float uniform() { return float(next64() >> 40) * (1.0f / 16777216.0f); } // [0, 1)
    float range(float a, float b) { return a + (b - a) * uniform(); }
    u32 below(u32 n) { return n ? next() % n : 0; }
};

// ---- shader source expansion ---------------------------------------------------------

void expand(const std::filesystem::path& file, std::set<std::string>& seen, std::string& out) {
    std::ifstream in(file);
    if (!in) throw BenchError("cannot read " + file.string());
    std::string line;
    while (std::getline(in, line)) {
        size_t i = line.find_first_not_of(" \t");
        if (i != std::string::npos && line.compare(i, 8, "#include") == 0) {
            const size_t q0 = line.find('"', i);
            if (q0 != std::string::npos) {
                const size_t q1 = line.find('"', q0 + 1);
                const std::string rel = line.substr(q0 + 1, q1 - q0 - 1);
                const std::filesystem::path p = std::filesystem::path(SOC_SOURCE_DIR) / "src" / rel;
                if (seen.insert(p.string()).second) {
                    out += "// ---- begin " + rel + "\n";
                    expand(p, seen, out);
                    out += "// ---- end " + rel + "\n";
                }
                continue;
            }
        }
        if (i != std::string::npos && line.compare(i, 12, "#pragma once") == 0) continue;
        out += line;
        out += '\n';
    }
}

/// Writes the expanded shader to a temp file and returns its path relative to the harness shader directory.
std::string expandedShader() {
    std::set<std::string> seen;
    std::string src;
    expand(std::filesystem::path(SOC_SOURCE_DIR) / "shaders" / "gpu_scene.metal", seen, src);
    const std::filesystem::path tmp = std::filesystem::temp_directory_path() / ("f5_k2_gpu_scene_" + std::to_string(getpid()) + ".metal");
    {
        std::ofstream o(tmp);
        o << src;
    }
    const char* env = std::getenv("SOC_SHADER_DIR");
    const std::filesystem::path base = env ? env : SOC_SHADER_DIR;
    return std::filesystem::relative(tmp, base).string();
}

// ---- scene -----------------------------------------------------------------------------

constexpr u32 kSentinel = ph::DRAW_COMMAND_SENTINEL;

struct Layout {
    const char* name;
    u32 slots;
    u32 buckets;
    bool holes;      // invalid slots inside the live part of a bucket
    bool emptyBuckets;
    u32 flags;       // cull flags
    bool render;     // execute the ICB and compare with direct draws
    u64 seed;
};

struct Scene {
    std::vector<ph::GPUInstance> inst;
    std::vector<ph::GPUMeshInfo> meshes;
    std::vector<ph::GPUDrawBucket> buckets;
    std::vector<u32> commands;       // bucket index or sentinel
    u32 classStart[3] = {}, classLen[3] = {};
    u32 mirrored = 0, valid = 0;
};

constexpr u32 kMaxMeshes = 1100;

struct Geometry { // one global vertex / index buffer for every layout
    std::vector<float> vertices;     // float2
    std::vector<u32> indices;
    std::vector<u32> vertexOffset, indexOffset, indexCount;
};

Geometry makeGeometry() {
    Geometry g;
    for (u32 m = 0; m < kMaxMeshes; ++m) {
        const u32 nTri = 2 + (m * 7 + m / 3) % 11;
        g.vertexOffset.push_back(static_cast<u32>(g.vertices.size() / 2));
        g.indexOffset.push_back(static_cast<u32>(g.indices.size()));
        g.indexCount.push_back(nTri * 3);
        const float phase = float(m) * 0.37f;
        g.vertices.push_back(0.0f);
        g.vertices.push_back(0.0f);
        for (u32 k = 0; k <= nTri; ++k) {
            const float a = phase + 4.712389f * float(k) / float(nTri);
            g.vertices.push_back(std::cos(a));
            g.vertices.push_back(std::sin(a));
        }
        for (u32 k = 0; k < nTri; ++k) {
            g.indices.push_back(0);
            g.indices.push_back(1 + k);
            g.indices.push_back(2 + k);
        }
    }
    return g;
}

Scene makeScene(const Layout& L, const Geometry& geo) {
    Scene s;
    Rng rng(L.seed);
    const u32 B = L.buckets;
    // Bucket capacities: random, summing to exactly `slots`.
    std::vector<u32> cap(B, 1);
    for (u32 i = B; i < L.slots; ++i) cap[rng.below(B)] += 1;
    // Classes: contiguous runs of buckets, class 2 / 1 / 0 each get about a third (small layouts: class 0 only).
    std::vector<u32> cls(B);
    for (u32 b = 0; b < B; ++b) cls[b] = B < 3 ? 0 : std::min(2u, b * 3 / B);
    u32 perClass[3] = {};
    for (u32 b = 0; b < B; ++b) perClass[cls[b]] += 1;
    const u32 meshCount = std::max({perClass[0], perClass[1], perClass[2], 8u});
    for (u32 m = 0; m < meshCount; ++m) {
        ph::GPUMeshInfo mi{};
        mi.indexOffset = geo.indexOffset[m];
        mi.indexCount  = geo.indexCount[m];
        mi.boundingSphere[0] = rng.range(-0.5f, 0.5f);
        mi.boundingSphere[1] = rng.range(-0.5f, 0.5f);
        mi.boundingSphere[2] = rng.range(-0.5f, 0.5f);
        mi.boundingSphere[3] = rng.range(0.5f, 1.5f);
        s.meshes.push_back(mi);
    }
    s.inst.resize(L.slots);
    u32 first = 0, nextMesh[3] = {0, 0, 0};
    u32 curClass = 0xFFFFFFFFu;
    const u32 emptyEvery = L.emptyBuckets ? 5 : 0;
    for (u32 b = 0; b < B; ++b) {
        if (cls[b] != curClass) {
            if (curClass != 0xFFFFFFFFu) {
                s.commands.push_back(kSentinel);
                s.classLen[curClass] = static_cast<u32>(s.commands.size()) - s.classStart[curClass];
            }
            // classes without buckets still get their sentinel
            for (u32 c = curClass == 0xFFFFFFFFu ? 0 : curClass + 1; c < cls[b]; ++c) {
                s.classStart[c] = static_cast<u32>(s.commands.size());
                s.commands.push_back(kSentinel);
                s.classLen[c] = 1;
            }
            curClass = cls[b];
            s.classStart[curClass] = static_cast<u32>(s.commands.size());
        }
        const u32 mesh = nextMesh[cls[b]]++;
        ph::GPUDrawBucket bk{};
        bk.firstSlot    = first;
        bk.capacity     = cap[b];
        bk.meshIndex    = mesh;
        bk.cullClass    = cls[b];
        bk.indexCount   = geo.indexCount[mesh];
        bk.indexOffset  = geo.indexOffset[mesh];
        bk.vertexOffset = geo.vertexOffset[mesh];
        bk.command      = static_cast<u32>(s.commands.size());
        s.buckets.push_back(bk);
        s.commands.push_back(b);
        // Live part: all but a random slack at the end (some buckets are entirely empty).
        u32 live = cap[b];
        if (emptyEvery && b % emptyEvery == 3) live = 0;
        else if (rng.uniform() < 0.5f) live = cap[b] - rng.below(cap[b] / 3 + 1);
        for (u32 i = 0; i < cap[b]; ++i) {
            ph::GPUInstance& in = s.inst[first + i];
            const bool isLive   = i < live && !(L.holes && rng.uniform() < 0.05f);
            if (!isLive) { // garbage that must never be dereferenced
                for (float& f : in.modelMatrix) f = rng.range(-1e6f, 1e6f);
                in.meshIndex = 0xFFFFFFF0u;
                in.materialIndex = rng.next();
                in.flags = (rng.next() & 0xFu) & ~ph::INSTANCE_FLAG_VALID;
                in.pad = 0;
                continue;
            }
            const glm::quat q = glm::normalize(glm::quat(rng.range(-1, 1) + 1.5f, rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)));
            glm::vec3 sc(std::exp(rng.range(std::log(0.01f), std::log(3.0f))), std::exp(rng.range(std::log(0.01f), std::log(3.0f))),
                         std::exp(rng.range(std::log(0.01f), std::log(3.0f))));
            bool mirrored = false;
            if (cls[b] == 1 || (cls[b] == 2 && rng.uniform() < 0.5f)) {
                sc[rng.below(3)] *= -1.0f;
                mirrored = true;
            }
            const glm::mat4 m = glm::translate(glm::mat4(1.0f), glm::vec3(rng.range(-500, 500), rng.range(-500, 500), rng.range(-500, 500))) *
                                glm::mat4_cast(q) * glm::scale(glm::mat4(1.0f), sc);
            std::memcpy(in.modelMatrix, glm::value_ptr(m), sizeof(in.modelMatrix));
            in.meshIndex     = mesh;
            in.materialIndex = 0;
            in.flags         = 1u | ph::INSTANCE_FLAG_VALID | (mirrored ? ph::INSTANCE_FLAG_MIRRORED : 0u);
            in.pad           = 0;
            s.mirrored += mirrored;
            s.valid += 1;
        }
        first += cap[b];
    }
    // Close the last class and add the sentinels of classes without buckets.
    s.commands.push_back(kSentinel);
    s.classLen[curClass] = static_cast<u32>(s.commands.size()) - s.classStart[curClass];
    for (u32 c = curClass + 1; c < 3; ++c) {
        s.classStart[c] = static_cast<u32>(s.commands.size());
        s.commands.push_back(kSentinel);
        s.classLen[c] = 1;
    }
    return s;
}

// ---- GPU rig -------------------------------------------------------------------------------

struct Pipes {
    MTL::ComputePipelineState *clear, *scatter, *flags, *scan, *write, *build;
};

Pipes makePipes(Context& ctx, MTL::Library* lib) {
    Pipes p{};
    p.clear   = ctx.compute(lib, ph::KERNEL_QUEUE_CLEAR);
    p.scatter = ctx.compute(lib, ph::KERNEL_SCATTER);
    p.flags   = ctx.compute(lib, ph::KERNEL_CULL_FLAGS);
    p.scan    = ctx.compute(lib, ph::KERNEL_CULL_SCAN);
    p.write   = ctx.compute(lib, ph::KERNEL_CULL_WRITE);
    p.build   = ctx.compute(lib, ph::KERNEL_DRAW_BUILD);
    for (auto* pso : {p.flags, p.scan, p.write})
        if (pso->maxTotalThreadsPerThreadgroup() < ph::SCENE_CULL_GROUP) throw BenchError("cull pipeline max threads per threadgroup < 1024");
    return p;
}

struct Rig {
    Context& ctx;
    Pipes fast{}, safe{};
    MTL::Library* renderLib = nullptr;
    MTL::RenderPipelineState* renderPso = nullptr;
    MTL::DepthStencilState* depthState = nullptr;
    MTL::Buffer *verts = nullptr, *idx = nullptr;
    std::string failures;
    u32 failCount = 0;
    explicit Rig(Context& c) : ctx(c) {}
    void fail(const std::string& w) {
        ++failCount;
        if (failures.size() < 1200) failures += w + "; ";
    }
};

void bindSlots(Context& ctx, MTL4::ComputeCommandEncoder* ce, MTL::ComputePipelineState* pso, std::initializer_list<std::pair<u32, const MTL::Buffer*>> binds) {
    for (const auto& [slot, buf] : binds) ctx.table()->setAddress(buf->gpuAddress(), slot);
    ce->setComputePipelineState(pso);
    ce->setArgumentTable(ctx.table());
}

void barrier(MTL4::ComputeCommandEncoder* ce) { ce->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice); }

template <class T>
MTL::Buffer* upload(Context& ctx, const std::vector<T>& v, size_t minBytes = 16) {
    MTL::Buffer* b = ctx.buffer(std::max(minBytes, v.size() * sizeof(T)));
    if (!v.empty()) std::memcpy(b->contents(), v.data(), v.size() * sizeof(T));
    return b;
}

// ---- queue clear + scatter -----------------------------------------------------------------

void checkQueueClear(Rig& r, const Pipes& pl, const std::string& tag) {
    Context& ctx = r.ctx;
    constexpr u32 kCap = 1000, kQueues = ph::SCENE_MAX_LEVELS - 1;
    const u32 stride = static_cast<u32>((ph::gpuQueueBytes(kCap) + 31) / 32 * 32);
    MTL::Buffer* queues   = ctx.buffer(size_t(stride) * kQueues);
    MTL::Buffer* counters = ctx.buffer(64);
    MTL::Buffer* params   = ctx.buffer(16);
    const ph::GPUQueueClearParams cp{kQueues, stride, {0, 0}};
    std::memcpy(params->contents(), &cp, sizeof(cp));
    auto* qb = static_cast<u8*>(queues->contents());
    std::memset(qb, 0xAB, queues->length());
    for (u32 q = 0; q < kQueues; ++q) {
        auto* h = reinterpret_cast<ph::GPUQueueHeader*>(qb + size_t(q) * stride);
        h->count = 1234 + q;
        h->capacity = kCap;
        h->overflow = 77 + q;
        h->pad0 = 0x5A5A5A5Au;
        h->groups[0] = 9;
        h->groups[1] = 8;
        h->groups[2] = 7;
        h->pad1 = 0xA5A5A5A5u;
    }
    std::memset(counters->contents(), 0xCD, 64);
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("k2 clear", NS::UTF8StringEncoding));
    bindSlots(ctx, ce, pl.clear, {{0, params}, {ph::SB_CLEAR_QUEUES, queues}, {ph::SB_CLEAR_COUNTERS, counters}});
    ce->dispatchThreads(MTL::Size::Make(std::max(kQueues, 8u), 1, 1), MTL::Size::Make(64, 1, 1));
    ce->endEncoding();
    ctx.submit();
    u32 bad = 0;
    for (u32 q = 0; q < kQueues; ++q) {
        const auto* h = reinterpret_cast<const ph::GPUQueueHeader*>(qb + size_t(q) * stride);
        bad += h->count != 0 || h->overflow != 0 || h->groups[0] != 0 || h->groups[1] != 0 || h->groups[2] != 0;
        bad += h->capacity != kCap || h->pad0 != 0x5A5A5A5Au || h->pad1 != 0xA5A5A5A5u;
        for (u32 e = 0; e < kCap; e += 97) bad += reinterpret_cast<const u32*>(h + 1)[e] != 0xABABABABu;
    }
    const auto* cn = static_cast<const u32*>(counters->contents());
    for (u32 i = 0; i < sizeof(ph::GPUSceneCounters) / 4; ++i) bad += cn[i] != 0;
    bad += cn[8] != 0xCDCDCDCDu; // beyond the counters: untouched
    if (bad) r.fail(tag + " queue clear: " + std::to_string(bad) + " wrong words");
    ctx.log("F5-K2 %s: queue clear %s", tag.c_str(), bad ? "FAILED" : "exact");
}

// Returns the number of mismatching words after a scatter of `count` records of `words` words each.
// `dropLast`: the kernel is told count - 1 records while the expectation has all (negative control).
u32 runScatter(Rig& r, const Pipes& pl, u32 count, u32 words, u32 dstElems, bool dropLast, u32* dispatchedCount = nullptr) {
    Context& ctx = r.ctx;
    Rng rng(0x5CA77E4 + count * 31 + words);
    std::vector<ph::GPUDeltaRecord> recs(std::max(count, 1u));
    std::vector<u32> slots(dstElems);
    for (u32 i = 0; i < dstElems; ++i) slots[i] = i;
    for (u32 i = 0; i < count; ++i) std::swap(slots[i], slots[i + rng.below(dstElems - i)]); // distinct random slots
    std::vector<u32> dstInit(size_t(dstElems) * words), expect;
    for (u32& w : dstInit) w = rng.next();
    expect = dstInit;
    for (u32 i = 0; i < count; ++i) {
        ph::GPUDeltaRecord& rec = recs[i];
        rec.slot = slots[i];
        rec.pad[0] = rec.pad[1] = rec.pad[2] = 0;
        for (u32 w = 0; w < 20; ++w) rec.payload[w] = rng.next();
        for (u32 w = 0; w < words; ++w) expect[size_t(rec.slot) * words + w] = rec.payload[w];
    }
    MTL::Buffer* recBuf = upload(ctx, recs, sizeof(ph::GPUDeltaRecord));
    MTL::Buffer* dst    = upload(ctx, dstInit);
    MTL::Buffer* params = ctx.buffer(16);
    const u32 n = dropLast && count > 0 ? count - 1 : count;
    const ph::GPUScatterParams sp{n, words, {0, 0}};
    std::memcpy(params->contents(), &sp, sizeof(sp));
    MTL4::CommandBuffer* cmd = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("k2 scatter", NS::UTF8StringEncoding));
    bindSlots(ctx, ce, pl.scatter, {{0, params}, {ph::SB_SCATTER_RECORDS, recBuf}, {ph::SB_SCATTER_DST, dst}});
    ce->dispatchThreads(MTL::Size::Make(std::max(n, 1u), 1, 1), MTL::Size::Make(ph::SCENE_SCATTER_GROUP, 1, 1));
    ce->endEncoding();
    ctx.submit();
    if (dispatchedCount) *dispatchedCount = n;
    const auto* got = static_cast<const u32*>(dst->contents());
    u32 bad = 0;
    for (size_t i = 0; i < expect.size(); ++i) bad += got[i] != expect[i];
    return bad;
}

// ---- cull + draw build ----------------------------------------------------------------------

struct Gpu {
    std::vector<u32> flags;
    std::vector<u32> visible, prefix;
    std::vector<u32> args;
    u32 counters[8] = {};
};

struct Buffers {
    MTL::Buffer *inst, *meshes, *cullParams, *flags, *groups, *counters, *visible, *prefix;
    MTL::Buffer *buckets, *commands, *drawParams, *drawArgs, *icbBox;
    MTL::IndirectCommandBuffer* icb = nullptr;
    u32 slots = 0, groupCount = 0, commandCount = 0;
};

MTL::IndirectCommandBuffer* makeIcb(Rig& r, u32 maxCommands) {
    MTL::IndirectCommandBufferDescriptor* d = MTL::IndirectCommandBufferDescriptor::alloc()->init();
    d->setCommandTypes(MTL::IndirectCommandTypeDrawIndexed);
    d->setInheritPipelineState(true);
    d->setInheritBuffers(true);
    d->setInheritDepthStencilState(true);
    d->setInheritCullMode(true);
    d->setInheritFrontFacingWinding(true);
    d->setMaxVertexBufferBindCount(4);
    d->setMaxFragmentBufferBindCount(0);
    MTL::IndirectCommandBuffer* icb = r.ctx.device()->newIndirectCommandBuffer(d, maxCommands, MTL::ResourceStorageModePrivate);
    if (!icb) icb = r.ctx.device()->newIndirectCommandBuffer(d, maxCommands, MTL::ResourceStorageModeShared);
    d->release();
    if (!icb) throw BenchError("newIndirectCommandBuffer failed");
    r.ctx.adopt(icb);
    return icb;
}

Buffers makeBuffers(Rig& r, const Scene& sc, const ph::GPUCullParams& params) {
    Context& ctx = r.ctx;
    Buffers b;
    b.slots        = params.slotCount;
    b.groupCount   = params.groupCount;
    b.commandCount = static_cast<u32>(sc.commands.size());
    b.inst         = upload(ctx, sc.inst);
    b.meshes       = upload(ctx, sc.meshes);
    b.cullParams   = ctx.buffer(256);
    b.flags        = ctx.buffer(size_t(b.slots) * 4 + 16);
    b.groups       = ctx.buffer(size_t(b.groupCount) * 4 + 16);
    b.counters     = ctx.buffer(64);
    b.visible      = ctx.buffer(size_t(b.slots) * 4 + 16);
    b.prefix       = ctx.buffer((size_t(b.slots) + 1) * 4 + 16);
    b.buckets      = upload(ctx, sc.buckets);
    b.commands     = upload(ctx, sc.commands);
    b.drawParams   = ctx.buffer(16);
    b.drawArgs     = ctx.buffer(size_t(b.commandCount) * 8 + 16);
    b.icbBox       = ctx.buffer(16);
    b.icb          = makeIcb(r, b.commandCount);
    const MTL::ResourceID rid = b.icb->gpuResourceID();
    std::memcpy(b.icbBox->contents(), &rid, sizeof(rid));
    std::memcpy(b.cullParams->contents(), &params, sizeof(params));
    const ph::GPUDrawParams dp{b.commandCount, static_cast<u32>(sc.buckets.size()), b.slots, 0};
    std::memcpy(b.drawParams->contents(), &dp, sizeof(dp));
    return b;
}

void setDrawCommandCount(Buffers& b, u32 commandCount, u32 bucketCount) {
    const ph::GPUDrawParams dp{commandCount, bucketCount, b.slots, 0};
    std::memcpy(b.drawParams->contents(), &dp, sizeof(dp));
}

void encodeKernels(Rig& r, MTL4::ComputeCommandEncoder* ce, const Pipes& pl, Buffers& b, MTL::Buffer* clearQueues, MTL::Buffer* clearParams,
                   MTL::Buffer* indexBuffer, u32 commandCount) {
    Context& ctx = r.ctx;
    bindSlots(ctx, ce, pl.clear, {{0, clearParams}, {ph::SB_CLEAR_QUEUES, clearQueues}, {ph::SB_CLEAR_COUNTERS, b.counters}});
    ce->dispatchThreads(MTL::Size::Make(8, 1, 1), MTL::Size::Make(64, 1, 1));
    barrier(ce);
    bindSlots(ctx, ce, pl.flags,
              {{0, b.cullParams}, {ph::SB_CULL_INSTANCES, b.inst}, {ph::SB_CULL_MESHES, b.meshes}, {ph::SB_CULL_FLAGS, b.flags},
               {ph::SB_CULL_GROUPS, b.groups}, {ph::SB_CULL_COUNTERS, b.counters}});
    ce->dispatchThreadgroups(MTL::Size::Make(b.groupCount, 1, 1), MTL::Size::Make(ph::SCENE_CULL_GROUP, 1, 1));
    barrier(ce);
    bindSlots(ctx, ce, pl.scan, {{0, b.cullParams}, {ph::SB_CULL_GROUPS, b.groups}});
    ce->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(ph::SCENE_CULL_GROUP, 1, 1));
    barrier(ce);
    bindSlots(ctx, ce, pl.write,
              {{0, b.cullParams}, {ph::SB_CULL_FLAGS, b.flags}, {ph::SB_CULL_GROUPS, b.groups}, {ph::SB_CULL_VISIBLE, b.visible},
               {ph::SB_CULL_PREFIX, b.prefix}});
    ce->dispatchThreadgroups(MTL::Size::Make(b.groupCount, 1, 1), MTL::Size::Make(ph::SCENE_CULL_GROUP, 1, 1));
    // ICB reset on the GPU timeline, then the draw build.
    ce->barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit, MTL::StageDispatch | MTL::StageBlit, MTL4::VisibilityOptionDevice);
    ce->resetCommandsInBuffer(b.icb, NS::Range::Make(0, b.commandCount));
    ce->barrierAfterEncoderStages(MTL::StageDispatch | MTL::StageBlit, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    bindSlots(ctx, ce, pl.build,
              {{0, b.drawParams}, {ph::SB_DRAW_BUCKETS, b.buckets}, {ph::SB_DRAW_COMMANDS, b.commands}, {ph::SB_DRAW_PREFIX, b.prefix},
               {ph::SB_DRAW_ICB, b.icbBox}, {ph::SB_DRAW_INDICES, indexBuffer}, {ph::SB_DRAW_ARGS, b.drawArgs}, {ph::SB_DRAW_COUNTERS, b.counters}});
    ce->dispatchThreads(MTL::Size::Make(std::max(commandCount, 1u), 1, 1), MTL::Size::Make(ph::SCENE_DRAW_GROUP, 1, 1));
}

void fillGarbage(Buffers& b) {
    for (MTL::Buffer* x : {b.flags, b.groups, b.visible, b.prefix, b.drawArgs}) std::memset(x->contents(), 0xFF, x->length());
    std::memset(b.counters->contents(), 0xCD, 64);
}

Gpu readback(const Buffers& b, u32 commandCount) {
    Gpu g;
    const auto* f = static_cast<const u32*>(b.flags->contents());
    g.flags.assign(f, f + b.slots);
    const auto* p = static_cast<const u32*>(b.prefix->contents());
    g.prefix.assign(p, p + b.slots + 1);
    const u32 total = g.prefix[b.slots];
    const auto* v   = static_cast<const u32*>(b.visible->contents());
    g.visible.assign(v, v + std::min(total, b.slots));
    const auto* a = static_cast<const u32*>(b.drawArgs->contents());
    g.args.assign(a, a + size_t(commandCount) * 2);
    std::memcpy(g.counters, b.counters->contents(), sizeof(g.counters));
    return g;
}

/// flags value -> reference "result" encoding (invalid 255, reason 0..3), or 254 for a malformed flag word.
u8 resultFromFlag(u32 f) {
    switch (f) {
    case 0: return ph::CULL_RESULT_INVALID;
    case 1: return 0;
    case 2: return 1;
    case 4: return 2;
    case 6: return 3;
    default: return 254;
    }
}

struct Check {
    u32 outsideBand = 0, insideBand = 0, structural = 0;
    u32 visible = 0;
};

/// Compares everything of one run; returns outside-band mismatches in .outsideBand.
Check checkRun(Rig& r, const Scene& sc, const ph::GPUCullParams& params, const Gpu& g, const std::vector<u8>& refResult,
               const std::vector<float>& refMargin, const std::string& tag, bool checkDraw) {
    Check c;
    const u32 n = params.slotCount;
    std::vector<u8> gpuResult(n);
    u32 count[4] = {}, tested = 0;
    for (u32 i = 0; i < n; ++i) {
        gpuResult[i] = resultFromFlag(g.flags[i]);
        if (gpuResult[i] == 254) {
            ++c.structural;
            continue;
        }
        if (gpuResult[i] != ph::CULL_RESULT_INVALID) {
            ++tested;
            ++count[gpuResult[i]];
        }
        if (gpuResult[i] != refResult[i]) {
            if (gpuResult[i] == ph::CULL_RESULT_INVALID || refResult[i] == ph::CULL_RESULT_INVALID) ++c.structural;
            else if (std::fabs(refMargin[i]) < ph::CULL_BAND) ++c.insideBand;
            else ++c.outsideBand;
        }
    }
    c.visible = count[0];
    // Counters: exactly the GPU's own flags.
    const u32 exp[5] = {tested, count[0], count[1], count[2], count[3]};
    for (u32 k = 0; k < 5; ++k)
        if (g.counters[k] != exp[k]) {
            ++c.structural;
            r.fail(tag + " counter " + std::to_string(k) + " = " + std::to_string(g.counters[k]) + " expected " + std::to_string(exp[k]));
        }
    if (g.counters[6] != 0 || g.counters[7] != 0) {
        ++c.structural;
        r.fail(tag + " counters 6/7 not cleared");
    }
    // Compaction: exact against the GPU's own flags.
    std::vector<u32> visible, prefix;
    ph::compactReference(gpuResult, visible, prefix);
    if (g.prefix != prefix) {
        ++c.structural;
        r.fail(tag + " prefix differs from compactReference");
    }
    if (g.visible != visible) {
        ++c.structural;
        r.fail(tag + " visible list differs from compactReference");
    }
    if (checkDraw) {
        std::vector<u32> args;
        ph::drawArgsReference(sc.buckets, sc.commands, g.prefix, args);
        if (g.args != args) {
            u32 firstBad = 0;
            while (firstBad < args.size() && g.args[firstBad] == args[firstBad]) ++firstBad;
            ++c.structural;
            r.fail(tag + " draw args differ from drawArgsReference (first at word " + std::to_string(firstBad) + ")");
        }
        u32 nonEmpty = 0;
        for (size_t k = 0; k < args.size(); k += 2) nonEmpty += args[k] != 0;
        if (g.counters[ph::SCENE_COUNTER_DRAW_COMMANDS] != nonEmpty) {
            ++c.structural;
            r.fail(tag + " drawCommands counter " + std::to_string(g.counters[5]) + " expected " + std::to_string(nonEmpty));
        }
    }
    return c;
}

// ---- rendering ------------------------------------------------------------------------------

constexpr u32 kRes = 512;

struct Target {
    MTL::Texture *color = nullptr, *depth = nullptr;
    MTL::Buffer *drawn = nullptr, *readback = nullptr;
};

Target makeTarget(Rig& r, u32 slots) {
    Target t;
    auto* cd = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatR32Uint, kRes, kRes, false);
    cd->setUsage(MTL::TextureUsageRenderTarget);
    cd->setStorageMode(MTL::StorageModePrivate);
    t.color = r.ctx.texture(cd);
    auto* dd = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatDepth32Float, kRes, kRes, false);
    dd->setUsage(MTL::TextureUsageRenderTarget);
    dd->setStorageMode(MTL::StorageModePrivate);
    t.depth    = r.ctx.texture(dd);
    t.drawn    = r.ctx.buffer(size_t(slots) * 4 + 16);
    t.readback = r.ctx.buffer(size_t(kRes) * kRes * 4);
    return t;
}

void setupRender(Rig& r, const Geometry& geo) {
    Context& ctx = r.ctx;
    r.renderLib = f5::f5Library(ctx, "k2_render.metal");
    auto* rd = MTL4::RenderPipelineDescriptor::alloc()->init();
    rd->setVertexFunctionDescriptor(ctx.function(r.renderLib, "k2_vs"));
    rd->setFragmentFunctionDescriptor(ctx.function(r.renderLib, "k2_fs"));
    rd->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatR32Uint);
    rd->setSupportIndirectCommandBuffers(MTL4::IndirectCommandBufferSupportStateEnabled);
    r.renderPso = ctx.render(rd);
    rd->release();
    auto* dsd = MTL::DepthStencilDescriptor::alloc()->init();
    dsd->setDepthCompareFunction(MTL::CompareFunctionGreater); // reverse-Z
    dsd->setDepthWriteEnabled(true);
    r.depthState = ctx.device()->newDepthStencilState(dsd);
    dsd->release();
    ctx.keep(r.depthState);
    r.verts = upload(ctx, geo.vertices);
    r.idx   = upload(ctx, geo.indices);
}

/// Render pass: `icb` != nullptr executes the ICB class ranges, else direct draws from `args`.
void encodeRender(Rig& r, MTL4::CommandBuffer* cmd, const Scene& sc, const Buffers& b, Target& t, bool useIcb, const std::vector<u32>* args,
                  bool queueBarrier) {
    Context& ctx = r.ctx;
    auto* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c  = pd->colorAttachments()->object(0);
    c->setTexture(t.color);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionStore);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    auto* d = pd->depthAttachment();
    d->setTexture(t.depth);
    d->setLoadAction(MTL::LoadActionClear);
    d->setStoreAction(MTL::StoreActionDontCare);
    d->setClearDepth(0.0);
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    re->setLabel(NS::String::string("k2 render", NS::UTF8StringEncoding));
    pd->release();
    re->setViewport(MTL::Viewport{0.0, 0.0, double(kRes), double(kRes), 0.0, 1.0});
    MTL4::ArgumentTable* tbl = ctx.table();
    u32 slot = 0;
    for (const MTL::Buffer* buf : {b.inst, b.visible, r.verts, t.drawn}) tbl->setAddress(buf->gpuAddress(), slot++);
    re->setRenderPipelineState(r.renderPso);
    re->setDepthStencilState(r.depthState);
    re->setArgumentTable(tbl, MTL::RenderStageVertex | MTL::RenderStageFragment);
    if (queueBarrier) re->barrierAfterQueueStages(MTL::StageDispatch | MTL::StageBlit, MTL::StageVertex, MTL4::VisibilityOptionDevice);
    // State tracked from Metal's defaults (clockwise, no culling): the validation layer rejects redundant state.
    MTL::Winding winding = MTL::WindingClockwise;
    MTL::CullMode cull   = MTL::CullModeNone;
    const auto setState  = [&](u32 cls) {
        const MTL::CullMode want = cls == 0 ? MTL::CullModeBack : cls == 1 ? MTL::CullModeFront : MTL::CullModeNone;
        if (winding != MTL::WindingCounterClockwise) re->setFrontFacingWinding(winding = MTL::WindingCounterClockwise);
        if (cull != want) re->setCullMode(cull = want);
    };
    const MTL::GPUAddress idxBase = r.idx->gpuAddress();
    const NS::UInteger idxLen     = r.idx->length();
    for (u32 cl = 0; cl < 3; ++cl) {
        setState(cl);
        if (useIcb) {
            re->executeCommandsInBuffer(b.icb, NS::Range::Make(sc.classStart[cl], sc.classLen[cl]));
        } else {
            for (u32 k = sc.classStart[cl]; k < sc.classStart[cl] + sc.classLen[cl]; ++k) {
                const u32 bi = sc.commands[k];
                if (bi == kSentinel || (*args)[k * 2] == 0) continue;
                const ph::GPUDrawBucket& bk = sc.buckets[bi];
                re->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, bk.indexCount, MTL::IndexTypeUInt32, idxBase + u64(bk.indexOffset) * 4,
                                          NS::UInteger(bk.indexCount) * 4, (*args)[k * 2], bk.vertexOffset, (*args)[k * 2 + 1]);
            }
        }
    }
    re->endEncoding();
    (void)idxLen;
}

void encodeReadback(Rig& r, MTL4::CommandBuffer* cmd, Target& t) {
    MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
    ce->setLabel(NS::String::string("k2 readback", NS::UTF8StringEncoding));
    ce->barrierAfterQueueStages(MTL::StageFragment | MTL::StageVertex | MTL::StageDispatch, MTL::StageBlit, MTL4::VisibilityOptionDevice);
    ce->copyFromTexture(t.color, 0, 0, MTL::Origin::Make(0, 0, 0), MTL::Size::Make(kRes, kRes, 1), t.readback, 0, kRes * 4, kRes * kRes * 4);
    ce->endEncoding();
    (void)r;
}

struct Render {
    std::vector<u32> image, drawn;
};

Render readRender(const Target& t, u32 slots) {
    Render o;
    const auto* px = static_cast<const u32*>(t.readback->contents());
    o.image.assign(px, px + size_t(kRes) * kRes);
    const auto* dr = static_cast<const u32*>(t.drawn->contents());
    o.drawn.assign(dr, dr + slots);
    return o;
}

// ---- one layout -----------------------------------------------------------------------------

struct Totals {
    u64 outside = 0, inside = 0, structural = 0;
    u32 layouts = 0;
    u64 maxSlots = 0;
    bool controlFlip = true, controlScatter = true, controlDrop = true, controlCorrupt = true;
    std::string controlDetail;
};

ph::GPUCullParams cameraParams(Rng& rng, u32 flags, u32 slots, float maxDistance) {
    ph::Camera cam(glm::radians(60.0f), 16.0f / 9.0f, 0.05f, 1000.0f);
    cam.setPosition(glm::vec3(rng.range(-100, 100), rng.range(-100, 100), rng.range(-100, 100)));
    cam.setYawPitch(rng.range(-180, 180), rng.range(-45, 45));
    cam.updateMatrices();
    return ph::makeCullParams(cam.getViewProjection(), cam.getProjection()[1][1], 1080, cam.getPosition(), cam.getFront(), 0.05f, flags, maxDistance,
                              2.0f, slots);
}

void runLayout(Rig& r, Totals& tot, const Geometry& geo, const Layout& L, bool withSafe) {
    Context& ctx = r.ctx;
    const Scene sc = makeScene(L, geo);
    Rng crng(L.seed ^ 0xCA3E5A);
    ph::GPUCullParams params = cameraParams(crng, L.flags, L.slots, 400.0f);
    ctx.log("F5-K2 layout %s: %u slots, %u buckets, %u valid, %u mirrored, flags %u, %u groups", L.name, L.slots, L.buckets, sc.valid, sc.mirrored, L.flags,
            params.groupCount);
    Buffers b = makeBuffers(r, sc, params);
    const u32 queueStride = 4032;
    MTL::Buffer* qbuf = ctx.buffer(size_t(queueStride) * 7);
    MTL::Buffer* qpar = ctx.buffer(16);
    const ph::GPUQueueClearParams qp{7, queueStride, {0, 0}};
    std::memcpy(qpar->contents(), &qp, sizeof(qp));
    std::memset(qbuf->contents(), 0, qbuf->length());
    MTL::Buffer* indexBuf = r.idx ? r.idx : upload(ctx, geo.indices);

    std::vector<u8> refResult;
    std::vector<float> refMargin;
    ph::cullReference(sc.inst, sc.meshes, params, refResult, &refMargin);
    u32 refVisible = 0, refBand = 0;
    for (u32 i = 0; i < L.slots; ++i) {
        refVisible += refResult[i] == 0;
        refBand += refResult[i] != ph::CULL_RESULT_INVALID && std::fabs(refMargin[i]) < ph::CULL_BAND;
    }

    for (int mode = 0; mode < (withSafe ? 2 : 1); ++mode) {
        const Pipes& pl = mode == 0 ? r.fast : r.safe;
        const std::string tag = std::string(L.name) + (mode == 0 ? "/fast" : "/safe");
        fillGarbage(b);
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
        ce->setLabel(NS::String::string("k2 kernels", NS::UTF8StringEncoding));
        encodeKernels(r, ce, pl, b, qbuf, qpar, indexBuf, b.commandCount);
        ce->endEncoding();
        const bool render = L.render && mode == 0;
        Target icbTarget;
        if (render) {
            icbTarget = makeTarget(r, b.slots);
            std::memset(icbTarget.drawn->contents(), 0, icbTarget.drawn->length());
            encodeRender(r, cmd, sc, b, icbTarget, true, nullptr, true);
            encodeReadback(r, cmd, icbTarget);
        }
        ctx.submit();
        const Gpu g = readback(b, b.commandCount);
        const Check c = checkRun(r, sc, params, g, refResult, refMargin, tag, true);
        tot.outside += c.outsideBand;
        tot.inside += c.insideBand;
        tot.structural += c.structural;
        ctx.log("F5-K2 %s: GPU visible %u (CPU %u), mismatches outside band %u, inside band %u (CPU slots in band %u), structural %u", tag.c_str(), c.visible,
                refVisible, c.outsideBand, c.insideBand, refBand, c.structural);
        if (c.outsideBand) r.fail(tag + ": " + std::to_string(c.outsideBand) + " flags differ from cullReference outside the band");
        if (c.structural) r.fail(tag + ": " + std::to_string(c.structural) + " structural errors");

        if (render) {
            // Direct draws from the args of the GPU's own prefix, in a later command buffer.
            std::vector<u32> args;
            ph::drawArgsReference(sc.buckets, sc.commands, g.prefix, args);
            Target direct = makeTarget(r, b.slots);
            std::memset(direct.drawn->contents(), 0, direct.drawn->length());
            MTL4::CommandBuffer* c2 = ctx.beginCommands();
            encodeRender(r, c2, sc, b, direct, false, &args, false);
            encodeReadback(r, c2, direct);
            ctx.submit();
            const Render a = readRender(icbTarget, b.slots), d = readRender(direct, b.slots);
            u32 pixDiff = 0, flagBad = 0, nonZero = 0;
            for (size_t i = 0; i < a.image.size(); ++i) {
                pixDiff += a.image[i] != d.image[i];
                nonZero += d.image[i] != 0;
            }
            for (u32 i = 0; i < b.slots; ++i) {
                const bool expectDrawn = g.flags[i] == 1;
                flagBad += (a.drawn[i] == 1) != expectDrawn || (d.drawn[i] == 1) != expectDrawn;
            }
            ctx.log("F5-K2 %s: ICB vs direct draws: %u pixels differ, %u drawn flags wrong, %u pixels non-empty", tag.c_str(), pixDiff, flagBad, nonZero);
            if (pixDiff || flagBad || nonZero == 0) r.fail(tag + ": ICB render differs from direct draws (px " + std::to_string(pixDiff) + ", flags " + std::to_string(flagBad) + ")");
            tot.structural += pixDiff + flagBad;

            // Negative control: drop the last non-empty command (draw build told fewer commands): the ICB image must differ.
            u32 lastNonEmpty = 0;
            for (u32 k = 0; k < b.commandCount; ++k)
                if (args[k * 2] != 0) lastNonEmpty = k;
            setDrawCommandCount(b, lastNonEmpty, static_cast<u32>(sc.buckets.size()));
            fillGarbage(b);
            Target dropped = makeTarget(r, b.slots);
            std::memset(dropped.drawn->contents(), 0, dropped.drawn->length());
            MTL4::CommandBuffer* c3 = ctx.beginCommands();
            MTL4::ComputeCommandEncoder* ce3 = c3->computeCommandEncoder();
            ce3->setLabel(NS::String::string("k2 kernels", NS::UTF8StringEncoding));
            encodeKernels(r, ce3, pl, b, qbuf, qpar, indexBuf, lastNonEmpty);
            ce3->endEncoding();
            encodeRender(r, c3, sc, b, dropped, true, nullptr, true);
            encodeReadback(r, c3, dropped);
            ctx.submit();
            const Render e = readRender(dropped, b.slots);
            u32 pixDiff2 = 0, flagDiff2 = 0;
            for (size_t i = 0; i < e.image.size(); ++i) pixDiff2 += e.image[i] != d.image[i];
            for (u32 i = 0; i < b.slots; ++i) flagDiff2 += e.drawn[i] != d.drawn[i];
            const bool detected = pixDiff2 > 0 && flagDiff2 > 0;
            ctx.log("F5-K2 control: ICB without command %u: %u pixels, %u drawn flags differ -> %s", lastNonEmpty, pixDiff2, flagDiff2,
                    detected ? "detected" : "NOT detected");
            tot.controlDrop &= detected;
            setDrawCommandCount(b, b.commandCount, static_cast<u32>(sc.buckets.size()));
        }
    }

    // ---- negative controls on the cull result -----------------------------------------------------
    if (L.flags & ph::CULL_FLAG_FRUSTUM && L.slots >= 4096 && refVisible > 0) {
        // Flip the left plane: the GPU result must differ from the unflipped reference outside the band.
        ph::GPUCullParams flipped = params;
        for (int k = 0; k < 4; ++k) flipped.planes[k] = -flipped.planes[k];
        std::memcpy(b.cullParams->contents(), &flipped, sizeof(flipped));
        fillGarbage(b);
        MTL4::CommandBuffer* cmd = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* ce = cmd->computeCommandEncoder();
        ce->setLabel(NS::String::string("k2 kernels", NS::UTF8StringEncoding));
        encodeKernels(r, ce, r.fast, b, qbuf, qpar, indexBuf, b.commandCount);
        ce->endEncoding();
        ctx.submit();
        const Gpu g = readback(b, b.commandCount);
        u32 diff = 0;
        for (u32 i = 0; i < L.slots; ++i)
            diff += resultFromFlag(g.flags[i]) != refResult[i] && refResult[i] != ph::CULL_RESULT_INVALID && std::fabs(refMargin[i]) >= ph::CULL_BAND;
        const bool detected = diff > std::max(1u, refVisible / 100);
        ctx.log("F5-K2 control (%s): left plane flipped -> %u slots differ outside the band (CPU visible %u) -> %s", L.name, diff, refVisible,
                detected ? "detected" : "NOT detected");
        tot.controlFlip &= detected;
        std::memcpy(b.cullParams->contents(), &params, sizeof(params));
        // Compaction control: one corrupted visible entry must break the exact comparison.
        fillGarbage(b);
        cmd = ctx.beginCommands();
        ce  = cmd->computeCommandEncoder();
        ce->setLabel(NS::String::string("k2 kernels", NS::UTF8StringEncoding));
        encodeKernels(r, ce, r.fast, b, qbuf, qpar, indexBuf, b.commandCount);
        ce->endEncoding();
        ctx.submit();
        Gpu g2 = readback(b, b.commandCount);
        if (g2.visible.size() > 1) {
            g2.visible[g2.visible.size() / 2] ^= 1u;
            std::vector<u8> gr(L.slots);
            for (u32 i = 0; i < L.slots; ++i) gr[i] = resultFromFlag(g2.flags[i]);
            std::vector<u32> vis, pre;
            ph::compactReference(gr, vis, pre);
            const bool det = vis != g2.visible;
            tot.controlCorrupt &= det;
            ctx.log("F5-K2 control (%s): corrupted visible entry -> %s", L.name, det ? "detected" : "NOT detected");
        }
    }
    ++tot.layouts;
    tot.maxSlots = std::max<u64>(tot.maxSlots, L.slots);
}

void benchK2(Context& ctx, Report& rep) {
    Rig r(ctx);
    const std::string rel = expandedShader();
    MTL::Library* fast = ctx.library(rel, true);
    MTL::Library* safe = ctx.library(rel, false);
    r.fast             = makePipes(ctx, fast);
    r.safe             = makePipes(ctx, safe);
    const Geometry geo = makeGeometry();
    setupRender(r, geo);

    Totals tot;
    // ---- queue clear ---------------------------------------------------------------------------
    checkQueueClear(r, r.fast, "fast");
    checkQueueClear(r, r.safe, "safe");

    // ---- scatter -------------------------------------------------------------------------------
    {
        u32 bad = 0;
        for (const auto& [count, words, elems] : {std::tuple<u32, u32, u32>{0, 20, 100}, {1, 20, 1}, {37, 20, 5000}, {50000, 20, 262144}, {4099, 1, 8000}, {777, 5, 1000}}) {
            const u32 badRun = runScatter(r, r.fast, count, words, elems, false) + runScatter(r, r.safe, count, words, elems, false);
            if (badRun) r.fail("scatter count " + std::to_string(count) + " words " + std::to_string(words) + ": " + std::to_string(badRun) + " wrong words");
            bad += badRun;
        }
        ctx.log("F5-K2 scatter (6 shapes x fast/safe incl. count 0): %u wrong words", bad);
        // Negative control: the kernel gets count - 1 records, the expectation has all of them.
        const u32 dropBad = runScatter(r, r.fast, 5000, 20, 20000, true);
        tot.controlScatter = dropBad > 0;
        ctx.log("F5-K2 control: scatter without its last record -> %u wrong words -> %s", dropBad, dropBad ? "detected" : "NOT detected");
    }

    // ---- layouts -------------------------------------------------------------------------------
    const u32 all = ph::CULL_FLAG_FRUSTUM | ph::CULL_FLAG_DISTANCE | ph::CULL_FLAG_SIZE;
    std::vector<Layout> layouts = {
        {"one_slot", 1, 1, false, false, all, false, 11},
        {"tiny_33", 33, 3, false, false, ph::CULL_FLAG_FRUSTUM, false, 21},
        {"g2_empty_buckets", 1025, 7, false, true, all, false, 12},
        {"holes_4099", 4099, 9, true, true, all, false, 13},
        {"render_200k", 200000, 20, false, false, all, true, 14},
        {"b1000_100k", 100000, 1000, false, true, all, false, 15},
        {"c1_25M", 1250000, 40, false, true, all, false, 16},
        {"d4M", 4194304, 60, true, true, all, false, 17},
    };
    for (u32 f = 0; f < 8; ++f) layouts.push_back({"flags_combo", 70001, 11, true, true, f, false, 100 + f});
    for (const Layout& L : layouts) runLayout(r, tot, geo, L, L.slots <= 1250000);

    // ---- report -----------------------------------------------------------------------------------
    rep.value("k2.layouts", "count", tot.layouts, {}, false);
    rep.value("k2.max_slots", "count", double(tot.maxSlots), {}, false);
    rep.value("k2.flag_mismatch_outside_band", "count", double(tot.outside), {}, false);
    rep.value("k2.flag_mismatch_inside_band", "count", double(tot.inside), {}, false);
    rep.value("k2.structural_errors", "count", double(tot.structural), {}, false);
    const bool controls = tot.controlFlip && tot.controlScatter && tot.controlDrop && tot.controlCorrupt;
    rep.negative(controls, std::string("flipped left plane detected: ") + (tot.controlFlip ? "yes" : "NO") + "; dropped scatter record detected: " +
                               (tot.controlScatter ? "yes" : "NO") + "; ICB without one command detected (image + drawn flags): " +
                               (tot.controlDrop ? "yes" : "NO") + "; corrupted visible entry detected: " + (tot.controlCorrupt ? "yes" : "NO"));
    if (r.failCount || tot.outside || tot.structural) rep.status(Status::Failed, r.failures.empty() ? "mismatches" : r.failures);
    rep.note("Correctness only (no timings).  Flags compared with cullReference() (fast math and MathModeSafe, 'inside band' = |margin| < CULL_BAND, "
             "reported not failed); counters, visible list, prefix and draw args EXACT against the references applied to the GPU's own flags/prefix; "
             "ICB executed in the same command buffer (class ranges with CPU state) vs direct draws.");
}

} // namespace

} // namespace f5k2

namespace soc {
SOC_BENCH("F5-K2", "kernels.gpu_scene", "F5-K2: production GPU scene kernels (cull, compaction, draw build, scatter, clear) vs portable references",
          f5k2::benchK2);
} // namespace soc
