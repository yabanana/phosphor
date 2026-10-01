// F5-K3: GPU correctness check of shaders/transforms.metal (F5.2 hierarchy +
// motion, F5.4 queues with indirect dispatch).  Correctness only, no timings.
//
// The production kernels are compiled from their real source: the harness
// compiles MSL from source WITHOUT include paths, so this file expands
// `#include "renderer/..."` itself (include expander below) and loads the
// expanded text (a TMPDIR file reached through a relative path from the
// harness shader directory).
//
// The chain is encoded exactly as the host will (Dispatch -> Dispatch encoder
// barriers, every level always encoded, queue 0 written by the CPU with
// gpuQueueWrite, queues 1..7 in one buffer cleared each frame):
//   k3_queue_clear -> scene_motion -> for L = 0..7: scene_queue_args(L) +
//   scene_hier_level(L) (dispatchThreadgroups(indirect from the queue header)).
// (k3_queue_clear stands in for scene_queue_clear of shaders/gpu_scene.metal.)
//
// Checks
//   1. Random forests, depth 0..7 (7 = SCENE_MAX_LEVELS - 1), 10%..100% of the
//      roots with motion, shuffled slot order (a child's slot may be lower than
//      its parent's), chains / wide fan-out, partial dirty-root sets, a
//      duplicated dirty-root list, no motion, no hierarchy: every GPUInstance
//      of the instance buffer compared BIT FOR BIT with referenceWorlds (flags
//      and the other fields must stay untouched), counters (nodesUpdated),
//      per-queue counts, no overflow.
//   2. Queue overflow on a real forest (small capacities): counts, overflow,
//      stored entries valid, the gap words after each queue's capacity intact,
//      processed nodes exact, unprocessed untouched.
//   3. F5.4 synthetic producer/consumer chain with data-dependent counts (some
//      entries append nothing): exact multiset equality, empty queues
//      (indirect dispatch of 0 groups), capacity == total and total - 1
//      (boundary), tiny capacities (overflow counted, never out of bounds).
// Negative controls (all must be detected; built in):
//   a. a dropped child per SIMD-group (PHOSPHOR_TRANSFORMS_TEST_DROP_APPEND
//      compiled into scene_hier_level): the instance comparison must fail;
//   b. one flipped bit of the expected result: the comparison must fail;
//   c. a tampered expected overflow: the queue-chain check must fail;
//   d. no barriers between the dispatches: reported as detected (wrong
//      results) or INCONCLUSIVE (the race did not show; not a proof of safety).

#include "f5_common.h"

#include "renderer/gpu_queue.h"
#include "renderer/gpu_scene_layout.h"
#include "renderer/transform_math.h"
#include "renderer/transform_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef SOC_SOURCE_DIR
#define SOC_SOURCE_DIR "."
#endif

namespace soc {
namespace {

using phosphor::GPUInstance;
using phosphor::GPUMotion;
using phosphor::GPUMotionFrame;
using phosphor::GPUQueueHeader;
using phosphor::GPUSceneCounters;
using phosphor::GPUTransformNode;

constexpr u32 kLevels  = phosphor::SCENE_MAX_LEVELS; // 8 dispatch pairs
constexpr u32 kQueues  = kLevels - 1;                // queues 1..7 in one buffer
constexpr u32 kGapWords = 8;                         // sentinel words after each queue's capacity
constexpr u32 kSentinel = 0xCDCDCDCDu;

// ---- include expander -----------------------------------------------------------------------

std::string readFile(const std::filesystem::path& p) {
    std::ifstream in(p);
    if (!in) throw BenchError("cannot read " + p.string());
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Inlines `#include "renderer/x.h"` (resolved in <repo>/src) once; drops `#pragma once`;
// keeps `#include <...>`.
void expandInto(const std::string& text, std::set<std::string>& done, std::string& out) {
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        size_t i = line.find_first_not_of(" \t");
        if (i != std::string::npos && line.compare(i, 8, "#include") == 0) {
            const size_t q1 = line.find('"', i), q2 = q1 == std::string::npos ? q1 : line.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos) {
                const std::string name = line.substr(q1 + 1, q2 - q1 - 1);
                if (done.insert(name).second)
                    expandInto(readFile(std::filesystem::path(SOC_SOURCE_DIR) / "src" / name), done, out);
                continue;
            }
        }
        if (i != std::string::npos && line.compare(i, 12, "#pragma once") == 0) continue;
        out += line;
        out += '\n';
    }
}

// Loads `file` (relative to the repo root) with its includes expanded and `defines` prepended.
MTL::Library* expandedLibrary(Context& ctx, const std::string& file, const std::string& tag, const std::string& defines) {
    std::set<std::string> done;
    std::string text = defines;
    expandInto(readFile(std::filesystem::path(SOC_SOURCE_DIR) / file), done, text);
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("phosphor_k3_" + tag + "_" + std::to_string(::getpid()) + ".metal");
    {
        std::ofstream o(path);
        o << text;
    }
    // Context::library reads SOC_SHADER_DIR + "/" + file: climb out of it with ".." (extra
    // ".." at the root stay at the root) and continue with the absolute path.
    std::string rel;
    for (int i = 0; i < 40; ++i) rel += "../";
    rel += path.string().substr(1);
    MTL::Library* lib = ctx.library(rel);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return lib;
}

// ---- random data -------------------------------------------------------------------------------

float u01(u64& s) { return float(xorshift64(s) >> 40) / 16777216.0f; }
float uf(u64& s, float lo, float hi) { return lo + (hi - lo) * u01(s); }
u32 un(u64& s, u32 n) { return n ? u32(xorshift64(s) % n) : 0u; }

// TRS matrix, column-major, like TransformComponent::updateMatrix without glm: rotation from a
// random unit quaternion, scale (30% non-uniform, 15% mirrored).
void randomTrs(u64& s, float* m) {
    float q[4];
    float len2 = 0;
    for (float& c : q) {
        c = uf(s, -1, 1);
        len2 += c * c;
    }
    if (len2 < 1e-4f) { q[0] = 1; q[1] = q[2] = q[3] = 0; len2 = 1; }
    const float inv = 1.0f / std::sqrt(len2);
    const float x = q[0] * inv, y = q[1] * inv, z = q[2] * inv, w = q[3] * inv;
    float sc[3] = {uf(s, 0.5f, 1.5f), 0, 0};
    sc[1] = sc[2] = sc[0];
    if (un(s, 100) < 30) { sc[1] = uf(s, 0.5f, 1.5f); sc[2] = uf(s, 0.5f, 1.5f); }
    if (un(s, 100) < 15) sc[0] = -sc[0];
    const float r[9] = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
                        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
                        2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)};
    for (u32 c = 0; c < 3; ++c) {
        for (u32 i = 0; i < 3; ++i) m[4 * c + i] = r[3 * c + i] * sc[c];
        m[4 * c + 3] = 0;
    }
    m[12] = uf(s, -10, 10);
    m[13] = uf(s, -10, 10);
    m[14] = uf(s, -10, 10);
    m[15] = 1;
}

void garbage(u64& s, void* p, size_t bytes) {
    auto* w = static_cast<u32*>(p);
    for (size_t i = 0; i < bytes / 4; ++i) w[i] = u32(xorshift64(s) >> 20);
}

struct Cfg {
    const char* name;
    u32    slots;
    u32    maxDepth;    // 0: no hierarchy
    double motionFrac;  // of the roots
    double dirtyFrac;   // of the static roots with children (motion roots with children are always dirty)
    u32    shape;       // 0 random parents, 1 wide fan-out (few heavy parents), 2 chains
    bool   dup;         // duplicated dirty-root entries
    u64    seed;
};

struct Forest {
    u32 slots = 0;
    std::vector<GPUInstance>      instances; // initial GPU contents
    std::vector<GPUTransformNode> nodes;
    std::vector<GPUMotion>        motions;
    std::vector<u32>              motionSlots, childOffsets, childSlots, parent, depth, dirty, mult;
    u32 maxDepth = 0;
    [[nodiscard]] phosphor::HierarchyView view() const {
        return {instances, nodes, motions, motionSlots, childOffsets, childSlots};
    }
    [[nodiscard]] u32 rootOf(u32 s) const {
        while (parent[s] != ~0u) s = parent[s];
        return s;
    }
};

Forest buildForest(const Cfg& c) {
    Forest f;
    u64 rng = c.seed * 0x9E3779B97F4A7C15ull + 1;
    const u32 N = c.slots;
    f.slots = N;
    f.maxDepth = c.maxDepth;

    // Logical nodes in level order, then a random permutation to slot space.
    std::vector<u32> level(N, 0), lparent(N, ~0u);
    std::vector<u32> levelStart(c.maxDepth + 2, 0);
    if (c.maxDepth == 0) {
        levelStart[1] = N;
    } else {
        const u32 roots = std::max<u32>(1, N * 3 / 10);
        levelStart[1] = roots;
        const u32 rest = N - roots;
        for (u32 d = 1; d <= c.maxDepth; ++d) {
            const u32 cnt = std::max<u32>(1, d == c.maxDepth ? N - levelStart[d] : rest / c.maxDepth);
            levelStart[d + 1] = levelStart[d] + cnt;
        }
    }
    for (u32 d = 0; d <= c.maxDepth; ++d)
        for (u32 i = levelStart[d]; i < levelStart[d + 1]; ++i) level[i] = d;
    for (u32 d = 1; d <= c.maxDepth; ++d) {
        const u32 pc = levelStart[d] - levelStart[d - 1], ps = levelStart[d - 1];
        for (u32 i = levelStart[d]; i < levelStart[d + 1]; ++i) {
            const u32 k = i - levelStart[d];
            u32 p;
            if (k == 0) p = 0;                                    // guarantees the maximum depth
            else if (c.shape == 1) p = un(rng, 100) < 90 ? un(rng, std::min<u32>(pc, 3)) : un(rng, pc);
            else if (c.shape == 2) p = k % pc;
            else p = un(rng, pc);
            lparent[i] = ps + p;
        }
    }
    std::vector<u32> perm(N); // logical -> slot
    for (u32 i = 0; i < N; ++i) perm[i] = i;
    for (u32 i = N; i > 1; --i) std::swap(perm[i - 1], perm[un(rng, i)]);

    f.parent.assign(N, ~0u);
    f.depth.assign(N, 0);
    for (u32 i = 0; i < N; ++i) {
        f.depth[perm[i]] = level[i];
        if (lparent[i] != ~0u) f.parent[perm[i]] = perm[lparent[i]];
    }
    f.childOffsets.assign(N + 1, 0);
    for (u32 s = 0; s < N; ++s)
        if (f.parent[s] != ~0u) ++f.childOffsets[f.parent[s] + 1];
    for (u32 s = 0; s < N; ++s) f.childOffsets[s + 1] += f.childOffsets[s];
    f.childSlots.assign(f.childOffsets[N], 0);
    {
        std::vector<u32> fill(N, 0);
        for (u32 s = 0; s < N; ++s)
            if (f.parent[s] != ~0u) f.childSlots[f.childOffsets[f.parent[s]] + fill[f.parent[s]]++] = s;
    }

    f.instances.resize(N);
    f.nodes.resize(N);
    f.motions.resize(N);
    for (u32 s = 0; s < N; ++s) {
        GPUInstance& in = f.instances[s];
        if (f.parent[s] == ~0u) {
            randomTrs(rng, in.modelMatrix); // static root: the mirror matrix
        } else {
            for (float& v : in.modelMatrix) v = uf(rng, -100, 100); // child: stale, must be recomputed
        }
        in.meshIndex = u32(xorshift64(rng));
        in.materialIndex = u32(xorshift64(rng));
        in.flags = u32(xorshift64(rng));
        in.pad = u32(xorshift64(rng));
        garbage(rng, &f.nodes[s], sizeof(GPUTransformNode)); // roots: unused, must not be read
        garbage(rng, &f.motions[s], sizeof(GPUMotion));
        if (f.parent[s] != ~0u) {
            randomTrs(rng, f.nodes[s].local);
            f.nodes[s].parentSlot = f.parent[s];
            f.nodes[s].depth = f.depth[s];
        }
    }
    // Motion roots.
    for (u32 s = 0; s < N; ++s) {
        if (f.parent[s] != ~0u || u01(rng) >= c.motionFrac) continue;
        GPUMotion& m = f.motions[s];
        for (float& v : m.centre) v = uf(rng, -20, 20);
        m.radius = uf(rng, 0, 15);
        const float ph = uf(rng, 0, 6.28f);
        m.cosPhase = std::cos(ph);
        m.sinPhase = std::sin(ph);
        m.height = uf(rng, -3, 3);
        m.speedClass = un(rng, phosphor::SCENE_MOTION_CLASSES);
        float base[16];
        randomTrs(rng, base);
        for (u32 j = 0; j < 4; ++j)
            for (u32 i = 0; i < 3; ++i) m.base[3 * j + i] = base[4 * j + i];
        f.motionSlots.push_back(s);
    }
    std::vector<u8> isMotion(N, 0);
    for (u32 s : f.motionSlots) isMotion[s] = 1;
    // Dirty roots (queue 0): roots with children that move, plus a random part of the static ones.
    f.mult.assign(N, 0);
    for (u32 s = 0; s < N; ++s) {
        if (f.parent[s] != ~0u || f.childOffsets[s + 1] == f.childOffsets[s]) continue;
        if (isMotion[s] || u01(rng) < c.dirtyFrac) {
            f.dirty.push_back(s);
            f.mult[s] = 1;
            if (c.dup && un(rng, 3) == 0) {
                f.dirty.push_back(s);
                f.mult[s] = 2;
            }
        }
    }
    for (u32 i = u32(f.dirty.size()); i > 1; --i) std::swap(f.dirty[i - 1], f.dirty[un(rng, i)]);
    return f;
}

// ---- GPU side --------------------------------------------------------------------------------

struct QueueSet {
    Context&     ctx;
    u32          cap;    // entries each queue holds
    u32          stride; // bytes between queues 1..7 (a multiple of 32)
    MTL::Buffer *q0 = nullptr, *qs = nullptr;
    QueueSet(Context& c, u32 capacity) : ctx(c), cap(capacity) {
        stride = u32(phosphor::gpuQueueBytes(cap + kGapWords));
        q0 = ctx.buffer(stride);
        qs = ctx.buffer(size_t(stride) * kQueues);
        // Garbage in counts and entries (a clear that does not clear must show), the capacity
        // in place, the sentinel in the gap words after every capacity.
        u64 g = 0x1234567;
        garbage(g, q0->contents(), stride);
        garbage(g, qs->contents(), size_t(stride) * kQueues);
        for (u32 L = 0; L < kLevels; ++L) {
            u32* w = words(L);
            w[phosphor::GPU_QUEUE_WORD_CAPACITY] = cap;
            for (u32 k = 0; k < kGapWords; ++k) w[phosphor::GPU_QUEUE_WORD_ENTRIES + cap + k] = kSentinel;
        }
    }
    [[nodiscard]] u32* words(u32 L) const {
        return L == 0 ? static_cast<u32*>(q0->contents())
                      : reinterpret_cast<u32*>(static_cast<u8*>(qs->contents()) + size_t(L - 1) * stride);
    }
    [[nodiscard]] u64 addr(u32 L) const { return L == 0 ? q0->gpuAddress() : qs->gpuAddress() + u64(L - 1) * stride; }
    void writeQueue0(const std::vector<u32>& entries) const {
        // The CPU-filled queue keeps its gap sentinel: gpuQueueWrite stores at most `cap` entries.
        phosphor::gpuQueueWrite(q0->contents(), entries.data(), u32(entries.size()), cap, phosphor::SCENE_HIER_GROUP);
    }
};

struct Readback {
    std::vector<GPUInstance> instances;
    GPUSceneCounters counters{};
    GPUQueueHeader header[kLevels];
    std::vector<u32> entries[kLevels]; // stored entries (first min(count, cap))
    bool gapOk = true;
    std::string error;
};

struct Pipes {
    MTL::ComputePipelineState *motion, *args, *hier, *clear, *synth;
};

struct HierRun {
    Context&  ctx;
    Pipes     pipes;
    QueueSet  queues;
    MTL::Buffer *instances, *nodes, *motions, *motionSlots, *childOffsets, *childSlots, *counters, *frame, *levelParams,
        *clearParams;

    HierRun(Context& c, const Pipes& p, const Forest& f, u32 capacity, const float sinCos[])
        : ctx(c), pipes(p), queues(c, capacity) {
        const u32 N = f.slots;
        auto upload = [&](const void* src, size_t bytes) {
            MTL::Buffer* b = ctx.buffer(std::max<size_t>(bytes, 16));
            if (bytes) std::memcpy(b->contents(), src, bytes);
            return b;
        };
        instances    = upload(f.instances.data(), size_t(N) * sizeof(GPUInstance));
        nodes        = upload(f.nodes.data(), size_t(N) * sizeof(GPUTransformNode));
        motions      = upload(f.motions.data(), size_t(N) * sizeof(GPUMotion));
        motionSlots  = upload(f.motionSlots.data(), f.motionSlots.size() * 4);
        childOffsets = upload(f.childOffsets.data(), f.childOffsets.size() * 4);
        childSlots   = upload(f.childSlots.data(), f.childSlots.size() * 4);
        counters     = ctx.buffer(sizeof(GPUSceneCounters));
        std::memset(counters->contents(), 0x5A, sizeof(GPUSceneCounters)); // the clear kernel must zero it
        GPUMotionFrame fr{};
        fr.motionCount = u32(f.motionSlots.size());
        std::memcpy(fr.sinCos, sinCos, sizeof(fr.sinCos));
        frame = upload(&fr, sizeof(fr));
        u32 lp[kLevels * 4] = {};
        for (u32 L = 0; L < kLevels; ++L) lp[L * 4] = L;
        levelParams = upload(lp, sizeof(lp));
        phosphor::GPUQueueClearParams cp{kQueues, queues.stride, {0, 0}};
        clearParams = upload(&cp, sizeof(cp));
        queues.writeQueue0(f.dirty);
        ctx.commitResidency();
    }

    static void barrier(MTL4::ComputeCommandEncoder* e, bool on) {
        if (on) e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    }

    // clear -> motion -> levels, as Engine::buildFrameGraph will record it.
    void encode(const Forest& f, bool barriers, u32 levels = kLevels) {
        MTL4::CommandBuffer* c = ctx.beginCommands();
        MTL4::ComputeCommandEncoder* e = c->computeCommandEncoder();
        MTL4::ArgumentTable* tb = ctx.table();
        const auto Tg = MTL::Size::Make(phosphor::SCENE_HIER_GROUP, 1, 1);

        tb->setAddress(clearParams->gpuAddress(), 0);
        tb->setAddress(queues.addr(1), phosphor::SB_CLEAR_QUEUES);
        tb->setAddress(counters->gpuAddress(), phosphor::SB_CLEAR_COUNTERS);
        e->setComputePipelineState(pipes.clear);
        e->setArgumentTable(tb);
        e->dispatchThreads(MTL::Size::Make(kQueues, 1, 1), MTL::Size::Make(kQueues, 1, 1));
        barrier(e, barriers);

        tb->setAddress(frame->gpuAddress(), 0);
        tb->setAddress(motionSlots->gpuAddress(), phosphor::SB_MOTION_SLOTS);
        tb->setAddress(motions->gpuAddress(), phosphor::SB_MOTION_RECORDS);
        tb->setAddress(instances->gpuAddress(), phosphor::SB_MOTION_INSTANCES);
        e->setComputePipelineState(pipes.motion);
        e->setArgumentTable(tb);
        e->dispatchThreads(MTL::Size::Make(std::max<size_t>(f.motionSlots.size(), 1), 1, 1), Tg); // 1 thread when empty
        barrier(e, barriers);

        for (u32 L = 0; L < levels; ++L) {
            const u64 in  = queues.addr(L);
            const u64 out = queues.addr(std::min(L + 1, kQueues)); // level 7: queue 8 does not exist, a dummy
            tb->setAddress(levelParams->gpuAddress() + u64(L) * 16, 0);
            tb->setAddress(in, phosphor::SB_HIER_QUEUE_IN);
            e->setComputePipelineState(pipes.args);
            e->setArgumentTable(tb);
            e->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
            barrier(e, barriers);

            tb->setAddress(out, phosphor::SB_HIER_QUEUE_OUT);
            tb->setAddress(nodes->gpuAddress(), phosphor::SB_HIER_NODES);
            tb->setAddress(instances->gpuAddress(), phosphor::SB_HIER_INSTANCES);
            tb->setAddress(childOffsets->gpuAddress(), phosphor::SB_HIER_CHILD_OFFSETS);
            tb->setAddress(childSlots->gpuAddress(), phosphor::SB_HIER_CHILD_SLOTS);
            tb->setAddress(counters->gpuAddress(), phosphor::SB_HIER_COUNTERS);
            e->setComputePipelineState(pipes.hier);
            e->setArgumentTable(tb);
            e->dispatchThreadgroups(in + phosphor::GPU_QUEUE_ARGS_OFFSET, Tg);
            barrier(e, barriers);
        }
        e->endEncoding();
        ctx.submit();
    }

    Readback read(const Forest& f) const {
        Readback r;
        r.instances.resize(f.slots);
        std::memcpy(r.instances.data(), instances->contents(), size_t(f.slots) * sizeof(GPUInstance));
        std::memcpy(&r.counters, counters->contents(), sizeof(GPUSceneCounters));
        readQueues(queues, r);
        return r;
    }

    static void readQueues(const QueueSet& qs, Readback& r) {
        for (u32 L = 0; L < kLevels; ++L) {
            const u32* w = qs.words(L);
            std::memcpy(&r.header[L], w, sizeof(GPUQueueHeader));
            const u32 stored = std::min(r.header[L].count, qs.cap);
            if (r.header[L].capacity != qs.cap) r.error += " queue " + std::to_string(L) + " capacity changed;";
            r.entries[L].assign(w + phosphor::GPU_QUEUE_WORD_ENTRIES, w + phosphor::GPU_QUEUE_WORD_ENTRIES + stored);
            for (u32 k = 0; k < kGapWords; ++k)
                if (w[phosphor::GPU_QUEUE_WORD_ENTRIES + qs.cap + k] != kSentinel) {
                    r.gapOk = false;
                    r.error += " write past the capacity of queue " + std::to_string(L) + ";";
                }
        }
    }
};

// ---- comparison -------------------------------------------------------------------------------

struct Mismatch {
    u32 slots = 0;
    u32 first = ~0u;
    std::string text() const {
        return slots ? std::to_string(slots) + " slots differ (first " + std::to_string(first) + ")" : "none";
    }
};

// Expected instance contents after the chain.  `processed` != null: only the listed child slots
// were recomputed (overflow runs); otherwise every child below a dirty root.
std::vector<GPUInstance> expectedInstances(const Forest& f, const float* sinCos, const std::vector<u8>* processed) {
    std::vector<float> world;
    phosphor::referenceWorlds(f.view(), sinCos, world);
    std::vector<GPUInstance> out = f.instances;
    for (u32 s = 0; s < f.slots; ++s) {
        bool computed;
        if (f.parent[s] == ~0u) computed = true; // roots: motion world or the mirror matrix
        else if (processed) computed = (*processed)[s] != 0;
        else computed = f.mult[f.rootOf(s)] != 0;
        if (computed) std::memcpy(out[s].modelMatrix, &world[size_t(s) * 16], 64);
    }
    return out;
}

Mismatch compareInstances(const std::vector<GPUInstance>& got, const std::vector<GPUInstance>& want) {
    Mismatch m;
    for (u32 s = 0; s < want.size(); ++s)
        if (std::memcmp(&got[s], &want[s], sizeof(GPUInstance)) != 0) {
            if (!m.slots) m.first = s;
            ++m.slots;
        }
    return m;
}

// Expected queue counts for a full-recompute run: queue L = nodes of depth L under dirty roots,
// weighted by the multiplicity of their root; level L processes queue L.
std::string checkHierQueues(const Forest& f, const Readback& r, u32 capacity, bool tamperOverflow = false) {
    std::string bad = r.error;
    std::vector<u64> perDepth(kLevels, 0);
    u64 updated = 0;
    for (u32 s = 0; s < f.slots; ++s) {
        if (f.parent[s] == ~0u) continue;
        const u32 m = f.mult[f.rootOf(s)];
        perDepth[f.depth[s]] += m;
        updated += m;
    }
    for (u32 L = 1; L < kLevels; ++L) {
        u32 over = u32(perDepth[L] > capacity ? perDepth[L] - capacity : 0);
        if (tamperOverflow && L == 1) ++over;
        if (r.header[L].count != perDepth[L] || r.header[L].overflow != over)
            bad += " queue " + std::to_string(L) + " count " + std::to_string(r.header[L].count) + "/" +
                   std::to_string(perDepth[L]) + " overflow " + std::to_string(r.header[L].overflow) + "/" + std::to_string(over) + ";";
        const u32 groups = phosphor::gpuQueueGroups(r.header[L].count, capacity, phosphor::SCENE_HIER_GROUP);
        if (r.header[L].groups[0] != groups || r.header[L].groups[1] != 1 || r.header[L].groups[2] != 1)
            bad += " queue " + std::to_string(L) + " groups " + std::to_string(r.header[L].groups[0]) + "/" + std::to_string(groups) + ";";
    }
    if (r.counters.nodesUpdated != updated)
        bad += " nodesUpdated " + std::to_string(r.counters.nodesUpdated) + "/" + std::to_string(updated) + ";";
    if (r.counters.queueOverflow != 0) bad += " queueOverflow " + std::to_string(r.counters.queueOverflow) + ";";
    if (r.header[0].count != f.dirty.size()) bad += " queue 0 count changed;";
    return bad;
}

u32 round8(u32 v) { return (v + 7u) & ~7u; }

struct Outcome {
    bool   ok = true;
    std::string what;
};

Outcome runForest(Context& ctx, const Pipes& pipes, const Cfg& c, const Forest& f, bool barriers = true) {
    float sinCos[phosphor::SCENE_MOTION_CLASSES * 2];
    phosphor::motionSinCosTable(7.123 + double(c.seed), sinCos);
    // Capacity: every queue can hold every entry of the run (duplicated roots double them).
    const u32 cap = round8(2 * f.slots + 64);
    HierRun run(ctx, pipes, f, cap, sinCos);
    run.encode(f, barriers);
    const Readback r = run.read(f);
    const std::vector<GPUInstance> want = expectedInstances(f, sinCos, nullptr);
    const Mismatch mm = compareInstances(r.instances, want);
    std::string bad = checkHierQueues(f, r, cap);
    Outcome o;
    if (mm.slots) bad = "instances: " + mm.text() + ";" + bad;
    o.ok = bad.empty();
    o.what = std::string(c.name) + ": " + bad;
    return o;
}

// ---- synthetic F5.4 chain (k3_synth_stage) ---------------------------------------------------

u32 k3hash(u32 x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
u32 k3count(u32 v, u32 stage, u32 mod) {
    const u32 h = k3hash(v * 0x9E3779B1U + stage * 0x85EBCA6BU + 12345U);
    return (h & 3u) == 0u ? 0u : (h >> 2) % mod;
}
u32 k3value(u32 v, u32 j, u32 stage) { return k3hash(v ^ (j * 0x27d4eb2fU) ^ (stage << 24)); }

struct SynthResult {
    std::string bad;
    u64 total[kLevels] = {};
    u64 overflow = 0;
};

// Runs `stages` synthetic stages from `seedEntries` (queue 0) and checks every queue against the
// CPU model built from the entries actually stored in the previous queue.
SynthResult runSynth(Context& ctx, const Pipes& pipes, const std::vector<u32>& seedEntries, u32 capacity, u32 stages, u32 mod,
                     bool tamperOverflow = false) {
    QueueSet qs(ctx, capacity);
    MTL::Buffer* counters = ctx.buffer(sizeof(GPUSceneCounters));
    u32 lp[kLevels * 4] = {};
    u32 sp[kLevels * 4] = {};
    for (u32 L = 0; L < kLevels; ++L) {
        lp[L * 4] = L;
        sp[L * 4 + 0] = L;
        sp[L * 4 + 1] = mod;
    }
    MTL::Buffer* levelParams = ctx.buffer(sizeof(lp));
    MTL::Buffer* synthParams = ctx.buffer(sizeof(sp));
    std::memcpy(levelParams->contents(), lp, sizeof(lp));
    std::memcpy(synthParams->contents(), sp, sizeof(sp));
    phosphor::GPUQueueClearParams cp{kQueues, qs.stride, {0, 0}};
    MTL::Buffer* clearParams = ctx.buffer(sizeof(cp));
    std::memcpy(clearParams->contents(), &cp, sizeof(cp));
    qs.writeQueue0(seedEntries);
    ctx.commitResidency();

    MTL4::CommandBuffer* c = ctx.beginCommands();
    MTL4::ComputeCommandEncoder* e = c->computeCommandEncoder();
    MTL4::ArgumentTable* tb = ctx.table();
    auto barrier = [&] { e->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice); };
    tb->setAddress(clearParams->gpuAddress(), 0);
    tb->setAddress(qs.addr(1), phosphor::SB_CLEAR_QUEUES);
    tb->setAddress(counters->gpuAddress(), phosphor::SB_CLEAR_COUNTERS);
    e->setComputePipelineState(pipes.clear);
    e->setArgumentTable(tb);
    e->dispatchThreads(MTL::Size::Make(kQueues, 1, 1), MTL::Size::Make(kQueues, 1, 1));
    barrier();
    for (u32 s = 0; s < stages; ++s) {
        tb->setAddress(levelParams->gpuAddress() + u64(s) * 16, 0);
        tb->setAddress(qs.addr(s), phosphor::SB_HIER_QUEUE_IN);
        e->setComputePipelineState(pipes.args);
        e->setArgumentTable(tb);
        e->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
        barrier();
        tb->setAddress(synthParams->gpuAddress() + u64(s) * 16, 0);
        tb->setAddress(qs.addr(s + 1), phosphor::SB_HIER_QUEUE_OUT);
        e->setComputePipelineState(pipes.synth);
        e->setArgumentTable(tb);
        e->dispatchThreadgroups(qs.addr(s) + phosphor::GPU_QUEUE_ARGS_OFFSET, MTL::Size::Make(phosphor::SCENE_HIER_GROUP, 1, 1));
        barrier();
    }
    e->endEncoding();
    ctx.submit();

    Readback r;
    HierRun::readQueues(qs, r);
    SynthResult res;
    res.bad = r.error;
    // Queue 0 as written by the CPU.
    {
        const u64 n = seedEntries.size();
        const u32 stored = u32(std::min<u64>(n, capacity));
        if (r.header[0].count != n || r.header[0].overflow != n - stored || r.entries[0].size() != stored ||
            !std::equal(r.entries[0].begin(), r.entries[0].end(), seedEntries.begin()))
            res.bad += " queue 0 header/entries wrong;";
        res.total[0] = n;
    }
    for (u32 s = 0; s < stages; ++s) {
        // Offered to queue s+1 = what the consumer of the entries stored in queue s appends.
        std::vector<u32> offered;
        for (const u32 v : r.entries[s]) {
            const u32 n = k3count(v, s, mod);
            for (u32 j = 0; j < n; ++j) offered.push_back(k3value(v, j, s));
        }
        const u64 total = offered.size();
        const u32 stored = u32(std::min<u64>(total, capacity));
        u32 expOverflow = u32(total - stored);
        if (tamperOverflow) ++expOverflow;
        const GPUQueueHeader& h = r.header[s + 1];
        const std::string q = " queue " + std::to_string(s + 1);
        if (h.count != total) res.bad += q + " count " + std::to_string(h.count) + "/" + std::to_string(total) + ";";
        if (h.overflow != expOverflow) res.bad += q + " overflow " + std::to_string(h.overflow) + "/" + std::to_string(expOverflow) + ";";
        if (r.entries[s + 1].size() != stored) res.bad += q + " stored entries count wrong;";
        // Stored entries: a sub-multiset of the offered ones (all of them without overflow).
        std::vector<u32> a = offered, b = r.entries[s + 1];
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (!std::includes(a.begin(), a.end(), b.begin(), b.end())) res.bad += q + " stores an entry nobody appended;";
        if (expOverflow == 0 && !tamperOverflow && a != b) res.bad += q + " entries differ from the offered multiset;";
        // Groups are written by the args kernel of the stage that consumes the queue.
        if (s + 1 < stages) {
            const u32 g = phosphor::gpuQueueGroups(h.count, capacity, phosphor::SCENE_HIER_GROUP);
            if (h.groups[0] != g || h.groups[1] != 1 || h.groups[2] != 1) res.bad += q + " groups wrong;";
        }
        res.total[s + 1] = total;
        res.overflow += total - stored;
    }
    // Queues past the last stage stay cleared.
    for (u32 L = stages + 1; L < kLevels; ++L)
        if (r.header[L].count != 0 || r.header[L].overflow != 0) res.bad += " queue " + std::to_string(L) + " not empty;";
    return res;
}

// ---- overflow on the real forest ------------------------------------------------------------

std::string runForestOverflow(Context& ctx, const Pipes& pipes, const Cfg& c, const Forest& f, u32 capacity, u64* overflowOut) {
    float sinCos[phosphor::SCENE_MOTION_CLASSES * 2];
    phosphor::motionSinCosTable(1.5, sinCos);
    HierRun run(ctx, pipes, f, capacity, sinCos);
    run.encode(f, true);
    const Readback r = run.read(f);
    std::string bad = r.error;
    std::vector<u8> processed(f.slots, 0);
    u64 overflow = 0;
    for (u32 L = 1; L < kLevels; ++L) {
        // Offered to queue L = children of the entries stored in queue L-1.
        u64 offered = 0;
        std::set<u32> validChildren;
        for (const u32 p : r.entries[L - 1]) {
            offered += f.childOffsets[p + 1] - f.childOffsets[p];
            for (u32 i = f.childOffsets[p]; i < f.childOffsets[p + 1]; ++i) validChildren.insert(f.childSlots[i]);
        }
        const u32 stored = u32(std::min<u64>(offered, capacity));
        if (r.header[L].count != offered || r.header[L].overflow != offered - stored || r.entries[L].size() != stored)
            bad += " queue " + std::to_string(L) + " count/overflow wrong (" + std::to_string(r.header[L].count) + "/" +
                   std::to_string(offered) + ");";
        std::set<u32> seen;
        for (const u32 s : r.entries[L]) {
            if (!validChildren.count(s)) bad += " queue " + std::to_string(L) + " holds a non-child;";
            if (!seen.insert(s).second) bad += " duplicate entry in queue " + std::to_string(L) + ";";
            processed[s] = 1;
        }
        overflow += offered - stored;
    }
    if (r.counters.queueOverflow != overflow) bad += " counters.queueOverflow " + std::to_string(r.counters.queueOverflow) + "/" + std::to_string(overflow) + ";";
    u64 updated = 0;
    for (u8 p : processed) updated += p;
    if (r.counters.nodesUpdated != updated) bad += " nodesUpdated " + std::to_string(r.counters.nodesUpdated) + "/" + std::to_string(updated) + ";";
    const Mismatch mm = compareInstances(r.instances, expectedInstances(f, sinCos, &processed));
    if (mm.slots) bad += " instances: " + mm.text() + ";";
    if (overflowOut) *overflowOut = overflow;
    return bad.empty() ? "" : std::string(c.name) + " cap " + std::to_string(capacity) + ":" + bad;
}

// ---- the benchmark -------------------------------------------------------------------------------

void benchTransforms(Context& ctx, Report& rep) {
    MTL::Library* lib     = expandedLibrary(ctx, "shaders/transforms.metal", "transforms", "");
    MTL::Library* libDrop = expandedLibrary(ctx, "shaders/transforms.metal", "transforms_drop", "#define PHOSPHOR_TRANSFORMS_TEST_DROP_APPEND 1\n");
    MTL::Library* libQ    = expandedLibrary(ctx, "bench/f5_spike/shaders/k3_queues.metal", "queues", "");

    Pipes pipes{};
    pipes.motion = ctx.compute(lib, phosphor::KERNEL_MOTION);
    pipes.args   = ctx.compute(lib, phosphor::KERNEL_QUEUE_ARGS);
    pipes.hier   = ctx.compute(lib, phosphor::KERNEL_HIER_LEVEL);
    pipes.clear  = ctx.compute(libQ, "k3_queue_clear");
    pipes.synth  = ctx.compute(libQ, "k3_synth_stage");
    Pipes dropPipes = pipes;
    dropPipes.hier = ctx.compute(libDrop, phosphor::KERNEL_HIER_LEVEL);

    std::string failures;
    u32 cases = 0, exact = 0;

    // 1. Random forests ------------------------------------------------------------------------
    std::vector<Cfg> cfgs;
    u64 seed = 1;
    const u32 N = ctx.quick() ? 3000 : 20000;
    for (u32 d = 0; d <= phosphor::SCENE_MAX_LEVELS - 1; ++d)
        for (const double mf : {0.1, 0.5, 1.0})
            cfgs.push_back({"random", N + 37 * d, d, mf, 0.5, 0, false, seed++});
    cfgs.push_back({"wide", N, 3, 0.3, 1.0, 1, false, seed++});
    cfgs.push_back({"wide-deep", N, 7, 1.0, 1.0, 1, false, seed++});
    cfgs.push_back({"chains", N, 7, 0.5, 1.0, 2, false, seed++});
    cfgs.push_back({"dup-dirty", N, 5, 0.5, 1.0, 0, true, seed++});
    cfgs.push_back({"no-motion", N, 4, 0.0, 0.3, 0, false, seed++});
    cfgs.push_back({"all-clean", N, 4, 0.0, 0.0, 0, false, seed++}); // empty queue 0 everywhere
    cfgs.push_back({"tiny", 1, 0, 1.0, 0.0, 0, false, seed++});
    cfgs.push_back({"tiny2", 5, 1, 1.0, 1.0, 0, false, seed++});

    Forest negForest;
    Cfg negCfg{};
    for (const Cfg& c : cfgs) {
        const Forest f = buildForest(c);
        const Outcome o = runForest(ctx, pipes, c, f);
        ++cases;
        if (o.ok) ++exact;
        else failures += " " + o.what;
        if (c.maxDepth == 7 && c.shape == 0 && c.motionFrac == 1.0) { negForest = f; negCfg = c; }
    }
    ctx.log("[F5-K3] %u forests: %u bit-exact", cases, exact);
    rep.value("forests.cases", "count", cases);
    rep.value("forests.bit_exact", "count", exact);

    // 2. Overflow on a real forest ---------------------------------------------------------------
    u32 overflowRuns = 0;
    {
        ctx.log("[F5-K3] overflow runs");
        const Cfg c{"overflow", ctx.quick() ? 4000u : 20000u, 4, 0.5, 1.0, 1, false, seed++};
        const Forest f = buildForest(c);
        for (const u32 cap : {8u, 64u, 200u, 1000u, 5000u}) {
            u64 dropped = 0;
            ctx.log("[F5-K3]   overflow cap %u", cap);
            const std::string bad = runForestOverflow(ctx, pipes, c, f, cap, &dropped);
            ++overflowRuns;
            if (!bad.empty()) failures += " " + bad;
            if (cap == 8 && dropped == 0) failures += " overflow run dropped nothing (the control is vacuous);";
        }
    }
    rep.value("overflow.cases", "count", overflowRuns);

    // 3. Synthetic producer/consumer chain --------------------------------------------------------
    u32 synthRuns = 0;
    {
        u64 rng = 99;
        auto seedEntries = [&](u32 n) {
            std::vector<u32> v(n);
            for (u32& e : v) e = u32(xorshift64(rng));
            return v;
        };
        const u32 stages = 4, mod = 5;
        const std::vector<u32> big = seedEntries(ctx.quick() ? 2000 : 12000);
        auto check = [&](const char* what, const SynthResult& r) {
            ++synthRuns;
            ctx.log("[F5-K3]   synth %s: %s", what, r.bad.empty() ? "ok" : r.bad.c_str());
            if (!r.bad.empty()) failures += std::string(" synth ") + what + ":" + r.bad;
        };
        const u32 bigCap = round8(1u << 17);
        SynthResult full = runSynth(ctx, pipes, big, bigCap, stages, mod);
        check("data-dependent", full);
        // Boundary: capacity exactly the total of one level, then one less.
        for (u32 s = 1; s <= stages; ++s) {
            const u32 total = u32(full.total[s]);
            if (total == 0) failures += " synth: empty level in the data-dependent chain (vacuous);";
            const u32 capExact = round8(total);
            // capacity must be a multiple of 8: pick the exact total when it is, else check the rounded one
            check("cap = total (rounded up to 8)", runSynth(ctx, pipes, big, capExact, stages, mod));
            if (capExact > 8) {
                SynthResult r = runSynth(ctx, pipes, big, capExact - 8, stages, mod);
                check("cap < total", r);
                if (r.overflow == 0) failures += " synth: capacity below the total did not overflow;";
            }
        }
        check("tiny capacity", runSynth(ctx, pipes, big, 8, stages, mod));
        check("small capacity", runSynth(ctx, pipes, big, 256, stages, mod));
        check("empty queue 0", runSynth(ctx, pipes, {}, 64, stages, mod));
        check("single entry", runSynth(ctx, pipes, seedEntries(1), 64, stages, mod));
        check("mod 1 (never appends)", runSynth(ctx, pipes, big, bigCap, stages, 1));
        // Entries offered to queue 0 beyond its capacity are dropped by the CPU helper (count/overflow kept).
        check("queue 0 over capacity", runSynth(ctx, pipes, big, 512, stages, mod));
    }
    rep.value("synth.cases", "count", synthRuns);

    // Negative controls ----------------------------------------------------------------------------
    std::string negDetail;
    bool negPass = true;
    {
        // a. dropped append
        float sinCos[phosphor::SCENE_MOTION_CLASSES * 2];
        phosphor::motionSinCosTable(7.123 + double(negCfg.seed), sinCos);
        const u32 cap = round8(2 * negForest.slots + 64);
        HierRun run(ctx, dropPipes, negForest, cap, sinCos);
        run.encode(negForest, true);
        const Readback r = run.read(negForest);
        const Mismatch mm = compareInstances(r.instances, expectedInstances(negForest, sinCos, nullptr));
        const bool a = mm.slots > 0;
        negDetail += std::string("dropped append: ") + (a ? "detected (" + mm.text() + ")" : "NOT DETECTED") + "; ";
        negPass = negPass && a;
        // b. one flipped bit in the expectation
        std::vector<GPUInstance> want = expectedInstances(negForest, sinCos, nullptr);
        HierRun run2(ctx, pipes, negForest, cap, sinCos);
        run2.encode(negForest, true);
        const Readback r2 = run2.read(negForest);
        const bool clean = compareInstances(r2.instances, want).slots == 0;
        u32 bits;
        std::memcpy(&bits, &want[negForest.slots / 2].modelMatrix[5], 4);
        bits ^= 1u;
        std::memcpy(&want[negForest.slots / 2].modelMatrix[5], &bits, 4);
        const bool b = clean && compareInstances(r2.instances, want).slots == 1;
        negDetail += std::string("flipped bit: ") + (b ? "detected" : "NOT DETECTED") + "; ";
        negPass = negPass && b;
        // c. tampered expected overflow
        const bool c = !checkHierQueues(negForest, r2, cap, true).empty();
        negDetail += std::string("tampered overflow (forest): ") + (c ? "detected" : "NOT DETECTED") + "; ";
        negPass = negPass && c;
        std::vector<u32> seedE(500);
        u64 g = 5;
        for (u32& e : seedE) e = u32(xorshift64(g));
        const bool c2 = !runSynth(ctx, pipes, seedE, 1024, 3, 5, true).bad.empty();
        negDetail += std::string("tampered overflow (synthetic): ") + (c2 ? "detected" : "NOT DETECTED") + "; ";
        negPass = negPass && c2;
        // d. no barriers
        u32 racy = 0;
        const u32 reps = 20;
        want = expectedInstances(negForest, sinCos, nullptr);
        for (u32 k = 0; k < reps; ++k) {
            HierRun r3(ctx, pipes, negForest, cap, sinCos);
            r3.encode(negForest, false);
            const Readback rb = r3.read(negForest);
            racy += compareInstances(rb.instances, want).slots > 0;
        }
        negDetail += racy ? "no barriers: detected in " + std::to_string(racy) + "/" + std::to_string(reps) + " runs"
                          : "no barriers: INCONCLUSIVE (0/" + std::to_string(reps) + " runs raced; not a proof of safety)";
        rep.value("negative.nobarrier.wrong_runs", "count", racy);
        if (!racy) rep.note("negative control d (no barriers) INCONCLUSIVE: the race did not show in " + std::to_string(reps) + " runs");
    }
    rep.negative(negPass, negDetail);

    rep.note("scene_hier_level: level L computes the nodes of depth L (level 0 only expands the roots), so depth 7 needs "
             "levels 0..7 (8 dispatch pairs); the last level appends nothing.  Encoded as the host will.");
    rep.note("k3_queue_clear stands in for scene_queue_clear (shaders/gpu_scene.metal, other task).");
    if (!ctx.apple10()) rep.note("apple9 path: the kernels use no Apple10 feature, same code");
    if (!failures.empty()) {
        ctx.log("[F5-K3] FAILURES:%s", failures.c_str());
        rep.status(Status::Failed, failures);
    } else if (!negPass) {
        rep.status(Status::Failed, "negative control not detected: " + negDetail);
    }
}

} // namespace

SOC_BENCH("F5-K3", "kernels.transforms",
          "GPU check of scene_motion / scene_queue_args / scene_hier_level against the CPU reference, bit for bit",
          benchTransforms);

} // namespace soc
