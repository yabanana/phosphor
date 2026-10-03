// meshlet.metal -- F6 mesh path: meshlet candidates (F6.6), object shader
// culling (F6.2), mesh emission (F6.3), phase-B compaction (F6.5).  Contract:
// renderer/meshlet_layout.h (names, slots, sizes, pass order), the culling
// math is renderer/meshlet_cull_math.h (shared with the C++ reference).
//
// Dispatch rules (the host encodes every one of them every frame):
//   meshlet_cand_count / meshlet_cand_write  dispatchThreadgroups(groupCount, 1024): whole groups
//   meshlet_cand_scan / meshlet_b_scan       dispatchThreadgroups(1, 1024)
//   meshlet_b_count / meshlet_b_write        dispatchThreadgroups(candidateGroups, 1024)
//   draws                                    drawMeshThreadgroups(indirect args, 32, 128)
// Between consecutive kernels: Dispatch -> Dispatch barrier.

#include <metal_stdlib>

#include "renderer/gpu_types.h"
#include "renderer/visibility_math.h"
#include "renderer/temporal_layout.h"
#include "renderer/meshlet_cull_math.h"
#include "renderer/meshlet_layout.h"
// The fragment stage is forward_fs: the variant constants are declared here
// too so a specialised function descriptor applies to every stage alike.
#include "pipeline/forward_variants.generated.metal.h"

using namespace metal;
using namespace phosphor;
#include "surface_geometry.h"

static_assert(MB_PARAMS == 0 && MB_INSTANCES == 1 && MB_MESHES == 2 && MB_MATERIALS == 3 && MB_SCENE_FLAGS == 4 &&
                  MB_GROUP_SUMS == 5 && MB_RANGES == 6 && MB_ARGS == 7 && MB_COUNTERS == 8 && MB_GATE == 9 &&
                  MB_CANDIDATES == 10 && MB_B_FLAGS == 11 && MB_B_SUMS == 12 && MB_B_LIST == 13,
              "meshlet kernel slots");
static_assert(MR_FRAME == 0 && MR_VERTICES == 1 && MR_INSTANCES == 2 && MR_MESHLETS == 6 &&
                  MR_MESHLET_VERTICES == 7 && MR_MESHLET_TRIANGLES == 8 && MR_BOUNDS == 9 && MR_PARAMS == 10 &&
                  MR_RANGE == 11 && MR_LIST == 12 && MR_B_FLAGS == 13 && MR_COUNTERS == 14 && MR_DECISIONS == 15 &&
                  MR_TEX_HIZ == 0,
              "meshlet draw slots");
static_assert(MESHLET_SCAN_GROUP == 1024 && MESHLET_OBJECT_GROUP == 32 && MESHLET_MESH_GROUP == 128,
              "kernels written for these sizes");

// ---- candidate list -------------------------------------------------------------------

// Cull class of a slot, as SceneStore decides its bucket (double-sided
// material -> None, mirrored -> BackMirrored, else Back).
static uint slotClass(const device GPUInstance& in, const device GPUMaterial* materials) {
    if ((materials[in.materialIndex].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u) return 2u;
    return (in.flags & INSTANCE_FLAG_MIRRORED) != 0u ? 1u : 0u;
}

// Meshlets of a slot the instance cull kept (0 otherwise) and its class.
static uint slotWeight(constant GPUMeshletCullParams& p, uint slot, const device GPUInstance* instances,
                       const device GPUMeshInfo* meshes, const device GPUMaterial* materials,
                       const device uint* sceneFlags, thread uint& cls) {
    cls = 0u;
    if (slot >= p.slotCount || (sceneFlags[slot] & 1u) == 0u) return 0u;
    const device GPUInstance& in = instances[slot];
    if ((in.flags & INSTANCE_FLAG_VALID) == 0u) return 0u;
    cls = slotClass(in, materials);
    return min(meshes[in.meshIndex].meshletCount, MESHLET_MAX_PER_MESH);
}

// (1) per group of 1024 slots: the meshlets of each class (3 sums).
kernel void meshlet_cand_count(constant GPUMeshletCullParams& p [[buffer(0)]],
                               const device GPUInstance* instances [[buffer(1)]],
                               const device GPUMeshInfo* meshes [[buffer(2)]],
                               const device GPUMaterial* materials [[buffer(3)]],
                               const device uint* sceneFlags [[buffer(4)]], device uint* groupSums [[buffer(5)]],
                               uint tid [[thread_position_in_grid]], uint gid [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[3][32];
    uint cls     = 0u;
    const uint w = slotWeight(p, tid, instances, meshes, materials, sceneFlags, cls);
    for (uint c = 0; c < 3u; ++c) {
        const uint s = simd_sum(cls == c ? w : 0u);
        if (lane == 0u) sums[c][sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        for (uint c = 0; c < 3u; ++c) {
            const uint t = simd_sum(sums[c][lane]);
            if (lane == 0u) groupSums[gid * 3u + c] = t;
        }
    }
}

// Exclusive scan, in place, of `groups` triples (3 independent channels) by ONE
// group of 1024 threads in chunks with running carries; returns the 3 totals
// (valid in every thread).
static uint3 scanTriples(device uint* sums, uint groups, uint tid, uint sg, uint lane, threadgroup uint (&part)[3][32],
                         threadgroup uint (&carry)[3], threadgroup uint (&chunk)[3]) {
    if (tid < 3u) carry[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint base = 0u; base < groups; base += MESHLET_SCAN_GROUP) {
        const uint idx = base + tid;
        uint excl[3];
        uint v[3];
        for (uint c = 0; c < 3u; ++c) {
            v[c]    = idx < groups ? sums[idx * 3u + c] : 0u;
            excl[c] = simd_prefix_exclusive_sum(v[c]);
            if (lane == 31u) part[c][sg] = excl[c] + v[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {
            for (uint c = 0; c < 3u; ++c) {
                const uint s  = part[c][lane];
                part[c][lane] = simd_prefix_exclusive_sum(s);
                const uint t  = simd_sum(s);
                if (lane == 0u) chunk[c] = t;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idx < groups) {
            for (uint c = 0; c < 3u; ++c) sums[idx * 3u + c] = carry[c] + part[c][sg] + excl[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 3u) carry[tid] += chunk[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return uint3(carry[0], carry[1], carry[2]);
}

static void writeArgs(device uint* args, uint draw, uint count) {
    args[draw * 3u + 0u] = (count + MESHLET_OBJECT_GROUP - 1u) / MESHLET_OBJECT_GROUP;
    args[draw * 3u + 1u] = 1u;
    args[draw * 3u + 2u] = 1u;
}

// (2) ONE group: scan the class sums of every slot group, place the class
// regions back to back, write the phase-A ranges + indirect arguments and the
// overflow gate.  Overflow (more candidates than the capacity, impossible
// with a capacity from meshletCandidateCapacity unless the data is corrupt):
// no mesh draw at all this frame and the gate tells the F5 draw build to
// write the indexed draws instead -- the frame stays complete.
kernel void meshlet_cand_scan(constant GPUMeshletCullParams& p [[buffer(0)]], device uint* groupSums [[buffer(5)]],
                              device GPUMeshletDrawRange* ranges [[buffer(6)]], device uint* args [[buffer(7)]],
                              device atomic_uint* counters [[buffer(8)]], device uint* gate [[buffer(9)]],
                              uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint part[3][32];
    threadgroup uint carry[3];
    threadgroup uint chunk[3];
    const uint3 total = scanTriples(groupSums, p.groupCount, tid, sg, lane, part, carry, chunk);
    if (tid == 0u) {
        const ulong sum     = ulong(total.x) + ulong(total.y) + ulong(total.z);
        const bool overflow = sum > ulong(p.candidateCapacity);
        uint first          = 0u;
        for (uint c = 0; c < 3u; ++c) {
            const uint count = overflow ? 0u : total[c];
            ranges[c]        = GPUMeshletDrawRange{first, count, c, MESHLET_PHASE_A};
            writeArgs(args, c, count);
            // Spike S2 mesh-only variant: one mesh threadgroup per candidate.
            // 2D grid (x <= 32768 threadgroups): a 1D mesh grid of ~700K
            // threadgroups drew only part of bench 8 (grid dimension limit).
            args[MESHLET_ARGS_DIRECT + c * 3u + 0u] = min(count, 32768u);
            args[MESHLET_ARGS_DIRECT + c * 3u + 1u] = (count + 32767u) / 32768u;
            args[MESHLET_ARGS_DIRECT + c * 3u + 2u] = 1u;
            // Phase B starts empty (meshlet_b_scan fills it when it runs).
            ranges[3u + c] = GPUMeshletDrawRange{0u, 0u, c, MESHLET_PHASE_B};
            writeArgs(args, 3u + c, 0u);
            first += count;
        }
        gate[0] = overflow ? 1u : 0u;
        atomic_store_explicit(&counters[MESHLET_COUNTER_CANDIDATES], overflow ? 0u : uint(sum), memory_order_relaxed);
        atomic_store_explicit(&counters[MESHLET_COUNTER_OVERFLOW], overflow ? 1u : 0u, memory_order_relaxed);
    }
}

// (3) every kept slot writes its meshlets at classStart + group offset + the
// intra-group exclusive prefix of its class: stable (class, slot, meshlet) order.
kernel void meshlet_cand_write(constant GPUMeshletCullParams& p [[buffer(0)]],
                               const device GPUInstance* instances [[buffer(1)]],
                               const device GPUMeshInfo* meshes [[buffer(2)]],
                               const device GPUMaterial* materials [[buffer(3)]],
                               const device uint* sceneFlags [[buffer(4)]], const device uint* groupOffsets [[buffer(5)]],
                               const device GPUMeshletDrawRange* ranges [[buffer(6)]], const device uint* gate [[buffer(9)]],
                               device GPUMeshletCandidate* candidates [[buffer(10)]], uint tid [[thread_position_in_grid]],
                               uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint part[3][32];
    uint cls     = 0u;
    const uint w = slotWeight(p, tid, instances, meshes, materials, sceneFlags, cls);
    uint mine    = 0u;
    for (uint c = 0; c < 3u; ++c) {
        const uint v    = cls == c ? w : 0u;
        const uint excl = simd_prefix_exclusive_sum(v);
        if (lane == 31u) part[c][sg] = excl + v;
        if (cls == c) mine = excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        for (uint c = 0; c < 3u; ++c) part[c][lane] = simd_prefix_exclusive_sum(part[c][lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (w == 0u || gate[0] != 0u) return;
    const uint pos = ranges[cls].first + groupOffsets[gid * 3u + cls] + part[cls][sg] + mine;
    if (ulong(pos) + ulong(w) > ulong(p.candidateCapacity)) return; // never written out of bounds
    const uint first = meshes[instances[tid].meshIndex].meshletOffset;
    for (uint j = 0; j < w; ++j) candidates[pos + j] = GPUMeshletCandidate{tid, first + j};
    // Self-check negative control (--debug-meshlets-corrupt id): the first
    // candidate names another meshlet of the same mesh (or, for a 1-meshlet
    // mesh, the neighbouring slot): in range, wrong.
    if (p.corruptId != 0u && pos == 0u) {
        candidates[0] = w > 1u ? GPUMeshletCandidate{tid, first + 1u}
                               : GPUMeshletCandidate{tid + 1u < p.slotCount ? tid + 1u : 0u, first};
    }
}

// ---- phase B list ------------------------------------------------------------------

// Class of candidate `i` from the phase-A ranges (contiguous regions); 3 = none.
static uint candidateClass(const device GPUMeshletDrawRange* ranges, uint i) {
    for (uint c = 0; c < 3u; ++c) {
        if (i >= ranges[c].first && i < ranges[c].first + ranges[c].count) return c;
    }
    return 3u;
}

// (1) per group of 1024 candidates: the history-rejected ones of each class.
kernel void meshlet_b_count(constant GPUMeshletCullParams& p [[buffer(0)]],
                            const device GPUMeshletDrawRange* ranges [[buffer(6)]],
                            const device uint* bFlags [[buffer(11)]], device uint* bSums [[buffer(12)]],
                            uint tid [[thread_position_in_grid]], uint gid [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sums[3][32];
    const uint cls = tid < p.candidateCapacity ? candidateClass(ranges, tid) : 3u;
    const uint f   = cls < 3u && bFlags[tid] != 0u ? 1u : 0u;
    for (uint c = 0; c < 3u; ++c) {
        const uint s = simd_sum(cls == c ? f : 0u);
        if (lane == 0u) sums[c][sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        for (uint c = 0; c < 3u; ++c) {
            const uint t = simd_sum(sums[c][lane]);
            if (lane == 0u) bSums[gid * 3u + c] = t;
        }
    }
}

// (2) ONE group: scan, phase-B ranges and indirect arguments.
kernel void meshlet_b_scan(constant GPUMeshletCullParams& p [[buffer(0)]], device GPUMeshletDrawRange* ranges [[buffer(6)]],
                           device uint* args [[buffer(7)]], device uint* bSums [[buffer(12)]],
                           uint tid [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint part[3][32];
    threadgroup uint carry[3];
    threadgroup uint chunk[3];
    const uint3 total = scanTriples(bSums, p.candidateGroups, tid, sg, lane, part, carry, chunk);
    if (tid == 0u) {
        uint first = 0u;
        for (uint c = 0; c < 3u; ++c) {
            ranges[3u + c] = GPUMeshletDrawRange{first, total[c], c, MESHLET_PHASE_B};
            writeArgs(args, 3u + c, total[c]);
            first += total[c];
        }
    }
}

// (3) stable scatter of the history-rejected candidates into the B list.
kernel void meshlet_b_write(constant GPUMeshletCullParams& p [[buffer(0)]],
                            const device GPUMeshletDrawRange* ranges [[buffer(6)]],
                            const device GPUMeshletCandidate* candidates [[buffer(10)]],
                            const device uint* bFlags [[buffer(11)]], const device uint* bOffsets [[buffer(12)]],
                            device GPUMeshletCandidate* bList [[buffer(13)]], uint tid [[thread_position_in_grid]],
                            uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint part[3][32];
    const uint cls = tid < p.candidateCapacity ? candidateClass(ranges, tid) : 3u;
    const uint f   = cls < 3u && bFlags[tid] != 0u ? 1u : 0u;
    uint mine      = 0u;
    for (uint c = 0; c < 3u; ++c) {
        const uint v    = cls == c ? f : 0u;
        const uint excl = simd_prefix_exclusive_sum(v);
        if (lane == 31u) part[c][sg] = excl + v;
        if (cls == c) mine = excl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        for (uint c = 0; c < 3u; ++c) part[c][lane] = simd_prefix_exclusive_sum(part[c][lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (f == 0u) return;
    const uint pos = ranges[3u + cls].first + bOffsets[gid * 3u + cls] + part[cls][sg] + mine;
    if (pos < p.candidateCapacity) bList[pos] = candidates[tid];
}

// ---- object shader ------------------------------------------------------------------

struct MeshletPayload {
    uint slot[MESHLET_OBJECT_GROUP];
    uint meshlet[MESHLET_OBJECT_GROUP];
    uint decision[MESHLET_OBJECT_GROUP];
};
static_assert(sizeof(MeshletPayload) == MESHLET_PAYLOAD_BYTES, "payload layout");

// Minimum of the pyramid texels of a footprint (at most HIZ_TEST_SPAN^2 reads).
static float hizMin(texture2d<float, access::read> hiz, HiZFootprint f) {
    float m = 1.0f;
    for (uint y = f.y0; y <= f.y1 && y < f.y0 + HIZ_TEST_SPAN; ++y) {
        for (uint x = f.x0; x <= f.x1 && x < f.x0 + HIZ_TEST_SPAN; ++x) m = min(m, hiz.read(uint2(x, y), f.level).x);
    }
    return m;
}

// One SIMD-group: candidate (range.first + group * 32 + lane).  Phase A tests
// frustum, cone (not for double-sided materials) and, with two-phase culling
// and a valid history, the history pyramid under the history's view (a hint:
// every history rejection is retested in phase B); it writes the B flag of
// every candidate it owns.  Phase B tests the B list against the current
// pyramid (phase A's depth) under the current view.  Visible meshlets are
// compacted into the payload with a SIMD prefix sum; one mesh threadgroup per
// visible meshlet.
template <uint Pass>
static void
meshletObject(object_data MeshletPayload &payload, mesh_grid_properties grid, constant GPUMeshletCullParams &p,
              const device GPUMeshletDrawRange &range, const device GPUMeshletCandidate *list, device uint *bFlags,
              device atomic_uint *counters, device uint *decisions, const device GPUInstance *instances,
              const device GPUMaterial *materials, const device GPUMeshlet *meshlets,
              const device GPUMeshletBounds *bounds, texture2d<float, access::read> hiz, uint group, uint lane) {
    const uint idx    = group * MESHLET_OBJECT_GROUP + lane;
    const bool valid  = idx < range.count;
    const bool phaseB = range.phase == MESHLET_PHASE_B;
    uint decision     = phaseB ? MESHLET_DECISION_DRAWN_B : MESHLET_DECISION_DRAWN_A;
    GPUMeshletCandidate cand{0u, 0u};
    if (valid) {
        cand                          = list[range.first + idx];
        const device GPUInstance& in  = instances[cand.slot];
        const device GPUMeshletBounds& b = bounds[cand.meshlet];
        const GPUWorldSphere w        = meshletWorldSphere(in.modelMatrix, b);
        if (!phaseB) {
            if ((p.flags & MESHLET_CULL_FRUSTUM) != 0u && meshletFrustumCulled(p, w)) {
                decision = MESHLET_DECISION_FRUSTUM;
            } else if ((p.flags & MESHLET_CULL_CONE) != 0u && range.cullClass != 2u &&
                       meshletConeCulled(in.modelMatrix, b, p.cameraPosition)) {
                decision = MESHLET_DECISION_CONE;
            } else {
                const HiZFootprint cur = hizFootprint(p.viewProj, w, p);
                if ((p.flags & MESHLET_CULL_SIZE) != 0u && cur.usable != 0u && cur.area < p.minPixels * p.minPixels) {
                    decision = MESHLET_DECISION_SIZE;
                } else if ((p.flags & (MESHLET_CULL_OCCLUSION | MESHLET_CULL_HISTORY_VALID)) ==
                           (MESHLET_CULL_OCCLUSION | MESHLET_CULL_HISTORY_VALID)) {
                    const HiZFootprint f = hizFootprint(p.prevViewProj, w, p);
                    if (f.usable != 0u && hizOccluded(f.nearestDepth, hizMin(hiz, f))) decision = MESHLET_DECISION_HISTORY;
                }
            }
            if (Pass != 2u && (p.flags & MESHLET_CULL_OCCLUSION) != 0u) {
                bFlags[range.first + idx] = decision == MESHLET_DECISION_HISTORY ? 1u : 0u;
            }
        } else {
            const HiZFootprint f = hizFootprint(p.viewProj, w, p);
            if (f.usable != 0u && hizOccluded(f.nearestDepth, hizMin(hiz, f))) decision = MESHLET_DECISION_OCCLUDED;
        }
    }
    if (Pass != 2u && valid && (p.flags & MESHLET_CULL_RECORD) != 0u) {
        decisions[(phaseB ? p.candidateCapacity : 0u) + range.first + idx] = decision;
    }
    const bool drawn = valid && (decision == MESHLET_DECISION_DRAWN_A || decision == MESHLET_DECISION_DRAWN_B);
    // Debug view (cull): also the cone rejects in phase A and the occluded
    // meshlets in phase B, coloured by decision (a visible occluded meshlet
    // on screen is a conservativeness bug).
    const bool debugExtra = (p.flags & MESHLET_CULL_DEBUG_ALL) != 0u && valid &&
                            (decision == MESHLET_DECISION_CONE || decision == MESHLET_DECISION_OCCLUDED);
    const bool masked = valid && materials[instances[cand.slot].materialIndex].alphaCutoff > 0.0f;
    const bool materialPass = Pass == 0u || (Pass == 2u ? masked : !masked);
    const bool emit = (drawn || debugExtra) && materialPass;
    const uint pos  = simd_prefix_exclusive_sum(emit ? 1u : 0u);
    if (emit) {
        payload.slot[pos]     = cand.slot;
        payload.meshlet[pos]  = cand.meshlet;
        payload.decision[pos] = Pass == 0u ? decision : (phaseB ? p.candidateCapacity : 0u) + range.first + idx;
    }
    const uint total = simd_sum(emit ? 1u : 0u);
    // Counters: one atomic per SIMD-group and counter.
    const uint prims = simd_sum(drawn ? meshlets[cand.meshlet].triangleCount : 0u);
    const uint nDraw = simd_sum(drawn ? 1u : 0u);
    if (lane == 0u)
        grid.set_threadgroups_per_grid(uint3(total, 1u, 1u));
    if (lane == 0u && Pass != 2u) {
        if (!phaseB) {
            atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_GROUPS_A], 1u, memory_order_relaxed);
            if (nDraw) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_DRAWN_A], nDraw, memory_order_relaxed);
            if (prims) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_PRIMS_A], prims, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_GROUPS_B], 1u, memory_order_relaxed);
            if (nDraw) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_DRAWN_B], nDraw, memory_order_relaxed);
            if (prims) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_PRIMS_B], prims, memory_order_relaxed);
        }
    }
    const uint nF = simd_sum(valid && decision == MESHLET_DECISION_FRUSTUM ? 1u : 0u);
    const uint nC = simd_sum(valid && decision == MESHLET_DECISION_CONE ? 1u : 0u);
    const uint nH = simd_sum(valid && decision == MESHLET_DECISION_HISTORY ? 1u : 0u);
    const uint nT = simd_sum(valid && phaseB ? 1u : 0u);
    const uint nO = simd_sum(valid && decision == MESHLET_DECISION_OCCLUDED ? 1u : 0u);
    const uint nS = simd_sum(valid && decision == MESHLET_DECISION_SIZE ? 1u : 0u);
    if (lane == 0u && Pass != 2u) {
        if (nF) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_FRUSTUM], nF, memory_order_relaxed);
        if (nC) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_CONE], nC, memory_order_relaxed);
        if (nH) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_HISTORY], nH, memory_order_relaxed);
        if (nT) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_TESTED_B], nT, memory_order_relaxed);
        if (nO) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_OCCLUDED_B], nO, memory_order_relaxed);
        if (nS) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_SIZE], nS, memory_order_relaxed);
    }
}

[[object]] void
meshlet_object(object_data MeshletPayload &payload [[payload]], mesh_grid_properties grid,
               constant GPUMeshletCullParams &p [[buffer(10)]], const device GPUMeshletDrawRange &range [[buffer(11)]],
               const device GPUMeshletCandidate *list [[buffer(12)]], device uint *bFlags [[buffer(13)]],
               device atomic_uint *counters [[buffer(14)]], device uint *decisions [[buffer(15)]],
               const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
               const device GPUMeshlet *meshlets [[buffer(6)]], const device GPUMeshletBounds *bounds [[buffer(9)]],
               texture2d<float, access::read> hiz [[texture(0)]], uint group [[threadgroup_position_in_grid]],
               uint lane [[thread_index_in_threadgroup]]) {
    meshletObject<0u>(payload, grid, p, range, list, bFlags, counters, decisions, instances, materials, meshlets,
                      bounds, hiz, group, lane);
}

[[object]] void visibility_object_opaque(
    object_data MeshletPayload &payload [[payload]], mesh_grid_properties grid,
    constant GPUMeshletCullParams &p [[buffer(10)]], const device GPUMeshletDrawRange &range [[buffer(11)]],
    const device GPUMeshletCandidate *list [[buffer(12)]], device uint *bFlags [[buffer(13)]],
    device atomic_uint *counters [[buffer(14)]], device uint *decisions [[buffer(15)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device GPUMeshletBounds *bounds [[buffer(9)]],
    texture2d<float, access::read> hiz [[texture(0)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    meshletObject<1u>(payload, grid, p, range, list, bFlags, counters, decisions, instances, materials, meshlets,
                      bounds, hiz, group, lane);
}

[[object]] void visibility_object_alpha(
    object_data MeshletPayload &payload [[payload]], mesh_grid_properties grid,
    constant GPUMeshletCullParams &p [[buffer(10)]], const device GPUMeshletDrawRange &range [[buffer(11)]],
    const device GPUMeshletCandidate *list [[buffer(12)]], device uint *bFlags [[buffer(13)]],
    device atomic_uint *counters [[buffer(14)]], device uint *decisions [[buffer(15)]],
    const device GPUInstance *instances [[buffer(2)]], const device GPUMaterial *materials [[buffer(3)]],
    const device GPUMeshlet *meshlets [[buffer(6)]], const device GPUMeshletBounds *bounds [[buffer(9)]],
    texture2d<float, access::read> hiz [[texture(0)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    meshletObject<2u>(payload, grid, p, range, list, bFlags, counters, decisions, instances, materials, meshlets,
                      bounds, hiz, group, lane);
}

// ---- mesh shaders ---------------------------------------------------------------------

// Must match VertexOut of shaders/forward.metal member for member (the
// fragment stage is forward_fs).
struct VertexOut {
    float4 position [[position, invariant]];
    float3 worldPos;
    float3 normal;
    float4 tangent;
    float2 uv;
    uint   materialIndex [[flat]];
    uint   mirrored [[flat]];
};

static float4x4 loadMatrix(const device float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

static float4x4 loadMatrix(constant float* m) {
    return float4x4(float4(m[0],  m[1],  m[2],  m[3]),
                    float4(m[4],  m[5],  m[6],  m[7]),
                    float4(m[8],  m[9],  m[10], m[11]),
                    float4(m[12], m[13], m[14], m[15]));
}

using MeshletMesh = metal::mesh<VertexOut, void, MESHLET_MESH_GROUP, MESHLET_MESH_GROUP, topology::triangle>;

// One meshlet per mesh threadgroup: thread t emits vertex t (< vertexCount)
// and triangle t (< triangleCount); exactly the meshlet's primitives are
// declared (no padding primitives; culled meshlets were already omitted by
// the object shader).  The vertex maths are forward_vs's.
[[mesh]] void meshlet_mesh(MeshletMesh out, const object_data MeshletPayload& payload [[payload]],
                           uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                           constant FrameConstants& frame [[buffer(0)]], const device GPUVertex* vertices [[buffer(1)]],
                           const device GPUInstance* instances [[buffer(2)]],
                           const device GPUMeshlet* meshlets [[buffer(6)]],
                           const device uint* meshletVertices [[buffer(7)]],
                           const device uchar* meshletTriangles [[buffer(8)]]) {
    const GPUMeshlet m           = meshlets[payload.meshlet[gid]];
    const device GPUInstance& gi = instances[payload.slot[gid]];
    if (tid == 0u) out.set_primitive_count(m.triangleCount);
    if (tid < m.vertexCount) {
        const device GPUVertex& v = vertices[meshletVertices[m.vertexOffset + tid]];
        const float4x4 model       = loadMatrix(gi.modelMatrix);
        const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        const float4 world         = model * float4(v.px, v.py, v.pz, 1.0);
        VertexOut o;
        o.position      = loadMatrix(frame.viewProjection) * world;
        o.worldPos      = world.xyz;
        o.normal = surfaceNormal(model, float3(v.nx, v.ny, v.nz));
        o.tangent = float4(normalMatrix * float3(v.tx, v.ty, v.tz),
                           v.tw * ((gi.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
        o.uv            = float2(v.u, v.v);
        o.materialIndex = gi.materialIndex;
        o.mirrored      = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
        out.set_vertex(tid, o);
    }
    if (tid < m.triangleCount) {
        const uint base = m.triangleOffset + tid * 3u;
        out.set_index(tid * 3u + 0u, meshletTriangles[base + 0u]);
        out.set_index(tid * 3u + 1u, meshletTriangles[base + 1u]);
        out.set_index(tid * 3u + 2u, meshletTriangles[base + 2u]);
    }
}

// --meshlet-triangle-cull on (F6.3 option, OFF by default: measured slower
// on M5 Max, docs/opt-log.md "F6"): thread t transforms vertex t (<
// vertexCount) and tests triangle t (< triangleCount).  Triangles the
// rasterizer would cull anyway are not emitted -- back faces (class Back) or
// front faces (class BackMirrored, front-face culled), decided on the
// projected triangle with a margin larger than the raster snapping error
// (2 x area in pixels beyond 0.02 x the perimeter in pixels; a vertex
// behind the camera or a near-zero area keeps the triangle) -- and the kept
// ones are compacted in their original order (SIMD prefix sums), so the
// declared primitive count is the exact output.  Double-sided materials
// (class None) keep every triangle.  The vertex maths are forward_vs's.
[[mesh]] void meshlet_mesh_tricull(MeshletMesh out, const object_data MeshletPayload& payload [[payload]],
                           uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           constant FrameConstants& frame [[buffer(0)]], const device GPUVertex* vertices [[buffer(1)]],
                           const device GPUInstance* instances [[buffer(2)]],
                           const device GPUMaterial* materials [[buffer(3)]],
                           const device GPUMeshlet* meshlets [[buffer(6)]],
                           const device uint* meshletVertices [[buffer(7)]],
                           const device uchar* meshletTriangles [[buffer(8)]],
                           constant GPUMeshletCullParams& p [[buffer(10)]],
                           device atomic_uint* counters [[buffer(14)]]) {
    threadgroup float3 clip[MESHLET_MESH_GROUP]; // x, y, w of each vertex
    threadgroup uint partial[MESHLET_MESH_GROUP / 32u];
    const GPUMeshlet m           = meshlets[payload.meshlet[gid]];
    const device GPUInstance& gi = instances[payload.slot[gid]];
    if (tid < m.vertexCount) {
        const device GPUVertex& v = vertices[meshletVertices[m.vertexOffset + tid]];
        const float4x4 model       = loadMatrix(gi.modelMatrix);
        const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        const float4 world         = model * float4(v.px, v.py, v.pz, 1.0);
        VertexOut o;
        o.position      = loadMatrix(frame.viewProjection) * world;
        o.worldPos      = world.xyz;
        o.normal = surfaceNormal(model, float3(v.nx, v.ny, v.nz));
        o.tangent = float4(normalMatrix * float3(v.tx, v.ty, v.tz),
                           v.tw * ((gi.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
        o.uv            = float2(v.u, v.v);
        o.materialIndex = gi.materialIndex;
        o.mirrored      = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
        out.set_vertex(tid, o);
        clip[tid] = float3(o.position.x, o.position.y, o.position.w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Facing cull of triangle `tid` (class from the instance, as SceneStore).
    const bool doubleSided = (materials[gi.materialIndex].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u;
    const float facing     = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0u ? -1.0f : 1.0f; // +: back faces culled
    uchar i0 = 0, i1 = 0, i2 = 0;
    bool keep = false;
    if (tid < m.triangleCount) {
        const uint base = m.triangleOffset + tid * 3u;
        i0   = meshletTriangles[base + 0u];
        i1   = meshletTriangles[base + 1u];
        i2   = meshletTriangles[base + 2u];
        keep = true;
        const float3 a = clip[i0], b = clip[i1], c = clip[i2];
        if (!doubleSided && a.z > 0.0f && b.z > 0.0f && c.z > 0.0f) {
            // Pixel coordinates (y up, the NDC orientation of the CCW front faces).
            const float2 h  = float2(p.viewport[0], p.viewport[1]) * 0.5f;
            const float2 pa = a.xy / a.z * h, pb = b.xy / b.z * h, pc = c.xy / c.z * h;
            const float area2 = (pb.x - pa.x) * (pc.y - pa.y) - (pc.x - pa.x) * (pb.y - pa.y); // > 0: front facing
            const float perim = length(pb - pa) + length(pc - pb) + length(pa - pc);
            if (isfinite(area2) && isfinite(perim) && facing * area2 < -0.02f * perim) keep = false;
        }
    }
    // Stable compaction of the kept triangles.
    const uint k    = keep ? 1u : 0u;
    const uint excl = simd_prefix_exclusive_sum(k);
    if (lane == 31u) partial[sg] = excl + k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0u, total = 0u;
    for (uint s = 0; s < MESHLET_MESH_GROUP / 32u; ++s) {
        if (s < sg) before += partial[s];
        total += partial[s];
    }
    if (keep) {
        const uint t = before + excl;
        out.set_index(t * 3u + 0u, i0);
        out.set_index(t * 3u + 1u, i1);
        out.set_index(t * 3u + 2u, i2);
    }
    if (tid == 0u) {
        out.set_primitive_count(total);
        if (total) atomic_fetch_add_explicit(&counters[MESHLET_COUNTER_EMITTED], total, memory_order_relaxed);
    }
}

// Spike S2 (--meshlet-object off): no object stage; mesh threadgroup g draws
// candidate range.first + g of the phase-A list (no culling at all).
[[mesh]] void meshlet_mesh_direct(MeshletMesh out, uint tid [[thread_index_in_threadgroup]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
                                  constant FrameConstants& frame [[buffer(0)]],
                                  const device GPUVertex* vertices [[buffer(1)]],
                                  const device GPUInstance* instances [[buffer(2)]],
                                  const device GPUMeshlet* meshlets [[buffer(6)]],
                                  const device uint* meshletVertices [[buffer(7)]],
                                  const device uchar* meshletTriangles [[buffer(8)]],
                                  const device GPUMeshletDrawRange& range [[buffer(11)]],
                                  const device GPUMeshletCandidate* list [[buffer(12)]]) {
    const uint gid = tg.y * grid.x + tg.x;
    if (gid >= range.count) {
        if (tid == 0u) out.set_primitive_count(0u);
        return;
    }
    const GPUMeshletCandidate c  = list[range.first + gid];
    const GPUMeshlet m           = meshlets[c.meshlet];
    const device GPUInstance& gi = instances[c.slot];
    if (tid == 0u) out.set_primitive_count(m.triangleCount);
    if (tid < m.vertexCount) {
        const device GPUVertex& v = vertices[meshletVertices[m.vertexOffset + tid]];
        const float4x4 model       = loadMatrix(gi.modelMatrix);
        const float3x3 normalMatrix = float3x3(model[0].xyz, model[1].xyz, model[2].xyz);
        const float4 world         = model * float4(v.px, v.py, v.pz, 1.0);
        VertexOut o;
        o.position      = loadMatrix(frame.viewProjection) * world;
        o.worldPos      = world.xyz;
        o.normal = surfaceNormal(model, float3(v.nx, v.ny, v.nz));
        o.tangent = float4(normalMatrix * float3(v.tx, v.ty, v.tz),
                           v.tw * ((gi.flags & INSTANCE_FLAG_MIRRORED) ? -1.0f : 1.0f));
        o.uv            = float2(v.u, v.v);
        o.materialIndex = gi.materialIndex;
        o.mirrored      = (gi.flags & INSTANCE_FLAG_MIRRORED) != 0 ? 1u : 0u;
        out.set_vertex(tid, o);
    }
    if (tid < m.triangleCount) {
        const uint base = m.triangleOffset + tid * 3u;
        out.set_index(tid * 3u + 0u, meshletTriangles[base + 0u]);
        out.set_index(tid * 3u + 1u, meshletTriangles[base + 1u]);
        out.set_index(tid * 3u + 2u, meshletTriangles[base + 2u]);
    }
}

// ---- debug views (F6.7) ---------------------------------------------------------------

struct DebugVertexOut {
    float4 position [[position, invariant]];
    float3 color [[flat]];
};

using MeshletDebugMesh = metal::mesh<DebugVertexOut, void, MESHLET_MESH_GROUP, MESHLET_MESH_GROUP, topology::triangle>;

static float3 hashColor(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return float3(float(x & 255u), float((x >> 8) & 255u), float((x >> 16) & 255u)) / 255.0 * 0.75 + 0.25;
}

static float3 decisionColor(uint d) {
    switch (d) {
    case MESHLET_DECISION_DRAWN_A:  return float3(0.20, 0.80, 0.25); // green: phase A
    case MESHLET_DECISION_DRAWN_B:  return float3(1.00, 0.65, 0.10); // orange: recovered by phase B
    case MESHLET_DECISION_CONE:     return float3(0.20, 0.40, 1.00); // blue: back-facing cone
    case MESHLET_DECISION_OCCLUDED: return float3(1.00, 0.05, 0.05); // red: occluded (must stay hidden)
    default:                        return float3(1.00, 0.00, 1.00);
    }
}

// --debug-view meshlets: a colour per (slot, meshlet); cull
// (MESHLET_CULL_DEBUG_ALL): a colour per decision.
[[mesh]] void meshlet_mesh_debug(MeshletDebugMesh out, const object_data MeshletPayload& payload [[payload]],
                                 uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                                 constant FrameConstants& frame [[buffer(0)]],
                                 constant GPUMeshletCullParams& p [[buffer(10)]],
                                 const device GPUVertex* vertices [[buffer(1)]],
                                 const device GPUInstance* instances [[buffer(2)]],
                                 const device GPUMeshlet* meshlets [[buffer(6)]],
                                 const device uint* meshletVertices [[buffer(7)]],
                                 const device uchar* meshletTriangles [[buffer(8)]]) {
    const GPUMeshlet m           = meshlets[payload.meshlet[gid]];
    const device GPUInstance& gi = instances[payload.slot[gid]];
    const float3 color = (p.flags & MESHLET_CULL_DEBUG_ALL) != 0u
                             ? decisionColor(payload.decision[gid])
                             : hashColor(payload.meshlet[gid] * 0x9E3779B9u ^ payload.slot[gid]);
    if (tid == 0u) out.set_primitive_count(m.triangleCount);
    if (tid < m.vertexCount) {
        const device GPUVertex& v = vertices[meshletVertices[m.vertexOffset + tid]];
        const float4 world        = loadMatrix(gi.modelMatrix) * float4(v.px, v.py, v.pz, 1.0);
        DebugVertexOut o;
        o.position = loadMatrix(frame.viewProjection) * world;
        o.color    = color;
        out.set_vertex(tid, o);
    }
    if (tid < m.triangleCount) {
        const uint base = m.triangleOffset + tid * 3u;
        out.set_index(tid * 3u + 0u, meshletTriangles[base + 0u]);
        out.set_index(tid * 3u + 1u, meshletTriangles[base + 1u]);
        out.set_index(tid * 3u + 2u, meshletTriangles[base + 2u]);
    }
}

fragment half4 meshlet_debug_fs(DebugVertexOut in [[stage_in]]) { return half4(half3(in.color), 1.0h); }

// F7: primitive data identifies the exact instance/meshlet reference and
// triangle. It does not depend on the rasterizer's primitive_id numbering.
struct VisibilityVertex {
    float4 position [[position, invariant]];
    float2 uv;
    uint materialIndex [[flat]];
};
struct VisibilityPrimitive {
    uint id [[flat]];
};
struct VisibilityFragment {
    VisibilityVertex vert;
    VisibilityPrimitive primitive;
};
using VisibilityMesh =
    metal::mesh<VisibilityVertex, VisibilityPrimitive, MESHLET_MESH_GROUP, MESHLET_MESH_GROUP, topology::triangle>;

[[mesh]] void
visibility_mesh(VisibilityMesh out, const object_data MeshletPayload &payload [[payload]],
                uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                constant FrameConstants &frame [[buffer(0)]], const device GPUVertex *vertices [[buffer(1)]],
                const device GPUInstance *instances [[buffer(2)]], const device GPUMeshlet *meshlets [[buffer(6)]],
                const device uint *meshletVertices [[buffer(7)]], const device uchar *triangles [[buffer(8)]]) {
    const GPUMeshlet m = meshlets[payload.meshlet[gid]];
    const GPUInstance gi = instances[payload.slot[gid]];
    if (tid == 0)
        out.set_primitive_count(m.triangleCount);
    if (tid < m.vertexCount) {
        const GPUVertex v = vertices[meshletVertices[m.vertexOffset + tid]];
        VisibilityVertex o;
        const float4x4 model =
            float4x4(float4(gi.modelMatrix[0], gi.modelMatrix[1], gi.modelMatrix[2], gi.modelMatrix[3]),
                     float4(gi.modelMatrix[4], gi.modelMatrix[5], gi.modelMatrix[6], gi.modelMatrix[7]),
                     float4(gi.modelMatrix[8], gi.modelMatrix[9], gi.modelMatrix[10], gi.modelMatrix[11]),
                     float4(gi.modelMatrix[12], gi.modelMatrix[13], gi.modelMatrix[14], gi.modelMatrix[15]));
        o.position = loadMatrix(frame.viewProjection) * model * float4(v.px, v.py, v.pz, 1);
        o.uv = float2(v.u, v.v);
        o.materialIndex = gi.materialIndex;
        out.set_vertex(tid, o);
    }
    if (tid < m.triangleCount) {
        for (uint k = 0; k < 3; ++k)
            out.set_index(tid * 3 + k, triangles[m.triangleOffset + tid * 3 + k]);
        out.set_primitive(tid, VisibilityPrimitive{visibilityPack(payload.decision[gid], tid)});
    }
}
fragment uint visibility_opaque_fs(VisibilityFragment in [[stage_in]]) {
    return in.primitive.id;
}
struct VisibilityTextureHandle {
    texture2d<float> tex;
};
fragment uint visibility_alpha_fs(VisibilityFragment in [[stage_in]], const device GPUMaterial *materials [[buffer(3)]],
                                  const device VisibilityTextureHandle *textures [[buffer(5)]],
                                  constant GPUTemporalParams &temporal [[buffer(16)]]) {
    const GPUMaterial material = materials[in.vert.materialIndex];
    constexpr sampler sampling(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));
    const float alpha =
        material.baseColor[3] *
        (material.baseColorTex == INVALID_TEXTURE_INDEX
             ? 1.0f
             : float(half(textures[material.baseColorTex].tex.sample(sampling, in.vert.uv, bias(temporal.mipBias)).a)));
    if (alpha < material.alphaCutoff)
        discard_fragment();
    return in.primitive.id;
}
