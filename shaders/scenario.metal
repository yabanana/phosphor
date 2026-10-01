// scenario.metal -- synthetic passes of the OPT-1 graph scenarios
// (src/rendergraph/scenario.h, src/platform/metal/scenario_passes.cpp).
//
// Every value written is a deterministic function of the pass seed, the
// pixel and the values read, so any correct schedule of a scenario (order,
// fusion, aliasing, queues, rematerialisation) produces the same image:
//   color/storage channel = byte of hash(acc, slot, pixel) / 255
//   acc = mix(seed, [fragment depth key], each texel read, each fetched
//             attachment) ^ (ALU chains & zero)
//   depth written by Geometry passes = 0.1 + 0.8 * k / 1023 (k per cell)
// Readers turn values back into integers (round(v * 255), depth key), so the
// storage format never changes a hash (all formats hold k/255 within half a
// step).  "Signal" outputs (depthOnly) hash only the pixel's depth key: a
// consumer can recompute them from the depth (OPT-1.2 rematerialisation).
//
// Bindings (Metal 4 argument tables): buffer(0) SynthArgs; texture(0..5)
// color inputs, texture(6..10) depth inputs, texture(11..12) storage outputs
// (GPUSynthArgs in renderer/gpu_types.h).

#include <metal_stdlib>
#include "renderer/gpu_types.h"

using namespace metal;

constant uint kFetchMask [[function_constant(0)]]; // color slots read per pixel
constant uint kOutMask   [[function_constant(1)]]; // color slots written

constant bool kFetch0 = (kFetchMask & 1u) != 0;
constant bool kFetch1 = (kFetchMask & 2u) != 0;
constant bool kFetch2 = (kFetchMask & 4u) != 0;
constant bool kFetch3 = (kFetchMask & 8u) != 0;
constant bool kOut0 = (kOutMask & 1u) != 0;
constant bool kOut1 = (kOutMask & 2u) != 0;
constant bool kOut2 = (kOutMask & 4u) != 0;
constant bool kOut3 = (kOutMask & 8u) != 0;
constant bool kOut4 = (kOutMask & 16u) != 0;

constant uint kMaxFootprint = 16;

using phosphor::GPUSynthArgs;
using phosphor::GPUSynthInput;

static uint mixh(uint h, uint v) {
    v *= 0xcc9e2d51u;
    v = (v << 15) | (v >> 17);
    v *= 0x1b873593u;
    h ^= v;
    h = (h << 13) | (h >> 19);
    return h * 5u + 0xe6546b64u;
}

static uint fmix(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

static uint quantize(float4 v) {
    const uint4 q = uint4(rint(saturate(v) * 255.0f));
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}

static uint depthKey(float z) {
    return uint(int(rint((z - 0.1f) * (1023.0f / 0.8f))));
}

static float4 outputValue(uint acc, uint slot, uint2 p) {
    const uint h = fmix(mixh(mixh(mixh(acc, slot + 1u), p.x), p.y));
    return float4(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u), float(h >> 24)) / 255.0f;
}

// What a format with `channels` channels returns when read: missing
// channels are 0, alpha 1.
static float4 asRead(float4 v, uint channels) {
    return float4(v.x, channels > 1 ? v.y : 0.0f, channels > 2 ? v.z : 0.0f, channels > 3 ? v.w : 1.0f);
}

// 4 independent LCG chains, seeded per thread (uniform chains would be
// computed once per SIMD-group: OPT-0, B-01).
static uint aluChains(uint seed, uint iterations) {
    uint a = seed, b = seed ^ 0x9e3779b9u, c = seed ^ 0x7f4a7c15u, d = seed ^ 0x94d049bbu;
    for (uint i = 0; i < iterations; ++i) {
        a = a * 1664525u + 1013904223u;
        b = b * 22695477u + 1u;
        c = c * 1103515245u + 12345u;
        d = d * 134775813u + 1u;
    }
    return a ^ b ^ c ^ d;
}

static uint readInputs(uint acc, uint2 p, constant GPUSynthArgs& a,
                       array<texture2d<float, access::read>, phosphor::SYNTH_COLOR_INPUTS> colorIn,
                       array<depth2d<float, access::read>, phosphor::SYNTH_DEPTH_INPUTS> depthIn) {
    for (uint i = 0; i < a.inputCount && i < phosphor::SYNTH_MAX_INPUTS; ++i) {
        const GPUSynthInput in = a.inputs[i];
        // A depth read only to recompute a signal is not part of the value
        // (the stored signal would not have carried it).
        if (in.kind == phosphor::SYNTH_INPUT_SOURCE) continue;
        const uint2 size = uint2(in.width, in.height);
        const uint2 out  = uint2(a.outWidth, a.outHeight);
        const uint2 base = p * size / out;
        const uint fx = clamp(size.x / out.x, 1u, kMaxFootprint);
        const uint fy = clamp(size.y / out.y, 1u, kMaxFootprint);
        for (uint y = 0; y < fy; ++y) {
            for (uint x = 0; x < fx; ++x) {
                const uint2 c = min(base + uint2(x, y), size - 1u);
                if (in.kind == phosphor::SYNTH_INPUT_COLOR) {
                    acc = mixh(acc, quantize(colorIn[in.bind].read(c)));
                } else if (in.kind == phosphor::SYNTH_INPUT_DEPTH) {
                    acc = mixh(acc, depthKey(depthIn[in.bind].read(c)));
                } else {
                    // Rematerialised signal: the producer's value function on
                    // the depth at the same texel, as its format would return it.
                    const GPUSynthInput d = a.inputs[in.rematDepth];
                    const uint key = depthKey(depthIn[d.bind].read(c));
                    const float4 v = asRead(outputValue(mixh(in.rematSeed, key), in.rematSlot, c), in.channels);
                    acc = mixh(acc, quantize(v));
                }
            }
        }
    }
    return acc;
}

// Seed of the depth-only signals of a non-Geometry pass.
static uint signalAcc(uint2 p, constant GPUSynthArgs& a, array<depth2d<float, access::read>, phosphor::SYNTH_DEPTH_INPUTS> depthIn) {
    if (a.signalDepthInput >= a.inputCount) return a.seed;
    const GPUSynthInput d = a.inputs[a.signalDepthInput];
    return mixh(a.seed, depthKey(depthIn[d.bind].read(min(p, uint2(d.width, d.height) - 1u))));
}

// --- Raster -------------------------------------------------------------------

struct VsOut {
    float4 position [[position]];
};

vertex VsOut synth_fullscreen_vs(uint vid [[vertex_id]]) {
    const float2 p = float2((vid << 1) & 2, vid & 2);
    return {float4(p * 2.0f - 1.0f, 0.0f, 1.0f)};
}

// A grid of gridW x gridH cells (two triangles each) covering the target;
// one flat depth per cell.
vertex VsOut synth_geometry_vs(uint vid [[vertex_id]], constant GPUSynthArgs& a [[buffer(0)]]) {
    const uint tri = vid / 3u, corner = vid % 3u;
    const uint cell = tri >> 1u;
    const uint cx = cell % a.gridW, cy = cell / a.gridW;
    uint2 c;
    if ((tri & 1u) == 0u) {
        c = corner == 0u ? uint2(0, 0) : corner == 1u ? uint2(1, 0) : uint2(0, 1);
    } else {
        c = corner == 0u ? uint2(1, 0) : corner == 1u ? uint2(1, 1) : uint2(0, 1);
    }
    const uint k = fmix(mixh(a.geometrySeed, cell)) & 1023u;
    const uint work = aluChains(mixh(a.seed, vid), a.vertexIterations) & a.zero;
    const float z = 0.1f + 0.8f * float(k + work) / 1023.0f;
    const float x = float(cx + c.x) / float(a.gridW) * 2.0f - 1.0f;
    const float y = float(cy + c.y) / float(a.gridH) * 2.0f - 1.0f;
    return {float4(x, y, z, 1.0f)};
}

struct FsIn {
    float4 position [[position]];
    float4 f0 [[color(0), function_constant(kFetch0)]];
    float4 f1 [[color(1), function_constant(kFetch1)]];
    float4 f2 [[color(2), function_constant(kFetch2)]];
    float4 f3 [[color(3), function_constant(kFetch3)]];
};

struct FsOut {
    float4 o0 [[color(0), function_constant(kOut0)]];
    float4 o1 [[color(1), function_constant(kOut1)]];
    float4 o2 [[color(2), function_constant(kOut2)]];
    float4 o3 [[color(3), function_constant(kOut3)]];
    float4 o4 [[color(4), function_constant(kOut4)]];
};

fragment FsOut synth_fs(FsIn in [[stage_in]], constant GPUSynthArgs& a [[buffer(0)]],
                        array<texture2d<float, access::read>, phosphor::SYNTH_COLOR_INPUTS> colorIn [[texture(0)]],
                        array<depth2d<float, access::read>, phosphor::SYNTH_DEPTH_INPUTS> depthIn [[texture(6)]]) {
    const uint2 p = uint2(in.position.xy);
    uint acc = a.seed;
    uint accZ = a.seed;
    if (a.geometry != 0u) {
        const uint key = depthKey(in.position.z);
        acc  = mixh(acc, key);
        accZ = mixh(accZ, key);
    } else {
        accZ = signalAcc(p, a, depthIn);
    }
    acc = readInputs(acc, p, a, colorIn, depthIn);
    if (kFetch0) acc = mixh(acc, quantize(in.f0));
    if (kFetch1) acc = mixh(acc, quantize(in.f1));
    if (kFetch2) acc = mixh(acc, quantize(in.f2));
    if (kFetch3) acc = mixh(acc, quantize(in.f3));
    acc ^= aluChains(mixh(a.seed, p.x | (p.y << 16)), a.iterations) & a.zero;

    FsOut out;
    if (kOut0) out.o0 = outputValue((a.depthOnlyMask & 1u) ? accZ : acc, 0u, p);
    if (kOut1) out.o1 = outputValue((a.depthOnlyMask & 2u) ? accZ : acc, 1u, p);
    if (kOut2) out.o2 = outputValue((a.depthOnlyMask & 4u) ? accZ : acc, 2u, p);
    if (kOut3) out.o3 = outputValue((a.depthOnlyMask & 8u) ? accZ : acc, 3u, p);
    if (kOut4) out.o4 = outputValue((a.depthOnlyMask & 16u) ? accZ : acc, 4u, p);
    return out;
}

// Depth-only Geometry passes (shadow maps, depth prepass).
fragment void synth_depth_fs() {}

// --- Compute ------------------------------------------------------------------

kernel void synth_cs(constant GPUSynthArgs& a [[buffer(0)]],
                     array<texture2d<float, access::read>, phosphor::SYNTH_COLOR_INPUTS> colorIn [[texture(0)]],
                     array<depth2d<float, access::read>, phosphor::SYNTH_DEPTH_INPUTS> depthIn [[texture(6)]],
                     array<texture2d<float, access::write>, phosphor::SYNTH_STORAGE_OUTPUTS> storageOut [[texture(11)]],
                     uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= a.outWidth || tid.y >= a.outHeight) return;
    uint acc = readInputs(a.seed, tid, a, colorIn, depthIn);
    const uint accZ = signalAcc(tid, a, depthIn);
    acc ^= aluChains(mixh(a.seed, tid.x | (tid.y << 16)), a.iterations) & a.zero;
    for (uint s = 0; s < a.storageCount && s < phosphor::SYNTH_STORAGE_OUTPUTS; ++s) {
        storageOut[s].write(outputValue((a.depthOnlyMask >> s) & 1u ? accZ : acc, s, tid), tid);
    }
}
