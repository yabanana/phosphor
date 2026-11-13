#pragma once
// F13 denoise history layout helpers, shared by C++ and MSL (gpu_types.h holds
// the structs). The state word of GPUDenoiseHistoryRecord packs
//   bits  0-15 length (<= maxHistory <= 256)
//   bits 16-19 flags  (bit0 numerical error, DENOISE_HISTORY_STOCHASTIC)
//   bits 20-22 sampled-lobe key (F15.1 ticket 08: lobe id | axis << 2, specular signal only, else 0)
//   bit  23    unused
//   bits 24-30 path   (REFLECTION_PATH_*, <= 5)
//   bit  31    valid
// losslessly for every value the temporal pass writes (tests/test_denoise.cpp).
#include "renderer/gpu_types.h"
namespace phosphor {
PHOSPHOR_GPU_CONSTANT u32 DENOISE_HISTORY_MAX_LENGTH = 0xffffu, DENOISE_HISTORY_MAX_FLAGS = 0xfu, DENOISE_HISTORY_MAX_PATH = 0x7fu, DENOISE_HISTORY_MAX_LOBE = 7u;
inline u32 denoisePackState(u32 length, u32 flags, u32 path, u32 valid, u32 lobe = 0u) {
    return (length & 0xffffu) | ((flags & 0xfu) << 16) | ((lobe & 7u) << 20) | ((path & 0x7fu) << 24) | (valid ? 0x80000000u : 0u);
}
inline u32 denoiseStateLength(u32 state) { return state & 0xffffu; }
inline u32 denoiseStateFlags(u32 state) { return (state >> 16) & 0xfu; }
inline u32 denoiseStateLobe(u32 state) { return (state >> 20) & 7u; }
inline u32 denoiseStatePath(u32 state) { return (state >> 24) & 0x7fu; }
inline u32 denoiseStateValid(u32 state) { return state >> 31; }
// Sampled-lobe key of a specular sample (GPUSpecularSample.flags): lobe id in bits 0-1, axis (refract) in bit 2.
// Two pixels (or a pixel and its history) are only comparable when their keys are equal.
inline u32 specularSampleLobeKey(u32 sampleFlags) {
    return ((sampleFlags >> SPECULAR_SAMPLE_LOBE_SHIFT) & SPECULAR_SAMPLE_LOBE_MASK) | ((sampleFlags & SPECULAR_SAMPLE_AXIS_REFRACT) ? 4u : 0u);
}
// The flag bits a transport writes for a sampled lobe (`refract` selects the refraction axis).
inline u32 specularSampleLobeFlags(u32 lobeId, bool refract) {
    return ((lobeId & SPECULAR_SAMPLE_LOBE_MASK) << SPECULAR_SAMPLE_LOBE_SHIFT) | (refract ? SPECULAR_SAMPLE_AXIS_REFRACT : 0u);
}
// Logical history of one pixel: every field the temporal pass computes. The
// CPU mirror (denoise_settings.h) and the MSL kernels run on this struct; only
// loads and stores go through the packed layout.
struct DenoiseHistory {
    float color[3], firstMoment;
    float secondMoment, variance; u32 length, signal;
    float normal[3], depth;
    u32 slot, instanceGeneration, materialRevision, signalRevision;
    u32 viewID, historyEpoch, valid, flags;
    float position[3], roughness;
    float hitDistance; u32 path, secondarySlot, secondaryGeneration;
    u32 lobe; // specularSampleLobeKey of the sample the record was accumulated from (specular signal)
};
inline GPUDenoiseHistoryHeader denoisePackHeader(DenoiseHistory h) {
    GPUDenoiseHistoryHeader out{};
    out.signal = h.signal; out.viewID = h.viewID; out.historyEpoch = h.historyEpoch; out.signalRevision = h.signalRevision;
    return out;
}
inline GPUDenoiseHistoryRecord denoisePackRecord(DenoiseHistory h) {
    GPUDenoiseHistoryRecord out{};
    for (u32 i = 0; i < 3; ++i) { out.color[i] = h.color[i]; out.normal[i] = h.normal[i]; out.position[i] = h.position[i]; }
    out.firstMoment = h.firstMoment; out.secondMoment = h.secondMoment; out.depth = h.depth;
    out.state = denoisePackState(h.length, h.flags, h.path, h.valid, h.lobe);
    out.slot = h.slot; out.instanceGeneration = h.instanceGeneration; out.materialRevision = h.materialRevision;
    return out;
}
inline GPUDenoiseSpecularHistory denoisePackSpecular(DenoiseHistory h) {
    GPUDenoiseSpecularHistory out{};
    out.roughness = h.roughness; out.hitDistance = h.hitDistance; out.secondarySlot = h.secondarySlot; out.secondaryGeneration = h.secondaryGeneration;
    return out;
}
// Variance loads as 0 (see GPUDenoiseHistoryRecord); a non-specular signal
// passes a zero extension, which it never reads.
inline DenoiseHistory denoiseUnpack(GPUDenoiseHistoryHeader header, GPUDenoiseHistoryRecord r, GPUDenoiseSpecularHistory spec) {
    DenoiseHistory h{};
    for (u32 i = 0; i < 3; ++i) { h.color[i] = r.color[i]; h.normal[i] = r.normal[i]; h.position[i] = r.position[i]; }
    h.firstMoment = r.firstMoment; h.secondMoment = r.secondMoment; h.variance = 0; h.depth = r.depth;
    h.length = denoiseStateLength(r.state); h.flags = denoiseStateFlags(r.state); h.path = denoiseStatePath(r.state); h.valid = denoiseStateValid(r.state); h.lobe = denoiseStateLobe(r.state);
    h.signal = header.signal; h.viewID = header.viewID; h.historyEpoch = header.historyEpoch; h.signalRevision = header.signalRevision;
    h.slot = r.slot; h.instanceGeneration = r.instanceGeneration; h.materialRevision = r.materialRevision;
    h.roughness = spec.roughness; h.hitDistance = spec.hitDistance; h.secondarySlot = spec.secondarySlot; h.secondaryGeneration = spec.secondaryGeneration;
    return h;
}
#ifndef __METAL_VERSION__
// Byte offsets inside one history buffer of `capacity` pixels.
inline u64 denoiseHistoryRecordsOffset() { return sizeof(GPUDenoiseHistoryHeader); }
inline u64 denoiseHistorySpecularOffset(u64 capacity) { return denoiseHistoryRecordsOffset() + capacity * sizeof(GPUDenoiseHistoryRecord); }
inline u64 denoiseHistoryBytes(u64 capacity, bool specular) {
    return denoiseHistorySpecularOffset(capacity) + (specular ? capacity * sizeof(GPUDenoiseSpecularHistory) : 0);
}
#endif
} // namespace phosphor
