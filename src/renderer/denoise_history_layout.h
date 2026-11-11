#pragma once
// F13 denoise history layout helpers, shared by C++ and MSL (gpu_types.h holds
// the structs). The state word of GPUDenoiseHistoryRecord packs
//   bits  0-15 length (<= maxHistory <= 256)
//   bits 16-23 flags  (bit0 numerical error, DENOISE_HISTORY_STOCHASTIC)
//   bits 24-30 path   (REFLECTION_PATH_*, <= 5)
//   bit  31    valid
// losslessly for every value the temporal pass writes (tests/test_denoise.cpp).
#include "renderer/gpu_types.h"
namespace phosphor {
PHOSPHOR_GPU_CONSTANT u32 DENOISE_HISTORY_MAX_LENGTH = 0xffffu, DENOISE_HISTORY_MAX_FLAGS = 0xffu, DENOISE_HISTORY_MAX_PATH = 0x7fu;
inline u32 denoisePackState(u32 length, u32 flags, u32 path, u32 valid) {
    return (length & 0xffffu) | ((flags & 0xffu) << 16) | ((path & 0x7fu) << 24) | (valid ? 0x80000000u : 0u);
}
inline u32 denoiseStateLength(u32 state) { return state & 0xffffu; }
inline u32 denoiseStateFlags(u32 state) { return (state >> 16) & 0xffu; }
inline u32 denoiseStatePath(u32 state) { return (state >> 24) & 0x7fu; }
inline u32 denoiseStateValid(u32 state) { return state >> 31; }
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
    out.state = denoisePackState(h.length, h.flags, h.path, h.valid);
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
    h.length = denoiseStateLength(r.state); h.flags = denoiseStateFlags(r.state); h.path = denoiseStatePath(r.state); h.valid = denoiseStateValid(r.state);
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
