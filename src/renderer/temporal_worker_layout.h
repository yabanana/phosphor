#pragma once
#include "core/types.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace phosphor::temporal_worker {
constexpr u32 Magic = 0x46585434; // FXT4
constexpr u32 Version = 1;
constexpr u32 Slots = 3;
constexpr u32 Planes = 6;
enum Plane : u32 { Color, Depth, Motion, Reactive, Exposure, Output };
constexpr u64 aligned(u64 bytes, u64 alignment) {
    return (bytes + alignment - 1) / alignment * alignment;
}
struct Layout {
    u32 magic = Magic, version = Version, width = 0, height = 0;
    std::array<u64, Planes> offsets{}, rows{};
    u64 slotBytes = 0, mappedBytes = 0;
    bool operator==(const Layout &) const = default;
};
inline Layout makeLayout(u32 width, u32 height) {
    if (width < 32 || height < 32 || width > 8192 || height > 8192)
        throw std::invalid_argument("MetalFX worker extent outside 32..8192");
    Layout l;
    l.width = width;
    l.height = height;
    constexpr u32 bpp[] = {8, 4, 4, 1, 2, 8};
    for (u32 i = 0; i < Planes; ++i) {
        l.offsets[i] = l.slotBytes;
        l.rows[i] = aligned(u64(i == Exposure ? 1 : width) * bpp[i], 256);
        l.slotBytes += l.rows[i] * (i == Exposure ? 1 : height);
    }
    l.slotBytes = aligned(l.slotBytes, 16384);
    l.mappedBytes = l.slotBytes * Slots;
    return l;
}
inline bool valid(const Layout &l) {
    try {
        return l == makeLayout(l.width, l.height);
    } catch (...) {
        return false;
    }
}
enum class Command : u32 { Stop, Frame, CrashForTest };
struct Request {
    u32 magic = Magic, version = Version;
    Command command = Command::Frame;
    u32 slot = 0, inputWidth = 0, inputHeight = 0, reset = 0, delayMs = 0;
    u64 ticket = 0;
    float jitterX = 0, jitterY = 0, motionScale = 1, reservedFloat = 0;
};
inline bool valid(const Request &r, const Layout &l) {
    return r.magic == Magic && r.version == Version && r.command == Command::Frame && r.ticket != 0 && r.slot < Slots &&
           r.inputWidth >= 32 && r.inputHeight >= 32 && r.inputWidth <= l.width && r.inputHeight <= l.height &&
           u64(r.inputWidth) * 2 >= l.width && u64(r.inputHeight) * 2 >= l.height && r.reset <= 1 &&
           r.delayMs <= 2000 && r.reservedFloat == 0 && std::isfinite(r.jitterX) && std::isfinite(r.jitterY) &&
           std::isfinite(r.motionScale);
}
struct Reply {
    u32 magic = Magic, version = Version, status = 0, reserved = 0;
    u64 ticket = 0, deviceBytes = 0, physicalFootprint = 0, gpuAllocations = 0;
};
static_assert(sizeof(Layout) == 128);
static_assert(sizeof(Request) == 56);
static_assert(sizeof(Reply) == 48);
} // namespace phosphor::temporal_worker
