#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace phosphor {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

constexpr u32 FRAMES_IN_FLIGHT      = 2;
constexpr u32 MAX_BINDLESS_TEXTURES  = 16384;
constexpr u32 MAX_BINDLESS_BUFFERS   = 4096;
constexpr u32 MAX_MESHLETS_PER_MESH  = 65536;
constexpr u32 MESHLET_MAX_VERTICES   = 64;
constexpr u32 MESHLET_MAX_TRIANGLES  = 124;

using EntityID = u32;
constexpr u32 INVALID_ENTITY = ~0u;

} // namespace phosphor
