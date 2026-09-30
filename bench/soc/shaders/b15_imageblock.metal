// B-15: imageblock (tile memory) capacity.  One tile kernel per imageblock
// size (K 32-bit words per pixel = 4K bytes, generated for K = 1..24, 28, 32, 48, 64):
// every thread fills its pixel's imageblock entry with hash(px, py, j),
// waits for the tile (imageblock barrier), then reads the entry of the pixel to
// its right (wrapping inside the tile), so the values really travel through tile
// memory, and writes the checksum of what it read to a device buffer at its own
// pixel.  The CPU recomputes every checksum.
#include <metal_stdlib>
using namespace metal;

struct TileParams {
    uint width;   // pixels per row of the output
    uint zero;    // 0 at run time
    uint pad0, pad1;
};

inline uint b15_hash(uint x, uint y, uint j) {
    uint h = x * 73856093u ^ y * 19349663u ^ (j * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

struct Ib1 { uint d[1]; };
kernel void b15_tile_1(imageblock<Ib1, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib1* mine = ib.data(tid);
    for (uint j = 0; j < 1u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib1* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 1u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib2 { uint d[2]; };
kernel void b15_tile_2(imageblock<Ib2, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib2* mine = ib.data(tid);
    for (uint j = 0; j < 2u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib2* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 2u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib3 { uint d[3]; };
kernel void b15_tile_3(imageblock<Ib3, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib3* mine = ib.data(tid);
    for (uint j = 0; j < 3u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib3* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 3u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib4 { uint d[4]; };
kernel void b15_tile_4(imageblock<Ib4, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib4* mine = ib.data(tid);
    for (uint j = 0; j < 4u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib4* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 4u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib5 { uint d[5]; };
kernel void b15_tile_5(imageblock<Ib5, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib5* mine = ib.data(tid);
    for (uint j = 0; j < 5u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib5* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 5u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib6 { uint d[6]; };
kernel void b15_tile_6(imageblock<Ib6, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib6* mine = ib.data(tid);
    for (uint j = 0; j < 6u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib6* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 6u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib7 { uint d[7]; };
kernel void b15_tile_7(imageblock<Ib7, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib7* mine = ib.data(tid);
    for (uint j = 0; j < 7u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib7* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 7u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib8 { uint d[8]; };
kernel void b15_tile_8(imageblock<Ib8, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib8* mine = ib.data(tid);
    for (uint j = 0; j < 8u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib8* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 8u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib9 { uint d[9]; };
kernel void b15_tile_9(imageblock<Ib9, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib9* mine = ib.data(tid);
    for (uint j = 0; j < 9u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib9* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 9u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib10 { uint d[10]; };
kernel void b15_tile_10(imageblock<Ib10, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib10* mine = ib.data(tid);
    for (uint j = 0; j < 10u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib10* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 10u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib11 { uint d[11]; };
kernel void b15_tile_11(imageblock<Ib11, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib11* mine = ib.data(tid);
    for (uint j = 0; j < 11u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib11* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 11u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib12 { uint d[12]; };
kernel void b15_tile_12(imageblock<Ib12, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib12* mine = ib.data(tid);
    for (uint j = 0; j < 12u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib12* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 12u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib13 { uint d[13]; };
kernel void b15_tile_13(imageblock<Ib13, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib13* mine = ib.data(tid);
    for (uint j = 0; j < 13u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib13* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 13u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib14 { uint d[14]; };
kernel void b15_tile_14(imageblock<Ib14, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib14* mine = ib.data(tid);
    for (uint j = 0; j < 14u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib14* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 14u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib15 { uint d[15]; };
kernel void b15_tile_15(imageblock<Ib15, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib15* mine = ib.data(tid);
    for (uint j = 0; j < 15u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib15* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 15u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib16 { uint d[16]; };
kernel void b15_tile_16(imageblock<Ib16, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib16* mine = ib.data(tid);
    for (uint j = 0; j < 16u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib16* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 16u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib17 { uint d[17]; };
kernel void b15_tile_17(imageblock<Ib17, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib17* mine = ib.data(tid);
    for (uint j = 0; j < 17u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib17* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 17u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib18 { uint d[18]; };
kernel void b15_tile_18(imageblock<Ib18, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib18* mine = ib.data(tid);
    for (uint j = 0; j < 18u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib18* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 18u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib19 { uint d[19]; };
kernel void b15_tile_19(imageblock<Ib19, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib19* mine = ib.data(tid);
    for (uint j = 0; j < 19u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib19* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 19u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib20 { uint d[20]; };
kernel void b15_tile_20(imageblock<Ib20, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib20* mine = ib.data(tid);
    for (uint j = 0; j < 20u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib20* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 20u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib21 { uint d[21]; };
kernel void b15_tile_21(imageblock<Ib21, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib21* mine = ib.data(tid);
    for (uint j = 0; j < 21u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib21* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 21u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib22 { uint d[22]; };
kernel void b15_tile_22(imageblock<Ib22, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib22* mine = ib.data(tid);
    for (uint j = 0; j < 22u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib22* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 22u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib23 { uint d[23]; };
kernel void b15_tile_23(imageblock<Ib23, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib23* mine = ib.data(tid);
    for (uint j = 0; j < 23u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib23* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 23u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib24 { uint d[24]; };
kernel void b15_tile_24(imageblock<Ib24, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib24* mine = ib.data(tid);
    for (uint j = 0; j < 24u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib24* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 24u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib28 { uint d[28]; };
kernel void b15_tile_28(imageblock<Ib28, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib28* mine = ib.data(tid);
    for (uint j = 0; j < 28u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib28* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 28u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib32 { uint d[32]; };
kernel void b15_tile_32(imageblock<Ib32, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib32* mine = ib.data(tid);
    for (uint j = 0; j < 32u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib32* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 32u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib48 { uint d[48]; };
kernel void b15_tile_48(imageblock<Ib48, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib48* mine = ib.data(tid);
    for (uint j = 0; j < 48u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib48* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 48u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}

struct Ib64 { uint d[64]; };
kernel void b15_tile_64(imageblock<Ib64, imageblock_layout_explicit> ib,
                        device uint* out [[buffer(0)]], constant TileParams& p [[buffer(1)]],
                        ushort2 tid [[thread_position_in_threadgroup]], ushort2 tg [[threadgroup_position_in_grid]],
                        ushort2 tsz [[threads_per_threadgroup]]) {
    const uint px = uint(tg.x) * tsz.x + tid.x, py = uint(tg.y) * tsz.y + tid.y;
    threadgroup_imageblock Ib64* mine = ib.data(tid);
    for (uint j = 0; j < 64u; ++j) mine->d[j] = b15_hash(px, py, j) + p.zero;
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    const ushort2 nb = ushort2((tid.x + 1) % tsz.x, tid.y);
    threadgroup_imageblock Ib64* other = ib.data(nb);
    uint sum = 0;
    for (uint j = 0; j < 64u; ++j) sum = sum * 31u + other->d[j];
    out[py * p.width + px] = sum;
}
