// graph_debug.metal -- synthetic passes of --debug-graph-transients (F2.2).
//
// Integer-only chain used to exercise transient aliasing on the device; the
// CPU twin of the arithmetic is src/rendergraph/graph_debug_reference.h.
// Keep the two in sync.  All arithmetic is uint32 (wraps like on the CPU).
//
// Bindings (Metal 4 argument tables):
//   compute: buffer(0) uint frame (constant), buffer(1) device uint[]
//            (row sums or readback), texture(0) the image.
//   raster : texture(0) image C.

#include <metal_stdlib>

using namespace metal;

constant uint kSize         = 256;
constant uint kChecksumMask = 0x5bd1e995u;

static uint fillValue(uint x, uint y, uint frame) {
    return (x * 73856093u) ^ (y * 19349663u) ^ (frame * 83492791u) ^ (x * y);
}

static uint expandValue(uint rowSum, uint x, uint y) {
    return (rowSum ^ (x * 2246822519u)) + y * 3266489917u;
}

static uint rasterValue(uint c) {
    return (c * 1664525u + 1013904223u) ^ (c >> 16);
}

// 1: A = fill(x, y, frame)
kernel void debug_fill(texture2d<uint, access::write> a [[texture(0)]],
                       constant uint& frame [[buffer(0)]],
                       uint2 tid [[thread_position_in_grid]]) {
    a.write(uint4(fillValue(tid.x, tid.y, frame), 0, 0, 0), tid);
}

// 2: B[y] = sum of row y of A
kernel void debug_reduce(texture2d<uint, access::read> a [[texture(0)]],
                         device uint* rows [[buffer(1)]],
                         uint y [[thread_position_in_grid]]) {
    uint sum = 0;
    for (uint x = 0; x < kSize; ++x) sum += a.read(uint2(x, y)).x;
    rows[y] = sum;
}

// 3: C = expand(B[y], x, y)
kernel void debug_expand(texture2d<uint, access::write> c [[texture(0)]],
                         device const uint* rows [[buffer(1)]],
                         uint2 tid [[thread_position_in_grid]]) {
    c.write(uint4(expandValue(rows[tid.y], tid.x, tid.y), 0, 0, 0), tid);
}

// 4: D = raster(C) through a fullscreen triangle into an R32Uint attachment
vertex float4 debug_vs(uint vid [[vertex_id]]) {
    const float2 p = float2((vid << 1) & 2, vid & 2);
    return float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
}

fragment uint debug_fs(float4 position [[position]],
                       texture2d<uint, access::read> c [[texture(0)]]) {
    return rasterValue(c.read(uint2(position.xy)).x);
}

// 5: readback = D ^ mask
kernel void debug_checksum(texture2d<uint, access::read> d [[texture(0)]],
                           device uint* out [[buffer(1)]],
                           uint2 tid [[thread_position_in_grid]]) {
    out[tid.y * kSize + tid.x] = d.read(tid).x ^ kChecksumMask;
}
