// B-08 memory hierarchy kernels (S-MEM-1..3, S-MEM-6 of
// docs/APPLE_SOC_PLAYBOOK.md): pointer chase (latency), streaming read,
// write and copy (bandwidth).  B-09 reuses b08_read.
#include <metal_stdlib>
using namespace metal;

struct MemParams {
    uint iters; // chase steps / passes
    uint zero;  // always 0 at run time: hides the pass structure from the compiler
    uint words; // float4/uint4 elements of the working set
    uint seed;
};

// One thread follows next[] (a random cycle over 128-byte lines: the word
// index of the next line) `iters` times.
kernel void b08_chase(device const uint* next [[buffer(0)]], constant MemParams& p [[buffer(1)]],
                      device uint* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint idx = p.seed; // start word: the host advances it so repeated runs touch new lines
    for (uint k = 0; k < p.iters; ++k) idx = next[idx];
    out[i] = idx;
}

// Streaming read of `words` uint4, `iters` passes.  Thread i reads i, i+n, ...
// (coalesced).  The address of each pass carries k * zero (a runtime zero) so
// the compiler can neither interchange the loops nor reuse loads across
// passes; the passes stay independent (no dependency chain between them).
// Integer accumulation: the CPU check is exact.
kernel void b08_read(device const uint4* src [[buffer(0)]], constant MemParams& p [[buffer(1)]],
                     device uint4* out [[buffer(2)]], uint i [[thread_position_in_grid]],
                     uint n [[threads_per_grid]]) {
    uint4 acc = 0;
    for (uint k = 0; k < p.iters; ++k) {
        const uint base = k * p.zero;
        for (uint j = i; j < p.words; j += n) acc += src[j + base];
    }
    out[i] = acc;
}

// Streaming write; pass k writes value(j, k); the last pass wins (checked).
kernel void b08_write(device uint4* dst [[buffer(0)]], constant MemParams& p [[buffer(1)]],
                      uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
    for (uint k = 0; k < p.iters; ++k) {
        const uint base = k * p.zero;
        for (uint j = i; j < p.words; j += n) {
            const uint v = j * 2654435761u + k * 40503u + p.seed;
            dst[j + base] = uint4(v, v ^ 0x55555555u, v * 3u, ~v);
        }
    }
}

// Copy `words` uint4 from buffer(0) to buffer(2).
kernel void b08_copy(device const uint4* src [[buffer(0)]], constant MemParams& p [[buffer(1)]],
                     device uint4* dst [[buffer(2)]], uint i [[thread_position_in_grid]],
                     uint n [[threads_per_grid]]) {
    for (uint k = 0; k < p.iters; ++k) {
        const uint base = k * p.zero;
        for (uint j = i; j < p.words; j += n) dst[j + base] = src[j + base];
    }
}
