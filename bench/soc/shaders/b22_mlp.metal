// B-22: per-pixel MLP 64 -> 64 -> 64 -> 3 over a batch of pixels, fully fused
// in one kernel with all the weights in threadgroup memory: the layer
// "GEMM" on the Neural Accelerator (matmul2d with a cooperative destination
// tensor, activation applied in registers) against the same network on
// simdgroup_matrix (shader ALUs).
//
// Everything is small-integer exact so the CPU can check the outputs:
// inputs and weights in {-1, 0, 1}, activation min(max(v, 0), 4) (exact in
// half and float), outputs float.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct MlpParams {
    uint tilesPerGroup; // 64-pixel tiles per threadgroup
    uint seed;
    uint pixels;
    uint pad;
};

constant constexpr int TP = 64;  // pixels per tile
constant constexpr int W = 64;   // hidden width
constant constexpr int NO = 8;   // output columns (3 used)

// Weights buffer layout (half): W1[64][64], W2[64][64], W3[64][8].
constant constexpr int W1_OFF = 0;
constant constexpr int W2_OFF = W * W;
constant constexpr int W3_OFF = 2 * W * W;
constant constexpr int W_TOTAL = 2 * W * W + W * NO;

inline half input(uint pix, uint k, uint seed) {
    uint h = (pix * 64u + k) * 0x9E3779B1u + seed;
    h ^= h >> 15;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return half(int(h % 3u) - 1);
}
inline float act(float v) { return min(max(v, 0.0f), 4.0f); }

// Fill the 64x64 input tile: thread t = pixel t/2, columns (t%2)*32 .. +32.
inline void loadInput(threadgroup half* x, uint tid, uint pixBase, uint seed) {
    const uint r = tid >> 1, k0 = (tid & 1u) * 32u;
    for (uint k = 0; k < 32; ++k) x[r * W + k0 + k] = input(pixBase + r, k0 + k, seed);
}
inline void loadWeights(threadgroup half* tw, device const half* w, uint tid) {
    for (uint i = tid; i < uint(W_TOTAL); i += 128u) tw[i] = w[i];
}

// ---- tensor ops ------------------------------------------------------------
kernel void mlp_tensor(device const half* w [[buffer(0)]], device float* out [[buffer(1)]],
                       constant MlpParams& p [[buffer(2)]], uint tgid [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half tw[W_TOTAL];
    threadgroup half tx[TP * W];
    loadWeights(tw, w, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto d64 = matmul2d_descriptor(TP, W, static_cast<int>(dynamic_extent));
    constexpr auto d8 = matmul2d_descriptor(TP, NO, static_cast<int>(dynamic_extent));
    matmul2d<d64, execution_simdgroups<4>> op64;
    matmul2d<d8, execution_simdgroups<4>> op8;

    auto X = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(tx, dextents<int32_t, 2>(W, TP));
    auto B1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(tw + W1_OFF, dextents<int32_t, 2>(W, W));
    auto B2 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(tw + W2_OFF, dextents<int32_t, 2>(W, W));
    auto B3 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(tw + W3_OFF, dextents<int32_t, 2>(NO, W));

    for (uint it = 0; it < p.tilesPerGroup; ++it) {
        const uint pixBase = (tgid * p.tilesPerGroup + it) * uint(TP);
        loadInput(tx, tid, pixBase, p.seed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto c1 = op64.get_destination_cooperative_tensor<decltype(X), decltype(B1), half>();
            op64.run(X, B1, c1);
            threadgroup_barrier(mem_flags::mem_threadgroup); // every simdgroup done reading X
            #pragma unroll
            for (uint16_t i = 0; i < c1.get_capacity(); ++i)
                if (c1.is_valid_element(i)) c1[i] = half(act(float(c1[i])));
            c1.store(X);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            auto c2 = op64.get_destination_cooperative_tensor<decltype(X), decltype(B2), half>();
            op64.run(X, B2, c2);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint16_t i = 0; i < c2.get_capacity(); ++i)
                if (c2.is_valid_element(i)) c2[i] = half(act(float(c2[i])));
            c2.store(X);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        {
            auto c3 = op8.get_destination_cooperative_tensor<decltype(X), decltype(B3), float>();
            op8.run(X, B3, c3);
            #pragma unroll
            for (uint16_t i = 0; i < c3.get_capacity(); ++i) {
                if (c3.is_valid_element(i)) {
                    const auto id = c3.get_multidimensional_index(i); // (column, row)
                    if (id[0] < 3) out[(pixBase + uint(id[1])) * 3u + uint(id[0])] = c3[i];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // before the next tile overwrites X
        }
    }
}

// ---- simdgroup_matrix -------------------------------------------------------
// Each simdgroup owns 16 pixels (2 row blocks) of the tile; the activation is
// applied to the accumulator's thread elements and written back in place.
inline void layerSimd(threadgroup half* x, threadgroup const half* wt, uint sg, uint lane) {
    simdgroup_float8x8 acc[2][8];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 8; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (int k = 0; k < W; k += 8) {
        simdgroup_half8x8 ma[2], mb[8];
        for (int i = 0; i < 2; ++i) simdgroup_load(ma[i], x + (sg * 16 + i * 8) * W + k, W);
        for (int j = 0; j < 8; ++j) simdgroup_load(mb[j], wt + k * W + j * 8, W);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 8; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup); // all loads of this simdgroup done before the in-place store
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 8; ++j) {
            simdgroup_half8x8 h;
            h.thread_elements()[0] = half(act(acc[i][j].thread_elements()[0]));
            h.thread_elements()[1] = half(act(acc[i][j].thread_elements()[1]));
            simdgroup_store(h, x + (sg * 16 + i * 8) * W + j * 8, W);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void mlp_simd(device const half* w [[buffer(0)]], device float* out [[buffer(1)]],
                     constant MlpParams& p [[buffer(2)]], uint tgid [[threadgroup_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half tw[W_TOTAL];
    threadgroup half tx[TP * W];
    threadgroup float to[4 * 16 * NO];
    loadWeights(tw, w, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint it = 0; it < p.tilesPerGroup; ++it) {
        const uint pixBase = (tgid * p.tilesPerGroup + it) * uint(TP);
        threadgroup_barrier(mem_flags::mem_threadgroup); // previous tile's reads done
        loadInput(tx, tid, pixBase, p.seed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        layerSimd(tx, tw + W1_OFF, sg, lane);
        layerSimd(tx, tw + W2_OFF, sg, lane);
        simdgroup_float8x8 acc[2];
        acc[0] = simdgroup_float8x8(0);
        acc[1] = simdgroup_float8x8(0);
        for (int k = 0; k < W; k += 8) {
            simdgroup_half8x8 ma[2], mb;
            for (int i = 0; i < 2; ++i) simdgroup_load(ma[i], tx + (sg * 16 + i * 8) * W + k, W);
            simdgroup_load(mb, tw + W3_OFF + k * NO, NO);
            for (int i = 0; i < 2; ++i) simdgroup_multiply_accumulate(acc[i], ma[i], mb, acc[i]);
        }
        for (int i = 0; i < 2; ++i) simdgroup_store(acc[i], to + (sg * 16 + i * 8) * NO, NO);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 16u * 3u; e += 32u) {
            const uint r = e / 3u, c = e % 3u;
            out[(pixBase + sg * 16u + r) * 3u + c] = to[(sg * 16u + r) * NO + c];
        }
    }
}
