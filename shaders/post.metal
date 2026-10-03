#include <metal_stdlib>
#include "renderer/post_layout.h"
using namespace metal;
using namespace phosphor;

// Exposure histogram: bin 0 is black/non-finite; 1..255 span log2 [-12,16].
kernel void exposure_clear(device uint *histogram [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i < 256)
        histogram[i] = 0;
}
kernel void exposure_histogram(texture2d<float, access::read> color [[texture(0)]],
                               device atomic_uint *histogram [[buffer(0)]], constant GPUPostParams &p [[buffer(1)]],
                               depth2d<float, access::read> depth [[texture(2)]],
                               uint2 pixel [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint localBins[256];
    atomic_store_explicit(&localBins[tid], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bin = 0;
    bool valid = pixel.x < p.inputWidth && pixel.y < p.inputHeight;
    if (valid) {
        const float4 value = color.read(pixel);
        const float luminance = dot(value.rgb, float3(0.2126f, 0.7152f, 0.0722f));
        if (depth.read(pixel) > 0 && isfinite(luminance) && luminance > exp2(-12.0f))
            bin = 1u + uint(clamp((log2(luminance) + 12.0f) / 28.0f, 0.0f, 1.0f) * 254.0f);
        atomic_fetch_add_explicit(&localBins[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = atomic_load_explicit(&localBins[tid], memory_order_relaxed);
    if (count)
        atomic_fetch_add_explicit(&histogram[tid], count, memory_order_relaxed);
}
kernel void exposure_reduce(const device uint *histogram [[buffer(0)]], constant GPUPostParams &p [[buffer(1)]],
                            device float *state [[buffer(2)]], texture2d<float, access::write> exposure [[texture(1)]],
                            uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint totals[8];
    threadgroup float weighted[8], weights[8];
    const uint count = tid == 0 ? 0u : histogram[tid];
    const uint prefix = simd_prefix_exclusive_sum(count);
    const uint total = simd_sum(count);
    if (lane == 0)
        totals[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint all = 0, before = prefix;
    for (uint i = 0; i < 8; ++i) {
        all += totals[i];
        if (i < simd)
            before += totals[i];
    }
    const float low = float(all) * 0.05f, high = float(all) * 0.95f;
    const float weight = max(0.0f, min(float(before + count), high) - max(float(before), low));
    const float logValue = -12.0f + (float(tid) - 0.5f) * (28.0f / 254.0f);
    const float wsum = simd_sum(weight * logValue), n = simd_sum(weight);
    if (lane == 0) {
        weighted[simd] = wsum;
        weights[simd] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0, used = 0;
        for (uint i = 0; i < 8; ++i) {
            sum += weighted[i];
            used += weights[i];
        }
        const float target = used > 0 ? clamp(0.18f / exp2(sum / used), p.minExposure, p.maxExposure) : 1.0f;
        const float previous = state[0];
        const float adapted = p.historyReset || !isfinite(previous) || previous <= 0
                                  ? target
                                  : mix(previous, target, 1.0f - exp(-max(p.deltaTime, 0.0f) * p.adaptSpeed));
        state[0] = (p.autoExposure & 1u) ? adapted : 1.0f;
        state[1] = target + ((p.autoExposure & 2u) ? max(1.0f, target * 0.1f) : 0.0f);
        state[2] = float(all);
        state[3] = used;
        exposure.write(float4(state[0] * p.manualExposure), uint2(0));
    }
}

// Native/spatial fallback while an asynchronous MetalFX request is pending.
// Content occupies the top-left input rectangle of a fixed-size allocation.
kernel void post_native(texture2d<float> input [[texture(0)]], texture2d<float, access::write> output [[texture(1)]],
                        constant GPUPostParams &p [[buffer(1)]], uint2 pixel [[thread_position_in_grid]]) {
    if (pixel.x >= p.outputWidth || pixel.y >= p.outputHeight)
        return;
    // Reference-only integer supersampling averages radiance before the
    // nonlinear display curve. A bilinear lookup is only a 2x2 box and
    // cannot represent the larger reference footprints.
    const uint scale = p.inputWidth / p.outputWidth;
    if (scale > 1u && scale <= 8u && p.inputWidth == p.outputWidth * scale && p.inputHeight == p.outputHeight * scale) {
        float4 sum = 0;
        for (uint y = 0; y < scale; ++y)
            for (uint x = 0; x < scale; ++x)
                sum += input.read(pixel * scale + uint2(x, y));
        output.write(sum / float(scale * scale), pixel);
        return;
    }
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    const float2 size = float2(input.get_width(), input.get_height());
    const float2 uv =
        clamp((float2(pixel) + 0.5f) / float2(p.outputWidth, p.outputHeight) * float2(p.inputWidth, p.inputHeight),
              float2(0.5f), float2(p.inputWidth, p.inputHeight) - 0.5f) /
        size;
    output.write(input.sample(linear, uv), pixel);
}

static float3 acesFit(float3 v) {
    return saturate((v * (2.51f * v + 0.03f)) / (v * (2.43f * v + 0.59f) + 0.14f));
}
// AgX minimal polynomial fit: Troy Sobotka's initial configuration and
// Benjamin Wrensch's approximation. This is not an exact Blender OCIO LUT.
// Sources/attribution: docs/plans/F7-F8-EXECUTION.md.
static float3 agxFit(float3 v) {
    const float3 inset = float3(dot(v, float3(0.8424790623f, 0.0784336f, 0.0792237451f)),
                                dot(v, float3(0.0423282423f, 0.8784686365f, 0.0791661275f)),
                                dot(v, float3(0.0423756549f, 0.0784336f, 0.8791429738f)));
    const float3 x = saturate((log2(max(inset, 1e-10f)) + 12.47393f) / 16.5f);
    const float3 y =
        ((((((15.5f * x - 40.14f) * x + 31.96f) * x - 6.868f) * x + 0.4298f) * x + 0.1191f) * x - 0.00232f);
    const float3 outset = float3(dot(y, float3(1.1968790051f, -0.0980208811f, -0.0990297441f)),
                                 dot(y, float3(-0.0528968518f, 1.1519031299f, -0.0989611768f)),
                                 dot(y, float3(-0.0529716355f, -0.0980434501f, 1.1510736726f)));
    return saturate(pow(max(outset, 0.0f), float3(2.2f)));
}
static float3 postMap(float3 color, constant GPUPostParams &p) {
    color = max(color, 0.0f);
    float3 mapped;
    if (p.tonemap == 1u)
        mapped = agxFit(color);
    else if (p.tonemap == 2u)
        mapped = saturate(color * (1.0f + color / (p.whitePoint * p.whitePoint)) / (1.0f + color));
    else
        mapped = acesFit(color);
    // Display-linear extended sRGB: retain the selected SDR curve through
    // middle grey, then roll scene highlights into the available headroom.
    if (p.headroom > 1.0f) {
        const float3 highlight = max(color - 1.0f, 0.0f);
        mapped += (p.headroom - 1.0f) * highlight / (highlight + p.headroom);
    }
    return clamp(mapped, 0.0f, p.headroom);
}
struct PostVertex {
    float4 position [[position]];
    float2 uv;
};
vertex PostVertex post_vs(uint id [[vertex_id]]) {
    const float2 uv = float2((id << 1) & 2, id & 2);
    return {float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), uv};
}
fragment half4 post_fs(PostVertex in [[stage_in]], texture2d<float> hdr [[texture(0)]],
                       texture2d<float, access::read> exposure [[texture(1)]],
                       constant GPUPostParams &p [[buffer(1)]]) {
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    const float2 uv = in.uv, step = 1.0f / float2(hdr.get_width(), hdr.get_height());
    const float3 c = hdr.sample(linear, uv).rgb;
    float3 value = c;
    if (p.sharpening > 0) {
        const float3 a = hdr.sample(linear, uv + float2(step.x, 0)).rgb,
                     b = hdr.sample(linear, uv - float2(step.x, 0)).rgb;
        const float3 d = hdr.sample(linear, uv + float2(0, step.y)).rgb,
                     e = hdr.sample(linear, uv - float2(0, step.y)).rgb;
        const float3 lo = min(c, min(min(a, b), min(d, e))), hi = max(c, max(max(a, b), max(d, e)));
        const float contrast = max(max(hi.x - lo.x, hi.y - lo.y), hi.z - lo.z) / max(max(max(hi.x, hi.y), hi.z), 0.01f);
        const float amount = p.sharpening * (1.0f - saturate(contrast));
        value = clamp(c + amount * (c - (a + b + d + e) * 0.25f), lo, hi);
    }
    return half4(half3(postMap(value * exposure.read(uint2(0)).x, p)), 1.0h);
}

// An 8-bit PNG cannot retain EDR values above reference white. Capture the
// displayed image (including UI), clipped to SDR and encoded by the sRGB RT.
fragment half4 post_capture_fs(PostVertex in [[stage_in]], texture2d<float> display [[texture(0)]]) {
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    return half4(half3(saturate(display.sample(nearest, in.uv).rgb)), 1.0h);
}
