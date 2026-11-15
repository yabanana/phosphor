#pragma once
// F15.1 (ticket 22): the constant-free core of the per-slot KHR_texture_transform helpers (see texture_transform.h).  It never
// references the reserved function constant, so any module (RT intersection functions, DI / GI / reflection kernels, the shadow
// fragments) can include it; only gpu_types.h and <metal_stdlib> are needed.
#include <metal_stdlib>
#include "renderer/gpu_types.h"
using namespace metal;
using namespace phosphor;

struct TexXf {
    const device GPUTextureTransformRecord *rec;
    uint mask;
};

static TexXf texXfNone() {
    TexXf x;
    x.rec = nullptr;
    x.mask = 0u;
    return x;
}

// The "On" functions never reference kTextureTransform: an entry point that only exists for scenes with transforms (the *_tt_fs alpha
// tests) uses them, so its descriptor needs no function constant (metal-tt cannot lower an unspecialised descriptor whose function
// references one).  The plain names compile the transform in only when the pipeline sets the reserved constant.

// The block of a material: the extension buffer address plus the material's first transform record.
static TexXf texXfForOn(const device GPUMaterialExtRecord *extensions, uint flags, uint index) {
    TexXf x = texXfNone();
    if (extensions != nullptr && (flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) != 0u) {
        x.rec = reinterpret_cast<const device GPUTextureTransformRecord *>(extensions + index);
        x.mask = x.rec[0].slotMask;
    }
    return x;
}

// The block of a LAYER sub-material: valid under either bit (MATERIAL_FLAG_LAYER_TRANSFORM alone = a uniform transform that direct users have baked).
static TexXf texXfForLayerOn(const device GPUMaterialExtRecord *extensions, uint flags, uint index) {
    TexXf x = texXfNone();
    if (extensions != nullptr && (flags & (MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM)) != 0u) {
        x.rec = reinterpret_cast<const device GPUTextureTransformRecord *>(extensions + index);
        x.mask = x.rec[0].slotMask;
    }
    return x;
}

static bool texXfHasOn(TexXf x, uint slot) {
    return ((x.mask >> slot) & 1u) != 0u;
}

static GPUTextureTransform texXfGet(TexXf x, uint slot) {
    return x.rec[slot / 3u].t[slot % 3u];
}

static float2 texXfApply(GPUTextureTransform t, float2 uv) {
    return float2(t.m[0] * uv.x + t.m[1] * uv.y + t.o[0], t.m[2] * uv.x + t.m[3] * uv.y + t.o[1]);
}

static float2 texXfLinear(GPUTextureTransform t, float2 d) {
    return float2(t.m[0] * d.x + t.m[1] * d.y, t.m[2] * d.x + t.m[3] * d.y);
}

// The slot's uv and derivatives (unchanged when the slot has no transform).
static void texXfUvOn(TexXf x, uint slot, thread float2 &uv, thread float2 &dx, thread float2 &dy) {
    if (texXfHasOn(x, slot)) {
        const GPUTextureTransform t = texXfGet(x, slot);
        uv = texXfApply(t, uv);
        dx = texXfLinear(t, dx);
        dy = texXfLinear(t, dy);
    }
}

// One point sample (level 0) at a secondary site: the uv of `slot` through the material's transform block, or `uv` itself when the
// material has none (a guarded alternative: the value is untouched for every material without MATERIAL_FLAG_TEXTURE_TRANSFORM).
static float2 texXfSlotUv(const device GPUMaterialExtRecord *extensions, uint flags, uint index, uint slot, float2 uv) {
    if (extensions != nullptr && (flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) != 0u) {
        const TexXf x = texXfForOn(extensions, flags, index);
        if (texXfHasOn(x, slot)) return texXfApply(texXfGet(x, slot), uv);
    }
    return uv;
}

// The tangent frame of a normal-like slot at a secondary site (texXfFrame of texture_transform.h without the constant): T = dP/du,
// B = -dP/dv; with uv' = M uv the texel axes are T' = inv00 T - inv10 B and B' = -inv01 T + inv11 B (renormalised).
static void texXfFrameOn(TexXf x, uint slot, thread float3 &T, thread float3 &B) {
    if (texXfHasOn(x, slot)) {
        const GPUTextureTransform t = texXfGet(x, slot);
        const float det = t.m[0] * t.m[3] - t.m[1] * t.m[2];
        if (fabs(det) > 1e-12f) {
            const float i00 = t.m[3] / det, i01 = -t.m[1] / det, i10 = -t.m[2] / det, i11 = t.m[0] / det;
            const float3 T2 = T * i00 - B * i10;
            const float3 B2 = B * i11 - T * i01;
            const float tl = length(T2), bl = length(B2);
            if (tl > 1e-12f && bl > 1e-12f) {
                T = T2 / tl;
                B = B2 / bl;
            }
        }
    }
}

// A slot's uv AND derivatives through the material's block, guarded by the material's flag (the values are untouched without one).  The layer
// stack's mask / height and every layer's own textures use this: a layer's slots go through the block of ITS sub-material.
static void texXfSlotUvDd(const device GPUMaterialExtRecord *extensions, uint flags, uint index, uint slot, thread float2 &uv, thread float2 &dx,
                          thread float2 &dy) {
    if (extensions != nullptr && (flags & (MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM)) != 0u)
        texXfUvOn(texXfForLayerOn(extensions, flags, index), slot, uv, dx, dy);
}

// A tangent-space normal vector read through a transformed normal-like slot, expressed in the mesh's tangent frame: the texel axes are
// T' = inv00 T - inv10 B and B' = -inv01 T + inv11 B (texXfFrameOn), so (a, b) -> a T' + b B' in the (T, B) basis.  Unchanged without a transform.
static float3 texXfRotateTn(const device GPUMaterialExtRecord *extensions, uint flags, uint index, uint slot, float3 tn) {
    if (extensions != nullptr && (flags & (MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM)) != 0u) {
        const TexXf x = texXfForLayerOn(extensions, flags, index);
        if (texXfHasOn(x, slot)) {
            const GPUTextureTransform t = texXfGet(x, slot);
            const float det = t.m[0] * t.m[3] - t.m[1] * t.m[2];
            if (fabs(det) > 1e-12f) {
                const float i00 = t.m[3] / det, i01 = -t.m[1] / det, i10 = -t.m[2] / det, i11 = t.m[0] / det;
                const float2 tp = float2(i00, -i10), bp = float2(-i01, i11);
                const float tl = length(tp), bl = length(bp);
                if (tl > 1e-12f && bl > 1e-12f) {
                    const float2 ts = tp / tl, bs = bp / bl;
                    return float3(tn.x * ts.x + tn.y * bs.x, tn.x * ts.y + tn.y * bs.y, tn.z);
                }
            }
        }
    }
    return tn;
}
