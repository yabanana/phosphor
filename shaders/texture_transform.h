#pragma once
// F15.1 (ticket 22): per-slot KHR_texture_transform.
//
// A material with MATERIAL_FLAG_TEXTURE_TRANSFORM owns a transform block in the extension buffer (renderer/gpu_types.h,
// GPUTextureTransformRecord): slot s at record s / 3.  uv' = M uv + o; the screen-space derivatives follow the linear part
// (dx' = M dx); the tangent frame of a normal-like slot follows the transformed UV (texXfFrame).  Everything is compiled out
// unless the pipeline sets the reserved constant TEXTURE_TRANSFORM (kTextureTransform, undefined => 0): the pipelines of every
// scene without such a material are the pre-ticket ones.
//
// Needs gpu_types.h, the generated forward_variants.generated.metal.h (kTextureTransform) and <metal_stdlib>.

#include "texture_transform_core.h"

// The "On" functions never reference kTextureTransform: an entry point that only exists for scenes with transforms (the *_tt_fs alpha
// tests) uses them, so its descriptor needs no function constant (metal-tt cannot lower an unspecialised descriptor whose function
// references one).  The plain names compile the transform in only when the pipeline sets the reserved constant.

static TexXf texXfFor(const device GPUMaterialExtRecord *extensions, uint flags, uint index) {
    return kTextureTransform != 0u ? texXfForOn(extensions, flags, index) : texXfNone();
}

// The block of a layer sub-material (valid for either bit of the flags): compiled out without the constant like texXfFor.
static TexXf texXfForLayer(const device GPUMaterialExtRecord *extensions, uint flags, uint index) {
    return kTextureTransform != 0u ? texXfForLayerOn(extensions, flags, index) : texXfNone();
}

static bool texXfHas(TexXf x, uint slot) {
    return kTextureTransform != 0u && texXfHasOn(x, slot);
}

// Unchanged when the feature is compiled out.
static void texXfUv(TexXf x, uint slot, thread float2 &uv, thread float2 &dx, thread float2 &dy) {
    if (kTextureTransform != 0u) texXfUvOn(x, slot, uv, dx, dy);
}

// The tangent frame of a normal-like slot.  T = dP/du, B = -dP/dv (glTF: bitangent towards decreasing V); with uv' = M uv the
// directions of increasing u' and decreasing v' are T' = inv00 T - inv10 B and B' = -inv01 T + inv11 B (renormalised: a scale
// or shear changes their length, not the texel directions).  A singular matrix keeps the frame.
static void texXfFrame(TexXf x, uint slot, thread float3 &T, thread float3 &B) {
    if (texXfHas(x, slot)) {
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
