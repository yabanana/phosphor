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

// The block of a material: the extension buffer address plus the material's first transform record.
static TexXf texXfFor(const device GPUMaterialExtRecord *extensions, uint flags, uint index) {
    TexXf x = texXfNone();
    if (kTextureTransform != 0u && extensions != nullptr && (flags & MATERIAL_FLAG_TEXTURE_TRANSFORM) != 0u) {
        x.rec = reinterpret_cast<const device GPUTextureTransformRecord *>(extensions + index);
        x.mask = x.rec[0].slotMask;
    }
    return x;
}

static bool texXfHas(TexXf x, uint slot) {
    return kTextureTransform != 0u && ((x.mask >> slot) & 1u) != 0u;
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

// The slot's uv and derivatives (unchanged when the slot has no transform or the feature is compiled out).
static void texXfUv(TexXf x, uint slot, thread float2 &uv, thread float2 &dx, thread float2 &dy) {
    if (texXfHas(x, slot)) {
        const GPUTextureTransform t = texXfGet(x, slot);
        uv = texXfApply(t, uv);
        dx = texXfLinear(t, dx);
        dy = texXfLinear(t, dy);
    }
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

// Sample a material texture with the slot's transform (the plain sample when the slot has none).
static half4 sampleSlot(const device TextureHandle *table, uint index, uint slot, TexXf xf, float2 uv, half4 fallback, float2 dx,
                        float2 dy) {
    if (index == INVALID_TEXTURE_INDEX) {
        return fallback;
    }
    texXfUv(xf, slot, uv, dx, dy);
    return half4(table[index].tex.sample(kMaterialSampler, uv, gradient2d(dx, dy)));
}
