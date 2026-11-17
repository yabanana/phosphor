#pragma once
// F15.4 (ticket 23): the layer stack's base resolve, shared by the primary resolve (material_shading.h, one sampler policy with
// derivative gradients) and every secondary-hit consumer (reflection hits, GI probe and reservoir hits, the reflection probe capture:
// level-0 policies).  The mask is sampled at the primitive uv with a clamp-to-edge sampler; every layer with a non-zero weight is an
// ordinary material of the library whose base parameters are sampled at uv * uvScale with the same expressions as the single-material
// path (so a weight vector e_j reproduces layer j exactly), blended in parameter space (renderer/layer_blend.h) and shaded ONCE by the
// base model.  Includers: `using namespace metal; using namespace phosphor;`, gpu_types.h and layer_blend.h before this file.
//
// The sampler policy is a struct with two static member templates (T = the bindless table element, TextureHandle or DITextureHandle, with
// a `tex` member):
//   half4 sample(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy)   a repeat-addressed layer texture
//   half4 mask(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy)     the clamp-to-edge mask

#include "texture_transform_core.h"

// Sampler policies of the coverage alpha at the sites that have no sampler of their own (shadow depth, RT any-hit, raster alpha tests):
// repeat-addressed layer textures and a clamp-to-edge mask, either at level 0 (`LayerAlphaLevel0`) or with the caller's derivative gradients
// (`LayerAlphaGrad`).  T is any bindless element with a `tex` member.
struct LayerAlphaLevel0 {
    static bool transforms() { return true; }
    template <typename T>
    static half4 sample(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::repeat);
        return index == INVALID_TEXTURE_INDEX ? fallback : half4(table[index].tex.sample(s, uv, level(0.0f)));
    }
    template <typename T>
    static half4 mask(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
        return index == INVALID_TEXTURE_INDEX ? fallback : half4(table[index].tex.sample(s, uv, level(0.0f)));
    }
};
struct LayerAlphaGrad {
    static bool transforms() { return true; }
    template <typename T>
    static half4 sample(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));
        return index == INVALID_TEXTURE_INDEX ? fallback : half4(table[index].tex.sample(s, uv, gradient2d(dx, dy)));
    }
    template <typename T>
    static half4 mask(const device T *table, uint index, float2 uv, half4 fallback, float2 dx, float2 dy) {
        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
        return index == INVALID_TEXTURE_INDEX ? fallback : half4(table[index].tex.sample(s, uv, gradient2d(dx, dy)));
    }
};

// Every sampler policy has `static bool transforms()`: whether the per-slot texture transforms of the stack and its layers are applied.  The
// secondary sites say true (a runtime flag test per slot); the primary resolve says kTextureTransform, so a scene without any transformed
// material (the scene store counts both transform flags) compiles the whole in-stack transform code out (it was dead there: no material has
// either flag, so the images are identical).
//
// F15.1 (ticket 22): per-slot texture transforms of a stack.  The mask goes through the stack material's block (slot TT_SLOT_LAYER_MASK), a layer's
// height through the same block (TT_SLOT_LAYER_HEIGHT0 + i, applied to uv * uvScale), and every texture of layer i through the block of ITS
// sub-material (applied to uv * uvScale; the normal vector is rotated into the mesh frame).  `extensions` = the extension buffer (null: none),
// `stackFlags` / `stackIndex` = the stack material's flags and textureTransformIndex.  A material without MATERIAL_FLAG_TEXTURE_TRANSFORM
// hands the sampler the very uv it always did.
template <typename Samples>
static void layerXfUv(const device GPUMaterialExtRecord *extensions, uint flags, uint index, uint slot, thread float2 &uv, thread float2 &dx, thread float2 &dy) {
    if (Samples::transforms()) texXfSlotUvDd(extensions, flags, index, slot, uv, dx, dy);
}
template <typename Samples>
static float3 layerXfTn(const device GPUMaterialExtRecord *extensions, uint flags, uint index, uint slot, float3 tn) {
    return Samples::transforms() ? texXfRotateTn(extensions, flags, index, slot, tn) : tn;
}

template <typename T, typename Samples>
static half4 layerSlotSample(const device T *table, uint index, const device GPUMaterialExtRecord *extensions, const device GPUMaterial &lm,
                             uint slot, float2 uv, float2 dx, float2 dy, half4 fallback) {
    layerXfUv<Samples>(extensions, lm.flags, lm.textureTransformIndex, slot, uv, dx, dy);
    return Samples::sample(table, index, uv, fallback, dx, dy);
}

struct LayerSurface {
    LayerWeights w;     // the final weights (normalised, height contrast applied)
    float3 baseColor;
    float roughness, metallic, occlusion;
    float3 tn;          // the tangent-space normal vector for normalize(T a + B b + N c)
    float3 emission;    // sum w_i emissive_i x emissive texture(uv_i); 0 when `wantEmission` is false or no layer emits
    float3 emitFactor[4], emitTexel[4]; // the per-layer factor and texel of that product (the primary resolve evaluates a sole layer's product inline, like a single material)
};

// The final layer weights (mask channels normalised, height contrast applied): shared by the base resolve and the coverage alpha.
template <typename T, typename Samples>
static LayerWeights resolveLayerWeights(GPUMaterialExt3 e3, const device T *textures, float2 uv, float2 dx, float2 dy,
                                        const device GPUMaterialExtRecord *extensions, uint stackFlags, uint stackIndex) {
    const uint count = min(e3.layerCount, LAYER_STACK_MAX);
    float2 uvm = uv, dxm = dx, dym = dy;
    layerXfUv<Samples>(extensions, stackFlags, stackIndex, TT_SLOT_LAYER_MASK, uvm, dxm, dym);
    const float4 mask = float4(Samples::mask(textures, e3.maskTex, uvm, half4(1.0h), dxm, dym));
    LayerWeights raw{{0.0f, 0.0f, 0.0f, 0.0f}};
    bool heightOn = false;
    for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count) {
            raw.w[i] = mask[(e3.packedMaskChannels >> (2u * i)) & 3u];
            heightOn = heightOn || e3.heightContrast[i] >= 0.0f;
        }
    }
    LayerWeights w = layerNormalise(raw, count);
    if (heightOn) {
        LayerHeights heights{{0.0f, 0.0f, 0.0f, 0.0f}};
        for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
            if (i < count && w.w[i] > 0.0f && e3.heightContrast[i] >= 0.0f) {
                const float s = e3.uvScale[i];
                float2 uvh = uv * s, dxh = dx * s, dyh = dy * s;
                layerXfUv<Samples>(extensions, stackFlags, stackIndex, TT_SLOT_LAYER_HEIGHT0 + i, uvh, dxh, dyh);
                const float texel = float(Samples::sample(textures, e3.heightTex[i], uvh, half4(0.5h), dxh, dyh).r);
                heights.h[i] = layerHeight(e3.heightContrast[i], texel);
            }
        }
        w = layerHeightBlend(w, heights, count);
    }
    return w;
}

// F15.4 (ticket 23) coverage of the stack: sum_i w_i a_i with a_i the alpha of layer i's base colour (factor x texture alpha at uv * uvScale
// through the layer's transform), the weights those of the base blend; the caller multiplies the stack's own record alpha and tests the STACK's
// cutoff.  A weight vector e_j gives layer j's alpha bit for bit.
template <typename T, typename Samples>
static float layerStackAlpha(GPUMaterialExt3 e3, const device GPUMaterial *materials, const device T *textures, float2 uv, float2 dx, float2 dy,
                             const device GPUMaterialExtRecord *extensions, uint stackFlags, uint stackIndex) {
    const uint count = min(e3.layerCount, LAYER_STACK_MAX);
    const LayerWeights w = resolveLayerWeights<T, Samples>(e3, textures, uv, dx, dy, extensions, stackFlags, stackIndex);
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count && w.w[i] > 0.0f) {
            const device GPUMaterial &lm = materials[e3.layerMaterial[i]];
            const float s = e3.uvScale[i];
            const half4 baseTex = layerSlotSample<T, Samples>(textures, lm.baseColorTex, extensions, lm, TT_SLOT_BASE_COLOR, uv * s, dx * s, dy * s, half4(1.0h));
            a[i] = lm.baseColor[3] * float(baseTex.a);
        }
    }
    return layerBlendScalar(w, a[0], a[1], a[2], a[3]);
}

template <typename T, typename Samples>
static LayerSurface resolveLayerBase(GPUMaterialExt3 e3, const device GPUMaterial *materials, const device T *textures, float2 uv, float2 dx,
                                     float2 dy, bool wantEmission, const device GPUMaterialExtRecord *extensions, uint stackFlags,
                                     uint stackIndex) {
    const uint count = min(e3.layerCount, LAYER_STACK_MAX);
    const LayerWeights w = resolveLayerWeights<T, Samples>(e3, textures, uv, dx, dy, extensions, stackFlags, stackIndex);
    float3 color[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)};
    float3 emit[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)};
    float3 emitF[4] = {float3(0.0f), float3(0.0f), float3(0.0f), float3(0.0f)}, emitT[4] = {float3(1.0f), float3(1.0f), float3(1.0f), float3(1.0f)};
    float rough[4] = {0.0f, 0.0f, 0.0f, 0.0f}, metal[4] = {0.0f, 0.0f, 0.0f, 0.0f}, occ[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    LayerNormals normals{{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
    for (uint i = 0; i < LAYER_STACK_MAX; ++i) {
        if (i < count && w.w[i] > 0.0f) {
            const device GPUMaterial &lm = materials[e3.layerMaterial[i]];
            const float s = e3.uvScale[i];
            const float2 uvi = uv * s, dxi = dx * s, dyi = dy * s;
            const half4 baseTex = layerSlotSample<T, Samples>(textures, lm.baseColorTex, extensions, lm, TT_SLOT_BASE_COLOR, uvi, dxi, dyi, half4(1.0h));
            const half4 mrTex = layerSlotSample<T, Samples>(textures, lm.metallicRoughnessTex, extensions, lm, TT_SLOT_METAL_ROUGH, uvi, dxi, dyi, half4(1.0h));
            const half4 aoTex = layerSlotSample<T, Samples>(textures, lm.occlusionTex, extensions, lm, TT_SLOT_OCCLUSION, uvi, dxi, dyi, half4(1.0h));
            const half4 nTex = layerSlotSample<T, Samples>(textures, lm.normalTex, extensions, lm, TT_SLOT_NORMAL, uvi, dxi, dyi, half4(0.5h, 0.5h, 1.0h, 1.0h));
            color[i] = float3(lm.baseColor[0], lm.baseColor[1], lm.baseColor[2]) * float3(baseTex.rgb);
            rough[i] = clamp(lm.roughness * float(mrTex.g), 0.04, 1.0);
            metal[i] = saturate(lm.metallic * float(mrTex.b));
            occ[i] = mix(1.0, float(aoTex.r), lm.occlusionStrength);
            float3 tn = float3(nTex.xyz) * 2.0 - 1.0;
            tn.xy *= lm.normalScale;
            tn = layerXfTn<Samples>(extensions, lm.flags, lm.textureTransformIndex, TT_SLOT_NORMAL, tn); // the layer's transform rotates its texel axes
            normals.tx[i] = tn.x;
            normals.ty[i] = tn.y;
            normals.tz[i] = tn.z;
            // The layer's own emission (a material of the library: factor x texture), the same expression as the single material's.
            if (wantEmission && (lm.emissive[0] > 0.0f || lm.emissive[1] > 0.0f || lm.emissive[2] > 0.0f)) {
                const half4 emTex = layerSlotSample<T, Samples>(textures, lm.emissiveTex, extensions, lm, TT_SLOT_EMISSIVE, uvi, dxi, dyi, half4(1.0h));
                emitF[i] = float3(lm.emissive[0], lm.emissive[1], lm.emissive[2]);
                emitT[i] = float3(emTex.rgb);
                emit[i] = emitF[i] * emitT[i];
            }
        }
    }
    LayerSurface out;
    out.w = w;
    out.baseColor = float3(layerBlendScalar(w, color[0].x, color[1].x, color[2].x, color[3].x),
                           layerBlendScalar(w, color[0].y, color[1].y, color[2].y, color[3].y),
                           layerBlendScalar(w, color[0].z, color[1].z, color[2].z, color[3].z));
    out.roughness = layerBlendScalar(w, rough[0], rough[1], rough[2], rough[3]);
    out.metallic = layerBlendScalar(w, metal[0], metal[1], metal[2], metal[3]);
    out.occlusion = layerBlendScalar(w, occ[0], occ[1], occ[2], occ[3]);
    const LayerNormalResult n = layerNormalBlend(w, normals);
    out.tn = float3(n.a, n.b, n.c);
    for (uint k = 0; k < LAYER_STACK_MAX; ++k) {
        out.emitFactor[k] = emitF[k];
        out.emitTexel[k] = emitT[k];
    }
    out.emission = float3(layerBlendScalar(w, emit[0].x, emit[1].x, emit[2].x, emit[3].x),
                          layerBlendScalar(w, emit[0].y, emit[1].y, emit[2].y, emit[3].y),
                          layerBlendScalar(w, emit[0].z, emit[1].z, emit[2].z, emit[3].z));
    return out;
}

// The stack coverage multiplier of a material at one alpha site: 1 unless `m` is a layer stack with a cutoff (the stack's own record alpha is
// the caller's factor).  Level-0 / gradient policies are the caller's choice.
template <typename T, typename Samples>
static float layerCoverage(const device GPUMaterial &m, const device GPUMaterial *materials, const device T *textures, float2 uv, float2 dx,
                           float2 dy, const device GPUMaterialExtRecord *extensions) {
    if ((m.flags & (MATERIAL_FEATURE_LAYERS << MATERIAL_FEATURE_SHIFT)) == 0u || !(m.alphaCutoff > 0.0f) || extensions == nullptr ||
        m.extensionIndex == MATERIAL_EXTENSION_INVALID)
        return 1.0f;
    const GPUMaterialExt3 e3 = *reinterpret_cast<const device GPUMaterialExt3 *>(extensions + m.extensionIndex + 3u);
    return layerStackAlpha<T, Samples>(e3, materials, textures, uv, dx, dy, extensions, m.flags, m.textureTransformIndex);
}
