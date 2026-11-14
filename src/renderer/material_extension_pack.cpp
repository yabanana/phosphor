#include "renderer/material_extension_pack.h"

#include <algorithm>

namespace phosphor {

namespace {

template <typename T>
GPUMaterialExtRecord toRecord(const T& v) {
    static_assert(sizeof(T) == sizeof(GPUMaterialExtRecord) && std::is_trivially_copyable_v<T>);
    GPUMaterialExtRecord r;
    std::memcpy(&r, &v, sizeof(T));
    return r;
}

void copy3(float* dst, const std::array<float, 3>& src) {
    for (size_t i = 0; i < 3; ++i) dst[i] = src[i];
}

u32 channelCode(TextureChannel c) {
    switch (c) {
        case TextureChannel::R: return 0;
        case TextureChannel::G: return 1;
        case TextureChannel::B: return 2;
        case TextureChannel::A: return 3;
        default: return 0; // validate() restricts layer masks to one channel
    }
}

} // namespace

MaterialExtGroup packMaterialExtensions(const MaterialDescription& d) {
    MaterialExtGroup out{};

    // Record 0: specular + ior, clearcoat, sheen.
    {
        const SpecularBlock  s{};
        const ClearcoatBlock c{};
        const SheenBlock     h{};
        const SpecularBlock&  sp = d.specular ? *d.specular : s;
        const ClearcoatBlock& cc = d.clearcoat ? *d.clearcoat : c;
        const SheenBlock&     sh = d.sheen ? *d.sheen : h;
        GPUMaterialExt0 r{};
        r.ior            = sp.ior;
        r.specularFactor = sp.specularFactor;
        copy3(r.specularColor, sp.specularColor);
        r.specularTex      = sp.specularTexture.index;
        r.specularColorTex = sp.specularColorTexture.index;
        r.clearcoatFactor        = cc.factor;
        r.clearcoatRoughness     = cc.roughness;
        r.clearcoatNormalScale   = cc.normalScale;
        r.clearcoatFactorTex     = cc.factorTexture.index;
        r.clearcoatRoughnessTex  = cc.roughnessTexture.index;
        r.clearcoatNormalTex     = cc.normalTexture.index;
        copy3(r.sheenColor, sh.color);
        r.sheenRoughness    = sh.roughness;
        r.sheenColorTex     = sh.colorTexture.index;
        r.sheenRoughnessTex = sh.roughnessTexture.index;
        out[0] = toRecord(r);
    }
    // Record 1: anisotropy, iridescence, transmission, volume.
    {
        const AnisotropyBlock   a{};
        const IridescenceBlock  i{};
        const TransmissionBlock t{};
        const VolumeBlock       v{};
        const AnisotropyBlock&   an = d.anisotropy ? *d.anisotropy : a;
        const IridescenceBlock&  ir = d.iridescence ? *d.iridescence : i;
        const TransmissionBlock& tr = d.transmission ? *d.transmission : t;
        const VolumeBlock&       vo = d.volume ? *d.volume : v;
        GPUMaterialExt1 r{};
        r.anisotropyStrength = an.strength;
        r.anisotropyRotation = an.rotation;
        r.anisotropyTex      = an.texture.index;
        r.iridescenceFactor       = ir.factor;
        r.iridescenceIor          = ir.ior;
        r.iridescenceThicknessMin = ir.thicknessMin;
        r.iridescenceThicknessMax = ir.thicknessMax;
        r.iridescenceFactorTex    = ir.factorTexture.index;
        r.iridescenceThicknessTex = ir.thicknessTexture.index;
        r.transmissionFactor = tr.factor;
        r.transmissionTex    = tr.texture.index;
        r.volumeThickness    = vo.thickness;
        r.volumeThicknessTex = vo.thicknessTexture.index;
        copy3(r.attenuationColor, vo.attenuationColor);
        r.attenuationDistance = vo.attenuationDistance;
        out[1] = toRecord(r);
    }
    // Record 2: subsurface.
    {
        const SubsurfaceBlock b{};
        const SubsurfaceBlock& s = d.subsurface ? *d.subsurface : b;
        GPUMaterialExt2 r{};
        r.mode   = s.mode == SubsurfaceMode::Volumetric ? 1u : 0u;
        r.weight = s.weight;
        copy3(r.color, s.color);
        r.radius = s.radius;
        copy3(r.radiusScale, s.radiusScale);
        r.anisotropy     = s.anisotropy;
        r.weightTex      = s.weightTexture.index;
        r.colorTex       = s.colorTexture.index;
        r.radiusScaleTex = s.radiusScaleTexture.index;
        out[2] = toRecord(r);
    }
    // Record 3: layer stack.
    {
        GPUMaterialExt3 r{};
        r.maskTex = INVALID_TEXTURE_INDEX;
        for (u32 k = 0; k < MATERIAL_LAYER_MAX; ++k) {
            r.layerMaterial[k]  = MATERIAL_EXTENSION_INVALID;
            r.heightContrast[k] = -1.0f;
            // uvScale / heightTex stay 0 for a layer that does not exist: the record of a material without a stack is byte-identical to
            // the pre-F15.4 one (snapshots, references and delta payloads of every other scene are unchanged).
        }
        if (d.layers) {
            const LayerStack& ls = *d.layers;
            r.maskTex    = ls.mask.index;
            r.layerCount = static_cast<u32>(std::min<size_t>(ls.layers.size(), MATERIAL_LAYER_MAX));
            for (u32 k = 0; k < r.layerCount; ++k) {
                const MaterialLayer& l = ls.layers[k];
                r.layerMaterial[k] = l.materialIndex;
                r.packedMaskChannels |= channelCode(l.maskChannel) << (2u * k);
                r.heightContrast[k] = l.heightContrast ? *l.heightContrast : -1.0f;
                r.uvScale[k]        = l.uvScale;
                r.heightTex[k]      = l.heightTexture.index;
            }
        }
        out[3] = toRecord(r);
    }
    return out;
}

} // namespace phosphor
