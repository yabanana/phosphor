#pragma once
#include "renderer/shadow_math.h"
#include "renderer/transmission_math.h"
// F15 closure (ticket 15): the receiver side of the CSM transmittance map.  A receiver that is itself a transmissive surface contributes its own
// surface (and, in the single-layer sum, only that texel's layers) to the map entry above it: the sun reflected by a pane behind another pane
// must be attenuated by the pane in front, not by its own Fresnel pass and tint again.  The guide producers (shadow_surface_guides,
// shadow_receiver_pack) evaluate this term once per pixel with the receiver's own uv and write it to a transient texture; the PCSS kernel
// subtracts it from every tap of the map (the ray-traced shadow ray starts at the receiver and never meets it).
// Returned: the ln RGB of the Fresnel pass and of factor x tint of the receiver surface, without the medium term (the medium term of a closed
// volume pairs its two faces: a receiver on the face the shadow ray ENTERS keeps it, which is exactly the ray-traced receiver inside the slab).
// Zero for an opaque receiver and for the face of a closed solid that faces the light (declared: the single-layer map cannot separate its
// back face from the layers above it; such a receiver is nearest in its texel unless another layer lies above it).
inline float3 shadowReceiverOwnLn(const device GPUMaterial* materials, uint materialIndex, uint materialCount,
                                  const device RtTextureHandle* textures, const device GPUMaterialExtRecord* extensions,
                                  float2 uv, float3 outward, float3 toLight) {
    if (materialIndex >= materialCount) return float3(0.0f);
    const device GPUMaterial& m = materials[materialIndex];
    if (!shadowCasterTransmissive(m.flags, m.extensionIndex, m.alphaCutoff)) return float3(0.0f);
    const uint features = (m.flags >> MATERIAL_FEATURE_SHIFT) & MATERIAL_FEATURES_ALL;
    const device GPUMaterialExt1& e1 = *reinterpret_cast<const device GPUMaterialExt1*>(extensions + m.extensionIndex + 1u);
    const bool solid = (features & MATERIAL_FEATURE_VOLUME) != 0u && e1.volumeThickness > 0.0f;
    const float cosine = dot(outward, toLight);
    const bool entry = cosine < 0.0f; // the shadow ray from the receiver goes against the outward normal: it enters the volume here
    if (solid && !entry) return float3(0.0f);
    float ior = kShadowDefaultIor;
    if ((features & MATERIAL_FEATURE_SPECULAR_IOR) != 0u)
        ior = reinterpret_cast<const device GPUMaterialExt0*>(extensions + m.extensionIndex)->ior;
    float factor = e1.transmissionFactor;
    if (e1.transmissionTex != INVALID_TEXTURE_INDEX) factor *= float(half(textures[e1.transmissionTex].tex.sample(kRtAlphaSampler, uv, level(0)).r));
    float metallic = m.metallic;
    if (m.metallicRoughnessTex != INVALID_TEXTURE_INDEX) metallic *= float(half(textures[m.metallicRoughnessTex].tex.sample(kRtAlphaSampler, uv, level(0)).b));
    float3 tint = float3(m.baseColor[0], m.baseColor[1], m.baseColor[2]);
    if (m.baseColorTex != INVALID_TEXTURE_INDEX) tint *= float3(half3(textures[m.baseColorTex].tex.sample(kRtAlphaSampler, uv, level(0)).rgb));
    factor = saturate(factor) * (1.0f - saturate(metallic));
    const TransmissionShadowLn ln = transmissionShadowSurfaceLn(abs(cosine), entry, false, ior, factor, tint.x, tint.y, tint.z,
        e1.attenuationColor[0], e1.attenuationColor[1], e1.attenuationColor[2], e1.attenuationDistance, 0.0f);
    return float3(ln.r, ln.g, ln.b);
}
