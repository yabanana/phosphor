// One exported alpha entry shared by all statically linked RT consumers.
#include "rt_common.h"
#include "texture_transform_core.h" // F15.1 (ticket 22): per-slot texture transforms at the secondary sites

// F15.1 (ticket 14): the transmittance of ONE transmissive triangle crossed by a RT_PROBE_SHADOW_TRANSMITTED ray, accumulated in log space
// into the payload (order independent: no sorting of the hits).  Hardware invokes the intersection function for every non-opaque triangle the
// ray meets; the transmissive instances are non-opaque (rt_write_instances) so that it sees them.  The caller returns false, which rejects the
// triangle and lets the traversal continue to the light; an opaque or alpha-masked caster that is accepted ends the ray (any-hit).  The facing
// of the hit (entry = the ray goes against the outward normal, exit = along it) comes from the world-space winding.
inline void rtAccumulateTransmittance(uint primitive, uint slot, float2 bary, float distance, ray_data RtPayload& payload,
                                      const device GPUMaterial& m, const device GPUInstance& instance, const device GPURtMesh& mesh,
                                      const device RtTextureHandle* textures, const device GPUVertex* vertices, const device uint* indices,
                                      const device GPUMaterialExtRecord* extensions, uint transmittanceFlags) {
    ++payload.ifInvocations;
    // The duplicate filter: Metal may invoke the function more than once for the same primitive (measured by the F15 ticket 14 spike).
    bool seen = false;
    for (uint i = 0; i < payload.listCount; ++i) seen |= all(payload.listed[i] == uint2(slot, primitive));
    if (!seen) {
        ++payload.ifDistinct;
        if (payload.listCount < kRtTransmitList) payload.listed[payload.listCount++] = uint2(slot, primitive);
        else ++payload.listOverflows;
    }
    if (seen && (transmittanceFlags & RT_TRANSMITTANCE_NO_DEDUP) == 0u) return;
    const uint base = mesh.indexOffset + 3u * primitive;
    const device GPUVertex& v0 = vertices[mesh.vertexOffset + indices[base]];
    const device GPUVertex& v1 = vertices[mesh.vertexOffset + indices[base + 1u]];
    const device GPUVertex& v2 = vertices[mesh.vertexOffset + indices[base + 2u]];
    const float3 a = rtWorldPoint(instance, rtPosition(v0)), b = rtWorldPoint(instance, rtPosition(v1)), c = rtWorldPoint(instance, rtPosition(v2));
    float3 n = cross(b - a, c - a);
    const float len2 = dot(n, n);
    if (!(len2 > 1e-30f)) return;
    n *= rsqrt(len2);
    if ((instance.flags & INSTANCE_FLAG_MIRRORED) != 0u) n = -n; // the world winding of a mirrored instance is reversed
    const float3 direction = normalize(float3(payload.direction[0], payload.direction[1], payload.direction[2]));
    const float cosine = dot(n, direction);
    const bool entry = cosine < 0.0f;
    const float2 uv0(v0.u, v0.v), uv1(v1.u, v1.v), uv2(v2.u, v2.v);
    const float2 uv = uv0 * (1.0f - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
    const uint features = (m.flags >> MATERIAL_FEATURE_SHIFT) & MATERIAL_FEATURES_ALL;
    const device GPUMaterialExt1& e1 = *reinterpret_cast<const device GPUMaterialExt1*>(extensions + m.extensionIndex + 1u);
    const bool solid = (features & MATERIAL_FEATURE_VOLUME) != 0u && e1.volumeThickness > 0.0f; // thin-walled: one surface, no medium
    float ior = kShadowDefaultIor;
    if ((features & MATERIAL_FEATURE_SPECULAR_IOR) != 0u)
        ior = reinterpret_cast<const device GPUMaterialExt0*>(extensions + m.extensionIndex)->ior;
    float factor = 1.0f;
    float3 tint(1.0f);
    if (entry || !solid) {
        factor = e1.transmissionFactor;
        // F15.1 (ticket 22): each slot samples its own KHR_texture_transform (the uv itself for a material without one).
        if (e1.transmissionTex != INVALID_TEXTURE_INDEX) factor *= float(half(textures[e1.transmissionTex].tex.sample(kRtAlphaSampler, texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_TRANSMISSION, uv), level(0)).r));
        float metallic = m.metallic;
        if (m.metallicRoughnessTex != INVALID_TEXTURE_INDEX) metallic *= float(half(textures[m.metallicRoughnessTex].tex.sample(kRtAlphaSampler, texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_METAL_ROUGH, uv), level(0)).b));
        tint = float3(m.baseColor[0], m.baseColor[1], m.baseColor[2]);
        if (m.baseColorTex != INVALID_TEXTURE_INDEX) tint *= float3(half3(textures[m.baseColorTex].tex.sample(kRtAlphaSampler, texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_BASE_COLOR, uv), level(0)).rgb));
        factor = saturate(factor) * (1.0f - saturate(metallic));
    }
    // Shared with the CSM transmittance pass (F15.1 ticket 15): the entry factor, the Fresnel pass and the signed medium term.
    const TransmissionShadowLn ln = transmissionShadowSurfaceLn(cosine, entry, solid, ior, factor, tint.x, tint.y, tint.z,
        e1.attenuationColor[0], e1.attenuationColor[1], e1.attenuationColor[2], e1.attenuationDistance, distance);
    const float ln0 = ln.r, ln1 = ln.g, ln2 = ln.b;
    payload.lnTransmittance[0] += ln0; payload.lnTransmittance[1] += ln1; payload.lnTransmittance[2] += ln2;
}

// The alpha-masked (alphaCutoff > 0) test shared by both linked entries.
template<typename Payload>
inline bool rtAlphaMaskedTest(uint primitive, float2 bary, float distance, ray_data Payload& payload, const device GPUMaterial& m,
                              const device GPUInstance& instance, const device GPURtMesh* meshes, const device RtTextureHandle* textures,
                              const device GPUVertex* vertices, const device uint* indices,
                              const device GPUMaterialExtRecord* extensions) {
    const device GPURtMesh& mesh = meshes[instance.meshIndex];
    if (primitive >= mesh.indexCount / 3u) return false;
    if (m.baseColorTex == INVALID_TEXTURE_INDEX) return m.baseColor[3] >= m.alphaCutoff;
    const uint base = mesh.indexOffset + 3u * primitive;
    const device GPUVertex& v0 = vertices[mesh.vertexOffset + indices[base]];
    const device GPUVertex& v1 = vertices[mesh.vertexOffset + indices[base + 1u]];
    const device GPUVertex& v2 = vertices[mesh.vertexOffset + indices[base + 2u]];
    // F15.1 (ticket 22): the base colour slot's KHR_texture_transform moves the vertex uvs (affine: the interpolation and the cone LOD
    // follow); a material without one keeps every value as it was.
    const float2 uv0 = texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_BASE_COLOR, float2(v0.u, v0.v)),
                 uv1 = texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_BASE_COLOR, float2(v1.u, v1.v)),
                 uv2 = texXfSlotUv(extensions, m.flags, m.textureTransformIndex, TT_SLOT_BASE_COLOR, float2(v2.u, v2.v));
    const float2 uv = uv0 * (1.0f - bary.x - bary.y) + uv1 * bary.x + uv2 * bary.y;
    const auto texture = textures[m.baseColorTex].tex;
    float lod = 0.0f;
    if (payload.type == RT_PROBE_PRIMARY && payload.coneWidth > 0.0f) {
        lod = rtConeLod(payload.coneWidth, distance,
                        rtWorldPoint(instance, rtPosition(v0)), rtWorldPoint(instance, rtPosition(v1)),
                        rtWorldPoint(instance, rtPosition(v2)), uv0, uv1, uv2,
                        uint2(texture.get_width(), texture.get_height()));
    }
    const float alpha = m.baseColor[3] * float(half(texture.sample(kRtAlphaSampler, uv, level(lod)).a));
    return alpha >= m.alphaCutoff;
}

[[intersection(triangle, triangle_data, instancing)]]
bool rt_alpha_generic(uint primitive [[primitive_id]], uint slot [[user_instance_id]],
                       float2 bary [[barycentric_coord]], float distance [[distance]],
                       ray_data RtPayload& payload [[payload]],
                       const device GPUMaterial* materials [[buffer(0)]],
                       const device RtTextureHandle* textures [[buffer(1)]],
                       const device GPUVertex* vertices [[buffer(2)]],
                       const device uint* indices [[buffer(3)]],
                       const device GPUInstance* instances [[buffer(4)]],
                       const device GPURtMesh* meshes [[buffer(5)]],
                       constant GPURtParams& params [[buffer(6)]],
                       const device GPUMaterialExtRecord* extensions [[buffer(7)]]) {
    ++payload.alphaTests;
    if (slot >= params.slotCount) return false;
    const device GPUInstance& instance = instances[slot];
    if (instance.meshIndex >= params.meshCount || instance.materialIndex >= params.materialCount) return false;
    const device GPUMaterial& m = materials[instance.materialIndex];
    // F15.1 (ticket 14): a transmissive instance is non-opaque only so that transmittance rays see it.
    const bool transmissive = (m.flags & (MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT)) != 0u &&
                              m.extensionIndex != MATERIAL_EXTENSION_INVALID;
    if (m.alphaCutoff <= 0.0f) {
        if (transmissive && payload.type == RT_PROBE_SHADOW_TRANSMITTED) {
            const device GPURtMesh& mesh = meshes[instance.meshIndex];
            if (primitive < mesh.indexCount / 3u) {
                rtAccumulateTransmittance(primitive, slot, bary, distance, payload, m, instance, mesh, textures, vertices, indices, extensions,
                                          params.transmittanceFlags);
                // The spike's control: a duplicated invocation of the same triangle (hardware does not produce one on the M5 Max, see the opt log).
                if ((params.transmittanceFlags & RT_TRANSMITTANCE_INJECT_DUPLICATE) != 0u)
                    rtAccumulateTransmittance(primitive, slot, bary, distance, payload, m, instance, mesh, textures, vertices, indices, extensions,
                                              params.transmittanceFlags);
            }
            return false;
        }
        // Must stay zero: opaque instances have the hardware Opaque option (a transmissive instance is the declared exception).
        if (!transmissive) ++payload.opaqueAlphaTests;
        return true;
    }
    return rtAlphaMaskedTest(primitive, bary, distance, payload, m, instance, meshes, textures, vertices, indices, extensions);
}

// F15 ticket 24: the entry of every pipeline that never traces a RT_PROBE_SHADOW_TRANSMITTED ray.  The payload is the 16-byte
// RtPayloadLite and a transmissive instance behaves as for any non-transmittance ray of rt_alpha_generic (an accepted hit).
[[intersection(triangle, triangle_data, instancing)]]
bool rt_alpha_lite(uint primitive [[primitive_id]], uint slot [[user_instance_id]],
                    float2 bary [[barycentric_coord]], float distance [[distance]],
                    ray_data RtPayloadLite& payload [[payload]],
                    const device GPUMaterial* materials [[buffer(0)]],
                    const device RtTextureHandle* textures [[buffer(1)]],
                    const device GPUVertex* vertices [[buffer(2)]],
                    const device uint* indices [[buffer(3)]],
                    const device GPUInstance* instances [[buffer(4)]],
                    const device GPURtMesh* meshes [[buffer(5)]],
                    constant GPURtParams& params [[buffer(6)]],
                    const device GPUMaterialExtRecord* extensions [[buffer(7)]]) { // F15.1 (ticket 22): the IFT binds it for every table
    ++payload.alphaTests;
    if (slot >= params.slotCount) return false;
    const device GPUInstance& instance = instances[slot];
    if (instance.meshIndex >= params.meshCount || instance.materialIndex >= params.materialCount) return false;
    const device GPUMaterial& m = materials[instance.materialIndex];
    if (m.alphaCutoff <= 0.0f) {
        const bool transmissive = (m.flags & (MATERIAL_FEATURE_TRANSMISSION << MATERIAL_FEATURE_SHIFT)) != 0u &&
                                  m.extensionIndex != MATERIAL_EXTENSION_INVALID;
        if (!transmissive) ++payload.opaqueAlphaTests;
        return true;
    }
    return rtAlphaMaskedTest(primitive, bary, distance, payload, m, instance, meshes, textures, vertices, indices, extensions);
}
