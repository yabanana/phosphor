// One exported alpha entry shared by all statically linked RT consumers.
#include "rt_common.h"

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
                       constant GPURtParams& params [[buffer(6)]]) {
    ++payload.alphaTests;
    if (slot >= params.slotCount) return false;
    const device GPUInstance& instance = instances[slot];
    if (instance.meshIndex >= params.meshCount || instance.materialIndex >= params.materialCount) return false;
    const device GPUMaterial& m = materials[instance.materialIndex];
    if (m.alphaCutoff <= 0.0f) {
        // Must stay zero: opaque instances have the hardware Opaque option.
        ++payload.opaqueAlphaTests;
        return true;
    }
    const device GPURtMesh& mesh = meshes[instance.meshIndex];
    if (primitive >= mesh.indexCount / 3u) return false;
    if (m.baseColorTex == INVALID_TEXTURE_INDEX) return m.baseColor[3] >= m.alphaCutoff;
    const uint base = mesh.indexOffset + 3u * primitive;
    const device GPUVertex& v0 = vertices[mesh.vertexOffset + indices[base]];
    const device GPUVertex& v1 = vertices[mesh.vertexOffset + indices[base + 1u]];
    const device GPUVertex& v2 = vertices[mesh.vertexOffset + indices[base + 2u]];
    const float2 uv0(v0.u, v0.v), uv1(v1.u, v1.v), uv2(v2.u, v2.v);
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

