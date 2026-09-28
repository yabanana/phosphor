#version 460
#extension GL_EXT_ray_tracing                              : require
#extension GL_EXT_buffer_reference                         : require
#extension GL_EXT_buffer_reference2                        : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8    : require
#extension GL_EXT_scalar_block_layout                      : require
#extension GL_EXT_nonuniform_qualifier                     : require

// ---------------------------------------------------------------------------
// DDGI Probe Ray Closest-Hit Shader
//
// Fetches the hit surface position and normal from the vertex data via BDA,
// computes a simple direct lighting contribution (one directional light),
// and returns radiance + hit distance through the payload.
// ---------------------------------------------------------------------------

// -- Buffer references (must match types.glsl) --------------------------------

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBufferArray {
    float data[];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer InstanceBuffer {
    mat4 modelMatrix;
    uint meshIndex;
    uint materialIndex;
    uint flags;
    uint pad;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MeshInfoBuffer {
    uint meshletCount;
    uint meshletOffset;
    uint vertexOffset;
    uint indexOffset;
    vec4 boundingSphere;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MaterialBuffer {
    vec4  baseColor;
    float metallic;
    float roughness;
    float normalScale;
    float occlusionStrength;
    uint  baseColorTex;
    uint  normalTex;
    uint  metallicRoughnessTex;
    uint  occlusionTex;
    uint  emissiveTex;
    vec3  emissive;
    float alphaCutoff;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer LightBuffer {
    uint  type;
    vec3  position;
    vec3  direction;
    vec3  color;
    float intensity;
    float range;
    float innerCone;
    float outerCone;
    uint  shadowMapIndex;
};

// -- Scene globals (BDA root pointer) ----------------------------------------

layout(set = 1, binding = 3, scalar) readonly buffer SceneGlobalsRef {
    uint64_t vertexAddr;
    uint64_t meshletAddr;
    uint64_t meshletVertexAddr;
    uint64_t meshletTriangleAddr;
    uint64_t instanceAddr;
    uint64_t materialAddr;
    uint64_t meshInfoAddr;
    uint64_t lightAddr;
    uint64_t meshletBoundsAddr;
    uint     instanceCount;
    uint     lightCount;
    uint     meshletTotalCount;
    uint     scenePad;
} sceneGlobals;

// -- Ray payload -------------------------------------------------------------

layout(location = 0) rayPayloadInEXT vec4 payload;

// -- Hit attributes (barycentrics) -------------------------------------------

hitAttributeEXT vec2 hitAttribs;

// ---------------------------------------------------------------------------
// Vertex loading helper (12 floats = 48 bytes per vertex)
// ---------------------------------------------------------------------------

struct SimpleVertex {
    vec3 position;
    vec3 normal;
};

SimpleVertex loadVertexSimple(uint64_t vertexAddr, uint index) {
    VertexBufferArray vbuf = VertexBufferArray(vertexAddr);
    uint base = index * 12;

    SimpleVertex v;
    v.position = vec3(vbuf.data[base + 0], vbuf.data[base + 1], vbuf.data[base + 2]);
    v.normal   = vec3(vbuf.data[base + 3], vbuf.data[base + 4], vbuf.data[base + 5]);
    return v;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

void main() {
    // -- Retrieve instance and primitive info --
    uint instanceID  = gl_InstanceCustomIndexEXT;
    uint primitiveID = gl_PrimitiveID;

    // Load instance data
    InstanceBuffer inst = InstanceBuffer(
        sceneGlobals.instanceAddr + uint64_t(instanceID) * 80);
    uint meshIdx     = inst.meshIndex;
    uint materialIdx = inst.materialIndex;
    mat4 modelMatrix = inst.modelMatrix;

    // Load mesh info to get vertex offset
    MeshInfoBuffer meshInfo = MeshInfoBuffer(
        sceneGlobals.meshInfoAddr + uint64_t(meshIdx) * 32);
    uint vertexOffset = meshInfo.vertexOffset;

    // -- Compute hit barycentrics --
    vec3 barycentrics = vec3(1.0 - hitAttribs.x - hitAttribs.y,
                             hitAttribs.x,
                             hitAttribs.y);

    // -- Load triangle vertices --
    // gl_PrimitiveID is the triangle index within this geometry.
    // The BLAS was built from the mesh's own vertex/index range, so
    // the vertex indices here are relative to the mesh's vertex buffer.
    uint i0 = primitiveID * 3 + 0 + vertexOffset;
    uint i1 = primitiveID * 3 + 1 + vertexOffset;
    uint i2 = primitiveID * 3 + 2 + vertexOffset;

    SimpleVertex v0 = loadVertexSimple(sceneGlobals.vertexAddr, i0);
    SimpleVertex v1 = loadVertexSimple(sceneGlobals.vertexAddr, i1);
    SimpleVertex v2 = loadVertexSimple(sceneGlobals.vertexAddr, i2);

    // -- Interpolate position and normal --
    vec3 localPos = v0.position * barycentrics.x +
                    v1.position * barycentrics.y +
                    v2.position * barycentrics.z;

    vec3 localNorm = normalize(
        v0.normal * barycentrics.x +
        v1.normal * barycentrics.y +
        v2.normal * barycentrics.z);

    // Transform to world space
    vec3 worldPos  = (modelMatrix * vec4(localPos, 1.0)).xyz;
    vec3 worldNorm = normalize(mat3(modelMatrix) * localNorm);

    // -- Material base colour --
    vec3 albedo = vec3(0.8);
    if (sceneGlobals.materialAddr != 0) {
        MaterialBuffer mat = MaterialBuffer(
            sceneGlobals.materialAddr + uint64_t(materialIdx) * 80);
        albedo = mat.baseColor.rgb;
    }

    // -- Direct lighting (first directional light, or fallback sun) --
    vec3 sunDir   = normalize(vec3(0.4, 0.8, 0.3));
    vec3 sunColor = vec3(1.0, 0.95, 0.85);
    float sunIntensity = 3.0;

    // Try to use scene light if available
    if (sceneGlobals.lightCount > 0 && sceneGlobals.lightAddr != 0) {
        LightBuffer light = LightBuffer(sceneGlobals.lightAddr);
        if (light.type == 0) { // directional
            sunDir       = normalize(-light.direction);
            sunColor     = light.color;
            sunIntensity = light.intensity;
        }
    }

    float NdotL = max(dot(worldNorm, sunDir), 0.0);
    vec3 directLighting = albedo * sunColor * sunIntensity * NdotL;

    // Add a small ambient term to avoid pure-black hits
    vec3 ambient = albedo * 0.05;

    vec3 radiance = directLighting + ambient;

    // -- Hit distance --
    float hitDist = gl_HitTEXT;

    payload = vec4(radiance, hitDist);
}
