#ifndef TYPES_GLSL
#define TYPES_GLSL

#extension GL_EXT_buffer_reference          : require
#extension GL_EXT_buffer_reference2         : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout       : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

// ---------------------------------------------------------------------------
// Buffer-reference GPU structs (must match C++ gpu_scene.h layout exactly)
// ---------------------------------------------------------------------------

// 48 bytes per vertex
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBuffer {
    float px, py, pz;      // position
    float nx, ny, nz;      // normal
    float tx, ty, tz, tw;  // tangent + handedness
    float u, v;             // texcoord
};

// For strided access into the vertex buffer
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBufferArray {
    float data[];
};

// 16 bytes per meshlet
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MeshletBuffer {
    uint vertexOffset;
    uint vertexCount;
    uint triangleOffset;
    uint triangleCount;
};

// 48 bytes per meshlet bounds
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MeshletBoundsBuffer {
    vec3  center;
    float radius;
    vec3  coneApex;
    float coneCutoff;
    vec3  coneAxis;
    float pad;
};

// Meshlet vertex index buffer (uint per entry)
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MeshletVertexBuffer {
    uint indices[];
};

// Meshlet triangle buffer (byte-packed, 3 bytes per triangle)
layout(buffer_reference, scalar, buffer_reference_align = 1) readonly buffer MeshletTriangleBuffer {
    uint8_t indices[];
};

// 80 bytes per instance
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer InstanceBuffer {
    mat4 modelMatrix;
    uint meshIndex;
    uint materialIndex;
    uint flags;
    uint pad;
};

// 32 bytes per mesh info
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MeshInfoBuffer {
    uint meshletCount;
    uint meshletOffset;
    uint vertexOffset;
    uint indexOffset;
    vec4 boundingSphere;    // center xyz + radius w
};

// 80 bytes per material
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

// GPULight
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer LightBuffer {
    uint  type;             // 0=directional, 1=point, 2=spot
    vec3  position;
    vec3  direction;
    vec3  color;
    float intensity;
    float range;
    float innerCone;
    float outerCone;
    uint  shadowMapIndex;
};

// SceneGlobals — root pointer table
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer SceneGlobalsBuffer {
    uint64_t vertexAddr;
    uint64_t meshletAddr;
    uint64_t meshletVertexAddr;
    uint64_t meshletTriangleAddr;
    uint64_t instanceAddr;
    uint64_t materialAddr;
    uint64_t meshInfoAddr;
    uint64_t lightAddr;
    uint64_t meshletBoundsAddr;
    uint    instanceCount;
    uint    lightCount;
    uint    meshletTotalCount;
    uint    pad;
};

// ---------------------------------------------------------------------------
// Convenience structs for loading data without BDA (plain GLSL)
// ---------------------------------------------------------------------------

struct GPUVertex {
    float px, py, pz;
    float nx, ny, nz;
    float tx, ty, tz, tw;
    float u, v;
};

struct Meshlet {
    uint vertexOffset;
    uint vertexCount;
    uint triangleOffset;
    uint triangleCount;
};

struct MeshletBounds {
    vec3  center;
    float radius;
    vec3  coneApex;
    float coneCutoff;
    vec3  coneAxis;
    float pad;
};

struct GPUInstance {
    mat4 modelMatrix;
    uint meshIndex;
    uint materialIndex;
    uint flags;
    uint pad;
};

struct GPUMeshInfo {
    uint meshletCount;
    uint meshletOffset;
    uint vertexOffset;
    uint indexOffset;
    vec4 boundingSphere;
};

struct GPUMaterial {
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

struct GPULight {
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

// ---------------------------------------------------------------------------
// Push constants — 128 bytes, bound to all pipelines
// ---------------------------------------------------------------------------

layout(push_constant) uniform PushConstants {
    mat4     viewProjection;           // 64 bytes
    vec4     cameraPosition;           // 16 bytes (w = time)
    uint64_t sceneGlobalsAddress;      // 8 bytes
    uint64_t vertexBufferAddress;      // 8 bytes
    uint64_t meshletBufferAddress;     // 8 bytes
    uvec2    resolution;               // 8 bytes
    uint     frameIndex;               // 4 bytes
    uint     lightCount;               // 4 bytes
    float    exposure;                 // 4 bytes
    uint     debugMode;                // 4 bytes
} pc;                                  // total: 128 bytes

// ---------------------------------------------------------------------------
// Helper: load a vertex from the vertex buffer via BDA
// ---------------------------------------------------------------------------

GPUVertex loadVertex(uint64_t vertexAddr, uint index) {
    // 48 bytes = 12 floats per vertex
    VertexBufferArray vbuf = VertexBufferArray(vertexAddr);
    uint base = index * 12;

    GPUVertex vtx;
    vtx.px = vbuf.data[base + 0];
    vtx.py = vbuf.data[base + 1];
    vtx.pz = vbuf.data[base + 2];
    vtx.nx = vbuf.data[base + 3];
    vtx.ny = vbuf.data[base + 4];
    vtx.nz = vbuf.data[base + 5];
    vtx.tx = vbuf.data[base + 6];
    vtx.ty = vbuf.data[base + 7];
    vtx.tz = vbuf.data[base + 8];
    vtx.tw = vbuf.data[base + 9];
    vtx.u  = vbuf.data[base + 10];
    vtx.v  = vbuf.data[base + 11];
    return vtx;
}

// ---------------------------------------------------------------------------
// Helper: load a meshlet from the meshlet buffer via BDA
// ---------------------------------------------------------------------------

Meshlet loadMeshlet(uint64_t meshletAddr, uint index) {
    // 16 bytes = 4 uints per meshlet
    MeshletBuffer mbuf = MeshletBuffer(meshletAddr + uint64_t(index) * 16);
    Meshlet m;
    m.vertexOffset   = mbuf.vertexOffset;
    m.vertexCount    = mbuf.vertexCount;
    m.triangleOffset = mbuf.triangleOffset;
    m.triangleCount  = mbuf.triangleCount;
    return m;
}

// ---------------------------------------------------------------------------
// Helper: load meshlet bounds via BDA
// ---------------------------------------------------------------------------

MeshletBounds loadMeshletBounds(uint64_t boundsAddr, uint index) {
    MeshletBoundsBuffer bbuf = MeshletBoundsBuffer(boundsAddr + uint64_t(index) * 48);
    MeshletBounds b;
    b.center    = bbuf.center;
    b.radius    = bbuf.radius;
    b.coneApex  = bbuf.coneApex;
    b.coneCutoff = bbuf.coneCutoff;
    b.coneAxis  = bbuf.coneAxis;
    return b;
}

// ---------------------------------------------------------------------------
// Helper: load an instance via BDA
// ---------------------------------------------------------------------------

GPUInstance loadInstance(uint64_t instanceAddr, uint index) {
    InstanceBuffer ibuf = InstanceBuffer(instanceAddr + uint64_t(index) * 80);
    GPUInstance inst;
    inst.modelMatrix   = ibuf.modelMatrix;
    inst.meshIndex     = ibuf.meshIndex;
    inst.materialIndex = ibuf.materialIndex;
    inst.flags         = ibuf.flags;
    return inst;
}

// ---------------------------------------------------------------------------
// Helper: load material via BDA
// ---------------------------------------------------------------------------

GPUMaterial loadMaterial(uint64_t materialAddr, uint index) {
    MaterialBuffer mbuf = MaterialBuffer(materialAddr + uint64_t(index) * 80);
    GPUMaterial mat;
    mat.baseColor          = mbuf.baseColor;
    mat.metallic           = mbuf.metallic;
    mat.roughness          = mbuf.roughness;
    mat.normalScale        = mbuf.normalScale;
    mat.occlusionStrength  = mbuf.occlusionStrength;
    mat.baseColorTex       = mbuf.baseColorTex;
    mat.normalTex          = mbuf.normalTex;
    mat.metallicRoughnessTex = mbuf.metallicRoughnessTex;
    mat.occlusionTex       = mbuf.occlusionTex;
    mat.emissiveTex        = mbuf.emissiveTex;
    mat.emissive           = mbuf.emissive;
    mat.alphaCutoff        = mbuf.alphaCutoff;
    return mat;
}

// ---------------------------------------------------------------------------
// Helper: load light via BDA
// ---------------------------------------------------------------------------

GPULight loadLight(uint64_t lightAddr, uint index) {
    // GPULight is 17 floats + 2 uints = 76 bytes, but align to 80
    LightBuffer lbuf = LightBuffer(lightAddr + uint64_t(index) * 80);
    GPULight light;
    light.type           = lbuf.type;
    light.position       = lbuf.position;
    light.direction      = lbuf.direction;
    light.color          = lbuf.color;
    light.intensity      = lbuf.intensity;
    light.range          = lbuf.range;
    light.innerCone      = lbuf.innerCone;
    light.outerCone      = lbuf.outerCone;
    light.shadowMapIndex = lbuf.shadowMapIndex;
    return light;
}

#endif // TYPES_GLSL
