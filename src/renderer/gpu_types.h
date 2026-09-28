#pragma once

// ---------------------------------------------------------------------------
// GPU-visible data layouts shared by C++ and the Metal Shading Language.
//
// This header is compiled by both the host compiler and the Metal compiler
// (shaders include it through the `src` include path), so every struct has a
// single definition.  Only scalar members are used: MSL's float3 is 16-byte
// aligned while glm::vec3 is not, and scalars keep both sides identical.
// ---------------------------------------------------------------------------

#ifdef __METAL_VERSION__
#include <metal_stdlib>
namespace phosphor {
using u32 = uint;
} // namespace phosphor
#define PHOSPHOR_STATIC_ASSERT(cond, msg)
// Program-scope variables must live in the constant address space in MSL.
#define PHOSPHOR_GPU_CONSTANT constant
#else
#include "core/types.h"
#define PHOSPHOR_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#define PHOSPHOR_GPU_CONSTANT inline constexpr
#endif

namespace phosphor {

struct GPUVertex {
    float px, py, pz;     // position
    float nx, ny, nz;     // normal
    float tx, ty, tz, tw; // tangent + handedness
    float u, v;           // UV
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUVertex) == 48, "GPUVertex layout");

struct GPUInstance {
    float modelMatrix[16]; // column-major
    u32 meshIndex;
    u32 materialIndex;
    u32 flags;
    u32 pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUInstance) == 80, "GPUInstance layout");

PHOSPHOR_GPU_CONSTANT u32 INVALID_TEXTURE_INDEX = 0xFFFFFFFFu;

struct GPUMaterial {
    float baseColor[4];
    float metallic;
    float roughness;
    float normalScale;
    float occlusionStrength;
    u32 baseColorTex;          // bindless index or INVALID_TEXTURE_INDEX
    u32 normalTex;
    u32 metallicRoughnessTex;
    u32 occlusionTex;
    u32 emissiveTex;
    float emissive[3];
    float alphaCutoff;
    float pad[3];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterial) == 80, "GPUMaterial layout");

struct GPUMeshInfo {
    u32 meshletCount;
    u32 meshletOffset;     // into the global meshlet buffer
    u32 vertexOffset;      // into the global vertex buffer
    u32 indexOffset;       // into the global index buffer (mesh-local indices)
    u32 indexCount;
    u32 pad[3];
    float boundingSphere[4]; // center xyz + radius
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshInfo) == 48, "GPUMeshInfo layout");

PHOSPHOR_GPU_CONSTANT u32 LIGHT_DIRECTIONAL = 0;
PHOSPHOR_GPU_CONSTANT u32 LIGHT_POINT       = 1;
PHOSPHOR_GPU_CONSTANT u32 LIGHT_SPOT        = 2;

struct GPULight {
    u32 type;
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float range;
    float innerCone;
    float outerCone;
    u32 shadowMapIndex;
    u32 pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPULight) == 64, "GPULight layout");

// Per-frame constants, bound once per pass.
struct FrameConstants {
    float viewProjection[16];
    float view[16];
    float cameraPosition[4]; // w = time in seconds
    u32 lightCount;
    u32 debugMode;
    float exposure;
    u32 frameIndex;
};
PHOSPHOR_STATIC_ASSERT(sizeof(FrameConstants) == 160, "FrameConstants layout");

} // namespace phosphor
