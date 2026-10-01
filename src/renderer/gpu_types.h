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

// GPUInstance::flags: bits 0-2 mirror MeshInstanceComponent (visible, casts
// shadows, static); the renderer adds the bits below.
// Mirrored: the model matrix has a negative determinant, which reverses the
// triangle winding on screen, so front_facing must be inverted.
PHOSPHOR_GPU_CONSTANT u32 INSTANCE_FLAG_MIRRORED = 1u << 3;

PHOSPHOR_GPU_CONSTANT u32 INVALID_TEXTURE_INDEX = 0xFFFFFFFFu;

// GPUMaterial::flags bits.
// DOUBLE_SIDED: glTF `doubleSided`; back-face culling is disabled for the
// material and back faces are lit with the flipped normal.
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FLAG_DOUBLE_SIDED = 1u << 0;

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
    u32 flags;                 // MATERIAL_FLAG_*
    float pad[2];
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

// Constants of the debug overlays (F4.7, shaders/overlay.metal); the scales
// and the palette are in diagnostics/overlay_math.h.
struct OverlayConstants {
    u32 kind;   // overlay::KIND_*
    u32 width;  // target size in pixels
    u32 height;
    u32 tilesX; // tile grid (tile-cost heatmap)
    u32 tilesY;
    float alpha;
    u32 pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(OverlayConstants) == 32, "OverlayConstants layout");

// Synthetic passes of the OPT-1 graph scenarios (shaders/scenario.metal,
// platform/metal/scenario_passes.cpp, value model in rendergraph/scenario.h).
PHOSPHOR_GPU_CONSTANT u32 SYNTH_MAX_INPUTS     = 8;
PHOSPHOR_GPU_CONSTANT u32 SYNTH_COLOR_INPUTS   = 6; // texture(0..5)
PHOSPHOR_GPU_CONSTANT u32 SYNTH_DEPTH_INPUTS   = 5; // texture(6..10)
PHOSPHOR_GPU_CONSTANT u32 SYNTH_STORAGE_OUTPUTS = 2; // texture(11..12)
PHOSPHOR_GPU_CONSTANT u32 SYNTH_INPUT_COLOR = 0;
PHOSPHOR_GPU_CONSTANT u32 SYNTH_INPUT_DEPTH = 1;
PHOSPHOR_GPU_CONSTANT u32 SYNTH_INPUT_REMAT = 2;
PHOSPHOR_GPU_CONSTANT u32 SYNTH_INPUT_SOURCE = 3; // depth read only as a remat source (not hashed)

struct GPUSynthInput {
    u32 width;      // texture size (remat: the depth's size)
    u32 height;
    u32 kind;       // SYNTH_INPUT_*
    u32 bind;       // index in the color or depth input array
    u32 channels;   // remat: channels of the rematerialised format (1..4)
    u32 rematDepth; // remat: input index of the depth
    u32 rematSeed;  // remat: producer's seed and slot
    u32 rematSlot;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUSynthInput) == 32, "GPUSynthInput layout");

struct GPUSynthArgs {
    u32 outWidth;         // raster target / dispatch size
    u32 outHeight;
    u32 seed;
    u32 iterations;       // ALU steps per pixel/thread
    u32 zero;             // runtime 0: keeps the ALU chains, changes no value
    u32 inputCount;
    u32 geometry;         // 1: Geometry pass (the fragment hashes its depth key)
    u32 gridW;
    u32 gridH;
    u32 geometrySeed;
    u32 vertexIterations; // ALU steps per vertex (Geometry)
    u32 depthOnlyMask;    // outputs (color slots / storage indices) that are signals
    u32 signalDepthInput; // non-Geometry signals: input giving the depth
    u32 storageCount;
    u32 pad[2];
    GPUSynthInput inputs[8]; // SYNTH_MAX_INPUTS
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUSynthArgs) == 320, "GPUSynthArgs layout");

} // namespace phosphor
