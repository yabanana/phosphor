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
    u32 generation; // F8: globally unique entity incarnation within this world
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUInstance) == 80, "GPUInstance layout");

// GPUInstance::flags: bits 0-2 mirror MeshInstanceComponent (visible, casts
// shadows, static); the renderer adds the bits below.
// Mirrored: the model matrix has a negative determinant, which reverses the
// triangle winding on screen, so front_facing must be inverted.
PHOSPHOR_GPU_CONSTANT u32 INSTANCE_FLAG_MIRRORED = 1u << 3;
// F5: the slot holds a live instance (scene store slots of a bucket's slack
// have it clear and are never drawn or culled in).
PHOSPHOR_GPU_CONSTANT u32 INSTANCE_FLAG_VALID = 1u << 4;

PHOSPHOR_GPU_CONSTANT u32 INVALID_TEXTURE_INDEX = 0xFFFFFFFFu;

// GPUMaterial::flags bits.
// DOUBLE_SIDED: glTF `doubleSided`; back-face culling is disabled for the
// material and back faces are lit with the flipped normal.
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FLAG_DOUBLE_SIDED = 1u << 0;
/// F15.1 (ticket 22): the material carries per-slot KHR_texture_transform data: GPUMaterial::textureTransformIndex is the first of
/// the MATERIAL_TT_RECORDS records of its transform block in the extension buffer (GPUTextureTransformRecord).
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FLAG_TEXTURE_TRANSFORM = 1u << 1;
// UNLIT: KHR_materials_unlit, lowered onto the emission path (renderer/material_description.h lowerUnlit).  The emission is the
// displayed colour only: the surface is never a sampled emitter and GI rays hitting it read no emission.
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FLAG_UNLIT = 1u << 2;

// F15.1 (ticket 05): material FEATURE bits live in GPUMaterial::flags from bit
// MATERIAL_FEATURE_SHIFT: bit (8 + MaterialFeature value) is set while the
// material has that feature active (renderer/material_description.h keeps the
// enum; a static_assert there checks these constants).  The same bit positions
// without the shift form the scene feature union (SceneStore::featureUnion) and
// the value of the reserved function constant MATERIAL_FEATURES.
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_SHIFT        = 8;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_SPECULAR_IOR = 1u << 0;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_CLEARCOAT    = 1u << 1;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_SHEEN        = 1u << 2;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_ANISOTROPY   = 1u << 3;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_IRIDESCENCE  = 1u << 4;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_TRANSMISSION = 1u << 5;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_VOLUME       = 1u << 6;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_SUBSURFACE   = 1u << 7;
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURE_LAYERS       = 1u << 8;
/// Every feature bit: the value of the all-lobes reference pipeline.
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_FEATURES_ALL         = (1u << 9) - 1u;
/// F15.1 (ticket 22), HOST ONLY: marker or-ed into the feature union handed to FeaturePipelines::serve when a live material carries
/// per-slot texture transforms (SceneStore::pipelineFeatureUnion).  Never a MATERIAL_FEATURES constant value: serve() strips it and sets
/// the reserved constant TEXTURE_TRANSFORM instead.  Outside MATERIAL_FEATURES_ALL.
PHOSPHOR_GPU_CONSTANT u32 FEATURE_UNION_TEXTURE_TRANSFORM = 1u << 31;
/// GPUMaterial::extensionIndex of a material without an extension group.
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_EXTENSION_INVALID    = 0xFFFFFFFFu;
/// Records of the extension group of one extended material (fixed, so
/// toggling features never reallocates the group).
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_EXT_RECORDS          = 4;

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
    u32 flags;                 // MATERIAL_FLAG_* | feature bits << MATERIAL_FEATURE_SHIFT
    u32 extensionIndex;        // first of the MATERIAL_EXT_RECORDS extension records, or MATERIAL_EXTENSION_INVALID
    u32 textureTransformIndex; // F15.1 (ticket 22): first record of the transform block when MATERIAL_FLAG_TEXTURE_TRANSFORM is set, else 0
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterial) == 80, "GPUMaterial layout");

// ---------------------------------------------------------------------------
// F15.1 (ticket 05): material extension records (decision 07, ADR-0002).
//
// A separate GPU buffer of 80-byte records; an extended material owns a group
// of MATERIAL_EXT_RECORDS consecutive records starting at
// GPUMaterial::extensionIndex.  The layouts of EVERY F15 feature are frozen
// here: later lobe tickets only read fields.  Texture fields are bindless
// indices (INVALID_TEXTURE_INDEX = factor only).  Each record is one
// GPUDeltaRecord payload.
// ---------------------------------------------------------------------------

/// Record 0: KHR_materials_ior + KHR_materials_specular, clearcoat, sheen.
struct GPUMaterialExt0 {
    float ior;                 // 0 = the permanent special mode (Fresnel evaluates to 1)
    float specularFactor;
    float specularColor[3];
    u32   specularTex;         // A channel
    u32   specularColorTex;    // RGB (sRGB)
    float clearcoatFactor;
    float clearcoatRoughness;
    float clearcoatNormalScale;
    u32   clearcoatFactorTex;  // R
    u32   clearcoatRoughnessTex; // G
    u32   clearcoatNormalTex;
    float sheenColor[3];
    float sheenRoughness;
    u32   sheenColorTex;       // RGB (sRGB)
    u32   sheenRoughnessTex;   // A
    u32   pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterialExt0) == 80, "GPUMaterialExt0 layout");

/// Record 1: anisotropy, iridescence, transmission, volume.
struct GPUMaterialExt1 {
    float anisotropyStrength;
    float anisotropyRotation;  // radians, counter-clockwise from the tangent
    u32   anisotropyTex;       // RGB: direction rg, strength b
    float iridescenceFactor;
    float iridescenceIor;
    float iridescenceThicknessMin; // nm
    float iridescenceThicknessMax; // nm
    u32   iridescenceFactorTex;    // R
    u32   iridescenceThicknessTex; // G
    float transmissionFactor;
    u32   transmissionTex;     // R
    float volumeThickness;
    u32   volumeThicknessTex;  // G
    float attenuationColor[3];
    float attenuationDistance; // +inf: no attenuation
    u32   pad[3];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterialExt1) == 80, "GPUMaterialExt1 layout");

/// Record 2: subsurface (OpenPBR semantics, ADR-0005).  mode: 0 thin, 1 volumetric.
struct GPUMaterialExt2 {
    u32   mode;
    float weight;
    float color[3];
    float radius;              // metres
    float radiusScale[3];
    float anisotropy;
    u32   weightTex;           // A
    u32   colorTex;            // RGB (sRGB)
    u32   radiusScaleTex;      // RGB (sRGB)
    u32   pad[7];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterialExt2) == 80, "GPUMaterialExt2 layout");

/// Record 3: layer stack.  packedMaskChannels: 2 bits per layer (layer i at
/// bits 2i..2i+1; 0 R, 1 G, 2 B, 3 A).  heightContrast[i] in metres, or -1
/// when the layer has none (plain mask blend).  Unused layers have material
/// index MATERIAL_EXTENSION_INVALID.
struct GPUMaterialExt3 {
    u32   maskTex;
    u32   layerCount;          // 0 .. 4
    u32   layerMaterial[4];
    u32   packedMaskChannels;
    float heightContrast[4];
    float uvScale[4];          // F15.4 (ticket 23): per-layer texture tiling (uv * uvScale); 0 for a layer that does not exist
    u32   heightTex[4];        // F15.4: per-layer height texture (R): INVALID_TEXTURE_INDEX when the layer has none, 0 for a layer that does not exist
    u32   pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterialExt3) == 80, "GPUMaterialExt3 layout");

// ---------------------------------------------------------------------------
// F15.1 (ticket 22): per-slot KHR_texture_transform.  A material with non-uniform transforms (slots that differ, or a mix of
// transformed and plain slots) owns a TRANSFORM BLOCK of MATERIAL_TT_RECORDS consecutive records in the extension buffer (two
// adjacent extension groups, allocated and freed together by the SceneStore).  Slot s (TT_SLOT_*) is stored at record s / 3,
// transform s % 3.  Record 0 word `slotMask` has bit s set when slot s has a non-identity transform; the other slots sample the
// plain UV.  uv' = M uv + o with M = R S of the specification (m = {m00, m01, m10, m11}, row major).
// ---------------------------------------------------------------------------
PHOSPHOR_GPU_CONSTANT u32 MATERIAL_TT_RECORDS = 2u * MATERIAL_EXT_RECORDS;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_BASE_COLOR       = 0;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_METAL_ROUGH      = 1;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_NORMAL           = 2;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_OCCLUSION        = 3;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_EMISSIVE         = 4;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SPECULAR         = 5;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SPECULAR_COLOR   = 6;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_COAT             = 7;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_COAT_ROUGHNESS   = 8;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_COAT_NORMAL      = 9;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SHEEN_COLOR      = 10;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SHEEN_ROUGHNESS  = 11;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_ANISOTROPY       = 12;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_IRIDESCENCE      = 13;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_IRIDESCENCE_THICKNESS = 14;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_TRANSMISSION     = 15;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_VOLUME_THICKNESS = 16;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SUBSURFACE_WEIGHT = 17;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SUBSURFACE_COLOR = 18;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_SUBSURFACE_RADIUS = 19;
PHOSPHOR_GPU_CONSTANT u32 TT_SLOT_COUNT            = 20;

struct GPUTextureTransform {
    float m[4];                // m00, m01, m10, m11
    float o[2];                // offset
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUTextureTransform) == 24, "GPUTextureTransform layout");

/// One record of a transform block (three transforms and two spare words; `slotMask` is only meaningful in record 0).
struct GPUTextureTransformRecord {
    GPUTextureTransform t[3];
    u32 slotMask;
    u32 reserved;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUTextureTransformRecord) == 80, "GPUTextureTransformRecord layout");

/// One record of the extension buffer, untyped (the scatter payload).
struct GPUMaterialExtRecord {
    u32 w[20];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMaterialExtRecord) == 80, "GPUMaterialExtRecord layout");

// ---------------------------------------------------------------------------
// F15.1 (ticket 06): the per-pixel lobe guide (decision 07: "an additional
// guide buffer allocated and read only when the scene feature union is
// non-empty").  The producers of the DI surfaces (material lighting guides
// kernel, indexed overflow receivers) write one record per pixel with the
// RESOLVED feature parameters (textures sampled, layers blended, instance
// scale applied) of the pixel's material; DI, the specular transport, the
// composite and the denoiser read it (tickets 07/08 onwards).  Base
// parameters (albedo, metallic, roughness, shading normal) stay in
// GPUDISurface.  The layout of EVERY F15 feature is frozen here; packing and
// unpacking live in renderer/lobe_guide_pack.h.
//
// Rules: `features` holds the feature bits of the pixel's material that the
// producer resolved (0 = base material: every other word is 0); the words of
// a feature whose bit is clear are 0.  Half fields hold IEEE binary16 bits
// (low half = first named value); oct fields hold a unit vector in WORLD
// space, octahedral snorm16x2 (low half = x).
// ---------------------------------------------------------------------------
PHOSPHOR_GPU_CONSTANT u32 LOBE_GUIDE_FEATURE_MASK = 0xFFFFu;   // features: MATERIAL_FEATURE_* bits
PHOSPHOR_GPU_CONSTANT u32 LOBE_GUIDE_FLAG_SUBSURFACE_VOLUMETRIC = 1u << 16; // GPUMaterialExt2::mode == 1
// Features whose parameters the resolve (shadeSurface) resolves into LobeParams today; the producers
// write `material bits & kMaterialFeatures & this`.  Later lobe tickets extend it.
PHOSPHOR_GPU_CONSTANT u32 LOBE_GUIDE_RESOLVED_FEATURES = MATERIAL_FEATURE_SPECULAR_IOR | MATERIAL_FEATURE_CLEARCOAT | MATERIAL_FEATURE_SHEEN |
    MATERIAL_FEATURE_ANISOTROPY | MATERIAL_FEATURE_IRIDESCENCE | MATERIAL_FEATURE_SUBSURFACE | // SUBSURFACE: the thin mode (ticket 16)
    MATERIAL_FEATURE_TRANSMISSION | MATERIAL_FEATURE_VOLUME; // ticket 13

struct GPULobeGuide {
    u32   features;               // MATERIAL_FEATURE_* (bits 0..15) | LOBE_GUIDE_FLAG_* (bits 16..31)
    float ior;                    // KHR_materials_ior (0 = special mode); also the transmission/volume IOR
    u32   specularFactorCoat;     // half2: specularFactor, clearcoatFactor
    u32   specularColorRG;        // half2: specularColor r, g
    u32   specularColorBCoatRoughness; // half2: specularColor b, clearcoatRoughness
    u32   clearcoatNormal;        // oct: coat normal (the shading normal without a coat normal texture)
    u32   sheenColorRG;           // half2: sheenColor r, g
    u32   sheenColorBRoughness;   // half2: sheenColor b, sheenRoughness
    u32   anisotropyDirection;    // oct: anisotropy direction (tangent frame rotated by the rotation/texture), orthogonal to N
    u32   anisotropyIridescence;  // half2: anisotropyStrength, iridescenceFactor
    u32   iridescence;            // half2: iridescenceIor, iridescenceThickness (nm, resolved from min/max/texture)
    u32   transmissionThickness;  // half2: transmissionFactor, volume thickness in WORLD units (mesh thickness * instance scale)
    u32   attenuationRG;          // half2: attenuationColor r, g
    u32   attenuationBDistance;   // half2: attenuationColor b, attenuationDistance (world units, +inf = none)
    u32   subsurfaceWeightRadius; // half2: subsurface weight, radius (metres)
    u32   subsurfaceColorRG;      // half2: subsurface colour r, g
    u32   subsurfaceColorBAnisotropy; // half2: subsurface colour b, anisotropy
    u32   subsurfaceScaleRG;      // half2: radiusScale r, g
    u32   subsurfaceScaleB;       // half2: radiusScale b, 0
    u32   reserved;               // 0
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPULobeGuide) == 80, "GPULobeGuide layout");

// F15.1 (ticket 17): screen-space diffusion of volumetric subsurface (shaders/subsurface.metal).  Reverse-Z infinite projection: view depth
// z = nearPlane / depth; metres per pixel at depth z are 2 z tanHalfFovX / width (square pixels).
struct GPUSubsurfaceParams {
    u32   width, height;               // the rendered (logical) image
    u32   outputWidth, outputHeight;   // the attachments
    float nearPlane, tanHalfFovX, tanHalfFovY;
    u32   pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUSubsurfaceParams) == 32, "GPUSubsurfaceParams layout");

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

// F9: instance descriptors are exactly Metal's packed indirect layout. Resource
// IDs use two 32-bit words so the portable/shared header needs no API types.
PHOSPHOR_GPU_CONSTANT u32 RT_MASK_PRIMARY = 1u;
PHOSPHOR_GPU_CONSTANT u32 RT_MASK_SHADOW = 2u;
PHOSPHOR_GPU_CONSTANT u32 RT_MASK_INDIRECT = 4u;
PHOSPHOR_GPU_CONSTANT u32 RT_MASK_ALL = 7u;
PHOSPHOR_GPU_CONSTANT u32 RT_CORRUPT_NONE = 0u;
PHOSPHOR_GPU_CONSTANT u32 RT_CORRUPT_TRANSFORM = 1u;
PHOSPHOR_GPU_CONSTANT u32 RT_CORRUPT_MASK = 2u;
PHOSPHOR_GPU_CONSTANT u32 RT_CORRUPT_BLAS = 3u;
PHOSPHOR_GPU_CONSTANT u32 RT_PROBE_PRIMARY = 0u;
PHOSPHOR_GPU_CONSTANT u32 RT_PROBE_SHADOW = 1u;
PHOSPHOR_GPU_CONSTANT u32 RT_PROBE_AO = 2u;
PHOSPHOR_GPU_CONSTANT u32 RT_PROBE_DIFFUSE = 3u;
// F15.1 (ticket 14): a visibility ray that ACCUMULATES the RGB transmittance of transmissive casters (any-hit, the intersection function rejects
// the transmissive triangles after multiplying their factor into the payload) instead of stopping at them; opaque and alpha-masked casters
// stop it as RT_PROBE_SHADOW does.  Every other ray type sees a transmissive surface as an opaque hit.
PHOSPHOR_GPU_CONSTANT u32 RT_PROBE_SHADOW_TRANSMITTED = 4u;
// GPURtParams::transmittanceFlags.
PHOSPHOR_GPU_CONSTANT u32 RT_TRANSMITTANCE_NO_DEDUP = 1u << 0; // negative control: no per-primitive duplicate filter (--debug-transmittance-dedup off)
PHOSPHOR_GPU_CONSTANT u32 RT_TRANSMITTANCE_INJECT_DUPLICATE = 1u << 1; // spike control: every transmissive triangle invokes the accumulation twice (--debug-transmittance-inject)
PHOSPHOR_GPU_CONSTANT u32 RT_MESH_ALPHA = 1u;

struct GPURtInstanceDesc {
    float transform[12]; // column-major packed float4x3
    u32 options, mask, intersectionFunctionTableOffset, userID;
    u32 blasLo, blasHi;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtInstanceDesc) == 72, "GPURtInstanceDesc layout");

struct GPURtMesh {
    u32 blasLo, blasHi;
    u32 vertexOffset, indexOffset; // original GPUVertex stream, RT index stream
    u32 indexCount, flags, proxyLevel, pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtMesh) == 32, "GPURtMesh layout");

struct GPURtParams {
    u32 slotCount, meshCount, materialCount, corruption;
    u32 transmittanceFlags, pad0, pad1, pad2; // F15.1 (ticket 14): RT_TRANSMITTANCE_*
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtParams) == 32, "GPURtParams layout");

struct GPURtRay {
    float ox, oy, oz, tmin;
    float dx, dy, dz, tmax;
    u32 mask, type;
    float coneWidth; // angular footprint; zero selects LOD 0
    u32 pad; // source pixel index, preserved by secondary rays
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtRay) == 48, "GPURtRay layout");

struct GPURtHit {
    float t, u, v; // miss: t < 0, slot = ~0u
    u32 slot, primitive, generation, frontFacing, hit;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtHit) == 32, "GPURtHit layout");

struct GPURtProbeParams {
    float inverseViewProjection[16];
    float cameraPosition[4];
    float lightDirection[4]; // xyz direction toward light; w = secondary tmax
    u32 width, height, rayCount, probeType;
    u32 slotCount, meshCount, frameIndex, flags; // bit0 alpha counters, bits1..2 debug mode, bit3 sampled primary
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtProbeParams) == 128, "GPURtProbeParams layout");

struct GPURtCounters {
    u32 activeInstances, maskedInstances, invalidMesh, rays;
    u32 hits, alphaTests, opaqueAlphaTests, pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURtCounters) == 32, "GPURtCounters layout");

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
    // F15: GPU address of the DFG table buffer (renderer/dfg_table.h values,
    // owned by SceneRenderer). Fragment functions must read the table through
    // it: MTL_SHADER_VALIDATION corrupts their reads of program-scope constant
    // arrays (measured 2026-11-13, see shaders/materials/bsdf.h).
#if defined(__METAL_VERSION__)
    const constant float* dfgTable;
#else
    u64 dfgTable;
#endif
    // F15.1 (ticket 05): GPU address of the material extension buffer
    // (GPUMaterialExtRecord[]); read only by materials whose flags carry
    // feature bits, so it may stay 0 for scenes without features.
#if defined(__METAL_VERSION__)
    const device GPUMaterialExtRecord* materialExtensions;
#else
    u64 materialExtensions;
#endif
    // F15.1 (ticket 10): GPU address of the sheen albedo table buffer (renderer/sheen_table.h values, owned by SceneRenderer),
    // read by fragment functions for the same reason as dfgTable.
#if defined(__METAL_VERSION__)
    const constant float* sheenTable;
#else
    u64 sheenTable;
#endif
};
PHOSPHOR_STATIC_ASSERT(sizeof(FrameConstants) == 184, "FrameConstants layout");

// ---------------------------------------------------------------------------
// F5: persistent GPU scene (renderer/scene_store.h, renderer/gpu_scene_layout.h)
// ---------------------------------------------------------------------------

/// Deepest transform hierarchy level the GPU processes (roots = level 0).
PHOSPHOR_GPU_CONSTANT u32 SCENE_MAX_LEVELS = 8;
/// Speed classes of the procedural motion: the CPU evaluates sin/cos of each
/// class's angle once per frame, the GPU only multiplies and adds (bit-exact
/// CPU mirror, F5.2).
PHOSPHOR_GPU_CONSTANT u32 SCENE_MOTION_CLASSES = 64;
/// Cull classes (CullClass): Back, BackMirrored, None.  The ICB holds one
/// fixed command range per class, each ending with a sentinel command that
/// is always reset (F5.3, D2).
PHOSPHOR_GPU_CONSTANT u32 SCENE_CULL_CLASSES = 3;

/// One CPU->GPU delta: `slot` (element index in the destination buffer) and
/// up to 20 words of payload (GPUInstance, GPUMaterial, GPUTransformNode and
/// GPUMotion are all 80 bytes).  Applied by kernel scene_scatter (D1).
struct GPUDeltaRecord {
    u32 slot;
    u32 pad[3];
    u32 payload[20];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDeltaRecord) == 96, "GPUDeltaRecord layout");

/// Hierarchy node of a slot whose entity has a parent (F5.2): world =
/// world(parentSlot) * local, computed on the GPU with the product of
/// renderer/transform_math.h (fp contract off: bit-identical to glm).
struct GPUTransformNode {
    float local[16];  // column-major
    u32   parentSlot; // slot of the parent instance
    u32   depth;      // 1 .. SCENE_MAX_LEVELS - 1
    u32   pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUTransformNode) == 80, "GPUTransformNode layout");

/// Procedural motion of a root (F5.6): world = translate(orbit) * rotateY(a)
/// * base, with (sin a, cos a) of the slot's speed class for this frame and
/// orbit = centre + (r (cos a cosP - sin a sinP), h, r (sin a cosP + cos a sinP)).
struct GPUMotion {
    float centre[3];
    float radius;
    float cosPhase;
    float sinPhase;
    float height;
    u32   speedClass;  // < SCENE_MOTION_CLASSES
    float base[12];    // 3x4 column-major: three basis columns + translation
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMotion) == 80, "GPUMotion layout");

/// A draw bucket = (cull class, mesh): a contiguous slot region, one ICB
/// command.  Buckets are sorted by (class, mesh); the command index of a
/// bucket is its index plus the sentinels of the classes before it.
struct GPUDrawBucket {
    u32 firstSlot;
    u32 capacity;      // slots of the region (live + slack)
    u32 meshIndex;
    u32 cullClass;
    u32 indexCount;
    u32 indexOffset;   // into the global index buffer
    u32 vertexOffset;  // base vertex
    u32 command;       // ICB command index
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDrawBucket) == 32, "GPUDrawBucket layout");

// GPUCullParams::flags
PHOSPHOR_GPU_CONSTANT u32 CULL_FLAG_FRUSTUM  = 1u << 0;
PHOSPHOR_GPU_CONSTANT u32 CULL_FLAG_DISTANCE = 1u << 1;
PHOSPHOR_GPU_CONSTANT u32 CULL_FLAG_SIZE     = 1u << 2;

/// Instance culling (F5.5, renderer/cull_math.h): planes of the reverse-Z
/// infinite frustum (left, right, bottom, top, near; xyz normal, w distance,
/// normalised), camera, thresholds.
struct GPUCullParams {
    float planes[20];
    float cameraPosition[3];
    float maxDistance;     // CULL_FLAG_DISTANCE: culled if distance - radius > maxDistance
    float minPixels;       // CULL_FLAG_SIZE: culled if projected diameter < minPixels
    float pixelScale;      // projection[1][1] * viewportHeight / 2
    float nearPlane;
    u32   slotCount;       // slots to test (scene store capacity)
    u32   groupCount;      // threadgroups of scene_cull_flags (SCENE_CULL_GROUP threads each)
    u32   flags;           // CULL_FLAG_*
    u32   pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUCullParams) == 128, "GPUCullParams layout");

/// Counters written by the GPU passes of a frame and read back by the CPU
/// METAL_FRAMES_IN_FLIGHT frames later (report, panel).
struct GPUSceneCounters {
    u32 tested;          // valid slots tested
    u32 visible;
    u32 culledFrustum;
    u32 culledDistance;
    u32 culledSize;
    u32 drawCommands;    // non-empty ICB commands written
    u32 nodesUpdated;    // hierarchy nodes recomputed
    u32 queueOverflow;   // entries dropped by a full GPU queue (must stay 0)
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUSceneCounters) == 32, "GPUSceneCounters layout");

// ---------------------------------------------------------------------------
// F6: meshlets, mesh rendering and two-phase occlusion culling
// (renderer/meshlet_layout.h has the kernel/pass contract).
// ---------------------------------------------------------------------------

/// One meshlet (renderer/meshlet_builder.h): a range of the meshlet vertex
/// buffer (global vertex indices) and of the packed triangle buffer (3 bytes
/// per triangle, local vertex indices; triangleOffset is a BYTE offset).
struct GPUMeshlet {
    u32 vertexOffset;
    u32 vertexCount;
    u32 triangleOffset;
    u32 triangleCount;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshlet) == 16, "GPUMeshlet layout");

/// Bounds of a meshlet in mesh space (meshoptimizer v1.3 semantics, see
/// renderer/meshlet_cull_reference.h): bounding sphere, and the normal cone
/// for back-facing rejection -- all triangles face away from a camera at
/// `cam` when dot(normalize(coneApex - cam), coneAxis) >= coneCutoff.
/// coneCutoff >= 1 or a zero axis means "no usable cone" (never rejected).
struct GPUMeshletBounds {
    float center[3];
    float radius;
    float coneApex[3];
    float coneCutoff;
    float coneAxis[3];
    float pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshletBounds) == 48, "GPUMeshletBounds layout");

/// One meshlet of one instance.  `meshlet` is the GLOBAL meshlet index
/// (GPUMeshInfo::meshletOffset + local index), so candidates of different
/// instances never collide.  No generation is stored: the two-phase history
/// is a depth pyramid, never per-meshlet visibility, so a recycled slot
/// inherits nothing from the entity that used it before.
struct GPUMeshletCandidate {
    u32 slot;
    u32 meshlet;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshletCandidate) == 8, "GPUMeshletCandidate layout");

/// Candidates of one indirect mesh draw (one cull class of one phase):
/// list[first .. first + count) of the phase's list (A: candidates, B: the
/// history-rejected list).  Written by the GPU (meshlet_cand_scan /
/// meshlet_b_scan), read by the object shader.
struct GPUMeshletDrawRange {
    u32 first;
    u32 count;
    u32 cullClass; // CullClass
    u32 phase;     // MESHLET_PHASE_*
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshletDrawRange) == 16, "GPUMeshletDrawRange layout");

PHOSPHOR_GPU_CONSTANT u32 MESHLET_PHASE_A = 0;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_PHASE_B = 1;

// GPUMeshletCullParams::flags
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_FRUSTUM       = 1u << 0; // sphere vs the 5 planes
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_CONE = 1u << 1; // normal cone in mesh space (invertible affine transforms)
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_OCCLUSION     = 1u << 2; // two-phase Hi-Z test
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_HISTORY_VALID = 1u << 3; // phase A may use the history pyramid
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_DEBUG_ALL     = 1u << 4; // debug view: emit every candidate, coloured by decision
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_SIZE          = 1u << 5; // APPROXIMATE: projected bound smaller than minPixels
PHOSPHOR_GPU_CONSTANT u32 MESHLET_CULL_RECORD        = 1u << 6; // self-check frame: record every decision (MR_DECISIONS)

// Decision of the culling for one candidate (counters, debug views).
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_DRAWN_A   = 0; // drawn in phase A
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_DRAWN_B   = 1; // history-rejected, recovered in phase B
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_FRUSTUM   = 2;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_CONE      = 3;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_OCCLUDED  = 4; // rejected by the current pyramid in phase B
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_HISTORY   = 5; // phase A only: deferred to phase B
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DECISION_SIZE      = 6; // approximate size cull (never in the exact preset)

/// Per-frame parameters of the meshlet passes (candidates, object shaders).
/// Planes as GPUCullParams (left, right, bottom, top, near of the reverse-Z
/// infinite frustum).  The Hi-Z pyramid has power-of-two level-0 dimensions
/// >= ceil(viewport / 2): level-0 texel (x, y) covers the pixels
/// [2x, 2x + 2) x [2y, 2y + 2) (pixels outside the viewport count as depth
/// 1, the neutral value of the min), level L texel covers 2^(L+1) pixels per
/// side.  Reverse-Z: every texel holds the MINIMUM (farthest) depth.
struct GPUMeshletCullParams {
    float viewProj[16];      // current frame
    float prevViewProj[16];  // view the history pyramid was rendered with
    float planes[20];
    float cameraPosition[3];
    float nearPlane;
    float viewport[2];       // drawable size in pixels
    u32   hizSize[2];        // level-0 size of the pyramids (texels)
    u32   hizLevels;
    u32   flags;             // MESHLET_CULL_*
    u32   slotCount;         // scene store slots (candidate kernels)
    u32   groupCount;        // ceil(slotCount / MESHLET_SCAN_GROUP)
    u32   candidateCapacity; // entries of the candidate / B lists
    u32   candidateGroups;   // ceil(candidateCapacity / MESHLET_SCAN_GROUP)
    float minPixels;         // MESHLET_CULL_SIZE (approximate; off in the exact preset)
    u32   corruptId;         // self-check negative control: 1 = candidate 0 gets a wrong (in-range) id
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshletCullParams) == 272, "GPUMeshletCullParams layout");

/// Counters of the meshlet passes of a frame (word indices MESHLET_COUNTER_*
/// in renderer/meshlet_layout.h), read back METAL_FRAMES_IN_FLIGHT frames
/// later.  Identities (checked by the self-check): drawnA + frustum + cone +
/// historyRejected == candidates (phase A tests every candidate); testedB ==
/// historyRejected; drawnB + occludedB == testedB.
struct GPUMeshletCounters {
    u32 candidates;       // candidate meshlets this frame (all classes)
    u32 overflow;         // 1: the candidates exceeded the capacity (indexed fallback drew the frame)
    u32 drawnA;
    u32 frustum;
    u32 cone;
    u32 historyRejected;  // phase A: occluded by the history pyramid -> phase B
    u32 testedB;
    u32 drawnB;           // recovered by phase B
    u32 occludedB;        // final rejects of phase B
    u32 primitivesA;      // triangles emitted by the mesh shaders
    u32 primitivesB;
    u32 objectGroupsA;    // object threadgroups that ran
    u32 objectGroupsB;
    u32 emitted;          // triangles the mesh shaders kept after per-triangle facing culling (both phases)
    u32 sizeCulled;       // MESHLET_CULL_SIZE rejections (approximate; 0 in the exact preset)
    u32 pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUMeshletCounters) == 64, "GPUMeshletCounters layout");

/// One Hi-Z reduction step (shaders/hiz.metal): level `dstLevel` from level
/// dstLevel - 1 (or from the depth texture when dstLevel == 0).
struct GPUHiZParams {
    u32 srcSize[2];   // depth (dstLevel 0) or source level size
    u32 dstSize[2];
    u32 dstLevel;
    u32 levels;       // levels of the pyramid
    u32 pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUHiZParams) == 32, "GPUHiZParams layout");

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

// ---------------------------------------------------------------------------
// F10: visibility is one signal per selected light and view; 1 = unoccluded.
// The guide comes from raster geometry BEFORE material resolve. No normal map
// or RT proxy primitive may supply the geometric normal used for ray offsets.
// ---------------------------------------------------------------------------
PHOSPHOR_GPU_CONSTANT u32 SHADOW_CASCADES = 4u;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_CSM = 1u << 0;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_RT = 1u << 1;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_HISTORY_VALID = 1u << 2;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_CONTACT = 1u << 3;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_LIGHT_VALID = 1u << 4;
// F15.1 (ticket 15): the scene has transmissive casters and the CSM path keeps them out of the opaque depth maps: the cascades carry a
// single-layer transmittance map (nearest transmissive depth + summed ln RGB transmittance) the receiver multiplies in.
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_TRANSMITTANCE_MAP = 1u << 5;
// The receiver ignores the map (negative control `--debug-transmittance-map off`): transmissive casters then cast nothing.
PHOSPHOR_GPU_CONSTANT u32 SHADOW_FLAG_TRANSMITTANCE_IGNORED = 1u << 6;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_CORRUPT_NONE = 0u;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_CORRUPT_BIAS = 1u;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_CORRUPT_CASTER = 2u;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_CORRUPT_HISTORY = 3u;
PHOSPHOR_GPU_CONSTANT u32 SHADOW_CORRUPT_CACHE = 4u;

struct GPUShadowCascade {
    float viewProjection[16]; // orthographic reverse-Z; clear 0, compare Greater
    float splitNear, splitFar, texelWorld, depthRange;
    float center[3], radius; // snapped receiver bound in world space
    float depthMin, depthMax, biasWorld, normalBiasWorld;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowCascade) == 112, "GPUShadowCascade layout");

struct GPUShadowSurface {
    float position[3], viewDepth;
    float geometricNormal[3], reverseDepth;
    u32 slot, generation, primitive, valid; // primitive = meshlet-local triangle
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowSurface) == 48, "GPUShadowSurface layout");

struct GPUShadowHistory {
    float position[3], visibility;        // visibility = the luminance of the RGB transmittance (the variance estimator's scalar)
    float geometricNormal[3], secondMoment;
    u32 slot, generation, lightID, lightRevision;
    u32 sceneRevision, viewID, samples, valid;
    float transmittance[3], pad;          // F15.1 (ticket 14): the temporal mean of the RGB transmittance (= visibility on each channel without glass)
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowHistory) == 80, "GPUShadowHistory layout");
// F15 ticket 24: the history of a scene without transmissive materials: the 64-byte record of the pre-ticket-14 layout (the first 64 bytes of
// GPUShadowHistory).  The 80-byte stride cost `shadow_temporal` 0.06 ms on Sponza D2 (docs/opt-log.md): `shadow_temporal_scalar` reads and
// writes this layout in the same buffers (allocated for GPUShadowHistory), `shadow_temporal` keeps the RGB record of a scene with glass.
struct GPUShadowHistoryScalar {
    float position[3], visibility;
    float geometricNormal[3], secondMoment;
    u32 slot, generation, lightID, lightRevision;
    u32 sceneRevision, viewID, samples, valid;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowHistoryScalar) == 64, "GPUShadowHistoryScalar layout");

struct GPUShadowParams {
    float inverseViewProjection[16], viewProjection[16];
    float previousViewProjection[16], view[16];
    float cameraPosition[4];
    float lightDirection[4]; // xyz toward sun, w = angular radius in RADIANS
    u32 width, height, slotCount, meshCount;
    u32 candidateCapacity, materialCount, frameIndex, flags;
    u32 mapResolution, pcssSearchSamples, pcssFilterSamples, historyMaxSamples;
    u32 viewID, lightID, lightRevision, sceneRevision;
    float temporalNormalThreshold, temporalPositionThreshold, contactDistance, contactThickness;
    float depthBiasWorld, normalBiasWorld, contactStrength, maxTraceDistance;
    u32 contactSteps, cascadeIndex, casterSlot, meshletFirst;
    u32 meshletCount, meshletGridWidth, corruption, pad;
    GPUShadowCascade cascades[4];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowParams) == 864, "GPUShadowParams layout");

struct GPUShadowCounters {
    u32 guides, rays, hits, historyReused;
    u32 historyRejected, casters, contactHits, errors;
    // F15.1 (ticket 14): transmittance rays that crossed at least one transmissive triangle, intersection-function invocations on
    // transmissive triangles, of which distinct (slot, primitive) pairs, and rays whose hit list overflowed.
    u32 transmittedRays, ifInvocations, ifDistinct, listOverflows;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowCounters) == 48, "GPUShadowCounters layout");

// F10.5 cache region controls. Masks are 64 bits split into scalar u32 words;
// bit y*8+x identifies a tile in the 8x8 map. Revisions are FULL 64-bit values,
// split into words; shaders compare the tuple, never an age or hash alone.
struct GPUShadowCacheParams {
    u32 resolution, cascade, casterClass, slotCount; // class 0 static, 1 dynamic, 2 all
    u32 currentLo, currentHi, updateLo, updateHi;
    u32 readyLo, readyHi, corruption, pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowCacheParams) == 48, "GPUShadowCacheParams layout");
struct GPUShadowCacheTile {
    u32 lightLo, lightHi, casterLo, casterHi;
    u32 materialLo, materialHi, projectionLo, projectionHi;
    u32 valid, pad[3];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUShadowCacheTile) == 48, "GPUShadowCacheTile layout");


// F13 residual surface base omits external DI/GI and hemisphere diffuse.
// A positive raw assembly supplies full pre-reflection Lo for SSR separately.
PHOSPHOR_GPU_CONSTANT u32 RESOLVE_EXTERNAL_DIFFUSE = 1u<<5;
struct GPUResolveLightingParams {
    u32 flags, sunIndex, width, height; // bits: sun visibility, replace local DI, indirect irradiance
    float environment[3];               // declared isotropic environment radiance (bit16 = F13 path)
    float sunAngularRadius;             // disk of the shadow-casting directional light (F13 path)
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUResolveLightingParams) == 32, "GPUResolveLightingParams layout");
// F11: local-light sampling. Area emission is radiance (W/(m^2 sr));
// point/spot emission is intensity (W/sr). Distances are world-space metres.
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_RECTANGLE = 3u;
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_DISK = 4u;
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_TUBE = 5u;
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_TRIANGLE = 6u;
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_TWO_SIDED = 1u;
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_TEXTURED_EMISSION = 2u;
PHOSPHOR_GPU_CONSTANT u32 DI_LIGHT_MIRRORED = 4u;
PHOSPHOR_GPU_CONSTANT u32 DI_RESET_HISTORY = 1u;
PHOSPHOR_GPU_CONSTANT u32 DI_ENABLE_TEMPORAL = 2u;
PHOSPHOR_GPU_CONSTANT u32 DI_ENABLE_SPATIAL = 4u;
PHOSPHOR_GPU_CONSTANT u32 DI_ENABLE_VISIBILITY = 8u;
PHOSPHOR_GPU_CONSTANT u32 DI_USE_STBN = 16u;
// F15.1 (ticket 06): the lobe guide buffer exists and its records for this frame were written by a
// specialised producer (not by a base-pipeline fallback while compiling); consumers read it only
// then.  DI_LOBE_GUIDE_CORRUPT is the negative control of --debug-lobe-guide (+0.25 on ior, on the sheen roughness of sheen pixels and on the weight of thin subsurface pixels).
PHOSPHOR_GPU_CONSTANT u32 DI_LOBE_GUIDE_VALID = 64u;
PHOSPHOR_GPU_CONSTANT u32 DI_LOBE_GUIDE_CORRUPT = 128u;
// The lobe guide buffer is allocated and bound (di_receiver_pack clears the record of surfaces it invalidates).
PHOSPHOR_GPU_CONSTANT u32 DI_LOBE_GUIDE_BOUND = 256u;
// F15 closure (ticket 06): --debug-lobe-guide-delay makes the guide-writing overflow receivers spend a bounded, data-dependent time before
// the write (up to 16384 dependent fma per fragment, the longest for the first layer of the coplanar fixture), so that overlapping fragments
// finish out of order: the raster order group must still order the writes, its absence must not.
PHOSPHOR_GPU_CONSTANT u32 DI_LOBE_GUIDE_DELAY = 4096u;
// F13.6: RT reflections also see geometry-backed emitters (DI_LIGHT_TRIANGLE).
// Their specular lobe is then split by the reflection RT probability: DI keeps
// smoothstep(low, high, roughness), the RT path (taken with the complementary
// probability) adds the emission it hits. Analytic and punctual lights keep
// their full DI specular: no reflection ray can hit them.
PHOSPHOR_GPU_CONSTANT u32 DI_REFLECTED_EMITTER_SPECULAR = 32u;
PHOSPHOR_GPU_CONSTANT u32 DI_ERROR_WEIGHT = 1u;
PHOSPHOR_GPU_CONSTANT u32 DI_ERROR_ALIAS = 2u;
PHOSPHOR_GPU_CONSTANT u32 DI_ERROR_TARGET = 4u;
PHOSPHOR_GPU_CONSTANT u32 DI_PROPOSAL_VALID = 1u; // GPUDIReservoir.pad[1], distinct from selected valid

struct GPUSampledLight {
    u32 id, generation, type, flags;
    float position[3], range;
    float axisU[3], radius;
    float axisV[3], innerCone;
    float emission[3], outerCone;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUSampledLight) == 80, "GPUSampledLight layout");

// Indexed by the sampled-light slot; valid=0 for analytic/constant emitters.
// Local triangle geometry is transformed by light_emissive_update AFTER scene
// transforms, including GPU animation; never CPU guesses for moving emitters.
struct GPUEmissiveSurface {
    float p0[3]; u32 instanceSlot;
    float p1[3]; u32 instanceGeneration;
    float p2[3]; u32 materialIndex;
    float uv0[2], uv1[2];
    float uv2[2]; u32 valid, pad;
    u32 vertex0, vertex1, vertex2, geometryValid; // live raster vertex lookup for deformation
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUEmissiveSurface) == 96, "GPUEmissiveSurface layout");
struct GPUEmissiveUpdateParams {
    u32 lightCount, slotCount, materialCount, pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUEmissiveUpdateParams) == 16, "GPUEmissiveUpdateParams layout");

struct GPUAliasEntry {
    float probability; // Vose threshold for this uniform column
    float selectionPdf; // probability of choosing THIS entry's light
    u32 alias, lightIndex;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUAliasEntry) == 16, "GPUAliasEntry layout");

struct GPUDIReservoir {
    u32 lightIndex, lightID, lightGeneration, lightRevision;
    float u, v, target, weightSum;
    float normalization; // W = sumWeight / (M * target(selected))
    u32 M, age, valid;
    u32 viewID, historyEpoch, pad[2]; // pad[0]=DI_ERROR_*; pad[1]=counted proposal-domain validity
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDIReservoir) == 64, "GPUDIReservoir layout");

struct GPUDISurface {
    float position[3], depth; // world position; positive view distance
    float geometricNormal[3], roughness; // WORLD geometric normal for ray offset
    float shadingNormal[3], metallic; // material/normal-map result, matches resolve
    float albedo[3];
    u32 materialRevision;
    float viewDirection[3];
    u32 instanceSlot;
    // pad[0]: material occlusion (float bits); pad[1]: resolve emission
    // packed by renderer/emission_pack.h (F13.1), 0 where no guide carries it.
    u32 instanceGeneration, valid, pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDISurface) == 96, "GPUDISurface layout");

struct GPUDIParams {
    u32 width, height, lightCount, frameIndex;
    u32 candidateCount, spatialCount, spatialRadius, maxHistoryM;
    u32 flags, viewID, historyEpoch, lightRevision;
    u32 maxHistoryAge, stbnWidth, stbnHeight, stbnFrames;
    float depthRelativeThreshold, normalThreshold, targetFloor, pad0;
    u32 stbnDimensions, stbnSeed, slotCount, pad1; // pad1!=0 forces full-light brute iteration in cluster shade
    float emitterSpecularLow, emitterSpecularHigh; u32 pad2, pad3; // = ReflectionSettings rtRoughnessLow/High
    // F15.1 (ticket 22): GPU address of the material extension buffer: the per-slot texture transforms of the secondary-site texture lookups
    // (emitter / hit / capture surfaces); 0 when the scene has none.  Set by the host that owns the params.
#if defined(__METAL_VERSION__)
    const device GPUMaterialExtRecord* materialExtensions;
#else
    u64 materialExtensions;
#endif
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDIParams) == 120, "GPUDIParams layout");

struct GPULightCluster {
    u32 count, overflow, pad[2]; // overflow selects full local-light iteration
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPULightCluster) == 16, "GPULightCluster layout");

struct GPULightClusterParams {
    float view[16];
    float nearPlane, farPlane, tanHalfFovX, tanHalfFovY;
    u32 gridX, gridY, gridZ, capacity;
    u32 lightCount, pad[3];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPULightClusterParams) == 112, "GPULightClusterParams layout");

// F12: experimental GI ABI. Linear RGB, metres, world geometric normals.
// generation is the geometric probe-state epoch; cacheGeneration is the
// radiometric/history epoch. A radiance reset must not erase relocated geometry.
PHOSPHOR_GPU_CONSTANT u32 GI_MODE_DDGI = 1u;
PHOSPHOR_GPU_CONSTANT u32 GI_MODE_CACHE = 2u;
PHOSPHOR_GPU_CONSTANT u32 GI_MODE_RESTIR = 3u;
PHOSPHOR_GPU_CONSTANT u32 GI_CORRUPT_NONE = 0u;
PHOSPHOR_GPU_CONSTANT u32 GI_CORRUPT_CACHE = 1u;
PHOSPHOR_GPU_CONSTANT u32 GI_CORRUPT_PROBE = 2u;
PHOSPHOR_GPU_CONSTANT u32 GI_CORRUPT_PDF = 3u;
PHOSPHOR_GPU_CONSTANT u32 GI_PROBE_ACTIVE = 1u;
PHOSPHOR_GPU_CONSTANT u32 GI_PROBE_INACTIVE = 2u;
PHOSPHOR_GPU_CONSTANT u32 GI_SAMPLE_VALID = 1u;
// A receiver/proposal can be valid while every sampled contribution is zero.
// Keep that proposal count/history distinct from a positive selected endpoint.
PHOSPHOR_GPU_CONSTANT u32 GI_PROPOSAL_VALID = 2u;

// Physical ablation for independent leakage validation, never a product preset.
PHOSPHOR_GPU_CONSTANT u32 GI_DEBUG_NO_VISIBILITY = 1u;

struct GPUProbeGridParams {
    float origin[3], maxDistance;
    float spacing[3], hysteresis;
    u32 countX, countY, countZ, raysPerProbe;
    u32 irradianceTexels, distanceTexels, frameIndex, generation;
    float skyRadiance[3], normalBias;
    u32 slotCount, meshCount, materialCount, lightCount;
    float backfaceThreshold, minFrontDistance, relocationStep, maxRelocation;
    u32 width, height, mode, reset;
    u32 cacheCapacity, cacheProbeLimit, cacheMaxAge, cacheGeneration;
    u32 geometryRevision, lightRevision, materialRevision, viewRevision;
    u32 debugFlags, debugPadding[3];
    // F15.1 (ticket 22): GPU address of the material extension buffer: the per-slot texture transforms of the secondary-site texture lookups
    // (emitter / hit / capture surfaces); 0 when the scene has none.  Set by the host that owns the params.
#if defined(__METAL_VERSION__)
    const device GPUMaterialExtRecord* materialExtensions;
#else
    u64 materialExtensions;
#endif
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUProbeGridParams) == 184, "GPUProbeGridParams layout");

struct GPUProbeState {
    float offset[3]; u32 state;
    float relocationTravel; u32 generation, age, pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUProbeState) == 32, "GPUProbeState layout");
struct GPUProbeRay {
    float direction[3], distance; // negative distance = backface, miss = maxDistance
    float radiance[3]; u32 backface;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUProbeRay) == 32, "GPUProbeRay layout");

// Full key comparison after hashing; RADIANCE along directionBin, not irradiance.
// Single-writer resolve pass owns each table lane (no races/partially published RGB).
struct GPURadianceCacheEntry {
    u32 cellX, cellY, cellZ, normalBin;
    u32 directionBin, generation, lastFrame, samples;
    float radiance[3]; u32 state;
    u32 geometryRevision, lightRevision, materialRevision, pad;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPURadianceCacheEntry) == 64, "GPURadianceCacheEntry layout");

// F12 diffuse reconnection reservoir uses SECONDARY AREA measure. Fresnel/specular
// paths are excluded explicitly. sourceProposalArea records the selected path's
// original area density; it is diagnostic and is not substituted for W on reuse.
struct GPUGiReservoir {
    float position[3], proposalArea;
    float normal[3], proposalSolidAngle;
    float radiance[3], target;
    float sourcePosition[3], sourceProposalArea;
    float sourceNormal[3], W;
    float weightSum; u32 M, age, flags;
    u32 slot, instanceGeneration, geometryRevision, lightRevision;
    u32 materialRevision, viewRevision, pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUGiReservoir) == 128, "GPUGiReservoir layout");

struct GPUProbeTraceExtra {
    u32 sampledLightCount, cacheUpdateCount, cacheCandidateCount, frameSeed;
    float sunAngularRadius; u32 pad[3];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUProbeTraceExtra) == 32, "GPUProbeTraceExtra layout");

struct GPULightingCheckParams {
    u32 width, height, restir, lightCount;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPULightingCheckParams) == 16, "GPULightingCheckParams layout");

struct GPUGiCheckParams {
    u32 width, height, mode, corruption;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUGiCheckParams) == 16, "GPUGiCheckParams layout");

// F13 source-only signal ABI. All radiance is linear; distances are metres.
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_ENABLE_RT = 1u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_ENABLE_SSR = 2u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_ENABLE_CACHE = 4u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_ENABLE_PROBES = 8u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_ENABLE_GI = 16u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_CAPTURE_HALF = 32u; // reject total radiance outside HALF before conversion
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_SINGLE_SAMPLE = 64u; // one sample: the kernel stores the reduced record, signal and distance
// F13.1 single count of geometry emitters. EXCLUDE: non-legacy DI integrates
// emitters (the SSR source carries none). BAND: DI owns only
// smoothstep(rtRoughnessLow, rtRoughnessHigh, roughness) of their specular
// lobe and the reflection paths own the rest; without BAND, DI owns all of it.
// reflectionFallbackEmissionShare() is the share a non-RT path adds back.
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_EXCLUDE_DI_EMISSION = 128u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_DI_EMITTER_BAND = 512u;
// F15.1 (ticket 08): the lobe guide (buffer 20 of the "_lobe" reflection kernels) is valid this frame.
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_LOBE_GUIDE_VALID = 1024u;
// F15.1 (ticket 13, VERIFICATION ONLY, --debug-transmission-both-branches): a transmissive pixel evaluates BOTH branches of the Fresnel one-sample
// estimator at the camera-visible interface (each with its own unweighted contribution) instead of drawing one: the same expectation, no branch noise.
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_DEBUG_BOTH_BRANCHES = 2048u;
// F15.1 (ticket 22, C22-A1): bits 12-13 of the reflection flags carry the number of EXTRA specular bounces a reflection ray continues through when its
// secondary hit is mirror-like (0 = the single-bounce path of F13, the negative control; 3 = the engine default).
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_BOUNCE_SHIFT = 12u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_BOUNCE_MASK = 3u;
// F13.5 negative control: probe lookups ignore the box projection.
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_DEBUG_NO_PARALLAX = 256u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_PATH_NONE = 0u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_PATH_RT = 1u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_PATH_SSR = 2u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_PATH_CACHE = 3u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_PATH_PROBE = 4u;
PHOSPHOR_GPU_CONSTANT u32 REFLECTION_PATH_ENVIRONMENT = 5u;
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_VALID = 1u;
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_PREFILTERED = 2u; // deterministic split-sum probe path, no MC PDF
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_STOCHASTIC = 8u; // endpoint/path is a Monte Carlo outcome, not receiver identity
PHOSPHOR_GPU_CONSTANT u32 DENOISE_SIGNAL_DI = 0u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_SIGNAL_GI = 1u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_SIGNAL_SPECULAR = 2u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_SIGNAL_AO = 3u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_RESET = 1u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_CLAMP_HISTORY = 2u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_CURRENT_VIEW_DEPTH = 8u;
// F13.6: the scene changed continuously this frame (geometry or light motion)
// while the signal's hard content key did not. History survives per pixel
// unless it leaves the current neighbourhood (clampSigma * sigma, plus a 2%
// relative margin); kept histories adapt with at least twice temporalAlpha.
PHOSPHOR_GPU_CONSTANT u32 DENOISE_SOFT_CHANGE = 16u;
// F13.6: neither the content nor this view's camera changed since the previous
// frame. A soft signal's history then accumulates as a running mean (alpha =
// 1/length, up to maxHistory) instead of the temporalAlpha floor.
PHOSPHOR_GPU_CONSTANT u32 DENOISE_STATIONARY = 32u;
PHOSPHOR_GPU_CONSTANT u32 DENOISE_HISTORY_STOCHASTIC = 2u; // history flags: bit0 remains a numerical error
PHOSPHOR_GPU_CONSTANT u32 DENOISE_DIAGNOSTIC_ENDPOINTS = 4u; // opt-in old endpoint rejection for the quality negative
// F15.1 (ticket 08): the "_lobe" specular denoiser kernels run with a valid lobe guide (buffer 9 / 4).
PHOSPHOR_GPU_CONSTANT u32 DENOISE_LOBE_GUIDE_VALID = 64u;
// Diagnostics of the lobe contract (PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE=stripes|flip, lobe kernels only): the sampled
// lobe id the denoiser sees is forced to 1 on vertical stripes 16 px wide (static: neighbours across a stripe edge
// are rejected), or on the stripes of alternating frames (the history key changes every frame: history rejected).
PHOSPHOR_GPU_CONSTANT u32 DENOISE_DIAGNOSTIC_LOBE_STRIPES = 128u, DENOISE_DIAGNOSTIC_LOBE_FLIP = 256u;
// F15.1 (ticket 13) refract-axis negative (PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE=reflect-axis): the lobe kernels treat a refract-axis sample as a
// reflect-axis one, as a defective implementation without the refract rule would: the key loses its axis bit and the spatial axis of the
// edge-stopping weights is the mirror reflection of the view about the normal instead of the refracted ray.  The reprojection (motion
// vectors) never depends on the axis; the axis acts through the history/neighbour key and the cosine of the axes in the a-trous weights.
PHOSPHOR_GPU_CONSTANT u32 DENOISE_DIAGNOSTIC_LOBE_REFLECT_AXIS = 512u;

struct GPUReflectionParams {
    float viewProjection[16], inverseViewProjection[16], view[16];
    float cameraPosition[4];
    float maxDistance, ssrThickness, rtRoughnessLow, rtRoughnessHigh;
    u32 width, height, frameIndex, flags;
    u32 slotCount, meshCount, materialCount, lightCount;
    u32 ssrSteps, ssrBinarySteps, probeCount, seed;
    float environment[3], maxProbeMip;
    float regularizationScale; u32 ssrRoughSteps, boostSamples; float boostRoughness; // k: alpha' = max(alpha_secondary, k*alpha_camera); SSR steps at roughness 1; samples after a reset below boostRoughness
    // F15.1 (ticket 22): GPU address of the material extension buffer: the per-slot texture transforms of the secondary-site texture lookups
    // (emitter / hit / capture surfaces); 0 when the scene has none.  Set by the host that owns the params.
#if defined(__METAL_VERSION__)
    const device GPUMaterialExtRecord* materialExtensions;
#else
    u64 materialExtensions;
#endif
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUReflectionParams) == 312, "GPUReflectionParams layout");
struct GPUSpecularSample {
    float radiance[3], hitDistance; // weighted outgoing Lo; 0 distance for environment/miss
    float direction[3], proposalSolidAngle;
    u32 path, secondarySlot, secondaryGeneration, flags;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUSpecularSample) == 48, "GPUSpecularSample layout");
struct GPUAOParams {
    float viewProjection[16], inverseViewProjection[16];
    float radius, originBias, thickness, pixelScale;
    u32 width, height, samples, slices;
    u32 steps, frameIndex, flags, slotCount;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUAOParams) == 176, "GPUAOParams layout");
struct GPUReflectionProbe {
    float boxMin[3], blendDistance;
    float boxMax[3], mipCount;
    float capturePosition[3]; u32 generation;
    u32 cubeIndex, enabled, pad[2];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUReflectionProbe) == 64, "GPUReflectionProbe layout");
struct GPUProbeCaptureParams {
    float viewProjection[16], capturePosition[4];
    u32 lightCount, slotCount, materialCount, sampledLightCount;
    float environment[3]; u32 storageFlags; // formerly unused raster seed; ABI size unchanged
    // F15.1 (ticket 22): GPU address of the material extension buffer: the per-slot texture transforms of the secondary-site texture lookups
    // (emitter / hit / capture surfaces); 0 when the scene has none.  Set by the host that owns the params.
#if defined(__METAL_VERSION__)
    const device GPUMaterialExtRecord* materialExtensions;
#else
    u64 materialExtensions;
#endif
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUProbeCaptureParams) == 120, "GPUProbeCaptureParams layout");
struct GPUProbeFilterParams {
    u32 side, mip, sampleCount, cubeIndex;
    float roughness, sourceSide; u32 storageFlags, diagnosticReadback;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUProbeFilterParams) == 32, "GPUProbeFilterParams layout");
struct GPUDenoiseParams {
    u32 width, height, frameIndex, signal;
    u32 viewID, historyEpoch, signalRevision, flags;
    float temporalAlpha, momentsAlpha, depthThreshold, normalThreshold;
    float roughnessThreshold, hitDistanceThreshold, luminancePhi, normalPhi;
    u32 maxHistory, atrousStep, minHistoryForVariance, pad0;
    float varianceFloor, clampSigma, planeThreshold, pad1;
    float currentViewDepthRow[4]; // world point -> current positive view depth (negated dot)
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDenoiseParams) == 112, "GPUDenoiseParams layout");
// F13 denoise history, split by access (budget closure). Each view/signal
// history buffer is [header][capacity records][capacity specular extensions,
// specular signal only]; denoise_history_layout.h packs the record state word
// and gives the offsets. The logical history (CPU DenoiseHistory, MSL
// DenoiseHistory) keeps every field; nothing that a later frame reads is lost.
// Header: the per-frame domain identity, identical for every pixel one temporal
// dispatch writes (its thread 0 stores it).
struct GPUDenoiseHistoryHeader {
    u32 signal, viewID, historyEpoch, signalRevision;
    u32 pad[12]; // 64 bytes keep the records 64-byte aligned
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDenoiseHistoryHeader) == 64, "GPUDenoiseHistoryHeader layout");
// Per-pixel state. The variance is not stored: the temporal pass writes it to
// the moments texture, and a record it marks valid always had a finite
// variance >= varianceFloor > 0 (the only property a later frame tested).
struct GPUDenoiseHistoryRecord {
    float color[3], firstMoment;
    float secondMoment; u32 state, slot, instanceGeneration; // state: length | flags | path | valid
    float normal[3], depth;
    float position[3]; u32 materialRevision;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDenoiseHistoryRecord) == 64, "GPUDenoiseHistoryRecord layout");
struct GPUDenoiseSpecularHistory {
    float roughness, hitDistance; u32 secondarySlot, secondaryGeneration;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUDenoiseSpecularHistory) == 16, "GPUDenoiseSpecularHistory layout");

// F14: ALL positions/radii/heights/distances are WORLD METRES; extinction and
// scattering coefficients are m^-1. RGB radiance and irradiance remain linear
// and pre-exposure. Atmosphere planet centre maps world origin to local ground.
PHOSPHOR_GPU_CONSTANT u32 ATMOSPHERE_SCHEMA = 1u;
PHOSPHOR_GPU_CONSTANT u32 ATMOSPHERE_ENABLE_MOON = 1u;
PHOSPHOR_GPU_CONSTANT u32 ATMOSPHERE_ENABLE_STARS = 2u;
PHOSPHOR_GPU_CONSTANT u32 ATMOSPHERE_REFERENCE_AERIAL = 4u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_HISTORY_VALID = 1u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_SHADOW_CSM = 2u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_SHADOW_RT = 4u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_ENABLE_GI = 8u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_ENABLE_LOCAL_LIGHTS = 16u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_UNITS = 1u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_HISTORY = 2u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_LIGHT = 3u;

struct GPUAtmosphereParams {
    float planetCenter[3], bottomRadius;
    float topRadius, rayleighScaleHeight, mieScaleHeight, mieG;
    float rayleighScattering[3], sunAngularRadius;
    float mieScattering[3], mieAbsorption;
    float ozoneAbsorption[3], ozoneCenterHeight;
    float groundAlbedo[3], ozoneHalfWidth;
    float sunDirection[3], pad0;
    float sunIrradiance[3], pad1;
    float moonDirection[3], moonAngularRadius;
    float moonIrradiance[3], moonPhase;
    float cameraPosition[3], timeSeconds;
    float inverseViewProjection[16], viewProjection[16];
    u32 transmittanceWidth, transmittanceHeight, multiWidth, multiHeight;
    u32 skyWidth, skyHeight, marchSteps, multiDirections;
    u32 outputWidth, outputHeight, parameterRevision, skyRevision;
    u32 frameIndex, viewID, flags, corruption;
    float starIntensity, starRotation, timeDelta, exposureEv100;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUAtmosphereParams) == 384, "GPUAtmosphereParams layout");

struct GPUFogParams {
    float inverseViewProjection[16], view[16], previousViewProjection[16], previousView[16];
    u32 gridX, gridY, gridZ, maxLocalLights;
    u32 outputWidth, outputHeight, frameIndex, viewID;
    float nearDistance, farDistance, densityAtBase, heightFalloff;
    float albedo[3], anisotropy;
    float heightBase, maxDensity, historyWeight, positionThreshold;
    float sunDirection[3], maxTraceDistance;
    float sunIrradiance[3], depthRelativeThreshold;
    float cameraPosition[3], timeSeconds;
    u32 lightCount, slotCount, flags, generation;
    u32 lightRevision, previousViewID, corruption, maxHistoryAge;
    float moonDirection[3], pad0;
    float moonIrradiance[3], pad1;
    // F15.1 (ticket 22): GPU address of the material extension buffer (per-slot texture transforms of textured emitters); 0 without one.
#if defined(__METAL_VERSION__)
    const device GPUMaterialExtRecord* materialExtensions;
#else
    u64 materialExtensions;
#endif
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUFogParams) == 456, "GPUFogParams layout");
struct GPUFogCell {
    float source[3], extinction; // source radiance per metre; scalar extinction m^-1
    float worldPosition[3], viewDepth;
    u32 viewID, generation, age, valid;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUFogCell) == 48, "GPUFogCell layout");
struct GPUFogIntegrated { float radiance[3], transmittance; };
PHOSPHOR_STATIC_ASSERT(sizeof(GPUFogIntegrated) == 16, "GPUFogIntegrated layout");

struct GPUCloudParams {
    float inverseViewProjection[16], previousViewProjection[16];
    float cameraPosition[3], timeSeconds;
    float wind[3], previousTimeSeconds;
    float baseHeight, topHeight, coverage, densityScale;
    float noiseScale, erosionScale, extinction, albedo;
    float anisotropy, maxDistance, terminationTransmittance, historyWeight;
    float depthRelativeThreshold, positionThreshold, lightStepDistance;
    u32 maxHistorySamples;
    u32 width, height, outputWidth, outputHeight;
    u32 marchSteps, lightSteps, seed, generation;
    u32 frameIndex, viewID, flags, corruption;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUCloudParams) == 272, "GPUCloudParams layout");
struct GPUCloudHistory {
    float worldPosition[3], opaqueDistance;
    float radiance[3], transmittance;
    u32 viewID, generation, samples, valid;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUCloudHistory) == 48, "GPUCloudHistory layout");
struct GPUVolumeCounters {
    u32 nonfinite, invalidUnits, invalidHistory, shadowRays;
    u32 froxels, cloudSamples, historyReused, historyRejected;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUVolumeCounters) == 32, "GPUVolumeCounters layout");
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_OMIT_LUT = 4u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_MOON = 5u;
// F14.2 negative: froxel history survives volume-epoch changes and resets
// (moving lights, clock jumps); only the geometric rejection remains.
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_STALE_FOG = 6u;
// F14.3 negative: cloud history reprojected outside the screen wraps to the
// opposite edge and skips every rejection (bounds, position, opaque depth,
// neighbourhood clamp).
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CORRUPT_CLOUD_EDGE = 7u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_TRANS = 0u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_MULTI = 1u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_SKY = 2u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_FOG = 3u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_SOLAR = 4u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_LUNAR = 5u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_NUMERIC_STARS = 6u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_DISK_RAYS = 3u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_STAR_GRID_SIDE = 8u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_STAR_RAYS = VOLUME_STAR_GRID_SIDE*VOLUME_STAR_GRID_SIDE;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_CELESTIAL_RAYS = 2u*VOLUME_DISK_RAYS+VOLUME_STAR_RAYS;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_LUT_SAMPLES = 5u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_FOG_SAMPLES = 3u;
// F14.2 slicing probe: a stratified grid of output pixels read around the fog
// composition (input radiance, depth, output radiance), offset every frame.
PHOSPHOR_GPU_CONSTANT u32 VOLUME_FOG_APPLY_X = 32u, VOLUME_FOG_APPLY_Y = 18u;
PHOSPHOR_GPU_CONSTANT u32 VOLUME_FOG_APPLY_SAMPLES = VOLUME_FOG_APPLY_X*VOLUME_FOG_APPLY_Y;
struct GPUVolumeDiagnosticParams {
    u32 transWidth,transHeight,multiWidth,multiHeight;
    u32 skyWidth,skyHeight,fogX,fogY;
    u32 fogZ,frameLo,frameHi,expectedPhysicsRevision;
    u32 expectedSkyRevision,sampleCount,corruption,homogeneous;
    float fixtureExtinction,fixtureSource[3];
    u32 fogIndices[3],stampKind;
    // Fixture extinction sigma(y) = fixtureExtinction * exp(-falloff * y),
    // source = sigma * fixtureSource / fixtureExtinction (constant albedo ratio).
    float fixtureHeightFalloff;
    u32 applyWidth,applyHeight,applyOffset;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUVolumeDiagnosticParams)==112,"GPUVolumeDiagnosticParams layout");
struct GPUFogApplySample {
    u32 x,y;float depth,pad;
    float before[4],after[4];
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUFogApplySample)==48,"GPUFogApplySample layout");
struct GPUVolumeNumericSample {
    float value[4];
    u32 kind,x,y,producedRevision;
};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUVolumeNumericSample)==32,"GPUVolumeNumericSample layout");

PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_ERROR = 1u<<31;
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_MIXED = 4u;
// Samples averaged into a boosted record (bits 16-23; 0 = one sample). The
// stationary specular history weights each frame by it.
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_COUNT_SHIFT = 16u, SPECULAR_SAMPLE_COUNT_MASK = 0xffu;
// F15.1 (ticket 08): the lobe the transport sampled for this pixel, in spare bits of GPUSpecularSample.flags
// (bit 31 stays SPECULAR_SAMPLE_ERROR).  bits 8-9: lobe id (0 = base specular, 1 = clearcoat, 2 = transmission,
// 3 reserved); bit 10: axis (0 = reflect, 1 = refract).  The "_lobe" denoiser kernels reject neighbours and history
// across a different (id, axis) key and take the edge-stopping roughness and spatial guide axis of the sampled lobe.
// Today the transport only samples lobe 0 / reflect, so every key is 0.
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_LOBE_SHIFT = 8u, SPECULAR_SAMPLE_LOBE_MASK = 3u;
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_AXIS_REFRACT = 1u << 10;
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_SAMPLE_LOBE_KEY_MASK = (3u << 8) | (1u << 10);
PHOSPHOR_GPU_CONSTANT u32 SPECULAR_LOBE_BASE = 0u, SPECULAR_LOBE_CLEARCOAT = 1u, SPECULAR_LOBE_TRANSMISSION = 2u;
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_GI = 1u;
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_CUSTOM = 2u;
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_DI = 4u;
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_SPEC = 8u;
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_AO = 16u;
// SSR source only: remove the receiver's packed emission (F13.1 single count).
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_EXCLUDE_EMISSION = 32u;
// F15.1 (ticket 08): the lobe guide (buffer 3 of reflection_composite_lobe) is valid this frame.
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_LOBE_GUIDE_VALID = 64u;
// F15.1 (ticket 19): texture 8 of the composite holds E(-N) for the thin subsurface pixels (DDGI); without it the pixel's own GI stands for it.
PHOSPHOR_GPU_CONSTANT u32 REFLECT_COMPOSE_GI_BACK = 128u;
struct GPUReflectionReduceParams {u32 width,height,samples,stridePixels;};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUReflectionReduceParams)==16,"GPUReflectionReduceParams layout");
struct GPUReflectionComposeParams {u32 width,height,backingWidth,backingHeight;u32 flags,pad[3];float environment[3];u32 pad2;};
PHOSPHOR_STATIC_ASSERT(sizeof(GPUReflectionComposeParams)==48,"GPUReflectionComposeParams layout");

} // namespace phosphor
