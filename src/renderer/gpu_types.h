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

} // namespace phosphor
