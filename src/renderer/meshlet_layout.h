#pragma once

// ---------------------------------------------------------------------------
// F6 -- contract between the meshlet shaders (shaders/meshlet.metal: candidate
// and phase-B kernels, object and mesh shaders; shaders/hiz.metal: pyramid)
// and the host (platform/metal/mesh_renderer, platform/metal/hiz_builder,
// Engine::declareFrameGraph).  Shared by C++ and MSL like gpu_scene_layout.h.
//
// Frame (--geometry-path mesh, needs gpu-driven on; F5 passes first):
//
//   Instance cull       F5 (scene_cull_flags/scan/write): the visible slots
//   Meshlet candidates  meshlet_cand_count, meshlet_cand_scan, meshlet_cand_write:
//                       every meshlet of every visible slot, in 3 contiguous
//                       class regions (Back, BackMirrored, None) of the
//                       candidate list, in slot order inside a class; phase-A
//                       draw ranges + indirect mesh arguments; overflow word
//   Draw build          F5 ICB build, GATED by the overflow word: resets only,
//                       unless the candidates overflowed (then the F5 indexed
//                       draws render the frame: no truncation, never a lost object)
//   Forward             phase A: per class, cull state + drawMeshThreadgroups
//                       (indirect); object shader: frustum + cone + history
//                       Hi-Z test, writes the B flag of every candidate it owns
//   Hi-Z A              pyramid of phase A's depth (shaders/hiz.metal)
//   Meshlet B           meshlet_b_count, meshlet_b_scan, meshlet_b_write: the
//                       history-rejected candidates, stable, per class; phase-B
//                       draw ranges + arguments
//   Forward B           phase B (depth/colour loaded): object shader retests the
//                       B list against the CURRENT pyramid (Hi-Z A)
//   Hi-Z final          pyramid of the final depth -> next frame's history
//
// --meshlet-cull frustum: Forward only (no Hi-Z, no phase B); off: object
// shader without tests.  Every dispatch / draw above is encoded every frame
// whatever the scene (empty ranges launch nothing): the CPU command count is
// constant (O8).  Barriers: compute -> compute Dispatch -> Dispatch; indirect
// mesh arguments and the lists are consumed at Vertex|Object|Mesh (spike S4);
// object-shader writes (B flags) -> compute: Object -> Dispatch.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"

namespace phosphor {

// --- Threadgroup sizes ---------------------------------------------------------
/// Candidate and phase-B kernels: 1024 threads (32 SIMD-groups) per group;
/// the scan kernels are ONE group that loops over the group sums.
PHOSPHOR_GPU_CONSTANT u32 MESHLET_SCAN_GROUP   = 1024;
/// Object shader: one SIMD-group, one candidate per thread; the visible
/// candidates are compacted into the payload with simd_prefix_exclusive_sum.
PHOSPHOR_GPU_CONSTANT u32 MESHLET_OBJECT_GROUP = 32;
/// Mesh shader: one vertex and one triangle per thread (>= the cook limits).
PHOSPHOR_GPU_CONSTANT u32 MESHLET_MESH_GROUP   = 128;
/// Payload of one object threadgroup: (slot, meshlet, decision) of up to 32 meshlets.
PHOSPHOR_GPU_CONSTANT u32 MESHLET_PAYLOAD_BYTES = MESHLET_OBJECT_GROUP * 12;
/// Classes x phases of the indirect mesh draws.
PHOSPHOR_GPU_CONSTANT u32 MESHLET_DRAWS = SCENE_CULL_CLASSES * 2;
/// Upper bound of the meshlets of one mesh (loop bound of meshlet_cand_write).
PHOSPHOR_GPU_CONSTANT u32 MESHLET_MAX_PER_MESH = 65536;

/// Hi-Z: power-of-two level 0 >= ceil(viewport / 2), at most 13 levels (8192).
PHOSPHOR_GPU_CONSTANT u32 HIZ_MAX_LEVELS = 13;
PHOSPHOR_GPU_CONSTANT u32 HIZ_GROUP      = 16; // 16 x 16 threads per reduction group
/// Phase A/B occlusion test: texels read per axis at most (the level is the
/// finest one whose footprint of the bound's rectangle fits in 4 x 4 texels).
PHOSPHOR_GPU_CONSTANT u32 HIZ_TEST_SPAN  = 4;

// --- Counter word indices (GPUMeshletCounters) -----------------------------------
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_CANDIDATES = 0;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_OVERFLOW   = 1;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_DRAWN_A    = 2;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_FRUSTUM    = 3;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_CONE       = 4;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_HISTORY    = 5;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_TESTED_B   = 6;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_DRAWN_B    = 7;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_OCCLUDED_B = 8;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_PRIMS_A    = 9;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_PRIMS_B    = 10;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_GROUPS_A   = 11;
PHOSPHOR_GPU_CONSTANT u32 MESHLET_COUNTER_GROUPS_B   = 12;

// --- Argument-table slots: candidate / phase-B kernels (one table) --------------
PHOSPHOR_GPU_CONSTANT u32 MB_PARAMS      = 0;  // GPUMeshletCullParams
PHOSPHOR_GPU_CONSTANT u32 MB_INSTANCES   = 1;  // GPUInstance[] by slot
PHOSPHOR_GPU_CONSTANT u32 MB_MESHES      = 2;  // GPUMeshInfo[]
PHOSPHOR_GPU_CONSTANT u32 MB_MATERIALS   = 3;  // GPUMaterial[] (double-sided -> class None)
PHOSPHOR_GPU_CONSTANT u32 MB_SCENE_FLAGS = 4;  // u32[slots]: F5 instance cull flags (bit 0 visible)
PHOSPHOR_GPU_CONSTANT u32 MB_GROUP_SUMS  = 5;  // u32[groups * 3]: per-class sums, then (scan) offsets
PHOSPHOR_GPU_CONSTANT u32 MB_RANGES      = 6;  // GPUMeshletDrawRange[MESHLET_DRAWS] (phase * 3 + class)
PHOSPHOR_GPU_CONSTANT u32 MB_ARGS        = 7;  // u32[MESHLET_DRAWS * 3]: threadgroups per grid (indirect)
PHOSPHOR_GPU_CONSTANT u32 MB_COUNTERS    = 8;  // GPUMeshletCounters
PHOSPHOR_GPU_CONSTANT u32 MB_GATE        = 9;  // u32[1]: nonzero = overflow (the F5 ICB fallback draws)
PHOSPHOR_GPU_CONSTANT u32 MB_CANDIDATES  = 10; // GPUMeshletCandidate[capacity]
PHOSPHOR_GPU_CONSTANT u32 MB_B_FLAGS     = 11; // u32[capacity]: 1 = history-rejected in phase A
PHOSPHOR_GPU_CONSTANT u32 MB_B_SUMS      = 12; // u32[candidateGroups * 3]
PHOSPHOR_GPU_CONSTANT u32 MB_B_LIST      = 13; // GPUMeshletCandidate[capacity]
PHOSPHOR_GPU_CONSTANT u32 MB_BIND_COUNT  = 14;

// --- Argument-table slots: mesh-path draws (object + mesh + fragment) ----------
// The fragment shader is forward_fs: slots 0, 3, 4, 5 as the forward pass.
PHOSPHOR_GPU_CONSTANT u32 MR_FRAME      = 0;  // FrameConstants
PHOSPHOR_GPU_CONSTANT u32 MR_VERTICES   = 1;  // GPUVertex[]
PHOSPHOR_GPU_CONSTANT u32 MR_INSTANCES  = 2;  // GPUInstance[]
PHOSPHOR_GPU_CONSTANT u32 MR_MATERIALS  = 3;
PHOSPHOR_GPU_CONSTANT u32 MR_LIGHTS     = 4;
PHOSPHOR_GPU_CONSTANT u32 MR_TEXTURES   = 5;
PHOSPHOR_GPU_CONSTANT u32 MR_MESHLETS   = 6;  // GPUMeshlet[] (global)
PHOSPHOR_GPU_CONSTANT u32 MR_MESHLET_VERTICES  = 7;  // u32[] global vertex indices
PHOSPHOR_GPU_CONSTANT u32 MR_MESHLET_TRIANGLES = 8;  // uchar[] packed local indices
PHOSPHOR_GPU_CONSTANT u32 MR_BOUNDS     = 9;  // GPUMeshletBounds[] (global)
PHOSPHOR_GPU_CONSTANT u32 MR_PARAMS     = 10; // GPUMeshletCullParams
PHOSPHOR_GPU_CONSTANT u32 MR_RANGE      = 11; // GPUMeshletDrawRange of this draw (address per draw)
PHOSPHOR_GPU_CONSTANT u32 MR_LIST       = 12; // GPUMeshletCandidate[]: candidates (A) or B list (B)
PHOSPHOR_GPU_CONSTANT u32 MR_B_FLAGS    = 13; // u32[] written by phase A
PHOSPHOR_GPU_CONSTANT u32 MR_COUNTERS   = 14; // GPUMeshletCounters
/// u32[2 * capacity]: with MESHLET_CULL_RECORD (self-check frames) phase A
/// writes the decision of candidate i at [i], phase B the decision of B-list
/// entry j at [capacity + j].
PHOSPHOR_GPU_CONSTANT u32 MR_DECISIONS  = 15;
PHOSPHOR_GPU_CONSTANT u32 MR_BIND_COUNT = 16;
/// Texture slot of the pyramid the object shader tests against (history in
/// phase A, Hi-Z A in phase B).
PHOSPHOR_GPU_CONSTANT u32 MR_TEX_HIZ    = 0;

// --- Hi-Z kernels (shaders/hiz.metal) -------------------------------------------
PHOSPHOR_GPU_CONSTANT u32 HZ_PARAMS  = 0;  // GPUHiZParams
PHOSPHOR_GPU_CONSTANT u32 HZ_TEX_SRC = 0;  // depth (level 0) or the pyramid (source level)
PHOSPHOR_GPU_CONSTANT u32 HZ_TEX_DST = 1;  // the pyramid (destination level)

// --- Render-graph pass names (renamed passes invalidate OPT-1 plans) ----------
#ifndef __METAL_VERSION__
inline constexpr const char* PASS_MESHLET_CANDIDATES = "Meshlet candidates";
inline constexpr const char* PASS_FORWARD_B          = "Forward B";
inline constexpr const char* PASS_HIZ_A              = "Hi-Z A";
inline constexpr const char* PASS_MESHLET_B          = "Meshlet B";
inline constexpr const char* PASS_HIZ_FINAL          = "Hi-Z final";

// --- Function names --------------------------------------------------------------
inline constexpr const char* KERNEL_MESHLET_CAND_COUNT = "meshlet_cand_count";
inline constexpr const char* KERNEL_MESHLET_CAND_SCAN  = "meshlet_cand_scan";
inline constexpr const char* KERNEL_MESHLET_CAND_WRITE = "meshlet_cand_write";
inline constexpr const char* KERNEL_MESHLET_B_COUNT    = "meshlet_b_count";
inline constexpr const char* KERNEL_MESHLET_B_SCAN     = "meshlet_b_scan";
inline constexpr const char* KERNEL_MESHLET_B_WRITE    = "meshlet_b_write";
inline constexpr const char* MESHLET_OBJECT_FN         = "meshlet_object";
inline constexpr const char* MESHLET_MESH_FN           = "meshlet_mesh";
inline constexpr const char* MESHLET_MESH_DEBUG_FN     = "meshlet_mesh_debug";
inline constexpr const char* MESHLET_DEBUG_FS          = "meshlet_debug_fs";
inline constexpr const char* KERNEL_HIZ_LEVEL0         = "hiz_level0";
inline constexpr const char* KERNEL_HIZ_REDUCE         = "hiz_reduce";
inline constexpr const char* KERNEL_HIZ_REDUCE_SAMPLER = "hiz_reduce_sampler";
#endif

} // namespace phosphor
