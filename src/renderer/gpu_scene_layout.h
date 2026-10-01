#pragma once

// ---------------------------------------------------------------------------
// F5 -- contract between the GPU scene kernels (shaders/gpu_scene.metal,
// shaders/transforms.metal) and the host (platform/metal/gpu_scene_buffers,
// SceneRenderer, Engine::buildFrameGraph).  Shared by C++ and MSL like
// gpu_types.h: kernel names, argument-table slots, threadgroup sizes, the
// small parameter blocks and the render-graph pass names.
//
// Frame (gpu-driven on; off runs only Scene update + Scene transforms):
//
//   Scene update      scene_queue_clear, scene_scatter x4 (instances,
//                     materials, nodes, motion) or full copies (D1)
//   Scene transforms  scene_motion, then for L = 0 .. SCENE_MAX_LEVELS-2:
//                     scene_queue_args(L) + scene_hier_level(L) (indirect)
//   Instance cull     scene_cull_flags, scene_cull_scan, scene_cull_write (D4)
//   Draw build        resetCommandsInBuffer + scene_draw_build (D2)
//   Forward           3 x (cull state + executeCommandsInBuffer(range))
//
// Every dispatch above is encoded every frame whatever the scene (empty
// work early-outs on the GPU): the CPU command count is constant (O8).
// Barriers (S5): compute -> compute Dispatch -> Dispatch; ICB and the visible
// list are read by the forward pass at the Vertex stage.
// ---------------------------------------------------------------------------

#include "renderer/gpu_types.h"

namespace phosphor {

// --- Threadgroup sizes ---------------------------------------------------------
PHOSPHOR_GPU_CONSTANT u32 SCENE_SCATTER_GROUP = 64;
PHOSPHOR_GPU_CONSTANT u32 SCENE_MOTION_GROUP  = 64;
PHOSPHOR_GPU_CONSTANT u32 SCENE_HIER_GROUP    = 64;
/// scene_cull_flags / scene_cull_write: one thread per slot, 1024 per group
/// (32 SIMD-groups); scene_cull_scan: ONE group of 1024 threads that scans
/// GPUCullParams::groupCount group counts (any count: it loops over chunks
/// of 1024, so 4M slots = 4096 groups are fine).
PHOSPHOR_GPU_CONSTANT u32 SCENE_CULL_GROUP    = 1024;
PHOSPHOR_GPU_CONSTANT u32 SCENE_DRAW_GROUP    = 64;

// --- Parameter blocks (argument slot 0 of each kernel) -------------------------

/// scene_scatter: `count` records of `words` payload words each are copied
/// to dst[slot * words .. + words).  count == 0: the dispatch has 1 thread
/// that returns.
struct GPUScatterParams {
    u32 count;
    u32 words;
    u32 pad[2];
};

/// scene_motion: per-frame motion table.
struct GPUMotionFrame {
    u32   motionCount;                        // entries of the motion slot list
    u32   pad[3];
    float sinCos[SCENE_MOTION_CLASSES * 2];   // (sin a, cos a) per speed class
};

/// scene_queue_clear: queues 1 .. SCENE_MAX_LEVELS-1 live back to back in
/// one buffer (queue L at (L - 1) * strideBytes; queue 0, the dirty roots,
/// is written by the CPU in the frame upload ring with its header and
/// groups).  Clears count / overflow / groups of each and the counters.
struct GPUQueueClearParams {
    u32 queues;       // SCENE_MAX_LEVELS - 1
    u32 strideBytes;  // gpuQueueBytes(capacity), a multiple of 32
    u32 pad[2];
};

/// scene_queue_args / scene_hier_level.
struct GPUHierParams {
    u32 level;   // 0: expand roots only (their world is final); >= 1: compute world, then expand
    u32 pad[3];
};

/// scene_draw_build.
struct GPUDrawParams {
    u32 commandCount;  // buckets + SCENE_CULL_CLASSES sentinels
    u32 bucketCount;
    u32 slotCount;     // prefix has slotCount + 1 entries
    u32 pad;
};

// --- Argument-table slots ------------------------------------------------------
// One MTL4 argument table per pass; addresses are set per dispatch.

// scene_queue_clear(queues of the frame, counters)
PHOSPHOR_GPU_CONSTANT u32 SB_CLEAR_QUEUES   = 1;  // queues 1 .. SCENE_MAX_LEVELS-1 (GPUQueueClearParams)
PHOSPHOR_GPU_CONSTANT u32 SB_CLEAR_COUNTERS = 2;  // GPUSceneCounters
// scene_scatter(params, records, dst)
PHOSPHOR_GPU_CONSTANT u32 SB_SCATTER_RECORDS = 1; // GPUDeltaRecord[]
PHOSPHOR_GPU_CONSTANT u32 SB_SCATTER_DST     = 2; // u32[] (instances / materials / nodes / motion)
// scene_motion(frame, motionSlots, motion, instances)
PHOSPHOR_GPU_CONSTANT u32 SB_MOTION_SLOTS     = 1; // u32[] slots with procedural motion
PHOSPHOR_GPU_CONSTANT u32 SB_MOTION_RECORDS   = 2; // GPUMotion[] by slot
PHOSPHOR_GPU_CONSTANT u32 SB_MOTION_INSTANCES = 3; // GPUInstance[] by slot (modelMatrix written)
// scene_queue_args(params, queueIn) and scene_hier_level(params, queueIn, queueOut, ...)
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_QUEUE_IN     = 1; // GPUQueueHeader + u32[] (level L)
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_QUEUE_OUT    = 2; // level L + 1
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_NODES        = 3; // GPUTransformNode[] by slot
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_INSTANCES    = 4; // GPUInstance[] by slot
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_CHILD_OFFSETS = 5; // u32[slots + 1] (CSR)
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_CHILD_SLOTS  = 6; // u32[]
PHOSPHOR_GPU_CONSTANT u32 SB_HIER_COUNTERS     = 7; // GPUSceneCounters (nodesUpdated, queueOverflow)
// scene_cull_flags / scan / write(params, ...)
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_INSTANCES = 1; // GPUInstance[]
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_MESHES    = 2; // GPUMeshInfo[]
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_FLAGS     = 3; // u32[slots] 1 = visible
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_GROUPS    = 4; // u32[groups] counts, then (scan) exclusive offsets
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_COUNTERS  = 5; // GPUSceneCounters
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_VISIBLE   = 6; // u32[slots] visible slots, slot order
PHOSPHOR_GPU_CONSTANT u32 SB_CULL_PREFIX    = 7; // u32[slots + 1] exclusive prefix of the flags
// scene_draw_build(params, buckets, commandBuckets, prefix, icb, indices, drawArgs, counters)
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_BUCKETS     = 1; // GPUDrawBucket[]
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_COMMANDS    = 2; // u32[commands]: bucket index or ~0u (sentinel)
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_PREFIX      = 3; // u32[slots + 1]
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_ICB         = 4; // GPUIcbContainer (MSL: { command_buffer icb; }, host: MTL::ResourceID)
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_INDICES     = 5; // u32[] global index buffer
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_ARGS        = 6; // u32[commands * 2]: instanceCount, baseInstance (self-check)
PHOSPHOR_GPU_CONSTANT u32 SB_DRAW_COUNTERS    = 7; // GPUSceneCounters (drawCommands)
PHOSPHOR_GPU_CONSTANT u32 SCENE_BIND_COUNT    = 8;

/// Forward pass: the visible list (identity in gpu-driven off) after the
/// F0 slots 0..5 of shaders/forward.metal.
PHOSPHOR_GPU_CONSTANT u32 FORWARD_BIND_VISIBLE = 6;

// --- Render-graph pass names (renamed passes invalidate OPT-1 plans) ----------
#ifndef __METAL_VERSION__
inline constexpr const char* PASS_SCENE_UPDATE     = "Scene update";
inline constexpr const char* PASS_SCENE_TRANSFORMS = "Scene transforms";
inline constexpr const char* PASS_INSTANCE_CULL    = "Instance cull";
inline constexpr const char* PASS_DRAW_BUILD       = "Draw build";
#endif

// --- Kernel names (pipe::PipelineDesc::functions[0]) ---------------------------
#ifndef __METAL_VERSION__
inline constexpr const char* KERNEL_QUEUE_CLEAR = "scene_queue_clear";
inline constexpr const char* KERNEL_SCATTER     = "scene_scatter";
inline constexpr const char* KERNEL_MOTION      = "scene_motion";
inline constexpr const char* KERNEL_QUEUE_ARGS  = "scene_queue_args";
inline constexpr const char* KERNEL_HIER_LEVEL  = "scene_hier_level";
inline constexpr const char* KERNEL_CULL_FLAGS  = "scene_cull_flags";
inline constexpr const char* KERNEL_CULL_SCAN   = "scene_cull_scan";
inline constexpr const char* KERNEL_CULL_WRITE  = "scene_cull_write";
inline constexpr const char* KERNEL_DRAW_BUILD  = "scene_draw_build";
#endif

} // namespace phosphor
