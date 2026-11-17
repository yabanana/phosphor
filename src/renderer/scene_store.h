#pragma once

#include "core/types.h"
#include "renderer/gpu_types.h"
#include "renderer/scene_extract.h" // CullClass

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace phosphor {

class ECS;
class GpuScene;

// ---------------------------------------------------------------------------
// SceneStore -- CPU mirror of the persistent GPU scene (F5.1).  Portable.
//
// Instances live in stable SLOTS grouped by draw bucket = (cull class, mesh),
// buckets sorted by (class, mesh): the canonical draw order of both
// submission paths is (class, mesh, slot), so --gpu-driven off and on give
// the same image (D2).  Each bucket owns a contiguous slot region with
// capacity (25% slack, at least 64); a removal leaves a hole, an add takes a
// hole of the bucket or the next free slot of its region; a full region
// without holes relocates the bucket
// to the end of the slot space with twice the capacity (a STRUCTURE change,
// counted, never expected in measured frames).  Slack slots have
// GPUInstance::flags without INSTANCE_FLAG_VALID.  A removal leaves a HOLE
// (a zeroed slot, reused by the next add to the bucket) instead of moving
// another instance: churn moves no slot, so it never touches the motion list
// or the hierarchy CSR (no structure change, bytes proportional to changes).
//
// sync() consumes the ECS change lists (scene/ecs.h): its cost is
// O(changed), never O(instances), except the first sync after clear() (a
// full build) and structure changes.  It produces this frame's delta records
// (GPUDeltaRecord, applied by kernel scene_scatter) for the instance,
// material, node and motion buffers, or asks for a full copy of a buffer
// when more than 1/8 of it changed (D1: one memcpy beats scattered records).
//
// Materials: the GpuScene library first (indices unchanged), then one
// persistent material per entity with a MaterialComponent, allocated once
// (free list) and re-sent only when the component changes.
//
// Material extensions (F15.1): an extended material (one with an active
// feature) owns a fixed group of MATERIAL_EXT_RECORDS records in a separate
// buffer (extensions(), GPUMaterial::extensionIndex = first record, feature
// bits in GPUMaterial::flags).  The group is allocated when the material first
// becomes extended (a counted structure change, never per frame) and kept
// while the material lives, so toggling features rewrites records but never
// reallocates; its records are produced by packMaterialExtensions from the
// library description (GpuScene::materialDescriptions) or the entity's
// MaterialComponent::extension and travel as deltas exactly like materials.
// featureUnion() is the OR of the feature bits of the live materials.
//
// Hierarchy (F5.2): an entity with a HierarchyComponent is a child: its
// slot has a GPUTransformNode (local matrix = its TransformComponent's
// worldMatrix field, parent slot, depth) and its world matrix is computed on
// the GPU; the store keeps the CSR child lists in slot space and, per
// frame, the dirty roots (queue 0): roots whose world matrix changed this
// frame (CPU delta or procedural motion) and that have children.  The cull
// class of a child comes from the product of the determinant signs along
// its chain (the CPU decides the bucket, the GPU never reads back).
//
// Motion (F5.6): roots with a MotionComponent get a GPUMotion record (base =
// TransformComponent::worldMatrix) and are listed in motionSlots(); the
// GPU writes their world matrix every frame (scene_motion).
// ---------------------------------------------------------------------------

// Placeholders (F5.2/F5.6): the GPUInstance::modelMatrix of a child or of a
// motion root in the mirror (and in delta records / full copies) is the
// IDENTITY matrix; the GPU overwrites it (scene_hier_level / scene_motion).
// Whenever such a slot's record is written or moved, its parent is listed in
// dirtyRoots() (and a full instance copy lists every root with children).
// Empty buckets are kept (their ICB command draws nothing) so a bucket
// emptied and refilled does not change the command layout.

struct SceneBucket {
    u32       mesh = 0;
    CullClass cull = CullClass::Back;
    u32       firstSlot = 0;
    u32       capacity  = 0;
    u32       count     = 0; // live instances
    u32       used      = 0; // slots [firstSlot, firstSlot + used) hold the live instances and the
                             // holes left by removals (zeroed records, reused first): never drawn
    u32       command   = 0; // ICB command index
};

/// What one sync() produced (report "scene" object, panel).
struct SceneSyncStats {
    u32  instances       = 0; // live instances
    u32  slots           = 0; // slot capacity
    u32  buckets         = 0;
    u32  materials       = 0;
    u32  instanceRecords = 0; // delta records written this frame, per buffer
    u32  materialRecords = 0;
    u32  extensionRecords = 0; // extension delta records this frame
    u32  nodeRecords     = 0;
    u32  motionRecords   = 0;
    u32  dirtyRoots      = 0;
    bool fullInstances   = false; // whole buffer copied this frame
    bool fullMaterials   = false;
    bool fullExtensions  = false;
    bool fullNodes       = false;
    bool fullMotions     = false;
    bool structure       = false; // bucket table / CSR / motion list / capacity changed
    bool csrRebuilt      = false; // hierarchy-related slots moved: the CSR was recomputed (O(slots))
    bool motionParentsChanged = false; // motionParentSlots() differs from the previous sync
    u64  uploadBytes     = 0;     // bytes the CPU must write this frame (records + full copies + structure)
};

class SceneStore {
public:
    struct ClassRange {
        u32 firstCommand = 0;
        u32 commandCount = 0; // buckets of the class + 1 sentinel
    };

    SceneStore();
    ~SceneStore();
    SceneStore(const SceneStore&) = delete;
    SceneStore& operator=(const SceneStore&) = delete;

    /// Drop everything (bench switch); the next sync() rebuilds from the ECS.
    void clear();

    /// Apply this frame's ECS changes (then the caller clears the ECS change
    /// lists with ECS::endFrame()).  `scene` gives the mesh infos and the
    /// material library.
    void sync(ECS& ecs, const GpuScene& scene);

    // --- Mirror (slot space; what the GPU buffers must hold) -------------------
    [[nodiscard]] std::span<const GPUInstance>      instances() const;  // slotCapacity()
    [[nodiscard]] std::span<const GPUMaterial>      materials() const;
    /// F15.1: extension records (extensionGroupCapacity() * MATERIAL_EXT_RECORDS).
    [[nodiscard]] std::span<const GPUMaterialExtRecord> extensions() const;
    [[nodiscard]] std::span<const GPUTransformNode> nodes() const;      // slotCapacity(); unused for roots
    [[nodiscard]] std::span<const GPUMotion>        motions() const;    // slotCapacity(); unused without motion
    [[nodiscard]] std::span<const SceneBucket>      buckets() const;    // sorted by (cull, mesh)
    [[nodiscard]] std::span<const GPUDrawBucket>    gpuBuckets() const; // same order, GPU layout
    /// Per ICB command: bucket index, or ~0u for the sentinel ending a class range.
    [[nodiscard]] std::span<const u32>              commandBuckets() const;
    [[nodiscard]] std::array<ClassRange, SCENE_CULL_CLASSES> classRanges() const;
    [[nodiscard]] u32 commandCount() const;
    [[nodiscard]] u32 slotCapacity() const;
    [[nodiscard]] u32 instanceCount() const;
    /// Slots whose world matrix the GPU computes from a GPUMotion.
    [[nodiscard]] std::span<const u32> motionSlots() const;
    /// CSR of children in slot space: children of slot s are
    /// childSlots()[childOffsets()[s] .. childOffsets()[s + 1]).
    [[nodiscard]] std::span<const u32> childOffsets() const; // slotCapacity() + 1
    [[nodiscard]] std::span<const u32> childSlots() const;
    /// Parents whose descendants must be recomputed this frame because of a
    /// CPU change (GPU queue 0).  Motion roots with children are NOT listed:
    /// they move every frame and are expanded from motionParentSlots().
    [[nodiscard]] std::span<const u32> dirtyRoots() const;
    /// Slots of the motion roots that have children (a persistent GPU queue,
    /// re-sent only when stats().motionParentsChanged).
    [[nodiscard]] std::span<const u32> motionParentSlots() const;
    [[nodiscard]] u32 maxDepth() const; // 0 = no hierarchy
    /// True if any material is emissive (forward variant selection).
    [[nodiscard]] bool hasEmissive() const;
    /// F15.1: OR of the feature bits (MaterialFeature bit values, unshifted) of
    /// the live materials; 0 for a scene without features (Sponza, benches).
    [[nodiscard]] u32 featureUnion() const;
    /// F15.1 (ticket 22): a live material carries per-slot KHR_texture_transform data (MATERIAL_FLAG_TEXTURE_TRANSFORM): the scene's
    /// shading pipelines compile the transform (reserved constant TEXTURE_TRANSFORM); false for Sponza and every featureless scene.
    [[nodiscard]] bool textureTransformInUse() const;
    /// featureUnion() with FEATURE_UNION_TEXTURE_TRANSFORM or-ed in when textureTransformInUse(): what FeaturePipelines::serve takes.
    [[nodiscard]] u32 pipelineFeatureUnion() const {
        return featureUnion() | (textureTransformInUse() ? FEATURE_UNION_TEXTURE_TRANSFORM : 0u) | (vertexColorInUse() ? FEATURE_UNION_VERTEX_COLOR : 0u);
    }
    /// F15 closure (COLOR_0): the synced GpuScene has a per-vertex colour stream.
    [[nodiscard]] bool vertexColorInUse() const;
    /// F15.1 (ticket 17): a live material carries the subsurface feature in VOLUMETRIC mode (record 2 mode 1) with a weight above 0 (or a
    /// weight texture): the scene needs the diffuse-irradiance target and the diffusion pass.
    [[nodiscard]] bool volumetricSubsurface() const;
    /// Extension groups handed out so far (live + free) and the capacity.
    [[nodiscard]] u32 extensionGroups() const;
    [[nodiscard]] u32 extensionGroupCapacity() const;

    /// Slot of an entity's instance (~0u: not in the store) and its GPU
    /// material index (the library index or its persistent per-entity one).
    [[nodiscard]] u32 slotOf(EntityID entity) const;
    [[nodiscard]] u32 materialIndexOf(EntityID entity) const;
    /// CSR rebuilds since clear() (hierarchy-related slot moves).
    [[nodiscard]] u64 csrRebuildCount() const;

    // --- This frame's deltas ----------------------------------------------------
    [[nodiscard]] std::span<const GPUDeltaRecord> instanceDeltas() const;
    [[nodiscard]] std::span<const GPUDeltaRecord> materialDeltas() const;
    [[nodiscard]] std::span<const GPUDeltaRecord> extensionDeltas() const;
    /// True when any material data changed this frame (base records or
    /// extension records, deltas or full copies): the single "material
    /// revision" test of every consumer that caches material-dependent state.
    [[nodiscard]] bool materialsChanged() const;
    [[nodiscard]] std::span<const GPUDeltaRecord> nodeDeltas() const;
    [[nodiscard]] std::span<const GPUDeltaRecord> motionDeltas() const;
    [[nodiscard]] const SceneSyncStats& stats() const;
    /// Incremented whenever slot capacity, the bucket table, the command
    /// layout, the CSR or the motion list change (buffers re-sized/re-sent,
    /// render graph key).
    [[nodiscard]] u64 structureVersion() const;

    // --- Self-check -------------------------------------------------------------
    /// O(instances): rebuild the expected mirror from the ECS and compare (a
    /// change the ECS did not report -- a missed touch -- shows up here).
    /// Empty string = equal; otherwise the first difference.
    [[nodiscard]] std::string verifyAgainstEcs(const ECS& ecs, const GpuScene& scene) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace phosphor
