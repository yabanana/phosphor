#include "renderer/scene_store.h"

#include "core/log.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_extension_pack.h"
#include "scene/components.h"
#include "scene/ecs.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <utility>

namespace phosphor {

namespace {

constexpr u32 NONE         = ~0u;
constexpr u32 MIN_BUCKET   = 64;       // slots of the smallest bucket region
constexpr u32 MAX_ENTITY_ID = 1u << 26; // sanity bound for hierarchy parents (the id indexes a vector)

[[nodiscard]] u64 makeKey(CullClass c, u32 mesh) { return (u64(static_cast<u32>(c)) << 32) | mesh; }
[[nodiscard]] bool negDet(const glm::mat4& m) { return glm::determinant(glm::mat3(m)) < 0.0f; }
[[nodiscard]] bool isEmissive(const GPUMaterial& m) {
    return m.emissive[0] != 0.0f || m.emissive[1] != 0.0f || m.emissive[2] != 0.0f;
}
template <typename T>
[[nodiscard]] bool sameBytes(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <typename T>
[[nodiscard]] bool isZero(const T& a) {
    const T z{};
    return std::memcmp(&a, &z, sizeof(T)) == 0;
}

void setIdentity(float* m) {
    std::memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

[[nodiscard]] u32 initialCapacity(u32 count) { return std::max(MIN_BUCKET, count + (count + 3) / 4); }

template <typename V>
void growVector(V& v, size_t n) {
    if (n > v.capacity()) v.reserve(std::max(n, v.capacity() * 2));
}

[[nodiscard]] std::string fmt(const char* f, ...) {
    char buf[320];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

// Records shared by the incremental path and verifyAgainstEcs().
GPUTransformNode makeNode(const TransformComponent& xf, u32 parentSlot, u32 depth) {
    GPUTransformNode n{};
    std::memcpy(n.local, &xf.worldMatrix[0][0], sizeof(n.local));
    n.parentSlot = parentSlot;
    n.depth      = depth;
    return n;
}

GPUMotion makeMotion(const MotionComponent& m, const TransformComponent& xf) {
    GPUMotion g{};
    g.centre[0]   = m.centre.x;
    g.centre[1]   = m.centre.y;
    g.centre[2]   = m.centre.z;
    g.radius      = m.radius;
    g.cosPhase    = std::cos(m.phase);
    g.sinPhase    = std::sin(m.phase);
    g.height      = m.height;
    g.speedClass  = m.speedClass;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 3; ++r) g.base[c * 3 + r] = xf.worldMatrix[c][r];
    }
    return g;
}

} // namespace

struct SceneStore::Impl {
    // --- Decided (desired) state of an entity vs. where it is placed ----------
    struct Decision {
        bool want   = false; // belongs in the store
        bool neg    = false; // cumulative determinant sign along the chain is negative
        bool motion = false;
        u8   depth  = 0;
        u64  key    = 0;     // bucket key (class, mesh)
        bool operator==(const Decision& o) const {
            return want == o.want && neg == o.neg && motion == o.motion && depth == o.depth && key == o.key;
        }
    };

    struct Ent {
        Decision d;
        u32  slot = NONE;      // placed slot
        u32  bucket = NONE;    // bucket id (stable) of the placed slot
        u32  matSlot = NONE;   // persistent per-entity material (relative to the library)
        u32  parent = NONE;    // linked hierarchy parent (HierarchyComponent), placed or not
        u32  firstChild = NONE, nextSib = NONE, prevSib = NONE;
        u32  liveChildren = 0; // children currently in the store
        u32  motionPos = NONE; // index in motionSlots_
        u32  motionParentPos = NONE; // index in motionParents_
        u32  dpStamp = 0;      // dirty-parent dedupe (frame)
        u32  decStamp = 0;     // full-build decision memo
        bool queued = false;
    };

    // --- Inputs of the current sync ---------------------------------------------
    const ComponentArray<TransformComponent>*    xf_  = nullptr;
    const ComponentArray<MeshInstanceComponent>* mi_  = nullptr;
    const ComponentArray<MaterialComponent>*     mat_ = nullptr;
    const ComponentArray<HierarchyComponent>*    hier_ = nullptr;
    const ComponentArray<MotionComponent>*       mot_ = nullptr;
    const GpuScene*                              scene_ = nullptr;
    const ECS *ecs_ = nullptr;
    std::unordered_map<u64, u8>                  warnedLayers_; // (material, dropped blocks): warned once

    // --- State ---------------------------------------------------------------------
    bool built_ = false;
    u64  geometryVersion_ = 0;
    u32  libRegion_ = 1;   // library materials (1 = the default when the library is empty)
    u32  meshCount_ = 0;
    u32  frame_ = 0;
    u32  buildStamp_ = 0;

    std::vector<Ent> ent_;

    std::vector<GPUInstance>      instances_;
    std::vector<GPUTransformNode> nodes_;
    std::vector<GPUMotion>        motions_;
    std::vector<GPUMaterial>      materials_;
    // F15.1 extension buffer: groups of MATERIAL_EXT_RECORDS records; matGroup_[i] =
    // first record of material i's group (NONE: not extended).
    std::vector<GPUMaterialExtRecord> ext_;
    std::vector<u32>              extStamp_, matGroup_, extFree_; // extFree_: free group ids
    // F15.1 (ticket 22): per-slot texture-transform blocks (MATERIAL_TT_RECORDS = two adjacent groups): matTt_[i] = first record of material
    // i's block (NONE: none), ttFree_ = first group id of each free block; ttCount_ = live materials with the transform flag.
    std::vector<u32>              matTt_, ttFree_;
    std::vector<u8>               layerTarget_; // library material i is a layer of some stack (its transform is evaluated on the GPU even when uniform)
    u32 ttCount_ = 0;
    bool vertexColors_ = false; // F15 closure (COLOR_0): the GpuScene has a colour stream
    std::vector<GPUDeltaRecord>   extRecs_;
    std::vector<u32>              extDirty_;
    u32 extHigh_ = 0, extCap_ = 0;            // groups handed out / capacity (groups)
    u32 featureCount_[MATERIAL_FEATURE_COUNT] = {};
    u64 editVersion_ = 0;                     // GpuScene::materialEditVersion mirrored
    bool extCapChanged_ = false;
    std::vector<u32>              slotEntity_;
    std::vector<u32>              instStamp_, nodeStamp_, motStamp_, rootStamp_, matStamp_;

    std::vector<SceneBucket>      buckets_;     // sorted by key
    std::vector<u32>              posIds_;      // position -> bucket id
    std::vector<u32>              idPos_;       // bucket id -> position
    std::unordered_map<u64, u32>  keyToId_;
    std::vector<GPUDrawBucket>    gpuBuckets_;
    std::vector<u32>              commandBuckets_;
    std::array<ClassRange, SCENE_CULL_CLASSES> classRanges_{};

    std::vector<u32> motionSlots_;
    std::vector<u32> motionParents_; // entities: motion roots with live children
    std::vector<u32> csrOffsets_, csrSlots_, csrScratch_, csrSlotsNew_, csrCursor_;
    std::vector<u32> dirtyRoots_;
    std::vector<std::vector<u32>> holes_; // per bucket id: free slots inside [firstSlot, firstSlot + used)
    std::vector<u32> motionParentSlots_, motionParentScratch_;
    bool motionParentsDirty_ = true; // the set changed or one of its slots moved

    std::vector<GPUDeltaRecord> instRecs_, matRecs_, nodeRecs_, motRecs_;
    std::vector<u32> instDirty_, nodeDirty_, motDirty_, matDirty_;
    std::vector<u32> work_, dirtyParents_;

    std::vector<u32> matFree_;
    u32 matHigh_ = 0, matCap_ = 0;
    u32 liveCount_ = 0, emissiveCount_ = 0;
    u32 depthCount_[SCENE_MAX_LEVELS] = {};
    u64 csrRebuilds_ = 0;
    u64 structureVersion_ = 0;

    bool structure_ = false, capacityChanged_ = false, matCapChanged_ = false;
    bool csrDirty_ = false, bucketsDirty_ = false, fullAll_ = false;
    SceneSyncStats stats_;

    // ========================================================================
    // helpers
    // ========================================================================
    void ensureEnt(u32 id) {
        if (id < ent_.size()) return;
        const size_t n = size_t(id) + 1;
        growVector(ent_, n);
        ent_.resize(n);
    }

    void enqueue(u32 e) {
        if (!ent_[e].queued) {
            ent_[e].queued = true;
            work_.push_back(e);
        }
    }

    void enqueueChildren(u32 e) {
        for (u32 c = ent_[e].firstChild; c != NONE; c = ent_[c].nextSib) enqueue(c);
    }

    void markInst(u32 s) { if (instStamp_[s] != frame_) { instStamp_[s] = frame_; instDirty_.push_back(s); } }
    void markNode(u32 s) { if (nodeStamp_[s] != frame_) { nodeStamp_[s] = frame_; nodeDirty_.push_back(s); } }
    void markMot(u32 s)  { if (motStamp_[s]  != frame_) { motStamp_[s]  = frame_; motDirty_.push_back(s); } }
    void markMat(u32 i)  { if (matStamp_[i]  != frame_) { matStamp_[i]  = frame_; matDirty_.push_back(i); } }

    void markDirtyParent(u32 p) {
        if (p == NONE) return;
        if (ent_[p].dpStamp != frame_) {
            ent_[p].dpStamp = frame_;
            dirtyParents_.push_back(p);
        }
    }

    // --- slot space ---------------------------------------------------------------
    void growSlots(u32 newEnd) {
        if (newEnd <= instances_.size()) return;
        growVector(instances_, newEnd);
        growVector(nodes_, newEnd);
        growVector(motions_, newEnd);
        growVector(slotEntity_, newEnd);
        growVector(instStamp_, newEnd);
        growVector(nodeStamp_, newEnd);
        growVector(motStamp_, newEnd);
        growVector(rootStamp_, newEnd);
        instances_.resize(newEnd);
        nodes_.resize(newEnd);
        motions_.resize(newEnd);
        slotEntity_.resize(newEnd, NONE);
        instStamp_.resize(newEnd, 0);
        nodeStamp_.resize(newEnd, 0);
        motStamp_.resize(newEnd, 0);
        rootStamp_.resize(newEnd, 0);
        capacityChanged_ = true;
        csrDirty_        = true;
        structure_       = true;
    }

    [[nodiscard]] u32 slotEnd() const { return static_cast<u32>(instances_.size()); }

    // --- materials ------------------------------------------------------------------
    void growMaterials(u32 newCap) {
        if (newCap <= matCap_) return;
        const size_t n = size_t(libRegion_) + newCap;
        growVector(materials_, n);
        growVector(matStamp_, n);
        growVector(matGroup_, n);
        growVector(matTt_, n);
        materials_.resize(n);
        matStamp_.resize(n, 0);
        matGroup_.resize(n, NONE);
        matTt_.resize(n, NONE);
        matCap_        = newCap;
        matCapChanged_ = true;
        structure_     = true; // buffer re-sized: the host re-sends it (and rebinds)
    }

    void markExt(u32 r) { if (extStamp_[r] != frame_) { extStamp_[r] = frame_; extDirty_.push_back(r); } }

    void growExt(u32 newCapGroups) {
        if (newCapGroups <= extCap_) return;
        const size_t n = size_t(newCapGroups) * MATERIAL_EXT_RECORDS;
        growVector(ext_, n);
        growVector(extStamp_, n);
        ext_.resize(n);
        extStamp_.resize(n, 0);
        extCap_        = newCapGroups;
        extCapChanged_ = true;
        structure_     = true; // buffer re-sized: the host re-sends it
    }

    void setExt(u32 r, const GPUMaterialExtRecord& rec) {
        if (sameBytes(ext_[r], rec)) return;
        ext_[r] = rec;
        markExt(r);
    }

    u32 allocGroup() {
        if (!extFree_.empty()) {
            const u32 g = extFree_.back();
            extFree_.pop_back();
            return g * MATERIAL_EXT_RECORDS;
        }
        if (extHigh_ == extCap_) growExt(std::max<u32>(4u, extCap_ + extCap_ / 2)); // a counted structure change
        return (extHigh_++) * MATERIAL_EXT_RECORDS;
    }

    /// Library materials referenced as a layer by some library description: no primitive bakes a layer's own transform into a mesh.
    static std::vector<u8> layerTargets(const GpuScene& scene) {
        std::vector<u8> out(std::max<size_t>(1u, scene.materials().size()), 0);
        for (const MaterialDescription& d : scene.materialDescriptions())
            if (d.layers)
                for (const MaterialLayer& l : d.layers->layers)
                    if (l.materialIndex < out.size()) out[l.materialIndex] = 1;
        return out;
    }
    /// The transform flags of a material: MATERIAL_FLAG_TEXTURE_TRANSFORM for a PerSlot one (its direct users keep their uvs untouched), and
    /// MATERIAL_FLAG_LAYER_TRANSFORM when it is a layer of a stack (PerSlot or uniform: nothing bakes a layer).  A uniform material that is both
    /// used directly (baked into the uvs of its primitives) and as a layer carries ONLY the layer bit: no double application.
    static u32 ttFlagBits(bool layerTarget, const MaterialDescription& d) {
        const TextureTransformStatus st = analyzeTextureTransforms(d).status;
        const bool perSlot = st == TextureTransformStatus::PerSlot;
        return (perSlot ? MATERIAL_FLAG_TEXTURE_TRANSFORM : 0u) |
               (layerTarget && (perSlot || st == TextureTransformStatus::Uniform) ? MATERIAL_FLAG_LAYER_TRANSFORM : 0u);
    }
    bool ttNeeded(u32 idx, const MaterialDescription& d) const {
        return ttFlagBits(idx < layerTarget_.size() && layerTarget_[idx] != 0, d) != 0u;
    }
    u32 ttBits(u32 idx, const MaterialDescription* d) const {
        return d ? ttFlagBits(idx < layerTarget_.size() && layerTarget_[idx] != 0, *d) : 0u;
    }

    /// A transform block: MATERIAL_TT_GROUPS adjacent groups (the pair is reused through ttFree_, never split).
    u32 allocTtBlock() {
        static_assert(MATERIAL_TT_RECORDS == MATERIAL_TT_GROUPS * MATERIAL_EXT_RECORDS);
        if (!ttFree_.empty()) {
            const u32 g = ttFree_.back();
            ttFree_.pop_back();
            return g * MATERIAL_EXT_RECORDS;
        }
        if (extHigh_ + MATERIAL_TT_GROUPS > extCap_) growExt(std::max<u32>(extHigh_ + MATERIAL_TT_GROUPS, std::max<u32>(4u, extCap_ + extCap_ / 2)));
        const u32 g = extHigh_;
        extHigh_ += MATERIAL_TT_GROUPS;
        return g * MATERIAL_EXT_RECORDS;
    }

    void countFeatures(const GPUMaterial& m, int delta) {
        const u32 bits = materialFeatureBits(m);
        for (u32 f = 0; f < MATERIAL_FEATURE_COUNT; ++f)
            if (bits & (1u << f)) featureCount_[f] = u32(int(featureCount_[f]) + delta);
    }

    void setMaterial(u32 idx, const GPUMaterial& gm) {
        GPUMaterial& cur = materials_[idx];
        if (sameBytes(cur, gm)) return;
        if (isEmissive(cur)) --emissiveCount_;
        if (isEmissive(gm)) ++emissiveCount_;
        countFeatures(cur, -1);
        countFeatures(gm, +1);
        constexpr u32 ttAny = MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM;
        if (cur.flags & ttAny) --ttCount_;
        if (gm.flags & ttAny) ++ttCount_;
        cur = gm;
        markMat(idx);
    }

    /// Install material `idx` from its base record and optional extension data:
    /// feature bits from the description, the extension group allocated when the
    /// material first becomes extended (kept afterwards: toggling a feature
    /// rewrites records, never the group), records packed and diffed.
    void applyMaterial(u32 idx, GPUMaterial gm, const MaterialDescription* desc) {
        // F15.4 (ticket 23): a stack's flags carry the blendable features of its layers next to its own (the layers are library materials).
        const u32 layerUnion = desc && desc->layers ? layerFeatureUnion(*desc->layers, scene_->materialDescriptions()) : 0u;
        const u32 mask = (desc ? featureMask(*desc) : 0u) | layerUnion;
        if (desc && desc->layers) { // the blocks of a layer that cannot blend are dropped, reported once (never silently)
            u32 dropped = 0;
            for (size_t k = 0; k < desc->layers->layers.size() && k < MATERIAL_LAYER_MAX; ++k) {
                const u32 index = desc->layers->layers[k].materialIndex;
                if (index < scene_->materialDescriptions().size()) dropped |= layerUnblendableMask(scene_->materialDescriptions()[index]);
            }
            if (dropped != 0 && warnedLayers_.emplace((u64(idx) << 32) | dropped, 1).second)
                LOG_WARN("SceneStore: layer stack material %u: layer blocks 0x%x (transmission, volume or volumetric subsurface) cannot blend in parameter space and are not applied", idx, dropped);
        }
        u32 group = matGroup_[idx];
        if (group == NONE && mask != 0) {
            group = allocGroup();
            matGroup_[idx] = group;
        }
        if (group != NONE) {
            const MaterialExtGroup recs = packMaterialExtensions(desc ? *desc : MaterialDescription{}, layerUnion);
            for (u32 k = 0; k < MATERIAL_EXT_RECORDS; ++k) setExt(group + k, recs[k]);
        }
        // F15.1 (ticket 22): the transform block of a PerSlot material (allocated on first need, kept, zeroed when the transforms go).
        const bool tt = desc && ttNeeded(idx, *desc);
        u32 block = matTt_[idx];
        if (tt && block == NONE) {
            block = allocTtBlock();
            matTt_[idx] = block;
        }
        if (block != NONE) {
            const TextureTransformBlock recs = tt ? packTextureTransforms(*desc, true) : TextureTransformBlock{};
            for (u32 k = 0; k < MATERIAL_TT_RECORDS; ++k) setExt(block + k, recs[k]);
        }
        gm.flags = (gm.flags & ~(MATERIAL_FEATURES_ALL << MATERIAL_FEATURE_SHIFT)) | (mask << MATERIAL_FEATURE_SHIFT);
        gm.flags = (gm.flags & ~(MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM)) | ttBits(idx, desc);
        gm.textureTransformIndex = tt ? block : 0u;
        gm.extensionIndex = group == NONE ? MATERIAL_EXTENSION_INVALID : group;
        setMaterial(idx, gm);
    }

    /// Library material `i` from the scene (base record + description, if any).
    void applyLibraryMaterial(u32 i) {
        const auto& lib = scene_->materials();
        const auto& desc = scene_->materialDescriptions();
        if (lib.empty()) {
            applyMaterial(0, toGPUMaterial(MaterialComponent{}), nullptr);
            return;
        }
        applyMaterial(i, lib[i], i < desc.size() ? &desc[i] : nullptr);
    }

    u32 allocMaterial() {
        if (!matFree_.empty()) {
            const u32 m = matFree_.back();
            matFree_.pop_back();
            return m;
        }
        if (matHigh_ == matCap_) growMaterials(std::max<u32>(64u, matCap_ + matCap_ / 2));
        return matHigh_++;
    }

    void freeMaterial(u32 e) {
        Ent& en = ent_[e];
        if (en.matSlot == NONE) return;
        const u32 idx = libRegion_ + en.matSlot;
        if (matGroup_[idx] != NONE) { // the group goes back to the pool, zeroed (verify checks it)
            const u32 g = matGroup_[idx];
            for (u32 k = 0; k < MATERIAL_EXT_RECORDS; ++k) setExt(g + k, GPUMaterialExtRecord{});
            extFree_.push_back(g / MATERIAL_EXT_RECORDS);
            matGroup_[idx] = NONE;
        }
        if (matTt_[idx] != NONE) {
            const u32 g = matTt_[idx];
            for (u32 k = 0; k < MATERIAL_TT_RECORDS; ++k) setExt(g + k, GPUMaterialExtRecord{});
            ttFree_.push_back(g / MATERIAL_EXT_RECORDS);
            matTt_[idx] = NONE;
        }
        setMaterial(idx, GPUMaterial{});
        matFree_.push_back(en.matSlot);
        en.matSlot = NONE;
    }

    // --- hierarchy links (entity space) ---------------------------------------------
    void link(u32 e, u32 p) {
        Ent& en = ent_[e];
        Ent& pe = ent_[p];
        en.parent = p;
        en.prevSib = NONE;
        en.nextSib = pe.firstChild;
        if (pe.firstChild != NONE) ent_[pe.firstChild].prevSib = e;
        pe.firstChild = e;
    }

    void unlink(u32 e) {
        Ent& en = ent_[e];
        if (en.parent == NONE) return;
        if (en.prevSib != NONE) ent_[en.prevSib].nextSib = en.nextSib;
        else ent_[en.parent].firstChild = en.nextSib;
        if (en.nextSib != NONE) ent_[en.nextSib].prevSib = en.prevSib;
        en.parent = en.prevSib = en.nextSib = NONE;
    }

    void syncLink(u32 e) {
        const HierarchyComponent* h = hier_->tryGet(e);
        u32 wp = h ? h->parent : NONE;
        if (wp != NONE && (wp == e || wp >= MAX_ENTITY_ID)) wp = NONE;
        if (wp == ent_[e].parent) return;
        if (wp != NONE) ensureEnt(wp);
        if (ent_[e].parent != NONE) {
            const u32 old = ent_[e].parent;
            if (ent_[e].slot != NONE) adjustLive(old, -1);
            unlink(e);
        }
        if (wp != NONE) {
            link(e, wp);
            if (ent_[e].slot != NONE) adjustLive(wp, +1);
        }
    }

    // --- motion lists ---------------------------------------------------------------
    void removeMotionEntry(u32 e) {
        Ent& en = ent_[e];
        if (en.motionPos == NONE) return;
        const u32 pos = en.motionPos;
        const u32 lastSlot = motionSlots_.back();
        if (pos + 1 != motionSlots_.size()) {
            motionSlots_[pos] = lastSlot;
            ent_[slotEntity_[lastSlot]].motionPos = pos;
        }
        motionSlots_.pop_back();
        en.motionPos = NONE;
        structure_ = true;
    }

    void updateMotionParent(u32 e) {
        Ent& en = ent_[e];
        const bool want = en.slot != NONE && en.d.motion && en.liveChildren > 0;
        if (want && en.motionParentPos == NONE) {
            en.motionParentPos = static_cast<u32>(motionParents_.size());
            motionParents_.push_back(e);
            motionParentsDirty_ = true;
        } else if (!want && en.motionParentPos != NONE) {
            const u32 pos = en.motionParentPos;
            const u32 last = motionParents_.back();
            motionParents_[pos] = last;
            ent_[last].motionParentPos = pos;
            motionParents_.pop_back();
            motionParentsDirty_ = true;
            en.motionParentPos = NONE; // (if last == e this reset comes after the fix-up)
        }
    }

    void syncLists(u32 e) {
        Ent& en = ent_[e];
        const bool wantMotion = en.slot != NONE && en.d.motion;
        if (wantMotion && en.motionPos == NONE) {
            en.motionPos = static_cast<u32>(motionSlots_.size());
            motionSlots_.push_back(en.slot);
            structure_ = true;
        } else if (!wantMotion) {
            removeMotionEntry(e);
        }
        updateMotionParent(e);
    }

    void adjustLive(u32 p, int delta) {
        ent_[p].liveChildren = static_cast<u32>(static_cast<int>(ent_[p].liveChildren) + delta);
        updateMotionParent(p);
    }

    // --- decision ---------------------------------------------------------------------
    [[nodiscard]] Decision decide(u32 e) const {
        Decision d;
        const MeshInstanceComponent* mi = mi_->tryGet(e);
        const TransformComponent*    xf = xf_->tryGet(e);
        if (!mi || !xf || !mi->isVisible() || mi->meshHandle >= meshCount_) return d;

        const HierarchyComponent* h = hier_->tryGet(e);
        bool neg = false;
        u32  depth = 0;
        if (h) {
            const u32 p = h->parent;
            if (p == NONE || p == e || p >= MAX_ENTITY_ID || ent_[e].parent != p) {
                LOG_WARN("SceneStore: entity %u has an invalid hierarchy parent %u, skipped", e, p);
                return d;
            }
            const Ent& pe = ent_[p];
            if (!pe.d.want) {
                if (!mi_->has(p)) {
                    LOG_WARN("SceneStore: parent %u of entity %u is not an instance, skipped", p, e);
                }
                return d;
            }
            depth = pe.d.depth + 1u;
            if (depth >= SCENE_MAX_LEVELS) {
                LOG_ERROR("SceneStore: entity %u at hierarchy depth %u exceeds the maximum %u, skipped",
                          e, depth, SCENE_MAX_LEVELS - 1);
                return d;
            }
            neg = pe.d.neg != negDet(xf->worldMatrix); // worldMatrix holds the local matrix of a child
        } else {
            neg = negDet(xf->worldMatrix);
        }

        bool doubleSided;
        if (const MaterialComponent* mc = mat_->tryGet(e)) {
            doubleSided = mc->doubleSided;
        } else {
            const u32 idx = mi->materialIndex < libRegion_ ? mi->materialIndex : 0u;
            doubleSided = (materials_[idx].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0;
        }

        d.want   = true;
        d.neg    = neg;
        d.depth  = static_cast<u8>(depth);
        d.motion = !h && mot_->has(e);
        const CullClass cls = doubleSided ? CullClass::None : (neg ? CullClass::BackMirrored : CullClass::Back);
        d.key = makeKey(cls, mi->meshHandle);
        return d;
    }

    void decideRec(u32 e) {
        if (ent_[e].decStamp == buildStamp_) return;
        ent_[e].decStamp = buildStamp_;
        const u32 p = ent_[e].parent;
        if (p != NONE && mi_->has(p)) decideRec(p);
        ent_[e].d = decide(e);
    }

    // --- buckets ------------------------------------------------------------------------
    [[nodiscard]] static u64 keyOf(const SceneBucket& b) { return makeKey(b.cull, b.mesh); }

    u32 createBucket(u64 key, u32 capacity) {
        SceneBucket b;
        b.mesh      = static_cast<u32>(key & 0xFFFFFFFFu);
        b.cull      = static_cast<CullClass>(key >> 32);
        b.capacity  = capacity;
        b.firstSlot = slotEnd();
        growSlots(b.firstSlot + capacity);
        const auto it = std::lower_bound(buckets_.begin(), buckets_.end(), key,
                                         [](const SceneBucket& x, u64 k) { return keyOf(x) < k; });
        const u32 pos = static_cast<u32>(it - buckets_.begin());
        const u32 id  = static_cast<u32>(idPos_.size());
        buckets_.insert(it, b);
        posIds_.insert(posIds_.begin() + pos, id);
        idPos_.push_back(pos);
        holes_.emplace_back();
        for (u32 p = pos; p < posIds_.size(); ++p) idPos_[posIds_[p]] = p;
        keyToId_[key] = id;
        bucketsDirty_ = true;
        structure_    = true;
        return id;
    }

    // An entity's slot changed (swap-and-pop or relocation); the mirror
    // content is already in place.
    void afterMoved(u32 e, u32 newSlot) {
        Ent& en = ent_[e];
        en.slot = newSlot;
        if (en.motionParentPos != NONE) motionParentsDirty_ = true;
        if (en.motionPos != NONE) {
            motionSlots_[en.motionPos] = newSlot;
            structure_ = true;
        }
        if (en.d.depth > 0 || en.liveChildren > 0) csrDirty_ = true;
        if (en.d.depth > 0) markDirtyParent(en.parent); // its matrix is a placeholder in the new slot
        if (en.firstChild != NONE) enqueueChildren(e);  // their parentSlot changes
    }

    void relocate(u32 pos) {
        SceneBucket& b0 = buckets_[pos];
        // Only a full bucket without holes relocates: used == count here.
        const u32 oldFirst = b0.firstSlot, count = b0.used, newCap = std::max(b0.capacity * 2, MIN_BUCKET);
        const u32 newFirst = slotEnd();
        growSlots(newFirst + newCap); // may reallocate nothing referenced below (b0 is in buckets_)
        SceneBucket& b = buckets_[pos];
        for (u32 i = 0; i < count; ++i) {
            const u32 from = oldFirst + i, to = newFirst + i;
            instances_[to]  = instances_[from];
            nodes_[to]      = nodes_[from];
            motions_[to]    = motions_[from];
            slotEntity_[to] = slotEntity_[from];
            instances_[from] = GPUInstance{};
            nodes_[from]     = GPUTransformNode{};
            motions_[from]   = GPUMotion{};
            slotEntity_[from] = NONE;
        }
        b.firstSlot = newFirst;
        b.capacity  = newCap;
        for (u32 i = 0; i < count; ++i) afterMoved(slotEntity_[newFirst + i], newFirst + i);
        bucketsDirty_ = true;
        structure_    = true;
        LOG_DEBUG("SceneStore: bucket (mesh %u, class %u) relocated to slot %u, capacity %u",
                  b.mesh, static_cast<u32>(b.cull), newFirst, newCap);
    }

    void addToBucket(u32 e) {
        const u64 key = ent_[e].d.key;
        auto it = keyToId_.find(key);
        const u32 id = it != keyToId_.end() ? it->second : createBucket(key, MIN_BUCKET);
        u32 pos = idPos_[id];
        std::vector<u32>& holes = holes_[id];
        u32 slot;
        if (!holes.empty()) {
            slot = holes.back();
            holes.pop_back();
            ++buckets_[pos].count;
        } else {
            if (buckets_[pos].used == buckets_[pos].capacity) {
                relocate(pos);
                pos = idPos_[id];
            }
            SceneBucket& b = buckets_[pos];
            slot = b.firstSlot + b.used++;
            ++b.count;
        }
        slotEntity_[slot] = e;
        Ent& en = ent_[e];
        en.slot   = slot;
        en.bucket = id;
        if (en.d.depth > 0 || en.liveChildren > 0) csrDirty_ = true;
    }

    // Place a decided-wanted, unplaced entity.
    void place(u32 e) {
        addToBucket(e);
        Ent& en = ent_[e];
        ++liveCount_;
        ++depthCount_[en.d.depth];
        if (en.parent != NONE) adjustLive(en.parent, +1);
    }

    // Remove a placed entity from the store (uses its current decision).
    void unplace(u32 e) {
        removeMotionEntry(e);
        {
            Ent& en = ent_[e];
            if (en.motionParentPos != NONE) {
                const u32 pos = en.motionParentPos;
                const u32 last = motionParents_.back();
                motionParents_[pos] = last;
                ent_[last].motionParentPos = pos;
                motionParents_.pop_back();
                motionParentsDirty_ = true;
                en.motionParentPos = NONE;
            }
        }
        freeMaterial(e);
        Ent& en = ent_[e];
        if (en.parent != NONE) adjustLive(en.parent, -1);
        --depthCount_[en.d.depth];
        --liveCount_;
        if (en.d.depth > 0 || en.liveChildren > 0) csrDirty_ = true;

        // No other instance moves: the slot becomes a hole of its bucket
        // (zeroed: never valid, never culled in, degenerate if drawn).
        SceneBucket& b = buckets_[idPos_[en.bucket]];
        const u32 hole = en.slot;
        instances_[hole]  = GPUInstance{};
        nodes_[hole]      = GPUTransformNode{};
        motions_[hole]    = GPUMotion{};
        slotEntity_[hole] = NONE;
        markInst(hole);
        markNode(hole);
        markMot(hole);
        if (hole + 1 == b.firstSlot + b.used) {
            --b.used; // the last used slot: no hole to remember
        } else {
            holes_[en.bucket].push_back(hole);
        }
        --b.count;
        en.slot   = NONE;
        en.bucket = NONE;
    }

    // --- records ------------------------------------------------------------------------
    void setInstance(u32 e, const GPUInstance& gi) {
        Ent& en = ent_[e];
        GPUInstance& cur = instances_[en.slot];
        if (sameBytes(cur, gi)) return;
        const bool matrixChanged = std::memcmp(cur.modelMatrix, gi.modelMatrix, sizeof(gi.modelMatrix)) != 0;
        cur = gi;
        markInst(en.slot);
        if (en.d.depth > 0) markDirtyParent(en.parent);      // placeholder overwrote the GPU world matrix
        else if (matrixChanged && en.liveChildren > 0) markDirtyParent(e); // root world changed
    }

    void writeRecords(u32 e) {
        Ent& en = ent_[e];
        const MeshInstanceComponent& mi = mi_->get(e);
        const TransformComponent&    xf = xf_->get(e);

        u32 matIdx;
        if (const MaterialComponent* mc = mat_->tryGet(e)) {
            if (en.matSlot == NONE) en.matSlot = allocMaterial();
            matIdx = libRegion_ + en.matSlot;
            applyMaterial(matIdx, toGPUMaterial(*mc), mc->extension.get()); // (may grow materials_; no Ent reference is invalidated)
        } else {
            freeMaterial(e);
            matIdx = mi.materialIndex < libRegion_ ? mi.materialIndex : 0u;
        }

        GPUInstance gi{};
        if (en.d.depth == 0 && !en.d.motion) std::memcpy(gi.modelMatrix, &xf.worldMatrix[0][0], sizeof(gi.modelMatrix));
        else setIdentity(gi.modelMatrix);
        gi.generation = ecs_->incarnation(e);
        gi.meshIndex     = mi.meshHandle;
        gi.materialIndex = matIdx;
        gi.flags         = mi.flags | (en.d.neg ? INSTANCE_FLAG_MIRRORED : 0u) | INSTANCE_FLAG_VALID;
        setInstance(e, gi);

        GPUTransformNode nn{};
        if (en.d.depth > 0) nn = makeNode(xf, ent_[en.parent].slot, en.d.depth);
        {
            GPUTransformNode& cur = nodes_[en.slot];
            if (!sameBytes(cur, nn)) {
                // A new parent or a root <-> child change alters the CSR (a local-only edit does not).
                if (cur.parentSlot != nn.parentSlot || (cur.depth == 0) != (nn.depth == 0)) csrDirty_ = true;
                cur = nn;
                markNode(en.slot);
                if (en.d.depth > 0) markDirtyParent(en.parent);
            }
        }

        GPUMotion mm{};
        if (en.d.motion) mm = makeMotion(mot_->get(e), xf);
        {
            GPUMotion& cur = motions_[en.slot];
            if (!sameBytes(cur, mm)) {
                cur = mm;
                markMot(en.slot);
            }
        }
    }

    // --- reconcile one entity with the ECS -------------------------------------------------
    void reconcile(u32 e) {
        syncLink(e);
        const Decision nd = decide(e);
        const Decision od = ent_[e].d;
        const u32 oldSlot = ent_[e].slot;

        if (ent_[e].slot != NONE && (!nd.want || nd.key != od.key)) unplace(e);
        ent_[e].d = nd;
        if (ent_[e].slot == NONE) {
            if (nd.want) place(e);
        } else if (od.depth != nd.depth) {
            --depthCount_[od.depth];
            ++depthCount_[nd.depth];
            csrDirty_ = true;
        }

        Ent& en = ent_[e];
        const bool kidsAffected = od.want != nd.want || od.neg != nd.neg || od.depth != nd.depth || oldSlot != en.slot;
        if (kidsAffected && en.firstChild != NONE) enqueueChildren(e);
        if (en.slot != NONE) writeRecords(e);
        syncLists(e);
    }

    // --- full build -------------------------------------------------------------------------
    void resetState() {
        ent_.clear();
        instances_.clear(); nodes_.clear(); motions_.clear(); materials_.clear(); slotEntity_.clear();
        instStamp_.clear(); nodeStamp_.clear(); motStamp_.clear(); rootStamp_.clear(); matStamp_.clear();
        buckets_.clear(); posIds_.clear(); idPos_.clear(); keyToId_.clear(); holes_.clear();
        gpuBuckets_.clear(); commandBuckets_.clear(); classRanges_ = {};
        motionSlots_.clear(); motionParents_.clear(); motionParentSlots_.clear();
        motionParentsDirty_ = true;
        csrOffsets_.clear(); csrSlots_.clear();
        dirtyRoots_.clear();
        instRecs_.clear(); matRecs_.clear(); nodeRecs_.clear(); motRecs_.clear();
        instDirty_.clear(); nodeDirty_.clear(); motDirty_.clear(); matDirty_.clear();
        ext_.clear(); extStamp_.clear(); matGroup_.clear(); extFree_.clear(); matTt_.clear(); ttFree_.clear(); ttCount_ = 0; extRecs_.clear(); extDirty_.clear();
        extHigh_ = extCap_ = 0;
        std::memset(featureCount_, 0, sizeof(featureCount_));
        work_.clear(); dirtyParents_.clear();
        matFree_.clear();
        matHigh_ = matCap_ = 0;
        liveCount_ = emissiveCount_ = 0;
        std::memset(depthCount_, 0, sizeof(depthCount_));
        built_ = false;
    }

    void build() {
        resetState();
        const auto& lib = scene_->materials();
        libRegion_ = std::max<u32>(1u, static_cast<u32>(lib.size()));
        meshCount_ = scene_->getMeshCount();
        geometryVersion_ = scene_->geometryVersion();
        editVersion_ = scene_->materialEditVersion();
        // Size the extension buffer for the extended materials of the scene (library and
        // entities) so the first sync is not a growth event; none: nothing allocated.
        u32 extended = 0;
        layerTarget_ = layerTargets(*scene_);
        {
            const auto& desc = scene_->materialDescriptions();
            for (u32 i = 0; i < lib.size() && i < desc.size(); ++i)
                extended += (featureMask(desc[i]) != 0 || (desc[i].layers && layerFeatureUnion(*desc[i].layers, desc) != 0)) ? 1u : 0u;
            for (EntityID e : mat_->entities())
                if (const MaterialComponent* mc = mat_->tryGet(e);
                    mc && mc->extension && (featureMask(*mc->extension) != 0 || (mc->extension->layers && layerFeatureUnion(*mc->extension->layers, desc) != 0)))
                    ++extended;
            for (u32 i = 0; i < lib.size() && i < desc.size(); ++i) extended += ttNeeded(i, desc[i]) ? MATERIAL_TT_GROUPS : 0u;
            for (EntityID e : mat_->entities())
                if (const MaterialComponent* mc = mat_->tryGet(e); mc && mc->extension && needsTextureTransformBlock(*mc->extension))
                    extended += MATERIAL_TT_GROUPS;
        }
        materials_.assign(lib.begin(), lib.end());
        if (materials_.empty()) materials_.push_back(toGPUMaterial(MaterialComponent{}));
        matStamp_.assign(materials_.size(), 0);
        matGroup_.assign(materials_.size(), NONE);
        matTt_.assign(materials_.size(), NONE);
        if (extended > 0) growExt(extended + extended / 4 + 1);
        // The store owns the feature bits and the extension index of the library records.
        for (u32 i = 0; i < materials_.size(); ++i) {
            GPUMaterial base = materials_[i];
            base.flags &= ~((MATERIAL_FEATURES_ALL << MATERIAL_FEATURE_SHIFT) | MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM);
            base.extensionIndex = MATERIAL_EXTENSION_INVALID;
            base.textureTransformIndex = 0;
            materials_[i] = base;
        }
        for (const GPUMaterial& m : materials_) emissiveCount_ += isEmissive(m) ? 1u : 0u;
        for (u32 i = 0; i < materials_.size(); ++i) applyLibraryMaterial(i);

        u32 maxId = 0;
        bool any = false;
        auto scan = [&](const auto& arr) {
            for (EntityID e : arr.entities()) { maxId = std::max(maxId, e); any = true; }
        };
        scan(*mi_); scan(*xf_); scan(*mat_); scan(*hier_); scan(*mot_);
        for (EntityID e : hier_->entities()) {
            const u32 p = hier_->get(e).parent;
            if (p != NONE && p < MAX_ENTITY_ID) maxId = std::max(maxId, p);
        }
        if (any) ensureEnt(maxId);

        for (EntityID e : hier_->entities()) syncLink(e);

        // Phase A: decisions, parents first.
        ++buildStamp_;
        for (EntityID e : mi_->entities()) decideRec(e);

        // Phase B: buckets (sorted by (class, mesh)), then slots in dense MeshInstance order.
        std::unordered_map<u64, u32> counts;
        u32 withMaterial = 0;
        for (EntityID e : mi_->entities()) {
            if (!ent_[e].d.want) continue;
            ++counts[ent_[e].d.key];
            if (mat_->has(e)) ++withMaterial;
        }
        std::vector<std::pair<u64, u32>> keys(counts.begin(), counts.end());
        std::sort(keys.begin(), keys.end());
        for (const auto& [key, n] : keys) createBucket(key, initialCapacity(n));
        growMaterials(withMaterial + withMaterial / 4);
        for (EntityID e : mi_->entities()) {
            if (ent_[e].d.want) place(e);
        }

        // Phase C: records and lists (every slot is known now).
        for (EntityID e : mi_->entities()) {
            if (ent_[e].slot == NONE) continue;
            writeRecords(e);
            syncLists(e);
        }
        built_   = true;
        fullAll_ = true;
        structure_ = true;
        bucketsDirty_ = true; // also for an empty scene: the class sentinels
        csrDirty_ = true;
    }

    // --- incremental ---------------------------------------------------------------------------
    void gather() {
        auto addAll = [&](const auto& arr) {
            const ComponentChanges ch = arr.changes();
            if (ch.all) {
                for (EntityID e : arr.entities()) { ensureEnt(e); enqueue(e); }
            }
            for (EntityID e : ch.added)   { ensureEnt(e); enqueue(e); }
            for (EntityID e : ch.removed) { ensureEnt(e); enqueue(e); }
            for (EntityID e : ch.changed) { ensureEnt(e); enqueue(e); }
        };
        addAll(*mi_);
        addAll(*hier_);
        addAll(*mot_);
        addAll(*xf_);
        addAll(*mat_);
    }

    void processWork() {
        for (size_t i = 0; i < work_.size(); ++i) {
            const u32 e = work_[i];
            ent_[e].queued = false;
            reconcile(e);
        }
        work_.clear();
    }

    // --- end of sync ---------------------------------------------------------------------------
    void rebuildBucketTables() {
        const auto& infos = scene_->meshInfos();
        gpuBuckets_.assign(buckets_.size(), GPUDrawBucket{});
        commandBuckets_.clear();
        u32 cmd = 0;
        u32 b = 0;
        for (u32 c = 0; c < SCENE_CULL_CLASSES; ++c) {
            classRanges_[c].firstCommand = cmd;
            for (; b < buckets_.size() && static_cast<u32>(buckets_[b].cull) == c; ++b) {
                SceneBucket& sb = buckets_[b];
                sb.command = cmd++;
                commandBuckets_.push_back(b);
                GPUDrawBucket& g = gpuBuckets_[b];
                g.firstSlot = sb.firstSlot;
                g.capacity  = sb.capacity;
                g.meshIndex = sb.mesh;
                g.cullClass = static_cast<u32>(sb.cull);
                g.command   = sb.command;
                if (sb.mesh < infos.size()) {
                    g.indexCount   = infos[sb.mesh].indexCount;
                    g.indexOffset  = infos[sb.mesh].indexOffset;
                    g.vertexOffset = infos[sb.mesh].vertexOffset;
                }
            }
            commandBuckets_.push_back(NONE); // sentinel
            ++cmd;
            classRanges_[c].commandCount = cmd - classRanges_[c].firstCommand;
        }
    }

    void rebuildCsr() {
        const u32 cap = slotEnd();
        csrScratch_.assign(size_t(cap) + 1, 0);
        for (u32 s = 0; s < cap; ++s) {
            if (slotEntity_[s] != NONE && nodes_[s].depth > 0) ++csrScratch_[nodes_[s].parentSlot + 1];
        }
        for (u32 s = 0; s < cap; ++s) csrScratch_[s + 1] += csrScratch_[s];
        csrCursor_.assign(csrScratch_.begin(), csrScratch_.end() - 1);
        std::vector<u32>& slots = csrSlotsNew_;
        slots.resize(csrScratch_[cap]);
        for (u32 s = 0; s < cap; ++s) {
            if (slotEntity_[s] != NONE && nodes_[s].depth > 0) slots[csrCursor_[nodes_[s].parentSlot]++] = s;
        }
        ++csrRebuilds_;
        stats_.csrRebuilt = true;
        if (csrScratch_ != csrOffsets_ || slots != csrSlots_) {
            csrOffsets_.swap(csrScratch_);
            csrSlots_.swap(slots);
            structure_ = true;
        }
    }

    template <typename Buf>
    void deltas(bool capChanged, const std::vector<u32>& dirty, const Buf& buf, std::vector<GPUDeltaRecord>& recs,
                bool& full, u32& count) {
        recs.clear();
        full = fullAll_ || capChanged || u64(dirty.size()) * 8 > buf.size();
        if (full) return;
        for (u32 s : dirty) {
            GPUDeltaRecord r{};
            r.slot = s;
            static_assert(sizeof(typename Buf::value_type) == sizeof(r.payload), "record payload size");
            std::memcpy(r.payload, &buf[s], sizeof(r.payload));
            recs.push_back(r);
        }
        count = static_cast<u32>(recs.size());
    }

    void finish() {
        if (bucketsDirty_) rebuildBucketTables();
        if (csrDirty_) rebuildCsr();

        stats_.structure = structure_ || fullAll_;
        deltas(capacityChanged_, instDirty_, instances_, instRecs_, stats_.fullInstances, stats_.instanceRecords);
        deltas(capacityChanged_, nodeDirty_, nodes_, nodeRecs_, stats_.fullNodes, stats_.nodeRecords);
        deltas(capacityChanged_, motDirty_, motions_, motRecs_, stats_.fullMotions, stats_.motionRecords);
        deltas(matCapChanged_, matDirty_, materials_, matRecs_, stats_.fullMaterials, stats_.materialRecords);
        deltas(extCapChanged_, extDirty_, ext_, extRecs_, stats_.fullExtensions, stats_.extensionRecords);

        // Dirty roots (GPU queue 0): parents whose descendants must be recomputed.
        dirtyRoots_.clear();
        auto addRoot = [&](u32 slot) {
            if (rootStamp_[slot] != frame_) {
                rootStamp_[slot] = frame_;
                dirtyRoots_.push_back(slot);
            }
        };
        // Motion roots with children are expanded every frame from their own
        // persistent GPU queue (motionParentSlots): never listed here, so a
        // static CPU scene writes no queue 0 at all.
        const auto isMotionParent = [&](u32 e) { return ent_[e].motionParentPos != NONE; };
        if (stats_.fullInstances) {
            // The full copy overwrote every child's world matrix with its placeholder.
            for (u32 s = 0; s < slotEnd(); ++s) {
                const u32 e = slotEntity_[s];
                if (e != NONE && ent_[e].d.depth == 0 && ent_[e].liveChildren > 0 && !isMotionParent(e)) addRoot(s);
            }
        }
        for (u32 p : dirtyParents_) {
            if (ent_[p].slot != NONE && !isMotionParent(p)) addRoot(ent_[p].slot);
        }
        // Rebuilt only when the set changed or one of its slots moved
        // (swap-and-pop, relocation), then compared: O(1) in a steady frame.
        stats_.motionParentsChanged = false;
        if (motionParentsDirty_) {
            motionParentsDirty_ = false;
            motionParentScratch_.clear();
            for (u32 e : motionParents_) motionParentScratch_.push_back(ent_[e].slot);
            stats_.motionParentsChanged = motionParentScratch_ != motionParentSlots_;
            if (stats_.motionParentsChanged) motionParentSlots_.swap(motionParentScratch_);
        }

        if (stats_.structure) ++structureVersion_;

        stats_.instances = liveCount_;
        stats_.slots     = slotEnd();
        stats_.buckets   = static_cast<u32>(buckets_.size());
        stats_.materials = static_cast<u32>(materials_.size());
        stats_.dirtyRoots = static_cast<u32>(dirtyRoots_.size());
        u64 bytes = u64(stats_.instanceRecords + stats_.materialRecords + stats_.extensionRecords + stats_.nodeRecords +
                        stats_.motionRecords) * sizeof(GPUDeltaRecord);
        if (stats_.fullInstances) bytes += u64(instances_.size()) * sizeof(GPUInstance);
        if (stats_.fullNodes)     bytes += u64(nodes_.size()) * sizeof(GPUTransformNode);
        if (stats_.fullMotions)   bytes += u64(motions_.size()) * sizeof(GPUMotion);
        if (stats_.motionParentsChanged) bytes += 32 + u64(motionParentSlots_.size()) * sizeof(u32);
        if (stats_.fullMaterials) bytes += u64(materials_.size()) * sizeof(GPUMaterial);
        if (stats_.fullExtensions) bytes += u64(ext_.size()) * sizeof(GPUMaterialExtRecord);
        if (stats_.structure) {
            bytes += u64(gpuBuckets_.size()) * sizeof(GPUDrawBucket) + u64(commandBuckets_.size()) * sizeof(u32) +
                     u64(csrOffsets_.size() + csrSlots_.size() + motionSlots_.size()) * sizeof(u32);
        }
        stats_.uploadBytes = bytes;
    }

    void sync(ECS& ecs, const GpuScene& scene) {
        ecs_ = &ecs;
        vertexColors_ = scene.hasVertexColors();
        const ECS& c = ecs; // const access: reading must not mark changes
        xf_   = &c.getArray<TransformComponent>();
        mi_   = &c.getArray<MeshInstanceComponent>();
        mat_  = &c.getArray<MaterialComponent>();
        hier_ = &c.getArray<HierarchyComponent>();
        mot_  = &c.getArray<MotionComponent>();
        scene_ = &scene;

        ++frame_;
        instRecs_.clear(); matRecs_.clear(); nodeRecs_.clear(); motRecs_.clear();
        instDirty_.clear(); nodeDirty_.clear(); motDirty_.clear(); matDirty_.clear();
        extRecs_.clear(); extDirty_.clear();
        dirtyParents_.clear();
        stats_ = SceneSyncStats{};
        structure_ = capacityChanged_ = matCapChanged_ = extCapChanged_ = csrDirty_ = bucketsDirty_ = fullAll_ = false;

        bool rebuild = !built_ || scene.geometryVersion() != geometryVersion_ ||
                       std::max<u32>(1u, static_cast<u32>(scene.materials().size())) != libRegion_;
        // A library edit that flips double-sided changes the cull class (bucket) of its
        // instances: rebuild (rare, an authoring-time edit).
        if (!rebuild && scene.materialEditVersion() != editVersion_) {
            const auto& lib = scene.materials();
            for (u32 i = 0; i < lib.size() && i < materials_.size(); ++i)
                if ((lib[i].flags ^ materials_[i].flags) & MATERIAL_FLAG_DOUBLE_SIDED) { rebuild = true; break; }
        }
        if (rebuild) {
            build();
        } else {
            // A runtime edit of a library material (GpuScene::setMaterialDescription): re-derive
            // the library records; unchanged ones diff to nothing (O(library), only on an edit).
            if (scene.materialEditVersion() != editVersion_) {
                editVersion_ = scene.materialEditVersion();
                layerTarget_ = layerTargets(*scene_);
                for (u32 i = 0; i < libRegion_; ++i) applyLibraryMaterial(i);
                // F15.4: a stack's flags and record 3 carry the features of its layers, which live in the library: re-derive the entity stacks.
                for (EntityID e : mat_->entities()) {
                    const MaterialComponent* mc = mat_->tryGet(e);
                    if (mc && mc->extension && mc->extension->layers && e < ent_.size() && ent_[e].matSlot != NONE)
                        applyMaterial(libRegion_ + ent_[e].matSlot, toGPUMaterial(*mc), mc->extension.get());
                }
            }
            gather();
            processWork();
        }
        finish();
        scene_ = nullptr;
    }

    // --- self check -----------------------------------------------------------------------------
    std::string verify(const ECS& ecs, const GpuScene& scene) const;
};

// ===========================================================================
// verifyAgainstEcs: expected state from the ECS alone, compared with the mirror
// ===========================================================================
std::string SceneStore::Impl::verify(const ECS& ecs, const GpuScene& scene) const {
    if (!built_) return "store was never synced";
    const auto& xf   = ecs.getArray<TransformComponent>();
    const auto& mi   = ecs.getArray<MeshInstanceComponent>();
    const auto& mat  = ecs.getArray<MaterialComponent>();
    const auto& hier = ecs.getArray<HierarchyComponent>();
    const auto& mot  = ecs.getArray<MotionComponent>();

    // Library.
    const auto& lib = scene.materials();
    std::vector<GPUMaterial> expLib(lib.begin(), lib.end());
    if (expLib.empty()) expLib.push_back(toGPUMaterial(MaterialComponent{}));
    const u32 libRegion = static_cast<u32>(expLib.size());
    if (libRegion != libRegion_) return fmt("library size %u != %u", libRegion, libRegion_);
    if (materials_.size() < libRegion) return "materials_ smaller than the library";
    // F15.1: extension groups in use (owner material -> checked below; unowned groups are zero).
    std::vector<u8> groupOwned(extHigh_, 0);
    const std::vector<u8> layerTargetsExp = layerTargets(scene); // F15.1 (ticket 22): layer sub-materials carry their transform block even when uniform
    // Expected record of a material: base data + feature bits of its description; the
    // extension index is allocation history, so it is taken from the mirror after
    // validating it (aligned, inside the used range, owned once, records equal the packing).
    const auto checkMaterial = [&](u32 idx, GPUMaterial exp, const MaterialDescription* desc) -> std::string {
        const u32 layerUnion = desc && desc->layers ? layerFeatureUnion(*desc->layers, scene.materialDescriptions()) : 0u;
        const u32 mask = (desc ? featureMask(*desc) : 0u) | layerUnion;
        exp.flags = (exp.flags & ~(MATERIAL_FEATURES_ALL << MATERIAL_FEATURE_SHIFT)) | (mask << MATERIAL_FEATURE_SHIFT);
        // F15.1 (ticket 22): the transform block (allocation history again: validated, then taken from the mirror).
        const bool tt = desc && needsTextureTransformBlock(*desc, idx < layerTargetsExp.size() && layerTargetsExp[idx] != 0);
        {
            const u32 blk = matTt_[idx];
            if (blk != NONE) {
                if (blk % MATERIAL_EXT_RECORDS != 0 || blk / MATERIAL_EXT_RECORDS + MATERIAL_TT_GROUPS > extHigh_)
                    return fmt("material %u: bad transform block %u", idx, blk);
                const u32 g = blk / MATERIAL_EXT_RECORDS;
                for (u32 k = 0; k < MATERIAL_TT_GROUPS; ++k) {
                    if (groupOwned[g + k]) return fmt("material %u: transform block %u shared", idx, g);
                    groupOwned[g + k] = 1;
                }
                const TextureTransformBlock want = tt ? packTextureTransforms(*desc, true) : TextureTransformBlock{};
                for (u32 k = 0; k < MATERIAL_TT_RECORDS; ++k)
                    if (!sameBytes(ext_[blk + k], want[k])) return fmt("material %u: transform record %u differs", idx, k);
            } else if (tt) {
                return fmt("material %u: transform needed without a block", idx);
            }
            exp.flags = (exp.flags & ~(MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM)) |
                        (desc ? ttFlagBits(idx < layerTargetsExp.size() && layerTargetsExp[idx] != 0, *desc) : 0u);
            exp.textureTransformIndex = tt ? blk : 0u;
        }
        const u32 got = materials_[idx].extensionIndex;
        if (got == MATERIAL_EXTENSION_INVALID) {
            if (mask != 0) return fmt("material %u: active features without an extension group", idx);
            if (matGroup_[idx] != NONE) return fmt("material %u: group tracked but not in the record", idx);
        } else {
            if (got % MATERIAL_EXT_RECORDS != 0 || got / MATERIAL_EXT_RECORDS >= extHigh_ || matGroup_[idx] != got)
                return fmt("material %u: bad extension index %u", idx, got);
            const u32 g = got / MATERIAL_EXT_RECORDS;
            if (groupOwned[g]) return fmt("material %u: extension group %u shared", idx, g);
            groupOwned[g] = 1;
            const MaterialExtGroup want = packMaterialExtensions(desc ? *desc : MaterialDescription{}, layerUnion);
            for (u32 k = 0; k < MATERIAL_EXT_RECORDS; ++k)
                if (!sameBytes(ext_[got + k], want[k])) return fmt("material %u: extension record %u differs", idx, k);
        }
        exp.extensionIndex = got;
        if (!sameBytes(materials_[idx], exp)) return fmt("material %u differs", idx);
        return {};
    };
    {
        const auto& libDesc = scene.materialDescriptions();
        for (u32 i = 0; i < libRegion; ++i) {
            GPUMaterial exp = expLib[i];
            exp.flags &= ~((MATERIAL_FEATURES_ALL << MATERIAL_FEATURE_SHIFT) | MATERIAL_FLAG_TEXTURE_TRANSFORM | MATERIAL_FLAG_LAYER_TRANSFORM);
            if (const std::string err = checkMaterial(i, exp, i < libDesc.size() && !lib.empty() ? &libDesc[i] : nullptr);
                !err.empty())
                return "library " + err;
        }
    }
    const u32 meshCount = scene.getMeshCount();

    // Expected decisions (independent recursion over the ECS).
    struct Exp { bool want = false, neg = false, motion = false; u32 depth = 0, cls = 0, mesh = 0, parent = NONE; u8 st = 0; };
    u32 maxId = 0;
    for (EntityID e : mi.entities()) {
        maxId = std::max(maxId, e);
        if (const HierarchyComponent* h = hier.tryGet(e); h && h->parent != NONE && h->parent < MAX_ENTITY_ID) {
            maxId = std::max(maxId, h->parent);
        }
    }
    std::vector<Exp> ex(mi.size() ? size_t(maxId) + 1 : 0);
    std::function<void(EntityID)> expOf = [&](EntityID e) {
        Exp& x = ex[e];
        if (x.st) return;
        x.st = 1;
        const MeshInstanceComponent* m = mi.tryGet(e);
        const TransformComponent* t = xf.tryGet(e);
        if (!m || !t || !m->isVisible() || m->meshHandle >= meshCount) return;
        bool neg;
        u32 depth = 0;
        const HierarchyComponent* h = hier.tryGet(e);
        if (h) {
            const u32 p = h->parent;
            if (p == NONE || p == e || p >= MAX_ENTITY_ID || p >= ex.size() || !mi.has(p)) return;
            expOf(p);
            if (!ex[p].want) return;
            depth = ex[p].depth + 1;
            if (depth >= SCENE_MAX_LEVELS) return;
            neg = ex[p].neg != negDet(t->worldMatrix);
            ex[e].parent = p;
        } else {
            neg = negDet(t->worldMatrix);
        }
        Exp& y = ex[e];
        bool ds;
        if (const MaterialComponent* mc = mat.tryGet(e)) ds = mc->doubleSided;
        else ds = (expLib[m->materialIndex < libRegion ? m->materialIndex : 0].flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0;
        y.want = true;
        y.neg = neg;
        y.depth = depth;
        y.motion = !h && mot.has(e);
        y.cls = static_cast<u32>(ds ? CullClass::None : (neg ? CullClass::BackMirrored : CullClass::Back));
        y.mesh = m->meshHandle;
    };
    for (EntityID e : mi.entities()) expOf(e);

    // Buckets: order, regions, ownership.
    if (buckets_.size() != posIds_.size() || buckets_.size() != idPos_.size()) return "bucket id tables inconsistent";
    std::vector<u32> owner(slotEnd(), NONE);
    for (u32 p = 0; p < buckets_.size(); ++p) {
        const SceneBucket& b = buckets_[p];
        if (p > 0 && !(keyOf(buckets_[p - 1]) < keyOf(b))) return fmt("buckets not sorted at %u", p);
        if (idPos_[posIds_[p]] != p) return fmt("bucket id table wrong at %u", p);
        if (b.count > b.used || b.used > b.capacity) return fmt("bucket %u count > used > capacity", p);
        if (uint64_t(b.firstSlot) + b.capacity > slotEnd()) return fmt("bucket %u region beyond slot space", p);
        for (u32 s = b.firstSlot; s < b.firstSlot + b.capacity; ++s) {
            if (owner[s] != NONE) return fmt("slot %u owned by two buckets", s);
            owner[s] = p;
        }
    }
    // Command layout.
    {
        std::vector<u32> cmds;
        u32 b = 0;
        for (u32 c = 0; c < SCENE_CULL_CLASSES; ++c) {
            if (classRanges_[c].firstCommand != cmds.size()) return fmt("class %u range start wrong", c);
            for (; b < buckets_.size() && static_cast<u32>(buckets_[b].cull) == c; ++b) {
                if (buckets_[b].command != cmds.size()) return fmt("bucket %u command index wrong", b);
                cmds.push_back(b);
            }
            cmds.push_back(NONE);
            if (classRanges_[c].commandCount != cmds.size() - classRanges_[c].firstCommand) return fmt("class %u range size wrong", c);
        }
        if (cmds != commandBuckets_) return "command buckets differ";
        if (gpuBuckets_.size() != buckets_.size()) return "gpu buckets size";
        const auto& infos = scene.meshInfos();
        for (u32 p = 0; p < buckets_.size(); ++p) {
            const SceneBucket& sb = buckets_[p];
            const GPUDrawBucket& g = gpuBuckets_[p];
            if (g.firstSlot != sb.firstSlot || g.capacity != sb.capacity || g.meshIndex != sb.mesh ||
                g.cullClass != static_cast<u32>(sb.cull) || g.command != sb.command ||
                g.indexCount != infos[sb.mesh].indexCount || g.indexOffset != infos[sb.mesh].indexOffset ||
                g.vertexOffset != infos[sb.mesh].vertexOffset) {
                return fmt("gpu bucket %u differs", p);
            }
        }
    }

    // Slots: live ones have an owner entity, the rest are zero.
    u32 live = 0;
    for (u32 s = 0; s < slotEnd(); ++s) {
        const u32 o = owner[s];
        const bool inLive = o != NONE && s < buckets_[o].firstSlot + buckets_[o].used && slotEntity_[s] != NONE;
        if (inLive) {
            ++live;
            const u32 e = slotEntity_[s];
            if (e == NONE || e >= ent_.size() || ent_[e].slot != s) return fmt("slot %u: entity mapping broken", s);
            if (!(instances_[s].flags & INSTANCE_FLAG_VALID)) return fmt("live slot %u without VALID", s);
        } else {
            if (slotEntity_[s] != NONE) return fmt("slack slot %u has an entity", s);
            if (!isZero(instances_[s]) || !isZero(nodes_[s]) || !isZero(motions_[s])) return fmt("slack slot %u not zero", s);
        }
    }
    u32 expCount = 0;
    for (EntityID e : mi.entities()) expCount += ex[e].want ? 1u : 0u;
    if (live != expCount || live != liveCount_) return fmt("live instances %u / counter %u != expected %u", live, liveCount_, expCount);

    // Entities.
    std::vector<std::vector<u32>> expChildren(slotEnd());
    std::vector<u32> expMotion;
    std::vector<u8> matUsed(matCap_, 0);
    u32 expMaxDepth = 0;
    for (EntityID e : mi.entities()) {
        const Exp& x = ex[e];
        if (!x.want) {
            if (e < ent_.size() && ent_[e].slot != NONE) return fmt("entity %u in the store but not expected", e);
            continue;
        }
        if (e >= ent_.size() || ent_[e].slot == NONE) return fmt("entity %u missing from the store", e);
        const Ent& en = ent_[e];
        const u32 s = en.slot;
        const auto it = keyToId_.find(makeKey(static_cast<CullClass>(x.cls), x.mesh));
        if (it == keyToId_.end()) return fmt("entity %u: no bucket for its (class, mesh)", e);
        const SceneBucket& b = buckets_[idPos_[it->second]];
        if (s < b.firstSlot || s >= b.firstSlot + b.used) return fmt("entity %u: slot %u outside its bucket", e, s);
        if (slotEntity_[s] != e) return fmt("entity %u: slotEntity mismatch", e);

        const MeshInstanceComponent& m = mi.get(e);
        const TransformComponent& t = xf.get(e);
        u32 matIdx;
        if (const MaterialComponent* mc = mat.tryGet(e)) {
            if (en.matSlot == NONE || en.matSlot >= matCap_) return fmt("entity %u: no persistent material", e);
            if (matUsed[en.matSlot]) return fmt("entity %u: material slot shared", e);
            matUsed[en.matSlot] = 1;
            matIdx = libRegion + en.matSlot;
            if (const std::string err = checkMaterial(matIdx, toGPUMaterial(*mc), mc->extension.get()); !err.empty())
                return fmt("entity %u: ", e) + err;
        } else {
            if (en.matSlot != NONE) return fmt("entity %u: stale per-entity material", e);
            matIdx = m.materialIndex < libRegion ? m.materialIndex : 0u;
        }
        const GPUInstance& gi = instances_[s];
        const u32 expFlags = m.flags | (x.neg ? INSTANCE_FLAG_MIRRORED : 0u) | INSTANCE_FLAG_VALID;
        if (gi.meshIndex != m.meshHandle || gi.materialIndex != matIdx || gi.flags != expFlags ||
            gi.generation != ecs.incarnation(e)) {
            return fmt("entity %u (slot %u): instance mesh/material/flags differ", e, s);
        }
        const bool computed = x.depth > 0 || x.motion; // matrix written by the GPU
        if (!computed && std::memcmp(gi.modelMatrix, &t.worldMatrix[0][0], sizeof(gi.modelMatrix)) != 0) {
            return fmt("entity %u (slot %u): model matrix differs from the ECS", e, s);
        }
        GPUTransformNode en2{};
        if (x.depth > 0) {
            const u32 ps = ent_[x.parent].slot;
            if (ps == NONE) return fmt("entity %u: parent %u not in the store", e, x.parent);
            en2 = makeNode(t, ps, x.depth);
            expChildren[ps].push_back(s);
            expMaxDepth = std::max(expMaxDepth, x.depth);
        }
        if (!sameBytes(nodes_[s], en2)) return fmt("entity %u (slot %u): node record differs", e, s);
        GPUMotion mm{};
        if (x.motion) {
            mm = makeMotion(mot.get(e), t);
            expMotion.push_back(s);
        }
        if (!sameBytes(motions_[s], mm)) return fmt("entity %u (slot %u): motion record differs", e, s);
    }

    // Materials: no stale per-entity records, emissive count.
    u32 emissive = 0;
    for (u32 i = 0; i < materials_.size(); ++i) {
        if (isEmissive(materials_[i])) ++emissive;
        if (i >= libRegion && !matUsed[i - libRegion] && !isZero(materials_[i])) return fmt("free material %u not zero", i);
    }
    if (emissive != emissiveCount_ || (emissive > 0) != (emissiveCount_ > 0)) return "emissive count differs";
    // F15.1: free and unowned extension groups are zero; the feature counters match the mirror.
    {
        std::vector<u8> freeSet(extHigh_, 0);
        for (u32 g : extFree_) {
            if (g >= extHigh_ || freeSet[g] || groupOwned[g]) return fmt("extension group %u: bad free list", g);
            freeSet[g] = 1;
        }
        for (u32 g : ttFree_) {
            if (g + MATERIAL_TT_GROUPS > extHigh_) return fmt("transform block %u: bad free list", g);
            for (u32 k = 0; k < MATERIAL_TT_GROUPS; ++k) {
                if (freeSet[g + k] || groupOwned[g + k]) return fmt("transform block %u: bad free list", g);
                freeSet[g + k] = 1;
            }
        }
        for (u32 g = 0; g < extHigh_; ++g) {
            if (!groupOwned[g] && !freeSet[g]) return fmt("extension group %u leaked", g);
            if (!groupOwned[g])
                for (u32 k = 0; k < MATERIAL_EXT_RECORDS; ++k)
                    if (!isZero(ext_[g * MATERIAL_EXT_RECORDS + k])) return fmt("free extension group %u not zero", g);
        }
        if (ext_.size() != size_t(extCap_) * MATERIAL_EXT_RECORDS) return "extension buffer size";
        u32 counts[MATERIAL_FEATURE_COUNT] = {};
        for (const GPUMaterial& m : materials_) {
            const u32 bits = materialFeatureBits(m);
            for (u32 f = 0; f < MATERIAL_FEATURE_COUNT; ++f) counts[f] += (bits >> f) & 1u;
        }
        for (u32 f = 0; f < MATERIAL_FEATURE_COUNT; ++f)
            if (counts[f] != featureCount_[f]) return fmt("feature %u count differs", f);
    }

    // CSR.
    if (csrOffsets_.size() != size_t(slotEnd()) + 1) return "CSR offsets size";
    if (!csrOffsets_.empty() && csrOffsets_.back() != csrSlots_.size()) return "CSR slots size";
    for (u32 s = 0; s < slotEnd(); ++s) {
        if (csrOffsets_[s + 1] < csrOffsets_[s]) return "CSR offsets not monotonic";
        std::vector<u32> got(csrSlots_.begin() + csrOffsets_[s], csrSlots_.begin() + csrOffsets_[s + 1]);
        std::sort(got.begin(), got.end());
        std::sort(expChildren[s].begin(), expChildren[s].end());
        if (got != expChildren[s]) return fmt("CSR children of slot %u differ", s);
    }

    // Motion list.
    std::vector<u32> gotMotion = motionSlots_;
    std::sort(gotMotion.begin(), gotMotion.end());
    std::sort(expMotion.begin(), expMotion.end());
    if (gotMotion != expMotion) return "motion slot list differs";
    u32 maxDepth = 0;
    for (u32 d = 1; d < SCENE_MAX_LEVELS; ++d) {
        if (depthCount_[d] > 0) maxDepth = d;
    }
    if (maxDepth != expMaxDepth) return fmt("max depth %u != %u", maxDepth, expMaxDepth);
    return {};
}

// ===========================================================================
// Public API
// ===========================================================================

SceneStore::SceneStore() : impl_(std::make_unique<Impl>()) {}
SceneStore::~SceneStore() = default;

void SceneStore::clear() {
    impl_->resetState();
    impl_->stats_ = SceneSyncStats{};
    impl_->csrRebuilds_ = 0;
    ++impl_->structureVersion_; // a rebuilt scene is a structure change for the host
}

void SceneStore::sync(ECS& ecs, const GpuScene& scene) { impl_->sync(ecs, scene); }

std::span<const GPUInstance>      SceneStore::instances() const { return impl_->instances_; }
std::span<const GPUMaterial>      SceneStore::materials() const { return impl_->materials_; }
std::span<const GPUTransformNode> SceneStore::nodes() const { return impl_->nodes_; }
std::span<const GPUMotion>        SceneStore::motions() const { return impl_->motions_; }
std::span<const SceneBucket>      SceneStore::buckets() const { return impl_->buckets_; }
std::span<const GPUDrawBucket>    SceneStore::gpuBuckets() const { return impl_->gpuBuckets_; }
std::span<const u32>              SceneStore::commandBuckets() const { return impl_->commandBuckets_; }
std::array<SceneStore::ClassRange, SCENE_CULL_CLASSES> SceneStore::classRanges() const { return impl_->classRanges_; }
u32 SceneStore::commandCount() const { return static_cast<u32>(impl_->commandBuckets_.size()); }
u32 SceneStore::slotCapacity() const { return impl_->slotEnd(); }
u32 SceneStore::instanceCount() const { return impl_->liveCount_; }
std::span<const u32> SceneStore::motionSlots() const { return impl_->motionSlots_; }
std::span<const u32> SceneStore::childOffsets() const { return impl_->csrOffsets_; }
std::span<const u32> SceneStore::childSlots() const { return impl_->csrSlots_; }
std::span<const u32> SceneStore::dirtyRoots() const { return impl_->dirtyRoots_; }
std::span<const u32> SceneStore::motionParentSlots() const { return impl_->motionParentSlots_; }
u32 SceneStore::maxDepth() const {
    u32 d = 0;
    for (u32 i = 1; i < SCENE_MAX_LEVELS; ++i) {
        if (impl_->depthCount_[i] > 0) d = i;
    }
    return d;
}
bool SceneStore::hasEmissive() const { return impl_->emissiveCount_ > 0; }
bool SceneStore::textureTransformInUse() const { return impl_->ttCount_ > 0; }
bool SceneStore::vertexColorInUse() const { return impl_->vertexColors_; }
u32 SceneStore::featureUnion() const {
    u32 u = 0;
    for (u32 f = 0; f < MATERIAL_FEATURE_COUNT; ++f)
        if (impl_->featureCount_[f] > 0) u |= 1u << f;
    return u;
}
bool SceneStore::volumetricSubsurface() const {
    if ((featureUnion() & MATERIAL_FEATURE_SUBSURFACE) == 0u) return false;
    for (const GPUMaterial& m : impl_->materials_) {
        if (!(materialFeatureBits(m) & MATERIAL_FEATURE_SUBSURFACE) || m.extensionIndex == MATERIAL_EXTENSION_INVALID ||
            size_t(m.extensionIndex) + 2 >= impl_->ext_.size())
            continue;
        GPUMaterialExt2 e2;
        std::memcpy(&e2, &impl_->ext_[m.extensionIndex + 2], sizeof(e2));
        if (e2.mode == 1u && (e2.weight > 0.0f || e2.weightTex != INVALID_TEXTURE_INDEX)) return true;
    }
    return false;
}
u32 SceneStore::extensionGroups() const { return impl_->extHigh_; }
u32 SceneStore::extensionGroupCapacity() const { return impl_->extCap_; }
std::span<const GPUMaterialExtRecord> SceneStore::extensions() const { return impl_->ext_; }

u32 SceneStore::slotOf(EntityID entity) const {
    return entity < impl_->ent_.size() ? impl_->ent_[entity].slot : NONE;
}
u32 SceneStore::materialIndexOf(EntityID entity) const {
    const u32 s = slotOf(entity);
    return s == NONE ? NONE : impl_->instances_[s].materialIndex;
}
u64 SceneStore::csrRebuildCount() const { return impl_->csrRebuilds_; }

std::span<const GPUDeltaRecord> SceneStore::instanceDeltas() const { return impl_->instRecs_; }
std::span<const GPUDeltaRecord> SceneStore::materialDeltas() const { return impl_->matRecs_; }
std::span<const GPUDeltaRecord> SceneStore::extensionDeltas() const { return impl_->extRecs_; }
bool SceneStore::materialsChanged() const {
    const SceneSyncStats& st = impl_->stats_;
    return st.fullMaterials || st.fullExtensions || !impl_->matRecs_.empty() || !impl_->extRecs_.empty();
}
std::span<const GPUDeltaRecord> SceneStore::nodeDeltas() const { return impl_->nodeRecs_; }
std::span<const GPUDeltaRecord> SceneStore::motionDeltas() const { return impl_->motRecs_; }
const SceneSyncStats& SceneStore::stats() const { return impl_->stats_; }
u64 SceneStore::structureVersion() const { return impl_->structureVersion_; }

std::string SceneStore::verifyAgainstEcs(const ECS& ecs, const GpuScene& scene) const {
    return impl_->verify(ecs, scene);
}

} // namespace phosphor
