#include "renderer/scene_store.h"

#include "core/log.h"
#include "renderer/gpu_scene.h"
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
        materials_.resize(n);
        matStamp_.resize(n, 0);
        matCap_        = newCap;
        matCapChanged_ = true;
        structure_     = true; // buffer re-sized: the host re-sends it (and rebinds)
    }

    void setMaterial(u32 idx, const GPUMaterial& gm) {
        GPUMaterial& cur = materials_[idx];
        if (sameBytes(cur, gm)) return;
        if (isEmissive(cur)) --emissiveCount_;
        if (isEmissive(gm)) ++emissiveCount_;
        cur = gm;
        markMat(idx);
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
        setMaterial(libRegion_ + en.matSlot, GPUMaterial{});
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
            setMaterial(matIdx, toGPUMaterial(*mc)); // (may grow materials_; no Ent reference is invalidated)
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
        materials_.assign(lib.begin(), lib.end());
        if (materials_.empty()) materials_.push_back(toGPUMaterial(MaterialComponent{}));
        matStamp_.assign(materials_.size(), 0);
        for (const GPUMaterial& m : materials_) emissiveCount_ += isEmissive(m) ? 1u : 0u;

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
        u64 bytes = u64(stats_.instanceRecords + stats_.materialRecords + stats_.nodeRecords + stats_.motionRecords) *
                    sizeof(GPUDeltaRecord);
        if (stats_.fullInstances) bytes += u64(instances_.size()) * sizeof(GPUInstance);
        if (stats_.fullNodes)     bytes += u64(nodes_.size()) * sizeof(GPUTransformNode);
        if (stats_.fullMotions)   bytes += u64(motions_.size()) * sizeof(GPUMotion);
        if (stats_.motionParentsChanged) bytes += 32 + u64(motionParentSlots_.size()) * sizeof(u32);
        if (stats_.fullMaterials) bytes += u64(materials_.size()) * sizeof(GPUMaterial);
        if (stats_.structure) {
            bytes += u64(gpuBuckets_.size()) * sizeof(GPUDrawBucket) + u64(commandBuckets_.size()) * sizeof(u32) +
                     u64(csrOffsets_.size() + csrSlots_.size() + motionSlots_.size()) * sizeof(u32);
        }
        stats_.uploadBytes = bytes;
    }

    void sync(ECS& ecs, const GpuScene& scene) {
        ecs_ = &ecs;
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
        dirtyParents_.clear();
        stats_ = SceneSyncStats{};
        structure_ = capacityChanged_ = matCapChanged_ = csrDirty_ = bucketsDirty_ = fullAll_ = false;

        const bool rebuild = !built_ || scene.geometryVersion() != geometryVersion_ ||
                             std::max<u32>(1u, static_cast<u32>(scene.materials().size())) != libRegion_;
        if (rebuild) {
            build();
        } else {
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
    for (u32 i = 0; i < libRegion; ++i) {
        if (!sameBytes(materials_[i], expLib[i])) return fmt("library material %u differs", i);
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
            if (!sameBytes(materials_[matIdx], toGPUMaterial(*mc))) return fmt("entity %u: material record differs", e);
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
std::span<const GPUDeltaRecord> SceneStore::nodeDeltas() const { return impl_->nodeRecs_; }
std::span<const GPUDeltaRecord> SceneStore::motionDeltas() const { return impl_->motRecs_; }
const SceneSyncStats& SceneStore::stats() const { return impl_->stats_; }
u64 SceneStore::structureVersion() const { return impl_->structureVersion_; }

std::string SceneStore::verifyAgainstEcs(const ECS& ecs, const GpuScene& scene) const {
    return impl_->verify(ecs, scene);
}

} // namespace phosphor
