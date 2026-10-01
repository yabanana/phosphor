#pragma once

#include "core/types.h"
#include <cassert>
#include <span>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <any>
#include <functional>

namespace phosphor {

// ---------------------------------------------------------------------------
// Per-frame change lists of one component array (F5.1).
//
//  - changed: entities whose component was (possibly) written since the last
//    clearChanges(): the non-const get()/modify()/markChanged().  An entity
//    appears at most once and never while it is in `added` (an added entity
//    is re-read by the consumer anyway).
//  - added / removed: add() / remove().  Net semantics against the state the
//    consumer last saw: add then remove in the same frame cancels (neither
//    list); remove then add reports the entity in BOTH lists (drop the old
//    component, take the new one); remove, add, remove reports it only in
//    `removed`.  Removing an entity also drops it from `changed`.
//  - all: a bulk writer took the non-const data() span, so every entity of
//    the array must be treated as changed (conservative fallback).
// ---------------------------------------------------------------------------
struct ComponentChanges {
    std::span<const EntityID> changed;
    std::span<const EntityID> added;
    std::span<const EntityID> removed;
    bool                      all = false;

    [[nodiscard]] bool empty() const {
        return !all && changed.empty() && added.empty() && removed.empty();
    }
};

// ---------------------------------------------------------------------------
// ComponentArray<T> -- Dense storage with sparse-to-dense mapping.
// Entity IDs map to packed indices via an unordered_map. Removal uses
// swap-and-pop to keep the array dense (cache-friendly for GPU upload).
//
// Change tracking (F5.1) costs O(1) per write and O(changes) per frame: two
// per-index positions (into the changed / added lists, ~0u = not listed)
// live next to the dense data and move with it on swap-and-pop.  READ
// through const references (std::as_const, const ECS&): the non-const
// accessors mark the entity changed.
// ---------------------------------------------------------------------------
template <typename T>
class ComponentArray {
public:
    using Changes = ComponentChanges;

    T& add(EntityID entity, T&& component) {
        assert(!has(entity) && "Entity already has this component");
        u32 index = static_cast<u32>(data_.size());
        entityToIndex_[entity] = index;
        entities_.push_back(entity);
        data_.push_back(std::move(component));
        changedPos_.push_back(kNotListed);
        addedPos_.push_back(static_cast<u32>(added_.size()));
        added_.push_back(entity);
        return data_.back();
    }

    /// Mutable access: marks the entity changed.
    [[nodiscard]] T& get(EntityID entity) {
        auto it = entityToIndex_.find(entity);
        if (it == entityToIndex_.end()) {
            throw std::runtime_error("ComponentArray::get -- entity does not have component");
        }
        markIndex(it->second);
        return data_[it->second];
    }

    /// Same as the non-const get(), spelled as intent.
    [[nodiscard]] T& modify(EntityID entity) { return get(entity); }

    [[nodiscard]] const T& get(EntityID entity) const {
        auto it = entityToIndex_.find(entity);
        if (it == entityToIndex_.end()) {
            throw std::runtime_error("ComponentArray::get -- entity does not have component");
        }
        return data_[it->second];
    }

    /// Const lookup that does not throw (nullptr when absent); never marks.
    [[nodiscard]] const T* tryGet(EntityID entity) const {
        auto it = entityToIndex_.find(entity);
        return it == entityToIndex_.end() ? nullptr : &data_[it->second];
    }

    /// Mark an entity changed after writing through a raw pointer/reference
    /// obtained without get() (no-op if the entity has no such component).
    void markChanged(EntityID entity) {
        auto it = entityToIndex_.find(entity);
        if (it != entityToIndex_.end()) markIndex(it->second);
    }

    [[nodiscard]] bool has(EntityID entity) const {
        return entityToIndex_.contains(entity);
    }

    void remove(EntityID entity) {
        auto it = entityToIndex_.find(entity);
        if (it == entityToIndex_.end()) return;

        u32 removedIndex = it->second;
        u32 lastIndex    = static_cast<u32>(data_.size()) - 1;

        // Change lists first, while the entity -> index map is intact.
        unlist(changed_, changedPos_, removedIndex);
        if (addedPos_[removedIndex] != kNotListed) {
            unlist(added_, addedPos_, removedIndex); // added and removed in one frame: cancels
        } else {
            removed_.push_back(entity);
        }

        if (removedIndex != lastIndex) {
            // Swap the last element into the removed slot
            data_[removedIndex]       = std::move(data_[lastIndex]);
            entities_[removedIndex]   = entities_[lastIndex];
            changedPos_[removedIndex] = changedPos_[lastIndex];
            addedPos_[removedIndex]   = addedPos_[lastIndex];

            // Update the swapped entity's index mapping
            entityToIndex_[entities_[removedIndex]] = removedIndex;
        }

        data_.pop_back();
        entities_.pop_back();
        changedPos_.pop_back();
        addedPos_.pop_back();
        entityToIndex_.erase(it);
    }

    /// Bulk mutable access: the whole array is assumed changed.
    [[nodiscard]] std::span<T> data() { all_ = true; return data_; }
    [[nodiscard]] std::span<const T> data() const { return data_; }

    [[nodiscard]] std::span<const EntityID> entities() const { return entities_; }

    [[nodiscard]] u32 size() const { return static_cast<u32>(data_.size()); }

    // --- Change tracking -------------------------------------------------------
    [[nodiscard]] Changes changes() const {
        return Changes{changed_, added_, removed_, all_};
    }

    /// Forget this frame's changes (O(changes)).
    void clearChanges() {
        for (EntityID e : changed_) changedPos_[entityToIndex_.at(e)] = kNotListed;
        for (EntityID e : added_)   addedPos_[entityToIndex_.at(e)]   = kNotListed;
        changed_.clear();
        added_.clear();
        removed_.clear();
        all_ = false;
    }

private:
    static constexpr u32 kNotListed = ~0u;

    void markIndex(u32 index) {
        if (addedPos_[index] != kNotListed || changedPos_[index] != kNotListed) return;
        changedPos_[index] = static_cast<u32>(changed_.size());
        changed_.push_back(entities_[index]);
    }

    // Remove the entry of dense index `index` from `list` (swap-and-pop on
    // the list), fixing the position of the entry that takes its place.
    void unlist(std::vector<EntityID>& list, std::vector<u32>& pos, u32 index) {
        const u32 p = pos[index];
        if (p == kNotListed) return;
        const EntityID moved = list.back();
        list[p] = moved;
        list.pop_back();
        if (moved != entities_[index]) pos[entityToIndex_.at(moved)] = p;
        pos[index] = kNotListed;
    }

    std::vector<T>                          data_;
    std::vector<EntityID>                   entities_;
    std::unordered_map<EntityID, u32>       entityToIndex_;

    std::vector<u32>        changedPos_; // per dense index: position in changed_
    std::vector<u32>        addedPos_;   // per dense index: position in added_
    std::vector<EntityID>   changed_, added_, removed_;
    bool                    all_ = false;
};

// ---------------------------------------------------------------------------
// ECS -- Manages entities and heterogeneous component arrays.
// Different ComponentArray<T> instances are stored via std::any, keyed by
// std::type_index. A parallel vector of "remove" callbacks enables
// destroyEntity to clean up across all arrays without knowing the types.
//
// Non-const getComponent()/getArray() hand out mutable access and therefore
// mark changes (see ComponentArray); read through const references.
// ---------------------------------------------------------------------------
class ECS {
public:
    EntityID createEntity();
    void     destroyEntity(EntityID entity);

    template <typename T>
    ComponentArray<T>& getArray() {
        std::type_index ti(typeid(T));
        auto it = arrays_.find(ti);
        if (it == arrays_.end()) {
            arrays_[ti] = ComponentArray<T>{};
            // Register a removal callback so destroyEntity can clean up
            removers_.push_back([ti, this](EntityID e) {
                auto& arr = std::any_cast<ComponentArray<T>&>(arrays_.at(ti));
                arr.remove(e);
            });
            enders_.push_back([ti, this]() {
                std::any_cast<ComponentArray<T>&>(arrays_.at(ti)).clearChanges();
            });
            it = arrays_.find(ti);
        }
        return std::any_cast<ComponentArray<T>&>(it->second);
    }

    /// Read-only view of an array: never registers it and never marks
    /// changes.  An unregistered type gives a shared empty array.
    template <typename T>
    [[nodiscard]] const ComponentArray<T>& getArray() const {
        std::type_index ti(typeid(T));
        auto it = arrays_.find(ti);
        if (it == arrays_.end()) {
            static const ComponentArray<T> kEmpty;
            return kEmpty;
        }
        return std::any_cast<const ComponentArray<T>&>(it->second);
    }

    template <typename T>
    T& addComponent(EntityID entity, T&& component) {
        return getArray<T>().add(entity, std::move(component));
    }

    /// Mutable access: marks the entity's component changed.
    template <typename T>
    [[nodiscard]] T& getComponent(EntityID entity) {
        return getArray<T>().get(entity);
    }

    template <typename T>
    [[nodiscard]] const T& getComponent(EntityID entity) const {
        std::type_index ti(typeid(T));
        auto it = arrays_.find(ti);
        if (it == arrays_.end()) {
            throw std::runtime_error("ECS::getComponent -- component type not registered");
        }
        return std::any_cast<const ComponentArray<T>&>(it->second).get(entity);
    }

    template <typename T>
    [[nodiscard]] bool hasComponent(EntityID entity) const {
        std::type_index ti(typeid(T));
        auto it = arrays_.find(ti);
        if (it == arrays_.end()) return false;
        return std::any_cast<const ComponentArray<T>&>(it->second).has(entity);
    }

    /// Mark an entity's component changed after writing it without
    /// getComponent() (raw pointer, cached reference).
    template <typename T>
    void markChanged(EntityID entity) { getArray<T>().markChanged(entity); }

    /// Clear every array's change lists; the engine calls it once per frame
    /// after the scene sync consumed them.
    void endFrame();

    [[nodiscard]] u32 entityCount() const { return entityCount_; }

private:
    EntityID nextEntity_  = 0;
    u32      entityCount_ = 0;
    std::vector<EntityID> freeIds_; // destroyed ids, reused by createEntity (LIFO)
    std::vector<u8>       alive_;   // per id

    std::unordered_map<std::type_index, std::any>   arrays_;
    std::vector<std::function<void(EntityID)>>      removers_;
    std::vector<std::function<void()>>              enders_;
};

} // namespace phosphor
