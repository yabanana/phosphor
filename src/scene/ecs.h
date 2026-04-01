#pragma once

#include "core/types.h"
#include <cassert>
#include <span>
#include <stdexcept>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <any>
#include <functional>

namespace phosphor {

// ---------------------------------------------------------------------------
// ComponentArray<T> -- Dense storage with sparse-to-dense mapping.
// Entity IDs map to packed indices via an unordered_map. Removal uses
// swap-and-pop to keep the array dense (cache-friendly for GPU upload).
// ---------------------------------------------------------------------------
template <typename T>
class ComponentArray {
public:
    T& add(EntityID entity, T&& component) {
        assert(!has(entity) && "Entity already has this component");
        u32 index = static_cast<u32>(data_.size());
        entityToIndex_[entity] = index;
        entities_.push_back(entity);
        data_.push_back(std::move(component));
        return data_.back();
    }

    [[nodiscard]] T& get(EntityID entity) {
        auto it = entityToIndex_.find(entity);
        if (it == entityToIndex_.end()) {
            throw std::runtime_error("ComponentArray::get -- entity does not have component");
        }
        return data_[it->second];
    }

    [[nodiscard]] const T& get(EntityID entity) const {
        auto it = entityToIndex_.find(entity);
        if (it == entityToIndex_.end()) {
            throw std::runtime_error("ComponentArray::get -- entity does not have component");
        }
        return data_[it->second];
    }

    [[nodiscard]] bool has(EntityID entity) const {
        return entityToIndex_.contains(entity);
    }

    void remove(EntityID entity) {
        auto it = entityToIndex_.find(entity);
        if (it == entityToIndex_.end()) return;

        u32 removedIndex = it->second;
        u32 lastIndex    = static_cast<u32>(data_.size()) - 1;

        if (removedIndex != lastIndex) {
            // Swap the last element into the removed slot
            data_[removedIndex]     = std::move(data_[lastIndex]);
            entities_[removedIndex] = entities_[lastIndex];

            // Update the swapped entity's index mapping
            entityToIndex_[entities_[removedIndex]] = removedIndex;
        }

        data_.pop_back();
        entities_.pop_back();
        entityToIndex_.erase(it);
    }

    [[nodiscard]] std::span<T> data() { return data_; }
    [[nodiscard]] std::span<const T> data() const { return data_; }

    [[nodiscard]] std::span<const EntityID> entities() const { return entities_; }

    [[nodiscard]] u32 size() const { return static_cast<u32>(data_.size()); }

private:
    std::vector<T>                          data_;
    std::vector<EntityID>                   entities_;
    std::unordered_map<EntityID, u32>       entityToIndex_;
};

// ---------------------------------------------------------------------------
// ECS -- Manages entities and heterogeneous component arrays.
// Different ComponentArray<T> instances are stored via std::any, keyed by
// std::type_index. A parallel vector of "remove" callbacks enables
// destroyEntity to clean up across all arrays without knowing the types.
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
            it = arrays_.find(ti);
        }
        return std::any_cast<ComponentArray<T>&>(it->second);
    }

    template <typename T>
    T& addComponent(EntityID entity, T&& component) {
        return getArray<T>().add(entity, std::move(component));
    }

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

    [[nodiscard]] u32 entityCount() const { return entityCount_; }

private:
    EntityID nextEntity_  = 0;
    u32      entityCount_ = 0;

    std::unordered_map<std::type_index, std::any>   arrays_;
    std::vector<std::function<void(EntityID)>>      removers_;
};

} // namespace phosphor
