#include "scene/ecs.h"

namespace phosphor {

EntityID ECS::createEntity() {
    ++entityCount_;
    // F5: destroyed ids are reused (LIFO), so tables indexed by EntityID
    // (the scene store) stop growing under steady churn: CPU heap flat (O7).
    EntityID e;
    if (!freeIds_.empty()) {
        e = freeIds_.back();
        freeIds_.pop_back();
    } else {
        e = nextEntity_++;
        alive_.push_back(0);
    }
    alive_[e] = 1;
    return e;
}

void ECS::destroyEntity(EntityID entity) {
    // Destroying twice must not hand the id out twice.
    if (entity >= alive_.size() || !alive_[entity]) return;
    for (auto& remover : removers_) {
        remover(entity);
    }
    alive_[entity] = 0;
    freeIds_.push_back(entity);
    if (entityCount_ > 0) {
        --entityCount_;
    }
}

void ECS::endFrame() {
    for (auto& ender : enders_) {
        ender();
    }
}

} // namespace phosphor
