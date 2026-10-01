#include "scene/ecs.h"

namespace phosphor {

EntityID ECS::createEntity() {
    ++entityCount_;
    return nextEntity_++;
}

void ECS::destroyEntity(EntityID entity) {
    for (auto& remover : removers_) {
        remover(entity);
    }
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
