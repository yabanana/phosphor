#include "platform/metal/residency_manager.h"

#include <stdexcept>

namespace phosphor {

namespace {

const char* setLabel(u32 cls) {
    return cls == static_cast<u32>(ResidencyClass::Static) ? "Phosphor static residency"
                                                          : "Phosphor streaming residency";
}

} // namespace

ResidencyManager::ResidencyManager(MTL::Device* device, MTL4::CommandQueue* queue) : queue_(queue) {
    for (u32 c = 0; c < COUNT; ++c) {
        MTL::ResidencySetDescriptor* desc = MTL::ResidencySetDescriptor::alloc()->init();
        desc->setLabel(NS::String::string(setLabel(c), NS::UTF8StringEncoding));
        desc->setInitialCapacity(256);
        NS::Error* error = nullptr;
        sets_[c] = device->newResidencySet(desc, &error);
        desc->release();
        if (!sets_[c]) {
            throw std::runtime_error("Failed to create residency set");
        }
        queue_->addResidencySet(sets_[c]);
    }
}

ResidencyManager::~ResidencyManager() {
    for (MTL::ResidencySet* set : sets_) {
        queue_->removeResidencySet(set);
        set->release();
    }
}

void ResidencyManager::add(const MTL::Allocation* allocation, ResidencyClass cls) {
    if (!allocation) return;
    const u32 c = static_cast<u32>(cls);
    sets_[c]->addAllocation(allocation);
    owner_[allocation] = cls;
    dirty_[c] = true;
}

void ResidencyManager::remove(const MTL::Allocation* allocation) {
    const auto it = owner_.find(allocation);
    if (it == owner_.end()) return;
    const u32 c = static_cast<u32>(it->second);
    sets_[c]->removeAllocation(allocation);
    owner_.erase(it);
    dirty_[c] = true;
}

void ResidencyManager::commit() {
    for (u32 c = 0; c < COUNT; ++c) {
        if (!dirty_[c]) continue;
        sets_[c]->commit();
        dirty_[c] = false;
        ++commits_[c];
    }
}

ResidencyManager::Stats ResidencyManager::stats(ResidencyClass cls) const {
    const u32 c = static_cast<u32>(cls);
    return {static_cast<u32>(sets_[c]->allocationCount()), sets_[c]->allocatedSize(), commits_[c]};
}

} // namespace phosphor
