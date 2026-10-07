#include "renderer/rt_proxy_transition.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_store.h"
#include "scene/ecs.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace phosphor {
namespace {
MaterialComponent component(const GPUMaterial& m) {
    MaterialComponent out;
    out.baseColorFactor = {m.baseColor[0],m.baseColor[1],m.baseColor[2],m.baseColor[3]};
    out.metallicFactor = m.metallic; out.roughnessFactor = m.roughness;
    out.normalScale = m.normalScale; out.occlusionStrength = m.occlusionStrength;
    out.baseColorTexIndex = m.baseColorTex; out.normalTexIndex = m.normalTex;
    out.metallicRoughnessTexIndex = m.metallicRoughnessTex; out.occlusionTexIndex = m.occlusionTex;
    out.emissiveTexIndex = m.emissiveTex;
    out.emissiveFactor = {m.emissive[0],m.emissive[1],m.emissive[2]};
    out.alphaCutoff = m.alphaCutoff;
    out.doubleSided = (m.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0;
    return out;
}
GPUMaterial masked(GPUMaterial m) {
    m.alphaCutoff = 0.5f;
    m.baseColor[3] = 0; // Deterministically transparent, independent of texture alpha.
    return m;
}
} // namespace

bool RtProxyTransitionCheck::fail(std::string message) {
    if (status_.error.empty()) status_.error = std::move(message);
    return false;
}

bool RtProxyTransitionCheck::arm(RtProxyTransition mode, ECS& ecs, GpuScene& scene,
                                 const SceneStore& store, const RtProxyGeometry& geometry) {
    *this = {};
    status_.mode = mode;
    if (mode == RtProxyTransition::None || u32(mode) > u32(RtProxyTransition::FullUpload))
        return fail("proxy transition mode is invalid");
    if (!geometry.manifestApplied) return fail("proxy transition requires an active measured manifest");
    if (geometry.meshes.size() != geometry.selections.size()) return fail("proxy transition geometry metadata is inconsistent");
    const auto& meshes = std::as_const(ecs).getArray<MeshInstanceComponent>();
    for (EntityID entity : meshes.entities()) {
        const auto& instance = meshes.get(entity);
        const u32 slot = store.slotOf(entity);
        if (slot >= store.instances().size() || !(store.instances()[slot].flags & INSTANCE_FLAG_VALID) ||
            instance.meshHandle >= geometry.meshes.size() || instance.meshHandle >= scene.meshInfos().size()) continue;
        const auto& selection = geometry.selections[instance.meshHandle];
        if (selection.level == RtProxyLevel::Full || selection.proxyIndexCount >= selection.sourceIndexCount) continue;
        if ((mode == RtProxyTransition::Reassign || mode == RtProxyTransition::FullUpload) &&
            ecs.hasComponent<MaterialComponent>(entity)) continue;
        if (target_ != INVALID_ENTITY && std::pair(instance.meshHandle,entity) >= std::pair(status_.mesh,target_)) continue;
        const u32 material = store.instances()[slot].materialIndex;
        if (material >= store.materials().size()) continue;
        target_ = entity;
        status_.mesh = instance.meshHandle;
        status_.sourceIndices = selection.sourceIndexCount;
        status_.initialIndices = selection.proxyIndexCount;
        baseMaterial_ = store.materials()[material];
    }
    if (target_ == INVALID_ENTITY) return fail("no live entity uses a genuinely reduced proxy");
    if (mode == RtProxyTransition::Mask || mode == RtProxyTransition::Emissive) {
        auto copy = component(baseMaterial_);
        if (ecs.hasComponent<MaterialComponent>(target_)) ecs.getComponent<MaterialComponent>(target_) = copy;
        else ecs.addComponent(target_, std::move(copy));
    } else if (mode == RtProxyTransition::Reassign) {
        preparedMaterial_ = scene.addMaterial(masked(baseMaterial_)); // Unused until frame eight.
    }
    return true;
}

bool RtProxyTransitionCheck::beforeSync(u32 frame, ECS& ecs, GpuScene& scene) {
    if (!status_.error.empty()) return false;
    if (status_.applied || frame < 8) return true;
    if (!ecs.hasComponent<MeshInstanceComponent>(target_)) return fail("transition entity disappeared");
    if (status_.mode == RtProxyTransition::Mask || status_.mode == RtProxyTransition::Emissive) {
        if (!ecs.hasComponent<MaterialComponent>(target_)) return fail("prepared material disappeared");
        auto& m = ecs.getComponent<MaterialComponent>(target_);
        if (status_.mode == RtProxyTransition::Mask) { m.alphaCutoff = 0.5f; m.baseColorFactor.a = 0; }
        else m.emissiveFactor = {2,0,0};
    } else {
        if (status_.mode == RtProxyTransition::FullUpload) preparedMaterial_ = scene.addMaterial(masked(baseMaterial_));
        ecs.getComponent<MeshInstanceComponent>(target_).materialIndex = preparedMaterial_;
    }
    status_.applied = true;
    return true;
}

bool RtProxyTransitionCheck::afterSync(const SceneStore& store, const RtProxyGeometry& geometry) {
    if (!status_.error.empty()) return false;
    if (!status_.applied || status_.promoted) return true;
    const auto& stats = store.stats();
    status_.materialFull = stats.fullMaterials;
    status_.instancesFull = stats.fullInstances;
    status_.materialRecords = stats.materialRecords;
    status_.instanceRecords = stats.instanceRecords;
    if (status_.mode == RtProxyTransition::FullUpload) {
        if (!stats.fullMaterials || !stats.fullInstances || !store.materialDeltas().empty() || !store.instanceDeltas().empty())
            return fail("full-upload control did not produce full buffers with empty delta arrays");
    } else if (status_.mode == RtProxyTransition::Reassign) {
        if (stats.fullInstances || store.instanceDeltas().empty()) return fail("reassignment did not use instance deltas");
    } else {
        if (stats.fullMaterials || store.materialDeltas().empty()) return fail("material edit did not use material deltas");
    }
    if (!geometry.manifestApplied || status_.mesh >= geometry.meshes.size() || status_.mesh >= geometry.selections.size())
        return fail("transition lost its measured proxy scene");
    status_.finalIndices = geometry.meshes[status_.mesh].indexCount;
    if (geometry.selections[status_.mesh].level != RtProxyLevel::Full || status_.finalIndices != status_.sourceIndices)
        return fail("protected mesh was not promoted to full source indices");
    expectedSlot_ = store.slotOf(target_);
    if (expectedSlot_ >= store.instances().size()) return fail("transition entity has no scene slot");
    const auto& instance = store.instances()[expectedSlot_];
    expectedGeneration_ = instance.generation;
    expectedMaterialIndex_ = instance.materialIndex;
    if (instance.meshIndex != status_.mesh || expectedMaterialIndex_ >= store.materials().size())
        return fail("transition scene assignment is invalid");
    expectedMaterial_ = store.materials()[expectedMaterialIndex_];
    if (status_.mode == RtProxyTransition::Emissive) {
        if (expectedMaterial_.emissive[0] != 2) return fail("emissive factor did not reach SceneStore");
    } else if (expectedMaterial_.alphaCutoff != 0.5f || expectedMaterial_.baseColor[3] != 0) {
        return fail("MASK change did not reach SceneStore");
    }
    if ((status_.mode == RtProxyTransition::Reassign || status_.mode == RtProxyTransition::FullUpload) &&
        expectedMaterialIndex_ != preparedMaterial_) return fail("material reassignment did not reach SceneStore");
    status_.promoted = true;
    return true;
}

bool RtProxyTransitionCheck::afterReadback(std::span<const GPUInstance> instances,
                                         std::span<const GPUMaterial> materials, bool checkerPassed) {
    if (!status_.error.empty()) return false;
    if (!status_.promoted || status_.verified) return true;
    if (!checkerPassed) return fail("RT checker failed after proxy transition");
    if (expectedSlot_ >= instances.size()) return fail("GPU transition slot is missing");
    const auto& instance = instances[expectedSlot_];
    if (!(instance.flags & INSTANCE_FLAG_VALID) || instance.generation != expectedGeneration_ ||
        instance.meshIndex != status_.mesh || instance.materialIndex != expectedMaterialIndex_)
        return fail("GPU transition instance assignment differs from the same-frame CPU snapshot");
    if (expectedMaterialIndex_ >= materials.size() ||
        std::memcmp(&materials[expectedMaterialIndex_], &expectedMaterial_, sizeof(GPUMaterial)) != 0)
        return fail("GPU transition material differs from the same-frame CPU snapshot");
    status_.verified = true;
    return true;
}

bool RtProxyTransitionCheck::finish() {
    return complete() || fail("proxy transition did not complete CPU promotion and GPU readback checks");
}

std::string RtProxyTransitionCheck::line() const {
    return "RT-PROXY-TRANSITION " + std::string(rtProxyTransitionName(status_.mode)) +
        " mesh " + std::to_string(status_.mesh) + " source " + std::to_string(status_.sourceIndices) +
        " before " + std::to_string(status_.initialIndices) + " after " + std::to_string(status_.finalIndices) +
        " material_full " + std::to_string(status_.materialFull) + " instances_full " + std::to_string(status_.instancesFull) +
        " material_records " + std::to_string(status_.materialRecords) + " instance_records " + std::to_string(status_.instanceRecords) +
        " applied " + std::to_string(status_.applied) + " promoted " + std::to_string(status_.promoted) +
        " verified " + std::to_string(status_.verified) + " | " + (complete() ? "PASS" : "FAIL") +
        (status_.error.empty() ? "" : ": " + status_.error);
}
} // namespace phosphor
