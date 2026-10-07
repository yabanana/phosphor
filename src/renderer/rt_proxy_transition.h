#pragma once

#include "core/launch_options.h"
#include "renderer/rt_proxy.h"
#include "scene/components.h"

namespace phosphor {
class ECS;
class GpuScene;
class SceneStore;

struct RtProxyTransitionStatus {
    RtProxyTransition mode = RtProxyTransition::None;
    u32 mesh = ~0u, sourceIndices = 0, initialIndices = 0, finalIndices = 0;
    u32 materialRecords = 0, instanceRecords = 0;
    bool applied = false, promoted = false, materialFull = false, instancesFull = false, verified = false;
    std::string error;
};

// Diagnostic controller only: ordinary rendering never constructs this class.
// Hooks: arm after initial RT load, beforeSync after benchmark simulation,
// afterSync AFTER the engine's ordinary protection/reload branch, afterReadback
// once that same frame's GPU instance/material snapshot and RtChecker are ready.
// Merely passing RtChecker is insufficient: both promotion and upload mode
// must have been observed. finish() fails closed if interrupted/incomplete.
class RtProxyTransitionCheck {
public:
    bool arm(RtProxyTransition mode, ECS&, GpuScene&, const SceneStore&, const RtProxyGeometry&);
    bool beforeSync(u32 framesOnBench, ECS&, GpuScene&);
    bool afterSync(const SceneStore&, const RtProxyGeometry&);
    bool afterReadback(std::span<const GPUInstance>, std::span<const GPUMaterial>, bool checkerPassed);
    bool finish();
    [[nodiscard]] const RtProxyTransitionStatus& status() const { return status_; }
    [[nodiscard]] bool complete() const { return status_.verified && status_.error.empty(); }
    [[nodiscard]] std::string line() const;
private:
    bool fail(std::string);
    RtProxyTransitionStatus status_;
    EntityID target_ = INVALID_ENTITY;
    u32 preparedMaterial_ = ~0u, expectedSlot_ = ~0u, expectedGeneration_ = 0, expectedMaterialIndex_ = ~0u;
    GPUMaterial baseMaterial_{}, expectedMaterial_{};
};
} // namespace phosphor
