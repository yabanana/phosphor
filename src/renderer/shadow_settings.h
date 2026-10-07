#pragma once
#include "renderer/gpu_types.h"
#include <array>
#include <span>
#include <vector>
#include <glm/glm.hpp>

namespace phosphor {
enum class ShadowTechnique : u32 { Off, Cascaded, RayTraced };
// Experimental presets only: none of these numerical choices has a measured
// quality/performance acceptance. The product baseline remains Off.
struct ShadowSettings {
    ShadowTechnique mode = ShadowTechnique::Off;
    u32 mapResolution = 1024, blockerSamples = 16, filterSamples = 16;
    float distance = 120.0f, splitLambda = 0.6f, casterReach = 500.0f;
    float sunAngularRadius = 0.00465f;
    float depthBiasWorld = 0.0005f, normalBiasWorld = 0.002f;
    u32 historySamples = 16;
    float temporalPositionThreshold = 0.02f, temporalNormalThreshold = 0.95f;
    bool contact = false, staticCache = false;
    u32 contactSteps = 16, cacheCapacity = 256, cacheUpdateBudget = 8;
    float contactDistance = 0.15f, contactThickness = 0.01f, contactStrength = 1.0f;
};
struct ShadowCamera {
    glm::mat4 inverseViewProjection{1.0f}; // unjittered reverse-Z projection
    glm::vec3 position{0.0f}, forward{0.0f, 0.0f, -1.0f};
    float nearPlane = 0.1f;
};
struct ShadowBounds { glm::vec3 minimum{0.0f}, maximum{0.0f}; };
void validateShadowSettings(const ShadowSettings& settings);
std::array<float, 5> shadowCascadeSplits(float nearPlane, float farPlane, float lambda);
std::array<GPUShadowCascade, 4> makeShadowCascades(const ShadowCamera& camera, glm::vec3 towardLight,
                                                const ShadowSettings& settings,
                                                std::span<const ShadowBounds> casterBounds = {});
u32 shadowCascadeIndex(float viewDepth, const std::array<GPUShadowCascade, 4>& cascades);
std::array<float, 3> shadowSolarDirection(glm::vec3 towardLight, float angularRadius, float u, float v);
// Dirty tile mask, 8x8 per cascade. Union old/new bounds for moved casters.
// Invalid bounds conservatively dirty every tile. No age-only invalidation.
u64 shadowDirtyTiles(const GPUShadowCascade& cascade, const ShadowBounds& bounds);

struct ShadowCacheKey {
    u32 view = 0, light = 0, cascade = 0, tile = 0;
    bool operator==(const ShadowCacheKey&) const = default;
};
struct ShadowCacheRevision {
    u64 light = 0, caster = 0, material = 0, projection = 0;
    bool operator==(const ShadowCacheRevision&) const = default;
};
enum class ShadowCacheAction : u32 { Cached, Update, DynamicFallback };
struct ShadowCacheDecision {
    ShadowCacheAction action = ShadowCacheAction::DynamicFallback;
    u32 entry = ~0u;
    u64 requiredCompletion = 0;
};
// Fixed-capacity LRU, no steady-state allocations. A reservation is invalid
// until publish() names its exact revision. Stale or budget-starved entries
// always use CURRENT dynamic shadow texels; old texels are never a fallback.
class ShadowStaticCache {
  public:
    explicit ShadowStaticCache(u32 capacity);
    void beginFrame(u64 frame, u64 completedSubmission, u32 updateBudget);
    ShadowCacheDecision request(ShadowCacheKey key, ShadowCacheRevision revision);
    void publish(u32 entry, ShadowCacheRevision revision, u64 submission);
    void read(u32 entry, u64 submission);
    void invalidate(ShadowCacheKey key);
    void invalidateLight(u32 light);
    void clear();
    [[nodiscard]] u32 capacity() const { return static_cast<u32>(entries_.size()); }
    [[nodiscard]] u32 updatesReserved() const { return updates_; }
  private:
    struct Entry {
        ShadowCacheKey key{};
        ShadowCacheRevision stored{}, requested{};
        u64 touched = 0, writer = 0, reader = 0;
        bool occupied = false, valid = false, reserved = false;
    };
    std::vector<Entry> entries_;
    u64 frame_ = 0, completed_ = 0;
    u32 budget_ = 0, updates_ = 0;
};
} // namespace phosphor
