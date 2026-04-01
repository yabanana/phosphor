#pragma once

#include "core/types.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace phosphor {

struct TransformComponent {
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // identity
    glm::vec3 scale{1.0f};
    glm::mat4 worldMatrix{1.0f}; // cached, computed from above

    void updateMatrix() {
        worldMatrix = glm::translate(glm::mat4{1.0f}, position)
                    * glm::mat4_cast(rotation)
                    * glm::scale(glm::mat4{1.0f}, scale);
    }
};

struct MeshInstanceComponent {
    u32 meshHandle    = ~0u;
    u32 materialIndex = ~0u;
    u32 flags         = 0; // bit 0: visible, bit 1: casts shadows, bit 2: static

    [[nodiscard]] bool isVisible()      const { return (flags & (1u << 0)) != 0; }
    [[nodiscard]] bool castsShadows()   const { return (flags & (1u << 1)) != 0; }
    [[nodiscard]] bool isStatic()       const { return (flags & (1u << 2)) != 0; }

    void setVisible(bool v)    { v ? (flags |= (1u << 0)) : (flags &= ~(1u << 0)); }
    void setCastsShadows(bool v){ v ? (flags |= (1u << 1)) : (flags &= ~(1u << 1)); }
    void setStatic(bool v)     { v ? (flags |= (1u << 2)) : (flags &= ~(1u << 2)); }
};

struct MaterialComponent {
    glm::vec4 baseColorFactor{1.0f};
    float metallicFactor          = 0.0f;
    float roughnessFactor         = 0.5f;
    float normalScale             = 1.0f;
    float occlusionStrength       = 1.0f;
    u32   baseColorTexIndex       = ~0u; // bindless index
    u32   normalTexIndex          = ~0u;
    u32   metallicRoughnessTexIndex = ~0u;
    u32   occlusionTexIndex       = ~0u;
    u32   emissiveTexIndex        = ~0u;
    glm::vec3 emissiveFactor{0.0f};
    float alphaCutoff             = 0.5f;
};

enum class LightType : u32 {
    Directional = 0,
    Point       = 1,
    Spot        = 2
};

struct LightComponent {
    LightType type       = LightType::Directional;
    glm::vec3 color{1.0f};
    float     intensity  = 1.0f;
    float     range      = 10.0f;
    float     innerConeAngle = 0.0f;      // for spot lights
    float     outerConeAngle = 0.7854f;   // ~45 degrees
    u32       shadowMapIndex = ~0u;
};

struct CameraComponent {
    float fovY      = glm::radians(60.0f);
    float nearPlane = 0.1f;
    float farPlane  = 1000.0f;
    bool  isActive  = false;
};

} // namespace phosphor
