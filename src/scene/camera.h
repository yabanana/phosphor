#pragma once

#include "core/types.h"
#include <glm/glm.hpp>
#include <array>

namespace phosphor {

class Input; // forward declaration

class Camera {
public:
    Camera(float fovY, float aspect, float nearPlane, float farPlane);

    // --- Update modes ---
    void updateFPS(const Input& input, float dt);
    void setOrbitMode(glm::vec3 target, float distance);
    void updateOrbit(const Input& input, float dt);

    // --- Matrix recomputation ---
    void updateMatrices();

    // --- Frustum ---
    [[nodiscard]] std::array<glm::vec4, 6> getFrustumPlanes() const;

    // --- Setters ---
    void setAspect(float aspect);
    void setJitter(glm::vec2 jitter);
    void setPosition(glm::vec3 pos);
    void setYawPitch(float yaw, float pitch);
    void setMoveSpeed(float speed);
    void setMouseSensitivity(float sensitivity);

    // --- Getters ---
    [[nodiscard]] const glm::mat4& getView()           const { return view_; }
    [[nodiscard]] const glm::mat4& getProjection()     const { return projection_; }
    [[nodiscard]] const glm::mat4& getViewProjection() const { return viewProjection_; }
    [[nodiscard]] const glm::mat4& getPrevViewProjection() const { return prevViewProjection_; }
    [[nodiscard]] glm::vec3        getPosition()       const { return position_; }
    [[nodiscard]] glm::vec3        getFront()          const { return front_; }
    [[nodiscard]] glm::vec3        getRight()          const { return right_; }
    [[nodiscard]] glm::vec3        getUp()             const { return up_; }
    [[nodiscard]] float            getNear()           const { return nearPlane_; }
    [[nodiscard]] float            getFar()            const { return farPlane_; }
    [[nodiscard]] float            getFovY()           const { return fovY_; }
    [[nodiscard]] float            getAspect()         const { return aspect_; }

private:
    void updateDirectionVectors();

    // --- Spatial ---
    glm::vec3 position_{0.0f, 0.0f, 5.0f};
    glm::vec3 front_{0.0f, 0.0f, -1.0f};
    glm::vec3 up_{0.0f, 1.0f, 0.0f};
    glm::vec3 right_{1.0f, 0.0f, 0.0f};
    glm::vec3 worldUp_{0.0f, 1.0f, 0.0f};

    float yaw_   = -90.0f; // degrees, -90 = looking along -Z
    float pitch_ = 0.0f;   // degrees

    // --- Projection ---
    float fovY_;
    float aspect_;
    float nearPlane_;
    float farPlane_;

    // --- Matrices ---
    glm::mat4 view_{1.0f};
    glm::mat4 projection_{1.0f};
    glm::mat4 viewProjection_{1.0f};
    glm::mat4 prevViewProjection_{1.0f};

    // --- TAA jitter ---
    glm::vec2 jitter_{0.0f};

    // --- FPS controls ---
    float moveSpeed_        = 5.0f;
    float sprintMultiplier_ = 3.0f;
    float mouseSensitivity_ = 0.1f;

    // --- Orbit mode ---
    bool      orbitMode_    = false;
    glm::vec3 orbitTarget_{0.0f};
    float     orbitDistance_ = 5.0f;
    float     orbitMinDist_  = 0.5f;
    float     orbitMaxDist_  = 100.0f;
};

} // namespace phosphor
