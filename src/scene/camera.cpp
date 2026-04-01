#include "scene/camera.h"
#include "core/input.h"

#include <glm/gtc/matrix_transform.hpp>
#include <SDL3/SDL_scancode.h>
#include <algorithm>
#include <cmath>

namespace phosphor {

Camera::Camera(float fovY, float aspect, float nearPlane, float farPlane)
    : fovY_(fovY)
    , aspect_(aspect)
    , nearPlane_(nearPlane)
    , farPlane_(farPlane)
{
    updateDirectionVectors();
    updateMatrices();
}

// ---------------------------------------------------------------------------
// FPS-style update: WASD movement, mouse look, shift to sprint
// ---------------------------------------------------------------------------
void Camera::updateFPS(const Input& input, float dt) {
    orbitMode_ = false;

    // --- Mouse look ---
    glm::vec2 mouseDelta = input.getMouseDelta();
    if (input.isMouseButtonDown(3)) { // right mouse button
        yaw_   += mouseDelta.x * mouseSensitivity_;
        pitch_ -= mouseDelta.y * mouseSensitivity_;
        pitch_  = std::clamp(pitch_, -89.0f, 89.0f);
        updateDirectionVectors();
    }

    // --- Movement ---
    float speed = moveSpeed_ * dt;
    if (input.isKeyDown(SDL_SCANCODE_LSHIFT)) {
        speed *= sprintMultiplier_;
    }

    if (input.isKeyDown(SDL_SCANCODE_W)) position_ += front_ * speed;
    if (input.isKeyDown(SDL_SCANCODE_S)) position_ -= front_ * speed;
    if (input.isKeyDown(SDL_SCANCODE_A)) position_ -= right_ * speed;
    if (input.isKeyDown(SDL_SCANCODE_D)) position_ += right_ * speed;
    if (input.isKeyDown(SDL_SCANCODE_E)) position_ += worldUp_ * speed;
    if (input.isKeyDown(SDL_SCANCODE_Q)) position_ -= worldUp_ * speed;
}

// ---------------------------------------------------------------------------
// Orbit mode setup
// ---------------------------------------------------------------------------
void Camera::setOrbitMode(glm::vec3 target, float distance) {
    orbitMode_    = true;
    orbitTarget_  = target;
    orbitDistance_ = distance;
}

// ---------------------------------------------------------------------------
// Orbit update: left-drag rotates around target, scroll zooms
// ---------------------------------------------------------------------------
void Camera::updateOrbit(const Input& input, float dt) {
    (void)dt;
    if (!orbitMode_) return;

    // --- Drag to rotate ---
    if (input.isMouseButtonDown(1)) { // left mouse button
        glm::vec2 mouseDelta = input.getMouseDelta();
        yaw_   += mouseDelta.x * mouseSensitivity_;
        pitch_ -= mouseDelta.y * mouseSensitivity_;
        pitch_  = std::clamp(pitch_, -89.0f, 89.0f);
    }

    // --- Scroll to zoom ---
    float scroll = input.getScrollDelta();
    if (scroll != 0.0f) {
        orbitDistance_ -= scroll * orbitDistance_ * 0.1f;
        orbitDistance_  = std::clamp(orbitDistance_, orbitMinDist_, orbitMaxDist_);
    }

    // --- Compute camera position from spherical coordinates ---
    float yawRad   = glm::radians(yaw_);
    float pitchRad = glm::radians(pitch_);

    glm::vec3 offset;
    offset.x = std::cos(pitchRad) * std::cos(yawRad);
    offset.y = std::sin(pitchRad);
    offset.z = std::cos(pitchRad) * std::sin(yawRad);

    position_ = orbitTarget_ + offset * orbitDistance_;

    // Point the camera at the target
    front_ = glm::normalize(orbitTarget_ - position_);
    right_ = glm::normalize(glm::cross(front_, worldUp_));
    up_    = glm::normalize(glm::cross(right_, front_));
}

// ---------------------------------------------------------------------------
// Recompute all matrices. Stores the previous VP for TAA reprojection.
// ---------------------------------------------------------------------------
void Camera::updateMatrices() {
    prevViewProjection_ = viewProjection_;

    view_ = glm::lookAt(position_, position_ + front_, up_);

    // Reverse-Z infinite far plane projection for better depth precision.
    // With GLM_FORCE_DEPTH_ZERO_TO_ONE this maps near to 1 and far to 0.
    float f = 1.0f / std::tan(fovY_ * 0.5f);
    projection_ = glm::mat4(0.0f);
    projection_[0][0] =  f / aspect_;
    projection_[1][1] =  f;
    projection_[2][2] =  0.0f;              // reversed: far maps to 0
    projection_[2][3] = -1.0f;              // perspective divide
    projection_[3][2] =  nearPlane_;        // reversed: near maps to 1

    // Apply sub-pixel jitter for TAA (in clip space)
    projection_[2][0] += jitter_.x;
    projection_[2][1] += jitter_.y;

    viewProjection_ = projection_ * view_;
}

// ---------------------------------------------------------------------------
// Extract 6 frustum planes from the view-projection matrix.
// Planes are in world space, normals point inward. Plane eq: dot(n,P)+d >= 0
// Order: left, right, bottom, top, near, far
// ---------------------------------------------------------------------------
std::array<glm::vec4, 6> Camera::getFrustumPlanes() const {
    std::array<glm::vec4, 6> planes{};
    const glm::mat4& m = viewProjection_;

    // Left:   row3 + row0
    planes[0] = glm::vec4(m[0][3] + m[0][0], m[1][3] + m[1][0],
                           m[2][3] + m[2][0], m[3][3] + m[3][0]);
    // Right:  row3 - row0
    planes[1] = glm::vec4(m[0][3] - m[0][0], m[1][3] - m[1][0],
                           m[2][3] - m[2][0], m[3][3] - m[3][0]);
    // Bottom: row3 + row1
    planes[2] = glm::vec4(m[0][3] + m[0][1], m[1][3] + m[1][1],
                           m[2][3] + m[2][1], m[3][3] + m[3][1]);
    // Top:    row3 - row1
    planes[3] = glm::vec4(m[0][3] - m[0][1], m[1][3] - m[1][1],
                           m[2][3] - m[2][1], m[3][3] - m[3][1]);
    // Near:   row3 + row2
    planes[4] = glm::vec4(m[0][3] + m[0][2], m[1][3] + m[1][2],
                           m[2][3] + m[2][2], m[3][3] + m[3][2]);
    // Far:    row3 - row2
    planes[5] = glm::vec4(m[0][3] - m[0][2], m[1][3] - m[1][2],
                           m[2][3] - m[2][2], m[3][3] - m[3][2]);

    // Normalize each plane
    for (auto& p : planes) {
        float len = glm::length(glm::vec3(p));
        if (len > 0.0f) p /= len;
    }

    return planes;
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------
void Camera::setAspect(float aspect) { aspect_ = aspect; }
void Camera::setJitter(glm::vec2 jitter) { jitter_ = jitter; }
void Camera::setPosition(glm::vec3 pos) { position_ = pos; }

void Camera::setYawPitch(float yaw, float pitch) {
    yaw_   = yaw;
    pitch_ = std::clamp(pitch, -89.0f, 89.0f);
    updateDirectionVectors();
}

void Camera::setMoveSpeed(float speed) { moveSpeed_ = speed; }
void Camera::setMouseSensitivity(float sensitivity) { mouseSensitivity_ = sensitivity; }

// ---------------------------------------------------------------------------
// Internal: recompute front/right/up from yaw and pitch
// ---------------------------------------------------------------------------
void Camera::updateDirectionVectors() {
    float yawRad   = glm::radians(yaw_);
    float pitchRad = glm::radians(pitch_);

    front_.x = std::cos(pitchRad) * std::cos(yawRad);
    front_.y = std::sin(pitchRad);
    front_.z = std::cos(pitchRad) * std::sin(yawRad);
    front_   = glm::normalize(front_);

    right_ = glm::normalize(glm::cross(front_, worldUp_));
    up_    = glm::normalize(glm::cross(right_, front_));
}

} // namespace phosphor
