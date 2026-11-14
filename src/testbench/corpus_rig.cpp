#include "testbench/corpus_rig.h"

#include <algorithm>
#include <cmath>

namespace phosphor {

CorpusRigView frameCorpusRig(glm::vec3 minimum, glm::vec3 maximum, float yawDegrees, float pitchDegrees, float fovYRadians, float aspect) {
    const auto finite = [](glm::vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); };
    if (!finite(minimum) || !finite(maximum) || glm::any(glm::greaterThan(minimum, maximum))) {
        minimum = glm::vec3(-0.5f);
        maximum = glm::vec3(0.5f);
    }
    const glm::vec3 centre = 0.5f * (minimum + maximum), half = 0.5f * (maximum - minimum);
    const float yaw = glm::radians(yawDegrees), pitch = glm::radians(pitchDegrees);
    // `back` points from the target to the camera; yaw 0, pitch 0 is the glTF front (+Z).
    const glm::vec3 back(std::sin(yaw) * std::cos(pitch), std::sin(pitch), std::cos(yaw) * std::cos(pitch));
    const glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0, 1, 0), back));
    const glm::vec3 up = glm::cross(back, right);
    const auto extent = [&](glm::vec3 axis) { return glm::dot(glm::abs(axis), half); };
    const float tanHalf = std::tan(0.5f * fovYRadians);
    // The box's half-extent along the view axes bounds its projection; its depth half-extent keeps the near face in front of the camera.
    const float fit = std::max(extent(up) / tanHalf, extent(right) / (tanHalf * std::max(aspect, 0.1f)));
    CorpusRigView view;
    view.distance = std::max(fit * CorpusRig::margin + extent(back), 1e-3f);
    view.target = centre;
    view.position = centre + back * view.distance;
    view.towardSun = glm::normalize(-0.45f * right + 0.55f * up + 0.70f * back);
    return view;
}

} // namespace phosphor
