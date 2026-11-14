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
    // Exact fit over the eight corners: a corner at view offset (r, u, b) (b towards the camera) is inside the frame, with the margin, when
    // d - b >= margin * max(|r| / (tan(fov/2) * aspect), |u| / tan(fov/2)).  The box is not replaced by a sphere: a deep box (a draped cloth)
    // would otherwise be framed far too wide.
    const float tanHalf = std::tan(0.5f * fovYRadians);
    float distance = 1e-3f;
    for (unsigned corner = 0; corner < 8; ++corner) {
        const glm::vec3 offset((corner & 1) ? half.x : -half.x, (corner & 2) ? half.y : -half.y, (corner & 4) ? half.z : -half.z);
        const float r = std::abs(glm::dot(offset, right)), u = std::abs(glm::dot(offset, up)), b = glm::dot(offset, back);
        distance = std::max(distance, CorpusRig::margin * std::max(r / (tanHalf * std::max(aspect, 0.1f)), u / tanHalf) + b);
    }
    CorpusRigView view;
    view.distance = distance;
    view.target = centre;
    view.position = centre + back * view.distance;
    view.towardSun = glm::normalize(-0.45f * right + 0.55f * up + 0.70f * back);
    return view;
}

} // namespace phosphor
