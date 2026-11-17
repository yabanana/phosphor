#include "scene/gltf_lights.h"

#include <tiny_gltf.h>

#include <algorithm>
#include <cmath>

namespace phosphor {

GltfPunctualLight convertPunctualLight(const tinygltf::Light& in, float photometricToEngine) {
    GltfPunctualLight out;
    LightComponent& l = out.light;
    if (in.type == "directional") l.type = LightType::Directional;
    else if (in.type == "point") l.type = LightType::Point;
    else if (in.type == "spot") l.type = LightType::Spot;
    else {
        out.warnings.push_back("KHR_lights_punctual: unknown light type '" + in.type + "', light ignored");
        return out;
    }
    out.valid = true;

    l.color = glm::vec3(1.0f);
    for (size_t i = 0; i < 3 && i < in.color.size(); ++i) l.color[static_cast<int>(i)] = static_cast<float>(in.color[i]);

    double intensity = in.intensity;
    if (!(intensity >= 0.0)) {
        out.warnings.push_back("KHR_lights_punctual: negative intensity clamped to 0");
        intensity = 0.0;
    }
    l.intensity = static_cast<float>(intensity) * photometricToEngine;

    // A directional light has no range (the specification); point and spot: absent or <= 0 is infinite.
    l.range = (l.type != LightType::Directional && in.range > 0.0) ? static_cast<float>(in.range) : GLTF_LIGHT_INFINITE_RANGE;

    if (l.type == LightType::Spot) {
        constexpr double halfPi = 1.5707963267948966;
        double inner = in.spot.innerConeAngle, outer = in.spot.outerConeAngle;
        if (!(outer > 0.0 && outer <= halfPi) || !(inner >= 0.0 && inner < outer)) {
            out.warnings.push_back("KHR_lights_punctual: spot cone angles outside 0 <= inner < outer <= pi/2 were clamped");
            outer = std::clamp(std::isfinite(outer) ? outer : 0.7853981634, 1.0e-3, halfPi);
            inner = std::clamp(std::isfinite(inner) ? inner : 0.0, 0.0, outer * 0.999);
        }
        l.innerConeAngle = static_cast<float>(inner);
        l.outerConeAngle = static_cast<float>(outer);
    }
    return out;
}

} // namespace phosphor
