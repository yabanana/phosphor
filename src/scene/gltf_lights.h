#pragma once

// F15 closure: KHR_lights_punctual import.
//
// Maps a glTF punctual light onto the engine's LightComponent (portable, no Metal).  Units: the engine's light
// intensity is radiometric (directional W/m^2, point and spot W/sr); the glTF values are photometric (lux, candela)
// and the extension does not fix a luminous efficacy, so the import applies `photometricToEngine` (default 1: the
// authored number is used as is, the convention of the glTF Sample Viewer, with the exposure doing the rest; 1/683
// converts to watts for a physically scaled scene).  Direction and position come from the node's world matrix
// (the light shines along local -Z, the engine's rule in extractLights).

#include "scene/components.h"

#include <string>
#include <vector>

namespace tinygltf {
struct Light;
}

namespace phosphor {

/// "Infinite" range of a light without `range`: a finite window so the forward shader's range window and the
/// clustered DI bounds stay valid (1e5 m: the window is below 1e-8 at 1 km).
constexpr float GLTF_LIGHT_INFINITE_RANGE = 1.0e5f;

struct GltfPunctualLight {
    bool                     valid = false; // false: unknown type, nothing created
    LightComponent           light;
    std::vector<std::string> warnings;
};

/// Convert one light of the KHR_lights_punctual `lights` array.  Specification defaults: colour [1,1,1], intensity 1,
/// spot inner 0 and outer pi/4; range absent or <= 0 means infinite; a spot needs 0 <= inner < outer <= pi/2 (anything
/// else is clamped with a warning); a negative intensity is clamped to 0 with a warning.
[[nodiscard]] GltfPunctualLight convertPunctualLight(const tinygltf::Light& light, float photometricToEngine = 1.0f);

} // namespace phosphor
