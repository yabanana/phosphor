#pragma once

// F15.2 (ticket 21): exports the feature-class manifest of a test bench scene (renderer/material_class_manifest.h).
// Portable and GPU-free: the bench is set up on an ECS, the persistent SceneStore syncs it exactly like the engine
// does, and the live materials' feature masks are read from it.

#include "renderer/material_class_manifest.h"
#include "renderer/scene_store.h"

#include <string>
#include <string_view>
#include <vector>

namespace phosphor::matclass {

/// The entry of a scene held by `store` (every live material is one class member).
[[nodiscard]] SceneEntry entryFromStore(std::string name, std::string family, const SceneStore& store);

/// The fixture families of the F15 benches: "material" (MaterialValidation::scenarios()), "transmission" and "shadow".
[[nodiscard]] std::vector<std::string> fixtureFamilies();
/// Scenarios of a family, in declaration order.
[[nodiscard]] std::vector<std::string> fixtureScenarios(std::string_view family);
/// Sets the fixture bench up (throws std::invalid_argument for an unknown family or scenario), syncs a SceneStore and
/// returns its entry named "<family>:<scenario>".
[[nodiscard]] SceneEntry exportFixtureScene(std::string_view family, std::string_view scenario);
/// One manifest per family (scenes in declaration order).
[[nodiscard]] Manifest exportFixtureFamily(std::string_view family);

} // namespace phosphor::matclass
