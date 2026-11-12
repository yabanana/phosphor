#pragma once
#include <glm/glm.hpp>
namespace phosphor {
// F13.5: mesh-local axis-aligned bounds of the uploaded positions (CPU only).
// Reflection probes fit their projection box to the world AABB of these
// boxes; the culling bounding sphere is far too loose for a room-sized box.
struct MeshLocalBounds { glm::vec3 minimum{0}, maximum{0}; };
} // namespace phosphor
