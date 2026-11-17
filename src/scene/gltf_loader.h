#pragma once

#include "core/types.h"
#include "renderer/gpu_scene.h"
#include "scene/gltf_material_extensions.h"
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <span>

// Forward-declare tinygltf types to avoid pulling the header into every TU
namespace tinygltf {
class Model;
class TinyGLTF;
struct Node;
struct Mesh;
struct Material;
} // namespace tinygltf

namespace phosphor {

class GpuScene;
class TextureManager;
class ECS;

class GltfLoader {
public:
    GltfLoader(GpuScene& gpuScene, TextureManager& textures, ECS& ecs);

    /// Load a glTF 2.0 file (.gltf or .glb).
    /// Processes the full node hierarchy, uploading meshes, materials, and
    /// textures, and creating ECS entities with Transform + MeshInstance +
    /// Material components.
    /// @return true on success
    bool loadFromFile(const std::string& path);
    [[nodiscard]] std::span<const EntityID> createdEntities() const { return createdEntities_; }
    /// KHR_lights_punctual: create LightComponent entities for the file's lights (default on).  A declared
    /// lighting rig (the corpus rig) turns it off so every model of a corpus is lit by the same lights.
    void setImportPunctualLights(bool on) { importLights_ = on; }
    /// Number of punctual lights imported by the last load.
    [[nodiscard]] u32 importedLightCount() const { return importedLights_; }
    /// Engine light intensity = glTF intensity * this (lux -> W/m^2 and candela -> W/sr; default 1, see gltf_lights.h).
    void setPhotometricToEngine(float scale) { photometricToEngine_ = scale; }
    /// Why the last loadFromFile() failed (empty after a success).
    [[nodiscard]] const std::string& lastError() const { return lastError_; }
    /// How many glTF materials were parsed in the last load: each material is
    /// parsed once, however many primitives use it.
    [[nodiscard]] size_t materialParseCount() const { return materialParseCount_; }

private:
    void processNode(const tinygltf::Model& model, const tinygltf::Node& node,
                     glm::mat4 parentTransform);
    MeshHandle processPrimitive(const tinygltf::Model& model, int meshIndex, size_t primitiveIndex);
    u32  processMaterial(const tinygltf::Model& model, const tinygltf::Material& material);
    /// Parsed description of glTF material `index`, cached per file.
    const GltfMaterialParse& parsedMaterial(const tinygltf::Model& model, int index);
    u32  processTexture(const tinygltf::Model& model, int textureIndex, bool sRGB);

    GpuScene&       gpuScene_;
    TextureManager& textures_;
    ECS&            ecs_;
    std::string     lastError_;
    std::string     baseDir_; // directory of the file being loaded, with trailing slash

    // Caches to avoid reprocessing the same glTF object twice within a single file
    std::unordered_map<u64, MeshHandle> meshCache_;
    std::vector<EntityID> createdEntities_;
    std::unordered_map<int, u32>        materialCache_;
    std::unordered_map<int, GltfMaterialParse> parsedMaterials_;
    size_t                              materialParseCount_ = 0;
    std::unordered_map<u64, u32>        textureCache_;
    bool                                importLights_ = true;
    u32                                 importedLights_ = 0;
    bool                                spotFalloffNoted_ = false;
    float                               photometricToEngine_ = 1.0f;
};

} // namespace phosphor
