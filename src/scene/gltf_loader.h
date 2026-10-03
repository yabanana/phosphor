#pragma once

#include "core/types.h"
#include "renderer/gpu_scene.h"
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

private:
    void processNode(const tinygltf::Model& model, const tinygltf::Node& node,
                     glm::mat4 parentTransform);
    MeshHandle processPrimitive(const tinygltf::Model& model, int meshIndex, size_t primitiveIndex);
    u32  processMaterial(const tinygltf::Model& model, const tinygltf::Material& material);
    u32  processTexture(const tinygltf::Model& model, int textureIndex, bool sRGB);

    GpuScene&       gpuScene_;
    TextureManager& textures_;
    ECS&            ecs_;
    std::string     baseDir_; // directory of the file being loaded, with trailing slash

    // Caches to avoid reprocessing the same glTF object twice within a single file
    std::unordered_map<u64, MeshHandle> meshCache_;
    std::vector<EntityID> createdEntities_;
    std::unordered_map<int, u32>        materialCache_;
    std::unordered_map<u64, u32>        textureCache_;
};

} // namespace phosphor
