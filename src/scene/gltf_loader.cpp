// tinygltf implementation: define these before including the header.
// This translation unit also provides the stb_image implementation used by
// TextureManager, and lets tinygltf decode embedded images with it.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "scene/gltf_loader.h"
#include "scene/mesh_processing.h"
#include "scene/texture_manager.h"
#include "scene/ecs.h"
#include "scene/components.h"
#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cassert>

namespace phosphor {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read a typed accessor into a contiguous vector.
template <typename T>
static std::vector<T> readAccessor(const tinygltf::Model& model, int accessorIdx) {
    if (accessorIdx < 0) {
        return {};
    }
    const auto& accessor   = model.accessors[static_cast<size_t>(accessorIdx)];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer     = model.buffers[bufferView.buffer];

    const u8* base = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    size_t stride  = bufferView.byteStride;
    if (stride == 0) {
        stride = sizeof(T);
    }

    std::vector<T> result(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        std::memcpy(&result[i], base + i * stride, sizeof(T));
    }
    return result;
}

/// Build a glm::mat4 from a glTF node's TRS or matrix fields.
static glm::mat4 nodeTransform(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        // Column-major, same as glm
        return glm::make_mat4(node.matrix.data());
    }

    glm::mat4 T{1.0f};
    glm::mat4 R{1.0f};
    glm::mat4 S{1.0f};

    if (node.translation.size() == 3) {
        T = glm::translate(glm::mat4{1.0f},
            glm::vec3{static_cast<float>(node.translation[0]),
                       static_cast<float>(node.translation[1]),
                       static_cast<float>(node.translation[2])});
    }
    if (node.rotation.size() == 4) {
        glm::quat q{static_cast<float>(node.rotation[3]),
                     static_cast<float>(node.rotation[0]),
                     static_cast<float>(node.rotation[1]),
                     static_cast<float>(node.rotation[2])};
        R = glm::mat4_cast(q);
    }
    if (node.scale.size() == 3) {
        S = glm::scale(glm::mat4{1.0f},
            glm::vec3{static_cast<float>(node.scale[0]),
                       static_cast<float>(node.scale[1]),
                       static_cast<float>(node.scale[2])});
    }

    return T * R * S;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

GltfLoader::GltfLoader(GpuScene& gpuScene, TextureManager& textures, ECS& ecs)
    : gpuScene_(gpuScene)
    , textures_(textures)
    , ecs_(ecs)
{
}

// ---------------------------------------------------------------------------
// loadFromFile
// ---------------------------------------------------------------------------

bool GltfLoader::loadFromFile(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }

    if (!warn.empty()) {
        LOG_WARN("glTF warning (%s): %s", path.c_str(), warn.c_str());
    }
    if (!ok) {
        LOG_ERROR("Failed to load glTF: %s (%s)", path.c_str(), err.c_str());
        return false;
    }

    const size_t slash = path.find_last_of("/\\");
    baseDir_ = slash == std::string::npos ? std::string{} : path.substr(0, slash + 1);

    // Clear per-file caches
    meshCache_.clear();
    materialCache_.clear();
    textureCache_.clear();

    // Process the default scene (or scene 0)
    int sceneIdx = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIdx >= static_cast<int>(model.scenes.size())) {
        LOG_ERROR("glTF has no scenes");
        return false;
    }

    const auto& scene = model.scenes[static_cast<size_t>(sceneIdx)];
    for (int nodeIdx : scene.nodes) {
        processNode(model, model.nodes[static_cast<size_t>(nodeIdx)], glm::mat4{1.0f});
    }

    // Flush uploaded mesh data to the GPU
    gpuScene_.flushMeshData();

    LOG_INFO("Loaded glTF: %s (%zu meshes, %zu materials, %zu textures)",
             path.c_str(), model.meshes.size(), model.materials.size(),
             model.textures.size());
    return true;
}

// ---------------------------------------------------------------------------
// Node processing (recursive)
// ---------------------------------------------------------------------------

void GltfLoader::processNode(const tinygltf::Model& model,
                              const tinygltf::Node& node,
                              glm::mat4 parentTransform) {
    glm::mat4 localTransform = nodeTransform(node);
    glm::mat4 worldTransform = parentTransform * localTransform;

    // If this node has a mesh, create an entity for each primitive
    if (node.mesh >= 0) {
        MeshHandle meshHandle = processMesh(model,
            model.meshes[static_cast<size_t>(node.mesh)]);

        if (meshHandle != INVALID_MESH_HANDLE) {
            EntityID entity = ecs_.createEntity();

            // TransformComponent
            TransformComponent tc{};
            tc.worldMatrix = worldTransform;
            // Decompose for editing (approximate; non-uniform shear may lose data)
            tc.position = glm::vec3(worldTransform[3]);
            tc.scale = glm::vec3(
                glm::length(glm::vec3(worldTransform[0])),
                glm::length(glm::vec3(worldTransform[1])),
                glm::length(glm::vec3(worldTransform[2])));
            glm::mat3 rotMat(
                glm::vec3(worldTransform[0]) / tc.scale.x,
                glm::vec3(worldTransform[1]) / tc.scale.y,
                glm::vec3(worldTransform[2]) / tc.scale.z);
            tc.rotation = glm::quat_cast(rotMat);
            ecs_.addComponent(entity, std::move(tc));

            // Determine material index for the first primitive
            u32 materialIdx = 0;
            const auto& gltfMesh = model.meshes[static_cast<size_t>(node.mesh)];
            if (!gltfMesh.primitives.empty() && gltfMesh.primitives[0].material >= 0) {
                materialIdx = processMaterial(model,
                    model.materials[static_cast<size_t>(gltfMesh.primitives[0].material)]);
            }

            // MeshInstanceComponent
            MeshInstanceComponent mic{};
            mic.meshHandle = meshHandle;
            mic.materialIndex = materialIdx;
            mic.setVisible(true);
            mic.setCastsShadows(true);
            ecs_.addComponent(entity, std::move(mic));
        }
    }

    // Recurse into children
    for (int childIdx : node.children) {
        processNode(model, model.nodes[static_cast<size_t>(childIdx)], worldTransform);
    }
}

// ---------------------------------------------------------------------------
// Mesh processing
// ---------------------------------------------------------------------------

MeshHandle GltfLoader::processMesh(const tinygltf::Model& model,
                                    const tinygltf::Mesh& mesh) {
    // Use the mesh's index in the model as cache key
    int meshIdx = static_cast<int>(&mesh - model.meshes.data());
    auto it = meshCache_.find(meshIdx);
    if (it != meshCache_.end()) {
        return it->second;
    }

    // Accumulate geometry from all primitives in this mesh
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> uvs;
    std::vector<u32> indices;

    for (const auto& prim : mesh.primitives) {
        if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1) {
            LOG_WARN("Skipping non-triangle primitive (mode=%d) in mesh '%s'",
                     prim.mode, mesh.name.c_str());
            continue;
        }

        // Read the primitive into local streams (indices local to it), fill
        // what glTF allows to omit, then append to the mesh.
        GeometryStreams geom;
        bool hasNormals = false, hasTangents = false, hasUVs = false;

        // Positions (required)
        {
            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end()) {
                LOG_WARN("Primitive in mesh '%s' has no POSITION attribute, skipping",
                         mesh.name.c_str());
                continue;
            }
            geom.positions = readAccessor<glm::vec3>(model, posIt->second);
        }
        const u32 vertexCount = static_cast<u32>(geom.positions.size());

        // Normals (optional: flat normals are generated, as the spec requires)
        if (auto it = prim.attributes.find("NORMAL"); it != prim.attributes.end()) {
            geom.normals = readAccessor<glm::vec3>(model, it->second);
            hasNormals = geom.normals.size() == vertexCount;
        }
        // Tangents (optional: MikkTSpace, as the spec recommends)
        if (auto it = prim.attributes.find("TANGENT"); it != prim.attributes.end()) {
            geom.tangents = readAccessor<glm::vec4>(model, it->second);
            hasTangents = geom.tangents.size() == vertexCount;
        }
        // UVs (optional)
        if (auto it = prim.attributes.find("TEXCOORD_0"); it != prim.attributes.end()) {
            geom.uvs = readAccessor<glm::vec2>(model, it->second);
            hasUVs = geom.uvs.size() == vertexCount;
        }
        if (!hasUVs) geom.uvs.assign(vertexCount, glm::vec2{0.0f});

        // Indices (generate sequential ones for non-indexed primitives)
        if (prim.indices >= 0) {
            const auto& accessor = model.accessors[static_cast<size_t>(prim.indices)];
            const auto& bufView  = model.bufferViews[accessor.bufferView];
            const auto& buf      = model.buffers[bufView.buffer];
            const u8* base       = buf.data.data() + bufView.byteOffset + accessor.byteOffset;

            geom.indices.reserve(accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                u32 idx = 0;
                switch (accessor.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        uint16_t v = 0;
                        std::memcpy(&v, base + i * sizeof(uint16_t), sizeof(uint16_t));
                        idx = v;
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                        std::memcpy(&idx, base + i * sizeof(u32), sizeof(u32));
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        idx = base[i];
                        break;
                    }
                    default:
                        LOG_WARN("Unsupported index component type: %d", accessor.componentType);
                        idx = 0;
                        break;
                }
                geom.indices.push_back(idx < vertexCount ? idx : 0);
            }
        } else {
            for (u32 i = 0; i < vertexCount; ++i) geom.indices.push_back(i);
        }

        if (!hasNormals || !hasTangents) {
            completeGeometry(geom, hasNormals, hasTangents, hasUVs);
            LOG_INFO("Mesh '%s': generated%s%s (%zu vertices)", mesh.name.c_str(),
                     hasNormals ? "" : " flat normals", hasTangents ? "" : " MikkTSpace tangents",
                     geom.positions.size());
        }

        const u32 vertexBase = static_cast<u32>(positions.size());
        positions.insert(positions.end(), geom.positions.begin(), geom.positions.end());
        normals.insert(normals.end(), geom.normals.begin(), geom.normals.end());
        tangents.insert(tangents.end(), geom.tangents.begin(), geom.tangents.end());
        uvs.insert(uvs.end(), geom.uvs.begin(), geom.uvs.end());
        for (const u32 idx : geom.indices) indices.push_back(vertexBase + idx);
    }

    if (positions.empty()) {
        meshCache_[meshIdx] = INVALID_MESH_HANDLE;
        return INVALID_MESH_HANDLE;
    }

    MeshHandle handle = gpuScene_.uploadMesh(positions, normals, tangents, uvs, indices);
    meshCache_[meshIdx] = handle;
    return handle;
}

// ---------------------------------------------------------------------------
// Material processing
// ---------------------------------------------------------------------------

u32 GltfLoader::processMaterial(const tinygltf::Model& model,
                                 const tinygltf::Material& material) {
    int matIdx = static_cast<int>(&material - model.materials.data());
    auto it = materialCache_.find(matIdx);
    if (it != materialCache_.end()) {
        return it->second;
    }

    MaterialComponent mc{};

    // PBR metallic-roughness
    const auto& pbr = material.pbrMetallicRoughness;

    mc.baseColorFactor = glm::vec4{
        static_cast<float>(pbr.baseColorFactor[0]),
        static_cast<float>(pbr.baseColorFactor[1]),
        static_cast<float>(pbr.baseColorFactor[2]),
        static_cast<float>(pbr.baseColorFactor[3])
    };
    mc.metallicFactor  = static_cast<float>(pbr.metallicFactor);
    mc.roughnessFactor = static_cast<float>(pbr.roughnessFactor);

    // Base color texture (sRGB)
    if (pbr.baseColorTexture.index >= 0) {
        mc.baseColorTexIndex = processTexture(model, pbr.baseColorTexture.index, true);
    } else {
        mc.baseColorTexIndex = textures_.getDefaultWhite();
    }

    // Metallic-roughness texture (linear)
    if (pbr.metallicRoughnessTexture.index >= 0) {
        mc.metallicRoughnessTexIndex = processTexture(model, pbr.metallicRoughnessTexture.index, false);
    } else {
        mc.metallicRoughnessTexIndex = textures_.getDefaultMR();
    }

    // Normal map (linear)
    if (material.normalTexture.index >= 0) {
        mc.normalTexIndex = processTexture(model, material.normalTexture.index, false);
        mc.normalScale = static_cast<float>(material.normalTexture.scale);
    } else {
        mc.normalTexIndex = textures_.getDefaultNormal();
        mc.normalScale = 1.0f;
    }

    // Occlusion (linear)
    if (material.occlusionTexture.index >= 0) {
        mc.occlusionTexIndex = processTexture(model, material.occlusionTexture.index, false);
        mc.occlusionStrength = static_cast<float>(material.occlusionTexture.strength);
    } else {
        mc.occlusionTexIndex = textures_.getDefaultWhite();
        mc.occlusionStrength = 1.0f;
    }

    // Emissive (sRGB)
    if (material.emissiveTexture.index >= 0) {
        mc.emissiveTexIndex = processTexture(model, material.emissiveTexture.index, true);
    } else {
        // glTF: emissive = factor * texture, with a missing texture meaning 1.
        mc.emissiveTexIndex = textures_.getDefaultWhite();
    }
    mc.emissiveFactor = glm::vec3{
        static_cast<float>(material.emissiveFactor[0]),
        static_cast<float>(material.emissiveFactor[1]),
        static_cast<float>(material.emissiveFactor[2])
    };

    // Alpha cutoff
    mc.alphaCutoff = static_cast<float>(material.alphaCutoff);
    mc.doubleSided = material.doubleSided;

    // Build GPUMaterial from component and upload
    GPUMaterial gpu{};
    gpu.baseColor[0]  = mc.baseColorFactor.r;
    gpu.baseColor[1]  = mc.baseColorFactor.g;
    gpu.baseColor[2]  = mc.baseColorFactor.b;
    gpu.baseColor[3]  = mc.baseColorFactor.a;
    gpu.metallic      = mc.metallicFactor;
    gpu.roughness     = mc.roughnessFactor;
    gpu.normalScale   = mc.normalScale;
    gpu.occlusionStrength = mc.occlusionStrength;
    gpu.baseColorTex         = mc.baseColorTexIndex;
    gpu.normalTex            = mc.normalTexIndex;
    gpu.metallicRoughnessTex = mc.metallicRoughnessTexIndex;
    gpu.occlusionTex         = mc.occlusionTexIndex;
    gpu.emissiveTex          = mc.emissiveTexIndex;
    gpu.emissive[0]  = mc.emissiveFactor.r;
    gpu.emissive[1]  = mc.emissiveFactor.g;
    gpu.emissive[2]  = mc.emissiveFactor.b;
    gpu.alphaCutoff  = mc.alphaCutoff;
    gpu.flags        = mc.doubleSided ? MATERIAL_FLAG_DOUBLE_SIDED : 0u;

    const u32 materialIndex = gpuScene_.addMaterial(gpu);
    materialCache_[matIdx] = materialIndex;

    return materialIndex;
}

// ---------------------------------------------------------------------------
// Texture processing
// ---------------------------------------------------------------------------

u32 GltfLoader::processTexture(const tinygltf::Model& model, int textureIndex,
                                bool sRGB) {
    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size())) {
        return textures_.getDefaultWhite();
    }

    auto it = textureCache_.find(textureIndex);
    if (it != textureCache_.end()) {
        return it->second;
    }

    const auto& tex = model.textures[static_cast<size_t>(textureIndex)];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size())) {
        return textures_.getDefaultWhite();
    }

    const auto& image = model.images[static_cast<size_t>(tex.source)];

    u32 bindlessIndex = 0;
    if (!image.image.empty()) {
        // Image data is already decoded by tinygltf
        bindlessIndex = textures_.loadTextureFromMemory(
            image.image.data(),
            static_cast<u32>(image.width),
            static_cast<u32>(image.height),
            static_cast<u32>(image.component),
            sRGB);
    } else if (!image.uri.empty()) {
        // External file reference, relative to the glTF file
        bindlessIndex = textures_.loadTexture(baseDir_ + image.uri, sRGB);
    } else {
        bindlessIndex = textures_.getDefaultWhite();
    }

    textureCache_[textureIndex] = bindlessIndex;
    return bindlessIndex;
}

} // namespace phosphor
