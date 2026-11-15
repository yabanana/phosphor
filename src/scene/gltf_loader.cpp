// tinygltf implementation: define these before including the header.
// This translation unit also provides the stb_image implementation used by
// TextureManager, and lets tinygltf decode embedded images with it.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "scene/gltf_loader.h"
#include "scene/gltf_contained_fs.h"
#include "scene/gltf_material_extensions.h"
#include "scene/gltf_lights.h"
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
    lastError_.clear();
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    // Every external image and buffer URI resolves inside the asset directory
    // and is checked before anything is opened (review: the containment check
    // used to run after tinygltf had already read the files).
    ContainedFileSystem files(path);
    std::string fsError;
    if (!files.install(loader, fsError)) {
        lastError_ = "cannot install file-system callbacks: " + fsError;
        LOG_ERROR("glTF %s: %s", path.c_str(), lastError_.c_str());
        return false;
    }

    bool ok = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }

    if (!warn.empty()) {
        LOG_WARN("glTF warning (%s): %s", path.c_str(), warn.c_str());
    }
    if (!files.refusals().empty()) {
        lastError_ = files.refusals().front();
        LOG_ERROR("glTF %s: %s; refusing to load", path.c_str(), lastError_.c_str());
        return false;
    }
    if (!ok) {
        lastError_ = err;
        LOG_ERROR("Failed to load glTF: %s (%s)", path.c_str(), err.c_str());
        return false;
    }

    const size_t slash = path.find_last_of("/\\");
    baseDir_ = slash == std::string::npos ? std::string{} : path.substr(0, slash + 1);

    // glTF 2.0 section 3.2: an asset requiring an extension we do not support
    // SHOULD NOT be loaded.
    if (const auto required = unsupportedRequiredExtensions(model); !required.empty()) {
        std::string names;
        for (const auto& n : required) names += (names.empty() ? "" : ", ") + n.name + " (" + n.reason + ")";
        lastError_ = "requires extension(s) that cannot be honoured: " + names;
        LOG_ERROR("glTF %s %s; refusing to load", path.c_str(), lastError_.c_str());
        return false;
    }
    // Image URIs resolve only inside the asset directory.
    for (const auto& image : model.images) {
        std::string uriError;
        if (!image.uri.empty() && !isImageUriContained(baseDir_, image.uri, uriError)) {
            lastError_ = uriError;
            LOG_ERROR("glTF %s: %s; refusing to load", path.c_str(), uriError.c_str());
            return false;
        }
    }
    const auto unknownOptional = unknownOptionalExtensions(model);
    gpuScene_.addAssetUnknownExtensions(unknownOptional);

    // Clear per-file caches
    meshCache_.clear();
    createdEntities_.clear();
    importedLights_ = 0;
    materialCache_.clear();
    parsedMaterials_.clear();
    materialParseCount_ = 0;
    textureCache_.clear();
    // The documented texture defaults a material description falls back to when a
    // slot has no texture (also what an override that removes a texture binds).
    gpuScene_.setBaseTextureDefaults(textures_.getDefaultWhite(), textures_.getDefaultMR(),
                                     textures_.getDefaultNormal());

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

    LOG_INFO("%s", formatSupportSummary(gpuScene_.materialSupport(), gpuScene_.assetUnknownExtensions()).c_str());
    for (const auto& report : gpuScene_.materialSupport())
        if (report.valid()) LOG_DEBUG("%s", formatSupportDetail(report).c_str());

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

    // A draw has one material: keep each glTF primitive separate. Nodes
    // referencing the same mesh reuse the per-primitive geometry cache.
    if (node.mesh >= 0) {
        const auto& mesh = model.meshes.at(static_cast<size_t>(node.mesh));
        for (size_t p = 0; p < mesh.primitives.size(); ++p) {
            const MeshHandle handle = processPrimitive(model, node.mesh, p);
            if (handle == INVALID_MESH_HANDLE) continue;
            const EntityID entity = ecs_.createEntity();
            createdEntities_.push_back(entity);
            TransformComponent transform{};
            transform.worldMatrix = worldTransform;
            transform.position = glm::vec3(worldTransform[3]);
            // The exact affine matrix is authoritative; do not decompose a
            // reflected/sheared transform into a lossy rotation and scale.
            ecs_.addComponent(entity, std::move(transform));
            MeshInstanceComponent instance{};
            instance.meshHandle = handle;
            const int material = mesh.primitives[p].material;
            if (material >= 0) {
                instance.materialIndex = processMaterial(model, model.materials.at(static_cast<size_t>(material)));
            } else {
                // The implicit glTF default must not alias an unrelated
                // explicit material at index zero.
                constexpr int defaultKey = -1;
                const auto found = materialCache_.find(defaultKey);
                if (found != materialCache_.end()) instance.materialIndex = found->second;
                else {
                    GPUMaterial value{};
                    value.baseColor[0] = value.baseColor[1] = value.baseColor[2] = value.baseColor[3] = 1.0f;
                    value.metallic = value.roughness = value.normalScale = value.occlusionStrength = 1.0f;
                    value.baseColorTex = value.normalTex = value.metallicRoughnessTex =
                        value.occlusionTex = value.emissiveTex = INVALID_TEXTURE_INDEX;
                    instance.materialIndex = gpuScene_.addMaterial(value);
                    materialCache_[defaultKey] = instance.materialIndex;
                }
            }
            instance.setVisible(true);
            instance.setCastsShadows(true);
            ecs_.addComponent(entity, std::move(instance));
        }
    }

    // KHR_lights_punctual: the light sits at the node origin and shines along its local -Z (extractLights).
    if (importLights_ && node.light >= 0) {
        if (node.light >= static_cast<int>(model.lights.size())) {
            LOG_WARN("glTF node '%s' references light %d which does not exist", node.name.c_str(), node.light);
        } else {
            const auto converted = convertPunctualLight(model.lights[static_cast<size_t>(node.light)], photometricToEngine_);
            for (const std::string& w : converted.warnings) LOG_WARN("glTF node '%s': %s", node.name.c_str(), w.c_str());
            if (converted.valid) {
                if (converted.light.type == LightType::Spot && !spotFalloffNoted_) {
                    spotFalloffNoted_ = true;
                    const FeatureSupportEntry& e = currentSupportTable().punctualLights;
                    LOG_INFO("KHR_lights_punctual: %s (%s): spot cone falloff is smoothstep(cosOuter, cosInner), the informative formula of the extension is t^2",
                             supportLevelName(e.level), e.approximation);
                }
                const EntityID entity = ecs_.createEntity();
                createdEntities_.push_back(entity);
                TransformComponent transform{};
                transform.worldMatrix = worldTransform;
                transform.position = glm::vec3(worldTransform[3]);
                ecs_.addComponent(entity, std::move(transform));
                ecs_.addComponent(entity, LightComponent(converted.light));
                ++importedLights_;
            }
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

MeshHandle GltfLoader::processPrimitive(const tinygltf::Model& model,
                                        int meshIndex, size_t primitiveIndex) {
    const auto& mesh = model.meshes.at(static_cast<size_t>(meshIndex));
    const u64 key = (static_cast<u64>(meshIndex) << 32) | static_cast<u64>(primitiveIndex);
    const auto found = meshCache_.find(key);
    if (found != meshCache_.end()) return found->second;

    // Decode exactly one primitive, preserving its material and topology.
    const auto& prim = mesh.primitives.at(primitiveIndex);
    if (prim.mode != TINYGLTF_MODE_TRIANGLES && prim.mode != -1) {
        LOG_WARN("Skipping non-triangle primitive (mode=%d) in mesh '%s'",
                 prim.mode, mesh.name.c_str());
        return INVALID_MESH_HANDLE;
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
            return INVALID_MESH_HANDLE;
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

    // KHR_texture_transform and texCoord (F15.1 ticket 22).  The engine has ONE UV stream per mesh (UV0), so the UV set that every
    // texture of the primitive's material samples (glTF texCoord, usually 0) is copied into UV0 when it is another one; a material
    // that mixes UV sets is reported unsupported and keeps UV0.  A transform shared by every texture of the material is baked into
    // UV0 (affine maps commute with interpolation: exact); anything else (slots that differ, transformed and plain slots mixed) is
    // evaluated per slot on the GPU from the material's transform block and leaves the UVs untouched.
    if (prim.material >= 0 && prim.material < static_cast<int>(model.materials.size())) {
        const auto analysis = analyzeTextureTransforms(parsedMaterial(model, prim.material).description);
        if (analysis.status != TextureTransformStatus::Unsupported && analysis.uvSet != 0) {
            const std::string name = "TEXCOORD_" + std::to_string(analysis.uvSet);
            if (auto it = prim.attributes.find(name); it != prim.attributes.end()) {
                auto set = readAccessor<glm::vec2>(model, it->second);
                if (set.size() == vertexCount) {
                    geom.uvs = std::move(set);
                    hasUVs   = true;
                }
            }
        }
        if (hasUVs && analysis.status == TextureTransformStatus::Uniform) bakeUvTransform(analysis.transform, geom.uvs);
    }

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

    // KHR_materials_anisotropy needs a tangent frame at every vertex: supplied tangents that are degenerate (zero,
    // parallel to the normal, non-finite) and MikkTSpace results on degenerate UVs would shade it as strength 0.
    if (!geom.positions.empty() && prim.material >= 0 && prim.material < static_cast<int>(model.materials.size()) &&
        parsedMaterial(model, prim.material).description.anisotropy) {
        if (const u32 repaired = repairTangentFrames(geom); repaired > 0)
            LOG_WARN("Mesh '%s': %u vertices had no usable tangent frame, a perpendicular of the normal was substituted "
                     "(KHR_materials_anisotropy)", mesh.name.c_str(), repaired);
    }

    if (geom.positions.empty()) {
        meshCache_[key] = INVALID_MESH_HANDLE;
        return INVALID_MESH_HANDLE;
    }

    MeshHandle handle = gpuScene_.uploadMesh(geom.positions, geom.normals, geom.tangents, geom.uvs, geom.indices);
    meshCache_[key] = handle;
    return handle;
}

// ---------------------------------------------------------------------------
// Material processing
// ---------------------------------------------------------------------------

const GltfMaterialParse& GltfLoader::parsedMaterial(const tinygltf::Model& model, int index) {
    const auto found = parsedMaterials_.find(index);
    if (found != parsedMaterials_.end()) return found->second;
    ++materialParseCount_;
    // Texture slots resolve through the same cache as everything else: core
    // slots first, then the extension slots.
    GltfMaterialParse parsed = parseGltfMaterial(model.materials.at(static_cast<size_t>(index)),
                                                 [&](int textureIndex, bool sRGB) {
                                                     return processTexture(model, textureIndex, sRGB);
                                                 });
    return parsedMaterials_.emplace(index, std::move(parsed)).first->second;
}

u32 GltfLoader::processMaterial(const tinygltf::Model& model,
                                 const tinygltf::Material& material) {
    int matIdx = static_cast<int>(&material - model.materials.data());
    auto it = materialCache_.find(matIdx);
    if (it != materialCache_.end()) {
        return it->second;
    }

    // F15: the description of the material (core slots, extensions, texture
    // slots) is parsed once per material; KHR_materials_emissive_strength is
    // folded into the emissive factor.
    const GltfMaterialParse& parsed = parsedMaterial(model, matIdx);
    const BaseMaterial& base = parsed.description.base;
    for (const std::string& w : parsed.info.warnings)
        LOG_WARN("glTF material '%s': %s", material.name.c_str(), w.c_str());

    // A slot without a (resolvable) texture binds the documented default.
    auto textureOr = [](const MaterialTextureRef& ref, u32 fallback) {
        return ref.index != INVALID_TEXTURE_INDEX ? ref.index : fallback;
    };

    MaterialComponent mc{};
    mc.baseColorFactor = glm::vec4{base.baseColor[0], base.baseColor[1], base.baseColor[2], base.baseColor[3]};
    mc.metallicFactor  = base.metallic;
    mc.roughnessFactor = base.roughness;
    mc.baseColorTexIndex         = textureOr(base.baseColorTexture, textures_.getDefaultWhite());
    mc.metallicRoughnessTexIndex = textureOr(base.metallicRoughnessTexture, textures_.getDefaultMR());
    mc.normalTexIndex            = textureOr(base.normalTexture, textures_.getDefaultNormal());
    mc.normalScale               = base.normalScale;
    mc.occlusionTexIndex         = textureOr(base.occlusionTexture, textures_.getDefaultWhite());
    mc.occlusionStrength         = base.occlusionStrength;
    // glTF: emissive = factor * texture, with a missing texture meaning 1.
    mc.emissiveTexIndex          = textureOr(base.emissiveTexture, textures_.getDefaultWhite());
    // The description keeps an authored negative emissive strength; the GPU never gets a negative emission.
    mc.emissiveFactor = glm::max(glm::vec3{base.emissive[0], base.emissive[1], base.emissive[2]}, glm::vec3{0.0f});

    // Alpha cutoff (the description already applies it to MASK only)
    mc.alphaCutoff = base.alphaCutoff;
    if (material.alphaMode == "BLEND") {
        LOG_WARN("glTF BLEND material rendered opaque until F16 transparency is implemented");
    }
    mc.doubleSided = base.doubleSided;

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
    gpu.flags        = (mc.doubleSided ? MATERIAL_FLAG_DOUBLE_SIDED : 0u) | (base.unlit ? MATERIAL_FLAG_UNLIT : 0u);

    MaterialSupportReport support = buildSupportReport(parsed.description, parsed.info);
    const u32 materialIndex = gpuScene_.addMaterial(gpu, parsed.description, material.name, std::move(support));
    materialCache_[matIdx] = materialIndex;

    return materialIndex;
}

// ---------------------------------------------------------------------------
// Texture processing
// ---------------------------------------------------------------------------

u32 GltfLoader::processTexture(const tinygltf::Model& model, int textureIndex,
                                bool sRGB) {
    // Unresolvable textures (index out of range, no source image) are "no texture".
    if (textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size())) {
        return INVALID_TEXTURE_INDEX;
    }

    const u64 key = (static_cast<u64>(textureIndex) << 1) | (sRGB ? 1u : 0u);
    auto it = textureCache_.find(key);
    if (it != textureCache_.end()) {
        return it->second;
    }

    const auto& tex = model.textures[static_cast<size_t>(textureIndex)];
    if (tex.source < 0 || tex.source >= static_cast<int>(model.images.size())) {
        return INVALID_TEXTURE_INDEX;
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

    textureCache_[key] = bindlessIndex;
    return bindlessIndex;
}

} // namespace phosphor
