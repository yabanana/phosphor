#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace phosphor {

MeshHandle GpuScene::uploadMesh(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec4>& tangents,
    const std::vector<glm::vec2>& uvs,
    const std::vector<u32>& indices,
    const std::vector<glm::vec4>& colors) {

    assert(!positions.empty());
    assert(indices.size() % 3 == 0);

    const size_t vertexCount = positions.size();
    const bool hasNormals  = normals.size() == vertexCount;
    const bool hasTangents = tangents.size() == vertexCount;
    const bool hasUVs      = uvs.size() == vertexCount;

    const u32 vertexOffset        = static_cast<u32>(vertices_.size());
    const u32 indexOffset         = static_cast<u32>(indices_.size());
    const u32 meshletOffset       = static_cast<u32>(meshlets_.size());
    const u32 meshletVertexOffset = static_cast<u32>(meshletVertices_.size());
    const u32 meshletTriOffset    = static_cast<u32>(meshletTriangles_.size());

    vertices_.reserve(vertices_.size() + vertexCount);
    // COLOR_0: the stream appears with the first coloured mesh (earlier vertices become white) and then stays parallel to vertices_.
    const bool coloured = colors.size() == vertexCount && !colors.empty();
    if (coloured || !vertexColors_.empty()) {
        if (vertexColors_.empty()) vertexColors_.assign(vertices_.size(), glm::vec4(1.0f));
        if (coloured) vertexColors_.insert(vertexColors_.end(), colors.begin(), colors.end());
        else vertexColors_.insert(vertexColors_.end(), vertexCount, glm::vec4(1.0f));
    }
    for (size_t i = 0; i < vertexCount; ++i) {
        GPUVertex v{};
        v.px = positions[i].x;
        v.py = positions[i].y;
        v.pz = positions[i].z;

        const glm::vec3 n = hasNormals ? normals[i] : glm::vec3(0.0f, 1.0f, 0.0f);
        v.nx = n.x;
        v.ny = n.y;
        v.nz = n.z;

        const glm::vec4 t = hasTangents ? tangents[i] : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        v.tx = t.x;
        v.ty = t.y;
        v.tz = t.z;
        v.tw = t.w;

        const glm::vec2 uv = hasUVs ? uvs[i] : glm::vec2(0.0f);
        v.u = uv.x;
        v.v = uv.y;
        vertices_.push_back(v);
    }

    indices_.insert(indices_.end(), indices.begin(), indices.end());

    MeshletBuildResult meshletData = MeshletBuilder::build(
        reinterpret_cast<const float*>(positions.data()),
        vertexCount, sizeof(glm::vec3),
        indices.data(), indices.size(), meshletOptions_);

    // Bounding sphere around the vertex centroid.
    glm::vec3 center{0.0f};
    for (const auto& p : positions) center += p;
    center /= static_cast<float>(vertexCount);
    float radiusSq = 0.0f;
    for (const auto& p : positions) {
        const glm::vec3 d = p - center;
        radiusSq = std::max(radiusSq, glm::dot(d, d));
    }

    // Meshlet vertex indices are mesh-local; make them global.
    for (u32 v : meshletData.meshletVertices) {
        meshletVertices_.push_back(v + vertexOffset);
    }
    meshletTriangles_.insert(meshletTriangles_.end(),
                             meshletData.meshletTriangles.begin(),
                             meshletData.meshletTriangles.end());
    for (const auto& m : meshletData.meshlets) {
        Meshlet g = m;
        g.vertexOffset   += meshletVertexOffset;
        g.triangleOffset += meshletTriOffset;
        meshlets_.push_back(g);
    }
    meshletBounds_.insert(meshletBounds_.end(),
                          meshletData.bounds.begin(), meshletData.bounds.end());

    GPUMeshInfo info{};
    info.meshletCount  = static_cast<u32>(meshletData.meshlets.size());
    info.meshletOffset = meshletOffset;
    info.vertexOffset  = vertexOffset;
    info.indexOffset   = indexOffset;
    info.indexCount    = static_cast<u32>(indices.size());
    info.boundingSphere[0] = center.x;
    info.boundingSphere[1] = center.y;
    info.boundingSphere[2] = center.z;
    info.boundingSphere[3] = std::sqrt(radiusSq);
    meshInfos_.push_back(info);
    MeshLocalBounds box{};
    if (!positions.empty()) box = {positions.front(), positions.front()};
    for (const auto& p : positions) { box.minimum = glm::min(box.minimum, p); box.maximum = glm::max(box.maximum, p); }
    meshBounds_.push_back(box);

    ++geometryVersion_;

    const MeshHandle handle = static_cast<MeshHandle>(meshInfos_.size() - 1);
    LOG_INFO("Uploaded mesh %u: %zu vertices, %zu meshlets",
             handle, vertexCount, meshletData.meshlets.size());
    return handle;
}

void GpuScene::updateInstances(const std::vector<GPUInstance>& instances) {
    instances_ = instances;
}

void GpuScene::updateLights(const std::vector<GPULight>& lights) {
    lights_ = lights;
}

void GpuScene::updateMaterials(const std::vector<GPUMaterial>& materials) {
    materials_ = materials;
    // The library was replaced: descriptions of the old one no longer apply.
    materialDescriptions_.assign(materials_.size(), MaterialDescription{});
    materialNames_.assign(materials_.size(), std::string{});
    materialSupport_.assign(materials_.size(), MaterialSupportReport{});
    materialByName_.clear();
}

u32 GpuScene::addMaterial(const GPUMaterial& material) {
    materials_.push_back(material);
    materialDescriptions_.emplace_back();
    materialNames_.emplace_back();
    materialSupport_.emplace_back();
    return static_cast<u32>(materials_.size() - 1);
}

u32 GpuScene::addMaterial(const GPUMaterial& material, MaterialDescription description, std::string name,
                          MaterialSupportReport support) {
    materials_.push_back(material);
    const u32 index = static_cast<u32>(materials_.size() - 1);
    if (!name.empty()) materialByName_.emplace(name, index); // first one wins
    materialDescriptions_.push_back(std::move(description));
    materialNames_.push_back(std::move(name));
    materialSupport_.push_back(std::move(support));
    return index;
}

std::optional<u32> GpuScene::findMaterial(const std::string& name) const {
    const auto it = materialByName_.find(name);
    if (it == materialByName_.end()) return std::nullopt;
    return it->second;
}

void GpuScene::addAssetUnknownExtensions(const std::vector<std::string>& names) {
    for (const std::string& n : names)
        if (std::find(assetUnknownExtensions_.begin(), assetUnknownExtensions_.end(), n) == assetUnknownExtensions_.end())
            assetUnknownExtensions_.push_back(n);
}

void GpuScene::setBaseTextureDefaults(u32 white, u32 metallicRoughness, u32 normal) {
    baseTextureDefaults_ = {white, metallicRoughness, normal};
}

bool GpuScene::setMaterialDescription(u32 index, const MaterialDescription& authored) {
    if (index >= materials_.size()) return false;
    MaterialDescription d = authored;
    lowerUnlit(d);
    GPUMaterial& g = materials_[index];
    const BaseMaterial& b = d.base;
    for (size_t i = 0; i < 4; ++i) g.baseColor[i] = b.baseColor[i];
    g.metallic          = b.metallic;
    g.roughness         = b.roughness;
    g.normalScale       = b.normalScale;
    g.occlusionStrength = b.occlusionStrength;
    // The description keeps an authored negative emission; the GPU never gets one.
    for (size_t i = 0; i < 3; ++i) g.emissive[i] = std::max(0.0f, b.emissive[i]);
    g.alphaCutoff = b.alphaCutoff;
    g.flags = b.doubleSided ? (g.flags | MATERIAL_FLAG_DOUBLE_SIDED) : (g.flags & ~MATERIAL_FLAG_DOUBLE_SIDED);
    g.flags = b.unlit ? (g.flags | MATERIAL_FLAG_UNLIT) : (g.flags & ~MATERIAL_FLAG_UNLIT);
    // A slot with a texture binds it; one without binds the documented default
    // (an override that removes a texture must not leave the old one bound); a
    // path that is not resolved yet keeps what is bound.
    auto slot = [](const MaterialTextureRef& ref, u32 current, u32 fallback) {
        if (!ref.present()) return fallback;
        return ref.index != INVALID_TEXTURE_INDEX ? ref.index : current;
    };
    g.baseColorTex         = slot(b.baseColorTexture, g.baseColorTex, baseTextureDefaults_.white);
    g.metallicRoughnessTex = slot(b.metallicRoughnessTexture, g.metallicRoughnessTex, baseTextureDefaults_.metallicRoughness);
    g.normalTex            = slot(b.normalTexture, g.normalTex, baseTextureDefaults_.normal);
    g.occlusionTex         = slot(b.occlusionTexture, g.occlusionTex, baseTextureDefaults_.white);
    g.emissiveTex          = slot(b.emissiveTexture, g.emissiveTex, baseTextureDefaults_.white);
    materialDescriptions_[index] = d;
    ++materialEditVersion_;

    // The support entry follows the new description.
    MaterialSourceInfo info;
    info.name              = materialNames_[index];
    info.unknownExtensions = materialSupport_[index].unknownExtensions;
    for (const FeatureSupport& e : materialSupport_[index].encodings)
        if (e.feature == "emissive_strength") info.emissiveStrengthUsed = true;
    {
        const auto analysis = analyzeTextureTransforms(d);
        info.transformStatus = analysis.status;
        info.transformReason = analysis.reason;
    }
    info.warnings        = validationWarnings(d);
    materialSupport_[index] = buildSupportReport(d, info);
    return true;
}

void GpuScene::clear() {
    vertices_.clear();
    vertexColors_.clear();
    indices_.clear();
    meshlets_.clear();
    meshletVertices_.clear();
    meshletTriangles_.clear();
    meshletBounds_.clear();
    meshInfos_.clear();
    meshBounds_.clear();
    instances_.clear();
    materials_.clear();
    materialDescriptions_.clear();
    materialNames_.clear();
    materialSupport_.clear();
    materialByName_.clear();
    assetUnknownExtensions_.clear();
    baseTextureDefaults_ = {};
    materialLibrarySealed_ = false;
    lights_.clear();
    sampledLights_.clear();
    ++geometryVersion_;
}

} // namespace phosphor
