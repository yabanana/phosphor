#include "renderer/gpu_scene.h"
#include "core/log.h"

#include <cassert>
#include <cmath>

namespace phosphor {

MeshHandle GpuScene::uploadMesh(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals,
    const std::vector<glm::vec4>& tangents,
    const std::vector<glm::vec2>& uvs,
    const std::vector<u32>& indices) {

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
}

u32 GpuScene::addMaterial(const GPUMaterial& material) {
    materials_.push_back(material);
    return static_cast<u32>(materials_.size() - 1);
}

void GpuScene::clear() {
    vertices_.clear();
    indices_.clear();
    meshlets_.clear();
    meshletVertices_.clear();
    meshletTriangles_.clear();
    meshletBounds_.clear();
    meshInfos_.clear();
    instances_.clear();
    materials_.clear();
    lights_.clear();
    ++geometryVersion_;
}

} // namespace phosphor
