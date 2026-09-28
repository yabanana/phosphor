#include <doctest/doctest.h>

#include "renderer/gpu_scene.h"
#include "scene/procedural.h"

using namespace phosphor;

namespace {

MeshHandle upload(GpuScene& scene, const MeshData& m) {
    return scene.uploadMesh(m.positions, m.normals, m.tangents, m.uvs, m.indices);
}

} // namespace

TEST_CASE("uploadMesh records per-mesh offsets into the global arrays") {
    GpuScene scene;
    const MeshData cube   = ProceduralMeshes::generateCube(1.0f);
    const MeshData sphere = ProceduralMeshes::generateSphere(1.0f, 16, 8);

    const u64 v0 = scene.geometryVersion();
    const MeshHandle a = upload(scene, cube);
    const MeshHandle b = upload(scene, sphere);
    CHECK(scene.geometryVersion() > v0);

    REQUIRE(a == 0);
    REQUIRE(b == 1);
    REQUIRE(scene.getMeshCount() == 2);

    const GPUMeshInfo& ia = scene.meshInfos()[a];
    const GPUMeshInfo& ib = scene.meshInfos()[b];
    CHECK(ia.vertexOffset == 0);
    CHECK(ia.indexOffset == 0);
    CHECK(ia.indexCount == cube.indices.size());
    CHECK(ib.vertexOffset == cube.positions.size());
    CHECK(ib.indexOffset == cube.indices.size());
    CHECK(ib.indexCount == sphere.indices.size());
    CHECK(ib.meshletOffset == ia.meshletCount);
    CHECK(scene.vertices().size() == cube.positions.size() + sphere.positions.size());
    CHECK(scene.indices().size() == cube.indices.size() + sphere.indices.size());
}

TEST_CASE("meshlet data stays inside the global buffers") {
    GpuScene scene;
    upload(scene, ProceduralMeshes::generateTorus(1.5f, 0.5f, 32, 16));
    upload(scene, ProceduralMeshes::generatePlane(4.0f, 4.0f, 8, 8));

    const auto& meshlets = scene.meshlets();
    REQUIRE(!meshlets.empty());
    CHECK(scene.meshletBounds().size() == meshlets.size());

    for (const GPUMeshInfo& info : scene.meshInfos()) {
        for (u32 m = info.meshletOffset; m < info.meshletOffset + info.meshletCount; ++m) {
            const Meshlet& ml = meshlets[m];
            REQUIRE(ml.vertexOffset + ml.vertexCount <= scene.meshletVertices().size());
            REQUIRE(ml.triangleOffset + ml.triangleCount * 3 <= scene.meshletTriangles().size());
            CHECK(ml.vertexCount <= MESHLET_MAX_VERTICES);
            CHECK(ml.triangleCount <= MESHLET_MAX_TRIANGLES);
            for (u32 v = 0; v < ml.vertexCount; ++v) {
                const u32 global = scene.meshletVertices()[ml.vertexOffset + v];
                // Global vertex index must land inside this mesh's vertex range.
                CHECK(global >= info.vertexOffset);
                CHECK(global < scene.vertices().size());
            }
            for (u32 t = 0; t < ml.triangleCount * 3; ++t) {
                CHECK(scene.meshletTriangles()[ml.triangleOffset + t] < ml.vertexCount);
            }
        }
    }
}

TEST_CASE("addMaterial appends to the library and clear resets everything") {
    GpuScene scene;
    GPUMaterial m{};
    CHECK(scene.addMaterial(m) == 0);
    CHECK(scene.addMaterial(m) == 1);
    upload(scene, ProceduralMeshes::generateCube(0.5f));

    const u64 before = scene.geometryVersion();
    scene.clear();
    CHECK(scene.geometryVersion() > before);
    CHECK(scene.materials().empty());
    CHECK(scene.getMeshCount() == 0);
    CHECK(scene.vertices().empty());
}

TEST_CASE("procedural meshes are well formed") {
    const MeshData meshes[] = {
        ProceduralMeshes::generateTorus(1.0f, 0.25f, 24, 12),
        ProceduralMeshes::generateSphere(1.0f, 16, 8),
        ProceduralMeshes::generateCube(1.0f),
        ProceduralMeshes::generatePlane(2.0f, 2.0f, 3, 3),
    };
    for (const MeshData& m : meshes) {
        REQUIRE(m.indices.size() % 3 == 0);
        CHECK(m.normals.size() == m.positions.size());
        for (u32 i : m.indices) CHECK(i < m.positions.size());
        for (const auto& n : m.normals) CHECK(glm::length(n) == doctest::Approx(1.0f).epsilon(1e-3));
    }
}
