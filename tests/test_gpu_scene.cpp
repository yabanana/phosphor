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

// ---------------------------------------------------------------------------
// Review (ticket 04): setMaterialDescription semantics.
// ---------------------------------------------------------------------------
namespace {
constexpr u32 kWhite = 90, kMr = 91, kNormal = 92;

GPUMaterial texturedMaterial() {
    GPUMaterial g{};
    g.baseColorTex = 10;
    g.metallicRoughnessTex = 11;
    g.normalTex = 12;
    g.occlusionTex = 13;
    g.emissiveTex = 14;
    return g;
}

MaterialDescription describedWithAllTextures() {
    MaterialDescription d;
    d.base.baseColorTexture.index         = 10;
    d.base.metallicRoughnessTexture.index = 11;
    d.base.normalTexture.index            = 12;
    d.base.occlusionTexture.index         = 13;
    d.base.emissiveTexture.index          = 14;
    return d;
}
} // namespace

TEST_CASE("GpuScene::setMaterialDescription: a removed texture binds the documented default, never the old one") {
    GpuScene scene;
    scene.setBaseTextureDefaults(kWhite, kMr, kNormal);
    scene.addMaterial(texturedMaterial(), describedWithAllTextures(), "m", MaterialSupportReport{});

    MaterialDescription edited = describedWithAllTextures();
    edited.base.normalTexture = textureSlot(TextureColorSpace::Linear, TextureChannel::Rgb); // removed
    edited.base.emissiveTexture = textureSlot(TextureColorSpace::SRGB, TextureChannel::Rgb); // removed
    REQUIRE(scene.setMaterialDescription(0, edited));
    const GPUMaterial& g = scene.materials()[0];
    CHECK(g.normalTex == kNormal);
    CHECK(g.emissiveTex == kWhite);
    CHECK(g.baseColorTex == 10u); // untouched slots keep their texture
    CHECK(g.metallicRoughnessTex == 11u);
    CHECK(g.occlusionTex == 13u);

    // Removing every texture binds the three defaults.
    MaterialDescription bare;
    REQUIRE(scene.setMaterialDescription(0, bare));
    CHECK(g.baseColorTex == kWhite);
    CHECK(g.metallicRoughnessTex == kMr);
    CHECK(g.normalTex == kNormal);
    CHECK(g.occlusionTex == kWhite);
    CHECK(g.emissiveTex == kWhite);
}

TEST_CASE("GpuScene::setMaterialDescription: without registered defaults a removed texture is the shader fallback") {
    GpuScene scene;
    scene.addMaterial(texturedMaterial(), describedWithAllTextures(), "m", MaterialSupportReport{});
    REQUIRE(scene.setMaterialDescription(0, MaterialDescription{}));
    CHECK(scene.materials()[0].baseColorTex == INVALID_TEXTURE_INDEX);
    CHECK(scene.materials()[0].normalTex == INVALID_TEXTURE_INDEX);
}

TEST_CASE("GpuScene::setMaterialDescription: a path that is not resolved yet keeps the current binding") {
    GpuScene scene;
    scene.setBaseTextureDefaults(kWhite, kMr, kNormal);
    scene.addMaterial(texturedMaterial(), describedWithAllTextures(), "m", MaterialSupportReport{});
    MaterialDescription d = describedWithAllTextures();
    d.base.baseColorTexture.index = INVALID_TEXTURE_INDEX;
    d.base.baseColorTexture.path  = "wax.png";
    REQUIRE(scene.setMaterialDescription(0, d));
    CHECK(scene.materials()[0].baseColorTex == 10u);
}

TEST_CASE("GpuScene::setMaterialDescription: the support entry is recomputed from the new description") {
    GpuScene scene;
    MaterialDescription plain;
    scene.addMaterial(GPUMaterial{}, plain, "wax", buildSupportReport(plain, MaterialSourceInfo{}));
    REQUIRE(scene.materialSupport()[0].features.empty());

    MaterialDescription sheened;
    sheened.sheen = SheenBlock{};
    sheened.sheen->color = {0.5f, 0.5f, 0.5f};
    REQUIRE(scene.setMaterialDescription(0, sheened));
    REQUIRE(scene.materialSupport()[0].features.size() == 1);
    CHECK(scene.materialSupport()[0].features[0].feature == "sheen");
    CHECK(scene.materialSupport()[0].features[0].level == currentSupportTable()
                                                              .features[static_cast<size_t>(MaterialFeature::Sheen)].level);
    CHECK(scene.materialSupport()[0].name == "wax");

    // A removed feature disappears from the report.
    REQUIRE(scene.setMaterialDescription(0, plain));
    CHECK(scene.materialSupport()[0].features.empty());
}

TEST_CASE("GpuScene::setMaterialDescription: the GPU emissive is clamped to >= 0, the description keeps the value") {
    GpuScene scene;
    scene.addMaterial(GPUMaterial{}, MaterialDescription{}, "m", MaterialSupportReport{});
    MaterialDescription d;
    d.base.emissive = {-2.0f, 0.5f, 3.0f};
    REQUIRE(scene.setMaterialDescription(0, d));
    CHECK(scene.materials()[0].emissive[0] == 0.0f);
    CHECK(scene.materials()[0].emissive[1] == 0.5f);
    CHECK(scene.materials()[0].emissive[2] == 3.0f);
    CHECK(scene.materialDescriptions()[0].base.emissive[0] == -2.0f);
}

TEST_CASE("GpuScene::setMaterialDescription after the store mirrored the library bumps the edit version") {
    GpuScene scene;
    GPUMaterial g{};
    g.roughness = 0.9f;
    scene.addMaterial(g, MaterialDescription{}, "m", MaterialSupportReport{});
    CHECK_FALSE(scene.materialLibrarySealed());
    CHECK(scene.materialEditVersion() == 0);
    MaterialDescription d;
    d.base.roughness = 0.1f;
    CHECK(scene.setMaterialDescription(0, d));
    CHECK(scene.materials()[0].roughness == doctest::Approx(0.1f));
    CHECK(scene.materialEditVersion() == 1);

    // F15.1 (ticket 05): edits stay valid once sealed; SceneStore re-derives the library.
    scene.sealMaterialLibrary();
    CHECK(scene.materialLibrarySealed());
    d.base.roughness = 0.7f;
    CHECK(scene.setMaterialDescription(0, d));
    CHECK(scene.materials()[0].roughness == doctest::Approx(0.7f));
    CHECK(scene.materialEditVersion() == 2);

    scene.clear();
    CHECK_FALSE(scene.materialLibrarySealed());
}
