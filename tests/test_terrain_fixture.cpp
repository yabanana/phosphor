// F15.4 (ticket 23): the terrain fixture and its procedural data (scene/terrain_textures.h, testbench/terrain_fixture.h).
#include <doctest/doctest.h>
#include "renderer/gpu_scene.h"
#include "renderer/material_extension_pack.h"
#include "renderer/scene_store.h"
#include "scene/ecs.h"
#include "scene/terrain_textures.h"
#include "scene/texture_manager.h"
#include "testbench/terrain_fixture.h"

#include <array>
#include <cmath>
#include <cstring>
#include <set>

using namespace phosphor;

namespace {
class DiscardTextures final : public TextureManager {
public:
    [[nodiscard]] u32 textureCount() const override { return count_; }

protected:
    u32 createTexture(const u8*, u32, u32, bool) override { return count_++; }

private:
    u32 count_ = 0;
};

struct Built {
    ECS ecs;
    GpuScene scene;
    DiscardTextures textures;
    SceneStore store;
    TerrainFixture fixture;
    explicit Built(const std::string& name) : fixture(name) {
        fixture.setup(ecs, scene, textures);
        store.sync(ecs, scene);
    }
    ~Built() { fixture.teardown(ecs, scene); }
};

double channelMean(const terrain::Image& image, u32 channel) {
    double sum = 0.0;
    for (size_t i = channel; i < image.rgba.size(); i += 4) sum += image.rgba[i];
    return sum / (255.0 * static_cast<double>(image.rgba.size() / 4));
}
} // namespace

TEST_CASE("terrain scenarios: every listed scenario parses, options are validated") {
    for (std::string_view s : TerrainFixture::scenarios()) CHECK_MESSAGE(TerrainFixture::validScenario(s), std::string(s));
    TerrainOptions o;
    std::string error;
    REQUIRE(TerrainFixture::parse("terrain/weights=1,0,0.5,0/layers=3/height=off/coat=zero", o, error));
    CHECK(o.constantMask);
    CHECK(o.weights == std::array<u8, 4>{255, 0, 128, 0});
    CHECK(o.layers == 3);
    CHECK_FALSE(o.height);
    CHECK(o.coat == TerrainOptions::Coat::Zero);
    for (const char* bad : {"terrain/weights=1,0,0", "terrain/weights=2,0,0,0", "terrain/single=4", "terrain/layers=0", "terrain/coat=maybe",
                            "terrain/ramp=on/furnace=on", "terrain/ior=1.4", "terrain/coat=spec/ior=4", "terrain/ramp=on/single=2", "terrain/furnace=on/single=0",
                            "terrain/single=0/weights=1,0,0,0", "terrain/unknown=1", "terrain/height", "chess-backlight"})
        CHECK_MESSAGE(!TerrainFixture::validScenario(bad), bad);
    CHECK(TerrainFixture::environmentRadiance("terrain/furnace=on") == 1.0f);
    CHECK(TerrainFixture::environmentRadiance("terrain/ramp=on") == 0.0f);
    CHECK(TerrainFixture::environmentRadiance("terrain") == TerrainFixture::PatchEnvironment);
    CHECK(TerrainFixture::environmentRadiance("nonsense") == 0.0f);
}

// The scenes of the layers-* keys of tools/f15_check.py (tools/test_f15_check.py compares this list with the runner).
constexpr std::array<std::string_view, 20> kRunnerScenes{
    "terrain/single=0/coat=off", "terrain/single=1/coat=off", "terrain/single=2/coat=off", "terrain/single=3/coat=off",
    "terrain/weights=1,0,0,0/coat=off", "terrain/weights=0,1,0,0/coat=off", "terrain/weights=0,0,1,0/coat=off", "terrain/weights=0,0,0,1/coat=off",
    "terrain/weights=1,1,0,0/layers=2/coat=off", "terrain/weights=1,1,0,0/coat=off", "terrain/coat=off", "terrain", "terrain/coat=zero",
    "terrain/ramp=on/single=0", "terrain/ramp=on/single=1", "terrain/ramp=reverse", "terrain/ramp=on", "terrain/furnace=on", "terrain/coat=spec", "terrain/coat=spec/ior=1.4",
};

TEST_CASE("terrain scenarios: the runner's scenes are valid") {
    for (std::string_view s : kRunnerScenes) CHECK_MESSAGE(TerrainFixture::validScenario(s), std::string(s));
}

TEST_CASE("terrain data: deterministic, layer tilings are powers of two, the mask uses every layer") {
    const terrain::Image a = terrain::proceduralMask(64), b = terrain::proceduralMask(64);
    CHECK(a.rgba == b.rgba);
    CHECK(terrain::proceduralCavity(32).rgba == terrain::proceduralCavity(32).rgba);
    CHECK(terrain::proceduralLayer(1, 32).color.rgba == terrain::proceduralLayer(1, 32).color.rgba);
    for (u32 i = 0; i < terrain::kLayerCount; ++i) {
        const float s = terrain::layerSpec(i).uvScale;
        CHECK(std::log2(s) == doctest::Approx(std::round(std::log2(s)))); // exact scaling of the mesh uvs of the single twin
        CHECK(channelMean(a, i) > 0.03);                                 // every layer appears on the patch
    }
    // The cavity map covers a part of the patch, not all and not nothing.
    const double coat = channelMean(terrain::proceduralCavity(64), 0);
    CHECK(coat > 0.02);
    CHECK(coat < 0.6);
    CHECK(terrain::cc0Path("/d", 1, "NormalGL") == "/d/Rock030/Rock030_1K-JPG_NormalGL.jpg");
    CHECK_THROWS(terrain::layerSpec(4));
}

TEST_CASE("terrain patch: unit normals orthogonal to unit tangents, uvs scaled exactly") {
    const MeshData plain = terrain::makePatch(16), scaled = terrain::makePatch(16, 8.0f);
    REQUIRE(plain.positions.size() == scaled.positions.size());
    for (size_t i = 0; i < plain.positions.size(); ++i) {
        CHECK(glm::length(plain.normals[i]) == doctest::Approx(1.0f).epsilon(1e-5));
        CHECK(glm::dot(plain.normals[i], glm::vec3(plain.tangents[i])) == doctest::Approx(0.0f).epsilon(1e-4));
        CHECK(scaled.uvs[i].x == plain.uvs[i].x * 8.0f);
        CHECK(scaled.uvs[i].y == plain.uvs[i].y * 8.0f);
    }
}

TEST_CASE("terrain fixture: the stack's feature class, the single twin and the plain fixtures") {
    {
        Built b("terrain");
        CHECK(b.store.featureUnion() == (MATERIAL_FEATURE_LAYERS | MATERIAL_FEATURE_CLEARCOAT));
        // The stack material's record 3 names the four library layers, tiling and height textures.
        bool found = false;
        for (const GPUMaterial& m : b.store.materials()) {
            if (!(materialFeatureBits(m) & MATERIAL_FEATURE_LAYERS)) continue;
            found = true;
            const GPUMaterialExt3 e3 = unpackExtRecord<GPUMaterialExt3>(b.store.extensions()[m.extensionIndex + 3]);
            CHECK(e3.layerCount == 4);
            for (u32 i = 0; i < 4; ++i) {
                CHECK(e3.layerMaterial[i] < b.scene.materials().size());
                CHECK(e3.uvScale[i] == terrain::layerSpec(i).uvScale);
                CHECK(e3.heightContrast[i] == terrain::layerSpec(i).heightContrast);
                CHECK(e3.heightTex[i] != INVALID_TEXTURE_INDEX);
                CHECK(((e3.packedMaskChannels >> (2 * i)) & 3u) == i);
            }
        }
        CHECK(found);
    }
    CHECK(Built("terrain/coat=off").store.featureUnion() == MATERIAL_FEATURE_LAYERS);
    CHECK(Built("terrain/coat=zero").store.featureUnion() == (MATERIAL_FEATURE_LAYERS | MATERIAL_FEATURE_CLEARCOAT));
    CHECK(Built("terrain/coat=spec").store.featureUnion() == (MATERIAL_FEATURE_LAYERS | MATERIAL_FEATURE_SPECULAR_IOR));
    CHECK(Built("terrain/single=2/coat=off").store.featureUnion() == 0u);
    CHECK(Built("terrain/ramp=on").store.featureUnion() == MATERIAL_FEATURE_LAYERS);
    CHECK(Built("terrain/furnace=on").store.featureUnion() == MATERIAL_FEATURE_LAYERS);
    // height=off leaves the layers without a height contrast.
    Built plain("terrain/height=off/coat=off");
    for (const GPUMaterial& m : plain.store.materials())
        if (materialFeatureBits(m) & MATERIAL_FEATURE_LAYERS) {
            const GPUMaterialExt3 e3 = unpackExtRecord<GPUMaterialExt3>(plain.store.extensions()[m.extensionIndex + 3]);
            for (u32 i = 0; i < 4; ++i) CHECK(e3.heightContrast[i] == -1.0f);
        }
}

TEST_CASE("terrain fixture: a missing CC0 file is an error, not a silent stand-in") {
    ECS ecs;
    GpuScene scene;
    DiscardTextures textures;
    TerrainFixture fixture("terrain", "/nonexistent-terrain-materials");
    CHECK_THROWS(fixture.setup(ecs, scene, textures));
}
