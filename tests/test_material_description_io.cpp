#include "renderer/gpu_scene.h"
#include "renderer/material_description_io.h"

#include <doctest/doctest.h>
#include <json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace phosphor;

namespace {

NamedMaterial richMaterial() {
    NamedMaterial m;
    m.name = "king";
    MaterialDescription& d = m.description;
    d.base.baseColor = {0.9f, 0.8f, 0.7f, 1.0f};
    d.base.metallic  = 0.0f;
    d.base.roughness = 0.35f;
    d.base.emissive  = {2.5f, 0.0f, 0.0f}; // HDR emitter
    d.base.baseColorTexture.path = "tex/king_albedo.png";
    UvTransform xf;
    xf.offset = {0.25f, 0.5f};
    xf.scale  = {2.0f, 2.0f};
    d.base.baseColorTexture.transform = xf;
    d.specular = SpecularBlock{};
    d.specular->ior = 1.45f;
    d.clearcoat = ClearcoatBlock{};
    d.clearcoat->factor = 0.5f;
    d.clearcoat->normalTexture.path = "tex/coat_n.png";
    d.sheen = SheenBlock{};
    d.sheen->color = {0.1f, 0.2f, 0.3f};
    d.anisotropy = AnisotropyBlock{};
    d.anisotropy->strength = 0.6f;
    d.anisotropy->rotation = 0.5f;
    d.iridescence = IridescenceBlock{};
    d.iridescence->factor = 1.0f;
    d.iridescence->thicknessMin = 500.0f; // allowed to exceed the maximum
    d.transmission = TransmissionBlock{};
    d.transmission->factor = 0.75f;
    d.volume = VolumeBlock{};
    d.volume->thickness = 0.5f; // attenuationDistance stays +inf
    d.subsurface = SubsurfaceBlock{};
    d.subsurface->weight = 1.0f;
    d.subsurface->mode = SubsurfaceMode::Volumetric;
    d.subsurface->radius = 0.02f;
    d.subsurface->radiusScale = {1.0f, 0.5f, 0.25f};
    d.subsurface->anisotropy = 0.3f;
    d.subsurface->source = SubsurfaceSource::KhrMaterialsScatterDraft;
    d.subsurface->radiusScaleTexture.path = "tex/king_albedo.png";
    d.subsurface->draft = ScatterDraftParams{};
    d.subsurface->draft->scatterStrength = 0.9f;
    d.layers = LayerStack{};
    d.layers->mask.path = "tex/mask.png";
    d.layers->layers.push_back({"grass", ~0u, TextureChannel::R, std::nullopt});
    d.layers->layers.push_back({"rock", ~0u, TextureChannel::G, 0.05f});
    return m;
}

const char* OVERRIDES_HEAD = R"({"schema":"phosphor-material-overrides","version":1,"overrides":[)";

} // namespace

TEST_CASE("material description file: save and load round-trip") {
    const NamedMaterial original = richMaterial();
    REQUIRE(validate(original.description).empty());
    std::string json, error;
    REQUIRE_MESSAGE(saveMaterialDescriptions({original}, json, error), error);

    const auto doc = nlohmann::json::parse(json);
    CHECK(doc.at("schema") == "phosphor-material-description");
    CHECK(doc.at("version") == 1);
    CHECK(doc.at("materials").at(0).at("name") == "king");
    CHECK(doc.at("materials").at(0).at("volume").at("attenuationDistance") == "inf");

    std::vector<NamedMaterial> loaded;
    REQUIRE_MESSAGE(loadMaterialDescriptions(json, loaded, error), error);
    REQUIRE(loaded.size() == 1);
    const MaterialDescription& d = loaded[0].description;
    CHECK(loaded[0].name == "king");
    CHECK(d.base.roughness == doctest::Approx(0.35f));
    CHECK(d.base.emissive[0] == doctest::Approx(2.5f));
    CHECK(d.base.baseColorTexture.path == "tex/king_albedo.png");
    CHECK(d.base.baseColorTexture.index == INVALID_TEXTURE_INDEX);
    REQUIRE(d.base.baseColorTexture.transform.has_value());
    CHECK(d.base.baseColorTexture.transform->offset[1] == doctest::Approx(0.5f));
    CHECK(d.specular->ior == doctest::Approx(1.45f));
    CHECK(d.clearcoat->normalTexture.path == "tex/coat_n.png");
    CHECK(d.sheen->color[2] == doctest::Approx(0.3f));
    CHECK(d.anisotropy->rotation == doctest::Approx(0.5f));
    CHECK(d.iridescence->thicknessMin == doctest::Approx(500.0f));
    CHECK(d.transmission->factor == doctest::Approx(0.75f));
    CHECK(std::isinf(d.volume->attenuationDistance));
    CHECK(d.subsurface->mode == SubsurfaceMode::Volumetric);
    CHECK(d.subsurface->source == SubsurfaceSource::KhrMaterialsScatterDraft);
    CHECK(d.subsurface->draft->scatterStrength == doctest::Approx(0.9f));
    CHECK(d.subsurface->radiusScaleTexture.path == "tex/king_albedo.png");
    REQUIRE(d.layers.has_value());
    REQUIRE(d.layers->layers.size() == 2);
    CHECK(d.layers->layers[1].materialName == "rock");
    CHECK(d.layers->layers[1].maskChannel == TextureChannel::G);
    CHECK(d.layers->layers[1].heightContrast == std::optional<float>(0.05f));
    CHECK_FALSE(d.layers->layers[0].heightContrast.has_value());
    CHECK(d.layers->mask.path == "tex/mask.png");
    CHECK(featureMask(d) == featureMask(original.description));

    // Stable: saving the loaded description gives the same text.
    std::string again;
    REQUIRE(saveMaterialDescriptions(loaded, again, error));
    CHECK(again == json);
}

TEST_CASE("material description file: a bindless index without a path cannot be saved") {
    NamedMaterial m;
    m.name = "from-gltf";
    m.description.base.baseColorTexture.index = 7;
    std::string json, error;
    CHECK_FALSE(saveMaterialDescriptions({m}, json, error));
    CHECK(error.find("baseColorTexture") != std::string::npos);
}

TEST_CASE("material description file: invalid files are errors") {
    std::vector<NamedMaterial> out;
    std::string error;
    auto load = [&](const std::string& materialsJson) {
        error.clear();
        return loadMaterialDescriptions(R"({"schema":"phosphor-material-description","version":1,"materials":)" +
                                            materialsJson + "}",
                                        out, error);
    };
    CHECK(load(R"([{"name":"a"}])"));
    CHECK_FALSE(load(R"([{"name":"a","clearcoat":{"factor":1.5}}])"));
    CHECK(error.find("clearcoat.factor") != std::string::npos);
    CHECK_FALSE(load(R"([{"name":"a","clearcoat":{"factr":1}}])"));
    CHECK(error.find("unknown key 'factr'") != std::string::npos);
    CHECK_FALSE(load(R"([{"name":"a","sparkle":{}}])"));
    CHECK_FALSE(load(R"([{"name":"a","clearcoat":{"factor":"high"}}])"));
    CHECK_FALSE(load(R"([{"name":"a","specular":{"ior":0.5}}])"));
    CHECK_FALSE(load(R"([{"name":"a","subsurface":{"mode":"sideways"}}])"));
    CHECK_FALSE(load(R"([{"name":"a","volume":{"thickness":1}}])")); // needs transmission
    CHECK_FALSE(load(R"([{"name":"a"},{"name":"a"}])"));
    CHECK_FALSE(load(R"([{"base":{}}])"));
    CHECK_FALSE(load(R"([{"name":"a","base":{"baseColorTexture":{"texCoord":1}}}])")); // no path
    CHECK_FALSE(loadMaterialDescriptions("{not json", out, error));
    CHECK_FALSE(loadMaterialDescriptions(R"({"schema":"other","version":1,"materials":[]})", out, error));
    CHECK_FALSE(loadMaterialDescriptions(R"({"schema":"phosphor-material-description","version":2,"materials":[]})",
                                         out, error));
    CHECK(error.find("version") != std::string::npos);
}

TEST_CASE("material description file: name-keyed overrides") {
    std::vector<NamedMaterial> materials(2);
    materials[0].name = "wax";
    materials[1].name = "stone";
    materials[1].description.base.roughness = 0.9f;
    std::string error;
    std::vector<size_t> changed;

    SUBCASE("changes only the listed fields and creates blocks") {
        const std::string json = std::string(OVERRIDES_HEAD) +
            R"({"name":"wax","base":{"roughness":0.25},"clearcoat":{"factor":1,"roughness":0.1}}]})";
        REQUIRE_MESSAGE(applyMaterialOverrides(json, materials, error, &changed), error);
        CHECK(changed == std::vector<size_t>{0});
        CHECK(materials[0].description.base.roughness == doctest::Approx(0.25f));
        CHECK(materials[0].description.base.metallic == 1.0f); // untouched
        REQUIRE(materials[0].description.clearcoat);
        CHECK(materials[0].description.clearcoat->factor == 1.0f);
        CHECK(materials[1].description.base.roughness == doctest::Approx(0.9f));
        CHECK(featureMask(materials[0].description) == featureBit(MaterialFeature::Clearcoat));
    }
    SUBCASE("null removes a block, a path sets a texture slot") {
        materials[0].description.clearcoat = ClearcoatBlock{};
        materials[0].description.clearcoat->factor = 1.0f;
        materials[0].description.base.normalTexture.index = 12;
        const std::string json = std::string(OVERRIDES_HEAD) +
            R"({"name":"wax","clearcoat":null,"base":{"baseColorTexture":{"path":"wax.png"}}}]})";
        REQUIRE_MESSAGE(applyMaterialOverrides(json, materials, error), error);
        CHECK_FALSE(materials[0].description.clearcoat.has_value());
        CHECK(materials[0].description.base.baseColorTexture.path == "wax.png");
        CHECK(materials[0].description.base.normalTexture.index == 12u); // not mentioned: kept
    }
    SUBCASE("an unknown name is an error and nothing changes") {
        const std::string json = std::string(OVERRIDES_HEAD) +
            R"({"name":"wax","base":{"roughness":0.1}},{"name":"marble","base":{"roughness":0.1}}]})";
        CHECK_FALSE(applyMaterialOverrides(json, materials, error));
        CHECK(error.find("unknown material 'marble'") != std::string::npos);
        CHECK(materials[0].description.base.roughness == 1.0f); // atomic
    }
    SUBCASE("an invalid value is an error and nothing changes") {
        const std::string json = std::string(OVERRIDES_HEAD) +
            R"({"name":"wax","base":{"roughness":2.0}}]})";
        CHECK_FALSE(applyMaterialOverrides(json, materials, error));
        CHECK(error.find("base.roughness") != std::string::npos);
        CHECK(materials[0].description.base.roughness == 1.0f);
    }
    SUBCASE("unknown keys and a wrong schema are errors") {
        CHECK_FALSE(applyMaterialOverrides(std::string(OVERRIDES_HEAD) + R"({"name":"wax","shine":1}]})", materials, error));
        CHECK_FALSE(applyMaterialOverrides(R"({"schema":"phosphor-material-description","version":1,"materials":[]})",
                                           materials, error));
    }
    SUBCASE("a companion violation after the override is an error") {
        const std::string json = std::string(OVERRIDES_HEAD) +
            R"({"name":"wax","volume":{"thickness":1}}]})";
        CHECK_FALSE(applyMaterialOverrides(json, materials, error));
        CHECK(error.find("volume requires") != std::string::npos);
    }
}

TEST_CASE("material description file: file helpers resolve texture paths against the file directory") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "phosphor_matdesc_io";
    fs::create_directories(dir);
    NamedMaterial m;
    m.name = "m";
    m.description.base.baseColorTexture.path = "a.png";
    const std::string file = (dir / "materials.json").string();
    std::string error;
    REQUIRE_MESSAGE(saveMaterialDescriptionFile(file, {m}, error), error);
    std::vector<NamedMaterial> loaded;
    REQUIRE_MESSAGE(loadMaterialDescriptionFile(file, loaded, error), error);
    CHECK(loaded[0].description.base.baseColorTexture.path == (dir / "a.png").generic_string());
    CHECK_FALSE(loadMaterialDescriptionFile((dir / "missing.json").string(), loaded, error));
    fs::remove_all(dir);
}

TEST_CASE("material overrides apply to a loaded scene's GPU records") {
    GpuScene scene;
    GPUMaterial g{};
    g.roughness = 1.0f;
    g.baseColorTex = 3;
    g.flags = MATERIAL_FLAG_DOUBLE_SIDED;
    MaterialDescription d;
    scene.addMaterial(g, d, "wax", MaterialSupportReport{});
    scene.addMaterial(g); // unnamed
    const std::string json = std::string(OVERRIDES_HEAD) +
        R"({"name":"wax","base":{"roughness":0.4,"emissive":[1,2,3],"doubleSided":false,"baseColorTexture":{"path":"wax.png"}},
            "sheen":{"color":[0.5,0.5,0.5]}}]})";
    std::string error;
    const auto resolve = [](const std::string& path, bool sRGB) {
        return path == "wax.png" && sRGB ? 42u : 0u;
    };
    REQUIRE_MESSAGE(applyMaterialOverridesToScene(json, scene, resolve, error), error);
    CHECK(scene.materials()[0].roughness == doctest::Approx(0.4f));
    CHECK(scene.materials()[0].emissive[2] == doctest::Approx(3.0f));
    CHECK((scene.materials()[0].flags & MATERIAL_FLAG_DOUBLE_SIDED) == 0u);
    CHECK(scene.materials()[0].baseColorTex == 42u); // baseColor is sRGB
    CHECK(featureMask(scene.materialDescriptions()[0]) == featureBit(MaterialFeature::Sheen));
    CHECK(scene.materials()[1].roughness == 1.0f);

    CHECK_FALSE(applyMaterialOverridesToScene(std::string(OVERRIDES_HEAD) + R"({"name":"ghost"}]})", scene, resolve, error));
}
