// F15 closure (loader-misc): KHR_materials_unlit, KHR_lights_punctual and the tangent frame repair of anisotropic
// primitives.  Expected values come from the extension texts (unlit: the colour is displayed as is; lights: units,
// defaults, direction along the node's -Z) and from geometry, not from the code under test.
#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "renderer/material_description.h"
#include "renderer/material_description_io.h"
#include "renderer/material_support.h"
#include "renderer/scene_extract.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/gltf_lights.h"
#include "scene/gltf_loader.h"
#include "scene/gltf_material_extensions.h"
#include "scene/mesh_processing.h"

#include <doctest/doctest.h>
#include <json.hpp>
#include <tiny_gltf.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace phosphor;
using phosphor::test::NullTextureManager;

namespace {

std::string base64(const std::vector<unsigned char>& data) {
    static const char* kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < data.size(); i += 3) {
        const u32 n = (u32{data[i]} << 16) | (i + 1 < data.size() ? u32{data[i + 1]} << 8 : 0) |
                      (i + 2 < data.size() ? u32{data[i + 2]} : 0);
        out += kAlphabet[(n >> 18) & 63];
        out += kAlphabet[(n >> 12) & 63];
        out += i + 1 < data.size() ? kAlphabet[(n >> 6) & 63] : '=';
        out += i + 2 < data.size() ? kAlphabet[n & 63] : '=';
    }
    return out;
}

template <typename T> void append(std::vector<unsigned char>& bytes, const std::vector<T>& values) {
    const size_t at = bytes.size();
    bytes.resize(at + values.size() * sizeof(T));
    std::memcpy(bytes.data() + at, values.data(), values.size() * sizeof(T));
}

// A +Z facing quad with POSITION, NORMAL, TEXCOORD_0, optional TANGENT, indices.  `uvs` lets a test degenerate the
// mapping.  `nodesJson` replaces the node list, `extra` is spliced at the top level.
struct Quad {
    std::vector<float> uvs       = {0, 1, 1, 1, 1, 0, 0, 0};
    std::vector<float> tangents; // 16 floats or empty
    std::string        materialJson;
    std::string        nodesJson = R"([{"mesh":0}])";
    std::string        extra;
    std::string        scenesNodes = "[0]";
};

std::string quadJson(const Quad& q) {
    const std::vector<float> positions = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    const std::vector<float> normals   = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};
    const std::vector<unsigned short> indices = {0, 1, 2, 0, 2, 3};
    std::vector<unsigned char> bytes;
    append(bytes, positions); // 0..48
    append(bytes, normals);   // 48..96
    append(bytes, q.uvs);     // 96..128
    append(bytes, indices);   // 128..140
    const size_t tangentOffset = bytes.size();
    if (!q.tangents.empty()) append(bytes, q.tangents); // 140..204
    std::string attrs = R"("POSITION":0,"NORMAL":1,"TEXCOORD_0":2)";
    std::string views = R"({"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":48},
        {"buffer":0,"byteOffset":96,"byteLength":32},{"buffer":0,"byteOffset":128,"byteLength":12})";
    std::string accessors = R"({"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1,-1,0],"max":[1,1,0]},
        {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},
        {"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},
        {"bufferView":3,"componentType":5123,"count":6,"type":"SCALAR"})";
    if (!q.tangents.empty()) {
        attrs += R"(,"TANGENT":4)";
        views += R"(,{"buffer":0,"byteOffset":)" + std::to_string(tangentOffset) + R"(,"byteLength":64})";
        accessors += R"(,{"bufferView":4,"componentType":5126,"count":4,"type":"VEC4"})";
    }
    return R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":)" + q.scenesNodes + R"(}],"nodes":)" + q.nodesJson +
           R"(,"meshes":[{"primitives":[{"attributes":{)" + attrs + R"(},"indices":3)" +
           (q.materialJson.empty() ? "" : R"(,"material":0)") + R"(}]}],)" +
           (q.materialJson.empty() ? "" : R"("materials":[)" + q.materialJson + "],") + q.extra + R"("buffers":[{"byteLength":)" +
           std::to_string(bytes.size()) + R"(,"uri":"data:application/octet-stream;base64,)" + base64(bytes) + R"("}],"bufferViews":[)" +
           views + R"(],"accessors":[)" + accessors + "]}";
}

struct Loaded {
    GpuScene scene;
    NullTextureManager textures;
    ECS ecs;
    u32 lights = 0;
    bool ok = false;
};

void load(Loaded& l, const std::string& json, const char* name) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    {
        std::ofstream out(path);
        out << json;
    }
    l.textures.createDefaultTextures();
    GltfLoader loader(l.scene, l.textures, l.ecs);
    l.ok = loader.loadFromFile(path.string());
    l.lights = loader.importedLightCount();
    std::filesystem::remove(path);
}

tinygltf::Model parseModel(const std::string& json) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    REQUIRE_MESSAGE(loader.LoadASCIIFromString(&model, &err, &warn, json.c_str(), static_cast<unsigned>(json.size()), ""), err);
    return model;
}

GltfMaterialParse parseOne(const std::string& materialJson) {
    const tinygltf::Model model = parseModel(R"({"asset":{"version":"2.0"},"materials":[)" + materialJson + "]}");
    return parseGltfMaterial(model.materials.at(0), [](int index, bool) { return static_cast<u32>(index + 100); });
}

const char* kUnlitMaterial = R"({"name":"flat","emissiveFactor":[0.9,0.9,0.9],
    "pbrMetallicRoughness":{"baseColorFactor":[0.8,0.4,0.2,1],"metallicFactor":1,"roughnessFactor":0.1},
    "extensions":{"KHR_materials_unlit":{},"KHR_materials_clearcoat":{"clearcoatFactor":1}}})";

} // namespace

// --- KHR_materials_unlit ---------------------------------------------------------------------------------------------

TEST_CASE("unlit: the base colour is displayed as emission, nothing else reflects light") {
    const auto p = parseOne(kUnlitMaterial);
    const MaterialDescription& d = p.description;
    CHECK(d.base.unlit);
    // Displayed colour = base colour factor (the extension's README); the authored emission and the other extensions are ignored.
    CHECK(d.base.emissive[0] == doctest::Approx(0.8f));
    CHECK(d.base.emissive[1] == doctest::Approx(0.4f));
    CHECK(d.base.emissive[2] == doctest::Approx(0.2f));
    CHECK(d.base.baseColor[0] == 0.0f);
    CHECK(d.base.baseColor[1] == 0.0f);
    CHECK(d.base.baseColor[2] == 0.0f);
    CHECK(d.base.baseColor[3] == 1.0f); // alpha stays the material's alpha
    CHECK(d.base.metallic == 0.0f);
    REQUIRE(d.specular.has_value());
    CHECK(d.specular->specularFactor == 0.0f); // F0 = F90 = 0: no reflection of any light
    CHECK_FALSE(d.clearcoat.has_value());
    // The report encodes it as an approximation (no COLOR_0 stream) and lists no feature.
    const auto report = buildSupportReport(d, p.info);
    CHECK(report.features.empty());
    bool saw = false;
    for (const auto& e : report.encodings)
        if (e.feature == "unlit") {
            saw = true;
            CHECK(e.level == SupportLevel::Implemented); // COLOR_0 is applied now: no approximation left
        }
    CHECK(saw);
    CHECK(p.info.unknownExtensions.empty());

    SUBCASE("negative control: the same material without the extension keeps its lit description") {
        const auto lit = parseOne(R"({"emissiveFactor":[0.9,0.9,0.9],
            "pbrMetallicRoughness":{"baseColorFactor":[0.8,0.4,0.2,1],"metallicFactor":1}})");
        CHECK_FALSE(lit.description.base.unlit);
        CHECK(lit.description.base.baseColor[0] == doctest::Approx(0.8f));
        CHECK(lit.description.base.emissive[0] == doctest::Approx(0.9f));
        CHECK(lit.description.base.metallic == doctest::Approx(1.0f));
    }
    SUBCASE("lowering is idempotent") {
        MaterialDescription again = d;
        lowerUnlit(again);
        CHECK(again.base.emissive == d.base.emissive);
        CHECK(again.base.baseColor == d.base.baseColor);
        CHECK(featureMask(again) == featureMask(d));
    }
    SUBCASE("the description file round trips the flag, and an unlit file entry stays unlit") {
        std::vector<NamedMaterial> in{{"flat", d}};
        std::string json, error;
        REQUIRE(saveMaterialDescriptions(in, json, error));
        CHECK(json.find("unlit") != std::string::npos);
        std::vector<NamedMaterial> out;
        REQUIRE(loadMaterialDescriptions(json, out, error));
        REQUIRE(out.size() == 1);
        CHECK(out[0].description.base.unlit);
        CHECK(out[0].description.base.emissive[0] == doctest::Approx(0.8f));
    }
}

TEST_CASE("unlit: a lit description file stays free of the flag") {
    std::vector<NamedMaterial> in{{"plain", MaterialDescription{}}};
    std::string json, error;
    REQUIRE(saveMaterialDescriptions(in, json, error));
    CHECK(json.find("unlit") == std::string::npos);
}

TEST_CASE("unlit: the loader sets the GPU flag, folds alpha and the quad is no sampled emitter flag victim") {
    Quad q;
    q.materialJson = kUnlitMaterial;
    q.extra = R"("extensionsUsed":["KHR_materials_unlit","KHR_materials_clearcoat"],"extensionsRequired":["KHR_materials_unlit"],)";
    Loaded l;
    load(l, quadJson(q), "phosphor_unlit_quad.gltf");
    REQUIRE(l.ok); // a required KHR_materials_unlit is no longer refused
    REQUIRE(l.scene.materials().size() == 1);
    const GPUMaterial& m = l.scene.materials()[0];
    CHECK((m.flags & MATERIAL_FLAG_UNLIT) != 0);
    CHECK(m.emissive[0] == doctest::Approx(0.8f));
    CHECK(m.baseColor[0] == 0.0f);
    CHECK(m.metallic == 0.0f);
    CHECK(featureMask(l.scene.materialDescriptions()[0]) == featureBit(MaterialFeature::SpecularIor)); // the lowering's internal zero specular block only
    CHECK(l.scene.assetUnknownExtensions().empty());

    Quad lit;
    lit.materialJson = R"({"emissiveFactor":[0.9,0.9,0.9]})";
    Loaded l2;
    load(l2, quadJson(lit), "phosphor_lit_quad.gltf");
    REQUIRE(l2.ok);
    CHECK((l2.scene.materials()[0].flags & MATERIAL_FLAG_UNLIT) == 0); // negative control
}

// --- KHR_lights_punctual ---------------------------------------------------------------------------------------------

TEST_CASE("lights: specification defaults and units") {
    tinygltf::Light sun;
    sun.type = "directional";
    const auto a = convertPunctualLight(sun);
    REQUIRE(a.valid);
    CHECK(a.light.type == LightType::Directional);
    CHECK(a.light.color == glm::vec3(1.0f));
    CHECK(a.light.intensity == 1.0f);
    CHECK(a.warnings.empty());

    tinygltf::Light lamp;
    lamp.type = "point";
    lamp.color = {1.0, 0.5, 0.25};
    lamp.intensity = 40.0;
    lamp.range = 7.0;
    const auto b = convertPunctualLight(lamp);
    CHECK(b.light.type == LightType::Point);
    CHECK(b.light.intensity == doctest::Approx(40.0f));
    CHECK(b.light.color.g == doctest::Approx(0.5f));
    CHECK(b.light.range == doctest::Approx(7.0f));
    CHECK(convertPunctualLight(lamp, 1.0f / 683.0f).light.intensity == doctest::Approx(40.0f / 683.0f));

    lamp.range = 0.0; // absent in the file: infinite
    CHECK(convertPunctualLight(lamp).light.range == GLTF_LIGHT_INFINITE_RANGE);

    tinygltf::Light spot;
    spot.type = "spot";
    const auto c = convertPunctualLight(spot); // defaults: inner 0, outer pi/4
    CHECK(c.light.type == LightType::Spot);
    CHECK(c.light.innerConeAngle == doctest::Approx(0.0f));
    CHECK(c.light.outerConeAngle == doctest::Approx(0.7853982f));
    CHECK(c.warnings.empty());

    SUBCASE("invalid cones and negative intensity are clamped with a warning") {
        spot.spot.innerConeAngle = 1.2;
        spot.spot.outerConeAngle = 0.5; // inner > outer
        spot.intensity = -3.0;
        const auto d = convertPunctualLight(spot);
        CHECK(d.warnings.size() == 2);
        CHECK(d.light.outerConeAngle == doctest::Approx(0.5f));
        CHECK(d.light.innerConeAngle < d.light.outerConeAngle);
        CHECK(d.light.intensity == 0.0f);
    }
    SUBCASE("an unknown type creates nothing") {
        tinygltf::Light odd;
        odd.type = "area";
        const auto d = convertPunctualLight(odd);
        CHECK_FALSE(d.valid);
        CHECK(d.warnings.size() == 1);
    }
}

TEST_CASE("lights: nodes become LightComponents shining along the node's -Z") {
    // Node 0 carries the quad, node 1 a spot rotated 90 degrees about +X (local -Z -> world +Y... see below), node 2 a point
    // light under a scaled, translated parent.
    Quad q;
    q.materialJson = R"({"name":"m"})";
    // Rotation 90 deg about X: quaternion (sin45,0,0,cos45).  Rz(-1) rotates (0,0,-1) to (0,1,0)... computed below with glm.
    q.nodesJson = R"([{"mesh":0},
        {"extensions":{"KHR_lights_punctual":{"light":0}},"translation":[1,2,3],"rotation":[0.70710678,0,0,0.70710678]},
        {"children":[3],"translation":[10,0,0],"scale":[2,2,2]},
        {"extensions":{"KHR_lights_punctual":{"light":1}},"translation":[0,1,0]},
        {"extensions":{"KHR_lights_punctual":{"light":2}}}])";
    q.scenesNodes = "[0,1,2,4]";
    q.extra = R"("extensionsUsed":["KHR_lights_punctual"],"extensionsRequired":["KHR_lights_punctual"],
        "extensions":{"KHR_lights_punctual":{"lights":[
          {"type":"spot","color":[1,0.9,0.8],"intensity":120,"range":15,"spot":{"innerConeAngle":0.2,"outerConeAngle":0.6}},
          {"type":"point","intensity":5},
          {"type":"directional","intensity":3}]}},)";
    Loaded l;
    load(l, quadJson(q), "phosphor_lights_nodes.gltf");
    REQUIRE(l.ok);
    CHECK(l.lights == 3);
    CHECK(l.scene.assetUnknownExtensions().empty());

    std::vector<GPULight> lights;
    extractLights(l.ecs, lights);
    REQUIRE(lights.size() == 3);
    const GPULight* spotL = nullptr;
    const GPULight* pointL = nullptr;
    const GPULight* sunL = nullptr;
    for (const GPULight& g : lights) {
        if (g.type == static_cast<u32>(LightType::Spot)) spotL = &g;
        if (g.type == static_cast<u32>(LightType::Point)) pointL = &g;
        if (g.type == static_cast<u32>(LightType::Directional)) sunL = &g;
    }
    REQUIRE(spotL);
    REQUIRE(pointL);
    REQUIRE(sunL);
    // 90 degrees about +X maps local -Z = (0,0,-1) to (0,1,0)?  R_x(90): y' = y cos - z sin, z' = y sin + z cos
    // -> (0, 0*0 - (-1)*1, 0*1 + (-1)*0) = (0, 1, 0).
    CHECK(spotL->direction[0] == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(spotL->direction[1] == doctest::Approx(1.0f).epsilon(1e-5));
    CHECK(spotL->direction[2] == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(spotL->position[0] == doctest::Approx(1.0f));
    CHECK(spotL->position[1] == doctest::Approx(2.0f));
    CHECK(spotL->position[2] == doctest::Approx(3.0f));
    CHECK(spotL->intensity == doctest::Approx(120.0f));
    CHECK(spotL->range == doctest::Approx(15.0f));
    CHECK(spotL->innerCone == doctest::Approx(0.2f));
    CHECK(spotL->outerCone == doctest::Approx(0.6f));
    // The scaled parent moves the child: translation (10,0,0) + 2 * (0,1,0).
    CHECK(pointL->position[0] == doctest::Approx(10.0f));
    CHECK(pointL->position[1] == doctest::Approx(2.0f));
    CHECK(pointL->position[2] == doctest::Approx(0.0f));
    // The unrotated directional light shines along -Z.
    CHECK(sunL->direction[2] == doctest::Approx(-1.0f));
    CHECK(sunL->intensity == doctest::Approx(3.0f));

    SUBCASE("negative control: a +Z mapping (or a lost rotation) is not what the importer produces") {
        CHECK(spotL->direction[2] != doctest::Approx(1.0f));
        CHECK(spotL->direction[1] != doctest::Approx(0.0f)); // not the unrotated -Z / +Z axis
        CHECK(sunL->direction[2] != doctest::Approx(1.0f));
    }
}

TEST_CASE("lights: importing can be switched off (declared rig) and a missing light index is skipped") {
    Quad q;
    q.nodesJson = R"([{"mesh":0},{"extensions":{"KHR_lights_punctual":{"light":0}}},{"extensions":{"KHR_lights_punctual":{"light":9}}}])";
    q.scenesNodes = "[0,1,2]";
    q.extra = R"("extensionsUsed":["KHR_lights_punctual"],"extensions":{"KHR_lights_punctual":{"lights":[{"type":"point"}]}},)";
    {
        Loaded l;
        load(l, quadJson(q), "phosphor_lights_missing.gltf");
        REQUIRE(l.ok);
        CHECK(l.lights == 1);
    }
    {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / "phosphor_lights_off.gltf";
        {
            std::ofstream out(path);
            out << quadJson(q);
        }
        GpuScene scene;
        NullTextureManager textures;
        textures.createDefaultTextures();
        ECS ecs;
        GltfLoader loader(scene, textures, ecs);
        loader.setImportPunctualLights(false);
        REQUIRE(loader.loadFromFile(path.string()));
        std::filesystem::remove(path);
        CHECK(loader.importedLightCount() == 0);
        CHECK(ecs.getArray<LightComponent>().size() == 0);
    }
}

// --- tangent repair (ticket 11) ---------------------------------------------------------------------------------------

namespace {
const char* kAnisoMaterial = R"({"name":"brushed","extensions":{"KHR_materials_anisotropy":{"anisotropyStrength":0.8}}})";

bool usable(const GPUVertex& v) {
    return tangentFrameUsable(glm::vec3(v.nx, v.ny, v.nz), glm::vec4(v.tx, v.ty, v.tz, v.tw));
}
} // namespace

TEST_CASE("tangent frames: degenerate tangents are repaired on anisotropic primitives only") {
    SUBCASE("supplied zero tangents") {
        Quad q;
        q.tangents = std::vector<float>(16, 0.0f);
        q.materialJson = kAnisoMaterial;
        q.extra = R"("extensionsUsed":["KHR_materials_anisotropy"],)";
        Loaded l;
        load(l, quadJson(q), "phosphor_tangent_zero_aniso.gltf");
        REQUIRE(l.ok);
        REQUIRE(l.scene.vertices().size() == 4);
        for (const GPUVertex& v : l.scene.vertices()) {
            CHECK(usable(v));
            // perpendicular to the +Z normal, unit length, handedness kept at +1
            CHECK(v.tx * v.nx + v.ty * v.ny + v.tz * v.nz == doctest::Approx(0.0f).epsilon(1e-5));
            CHECK(std::sqrt(v.tx * v.tx + v.ty * v.ty + v.tz * v.tz) == doctest::Approx(1.0f).epsilon(1e-5));
            CHECK(v.tw == doctest::Approx(1.0f));
        }

        // Negative control (the state before the fix): the same data on a material without anisotropy is left untouched.
        Quad plain = q;
        plain.materialJson = R"({"name":"plain"})";
        plain.extra.clear();
        Loaded p;
        load(p, quadJson(plain), "phosphor_tangent_zero_plain.gltf");
        REQUIRE(p.ok);
        REQUIRE(p.scene.vertices().size() == 4);
        for (const GPUVertex& v : p.scene.vertices()) CHECK_FALSE(usable(v));
    }
    SUBCASE("a tangent parallel to the normal and a valid one next to it") {
        Quad q;
        q.tangents = {0, 0, 1, -1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1}; // vertex 0: parallel to N with w=-1; others fine
        q.materialJson = kAnisoMaterial;
        q.extra = R"("extensionsUsed":["KHR_materials_anisotropy"],)";
        Loaded l;
        load(l, quadJson(q), "phosphor_tangent_parallel.gltf");
        REQUIRE(l.ok);
        u32 minusOne = 0, usableCount = 0;
        for (const GPUVertex& v : l.scene.vertices()) {
            usableCount += usable(v) ? 1u : 0u;
            minusOne += v.tw < 0.0f ? 1u : 0u;
        }
        CHECK(usableCount == l.scene.vertices().size());
        CHECK(minusOne >= 1); // the authored handedness of the repaired vertex survives
        // untouched vertices keep their authored tangent exactly
        u32 xAxis = 0;
        for (const GPUVertex& v : l.scene.vertices()) xAxis += (v.tx == 1.0f && v.ty == 0.0f && v.tz == 0.0f) ? 1u : 0u;
        CHECK(xAxis >= 3);
    }
    SUBCASE("generated tangents on a degenerate UV mapping (all vertices share one UV)") {
        Quad q;
        q.uvs = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
        q.materialJson = kAnisoMaterial;
        q.extra = R"("extensionsUsed":["KHR_materials_anisotropy"],)";
        Loaded l;
        load(l, quadJson(q), "phosphor_tangent_degenerate_uv.gltf");
        REQUIRE(l.ok);
        REQUIRE(l.scene.vertices().size() >= 4);
        for (const GPUVertex& v : l.scene.vertices()) CHECK(usable(v));
    }
}

TEST_CASE("tangent frames: repairTangentFrames unit behaviour") {
    GeometryStreams g;
    g.normals  = {{0, 0, 1}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}};
    g.tangents = {{0, 0, 0, 1}, {0, 5, 0, -1}, {1, 0, 0, 1}, {std::nanf(""), 0, 0, 1}};
    CHECK_FALSE(tangentFrameUsable(g.normals[0], g.tangents[0]));
    CHECK_FALSE(tangentFrameUsable(g.normals[1], g.tangents[1]));
    CHECK(tangentFrameUsable(g.normals[2], g.tangents[2]));
    CHECK_FALSE(tangentFrameUsable(g.normals[3], g.tangents[3]));
    CHECK(repairTangentFrames(g) == 3);
    for (size_t i = 0; i < 4; ++i) CHECK(tangentFrameUsable(g.normals[i], g.tangents[i]));
    CHECK(g.tangents[1].w == -1.0f);
    CHECK(g.tangents[2] == glm::vec4(1, 0, 0, 1));
    CHECK(repairTangentFrames(g) == 0); // idempotent
}

TEST_CASE("lights: the spot falloff deviation from the informative t^2 is declared in the support table") {
    const MaterialSupportTable& t = currentSupportTable();
    CHECK(t.punctualLights.level == SupportLevel::Approximated);
    CHECK(std::string(t.punctualLights.approximation) == "spot_falloff_smoothstep_not_informative_t2");
    CHECK(isExtensionRendered("KHR_lights_punctual", t));
}
