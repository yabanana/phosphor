#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "scene/ecs.h"
#include "scene/components.h"
#include "scene/gltf_loader.h"
#include "renderer/material_description.h"

#include <doctest/doctest.h>
#include <json.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "scene/gltf_contained_fs.h"
#include <tiny_gltf.h>

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

template <typename T>
void append(std::vector<unsigned char>& bytes, const std::vector<T>& values) {
    const size_t at = bytes.size();
    bytes.resize(at + values.size() * sizeof(T));
    std::memcpy(bytes.data() + at, values.data(), values.size() * sizeof(T));
}

// A quad facing +Z with POSITION, TEXCOORD_0 and indices only: no NORMAL,
// no TANGENT.  glTF UVs: origin at the top-left, V grows downwards.
std::vector<unsigned char> quadBytes() {
    const std::vector<float> positions = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    const std::vector<float> uvs       = {0, 1, 1, 1, 1, 0, 0, 0};
    const std::vector<unsigned short> indices = {0, 1, 2, 0, 2, 3}; // counter-clockwise about +Z
    std::vector<unsigned char> bytes;
    append(bytes, positions); // 48 bytes
    append(bytes, uvs);       // 32 bytes
    append(bytes, indices);   // 12 bytes
    return bytes;
}

std::string quadWithoutNormalsOrTangents(const std::string& materialJson = "") {
    const std::vector<unsigned char> bytes = quadBytes();
    return R"({
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}],
  "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2)" +
           std::string(materialJson.empty() ? "" : R"(, "material": 0)") + R"(}]}],)" +
           std::string(materialJson.empty() ? "" : R"(
  "materials": [)" + materialJson + R"(],)") + R"(
  "buffers": [{"byteLength": 92, "uri": "data:application/octet-stream;base64,)" + base64(bytes) + R"("}],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0,  "byteLength": 48},
    {"buffer": 0, "byteOffset": 48, "byteLength": 32},
    {"buffer": 0, "byteOffset": 80, "byteLength": 12}
  ],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
    {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}
  ]
})";
}

} // namespace

TEST_CASE("glTF loader: missing normals and tangents are generated per the spec") {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "phosphor_quad_no_tangents.gltf";
    {
        std::ofstream out(path);
        out << quadWithoutNormalsOrTangents();
    }

    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    REQUIRE(loader.loadFromFile(path.string()));
    std::filesystem::remove(path);

    REQUIRE(scene.getMeshCount() == 1);
    const auto& vertices = scene.vertices();
    REQUIRE(vertices.size() == 4); // flat normals and tangents agree: nothing split
    for (const GPUVertex& v : vertices) {
        CAPTURE(v.px);
        CAPTURE(v.py);
        // Flat normal of a counter-clockwise quad facing +Z.
        CHECK(v.nx == doctest::Approx(0.0f));
        CHECK(v.ny == doctest::Approx(0.0f));
        CHECK(v.nz == doctest::Approx(1.0f));
        // MikkTSpace tangent along +U = +X; bitangent cross(N, T) * w = +Y,
        // i.e. towards decreasing V (up in the image), so w = +1.
        CHECK(v.tx == doctest::Approx(1.0f));
        CHECK(v.ty == doctest::Approx(0.0f));
        CHECK(v.tz == doctest::Approx(0.0f));
        CHECK(v.tw == doctest::Approx(1.0f));
    }
}

TEST_CASE("glTF loader: doubleSided materials carry the flag to the GPU material") {
    for (const bool doubleSided : {false, true}) {
        CAPTURE(doubleSided);
        const std::filesystem::path path = std::filesystem::temp_directory_path() / "phosphor_quad_double_sided.gltf";
        {
            std::ofstream out(path);
            out << quadWithoutNormalsOrTangents(std::string(R"({"doubleSided": )") +
                                                (doubleSided ? "true" : "false") + "}");
        }

        GpuScene scene;
        NullTextureManager textures;
        textures.createDefaultTextures();
        ECS ecs;
        GltfLoader loader(scene, textures, ecs);
        REQUIRE(loader.loadFromFile(path.string()));
        std::filesystem::remove(path);

        REQUIRE(!scene.materials().empty());
        const bool flagged = (scene.materials().back().flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0;
        CHECK(flagged == doubleSided);
    }
}

TEST_CASE("glTF loader preserves primitive materials and reuses geometry across nodes") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(R"({"pbrMetallicRoughness":{"baseColorFactor":[1,0,0,1]}})"));
    auto primitive = model["meshes"][0]["primitives"][0];
    primitive["material"] = 1;
    model["meshes"][0]["primitives"].push_back(primitive);
    model["materials"].push_back(nlohmann::json::parse(R"({"pbrMetallicRoughness":{"baseColorFactor":[0,1,0,1]}})"));
    model["nodes"].push_back({{"mesh",0}, {"matrix",{-1,0,0,0, 0.5,1,0,0, 0,0,1,0, 3,0,0,1}}});
    model["scenes"][0]["nodes"].push_back(1);
    const auto path = std::filesystem::temp_directory_path() / "phosphor_primitive_materials.gltf";
    { std::ofstream out(path); out << model; }
    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    REQUIRE(loader.loadFromFile(path.string()));
    std::filesystem::remove(path);
    REQUIRE(loader.createdEntities().size() == 4);
    REQUIRE(scene.getMeshCount() == 2);
    for (size_t i = 0; i < 4; ++i) {
        const auto id = loader.createdEntities()[i];
        const auto& instance = ecs.getComponent<MeshInstanceComponent>(id);
        const auto& material = scene.materials()[instance.materialIndex];
        CHECK(material.baseColor[0] == (i % 2 == 0 ? 1.0f : 0.0f));
        CHECK(material.baseColor[1] == (i % 2 == 0 ? 0.0f : 1.0f));
        if (i >= 2) {
            const auto& matrix = ecs.getComponent<TransformComponent>(id).worldMatrix;
            CHECK(matrix[0][0] == -1.0f);
            CHECK(matrix[1][0] == 0.5f);
            CHECK(matrix[3][0] == 3.0f);
        }
    }
}

TEST_CASE("glTF alpha cutoff applies only to MASK, never OPAQUE") {
    for (const auto* mode : {"OPAQUE", "MASK"}) {
        const auto path = std::filesystem::temp_directory_path() / "phosphor_alpha_mode.gltf";
        { std::ofstream out(path); out << quadWithoutNormalsOrTangents(
            std::string(R"({"alphaMode":")") + mode + R"(","alphaCutoff":0.7})"); }
        GpuScene scene;
        NullTextureManager textures;
        textures.createDefaultTextures();
        ECS ecs;
        GltfLoader loader(scene, textures, ecs);
        REQUIRE(loader.loadFromFile(path.string()));
        std::filesystem::remove(path);
        CHECK(scene.materials().back().alphaCutoff == doctest::Approx(std::string(mode) == "MASK" ? 0.7f : 0.0f));
    }
}

TEST_CASE("glTF implicit default material does not reuse a preceding explicit material") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(R"({"pbrMetallicRoughness":{"baseColorFactor":[1,0,0,1]}})"));
    auto primitive = model["meshes"][0]["primitives"][0];
    primitive.erase("material");
    model["meshes"][0]["primitives"].push_back(primitive);
    const auto path = std::filesystem::temp_directory_path() / "phosphor_default_material.gltf";
    { std::ofstream out(path); out << model; }
    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    REQUIRE(loader.loadFromFile(path.string()));
    std::filesystem::remove(path);
    REQUIRE(loader.createdEntities().size() == 2);
    const auto first = ecs.getComponent<MeshInstanceComponent>(loader.createdEntities()[0]).materialIndex;
    const auto second = ecs.getComponent<MeshInstanceComponent>(loader.createdEntities()[1]).materialIndex;
    REQUIRE(first != second);
    CHECK(scene.materials()[second].baseColor[1] == 1.0f);
    CHECK(scene.materials()[second].metallic == 1.0f);
    CHECK(scene.materials()[second].alphaCutoff == 0.0f);
}

TEST_CASE("glTF texture reuse keeps sRGB and linear interpretations distinct") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0}},"normalTexture":{"index":0}})"));
    model["images"] = {{{"uri", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j4L8AAAAASUVORK5CYII="}}};
    model["textures"] = {{{"source",0}}};
    const auto path = std::filesystem::temp_directory_path() / "phosphor_texture_color_space.gltf";
    { std::ofstream out(path); out << model; }
    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    REQUIRE(loader.loadFromFile(path.string()));
    std::filesystem::remove(path);
    const auto& material = scene.materials().back();
    REQUIRE(material.baseColorTex != material.normalTex);
    CHECK(textures.uploads[material.baseColorTex].sRGB);
    CHECK_FALSE(textures.uploads[material.normalTex].sRGB);
}

namespace {
bool loadModelJson(const nlohmann::json& model, GpuScene& scene, const char* name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    { std::ofstream out(path); out << model; }
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    const bool ok = loader.loadFromFile(path.string());
    std::filesystem::remove(path);
    return ok;
}
const char* kTinyPng = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==";
} // namespace

TEST_CASE("glTF loader (F15): materials keep description, name and support report; core-only mask is empty") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"name":"plain","pbrMetallicRoughness":{"baseColorFactor":[1,0,0,1]}})"));
    GpuScene scene;
    REQUIRE(loadModelJson(model, scene, "phosphor_f15_core_only.gltf"));
    REQUIRE(scene.materials().size() == 1);
    REQUIRE(scene.materialDescriptions().size() == 1);
    CHECK(featureMask(scene.materialDescriptions()[0]) == 0);
    CHECK(scene.materialNames()[0] == "plain");
    CHECK(scene.findMaterial("plain") == std::optional<u32>(0u));
    CHECK_FALSE(scene.findMaterial("nope").has_value());
    CHECK(scene.materialDescriptions()[0].base.baseColor[0] == 1.0f);
    REQUIRE(scene.materialSupport()[0].valid());
    CHECK(scene.materialSupport()[0].features.empty());
}

TEST_CASE("glTF loader (F15): extension materials are described and emissive strength is folded") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"name":"coat","emissiveFactor":[1,0.5,0],"extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":1},
            "KHR_materials_emissive_strength":{"emissiveStrength":4}}})"));
    model["extensionsUsed"] = {"KHR_materials_clearcoat", "KHR_materials_emissive_strength", "KHR_lights_punctual"};
    GpuScene scene;
    REQUIRE(loadModelJson(model, scene, "phosphor_f15_ext.gltf"));
    CHECK(featureMask(scene.materialDescriptions()[0]) == featureBit(MaterialFeature::Clearcoat));
    CHECK(scene.materials()[0].emissive[0] == doctest::Approx(4.0f));
    CHECK(scene.materials()[0].emissive[1] == doctest::Approx(2.0f));
    REQUIRE(scene.assetUnknownExtensions().size() == 1);
    CHECK(scene.assetUnknownExtensions()[0] == "KHR_lights_punctual");
}

TEST_CASE("glTF loader (F15): extensionsRequired with an unsupported extension is refused") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(R"({"name":"m"})"));
    model["extensionsUsed"]     = {"KHR_materials_unlit"};
    model["extensionsRequired"] = {"KHR_materials_unlit"};
    GpuScene scene;
    CHECK_FALSE(loadModelJson(model, scene, "phosphor_f15_required.gltf"));
    CHECK(scene.getMeshCount() == 0);
}

TEST_CASE("glTF loader (F15): a uniform KHR_texture_transform is baked into UV0") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{"offset":[0.5,0.25],"scale":[2,3]}}}}})"));
    model["textures"] = nlohmann::json::array({nlohmann::json{{"source", 0}}});
    model["images"]   = nlohmann::json::array({nlohmann::json{{"uri", kTinyPng}}});
    GpuScene scene;
    REQUIRE(loadModelJson(model, scene, "phosphor_f15_uv_bake.gltf"));
    REQUIRE(scene.vertices().size() == 4);
    // Quad UVs (0,1) (1,1) (1,0) (0,0); uv' = offset + scale * uv.
    int seen = 0;
    for (const GPUVertex& v : scene.vertices()) {
        if (v.px == -1 && v.py == -1) { CHECK(v.u == doctest::Approx(0.5f)); CHECK(v.v == doctest::Approx(3.25f)); ++seen; }
        if (v.px == 1 && v.py == -1)  { CHECK(v.u == doctest::Approx(2.5f)); CHECK(v.v == doctest::Approx(3.25f)); ++seen; }
        if (v.px == 1 && v.py == 1)   { CHECK(v.u == doctest::Approx(2.5f)); CHECK(v.v == doctest::Approx(0.25f)); ++seen; }
        if (v.px == -1 && v.py == 1)  { CHECK(v.u == doctest::Approx(0.5f)); CHECK(v.v == doctest::Approx(0.25f)); ++seen; }
    }
    CHECK(seen == 4);
}

TEST_CASE("glTF loader (F15): a non-uniform transform leaves UVs untouched and is reported unsupported") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{"scale":[2,2]}}}},
            "normalTexture":{"index":0}})"));
    model["textures"] = nlohmann::json::array({nlohmann::json{{"source", 0}}});
    model["images"]   = nlohmann::json::array({nlohmann::json{{"uri", kTinyPng}}});
    GpuScene scene;
    REQUIRE(loadModelJson(model, scene, "phosphor_f15_uv_nonuniform.gltf"));
    for (const GPUVertex& v : scene.vertices()) CHECK((v.u == 0.0f || v.u == 1.0f));
    bool unsupported = false;
    for (const auto& f : scene.materialSupport()[0].encodings)
        if (f.feature == "texture_transform") unsupported = f.level == SupportLevel::Unsupported;
    CHECK(unsupported);
}

TEST_CASE("glTF loader (F15): image URIs outside the asset directory are refused") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}})"));
    model["textures"] = nlohmann::json::array({nlohmann::json{{"source", 0}}});
    model["images"]   = nlohmann::json::array({nlohmann::json{{"uri", "../escape.png"}}});
    GpuScene scene;
    CHECK_FALSE(loadModelJson(model, scene, "phosphor_f15_uri_escape.gltf"));
}

// ---------------------------------------------------------------------------
// Review (ticket 04): URI containment before any read, for images AND buffers.
// ---------------------------------------------------------------------------
namespace {

namespace fs = std::filesystem;

// asset/ (the glTF and an inside buffer/image) next to outside/ (sentinels).
struct UriSandbox {
    fs::path root, asset, outside;
    explicit UriSandbox(const char* name) {
        root    = fs::temp_directory_path() / name;
        asset   = root / "asset";
        outside = root / "outside";
        fs::remove_all(root);
        fs::create_directories(asset / "data");
        fs::create_directories(outside);
        write(outside / "secret.bin", quadBytes());
        write(outside / "secret.png", {1, 2, 3});
        write(asset / "data" / "quad.bin", quadBytes());
    }
    ~UriSandbox() { fs::remove_all(root); }
    static void write(const fs::path& p, const std::vector<unsigned char>& bytes) {
        std::ofstream out(p, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    // The quad asset with its buffer at `bufferUri` and optionally one image URI.
    fs::path writeGltf(const std::string& bufferUri, const std::string& imageUri = "") const {
        auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents());
        model["buffers"][0]["uri"] = bufferUri;
        if (!imageUri.empty()) {
            model["images"]   = nlohmann::json::array({nlohmann::json{{"uri", imageUri}}});
            model["textures"] = nlohmann::json::array({nlohmann::json{{"source", 0}}});
        }
        const fs::path path = asset / "model.gltf";
        std::ofstream out(path);
        out << model;
        return path;
    }
};

struct RawLoad {
    bool                     ok = false;
    std::vector<std::string> refusals, reads;
    std::string              err;
};

RawLoad rawLoad(const fs::path& path) {
    ContainedFileSystem files(path.string());
    tinygltf::TinyGLTF loader;
    std::string installError;
    REQUIRE(files.install(loader, installError));
    tinygltf::Model model;
    RawLoad r;
    std::string warn;
    r.ok       = loader.LoadASCIIFromFile(&model, &r.err, &warn, path.string());
    r.refusals = files.refusals();
    r.reads    = files.reads();
    return r;
}

bool readSomething(const std::vector<std::string>& reads, const std::string& needle) {
    for (const auto& r : reads)
        if (r.find(needle) != std::string::npos) return true;
    return false;
}

} // namespace

TEST_CASE("glTF contained file system: an escaping buffer or image URI is refused without opening the file") {
    UriSandbox box("phosphor_uri_fs_escape");
    SUBCASE("buffer with a .. segment") {
        const auto r = rawLoad(box.writeGltf("../outside/secret.bin"));
        CHECK_FALSE(r.ok);
        REQUIRE(r.refusals.size() == 1);
        CHECK(r.refusals[0].find("../outside/secret.bin") != std::string::npos);
        CHECK_FALSE(readSomething(r.reads, "secret.bin"));
    }
    SUBCASE("buffer with a percent-encoded .. segment") {
        const auto r = rawLoad(box.writeGltf("%2e%2e/outside/secret.bin"));
        CHECK_FALSE(r.ok);
        CHECK_FALSE(r.refusals.empty());
        CHECK_FALSE(readSomething(r.reads, "secret.bin"));
    }
    SUBCASE("buffer with an absolute path") {
        const auto r = rawLoad(box.writeGltf((box.outside / "secret.bin").string()));
        CHECK_FALSE(r.ok);
        CHECK_FALSE(r.refusals.empty());
        CHECK_FALSE(readSomething(r.reads, "secret.bin"));
    }
    SUBCASE("buffer through a symlink to the outside") {
        std::error_code ec;
        fs::create_directory_symlink(box.outside, box.asset / "link", ec);
        if (ec) return; // symlinks unavailable on this file system
        const auto r = rawLoad(box.writeGltf("link/secret.bin"));
        CHECK_FALSE(r.ok);
        REQUIRE_FALSE(r.refusals.empty());
        CHECK(r.refusals[0].find("outside the asset directory") != std::string::npos);
        CHECK_FALSE(readSomething(r.reads, "secret.bin"));
    }
    SUBCASE("image with a .. segment (inside buffer is fine)") {
        const auto r = rawLoad(box.writeGltf("data/quad.bin", "../outside/secret.png"));
        REQUIRE_FALSE(r.refusals.empty());
        CHECK(r.refusals[0].find("../outside/secret.png") != std::string::npos);
        CHECK_FALSE(readSomething(r.reads, "secret.png"));
    }
}

TEST_CASE("glTF contained file system: files inside the asset directory are read, nothing from the current directory") {
    UriSandbox box("phosphor_uri_fs_inside");
    const auto r = rawLoad(box.writeGltf("data/quad.bin"));
    CHECK(r.ok);
    CHECK(r.refusals.empty());
    CHECK(readSomething(r.reads, "model.gltf"));
    CHECK(readSomething(r.reads, "data/quad.bin"));
    // A missing buffer is a plain failure, not a refusal and never found elsewhere.
    const auto missing = rawLoad(box.writeGltf("data/absent.bin"));
    CHECK_FALSE(missing.ok);
    CHECK(missing.refusals.empty());
    // The cwd fallback must not rescue it: put a same-named file in the cwd.
    const fs::path cwd = fs::current_path();
    fs::current_path(box.outside);
    UriSandbox::write(box.outside / "elsewhere.bin", quadBytes());
    const auto fallback = rawLoad(box.writeGltf("elsewhere.bin"));
    fs::current_path(cwd);
    CHECK_FALSE(fallback.ok);
    CHECK_FALSE(readSomething(fallback.reads, "elsewhere.bin"));
}

TEST_CASE("glTF loader: an escaping buffer URI fails the load and names the URI") {
    UriSandbox box("phosphor_uri_loader_escape");
    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    // The sentinel is a valid buffer: without containment it would load.
    CHECK_FALSE(loader.loadFromFile(box.writeGltf("../outside/secret.bin").string()));
    CHECK(loader.lastError().find("../outside/secret.bin") != std::string::npos);
    CHECK(scene.getMeshCount() == 0);
    // The same asset with an inside buffer loads.
    CHECK(loader.loadFromFile(box.writeGltf("data/quad.bin").string()));
    CHECK(loader.lastError().empty());
    CHECK(scene.getMeshCount() == 1);
}

TEST_CASE("glTF loader: a required extension whose feature is not rendered refuses the asset, an optional one loads") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"extensions":{"KHR_materials_transmission":{"transmissionFactor":0.5}}})"));
    model["extensionsUsed"] = {"KHR_materials_transmission"};
    SUBCASE("optional") {
        GpuScene scene;
        CHECK(loadModelJson(model, scene, "phosphor_f15_optional_transmission.gltf"));
    }
    SUBCASE("required") {
        // Transmission is rendered since F15.1 ticket 13: a required extension that nothing renders (dispersion is F16) is the refused case.
        model["extensionsUsed"] = {"KHR_materials_dispersion"};
        model["extensionsRequired"] = {"KHR_materials_dispersion"};
        const auto path = std::filesystem::temp_directory_path() / "phosphor_f15_required_dispersion.gltf";
        { std::ofstream out(path); out << model; }
        GpuScene scene;
        NullTextureManager textures;
        textures.createDefaultTextures();
        ECS ecs;
        GltfLoader loader(scene, textures, ecs);
        CHECK_FALSE(loader.loadFromFile(path.string()));
        std::filesystem::remove(path);
        CHECK(loader.lastError().find("KHR_materials_dispersion") != std::string::npos);
        CHECK(scene.getMeshCount() == 0);
    }
}

// ---------------------------------------------------------------------------
// Review (ticket 04): out-of-range input and once-per-material parsing.
// ---------------------------------------------------------------------------
namespace {
bool anyContains(const std::vector<std::string>& list, const std::string& needle) {
    for (const auto& s : list)
        if (s.find(needle) != std::string::npos) return true;
    return false;
}
} // namespace

TEST_CASE("glTF loader: a negative emissiveStrength keeps the authored value but the GPU emissive is clamped to 0") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"name":"neg","emissiveFactor":[1,0.5,0],"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":-2}}})"));
    model["extensionsUsed"] = {"KHR_materials_emissive_strength"};
    GpuScene scene;
    REQUIRE(loadModelJson(model, scene, "phosphor_f15_negative_emissive.gltf"));
    // description: authored -2 * (1, 0.5, 0)
    CHECK(scene.materialDescriptions()[0].base.emissive[0] == doctest::Approx(-2.0f));
    CHECK(scene.materialDescriptions()[0].base.emissive[1] == doctest::Approx(-1.0f));
    // GPU: never negative
    for (float e : scene.materials()[0].emissive) CHECK(e == 0.0f);
    CHECK(anyContains(scene.materialSupport()[0].warnings, "emissiveStrength"));
}

TEST_CASE("glTF loader: a negative texCoord is a warning listed in the support report") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"name":"neg","pbrMetallicRoughness":{"baseColorTexture":{"index":0,"texCoord":-1}}})"));
    model["textures"] = nlohmann::json::array({nlohmann::json{{"source", 0}}});
    model["images"]   = nlohmann::json::array({nlohmann::json{{"uri", kTinyPng}}});
    GpuScene scene;
    REQUIRE(loadModelJson(model, scene, "phosphor_f15_negative_texcoord.gltf"));
    CHECK(anyContains(scene.materialSupport()[0].warnings, "negative texCoord"));
    CHECK(scene.materialDescriptions()[0].base.baseColorTexture.texCoord == 0u);
}

TEST_CASE("glTF loader: a texture index out of range is no texture, with a warning") {
    // textures[] has one entry; index 7 does not exist (core slot and extension slot).
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(
        R"({"name":"oob","pbrMetallicRoughness":{"baseColorTexture":{"index":7}},
            "extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":1,"clearcoatTexture":{"index":9}}}})"));
    model["textures"] = nlohmann::json::array({nlohmann::json{{"source", 0}}});
    model["images"]   = nlohmann::json::array({nlohmann::json{{"uri", kTinyPng}}});
    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    const auto path = std::filesystem::temp_directory_path() / "phosphor_f15_oob_texture.gltf";
    { std::ofstream out(path); out << model; }
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    REQUIRE(loader.loadFromFile(path.string()));
    std::filesystem::remove(path);
    const auto& d = scene.materialDescriptions()[0];
    CHECK_FALSE(d.base.baseColorTexture.present());
    REQUIRE(d.clearcoat.has_value());
    CHECK_FALSE(d.clearcoat->factorTexture.present());
    // The GPU record binds the documented default of the slot.
    CHECK(scene.materials()[0].baseColorTex == textures.getDefaultWhite());
    CHECK(anyContains(scene.materialSupport()[0].warnings, "baseColorTexture"));
    CHECK(anyContains(scene.materialSupport()[0].warnings, "clearcoatTexture"));
}

TEST_CASE("glTF loader: each material is parsed once, however many primitives use it") {
    auto model = nlohmann::json::parse(quadWithoutNormalsOrTangents(R"({"name":"shared"})"));
    auto primitive = model["meshes"][0]["primitives"][0];
    model["meshes"][0]["primitives"].push_back(primitive);
    model["meshes"][0]["primitives"].push_back(primitive);
    model["materials"].push_back(nlohmann::json::parse(R"({"name":"unused"})"));
    model["nodes"].push_back({{"mesh", 0}});
    model["scenes"][0]["nodes"].push_back(1);
    const auto path = std::filesystem::temp_directory_path() / "phosphor_f15_parse_once.gltf";
    { std::ofstream out(path); out << model; }
    GpuScene scene;
    NullTextureManager textures;
    textures.createDefaultTextures();
    ECS ecs;
    GltfLoader loader(scene, textures, ecs);
    REQUIRE(loader.loadFromFile(path.string()));
    std::filesystem::remove(path);
    CHECK(loader.createdEntities().size() == 6); // two nodes x three primitives
    CHECK(scene.materials().size() == 1);
    CHECK(loader.materialParseCount() == 1);
}
