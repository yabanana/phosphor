#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "scene/ecs.h"
#include "scene/components.h"
#include "scene/gltf_loader.h"

#include <doctest/doctest.h>
#include <json.hpp>

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

template <typename T>
void append(std::vector<unsigned char>& bytes, const std::vector<T>& values) {
    const size_t at = bytes.size();
    bytes.resize(at + values.size() * sizeof(T));
    std::memcpy(bytes.data() + at, values.data(), values.size() * sizeof(T));
}

// A quad facing +Z with POSITION, TEXCOORD_0 and indices only: no NORMAL,
// no TANGENT.  glTF UVs: origin at the top-left, V grows downwards.
std::string quadWithoutNormalsOrTangents(const std::string& materialJson = "") {
    const std::vector<float> positions = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    const std::vector<float> uvs       = {0, 1, 1, 1, 1, 0, 0, 0};
    const std::vector<unsigned short> indices = {0, 1, 2, 0, 2, 3}; // counter-clockwise about +Z
    std::vector<unsigned char> bytes;
    append(bytes, positions); // 48 bytes
    append(bytes, uvs);       // 32 bytes
    append(bytes, indices);   // 12 bytes
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
