#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "scene/ecs.h"
#include "scene/gltf_loader.h"

#include <doctest/doctest.h>

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
