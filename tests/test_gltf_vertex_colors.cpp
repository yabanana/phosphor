// F15 closure: glTF COLOR_0 (vertex colour) import.  Expected values come from the specification (VEC3 alpha 1, normalised unsigned byte / short
// divided by 255 / 65535, floats as they are), not from the code under test.
#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "scene/ecs.h"
#include "scene/gltf_loader.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace phosphor;
using phosphor::test::NullTextureManager;

namespace {

std::string b64(const std::vector<unsigned char>& data) {
    static const char* k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < data.size(); i += 3) {
        const uint32_t n = (uint32_t(data[i]) << 16) | (i + 1 < data.size() ? uint32_t(data[i + 1]) << 8 : 0) | (i + 2 < data.size() ? uint32_t(data[i + 2]) : 0);
        out += k[(n >> 18) & 63];
        out += k[(n >> 12) & 63];
        out += i + 1 < data.size() ? k[(n >> 6) & 63] : '=';
        out += i + 2 < data.size() ? k[n & 63] : '=';
    }
    return out;
}

template <class T> void put(std::vector<unsigned char>& b, const std::vector<T>& v) {
    const size_t at = b.size();
    b.resize(at + v.size() * sizeof(T));
    std::memcpy(b.data() + at, v.data(), v.size() * sizeof(T));
}

// Two quads (two primitives, two nodes): the first optionally carries COLOR_0 of `colorType` / `componentType` (normalised for the integers),
// the second never does.  `colorBytes` is the raw accessor data of the first quad (4 vertices).
std::string twoQuads(const std::string& colorType, int componentType, bool normalized, const std::vector<unsigned char>& colorBytes, bool firstHasColor = true) {
    std::vector<unsigned char> bytes;
    put(bytes, std::vector<float>{-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0});          // 0..48 positions
    put(bytes, std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1});              // 48..96 normals
    put(bytes, std::vector<float>{0, 1, 1, 1, 1, 0, 0, 0});                          // 96..128 uvs
    put(bytes, std::vector<unsigned short>{0, 1, 2, 0, 2, 3});                       // 128..140 indices
    const size_t colorOffset = bytes.size();
    bytes.insert(bytes.end(), colorBytes.begin(), colorBytes.end());
    const std::string colorAttr = firstHasColor ? R"(,"COLOR_0":4)" : "";
    return R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],"nodes":[{"mesh":0},{"mesh":1,"translation":[3,0,0]}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2)" + colorAttr + R"(},"indices":3}]},
                {"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2},"indices":3}]}],
      "buffers":[{"byteLength":)" + std::to_string(bytes.size()) + R"(,"uri":"data:application/octet-stream;base64,)" + b64(bytes) + R"("}],
      "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":48},{"buffer":0,"byteOffset":96,"byteLength":32},
                     {"buffer":0,"byteOffset":128,"byteLength":12},{"buffer":0,"byteOffset":)" + std::to_string(colorOffset) + R"(,"byteLength":)" +
           std::to_string(colorBytes.size()) + R"(}],
      "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1,-1,0],"max":[1,1,0]},{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},
                   {"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},{"bufferView":3,"componentType":5123,"count":6,"type":"SCALAR"},
                   {"bufferView":4,"componentType":)" + std::to_string(componentType) + R"(,"count":4,"type":")" + colorType + R"(")" +
           (normalized ? R"(,"normalized":true)" : "") + "}]}";
}

struct Loaded {
    GpuScene scene;
    NullTextureManager textures;
    ECS ecs;
    bool ok = false;
};

void load(Loaded& l, const std::string& json, const char* name) {
    const auto path = std::filesystem::temp_directory_path() / name;
    { std::ofstream(path) << json; }
    l.textures.createDefaultTextures();
    GltfLoader loader(l.scene, l.textures, l.ecs);
    l.ok = loader.loadFromFile(path.string());
    std::filesystem::remove(path);
}

} // namespace

TEST_CASE("COLOR_0: float VEC4 is imported as is, the second (uncoloured) mesh is white, the stream is parallel to the vertices") {
    std::vector<unsigned char> raw;
    put(raw, std::vector<float>{1, 0, 0, 1, 0, 1, 0, 0.5f, 0, 0, 1, 0.25f, 0.2f, 0.4f, 0.6f, 0.8f});
    Loaded l;
    load(l, twoQuads("VEC4", 5126, false, raw), "phosphor_color0_float4.gltf");
    REQUIRE(l.ok);
    REQUIRE(l.scene.hasVertexColors());
    REQUIRE(l.scene.vertexColors().size() == l.scene.vertices().size());
    CHECK(l.scene.vertices().size() == 8);
    const auto& c = l.scene.vertexColors();
    CHECK(c[0] == glm::vec4(1, 0, 0, 1));
    CHECK(c[1] == glm::vec4(0, 1, 0, 0.5f));
    CHECK(c[2] == glm::vec4(0, 0, 1, 0.25f));
    CHECK(c[3].a == doctest::Approx(0.8f));
    for (size_t i = 4; i < 8; ++i) CHECK(c[i] == glm::vec4(1.0f)); // the uncoloured mesh: white, alpha 1 (identity of the product)
}

TEST_CASE("COLOR_0: VEC3 gets alpha 1, normalised unsigned byte and short are divided by 255 and 65535") {
    SUBCASE("float VEC3") {
        std::vector<unsigned char> raw;
        put(raw, std::vector<float>{0.1f, 0.2f, 0.3f, 1, 1, 1, 0, 0, 0, 0.5f, 0.5f, 0.5f});
        Loaded l;
        load(l, twoQuads("VEC3", 5126, false, raw), "phosphor_color0_float3.gltf");
        REQUIRE(l.ok);
        CHECK(l.scene.vertexColors()[0] == glm::vec4(0.1f, 0.2f, 0.3f, 1.0f));
    }
    SUBCASE("unsigned byte VEC4 normalised") {
        std::vector<unsigned char> raw = {255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 51, 102, 153, 204};
        Loaded l;
        load(l, twoQuads("VEC4", 5121, true, raw), "phosphor_color0_ubyte.gltf");
        REQUIRE(l.ok);
        const auto& c = l.scene.vertexColors();
        CHECK(c[1].g == doctest::Approx(1.0f));
        CHECK(c[1].a == doctest::Approx(128.0f / 255.0f));
        CHECK(c[3].r == doctest::Approx(0.2f));
        CHECK(c[3].a == doctest::Approx(0.8f));
    }
    SUBCASE("unsigned short VEC3 normalised") {
        std::vector<unsigned char> raw;
        put(raw, std::vector<unsigned short>{65535, 0, 0, 0, 65535, 0, 0, 0, 65535, 32768, 32768, 32768});
        Loaded l;
        load(l, twoQuads("VEC3", 5123, true, raw), "phosphor_color0_ushort.gltf");
        REQUIRE(l.ok);
        const auto& c = l.scene.vertexColors();
        CHECK(c[0] == glm::vec4(1, 0, 0, 1));
        CHECK(c[3].r == doctest::Approx(32768.0f / 65535.0f));
        CHECK(c[3].a == 1.0f);
    }
}

TEST_CASE("COLOR_0: a scene without vertex colours has no stream (zero cost), an invalid COLOR_0 is ignored with the stream absent") {
    {
        Loaded l;
        load(l, twoQuads("VEC4", 5126, false, {}, false), "phosphor_color0_none.gltf");
        REQUIRE(l.ok);
        CHECK_FALSE(l.scene.hasVertexColors());
        CHECK(l.scene.vertexColors().empty());
    }
    {   // negative control: SCALAR type is not a colour: ignored, no stream
        std::vector<unsigned char> raw;
        put(raw, std::vector<float>{1, 2, 3, 4});
        Loaded l;
        load(l, twoQuads("SCALAR", 5126, false, raw), "phosphor_color0_scalar.gltf");
        REQUIRE(l.ok);
        CHECK_FALSE(l.scene.hasVertexColors());
    }
}

TEST_CASE("COLOR_0: GpuScene::clear drops the stream") {
    std::vector<unsigned char> raw;
    put(raw, std::vector<float>{1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1});
    Loaded l;
    load(l, twoQuads("VEC4", 5126, false, raw), "phosphor_color0_clear.gltf");
    REQUIRE(l.scene.hasVertexColors());
    l.scene.clear();
    CHECK_FALSE(l.scene.hasVertexColors());
}

#include "pipeline/forward_variants.h"
#include "renderer/scene_store.h"

TEST_CASE("COLOR_0: the scene store marks the pipeline union only for a coloured scene; withVertexColor swaps the twin entry points of consistent pairs") {
    {
        std::vector<unsigned char> raw;
        put(raw, std::vector<float>{1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1});
        Loaded l;
        load(l, twoQuads("VEC4", 5126, false, raw), "phosphor_color0_store.gltf");
        REQUIRE(l.ok);
        SceneStore store;
        store.sync(l.ecs, l.scene);
        CHECK(store.vertexColorInUse());
        CHECK((store.pipelineFeatureUnion() & FEATURE_UNION_VERTEX_COLOR) != 0u);
        CHECK(store.featureUnion() == 0u); // a colour stream is no material feature: the lobe union stays 0
    }
    {   // negative control: no COLOR_0, no marker, the union is what it was before this feature
        Loaded l;
        load(l, twoQuads("VEC4", 5126, false, {}, false), "phosphor_color0_store_none.gltf");
        REQUIRE(l.ok);
        SceneStore store;
        store.sync(l.ecs, l.scene);
        CHECK_FALSE(store.vertexColorInUse());
        CHECK(store.pipelineFeatureUnion() == store.featureUnion());
    }
    pipe::PipelineDesc indexed;
    indexed.kind = pipe::PipelineKind::Render;
    indexed.functions = {"forward_surface_vs", "forward_di_receiver_guide_fs", ""};
    const auto twin = pipe::forward::withVertexColor(indexed);
    CHECK(twin.functions[0] == "forward_surface_vs_vc");
    CHECK(twin.functions[1] == "forward_di_receiver_guide_fs_vc");
    CHECK(twin.constantCount == indexed.constantCount + 1);
    pipe::PipelineDesc meshForward;
    meshForward.kind = pipe::PipelineKind::Render;
    meshForward.functions = {"meshlet_object", "meshlet_mesh", "forward_fs"};
    CHECK(pipe::forward::withVertexColor(meshForward).functions[1] == "meshlet_mesh_vc");
    CHECK(pipe::forward::withVertexColor(meshForward).functions[2] == "forward_fs_vc");
    pipe::PipelineDesc debugMesh;
    debugMesh.kind = pipe::PipelineKind::Mesh;
    debugMesh.functions = {"meshlet_object", "meshlet_mesh_debug", "meshlet_debug_fs"}; // a pair without twins is left alone
    CHECK(pipe::forward::withVertexColor(debugMesh).functions[1] == "meshlet_mesh_debug");
    pipe::PipelineDesc alpha;
    alpha.kind = pipe::PipelineKind::Render;
    alpha.functions = {"visibility_object_alpha", "visibility_mesh", "visibility_alpha_lit_tt_fs"};
    const auto alphaTwin = pipe::forward::withVertexColor(alpha);
    CHECK(alphaTwin.functions[1] == "visibility_mesh_vc");
    CHECK(alphaTwin.functions[2] == "visibility_alpha_lit_tt_vc_fs");
    pipe::PipelineDesc kernel;
    kernel.kind = pipe::PipelineKind::Compute;
    kernel.functions = {"visibility_lit_resolve", "", ""};
    CHECK(pipe::forward::withVertexColor(kernel).functions[0] == "visibility_lit_resolve"); // compute kernels only get the constant
}
