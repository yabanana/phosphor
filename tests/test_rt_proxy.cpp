#include <doctest/doctest.h>

#include "renderer/gpu_scene.h"
#include "renderer/rt_proxy.h"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>

using namespace phosphor;

namespace {

struct Grid {
    std::vector<GPUVertex> vertices;
    std::vector<u32> indices;
};
Grid grid(u32 n = 12) {
    Grid g;
    for (u32 y = 0; y <= n; ++y) {
        for (u32 x = 0; x <= n; ++x) {
            GPUVertex v{};
            v.px = float(x); v.pz = float(y); v.py = 0.01f * float(x * x + y * y);
            v.ny = 1; v.tx = v.tw = 1; v.u = float(x) / n; v.v = float(y) / n;
            g.vertices.push_back(v);
        }
    }
    for (u32 y = 0; y < n; ++y) {
        for (u32 x = 0; x < n; ++x) {
            const u32 a = y * (n + 1) + x, b = a + 1, c = a + n + 1, d = c + 1;
            g.indices.insert(g.indices.end(), {a, c, b, b, c, d});
        }
    }
    return g;
}
void addGrid(GpuScene& scene, u32 n = 12, float translateX = 0) {
    auto g = grid(n);
    std::vector<glm::vec3> positions, normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> uvs;
    for (const auto& v : g.vertices) {
        positions.push_back({v.px + translateX, v.py, v.pz});
        normals.push_back({v.nx, v.ny, v.nz});
        tangents.push_back({v.tx, v.ty, v.tz, v.tw});
        uvs.push_back({v.u, v.v});
    }
    scene.uploadMesh(positions, normals, tangents, uvs, g.indices);
}
std::map<std::pair<u32, u32>, u32> borders(std::span<const u32> indices) {
    std::map<std::pair<u32, u32>, u32> edges;
    for (size_t i = 0; i < indices.size(); i += 3) {
        for (size_t e = 0; e < 3; ++e) {
            const auto a = indices[i + e], b = indices[i + (e + 1) % 3];
            edges[{std::min(a, b), std::max(a, b)}]++;
        }
    }
    std::erase_if(edges, [](const auto& e) { return e.second != 1; });
    return edges;
}
RtProxyManifest manifestFor(const GpuScene& scene) {
    std::vector<RtProxyLevel> levels(scene.getMeshCount(), RtProxyLevel::R10);
    // Synthetic test evidence only. Production manifests require actual S4 rays.
    return rtMakeProxyManifest(scene, levels, {100, 100, 0.1, 0.1, 0.1, 0.1}, "unit-grid", "synthetic-unit-fixture");
}
struct TempFile {
    std::filesystem::path directory;
    std::filesystem::path path;
    TempFile() {
        static u32 serial = 0;
        directory = std::filesystem::temp_directory_path() /
            ("phosphor-rt-proxy-" + std::to_string(reinterpret_cast<uintptr_t>(this)) + "-" + std::to_string(serial++));
        std::filesystem::create_directory(directory);
        path = directory / "proxy.json";
    }
    ~TempFile() { std::error_code e; std::filesystem::remove_all(directory, e); }
};

} // namespace

TEST_CASE("RT proxy index-only cook preserves borders and original attributes") {
    const auto g = grid();
    const auto boundary = borders(g.indices);
    REQUIRE_FALSE(boundary.empty());
    for (auto level : {RtProxyLevel::R10, RtProxyLevel::R25, RtProxyLevel::R50, RtProxyLevel::Full}) {
        const auto result = rtCookProxyMesh(g.vertices, g.indices, level);
        CHECK_FALSE(result.indices.empty());
        CHECK(result.indices.size() % 3 == 0);
        CHECK(result.indices.size() <= g.indices.size());
        CHECK(std::all_of(result.indices.begin(), result.indices.end(), [&](u32 v) { return v < g.vertices.size(); }));
        CHECK(borders(result.indices) == boundary);
        CHECK(result.relativeError >= 0);
        CHECK(result.objectSpaceError >= 0);
        if (level == RtProxyLevel::Full) {
            CHECK(result.indices == g.indices);
            CHECK(result.relativeError == 0);
        } else {
            CHECK(result.indices.size() < g.indices.size());
        }
    }
}

TEST_CASE("RT proxy invalid geometry cannot reach meshoptimizer assertions") {
    auto g = grid(1);
    CHECK_THROWS_AS(rtCookProxyMesh({}, g.indices, RtProxyLevel::R10), std::invalid_argument);
    CHECK_THROWS_AS(rtCookProxyMesh(g.vertices, {}, RtProxyLevel::R10), std::invalid_argument);
    CHECK_THROWS_AS(rtCookProxyMesh(g.vertices, g.indices, RtProxyLevel(100)), std::invalid_argument);
    const auto full = rtCookProxyMesh(g.vertices, g.indices, RtProxyLevel::R10);
    CHECK(full.indices == g.indices);
    g.indices.push_back(0);
    CHECK_THROWS_AS(rtCookProxyMesh(g.vertices, g.indices, RtProxyLevel::Full), std::invalid_argument);
    g.indices.pop_back();
    g.indices[0] = 0xffffffff;
    CHECK_THROWS_AS(rtCookProxyMesh(g.vertices, g.indices, RtProxyLevel::R10), std::invalid_argument);
    g.indices[0] = 0;
    g.vertices[0].px = std::numeric_limits<float>::infinity();
    CHECK_THROWS_AS(rtCookProxyMesh(g.vertices, g.indices, RtProxyLevel::R10), std::invalid_argument);
}

TEST_CASE("RT proxy S4 promotion concentrates on blame and stops at full") {
    std::array levels{RtProxyLevel::R10, RtProxyLevel::R10, RtProxyLevel::Full};
    std::array<double, 3> blame{99, 1, 10000};
    CHECK(rtProxyPromote(levels, blame));
    CHECK(levels[0] == RtProxyLevel::R25);
    CHECK(levels[1] == RtProxyLevel::R10);
    CHECK(levels[2] == RtProxyLevel::Full);
    CHECK(rtProxyPromote(levels, blame));
    CHECK(rtProxyPromote(levels, blame));
    CHECK(levels[0] == RtProxyLevel::Full);
    CHECK(rtProxyPromote(levels, blame));
    CHECK(levels[1] == RtProxyLevel::R25); // full meshes no longer dilute eligible blame
    CHECK(rtProxyPromote(levels, blame));
    CHECK(rtProxyPromote(levels, blame));
    CHECK_FALSE(rtProxyPromote(levels, blame));
    CHECK_THROWS_AS(rtProxyPromote(levels, std::span(blame).first(2)), std::invalid_argument);
    blame[0] = -1;
    CHECK_THROWS_AS(rtProxyPromote(levels, blame), std::invalid_argument);

    // Each offender is <2%; the highest still advances deterministically.
    std::vector<RtProxyLevel> many(100, RtProxyLevel::R10);
    std::vector<double> equal(100, 1);
    CHECK(rtProxyPromote(many, equal));
    CHECK(std::count(many.begin(), many.end(), RtProxyLevel::R25) == 1);
    CHECK(many[0] == RtProxyLevel::R25);
    std::fill(equal.begin(), equal.end(), 0);
    CHECK_FALSE(rtProxyPromote(many, equal));
}

TEST_CASE("RT proxy acceptance rejects empty populations and bad measurements") {
    RtProxyMeasurements m{1000, 800, 0.5, 0.2, 1.0, 0.5};
    CHECK(rtProxyMeetsThresholds(m));
    m.shadowPercent = 0.500001;
    CHECK_FALSE(rtProxyMeetsThresholds(m));
    m.shadowPercent = 0;
    m.primaryRays = 0;
    CHECK_FALSE(rtProxyMeetsThresholds(m));
    m.primaryRays = 100;
    m.distance95Cm = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(rtProxyMeetsThresholds(m));
    m.distance95Cm = -1;
    CHECK_FALSE(rtProxyMeetsThresholds(m));
}

TEST_CASE("RT proxy aggregate offsets share vertices but never modify raster data") {
    GpuScene scene;
    addGrid(scene);
    addGrid(scene, 8, 30);
    const auto oldIndices = scene.indices();
    const auto oldVersion = scene.geometryVersion();
    auto manifest = manifestFor(scene);
    auto proxy = rtBuildProxyGeometry(scene, &manifest);
    REQUIRE(proxy.manifestApplied);
    REQUIRE(proxy.meshes.size() == 2);
    CHECK(proxy.proxyTriangles < proxy.fullTriangles);
    CHECK(proxy.meshes[0].indexOffset == 0);
    CHECK(proxy.meshes[1].indexOffset == proxy.meshes[0].indexCount);
    for (u32 i = 0; i < 2; ++i) {
        CHECK(proxy.meshes[i].vertexOffset == scene.meshInfos()[i].vertexOffset);
        CHECK(proxy.meshes[i].meshletCount == 0);
        CHECK(proxy.meshes[i].indexCount == manifest.meshes[i].proxyIndexCount);
    }
    CHECK(scene.indices() == oldIndices);
    CHECK(scene.geometryVersion() == oldVersion);
    const auto full = rtBuildProxyGeometry(scene);
    CHECK_FALSE(full.manifestApplied);
    CHECK(full.indices == scene.indices());
    CHECK(full.proxyTriangles == full.fullTriangles);
}

TEST_CASE("RT proxy manifest validates exact scene and every regenerated index stream") {
    GpuScene scene;
    addGrid(scene);
    const auto good = manifestFor(scene);
    REQUIRE(rtBuildProxyGeometry(scene, &good).manifestApplied);
    GpuScene other;
    addGrid(other, 12, 1);
    CHECK_FALSE(rtBuildProxyGeometry(other, &good).manifestApplied);
    const std::vector<std::function<void(RtProxyManifest&)>> corrupt = {
        [](auto& m) { ++m.meshoptimizerVersion; },
        [](auto& m) { m.meshes.clear(); },
        [](auto& m) { m.meshes[0].proxyIndexCount += 3; },
        [](auto& m) { m.meshes[0].indexFingerprint = rtProxyIndexFingerprint(std::array<u32, 3>{1, 2, 3}); },
        [](auto& m) { m.meshes[0].objectSpaceError += 1; },
        [](auto& m) { m.thresholds.shadowPercent = 100; },
        [](auto& m) { m.measured.primaryBadPercent = 10; },
        [](auto& m) { m.measurementScope = "unmeasured"; },
        [](auto& m) { m.meshes[0].mesh = 3; },
        [](auto& m) { m.meshes[0].relativeError = std::numeric_limits<float>::infinity(); },
        [](auto& m) { m.meshes[0].level = RtProxyLevel::Full; },
    };
    for (size_t i = 0; i < corrupt.size(); ++i) {
        CAPTURE(i);
        auto bad = good;
        corrupt[i](bad);
        const auto result = rtBuildProxyGeometry(scene, &bad);
        CHECK_FALSE(result.manifestApplied);
        CHECK(result.indices == scene.indices());
        CHECK_FALSE(result.diagnostic.empty());
    }
}

TEST_CASE("RT proxy JSON round trip and fail-closed malformed files") {
    GpuScene scene;
    addGrid(scene);
    const auto manifest = manifestFor(scene);
    TempFile file;
    std::string error;
    REQUIRE(rtWriteProxyManifest(file.path.string(), manifest, error));
    RtProxyManifest loaded;
    REQUIRE(rtReadProxyManifest(file.path.string(), loaded, error));
    CHECK(rtBuildProxyGeometry(scene, &loaded).manifestApplied);
    std::ifstream input(file.path);
    auto doc = nlohmann::json::parse(input);
    SUBCASE("round trip") { return; }
    SUBCASE("missing measurement") { doc.erase("measured"); }
    SUBCASE("fractional count") { doc["meshes"][0]["proxy_indices"] = 1.5; }
    SUBCASE("negative count") { doc["meshes"][0]["mesh"] = -1; }
    SUBCASE("wrapped count") { doc["meshes"][0]["mesh"] = uint64_t(1) << 32; }
    SUBCASE("unknown level") { doc["meshes"][0]["level"] = "sloppy"; }
    SUBCASE("unversioned") { doc["schema"] = 0; }
    SUBCASE("empty population") { doc["measured"]["shadow_receivers"] = 0; }
    SUBCASE("null error") { doc["meshes"][0]["relative_error"] = nullptr; }
    SUBCASE("not an array") { doc["meshes"] = nlohmann::json::object(); }
    std::ofstream(file.path) << doc.dump();
    CHECK_FALSE(rtReadProxyManifest(file.path.string(), loaded, error));
    CHECK_FALSE(error.empty());
    CHECK(loaded.meshes.empty());
}

TEST_CASE("RT proxy protects all mesh users with alpha or emission") {
    std::array<GPUMaterial, 4> materials{};
    for (auto& m : materials) m.emissiveTex = INVALID_TEXTURE_INDEX;
    materials[1].alphaCutoff = 0.5f;
    materials[2].emissive[2] = 1;
    materials[3].emissiveTex = 0;
    std::array<GPUInstance, 7> instances{};
    instances[0].meshIndex = 0; instances[0].materialIndex = 0;
    instances[1].meshIndex = 1; instances[1].materialIndex = 1;
    instances[2].meshIndex = 2; instances[2].materialIndex = 2;
    instances[3].meshIndex = 3; instances[3].materialIndex = 3;
    instances[4].meshIndex = 0; instances[4].materialIndex = 4;
    instances[5].meshIndex = 1; instances[5].materialIndex = 0;
    instances[6].meshIndex = 500; instances[6].materialIndex = 1;
    CHECK(rtProxyProtectedMeshes(4, instances, materials) == std::vector<u32>{0, 1, 2, 3});
    GpuScene scene;
    addGrid(scene);
    addGrid(scene);
    auto manifest = manifestFor(scene);
    auto result = rtBuildProxyGeometry(scene, &manifest, std::array<u32, 1>{1});
    REQUIRE(result.manifestApplied);
    CHECK(result.selections[0].level == RtProxyLevel::R10);
    CHECK(result.selections[1].level == RtProxyLevel::Full);
    CHECK(result.meshes[1].indexCount == scene.meshInfos()[1].indexCount);
    CHECK(result.proxyTriangles < result.fullTriangles);
}
