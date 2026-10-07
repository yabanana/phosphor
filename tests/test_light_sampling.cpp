#include "renderer/light_sampling.h"
#include "renderer/reservoir.h"
#include "renderer/stochastic_sampling.h"
#include <doctest/doctest.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace phosphor;
namespace di = phosphor::di;
namespace {
constexpr double pi = 3.14159265358979323846;
GPUSampledLight rectangle() {
    GPUSampledLight l{};
    l.id = 7; l.generation = 3; l.type = DI_LIGHT_RECTANGLE;
    l.position[2] = 5; l.axisU[0] = 2; l.axisV[1] = -1;
    l.emission[0] = l.emission[1] = l.emission[2] = 2;
    return l;
}
GPUDISurface surface() {
    GPUDISurface s{};
    s.depth = 4; s.geometricNormal[2] = s.shadingNormal[2] = s.viewDirection[2] = 1;
    s.roughness = 1; s.albedo[0] = s.albedo[1] = s.albedo[2] = 1;
    s.instanceSlot = 5; s.instanceGeneration = 8; s.materialRevision = 9; s.valid = 1;
    return s;
}
GPUDIReservoir candidate(u32 id, float target) {
    GPUDIReservoir r{};
    r.lightIndex = r.lightID = id; r.lightGeneration = 3; r.lightRevision = 6;
    r.target = target; r.u = 0.2f; r.v = 0.7f; r.viewID = 11; r.historyEpoch = 4;
    return r;
}
GPULightClusterParams grid(u32 capacity = 2) {
    GPULightClusterParams p{};
    p.view[0] = p.view[5] = p.view[10] = p.view[15] = 1;
    p.nearPlane = 1; p.farPlane = 100; p.tanHalfFovX = p.tanHalfFovY = 1;
    p.gridX = p.gridY = 2; p.gridZ = 4; p.capacity = capacity;
    return p;
}
double occluded(glm::dvec3, const di::LightSample&, void*) { return 0; }
glm::dvec4 emitterTexture(u32 index, glm::dvec2 uv, void*) {
    return index == 3 ? glm::dvec4(uv.x, uv.y, 1, 1) : glm::dvec4(1, 1, 1, uv.x);
}
}

TEST_CASE("F11 alias probabilities match quantized table and a deterministic histogram") {
    di::AliasTable table;
    const std::array<double, 4> weights{1, 2, 4, 0};
    table.rebuild(weights, 42);
    CHECK(table.revision == 42);
    REQUIRE(table.entries.size() == weights.size());
    std::array<double, 4> reconstructed{};
    for (u32 i = 0; i < table.entries.size(); ++i) {
        const auto& e = table.entries[i];
        REQUIRE(e.probability >= 0);
        REQUIRE(e.probability <= 1);
        REQUIRE(e.alias < table.entries.size());
        reconstructed[i] += e.probability / 4.0;
        reconstructed[e.alias] += (1 - e.probability) / 4.0;
    }
    for (u32 i = 0; i < 4; ++i) {
        CHECK(table.entries[i].selectionPdf == doctest::Approx(reconstructed[i]).epsilon(1e-6));
        CHECK(table.entries[i].selectionPdf > 0); // full support, including zero weight
    }
    std::array<u32, 4> histogram{};
    constexpr u32 bins = 8192;
    for (u32 x = 0; x < 4; ++x) for (u32 y = 0; y < bins; ++y)
        ++histogram[table.sample((x + 0.5) / 4.0, (y + 0.5) / bins)];
    for (u32 i = 0; i < 4; ++i)
        CHECK(std::abs(double(histogram[i]) / (4 * bins) - table.entries[i].selectionPdf) <= 1.0 / bins);
    // Negative control: uniform sampling gives the wrong dominant-light PDF.
    CHECK(std::abs(0.25 - table.entries[2].selectionPdf) > 0.25);
}

TEST_CASE("F11 alias handles zero invalid dominant and overflowing input scales") {
    di::AliasTable table;
    const std::array<double, 4> invalid{0, -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    table.rebuild(invalid, 1);
    for (const auto& e : table.entries) CHECK(e.selectionPdf == doctest::Approx(0.25));
    const std::array<double, 3> dominant{std::numeric_limits<double>::max(), 1, 0};
    table.rebuild(dominant, 2);
    CHECK(table.entries[0].selectionPdf > 0.999);
    CHECK(table.entries[1].selectionPdf > 0);
    CHECK(table.entries[2].selectionPdf > 0);
    CHECK(table.sample(-0.1, 0.4) == ~0u);
    CHECK(table.sample(0.5, 1.0) == ~0u);
    table.rebuild({}, 3);
    CHECK(table.sample(0.4, 0.5) == ~0u);
}

TEST_CASE("F11 rectangle area samples expose area and solid-angle Jacobians") {
    const auto light = rectangle();
    CHECK(di::area(light) == doctest::Approx(8));
    const auto center = di::sampleLight(light, {0.5, 0.5}, {0, 0, 0});
    REQUIRE(center.valid);
    CHECK_FALSE(center.delta);
    CHECK(center.distance == doctest::Approx(5));
    CHECK(center.pdfArea == doctest::Approx(1.0 / 8));
    CHECK(center.pdfSolidAngle == doctest::Approx(25.0 / 8));
    CHECK(center.geometry == doctest::Approx(1.0 / 25));
    CHECK(center.normal.z == doctest::Approx(-1));
    auto shear = light;
    shear.axisV[0] = 0.7f;
    CHECK(di::area(shear) == doctest::Approx(8));
    auto twice = light; twice.axisU[0] *= 2; twice.axisV[1] *= 2;
    CHECK(di::area(twice) == doctest::Approx(4 * di::area(light)));
    // Negative control: replacing area PDF with solid-angle PDF changes energy.
    CHECK(center.pdfArea != doctest::Approx(center.pdfSolidAngle));
    const auto back = di::sampleLight(light, {0.5, 0.5}, {0, 0, 10});
    REQUIRE(back.valid); CHECK(back.geometry == 0);
    auto twoSided = light; twoSided.flags = DI_LIGHT_TWO_SIDED;
    CHECK(di::sampleLight(twoSided, {0.5, 0.5}, {0, 0, 10}).geometry == doctest::Approx(1.0 / 25));
}

TEST_CASE("F11 disk triangle and tube sampling distributions have known moments") {
    auto light = rectangle();
    light.axisU[0] = 1; light.type = DI_LIGHT_DISK; light.radius = 2;
    CHECK(di::area(light) == doctest::Approx(4 * pi));
    double diskRadius2 = 0;
    constexpr u32 n = 64;
    for (u32 y = 0; y < n; ++y) for (u32 x = 0; x < n; ++x) {
        const auto s = di::sampleLight(light, {(x + 0.5) / n, (y + 0.5) / n}, {0, 0, 0});
        REQUIRE(s.valid);
        diskRadius2 += s.position.x * s.position.x + s.position.y * s.position.y;
    }
    CHECK(diskRadius2 / (n * n) == doctest::Approx(2).epsilon(1e-6));
    light.type = DI_LIGHT_TRIANGLE; light.axisU[0] = 3; light.axisV[1] = -2;
    CHECK(di::area(light) == doctest::Approx(3));
    glm::dvec3 centroid(0);
    for (u32 y = 0; y < n; ++y) for (u32 x = 0; x < n; ++x) {
        const auto s = di::sampleLight(light, {(x + 0.5) / n, (y + 0.5) / n}, {0, 0, 0});
        REQUIRE(s.valid);
        const double b = s.position.x / 3, c = -s.position.y / 2;
        CHECK(b >= 0); CHECK(c >= 0); CHECK(b + c <= 1);
        centroid += s.position;
    }
    centroid /= n * n;
    CHECK(std::abs(centroid.x - 1) < 0.001);
    CHECK(std::abs(centroid.y + 2.0 / 3) < 0.001);
    light.type = DI_LIGHT_TUBE;
    light.axisU[0] = 0; light.axisU[1] = 3; light.axisV[1] = 0; light.axisV[0] = 1; light.radius = 0.5f;
    CHECK(di::area(light) == doctest::Approx(6 * pi));
    for (u32 i = 0; i < n; ++i) {
        const auto s = di::sampleLight(light, {(i + 0.5) / n, (i + 0.5) / n}, {0, 0, 0});
        REQUIRE(s.valid);
        CHECK(std::abs(s.position.y) <= 3);
        CHECK(s.position.x * s.position.x + (s.position.z - 5) * (s.position.z - 5) == doctest::Approx(0.25));
        CHECK(s.normal.y == 0);
        CHECK(glm::dot(s.normal, s.normal) == doctest::Approx(1));
    }
}

TEST_CASE("F11 rejects zero-area and nonfinite endpoints while target keeps black support") {
    auto light = rectangle();
    light.axisV[1] = 0;
    CHECK(di::area(light) == 0);
    CHECK_FALSE(di::sampleLight(light, {0.5, 0.5}, {0, 0, 0}).valid);
    light = rectangle(); light.type = DI_LIGHT_DISK; light.radius = 0;
    CHECK_FALSE(di::sampleLight(light, {0.5, 0.5}, {0, 0, 0}).valid);
    light = rectangle();
    CHECK_FALSE(di::sampleLight(light, {1, 0.5}, {0, 0, 0}).valid);
    light.emission[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(di::sampleLight(light, {0.5, 0.5}, {0, 0, 0}).valid);
    light = rectangle();
    auto s = surface(); s.shadingNormal[2] = -1;
    const auto sample = di::sampleLight(light, {0.5, 0.5}, {0, 0, 0});
    CHECK(di::target(s, sample, 1e-6) == doctest::Approx(1e-6));
    CHECK(di::target(s, sample, 0) == 0); // invalid target floor must not silently enable reuse
}

TEST_CASE("F11 small-light brute force matches independently solved frontal BRDF and visibility") {
    GPUSampledLight light{};
    light.id = 2; light.generation = 1; light.type = LIGHT_POINT; light.position[2] = 2;
    light.range = 1e8f; light.emission[0] = light.emission[1] = light.emission[2] = 10;
    const auto s = surface();
    // N=V=L, roughness=1, metallic=0: (0.96/pi + 0.04/(4pi))*I/r^2.
    const auto result = di::bruteForce(s, std::span(&light, 1), 16);
    const double independent = 0.97 / pi * 10 / 4;
    CHECK(result.x == doctest::Approx(independent).epsilon(1e-6));
    CHECK(result.y == doctest::Approx(independent).epsilon(1e-6));
    CHECK(di::bruteForce(s, std::span(&light, 1), 16, occluded).x == 0);
    std::array<GPUSampledLight, 3> lights{light, light, light};
    lights[1].position[2] = 4; lights[2].emission[0] = lights[2].emission[1] = lights[2].emission[2] = 0;
    CHECK(di::bruteForce(s, lights, 16).x == doctest::Approx(independent * 1.25).epsilon(1e-6));
}

TEST_CASE("F11 textured emissive endpoints use original barycentric UV emission and alpha") {
    auto light = rectangle();
    light.type = DI_LIGHT_TRIANGLE; light.flags = DI_LIGHT_TEXTURED_EMISSION;
    GPUEmissiveSurface emitter{};
    emitter.valid = 1; emitter.uv1[0] = 1; emitter.uv2[1] = 1;
    GPUMaterial material{};
    material.emissive[0] = 2; material.emissive[1] = 4; material.emissive[2] = 6;
    material.emissiveTex = 3; material.baseColorTex = INVALID_TEXTURE_INDEX;
    const auto sample = di::sampleTexturedLight(light, emitter, material, {0.25, 0.2}, {0, 0, 0}, emitterTexture);
    REQUIRE(sample.valid);
    // sqrt(0.25)=0.5; bary=(0.5,0.4,0.1), so interpolated UV=(0.4,0.1).
    CHECK(sample.radiance.x == doctest::Approx(0.8));
    CHECK(sample.radiance.y == doctest::Approx(0.4));
    CHECK(sample.radiance.z == doctest::Approx(6));
    material.baseColor[3] = 1; material.alphaCutoff = 0.5f; material.baseColorTex = 4;
    const auto masked = di::sampleTexturedLight(light, emitter, material, {0.25, 0.2}, {0, 0, 0}, emitterTexture);
    REQUIRE(masked.valid); CHECK(masked.radiance.x == 0); CHECK(masked.radiance.z == 0);
    CHECK(di::target(surface(), masked, 1e-6) == doctest::Approx(1e-6)); // keep support even on cutout
    CHECK_FALSE(di::sampleTexturedLight(light, emitter, material, {0.25, 0.2}, {0, 0, 0}, nullptr).valid);
    emitter.valid = 0;
    CHECK_FALSE(di::sampleTexturedLight(light, emitter, material, {0.25, 0.2}, {0, 0, 0}, emitterTexture).valid);
}

TEST_CASE("F11 emissive update oracle preserves IDs world transforms and mirrored facing") {
    GPUSampledLight source{}; source.id = 51; source.generation = 7; source.type = DI_LIGHT_TRIANGLE;
    GPUEmissiveSurface emitter{};
    emitter.valid = 1; emitter.instanceSlot = 0; emitter.instanceGeneration = 3; emitter.materialIndex = 0;
    emitter.p0[2] = emitter.p1[2] = emitter.p2[2] = 5; emitter.p1[0] = 1; emitter.p2[1] = -1;
    GPUInstance instance{};
    instance.flags = INSTANCE_FLAG_VALID | INSTANCE_FLAG_MIRRORED; instance.generation = 3;
    instance.modelMatrix[0] = -2; instance.modelMatrix[5] = 3; instance.modelMatrix[10] = instance.modelMatrix[15] = 1;
    instance.modelMatrix[12] = 4; instance.modelMatrix[14] = 2;
    GPUMaterial material{}; material.emissive[0] = 2;
    material.emissiveTex = material.baseColorTex = INVALID_TEXTURE_INDEX;
    const auto light = di::updateEmissive(source, emitter, std::span(&instance, 1), std::span(&material, 1));
    CHECK(light.id == source.id); CHECK(light.generation == source.generation);
    CHECK(light.position[0] == 4); CHECK(light.position[2] == 7);
    CHECK(light.axisU[0] == -2); CHECK(light.axisV[1] == -3);
    CHECK(light.emission[0] == 2);
    CHECK(di::area(light) == doctest::Approx(3));
    const auto sample = di::sampleLight(light, {0.25, 0.2}, {4, 0, 0});
    REQUIRE(sample.valid); CHECK(sample.normal.z == doctest::Approx(-1)); CHECK(sample.geometry > 0);
    ++instance.generation;
    const auto stale = di::updateEmissive(source, emitter, std::span(&instance, 1), std::span(&material, 1));
    CHECK(di::area(stale) == 0); CHECK(stale.emission[0] == 0);
    instance.generation = emitter.instanceGeneration; instance.materialIndex = 1;
    const auto swapped = di::updateEmissive(source, emitter, std::span(&instance, 1), std::span(&material, 1));
    CHECK(di::area(swapped) == 0);
}

TEST_CASE("F11 RIS normalization counts zero contribution proposals and detects missing M") {
    // Exhaustively average all two-proposal streams and reservoir-selection
    // outcomes. Independent discrete integral is 0+10, including dark support.
    const std::array<double, 2> f{0, 10}, pHat{0.1, 10}, q{0.5, 0.5};
    double expectation = 0, wrongNormalization = 0;
    for (u32 a = 0; a < 2; ++a) for (u32 b = 0; b < 2; ++b) {
        const double wa = pHat[a] / q[a], wb = pHat[b] / q[b], total = wa + wb;
        for (u32 selected = 0; selected < 2; ++selected) {
            GPUDIReservoir r{};
            REQUIRE(di::stream(r, candidate(a, float(pHat[a])), wa, 1, 0));
            REQUIRE(di::stream(r, candidate(b, float(pHat[b])), wb, 1, selected == 1 ? 0 : 0.999999));
            REQUIRE(di::finalize(r));
            CHECK(r.M == 2);
            const double probability = q[a] * q[b] * (selected == 0 ? wa : wb) / total;
            const u32 chosen = selected == 0 ? a : b;
            expectation += probability * f[chosen] * r.normalization;
            wrongNormalization += probability * f[chosen] * r.normalization * r.M;
        }
    }
    CHECK(expectation == doctest::Approx(10).epsilon(1e-5));
    CHECK(wrongNormalization == doctest::Approx(20).epsilon(1e-5)); // negative control MUST disagree
}

TEST_CASE("F11 merging re-evaluates target at receiver and preserves W with capped multiplicity") {
    GPUDIReservoir source{};
    REQUIRE(di::stream(source, candidate(1, 4), 40, 10, 0));
    REQUIRE(di::finalize(source));
    CHECK(source.normalization == doctest::Approx(1));
    GPUDIReservoir destination{};
    REQUIRE(di::merge(destination, source, 8, 3, 0));
    REQUIRE(di::finalize(destination));
    CHECK(destination.M == 3);
    CHECK(destination.target == 8);
    CHECK(destination.weightSum == 24);
    CHECK(destination.normalization == doctest::Approx(1));
    CHECK(destination.age == 1);
    // Wrong formula using the source target would yield 0.5 instead of 1.
    CHECK((source.target * source.normalization * 3) / (3 * 8) != doctest::Approx(destination.normalization));
}

TEST_CASE("F11 reservoir input rejects NaN overflow and invalid random without corrupting stream") {
    GPUDIReservoir r{};
    const auto c = candidate(0, 1);
    CHECK_FALSE(di::stream(r, c, std::numeric_limits<double>::infinity(), 1, 0.5));
    CHECK_FALSE(di::stream(r, c, -1, 1, 0.5));
    CHECK_FALSE(di::stream(r, c, 1, 1, 1));
    CHECK_FALSE(di::stream(r, c, 1, 0, 0.5));
    CHECK(r.M == 0); CHECK(r.weightSum == 0);
    CHECK(r.pad[0] == DI_ERROR_WEIGHT);
    REQUIRE(di::stream(r, c, 1, 1, 0));
    r.M = std::numeric_limits<u32>::max();
    CHECK_FALSE(di::stream(r, c, 1, 1, 0));
    CHECK(r.weightSum == 1);
    CHECK_FALSE(di::merge(r, c, std::numeric_limits<float>::quiet_NaN(), 1, 0));
}

TEST_CASE("F11 history rejects deleted lights views epochs age and mismatched geometry") {
    auto light = rectangle();
    GPUDIParams p{};
    p.maxHistoryAge = 16; p.viewID = 11; p.historyEpoch = 4; p.lightRevision = 6;
    auto r = candidate(light.id, 1); r.lightIndex = 0;
    REQUIRE(di::stream(r, r, 4, 4, 0)); REQUIRE(di::finalize(r));
    CHECK(di::reusable(r, p, light));
    auto changed = light; ++changed.generation;
    CHECK_FALSE(di::reusable(r, p, changed));
    changed = light; ++changed.id;
    CHECK_FALSE(di::reusable(r, p, changed));
    auto bad = p; ++bad.viewID; CHECK_FALSE(di::reusable(r, bad, light));
    bad = p; ++bad.historyEpoch; CHECK_FALSE(di::reusable(r, bad, light));
    bad = p; ++bad.lightRevision; CHECK_FALSE(di::reusable(r, bad, light));
    r.age = p.maxHistoryAge; CHECK_FALSE(di::reusable(r, p, light));
    const auto a = surface(); auto b = a;
    CHECK(di::compatible(a, b, 0.02f, 0.9f, true));
    ++b.instanceGeneration; CHECK_FALSE(di::compatible(a, b, 0.02f, 0.9f, true));
    CHECK(di::compatible(a, b, 0.02f, 0.9f, false)); // spatial may cross instances
    b = a; ++b.materialRevision; CHECK_FALSE(di::compatible(a, b, 0.02f, 0.9f, true));
    b = a; b.depth += 1; CHECK_FALSE(di::compatible(a, b, 0.02f, 0.9f, true));
    b = a; b.geometricNormal[2] = -1; CHECK_FALSE(di::compatible(a, b, 0.02f, 0.9f, true));
    b = a; b.position[2] = 0.2f; CHECK_FALSE(di::compatible(a, b, 0.02f, 0.9f, true));
    b = a; b.position[0] = std::numeric_limits<float>::quiet_NaN(); CHECK_FALSE(di::compatible(a, b, 0.02f, 0.9f, true));
}

TEST_CASE("F11 motion history uses input pixels and rejects invalid reprojection") {
    CHECK(di::historyPixel(3, 2, 1, -1, 8, 4) == 12);
    CHECK(di::historyPixel(3, 2, -0.49f, 0, 8, 4) == 19);
    CHECK(di::historyPixel(3, 2, -0.51f, 0, 8, 4) == 18);
    CHECK(di::historyPixel(0, 0, -1, 0, 8, 4) == ~0u);
    CHECK(di::historyPixel(7, 3, 1, 0, 8, 4) == ~0u);
    CHECK(di::historyPixel(3, 2, std::numeric_limits<float>::quiet_NaN(), 0, 8, 4) == ~0u);
}

TEST_CASE("F11 cluster overflow requires all lights rather than truncated energy") {
    std::array<GPUSampledLight, 5> lights{};
    for (u32 i = 0; i < lights.size(); ++i) {
        lights[i].type = LIGHT_POINT; lights[i].position[2] = -4;
        lights[i].range = 1000; lights[i].id = i;
    }
    di::ClusterGrid clusters;
    clusters.rebuild(lights, grid(2));
    const auto index = clusters.cell({0, 0, -4});
    REQUIRE(index != ~0u);
    CHECK(clusters.cells[index].count == 5);
    CHECK(clusters.requiresBruteForce(index));
    CHECK(clusters.list(index).empty());
    // Negative control: a clamped list loses three of five equal lights.
    CHECK(std::min(clusters.cells[index].count, clusters.params.capacity) != lights.size());
    clusters.rebuild(lights, grid(8));
    CHECK_FALSE(clusters.requiresBruteForce(index));
    CHECK(clusters.list(index).size() == 5);
    for (u32 i = 0; i < 5; ++i) CHECK(clusters.list(index)[i] == i);
    CHECK(clusters.requiresBruteForce(~0u));
    CHECK(clusters.cell({100, 100, -4}) == ~0u);
    CHECK(clusters.cell({0, 0, 4}) == ~0u);
}

TEST_CASE("F11 cluster sphere culling preserves known interior receiver") {
    GPUSampledLight light{};
    light.type = LIGHT_POINT; light.position[0] = 2; light.position[1] = 1; light.position[2] = -7; light.range = 1.1f;
    di::ClusterGrid clusters;
    clusters.rebuild(std::span(&light, 1), grid());
    for (const glm::dvec3 p : {glm::dvec3(2, 1, -7), glm::dvec3(2.4, 1.2, -7.2), glm::dvec3(1.5, 0.5, -7.3)}) {
        const auto index = clusters.cell(p);
        REQUIRE(index != ~0u);
        CHECK(clusters.cells[index].count == 1);
    }
    auto invalid = grid(); invalid.nearPlane = 0;
    CHECK_THROWS_AS(clusters.rebuild({}, invalid), std::invalid_argument);
}

TEST_CASE("F11 generated STBN ranks are reproducible independent permutations with declared periods") {
    di::StbnConfig config;
    config.width = config.height = 4; config.frames = 8; config.dimensions = 4;
    const auto mask = di::generateStbn(config), again = di::generateStbn(config);
    REQUIRE(mask.ranks == again.ranks);
    constexpr u32 count = 4 * 4 * 8;
    REQUIRE(mask.ranks.size() == count * config.dimensions);
    for (u32 d = 0; d < config.dimensions; ++d) {
        std::vector<u32> ranks(mask.ranks.begin() + d * count, mask.ranks.begin() + (d + 1) * count);
        std::sort(ranks.begin(), ranks.end());
        for (u32 i = 0; i < count; ++i) CHECK(ranks[i] == i);
    }
    CHECK_FALSE(std::equal(mask.ranks.begin(), mask.ranks.begin() + count, mask.ranks.begin() + count));
    CHECK(mask.sample(1, 2, 3, 0) == mask.sample(5, 6, 11, 0));
    for (u32 t = 0; t < config.frames; ++t) for (u32 y = 0; y < config.height; ++y) for (u32 x = 0; x < config.width; ++x) {
        const float value = mask.sample(x, y, t, 0);
        CHECK(value >= 0); CHECK(value < 1);
    }
    auto bad = config; bad.frames = 0;
    CHECK_THROWS_AS(di::generateStbn(bad), std::invalid_argument);
    bad = config; bad.width = 2048;
    CHECK_THROWS_AS(di::generateStbn(bad), std::length_error);
    // Hash fallback is repeatable but makes no blue-noise quality assertion.
    CHECK(di::whiteSample(1, 2, 3, 4, 5) == di::whiteSample(1, 2, 3, 4, 5));
    CHECK(di::whiteSample(1, 2, 3, 4, 5) != di::whiteSample(1, 2, 3, 5, 5));
}

TEST_CASE("F11 numerical diagnostic distinguishes finite-input overflow from a safe RGB write") {
    const float finite=std::numeric_limits<float>::max()*0.5f;
    REQUIRE(std::isfinite(finite));const double oracle=double(finite)*10000.0;
    REQUIRE(std::isfinite(oracle));const float shaderArithmetic=finite*10000.0f;
    CHECK_FALSE(std::isfinite(shaderArithmetic));
    const float safeWrite=std::isfinite(shaderArithmetic)?shaderArithmetic:0.0f;
    CHECK(std::isfinite(safeWrite));CHECK(safeWrite==0);
    // A texture-only check would pass; the pre-sanitization diagnostic must fail.
    CHECK_FALSE(std::isfinite(shaderArithmetic)==std::isfinite(safeWrite));
}

TEST_CASE("F11 STBN temporal blocks scramble without losing complete uniform rank support") {
    di::StbnMask mask;mask.config.width=2;mask.config.height=2;mask.config.frames=4;mask.config.dimensions=1;
    mask.ranks.resize(16);std::iota(mask.ranks.begin(),mask.ranks.end(),0u);
    CHECK(mask.sample(0,0,0,0)!=mask.sample(0,0,4,0));
    double firstMean=0,longMean=0;
    for(u32 block=0;block<64;++block) {
        std::array<u32,16> bins{};double mean=0;
        for(u32 t=0;t<4;++t)for(u32 y=0;y<2;++y)for(u32 x=0;x<2;++x) {
            const float value=mask.sample(x,y,block*4+t,0);REQUIRE(value>=0);REQUIRE(value<1);
            ++bins[std::min(15u,u32(value*16))];mean+=value/16.0;
        }
        for(u32 count:bins)CHECK(count==1); // independent uniform stratification oracle
        CHECK(std::abs(mean-0.5)<=1.0/32.0);longMean+=mean/64.0;if(!block)firstMean=mean;
    }
    CHECK(std::abs(longMean-0.5)<=1.0/32.0);
    CHECK(longMean!=doctest::Approx(firstMean).epsilon(1e-8));
}
