#include <doctest/doctest.h>

#include "renderer/rt_check.h"

#include <array>
#include <cmath>
#include <limits>

using namespace phosphor;

namespace {

std::vector<u8> rgba(std::initializer_list<u8> alpha) {
    std::vector<u8> result;
    for (u8 a : alpha) result.insert(result.end(), {255, 255, 255, a});
    return result;
}
GPUInstance instance(u32 material, float z = 0) {
    GPUInstance i{};
    i.modelMatrix[0] = i.modelMatrix[5] = i.modelMatrix[10] = i.modelMatrix[15] = 1;
    i.modelMatrix[14] = z;
    i.flags = INSTANCE_FLAG_VALID | 3;
    i.generation = material + 1;
    i.materialIndex = material;
    return i;
}
GPURtRay ray(float x = -0.5f, float y = -0.5f) {
    return {x, y, 2, 0, 0, 0, -1, 100, RT_MASK_PRIMARY, RT_PROBE_PRIMARY, 0, 0};
}
struct Scene {
    std::vector<GPUVertex> vertices = std::vector<GPUVertex>(4);
    std::array<u32, 6> indices{0,1,2,0,2,3};
    std::array<GPURtMesh, 1> meshes{};
    std::array<GPUInstance, 2> instances{instance(0), instance(1, -1)};
    std::array<GPUMaterial, 2> materials{};
    std::vector<RtCpuTexture> textures;
    RtChecker checker;
    RtReference reference;
    Scene() {
        vertices[0].px = -1; vertices[0].py = -1;
        vertices[1].px = 1; vertices[1].py = -1; vertices[1].u = 1;
        vertices[2].px = 1; vertices[2].py = 1; vertices[2].u = vertices[2].v = 1;
        vertices[3].px = -1; vertices[3].py = 1; vertices[3].v = 1;
        meshes[0].indexCount = 6;
        for (auto& m : materials) { m.baseColor[3] = 1; m.baseColorTex = INVALID_TEXTURE_INDEX; }
        materials[0].alphaCutoff = 0.5;
        materials[0].baseColorTex = 0;
        textures.push_back(rtMakeCpuTexture(rgba({0,255,255,0}), 2, 2, true));
        textures[0].exactMips = true; // These are deliberately supplied fixture mip bytes.
        checker.setGeometry(vertices, indices, meshes);
        reference.setGeometry(vertices, indices, meshes);
        refresh();
    }
    void refresh() { reference.setMaterials(materials); reference.setInstances(instances); }
    GPURtHit expected(GPURtRay r, bool frontAccepted) {
        return reference.nearest(r, [=](const auto&, const auto& h) { return h.material == 1 || frontAccepted; }).gpu();
    }
    RtCheckResult check(const GPURtRay& r, const GPURtHit& h) {
        return checker.check(std::span(&r,1), std::span(&h,1), instances, materials, textures);
    }
};

} // namespace

TEST_CASE("RT alpha sampler repeats negative UVs and interpolates mip levels") {
    const auto texture = rtMakeCpuTexture(rgba({0,255,255,0}), 2, 2, true);
    CHECK_FALSE(texture.exactMips);
    CHECK(rtSampleAlpha(texture, 0.25, 0.25, 0) == 0);
    CHECK(rtSampleAlpha(texture, 0.75, 0.25, 0) == 1);
    CHECK(rtSampleAlpha(texture, -0.25, 0.25, 0) == 1);
    CHECK(rtSampleAlpha(texture, 1.25, 0.25, 0) == 0);
    CHECK(rtSampleAlpha(texture, 0, 0.25, 0) == doctest::Approx(0.5));
    CHECK(rtSampleAlpha(texture, 0.5, 0.5, 0) == doctest::Approx(0.5));
    CHECK(rtSampleAlpha(texture, 0.25, 0.25, 1) == doctest::Approx(128.0/255));
    CHECK(rtSampleAlpha(texture, 0.25, 0.25, 0.5) == doctest::Approx(64.0/255));
    CHECK(rtSampleAlpha(texture, 0.25, 0.25, 99) == doctest::Approx(128.0/255));
    CHECK(rtSampleAlpha(texture, 0.25, 0.25, -99) == 0);
    const auto constant = rtMakeCpuTexture(rgba({137}), 1, 1);
    CHECK(constant.exactMips);
    CHECK(rtSampleAlpha(constant, -120.2, 180.8, 1000) == doctest::Approx(137.0/255));
}

TEST_CASE("RT alpha sampler rejects malformed mip chains and nonfinite inputs") {
    CHECK_THROWS_AS(rtMakeCpuTexture({}, 0, 2), std::invalid_argument);
    CHECK_THROWS_AS(rtMakeCpuTexture(rgba({0}), 2, 2), std::invalid_argument);
    auto texture = rtMakeCpuTexture(rgba({0,255,255,0}), 2, 2);
    CHECK(rtSampleAlpha(texture, 0.25, 0.25, 0) == 0);
    CHECK_THROWS_AS(rtSampleAlpha(texture, 0, 0, 1), std::invalid_argument);
    CHECK_THROWS_AS(rtSampleAlpha(texture, std::numeric_limits<double>::infinity(), 0, 0), std::invalid_argument);
    CHECK_THROWS_AS(rtSampleAlpha(texture, 0, 0, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    texture.mips.push_back({2, 2, rgba({0,0,0,0})});
    CHECK_THROWS_AS(rtSampleAlpha(texture, 0, 0, 1), std::invalid_argument);
    texture.mips[0].rgba8.pop_back();
    CHECK_THROWS_AS(rtSampleAlpha(texture, 0, 0, 0), std::invalid_argument);
}

TEST_CASE("RT checker accepts alpha holes and catches opacity and transform corruption") {
    Scene s;
    for (bool opaque : {false, true}) {
        const auto r = ray(opaque ? 0.5f : -0.5f);
        auto h = s.expected(r, opaque);
        const auto result = s.check(r,h);
        CHECK(result.ok());
        CHECK(result.checked == 1);
        CHECK(result.ambiguous == 0);
        h = s.expected(r, !opaque);
        CHECK_FALSE(s.check(r,h).ok());
    }
    const auto r = ray();
    const auto h = s.expected(r, false);
    s.instances[1].modelMatrix[14] = -2; // Same-frame snapshot disagrees with the recorded hit.
    const auto bad = s.check(r,h);
    CHECK_FALSE(bad.ok());
    CHECK(bad.failures == 1);
    CHECK_FALSE(bad.firstError.empty());
}

TEST_CASE("RT checker ambiguity is explicit and never suppresses a corrupt named hit") {
    Scene s;
    s.textures[0] = rtMakeCpuTexture(rgba({128}), 1, 1);
    const auto r = ray();
    const auto front = s.expected(r, true), back = s.expected(r, false);
    for (const auto& h : {front, back}) {
        const auto result = s.check(r,h);
        CHECK(result.ambiguous == 1);
        CHECK(result.checked == 0);
        CHECK_FALSE(result.ok()); // An entirely ambiguous batch is not evidence.
        CHECK(result.failures == 1);
    }
    auto corrupt = front;
    ++corrupt.generation;
    auto bad = s.check(r,corrupt);
    CHECK(bad.ambiguous == 0);
    CHECK(bad.failures == 1);
    corrupt = front;
    corrupt.u = 0.99;
    bad = s.check(r,corrupt);
    CHECK(bad.ambiguous == 0);
    CHECK(bad.failures == 1);
    const std::array rays{r, ray(10,10)};
    const std::array hits{front, s.expected(rays[1],true)};
    const auto mixed = s.checker.check(rays,hits,s.instances,s.materials,s.textures);
    CHECK(mixed.ok());
    CHECK(mixed.checked == 1);
    CHECK(mixed.ambiguous == 1);
}

TEST_CASE("RT checker constant alpha uses exact threshold without texture ambiguity") {
    Scene s;
    s.materials[0].baseColorTex = INVALID_TEXTURE_INDEX;
    for (float alpha : {0.499f,0.5f}) {
        s.materials[0].baseColor[3] = alpha;
        const auto r = ray();
        const auto h = s.expected(r,alpha >= 0.5f);
        const auto result = s.check(r,h);
        CHECK(result.ok());
        CHECK(result.ambiguous == 0);
    }
}

TEST_CASE("RT primary cone uses real mip chain while shadow alpha always uses LOD zero") {
    Scene s;
    auto r = ray();
    r.coneWidth = 1; // distance2, world/texel1 => LOD1 (2x2 -> 1x1).
    s.materials[0].alphaCutoff = 0.4f;
    s.refresh();
    auto h = s.expected(r,true); // averaged mip alpha128/255 passes, base texel0 would fail
    CHECK(s.check(r,h).ok());
    s.textures[0].exactMips = false;
    const auto approximate = s.check(r,h);
    CHECK_FALSE(approximate.ok());
    CHECK(approximate.unsupported == 1);
    r.type = RT_PROBE_SHADOW;
    r.mask = RT_MASK_SHADOW;
    h = s.expected(r,false);
    CHECK(s.check(r,h).ok());
    h = s.expected(r,true);
    CHECK_FALSE(s.check(r,h).ok());
}

TEST_CASE("RT checker validates nonnearest shadow occluders and reports unsupported textures") {
    Scene s;
    auto r = ray(0.5f);
    r.type = RT_PROBE_SHADOW; r.mask = RT_MASK_SHADOW;
    const auto far = s.expected(r,false); // both planes accept but any-hit may name the farther plane
    CHECK(s.check(r,far).ok());
    s.textures.clear();
    const auto result = s.check(r,far);
    CHECK_FALSE(result.ok());
    CHECK(result.unsupported == 1);
    CHECK(result.failures == 1);
    CHECK_FALSE(result.firstError.empty());
}

TEST_CASE("RT checker rejects empty mismatched and structurally corrupt samples") {
    Scene s;
    const auto r = ray(0.5f);
    const auto h = s.expected(r,true);
    CHECK_FALSE(s.checker.check({}, {}, s.instances, s.materials, s.textures).ok());
    CHECK_FALSE(s.checker.check(std::span(&r,1), {}, s.instances, s.materials, s.textures).ok());
    auto bad = h;
    bad.slot = 100;
    CHECK_FALSE(s.check(r,bad).ok());
    bad = h;
    bad.frontFacing ^= 1;
    CHECK_FALSE(s.check(r,bad).ok());
    auto badRay = r;
    badRay.mask = 0;
    CHECK_FALSE(s.check(badRay,h).ok());
    const std::array<u32,2> masks{0,0};
    CHECK_FALSE(s.checker.check(std::span(&r,1), std::span(&h,1), s.instances, s.materials, s.textures, masks).ok());
}

TEST_CASE("RT checker reports valid edge ties without accepting farther primary or AO hits") {
    Scene s;
    s.materials[0].alphaCutoff = 0;
    s.refresh();
    auto r = ray();
    auto h = s.expected(r,true);
    h.primitive = 1;
    h.u = 0.25f;
    h.v = 0;
    const auto tied = s.check(r,h);
    CHECK(tied.ok());
    CHECK(tied.edgeTies == 1);
    const auto far = s.expected(r,false);
    CHECK_FALSE(s.check(r,far).ok());
    r.type = RT_PROBE_AO;
    r.mask = RT_MASK_INDIRECT;
    CHECK_FALSE(s.check(r,far).ok());
}

TEST_CASE("RT checker without configured geometry cannot report a pass") {
    RtChecker checker;
    const auto r = ray();
    const GPURtHit miss{-1,0,0,~0u,~0u,0,0,0};
    const auto result = checker.check(std::span(&r,1),std::span(&miss,1),{},{},{});
    CHECK_FALSE(result.ok());
    CHECK(result.checked == 0);
    CHECK(result.failures == 1);
}

TEST_CASE("RT checker skips only the exact inactive secondary sentinel") {
    Scene s;
    GPURtRay sentinel{};
    sentinel.type = RT_PROBE_SHADOW;
    sentinel.mask = RT_MASK_SHADOW;
    sentinel.tmax = -1;
    GPURtHit skipped{-2,0,0,~0u,~0u,0,0,0};
    const auto allSkipped = s.check(sentinel,skipped);
    CHECK(allSkipped.skipped == 1);
    CHECK(allSkipped.checked == 0);
    CHECK_FALSE(allSkipped.ok());
    const std::array rays{sentinel,ray(0.5f)};
    const std::array hits{skipped,s.expected(rays[1],true)};
    const auto mixed = s.checker.check(rays,hits,s.instances,s.materials,s.textures);
    CHECK(mixed.ok());
    CHECK(mixed.checked == 1);
    CHECK(mixed.skipped == 1);
    sentinel.pad = 123; // source-pixel metadata is preserved by secondary generation
    CHECK(s.check(sentinel, skipped).skipped == 1);
    auto malformed = sentinel;
    malformed.dx = 1;
    CHECK(s.check(malformed,skipped).skipped == 0);
    CHECK_FALSE(s.check(malformed,skipped).ok());
    malformed = sentinel;
    malformed.type = RT_PROBE_PRIMARY;
    CHECK(s.check(malformed,skipped).skipped == 0);
    CHECK_FALSE(s.check(malformed,skipped).ok());
    skipped.t = -1;
    CHECK(s.check(sentinel,skipped).skipped == 0);
    CHECK_FALSE(s.check(sentinel,skipped).ok());
}
