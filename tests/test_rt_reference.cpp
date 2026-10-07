#include <doctest/doctest.h>
#include "renderer/rt_reference.h"

#include <array>
#include <cmath>
#include <vector>

using namespace phosphor;

namespace {
std::vector<GPUVertex> square() {
    std::vector<GPUVertex> v(4);
    v[0].px = -1; v[0].py = -1;
    v[1].px = 1; v[1].py = -1; v[1].u = 1;
    v[2].px = 1; v[2].py = 1; v[2].u = 1; v[2].v = 1;
    v[3].px = -1; v[3].py = 1; v[3].v = 1;
    return v;
}
GPUInstance instance(float x = 0, float y = 0, float z = 0) {
    GPUInstance i{};
    i.modelMatrix[0] = i.modelMatrix[5] = i.modelMatrix[10] = i.modelMatrix[15] = 1;
    i.modelMatrix[12] = x; i.modelMatrix[13] = y; i.modelMatrix[14] = z;
    i.flags = INSTANCE_FLAG_VALID | 3;
    i.generation = 7;
    return i;
}
GPURtRay ray(float x = 0, float y = 0, float z = 2) {
    return {x, y, z, 0, 0, 0, -1, 1000, RT_MASK_PRIMARY, RT_PROBE_PRIMARY, 0, 0};
}
void geometry(RtReference& ref) {
    const auto v = square();
    const std::array<u32, 6> indices{0,1,2,0,2,3};
    GPURtMesh mesh{};
    mesh.indexCount = 6;
    ref.setGeometry(v, indices, std::span(&mesh, 1));
    const GPUMaterial material{};
    ref.setMaterials(std::span(&material, 1));
}
}

TEST_CASE("RT reference uses mesh-local indices and global stream offsets") {
    RtReference ref;
    auto vertices = square();
    vertices.insert(vertices.begin(), GPUVertex{});
    const std::array<u32, 7> indices{999,0,1,2,0,2,3};
    GPURtMesh mesh{};
    mesh.vertexOffset = mesh.indexOffset = 1;
    mesh.indexCount = 6;
    ref.setGeometry(vertices, indices, std::span(&mesh, 1));
    const GPUMaterial material{};
    ref.setMaterials(std::span(&material, 1));
    auto i = instance();
    ref.setInstances(std::span(&i, 1));
    const auto h = ref.nearest(ray(0.5f, 0));
    REQUIRE(h.hit());
    CHECK(h.t == doctest::Approx(2));
    CHECK(h.texU == doctest::Approx(0.75));
    CHECK(h.texV == doctest::Approx(0.5));
    CHECK(h.generation == 7);
    CHECK(h.frontFacing);
    CHECK(ref.check(ray(0.5f, 0), h.gpu()).ok);
    CHECK(ref.meshCount() == 1);
    CHECK(ref.triangleCount() == 2);
}

TEST_CASE("RT watertight shared edge and vertices always hit") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    ref.setInstances(std::span(&i, 1));
    for (const float coordinate : {-1.0f, -0.9f, 0.0f, 0.7f, 1.0f}) {
        const auto r = ray(coordinate, coordinate);
        const auto h = ref.nearest(r);
        REQUIRE(h.hit());
        CHECK(h.primitive == 0); // Deterministic CPU tie break.
        auto alternate = h.gpu();
        alternate.primitive = 1;
        alternate.u = (coordinate + 1) / 2;
        alternate.v = 0;
        const auto check = ref.check(r, alternate);
        CHECK(check.ok);
        CHECK(check.edgeTie);
    }
}

TEST_CASE("RT reference preserves t under nonuniform scale and shear") {
    RtReference ref;
    geometry(ref);
    auto i = instance(4, 5, 6);
    i.modelMatrix[0] = 3;
    i.modelMatrix[5] = 0.25f;
    i.modelMatrix[10] = 7;
    i.modelMatrix[4] = 0.5f;
    ref.setInstances(std::span(&i, 1));
    auto r = ray(4, 5, 10);
    r.dz = -2;
    const auto h = ref.nearest(r);
    REQUIRE(h.hit());
    CHECK(h.t == doctest::Approx(2));
    CHECK(h.frontFacing);
    CHECK(ref.check(r, h.gpu()).ok);
}

TEST_CASE("RT primary culling uses object winding but mirrored hits report world winding") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    i.modelMatrix[0] = -1;
    i.flags |= INSTANCE_FLAG_MIRRORED;
    ref.setInstances(std::span(&i, 1));
    REQUIRE(ref.nearest(ray()).hit()); // Object front is visible despite reversed world winding.
    CHECK_FALSE(ref.nearest(ray()).frontFacing);
    CHECK(ref.check(ray(), ref.nearest(ray()).gpu()).ok);
    auto reverse = ray(0,0,-2); reverse.dz = 1;
    CHECK_FALSE(ref.nearest(reverse).hit());
    auto shadow = ray();
    shadow.mask = RT_MASK_SHADOW;
    shadow.type = RT_PROBE_SHADOW;
    auto h = ref.nearest(shadow);
    REQUIRE(h.hit());
    CHECK_FALSE(h.frontFacing);
    CHECK(ref.check(shadow, h.gpu()).ok);
    GPUMaterial material{};
    material.flags = MATERIAL_FLAG_DOUBLE_SIDED;
    ref.setMaterials(std::span(&material, 1));
    CHECK(ref.nearest(reverse).hit());
    CHECK(ref.nearest(reverse).frontFacing);
    h = ref.nearest(ray());
    REQUIRE(h.hit());
    CHECK_FALSE(h.frontFacing);
    auto bad = h.gpu();
    bad.frontFacing = 1;
    CHECK_FALSE(ref.check(ray(), bad).ok);
}

TEST_CASE("RT visibility flags and masks exclude dead instances and preserve shadow-only occluders") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    i.flags = INSTANCE_FLAG_VALID | 2; // Invisible, still casts shadows.
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray()));
    auto shadow = ray();
    shadow.mask = RT_MASK_SHADOW; shadow.type = RT_PROBE_SHADOW;
    CHECK(ref.any(shadow));
    i.flags = INSTANCE_FLAG_VALID | 1;
    ref.setInstances(std::span(&i, 1));
    CHECK(ref.any(ray()));
    CHECK_FALSE(ref.any(shadow));
    i.flags = 3; // Dead slot, even when override requests all masks.
    const u32 mask = RT_MASK_ALL;
    ref.setInstances(std::span(&i, 1), std::span(&mask, 1));
    CHECK_FALSE(ref.any(ray()));
}

TEST_CASE("RT delete reuse checks incarnation and owns same-frame instance snapshot") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    ref.setInstances(std::span(&i, 1));
    const auto previous = ref.nearest(ray()).gpu();
    i.flags = 0;
    CHECK(ref.any(ray())); // The CPU reference copied the previous readback.
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray()));
    CHECK_FALSE(ref.check(ray(), previous).ok);
    i = instance(); i.generation = 8;
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.check(ray(), previous).ok);
    CHECK(ref.check(ray(), ref.nearest(ray()).gpu()).ok);
}

TEST_CASE("RT alpha callback rejects front card and accepts farther opaque surface") {
    RtReference ref;
    geometry(ref);
    const std::array<GPUInstance, 2> instances{instance(0,0,1), instance(0,0,0)};
    ref.setInstances(instances);
    u32 rejected = 0;
    RtReference::Filter alpha = [&](const GPURtRay&, const RtReferenceHit& h) {
        if (h.slot == 0 && h.texU > 0.4 && h.texU < 0.6) { ++rejected; return false; }
        return true;
    };
    const auto r = ray();
    auto h = ref.nearest(r, alpha);
    REQUIRE(h.hit());
    CHECK(h.slot == 1);
    CHECK(h.t == doctest::Approx(2));
    CHECK(rejected > 0);
    CHECK(ref.any(r, alpha));
    CHECK(ref.check(r, h.gpu(), alpha).ok);
    CHECK_FALSE(ref.check(r, ref.nearest(r).gpu(), alpha).ok);
}

TEST_CASE("RT check rejects forged IDs distances barycentrics and missing hits") {
    RtReference ref;
    geometry(ref);
    auto i = instance(); ref.setInstances(std::span(&i,1));
    const auto r = ray(0.75f, -0.5f);
    const auto good = ref.nearest(r).gpu();
    REQUIRE(good.hit);
    auto bad = good; bad.slot = 12;
    CHECK_FALSE(ref.check(r,bad).ok);
    bad = good; bad.primitive = 1; // Other triangle does not contain this point.
    CHECK_FALSE(ref.check(r,bad).ok);
    bad = good; bad.t += 0.2f;
    CHECK_FALSE(ref.check(r,bad).ok);
    bad = good; bad.u += 0.1f;
    CHECK_FALSE(ref.check(r,bad).ok);
    bad = good; bad.hit = 0; bad.t = -1;
    CHECK_FALSE(ref.check(r,bad).ok);
    bad = good; bad.t = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(ref.check(r,bad).ok);
    CHECK(ref.check(ray(10,10), RtReferenceHit{}.gpu()).ok);
}

TEST_CASE("RT reference handles parallel rays origin plane finite ranges and invalid transforms") {
    RtReference ref;
    geometry(ref);
    auto i = instance(); ref.setInstances(std::span(&i,1));
    auto r = ray(); r.dx = 1; r.dz = 0;
    CHECK_FALSE(ref.any(r));
    r = ray(0,0,0); // tmin is exclusive, self hit at zero rejected.
    CHECK_FALSE(ref.any(r));
    r = ray(); r.tmax = 1;
    CHECK_FALSE(ref.any(r));
    r = ray(); r.tmin = 2;
    CHECK_FALSE(ref.any(r));
    r = ray(); r.dx = r.dy = r.dz = 0;
    CHECK_FALSE(ref.any(r));
    i.modelMatrix[0] = 0;
    ref.setInstances(std::span(&i,1));
    CHECK_FALSE(ref.any(ray()));
    i = instance(); i.meshIndex = 99;
    ref.setInstances(std::span(&i,1));
    CHECK_FALSE(ref.any(ray()));
}

TEST_CASE("RT top-level BVH handles 100K instances without duplicating triangles") {
    RtReference ref;
    geometry(ref);
    std::vector<GPUInstance> instances;
    instances.reserve(100000);
    for (u32 i = 0; i < 100000; ++i) instances.push_back(instance(float(i % 1000) * 3, float(i / 1000) * 3, 0));
    ref.setInstances(instances);
    CHECK(ref.instanceCount() == 100000);
    CHECK(ref.triangleCount() == 2);
    for (u32 target : {0u, 1u, 999u, 54321u, 99999u}) {
        const auto r = ray(float(target % 1000) * 3, float(target / 1000) * 3);
        const auto h = ref.nearest(r);
        REQUIRE(h.hit());
        CHECK(h.slot == target);
        CHECK(ref.check(r, h.gpu()).ok);
    }
}

TEST_CASE("RT reference rejects malformed geometry and clears stale snapshots on replacement") {
    RtReference ref;
    geometry(ref);
    auto i = instance(); ref.setInstances(std::span(&i,1));
    auto vertices = square();
    std::array<u32, 3> indices{0,1,99};
    GPURtMesh mesh{}; mesh.indexCount = 3;
    CHECK_THROWS(ref.setGeometry(vertices, indices, std::span(&mesh,1)));
    CHECK(ref.any(ray())); // Strong exception guarantee.
    indices[2] = 2;
    ref.setGeometry(vertices, indices, std::span(&mesh,1));
    CHECK(ref.instanceCount() == 0);
    CHECK_FALSE(ref.any(ray()));
    const std::array<u32,2> invalidMasks{1,1};
    CHECK_THROWS(ref.setInstances(std::span(&i,1), invalidMasks));
}

TEST_CASE("RT shadow validation accepts a farther any-hit but closest rays reject it") {
    RtReference ref;
    geometry(ref);
    const std::array<GPUInstance, 2> instances{instance(0,0,1), instance(0,0,0)};
    ref.setInstances(instances);
    auto r = ray(0.5f, 0);
    r.type = RT_PROBE_SHADOW;
    r.mask = RT_MASK_SHADOW;
    // Compute the farther candidate independently by filtering out the near one.
    const auto far = ref.nearest(r, [](const GPURtRay&, const RtReferenceHit& h) { return h.slot == 1; }).gpu();
    REQUIRE(far.hit);
    CHECK(far.t == doctest::Approx(2));
    CHECK(ref.nearest(r).t == doctest::Approx(1));
    const auto accepted = ref.check(r, far);
    CHECK(accepted.ok);
    CHECK_FALSE(accepted.edgeTie);
    for (u32 type : {RT_PROBE_PRIMARY, RT_PROBE_AO, RT_PROBE_DIFFUSE}) {
        r.type = type;
        CHECK_FALSE(ref.check(r, far).ok);
    }
    r.type = RT_PROBE_SHADOW;
    auto forged = far; forged.t = 1; // Distance must still belong to the named triangle.
    CHECK_FALSE(ref.check(r, forged).ok);
    forged = far; ++forged.generation;
    CHECK_FALSE(ref.check(r, forged).ok);
    forged = far; forged.u += 0.2f;
    CHECK_FALSE(ref.check(r, forged).ok);
    CHECK_FALSE(ref.check(r, far, [](const GPURtRay&, const RtReferenceHit& h) { return h.slot == 0; }).ok);
    r.tmax = 1.5f;
    CHECK_FALSE(ref.check(r, far).ok);
}

TEST_CASE("RT descriptor eligibility masks invalid material degenerate and nonfinite matrices") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    i.materialIndex = 1;
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray()));
    i = instance();
    i.modelMatrix[0] = 1e-21f;
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray()));
    i.modelMatrix[0] = -1e-21f;
    i.flags |= INSTANCE_FLAG_MIRRORED;
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray()));
    i = instance();
    i.modelMatrix[0] = -1;
    i.flags |= INSTANCE_FLAG_MIRRORED;
    ref.setInstances(std::span(&i, 1));
    CHECK(ref.any(ray())); // Negative determinant itself is valid.
    i.modelMatrix[12] = std::numeric_limits<float>::infinity();
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray()));
    i = instance();
    i.modelMatrix[0] = i.modelMatrix[5] = i.modelMatrix[10] = 1e20f;
    ref.setInstances(std::span(&i, 1));
    CHECK_FALSE(ref.any(ray())); // Descriptor float determinant overflows.
}
