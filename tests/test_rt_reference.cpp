#include <doctest/doctest.h>
#include "renderer/rt_reference.h"
#include "renderer/rt_check.h"

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

TEST_CASE("RT barycentric failure reports geometric diagnostics without relaxing acceptance") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    ref.setInstances(std::span(&i, 1));
    const auto r = ray(0.5f, 0);
    auto forged = ref.nearest(r).gpu();
    forged.u += 0.01f;
    const auto check = ref.check(r, forged);
    CHECK_FALSE(check.ok);
    CHECK(check.error.starts_with("barycentric mismatch:"));
    for (const char* field : {"cpu_uv=", "gpu_uv=", "abs_delta_uv=", "bary_tolerance=", "cpu_t=", "gpu_t=",
                              "edge_max_world=", "altitude_min_world=", "incidence_cos=",
                              "spatial_residual_world=", "bary_displacement_world=", "ray_perpendicular_residual_world=",
                              "coordinate_scale_world=", "fp32_ulp_world="})
        CHECK(check.error.find(field) != std::string::npos);
    // This known perturbation moves the reconstructed point by 0.02 world units.
    const auto key = check.error.find("bary_displacement_world=");
    REQUIRE(key != std::string::npos);
    CHECK(std::stod(check.error.substr(key + std::string("bary_displacement_world=").size())) == doctest::Approx(0.02).epsilon(1e-5));
    CHECK_FALSE(ref.check(r, forged, {}, 2e-4, 1e-5).ok);
}

TEST_CASE("RT grazing barycentric discrepancy is judged by spatial backward error") {
    RtReference ref;
    geometry(ref);
    auto i = instance(); ref.setInstances(std::span(&i, 1));
    auto r = ray(-3.5f, -0.25f, 0.001f);
    r.dx = 1; r.dz = -0.00025f;
    const auto exact = ref.nearest(r);
    REQUIRE(exact.hit());
    auto rounded = exact.gpu();
    // A float traversal may move along a nearly coplanar ray by much more
    // than its perpendicular geometric error. Couple t and barycentrics so
    // the reported point still lies on that ray within reconstruction error.
    rounded.t += 4e-5f;
    const double dt = double(rounded.t)-exact.t;
    rounded.u = static_cast<float>(exact.u + 0.5*dt*r.dx);
    REQUIRE(std::abs(double(rounded.u)-exact.u) > 1e-5);
    CHECK(ref.check(r, rounded).ok);
    auto bad = rounded; bad.u += 0.01f;
    CHECK_FALSE(ref.check(r, bad).ok);
    bad = rounded; bad.v += 0.01f; // Lateral displacement cannot hide behind grazing incidence.
    CHECK_FALSE(ref.check(r, bad).ok);
    bad = rounded; bad.t += 4e-5f; // Still inside S0 t tolerance, but no matching bary change.
    const auto inconsistent = ref.check(r, bad);
    CHECK_FALSE(inconsistent.ok);
    CHECK(inconsistent.error.find("reason=spatial-residual") != std::string::npos);
    bad = rounded; bad.t += 0.02f;
    CHECK_FALSE(ref.check(r, bad).ok); // S0 distance tolerance is unchanged.
}

TEST_CASE("RT spatial criterion scales with actual triangle geometry rather than parameter units") {
    RtReference ref;
    geometry(ref);
    auto i = instance();
    i.modelMatrix[0] = i.modelMatrix[5] = i.modelMatrix[10] = 0.001f;
    ref.setInstances(std::span(&i, 1));
    const auto r = ray(0.0005f, -0.00025f);
    const auto exact = ref.nearest(r);
    REQUIRE(exact.hit());
    auto gpu = exact.gpu(); gpu.u += 2e-5f;
    REQUIRE(std::abs(double(gpu.u)-exact.u) > 1e-5);
    CHECK(ref.check(r, gpu).ok); // 4e-8 world units despite a larger parameter error.
    gpu.u += 0.05f;
    CHECK_FALSE(ref.check(r, gpu).ok);
}

TEST_CASE("RT large-coordinate reconstruction budget is capped by the S0 world tolerance") {
    RtReference ref;
    geometry(ref);
    auto i = instance(1e7f, 0, 0);
    ref.setInstances(std::span(&i, 1));
    auto r = ray(1e7f, 0, 2);
    auto gpu = ref.nearest(r).gpu();
    REQUIRE(gpu.hit);
    CHECK(ref.check(r, gpu).ok);
    gpu.u += 0.001f; // 0.002 world units: below gamma16*large coordinates, above S0's 0.0004 cap.
    auto check = ref.check(r, gpu);
    CHECK_FALSE(check.ok);
    CHECK(check.error.find("reason=spatial-residual") != std::string::npos);
    CHECK(check.error.find("distance_world_cap=") != std::string::npos);
    // A small residual does not excuse barycentric weights outside the domain.
    r = ray(1e7f-1, -1, 2);
    gpu = ref.nearest(r).gpu();
    REQUIRE(gpu.hit);
    gpu.u = -2e-5f; // Only 4e-5 world units, but outside containment epsilon.
    check = ref.check(r, gpu);
    CHECK_FALSE(check.ok);
    CHECK(check.error.find("reason=weight-domain") != std::string::npos);
}

TEST_CASE("RT spatial checker rejects meaningful errors even below the former bary delta cutoff") {
    RtReference ref;
    geometry(ref);
    auto i = instance(); ref.setInstances(std::span(&i, 1));
    const auto r = ray(0.5f, 0);
    const auto exact = ref.nearest(r);
    auto gpu = exact.gpu();
    gpu.u += 4e-6f;
    REQUIRE(std::abs(double(gpu.u)-exact.u) < 1e-5);
    const auto check = ref.check(r, gpu);
    CHECK_FALSE(check.ok);
    CHECK(check.error.find("reason=spatial-residual") != std::string::npos);
}

namespace {
// F9 bench8 frame48/ray64 witness (owner decision D3, 2026-11-10): float32
// inputs frozen from the one-triangle Metal control. In exact arithmetic the
// ray hits at u = 1.02e-6, 2.1e-7 m from the edge, 0.056 ULP of the origin.
constexpr float witnessVertices[9] = {-.175000012f, -.150000006f, .303108871f, -.25000003f, 2.6226834e-8f, .433012694f,
                                      -.349999994f, -.150000006f, -3.05979704e-8f};
constexpr float witnessTransform[12] = {-.325449258f, .0220670607f, -1.0048188f, 1.00503695f, -.000158555209f, -.325523376f,
                                        -.00695022568f, -1.05620921f, -.0209445711f, 5.08867741f, 15.563817f, 44.1563911f};
constexpr float witnessOrigin[3] = {0, 12.9266081f, 58.169735f};
constexpr float witnessDirection[3] = {.341815978f, .1762207f, -.923097014f};

struct Witness {
    RtReference ref;
    std::vector<GPUInstance> instances;
    explicit Witness(bool secondBehind = false) {
        std::vector<GPUVertex> v(3);
        for (int k = 0; k < 3; ++k) { v[k].px = witnessVertices[3*k]; v[k].py = witnessVertices[3*k+1]; v[k].pz = witnessVertices[3*k+2]; }
        const auto quad = square();
        v.insert(v.end(), quad.begin(), quad.end());
        const std::array<u32, 9> indices{0, 1, 2, 0, 1, 2, 0, 2, 3};
        std::array<GPURtMesh, 2> meshes{};
        meshes[0].indexCount = 3;
        meshes[1].vertexOffset = 3; meshes[1].indexOffset = 3; meshes[1].indexCount = 6;
        ref.setGeometry(v, indices, meshes);
        GPUMaterial material{};
        material.flags = MATERIAL_FLAG_DOUBLE_SIDED; // the Metal control disables culling
        ref.setMaterials(std::span(&material, 1));
        instances.push_back(place(0));
        // Optional large square perpendicular to z, 5 m further along the ray and
        // crossed well inside: the exact result without the edge triangle is then
        // a hit, so a GPU miss must still fail.
        if (secondBehind) {
            GPUInstance back = instance();
            back.meshIndex = 1;
            const float t = 14.764225f + 5;
            back.modelMatrix[0] = back.modelMatrix[5] = 50;
            for (int row = 0; row < 3; ++row) back.modelMatrix[12+row] = witnessOrigin[row] + t*witnessDirection[row];
            instances.push_back(back);
        }
        ref.setInstances(instances);
    }
    static GPUInstance place(float along) {
        GPUInstance i = instance();
        for (int column = 0; column < 3; ++column)
            for (int row = 0; row < 3; ++row) i.modelMatrix[4*column+row] = witnessTransform[3*column+row];
        for (int row = 0; row < 3; ++row) i.modelMatrix[12+row] = witnessTransform[9+row] + along*witnessDirection[row];
        return i;
    }
    static GPURtRay ray(float inward = 0) {
        GPURtRay r{witnessOrigin[0], witnessOrigin[1], witnessOrigin[2], .0579443984f, witnessDirection[0], witnessDirection[1],
                   witnessDirection[2], 1e30f, RT_MASK_INDIRECT, RT_PROBE_DIFFUSE, 0, 0};
        // Move the origin toward vertex B in world space, as the Metal positive control does.
        float* o = &r.ox;
        for (int row = 0; row < 3; ++row)
            for (int k = 0; k < 3; ++k)
                o[row] += inward * witnessTransform[3*k+row] * (witnessVertices[3+k] - witnessVertices[k]);
        return r;
    }
    static GPURtHit miss() { return {-1, 0, 0, ~0u, ~0u, 0, 0, 0}; }
};
}

TEST_CASE("D3 edge rule: a GPU miss on the frozen F9 witness matches the exact result without the edge triangle") {
    Witness w;
    const auto r = Witness::ray();
    const auto h = w.ref.nearest(r);
    REQUIRE(h.hit());
    CHECK(h.t == doctest::Approx(14.764225).epsilon(1e-6));
    CHECK(std::min({h.u, h.v, 1 - h.u - h.v}) < 2e-6); // the frozen ray really is at the edge
    const auto excluded = w.ref.check(r, Witness::miss());
    CHECK(excluded.ok);
    CHECK(excluded.edgeExcluded);
    // Reporting the edge hit itself stays an ordinary pass.
    const auto direct = w.ref.check(r, h.gpu());
    CHECK(direct.ok);
    CHECK_FALSE(direct.edgeExcluded);
}

TEST_CASE("D3 edge rule never accepts an interior miss") {
    Witness w;
    const auto r = Witness::ray(0.005f); // the Metal control's interior positive case, about 1.1e-3 m inside
    REQUIRE(w.ref.nearest(r).hit());
    const auto check = w.ref.check(r, Witness::miss());
    CHECK_FALSE(check.ok);
    CHECK_FALSE(check.edgeExcluded);
    CHECK(check.error == "hit/miss mismatch");
}

TEST_CASE("D3 edge rule compares strictly with the result without that one triangle") {
    Witness w(true);
    const auto r = Witness::ray();
    const auto front = w.ref.nearest(r);
    REQUIRE(front.hit());
    // Without the edge triangle the exact result is the copy behind it: a miss is wrong.
    CHECK_FALSE(w.ref.check(r, Witness::miss()).ok);
    // Skipping the edge triangle and reporting the copy behind it is the admitted outcome.
    const auto back = w.ref.nearest(r, [&](const GPURtRay&, const RtReferenceHit& hit) { return hit.slot != front.slot; });
    REQUIRE(back.hit());
    CHECK(back.t == doctest::Approx(front.t + 5).epsilon(1e-6));
    const auto skipped = w.ref.check(r, back.gpu());
    CHECK(skipped.ok);
    CHECK(skipped.edgeExcluded);
    // The same farther hit is still refused when the ray is well inside the front triangle.
    const auto inside = Witness::ray(0.005f);
    const auto insideBack = w.ref.nearest(inside, [&](const GPURtRay&, const RtReferenceHit& hit) { return hit.slot != front.slot; });
    REQUIRE(insideBack.hit());
    CHECK_FALSE(w.ref.check(inside, insideBack.gpu()).ok);
}

TEST_CASE("D3 edge rule is counted and capped per frame by the checker") {
    std::vector<GPUVertex> v(3);
    for (int k = 0; k < 3; ++k) { v[k].px = witnessVertices[3*k]; v[k].py = witnessVertices[3*k+1]; v[k].pz = witnessVertices[3*k+2]; }
    const std::array<u32, 3> indices{0, 1, 2};
    GPURtMesh mesh{};
    mesh.indexCount = 3;
    RtChecker checker;
    checker.setGeometry(v, indices, std::span(&mesh, 1));
    GPUMaterial material{};
    material.flags = MATERIAL_FLAG_DOUBLE_SIDED;
    material.baseColorTex = INVALID_TEXTURE_INDEX;
    const std::vector<GPUInstance> instances{Witness::place(0)};
    const std::vector<RtCpuTexture> textures;
    const std::array<GPURtRay, 2> rays{Witness::ray(), Witness::ray(0.005f)};
    Witness w;
    const std::array<GPURtHit, 2> oneEdge{Witness::miss(), w.ref.nearest(rays[1]).gpu()};
    const auto ok = checker.check(rays, oneEdge, instances, std::span(&material, 1), textures);
    CHECK(ok.ok());
    CHECK(ok.edgeExclusions == 1);
    CHECK(ok.checked == 2);
    const std::array<GPURtRay, 2> twoEdge{Witness::ray(), Witness::ray()};
    const std::array<GPURtHit, 2> twoMiss{Witness::miss(), Witness::miss()};
    const auto capped = checker.check(twoEdge, twoMiss, instances, std::span(&material, 1), textures);
    CHECK_FALSE(capped.ok());
    CHECK(capped.edgeExclusions == 2);
    CHECK(capped.firstError.find("edge-excluded") != std::string::npos);
}
