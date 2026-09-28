#include <doctest/doctest.h>

#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "scene/components.h"
#include "scene/ecs.h"
#include "scene/procedural.h"

using namespace phosphor;

namespace {

EntityID addInstance(ECS& ecs, MeshHandle mesh, u32 materialIndex, bool withMaterial, bool visible = true) {
    const EntityID e = ecs.createEntity();
    TransformComponent xf{};
    xf.updateMatrix();
    ecs.addComponent(e, std::move(xf));
    MeshInstanceComponent mi{};
    mi.meshHandle = mesh;
    mi.materialIndex = materialIndex;
    mi.setVisible(visible);
    ecs.addComponent(e, std::move(mi));
    if (withMaterial) {
        MaterialComponent mc{};
        mc.roughnessFactor = 0.25f + 0.01f * static_cast<float>(e);
        ecs.addComponent(e, std::move(mc));
    }
    return e;
}

} // namespace

TEST_CASE("per-entity materials are appended after the library") {
    GpuScene scene;
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    const MeshHandle mesh = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
    GPUMaterial lib{};
    lib.roughness = 0.9f;
    scene.addMaterial(lib); // library index 0

    ECS ecs;
    addInstance(ecs, mesh, 0, /*withMaterial*/ false);
    addInstance(ecs, mesh, 0, /*withMaterial*/ true);

    FrameScene frame;
    extractFrameScene(ecs, scene, frame);

    REQUIRE(frame.instances.size() == 2);
    REQUIRE(frame.materials.size() == 2);
    CHECK(frame.materials[0].roughness == doctest::Approx(0.9f));
    CHECK(frame.instances[0].materialIndex == 0);
    CHECK(frame.instances[1].materialIndex == 1);
}

TEST_CASE("invisible instances, unknown meshes and bad material indices are handled") {
    GpuScene scene;
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    const MeshHandle mesh = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);

    ECS ecs;
    addInstance(ecs, mesh, 0, false, /*visible*/ false);
    addInstance(ecs, mesh + 7, 0, true);  // unknown mesh
    addInstance(ecs, mesh, 42, false);    // no library material 42

    FrameScene frame;
    extractFrameScene(ecs, scene, frame);

    REQUIRE(frame.instances.size() == 1);
    REQUIRE(!frame.materials.empty());
    CHECK(frame.instances[0].materialIndex < frame.materials.size());
}

TEST_CASE("instances are sorted by mesh and grouped into batches") {
    GpuScene scene;
    const MeshData cube = ProceduralMeshes::generateCube(1.0f);
    const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1);
    const MeshHandle a = scene.uploadMesh(cube.positions, cube.normals, cube.tangents, cube.uvs, cube.indices);
    const MeshHandle b = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);

    ECS ecs;
    addInstance(ecs, b, 0, true);
    addInstance(ecs, a, 0, true);
    addInstance(ecs, b, 0, true);
    addInstance(ecs, a, 0, true);
    addInstance(ecs, b, 0, true);

    FrameScene frame;
    extractFrameScene(ecs, scene, frame);

    REQUIRE(frame.batches.size() == 2);
    CHECK(frame.batches[0].meshIndex == a);
    CHECK(frame.batches[0].firstInstance == 0);
    CHECK(frame.batches[0].instanceCount == 2);
    CHECK(frame.batches[1].meshIndex == b);
    CHECK(frame.batches[1].firstInstance == 2);
    CHECK(frame.batches[1].instanceCount == 3);
    for (const DrawBatch& batch : frame.batches) {
        for (u32 i = batch.firstInstance; i < batch.firstInstance + batch.instanceCount; ++i) {
            CHECK(frame.instances[i].meshIndex == batch.meshIndex);
        }
    }
}

TEST_CASE("lights take position and forward direction from their transform") {
    GpuScene scene;
    ECS ecs;
    const EntityID e = ecs.createEntity();
    TransformComponent xf{};
    xf.position = glm::vec3(1.0f, 2.0f, 3.0f);
    xf.updateMatrix();
    ecs.addComponent(e, std::move(xf));
    LightComponent lc{};
    lc.type = LightType::Point;
    lc.intensity = 5.0f;
    ecs.addComponent(e, std::move(lc));

    FrameScene frame;
    extractFrameScene(ecs, scene, frame);

    REQUIRE(frame.lights.size() == 1);
    CHECK(frame.lights[0].type == LIGHT_POINT);
    CHECK(frame.lights[0].position[1] == doctest::Approx(2.0f));
    CHECK(frame.lights[0].direction[2] == doctest::Approx(-1.0f));
    CHECK(frame.lights[0].intensity == doctest::Approx(5.0f));
}

TEST_CASE("instances with a negative-determinant transform are flagged as mirrored") {
    GpuScene scene;
    const MeshData plane = ProceduralMeshes::generatePlane(1.0f, 1.0f, 1, 1);
    const MeshHandle mesh = scene.uploadMesh(plane.positions, plane.normals, plane.tangents, plane.uvs, plane.indices);

    ECS ecs;
    addInstance(ecs, mesh, 0, false);
    const EntityID mirrored = addInstance(ecs, mesh, 0, false);
    auto& xf = ecs.getComponent<TransformComponent>(mirrored);
    xf.scale = glm::vec3(1.0f, -1.0f, 1.0f); // e.g. a ceiling made from a floor plane
    xf.updateMatrix();

    FrameScene fs;
    extractFrameScene(ecs, scene, fs);
    REQUIRE(fs.instances.size() == 2);
    u32 mirroredCount = 0;
    for (const GPUInstance& gi : fs.instances) {
        CHECK((gi.flags & 1u) != 0); // component flags (visible) are preserved
        if (gi.flags & INSTANCE_FLAG_MIRRORED) ++mirroredCount;
    }
    CHECK(mirroredCount == 1);
}
