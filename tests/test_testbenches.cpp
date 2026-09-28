#include <doctest/doctest.h>

#include "null_texture_manager.h"
#include "renderer/gpu_scene.h"
#include "renderer/scene_extract.h"
#include "scene/ecs.h"
#include "testbench/testbench.h"

using namespace phosphor;
using phosphor::test::NullTextureManager;

TEST_CASE("every test bench builds a renderable scene") {
    for (int i = 0; i < testBenchCount(); ++i) {
        const auto type = static_cast<TestBenchType>(i);
        CAPTURE(testBenchName(type));

        ECS ecs;
        GpuScene scene;
        NullTextureManager textures;

        auto bench = createTestBench(type);
        REQUIRE(bench);
        bench->setup(ecs, scene, textures);
        bench->update(1.0f / 60.0f, ecs);

        FrameScene frame;
        extractFrameScene(ecs, scene, frame);

        CHECK(scene.getMeshCount() > 0);
        CHECK(!frame.instances.empty());
        CHECK(!frame.lights.empty());
        u32 drawn = 0;
        for (const DrawBatch& b : frame.batches) {
            CHECK(b.meshIndex < scene.getMeshCount());
            drawn += b.instanceCount;
        }
        CHECK(drawn == frame.instances.size());
        for (const GPUInstance& inst : frame.instances) {
            CHECK(inst.materialIndex < frame.materials.size());
        }
        for (const GPUMaterial& m : frame.materials) {
            for (u32 tex : {m.baseColorTex, m.normalTex, m.metallicRoughnessTex}) {
                CHECK((tex == INVALID_TEXTURE_INDEX || tex < textures.textureCount()));
            }
        }

        bench->teardown(ecs, scene);
    }
}
