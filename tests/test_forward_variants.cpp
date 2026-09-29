#include <doctest/doctest.h>

#include "pipeline/forward_variants.generated.h"
#include "pipeline/forward_variants.h"
#include "renderer/gpu_types.h"
#include "renderer/scene_extract.h"

#include <set>
#include <string>

using namespace phosphor;
using namespace phosphor::pipe;
using namespace phosphor::pipe::forward;

namespace {

GPULight makeLight(u32 type) {
    GPULight l{};
    l.type = type;
    return l;
}

GPUMaterial makeMaterial(float er, float eg, float eb) {
    GPUMaterial m{};
    m.emissive[0] = er;
    m.emissive[1] = eg;
    m.emissive[2] = eb;
    return m;
}

} // namespace

TEST_CASE("variant table: count and dense round trip") {
    CHECK(variantCount() == 42);
    std::set<u32> seen;
    for (u32 i = 0; i < variantCount(); ++i) {
        const Variant v = variantAt(i);
        CHECK(v.lightTypes >= 1);
        CHECK(v.lightTypes <= 7);
        CHECK(v.debugMode <= 2);
        CHECK(variantIndex(v) == i);
        seen.insert(i);
    }
    CHECK(seen.size() == 42);
    // And the other way round: every valid value combination maps to a unique index.
    std::set<u32> indices;
    for (u32 lt = 1; lt <= 7; ++lt) {
        for (bool em : {false, true}) {
            for (u32 dm = 0; dm <= 2; ++dm) {
                const Variant v{lt, em, dm};
                const u32 idx = variantIndex(v);
                CHECK(idx < variantCount());
                CHECK(variantAt(idx) == v);
                indices.insert(idx);
            }
        }
    }
    CHECK(indices.size() == 42);
}

TEST_CASE("generated axis table matches variants.def") {
    static_assert(gen::kAxisCount == 3);
    CHECK(std::string(gen::kAxes[gen::AXIS_LIGHT_TYPES].name) == "LIGHT_TYPES");
    CHECK(gen::kAxes[gen::AXIS_LIGHT_TYPES].constantIndex == 0);
    CHECK(gen::kAxes[gen::AXIS_LIGHT_TYPES].type == ConstantType::UInt);
    CHECK(gen::kAxes[gen::AXIS_LIGHT_TYPES].minValue == 1);
    CHECK(gen::kAxes[gen::AXIS_LIGHT_TYPES].maxValue == 7);
    CHECK(gen::kAxes[gen::AXIS_LIGHT_TYPES].genericValue == 7);
    CHECK_FALSE(gen::kAxes[gen::AXIS_LIGHT_TYPES].genericIsRuntime);

    CHECK(std::string(gen::kAxes[gen::AXIS_EMISSIVE].name) == "EMISSIVE");
    CHECK(gen::kAxes[gen::AXIS_EMISSIVE].constantIndex == 1);
    CHECK(gen::kAxes[gen::AXIS_EMISSIVE].type == ConstantType::Bool);
    CHECK(gen::kAxes[gen::AXIS_EMISSIVE].minValue == 0);
    CHECK(gen::kAxes[gen::AXIS_EMISSIVE].maxValue == 1);
    CHECK(gen::kAxes[gen::AXIS_EMISSIVE].genericValue == 1);

    CHECK(std::string(gen::kAxes[gen::AXIS_DEBUG_MODE].name) == "DEBUG_MODE");
    CHECK(gen::kAxes[gen::AXIS_DEBUG_MODE].constantIndex == 2);
    CHECK(gen::kAxes[gen::AXIS_DEBUG_MODE].type == ConstantType::UInt);
    CHECK(gen::kAxes[gen::AXIS_DEBUG_MODE].minValue == 0);
    CHECK(gen::kAxes[gen::AXIS_DEBUG_MODE].maxValue == 2);
    CHECK(gen::kAxes[gen::AXIS_DEBUG_MODE].genericIsRuntime);

    CHECK(gen::kVariantCount == 42);
    CHECK(gen::kSaltConstantIndex == 31);
    CHECK(saltConstantIndex() == 31);
}

TEST_CASE("sceneVariant derives the variant from lights, materials and debug mode") {
    FrameScene scene;
    // Empty scene: directional bit, no emissive.
    Variant v = sceneVariant(scene, 0);
    CHECK(v.lightTypes == LIGHT_DIRECTIONAL_BIT);
    CHECK_FALSE(v.emissive);
    CHECK(v.debugMode == 0);

    scene.lights.push_back(makeLight(LIGHT_POINT));
    scene.lights.push_back(makeLight(LIGHT_POINT));
    v = sceneVariant(scene, 1);
    CHECK(v.lightTypes == LIGHT_POINT_BIT);
    CHECK(v.debugMode == 1);

    scene.lights.push_back(makeLight(LIGHT_SPOT));
    scene.lights.push_back(makeLight(LIGHT_DIRECTIONAL));
    v = sceneVariant(scene, 2);
    CHECK(v.lightTypes == (LIGHT_DIRECTIONAL_BIT | LIGHT_POINT_BIT | LIGHT_SPOT_BIT));

    scene.materials.push_back(makeMaterial(0, 0, 0));
    CHECK_FALSE(sceneVariant(scene, 0).emissive);
    scene.materials.push_back(makeMaterial(0, 0, 0.5f));
    CHECK(sceneVariant(scene, 0).emissive);

    // Debug modes beyond the axis are clamped.
    CHECK(sceneVariant(scene, 99).debugMode == 2);
    CHECK(variantIndex(sceneVariant(scene, 99)) < variantCount());
}

TEST_CASE("pipelineDesc sets every axis constant, genericDesc none") {
    const Variant v{LIGHT_POINT_BIT | LIGHT_SPOT_BIT, false, 2};
    const PipelineDesc d = pipelineDesc(v, rg::Format::BGRA8Srgb);
    CHECK(d.kind == PipelineKind::Render);
    CHECK(d.functions[0] == "forward_vs");
    CHECK(d.functions[1] == "forward_fs");
    CHECK(d.label == "Forward v" + std::to_string(variantIndex(v)));
    REQUIRE(d.constantCount == 3);
    CHECK(d.constants[0] == FunctionConstant{0, ConstantType::UInt, 6});
    CHECK(d.constants[1] == FunctionConstant{1, ConstantType::Bool, 0});
    CHECK(d.constants[2] == FunctionConstant{2, ConstantType::UInt, 2});
    CHECK_FALSE(d.isGeneric());
    REQUIRE(d.colorCount == 1);
    CHECK(d.color[0].format == rg::Format::BGRA8Srgb);
    CHECK(d.color[0].blend == ColorOutput::Blend::Disabled);

    const PipelineDesc s = pipelineDesc(v, rg::Format::BGRA8Srgb, 1234);
    REQUIRE(s.constantCount == 4);
    CHECK(s.constants[3] == FunctionConstant{saltConstantIndex(), ConstantType::UInt, 1234});

    const PipelineDesc g = genericDesc(rg::Format::BGRA8Srgb);
    CHECK(g.constantCount == 0);
    CHECK(g.isGeneric());
    CHECK(g.functions[0] == "forward_vs");
    CHECK(g.functions[1] == "forward_fs");
    REQUIRE(g.colorCount == 1);
    CHECK(g.color[0].format == rg::Format::BGRA8Srgb);
}
