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

// F15.1 (ticket 05): the reserved constant MATERIAL_FEATURES.
#include "pipeline/pipeline_key.h"

TEST_CASE("reserved constant MATERIAL_FEATURES: index 22, not an axis, the 42 variants are unchanged") {
    CHECK(gen::kMaterialFeaturesConstantIndex == 22);
    CHECK(materialFeaturesConstantIndex() == 22);
    CHECK(gen::kAxisCount == 3);
    CHECK(gen::kVariantCount == 42);
    CHECK(MATERIAL_FEATURES_ALL == 0x1FFu);
}

TEST_CASE("withMaterialFeatures: union 0 keeps the descriptor and its key byte for byte") {
    for (u32 i = 0; i < variantCount(); ++i) {
        const PipelineDesc base = pipelineDesc(variantAt(i), rg::Format::BGRA8Srgb);
        const PipelineDesc same = withMaterialFeatures(base, 0);
        CHECK(canonicalString(same) == canonicalString(base));
        CHECK(pipelineKey(same) == pipelineKey(base));
        CHECK(same.constantCount == base.constantCount);
    }
    const PipelineDesc generic = genericDesc(rg::Format::BGRA8Srgb);
    CHECK(canonicalString(withMaterialFeatures(generic, 0)) == canonicalString(generic));
    CHECK(withMaterialFeatures(generic, 0).isGeneric());
}

TEST_CASE("withMaterialFeatures: a union appends constant 22; all-lobes is a distinct reference descriptor") {
    const PipelineDesc base = pipelineDesc(variantAt(5), rg::Format::BGRA8Srgb);
    const PipelineDesc spec = withMaterialFeatures(base, MATERIAL_FEATURE_SPECULAR_IOR);
    REQUIRE(spec.constantCount == base.constantCount + 1);
    CHECK(spec.constants[spec.constantCount - 1].index == 22);
    CHECK(spec.constants[spec.constantCount - 1].type == ConstantType::UInt);
    CHECK(spec.constants[spec.constantCount - 1].bits == 1u);
    // The first constants (the variant axes) are untouched.
    for (u32 i = 0; i < base.constantCount; ++i) CHECK(spec.constants[i] == base.constants[i]);
    CHECK(pipelineKey(spec) != pipelineKey(base));

    const PipelineDesc all = withAllLobes(base);
    CHECK(all.constants[all.constantCount - 1].bits == MATERIAL_FEATURES_ALL);
    CHECK(pipelineKey(all) != pipelineKey(spec));
    CHECK(pipelineKey(all) != pipelineKey(base));
    CHECK(canonicalString(all).find("c22:u=511") != std::string::npos);

    // Re-specialising replaces instead of appending a second constant 22.
    const PipelineDesc again = withMaterialFeatures(spec, MATERIAL_FEATURE_SPECULAR_IOR | MATERIAL_FEATURE_CLEARCOAT);
    CHECK(again.constantCount == spec.constantCount);
    CHECK(again.constants[again.constantCount - 1].bits == 3u);

    // The generic pipeline gets only constant 22: it is no longer "generic".
    const PipelineDesc g = withMaterialFeatures(genericDesc(rg::Format::BGRA8Srgb), MATERIAL_FEATURES_ALL);
    CHECK(g.constantCount == 1);
    CHECK_FALSE(g.isGeneric());
    // Stripping constants (the compile fallback of the cache) yields the base generic pipeline.
    CHECK(canonicalString(spec.generic()) == canonicalString(genericDesc(rg::Format::BGRA8Srgb)));
}

TEST_CASE("exact-all test mode maps base resolve kernels only when on (F15 closure round 4)") {
    PipelineDesc d;
    d.functions = {"visibility_resolve", "", ""};
    CHECK(inExactModuleIfAllScenes(d).functions[0] == "visibility_resolve"); // default: byte-identical
    setExactModuleAllScenes(true);
    CHECK(inExactModuleIfAllScenes(d).functions[0] == "phx_exact::visibility_resolve");
    d.functions[0] = "visibility_clear";
    CHECK(inExactModuleIfAllScenes(d).functions[0] == "visibility_clear");
    setExactModuleAllScenes(false);
    CHECK_FALSE(exactModuleAllScenes());
}
