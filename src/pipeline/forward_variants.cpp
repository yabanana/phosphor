#include "pipeline/forward_variants.h"

#include "pipeline/forward_variants.generated.h"
#include "renderer/gpu_types.h"
#include "renderer/scene_extract.h"

#include <algorithm>
#include <string>

namespace phosphor::pipe::forward {

static_assert(LIGHT_DIRECTIONAL_BIT == 1u << LIGHT_DIRECTIONAL && LIGHT_POINT_BIT == 1u << LIGHT_POINT &&
                  LIGHT_SPOT_BIT == 1u << LIGHT_SPOT,
              "light bits must follow the GPULight::type values");
static_assert(gen::kAxes[gen::AXIS_LIGHT_TYPES].maxValue == (LIGHT_DIRECTIONAL_BIT | LIGHT_POINT_BIT | LIGHT_SPOT_BIT),
              "variants.def LIGHT_TYPES range out of sync with the light bits");

namespace {

gen::AxisValues toValues(const Variant& v) {
    gen::AxisValues values{};
    values[gen::AXIS_LIGHT_TYPES] = v.lightTypes;
    values[gen::AXIS_EMISSIVE]    = v.emissive ? 1u : 0u;
    values[gen::AXIS_DEBUG_MODE]  = v.debugMode;
    return values;
}

} // namespace

u32 variantCount() { return gen::kVariantCount; }

u32 variantIndex(const Variant& v) { return gen::indexOf(toValues(v)); }

Variant variantAt(u32 index) {
    const gen::AxisValues values = gen::valuesAt(index % gen::kVariantCount);
    Variant v;
    v.lightTypes = values[gen::AXIS_LIGHT_TYPES];
    v.emissive   = values[gen::AXIS_EMISSIVE] != 0;
    v.debugMode  = values[gen::AXIS_DEBUG_MODE];
    return v;
}

Variant sceneVariant(std::span<const GPULight> lights, bool emissive, u32 debugMode) {
    Variant v;
    v.lightTypes = 0;
    for (const GPULight& light : lights) {
        switch (light.type) {
        case LIGHT_DIRECTIONAL: v.lightTypes |= LIGHT_DIRECTIONAL_BIT; break;
        case LIGHT_SPOT:        v.lightTypes |= LIGHT_SPOT_BIT; break;
        default:                v.lightTypes |= LIGHT_POINT_BIT; break; // the shader's else branch
        }
    }
    if (v.lightTypes == 0) {
        v.lightTypes = LIGHT_DIRECTIONAL_BIT;
    }
    v.emissive  = emissive;
    v.debugMode = std::min(debugMode, gen::kAxes[gen::AXIS_DEBUG_MODE].maxValue);
    return v;
}

Variant sceneVariant(const FrameScene& scene, u32 debugMode) {
    bool emissive = false;
    for (const GPUMaterial& m : scene.materials) {
        if (m.emissive[0] != 0.0f || m.emissive[1] != 0.0f || m.emissive[2] != 0.0f) {
            emissive = true;
            break;
        }
    }
    return sceneVariant(scene.lights, emissive, debugMode);
}

u16 saltConstantIndex() { return gen::kSaltConstantIndex; }

u16 materialFeaturesConstantIndex() { return gen::kMaterialFeaturesConstantIndex; }

std::string exactModuleKernel(const std::string& name) {
    return name == "visibility_resolve" || name == "visibility_lit_resolve" ? "phx_exact::" + name : name;
}

PipelineDesc inExactModule(const PipelineDesc& desc) {
    PipelineDesc out = desc;
    out.functions[0] = exactModuleKernel(desc.functions[0]);
    return out;
}

PipelineDesc withMaterialFeatures(const PipelineDesc& desc, u32 mask) {
    if (mask == 0) return desc;
    PipelineDesc out = inExactModule(desc);
    for (u32 i = 0; i < out.constantCount; ++i) {
        if (out.constants[i].index == gen::kMaterialFeaturesConstantIndex) {
            out.constants[i].bits = mask;
            return out;
        }
    }
    out.constant(gen::kMaterialFeaturesConstantIndex, ConstantType::UInt, mask);
    return out;
}

u16 textureTransformConstantIndex() { return gen::kTextureTransformConstantIndex; }

PipelineDesc withTextureTransform(const PipelineDesc& desc) {
    PipelineDesc out = desc;
    for (u32 i = 0; i < out.constantCount; ++i) {
        if (out.constants[i].index == gen::kTextureTransformConstantIndex) {
            out.constants[i].bits = 1u;
            return out;
        }
    }
    out.constant(gen::kTextureTransformConstantIndex, ConstantType::UInt, 1u);
    return out;
}

u16 vertexColorConstantIndex() { return gen::kVertexColorConstantIndex; }

namespace {
// The entry points that have a vertex-colour twin (shaders/forward_plain.inc, forward_surface.inc, meshlet.metal): their stage interface carries the
// colour only in the twin, so no function-constant interface member is needed (metal-tt cannot lower one in a constant-free descriptor).  A twin is only
// used for a consistent pair: the indexed vertex stages with their fragments, or the visibility mesh stage with the alpha tests.  The meshlet FORWARD
// pipeline (mesh stage + forward_fs) has no twin: it keeps its entry points (and ignores the colour stream: declared).
void applyVertexColorTwins(PipelineDesc& out) {
    const std::string vertexStage = out.functions[0];
    const bool indexed = vertexStage == "forward_vs" || vertexStage == "forward_surface_vs";
    const bool visibility = out.functions[1] == "visibility_mesh";
    static const char* const suffixed[] = {"forward_vs", "forward_fs", "forward_surface_vs", "forward_shadow_receiver_fs", "forward_shadow_receiver_tt_fs",
                                           "forward_surface_fs", "forward_surface_lit_fs", "forward_di_receiver_fs", "forward_di_receiver_guide_fs",
                                           "forward_di_receiver_guide_norog_fs"};
    static const char* const alpha[] = {"visibility_alpha_fs", "visibility_alpha_lit_fs", "visibility_alpha_tt_fs", "visibility_alpha_lit_tt_fs"};
    for (auto& f : out.functions) {
        if (indexed)
            for (const char* name : suffixed)
                if (f == name) f += "_vc";
        if (visibility) {
            if (f == "visibility_mesh") f = "visibility_mesh_vc";
            for (const char* name : alpha)
                if (f == name) f = f.substr(0, f.size() - 3) + "_vc_fs";
        }
    }
}
} // namespace

PipelineDesc withVertexColor(const PipelineDesc& desc) {
    PipelineDesc out = desc;
    applyVertexColorTwins(out);
    for (u32 i = 0; i < out.constantCount; ++i) {
        if (out.constants[i].index == gen::kVertexColorConstantIndex) {
            out.constants[i].bits = 1u;
            return out;
        }
    }
    out.constant(gen::kVertexColorConstantIndex, ConstantType::UInt, 1u);
    return out;
}

PipelineDesc withAllLobes(const PipelineDesc& desc) { return withMaterialFeatures(desc, MATERIAL_FEATURES_ALL); }

PipelineDesc genericDesc(rg::Format color) {
    PipelineDesc desc;
    desc.kind = PipelineKind::Render;
    desc.label = "Forward generic";
    desc.functions = {"forward_vs", "forward_fs"};
    desc.color[0] = ColorOutput{color, false, ColorOutput::Blend::Disabled, 0xF};
    desc.colorCount = 1;
    // F5.3: the same pipeline draws the CPU path and the GPU-built ICB.
    desc.indirectCommandBuffers = true;
    return desc;
}

PipelineDesc pipelineDesc(const Variant& v, rg::Format color, u32 salt) {
    PipelineDesc desc = genericDesc(color);
    desc.label = "Forward v" + std::to_string(variantIndex(v));
    // Filled directly (not through PipelineDesc::constant) to stay independent
    // of the descriptor's helper methods.
    const gen::AxisValues values = toValues(v);
    for (u32 i = 0; i < gen::kAxisCount; ++i) {
        desc.constants[desc.constantCount++] = FunctionConstant{gen::kAxes[i].constantIndex, gen::kAxes[i].type, values[i]};
    }
    if (salt != 0) {
        desc.constants[desc.constantCount++] = FunctionConstant{gen::kSaltConstantIndex, ConstantType::UInt, salt};
    }
    return desc;
}

} // namespace phosphor::pipe::forward
