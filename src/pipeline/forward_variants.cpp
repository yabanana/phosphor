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

Variant sceneVariant(const FrameScene& scene, u32 debugMode) {
    Variant v;
    v.lightTypes = 0;
    for (const GPULight& light : scene.lights) {
        switch (light.type) {
        case LIGHT_DIRECTIONAL: v.lightTypes |= LIGHT_DIRECTIONAL_BIT; break;
        case LIGHT_SPOT:        v.lightTypes |= LIGHT_SPOT_BIT; break;
        default:                v.lightTypes |= LIGHT_POINT_BIT; break; // the shader's else branch
        }
    }
    if (v.lightTypes == 0) {
        v.lightTypes = LIGHT_DIRECTIONAL_BIT;
    }
    v.emissive = false;
    for (const GPUMaterial& m : scene.materials) {
        if (m.emissive[0] != 0.0f || m.emissive[1] != 0.0f || m.emissive[2] != 0.0f) {
            v.emissive = true;
            break;
        }
    }
    v.debugMode = std::min(debugMode, gen::kAxes[gen::AXIS_DEBUG_MODE].maxValue);
    return v;
}

u16 saltConstantIndex() { return gen::kSaltConstantIndex; }

PipelineDesc genericDesc(rg::Format color) {
    PipelineDesc desc;
    desc.kind = PipelineKind::Render;
    desc.label = "Forward generic";
    desc.functions = {"forward_vs", "forward_fs"};
    desc.color[0] = ColorOutput{color, false, ColorOutput::Blend::Disabled, 0xF};
    desc.colorCount = 1;
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
