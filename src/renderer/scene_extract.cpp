#include "renderer/scene_extract.h"
#include "renderer/gpu_scene.h"
#include "scene/components.h"
#include "scene/ecs.h"

#include <algorithm>
#include <cstring>

namespace phosphor {

GPUMaterial toGPUMaterial(const MaterialComponent& mat) {
    GPUMaterial gm{};
    gm.baseColor[0]         = mat.baseColorFactor.r;
    gm.baseColor[1]         = mat.baseColorFactor.g;
    gm.baseColor[2]         = mat.baseColorFactor.b;
    gm.baseColor[3]         = mat.baseColorFactor.a;
    gm.metallic             = mat.metallicFactor;
    gm.roughness            = mat.roughnessFactor;
    gm.normalScale          = mat.normalScale;
    gm.occlusionStrength    = mat.occlusionStrength;
    gm.baseColorTex         = mat.baseColorTexIndex;
    gm.normalTex            = mat.normalTexIndex;
    gm.metallicRoughnessTex = mat.metallicRoughnessTexIndex;
    gm.occlusionTex         = mat.occlusionTexIndex;
    gm.emissiveTex          = mat.emissiveTexIndex;
    gm.emissive[0]          = mat.emissiveFactor.x;
    gm.emissive[1]          = mat.emissiveFactor.y;
    gm.emissive[2]          = mat.emissiveFactor.z;
    gm.alphaCutoff          = mat.alphaCutoff;
    return gm;
}

void extractFrameScene(ECS& ecs, const GpuScene& scene, FrameScene& out) {
    out.instances.clear();
    out.materials.clear();
    out.lights.clear();
    out.batches.clear();

    auto& transforms = ecs.getArray<TransformComponent>();
    auto& meshInsts  = ecs.getArray<MeshInstanceComponent>();
    auto& materials  = ecs.getArray<MaterialComponent>();
    auto& lights     = ecs.getArray<LightComponent>();

    const auto& library = scene.materials();
    out.materials.assign(library.begin(), library.end());

    const u32 meshCount = scene.getMeshCount();
    out.instances.reserve(meshInsts.size());

    for (u32 i = 0; i < meshInsts.size(); ++i) {
        const EntityID entity = meshInsts.entities()[i];
        const auto& inst = meshInsts.data()[i];

        if (!inst.isVisible()) continue;
        if (!transforms.has(entity)) continue;
        if (inst.meshHandle >= meshCount) continue;

        GPUInstance gi{};
        std::memcpy(gi.modelMatrix, &transforms.get(entity).worldMatrix[0][0], sizeof(gi.modelMatrix));
        gi.meshIndex = inst.meshHandle;
        gi.flags     = inst.flags;

        if (materials.has(entity)) {
            gi.materialIndex = static_cast<u32>(out.materials.size());
            out.materials.push_back(toGPUMaterial(materials.get(entity)));
        } else {
            gi.materialIndex = inst.materialIndex;
        }
        out.instances.push_back(gi);
    }

    // Guarantee every instance references a valid material.
    if (out.materials.empty()) {
        out.materials.push_back(toGPUMaterial(MaterialComponent{}));
    }
    const u32 materialCount = static_cast<u32>(out.materials.size());
    for (auto& gi : out.instances) {
        if (gi.materialIndex >= materialCount) gi.materialIndex = 0;
    }

    std::stable_sort(out.instances.begin(), out.instances.end(),
                     [](const GPUInstance& a, const GPUInstance& b) { return a.meshIndex < b.meshIndex; });

    for (u32 i = 0; i < out.instances.size(); ++i) {
        const u32 mesh = out.instances[i].meshIndex;
        if (out.batches.empty() || out.batches.back().meshIndex != mesh) {
            out.batches.push_back({mesh, i, 0});
        }
        ++out.batches.back().instanceCount;
    }

    out.lights.reserve(lights.size());
    for (u32 i = 0; i < lights.size(); ++i) {
        const EntityID entity = lights.entities()[i];
        const auto& lc = lights.data()[i];

        glm::vec3 position{0.0f};
        glm::vec3 direction{0.0f, -1.0f, 0.0f};
        if (transforms.has(entity)) {
            const auto& xform = transforms.get(entity);
            position  = xform.position;
            // Forward is -Z in local space.
            direction = glm::normalize(glm::mat3(xform.worldMatrix) * glm::vec3(0.0f, 0.0f, -1.0f));
        }

        GPULight gl{};
        gl.type           = static_cast<u32>(lc.type);
        gl.position[0]    = position.x;
        gl.position[1]    = position.y;
        gl.position[2]    = position.z;
        gl.direction[0]   = direction.x;
        gl.direction[1]   = direction.y;
        gl.direction[2]   = direction.z;
        gl.color[0]       = lc.color.r;
        gl.color[1]       = lc.color.g;
        gl.color[2]       = lc.color.b;
        gl.intensity      = lc.intensity;
        gl.range          = lc.range;
        gl.innerCone      = lc.innerConeAngle;
        gl.outerCone      = lc.outerConeAngle;
        gl.shadowMapIndex = lc.shadowMapIndex;
        out.lights.push_back(gl);
    }
}

} // namespace phosphor
