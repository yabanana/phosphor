#pragma once

#include "renderer/gpu_types.h"
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace phosphor {
// Capture linear texels before device-only upload or through a completed same-
// frame readback. RGB of sRGB textures MUST already be decoded, alpha is linear.
// No texture may be silently replaced with white/default during reference export.
struct ReferenceTexture {
    u32 index = 0, width = 0, height = 0;
    std::vector<float> rgba;
};
struct ReferenceCamera {
    float position[3]{}, direction[3]{0,0,-1}, up[3]{0,1,0};
    float fovYRadians = 1.04719755f, nearPlane = 0.1f, farPlane = 1000.f;
    float jitterPixels[2]{}; // rendered projection jitter, +X right/+Y down
    u32 width = 1920, height = 1080;
};
// Mirrors F11 sampling semantics without making portable export depend on a
// still-evolving shader layout. Root converts GPUSampledLight field-for-field.
struct ReferenceAreaLight {
    u32 id = 0, generation = 0, type = 3, flags = 0;
    float position[3]{}, range = 0;
    float axisU[3]{}, radius = 0;
    float axisV[3]{}, innerCone = 0;
    float emission[3]{}, outerCone = 0;
    u32 materialIndex = ~0u;
    float uv0[2]{}, uv1[2]{}, uv2[2]{};
};
struct OfflineReferenceScene {
    std::span<const GPUVertex> vertices;
    std::span<const u32> indices;
    std::span<const GPUMeshInfo> meshes;
    // Explicit completed same-frame WORLD matrices; never local/base motions.
    std::span<const GPUInstance> worldInstances;
    std::span<const GPUMaterial> materials;
    std::span<const GPULight> lights;
    std::span<const ReferenceAreaLight> sampledLights;
    std::span<const ReferenceTexture> textures;
    ReferenceCamera camera;
    float skyRadiance[3]{};
    u32 frame = 0;
    u64 geometryRevision = 0, materialRevision = 0, lightRevision = 0;
};
struct ReferenceExportResult { bool ok = false; std::string error; u32 meshes = 0, instances = 0, textures = 0; };
[[nodiscard]] ReferenceExportResult validateReferenceScene(const OfflineReferenceScene& scene);
// Refuses an existing destination; exports into a temporary sibling and renames
// only after all files succeed. PLY geometry is FULL raster geometry, never proxy.
[[nodiscard]] ReferenceExportResult exportOfflineReference(const OfflineReferenceScene& scene,
                                                            const std::filesystem::path& destination);
// PFM RGB, 32-bit LINEAR, bottom-to-top rows. No exposure/gamma/tonemapping.
[[nodiscard]] bool writeLinearPfm(const std::filesystem::path& path,u32 width,u32 height,
                                  std::span<const float> rgb,std::string& error);
} // namespace phosphor
