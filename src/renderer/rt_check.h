#pragma once

#include "renderer/rt_reference.h"

#include <span>
#include <string>
#include <vector>

namespace phosphor {

struct RtCpuMip {
    u32 width = 0, height = 0;
    std::vector<u8> rgba8;
};
struct RtCpuTexture {
    std::vector<RtCpuMip> mips;
    // True only for an actual GPU mip readback (or the trivial 1x1 texture).
    // Box-generated CPU mips are useful for tests, not a claim of Metal parity.
    bool exactMips = false;
};

// RGBA8 alpha is linear even for sRGB textures. The optional CPU box chain is
// explicitly approximate; MetalTextureManager reads GPU-generated mips back
// for actual cone-LOD diagnostics instead. Invalid dimensions/size throw.
RtCpuTexture rtMakeCpuTexture(std::span<const u8> rgba, u32 width, u32 height, bool boxMips = false);

// Independent CPU normalized repeat + bilinear/trilinear alpha sampling.
// LOD clamps to the complete chain; incomplete/malformed levels or nonfinite
// arguments throw std::invalid_argument. Does not apply half conversion.
double rtSampleAlpha(const RtCpuTexture& texture, double u, double v, double lod);

struct RtCheckResult {
    u64 checked = 0, failures = 0, ambiguous = 0, edgeTies = 0, unsupported = 0, skipped = 0;
    std::string firstError;
    [[nodiscard]] bool ok() const { return checked > 0 && failures == 0 && unsupported == 0; }
};

// Independent CPU alpha/traversal diagnostic. Geometry BVHs are built once;
// instances MUST be read back from the same frame as rays/hits, with the
// corresponding material snapshot. Inputs are
// borrowed only for check(); the reference owns its snapshot internally.
class RtChecker {
public:
    void setGeometry(std::span<const GPUVertex> vertices, std::span<const u32> indices,
                     std::span<const GPURtMesh> meshes);
    RtCheckResult check(std::span<const GPURtRay> rays, std::span<const GPURtHit> hits,
                        std::span<const GPUInstance> instances, std::span<const GPUMaterial> materials,
                        std::span<const RtCpuTexture> textures, std::span<const u32> masks = {});
private:
    RtReference reference_;
    std::vector<GPUVertex> vertices_;
    std::vector<u32> indices_;
    std::vector<GPURtMesh> meshes_;
};

} // namespace phosphor
