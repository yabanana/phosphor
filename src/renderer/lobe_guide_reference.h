#pragma once
// ---------------------------------------------------------------------------
// F15.1 (ticket 06): CPU expectation of the per-pixel lobe guide.
//
// The producers (visibility resolve guide kernel, indexed overflow receivers)
// write packLobeGuide(shadeSurface(...).lobes) for every surface.  This module
// rebuilds the same record from the CPU mirrors of the scene store (instance
// slot -> material -> extension record) so a readback of the guide and of the
// DI surfaces can be compared bit for bit (--debug-lobe-guide).  Portable
// (phosphor_core): no Metal.
//
// Scope = LOBE_GUIDE_RESOLVED_FEATURES (KHR_materials_ior/specular, KHR_materials_clearcoat and KHR_materials_sheen).  A
// pixel whose resolved fields depend on a texture (the material references a
// texture for one of them) is `unverified`: its record is not compared.
// ---------------------------------------------------------------------------
#include "core/types.h"
#include "renderer/gpu_types.h"
#include "renderer/lobe_guide_pack.h"

#include <span>
#include <string>
#include <vector>

namespace phosphor {

struct LobeGuideExpectation {
    LobeParams params{};
    /// A resolved field depends on a texture: the record cannot be rebuilt on the CPU.
    bool unverified = false;
    /// Clearcoat without a coat normal texture: the guide's coat normal is the pixel's shading normal when the base
    /// material has no normal map either (compared by angle; with a base normal map only its unit length is checked).
    bool coatNormalIsShading = false;
    bool coatNormalChecked = false; // the coat normal is compared/validated (clearcoat resolved)
};

/// The parameters shadeSurface resolves for `material` in a scene whose feature
/// union is `featureUnion`, from the CPU mirror of the extension buffer.
/// `extensionsValid` is set to false when the material's extension group is out of range.
[[nodiscard]] LobeGuideExpectation expectedLobeParams(const GPUMaterial& material,
                                                      std::span<const GPUMaterialExtRecord> extensions,
                                                      u32 featureUnion, bool* extensionsValid = nullptr);

struct LobeGuideCheckInput {
    u32 width = 0, height = 0;                       // DI surface extent (guide index = y * width + x)
    std::span<const GPUDISurface> surfaces;          // width * height
    std::span<const GPULobeGuide> guides;            // width * height
    std::span<const GPUInstance> instances;
    std::span<const GPUMaterial> materials;
    std::span<const GPUMaterialExtRecord> extensions;
    u32 featureUnion = 0;
};

struct LobeGuideCheckResult {
    u32 pixels = 0;       // valid DI surfaces
    u32 extended = 0;     // compared, expected features != 0
    u32 base = 0;         // compared, expected features == 0 (record must be zero)
    u32 unverified = 0;   // texture-dependent: not compared
    u32 invalid = 0;      // invalid/background pixels (record must be zero)
    u32 mismatches = 0;
    std::vector<std::string> firstMismatches; // at most 4
    [[nodiscard]] bool pass() const { return mismatches == 0; }
};

[[nodiscard]] LobeGuideCheckResult checkLobeGuide(const LobeGuideCheckInput& input);

/// `LOBE-GUIDE check frame F | pixels P extended E base B unverified U | PASS` (or FAIL + mismatches).
[[nodiscard]] std::string formatLobeGuideCheck(u64 frame, const LobeGuideCheckResult& result);

} // namespace phosphor
