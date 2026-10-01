#pragma once

#include "core/types.h"
#include "pipeline/pipeline_desc.h"
#include "renderer/gpu_types.h"

#include <span>

namespace phosphor {
struct FrameScene;
}

namespace phosphor::pipe::forward {

// ---------------------------------------------------------------------------
// Forward-pass variants (F3.3, O11).
//
// The axes are declared once in shaders/variants.def; tools/variant_gen turns
// them into `pipeline/forward_variants.generated.h` (C++ table: axes,
// function-constant indices, value ranges, generic values) and
// `pipeline/forward_variants.generated.metal.h` (MSL function constants, each
// with a runtime fallback when undefined, so the GENERIC pipeline -- no
// constants -- keeps today's behaviour).  Both land in
// ${CMAKE_BINARY_DIR}/generated, which is on the include path of phosphor_core
// and of the shader compiler.
//
// Axes (all remove only branches that are dead for the scene, so every
// variant renders the same pixels as the generic pipeline):
//   LIGHT_TYPES  uint 1..7  bitmask of light types present (1 dir, 2 point, 4 spot)
//   EMISSIVE     bool       any material with a non-zero emissive factor
//   DEBUG_MODE   uint 0..2  FrameConstants::debugMode (F1/F2/F3 keys)
// => 7 * 2 * 3 = 42 variants.  Per-material (per-draw) variants are F15.
//
// A reserved constant (index SALT_CONSTANT_INDEX, uint) is declared too:
// --pipeline-salt sets it on specialised variants so the OS shader cache
// cannot serve them (cold-compile measurements, F3.1).
// ---------------------------------------------------------------------------

constexpr u32 LIGHT_DIRECTIONAL_BIT = 1u;
constexpr u32 LIGHT_POINT_BIT       = 2u;
constexpr u32 LIGHT_SPOT_BIT        = 4u;

struct Variant {
    u32  lightTypes = LIGHT_DIRECTIONAL_BIT | LIGHT_POINT_BIT | LIGHT_SPOT_BIT;
    bool emissive   = true;
    u32  debugMode  = 0;
    bool operator==(const Variant&) const = default;
};

/// Number of valid variants (the generated table's size).
[[nodiscard]] u32 variantCount();
/// Dense index in [0, variantCount()) and back (round-trip exact).
[[nodiscard]] u32     variantIndex(const Variant& v);
[[nodiscard]] Variant variantAt(u32 index);

/// Variant needed to draw `scene` (lights and materials of the frame) with
/// `debugMode`.  A scene without lights uses LIGHT_DIRECTIONAL_BIT (the light
/// loop runs zero times either way).
[[nodiscard]] Variant sceneVariant(const FrameScene& scene, u32 debugMode);
/// F5: the same choice from the frame's lights and whether any material of
/// the persistent scene is emissive (SceneStore::hasEmissive).
[[nodiscard]] Variant sceneVariant(std::span<const GPULight> lights, bool emissive, u32 debugMode);

/// Reserved function-constant index of the salt (uint).
[[nodiscard]] u16 saltConstantIndex();

/// Descriptor of the forward pipeline for `v` (constants set) and of the
/// generic one (no constants), drawing into one colour attachment `color`.
/// `salt` != 0 adds the salt constant to the specialised descriptor.
[[nodiscard]] PipelineDesc pipelineDesc(const Variant& v, rg::Format color, u32 salt = 0);
[[nodiscard]] PipelineDesc genericDesc(rg::Format color);

} // namespace phosphor::pipe::forward
