#pragma once

#include "core/types.h"
#include "rendergraph/render_graph.h"

#include <array>
#include <string>

namespace phosphor::pipe {

// ---------------------------------------------------------------------------
// PipelineDesc -- API-agnostic description of a pipeline state (F3.2).
//
// Everything that selects a distinct GPU binary is here: the kind, the
// function names, the function-constant values (F3.3 variants) and the
// colour output state.  The Metal backend turns it into an MTL4 descriptor;
// the portable code hashes it into a PipelineKey (pipeline_key.h).
//
// "Unspecialized" output state (format, blend, write mask) describes a Metal 4
// flexible render pipeline: compiled once, then specialised to the real
// output state by the backend (newRenderPipelineStateBySpecialization).
// ---------------------------------------------------------------------------

enum class PipelineKind : u8 { Render, Compute };

/// Type of a function constant, as declared in MSL.
enum class ConstantType : u8 { Bool, UInt, Int, Float };

struct FunctionConstant {
    u16          index = 0;           // [[function_constant(index)]]
    ConstantType type  = ConstantType::UInt;
    u32          bits  = 0;           // value bits (bool: 0/1, float: bit pattern)
    bool operator==(const FunctionConstant&) const = default;
};

constexpr u32 MAX_FUNCTION_CONSTANTS = 8;
constexpr u32 MAX_COLOR_ATTACHMENTS  = 4;

/// Colour format of an attachment; `Unspecialized` = flexible pipeline.
/// rg::Format::Unknown means "no attachment at this index".
struct ColorOutput {
    rg::Format format         = rg::Format::Unknown;
    bool       unspecialized  = false; // format, blend and write mask left open
    enum class Blend : u8 { Disabled, AlphaOver } blend = Blend::Disabled;
    u8         writeMask      = 0xF;   // RGBA bits
    bool operator==(const ColorOutput&) const = default;
};

struct PipelineDesc {
    PipelineKind kind = PipelineKind::Render;
    std::string  label;                 // debug name; not part of the key
    /// Render: [0] vertex, [1] fragment.  Compute: [0] kernel, [1] empty.
    std::array<std::string, 2> functions;
    /// Specialisation constants; count == 0 means "generic" (every constant
    /// undefined: the shader falls back to its runtime path).
    std::array<FunctionConstant, MAX_FUNCTION_CONSTANTS> constants{};
    u32 constantCount = 0;
    std::array<ColorOutput, MAX_COLOR_ATTACHMENTS> color{};
    u32 colorCount = 0;

    /// Append a constant (asserts on overflow in debug builds).
    PipelineDesc& constant(u16 index, ConstantType type, u32 bits);
    /// Set colour attachment `index` (colorCount grows to cover it).
    PipelineDesc& output(u32 index, rg::Format format, ColorOutput::Blend blend = ColorOutput::Blend::Disabled);

    [[nodiscard]] bool isGeneric() const { return constantCount == 0; }
    /// True if any colour output is unspecialized (a flexible pipeline).
    [[nodiscard]] bool isFlexible() const;
    /// Copy with every colour output unspecialized (the flexible pipeline this
    /// one can be specialised from).
    [[nodiscard]] PipelineDesc flexible() const;
    /// Copy without function constants (the generic variant).
    [[nodiscard]] PipelineDesc generic() const;
};

} // namespace phosphor::pipe
