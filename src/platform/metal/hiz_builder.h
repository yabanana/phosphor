#pragma once

#include "core/types.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/metal_context.h"

#include <array>

namespace phosphor {

class PipelineCache;

// ---------------------------------------------------------------------------
// HiZBuilder -- conservative reverse-Z Hi-Z pyramids of the mesh path (F6.4).
//
// Two persistent R32Float pyramids (GpuMemory, RenderTargets), recreated
// when the drawable size changes: the CURRENT pyramid (phase A's depth, read
// by phase B) and the HISTORY pyramid (read by phase A, rewritten by Hi-Z
// final from the frame's final depth; the render graph orders the write
// after the read in a frame, and the persistent import orders frame n+1's
// read after frame n's write).  The self-check snapshots the history at the
// start of a check frame (encodeReadback).  Level 0 is power-of-two >=
// ceil(drawable / 2); sizes and the covering rule are
// renderer/meshlet_cull_math.h's.
//
// Backends (spike S3, docs/opt-log.md "F6 — Spike"): Compute = hiz_level0 +
// hiz_reduce_simd (SIMD-group shuffles + threadgroup memory, 5 levels per
// dispatch; Apple9); Sampler = hiz_level0 + one hiz_reduce_sampler dispatch
// per level (min-reduction sampler, Apple10 only: refused unless the
// EFFECTIVE family has it).  Both are bit-exact with the CPU reference.
// ---------------------------------------------------------------------------

class HiZBuilder {
public:
    enum class Backend : u8 { Compute, Sampler };

    HiZBuilder(MetalContext& context, PipelineCache& pipelines, Backend backend);
    ~HiZBuilder();
    HiZBuilder(const HiZBuilder&) = delete;
    HiZBuilder& operator=(const HiZBuilder&) = delete;

    /// (Re)create the pyramids for a drawable of width x height (graph
    /// compilation time, never in a measured frame).  Returns true when they
    /// were recreated: their contents are undefined (history invalid).
    bool resize(u32 width, u32 height);

    [[nodiscard]] u32 width0() const { return width0_; }
    [[nodiscard]] u32 height0() const { return height0_; }
    [[nodiscard]] u32 levels() const { return levels_; }
    [[nodiscard]] MTL::Texture* current() const { return current_; }
    [[nodiscard]] MTL::Texture* history() const { return history_; }
    /// Bytes of one pyramid read back level after level (row-major, tight).
    [[nodiscard]] u64 readbackBytes() const;
    /// Copy every level of `pyramid` into `dst` (tight rows, level after level).
    void encodeReadback(MTL4::ComputeCommandEncoder* encoder, MTL::Texture* pyramid, MTL::Buffer* dst) const;
    [[nodiscard]] Backend backend() const { return backend_; }
    [[nodiscard]] const char* backendName() const { return backend_ == Backend::Sampler ? "sampler" : "compute"; }
    [[nodiscard]] bool ready() const;
    /// Bytes of the two pyramids.
    [[nodiscard]] u64 bytes() const { return bytes_; }

    /// Encode the whole pyramid of `depth` into `dst` (one compute encoder;
    /// dispatch barriers between levels).  `table` 0 = Hi-Z A, 1 = Hi-Z final
    /// (separate argument tables per pass).  `corrupt` (self-check negative
    /// control): level-0 texel (0, 0) is forced to 1 (nearest), an unsafe pyramid.
    void encode(MTL4::ComputeCommandEncoder* encoder, MTL::Texture* depth, MTL::Texture* dst, u32 table,
                bool corrupt = false) const;
    /// Dispatches + barriers of one encode() (constant for a size, O8).
    [[nodiscard]] u32 commandCount() const;

private:
    void release();

    MetalContext&  context_;
    PipelineCache& pipelines_;
    Backend        backend_;
    pipe::PipelineHandle kLevel0_ = pipe::INVALID_PIPELINE, kReduce_ = pipe::INVALID_PIPELINE;
    std::array<MTL4::ArgumentTable*, 2> tables_{};
    MTL::Texture* current_ = nullptr;
    MTL::Texture* history_ = nullptr;
    MTL::Buffer*  params_  = nullptr; // GPUHiZParams per dispatch (256-byte strides); entry 0 corrupt copy last
    u32 width_ = 0, height_ = 0, width0_ = 0, height0_ = 0, levels_ = 0;
    u32 dispatches_ = 0; // reduce dispatches after level 0
    u64 bytes_ = 0;
};

} // namespace phosphor
