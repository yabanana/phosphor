#pragma once
#include <future>

#include "core/types.h"
#include "pipeline/compile_queue.h"
#include "pipeline/pipeline_registry.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace MTLFX {
class TemporalScalerDescriptor;
class TemporalDenoisedScalerDescriptor;
}
namespace MTL4FX {
class TemporalScaler;
class TemporalDenoisedScaler;
}

namespace phosphor {

class MetalContext;
class TemporalWorker;

// ---------------------------------------------------------------------------
// PipelineCache -- Metal 4 pipeline creation off the render thread (F3).
//
// Every pipeline of the engine is requested here with a portable
// pipe::PipelineDesc and used through a stable handle.  The render thread
// never calls the compiler or the archive after startup (O9):
//
//   request(desc)  -> new entry -> job on the compile threads (F3.1:
//                     maximumConcurrentCompilationTaskCount workers, QoS
//                     utility, lower than the render thread's):
//     1. MTL4Archive lookup of the full descriptor (F3.4) -> final object;
//     2. on a miss (F3.5, reason recorded) the full compile.  Meanwhile the
//        renderer draws a new variant with its GENERIC pipeline (same output
//        state, already final, bit-identical to the pre-F3 forward).  Only
//        when no generic pipeline with that output state is ready does the
//        request get a Metal 4 FLEXIBLE fallback: the generic variant
//        compiled once with an unspecialized output state and specialised
//        to the real one (newRenderPipelineStateBySpecialization, F3.2).
//        Measured: that path makes the validation layer warn and changes up
//        to 300k pixels by 1 LSB, so it is the exception, not the rule
//        (--debug-flexible-pipelines forces it).
//   beginFrame()   -> results become visible at frame start only; replaced
//                     objects go through MetalContext::deferRelease, so
//                     frames in flight keep theirs.  Passes look their
//                     pipeline up by handle while encoding, so the cached
//                     render graph never needs recompiling.
//
// Startup: waitReady()/waitAllFinal() block before the first frame.
// Harvest (--harvest-pipelines): the compiler records every descriptor with a
// MTL4PipelineDataSetSerializer, written as a .mtl4-json script for metal-tt
// (the archive is ignored so that every pipeline is really compiled).
// Hot reload (F3.6): reload(library) recompiles every entry against the new
// library as a new generation, committed atomically at a frame start.
// --pipeline-sync (negative control) resolves requests on the render thread.
// ---------------------------------------------------------------------------

class PipelineCache {
public:
    struct Options {
        std::string archivePath;          // empty: no archive
        std::string harvestPath;          // non-empty: record descriptors
        bool        sync           = false;
        bool        interactiveQos = false; // negative control (F3.1)
        bool        fallbackOnly   = false; // debug: never apply final objects
    };

    PipelineCache(MetalContext& context, const Options& options);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    /// Handle of `desc` (render thread).  A new descriptor starts resolving on
    /// the compile threads; its object appears at a later beginFrame().
    /// `loading`: the caller waits for the final object (loading time, like
    /// startup requests): no flexible fallback is specialised for it.
    pipe::PipelineHandle request(const pipe::PipelineDesc& desc, bool loading = false);

    /// Block until the entry has a usable object (fallback or final).  Startup
    /// and loading only.
    void waitReady(pipe::PipelineHandle handle);
    /// Block until every requested entry is usable (fallback or final); throws
    /// if one failed without a fallback.  Startup only.
    void waitAllReady();
    /// Block until every requested entry has its final object (or failed).
    void waitAllFinal();

    [[nodiscard]] MTL::RenderPipelineState*  render(pipe::PipelineHandle h) const;
    [[nodiscard]] MTL::ComputePipelineState* compute(pipe::PipelineHandle h) const;
    [[nodiscard]] bool isFinal(pipe::PipelineHandle h) const { return registry_.isFinal(h); }

    /// Frame start (render thread): apply finished compilations.  Returns the
    /// number of entries whose object changed.
    u32 beginFrame();
    /// Count a draw/dispatch that used a fallback object.
    void noteFallbackUse() { registry_.noteFallbackUse(); }
    /// Mark the end of startup: from now on compiles the render thread waits
    /// for are reported as render-thread compiles (hitches).
    void startupDone() { startupDone_ = true; }

    [[nodiscard]] const pipe::PipelineStats& stats() const { return registry_.stats(); }
    [[nodiscard]] u32  workerCount() const;
    [[nodiscard]] u32  entryCount() const { return registry_.size(); }
    /// Render-thread time spent inside request() so far (ms).
    [[nodiscard]] double requestMs() const { return requestMs_; }
    [[nodiscard]] bool archiveLoaded() const { return archive_ != nullptr; }
    [[nodiscard]] const std::string& archiveStatus() const { return archiveStatus_; }

    // F8: framework compilation shares the utility-QoS workers/compiler.
    // The future owns its result; discarding an obsolete request releases it.
    std::future<std::shared_ptr<MTL4FX::TemporalScaler>>
    requestTemporalScaler(MTLFX::TemporalScalerDescriptor *descriptor);

    // Destroy a scaler on a utility worker: its deallocation waits for MetalFX
    // initialisation running on other threads (up to ~150 ms measured on the
    // render thread). The GPU must already be done with it.
    void retireTemporalScaler(std::shared_ptr<MTL4FX::TemporalScaler> scaler);

#if defined(PHOSPHOR_METALFX_DENOISED_FACTORY) && PHOSPHOR_METALFX_DENOISED_FACTORY && !defined(PHOSPHOR_DISABLE_METALFX_DENOISED) && \
    __has_include(<MetalFX/MTL4FXTemporalDenoisedScaler.hpp>)
#define PHOSPHOR_METALFX_DENOISED_GATEWAY_AVAILABLE 1
    // F13 gateway: no ordinary TemporalScaler casts or F8 cycle workaround.
    std::future<std::shared_ptr<MTL4FX::TemporalDenoisedScaler>>
    requestTemporalDenoisedScaler(MTLFX::TemporalDenoisedScalerDescriptor* descriptor);
    void retireTemporalDenoisedScaler(std::shared_ptr<MTL4FX::TemporalDenoisedScaler> scaler);
#endif

    std::future<std::shared_ptr<TemporalWorker>> requestTemporalWorker(std::shared_ptr<TemporalWorker> worker);

    // --- F3.4 harvest -----------------------------------------------------------
    [[nodiscard]] bool harvesting() const { return serializer_ != nullptr; }
    [[nodiscard]] u32 generation() const { return registry_.generation(); }
    /// Write the recorded descriptors (every compile so far) as a pipelines
    /// script; false on error (logged).
    bool writeHarvest();

    // --- F3.6 hot reload ------------------------------------------------------
    /// Recompile every entry against `library` (retained by the cache) as a
    /// new generation; false if a reload is already pending.
    bool reload(MTL::Library* library);
    [[nodiscard]] bool reloadPending() const { return registry_.reloadPending(); }
    /// Library of the generation being served.
    [[nodiscard]] MTL::Library* library() const { return library_; }

private:
    struct Job; // resolution of one entry for one generation

    void submit(pipe::PipelineHandle handle, const pipe::PipelineDesc& desc, u32 generation,
                MTL::Library* library, bool allowFallback);
    void resolve(pipe::PipelineHandle handle, const pipe::PipelineDesc& desc, u32 generation,
                 MTL::Library* library, bool allowFallback);
    /// Post a completion and wake the render thread if it is waiting.
    void complete(const pipe::Completion& completion);
    /// Flexible pipeline of `generic` for `library` (compiled once, shared).
    MTL::RenderPipelineState* flexibleBase(const pipe::PipelineDesc& generic, MTL::Library* library,
                                           u32& compilerCalls, float& compileMs);
    NS::Object* compileFinal(const pipe::PipelineDesc& desc, MTL::Library* library, NS::Error** error);
    NS::Object* archiveLookup(const pipe::PipelineDesc& desc, MTL::Library* library, NS::Error** error);
    void waitForCompletions(const std::function<bool()>& done);
    /// Apply completions (render thread) and finish a committed/abandoned reload.
    u32 drain();

    MetalContext&        context_;
    Options              options_;
    MTL4::Compiler*      compiler_   = nullptr;
    MTL4::Archive*       archive_    = nullptr;
    MTL4::PipelineDataSetSerializer* serializer_ = nullptr;
    MTL::Library*        library_        = nullptr; // served generation
    MTL::Library*        pendingLibrary_ = nullptr; // reload in progress
    u32                  pendingGeneration_ = 0;
    std::string          archiveStatus_;
    bool                 startupDone_ = false;
    double               requestMs_   = 0.0;

    pipe::PipelineRegistry             registry_;
    std::unique_ptr<pipe::CompileQueue> queue_;

    // Completion signal for the blocking waits (startup, sync mode).
    std::mutex              waitMutex_;
    std::condition_variable waitCv_;
    u64                     posted_ = 0;

    // Flexible base pipelines per (generic key, library).
    std::mutex flexibleMutex_;
    struct FlexibleBase {
        MTL::Library*             library  = nullptr; // not retained: identity only
        MTL::RenderPipelineState* pipeline = nullptr;
    };
    std::unordered_map<u64, FlexibleBase> flexible_;
    std::unordered_map<u64, std::unique_ptr<std::once_flag>> flexibleOnce_;
    bool pruneFlexible_ = false; // a reload committed: drop the old library's bases
    /// Release the flexible bases of libraries no longer served, once no job
    /// of an older generation can still use them (render thread).
    void pruneFlexibleBases();
};

} // namespace phosphor
