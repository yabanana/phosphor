#pragma once

#include "core/types.h"
#include "pipeline/compile_queue.h"
#include "pipeline/pipeline_registry.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace phosphor {

class MetalContext;

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
//     2. on a miss (F3.5, reason recorded) a render pipeline first gets a
//        FALLBACK: the flexible pipeline of its generic variant
//        (unspecialized output state, compiled once per library) specialised
//        to the real output state (newRenderPipelineStateBySpecialization,
//        F3.2), then the full compile replaces it.
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
    pipe::PipelineHandle request(const pipe::PipelineDesc& desc);

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

    // --- F3.4 harvest -----------------------------------------------------------
    [[nodiscard]] bool harvesting() const { return serializer_ != nullptr; }
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

    MetalContext&        context_;
    Options              options_;
    MTL4::Compiler*      compiler_   = nullptr;
    MTL4::Archive*       archive_    = nullptr;
    MTL4::PipelineDataSetSerializer* serializer_ = nullptr;
    MTL::Library*        library_        = nullptr; // served generation
    MTL::Library*        pendingLibrary_ = nullptr; // reload in progress
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
    std::unordered_map<u64, MTL::RenderPipelineState*> flexible_;
    std::unordered_map<u64, std::unique_ptr<std::once_flag>> flexibleOnce_;
};

} // namespace phosphor
