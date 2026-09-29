#pragma once

#include "core/types.h"

#include <Metal/Metal.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// ShaderReloader -- shader hot reload, Debug builds (F3.6).
//
// A utility-QoS thread polls the shader directory (and src/renderer, where
// gpu_types.h lives) every 250 ms with FileWatcher.  On a change it rebuilds
// a complete metallib with the same compiler flags as cmake/App.cmake
// (passed in at build time), in a temporary directory, through runProcess
// (xcrun metal / metallib: no shell), then loads it with newLibrary.  The
// render thread picks the library up with takeLibrary() at a frame start and
// hands it to PipelineCache::reload(), which recompiles every pipeline off
// the render thread and swaps them atomically.  A compile error is logged
// with the compiler output and the running pipelines stay untouched.
// ---------------------------------------------------------------------------

class ShaderReloader {
public:
    /// `shaderDir`: directory of the .metal sources to watch and compile.
    ShaderReloader(MTL::Device* device, std::string shaderDir);
    ~ShaderReloader(); // stops and joins the thread

    ShaderReloader(const ShaderReloader&) = delete;
    ShaderReloader& operator=(const ShaderReloader&) = delete;

    /// A freshly built library (+1, the caller releases it), or null.
    [[nodiscard]] MTL::Library* takeLibrary();

    [[nodiscard]] u32 builds() const { return builds_.load(); }
    [[nodiscard]] u32 failures() const { return failures_.load(); }

    /// Compiler flags baked in by CMake (PHOSPHOR_SHADER_FLAGS).
    [[nodiscard]] static std::vector<std::string> compilerFlags();
    /// Build `sources` into `metallib` in `workDir`; false + `log` on error.
    static bool buildLibrary(const std::vector<std::string>& sources, const std::string& workDir,
                             const std::string& metallib, std::string& log);

private:
    void threadMain();
    void rebuild();

    MTL::Device* device_ = nullptr;
    std::string  shaderDir_;
    std::string  workDir_;

    std::thread             thread_;
    std::mutex              mutex_;
    std::condition_variable cv_;
    bool                    stop_ = false;
    MTL::Library*           ready_ = nullptr; // guarded by mutex_
    std::atomic<u32>        builds_{0};
    std::atomic<u32>        failures_{0};
};

} // namespace phosphor
