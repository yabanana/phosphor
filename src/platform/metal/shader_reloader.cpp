#include "platform/metal/shader_reloader.h"
#include "core/profile.h"

#include "core/file_watcher.h"
#include "core/log.h"
#include "core/process.h"

#include <pthread/qos.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <system_error>

#ifndef PHOSPHOR_SHADER_FLAGS
#define PHOSPHOR_SHADER_FLAGS ""
#endif
#ifndef PHOSPHOR_RENDERER_SOURCE_DIR
#define PHOSPHOR_RENDERER_SOURCE_DIR ""
#endif

namespace phosphor {

namespace fs = std::filesystem;

std::vector<std::string> ShaderReloader::compilerFlags() {
    // CMake joins PHOSPHOR_METAL_FLAGS with '|' (a ';' would split the
    // compile definition).
    std::vector<std::string> flags;
    const std::string all = PHOSPHOR_SHADER_FLAGS;
    size_t start = 0;
    while (start <= all.size() && !all.empty()) {
        const size_t end = all.find('|', start);
        const std::string flag = all.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!flag.empty()) flags.push_back(flag);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return flags;
}

bool ShaderReloader::buildLibrary(const std::vector<std::string>& sources, const std::string& workDir,
                                  const std::string& metallib, std::string& log) {
    std::vector<std::string> airs;
    for (const std::string& source : sources) {
        const std::string air = workDir + "/" + fs::path(source).stem().string() + ".air";
        std::vector<std::string> argv = {"xcrun", "-sdk", "macosx", "metal"};
        const std::vector<std::string> flags = compilerFlags();
        argv.insert(argv.end(), flags.begin(), flags.end());
        argv.insert(argv.end(), {"-c", source, "-o", air});
        const ProcessResult r = runProcess(argv);
        if (r.exitCode != 0) {
            log = r.output.empty() ? "xcrun metal failed to start" : r.output;
            return false;
        }
        airs.push_back(air);
    }
    std::vector<std::string> argv = {"xcrun", "-sdk", "macosx", "metallib"};
    argv.insert(argv.end(), airs.begin(), airs.end());
    argv.insert(argv.end(), {"-o", metallib});
    const ProcessResult r = runProcess(argv);
    if (r.exitCode != 0) {
        log = r.output.empty() ? "xcrun metallib failed to start" : r.output;
        return false;
    }
    return true;
}

ShaderReloader::ShaderReloader(MTL::Device* device, std::string shaderDir)
    : device_(device), shaderDir_(std::move(shaderDir)) {
    std::error_code ec;
    workDir_ = (fs::temp_directory_path(ec) / ("phosphor-hot-reload-" + std::to_string(::getpid()))).string();
    fs::create_directories(workDir_, ec);
    device_->retain();
    thread_ = std::thread([this] { threadMain(); });
    LOG_INFO("Shader hot reload: watching %s (build dir %s)", shaderDir_.c_str(), workDir_.c_str());
}

ShaderReloader::~ShaderReloader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
    if (ready_) ready_->release();
    device_->release();
    std::error_code ec;
    fs::remove_all(workDir_, ec);
}

MTL::Library* ShaderReloader::takeLibrary() {
    std::lock_guard<std::mutex> lock(mutex_);
    MTL::Library* library = ready_;
    ready_ = nullptr;
    return library;
}

void ShaderReloader::threadMain() {
    PH_THREAD_NAME("phosphor-shader-reload");
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    FileWatcher shaders(shaderDir_, {".metal", ".h"});
    FileWatcher types(PHOSPHOR_RENDERER_SOURCE_DIR, {".h"});
    shaders.poll(); // baselines
    types.poll();
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        cv_.wait_for(lock, std::chrono::milliseconds(250), [this] { return stop_; });
        if (stop_) break;
        lock.unlock();
        const bool changed = shaders.poll() | types.poll();
        if (changed) {
            for (const std::string& name : shaders.changed()) LOG_INFO("Shader changed: %s", name.c_str());
            for (const std::string& name : types.changed()) LOG_INFO("Shader header changed: %s", name.c_str());
            rebuild();
        }
        lock.lock();
    }
}

void ShaderReloader::rebuild() {
    PH_ZONE("Shader reload");
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::string> sources;
    std::error_code ec;
    for (const fs::directory_entry& e : fs::directory_iterator(shaderDir_, ec)) {
        if (e.is_regular_file() && e.path().extension() == ".metal") sources.push_back(e.path().string());
    }
    std::sort(sources.begin(), sources.end());
    const u32 build = builds_.load() + failures_.load() + 1;
    const std::string metallib = workDir_ + "/phosphor-" + std::to_string(build) + ".metallib";
    std::string log;
    if (!buildLibrary(sources, workDir_, metallib, log)) {
        ++failures_;
        LOG_ERROR("Shader hot reload: build failed, pipelines unchanged:\n%s", log.c_str());
        pool->release();
        return;
    }
    NS::Error* error = nullptr;
    MTL::Library* library = device_->newLibrary(NS::String::string(metallib.c_str(), NS::UTF8StringEncoding), &error);
    if (!library) {
        ++failures_;
        LOG_ERROR("Shader hot reload: cannot load %s: %s", metallib.c_str(),
                  error ? error->localizedDescription()->utf8String() : "unknown error");
        pool->release();
        return;
    }
    ++builds_;
    LOG_INFO("Shader hot reload: %zu shaders rebuilt in %.0f ms", sources.size(),
             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_) ready_->release(); // superseded before the render thread took it
        ready_ = library;
    }
    pool->release();
}

} // namespace phosphor
