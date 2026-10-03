#include "app/engine.h"
#include "platform/metal/temporal_worker.h"
#include <cstring>
#include "core/log.h"

#include <Foundation/Foundation.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char* argv[]) {
#if defined(PHOSPHOR_METAL_VALIDATION)
    // API validation must be requested before the Metal device is created.
    // Shader validation is much slower; opt in with MTL_SHADER_VALIDATION=1.
    setenv("MTL_DEBUG_LAYER", "1", /*overwrite*/ 0);
#endif
    // Outer pool for objects autoreleased outside the per-frame pools
    // (startup, bench switches).
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    int result = EXIT_SUCCESS;
    try {
        if (argc == 2 && std::strcmp(argv[1], "--metalfx-worker") == 0) {
            result = phosphor::runTemporalWorker();
            pool->release();
            return result;
        }
        phosphor::Engine engine(argc, argv);
        engine.run();
        result = engine.exitCode();
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal: %s", e.what());
        result = EXIT_FAILURE;
    }
    pool->release();
    if (phosphor::TemporalWorker::spawnedCount()) {
        const auto spawned = phosphor::TemporalWorker::spawnedCount(), reaped = phosphor::TemporalWorker::reapedCount();
        if (spawned != reaped || phosphor::TemporalWorker::failureCount() || phosphor::TemporalWorker::mappedBytes())
            result = EXIT_FAILURE;
        std::printf("METALFX-WORKERS spawned %llu reaped %llu peak-live %llu failures %llu shared-bytes %llu | %s\n",
                    static_cast<unsigned long long>(spawned), static_cast<unsigned long long>(reaped),
                    static_cast<unsigned long long>(phosphor::TemporalWorker::peakLiveCount()),
                    static_cast<unsigned long long>(phosphor::TemporalWorker::failureCount()),
                    static_cast<unsigned long long>(phosphor::TemporalWorker::mappedBytes()),
                    result == EXIT_SUCCESS ? "PASS" : "FAIL");
    }
    // stdout: the exit status as the app decided it, after the engine is
    // destroyed.  A run whose shell status differs from this line (or that
    // lacks it) needs the raw process status/signal and stderr to classify
    // its failure; absence alone does not prove a signal (F5/F6 review).
    std::printf("EXIT %d\n", result);
    std::fflush(stdout);
    return result;
}
