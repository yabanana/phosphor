#include "app/engine.h"
#include "core/log.h"

#include <Foundation/Foundation.hpp>

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
        phosphor::Engine engine(argc, argv);
        engine.run();
        result = engine.exitCode();
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal: %s", e.what());
        result = EXIT_FAILURE;
    }
    pool->release();
    return result;
}
