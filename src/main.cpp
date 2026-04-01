#include "core/engine.h"
#include "core/log.h"

#include <cstdlib>
#include <exception>

int main(int argc, char* argv[]) {
    try {
        phosphor::Engine engine(argc, argv);
        engine.run();
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal: %s", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
