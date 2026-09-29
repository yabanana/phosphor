#pragma once

#include <string>
#include <vector>

namespace phosphor {

// ---------------------------------------------------------------------------
// runProcess -- spawn a command and capture its output (F3.6: the hot-reload
// thread rebuilds the shader library with xcrun metal / metallib).
//
// POSIX posix_spawnp (PATH lookup), stdout and stderr merged into `output`,
// blocking until exit.  No shell: arguments are passed verbatim.  Never call
// it on the render thread.
// ---------------------------------------------------------------------------

struct ProcessResult {
    int         exitCode = -1; // -1: could not spawn; 128+N: killed by signal N
    std::string output;        // stdout + stderr
};

[[nodiscard]] ProcessResult runProcess(const std::vector<std::string>& argv);

} // namespace phosphor
