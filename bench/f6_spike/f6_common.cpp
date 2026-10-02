#include "f6_common.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <unistd.h>

namespace f6 {

namespace {

std::filesystem::path shaderBase() {
    const char* env = std::getenv("SOC_SHADER_DIR");
    return env ? env : SOC_SHADER_DIR;
}

void expand(const std::filesystem::path& file, std::set<std::string>& seen, std::string& out) {
    std::ifstream in(file);
    if (!in) throw soc::BenchError("cannot read " + file.string());
    std::string line;
    while (std::getline(in, line)) {
        const size_t i = line.find_first_not_of(" \t");
        if (i != std::string::npos && line.compare(i, 8, "#include") == 0) {
            const size_t q0 = line.find('"', i);
            if (q0 != std::string::npos) {
                const size_t q1 = line.find('"', q0 + 1);
                const std::string rel = line.substr(q0 + 1, q1 - q0 - 1);
                std::filesystem::path p = std::filesystem::path(SOC_SOURCE_DIR) / "src" / rel;
                if (!std::filesystem::exists(p)) p = std::filesystem::path(F6_GENERATED_DIR) / rel;
                if (seen.insert(p.string()).second) {
                    out += "// ---- begin " + rel + "\n";
                    expand(p, seen, out);
                    out += "// ---- end " + rel + "\n";
                }
                continue;
            }
        }
        if (i != std::string::npos && line.compare(i, 12, "#pragma once") == 0) continue;
        out += line;
        out += '\n';
    }
}

} // namespace

MTL::Library* f6Library(soc::Context& ctx, const std::string& file, bool fastMath) {
    return ctx.library("../../f6_spike/shaders/" + file, fastMath);
}

MTL::Library* engineLibrary(soc::Context& ctx, const std::string& file, bool fastMath) {
    std::set<std::string> seen;
    std::string src;
    expand(std::filesystem::path(SOC_SOURCE_DIR) / "shaders" / file, seen, src);
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / ("f6_engine_" + std::to_string(getpid()) + "_" + file);
    {
        std::ofstream o(tmp);
        o << src;
    }
    return ctx.library(std::filesystem::relative(tmp, shaderBase()).string(), fastMath);
}

} // namespace f6
