// class_manifest -- exports the feature-class manifests of the F15 fixtures (F15.2, ticket 21).
//
//   class_manifest export <output-dir>      writes <family>.json for every fixture family
//                                           (only when the content changed)
//   class_manifest list                     prints "<family>:<scenario>" of every fixture scene
//   class_manifest validate <manifest.json> exits 0 when the manifest passes the schema, 1 otherwise
//
// Portable (no GPU): the benches are set up on an ECS and a SceneStore reads the materials.
#include "renderer/material_class_export.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace phosphor;

namespace {

bool writeIfChanged(const std::filesystem::path& path, const std::string& content) {
    {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            std::stringstream ss;
            ss << in.rdbuf();
            if (ss.str() == content) return true;
        }
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
    out.flush();
    return static_cast<bool>(out);
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    try {
        if (mode == "export" && argc == 3) {
            for (const std::string& family : matclass::fixtureFamilies()) {
                const auto manifest = matclass::exportFixtureFamily(family);
                const std::filesystem::path path = std::filesystem::path(argv[2]) / (family + ".json");
                if (!writeIfChanged(path, matclass::serialize(manifest))) {
                    std::fprintf(stderr, "class_manifest: cannot write %s\n", path.string().c_str());
                    return 1;
                }
                std::printf("%s: %zu scenes\n", path.string().c_str(), manifest.scenes.size());
            }
            return 0;
        }
        if (mode == "list" && argc == 2) {
            for (const std::string& family : matclass::fixtureFamilies())
                for (const std::string& s : matclass::fixtureScenarios(family)) std::printf("%s:%s\n", family.c_str(), s.c_str());
            return 0;
        }
        if (mode == "validate" && argc == 3) {
            std::ifstream in(argv[2], std::ios::binary);
            if (!in) {
                std::fprintf(stderr, "class_manifest: cannot open %s\n", argv[2]);
                return 1;
            }
            std::stringstream ss;
            ss << in.rdbuf();
            matclass::Manifest m;
            std::string error;
            if (!matclass::parseManifest(ss.str(), m, error)) {
                std::fprintf(stderr, "class_manifest: %s: %s\n", argv[2], error.c_str());
                return 1;
            }
            std::printf("%s: valid, %zu scenes\n", argv[2], m.scenes.size());
            return 0;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "class_manifest: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "usage: class_manifest export <dir> | list | validate <manifest.json>\n");
    return 2;
}
