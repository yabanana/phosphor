// class_gen -- the material permutation generator of variant_gen (F15.2, ticket 21; ADR-0003).
//
//   variant_gen classes <material_classes.def> <output-dir>
//
// Reads the declared list `material_classes.def` and the feature-class manifests it names (renderer/
// material_class_manifest.h, exported by tools/class_manifest and by the app from the fixtures, the chess
// scene and the corpus) and writes, only when the content changed, into <output-dir>/pipeline/:
//   material_classes.generated.h        C++ class table, scene table and budgets
//   material_classes.generated.metal.h  MSL constants of the table
//   material_class_plan.json            the pipeline descriptors of the class table (kernel + function constants):
//                                       the targeted archive update (tools/material_class_archive.py) adds
//                                       exactly these to shaders/pipelines.mtl4-json
// Nothing depends on the time, the order of the file system or the host: two runs on the same inputs write the
// same bytes.  Exit code 1 on a malformed definition or manifest, or when the budget is exceeded.
//
// Line format of the .def (whitespace separated, '#' starts a comment):
//   budget archived <N>            max distinct material permutations (MATERIAL_FEATURES values) in the archive
//   budget active <N>              max feature classes of one scene that the per-class binning serves (the rest
//                                  falls back to the all-lobes pipeline, logged)
//   kernel <NAME>                  a resolve kernel that gets one pipeline per class (and the overflow pipeline)
//   manifest <path>                a manifest (relative to the .def); the order is irrelevant to the outputs
//   declare <NAME> <lobe>[|lobe..] a class that no manifest shows yet (declared list)
#include "renderer/material_class_manifest.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "class_gen.h"

namespace {

using phosphor::u32;
using namespace phosphor::matclass;

// RESOLVE_BIN values (renderer/visibility_layout.h VISIBILITY_BIN_CLASS / VISIBILITY_BIN_OVERFLOW; the engine static_asserts
// the generated copies against them).
constexpr u32 kBinClassFlag = 0x1000u;
constexpr u32 kBinOverflowValue = 0x2000u;

[[noreturn]] void fail(const std::string& file, int line, const std::string& msg) {
    if (line > 0) std::fprintf(stderr, "variant_gen: %s:%d: error: %s\n", file.c_str(), line, msg.c_str());
    else std::fprintf(stderr, "variant_gen: %s: error: %s\n", file.c_str(), msg.c_str());
    std::exit(1);
}

bool validName(const std::string& n) {
    if (n.empty() || !std::isupper(static_cast<unsigned char>(n[0]))) return false;
    for (char c : n)
        if (!std::isupper(static_cast<unsigned char>(c)) && !std::isdigit(static_cast<unsigned char>(c)) && c != '_') return false;
    return true;
}

bool parseUnsigned(const std::string& s, u32& out) {
    if (s.empty() || s.size() > 9) return false;
    for (char c : s)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    out = static_cast<u32>(std::stoul(s));
    return true;
}

std::string upperLobes(u32 mask) {
    std::string s;
    for (const std::string& n : lobeNames(mask)) {
        std::string u = n;
        for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        s += (s.empty() ? "" : "_") + u;
    }
    return s;
}

std::string hex(u32 v) {
    char b[16];
    std::snprintf(b, sizeof(b), "0x%X", v);
    return b;
}

std::string readFile(const std::filesystem::path& p, const std::string& what) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail(p.string(), 0, "cannot open " + what);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

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
    if (!out) {
        std::fprintf(stderr, "variant_gen: cannot write %s\n", path.string().c_str());
        return false;
    }
    return true;
}

} // namespace

int generateClasses(const std::string& defPath, const std::string& outputDir, const ClassGenConstants& fc) {
    using json = nlohmann::json;
    const std::filesystem::path defDir = std::filesystem::path(defPath).parent_path();
    std::ifstream in(defPath);
    if (!in) {
        std::fprintf(stderr, "variant_gen: cannot open %s\n", defPath.c_str());
        return 1;
    }

    u32 archivedBudget = 0, activeBudget = 0;
    std::vector<std::string> kernels, manifestPaths;
    struct Declared { std::string name; u32 mask; };
    std::vector<Declared> declared;
    std::set<std::string> declaredNames;

    std::string raw;
    int lineNo = 0;
    while (std::getline(in, raw)) {
        ++lineNo;
        if (const auto hash = raw.find('#'); hash != std::string::npos) raw.erase(hash);
        std::istringstream ls(raw);
        std::vector<std::string> tok;
        for (std::string t; ls >> t;) tok.push_back(t);
        if (tok.empty()) continue;
        if (tok[0] == "budget") {
            if (tok.size() != 3) fail(defPath, lineNo, "expected: budget <archived|active> <N>");
            u32 v = 0;
            if (!parseUnsigned(tok[2], v) || v == 0) fail(defPath, lineNo, "budget must be a positive integer");
            if (tok[1] == "archived") { if (archivedBudget) fail(defPath, lineNo, "duplicate budget archived"); archivedBudget = v; }
            else if (tok[1] == "active") { if (activeBudget) fail(defPath, lineNo, "duplicate budget active"); activeBudget = v; }
            else fail(defPath, lineNo, "unknown budget '" + tok[1] + "' (want archived or active)");
        } else if (tok[0] == "kernel") {
            if (tok.size() != 2) fail(defPath, lineNo, "expected: kernel <NAME>");
            if (std::find(kernels.begin(), kernels.end(), tok[1]) != kernels.end()) fail(defPath, lineNo, "duplicate kernel '" + tok[1] + "'");
            kernels.push_back(tok[1]);
        } else if (tok[0] == "manifest") {
            if (tok.size() != 2) fail(defPath, lineNo, "expected: manifest <path>");
            if (std::find(manifestPaths.begin(), manifestPaths.end(), tok[1]) != manifestPaths.end()) fail(defPath, lineNo, "duplicate manifest '" + tok[1] + "'");
            manifestPaths.push_back(tok[1]);
        } else if (tok[0] == "declare") {
            if (tok.size() != 3) fail(defPath, lineNo, "expected: declare <NAME> <lobe>[|lobe...]");
            if (!validName(tok[1])) fail(defPath, lineNo, "invalid class name '" + tok[1] + "'");
            if (!declaredNames.insert(tok[1]).second) fail(defPath, lineNo, "duplicate declared class '" + tok[1] + "'");
            u32 mask = 0;
            std::stringstream parts(tok[2]);
            for (std::string part; std::getline(parts, part, '|');) {
                const auto it = std::find_if(kLobeNames.begin(), kLobeNames.end(), [&](const char* n) { return part == n; });
                if (it == kLobeNames.end()) fail(defPath, lineNo, "unknown lobe '" + part + "'");
                mask |= 1u << (it - kLobeNames.begin());
            }
            if (mask == 0) fail(defPath, lineNo, "a declared class needs at least one lobe");
            declared.push_back({tok[1], mask});
        } else {
            fail(defPath, lineNo, "unknown directive '" + tok[0] + "' (want budget, kernel, manifest or declare)");
        }
    }
    if (!archivedBudget) fail(defPath, 0, "missing 'budget archived <N>'");
    if (!activeBudget) fail(defPath, 0, "missing 'budget active <N>'");
    if (kernels.empty()) fail(defPath, 0, "no 'kernel' declared");

    // ---- manifests --------------------------------------------------------
    std::vector<SceneEntry> scenes;
    std::set<std::string> sceneNames;
    for (const std::string& rel : manifestPaths) {
        const std::filesystem::path path = defDir / rel;
        Manifest m;
        std::string error;
        if (!parseManifest(readFile(path, "manifest"), m, error)) fail(path.string(), 0, error);
        for (SceneEntry& s : m.scenes) {
            if (!sceneNames.insert(s.name).second) fail(path.string(), 0, "scene '" + s.name + "' appears in two manifests");
            scenes.push_back(std::move(s));
        }
    }
    // Scenes are listed in name order (independent of the manifest order and of the host).
    std::sort(scenes.begin(), scenes.end(), [](const SceneEntry& a, const SceneEntry& b) { return a.name < b.name; });

    // ---- class table ------------------------------------------------------
    std::map<u32, u32> classScenes;       // per-material class mask (nonzero) -> scenes using it
    std::set<u32> unions;                 // scene unions (nonzero)
    bool anyVolumetric = false;
    // The resolve of a VOLUMETRIC scene also sets SSS_OUTPUT on every class pipeline of that scene, so a class gets the plain
    // pipeline when a plain scene (or the declared list) uses it and the SSS_OUTPUT one when a volumetric scene does.
    std::set<u32> volumetricClasses, plainClasses;
    u32 scenesOverBudget = 0;
    for (const SceneEntry& s : scenes) {
        if (s.unionMask) unions.insert(s.unionMask);
        for (const ClassUse& c : s.classes) {
            if (c.mask) ++classScenes[c.mask];
            if (c.mask) (s.volumetric ? volumetricClasses : plainClasses).insert(c.mask);
        }
        anyVolumetric |= s.volumetric;
        if (s.classes.size() > activeBudget) ++scenesOverBudget;
    }
    std::map<u32, std::string> declaredName;
    for (const Declared& d : declared) {
        if (!declaredName.emplace(d.mask, d.name).second) fail(defPath, 0, "declared classes '" + declaredName[d.mask] + "' and '" + d.name + "' have the same mask");
        classScenes.emplace(d.mask, 0u);
        plainClasses.insert(d.mask);
    }
    std::set<u32> permutations;
    for (const auto& [mask, count] : classScenes) permutations.insert(mask);
    permutations.insert(unions.begin(), unions.end());
    permutations.insert(phosphor::MATERIAL_FEATURES_ALL); // the all-lobes reference / overflow pipeline
    if (permutations.size() > archivedBudget) {
        fail(defPath, 0, "budget exceeded: " + std::to_string(permutations.size()) + " archived material permutations > " + std::to_string(archivedBudget));
    }

    const auto className = [&](u32 mask) {
        const auto it = declaredName.find(mask);
        return it != declaredName.end() ? it->second : "MC_" + upperLobes(mask);
    };

    // ---- C++ header -------------------------------------------------------
    std::ostringstream h;
    h << "// GENERATED by tools/variant_gen (classes) from shaders/material_classes.def and its manifests -- do not edit.\n"
         "#pragma once\n\n"
         "#include \"core/types.h\"\n\n"
         "#include <array>\n\n"
         "namespace phosphor::pipe::matclass::gen {\n\n"
         "inline constexpr u32 kArchivedPermutationBudget = " << archivedBudget << ";\n"
         "inline constexpr u32 kActiveClassBudget = " << activeBudget << ";\n"
         "inline constexpr u32 kAllLobesMask = " << hex(phosphor::MATERIAL_FEATURES_ALL) << ";\n"
         "inline constexpr u16 kMaterialFeaturesConstantIndex = " << fc.materialFeatures << ";\n"
         "inline constexpr u16 kResolveBinConstantIndex = " << fc.resolveBin << ";\n"
         "inline constexpr u16 kSssOutputConstantIndex = " << fc.sssOutput << ";\n"
         "inline constexpr u32 kBinClassFlag = " << hex(kBinClassFlag) << ";      // RESOLVE_BIN = kBinClassFlag | mask: one class bin\n"
         "inline constexpr u32 kBinOverflowValue = " << hex(kBinOverflowValue) << "; // RESOLVE_BIN of the all-lobes overflow bin\n\n"
         "struct ClassEntry {\n"
         "    const char* name;\n"
         "    u32         mask;   // feature mask of one material (bit n = MaterialFeature n)\n"
         "    u32         scenes; // manifest scenes that use it (0 = declared only)\n"
         "};\n\n"
         "inline constexpr u32 kClassCount = " << classScenes.size() << ";\n"
         "inline constexpr std::array<ClassEntry, kClassCount> kClasses{{\n";
    for (const auto& [mask, count] : classScenes) h << "    {\"" << className(mask) << "\", " << hex(mask) << ", " << count << "},\n";
    h << "}};\n\n"
         "/// Every archived MATERIAL_FEATURES value: class masks, scene unions and the all-lobes mask (ascending).\n"
         "inline constexpr u32 kPermutationCount = " << permutations.size() << ";\n"
         "inline constexpr std::array<u32, kPermutationCount> kPermutations{{";
    {
        bool first = true;
        for (u32 p : permutations) { h << (first ? "" : ", ") << hex(p); first = false; }
    }
    h << "}};\n\n"
         "struct SceneEntry {\n"
         "    const char* name;\n"
         "    u32         unionMask;\n"
         "    u32         classCount; // feature classes of the scene, the feature-free one included\n"
         "    bool        volumetric;\n"
         "};\n\n"
         "inline constexpr u32 kSceneCount = " << scenes.size() << ";\n"
         "inline constexpr u32 kScenesOverActiveBudget = " << scenesOverBudget << ";\n"
         "inline constexpr std::array<SceneEntry, kSceneCount> kScenes{{\n";
    for (const SceneEntry& s : scenes)
        h << "    {\"" << s.name << "\", " << hex(s.unionMask) << ", " << s.classes.size() << ", " << (s.volumetric ? "true" : "false") << "},\n";
    h << "}};\n\n"
         "constexpr bool isClassMask(u32 mask) {\n"
         "    for (const ClassEntry& c : kClasses)\n"
         "        if (c.mask == mask) return true;\n"
         "    return false;\n"
         "}\n"
         "constexpr bool isArchivedPermutation(u32 mask) {\n"
         "    for (u32 p : kPermutations)\n"
         "        if (p == mask) return true;\n"
         "    return false;\n"
         "}\n\n"
         "} // namespace phosphor::pipe::matclass::gen\n";

    // ---- MSL header -------------------------------------------------------
    std::ostringstream m;
    m << "// GENERATED by tools/variant_gen (classes) from shaders/material_classes.def and its manifests -- do not edit.\n"
         "//\n"
         "// The class table of the per-class binning (F15.2): kMaterialClassCount classes are generated, a scene\n"
         "// serves at most kMaterialClassActiveBudget of them through bins (the rest runs the all-lobes pipeline).\n"
         "#pragma once\n\n"
         "constant uint kMaterialClassCount = " << classScenes.size() << "u;\n"
         "constant uint kMaterialClassActiveBudget = " << activeBudget << "u;\n"
         "constant uint kMaterialArchivedPermutationBudget = " << archivedBudget << "u;\n";
    for (const auto& [mask, count] : classScenes) m << "constant uint " << className(mask) << "_MASK = " << hex(mask) << "u;\n";

    // ---- archive plan -----------------------------------------------------
    // One pipeline per (kernel, class) in bin mode 1 plus the all-lobes overflow pipeline (bin mode 2) per kernel;
    // classes of volumetric scenes (and the overflow pipeline when any scene is volumetric) also exist with SSS_OUTPUT.
    json plan;
    plan["schema"] = "phosphor.material-class-plan";
    plan["version"] = 1;
    plan["budgets"] = {{"archived_permutations", archivedBudget}, {"active_classes", activeBudget}};
    plan["permutations"] = std::vector<u32>(permutations.begin(), permutations.end());
    json pipelines = json::array();
    const auto entry = [&](const std::string& kernel, u32 mask, u32 binMode, bool sss) {
        // RESOLVE_BIN value: a class bin carries its own mask (the kernel filters on it), the overflow bin is a flag.
        const u32 binValue = binMode == 1 ? (kBinClassFlag | mask) : kBinOverflowValue;
        json constants = json::array();
        if (mask) constants.push_back({{"index", fc.materialFeatures}, {"value", mask}});
        if (sss) constants.push_back({{"index", fc.sssOutput}, {"value", 1}});
        constants.push_back({{"index", fc.resolveBin}, {"value", binValue}});
        std::sort(constants.begin(), constants.end(), [](const json& a, const json& b) { return a["index"].get<u32>() < b["index"].get<u32>(); });
        pipelines.push_back({{"kernel", kernel}, {"mask", mask}, {"bin_mode", binMode}, {"bin_value", binValue}, {"sss", sss}, {"constants", constants}});
    };
    // The feature-free class of an extended scene (a material without features next to extended ones) bins with the
    // scene's base resolve (no MATERIAL_FEATURES): a pipeline of its own because the bin constant is set.
    bool featureFreePlain = false, featureFreeVolumetric = false;
    for (const SceneEntry& s : scenes)
        if (s.unionMask != 0 && s.classes.front().mask == 0) (s.volumetric ? featureFreeVolumetric : featureFreePlain) = true;
    for (const std::string& kernel : kernels) {
        if (featureFreePlain) entry(kernel, 0, 1, false);
        if (featureFreeVolumetric) entry(kernel, 0, 1, true);
        for (const auto& [mask, count] : classScenes) {
            if (plainClasses.count(mask)) entry(kernel, mask, 1, false);
            if (volumetricClasses.count(mask)) entry(kernel, mask, 1, true);
        }
        entry(kernel, phosphor::MATERIAL_FEATURES_ALL, 2, false);
        if (anyVolumetric) entry(kernel, phosphor::MATERIAL_FEATURES_ALL, 2, true);
    }
    plan["pipelines"] = pipelines;

    const std::filesystem::path outDir = std::filesystem::path(outputDir) / "pipeline";
    if (!writeIfChanged(outDir / "material_classes.generated.h", h.str())) return 1;
    if (!writeIfChanged(outDir / "material_classes.generated.metal.h", m.str())) return 1;
    if (!writeIfChanged(outDir / "material_class_plan.json", plan.dump(2) + "\n")) return 1;
    if (scenesOverBudget)
        std::fprintf(stderr, "variant_gen: note: %u scene(s) have more than %u feature classes (the excess runs the all-lobes pipeline)\n", scenesOverBudget, activeBudget);
    return 0;
}
