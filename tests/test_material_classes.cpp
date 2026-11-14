// F15.2 (ticket 21): feature-class manifests, the permutation generator (tools/variant_gen classes) and its budgets.
#include <doctest/doctest.h>

#include "pipeline/material_classes.generated.h"
#include "renderer/material_class_export.h"
#include "renderer/material_class_manifest.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include <json.hpp>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace phosphor;
namespace fs = std::filesystem;

namespace {

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void spit(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out << text;
}

// Exit status of `variant_gen classes ...` (stdout and stderr discarded).
int runGenerator(const fs::path& def, const fs::path& out) {
    const std::string cmd = std::string("\"") + PHOSPHOR_VARIANT_GEN_BIN + "\" classes \"" + PHOSPHOR_SOURCE_DIR + "/shaders/variants.def\" \"" +
                            def.string() + "\" \"" + out.string() + "\" >/dev/null 2>&1";
    const int status = std::system(cmd.c_str());
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

struct TempDir {
    fs::path path;
    TempDir() {
        static int counter = 0;
        path = fs::temp_directory_path() / ("phosphor_matclass_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

matclass::Manifest sample() {
    matclass::Manifest m;
    m.source = "unit test";
    m.scenes.push_back(matclass::makeSceneEntry("material:a", "material", {MATERIAL_FEATURE_CLEARCOAT, MATERIAL_FEATURE_CLEARCOAT, 0}, false));
    m.scenes.push_back(matclass::makeSceneEntry("material:b", "material",
                                                {MATERIAL_FEATURE_SPECULAR_IOR | MATERIAL_FEATURE_SHEEN, MATERIAL_FEATURE_SUBSURFACE}, true));
    return m;
}

bool rejects(const nlohmann::json& j, const std::string& needle) {
    matclass::Manifest m;
    std::string error;
    const bool ok = matclass::parseManifest(j.dump(), m, error);
    CHECK_MESSAGE(!ok, "accepted: " << needle);
    CHECK_MESSAGE(error.find(needle) != std::string::npos, "error was: " << error << " (wanted '" << needle << "')");
    return !ok;
}

// A one-scene definition in a temp directory; `manifestJson` is the manifest text.
fs::path writeDef(const TempDir& t, const std::string& manifestJson, const std::string& extra = "", unsigned archived = 64) {
    spit(t.path / "m.json", manifestJson);
    spit(t.path / "x.def",
         "budget archived " + std::to_string(archived) + "\nbudget active 16\nkernel visibility_resolve\nmanifest m.json\n" + extra);
    return t.path / "x.def";
}

} // namespace

TEST_CASE("F15.2 manifest: serialisation is deterministic and round-trips") {
    const matclass::Manifest m = sample();
    const std::string text = matclass::serialize(m);
    CHECK(text == matclass::serialize(m));
    matclass::Manifest back;
    std::string error;
    REQUIRE_MESSAGE(matclass::parseManifest(text, back, error), error);
    CHECK(back == m);
    CHECK(m.scenes[0].unionMask == MATERIAL_FEATURE_CLEARCOAT);
    CHECK(m.scenes[0].classes.size() == 2); // the feature-free class and the clearcoat class
    CHECK(m.scenes[0].classes[1].materials == 2);
    CHECK(m.scenes[1].volumetric);
}

TEST_CASE("F15.2 manifest: malformed manifests and schema errors are rejected") {
    const nlohmann::json good = matclass::toJson(sample());
    matclass::Manifest m;
    std::string error;
    CHECK_FALSE(matclass::parseManifest("", m, error));
    CHECK_FALSE(matclass::parseManifest("{\"schema\":", m, error));
    CHECK_FALSE(matclass::parseManifest("[]", m, error));
    CHECK_FALSE(matclass::parseManifest(good.dump() + "x", m, error)); // trailing garbage

    auto j = good;
    j["schema"] = "something-else";
    rejects(j, "schema name");
    j = good;
    j["version"] = 2;
    rejects(j, "unsupported version");
    j = good;
    j["version"] = -1;
    rejects(j, "version");
    j = good;
    j["extra"] = 1;
    rejects(j, "unknown key 'extra'");
    j = good;
    j.erase("source");
    rejects(j, "missing key 'source'");
    j = good;
    j["scenes"] = nlohmann::json::array();
    rejects(j, "non-empty array");
    j = good;
    j["scenes"][0]["extra"] = true;
    rejects(j, "unknown key");
    j = good;
    j["scenes"][1]["name"] = "material:a";
    rejects(j, "duplicate scene name");
    j = good;
    j["scenes"][0]["union"] = 1;
    rejects(j, "OR of the class masks");
    j = good;
    j["scenes"][0]["classes"] = nlohmann::json::array();
    rejects(j, "non-empty array");
    j = good;
    j["scenes"][0]["classes"][1]["lobes"] = nlohmann::json::array({"sheen"});
    rejects(j, "do not match the mask");
    j = good;
    j["scenes"][0]["classes"][1]["materials"] = 0;
    rejects(j, "must be >= 1");
    j = good;
    std::swap(j["scenes"][1]["classes"][0], j["scenes"][1]["classes"][1]);
    rejects(j, "unique and ascending");
    j = good;
    j["scenes"][1]["classes"][1] = j["scenes"][1]["classes"][0];
    rejects(j, "unique and ascending");
    j = good;
    j["scenes"][0]["classes"][1]["mask"] = 0x200;
    rejects(j, "outside MATERIAL_FEATURES_ALL");
    j = good;
    j["scenes"][0]["volumetric"] = true; // no subsurface bit
    rejects(j, "no subsurface bit");
    j = good;
    j["scenes"][0]["volumetric"] = 1;
    rejects(j, "not a boolean");
    j = good;
    j["scenes"][0]["classes"][1]["mask"] = 1.5;
    rejects(j, "unsigned 32-bit");
}

TEST_CASE("F15.2 the committed fixture manifests are the current export of every fixture scene") {
    for (const std::string& family : matclass::fixtureFamilies()) {
        const fs::path path = fs::path(PHOSPHOR_SOURCE_DIR) / "shaders/material_classes" / (family + ".json");
        REQUIRE_MESSAGE(fs::exists(path), path);
        const std::string fresh = matclass::serialize(matclass::exportFixtureFamily(family));
        CHECK_MESSAGE(slurp(path) == fresh, path << " is stale: run build/release/class_manifest export shaders/material_classes");
        matclass::Manifest m;
        std::string error;
        CHECK_MESSAGE(matclass::parseManifest(slurp(path), m, error), error);
        CHECK(m.scenes.size() == matclass::fixtureScenarios(family).size());
    }
}

TEST_CASE("F15.2 generated table covers every fixture class, union and the all-lobes mask within the budgets") {
    namespace gen = pipe::matclass::gen;
    CHECK(gen::kArchivedPermutationBudget == 64);
    CHECK(gen::kActiveClassBudget == 16);
    CHECK(gen::kPermutationCount <= gen::kArchivedPermutationBudget);
    CHECK(gen::isArchivedPermutation(MATERIAL_FEATURES_ALL));
    CHECK(gen::kScenesOverActiveBudget == 0);
    std::set<u32> masks;
    u32 scenes = 0;
    for (const std::string& family : matclass::fixtureFamilies()) {
        for (const std::string& name : matclass::fixtureScenarios(family)) {
            const matclass::SceneEntry e = matclass::exportFixtureScene(family, name);
            ++scenes;
            CHECK_MESSAGE(e.classes.size() <= gen::kActiveClassBudget, e.name);
            if (e.unionMask) CHECK_MESSAGE(gen::isArchivedPermutation(e.unionMask), e.name);
            for (const matclass::ClassUse& c : e.classes) {
                masks.insert(c.mask);
                if (c.mask) CHECK_MESSAGE(gen::isClassMask(c.mask), e.name << " class " << c.mask);
            }
        }
    }
    CHECK(gen::kSceneCount == scenes);
    for (const auto& c : gen::kClasses) CHECK(masks.count(c.mask) == 1); // no class without a user (nothing is declared)
}

TEST_CASE("F15.2 generator: two runs write byte-identical outputs, independent of the manifest order") {
    TempDir t;
    const fs::path def = fs::path(PHOSPHOR_SOURCE_DIR) / "shaders/material_classes.def";
    REQUIRE(runGenerator(def, t.path / "a") == 0);
    REQUIRE(runGenerator(def, t.path / "b") == 0);
    for (const char* f : {"material_classes.generated.h", "material_classes.generated.metal.h", "material_class_plan.json"}) {
        const std::string a = slurp(t.path / "a/pipeline" / f), b = slurp(t.path / "b/pipeline" / f);
        CHECK_MESSAGE(!a.empty(), f);
        CHECK_MESSAGE(a == b, f << " differs between two runs");
    }
    // The manifest order of the definition does not matter: a reversed list gives the same bytes.
    std::string text = slurp(def);
    std::vector<std::string> lines;
    std::stringstream ss(text);
    for (std::string l; std::getline(ss, l);) lines.push_back(l);
    std::vector<std::string> manifestLines;
    for (const auto& l : lines)
        if (l.rfind("manifest ", 0) == 0) manifestLines.push_back(l);
    REQUIRE(manifestLines.size() == 3);
    std::string reordered;
    for (const auto& l : lines) {
        if (l.rfind("manifest ", 0) == 0) continue;
        reordered += l + "\n";
    }
    for (auto it = manifestLines.rbegin(); it != manifestLines.rend(); ++it) reordered += *it + "\n";
    // Relative manifest paths resolve against the definition's directory: write it next to a copy of the manifests.
    fs::copy(fs::path(PHOSPHOR_SOURCE_DIR) / "shaders/material_classes", t.path / "material_classes");
    spit(t.path / "reordered.def", reordered);
    REQUIRE(runGenerator(t.path / "reordered.def", t.path / "c") == 0);
    for (const char* f : {"material_classes.generated.h", "material_classes.generated.metal.h", "material_class_plan.json"})
        CHECK_MESSAGE(slurp(t.path / "a/pipeline" / f) == slurp(t.path / "c/pipeline" / f), f << " depends on the manifest order");
    // The build's own generated header is the same table.
    CHECK(slurp(t.path / "a/pipeline/material_classes.generated.h") ==
          slurp(fs::path(PHOSPHOR_BINARY_DIR) / "generated/pipeline/material_classes.generated.h"));
}

TEST_CASE("F15.2 generator: the archive plan names one class pipeline per kernel and class plus the all-lobes overflow") {
    const nlohmann::json plan = nlohmann::json::parse(slurp(fs::path(PHOSPHOR_BINARY_DIR) / "generated/pipeline/material_class_plan.json"));
    CHECK(plan["schema"] == "phosphor.material-class-plan");
    CHECK(plan["budgets"]["archived_permutations"] == 64);
    CHECK(plan["budgets"]["active_classes"] == 16);
    std::set<std::string> identities;
    std::set<u32> usedMasks;
    u32 overflow = 0;
    for (const auto& p : plan["pipelines"]) {
        const std::string kernel = p["kernel"];
        CHECK((kernel == "visibility_resolve" || kernel == "visibility_lit_resolve"));
        const u32 mask = p["mask"], bin = p["bin_mode"];
        const bool sss = p["sss"];
        CHECK(identities.insert(kernel + "/" + std::to_string(mask) + "/" + std::to_string(bin) + "/" + (sss ? "s" : "-")).second); // no duplicates
        u32 previous = 0;
        bool haveBin = false;
        for (const auto& c : p["constants"]) {
            const u32 index = c["index"];
            CHECK(index > previous); // ascending, unique
            previous = index;
            if (index == pipe::matclass::gen::kResolveBinConstantIndex) {
                haveBin = true;
                CHECK(c["value"] == p["bin_value"]);
                CHECK(c["value"] == (bin == 1 ? (pipe::matclass::gen::kBinClassFlag | mask) : pipe::matclass::gen::kBinOverflowValue));
            }
            if (index == pipe::matclass::gen::kMaterialFeaturesConstantIndex) CHECK(c["value"] == mask);
        }
        CHECK(haveBin);
        if (bin == 2) {
            ++overflow;
            CHECK(mask == MATERIAL_FEATURES_ALL);
        }
        usedMasks.insert(mask);
        if (mask) CHECK(pipe::matclass::gen::isArchivedPermutation(mask));
    }
    CHECK(overflow >= 2); // one per kernel
    for (const auto& c : pipe::matclass::gen::kClasses) CHECK(usedMasks.count(c.mask) == 1);
}

TEST_CASE("F15.2 generator: budgets, schema and definition errors fail the generation") {
    TempDir t;
    const std::string ok = matclass::serialize(sample());
    // Baseline: a valid input generates, the sample has 3 class masks, 2 unions (one equal to a class) and the all-lobes mask: 5 permutations.
    REQUIRE(runGenerator(writeDef(t, ok), t.path / "o1") == 0);
    CHECK(slurp(t.path / "o1/pipeline/material_classes.generated.h").find("kClassCount = 3") != std::string::npos);
    // Archived-permutation budget exceeded: 5 permutations > 4.
    CHECK(runGenerator(writeDef(t, ok, "", 4), t.path / "o2") != 0);
    CHECK(runGenerator(writeDef(t, ok, "", 5), t.path / "o3") == 0);
    // A declared class counts against the budget; a declared duplicate or unknown lobe is an error.
    CHECK(runGenerator(writeDef(t, ok, "declare EXTRA volume\n", 5), t.path / "o4") != 0);
    CHECK(runGenerator(writeDef(t, ok, "declare EXTRA volume\n", 6), t.path / "o5") == 0);
    CHECK(runGenerator(writeDef(t, ok, "declare EXTRA nothing\n"), t.path / "o6") != 0);
    CHECK(runGenerator(writeDef(t, ok, "declare DUP clearcoat\n"), t.path / "o7") == 0); // the same mask as a manifest class is fine ...
    // ... but two declarations of one mask are not.
    CHECK(runGenerator(writeDef(t, ok, "declare A volume\ndeclare B volume\n"), t.path / "o8") != 0);
    // Definition errors.
    CHECK(runGenerator(writeDef(t, ok, "bogus 1\n"), t.path / "o9") != 0);
    spit(t.path / "nokernel.def", "budget archived 64\nbudget active 16\nmanifest m.json\n");
    CHECK(runGenerator(t.path / "nokernel.def", t.path / "o10") != 0);
    spit(t.path / "nobudget.def", "budget active 16\nkernel visibility_resolve\nmanifest m.json\n");
    CHECK(runGenerator(t.path / "nobudget.def", t.path / "o11") != 0);
    spit(t.path / "missing.def", "budget archived 64\nbudget active 16\nkernel visibility_resolve\nmanifest absent.json\n");
    CHECK(runGenerator(t.path / "missing.def", t.path / "o12") != 0);
    // Manifest schema errors reach the generator through the shared parser.
    nlohmann::json bad = nlohmann::json::parse(ok);
    bad["version"] = 9;
    CHECK(runGenerator(writeDef(t, bad.dump()), t.path / "o13") != 0);
    CHECK(runGenerator(writeDef(t, "{not json"), t.path / "o14") != 0);
    // A scene named in two manifests is an error.
    spit(t.path / "m2.json", ok);
    CHECK(runGenerator(writeDef(t, ok, "manifest m2.json\n"), t.path / "o15") != 0);
    // A failed run does not write outputs.
    CHECK_FALSE(fs::exists(t.path / "o2/pipeline/material_classes.generated.h"));
}

TEST_CASE("F15.2 generator: a scene beyond the active-class budget is reported, not rejected") {
    TempDir t;
    std::vector<u32> masks;
    for (u32 m = 1; m <= 17; ++m) masks.push_back(m); // 17 classes in one scene
    matclass::Manifest m;
    m.source = "unit test";
    m.scenes.push_back(matclass::makeSceneEntry("material:wide", "material", masks, false));
    REQUIRE(runGenerator(writeDef(t, matclass::serialize(m)), t.path / "o") == 0);
    CHECK(slurp(t.path / "o/pipeline/material_classes.generated.h").find("kScenesOverActiveBudget = 1") != std::string::npos);
}

#include "renderer/material_class_table.h"

namespace {
std::vector<GPUMaterial> materialsWithMasks(const std::vector<u32>& masks) {
    std::vector<GPUMaterial> out(masks.size());
    for (size_t i = 0; i < masks.size(); ++i) {
        out[i] = GPUMaterial{};
        out[i].flags = masks[i] << MATERIAL_FEATURE_SHIFT;
    }
    return out;
}
} // namespace

TEST_CASE("F15.2 class table: bins by descending material count then ascending mask; the overflow bin is the last") {
    const auto mats = materialsWithMasks({0, 1, 1, 3, 3, 3, 2});
    const MaterialClassTable t = buildMaterialClassTable(mats);
    CHECK(t.count == 4);
    CHECK(t.masks[0] == 3); // 3 materials
    CHECK(t.masks[1] == 1); // 2
    CHECK(t.masks[2] == 0); // 1 each, ascending mask
    CHECK(t.masks[3] == 2);
    CHECK(t.materials[0] == 3);
    CHECK_FALSE(t.overflow());
    CHECK(t.binOf(1) == 1);
    CHECK(t.binOf(0) == 2);
    CHECK(t.binOf(7) == VISIBILITY_FEATURE_CLASSES); // not in the table: the overflow bin
    CHECK(buildMaterialClassTable(mats) == t);      // deterministic
}

TEST_CASE("F15.2 class table: beyond the budget the most used classes keep a bin and the rest overflow") {
    std::vector<u32> masks;
    for (u32 m = 1; m <= 20; ++m) masks.push_back(m);     // 20 classes with one material each ...
    masks.push_back(5); masks.push_back(5); masks.push_back(9); // ... mask 5 has 3 materials, mask 9 has 2
    const MaterialClassTable t = buildMaterialClassTable(materialsWithMasks(masks));
    CHECK(t.count == VISIBILITY_FEATURE_CLASSES);
    CHECK(t.masks[0] == 5);
    CHECK(t.masks[1] == 9);
    CHECK(t.overflowClasses == 20 - VISIBILITY_FEATURE_CLASSES);
    CHECK(t.overflowMaterials == t.overflowClasses);
    CHECK(t.overflow());
    // The kept classes are the 14 smallest masks of the single-material ones (ascending tie break).
    CHECK(t.binOf(1) < VISIBILITY_FEATURE_CLASSES);
    CHECK(t.binOf(20) == VISIBILITY_FEATURE_CLASSES);
    // A tighter budget only shrinks the table.
    CHECK(buildMaterialClassTable(materialsWithMasks(masks), 3).count == 3);
    // The empty scene has no class.
    CHECK(buildMaterialClassTable({}).count == 0);
}
