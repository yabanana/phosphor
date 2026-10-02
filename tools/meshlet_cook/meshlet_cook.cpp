// meshlet_cook -- F6 spike S1 (CPU part): meshlet cook comparison.
//
//   meshlet_cook [--json FILE] [--runs N] [--commit HASH]
//
// Cooks a fixed corpus of procedural meshes with every option set of the spike
// (MeshletBuilder, meshoptimizer v1.3) and reports cook time (median and min of
// N runs) and the MeshletBuilder::stats numbers plus the per-meshlet triangle
// count quantiles.  A markdown table goes to stdout, a JSON file with a
// manifest (commit, meshoptimizer version, compiler, build type, date,
// machine) to --json.  CPU only; portable (no Apple headers).  The GPU frame
// cost of the same option sets is S1's GPU part (bench/f6_spike), measured
// separately: this tool alone does not choose a cook.
//
// Run it in Release and silence the builder's log line:
//   ./build-agent-rel/meshlet_cook --runs 5 --json out.json 2>/dev/null

#include "renderer/meshlet_builder.h"
#include "scene/procedural.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf
#include <meshoptimizer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace phosphor;
using json = nlohmann::json;

#ifndef PHOSPHOR_BUILD_TYPE
#define PHOSPHOR_BUILD_TYPE "unknown"
#endif

namespace {

struct CorpusMesh {
    std::string name;
    std::string params;
    std::vector<glm::vec3> positions;
    std::vector<u32> indices;
};

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

CorpusMesh fromMeshData(std::string name, std::string params, const MeshData& m) {
    CorpusMesh c;
    c.name      = std::move(name);
    c.params    = std::move(params);
    c.positions = m.positions;
    c.indices   = m.indices;
    return c;
}

// Cube surface of side 2, N x N quads per face, vertices welded on the shared
// lattice (an edge/corner vertex exists once).  Triangles are CCW seen from
// outside.
CorpusMesh buildingCube(u32 n) {
    CorpusMesh c;
    c.name   = "building";
    c.params = "subdivided cube, " + std::to_string(n) + "x" + std::to_string(n) + " quads per face, welded";
    std::map<u64, u32> weld;
    auto vertex = [&](u32 x, u32 y, u32 z) {
        const u64 key = (static_cast<u64>(x) << 40) | (static_cast<u64>(y) << 20) | z;
        auto [it, fresh] = weld.try_emplace(key, static_cast<u32>(c.positions.size()));
        if (fresh) {
            const float s = 2.0f / static_cast<float>(n);
            c.positions.push_back({static_cast<float>(x) * s - 1.0f, static_cast<float>(y) * s - 1.0f,
                                   static_cast<float>(z) * s - 1.0f});
        }
        return it->second;
    };
    // (u axis, v axis, normal axis, normal side): u x v = outward normal.
    struct Face { int u, v, n; u32 side; };
    const Face faces[6] = {{1, 2, 0, n}, {2, 1, 0, 0}, {2, 0, 1, n}, {0, 2, 1, 0}, {0, 1, 2, n}, {1, 0, 2, 0}};
    for (const Face& f : faces) {
        auto at = [&](u32 i, u32 j) {
            u32 p[3];
            p[f.u] = i;
            p[f.v] = j;
            p[f.n] = f.side;
            return vertex(p[0], p[1], p[2]);
        };
        for (u32 j = 0; j < n; ++j) {
            for (u32 i = 0; i < n; ++i) {
                const u32 a = at(i, j), b = at(i + 1, j), d = at(i + 1, j + 1), e = at(i, j + 1);
                c.indices.insert(c.indices.end(), {a, b, d, a, d, e});
            }
        }
    }
    return c;
}

CorpusMesh triangleSoup(u32 triangles, u32 seed) {
    CorpusMesh c;
    c.name   = "soup";
    c.params = std::to_string(triangles) + " random triangles, seed " + std::to_string(seed) +
               ", centres uniform in [-1,1]^3, edge ~0.03, no shared vertices";
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(-1.0f, 1.0f), off(-0.03f, 0.03f);
    for (u32 t = 0; t < triangles; ++t) {
        const glm::vec3 centre{pos(rng), pos(rng), pos(rng)};
        for (u32 k = 0; k < 3; ++k) {
            c.positions.push_back(centre + glm::vec3{off(rng), off(rng), off(rng)});
            c.indices.push_back(static_cast<u32>(c.positions.size() - 1));
        }
    }
    return c;
}

std::vector<CorpusMesh> buildCorpus() {
    std::vector<CorpusMesh> corpus;
    corpus.push_back(fromMeshData("sphere", "generateSphere(radius 1, 256 slices x 128 stacks)",
                                  ProceduralMeshes::generateSphere(1.0f, 256, 128)));
    corpus.push_back(fromMeshData("torus", "generateTorus(major 1, minor 0.35, 256 x 128)",
                                  ProceduralMeshes::generateTorus(1.0f, 0.35f, 256, 128)));
    corpus.push_back(fromMeshData("icosphere", "generateIcosahedron(radius 1, subdivisions 6, smooth normals)",
                                  ProceduralMeshes::generateIcosahedron(1.0f, 6, false)));
    corpus.push_back(fromMeshData("plane", "generatePlane(100 x 100, 512 x 512 quads)",
                                  ProceduralMeshes::generatePlane(100.0f, 100.0f, 512, 512)));
    corpus.push_back(buildingCube(32));
    corpus.push_back(fromMeshData("slivers",
                                  "generatePlane(2000 x 1, 1000 x 100 quads): 2 x 0.01 cells, 200k triangles, "
                                  "edge aspect > 200",
                                  ProceduralMeshes::generatePlane(2000.0f, 1.0f, 1000, 100)));
    corpus.push_back(triangleSoup(100000, 12345));
    return corpus;
}

// ---------------------------------------------------------------------------
// Option sets
// ---------------------------------------------------------------------------

struct OptionSet {
    std::string label;
    MeshletBuildOptions options;
};

MeshletBuildOptions standard(u32 v, u32 t, float cone = 0.5f, bool optimize = false) {
    MeshletBuildOptions o;
    o.maxVertices  = v;
    o.maxTriangles = t;
    o.coneWeight   = cone;
    o.optimize     = optimize;
    return o;
}

MeshletBuildOptions spatial(u32 v, u32 t) {
    MeshletBuildOptions o;
    o.maxVertices  = v;
    o.maxTriangles = t;
    o.algorithm    = MeshletAlgorithm::Spatial;
    return o;
}

std::vector<OptionSet> optionSets() {
    std::vector<OptionSet> s;
    auto add = [&](std::string label, const MeshletBuildOptions& o) { s.push_back({std::move(label), o}); };
    add("standard 64/124 cone0.5 (baseline)", standard(64, 124));
    add("standard 64/64", standard(64, 64));
    add("standard 64/96", standard(64, 96));
    add("standard 64/128", standard(64, 128));
    add("standard 96/128", standard(96, 128));
    add("standard 128/128", standard(128, 128));
    add("standard 64/124 cone0.0", standard(64, 124, 0.0f));
    add("standard 64/124 cone1.0", standard(64, 124, 1.0f));
    add("spatial 64/124", spatial(64, 124));
    add("spatial 64/96", spatial(64, 96));
    add("spatial 64/128", spatial(64, 128));
    add("standard 64/124 optimize", standard(64, 124, 0.5f, true));
    return s;
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct Row {
    std::string mesh, option, name;
    u64 vertices = 0, triangles = 0;
    double cookMsMedian = 0.0, cookMsMin = 0.0;
    MeshletStats stats;
    double duplication = 0.0;      // vertexRefs / unique vertices
    double bytesPerTriangle = 0.0;
    u32 triP10 = 0, triP50 = 0, triP90 = 0;
};

u32 quantile(const std::vector<u32>& sorted, double p) {
    if (sorted.empty()) return 0;
    const size_t r = static_cast<size_t>(std::ceil(p * static_cast<double>(sorted.size())));
    return sorted[std::clamp<size_t>(r, 1, sorted.size()) - 1];
}

Row measure(const CorpusMesh& mesh, const OptionSet& set, u32 runs) {
    Row row;
    row.mesh      = mesh.name;
    row.option    = set.label;
    row.name      = meshletOptionsName(set.options);
    row.vertices  = mesh.positions.size();
    row.triangles = mesh.indices.size() / 3;
    std::vector<double> times;
    MeshletBuildResult result;
    for (u32 r = 0; r < runs; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        result = MeshletBuilder::build(&mesh.positions[0].x, mesh.positions.size(), sizeof(glm::vec3),
                                       mesh.indices.data(), mesh.indices.size(), set.options);
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    row.cookMsMin    = times.front();
    row.cookMsMedian = times[times.size() / 2];
    row.stats        = MeshletBuilder::stats(result, set.options);
    row.duplication  = row.vertices ? static_cast<double>(row.stats.vertexRefs) / static_cast<double>(row.vertices) : 0.0;
    row.bytesPerTriangle = row.stats.triangles ? static_cast<double>(row.stats.bytes) / static_cast<double>(row.stats.triangles) : 0.0;
    std::vector<u32> counts;
    counts.reserve(result.meshlets.size());
    for (const Meshlet& m : result.meshlets) counts.push_back(m.triangleCount);
    std::sort(counts.begin(), counts.end());
    row.triP10 = quantile(counts, 0.10);
    row.triP50 = quantile(counts, 0.50);
    row.triP90 = quantile(counts, 0.90);
    return row;
}

std::string popenLine(const char* cmd) {
    std::string out;
    if (FILE* p = popen(cmd, "r")) {
        char buf[256];
        while (std::fgets(buf, sizeof buf, p)) out += buf;
        pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string jsonPath, commit;
    u32 runs = 5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--json" && i + 1 < argc) jsonPath = argv[++i];
        else if (a == "--runs" && i + 1 < argc) runs = static_cast<u32>(std::max(1, std::atoi(argv[++i])));
        else if (a == "--commit" && i + 1 < argc) commit = argv[++i];
        else {
            std::fprintf(stderr, "usage: meshlet_cook [--json FILE] [--runs N] [--commit HASH]\n");
            return 2;
        }
    }
    if (commit.empty()) commit = popenLine("git rev-parse HEAD 2>/dev/null");

    const std::vector<OptionSet> sets = optionSets();
    for (const OptionSet& s : sets) {
        const std::string why = validateMeshletOptions(s.options);
        if (!why.empty()) {
            std::fprintf(stderr, "option set '%s' is illegal: %s\n", s.label.c_str(), why.c_str());
            return 1;
        }
    }

    std::vector<CorpusMesh> corpus = buildCorpus();
    std::vector<Row> rows;
    for (const CorpusMesh& mesh : corpus) {
        std::fprintf(stderr, "cooking %s (%zu triangles)\n", mesh.name.c_str(), mesh.indices.size() / 3);
        for (const OptionSet& set : sets) rows.push_back(measure(mesh, set, runs));
    }

    // Markdown, one table per mesh.
    for (const CorpusMesh& mesh : corpus) {
        std::printf("### %s -- %s\n\n", mesh.name.c_str(), mesh.params.c_str());
        std::printf("| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | "
                    "dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |\n");
        std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (const Row& r : rows) {
            if (r.mesh != mesh.name) continue;
            std::printf("| %s | %.2f / %.2f | %llu | %llu | %u | %.1f | %.1f | %.3f | %.3f | %.2f | %.3f | %.3f | "
                        "%u/%u/%u |\n",
                        r.option.c_str(), r.cookMsMedian, r.cookMsMin, static_cast<unsigned long long>(r.triangles),
                        static_cast<unsigned long long>(r.vertices), r.stats.meshlets, r.stats.avgTriangles,
                        r.stats.avgVertices, r.stats.fillTriangles, r.duplication, r.bytesPerTriangle,
                        r.stats.coneUsable, r.stats.coneCutoffMean, r.triP10, r.triP50, r.triP90);
        }
        std::printf("\n");
    }
    std::printf("sponza: not available (assets/README.md lists the source; no download in this phase)\n");

    if (!jsonPath.empty()) {
        json j;
        char date[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%S%z", std::localtime(&now));
        j["manifest"] = {
            {"tool", "meshlet_cook (F6 spike S1, CPU part)"},
            {"commit", commit},
            {"meshoptimizer_version", MESHOPTIMIZER_VERSION},
            {"compiler", __VERSION__},
            {"build_type", PHOSPHOR_BUILD_TYPE},
            {"date", date},
            {"machine", popenLine("sysctl -n machdep.cpu.brand_string 2>/dev/null")},
            {"memory_bytes", popenLine("sysctl -n hw.memsize 2>/dev/null")},
            {"os", popenLine("sw_vers -productVersion 2>/dev/null")},
            {"runs", runs},
            {"statistic", "cook ms = median and min of runs (MeshletBuilder::build only, single thread)"},
            {"sponza", "not available (assets/README.md lists the source; no download in this phase)"},
        };
        j["corpus"] = json::array();
        for (const CorpusMesh& m : corpus) {
            j["corpus"].push_back({{"name", m.name}, {"params", m.params}, {"vertices", m.positions.size()},
                                   {"triangles", m.indices.size() / 3}});
        }
        j["results"] = json::array();
        for (const Row& r : rows) {
            j["results"].push_back({
                {"mesh", r.mesh}, {"option_set", r.option}, {"options_name", r.name},
                {"cook_ms_median", r.cookMsMedian}, {"cook_ms_min", r.cookMsMin},
                {"triangles", r.triangles}, {"vertices", r.vertices}, {"meshlets", r.stats.meshlets},
                {"avg_triangles", r.stats.avgTriangles}, {"avg_vertices", r.stats.avgVertices},
                {"fill", r.stats.fillTriangles}, {"vertex_refs", r.stats.vertexRefs},
                {"vertex_duplication", r.duplication}, {"bytes", r.stats.bytes},
                {"bytes_per_triangle", r.bytesPerTriangle}, {"cone_usable", r.stats.coneUsable},
                {"cone_cutoff_mean", r.stats.coneCutoffMean},
                {"triangles_per_meshlet", {{"p10", r.triP10}, {"p50", r.triP50}, {"p90", r.triP90}}},
            });
        }
        FILE* f = std::fopen(jsonPath.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", jsonPath.c_str());
            return 1;
        }
        const std::string text = j.dump(2) + "\n";
        std::fwrite(text.data(), 1, text.size(), f);
        std::fclose(f);
    }
    return 0;
}
