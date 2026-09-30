// soc_model -- OPT-0.3 command line front end of the SoC cost model.
//
//   soc_model model   --results bench/results/X.json [--out model.json]
//   soc_model predict --model model.json --report engine_report.json
//                     [--ops bench/results/shader_ops.json] [--work work.json] [--out pred.json]
//
// `model` turns a soc_bench results file (merged runs) into the cost-model
// JSON.  `predict` computes, for every timed unit of an engine report (JSON
// schema v2, `passes[]`), the roofline lower bound next to the measured p50
// GPU time.  Portable (no Apple headers): builds and runs on Linux.

#include "diagnostics/soc_model.h"
#include "diagnostics/soc_results.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using namespace phosphor;
using namespace phosphor::soc;
using json = nlohmann::json;

namespace {

bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool writeOut(const std::string& path, const std::string& text) {
    if (path.empty()) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        return true;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << text;
    return bool(f);
}

std::map<std::string, std::string> parseArgs(int argc, char** argv, int first) {
    std::map<std::string, std::string> a;
    for (int i = first; i < argc; ++i) {
        if (std::strncmp(argv[i], "--", 2) == 0 && i + 1 < argc) {
            a[argv[i] + 2] = argv[i + 1];
            ++i;
        } else {
            std::fprintf(stderr, "soc_model: unexpected argument '%s'\n", argv[i]);
            a["error"] = "1";
        }
    }
    return a;
}

int usage() {
    std::fputs("usage: soc_model model --results FILE [--out model.json]\n"
               "       soc_model predict --model model.json --report report.json [--ops shader_ops.json]\n"
               "                         [--work work.json] [--out pred.json]\n",
               stderr);
    return 2;
}

// Work counters of one unit (engine report `work` object plus the optional
// --work file entry; the file wins field by field).
struct UnitWorkCounts {
    double draws = 0, instances = 0, indices = 0, vertices = 0, pixels = 0, threads = 0, lights = 0, fragments = 0;
};

UnitWorkCounts workFrom(const json& j, UnitWorkCounts w) {
    if (!j.is_object()) return w;
    auto get = [&](const char* k, double& dst) {
        if (j.contains(k) && j[k].is_number()) dst = j[k].get<double>();
    };
    get("draws", w.draws);
    get("instances", w.instances);
    get("indices", w.indices);
    get("vertices", w.vertices);
    get("pixels", w.pixels);
    get("threads", w.threads);
    get("lights", w.lights);
    get("fragments", w.fragments);
    return w;
}

std::string kindOf(const ShaderOps& s) {
    if (!s.kind.empty()) return s.kind;
    const std::string& n = s.name;
    auto ends = [&](const char* suf) {
        const size_t l = std::strlen(suf);
        return n.size() >= l && n.compare(n.size() - l, l, suf) == 0;
    };
    if (ends("_vs")) return "vertex";
    if (ends("_fs")) return "fragment";
    return "kernel";
}

int cmdModel(const std::map<std::string, std::string>& a) {
    if (!a.count("results")) return usage();
    std::string text;
    if (!readFile(a.at("results"), text)) {
        std::fprintf(stderr, "soc_model: cannot read %s\n", a.at("results").c_str());
        return 1;
    }
    Results r;
    std::string err;
    if (!fromJson(text, r, &err)) {
        std::fprintf(stderr, "soc_model: %s: %s\n", a.at("results").c_str(), err.c_str());
        return 1;
    }
    const SocCostModel m = SocCostModel::fromResults(r);
    const std::string out = a.count("out") ? a.at("out") : "";
    if (!writeOut(out, toJson(m))) {
        std::fprintf(stderr, "soc_model: cannot write %s\n", out.c_str());
        return 1;
    }
    if (!m.hasRoofs()) std::fputs("soc_model: warning: FP32 or DRAM roof missing, predictions will be 0\n", stderr);
    return 0;
}

int cmdPredict(const std::map<std::string, std::string>& a) {
    if (!a.count("model") || !a.count("report")) return usage();
    std::string text;
    SocCostModel model;
    std::string err;
    if (!readFile(a.at("model"), text)) err = "cannot read file";
    if (!err.empty() || !fromJson(text, model, &err)) {
        std::fprintf(stderr, "soc_model: model %s: %s\n", a.at("model").c_str(), err.c_str());
        return 1;
    }
    json report;
    try {
        if (!readFile(a.at("report"), text)) throw std::runtime_error("cannot read file");
        report = json::parse(text);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "soc_model: report %s: %s\n", a.at("report").c_str(), e.what());
        return 1;
    }
    std::vector<ShaderOps> ops;
    if (a.count("ops")) {
        if (!readFile(a.at("ops"), text) || !parseShaderOps(text, ops, &err)) {
            std::fprintf(stderr, "soc_model: ops %s: %s\n", a.at("ops").c_str(), err.c_str());
            return 1;
        }
    }
    json workFile = json::object();
    if (a.count("work")) {
        try {
            if (!readFile(a.at("work"), text)) throw std::runtime_error("cannot read file");
            workFile = json::parse(text);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "soc_model: work %s: %s\n", a.at("work").c_str(), e.what());
            return 1;
        }
    }

    auto num = [](double v) { return std::isfinite(v) ? json(v) : json(nullptr); };
    json units = json::array();
    if (!report.contains("passes") || !report["passes"].is_array()) {
        std::fputs("soc_model: report has no per-unit `passes` (run with --frames and GPU timing)\n", stderr);
        return 1;
    }
    for (const json& u : report["passes"]) {
        const std::string name = u.value("name", std::string("?"));
        // Engine report v3: `work` is an array with one entry per render-graph pass
        // ({"pass","draws",...}); v2-style single object also accepted.  Entries of a
        // unit are summed (lights: max), since the unit lists its shaders without a
        // per-pass mapping.
        UnitWorkCounts wc;
        if (u.contains("work")) {
            const json& wj = u["work"];
            if (wj.is_array()) {
                for (const json& e : wj) {
                    const UnitWorkCounts x = workFrom(e, {});
                    wc.draws += x.draws; wc.instances += x.instances; wc.indices += x.indices;
                    wc.vertices += x.vertices; wc.pixels += x.pixels; wc.threads += x.threads;
                    wc.fragments += x.fragments; wc.lights = std::max(wc.lights, x.lights);
                }
            } else {
                wc = workFrom(wj, wc);
            }
        }
        if (workFile.contains(name)) wc = workFrom(workFile[name], wc);

        PassWork w;
        w.dramBytes = u.value("dram_bytes", 0.0);
        w.draws = static_cast<u32>(wc.draws);
        const double vertexInv = wc.vertices > 0 ? wc.vertices : wc.indices;
        const double fragInv = wc.fragments > 0 ? wc.fragments : wc.pixels;
        json matched = json::array(), missing = json::array();
        if (u.contains("shaders")) {
            for (const json& sj : u["shaders"]) {
                const std::string sname = sj.get<std::string>();
                const ShaderOps* s = findShader(ops, sname);
                if (!s) {
                    missing.push_back(sname);
                    continue;
                }
                const std::string kind = kindOf(*s);
                const double inv = kind == "vertex" ? vertexInv : kind == "fragment" ? fragInv : wc.threads;
                const double trips = (sname.rfind("forward_fs", 0) == 0 && wc.lights > 0) ? wc.lights : 1.0;
                const OpCounts c = opsPerInvocation(*s, trips).scaled(inv);
                w.flops += c.flops;
                w.transcendentals += c.transcendentals;
                w.divides += c.divides;
                w.intOps += c.intOps;
                w.samples += c.samples;
                matched.push_back(s->name + " (" + kind + ", x" + std::to_string(static_cast<long long>(inv)) + ")");
            }
        }
        const PassPrediction p = model.predictPass(w);
        const double measured = u.contains("gpu_ms") ? u["gpu_ms"].value("p50", 0.0) : 0.0;
        json o;
        o["name"] = name;
        o["measured_ms"] = measured;
        o["predicted_ms"] = p.ms;
        o["bound"] = p.bound;
        o["dram_ms"] = p.dramMs;
        o["onchip_ms"] = p.onchipMs;
        o["alu_ms"] = p.aluMs;
        o["arithmetic_intensity"] = num(p.arithmeticIntensity);
        o["ratio"] = p.ms > 0 && measured > 0 ? num(measured / p.ms) : json(nullptr);
        o["flops"] = w.flops;
        o["transcendentals"] = w.transcendentals;
        o["divides"] = w.divides;
        o["int_ops"] = w.intOps;
        o["samples"] = w.samples;
        o["dram_bytes"] = w.dramBytes;
        o["achieved_tflops"] = measured > 0 ? num(w.flops / (measured * 1e-3) * 1e-12) : json(nullptr);
        o["achieved_gbps"] = measured > 0 ? num(w.dramBytes / (measured * 1e-3) * 1e-9) : json(nullptr);
        o["shaders_matched"] = matched;
        o["shaders_missing"] = missing;
        units.push_back(o);
    }

    json out;
    out["schema"] = "phosphor-soc-prediction";
    out["schema_version"] = 1;
    out["chip"] = model.machine.chip;
    out["slug"] = model.machine.slug;
    out["bench"] = report.value("bench", std::string());
    out["width"] = report.value("width", 0);
    out["height"] = report.value("height", 0);
    out["roofs"] = {{"f32_tflops", num(model.f32Fma.value)}, {"f16_tflops", num(model.f16Fma.value)},
                    {"dram_gbps", num(model.dramBw.value)},  {"onchip_gbps", num(model.onchipBw.value)},
                    {"ridge_flop_per_byte", num(model.ridgePoint())},
                    {"ridge_onchip_flop_per_byte", num(model.ridgePointOnchip())}};
    out["units"] = units;
    out["notes"] = json::array({
        "Predicted time is a LOWER BOUND: max(DRAM, on-chip, ALU) with no dispatch/barrier/pass overheads; ratio = measured p50 / predicted.",
        "ALU time = flops/FP32 FMA rate + transcendentals/rate + divides/rate + int ops/fastest measured int rate (they share the ALUs); a rate missing from the model contributes 0.",
        "Texture samples, loads/stores, register spills, occupancy, ALU/memory overlap losses and rasteriser/geometry limits are NOT modelled (they only make the real time larger).",
        "DRAM bytes come from the render graph estimate (`dram_bytes` of the report), not from a hardware counter; on-chip bytes are not estimated (0), so the on-chip roof never binds here.",
        "Ops are STATIC AIR counts per lane: every block once, BOTH sides of every branch (function-constant variants included), so the ALU term can exceed what a lane really executes: it is an estimate, not a strict bound. Loops with unknown trip count count once (forward_fs uses work.lights as trip count).",
        "Only outermost loops are scaled by the trip count; nested loops stay inside the outer body (inner trip counts are not modelled).",
        "Report v3 `work` entries of a unit are summed over its passes (lights = max) and applied to every shader of the unit by stage: a shader used by only some passes is over-counted.",
        "Invocations: vertex = work.vertices (else work.indices, i.e. no post-transform cache reuse and no instancing multiplier); fragment = work.fragments (else work.pixels: overdraw, early-z and helper lanes ignored); compute = work.threads.",
        "FLOP counting as in B-01: fma = 2, add/mul = 1; the roof is the measured independent-chain FP32 FMA rate at the sustained P-state of the characterisation run.",
        "Shaders not found in the ops file are listed in shaders_missing and contribute no work (lower bound stays a lower bound).",
    });
    const std::string outPath = a.count("out") ? a.at("out") : "";
    if (!writeOut(outPath, out.dump(1) + "\n")) {
        std::fprintf(stderr, "soc_model: cannot write %s\n", outPath.c_str());
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    const auto a = parseArgs(argc, argv, 2);
    if (a.count("error")) return usage();
    if (cmd == "model") return cmdModel(a);
    if (cmd == "predict") return cmdPredict(a);
    return usage();
}
