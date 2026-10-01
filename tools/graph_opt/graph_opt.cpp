// graph_opt -- OPT-1.1 offline graph optimiser front end (portable: no Apple
// headers, builds and runs on Linux).
//
//   graph_opt [--scenario N|all] [--views V[,V..]] [--size WxH] [--work F] [--wide]
//             [--model bench/results/m5max-macos27.2-model.json] [--cap N]
//             [--iterations N] [--seed N] [--out plans.json] [--merge]
//             [--report table.md] [--top K --top-dir DIR] [--gamma G] [--beta B]
//
// For every (scenario, views) pair it builds the scenario graph (the engine's
// Drawable import included) under each combination of build choices (remat,
// async), runs rg::optimize() and collects the best plan of the family
// (rg::scenarioFamily).  The plans go to --out (JSON, the format of
// --graph-plan; --merge replaces the plans of the same families in an
// existing file and keeps the others).  A markdown table per family (baseline
// "off" vs plan: DRAM, heap, max live, predicted frame ms, deltas, method,
// build choices, policies, then the candidates) goes to stdout and --report.
//
// --model takes the cost-model JSON of `soc_model model` (not the raw
// soc_bench results); without it the built-in M5 Max parameters are used.
// Defaults: all scenarios, 1 view, 2560x1440, work 1.  --cap sets the DP
// state cap, --iterations the annealing iterations; --gamma / --beta weigh the
// transient heap and the DRAM bytes in J (ms per GiB; memory-oriented plans).  --top K writes
// DIR/plans_<i>.json (i < K+1): the i-th best distinct candidate of every
// family, the baseline always among them (rg::topPlans) -- the plans that
// tools/graph_select.py measures on the device to adopt the best.  Exit code: 0 ok,
// 1 error (optimiser failure, I/O), 2 usage.

#include "diagnostics/soc_model.h"
#include "rendergraph/optimizer/optimizer.h"
#include "rendergraph/scenario.h"

#include "graph_opt/plan_merge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace phosphor;

namespace {

int usage() {
    std::fputs(
        "usage: graph_opt [--scenario N|all] [--views V[,V..]] [--size WxH] [--work F] [--wide]\n"
        "                 [--model model.json] [--cap N] [--iterations N] [--seed N]\n"
        "                 [--out plans.json] [--merge] [--report table.md] [--top K --top-dir DIR]\n"
        "                 [--gamma G] [--beta B]\n",
        stderr);
    return 2;
}

bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool writeFile(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << text;
    return bool(f);
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::string join(const std::vector<std::string>& v) {
    if (v.empty()) return "-";
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
    return s;
}

double mib(double bytes) { return bytes / (1024.0 * 1024.0); }

std::string delta(double base, double now) {
    if (base == 0.0) return now == 0.0 ? "0.0%" : "n/a";
    char b[32];
    std::snprintf(b, sizeof b, "%+.1f%%", (now - base) / base * 100.0);
    return b;
}

std::string fmt(const char* f, double v) {
    char b[48];
    std::snprintf(b, sizeof b, f, v);
    return b;
}

struct Family {
    std::string             name;
    u32                     scenario = 0;
    u32                     views    = 1;
    rg::OptimizeResult      result;
};

void appendFamily(std::string& md, const Family& f) {
    const rg::OptimizeResult& r = f.result;
    md += "### " + f.name + " (scenario " + std::to_string(f.scenario) + " " + rg::scenarioName(f.scenario) +
          ", views " + std::to_string(f.views) + ")\n\n";
    if (!r.ok) {
        md += "FAILED: " + r.error + "\n\n";
        return;
    }
    const rg::GraphCost& b = r.baseline;
    const rg::PlanMetrics& p = r.plan.predicted;
    md += "| | baseline (off) | plan | delta |\n|---|---|---|---|\n";
    auto row = [&](const char* label, double base, double now, const char* f2) {
        md += std::string("| ") + label + " | " + fmt(f2, base) + " | " + fmt(f2, now) + " | " + delta(base, now) +
              " |\n";
    };
    row("DRAM MiB", mib(b.dramBytes), mib(p.dramBytes), "%.2f");
    row("heap MiB", mib(b.heapBytes), mib(p.heapBytes), "%.2f");
    row("max live MiB", mib(b.maxLiveBytes), mib(p.maxLive), "%.2f");
    row("predicted frame ms", b.frameMs, p.timeMs, "%.4f");
    md += "\nmethod: " + r.plan.method + "; remat: " + join(r.plan.remat) + "; async: " + join(r.plan.async) +
          "; policies: alias=" + rg::aliasPolicyName(r.plan.aliasPolicy) +
          " barriers=" + rg::barrierPolicyName(r.plan.barrierPolicy) + "; key: " + [&] {
              char k[24];
              std::snprintf(k, sizeof k, "%016llx", static_cast<unsigned long long>(r.plan.key));
              return std::string(k);
          }() + "; planned passes: " + std::to_string(r.plan.order.size()) + "\n\n";

    md += "Candidates (" + std::to_string(r.candidates.size()) + "):\n\n";
    md += "| method | remat | async | alias | barriers | frame ms | DRAM MiB | heap MiB | J |\n"
          "|---|---|---|---|---|---|---|---|---|\n";
    for (const rg::PlanCandidate& c : r.candidates) {
        md += "| " + c.method + " | " + join(c.choices.remat) + " | " + join(c.choices.async) + " | " +
              rg::aliasPolicyName(c.aliasPolicy) + " | " + rg::barrierPolicyName(c.barrierPolicy) + " | " +
              fmt("%.4f", c.cost.frameMs) + " | " + fmt("%.2f", mib(c.cost.dramBytes)) + " | " +
              fmt("%.2f", mib(c.cost.heapBytes)) + " | " + fmt("%.4f", c.cost.J) + " |\n";
    }
    md += "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string scenarioArg = "all", viewsArg = "1", sizeArg, modelPath, outPath, reportPath, topDir;
    long        top = 0;
    double      gamma = -1, beta = -1;
    float       work = 1.0f;
    bool        wide = false, merge = false;
    long        cap = -1, iterations = -1, seed = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--scenario") { if (!next(scenarioArg)) return usage(); }
        else if (a == "--views") { if (!next(viewsArg)) return usage(); }
        else if (a == "--size") { if (!next(sizeArg)) return usage(); }
        else if (a == "--work") { if (!next(v)) return usage(); work = static_cast<float>(std::atof(v.c_str())); }
        else if (a == "--wide") wide = true;
        else if (a == "--merge") merge = true;
        else if (a == "--model") { if (!next(modelPath)) return usage(); }
        else if (a == "--cap") { if (!next(v)) return usage(); cap = std::atol(v.c_str()); }
        else if (a == "--iterations") { if (!next(v)) return usage(); iterations = std::atol(v.c_str()); }
        else if (a == "--seed") { if (!next(v)) return usage(); seed = std::atol(v.c_str()); }
        else if (a == "--out") { if (!next(outPath)) return usage(); }
        else if (a == "--report") { if (!next(reportPath)) return usage(); }
        else if (a == "--top") { if (!next(v)) return usage(); top = std::atol(v.c_str()); }
        else if (a == "--top-dir") { if (!next(topDir)) return usage(); }
        else if (a == "--gamma") { if (!next(v)) return usage(); gamma = std::atof(v.c_str()); }
        else if (a == "--beta") { if (!next(v)) return usage(); beta = std::atof(v.c_str()); }
        else return usage();
    }

    std::vector<u32> scenarios;
    if (scenarioArg == "all") {
        for (u32 s = 0; s < rg::scenarioCount(); ++s) scenarios.push_back(s);
    } else {
        for (const std::string& t : split(scenarioArg, ',')) {
            const long s = std::atol(t.c_str());
            if (s < 0 || s >= static_cast<long>(rg::scenarioCount())) {
                std::fprintf(stderr, "graph_opt: unknown scenario %s\n", t.c_str());
                return 2;
            }
            scenarios.push_back(static_cast<u32>(s));
        }
    }
    std::vector<u32> views;
    for (const std::string& t : split(viewsArg, ',')) {
        const long n = std::atol(t.c_str());
        if (n < 1 || n > 6) {
            std::fprintf(stderr, "graph_opt: views must be 1..6 (got %s)\n", t.c_str());
            return 2;
        }
        views.push_back(static_cast<u32>(n));
    }
    if (scenarios.empty() || views.empty()) return usage();
    u32 width = 2560, height = 1440;
    if (!sizeArg.empty() && std::sscanf(sizeArg.c_str(), "%ux%u", &width, &height) != 2) {
        std::fprintf(stderr, "graph_opt: bad --size %s (WxH)\n", sizeArg.c_str());
        return 2;
    }

    rg::GraphCostParams cost;
    if (!modelPath.empty()) {
        std::string text, err;
        soc::SocCostModel model;
        if (!readFile(modelPath, text)) err = "cannot read file";
        if (!err.empty() || !soc::fromJson(text, model, &err)) {
            std::fprintf(stderr, "graph_opt: model %s: %s\n", modelPath.c_str(), err.c_str());
            return 1;
        }
        cost = rg::GraphCostParams::fromSoc(model);
    }
    if (gamma >= 0) cost.gammaMsPerGiB = gamma;
    if (beta >= 0) cost.betaMsPerGiB = beta;

    std::vector<Family> families;
    for (u32 s : scenarios) {
        for (u32 v : views) {
            rg::ScenarioParams params;
            params.width   = width;
            params.height  = height;
            params.work    = work;
            params.wideHdr = wide;
            params.views   = v;

            Family fam;
            fam.name     = rg::scenarioFamily(s, params);
            fam.scenario = s;
            fam.views    = v;

            const rg::GraphBuilder builder = [&](const rg::BuildChoices& choices, rg::RenderGraph& graph,
                                                 std::string* error) {
                rg::ScenarioParams p = params;
                p.remat              = choices.remat;
                p.async              = choices.async; // explicit: [] = everything on graphics
                const rg::TextureRef drawable =
                    graph.importTexture("Drawable", {rg::Format::BGRA8Srgb, 3200, 1800}, rg::ImportOutput | rg::ImportPerFrame);
                rg::Scenario out;
                return rg::buildScenario(s, p, graph, drawable, out, {}, error);
            };

            rg::OptimizeOptions opt;
            opt.rematCandidates         = rg::scenarioRematCandidates(s);
            opt.asyncCandidates         = rg::scenarioAsyncCandidates(s);
            opt.baselineChoices.async   = rg::scenarioAsyncCandidates(s);
            opt.cost                    = cost;
            if (cap >= 0) opt.dpStateCap = static_cast<size_t>(cap);
            if (iterations >= 0) opt.annealIterations = static_cast<u32>(iterations);
            if (seed >= 0) opt.seed = static_cast<u32>(seed);

            fam.result = rg::optimize(fam.name, builder, opt);
            families.push_back(std::move(fam));
        }
    }

    std::string md = "## graph_opt: " + std::to_string(families.size()) + " famil" +
                     (families.size() == 1 ? "y" : "ies") + "\n\n";
    bool failed = false;
    std::vector<rg::GraphPlan> plans;
    for (const Family& f : families) {
        appendFamily(md, f);
        if (f.result.ok) plans.push_back(f.result.plan);
        else {
            failed = true;
            std::fprintf(stderr, "graph_opt: %s: %s\n", f.name.c_str(), f.result.error.c_str());
        }
    }
    std::fputs(md.c_str(), stdout);

    if (!reportPath.empty() && !writeFile(reportPath, md)) {
        std::fprintf(stderr, "graph_opt: cannot write %s\n", reportPath.c_str());
        return 1;
    }
    if (!outPath.empty()) {
        std::vector<rg::GraphPlan> result = plans;
        if (merge) {
            std::string text;
            if (readFile(outPath, text)) {
                std::vector<rg::GraphPlan> existing;
                std::string err;
                if (!rg::fromJson(text, existing, &err)) {
                    std::fprintf(stderr, "graph_opt: cannot merge into %s: %s\n", outPath.c_str(), err.c_str());
                    return 1;
                }
                result = rg::mergePlans(existing, plans);
            }
        }
        if (!writeFile(outPath, rg::toJson(result))) {
            std::fprintf(stderr, "graph_opt: cannot write %s\n", outPath.c_str());
            return 1;
        }
        std::fprintf(stderr, "graph_opt: wrote %zu plan(s) to %s\n", result.size(), outPath.c_str());
    }
    if (top > 0) {
        if (topDir.empty()) return usage();
        std::vector<std::vector<rg::GraphPlan>> files;
        for (const Family& f : families) {
            if (!f.result.ok) continue;
            const std::vector<rg::GraphPlan> best = rg::topPlans(f.result, static_cast<size_t>(top));
            for (size_t i = 0; i < best.size(); ++i) {
                if (files.size() <= i) files.resize(i + 1);
                files[i].push_back(best[i]);
            }
        }
        for (size_t i = 0; i < files.size(); ++i) {
            const std::string path = topDir + "/plans_" + std::to_string(i) + ".json";
            if (!writeFile(path, rg::toJson(files[i]))) {
                std::fprintf(stderr, "graph_opt: cannot write %s\n", path.c_str());
                return 1;
            }
            for (const rg::GraphPlan& p : files[i]) {
                std::fprintf(stderr, "graph_opt: %s [%zu] %s remat=%s async=%s alias=%s barriers=%s predicted %.4f ms\n",
                             path.c_str(), i, p.family.c_str(), join(p.remat).c_str(), join(p.async).c_str(),
                             rg::aliasPolicyName(p.aliasPolicy), rg::barrierPolicyName(p.barrierPolicy),
                             p.predicted.timeMs);
            }
        }
    }
    return failed ? 1 : 0;
}
