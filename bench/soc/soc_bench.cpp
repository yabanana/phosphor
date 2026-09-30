// OPT-0.1 SoC characterisation suite: command line (see harness.h and
// bench/soc/README.md).
//
//   soc_bench [--list] [--only B-01,B-08] [--quick] [--runs N] [--out FILE]
//             [--validate] [--force-family apple9] [--window] [--soak MIN]
//
// Exit status: 0 = every benchmark ran and every negative control passed;
// 1 = a benchmark failed or a negative control failed; 2 = usage error;
// 3 = --validate found messages (--validate: 1 = a benchmark failed; timing
// controls are listed but not enforced under the validation layers).

#include "harness.h"

#include "core/process.h"

#include <mach-o/dyld.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>

#ifndef SOC_SOURCE_DIR
#define SOC_SOURCE_DIR "."
#endif

namespace {

using namespace soc;

struct Cli {
    Options opts;
    bool list = false;
    bool validate = false;
    u32 runs = 3;
    std::string out;
    std::set<std::string> only;
    std::string args;
};

[[noreturn]] void usage(const std::string& why) {
    std::fprintf(stderr,
                 "soc_bench: %s\n"
                 "usage: soc_bench [--list] [--only B-01,B-08] [--quick] [--runs N] [--out FILE]\n"
                 "                 [--validate] [--force-family apple9] [--window] [--soak MIN]\n",
                 why.c_str());
    std::exit(2);
}

Cli parse(int argc, char** argv) {
    Cli c;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        c.args += (i > 1 ? " " : "") + a;
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(a + " needs a value");
            c.args += std::string(" ") + argv[i + 1];
            return argv[++i];
        };
        if (a == "--list") c.list = true;
        else if (a == "--quick") c.opts.quick = true;
        else if (a == "--validate") c.validate = true;
        else if (a == "--window") c.opts.window = true;
        else if (a == "--runs") c.runs = static_cast<u32>(std::max(1, std::atoi(next().c_str())));
        else if (a == "--out") c.out = next();
        else if (a == "--soak") c.opts.soakMinutes = std::atof(next().c_str());
        else if (a == "--force-family") {
            const std::string f = next();
            if (f != "apple9") usage("--force-family accepts only apple9");
            c.opts.forceApple9 = true;
        } else if (a == "--only") {
            std::stringstream ss(next());
            std::string id;
            while (std::getline(ss, id, ',')) c.only.insert(id);
        } else usage("unknown option " + a);
    }
    if (c.opts.quick) c.opts.repetitions = 5;
    return c;
}

std::string selfPath() {
    char buf[4096];
    uint32_t n = sizeof(buf);
    return _NSGetExecutablePath(buf, &n) == 0 ? buf : "soc_bench";
}

// --validate: run the quick suite again under the API + shader validation
// layers and count every output line that is not ours.
int validate(const Cli& c) {
    setenv("MTL_DEBUG_LAYER", "1", 1);
    setenv("MTL_SHADER_VALIDATION", "1", 1);
    setenv("MTL_DEBUG_LAYER_WARNING_MODE", "nslog", 1);
    const std::string json = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/soc_bench_validate_" +
                             std::to_string(getpid()) + ".json";
    std::vector<std::string> argv = {selfPath(), "--quick", "--runs", "1", "--out", json};
    if (!c.only.empty()) {
        std::string ids;
        for (const auto& id : c.only) ids += (ids.empty() ? "" : ",") + id;
        argv.push_back("--only");
        argv.push_back(ids);
    }
    if (c.opts.forceApple9) { argv.push_back("--force-family"); argv.push_back("apple9"); }
    if (c.opts.window) argv.push_back("--window");
    const phosphor::ProcessResult r = phosphor::runProcess(argv);
    std::istringstream in(r.output);
    std::string line;
    u32 messages = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line.rfind("[soc]", 0) == 0 || line.find("Validation Enabled") != std::string::npos) continue;
        ++messages;
        std::fprintf(stderr, "[soc] validation: %s\n", line.c_str());
    }
    // Pass = no validation message and no benchmark failed (wrong results,
    // aborts).  Timing controls are not meaningful under the validation
    // layers (they slow the GPU and the CPU unevenly): listed, not enforced.
    u32 failed = 0;
    std::ifstream f(json);
    std::stringstream text;
    text << f.rdbuf();
    phosphor::soc::Results res;
    std::string err;
    if (!phosphor::soc::fromJson(text.str(), res, &err)) {
        std::fprintf(stderr, "[soc] --validate: cannot read the child's results (%s), exit %d\n", err.c_str(), r.exitCode);
        return r.exitCode ? r.exitCode : 1;
    }
    std::remove(json.c_str());
    for (const auto& b : res.benchmarks) {
        if (b.status == Status::Failed) {
            ++failed;
            std::fprintf(stderr, "[soc] --validate: %s FAILED: %s\n", b.id.c_str(), b.notes.c_str());
        } else if (b.negative == Control::Fail) {
            std::fprintf(stderr, "[soc] --validate: %s timing control not enforced under validation: %s\n", b.id.c_str(),
                         b.negativeDetail.c_str());
        }
    }
    std::fprintf(stderr, "[soc] --validate: %zu benchmark(s), %u failed, %u validation message(s)\n",
                 res.benchmarks.size(), failed, messages);
    if (messages) return 3;
    return failed ? 1 : 0;
}

std::string gitCommit() {
    const std::string dir = SOC_SOURCE_DIR;
    phosphor::ProcessResult h = phosphor::runProcess({"git", "-C", dir, "rev-parse", "--short", "HEAD"});
    if (h.exitCode != 0) return "unknown";
    std::string commit = h.output.substr(0, h.output.find('\n'));
    phosphor::ProcessResult s = phosphor::runProcess({"git", "-C", dir, "status", "--porcelain", "--untracked-files=no"});
    if (s.exitCode == 0 && !s.output.empty()) commit += "-dirty";
    return commit;
}

std::string isoNow() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", std::localtime(&t));
    return buf;
}

phosphor::soc::Results runOnce(Context& ctx, const Cli& c, const std::vector<BenchInfo>& selected, u32 run) {
    phosphor::soc::Results r;
    const double w = ctx.warmUp();
    ctx.log("run %u: warm-up %.2f s%s", run + 1, std::fabs(w), w < 0 ? " (top P-state NOT reached)" : "");
    for (const BenchInfo& info : selected) {
        phosphor::soc::Benchmark b;
        b.id    = info.id;
        b.name  = info.name;
        b.title = info.title;
        Report rep(b);
        const double t0 = nowMs();
        ctx.beginBenchmark();
        ctx.gpuState().begin();
        try {
            info.fn(ctx, rep);
        } catch (const std::exception& e) {
            b.status = Status::Failed;
            rep.note(std::string("aborted: ") + e.what());
        }
        b.gpu = ctx.gpuState().end();
        ctx.endBenchmark();
        b.seconds = (nowMs() - t0) * 1e-3;
        ctx.log("%s %-22s %-11s neg=%-4s %6.1f s  %zu metrics  P-top %.0f%% %s%s", b.id.c_str(), b.name.c_str(),
                phosphor::soc::statusName(b.status), phosphor::soc::controlName(b.negative), b.seconds,
                b.metrics.size(), b.gpu.topStateShare * 100, b.notes.empty() ? "" : "| ", b.notes.c_str());
        if (b.negative == Control::Fail) ctx.log("   negative control FAILED: %s", b.negativeDetail.c_str());
        r.benchmarks.push_back(std::move(b));
    }
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const Cli c = parse(argc, argv);
    if (c.list) {
        for (const BenchInfo& b : registry()) std::printf("%s  %-24s %s\n", b.id, b.name, b.title);
        return 0;
    }
    if (c.validate) return validate(c);

    std::vector<BenchInfo> selected;
    for (const BenchInfo& b : registry())
        if (c.only.empty() || c.only.count(b.id)) selected.push_back(b);
    for (const auto& id : c.only) {
        bool known = false;
        for (const BenchInfo& b : registry()) known |= id == b.id;
        if (!known) usage("unknown benchmark " + id);
    }

    int exitCode = 0;
    try {
        Context ctx(c.opts);
        phosphor::soc::Machine machine = describeMachine(ctx);
        ctx.log("%s (%s, %u GPU cores), %s %s, power %s %d%%, thermal %s, IOReport %s, %zu benchmark(s) x %u run(s)%s",
                machine.chip.c_str(), machine.gpuFamily.c_str(), machine.gpuCores, machine.os.c_str(),
                machine.osBuild.c_str(), machine.powerSource.c_str(), machine.batteryPercent, machine.thermalStart.c_str(),
                ctx.gpuState().available() ? "ok" : "unavailable", selected.size(), c.runs, c.opts.quick ? " (quick)" : "");
        std::vector<phosphor::soc::Results> runs;
        for (u32 i = 0; i < c.runs; ++i) runs.push_back(runOnce(ctx, c, selected, i));
        phosphor::soc::Results merged = phosphor::soc::mergeRuns(runs);
        merged.machine = machine;
        merged.machine.thermalEnd = thermalStateName();
        merged.run.commit       = gitCommit();
        merged.run.date         = isoNow();
        merged.run.runs         = c.runs;
        merged.run.quick        = c.opts.quick;
        merged.run.forcedApple9 = c.opts.forceApple9;
        merged.run.args         = c.args;

        std::string out = c.out;
        if (out.empty()) {
            out = std::string(SOC_SOURCE_DIR) + "/bench/results/" + machine.slug + "-" + machine.osSlug +
                  (c.opts.quick ? "-quick" : "") + (c.opts.forceApple9 ? "-apple9" : "") + ".json";
        }
        std::ofstream f(out);
        f << phosphor::soc::toJson(merged);
        if (!f) {
            ctx.log("cannot write %s", out.c_str());
            exitCode = 1;
        } else {
            ctx.log("results: %s", out.c_str());
        }
        for (const auto& b : merged.benchmarks) {
            if (b.status == Status::Failed || b.negative == Control::Fail) exitCode = 1;
            for (const auto& m : b.metrics)
                if (c.runs > 1 && m.runCv > 0.02)
                    ctx.log("   %s %s: CV between runs %.1f%%", b.id.c_str(), m.name.c_str(), m.runCv * 100);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[soc] fatal: %s\n", e.what());
        return 1;
    }
    return exitCode;
}
