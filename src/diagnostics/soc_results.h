#pragma once

#include "core/types.h"

#include <map>
#include <string>
#include <vector>

namespace phosphor::soc {

// ---------------------------------------------------------------------------
// SoC characterisation results (OPT-0.1) -- portable schema of the JSON
// written by bench/soc (soc_bench) to bench/results/<chip>-<os>.json and read
// by the cost model (soc_model.h) and the tools.
//
// A benchmark (B-xx) holds named metrics.  A metric name is unique inside its
// benchmark and already encodes its parameters (e.g. "stream_bw.ws_64MiB");
// `params` repeats them as numbers for tools.  Within one run a metric keeps
// the statistics of its repetitions (`within`, value = within.median); across
// runs mergeRuns() keeps every run value, value = median of the run values
// and runCv = CV between runs.
// ---------------------------------------------------------------------------

inline constexpr u32 kResultsSchemaVersion = 1;

struct Stats {
    double median = 0, min = 0, max = 0, p10 = 0, p90 = 0, mean = 0;
    double cv = 0; // sample standard deviation / mean (0 with < 2 samples)
    u32    n  = 0;
};

/// Nearest-rank quantiles, mean and CV of `v` (any order; empty -> zeros).
[[nodiscard]] Stats computeStats(std::vector<double> v);

struct Metric {
    std::string name;
    std::string unit;          // "TFLOPS", "GB/s", "ns", "ms", "us", "count", "ratio", "W", "MHz", ...
    double      value = 0;     // one run: within.median; merged: median of `runs`
    Stats       within;        // repetitions inside one run
    std::vector<double> runs;  // one value per run (merged results)
    double      runCv = 0;     // CV between runs (0 with one run)
    bool        higherIsBetter = true;
    std::map<std::string, double> params;
};

enum class Status { Ok, Partial, Unsupported, Failed };
enum class Control { Pass, Fail, NotApplicable };

[[nodiscard]] const char* statusName(Status s);
[[nodiscard]] const char* controlName(Control c);
[[nodiscard]] Status statusFromString(const std::string& s);
[[nodiscard]] Control controlFromString(const std::string& s);

/// GPU clock state over the measured part of a benchmark (IOReport).
struct GpuWindow {
    double topStateShare = -1; // share of busy time in the highest P-state
    double meanMHz       = -1; // residency-weighted frequency of busy time
    double activeShare   = -1; // busy time / wall time
    double watts         = -1; // GPU energy / wall time
};

struct Benchmark {
    std::string id;     // "B-01"
    std::string name;   // "alu.throughput"
    std::string title;  // one line
    Status      status = Status::Ok;
    std::string notes;  // why partial/unsupported, fallbacks used, caveats
    Control     negative = Control::NotApplicable;
    std::string negativeDetail;
    GpuWindow   gpu;
    double      seconds = 0; // wall time of the benchmark (one run)
    std::vector<Metric> metrics;

    [[nodiscard]] const Metric* find(const std::string& metric) const;
};

struct Machine {
    std::string chip;           // "Apple M5 Max"
    std::string slug;           // "m5max"
    std::string gpuFamily;      // "Apple10"
    u32         gpuCores = 0;
    u32         cpuPerformanceCores = 0; // perflevel0
    u32         cpuEfficiencyCores  = 0; // perflevel1 (named "Performance" on M5 Pro/Max: see cpuLevelNames)
    std::vector<std::string> cpuLevelNames;
    u64         memoryBytes = 0;
    std::string os;             // "macOS 27.2"
    std::string osBuild;        // "26B5091g"
    std::string osSlug;         // "macos27.2"
    std::string sdk;            // "27.0"
    std::string powerSource;    // "AC" / "Battery"
    i32         batteryPercent = -1;
    std::string thermalStart;   // "nominal" / "fair" / "serious" / "critical"
    std::string thermalEnd;
    std::vector<double> gpuPStateMHz; // P1..Pn from the pmgr table (ioreg)
};

struct RunInfo {
    std::string commit;
    std::string date;           // ISO 8601
    u32         runs = 1;
    bool        quick = false;
    bool        forcedApple9 = false;
    std::string args;
};

struct Results {
    u32     schemaVersion = kResultsSchemaVersion;
    Machine machine;
    RunInfo run;
    std::vector<Benchmark> benchmarks;

    [[nodiscard]] const Benchmark* find(const std::string& id) const;
    [[nodiscard]] const Metric* find(const std::string& id, const std::string& metric) const;
};

[[nodiscard]] std::string toJson(const Results& r);
/// false (and *error) on malformed JSON, a missing field or another schema.
[[nodiscard]] bool fromJson(const std::string& text, Results& out, std::string* error = nullptr);

/// Merge N runs of the same suite: machine/run info from the first (thermal
/// end from the last), benchmarks and metrics matched by id and name.  Per
/// metric: runs = the run values, value = their median, runCv = their CV,
/// within = the within-run stats of the median run.  Benchmark status = the
/// worst, negative control = Fail if any run failed, notes/details joined
/// when they differ.
[[nodiscard]] Results mergeRuns(const std::vector<Results>& runs);

/// "Apple M5 Max" -> "m5max", "macOS 27.2" -> "macos27.2".
[[nodiscard]] std::string slugify(const std::string& s);

} // namespace phosphor::soc
