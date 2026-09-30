#include "diagnostics/soc_results.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <cctype>
#include <cmath>

namespace phosphor::soc {

using json = nlohmann::json;

Stats computeStats(std::vector<double> v) {
    Stats s;
    s.n = static_cast<u32>(v.size());
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const auto rank = [&](double q) {
        const size_t i = static_cast<size_t>(std::ceil(q * static_cast<double>(v.size()))) ;
        return v[std::min(v.size() - 1, i == 0 ? 0 : i - 1)];
    };
    s.min    = v.front();
    s.max    = v.back();
    s.median = rank(0.5);
    s.p10    = rank(0.1);
    s.p90    = rank(0.9);
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    if (v.size() > 1 && s.mean != 0) {
        double sq = 0;
        for (double x : v) sq += (x - s.mean) * (x - s.mean);
        s.cv = std::sqrt(sq / static_cast<double>(v.size() - 1)) / std::fabs(s.mean);
    }
    return s;
}

const char* statusName(Status s) {
    switch (s) {
    case Status::Ok:          return "ok";
    case Status::Partial:     return "partial";
    case Status::Unsupported: return "unsupported";
    case Status::Failed:      return "failed";
    }
    return "failed";
}
const char* controlName(Control c) {
    switch (c) {
    case Control::Pass:          return "pass";
    case Control::Fail:          return "fail";
    case Control::NotApplicable: return "n/a";
    }
    return "n/a";
}
Status statusFromString(const std::string& s) {
    if (s == "ok") return Status::Ok;
    if (s == "partial") return Status::Partial;
    if (s == "unsupported") return Status::Unsupported;
    return Status::Failed;
}
Control controlFromString(const std::string& s) {
    if (s == "pass") return Control::Pass;
    if (s == "fail") return Control::Fail;
    return Control::NotApplicable;
}

const Metric* Benchmark::find(const std::string& metric) const {
    for (const Metric& m : metrics)
        if (m.name == metric) return &m;
    return nullptr;
}
const Benchmark* Results::find(const std::string& id) const {
    for (const Benchmark& b : benchmarks)
        if (b.id == id) return &b;
    return nullptr;
}
const Metric* Results::find(const std::string& id, const std::string& metric) const {
    const Benchmark* b = find(id);
    return b ? b->find(metric) : nullptr;
}

namespace {

json statsJson(const Stats& s) {
    return {{"median", s.median}, {"min", s.min}, {"max", s.max}, {"p10", s.p10}, {"p90", s.p90},
            {"mean", s.mean}, {"cv", s.cv}, {"n", s.n}};
}
Stats statsFrom(const json& j) {
    Stats s;
    s.median = j.at("median").get<double>();
    s.min    = j.at("min").get<double>();
    s.max    = j.at("max").get<double>();
    s.p10    = j.at("p10").get<double>();
    s.p90    = j.at("p90").get<double>();
    s.mean   = j.at("mean").get<double>();
    s.cv     = j.at("cv").get<double>();
    s.n      = j.at("n").get<u32>();
    return s;
}

} // namespace

std::string toJson(const Results& r) {
    json j;
    j["schema"]         = "phosphor-soc-results";
    j["schema_version"] = r.schemaVersion;
    const Machine& m = r.machine;
    j["machine"] = {{"chip", m.chip},
                    {"slug", m.slug},
                    {"gpu_family", m.gpuFamily},
                    {"gpu_cores", m.gpuCores},
                    {"cpu_perflevel0_cores", m.cpuPerformanceCores},
                    {"cpu_perflevel1_cores", m.cpuEfficiencyCores},
                    {"cpu_level_names", m.cpuLevelNames},
                    {"memory_bytes", m.memoryBytes},
                    {"os", m.os},
                    {"os_build", m.osBuild},
                    {"os_slug", m.osSlug},
                    {"sdk", m.sdk},
                    {"power_source", m.powerSource},
                    {"battery_percent", m.batteryPercent},
                    {"thermal_start", m.thermalStart},
                    {"thermal_end", m.thermalEnd},
                    {"gpu_pstate_mhz", m.gpuPStateMHz}};
    j["run"] = {{"commit", r.run.commit}, {"date", r.run.date},   {"runs", r.run.runs},
                {"quick", r.run.quick},   {"forced_apple9", r.run.forcedApple9}, {"args", r.run.args}};
    json benches = json::array();
    for (const Benchmark& b : r.benchmarks) {
        json metrics = json::array();
        for (const Metric& mt : b.metrics) {
            json params = json::object();
            for (const auto& [k, v] : mt.params) params[k] = v;
            metrics.push_back({{"name", mt.name},
                               {"unit", mt.unit},
                               {"value", mt.value},
                               {"higher_is_better", mt.higherIsBetter},
                               {"within", statsJson(mt.within)},
                               {"runs", mt.runs},
                               {"run_cv", mt.runCv},
                               {"params", params}});
        }
        benches.push_back({{"id", b.id},
                           {"name", b.name},
                           {"title", b.title},
                           {"status", statusName(b.status)},
                           {"notes", b.notes},
                           {"negative_control", {{"status", controlName(b.negative)}, {"detail", b.negativeDetail}}},
                           {"gpu", {{"top_state_share", b.gpu.topStateShare},
                                    {"mean_mhz", b.gpu.meanMHz},
                                    {"active_share", b.gpu.activeShare},
                                    {"watts", b.gpu.watts}}},
                           {"seconds", b.seconds},
                           {"metrics", metrics}});
    }
    j["benchmarks"] = benches;
    return j.dump(1) + "\n";
}

bool fromJson(const std::string& text, Results& out, std::string* error) {
    try {
        const json j = json::parse(text);
        if (j.value("schema", "") != "phosphor-soc-results") throw std::runtime_error("not a phosphor-soc-results file");
        Results r;
        r.schemaVersion = j.at("schema_version").get<u32>();
        if (r.schemaVersion != kResultsSchemaVersion)
            throw std::runtime_error("schema_version " + std::to_string(r.schemaVersion) + " (expected " +
                                     std::to_string(kResultsSchemaVersion) + ")");
        const json& m = j.at("machine");
        r.machine.chip                = m.at("chip").get<std::string>();
        r.machine.slug                = m.at("slug").get<std::string>();
        r.machine.gpuFamily           = m.at("gpu_family").get<std::string>();
        r.machine.gpuCores            = m.at("gpu_cores").get<u32>();
        r.machine.cpuPerformanceCores = m.at("cpu_perflevel0_cores").get<u32>();
        r.machine.cpuEfficiencyCores  = m.at("cpu_perflevel1_cores").get<u32>();
        r.machine.cpuLevelNames       = m.at("cpu_level_names").get<std::vector<std::string>>();
        r.machine.memoryBytes         = m.at("memory_bytes").get<u64>();
        r.machine.os                  = m.at("os").get<std::string>();
        r.machine.osBuild             = m.at("os_build").get<std::string>();
        r.machine.osSlug              = m.at("os_slug").get<std::string>();
        r.machine.sdk                 = m.at("sdk").get<std::string>();
        r.machine.powerSource         = m.at("power_source").get<std::string>();
        r.machine.batteryPercent      = m.at("battery_percent").get<i32>();
        r.machine.thermalStart        = m.at("thermal_start").get<std::string>();
        r.machine.thermalEnd          = m.at("thermal_end").get<std::string>();
        r.machine.gpuPStateMHz        = m.at("gpu_pstate_mhz").get<std::vector<double>>();
        const json& run = j.at("run");
        r.run.commit       = run.at("commit").get<std::string>();
        r.run.date         = run.at("date").get<std::string>();
        r.run.runs         = run.at("runs").get<u32>();
        r.run.quick        = run.at("quick").get<bool>();
        r.run.forcedApple9 = run.at("forced_apple9").get<bool>();
        r.run.args         = run.at("args").get<std::string>();
        for (const json& jb : j.at("benchmarks")) {
            Benchmark b;
            b.id             = jb.at("id").get<std::string>();
            b.name           = jb.at("name").get<std::string>();
            b.title          = jb.at("title").get<std::string>();
            b.status         = statusFromString(jb.at("status").get<std::string>());
            b.notes          = jb.at("notes").get<std::string>();
            b.negative       = controlFromString(jb.at("negative_control").at("status").get<std::string>());
            b.negativeDetail = jb.at("negative_control").at("detail").get<std::string>();
            const json& g = jb.at("gpu");
            b.gpu.topStateShare = g.at("top_state_share").get<double>();
            b.gpu.meanMHz       = g.at("mean_mhz").get<double>();
            b.gpu.activeShare   = g.at("active_share").get<double>();
            b.gpu.watts         = g.at("watts").get<double>();
            b.seconds           = jb.at("seconds").get<double>();
            for (const json& jm : jb.at("metrics")) {
                Metric mt;
                mt.name           = jm.at("name").get<std::string>();
                mt.unit           = jm.at("unit").get<std::string>();
                mt.value          = jm.at("value").get<double>();
                mt.higherIsBetter = jm.at("higher_is_better").get<bool>();
                mt.within         = statsFrom(jm.at("within"));
                mt.runs           = jm.at("runs").get<std::vector<double>>();
                mt.runCv          = jm.at("run_cv").get<double>();
                for (const auto& [k, v] : jm.at("params").items()) mt.params[k] = v.get<double>();
                b.metrics.push_back(std::move(mt));
            }
            r.benchmarks.push_back(std::move(b));
        }
        out = std::move(r);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

namespace {

int severity(Status s) {
    switch (s) {
    case Status::Ok:          return 0;
    case Status::Partial:     return 1;
    case Status::Unsupported: return 2;
    case Status::Failed:      return 3;
    }
    return 3;
}

void joinText(std::string& into, const std::string& add) {
    if (add.empty() || into.find(add) != std::string::npos) return;
    into += into.empty() ? add : " | " + add;
}

} // namespace

Results mergeRuns(const std::vector<Results>& runs) {
    Results out;
    if (runs.empty()) return out;
    out = runs.front();
    out.run.runs             = static_cast<u32>(runs.size());
    out.machine.thermalEnd   = runs.back().machine.thermalEnd;
    for (Benchmark& b : out.benchmarks) {
        for (Metric& m : b.metrics) {
            std::vector<double> values;
            std::vector<Stats> withins;
            for (const Results& r : runs) {
                if (const Metric* x = r.find(b.id, m.name)) {
                    values.push_back(x->value);
                    withins.push_back(x->within);
                }
            }
            m.runs = values;
            const Stats s = computeStats(values);
            m.value = s.median;
            m.runCv = s.cv;
            // Within-run stats of the run whose value is the median.
            for (size_t i = 0; i < values.size(); ++i)
                if (values[i] == s.median) { m.within = withins[i]; break; }
        }
        for (size_t k = 1; k < runs.size(); ++k) {
            const Benchmark* o = runs[k].find(b.id);
            if (!o) {
                joinText(b.notes, "missing in run " + std::to_string(k + 1));
                b.status = Status::Failed;
                continue;
            }
            if (severity(o->status) > severity(b.status)) b.status = o->status;
            if (o->negative == Control::Fail) b.negative = Control::Fail;
            joinText(b.notes, o->notes);
            joinText(b.negativeDetail, o->negativeDetail);
            b.seconds += o->seconds;
            // GPU window: keep the worst top-state share (the DVFS risk).
            if (o->gpu.topStateShare >= 0 && (b.gpu.topStateShare < 0 || o->gpu.topStateShare < b.gpu.topStateShare))
                b.gpu = o->gpu;
        }
    }
    return out;
}

std::string slugify(const std::string& s) {
    std::string out;
    std::string t = s;
    if (t.rfind("Apple ", 0) == 0) t = t.substr(6);
    for (char c : t) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '.') out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

} // namespace phosphor::soc
