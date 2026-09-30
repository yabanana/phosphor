#include "diagnostics/soc_model.h"

#include <json.hpp> // nlohmann/json, shipped with tinygltf

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace phosphor::soc {

using json = nlohmann::json;

namespace {

struct FieldDef {
    const char* key;
    Field SocCostModel::*member;
    const char* bench;
    const char* metric;
};

const FieldDef kFields[] = {
    {"f32_fma_tflops", &SocCostModel::f32Fma, "B-01", "f32.fma.indep"},
    {"f16_fma_tflops", &SocCostModel::f16Fma, "B-01", "f16.fma.indep"},
    {"f32_add_tops", &SocCostModel::f32Add, "B-01", "f32.add.indep"},
    {"i32_add_tops", &SocCostModel::i32Add, "B-01", "i32.add.indep"},
    {"i32_mul_tops", &SocCostModel::i32Mul, "B-01", "i32.mul.indep"},
    {"f32_transcendental_tops", &SocCostModel::f32Transcendental, "B-03", "f32.transcendental.fast"},
    {"f32_div_tops", &SocCostModel::f32Div, "B-03", "f32.div.fast"},
    {"dram_bw_gbps", &SocCostModel::dramBw, "B-08", "dram_bw"},
    {"onchip_bw_gbps", &SocCostModel::onchipBw, "B-08", "onchip_bw"},
    {"latency_l1_ns", &SocCostModel::latencyL1, "B-08", "latency_l1"},
    {"latency_dram_ns", &SocCostModel::latencyDram, "B-08", "latency_dram"},
    {"slc_size_mib", &SocCostModel::slcSizeMiB, "B-08", "slc_size_estimate"},
    {"write_bw_dram_gbps", &SocCostModel::writeBwDram, "B-08", "write_bw.dram"},
    {"copy_bw_dram_gbps", &SocCostModel::copyBwDram, "B-08", "copy_bw.dram"},
    {"empty_pass_us", &SocCostModel::emptyPassUs, "B-14", "empty_pass.us"},
    {"dispatch_empty_us", &SocCostModel::dispatchEmptyUs, "B-17", "dispatch.empty.us"},
    {"dispatch_indirect_us", &SocCostModel::dispatchIndirectUs, "B-17", "dispatch.indirect.us"},
    {"barrier_encoder_us", &SocCostModel::barrierEncoderUs, "B-18", "barrier.encoder.us"},
    {"barrier_queue_us", &SocCostModel::barrierQueueUs, "B-18", "barrier.queue.us"},
    {"commit_gpu_us", &SocCostModel::commitGpuUs, "B-28", "commit.gpu_us"},
    {"commit_to_cpu_us", &SocCostModel::commitToCpuUs, "B-28", "commit_to_cpu.us"},
    {"rays_coherent_grays", &SocCostModel::raysCoherent, "B-20", "rays.coherent"},
    {"rays_incoherent_grays", &SocCostModel::raysIncoherent, "B-20", "rays.incoherent"},
    {"gemm_f16_tops", &SocCostModel::gemmF16, "B-22", "gemm.f16.tops"},
    {"gemm_bf16_tops", &SocCostModel::gemmBf16, "B-22", "gemm.bf16.tops"},
    {"gemm_i8_tops", &SocCostModel::gemmI8, "B-22", "gemm.i8.tops"},
    {"gemm_simd_f16_tops", &SocCostModel::gemmSimdF16, "B-22", "gemm.simd_f16.tops"},
};

double runCvOf(const Metric& m) {
    if (m.runs.size() >= 2) return m.runCv;
    return m.within.n >= 2 ? m.within.cv : kNaN;
}

double nanIfNull(const json& j) { return j.is_number() ? j.get<double>() : kNaN; }
json numOrNull(double v) { return std::isfinite(v) ? json(v) : json(nullptr); }

json fieldJson(const Field& f) {
    return {{"value", numOrNull(f.value)}, {"run_cv", numOrNull(f.runCv)}, {"unit", f.unit}, {"source", f.source}};
}
Field fieldFrom(const json& j) {
    Field f;
    f.value = nanIfNull(j.at("value"));
    f.runCv = nanIfNull(j.at("run_cv"));
    f.unit = j.at("unit").get<std::string>();
    f.source = j.at("source").get<std::string>();
    return f;
}

// Seconds-normalised rate of a field in "ops per second" (or bytes/s).
double perSecond(const Field& f, double scale) { return f.has() && f.value > 0 ? f.value * scale : 0.0; }

} // namespace

SocCostModel SocCostModel::fromResults(const Results& r) {
    SocCostModel m;
    m.machine.chip = r.machine.chip;
    m.machine.slug = r.machine.slug;
    m.machine.gpuFamily = r.machine.gpuFamily;
    m.machine.os = r.machine.os;
    m.machine.powerSource = r.machine.powerSource;
    m.machine.gpuCores = r.machine.gpuCores;
    if (!r.machine.gpuPStateMHz.empty())
        m.machine.topPStateMHz = *std::max_element(r.machine.gpuPStateMHz.begin(), r.machine.gpuPStateMHz.end());

    for (const FieldDef& d : kFields) {
        const Metric* mt = r.find(d.bench, d.metric);
        if (!mt) continue;
        Field& f = m.*(d.member);
        f.value = mt->value;
        f.runCv = runCvOf(*mt);
        f.unit = mt->unit;
        f.source = std::string(d.bench) + "/" + d.metric;
    }

    if (const Benchmark* b = r.find("B-14")) {
        for (const Metric& mt : b->metrics) {
            const bool isStore = mt.name.rfind("store.", 0) == 0;
            const bool isLoad = mt.name.rfind("load.", 0) == 0;
            if (!isStore && !isLoad) continue;
            const size_t k = mt.name.find('.');
            const size_t k2 = mt.name.find('.', k + 1);
            if (k2 == std::string::npos) continue;
            TbdrEntry e;
            e.kind = mt.name.substr(0, k);
            e.format = mt.name.substr(k + 1, k2 - k - 1);
            e.res = mt.name.substr(k2 + 1);
            e.gbps = mt.value;
            e.runCv = runCvOf(mt);
            m.tbdr.push_back(e);
        }
    }
    return m;
}

PassPrediction SocCostModel::predictPass(const PassWork& w) const {
    PassPrediction p;
    const double dramRate = perSecond(dramBw, 1e9);
    const double onchipRate = perSecond(onchipBw, 1e9);
    if (dramRate > 0) p.dramMs = w.dramBytes / dramRate * 1e3;
    if (onchipRate > 0) p.onchipMs = w.onchipBytes / onchipRate * 1e3;

    // ALU: the classes share the ALUs, times add up.  Integer work uses the
    // fastest measured integer rate (a lower bound).
    const double fmaRate = perSecond(f32Fma, 1e12);
    const double transRate = perSecond(f32Transcendental, 1e12);
    const double divRate = perSecond(f32Div, 1e12);
    const double intRate = std::max(perSecond(i32Add, 1e12), perSecond(i32Mul, 1e12));
    double aluS = 0;
    if (fmaRate > 0) aluS += w.flops / fmaRate;
    if (transRate > 0) aluS += w.transcendentals / transRate;
    if (divRate > 0) aluS += w.divides / divRate;
    if (intRate > 0) aluS += w.intOps / intRate;
    p.aluMs = aluS * 1e3;

    p.ms = std::max({p.dramMs, p.onchipMs, p.aluMs});
    if (p.ms > 0) p.bound = p.ms == p.dramMs ? "dram" : p.ms == p.onchipMs ? "onchip" : "alu";
    if (w.dramBytes > 0) p.arithmeticIntensity = w.flops / w.dramBytes;
    return p;
}

double SocCostModel::ridgePoint() const {
    if (!hasRoofs() || dramBw.value <= 0) return kNaN;
    return f32Fma.value * 1e12 / (dramBw.value * 1e9);
}
double SocCostModel::ridgePointOnchip() const {
    if (!hasFp32() || !hasOnchipBw() || onchipBw.value <= 0) return kNaN;
    return f32Fma.value * 1e12 / (onchipBw.value * 1e9);
}

std::string toJson(const SocCostModel& m) {
    json j;
    j["schema"] = "phosphor-soc-model";
    j["schema_version"] = 1;
    j["machine"] = {{"chip", m.machine.chip},       {"slug", m.machine.slug},
                    {"gpu_family", m.machine.gpuFamily}, {"os", m.machine.os},
                    {"power_source", m.machine.powerSource}, {"gpu_cores", m.machine.gpuCores},
                    {"top_pstate_mhz", numOrNull(m.machine.topPStateMHz)}};
    json fields = json::object();
    for (const FieldDef& d : kFields) fields[d.key] = fieldJson(m.*(d.member));
    j["fields"] = fields;
    json t = json::array();
    for (const TbdrEntry& e : m.tbdr)
        t.push_back({{"kind", e.kind}, {"format", e.format}, {"res", e.res}, {"gbps", e.gbps}, {"run_cv", numOrNull(e.runCv)}});
    j["tbdr"] = t;
    j["derived"] = {{"ridge_point_flop_per_byte", numOrNull(m.ridgePoint())},
                    {"ridge_point_onchip_flop_per_byte", numOrNull(m.ridgePointOnchip())}};
    return j.dump(1) + "\n";
}

bool fromJson(const std::string& text, SocCostModel& out, std::string* error) {
    try {
        const json j = json::parse(text);
        if (j.at("schema").get<std::string>() != "phosphor-soc-model") throw std::runtime_error("not a soc model");
        SocCostModel m;
        const json& mj = j.at("machine");
        m.machine.chip = mj.at("chip").get<std::string>();
        m.machine.slug = mj.at("slug").get<std::string>();
        m.machine.gpuFamily = mj.at("gpu_family").get<std::string>();
        m.machine.os = mj.at("os").get<std::string>();
        m.machine.powerSource = mj.at("power_source").get<std::string>();
        m.machine.gpuCores = mj.at("gpu_cores").get<u32>();
        m.machine.topPStateMHz = nanIfNull(mj.at("top_pstate_mhz"));
        const json& fj = j.at("fields");
        for (const FieldDef& d : kFields) m.*(d.member) = fieldFrom(fj.at(d.key));
        for (const json& e : j.at("tbdr")) {
            TbdrEntry t;
            t.kind = e.at("kind").get<std::string>();
            t.format = e.at("format").get<std::string>();
            t.res = e.at("res").get<std::string>();
            t.gbps = e.at("gbps").get<double>();
            t.runCv = nanIfNull(e.at("run_cv"));
            m.tbdr.push_back(t);
        }
        out = std::move(m);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

OpCounts& OpCounts::operator+=(const OpCounts& o) {
    flops += o.flops;
    transcendentals += o.transcendentals;
    divides += o.divides;
    intOps += o.intOps;
    samples += o.samples;
    loads += o.loads;
    stores += o.stores;
    return *this;
}
OpCounts OpCounts::scaled(double k) const {
    return {flops * k, transcendentals * k, divides * k, intOps * k, samples * k, loads * k, stores * k};
}

OpCounts opsPerInvocation(const ShaderOps& s, double loopTrips, bool lowerBound) {
    const bool useMin = lowerBound && s.hasMin;
    OpCounts body;
    for (const OpCounts& l : useMin ? s.loopsMin : s.loops) body += l;
    OpCounts r = useMin ? s.staticsMin : s.statics;
    OpCounts extra = body.scaled(loopTrips - 1.0);
    r += extra;
    auto clamp = [](double& v) { v = std::max(v, 0.0); };
    clamp(r.flops); clamp(r.transcendentals); clamp(r.divides); clamp(r.intOps);
    clamp(r.samples); clamp(r.loads); clamp(r.stores);
    return r;
}

namespace {
OpCounts countsFrom(const json& j) {
    OpCounts c;
    auto get = [&](const char* k) { return j.contains(k) && j[k].is_number() ? j[k].get<double>() : 0.0; };
    c.flops = get("flops");
    c.transcendentals = get("transcendentals");
    c.divides = get("divides");
    c.intOps = get("int_ops");
    c.samples = get("samples");
    c.loads = get("loads");
    c.stores = get("stores");
    return c;
}
} // namespace

bool parseShaderOps(const std::string& text, std::vector<ShaderOps>& out, std::string* error) {
    try {
        const json j = json::parse(text);
        std::vector<ShaderOps> v;
        for (const auto& [name, f] : j.at("functions").items()) {
            ShaderOps s;
            s.name = name;
            s.kind = f.value("kind", std::string());
            s.statics = countsFrom(f.at("static"));
            if (f.contains("loops"))
                for (const json& l : f["loops"]) s.loops.push_back(countsFrom(l.at("per_iteration")));
            if (f.contains("static_min")) {
                s.hasMin = true;
                s.staticsMin = countsFrom(f.at("static_min"));
                for (const json& l : f["loops"]) s.loopsMin.push_back(countsFrom(l.at("per_iteration_min")));
            }
            v.push_back(std::move(s));
        }
        out = std::move(v);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

const ShaderOps* findShader(const std::vector<ShaderOps>& all, const std::string& shader) {
    const ShaderOps* best = nullptr;
    for (const ShaderOps& s : all) {
        if (s.name == shader) return &s;
        if (shader.rfind(s.name, 0) == 0 && (!best || s.name.size() > best->name.size())) best = &s;
    }
    return best;
}

} // namespace phosphor::soc
