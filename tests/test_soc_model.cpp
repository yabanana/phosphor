#include "diagnostics/soc_model.h"

#include <doctest/doctest.h>

#include <cmath>
#include <string>

using namespace phosphor;
using namespace phosphor::soc;

namespace {

void add(Benchmark& b, const std::string& name, const std::string& unit, double v, double runCv = 0.01) {
    Metric m;
    m.name = name;
    m.unit = unit;
    m.value = v;
    m.runs = {v, v, v};
    m.runCv = runCv;
    b.metrics.push_back(m);
}

Benchmark& bench(Results& r, const std::string& id) {
    r.benchmarks.push_back({});
    r.benchmarks.back().id = id;
    return r.benchmarks.back();
}

// 20 TFLOPS FP32, 500 GB/s DRAM -> ridge 40 FLOP/byte.
Results synthetic() {
    Results r;
    r.machine.chip = "Apple Test";
    r.machine.slug = "test";
    r.machine.gpuCores = 40;
    r.machine.gpuPStateMHz = {338, 1000, 1620};
    Benchmark& b1 = bench(r, "B-01");
    add(b1, "f32.fma.indep", "TFLOPS", 20.0);
    add(b1, "f16.fma.indep", "TFLOPS", 40.0);
    add(b1, "f32.add.indep", "Top/s", 10.0);
    add(b1, "i32.add.indep", "Top/s", 8.0);
    add(b1, "i32.mul.indep", "Top/s", 4.0);
    Benchmark& b3 = bench(r, "B-03");
    add(b3, "f32.transcendental.fast", "Top/s", 2.5);
    add(b3, "f32.div.fast", "Top/s", 5.0);
    Benchmark& b8 = bench(r, "B-08");
    add(b8, "dram_bw", "GB/s", 500.0);
    add(b8, "onchip_bw", "GB/s", 2000.0);
    add(b8, "latency_l1", "ns", 20.0);
    add(b8, "latency_dram", "ns", 300.0);
    add(b8, "slc_size_estimate", "MiB", 64.0);
    Benchmark& b14 = bench(r, "B-14");
    add(b14, "empty_pass.us", "us", 5.0);
    add(b14, "store.rgba8.1080p", "GB/s", 300.0);
    add(b14, "load.rgba16f.4k", "GB/s", 250.0);
    Benchmark& b17 = bench(r, "B-17");
    add(b17, "dispatch.empty.us", "us", 3.0);
    return r;
}

} // namespace

TEST_CASE("SocCostModel: extraction") {
    const SocCostModel m = SocCostModel::fromResults(synthetic());
    CHECK(m.hasFp32());
    CHECK(m.hasRoofs());
    CHECK(m.f32Fma.value == 20.0);
    CHECK(m.f32Fma.source == "B-01/f32.fma.indep");
    CHECK(m.f32Fma.runCv == doctest::Approx(0.01));
    CHECK(m.f16Fma.value == 40.0);
    CHECK(m.i32Mul.value == 4.0);
    CHECK(m.f32Transcendental.value == 2.5);
    CHECK(m.dramBw.value == 500.0);
    CHECK(m.onchipBw.value == 2000.0);
    CHECK(m.slcSizeMiB.value == 64.0);
    CHECK(m.emptyPassUs.value == 5.0);
    CHECK(m.dispatchEmptyUs.value == 3.0);
    CHECK(m.machine.gpuCores == 40);
    CHECK(m.machine.topPStateMHz == 1620.0);
    REQUIRE(m.tbdr.size() == 2);
    CHECK(m.tbdr[0].kind == "store");
    CHECK(m.tbdr[0].format == "rgba8");
    CHECK(m.tbdr[0].res == "1080p");
    CHECK(m.tbdr[0].gbps == 300.0);
    CHECK(m.tbdr[1].kind == "load");
    CHECK(m.tbdr[1].format == "rgba16f");
}

TEST_CASE("SocCostModel: missing metrics stay unset") {
    Results r = synthetic();
    r.benchmarks.erase(r.benchmarks.begin() + 2); // B-08
    const SocCostModel m = SocCostModel::fromResults(r);
    CHECK(m.hasFp32());
    CHECK_FALSE(m.hasDramBw());
    CHECK_FALSE(m.hasOnchipBw());
    CHECK_FALSE(m.hasRoofs());
    CHECK(std::isnan(m.dramBw.value));
    CHECK(std::isnan(m.ridgePoint()));
    CHECK_FALSE(m.gemmF16.has());
    CHECK_FALSE(m.raysCoherent.has());
    const SocCostModel empty = SocCostModel::fromResults(Results{});
    CHECK_FALSE(empty.hasFp32());
    CHECK(empty.tbdr.empty());
    CHECK(std::isnan(empty.machine.topPStateMHz));
    // No rates: the bound is 0 (still a lower bound), never an invented number.
    PassWork w;
    w.flops = 1e12;
    w.dramBytes = 1e9;
    const PassPrediction p = empty.predictPass(w);
    CHECK(p.ms == 0.0);
    CHECK(p.bound == "none");
}

TEST_CASE("SocCostModel: ridge point") {
    const SocCostModel m = SocCostModel::fromResults(synthetic());
    CHECK(m.ridgePoint() == doctest::Approx(40.0));       // 20e12 / 500e9
    CHECK(m.ridgePointOnchip() == doctest::Approx(10.0)); // 20e12 / 2000e9
}

TEST_CASE("SocCostModel: predictions per regime") {
    const SocCostModel m = SocCostModel::fromResults(synthetic());
    { // DRAM bound: 1 GB at 500 GB/s = 2 ms; 1 GFLOP at 20 TFLOPS = 0.05 ms
        PassWork w;
        w.dramBytes = 1e9;
        w.flops = 1e9;
        const PassPrediction p = m.predictPass(w);
        CHECK(p.bound == "dram");
        CHECK(p.ms == doctest::Approx(2.0));
        CHECK(p.aluMs == doctest::Approx(0.05));
        CHECK(p.arithmeticIntensity == doctest::Approx(1.0));
    }
    { // ALU bound: 40 GFLOP = 2 ms, 10 MB = 0.02 ms
        PassWork w;
        w.dramBytes = 1e7;
        w.flops = 4e10;
        const PassPrediction p = m.predictPass(w);
        CHECK(p.bound == "alu");
        CHECK(p.ms == doctest::Approx(2.0));
        CHECK(p.arithmeticIntensity == doctest::Approx(4000.0));
    }
    { // on-chip bound: 4 GB at 2000 GB/s = 2 ms
        PassWork w;
        w.onchipBytes = 4e9;
        w.dramBytes = 1e8; // 0.2 ms
        const PassPrediction p = m.predictPass(w);
        CHECK(p.bound == "onchip");
        CHECK(p.ms == doctest::Approx(2.0));
    }
    { // ALU terms add up: 1 ms + 1 ms + 1 ms + 0.5 ms
        PassWork w;
        w.flops = 2e10;
        w.transcendentals = 2.5e9;
        w.divides = 5e9;
        w.intOps = 4e9;
        const PassPrediction p = m.predictPass(w);
        CHECK(p.aluMs == doctest::Approx(3.5));
        CHECK(p.bound == "alu");
        CHECK(std::isnan(p.arithmeticIntensity)); // no DRAM traffic
    }
    { // no work at all
        const PassPrediction p = m.predictPass({});
        CHECK(p.ms == 0.0);
        CHECK(p.bound == "none");
    }
    { // at the ridge point both roofs give the same time
        PassWork w;
        w.dramBytes = 1e9;
        w.flops = 1e9 * m.ridgePoint();
        const PassPrediction p = m.predictPass(w);
        CHECK(p.aluMs == doctest::Approx(p.dramMs));
    }
}

TEST_CASE("SocCostModel: prediction is a lower bound of a measurement with overheads") {
    const SocCostModel m = SocCostModel::fromResults(synthetic());
    u32 seed = 12345;
    auto rnd = [&] {
        seed = seed * 1664525u + 1013904223u;
        return double(seed >> 8) / double(1u << 24);
    };
    for (int i = 0; i < 500; ++i) {
        PassWork w;
        w.flops = rnd() * 1e11;
        w.transcendentals = rnd() * 1e9;
        w.divides = rnd() * 1e9;
        w.intOps = rnd() * 1e9;
        w.dramBytes = rnd() * 2e9;
        w.onchipBytes = rnd() * 4e9;
        const PassPrediction p = m.predictPass(w);
        // A "real" pass: the same terms without perfect overlap (sum) plus dispatch and pass overheads.
        const double measured =
            p.dramMs + p.onchipMs + p.aluMs + (m.dispatchEmptyUs.value + m.emptyPassUs.value) * 1e-3;
        CHECK(p.ms <= measured);
        CHECK(p.ms >= p.dramMs);
        CHECK(p.ms >= p.onchipMs);
        CHECK(p.ms >= p.aluMs);
    }
    // Doubling the work doubles the bound (linearity of the roofs).
    PassWork w;
    w.flops = 1e10;
    w.dramBytes = 3e8;
    PassWork w2 = w;
    w2.flops *= 2;
    w2.dramBytes *= 2;
    CHECK(m.predictPass(w2).ms == doctest::Approx(2.0 * m.predictPass(w).ms));
}

TEST_CASE("SocCostModel: JSON round trip") {
    const SocCostModel m = SocCostModel::fromResults(synthetic());
    const std::string j = toJson(m);
    SocCostModel back;
    std::string err;
    REQUIRE_MESSAGE(fromJson(j, back, &err), err);
    CHECK(back.f32Fma.value == 20.0);
    CHECK(back.dramBw.value == 500.0);
    CHECK(back.dramBw.source == "B-08/dram_bw");
    CHECK_FALSE(back.gemmI8.has()); // null -> NaN
    CHECK(back.machine.topPStateMHz == 1620.0);
    REQUIRE(back.tbdr.size() == 2);
    CHECK(back.tbdr[1].gbps == 250.0);
    CHECK(back.ridgePoint() == doctest::Approx(40.0));
    CHECK(j.find("ridge_point_flop_per_byte") != std::string::npos);
    CHECK_FALSE(fromJson("{\"schema\":\"other\"}", back));
    CHECK_FALSE(fromJson("not json", back));
}

TEST_CASE("Shader ops: parse, lookup, loop trips") {
    const char* text = R"({"functions": {
        "forward_fs": {"kind": "fragment",
                       "static": {"flops": 100, "transcendentals": 4, "divides": 1, "int_ops": 10, "samples": 3},
                       "loops": [{"per_iteration": {"flops": 30, "transcendentals": 2, "samples": 1}}]},
        "forward_vs": {"kind": "vertex", "static": {"flops": 50}}}})";
    std::vector<ShaderOps> ops;
    std::string err;
    REQUIRE_MESSAGE(parseShaderOps(text, ops, &err), err);
    REQUIRE(ops.size() == 2);
    const ShaderOps* fs = findShader(ops, "forward_fs");
    REQUIRE(fs);
    CHECK(fs->kind == "fragment");
    CHECK(findShader(ops, "forward_fs_variant3") == fs); // prefix match
    CHECK(findShader(ops, "nothing") == nullptr);
    CHECK(opsPerInvocation(*fs, 1).flops == 100);
    const OpCounts eight = opsPerInvocation(*fs, 8);
    CHECK(eight.flops == 100 + 7 * 30);
    CHECK(eight.transcendentals == 4 + 7 * 2);
    CHECK(eight.samples == 3 + 7);
    CHECK(opsPerInvocation(*fs, 0).flops == 70); // loop body not executed at all
    CHECK_FALSE(parseShaderOps("{}", ops));
}

TEST_CASE("Shader ops: the lower bound uses the cheapest path (OPT-0.4)") {
    const char* text = R"({"functions": {
        "forward_fs": {"kind": "fragment",
                       "static": {"flops": 100}, "static_min": {"flops": 40},
                       "loops": [{"per_iteration": {"flops": 30}, "per_iteration_min": {"flops": 10}}]},
        "old_fs": {"kind": "fragment", "static": {"flops": 100}, "loops": [{"per_iteration": {"flops": 30}}]}}})";
    std::vector<ShaderOps> ops;
    REQUIRE(parseShaderOps(text, ops));
    const ShaderOps* fs = findShader(ops, "forward_fs");
    REQUIRE(fs);
    CHECK(fs->hasMin);
    CHECK(opsPerInvocation(*fs, 4, true).flops == 40 + 3 * 10);  // lower bound
    CHECK(opsPerInvocation(*fs, 4, false).flops == 100 + 3 * 30); // estimate
    CHECK(opsPerInvocation(*fs, 4).flops == 100 + 3 * 30);        // default: estimate (unchanged callers)
    const ShaderOps* old = findShader(ops, "old_fs");             // JSON without *_min: falls back
    REQUIRE(old);
    CHECK_FALSE(old->hasMin);
    CHECK(opsPerInvocation(*old, 4, true).flops == 100 + 3 * 30);
}
