// F5 (schema 5): the one-line summary.  The JSON checks of the report live in
// test_launch_options.cpp next to the JSON syntax checker they share.
#include "diagnostics/bench_report.h"

#include <doctest/doctest.h>
#include <json.hpp>
#include <limits>

#include <string>

using namespace phosphor;

TEST_CASE("bench report line: scene part is appended only when present") {
    BenchReport r;
    r.bench     = "1M Instances (dynamic)";
    r.gpuTiming = true;
    const std::string plain = formatReportLine(r);
    CHECK(plain.find("scene") == std::string::npos);

    r.scene.present          = true;
    r.scene.mode             = "on";
    r.scene.instances        = 1000000;
    r.scene.buckets          = 24;
    r.scene.visible.mean     = 350000.0f;
    r.scene.uploadBytes.mean = 4096.0f;
    r.scene.cpuCommands.mean = 3.0f;
    const std::string line = formatReportLine(r);
    // The existing line (what tools parse) is an unchanged prefix.
    CHECK(line.compare(0, plain.size(), plain) == 0);
    CHECK(line.substr(plain.size()) ==
          " | scene on 1000000 inst 24 buckets visible 350000 upload 4096 B cpu-cmds 3");
}

TEST_CASE("F9 bench report emits all typed RT evidence only when present") {
    BenchReport report;
    auto document = nlohmann::json::parse(reportToJson(report));
    CHECK(document.at("schema_version") == BENCH_REPORT_SCHEMA_VERSION);
    CHECK_FALSE(document.contains("rt"));
    auto& rt = report.rt;
    rt.present = true;
    rt.enabled = true;
    rt.proxyMode = "manifest";
    rt.probe = "shadow";
    rt.alphaStrategy = "generic \"A\"\n";
    rt.effectiveFamily = "apple9";
    rt.blasCount = 103;
    rt.compactedCount = 102;
    rt.instances = 100000;
    rt.capacity = 131072;
    rt.checks = 20;
    rt.checkFailures = 2;
    rt.blasBytes = 1ull << 34;
    rt.uncompactedBytes = 1ull << 35;
    rt.blasScratchBytes = 1ull << 33;
    rt.tlasBytes = 8192;
    rt.tlasScratchBytes = 4096;
    rt.fullTriangles = 262267;
    rt.proxyTriangles = 150000;
    rt.tlasBuilds = 3;
    rt.tlasRefits = 97;
    rt.blasBuilds = (1ull << 54) + 11;
    rt.blasRefits = 27;
    rt.compactions = 102;
    rt.proxyMeshes = 80;
    rt.probeRays = (1ull << 54) + 9; // retain exact integers beyond double precision
    rt.visibilityCompared = 2000000;
    rt.visibilityMismatches = 7;
    rt.alphaTests = 30000;
    rt.opaqueAlphaTests = 0;
    rt.blasBuildMs = 2.5;
    rt.proxyShadowErrorPct = 0.125;
    rt.proxyPrimaryErrorPct = 0.0625;
    rt.proxyDt95 = 0.5;
    rt.proxyAcnePct = 0.25;
    rt.tlasUpdateMs = summarize({0.125f, 0.25f});
    rt.probeMs = summarize({0.5f, 1.0f});
    rt.probeNsPerRay = summarize({0.25f, 0.5f});
    document = nlohmann::json::parse(reportToJson(report));
    const nlohmann::json expected = {
        {"enabled", true}, {"proxy_mode", "manifest"}, {"probe", "shadow"},
        {"alpha_strategy", "generic \"A\"\n"}, {"effective_family", "apple9"},
        {"blas_count", 103}, {"compacted_count", 102}, {"instances", 100000}, {"capacity", 131072},
        {"checks", 20}, {"check_failures", 2}, {"blas_bytes", 1ull << 34}, {"uncompacted_bytes", 1ull << 35},
        {"blas_scratch_bytes", 1ull << 33}, {"tlas_bytes", 8192}, {"tlas_scratch_bytes", 4096},
        {"full_triangles", 262267}, {"proxy_triangles", 150000}, {"tlas_builds", 3}, {"tlas_refits", 97},
        {"blas_builds", (1ull << 54) + 11}, {"blas_refits", 27}, {"compactions", 102},
        {"proxy_meshes", 80}, {"probe_rays", (1ull << 54) + 9}, {"visibility_compared", 2000000},
        {"visibility_mismatches", 7}, {"alpha_tests", 30000}, {"opaque_alpha_tests", 0}, {"blas_build_ms", 2.5},
        {"proxy_shadow_error_pct", 0.125}, {"proxy_primary_error_pct", 0.0625}, {"proxy_dt95_cm", 0.5},
        {"proxy_acne_pct", 0.25},
        {"tlas_update_ms", {{"mean", 0.1875}, {"min", 0.125}, {"p50", 0.125}, {"p95", 0.25}, {"p99", 0.25}, {"max", 0.25}}},
        {"probe_ms", {{"mean", 0.75}, {"min", 0.5}, {"p50", 0.5}, {"p95", 1.0}, {"p99", 1.0}, {"max", 1.0}}},
        {"probe_ns_per_ray", {{"mean", 0.375}, {"min", 0.25}, {"p50", 0.25}, {"p95", 0.5}, {"p99", 0.5}, {"max", 0.5}}},
    };
    CHECK(document.at("rt") == expected);
    CHECK(document.at("rt").at("probe_rays").get<u64>() == (1ull << 54) + 9);
    CHECK(document.at("rt").at("blas_builds").get<u64>() == (1ull << 54) + 11);
    rt.enabled = false;
    CHECK_FALSE(nlohmann::json::parse(reportToJson(report)).at("rt").at("enabled").get<bool>());
    rt.present = false;
    CHECK_FALSE(nlohmann::json::parse(reportToJson(report)).contains("rt"));
}

TEST_CASE("F9 RT report keeps JSON valid for non-finite diagnostic samples") {
    BenchReport report;
    auto& rt = report.rt;
    rt.present = true;
    rt.blasBuildMs = std::numeric_limits<float>::infinity();
    rt.proxyShadowErrorPct = std::numeric_limits<float>::quiet_NaN();
    rt.proxyPrimaryErrorPct = -std::numeric_limits<float>::infinity();
    rt.proxyDt95 = std::numeric_limits<float>::quiet_NaN();
    rt.proxyAcnePct = std::numeric_limits<float>::infinity();
    rt.tlasUpdateMs.mean = std::numeric_limits<float>::infinity();
    rt.probeMs.max = std::numeric_limits<float>::quiet_NaN();
    rt.probeNsPerRay.p99 = std::numeric_limits<float>::infinity();
    const auto object = nlohmann::json::parse(reportToJson(report)).at("rt");
    for (const char* key : {"blas_build_ms", "proxy_shadow_error_pct", "proxy_primary_error_pct", "proxy_dt95_cm", "proxy_acne_pct"})
        CHECK(object.at(key) == 0);
    CHECK(object.at("tlas_update_ms").at("mean") == 0);
    CHECK(object.at("probe_ms").at("max") == 0);
    CHECK(object.at("probe_ns_per_ray").at("p99") == 0);
}

TEST_CASE("F10-F12 report labels opt-in lighting and the denoise boundary") {
    BenchReport r;auto json=nlohmann::json::parse(reportToJson(r));CHECK_FALSE(json.contains("lighting"));
    r.lighting.present=true;r.lighting.shadows="rt";r.lighting.direct="restir";r.lighting.gi="ddgi";
    r.lighting.checks=3;r.lighting.failures=1;json=nlohmann::json::parse(reportToJson(r));
    CHECK(json["lighting"]["shadows"]=="rt");CHECK(json["lighting"]["direct"]=="restir");
    CHECK(json["lighting"]["failures"]==1);CHECK(json["lighting"]["experimental"]==true);
    CHECK(json["lighting"]["full_lighting_denoise"]=="F13_SOURCE_UNVERIFIED");
}


TEST_CASE("F12 report identifies physical visibility ablation independently of check failures") {
    BenchReport report;report.lighting.present=true;
    auto json=nlohmann::json::parse(reportToJson(report));
    CHECK(json["lighting"]["gi_visibility_disabled"]==false);
    report.lighting.giVisibilityDisabled=true;report.lighting.checks=512;
    json=nlohmann::json::parse(reportToJson(report));
    CHECK(json["lighting"]["gi_visibility_disabled"]==true);
    CHECK(json["lighting"]["checks"]==512);CHECK(json["lighting"]["failures"]==0);
}
