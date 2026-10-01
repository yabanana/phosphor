// F5 (schema 5): the one-line summary.  The JSON checks of the report live in
// test_launch_options.cpp next to the JSON syntax checker they share.
#include "diagnostics/bench_report.h"

#include <doctest/doctest.h>

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
