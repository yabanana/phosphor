#include "diagnostics/bench_report.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace phosphor {

namespace {

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

std::string summaryToJson(const TimingSummary& t) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "{\"mean\": %.4f, \"min\": %.4f, \"p50\": %.4f, \"p99\": %.4f, \"max\": %.4f}",
                  t.mean, t.min, t.p50, t.p99, t.max);
    return buf;
}

} // namespace

TimingSummary summarize(std::vector<float> values) {
    TimingSummary s;
    if (values.empty()) return s;

    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    // Nearest-rank percentile: the smallest value with at least p of the samples at or below it.
    auto rank = [&](double p) {
        const size_t r = static_cast<size_t>(std::ceil(p * static_cast<double>(n)));
        return values[std::clamp<size_t>(r, 1, n) - 1];
    };
    s.mean = static_cast<float>(std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n));
    s.min  = values.front();
    s.p50  = rank(0.50);
    s.p99  = rank(0.99);
    s.max  = values.back();
    return s;
}

void summarizeSamples(const std::vector<FrameSample>& samples, BenchReport& report) {
    std::vector<float> frame, cpu, gpu, wait;
    frame.reserve(samples.size());
    cpu.reserve(samples.size());
    gpu.reserve(samples.size());
    wait.reserve(samples.size());
    for (const FrameSample& s : samples) {
        frame.push_back(s.frameMs);
        cpu.push_back(s.cpuMs);
        gpu.push_back(s.gpuMs);
        wait.push_back(s.waitMs);
    }
    report.frames  = static_cast<u32>(samples.size());
    report.frameMs = summarize(std::move(frame));
    report.cpuMs   = summarize(std::move(cpu));
    report.gpuMs   = summarize(std::move(gpu));
    report.waitMs  = summarize(std::move(wait));
    report.fps     = report.frameMs.mean > 0.0f ? 1000.0f / report.frameMs.mean : 0.0f;
}

std::string formatReportLine(const BenchReport& r) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "%s | %ux%u vsync=%s ui=%s | %u frames | %.1f fps | "
                  "frame %.3f ms (p99 %.3f) | CPU %.3f ms (p99 %.3f) | GPU %.3f ms (p99 %.3f) | "
                  "wait %.3f ms (p99 %.3f)",
                  r.bench.c_str(), r.width, r.height, r.vsync ? "on" : "off", r.ui ? "on" : "off",
                  r.frames, r.fps, r.frameMs.mean, r.frameMs.p99, r.cpuMs.mean, r.cpuMs.p99,
                  r.gpuMs.mean, r.gpuMs.p99, r.waitMs.mean, r.waitMs.p99);
    return buf;
}

std::string reportToJson(const BenchReport& r) {
    char head[512];
    std::snprintf(head, sizeof(head),
                  "{\n  \"bench\": \"%s\",\n  \"device\": \"%s\",\n  \"width\": %u,\n  \"height\": %u,\n"
                  "  \"vsync\": %s,\n  \"ui\": %s,\n  \"frames\": %u,\n  \"fps\": %.2f,\n",
                  jsonEscape(r.bench).c_str(), jsonEscape(r.device).c_str(), r.width, r.height,
                  r.vsync ? "true" : "false", r.ui ? "true" : "false", r.frames, r.fps);
    return std::string(head) +
           "  \"frame_ms\": " + summaryToJson(r.frameMs) + ",\n" +
           "  \"cpu_ms\": " + summaryToJson(r.cpuMs) + ",\n" +
           "  \"gpu_ms\": " + summaryToJson(r.gpuMs) + ",\n" +
           "  \"wait_ms\": " + summaryToJson(r.waitMs) + "\n}\n";
}

} // namespace phosphor
