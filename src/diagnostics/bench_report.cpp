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
    for (const char c : s) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char esc[8];
                std::snprintf(esc, sizeof(esc), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
                out += esc;
            } else {
                out.push_back(c);
            }
        }
    }
    return out;
}

float finiteOrZero(float v) { return std::isfinite(v) ? v : 0.0f; }

std::string summaryToJson(const TimingSummary& t) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "{\"mean\": %.4f, \"min\": %.4f, \"p50\": %.4f, \"p99\": %.4f, \"max\": %.4f}",
                  finiteOrZero(t.mean), finiteOrZero(t.min), finiteOrZero(t.p50), finiteOrZero(t.p99),
                  finiteOrZero(t.max));
    return buf;
}

std::string passToJson(const PassReport& p) {
    auto list = [](const std::vector<std::string>& v) {
        std::string out = "[";
        for (size_t i = 0; i < v.size(); ++i) out += (i ? ", \"" : "\"") + jsonEscape(v[i]) + "\"";
        return out + "]";
    };
    return "    {\"name\": \"" + jsonEscape(p.name) + "\", \"queue\": \"" + jsonEscape(p.queue) +
           "\", \"fused\": " + (p.fused ? "true" : "false") + ", \"passes\": " + list(p.passes) +
           ", \"shaders\": " + list(p.shaders) + ", \"dram_bytes\": " +
           std::to_string(static_cast<unsigned long long>(p.dramBytes)) +
           ", \"frames\": " + std::to_string(p.frames) + ", \"gpu_ms\": " +
           summaryToJson(p.gpuMs) + "}";
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
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "%s | %ux%u vsync=%s ui=%s | %u frames | %.1f fps | "
                  "frame %.3f ms (p99 %.3f) | CPU %.3f ms (p99 %.3f) | GPU %.3f ms (p99 %.3f) | "
                  "wait %.3f ms (p99 %.3f) | GPU allocations %llu | CPU heap %+lld blocks %+lld bytes",
                  r.bench.c_str(), r.width, r.height, r.vsync ? "on" : "off", r.ui ? "on" : "off",
                  r.frames, r.fps, r.frameMs.mean, r.frameMs.p99, r.cpuMs.mean, r.cpuMs.p99,
                  r.gpuMs.mean, r.gpuMs.p99, r.waitMs.mean, r.waitMs.p99,
                  static_cast<unsigned long long>(r.gpuAllocations), static_cast<long long>(r.cpuHeapBlocksDelta),
                  static_cast<long long>(r.cpuHeapBytesDelta));
    std::string line = buf;
    if (r.gpuTiming) {
        char extra[96];
        std::snprintf(extra, sizeof(extra), " | GPU passes %.3f ms (p99 %.3f)", r.gpuPassSumMs.mean,
                      r.gpuPassSumMs.p99);
        line += extra;
    }
    return line;
}

std::string reportToJson(const BenchReport& r) {
    const std::string bench  = jsonEscape(r.bench);
    const std::string device = jsonEscape(r.device);
    static constexpr char headFormat[] =
                  "{\n  \"schema_version\": %u,\n  \"bench\": \"%s\",\n  \"device\": \"%s\",\n  \"width\": %u,\n  \"height\": %u,\n"
                  "  \"vsync\": %s,\n  \"ui\": %s,\n  \"frames\": %u,\n  \"fps\": %.2f,\n"
                  "  \"gpu_allocations\": %llu,\n  \"cpu_heap_blocks_delta\": %lld,\n"
                  "  \"cpu_heap_bytes_delta\": %lld,\n";
    auto formatHead = [&](char* dst, size_t size) {
        return std::snprintf(dst, size, headFormat, BENCH_REPORT_SCHEMA_VERSION, bench.c_str(), device.c_str(),
                             r.width, r.height, r.vsync ? "true" : "false", r.ui ? "true" : "false", r.frames,
                             r.fps, static_cast<unsigned long long>(r.gpuAllocations),
                             static_cast<long long>(r.cpuHeapBlocksDelta),
                             static_cast<long long>(r.cpuHeapBytesDelta));
    };
    // Sized to fit: a truncated head would make the JSON invalid.
    std::string head(static_cast<size_t>(std::max(formatHead(nullptr, 0), 0)) + 1, '\0');
    formatHead(head.data(), head.size());
    head.pop_back();
    std::string timing = std::string("  \"gpu_timing\": ") + (r.gpuTiming ? "true" : "false") +
                         ",\n  \"gpu_timing_unfused\": " + (r.gpuTimingUnfused ? "true" : "false");
    if (r.gpuTiming) {
        timing += ",\n  \"passes\": [";
        for (size_t i = 0; i < r.passes.size(); ++i) timing += (i ? ",\n" : "\n") + passToJson(r.passes[i]);
        timing += r.passes.empty() ? "]" : "\n  ]";
        timing += ",\n  \"gpu_pass_sum_ms\": " + summaryToJson(r.gpuPassSumMs) +
                  ",\n  \"gpu_frame_span_ms\": " + summaryToJson(r.gpuFrameSpanMs);
    }
    return head +
           "  \"frame_ms\": " + summaryToJson(r.frameMs) + ",\n" +
           "  \"cpu_ms\": " + summaryToJson(r.cpuMs) + ",\n" +
           "  \"gpu_ms\": " + summaryToJson(r.gpuMs) + ",\n" +
           "  \"wait_ms\": " + summaryToJson(r.waitMs) +
           (r.pipelinesJson.empty() ? std::string() : ",\n  \"pipelines\": " + r.pipelinesJson) + ",\n" + timing +
           "\n}\n";
}

} // namespace phosphor
