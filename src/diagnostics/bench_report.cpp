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
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "{\"mean\": %.4f, \"min\": %.4f, \"p50\": %.4f, \"p95\": %.4f, \"p99\": %.4f, \"max\": %.4f}",
                  finiteOrZero(t.mean), finiteOrZero(t.min), finiteOrZero(t.p50), finiteOrZero(t.p95),
                  finiteOrZero(t.p99), finiteOrZero(t.max));
    return buf;
}

std::string workToJson(const std::vector<PassWork>& work) {
    if (work.empty()) return "";
    std::string out = ", \"work\": [";
    for (size_t i = 0; i < work.size(); ++i) {
        const PassWork& w = work[i];
        auto n = [](u64 v) { return std::to_string(static_cast<unsigned long long>(v)); };
        out += (i ? ", " : "") + std::string("{\"pass\": \"") + jsonEscape(w.pass) + "\", \"draws\": " + n(w.draws) +
               ", \"instances\": " + n(w.instances) + ", \"indices\": " + n(w.indices) + ", \"vertices\": " +
               n(w.vertices) + ", \"pixels\": " + n(w.pixels) + ", \"threads\": " + n(w.threads) +
               ", \"lights\": " + std::to_string(w.lights) + "}";
    }
    return out + "]";
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
           summaryToJson(p.gpuMs) + workToJson(p.work) + "}";
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
    s.p95  = rank(0.95);
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
                  "frame %.3f ms (p95 %.3f p99 %.3f) | CPU %.3f ms (p99 %.3f) | GPU %.3f ms (p99 %.3f) | "
                  "wait %.3f ms (p99 %.3f) | GPU allocations %llu | CPU heap %+lld blocks %+lld bytes",
                  r.bench.c_str(), r.width, r.height, r.vsync ? "on" : "off", r.ui ? "on" : "off",
                  r.frames, r.fps, r.frameMs.mean, r.frameMs.p95, r.frameMs.p99, r.cpuMs.mean, r.cpuMs.p99,
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
    if (r.scene.present) {
        // Short, `|`-separated like the rest, appended last so the fields
        // above keep their positions for the scripts that parse them.
        char extra[256];
        std::snprintf(extra, sizeof(extra),
                      " | scene %s %u inst %u buckets visible %.0f upload %.0f B cpu-cmds %.0f",
                      r.scene.mode.c_str(), r.scene.instances, r.scene.buckets, r.scene.visible.mean,
                      r.scene.uploadBytes.mean, r.scene.cpuCommands.mean);
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
    std::string graph;
    if (r.graph.present) {
        const GraphReport& g = r.graph;
        char nums[512];
        std::snprintf(nums, sizeof nums,
                      "\"passes\": %u, \"render_passes\": %u, \"memoryless\": %u, \"barriers\": %u, "
                      "\"dram_bytes\": %llu, \"heap_bytes\": %llu, \"heap_unaliased_bytes\": %llu, "
                      "\"max_live_bytes\": %llu",
                      g.passes, g.renderPasses, g.memoryless, g.barrierCount,
                      static_cast<unsigned long long>(g.dramBytes), static_cast<unsigned long long>(g.heapBytes),
                      static_cast<unsigned long long>(g.heapUnaliasedBytes),
                      static_cast<unsigned long long>(g.maxLiveBytes));
        graph = ",\n  \"graph\": {\"mode\": \"" + jsonEscape(g.mode) + "\", \"family\": \"" + jsonEscape(g.family) +
                "\", \"plan\": \"" + jsonEscape(g.plan) + "\", \"alias\": \"" + jsonEscape(g.alias) +
                "\", \"barriers_policy\": \"" + jsonEscape(g.barriers) + "\", " + nums + "}";
    }
    std::string scene;
    if (r.scene.present) {
        const SceneReport& sc = r.scene;
        auto u = [](u64 v) { return std::to_string(static_cast<unsigned long long>(v)); };
        scene = ",\n  \"scene\": {\"mode\": \"" + jsonEscape(sc.mode) + "\", \"instances\": " + u(sc.instances) +
                ", \"slots\": " + u(sc.slots) + ", \"buckets\": " + u(sc.buckets) +
                ", \"materials\": " + u(sc.materials) + ", \"commands\": " + u(sc.commands) +
                ", \"structure_changes\": " + u(sc.structureChanges) + ", \"queue_overflow\": " + u(sc.queueOverflow) +
                ", \"upload_bytes\": " + summaryToJson(sc.uploadBytes) +
                ", \"delta_records\": " + summaryToJson(sc.deltaRecords) +
                ", \"visible\": " + summaryToJson(sc.visible) +
                ", \"culled_frustum\": " + summaryToJson(sc.culledFrustum) +
                ", \"culled_distance\": " + summaryToJson(sc.culledDistance) +
                ", \"culled_size\": " + summaryToJson(sc.culledSize) +
                ", \"draw_commands\": " + summaryToJson(sc.drawCommands) +
                ", \"cpu_commands\": " + summaryToJson(sc.cpuCommands) + "}";
    }
    std::string phases;
    if (r.cpuPhases.present) {
        const CpuPhasesReport& c = r.cpuPhases;
        phases = ",\n  \"cpu_phases\": {\"sim\": " + summaryToJson(c.sim) +
                 ", \"scene_sync\": " + summaryToJson(c.sceneSync) + ", \"prepare\": " + summaryToJson(c.prepare) +
                 ", \"ui\": " + summaryToJson(c.ui) + ", \"graph\": " + summaryToJson(c.graph) +
                 ", \"submit\": " + summaryToJson(c.submit) + "}";
    }
    std::string hardware;
    if (r.hardware.present) {
        const DeviceReport& h = r.hardware;
        std::string unverified = "[";
        for (size_t i = 0; i < h.unverifiedDevices.size(); ++i) {
            unverified += (i ? ", \"" : "\"") + jsonEscape(h.unverifiedDevices[i]) + "\"";
        }
        unverified += "]";
        hardware = ",\n  \"hardware\": {\"physical_device\": \"" + jsonEscape(h.physicalDevice) +
                   "\", \"physical_family\": \"" + jsonEscape(h.physicalFamily) +
                   "\", \"memory_bytes\": " + std::to_string(static_cast<unsigned long long>(h.memoryBytes)) +
                   ", \"effective_capabilities\": \"" + jsonEscape(h.effectiveCapabilities) + "\", \"preset\": \"" +
                   jsonEscape(h.preset) + "\", \"validation_scope\": \"" + jsonEscape(h.validationScope) +
                   "\", \"unverified_devices\": " + unverified + "}";
    }
    std::string meshlets;
    if (r.meshlets.present) {
        const MeshletReport& m = r.meshlets;
        auto u = [](u64 v) { return std::to_string(static_cast<unsigned long long>(v)); };
        meshlets = ",\n  \"meshlets\": {\"path\": \"" + jsonEscape(m.path) + "\", \"cull\": \"" + jsonEscape(m.cull) +
                   "\", \"hiz_requested\": \"" + jsonEscape(m.hizRequested) + "\", \"hiz_effective\": \"" +
                   jsonEscape(m.hizEffective) + "\", \"cook\": \"" + jsonEscape(m.cook) + "\", \"meshlets\": " +
                   u(m.meshlets) + ", \"candidate_capacity\": " + u(m.candidateCapacity) +
                   ", \"overflow_frames\": " + u(m.overflowFrames) + ", \"history_resets\": " + u(m.historyResets) +
                   ", \"checks\": " + u(m.checks) + ", \"check_failures\": " + u(m.checkFailures) +
                   ", \"candidates\": " + summaryToJson(m.candidates) + ", \"drawn_a\": " + summaryToJson(m.drawnA) +
                   ", \"frustum\": " + summaryToJson(m.frustum) + ", \"cone\": " + summaryToJson(m.cone) +
                   ", \"history_rejected\": " + summaryToJson(m.historyRejected) +
                   ", \"drawn_b\": " + summaryToJson(m.drawnB) + ", \"occluded_b\": " + summaryToJson(m.occludedB) +
                   ", \"primitives\": " + summaryToJson(m.primitives) + ", \"emitted\": " + summaryToJson(m.emitted) +
                   ", \"size_culled\": " + summaryToJson(m.sizeCulled) + "}";
    }
    std::string rendering;
    if (r.rendering.present) {
        const auto &v = r.rendering;
        const auto boolean = [](bool x) { return x ? "true" : "false"; };
        rendering =
            ",\n  \"rendering\": {\"offscreen\": " + std::string(boolean(v.offscreen)) + ", \"path\": \"" +
            jsonEscape(v.path) + "\", \"asset\": \"" + jsonEscape(v.asset) +
            "\", \"material_binning_requested\": " + boolean(v.materialBinning) + ", \"post\": " + boolean(v.post) +
            ", \"upscaler_effective\": \"" + jsonEscape(v.upscaler) + "\", \"tonemap\": \"" + jsonEscape(v.tonemap) +
            "\", \"input_width_last\": " + std::to_string(v.inputWidth) +
            ", \"input_height_last\": " + std::to_string(v.inputHeight) + ", \"views\": " + std::to_string(v.views) +
            ", \"frames_in_flight\": " + std::to_string(v.framesInFlight) +
            ", \"gpu_failures\": " + std::to_string(v.gpuFailures) + ", \"auto_exposure\": " + boolean(v.autoExposure) +
            ", \"exposure_last\": " + std::to_string(v.exposure) + ", \"edr\": " + boolean(v.edr) +
            ", \"display_headroom_last\": " + std::to_string(v.headroom) +
            ", \"display_potential_headroom\": " + std::to_string(v.potentialHeadroom) +
            ", \"temporal_frames_total\": " + std::to_string(v.temporalFrames) +
            ", \"native_frames_total\": " + std::to_string(v.nativeFrames) +
            ", \"history_resets_total\": " + std::to_string(v.historyResets) +
            ", \"binned_frames_total\": " + std::to_string(v.binnedFrames) +
            ", \"generic_frames_total\": " + std::to_string(v.genericFrames) +
            ", \"guide_checks\": " + std::to_string(v.guideChecks) +
            ", \"shaded_pixels_last\": " + std::to_string(v.shadedPixels) +
            ", \"reused_pixels_last\": " + std::to_string(v.reusedPixels) +
            ", \"device_allocated_bytes_last\": " + std::to_string(v.deviceAllocatedBytes) +
            ", \"engine_resource_bytes_last\": " + std::to_string(v.engineResourceBytes) +
            ", \"command_buffer_rebuilds_total\": " + std::to_string(v.commandBufferRebuilds) +
            ", \"command_buffer_rebuilds_measured\": " + std::to_string(v.commandBufferRebuildsMeasured) +
            ", \"guide_failures\": " + std::to_string(v.guideFailures) + "}";
    }
    return head + "  \"frame_ms\": " + summaryToJson(r.frameMs) + ",\n" + "  \"cpu_ms\": " + summaryToJson(r.cpuMs) +
           ",\n" + "  \"gpu_ms\": " + summaryToJson(r.gpuMs) + ",\n" + "  \"wait_ms\": " + summaryToJson(r.waitMs) +
           (r.pipelinesJson.empty() ? std::string() : ",\n  \"pipelines\": " + r.pipelinesJson) + ",\n" + timing +
           graph + scene + phases + hardware + meshlets + rendering + "\n}\n";
}

} // namespace phosphor
