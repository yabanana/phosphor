// B-23: Neural Engine through Core ML.  Models are written in code (a
// NeuralNetwork .mlmodel protobuf encoded by hand, as in the OPT-0 spike):
//   conv    4 x [conv 3x3 256->256 + ReLU] on 256x64x64   (19.3 GFLOP)
//   conv8   the same with 8 layers                        (2x layers control)
//   mlp     4 x [innerProduct 2048->2048 + ReLU]          (small, latency bound)
//   pix     6 x [conv 1x1 64->64 + ReLU] on 64x128x128    (per-pixel net)
// For computeUnits cpuOnly / cpuAndGPU / cpuAndNeuralEngine / all: load time,
// latency (batch 1, p50/p90), throughput with an MLArrayBatchProvider, effective
// TFLOPS, MLComputePlan preferred device per layer; then ANE inference while
// the GPU runs a heavy compute load (slowdown of both).
//
// Serves S-ANE-1 of docs/APPLE_SOC_PLAYBOOK.md (§14).

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "harness.h"

#include <atomic>
#include <cstdlib>
#include <cmath>
#include <thread>

namespace soc {
namespace {

// --- minimal protobuf writer (Core ML model spec) ------------------------------
struct Pb {
    std::string b;
    void varint(uint64_t v) {
        while (v >= 0x80) { b.push_back(char((v & 0x7f) | 0x80)); v >>= 7; }
        b.push_back(char(v));
    }
    void tag(int field, int wire) { varint(uint64_t(field) << 3 | uint64_t(wire)); }
    void u(int field, uint64_t v) { tag(field, 0); varint(v); }
    void bytes(int field, const std::string& s) { tag(field, 2); varint(s.size()); b += s; }
    void msg(int field, const Pb& m) { bytes(field, m.b); }
    void floats(int field, const std::vector<float>& v) {
        bytes(field, std::string(reinterpret_cast<const char*>(v.data()), v.size() * 4));
    }
    void i64s(int field, const std::vector<int64_t>& v) {
        Pb p;
        for (int64_t x : v) p.varint(uint64_t(x));
        bytes(field, p.b);
    }
};

struct ModelSpec {
    const char* name;
    bool mlp;                 // innerProduct instead of convolution
    int c, h, w;              // input shape (mlp: c = features, h = w = 1)
    int layers;
    int k;                    // kernel size (conv)
    [[nodiscard]] double flop() const {
        return mlp ? 2.0 * c * c * layers : 2.0 * c * c * k * k * h * w * layers;
    }
    [[nodiscard]] std::vector<int64_t> shape() const { return mlp ? std::vector<int64_t>{c} : std::vector<int64_t>{c, h, w}; }
    [[nodiscard]] size_t elements() const { return size_t(c) * size_t(mlp ? 1 : h * w); }
};

Pb arrayFeature(const char* name, const std::vector<int64_t>& shape) {
    Pb arr;
    arr.i64s(1, shape);
    arr.u(2, 65568); // FLOAT32
    Pb type;
    type.msg(5, arr);
    Pb fd;
    fd.bytes(1, name);
    fd.msg(3, type);
    return fd;
}

std::string buildModel(const ModelSpec& s) {
    Pb desc;
    desc.msg(1, arrayFeature("x", s.shape()));
    desc.msg(10, arrayFeature("y", s.shape()));
    Pb nn;
    // He-style uniform weights: the signal keeps its magnitude through the
    // layers (FP16 on the ANE must not underflow).
    const size_t fanIn = s.mlp ? size_t(s.c) : size_t(s.c) * s.k * s.k;
    std::vector<float> w(size_t(s.c) * fanIn);
    uint32_t seed = 1;
    const float a = std::sqrt(6.0f / float(fanIn));
    for (float& x : w) { seed = seed * 1664525u + 1013904223u; x = (float((seed >> 8) & 0xffff) / 65536.0f - 0.5f) * 2.0f * a; }
    std::string prev = "x";
    for (int l = 0; l < s.layers; ++l) {
        const std::string mid = "m" + std::to_string(l);
        const std::string out = l + 1 == s.layers ? "y" : "r" + std::to_string(l);
        Pb weights;
        weights.floats(1, w);
        Pb layer;
        layer.bytes(1, "layer" + std::to_string(l));
        layer.bytes(2, prev);
        layer.bytes(3, mid);
        if (s.mlp) {
            Pb ip;                  // InnerProductLayerParams
            ip.u(1, uint64_t(s.c)); // inputChannels
            ip.u(2, uint64_t(s.c)); // outputChannels
            ip.msg(20, weights);
            layer.msg(140, ip);
        } else {
            Pb conv;
            conv.u(1, uint64_t(s.c));
            conv.u(2, uint64_t(s.c));
            conv.u(10, 1);
            conv.i64s(20, {s.k, s.k});
            conv.i64s(30, {1, 1});
            conv.i64s(40, {1, 1});
            conv.msg(51, Pb{}); // same padding
            conv.msg(90, weights);
            layer.msg(100, conv);
        }
        nn.msg(1, layer);
        Pb act;
        act.msg(10, Pb{}); // ReLU
        Pb alayer;
        alayer.bytes(1, "relu" + std::to_string(l));
        alayer.bytes(2, mid);
        alayer.bytes(3, out);
        alayer.msg(130, act);
        nn.msg(1, alayer);
        prev = out;
    }
    Pb model;
    model.u(1, 4);
    model.msg(2, desc);
    model.msg(500, nn);
    return model.b;
}

const char* deviceName(id<MLComputeDeviceProtocol> d) {
    if (!d) return "none";
    if ([(id)d isKindOfClass:[MLNeuralEngineComputeDevice class]]) return "ANE";
    if ([(id)d isKindOfClass:[MLGPUComputeDevice class]]) return "GPU";
    if ([(id)d isKindOfClass:[MLCPUComputeDevice class]]) return "CPU";
    return "?";
}

struct UnitCfg {
    MLComputeUnits units;
    const char* tag;   // metric prefix
    const char* label; // Core ML name
};
constexpr UnitCfg kUnits[] = {{MLComputeUnitsCPUOnly, "cpu_coreml", "cpuOnly"},
                              {MLComputeUnitsCPUAndGPU, "gpu_coreml", "cpuAndGPU"},
                              {MLComputeUnitsCPUAndNeuralEngine, "ane", "cpuAndNeuralEngine"},
                              {MLComputeUnitsAll, "all_coreml", "all"}};

/// Preferred device of every layer, as "ANE=8 GPU=0 CPU=0" plus the conv/innerProduct layers' device.
struct PlanResult {
    std::map<std::string, int> counts;
    std::map<std::string, int> heavy; // conv / innerProduct layers only
    bool ok = false;
    std::string error;
};
PlanResult computePlan(NSURL* compiled, MLModelConfiguration* cfg) {
    __block PlanResult res;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled configuration:cfg completionHandler:^(MLComputePlan* p, NSError* e) {
        if (p && p.modelStructure.neuralNetwork) {
            for (MLModelStructureNeuralNetworkLayer* l in p.modelStructure.neuralNetwork.layers) {
                MLComputePlanDeviceUsage* du = [p computeDeviceUsageForNeuralNetworkLayer:l];
                const char* dn = deviceName(du.preferredComputeDevice);
                res.counts[dn]++;
                if (![l.name hasPrefix:@"relu"]) res.heavy[dn]++;
            }
            res.ok = true;
        } else {
            res.error = e ? e.localizedDescription.UTF8String : "no neuralNetwork structure";
        }
        dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    return res;
}
std::string planString(const PlanResult& p) {
    std::string s;
    for (auto& [k, v] : p.counts) s += k + "=" + std::to_string(v) + " ";
    return s + "(heavy layers: " + [&] {
        std::string h;
        for (auto& [k, v] : p.heavy) h += k + "=" + std::to_string(v) + " ";
        return h;
    }() + ")";
}

Stats scaleStats(const Stats& t, double k) { // rate = k / time
    Stats s = t;
    s.median = k / t.median;
    s.min = k / t.max;
    s.max = k / t.min;
    s.p10 = k / t.p90;
    s.p90 = k / t.p10;
    s.mean = k / t.mean;
    return s;
}

// --- GPU load for the concurrency test ------------------------------------------------
NSString* const kLoadSrc = @"#include <metal_stdlib>\nusing namespace metal;\n"
                            "kernel void load(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
                            "  float v = float(i & 15) * 0.0625f + 1.0f, w = 0.9990234375f;\n"
                            "  for (uint k = 0; k < iters; ++k) { v = fma(v, w, 0.0009765625f); }\n"
                            "  out[i] = v;\n}\n";

struct GpuLoad {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pso;
    id<MTLBuffer> out;
    uint32_t iters = 1024;
    static constexpr uint32_t kThreads = 1u << 20;

    explicit GpuLoad(MTL::Device* d) {
        dev = (__bridge id<MTLDevice>)d;
        queue = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kLoadSrc options:nil error:&err];
        if (!lib) throw BenchError(std::string("GPU load shader: ") + err.localizedDescription.UTF8String);
        pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"load"] error:&err];
        if (!pso) throw BenchError("GPU load pipeline");
        out = [dev newBufferWithLength:kThreads * 4 options:MTLResourceStorageModeShared];
    }
    /// One dispatch of the current length; returns the GPU time in ms.
    double once() {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e dispatchThreads:MTLSizeMake(kThreads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    }
    void calibrate(double targetMs) {
        once();
        const double t = once();
        iters = std::max<uint32_t>(64, uint32_t(double(iters) * targetMs / std::max(0.01, t)));
    }
};

void benchAne(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    const int nPred = quick ? 20 : 50;
    const int batch = quick ? 8 : 16;
    // Core ML's own GPU kernels (cpuAndGPU / all) make the Metal API + shader validation layers print
    // 'unused binding' / 'endEncoding without encoding GPU work' messages from inside Apple frameworks:
    // under --validate (MTL_SHADER_VALIDATION set by soc_bench) those two configurations are skipped.
    const bool validating = std::getenv("MTL_SHADER_VALIDATION") != nullptr;
    if (validating) rep.note("--validate run: Core ML cpuAndGPU and all configurations skipped (Apple-internal validation messages), metrics for them are absent");

    std::vector<ModelSpec> specs = {{"conv", false, 256, 64, 64, 4, 3},
                                    {"conv8", false, 256, 64, 64, 8, 3},
                                    {"mlp", true, 2048, 1, 1, 4, 1},
                                    {"pix", false, 64, 128, 128, 6, 1}};

    NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"soc_b23_%d", getpid()]];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
    struct Cleanup {
        NSString* d;
        ~Cleanup() { [[NSFileManager defaultManager] removeItemAtPath:d error:nil]; }
    } cleanup{dir};

    bool planOk = true, outputOk = true;
    std::string planDetail, outputDetail;
    std::map<std::string, double> aneLatency, cpuLatency; // model -> minimum ms (least disturbed by other clients)
    double aneConvLatency = 0;
    NSURL* convCompiled = nil;

    for (const ModelSpec& s : specs) {
        NSString* path = [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"%s.mlmodel", s.name]];
        const std::string bytes = buildModel(s);
        [[NSData dataWithBytes:bytes.data() length:bytes.size()] writeToFile:path atomically:YES];
        NSError* err = nil;
        const double c0 = nowMs();
        NSURL* compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&err];
        if (!compiled) throw BenchError(std::string("compile ") + s.name + ": " + err.localizedDescription.UTF8String);
        const double compileMs = nowMs() - c0;
        if (std::string(s.name) == "conv") convCompiled = compiled;
        rep.value(std::string("coreml.") + s.name + ".compile_ms", "ms", compileMs, {{"layers", double(s.layers)}, {"gflop", s.flop() * 1e-9}}, false);

        // Input: deterministic, non-negative (ReLU keeps signal).
        MLMultiArray* x = [[MLMultiArray alloc] initWithShape:(s.mlp ? @[ @(s.c) ] : @[ @(s.c), @(s.h), @(s.w) ]) dataType:MLMultiArrayDataTypeFloat32 error:&err];
        float* xp = static_cast<float*>(x.dataPointer);
        for (size_t i = 0; i < s.elements(); ++i) xp[i] = float(i % 97) * 0.01f;
        MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x" : x} error:&err];
        NSMutableArray<id<MLFeatureProvider>>* items = [NSMutableArray array];
        for (int i = 0; i < batch; ++i) [items addObject:in];
        MLArrayBatchProvider* batchIn = [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:items];

        // Sampled output elements (flat indices) for the CPU comparison.
        std::vector<size_t> idx;
        for (size_t i = 0; i < 2048; ++i) idx.push_back((i * 2654435761ull + 12345) % s.elements());
        std::vector<float> ref;
        double refMax = 0;

        for (const UnitCfg& u : kUnits) {
            if (validating && (u.units == MLComputeUnitsCPUAndGPU || u.units == MLComputeUnitsAll)) continue;
            MLModelConfiguration* cfg = [MLModelConfiguration new];
            cfg.computeUnits = u.units;
            const std::string base = std::string(u.tag) + "." + s.name;
            const PlanResult plan = computePlan(compiled, cfg);
            if (!plan.ok) rep.note(base + ": MLComputePlan unavailable: " + plan.error);
            // Load time: first load (may compile for the device) and the median of the next ones.
            std::vector<double> loads;
            MLModel* m = nil;
            for (int i = 0; i < (quick ? 2 : 3); ++i) {
                const double l0 = nowMs();
                m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
                loads.push_back(nowMs() - l0);
                if (!m) throw BenchError(std::string("load failed ") + base + ": " + err.localizedDescription.UTF8String);
            }
            rep.value(base + ".load_first_ms", "ms", loads[0], {}, false);
            rep.value(base + ".load_ms", "ms", phosphor::soc::computeStats(std::vector<double>(loads.begin() + 1, loads.end())).median, {}, false);
            // Warm-up: at least 5 predictions and 300 ms (the ANE/GPU clocks ramp up under load).
            {
                const double w0 = nowMs();
                for (int i = 0; i < 5 || nowMs() - w0 < 300.0; ++i) [m predictionFromFeatures:in error:&err];
            }
            // Latency, batch 1.
            id<MLFeatureProvider> last = nil;
            const Stats lat = ctx.measure(
                [&] {
                    const double p0 = nowMs();
                    last = [m predictionFromFeatures:in error:&err];
                    return nowMs() - p0;
                },
                uint32_t(nPred));
            if (!last) throw BenchError(std::string("prediction failed ") + base);
            std::map<std::string, double> params = {{"layers", double(s.layers)}, {"gflop", s.flop() * 1e-9}, {"batch", 1}};
            rep.metric(base + ".latency_ms", "ms", lat, params, false);
            rep.metric(base + ".tflops", "TFLOPS", scaleStats(lat, s.flop() * 1e-12 / 1e-3), params);
            // Throughput with a batch provider.
            const Stats bt = ctx.measure(
                [&] {
                    const double p0 = nowMs();
                    id<MLBatchProvider> o = [m predictionsFromBatch:batchIn error:&err];
                    const double dt = nowMs() - p0;
                    if (!o || o.count != batch) throw BenchError(std::string("batch prediction failed ") + base);
                    return dt;
                },
                uint32_t(quick ? 3 : 5));
            std::map<std::string, double> bparams = {{"layers", double(s.layers)}, {"gflop", s.flop() * 1e-9}, {"batch", double(batch)}};
            rep.metric(base + ".batch_ms_per_item", "ms", scaleStats(bt, 1.0 * batch), bparams, false);
            rep.metric(base + ".batch_tflops", "TFLOPS", scaleStats(bt, s.flop() * batch * 1e-12 / 1e-3), bparams);
            // Output vs the CPU-only (FP32) reference.
            MLMultiArray* y = [last featureValueForName:@"y"].multiArrayValue;
            std::vector<float> got;
            for (size_t i : idx) got.push_back([y[i] floatValue]);
            if (u.units == MLComputeUnitsCPUOnly) {
                ref = got;
                for (float v : ref) refMax = std::max(refMax, double(std::fabs(v)));
                if (!(refMax > 0)) { outputOk = false; outputDetail += base + ": CPU reference all zero; "; }
                cpuLatency[s.name] = lat.min;
            } else if (!ref.empty()) {
                double maxErr = 0;
                for (size_t i = 0; i < got.size(); ++i) maxErr = std::max(maxErr, double(std::fabs(got[i] - ref[i])));
                const double rel = refMax > 0 ? maxErr / refMax : 1.0;
                rep.value(base + ".max_rel_err_vs_cpu", "ratio", rel, {}, false);
                if (!(rel < 0.05)) { outputOk = false; outputDetail += base + " err " + std::to_string(rel).substr(0, 6) + "; "; }
            }
            if (u.units == MLComputeUnitsCPUAndNeuralEngine) {
                aneLatency[s.name] = lat.min;
                if (std::string(s.name) == "conv") aneConvLatency = lat.median;
                if (!(plan.ok && plan.heavy.count("ANE") && plan.heavy.at("ANE") == s.layers && plan.heavy.size() == 1)) {
                    planOk = false;
                    planDetail += std::string(s.name) + ": " + (plan.ok ? planString(plan) : "no plan") + "; ";
                }
            }
            rep.note(base + " plan (" + u.label + "): " + (plan.ok ? planString(plan) : "unavailable"));
            ctx.log("%s: p50 %.3f ms p90 %.3f, %.2f TFLOPS, batch %.2f TFLOPS, load %.0f/%.0f ms", base.c_str(), lat.median, lat.p90,
                    s.flop() * 1e-12 / (lat.median * 1e-3), s.flop() * batch * 1e-12 / (bt.median * 1e-3), loads[0], loads.back());
        }
    }
    if (aneConvLatency > 0) rep.value("ane.latency_ms", "ms", aneConvLatency, {{"model_conv_layers", 4}, {"batch", 1}}, false);
    rep.note("ane.latency_ms = p50 latency of the 4-layer conv model with computeUnits cpuAndNeuralEngine (same as ane.conv.latency_ms)");
    rep.note("tflops = effective (model FLOP / measured time, incl. Core ML overhead); FP32 in/out, Core ML runs FP16 on ANE/GPU");

    // --- concurrency: ANE inference while the GPU runs a heavy compute load -----------------
    double aneSlowdown = 0, gpuSlowdown = 0;
    bool concOk = false;
    if (convCompiled) {
        NSError* err = nil;
        MLModelConfiguration* cfg = [MLModelConfiguration new];
        cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel* m = [MLModel modelWithContentsOfURL:convCompiled configuration:cfg error:&err];
        const ModelSpec& s = specs[0];
        MLMultiArray* x = [[MLMultiArray alloc] initWithShape:@[ @(s.c), @(s.h), @(s.w) ] dataType:MLMultiArrayDataTypeFloat32 error:&err];
        float* xp = static_cast<float*>(x.dataPointer);
        for (size_t i = 0; i < s.elements(); ++i) xp[i] = float(i % 97) * 0.01f;
        MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x" : x} error:&err];
        for (int i = 0; i < 5; ++i) [m predictionFromFeatures:in error:&err];

        GpuLoad gpu(ctx.device());
        gpu.calibrate(2.0);
        const double window = quick ? 0.6 : 2.0; // seconds per phase
        auto gpuLoop = [&](std::atomic<bool>& stop, std::atomic<u64>& count, std::atomic<double>& busyMs) {
            while (!stop.load()) { busyMs.store(busyMs.load() + gpu.once()); count.fetch_add(1); }
        };
        auto aneLoop = [&](double seconds, std::vector<double>& samples) {
            const double t0 = nowMs();
            while (nowMs() - t0 < seconds * 1e3) {
                const double p0 = nowMs();
                [m predictionFromFeatures:in error:&err];
                samples.push_back(nowMs() - p0);
            }
        };
        // 1. GPU load alone: dispatches per second.
        std::atomic<bool> stop{false};
        std::atomic<u64> cnt{0};
        std::atomic<double> busy{0};
        double gpuAlone, gpuKernelMs;
        {
            std::thread th(gpuLoop, std::ref(stop), std::ref(cnt), std::ref(busy));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const u64 c0 = cnt.load();
            const double t0 = nowMs();
            std::this_thread::sleep_for(std::chrono::duration<double>(window));
            gpuAlone = double(cnt.load() - c0) / ((nowMs() - t0) * 1e-3);
            stop.store(true);
            th.join();
            gpuKernelMs = 1e3 / gpuAlone;
        }
        // 2. ANE alone.
        std::vector<double> alone;
        aneLoop(window, alone);
        // 3. Both.
        std::vector<double> both;
        stop.store(false);
        cnt.store(0);
        double gpuBoth;
        {
            std::thread th(gpuLoop, std::ref(stop), std::ref(cnt), std::ref(busy));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const u64 c0 = cnt.load();
            const double t0 = nowMs();
            aneLoop(window, both);
            gpuBoth = double(cnt.load() - c0) / ((nowMs() - t0) * 1e-3);
            stop.store(true);
            th.join();
        }
        if (!alone.empty() && !both.empty() && gpuAlone > 0 && gpuBoth > 0) {
            const Stats sa = phosphor::soc::computeStats(alone), sb = phosphor::soc::computeStats(both);
            aneSlowdown = sb.median / sa.median;
            gpuSlowdown = gpuAlone / gpuBoth;
            concOk = true;
            rep.metric("concurrency.ane_latency_ms.alone", "ms", sa, {{"gpu_load", 0}}, false);
            rep.metric("concurrency.ane_latency_ms.with_gpu_load", "ms", sb, {{"gpu_load", 1}}, false);
            rep.value("concurrency.ane_slowdown", "ratio", aneSlowdown, {}, false);
            rep.value("concurrency.gpu_dispatches_per_s.alone", "1/s", gpuAlone, {{"kernel_ms", gpuKernelMs}});
            rep.value("concurrency.gpu_dispatches_per_s.with_ane", "1/s", gpuBoth, {{"kernel_ms", gpuKernelMs}});
            rep.value("concurrency.gpu_slowdown", "ratio", gpuSlowdown, {}, false);
            ctx.log("concurrency: ANE %.3f -> %.3f ms (x%.2f), GPU load %.0f -> %.0f dispatches/s (x%.2f)", sa.median, sb.median, aneSlowdown,
                    gpuAlone, gpuBoth, gpuSlowdown);
            rep.note("concurrency: the GPU load is a continuous ~2 ms FMA compute dispatch loop on its own MTLCommandQueue (Metal 3 API, separate thread); ANE inference = conv model, cpuAndNeuralEngine, batch 1; " +
                     std::to_string(int(window * 1000)) + " ms per phase");
        }
    }
    if (!concOk) rep.status(Status::Partial, "concurrency test produced no samples");

    // --- controls -----------------------------------------------------------------------------
    const double r8ane = aneLatency["conv8"] / aneLatency["conv"], r8cpu = cpuLatency["conv8"] / cpuLatency["conv"];
    // ANE latency = fixed Core ML/ANE overhead + compute: doubling the layers must add
    // clearly (ratio > 1.25) but cannot exceed 2x; CPU-only is compute bound (~2x) but shares cores with every other process.
    const bool layersOk = r8ane > 1.25 && r8ane < 2.3 && r8cpu > 1.3 && r8cpu < 3.5;
    if (aneLatency["conv8"] > aneLatency["conv"]) {
        const ModelSpec& c4 = specs[0];
        rep.value("ane.conv.marginal_tflops", "TFLOPS", c4.flop() * 1e-12 / ((aneLatency["conv8"] - aneLatency["conv"]) * 1e-3), {{"layers_added", 4}});
        rep.value("ane.conv.fixed_overhead_ms", "ms", 2 * aneLatency["conv"] - aneLatency["conv8"], {}, false);
        rep.note("ane.conv.marginal_tflops = FLOP of 4 layers / (min latency(8 layers) - min latency(4 layers)); fixed_overhead_ms = latency extrapolated to 0 layers");
    }
    rep.negative(planOk && outputOk && layersOk,
                 std::string("MLComputePlan puts every conv/innerProduct layer on ANE for cpuAndNeuralEngine: ") + (planOk ? "yes" : "NO (" + planDetail + ")") +
                     "; sampled outputs of GPU/ANE/all within 5% of max|CPU FP32|: " + (outputOk ? "yes" : "NO (" + outputDetail + ")") +
                     "; 8 vs 4 layers min-latency ratio: ANE " + std::to_string(r8ane).substr(0, 4) + "x (1.25..2.3, fixed overhead), CPU " + std::to_string(r8cpu).substr(0, 4) +
                     "x (1.3..3.5, cores shared with other processes)" + (layersOk ? "" : " OUT OF RANGE"));
    if (!outputOk) rep.status(Status::Failed, "Core ML output differs from the CPU reference");
    (void)cpuLatency;
}

} // namespace

SOC_BENCH("B-23", "ane.coreml", "Neural Engine via Core ML: latency, throughput, TFLOPS, concurrency with GPU load", benchAne);

} // namespace soc
