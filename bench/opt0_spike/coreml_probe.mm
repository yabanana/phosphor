// OPT-0 spike 3: can the Neural Engine be measured without external models?
// A MEASUREMENT PROBE, not engine code.
//
// Builds a Core ML NeuralNetwork model in code (the .mlmodel protobuf is
// encoded by hand: 4 x [conv 3x3 256->256 "same" + ReLU] on a 256x64x64
// input), compiles it, asks MLComputePlan which device each layer prefers,
// and times predictions for every compute-unit setting.
//
// Usage: coreml_probe [iterations]

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <mach/mach_time.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// --- minimal protobuf writer -------------------------------------------------
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
    void floats(int field, const std::vector<float>& v) {   // packed repeated float
        std::string s(reinterpret_cast<const char*>(v.data()), v.size() * 4);
        bytes(field, s);
    }
    void i64s(int field, const std::vector<int64_t>& v) {   // packed repeated int64
        Pb p;
        for (int64_t x : v) p.varint(uint64_t(x));
        bytes(field, p.b);
    }
};

constexpr int kC = 256, kH = 64, kW = 64, kLayers = 4;

Pb arrayFeature(const char* name) {
    Pb arr;                       // ArrayFeatureType
    arr.i64s(1, {kC, kH, kW});    // shape
    arr.u(2, 65568);              // FLOAT32
    Pb type;                      // FeatureType
    type.msg(5, arr);             // multiArrayType
    Pb fd;                        // FeatureDescription
    fd.bytes(1, name);
    fd.msg(3, type);
    return fd;
}

std::string buildModel() {
    Pb desc;
    desc.msg(1, arrayFeature("x"));
    desc.msg(10, arrayFeature("y"));
    Pb nn;
    std::vector<float> w(size_t(kC) * kC * 9);
    uint32_t seed = 1;
    for (float& x : w) { seed = seed * 1664525u + 1013904223u; x = (float((seed >> 8) & 0xffff) / 65536.0f - 0.5f) * 0.02f; }
    std::string prev = "x";
    for (int l = 0; l < kLayers; ++l) {
        const std::string convOut = "c" + std::to_string(l);
        const std::string out = l + 1 == kLayers ? "y" : "r" + std::to_string(l);
        Pb weights;
        weights.floats(1, w);
        Pb conv;                  // ConvolutionLayerParams
        conv.u(1, kC);            // outputChannels
        conv.u(2, kC);            // kernelChannels
        conv.u(10, 1);            // nGroups
        conv.i64s(20, {3, 3});    // kernelSize
        conv.i64s(30, {1, 1});    // stride
        conv.i64s(40, {1, 1});    // dilationFactor
        conv.msg(51, Pb{});       // same padding
        conv.msg(90, weights);
        Pb layer;
        layer.bytes(1, "conv" + std::to_string(l));
        layer.bytes(2, prev);
        layer.bytes(3, convOut);
        layer.msg(100, conv);
        nn.msg(1, layer);
        Pb relu;
        Pb act;
        act.msg(10, relu);        // ActivationParams.ReLU
        Pb alayer;
        alayer.bytes(1, "relu" + std::to_string(l));
        alayer.bytes(2, convOut);
        alayer.bytes(3, out);
        alayer.msg(130, act);
        nn.msg(1, alayer);
        prev = out;
    }
    Pb model;
    model.u(1, 4);                // specificationVersion
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

double nowMs() {
    static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    return double(mach_absolute_time()) * tb.numer / tb.denom * 1e-6;
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        const int iters = argc > 1 ? std::atoi(argv[1]) : 50;
        NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:@"opt0_coreml_probe"];
        [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
        NSString* path = [dir stringByAppendingPathComponent:@"conv4.mlmodel"];
        const std::string bytes = buildModel();
        [[NSData dataWithBytes:bytes.data() length:bytes.size()] writeToFile:path atomically:YES];
        NSError* err = nil;
        const double c0 = nowMs();
        NSURL* compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&err];
        if (!compiled) { std::printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        std::printf("model: %zu bytes, compiled in %.1f ms, %.2f GFLOP per prediction\n", bytes.size(), nowMs() - c0,
                    2.0 * kC * kC * 9 * kH * kW * kLayers * 1e-9);

        const struct { MLComputeUnits u; const char* n; } units[] = {
            {MLComputeUnitsCPUOnly, "cpuOnly"}, {MLComputeUnitsCPUAndGPU, "cpuAndGPU"},
            {MLComputeUnitsCPUAndNeuralEngine, "cpuAndNeuralEngine"}, {MLComputeUnitsAll, "all"}};

        MLMultiArray* x = [[MLMultiArray alloc] initWithShape:@[@(kC), @(kH), @(kW)] dataType:MLMultiArrayDataTypeFloat32 error:&err];
        float* xp = static_cast<float*>(x.dataPointer);
        for (int i = 0; i < kC * kH * kW; ++i) xp[i] = float(i % 97) * 0.01f;
        MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x" : x} error:&err];

        for (auto& cu : units) {
            MLModelConfiguration* cfg = [MLModelConfiguration new];
            cfg.computeUnits = cu.u;
            // Which device does each layer prefer?
            __block std::string plan;
            dispatch_semaphore_t sem = dispatch_semaphore_create(0);
            [MLComputePlan loadContentsOfURL:compiled configuration:cfg completionHandler:^(MLComputePlan* p, NSError* e) {
                if (p && p.modelStructure.neuralNetwork) {
                    for (MLModelStructureNeuralNetworkLayer* l in p.modelStructure.neuralNetwork.layers) {
                        MLComputePlanDeviceUsage* du = [p computeDeviceUsageForNeuralNetworkLayer:l];
                        plan += std::string(" ") + l.name.UTF8String + "=" + deviceName(du.preferredComputeDevice);
                    }
                } else {
                    plan = std::string(" plan unavailable: ") + (e ? e.localizedDescription.UTF8String : "?");
                }
                dispatch_semaphore_signal(sem);
            }];
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);

            const double l0 = nowMs();
            MLModel* m = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
            const double loadMs = nowMs() - l0;
            if (!m) { std::printf("%s: load failed %s\n", cu.n, err.localizedDescription.UTF8String); continue; }
            for (int i = 0; i < 5; ++i) [m predictionFromFeatures:in error:&err];   // warm-up
            std::vector<double> t;
            float check = 0;
            for (int i = 0; i < iters; ++i) {
                const double p0 = nowMs();
                id<MLFeatureProvider> o = [m predictionFromFeatures:in error:&err];
                t.push_back(nowMs() - p0);
                MLMultiArray* y = [o featureValueForName:@"y"].multiArrayValue;
                check = y ? [y[123] floatValue] : -1;
            }
            std::sort(t.begin(), t.end());
            const double med = t[t.size() / 2];
            std::printf("%-20s load %7.1f ms | predict p50 %7.3f ms p10 %7.3f p90 %7.3f | %6.2f TFLOPS eff | y[123]=%.5f |%s\n",
                        cu.n, loadMs, med, t[t.size() / 10], t[t.size() * 9 / 10],
                        2.0 * kC * kC * 9 * kH * kW * kLayers / (med * 1e-3) * 1e-12, check, plan.c_str());
        }
    }
    return 0;
}
