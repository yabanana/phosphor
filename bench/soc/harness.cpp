// OPT-0.1 SoC suite harness (see harness.h).

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include "harness.h"

#include "core/process.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#ifndef SOC_SHADER_DIR
#define SOC_SHADER_DIR "bench/soc/shaders"
#endif
#ifndef SOC_SOURCE_DIR
#define SOC_SOURCE_DIR "."
#endif
#ifndef SOC_SDK_VERSION
#define SOC_SDK_VERSION "unknown"
#endif

extern "C" {
typedef struct IOReportSubscription* IOReportSubscriptionRef;
CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
void IOReportMergeChannels(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
IOReportSubscriptionRef IOReportCreateSubscription(void*, CFMutableDictionaryRef, CFMutableDictionaryRef*, uint64_t,
                                                   CFTypeRef);
CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
int32_t IOReportStateGetCount(CFDictionaryRef);
CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);
int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t*);
}

namespace soc {

namespace {

constexpr u32 kHeapEntries = 4096;
constexpr u32 kBusyThreads = 1u << 20;
constexpr u32 kBusyIters   = 1000; // ~0.77 ms at the top P-state of an M5 Max

NS::String* nsstr(const std::string& s) { return NS::String::string(s.c_str(), NS::UTF8StringEncoding); }

std::string errorText(NS::Error* e) {
    return e && e->localizedDescription() ? e->localizedDescription()->utf8String() : "unknown error";
}

std::string cfString(CFStringRef r) {
    if (!r) return "";
    char buf[256];
    return CFStringGetCString(r, buf, sizeof(buf), kCFStringEncodingUTF8) ? buf : "";
}

std::string sysctlString(const char* name) {
    char buf[256] = {};
    size_t len = sizeof(buf);
    if (sysctlbyname(name, buf, &len, nullptr, 0) != 0) return "";
    return buf;
}

u64 sysctlU64(const char* name) {
    u64 v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
    if (len == 4) return static_cast<u32>(v);
    return v;
}

// GPU P-state table: pmgr "voltage-states9" = (Hz, mV) pairs, first = off.
std::vector<double> readPStateTable() {
    std::vector<double> mhz;
    io_service_t pmgr = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("pmgr"));
    if (!pmgr) return mhz;
    CFTypeRef data = IORegistryEntryCreateCFProperty(pmgr, CFSTR("voltage-states9"), kCFAllocatorDefault, 0);
    if (data && CFGetTypeID(data) == CFDataGetTypeID()) {
        const auto* bytes = CFDataGetBytePtr(static_cast<CFDataRef>(data));
        const CFIndex n = CFDataGetLength(static_cast<CFDataRef>(data)) / 8;
        for (CFIndex i = 0; i < n; ++i) {
            u32 hz = 0;
            std::memcpy(&hz, bytes + i * 8, 4);
            if (hz) mhz.push_back(double(hz) / 1e6);
        }
    }
    if (data) CFRelease(data);
    IOObjectRelease(pmgr);
    return mhz;
}

u32 gpuCoreCount() {
    io_service_t agx = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
    if (!agx) return 0;
    u32 cores = 0;
    CFTypeRef v = IORegistryEntryCreateCFProperty(agx, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (v && CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue(static_cast<CFNumberRef>(v), kCFNumberSInt32Type, &cores);
    if (v) CFRelease(v);
    IOObjectRelease(agx);
    return cores;
}

void powerSource(std::string& source, i32& percent) {
    source  = "unknown";
    percent = -1;
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return;
    CFStringRef type = IOPSGetProvidingPowerSourceType(info);
    if (type) source = CFStringCompare(type, CFSTR(kIOPMACPowerKey), 0) == kCFCompareEqualTo ? "AC" : "Battery";
    CFArrayRef list = IOPSCopyPowerSourcesList(info);
    for (CFIndex i = 0; list && i < CFArrayGetCount(list); ++i) {
        CFDictionaryRef d = IOPSGetPowerSourceDescription(info, CFArrayGetValueAtIndex(list, i));
        if (!d) continue;
        auto cap = static_cast<CFNumberRef>(CFDictionaryGetValue(d, CFSTR(kIOPSCurrentCapacityKey)));
        if (cap) CFNumberGetValue(cap, kCFNumberSInt32Type, &percent);
    }
    if (list) CFRelease(list);
    CFRelease(info);
}

struct BusyParams {
    u32 iters, zero;
};

} // namespace

u64 xorshift64(u64& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

double nowMs() {
    static const mach_timebase_info_data_t tb = [] {
        mach_timebase_info_data_t t;
        mach_timebase_info(&t);
        return t;
    }();
    return double(mach_absolute_time()) * tb.numer / tb.denom * 1e-6;
}

std::string thermalStateName() {
    switch (NS::ProcessInfo::processInfo()->thermalState()) {
    case NS::ProcessInfoThermalStateNominal:  return "nominal";
    case NS::ProcessInfoThermalStateFair:     return "fair";
    case NS::ProcessInfoThermalStateSerious:  return "serious";
    case NS::ProcessInfoThermalStateCritical: return "critical";
    }
    return "unknown";
}

// --- GpuState -------------------------------------------------------------------

GpuState::GpuState() {
    mhz_ = readPStateTable();
    CFDictionaryRef states = IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0);
    CFDictionaryRef energy = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), nullptr, 0, 0, 0);
    if (!states) return;
    CFMutableDictionaryRef channels = CFDictionaryCreateMutableCopy(nullptr, 0, states);
    if (energy) IOReportMergeChannels(channels, energy, nullptr);
    CFMutableDictionaryRef subbed = nullptr;
    sub_    = IOReportCreateSubscription(nullptr, channels, &subbed, 0, nullptr);
    subbed_ = subbed;
    CFRelease(channels);
    CFRelease(states);
    if (energy) CFRelease(energy);
}

GpuState::~GpuState() {
    if (start_) CFRelease(static_cast<CFDictionaryRef>(start_));
    if (subbed_) CFRelease(static_cast<CFMutableDictionaryRef>(subbed_));
    if (sub_) CFRelease(static_cast<CFTypeRef>(sub_));
}

void GpuState::begin() {
    if (!sub_) return;
    if (start_) CFRelease(static_cast<CFDictionaryRef>(start_));
    start_     = (void*)IOReportCreateSamples(static_cast<IOReportSubscriptionRef>(sub_),
                                              static_cast<CFMutableDictionaryRef>(subbed_), nullptr);
    startTime_ = nowMs();
}

phosphor::soc::GpuWindow GpuState::end() {
    phosphor::soc::GpuWindow w;
    if (!sub_ || !start_) return w;
    CFDictionaryRef now = IOReportCreateSamples(static_cast<IOReportSubscriptionRef>(sub_),
                                                static_cast<CFMutableDictionaryRef>(subbed_), nullptr);
    const double secs = (nowMs() - startTime_) * 1e-3;
    CFDictionaryRef d = IOReportCreateSamplesDelta(static_cast<CFDictionaryRef>(start_), now, nullptr);
    CFRelease(static_cast<CFDictionaryRef>(start_));
    start_ = nullptr;
    CFRelease(now);
    auto arr = static_cast<CFArrayRef>(CFDictionaryGetValue(d, CFSTR("IOReportChannels")));
    for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
        auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(arr, i));
        const std::string name = cfString(IOReportChannelGetChannelName(c));
        if (name == "GPUPH") {
            const int32_t n = IOReportStateGetCount(c);
            double total = 0, busy = 0, weighted = 0;
            int top = 0;
            std::vector<double> res(size_t(std::max(n, 0)));
            for (int32_t q = 0; q < n; ++q) {
                res[size_t(q)] = double(IOReportStateGetResidency(c, q));
                total += res[size_t(q)];
                if (q > 0 && res[size_t(q)] > 0) top = q;
            }
            for (int32_t q = 1; q < n; ++q) {
                busy += res[size_t(q)];
                const size_t k = size_t(q - 1);
                if (k < mhz_.size()) weighted += res[size_t(q)] * mhz_[k];
            }
            // "Top" = the highest state of the table (P13 on M5 Max) when the
            // table is known, else the highest state visited.
            const int topIndex = mhz_.empty() ? top : int(mhz_.size());
            w.topStateShare = busy > 0 && topIndex < n ? res[size_t(topIndex)] / busy : 0;
            w.meanMHz       = busy > 0 && !mhz_.empty() ? weighted / busy : -1;
            w.activeShare   = total > 0 ? busy / total : 0;
        } else if (name == "GPU Energy") {
            w.watts = secs > 0 ? double(IOReportSimpleGetIntegerValue(c, nullptr)) * 1e-9 / secs : -1;
        }
    }
    CFRelease(d);
    return w;
}

// --- Context --------------------------------------------------------------------

struct Context::Impl {
    std::map<std::string, MTL::Library*> libraries;
    std::map<std::string, MTL::ComputePipelineState*> pipelines;
    // Per benchmark.
    std::vector<NS::Object*> owned;
    std::vector<MTL::Allocation*> resident;
    NS::AutoreleasePool* pool = nullptr;
    double lastGpuMs = 0;
};

Context::Context(const Options& options) : options_(options), impl_(std::make_unique<Impl>()) {
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_ || !device_->supportsFamily(MTL::GPUFamilyMetal4)) throw BenchError("no Metal 4 device");
    queue_ = device_->newMTL4CommandQueue();
    NS::Error* err = nullptr;
    MTL4::CompilerDescriptor* cd = MTL4::CompilerDescriptor::alloc()->init();
    compiler_ = device_->newCompiler(cd, &err);
    cd->release();
    if (!compiler_) throw BenchError("newCompiler: " + errorText(err));
    allocator_ = device_->newCommandAllocator();
    cmd_       = device_->newCommandBuffer();
    event_     = device_->newSharedEvent();
    MTL::ResidencySetDescriptor* rd = MTL::ResidencySetDescriptor::alloc()->init();
    rd->setLabel(nsstr("soc_bench"));
    residency_ = device_->newResidencySet(rd, &err);
    rd->release();
    if (!residency_) throw BenchError("newResidencySet: " + errorText(err));
    queue_->addResidencySet(residency_);
    MTL4::CounterHeapDescriptor* hd = MTL4::CounterHeapDescriptor::alloc()->init();
    hd->setType(MTL4::CounterHeapTypeTimestamp);
    hd->setCount(kHeapEntries);
    heap_ = device_->newCounterHeap(hd, &err);
    hd->release();
    if (!heap_) throw BenchError("newCounterHeap: " + errorText(err));
    tickNs_ = 1e9 / double(device_->queryTimestampFrequency());
    MTL4::ArgumentTableDescriptor* ad = MTL4::ArgumentTableDescriptor::alloc()->init();
    ad->setMaxBufferBindCount(8);
    ad->setMaxTextureBindCount(8);
    ad->setMaxSamplerStateBindCount(4);
    table_ = device_->newArgumentTable(ad, &err);
    ad->release();
    if (!table_) throw BenchError("newArgumentTable: " + errorText(err));
    scratch_ = device_->newBuffer(64 * 1024, MTL::ResourceStorageModePrivate);
    busyParams_ = device_->newBuffer(256, MTL::ResourceStorageModeShared);
    const BusyParams bp{kBusyIters, 0};
    std::memcpy(busyParams_->contents(), &bp, sizeof(bp));
    residency_->addAllocation(scratch_);
    residency_->addAllocation(busyParams_);
    residency_->commit();
    MTL::Library* lib = library("harness.metal");
    anchor_ = compute(lib, "soc_anchor");
    busy_   = compute(lib, "soc_busy");
}

Context::~Context() {
    for (auto& [k, p] : impl_->pipelines) p->release();
    for (auto& [k, l] : impl_->libraries) l->release();
    if (scratch_) scratch_->release();
    if (busyParams_) busyParams_->release();
    if (table_) table_->release();
    if (heap_) heap_->release();
    if (residency_) residency_->release();
    if (event_) event_->release();
    if (cmd_) cmd_->release();
    if (allocator_) allocator_->release();
    if (compiler_) compiler_->release();
    if (queue_) queue_->release();
    if (device_) device_->release();
}

bool Context::apple10() const { return !options_.forceApple9 && device_->supportsFamily(MTL::GPUFamilyApple10); }

MTL::Library* Context::library(const std::string& file, bool fastMath) {
    const std::string key = file + (fastMath ? "" : ":safe");
    auto it = impl_->libraries.find(key);
    if (it != impl_->libraries.end()) return it->second;
    const char* env = std::getenv("SOC_SHADER_DIR");
    const std::string path = std::string(env ? env : SOC_SHADER_DIR) + "/" + file;
    std::ifstream in(path);
    if (!in) throw BenchError("cannot read shader " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    NS::Error* err = nullptr;
    MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
    opts->setLanguageVersion(MTL::LanguageVersion4_0);
    if (!fastMath) opts->setMathMode(MTL::MathModeSafe);
    MTL::Library* lib = device_->newLibrary(nsstr(ss.str()), opts, &err);
    opts->release();
    if (!lib) throw BenchError("MSL " + file + ": " + errorText(err));
    impl_->libraries[key] = lib;
    return lib;
}

MTL4::LibraryFunctionDescriptor* Context::function(MTL::Library* library, const std::string& name) {
    MTL4::LibraryFunctionDescriptor* f = MTL4::LibraryFunctionDescriptor::alloc()->init();
    f->setLibrary(library);
    f->setName(nsstr(name));
    keep(f);
    return f;
}

MTL::ComputePipelineState* Context::compute(MTL::Library* library, const std::string& name,
                                            const MTL::FunctionConstantValues* constants) {
    std::string key = std::to_string(reinterpret_cast<uintptr_t>(library)) + ":" + name;
    if (constants) key += ":" + std::to_string(reinterpret_cast<uintptr_t>(constants));
    auto it = impl_->pipelines.find(key);
    if (it != impl_->pipelines.end()) return it->second;
    NS::Error* err = nullptr;
    MTL4::LibraryFunctionDescriptor* f = MTL4::LibraryFunctionDescriptor::alloc()->init();
    f->setLibrary(library);
    f->setName(nsstr(name));
    MTL4::FunctionDescriptor* fd = f;
    MTL4::SpecializedFunctionDescriptor* sf = nullptr;
    if (constants) {
        sf = MTL4::SpecializedFunctionDescriptor::alloc()->init();
        sf->setFunctionDescriptor(f);
        sf->setConstantValues(constants);
        fd = sf;
    }
    MTL4::ComputePipelineDescriptor* d = MTL4::ComputePipelineDescriptor::alloc()->init();
    d->setComputeFunctionDescriptor(fd);
    MTL::ComputePipelineState* p = compiler_->newComputePipelineState(d, nullptr, &err);
    d->release();
    if (sf) sf->release();
    f->release();
    if (!p) throw BenchError("compute pipeline " + name + ": " + errorText(err));
    // Pipelines with constants are per benchmark (the constants object is
    // the caller's); plain ones live for the whole process.
    if (constants) keep(p);
    else impl_->pipelines[key] = p;
    return p;
}

MTL::RenderPipelineState* Context::render(const MTL4::RenderPipelineDescriptor* desc) {
    NS::Error* err = nullptr;
    MTL::RenderPipelineState* p = compiler_->newRenderPipelineState(desc, nullptr, &err);
    if (!p) throw BenchError("render pipeline: " + errorText(err));
    keep(p);
    return p;
}

void Context::keep(NS::Object* object) {
    if (object) impl_->owned.push_back(object);
}

void Context::adopt(MTL::Allocation* allocation) {
    if (!allocation) return;
    residency_->addAllocation(allocation);
    impl_->resident.push_back(allocation);
    keep(allocation);
    residency_->commit();
}

void Context::commitResidency() { residency_->commit(); }

MTL::Buffer* Context::buffer(size_t bytes, MTL::ResourceOptions options) {
    MTL::Buffer* b = device_->newBuffer(std::max<size_t>(bytes, 16), options);
    if (!b) throw BenchError("newBuffer(" + std::to_string(bytes) + ") failed");
    adopt(b);
    return b;
}

MTL::Buffer* Context::randomBuffer(size_t bytes, u64 seed) {
    MTL::Buffer* b = buffer(bytes, MTL::ResourceStorageModeShared);
    u64 s = seed ? seed : 1;
    auto* w = static_cast<u64*>(b->contents());
    for (size_t i = 0; i < bytes / 8; ++i) w[i] = xorshift64(s);
    auto* tail = static_cast<u8*>(b->contents());
    for (size_t i = bytes & ~size_t(7); i < bytes; ++i) tail[i] = static_cast<u8>(xorshift64(s));
    return b;
}

MTL::Texture* Context::texture(const MTL::TextureDescriptor* desc) {
    MTL::Texture* t = device_->newTexture(desc);
    if (!t) throw BenchError("newTexture failed");
    adopt(t);
    return t;
}

MTL::Heap* Context::heap(const MTL::HeapDescriptor* desc) {
    MTL::Heap* h = device_->newHeap(desc);
    if (!h) throw BenchError("newHeap failed");
    adopt(h);
    return h;
}

MTL4::CommandBuffer* Context::newCommandBuffer() {
    MTL4::CommandBuffer* c = device_->newCommandBuffer();
    keep(c);
    return c;
}
MTL4::CommandAllocator* Context::newAllocator() {
    MTL4::CommandAllocator* a = device_->newCommandAllocator();
    keep(a);
    return a;
}
MTL4::CommandQueue* Context::newQueue() {
    MTL4::CommandQueue* q = device_->newMTL4CommandQueue();
    q->addResidencySet(residency_);
    keep(q);
    return q;
}

MTL4::CommandBuffer* Context::beginCommands() {
    allocator_->reset();
    cmd_->beginCommandBuffer(allocator_);
    return cmd_;
}

double Context::submit() {
    cmd_->endCommandBuffer();
    std::atomic<bool> got{false};
    double gpuMs = 0;
    std::string error;
    MTL4::CommitOptions* o = MTL4::CommitOptions::alloc()->init();
    o->addFeedbackHandler([&](MTL4::CommitFeedback* fb) {
        if (fb->error()) error = errorText(fb->error());
        gpuMs = (fb->GPUEndTime() - fb->GPUStartTime()) * 1000.0;
        got   = true;
    });
    const MTL4::CommandBuffer* bufs[] = {cmd_};
    queue_->commit(bufs, 1, o);
    o->release();
    queue_->signalEvent(event_, ++eventValue_);
    if (!event_->waitUntilSignaledValue(eventValue_, 60000)) throw BenchError("GPU timeout (60 s)");
    while (!got) std::this_thread::yield();
    impl_->lastGpuMs = nowMs();
    if (!error.empty()) throw BenchError("GPU error: " + error);
    return gpuMs;
}

void Context::submitAsync(MTL4::CommandQueue* queue, MTL4::CommandBuffer* cmd) {
    cmd->endCommandBuffer();
    const MTL4::CommandBuffer* bufs[] = {cmd};
    queue->commit(bufs, 1);
    queue->signalEvent(event_, ++eventValue_);
}

void Context::waitIdle() {
    if (!event_->waitUntilSignaledValue(eventValue_, 60000)) throw BenchError("GPU timeout (60 s)");
    impl_->lastGpuMs = nowMs();
}

void Context::anchorDispatch(MTL4::ComputeCommandEncoder* enc) {
    table_->setAddress(scratch_->gpuAddress(), 0);
    enc->setComputePipelineState(anchor_);
    enc->setArgumentTable(table_);
    enc->dispatchThreads(MTL::Size::Make(1, 1, 1), MTL::Size::Make(1, 1, 1));
}

std::vector<u64> Context::readTimestamps(u32 first, u32 count) {
    std::vector<u64> v(count, 0);
    NS::Data* d = heap_->resolveCounterRange(NS::Range::Make(first, count));
    if (d) std::memcpy(v.data(), d->bytes(), std::min<size_t>(d->length(), size_t(count) * 8));
    return v;
}

double Context::ticksToMs(u64 a, u64 b) const { return (double(b) - double(a)) * tickNs_ * 1e-6; }

double Context::emptySpanMs() {
    if (emptySpanMs_ < 0) {
        std::vector<double> v;
        for (int i = 0; i < 31; ++i) {
            CommandTimer t(*this);
            t.begin();
            v.push_back(t.finish());
        }
        emptySpanMs_ = phosphor::soc::computeStats(v).median;
    }
    return emptySpanMs_;
}

void Context::keepWarm(double ms) {
    const double t0 = nowMs();
    do {
        MTL4::CommandBuffer* c = beginCommands();
        MTL4::ComputeCommandEncoder* e = c->computeCommandEncoder();
        table_->setAddress(scratch_->gpuAddress(), 0);
        table_->setAddress(busyParams_->gpuAddress(), 1);
        e->setComputePipelineState(busy_);
        e->setArgumentTable(table_);
        e->dispatchThreads(MTL::Size::Make(kBusyThreads, 1, 1), MTL::Size::Make(256, 1, 1));
        e->endEncoding();
        submit();
    } while (nowMs() - t0 < ms);
}

double Context::warmUp(double maxSeconds) {
    const double t0 = nowMs();
    if (!gpu_.available()) {
        keepWarm(2000);
        return -(nowMs() - t0) * 1e-3;
    }
    keepWarm(300);
    while (true) {
        gpu_.begin();
        keepWarm(250);
        const phosphor::soc::GpuWindow w = gpu_.end();
        const double el = (nowMs() - t0) * 1e-3;
        if (w.topStateShare >= 0.95) return el;
        if (el > maxSeconds) return -el;
    }
}

Stats Context::measure(const std::function<double()>& once, u32 reps) {
    if (reps == 0) reps = options_.repetitions;
    // The GPU clocks drop after idle gaps (measured >= 16 ms): re-warm.
    if (nowMs() - impl_->lastGpuMs > 5.0) keepWarm(30.0);
    std::vector<double> v;
    v.reserve(reps);
    for (u32 i = 0; i < reps; ++i) v.push_back(once());
    return phosphor::soc::computeStats(v);
}

void Context::log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fputs("[soc] ", stderr);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
}

void Context::beginBenchmark() { impl_->pool = NS::AutoreleasePool::alloc()->init(); }

void Context::endBenchmark() {
    // Everything the benchmark made: out of the residency set, then released
    // (the GPU is idle: every submit waited).
    try {
        waitIdle();
    } catch (...) {
    }
    for (MTL::Allocation* a : impl_->resident) residency_->removeAllocation(a);
    residency_->commit();
    for (auto it = impl_->owned.rbegin(); it != impl_->owned.rend(); ++it) (*it)->release();
    impl_->owned.clear();
    impl_->resident.clear();
    if (impl_->pool) impl_->pool->release();
    impl_->pool = nullptr;
}

// --- Timers ------------------------------------------------------------------

MTL4::ComputeCommandEncoder* ComputeTimer::begin() {
    MTL4::CommandBuffer* c = ctx_.beginCommands();
    enc_ = c->computeCommandEncoder();
    ctx_.anchorDispatch(enc_);
    enc_->writeTimestamp(MTL4::TimestampGranularityPrecise, ctx_.heap_, 0);
    enc_->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
    laps_ = 0;
    return enc_;
}

void ComputeTimer::lap() {
    if (laps_ + 1 >= kHeapEntries) throw BenchError("ComputeTimer: too many laps");
    enc_->writeTimestamp(MTL4::TimestampGranularityPrecise, ctx_.heap_, ++laps_);
    enc_->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
}

std::vector<double> ComputeTimer::finish() {
    enc_->endEncoding();
    ctx_.submit();
    const std::vector<u64> t = ctx_.readTimestamps(0, laps_ + 1);
    std::vector<double> ms(laps_);
    for (u32 i = 0; i < laps_; ++i) {
        if (t[i] == 0 || t[i + 1] < t[i]) throw BenchError("ComputeTimer: invalid timestamp");
        ms[i] = ctx_.ticksToMs(t[i], t[i + 1]);
    }
    return ms;
}

MTL4::CommandBuffer* CommandTimer::begin() {
    cmd_ = ctx_.beginCommands();
    MTL4::ComputeCommandEncoder* e = cmd_->computeCommandEncoder();
    ctx_.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, ctx_.heap_, 0);
    // Later encoders wait for the anchor (F4: without it the anchor may run
    // after the work it is supposed to precede).
    e->barrierAfterStages(MTL::StageDispatch,
                          MTL::StageVertex | MTL::StageObject | MTL::StageMesh | MTL::StageFragment | MTL::StageDispatch |
                              MTL::StageBlit | MTL::StageAccelerationStructure,
                          MTL4::VisibilityOptionNone);
    e->endEncoding();
    return cmd_;
}

double CommandTimer::finish() {
    MTL4::ComputeCommandEncoder* e = cmd_->computeCommandEncoder();
    e->barrierAfterQueueStages(MTL::StageVertex | MTL::StageObject | MTL::StageMesh | MTL::StageFragment |
                                   MTL::StageDispatch | MTL::StageBlit | MTL::StageAccelerationStructure,
                               MTL::StageDispatch, MTL4::VisibilityOptionNone);
    ctx_.anchorDispatch(e);
    e->writeTimestamp(MTL4::TimestampGranularityPrecise, ctx_.heap_, 1);
    e->endEncoding();
    ctx_.submit();
    const std::vector<u64> t = ctx_.readTimestamps(0, 2);
    if (t[0] == 0 || t[1] < t[0]) throw BenchError("CommandTimer: invalid timestamp");
    return ctx_.ticksToMs(t[0], t[1]);
}

// --- Report / registry ----------------------------------------------------------

phosphor::soc::Metric& Report::metric(const std::string& name, const std::string& unit, const Stats& stats,
                                      std::map<std::string, double> params, bool higherIsBetter) {
    for (const auto& m : b_.metrics)
        if (m.name == name) throw BenchError("duplicate metric " + name);
    phosphor::soc::Metric m;
    m.name           = name;
    m.unit           = unit;
    m.value          = stats.median;
    m.within         = stats;
    m.params         = std::move(params);
    m.higherIsBetter = higherIsBetter;
    b_.metrics.push_back(std::move(m));
    return b_.metrics.back();
}

phosphor::soc::Metric& Report::value(const std::string& name, const std::string& unit, double v,
                                     std::map<std::string, double> params, bool higherIsBetter) {
    return metric(name, unit, phosphor::soc::computeStats({v}), std::move(params), higherIsBetter);
}

void Report::status(Status s, const std::string& why) {
    b_.status = s;
    note(why);
}

void Report::note(const std::string& text) {
    if (text.empty()) return;
    b_.notes += b_.notes.empty() ? text : "; " + text;
}

void Report::negative(bool pass, const std::string& detail) {
    // Several controls: any failure fails the benchmark.
    if (!pass || b_.negative != Control::Fail) b_.negative = pass ? Control::Pass : Control::Fail;
    b_.negativeDetail += b_.negativeDetail.empty() ? detail : "; " + detail;
}

namespace {
std::vector<BenchInfo>& mutableRegistry() {
    static std::vector<BenchInfo> r;
    return r;
}
} // namespace

bool registerBench(const BenchInfo& info) {
    auto& r = mutableRegistry();
    r.push_back(info);
    std::sort(r.begin(), r.end(), [](const BenchInfo& a, const BenchInfo& b) { return std::strcmp(a.id, b.id) < 0; });
    return true;
}

const std::vector<BenchInfo>& registry() { return mutableRegistry(); }

// --- Machine ---------------------------------------------------------------------

phosphor::soc::Machine describeMachine(Context& ctx) {
    phosphor::soc::Machine m;
    m.chip = ctx.device()->name()->utf8String();
    m.slug = phosphor::soc::slugify(m.chip);
    m.gpuFamily = ctx.device()->supportsFamily(MTL::GPUFamilyApple10) ? "Apple10"
                  : ctx.device()->supportsFamily(MTL::GPUFamilyApple9) ? "Apple9"
                                                                        : "older";
    if (ctx.options().forceApple9) m.gpuFamily += " (forced Apple9 paths)";
    m.gpuCores            = gpuCoreCount();
    m.cpuPerformanceCores = static_cast<u32>(sysctlU64("hw.perflevel0.logicalcpu"));
    m.cpuEfficiencyCores  = static_cast<u32>(sysctlU64("hw.perflevel1.logicalcpu"));
    m.cpuLevelNames       = {sysctlString("hw.perflevel0.name"), sysctlString("hw.perflevel1.name")};
    m.memoryBytes         = sysctlU64("hw.memsize");
    m.os                  = "macOS " + sysctlString("kern.osproductversion");
    m.osBuild             = sysctlString("kern.osversion");
    m.osSlug              = phosphor::soc::slugify(m.os);
    m.sdk                 = SOC_SDK_VERSION;
    powerSource(m.powerSource, m.batteryPercent);
    m.thermalStart = thermalStateName();
    m.thermalEnd   = m.thermalStart;
    m.gpuPStateMHz = ctx.gpuState().pstateMHz();
    return m;
}

} // namespace soc
