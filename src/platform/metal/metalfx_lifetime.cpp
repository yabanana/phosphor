#include "platform/metal/metalfx_lifetime.h"
#include "core/log.h"
#include <Metal/Metal.hpp>
#include <MetalFX/MetalFX.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cstdio>
#include <new>
#ifdef __APPLE__
#include <objc/runtime.h>
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <malloc/malloc.h>
#endif

// Weak-reference entry points of the Objective-C runtime, specified by the
// Clang ARC ABI ("Runtime support"); the SDK has no C++ declaration of them.
extern "C" {
id objc_initWeak(id *location, id value);
id objc_loadWeakRetained(id *location);
void objc_destroyWeak(id *location);
}

namespace phosphor::metalfx {
namespace {
struct Counters {
    std::atomic<u64> adopted{0}, released{0}, cycleReleased{0}, wrapped{0}, retained{0}, unknownSignature{0};
    std::atomic<u64> maxReleaseMicros{0};
};
Counters &state() {
    static Counters c;
    return c;
}
id asId(MTL4FX::TemporalScaler *p) { return reinterpret_cast<id>(p); }
// The capture layer is the only Metal layer measured to wrap MetalFX scalers
// (API/shader validation and the HUD hand out the framework object itself).
bool captureLayerActive() {
    static const bool active = [] {
        auto *manager = MTL::CaptureManager::sharedCaptureManager();
        return manager && (manager->supportsDestination(MTL::CaptureDestinationDeveloperTools) ||
                           manager->supportsDestination(MTL::CaptureDestinationGPUTraceDocument));
    }();
    return active;
}
// Framework builds on which the self-reference was reproduced (bench/f8_spike
// matrix, plain release = live) and the release verified (--release-cycle,
// leaks, engine lifetime check). Add a version only after rerunning both.
constexpr const char *VerifiedFrameworkVersions[] = {"40.9"};
const std::string &loadedFrameworkVersion() {
    static const std::string version = [] {
        std::string v;
        auto *pool = NS::AutoreleasePool::alloc()->init();
        auto *bundle = NS::Bundle::bundle(
            NS::String::string("/System/Library/Frameworks/MetalFX.framework", NS::UTF8StringEncoding));
        auto *value = bundle ? bundle->objectForInfoDictionaryKey(NS::String::string("CFBundleVersion",
                                                                                     NS::UTF8StringEncoding))
                             : nullptr;
        if (value)
            v = static_cast<NS::String *>(value)->utf8String();
        pool->release();
        return v.empty() ? std::string("unknown") : v;
    }();
    return version;
}
} // namespace

const char *frameworkVersion() { return loadedFrameworkVersion().c_str(); }

bool selfReferenceReleaseEnabled() {
    static const bool enabled = [] {
        // Negative control: behave as on an unverified framework version.
        if (const char *e = std::getenv("PHOSPHOR_METALFX_UNVERIFIED"); e && *e == '1')
            return false;
        for (const char *v : VerifiedFrameworkVersions)
            if (loadedFrameworkVersion() == v)
                return true;
        return false;
    }();
    return enabled;
}

std::shared_ptr<MTL4FX::TemporalScaler> adoptTemporalScaler(MTL4FX::TemporalScaler *scaler) {
    if (!scaler)
        return {};
    // The caller owns exactly one reference (new... family), so anything above
    // one is held by the scaler's own implementation at creation time.
    const auto count = scaler->retainCount();
    const u32 internal = count > 1 ? static_cast<u32>(count - 1) : 0;
    state().adopted.fetch_add(1, std::memory_order_relaxed);
    if (captureLayerActive()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            LOG_WARN("GPU capture layer active: MetalFX scalers are wrapped, their internal self-reference cannot be "
                     "released and each recreation leaks until exit (lifetime result: unverified)");
        return std::shared_ptr<MTL4FX::TemporalScaler>(
            scaler, [](MTL4FX::TemporalScaler *p) { releaseTemporalScaler(p, 0, true); });
    }
    if (!selfReferenceReleaseEnabled()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            LOG_WARN("MetalFX %s is not a verified version for the self-reference release; plain release only "
                     "(rerun bench/f8_spike and tools/metalfx_lifetime_check.py, then list it)",
                     frameworkVersion());
        if (internal)
            state().unknownSignature.fetch_add(1, std::memory_order_relaxed);
        return std::shared_ptr<MTL4FX::TemporalScaler>(
            scaler, [](MTL4FX::TemporalScaler *p) { releaseTemporalScaler(p, 0); });
    }
    if (internal > 1) {
        state().unknownSignature.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("MetalFX temporal scaler created with %u internal references; only one is a known self-reference, "
                 "releasing normally",
                 internal);
    }
    return std::shared_ptr<MTL4FX::TemporalScaler>(
        scaler, [internal](MTL4FX::TemporalScaler *p) { releaseTemporalScaler(p, internal); });
}

ReleaseOutcome releaseTemporalScaler(MTL4FX::TemporalScaler *scaler, u32 internalReferences, bool wrapped) {
    if (!scaler)
        return ReleaseOutcome::Released;
    const auto begin = std::chrono::steady_clock::now();
    id weak = nullptr;
    objc_initWeak(&weak, asId(scaler));
    scaler->release();
    ReleaseOutcome outcome = wrapped ? ReleaseOutcome::Wrapped : ReleaseOutcome::Released;
    if (id alive = objc_loadWeakRetained(&weak)) {
        auto *object = reinterpret_cast<MTL4FX::TemporalScaler *>(alive);
        // Our probe plus, for the known defect, the single self-reference.
        const bool onlySelf = !wrapped && internalReferences == 1 && object->retainCount() == 2;
        object->release(); // the probe
        outcome = wrapped ? ReleaseOutcome::Wrapped : ReleaseOutcome::Retained;
        if (onlySelf) {
            // Nothing else owns the scaler: drop the reference its filter keeps
            // on it. Its later release during deallocation is a runtime no-op.
            object->release();
            id after = objc_loadWeakRetained(&weak);
            if (after)
                reinterpret_cast<MTL4FX::TemporalScaler *>(after)->release();
            else
                outcome = ReleaseOutcome::CycleReleased;
        }
    }
    objc_destroyWeak(&weak);
    const u64 micros = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count());
    for (u64 seen = state().maxReleaseMicros.load(std::memory_order_relaxed);
         micros > seen && !state().maxReleaseMicros.compare_exchange_weak(seen, micros, std::memory_order_relaxed);) {
    }
    switch (outcome) {
    case ReleaseOutcome::Released:
        state().released.fetch_add(1, std::memory_order_relaxed);
        break;
    case ReleaseOutcome::CycleReleased:
        state().cycleReleased.fetch_add(1, std::memory_order_relaxed);
        break;
    case ReleaseOutcome::Wrapped:
        state().wrapped.fetch_add(1, std::memory_order_relaxed);
        break;
    case ReleaseOutcome::Retained:
        state().retained.fetch_add(1, std::memory_order_relaxed);
        LOG_WARN("MetalFX temporal scaler still referenced after release (internal references at creation: %u)",
                 internalReferences);
        break;
    }
    return outcome;
}

namespace {
void* denoisedTimingRecord(MTL4FX::TemporalDenoisedScaler* scaler) {
#ifdef __APPLE__
    // Pin the actual loaded image, not CFBundleVersion alone: Apple can reuse
    // a marketing version across OS builds with different dealloc code.
    static constexpr unsigned char verifiedUUID[16]={0xea,0xb6,0x8c,0xbc,0x37,0x6a,0x37,0x91,0xbd,0x74,0xb0,0x94,0xfa,0x13,0xad,0x4e};
    if(const char* value=std::getenv("PHOSPHOR_METALFX_DENOISED_PLAIN_RELEASE");value&&std::strcmp(value,"1")==0)return nullptr;
    const Class cls=object_getClass(reinterpret_cast<id>(scaler));
    if(!cls||std::strcmp(class_getName(cls),"_M4FXTemporalDenoisingScalingEffect")!=0)return nullptr;
    const auto method=class_getInstanceMethod(cls,sel_registerName("dealloc"));
    Dl_info image{};
    if(!method||!dladdr(reinterpret_cast<const void*>(method_getImplementation(method)),&image)||!image.dli_fbase)return nullptr;
    const auto* header=static_cast<const mach_header_64*>(image.dli_fbase);
    if(header->magic!=MH_MAGIC_64)return nullptr;
    const auto* command=reinterpret_cast<const load_command*>(header+1);bool verified=false;
    for(uint32_t i=0;i<header->ncmds;++i) {
        if(command->cmd==LC_UUID&&command->cmdsize>=sizeof(uuid_command))
            verified=std::memcmp(reinterpret_cast<const uuid_command*>(command)->uuid,verifiedUUID,16)==0;
        command=reinterpret_cast<const load_command*>(reinterpret_cast<const char*>(command)+command->cmdsize);
    }
    if(!verified)return nullptr;
    const Ivar field=class_getInstanceVariable(cls,"_timingRecord");
    if(!field||std::strcmp(ivar_getTypeEncoding(field),"^{CHistoryRecord=fIIff[120f]ff}")!=0||
       ivar_getOffset(field)!=1072||class_getInstanceSize(cls)!=1248)return nullptr;
    // CHistoryRecord is 508 bytes of scalar timing values, no object pointers,
    // no destructor. The verified constructor allocates it with operator new;
    // neither dealloc nor .cxx_destruct destroys/deletes this particular field.
    void* record=nullptr;std::memcpy(&record,reinterpret_cast<const char*>(scaler)+ivar_getOffset(field),sizeof(record));
    if(record&&(malloc_size(record)<508||malloc_size(record)>640))return nullptr;
    return record;
#else
    (void)scaler;return nullptr;
#endif
}
}

std::shared_ptr<MTL4FX::TemporalDenoisedScaler> adoptTemporalDenoisedScaler(MTL4FX::TemporalDenoisedScaler* scaler) {
    if(!scaler)return {};
    void* const record=denoisedTimingRecord(scaler);
    return std::shared_ptr<MTL4FX::TemporalDenoisedScaler>(scaler,[record](auto* pointer) {
        // Adapter retirement guarantees GPU completion and uses the cache
        // worker's autorelease pool. Do not change any SDK ivar or swizzle a
        // method; let the framework perform its complete ordinary teardown.
        const bool unchanged=record&&denoisedTimingRecord(pointer)==record;
        id weak=nullptr;objc_initWeak(&weak,reinterpret_cast<id>(pointer));
        reinterpret_cast<NS::Object*>(pointer)->release();
        id alive=objc_loadWeakRetained(&weak);const bool destroyed=alive==nullptr;
        if(alive)reinterpret_cast<NS::Object*>(alive)->release();
        objc_destroyWeak(&weak);
        const bool reclaimed=unchanged&&destroyed;
        if(reclaimed)::operator delete(record);
        std::fprintf(stdout,"METALFX-DENOISED-LIFETIME weak_nil=%u timing_reclaimed=%u signature_matched=%u\n",
                     unsigned(destroyed),unsigned(reclaimed),unsigned(unchanged));
        // A surviving SDK owner or unknown image remains visible to leaks and
        // fails qualification; never free its memory merely to clear a report.
    });
}

LifetimeCounters counters() {
    const auto &c = state();
    return {c.adopted.load(),          c.released.load(), c.cycleReleased.load(), c.retained.load(),
            c.unknownSignature.load(), c.wrapped.load(),  c.maxReleaseMicros.load()};
}

} // namespace phosphor::metalfx
