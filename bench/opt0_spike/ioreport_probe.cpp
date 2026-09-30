// OPT-0 spike 2: energy and clock state without sudo (IOReport).
// A MEASUREMENT PROBE, not engine code.
//
// libIOReport is a private system library (in the SDK as libIOReport.tbd);
// tools such as macmon read the "Energy Model" and "GPU Stats" groups through
// it without root.  This probe lists the groups/channels it can subscribe to
// and samples energy (J -> W) and GPU P-state residency over an interval.
//
// Usage: ioreport_probe list            -- every group/subgroup/channel/unit
//        ioreport_probe sample [ms]     -- energy + GPU/CPU state residency delta

#include <CoreFoundation/CoreFoundation.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <chrono>

extern "C" {
typedef struct IOReportSubscription* IOReportSubscriptionRef;
CFDictionaryRef IOReportCopyAllChannels(uint64_t, uint64_t);
CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef group, CFStringRef subgroup, uint64_t, uint64_t, uint64_t);
void IOReportMergeChannels(CFDictionaryRef a, CFDictionaryRef b, CFTypeRef);
IOReportSubscriptionRef IOReportCreateSubscription(void*, CFMutableDictionaryRef channels,
                                                   CFMutableDictionaryRef* subbed, uint64_t, CFTypeRef);
CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef prev, CFDictionaryRef cur, CFTypeRef);
CFStringRef IOReportChannelGetGroup(CFDictionaryRef);
CFStringRef IOReportChannelGetSubGroup(CFDictionaryRef);
CFStringRef IOReportChannelGetChannelName(CFDictionaryRef);
CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef);
int32_t IOReportChannelGetFormat(CFDictionaryRef);
int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef, int32_t*);
int32_t IOReportStateGetCount(CFDictionaryRef);
CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef, int32_t);
int64_t IOReportStateGetResidency(CFDictionaryRef, int32_t);
}

namespace {

std::string s(CFStringRef r) {
    if (!r) return "";
    char buf[512];
    if (CFStringGetCString(r, buf, sizeof(buf), kCFStringEncodingUTF8)) return buf;
    return "?";
}

CFArrayRef channelsOf(CFDictionaryRef d) {
    return static_cast<CFArrayRef>(CFDictionaryGetValue(d, CFSTR("IOReportChannels")));
}

void list() {
    CFDictionaryRef all = IOReportCopyAllChannels(0, 0);
    if (!all) { std::printf("IOReportCopyAllChannels: null\n"); return; }
    CFArrayRef ch = channelsOf(all);
    const CFIndex n = ch ? CFArrayGetCount(ch) : 0;
    std::printf("channels: %ld\n", static_cast<long>(n));
    std::map<std::string, int> groups;
    for (CFIndex i = 0; i < n; ++i) {
        auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(ch, i));
        const std::string key = s(IOReportChannelGetGroup(c)) + " / " + s(IOReportChannelGetSubGroup(c));
        if (groups[key]++ < 6 || key.rfind("Energy Model", 0) == 0)
            std::printf("  [%s] %s (unit '%s', format %d)\n", key.c_str(), s(IOReportChannelGetChannelName(c)).c_str(),
                        s(IOReportChannelGetUnitLabel(c)).c_str(), IOReportChannelGetFormat(c));
    }
    for (auto& [k, v] : groups) std::printf("group %s: %d channels\n", k.c_str(), v);
    CFRelease(all);
}

// Busy CPU work on this thread so the sample shows a load.
volatile double g_sink = 0;

void sample(int ms, bool burn) {
    CFMutableDictionaryRef chans = CFDictionaryCreateMutableCopy(
        kCFAllocatorDefault, 0, IOReportCopyChannelsInGroup(CFSTR("Energy Model"), nullptr, 0, 0, 0));
    IOReportMergeChannels(chans, IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0), nullptr);
    CFMutableDictionaryRef subbed = nullptr;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(nullptr, chans, &subbed, 0, nullptr);
    if (!sub) { std::printf("IOReportCreateSubscription: null\n"); return; }
    CFDictionaryRef a = IOReportCreateSamples(sub, subbed, nullptr);
    if (std::getenv("RAW")) {  // absolute counter values of the first sample
        CFArrayRef rc = channelsOf(a);
        for (CFIndex i = 0; rc && i < CFArrayGetCount(rc); ++i) {
            auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(rc, i));
            if (s(IOReportChannelGetGroup(c)) != "Energy Model") continue;
            const int64_t v = IOReportSimpleGetIntegerValue(c, nullptr);
            if (v) std::printf("  raw %s = %lld %s\n", s(IOReportChannelGetChannelName(c)).c_str(),
                               static_cast<long long>(v), s(IOReportChannelGetUnitLabel(c)).c_str());
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (burn) {
        double x = 1.0;
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(ms)) x = x * 1.0000001 + 1e-9;
        g_sink = x;
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    CFDictionaryRef b = IOReportCreateSamples(sub, subbed, nullptr);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CFDictionaryRef d = IOReportCreateSamplesDelta(a, b, nullptr);
    CFArrayRef ch = channelsOf(d);
    const CFIndex n = ch ? CFArrayGetCount(ch) : 0;
    std::printf("interval %.3f s, %ld delta channels\n", secs, static_cast<long>(n));
    for (CFIndex i = 0; i < n; ++i) {
        auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(ch, i));
        const std::string grp = s(IOReportChannelGetGroup(c));
        const std::string name = s(IOReportChannelGetChannelName(c));
        const std::string unit = s(IOReportChannelGetUnitLabel(c));
        if (grp == "Energy Model") {
            const int64_t v = IOReportSimpleGetIntegerValue(c, nullptr);
            double j = double(v);
            if (unit == "mJ") j *= 1e-3; else if (unit == "uJ") j *= 1e-6; else if (unit == "nJ") j *= 1e-9;
            if (v != 0 || name == "CPU Energy" || name == "DRAM0" || name == "GPU0" || name == "PCPU" || name == "MCPU0") std::printf("  energy %-28s %12lld %s  -> %8.3f W\n", name.c_str(), static_cast<long long>(v),
                                    unit.c_str(), j / secs);
        } else {
            const int32_t k = IOReportStateGetCount(c);
            std::printf("  states %s/%s (%d):", s(IOReportChannelGetSubGroup(c)).c_str(), name.c_str(), k);
            int64_t tot = 0;
            for (int32_t q = 0; q < k; ++q) tot += IOReportStateGetResidency(c, q);
            for (int32_t q = 0; q < k; ++q) {
                const int64_t r = IOReportStateGetResidency(c, q);
                if (r || std::getenv("ALLSTATES")) std::printf(" %s=%.1f%%", s(IOReportStateGetNameForIndex(c, q)).c_str(), tot ? 100.0 * r / tot : 0.0);
            }
            std::printf("\n");
        }
    }
}

// Delta of every simple/state channel of one group (and optional subgroup).
void group(const char* g, const char* sg, int ms) {
    CFStringRef gs = CFStringCreateWithCString(nullptr, g, kCFStringEncodingUTF8);
    CFStringRef ss = sg ? CFStringCreateWithCString(nullptr, sg, kCFStringEncodingUTF8) : nullptr;
    CFMutableDictionaryRef chans =
        CFDictionaryCreateMutableCopy(kCFAllocatorDefault, 0, IOReportCopyChannelsInGroup(gs, ss, 0, 0, 0));
    CFMutableDictionaryRef subbed = nullptr;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(nullptr, chans, &subbed, 0, nullptr);
    if (!sub) { std::printf("no subscription\n"); return; }
    CFDictionaryRef a = IOReportCreateSamples(sub, subbed, nullptr);
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    CFDictionaryRef b = IOReportCreateSamples(sub, subbed, nullptr);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CFArrayRef ch = channelsOf(IOReportCreateSamplesDelta(a, b, nullptr));
    for (CFIndex i = 0; ch && i < CFArrayGetCount(ch); ++i) {
        auto c = static_cast<CFDictionaryRef>(CFArrayGetValueAtIndex(ch, i));
        const int32_t f = IOReportChannelGetFormat(c);
        const std::string head = s(IOReportChannelGetSubGroup(c)) + " / " + s(IOReportChannelGetChannelName(c));
        if (f == 1) {
            const int64_t v = IOReportSimpleGetIntegerValue(c, nullptr);
            if (v) std::printf("  %s = %lld %s (%.4g /s)\n", head.c_str(), static_cast<long long>(v),
                               s(IOReportChannelGetUnitLabel(c)).c_str(), v / secs);
        } else if (f == 2) {
            const int32_t k = IOReportStateGetCount(c);
            std::string line;
            for (int32_t q = 0; q < k; ++q) {
                const int64_t r = IOReportStateGetResidency(c, q);
                if (r) line += " " + s(IOReportStateGetNameForIndex(c, q)) + "=" + std::to_string(r);
            }
            if (!line.empty()) std::printf("  %s [%s]:%s\n", head.c_str(), s(IOReportChannelGetUnitLabel(c)).c_str(), line.c_str());
        }
    }
    std::printf("interval %.3f s\n", secs);
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "list";
    if (mode == "list") list();
    else if (mode == "sample") sample(argc > 2 ? std::atoi(argv[2]) : 1000, false);
    else if (mode == "burn") sample(argc > 2 ? std::atoi(argv[2]) : 1000, true);
    else if (mode == "group" && argc > 2) group(argv[2], argc > 3 && argv[3][0] ? argv[3] : nullptr, argc > 4 ? std::atoi(argv[4]) : 1000);
    else { std::fprintf(stderr, "usage: ioreport_probe list|sample [ms]|burn [ms]\n"); return 2; }
    return 0;
}
