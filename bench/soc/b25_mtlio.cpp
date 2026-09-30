// B-25: MTLIO (Metal fast resource loading) with every compression codec:
// cold (first read from the SSD) and warm (page cache) throughput of
// decompressed output, compression ratio on two realistic payloads, request
// size / chunk size (64 KiB, 1 MiB, 16 MiB), 1 vs 4 parallel IO queues, and
// the CPU time consumed per GB (getrusage) to tell whether decompression
// runs on our CPU.
//
// Serves S-IO-1..2 of docs/APPLE_SOC_PLAYBOOK.md (§13).
//
// Cold reads: the file is copied with F_NOCACHE (the copy is not left in the
// page cache), so the first read of the copy goes to the SSD (no purge/sudo).
// Files live in $TMPDIR and are removed afterwards (<= 2 GiB in total).

#include "harness.h"

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace soc {
namespace {

constexpr size_t KiB = 1024, MiB = 1024 * 1024;

struct Codec {
    const char* name;
    MTL::IOCompressionMethod method;
};
constexpr Codec kCodecs[] = {{"lz4", MTL::IOCompressionMethodLZ4},       {"lzfse", MTL::IOCompressionMethodLZFSE},
                             {"zlib", MTL::IOCompressionMethodZlib},     {"lzma", MTL::IOCompressionMethodLZMA},
                             {"lzbitmap", MTL::IOCompressionMethodLZBitmap}};

std::string sizeName(size_t bytes) { return bytes >= MiB ? std::to_string(bytes / MiB) + "MiB" : std::to_string(bytes / KiB) + "KiB"; }

double cpuSeconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return double(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) + double(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1e-6;
}

/// Smooth 16-bit ramps with 2 bits of noise (HDR texels / quantised data).
std::vector<u8> rampPayload(size_t bytes) {
    std::vector<u8> d(bytes);
    u32 s = 7;
    for (size_t i = 0; i + 1 < bytes; i += 2) {
        s = s * 1103515245u + 12345u;
        const u16 v = u16((i >> 5) + ((s >> 16) & 3));
        d[i] = u8(v);
        d[i + 1] = u8(v >> 8);
    }
    return d;
}

/// Mesh-like float buffer: 32-byte vertices (pos, normal, uv) of a smooth
/// wavy grid, i.e. neighbouring vertices differ slightly (the vertex data of a
/// real mesh before meshoptimizer quantisation).
std::vector<u8> meshPayload(size_t bytes) {
    struct V { float p[3], n[3], uv[2]; };
    static_assert(sizeof(V) == 32);
    std::vector<u8> d(bytes);
    const size_t count = bytes / sizeof(V);
    const size_t w = 256;
    V* v = reinterpret_cast<V*>(d.data());
    for (size_t i = 0; i < count; ++i) {
        const float x = float(i % w) / float(w), y = float(i / w) / float(w);
        const float h = 0.1f * std::sin(x * 12.0f) * std::cos(y * 9.0f);
        v[i] = {{x, h, y}, {-0.1f * 12.0f * std::cos(x * 12.0f) * std::cos(y * 9.0f), 1.0f, 0.1f * 9.0f * std::sin(x * 12.0f) * std::sin(y * 9.0f)}, {x, y}};
    }
    return d;
}

struct TempFile {
    std::string path;
    explicit TempFile(std::string p) : path(std::move(p)) {}
    ~TempFile() { if (!path.empty()) unlink(path.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

size_t fileSize(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fclose(f);
    return size_t(n);
}

bool writeFile(const std::string& path, const u8* data, size_t bytes, bool noCache) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;
    if (noCache) fcntl(fd, F_NOCACHE, 1);
    size_t off = 0;
    while (off < bytes) {
        const ssize_t k = write(fd, data + off, std::min<size_t>(bytes - off, 4 * MiB));
        if (k <= 0) { close(fd); return false; }
        off += size_t(k);
    }
    fsync(fd);
    close(fd);
    return true;
}

/// Copy `src` to `dst` with F_NOCACHE on the destination.
bool coldCopy(const std::string& src, const std::string& dst) {
    const size_t n = fileSize(src);
    std::vector<u8> buf(n);
    FILE* f = std::fopen(src.c_str(), "rb");
    if (!f) return false;
    const size_t got = std::fread(buf.data(), 1, n, f);
    std::fclose(f);
    return got == n && writeFile(dst, buf.data(), n, true);
}

struct Loader {
    Context& ctx;
    MTL::Buffer* dst;
    size_t bytes;

    // IO queues are process-wide and never released: releasing an IOCommandQueue
    // does not stop its IO dispatch threads (measured: ~20 threads per queue stay
    // blocked in IOGPU), and once ~90 dispatch threads are blocked the next
    // waitUntilCompleted deadlocks (hang on the 3rd suite run with per-run queues).
    MTL::IOCommandQueue* queue(u32 i) {
        static std::vector<MTL::IOCommandQueue*> queues;
        while (queues.size() <= i) {
            NS::Error* err = nullptr;
            auto* qd = MTL::IOCommandQueueDescriptor::alloc()->init();
            MTL::IOCommandQueue* q = ctx.device()->newIOCommandQueue(qd, &err);
            qd->release();
            if (!q) throw BenchError("newIOCommandQueue failed");
            queues.push_back(q);
        }
        return queues[i];
    }

    /// Loads the whole file as ceil(bytes/req) requests spread over `nq` IO
    /// queues (one IO command buffer each); returns wall ms of commit -> all complete
    /// and adds the CPU seconds of that span to *cpu.
    double load(MTL::IOFileHandle* h, size_t req, u32 nq, double* cpu) {
        std::vector<MTL::IOCommandBuffer*> cbs;
        for (u32 i = 0; i < nq; ++i) cbs.push_back(queue(i)->commandBuffer());
        const size_t nreq = (bytes + req - 1) / req;
        for (size_t r = 0; r < nreq; ++r) {
            const size_t off = r * req, len = std::min(req, bytes - off);
            cbs[r % nq]->loadBuffer(dst, off, len, h, off);
        }
        const double c0 = cpuSeconds();
        const double t0 = nowMs();
        for (auto* cb : cbs) cb->commit();
        for (auto* cb : cbs) cb->waitUntilCompleted();
        const double dt = nowMs() - t0;
        if (cpu) *cpu += cpuSeconds() - c0;
        for (auto* cb : cbs)
            if (cb->status() != MTL::IOStatusComplete) throw BenchError("MTLIO command buffer did not complete");
        return dt;
    }
};

struct ConfigResult {
    Stats cold, warm;
    double cpuPerGb = 0;
    bool exact = true;
};

void benchMtlio(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    const size_t payload = (quick ? 16 : 32) * MiB;
    const size_t meshBytes = (quick ? 4 : 8) * MiB;
    const u32 coldReps = quick ? 2 : 3, warmReps = quick ? 5 : 9;
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = (tmp && *tmp) ? tmp : "/tmp";
    const std::string stem = dir + (dir.back() == '/' ? "" : "/") + "soc_b25_" + std::to_string(getpid());

    const std::vector<u8> ramp = rampPayload(payload);
    const std::vector<u8> mesh = meshPayload(meshBytes);
    MTL::Buffer* dst = ctx.buffer(payload);
    Loader ld{ctx, dst, payload};

    bool allExact = true;
    std::string exactFail;
    std::map<std::string, double> warmBase; // codec -> warm GB/s at the base config
    double rawCold = 0, rawWarm = 0;

    
    // One configuration: cold (fresh F_NOCACHE copy each repetition) and warm.
    auto runConfig = [&](Loader& ld, const std::vector<u8>& ref, const std::string& file, MTL::IOCompressionMethod method, bool raw, size_t req, u32 nq, const std::string& label) {
        ConfigResult res;
        auto handle = [&](const std::string& p) {
            NS::Error* err = nullptr;
            NS::URL* url = NS::URL::fileURLWithPath(NS::String::string(p.c_str(), NS::UTF8StringEncoding));
            MTL::IOFileHandle* h = raw ? ctx.device()->newIOHandle(url, &err) : ctx.device()->newIOHandle(url, method, &err);
            if (!h) throw BenchError("newIOHandle failed for " + label);
            return h;
        };
        const size_t payload = ref.size();
        MTL::Buffer* dst = ld.dst;
        auto verify = [&]() { return std::memcmp(dst->contents(), ref.data(), payload) == 0; };
        const double gb = double(payload) * 1e-9;
        const std::string coldPath = file + ".cold";
        TempFile coldFile(coldPath);
        // cold
        std::vector<double> cold;
        for (u32 i = 0; i < coldReps; ++i) {
            if (!coldCopy(file, coldPath)) throw BenchError("cold copy failed");
            std::memset(dst->contents(), 0xA5, payload);
            MTL::IOFileHandle* h = handle(coldPath);
            const double ms = ld.load(h, req, nq, nullptr);
            h->release();
            if (!verify()) { res.exact = false; }
            cold.push_back(gb / (ms * 1e-3));
        }
        res.cold = phosphor::soc::computeStats(cold);
        // warm
        MTL::IOFileHandle* h = handle(file);
        ld.load(h, req, nq, nullptr); // populate the page cache
        double cpu = 0;
        res.warm = ctx.measure(
            [&] {
                std::memset(dst->contents(), 0xA5, payload);
                const double ms = ld.load(h, req, nq, &cpu);
                if (!verify()) res.exact = false;
                return gb / (ms * 1e-3);
            },
            warmReps);
        h->release();
        res.cpuPerGb = cpu / (gb * double(warmReps));
        if (!res.exact) { allExact = false; exactFail += label + " "; }
        return res;
    };

    auto report = [&](const std::string& base, const ConfigResult& r, size_t req, u32 nq, double ratio, size_t pbytes) {
        std::map<std::string, double> params = {{"request_bytes", double(req)}, {"queues", double(nq)}, {"payload_bytes", double(pbytes)}};
        if (ratio > 0) params["ratio"] = ratio;
        rep.metric(base + ".cold_gbs", "GB/s", r.cold, params);
        rep.metric(base + ".warm_gbs", "GB/s", r.warm, params);
        rep.value(base + ".cpu_s_per_gb", "s/GB", r.cpuPerGb, params, false);
    };

    // --- raw (uncompressed) -------------------------------------------------------
    {
        const size_t rawBytes = (quick ? 128 : 256) * MiB;
        const std::vector<u8> rawRamp = rampPayload(rawBytes);
        MTL::Buffer* rawDst = ctx.buffer(rawBytes);
        Loader ldRaw{ctx, rawDst, rawBytes};
        TempFile raw(stem + "_raw.bin");
        if (!writeFile(raw.path, rawRamp.data(), rawBytes, false)) throw BenchError("cannot write " + raw.path);
        for (size_t req : quick ? std::vector<size_t>{1 * MiB} : std::vector<size_t>{64 * KiB, 1 * MiB, 16 * MiB}) {
            for (u32 nq : {1u, 4u}) {
                const bool base = req == 1 * MiB && nq == 1;
                const ConfigResult r = runConfig(ldRaw, rawRamp, raw.path, MTL::IOCompressionMethodZlib, true, req, nq, "raw " + sizeName(req) + " q" + std::to_string(nq));
                report(base ? "mtlio.raw" : "mtlio.raw.req_" + sizeName(req) + ".q" + std::to_string(nq), r, req, nq, 1.0, rawBytes);
                if (base) { rawCold = r.cold.median; rawWarm = r.warm.median; }
                ctx.log("raw %s q%u: cold %.2f warm %.2f GB/s cpu %.3f s/GB", sizeName(req).c_str(), nq, r.cold.median, r.warm.median, r.cpuPerGb);
            }
        }
        rep.note("mtlio.raw = uncompressed " + std::to_string(rawBytes / MiB) + " MiB file (larger than the compressed payload so the SSD time dominates), 1 MiB requests, 1 queue (other sizes/queues have .req_*.q* suffixes)");
    }

    // --- compressed -------------------------------------------------------------------
    struct Cfg { size_t chunk; u32 nq; };
    std::vector<Cfg> cfgs;
    // Base config first: the default chunk (64 KiB), 1 queue.
    cfgs.push_back({64 * KiB, 1});
    cfgs.push_back({64 * KiB, 4});
    if (!quick)
        for (size_t c : {1 * MiB, 16 * MiB})
            for (u32 nq : {1u, 4u}) cfgs.push_back({c, nq});

    double compressMs = 0;
    for (const Codec& c : kCodecs) {
        double ratioRamp = 0;
        for (size_t chunk : {64 * KiB, 1 * MiB, 16 * MiB}) {
            bool used = false;
            for (const Cfg& cfg : cfgs) used |= cfg.chunk == chunk;
            if (!used) continue;
            TempFile f(stem + "_" + c.name + "_" + sizeName(chunk) + ".bin");
            const double t0 = nowMs();
            MTL::IOCompressionContext cc = MTL::IOCreateCompressionContext(f.path.c_str(), c.method, chunk);
            MTL::IOCompressionContextAppendData(cc, ramp.data(), ramp.size());
            if (MTL::IOFlushAndDestroyCompressionContext(cc) != MTL::IOCompressionStatusComplete) throw BenchError(std::string(c.name) + ": compression failed");
            const double cms = nowMs() - t0;
            const double ratio = double(payload) / double(fileSize(f.path));
            if (chunk == 64 * KiB) {
                ratioRamp = ratio;
                compressMs = cms;
                rep.value(std::string("mtlio.") + c.name + ".compress_mbs", "MB/s", double(payload) * 1e-6 / (cms * 1e-3), {{"chunk_bytes", double(chunk)}});
            }
            rep.value(std::string("mtlio.") + c.name + (chunk == 64 * KiB ? "" : ".chunk_" + sizeName(chunk)) + ".ratio_ramp", "ratio", ratio,
                      {{"chunk_bytes", double(chunk)}});
            for (const Cfg& cfg : cfgs) {
                if (cfg.chunk != chunk) continue;
                const bool base = chunk == 64 * KiB && cfg.nq == 1;
                const std::string label = std::string(c.name) + " " + sizeName(chunk) + " q" + std::to_string(cfg.nq);
                const ConfigResult r = runConfig(ld, ramp, f.path, c.method, false, chunk, cfg.nq, label);
                report(base ? std::string("mtlio.") + c.name
                            : std::string("mtlio.") + c.name + ".req_" + sizeName(chunk) + ".q" + std::to_string(cfg.nq),
                       r, chunk, cfg.nq, ratio, payload);
                if (base) warmBase[c.name] = r.warm.median;
                ctx.log("%s: ratio %.2f cold %.2f warm %.2f GB/s cpu %.3f s/GB (compress %.0f ms)", label.c_str(), ratio, r.cold.median,
                        r.warm.median, r.cpuPerGb, cms);
            }
        }
        // Mesh-like payload: ratio only (default chunk).
        {
            TempFile f(stem + "_mesh_" + c.name + ".bin");
            MTL::IOCompressionContext cc = MTL::IOCreateCompressionContext(f.path.c_str(), c.method, 64 * KiB);
            MTL::IOCompressionContextAppendData(cc, mesh.data(), mesh.size());
            if (MTL::IOFlushAndDestroyCompressionContext(cc) != MTL::IOCompressionStatusComplete) throw BenchError(std::string(c.name) + ": mesh compression failed");
            rep.value(std::string("mtlio.") + c.name + ".ratio_mesh", "ratio", double(mesh.size()) / double(fileSize(f.path)), {{"payload_bytes", double(mesh.size())}});
        }
        (void)ratioRamp;
    }
    (void)compressMs;

    rep.note("throughput = decompressed bytes / wall time of commit -> completion of all IO command buffers; request size = chunk size (whole file in payload/request requests); suffix .req_<size>.q<n> = other request size / IO queues");
    rep.note("cold: fresh F_NOCACHE copy per repetition (" + std::to_string(coldReps) + " reps), first read goes to the SSD; warm: page cache, median of " + std::to_string(warmReps) + " reps");
    rep.note("cpu_s_per_gb = getrusage(SELF) user+system of the process over the timed warm spans / GB decompressed: ~0 means decompression is not on this process's CPU threads");
    rep.note("payloads: 16-bit ramps + 2 bits noise (" + std::to_string(payload / MiB) + " MiB, throughput and ratio_ramp) and a wavy-grid 32-byte vertex buffer (" + std::to_string(mesh.size() / MiB) + " MiB, ratio_mesh)");

    // --- controls ---------------------------------------------------------------------------
    const double lz4 = warmBase["lz4"], lzma = warmBase["lzma"];
    const bool coldLeWarm = rawCold <= rawWarm * 1.05;
    const bool lzmaSlower = lzma < lz4;
    rep.negative(allExact && coldLeWarm && lzmaSlower,
                 std::string("bytes exact after every load: ") + (allExact ? "yes" : "NO (" + exactFail + ")") + "; raw cold " +
                     std::to_string(rawCold).substr(0, 5) + " <= warm " + std::to_string(rawWarm).substr(0, 5) + " GB/s: " + (coldLeWarm ? "yes" : "NO") +
                     "; lzma warm " + std::to_string(lzma).substr(0, 5) + " < lz4 warm " + std::to_string(lz4).substr(0, 5) + " GB/s: " + (lzmaSlower ? "yes" : "NO"));
    if (!allExact) rep.status(Status::Failed, "decompressed bytes differ from the source");
}

} // namespace

SOC_BENCH("B-25", "mtlio.decompress", "MTLIO: cold/warm throughput per codec, request size and IO queues; ratio; CPU per GB", benchMtlio);

} // namespace soc
