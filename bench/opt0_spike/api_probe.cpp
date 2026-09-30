// OPT-0 spike 3: Metal APIs available on the chip.  A MEASUREMENT PROBE,
// not engine code.
//
//   gemm [n]  : MetalPerformancePrimitives matmul2d (tensor ops, Neural
//               Accelerator on Apple10) for half/bfloat/int8 inputs, vs a
//               simdgroup_matrix FP16 GEMM on the shader ALUs; results
//               checked against the CPU on sampled entries.
//   mtlio     : MTLIO load of a file compressed with each codec
//               (MTLIOCreateCompressionContext), throughput and correctness.
//
// Usage: api_probe gemm [n] | mtlio [MiB]

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <mach/mach_time.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using u32 = uint32_t;
using u64 = uint64_t;

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }
[[noreturn]] void die(const std::string& m) { std::fprintf(stderr, "api_probe: %s\n", m.c_str()); std::exit(3); }

double nowMs() {
    static mach_timebase_info_data_t tb = [] { mach_timebase_info_data_t t; mach_timebase_info(&t); return t; }();
    return double(mach_absolute_time()) * tb.numer / tb.denom * 1e-6;
}

// C[M x N] = A[M x K] * B[K x N], row-major; tensors see extent(0) as the
// fastest (column) index, so A is (K, M), B is (N, K), C is (N, M).
const char* kGemmSource = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims { int m, n, k, pad; };

template <typename TIn, typename TOut>
inline void gemmTile(device TIn* a, device TIn* b, device TOut* c, constant Dims& d, uint2 tgid) {
    auto A = tensor<device TIn,  dextents<int32_t, 2>, tensor_inline>(a, dextents<int32_t, 2>(d.k, d.m));
    auto B = tensor<device TIn,  dextents<int32_t, 2>, tensor_inline>(b, dextents<int32_t, 2>(d.n, d.k));
    auto C = tensor<device TOut, dextents<int32_t, 2>, tensor_inline>(c, dextents<int32_t, 2>(d.n, d.m));
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
}

kernel void gemm_f16(device half* a [[buffer(0)]], device half* b [[buffer(1)]], device float* c [[buffer(2)]],
                     constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    gemmTile<half, float>(a, b, c, d, tgid);
}
kernel void gemm_bf16(device bfloat* a [[buffer(0)]], device bfloat* b [[buffer(1)]], device float* c [[buffer(2)]],
                      constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    gemmTile<bfloat, float>(a, b, c, d, tgid);
}
kernel void gemm_i8(device int8_t* a [[buffer(0)]], device int8_t* b [[buffer(1)]], device int32_t* c [[buffer(2)]],
                    constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    gemmTile<int8_t, int32_t>(a, b, c, d, tgid);
}

// Reference: simdgroup_matrix 8x8 FP16 GEMM on the shader ALUs, each
// simdgroup computes a 32x32 block of C (4x4 matrices), 4 simdgroups per
// threadgroup = 64x64 block.
kernel void gemm_simd_f16(device half* a [[buffer(0)]], device half* b [[buffer(1)]], device float* c [[buffer(2)]],
                          constant Dims& d [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    const int row0 = int(tgid.y) * 64 + int(sg / 2) * 32;
    const int col0 = int(tgid.x) * 64 + int(sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = simdgroup_float8x8(0);
    for (int k = 0; k < d.k; k += 8) {
        simdgroup_half8x8 ma[4], mb[4];
        for (int i = 0; i < 4; ++i) simdgroup_load(ma[i], a + (row0 + i * 8) * d.k + k, d.k);
        for (int j = 0; j < 4; ++j) simdgroup_load(mb[j], b + k * d.n + col0 + j * 8, d.n);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], c + (row0 + i * 8) * d.n + col0 + j * 8, d.n);
}
)MSL";

struct Dims { int m, n, k, pad; };

struct Gpu {
    MTL::Device* dev;
    MTL4::CommandQueue* q;
    MTL4::Compiler* comp;
    MTL4::CommandAllocator* alloc;
    MTL4::CommandBuffer* cb;
    MTL::SharedEvent* ev;
    MTL::ResidencySet* rs;
    MTL4::ArgumentTable* table;
    u64 evv = 0;
    Gpu() {
        dev = MTL::CreateSystemDefaultDevice();
        q = dev->newMTL4CommandQueue();
        NS::Error* err = nullptr;
        auto* cd = MTL4::CompilerDescriptor::alloc()->init();
        comp = dev->newCompiler(cd, &err);
        alloc = dev->newCommandAllocator();
        cb = dev->newCommandBuffer();
        ev = dev->newSharedEvent();
        auto* rd = MTL::ResidencySetDescriptor::alloc()->init();
        rs = dev->newResidencySet(rd, &err);
        q->addResidencySet(rs);
        auto* ad = MTL4::ArgumentTableDescriptor::alloc()->init();
        ad->setMaxBufferBindCount(8);
        table = dev->newArgumentTable(ad, &err);
    }
    MTL::Library* lib(const char* src) {
        NS::Error* err = nullptr;
        auto* o = MTL::CompileOptions::alloc()->init();
        o->setLanguageVersion(MTL::LanguageVersion4_0);
        MTL::Library* l = dev->newLibrary(str(src), o, &err);
        if (!l) die(std::string("MSL: ") + (err ? err->localizedDescription()->utf8String() : "?"));
        return l;
    }
    MTL::ComputePipelineState* pipe(MTL::Library* l, const char* name) {
        NS::Error* err = nullptr;
        auto* f = MTL4::LibraryFunctionDescriptor::alloc()->init();
        f->setLibrary(l);
        f->setName(str(name));
        auto* d = MTL4::ComputePipelineDescriptor::alloc()->init();
        d->setComputeFunctionDescriptor(f);
        auto* p = comp->newComputePipelineState(d, nullptr, &err);
        if (!p) die(std::string(name) + ": " + (err ? err->localizedDescription()->utf8String() : "?"));
        return p;
    }
    MTL::Buffer* buffer(size_t bytes) {
        MTL::Buffer* b = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
        rs->addAllocation(b);
        rs->commit();
        return b;
    }
    // Runs `enc` in one compute encoder; returns feedback GPU ms.
    template <typename F> double run(F&& enc) {
        alloc->reset();
        cb->beginCommandBuffer(alloc);
        auto* ce = cb->computeCommandEncoder();
        enc(ce);
        ce->endEncoding();
        cb->endCommandBuffer();
        std::atomic<bool> got{false};
        double ms = 0;
        std::string e;
        auto* o = MTL4::CommitOptions::alloc()->init();
        o->addFeedbackHandler([&](MTL4::CommitFeedback* fb) {
            if (fb->error()) e = fb->error()->localizedDescription()->utf8String();
            ms = (fb->GPUEndTime() - fb->GPUStartTime()) * 1e3;
            got = true;
        });
        const MTL4::CommandBuffer* bufs[] = {cb};
        q->commit(bufs, 1, o);
        o->release();
        q->signalEvent(ev, ++evv);
        ev->waitUntilSignaledValue(evv, 60000);
        while (!got) std::this_thread::yield();
        if (!e.empty()) die("GPU: " + e);
        return ms;
    }
};

uint16_t toHalf(float f) {
    __fp16 h = static_cast<__fp16>(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
uint16_t toBf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16); }

void caseGemm(int n) {
    Gpu g;
    MTL::Library* l = g.lib(kGemmSource);
    const size_t nn = size_t(n) * n;
    MTL::Buffer* a = g.buffer(nn * 2);
    MTL::Buffer* b = g.buffer(nn * 2);
    MTL::Buffer* c = g.buffer(nn * 4);
    MTL::Buffer* dims = g.buffer(256);
    Dims d{n, n, n, 0};
    std::memcpy(dims->contents(), &d, sizeof(d));
    // Small integer-valued inputs so FP16/BF16/INT8 products are exact.
    std::vector<float> fa(nn), fb(nn);
    for (size_t i = 0; i < nn; ++i) { fa[i] = float(int(i * 7 % 5) - 2); fb[i] = float(int(i * 3 % 7) - 3); }
    auto ref = [&](int r, int col) { double s = 0; for (int k = 0; k < n; ++k) s += double(fa[size_t(r) * n + k]) * fb[size_t(k) * n + col]; return s; };
    const double flop = 2.0 * n * double(n) * n;
    std::printf("gemm %dx%dx%d (%.1f GFLOP)\n", n, n, n, flop * 1e-9);
    struct K { const char* name; int type; int tgW; int tgH; };   // type 0 f16, 1 bf16, 2 i8
    for (K k : {K{"gemm_f16", 0, 32, 64}, K{"gemm_bf16", 1, 32, 64}, K{"gemm_i8", 2, 32, 64}, K{"gemm_simd_f16", 0, 64, 64}}) {
        auto* ps = g.pipe(l, k.name);
        for (size_t i = 0; i < nn; ++i) {
            if (k.type == 0) { static_cast<uint16_t*>(a->contents())[i] = toHalf(fa[i]); static_cast<uint16_t*>(b->contents())[i] = toHalf(fb[i]); }
            if (k.type == 1) { static_cast<uint16_t*>(a->contents())[i] = toBf16(fa[i]); static_cast<uint16_t*>(b->contents())[i] = toBf16(fb[i]); }
            if (k.type == 2) { static_cast<int8_t*>(a->contents())[i] = int8_t(fa[i]); static_cast<int8_t*>(b->contents())[i] = int8_t(fb[i]); }
        }
        auto enc = [&](MTL4::ComputeCommandEncoder* ce) {
            g.table->setAddress(a->gpuAddress(), 0);
            g.table->setAddress(b->gpuAddress(), 1);
            g.table->setAddress(c->gpuAddress(), 2);
            g.table->setAddress(dims->gpuAddress(), 3);
            ce->setComputePipelineState(ps);
            ce->setArgumentTable(g.table);
            ce->dispatchThreadgroups(MTL::Size::Make(n / k.tgW, n / k.tgH, 1), MTL::Size::Make(128, 1, 1));
        };
        std::memset(c->contents(), 0, nn * 4);
        for (int w = 0; w < 30; ++w) g.run(enc);   // warm clocks
        std::vector<double> t;
        for (int r = 0; r < 15; ++r) t.push_back(g.run(enc));
        std::sort(t.begin(), t.end());
        u32 bad = 0;
        for (int s = 0; s < 64; ++s) {
            const int r = (s * 977) % n, col = (s * 613 + 5) % n;
            const double got = k.type == 2 ? double(static_cast<int32_t*>(c->contents())[size_t(r) * n + col])
                                           : double(static_cast<float*>(c->contents())[size_t(r) * n + col]);
            if (std::fabs(got - ref(r, col)) > 1e-3 * std::max(1.0, std::fabs(ref(r, col)))) ++bad;
        }
        std::printf("  %-14s p50 %8.3f ms  min %8.3f  -> %6.2f T(FL)OPS  wrong %u/64\n", k.name, t[t.size() / 2], t[0],
                    flop / (t[t.size() / 2] * 1e-3) * 1e-12, bad);
    }
}

void caseMtlio(size_t mib) {
    Gpu g;
    const size_t bytes = mib << 20;
    std::vector<uint8_t> data(bytes);
    // Compressible but not trivial: a texture-like mix of gradients and noise.
    uint32_t s = 7;
    // Smooth 16-bit ramps with 2 bits of noise in the low byte: like HDR texels / vertex data.
    for (size_t i = 0; i < bytes; i += 2) {
        s = s * 1103515245u + 12345u;
        const uint16_t v = uint16_t((i >> 5) + ((s >> 16) & 3));
        data[i] = uint8_t(v);
        data[i + 1] = uint8_t(v >> 8);
    }
    MTL::Buffer* dst = g.buffer(bytes);
    NS::Error* err = nullptr;
    auto* qd = MTL::IOCommandQueueDescriptor::alloc()->init();
    MTL::IOCommandQueue* ioq = g.dev->newIOCommandQueue(qd, &err);
    if (!ioq) die("newIOCommandQueue");
    const char* dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    std::printf("mtlio: %zu MiB payload, chunk %zu KiB\n", mib, MTL::IOCompressionContextDefaultChunkSize() >> 10);
    struct C { MTL::IOCompressionMethod m; const char* n; };
    for (C c : {C{MTL::IOCompressionMethodLZ4, "lz4"}, C{MTL::IOCompressionMethodLZFSE, "lzfse"},
                C{MTL::IOCompressionMethodZlib, "zlib"}, C{MTL::IOCompressionMethodLZMA, "lzma"},
                C{MTL::IOCompressionMethodLZBitmap, "lzbitmap"}}) {
        const std::string path = std::string(dir) + "/opt0_mtlio_" + c.n + ".bin";
        const double c0 = nowMs();
        MTL::IOCompressionContext ctx = MTL::IOCreateCompressionContext(path.c_str(), c.m, MTL::IOCompressionContextDefaultChunkSize());
        MTL::IOCompressionContextAppendData(ctx, data.data(), data.size());
        if (MTL::IOFlushAndDestroyCompressionContext(ctx) != MTL::IOCompressionStatusComplete) { std::printf("  %s: compress failed\n", c.n); continue; }
        const double compMs = nowMs() - c0;
        FILE* f = std::fopen(path.c_str(), "rb");
        std::fseek(f, 0, SEEK_END);
        const long fileBytes = std::ftell(f);
        std::fclose(f);
        // Copy with F_NOCACHE so the copy is not in the page cache: the first
        // load of it reads the SSD (no purge/sudo needed).
        const std::string cold = path + ".cold";
        {
            FILE* in = std::fopen(path.c_str(), "rb");
            const int fd = open(cold.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            fcntl(fd, F_NOCACHE, 1);
            std::vector<char> buf(1 << 20);
            size_t k;
            while ((k = std::fread(buf.data(), 1, buf.size(), in)) > 0) write(fd, buf.data(), k);
            fsync(fd);
            close(fd);
            std::fclose(in);
        }
        double coldMs = 0;
        {
            MTL::IOFileHandle* hc = g.dev->newIOHandle(NS::URL::fileURLWithPath(str(cold.c_str())), c.m, &err);
            MTL::IOCommandBuffer* io = ioq->commandBuffer();
            io->loadBuffer(dst, 0, bytes, hc, 0);
            const double t0 = nowMs();
            io->commit();
            io->waitUntilCompleted();
            coldMs = nowMs() - t0;
            hc->release();
            unlink(cold.c_str());
        }
        NS::URL* url = NS::URL::fileURLWithPath(str(path.c_str()));
        MTL::IOFileHandle* h = g.dev->newIOHandle(url, c.m, &err);
        if (!h) { std::printf("  %s: newIOHandle failed\n", c.n); continue; }
        std::vector<double> t;
        for (int r = 0; r < 5; ++r) {
            std::memset(dst->contents(), 0, bytes);
            MTL::IOCommandBuffer* io = ioq->commandBuffer();
            io->loadBuffer(dst, 0, bytes, h, 0);
            const double t0 = nowMs();
            io->commit();
            io->waitUntilCompleted();
            t.push_back(nowMs() - t0);
            if (io->status() != MTL::IOStatusComplete) die(std::string(c.n) + ": IO status");
        }
        std::sort(t.begin(), t.end());
        const bool ok = std::memcmp(dst->contents(), data.data(), bytes) == 0;
        std::printf("  %-9s ratio %5.2f  compress %8.1f ms  cold %7.2f ms (%6.2f GB/s) warm p50 %7.2f ms -> %6.2f GB/s out  %s\n", c.n,
                    double(bytes) / double(fileBytes), compMs, coldMs, double(bytes) / (coldMs * 1e-3) * 1e-9, t[t.size() / 2], double(bytes) / (t[t.size() / 2] * 1e-3) * 1e-9,
                    ok ? "exact" : "MISMATCH");
        h->release();
        unlink(path.c_str());
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string c = argc > 1 ? argv[1] : "gemm";
    if (c == "gemm") caseGemm(argc > 2 ? std::atoi(argv[2]) : 4096);
    else if (c == "mtlio") caseMtlio(argc > 2 ? size_t(std::atoi(argv[2])) : 256);
    else die("usage: api_probe gemm [n] | mtlio [MiB]");
    return 0;
}
