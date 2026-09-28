// F2.3 barrier legality spike: a MEASUREMENT TOOL, not engine code.
//
// It deliberately breaks two engine rules so it stays self-contained:
//   * it creates buffers/textures/heaps directly through MTL::Device
//     (the engine goes through GpuMemory), and
//   * it compiles its MSL from a source string at runtime.
//
// One process runs ONE case/variant (so a validation abort only kills that
// run) and prints a single `SPIKE ...` line.  See README.md.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

using u8  = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

constexpr u32 kDim = 2048;             // 2048 x 2048 targets, 4M-element buffers
constexpr u32 kN   = kDim * kDim;

struct Config {
    u32 reps  = 20;    // repetitions encoded in one command buffer
    u32 iters = 3000;  // producer delay-loop iterations per thread
    u32 runs  = 1;     // whole-run repetitions (cost cases take the median)
} g_cfg;

NS::String* str(const char* s) { return NS::String::string(s, NS::UTF8StringEncoding); }

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "barrier_spike: %s\n", msg.c_str());
    std::exit(3);
}

// ---------------------------------------------------------------------------
// MSL.  produce(i, p) = val(i, salt) ^ (LCG chain over p.iters & p.zero); zero
// is 0 at runtime, so the compiler cannot drop the loop but the value is known.
// Values are in [2^23, 2^24): non-zero and exact as float32 (R32Float/Depth32).
// ---------------------------------------------------------------------------
const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params { uint n; uint salt; uint iters; uint zero; uint width; uint rev; uint pad0; uint pad1; };

inline uint val(uint i, uint salt) { return 0x800000u | (((i * 2654435761u) ^ (salt * 40503u)) & 0x7FFFFFu); }
inline uint produce(uint i, constant Params& p) {
    uint a = i | 1u;
    for (uint k = 0; k < p.iters; ++k) a = a * 1664525u + 1013904223u;
    return val(i, p.salt) ^ (a & p.zero);
}
inline uint srcIndex(uint i, constant Params& p) { return p.rev ? (p.n - 1u - i) : i; }

// ---- compute ----
kernel void k_fill(device uint* out [[buffer(0)]], constant Params& p [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    out[i] = produce(i, p);
}
kernel void k_copy(device const uint* src [[buffer(0)]], device uint* out [[buffer(1)]],
                   constant Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    out[i] = src[srcIndex(i, p)];
}
kernel void k_tex_to_buf(texture2d<float, access::read> t [[texture(0)]], device uint* out [[buffer(0)]],
                         constant Params& p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.width) return;
    out[gid.y * p.width + gid.x] = uint(t.read(gid).x);
}
kernel void k_inc(device uint* buf [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    buf[i] = buf[i] + 1u;
}

// ---- vertex ----
struct VOut { float4 pos [[position]]; };
vertex VOut v_fullscreen(uint vid [[vertex_id]]) {
    float2 q = float2(float((vid << 1) & 2u), float(vid & 2u));
    VOut o; o.pos = float4(q * 2.0 - 1.0, 0.5, 1.0); return o;
}
struct PVOut { float4 pos [[position]]; float ps [[point_size]]; };
vertex PVOut v_pts_write(uint vid [[vertex_id]], device uint* tmp [[buffer(0)]], constant Params& p [[buffer(1)]]) {
    tmp[vid] = produce(vid, p);
    PVOut o; o.pos = float4(2.0, 2.0, 0.0, 1.0); o.ps = 1.0; return o;   // clipped away
}
vertex PVOut v_pts_read(uint vid [[vertex_id]], device const uint* src [[buffer(0)]],
                        device uint* out [[buffer(1)]], constant Params& p [[buffer(2)]]) {
    out[vid] = src[srcIndex(vid, p)];
    PVOut o; o.pos = float4(2.0, 2.0, 0.0, 1.0); o.ps = 1.0; return o;
}

// ---- fragment ----
fragment half4 f_dummy() { return half4(0.0h); }
inline uint pixelIndex(float4 pos, constant Params& p) { return uint(pos.y) * p.width + uint(pos.x); }
fragment half4 f_write(float4 pos [[position]], device uint* tmp [[buffer(0)]], constant Params& p [[buffer(1)]]) {
    uint i = pixelIndex(pos, p);
    tmp[i] = produce(i, p);
    return half4(0.0h);
}
fragment half4 f_read(float4 pos [[position]], device const uint* src [[buffer(0)]],
                      device uint* out [[buffer(1)]], constant Params& p [[buffer(2)]]) {
    uint i = pixelIndex(pos, p);
    out[i] = src[srcIndex(i, p)];
    return half4(0.0h);
}
fragment float4 f_write_color(float4 pos [[position]], constant Params& p [[buffer(0)]]) {
    return float4(float(produce(pixelIndex(pos, p), p)), 0.0, 0.0, 0.0);
}
struct DOut { float d [[depth(any)]]; };
fragment DOut f_write_depth(float4 pos [[position]], constant Params& p [[buffer(0)]]) {
    DOut o; o.d = float(produce(pixelIndex(pos, p), p)) * (1.0 / 16777216.0); return o;
}
fragment half4 f_read_depth(float4 pos [[position]], depth2d<float, access::read> d [[texture(0)]],
                            device uint* out [[buffer(0)]], constant Params& p [[buffer(1)]]) {
    out[pixelIndex(pos, p)] = uint(d.read(uint2(pos.xy)) * 16777216.0);
    return half4(0.0h);
}
fragment half4 f_read_tex_uint(float4 pos [[position]], texture2d<uint, access::read> t [[texture(0)]],
                               device uint* out [[buffer(0)]], constant Params& p [[buffer(1)]]) {
    out[pixelIndex(pos, p)] = t.read(uint2(pos.xy)).x;
    return half4(0.0h);
}
)MSL";

struct Params { u32 n, salt, iters, zero, width, rev, pad0, pad1; };
constexpr size_t kParamStride = 256;

u32 val(u32 i, u32 salt) { return 0x800000u | (((i * 2654435761u) ^ (salt * 40503u)) & 0x7FFFFFu); }

// ---------------------------------------------------------------------------
// Variant parsing: none | <q|p|e>_<after>_<before>[_<alias|both>]
// q = barrierAfterQueueStages at the start of the consumer encoder
// p = barrierAfterStages at the end of the producer encoder
// e = barrierAfterEncoderStages inside one encoder
// ---------------------------------------------------------------------------
struct Barrier {
    char kind = 'n';
    MTL::Stages after  = 0;
    MTL::Stages before = 0;
    MTL4::VisibilityOptions vis = MTL4::VisibilityOptionDevice;
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t at = s.find(sep, start);
        out.push_back(s.substr(start, at == std::string::npos ? at : at - start));
        if (at == std::string::npos) break;
        start = at + 1;
    }
    return out;
}

MTL::Stages parseStages(const std::string& s) {
    MTL::Stages r = 0;
    for (const std::string& t : split(s, '+')) {
        if (t == "vertex") r |= MTL::StageVertex;
        else if (t == "fragment") r |= MTL::StageFragment;
        else if (t == "tile") r |= MTL::StageTile;
        else if (t == "dispatch") r |= MTL::StageDispatch;
        else if (t == "blit") r |= MTL::StageBlit;
        else if (t == "object") r |= MTL::StageObject;
        else if (t == "mesh") r |= MTL::StageMesh;
        else if (t == "all") r |= MTL::StageAll;
        else die("unknown stage '" + t + "'");
    }
    return r;
}

Barrier parseBarrier(const std::string& variant) {
    Barrier b;
    const std::vector<std::string> t = split(variant, '_');
    if (t[0] == "none") return b;
    if (t.size() < 3 || t[0].size() != 1 || !std::strchr("qpe", t[0][0])) die("bad variant '" + variant + "'");
    b.kind   = t[0][0];
    b.after  = parseStages(t[1]);
    b.before = parseStages(t[2]);
    if (t.size() > 3) {
        if (t[3] == "alias") b.vis = MTL4::VisibilityOptionResourceAlias;
        else if (t[3] == "both") b.vis = MTL4::VisibilityOptionDevice | MTL4::VisibilityOptionResourceAlias;
        else if (t[3] == "none") b.vis = MTL4::VisibilityOptionNone;
        else die("bad visibility '" + t[3] + "'");
    }
    return b;
}

// Consumer-side (start of the consumer encoder).
void barrierBefore(MTL4::CommandEncoder* enc, const Barrier& b) {
    if (b.kind == 'q') enc->barrierAfterQueueStages(b.after, b.before, b.vis);
}
// Producer-side (end of the producer encoder).
void barrierAtEnd(MTL4::CommandEncoder* enc, const Barrier& b) {
    if (b.kind == 'p') enc->barrierAfterStages(b.after, b.before, b.vis);
}
// Inside one encoder, between two commands.
void barrierBetween(MTL4::CommandEncoder* enc, const Barrier& b) {
    if (b.kind == 'e') enc->barrierAfterEncoderStages(b.after, b.before, b.vis);
}

// ---------------------------------------------------------------------------
// Rig: device, queue, compiler, pipelines, one command buffer per run.
// ---------------------------------------------------------------------------
struct Rig {
    MTL::Device* dev          = nullptr;
    MTL4::CommandQueue* queue = nullptr;
    MTL4::Compiler* compiler  = nullptr;
    MTL::Library* lib         = nullptr;
    MTL4::CommandAllocator* allocator = nullptr;
    MTL4::CommandBuffer* cb   = nullptr;
    MTL::SharedEvent* event   = nullptr;
    MTL::ResidencySet* rs     = nullptr;
    std::vector<const MTL::Allocation*> allocs;
    std::vector<NS::Object*> temps;                    // released after the run
    std::map<std::string, NS::Object*> pipelines;
    u64 eventValue = 0;
    std::string gpuError;

    Rig() {
        dev = MTL::CreateSystemDefaultDevice();
        if (!dev) die("no Metal device");
        if (!dev->supportsFamily(MTL::GPUFamilyMetal4)) die("device does not support Metal 4");
        queue = dev->newMTL4CommandQueue();
        NS::Error* err = nullptr;
        MTL4::CompilerDescriptor* cd = MTL4::CompilerDescriptor::alloc()->init();
        compiler = dev->newCompiler(cd, &err);
        cd->release();
        if (!compiler) die("newCompiler failed");
        MTL::CompileOptions* opts = MTL::CompileOptions::alloc()->init();
        opts->setLanguageVersion(MTL::LanguageVersion4_0);
        lib = dev->newLibrary(str(kShaderSource), opts, &err);
        opts->release();
        if (!lib) die(std::string("MSL compile failed: ") + (err ? err->localizedDescription()->utf8String() : "?"));
        allocator = dev->newCommandAllocator();
        cb        = dev->newCommandBuffer();
        event     = dev->newSharedEvent();
    }

    // ---- resources (bench tool: created directly with the device) ----
    MTL::Buffer* buffer(u64 size, MTL::ResourceOptions opts) {
        MTL::Buffer* b = dev->newBuffer(size, opts);
        if (!b) die("newBuffer failed");
        allocs.push_back(b);
        temps.push_back(b);
        return b;
    }
    MTL::Buffer* priv(u64 size) { return buffer(size, MTL::ResourceStorageModePrivate); }
    MTL::Buffer* shared(u64 size, int fill = 0) {
        MTL::Buffer* b = buffer(size, MTL::ResourceStorageModeShared);
        std::memset(b->contents(), fill, size);
        return b;
    }
    MTL::Texture* texture(MTL::PixelFormat fmt, u32 dim, MTL::TextureUsage usage) {
        MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(fmt, dim, dim, false);
        d->setUsage(usage);
        d->setStorageMode(MTL::StorageModePrivate);
        MTL::Texture* t = dev->newTexture(d);
        if (!t) die("newTexture failed");
        allocs.push_back(t);
        temps.push_back(t);
        return t;
    }
    MTL::Heap* placementHeap(u64 size) {
        MTL::HeapDescriptor* d = MTL::HeapDescriptor::alloc()->init();
        d->setType(MTL::HeapTypePlacement);
        d->setStorageMode(MTL::StorageModePrivate);
        d->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
        d->setSize(size);
        MTL::Heap* h = dev->newHeap(d);
        d->release();
        if (!h) die("newHeap failed");
        allocs.push_back(h);
        temps.push_back(h);
        return h;
    }

    // ---- pipelines ----
    MTL4::LibraryFunctionDescriptor* fn(const char* name) {
        MTL4::LibraryFunctionDescriptor* f = MTL4::LibraryFunctionDescriptor::alloc()->init();
        f->setLibrary(lib);
        f->setName(str(name));
        return f;
    }
    MTL::ComputePipelineState* compute(const char* name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return static_cast<MTL::ComputePipelineState*>(it->second);
        MTL4::LibraryFunctionDescriptor* f = fn(name);
        MTL4::ComputePipelineDescriptor* d = MTL4::ComputePipelineDescriptor::alloc()->init();
        d->setComputeFunctionDescriptor(f);
        NS::Error* err = nullptr;
        MTL::ComputePipelineState* p = compiler->newComputePipelineState(d, nullptr, &err);
        d->release();
        f->release();
        if (!p) die(std::string("compute pipeline ") + name + ": " + (err ? err->localizedDescription()->utf8String() : "?"));
        pipelines[name] = p;
        return p;
    }
    // colorFormat == PixelFormatInvalid: no colour attachment (depth-only pass).
    MTL::RenderPipelineState* render(const char* vs, const char* fs, MTL::PixelFormat colorFormat) {
        const std::string key = std::string(vs) + "/" + fs + "/" + std::to_string(int(colorFormat));
        auto it = pipelines.find(key);
        if (it != pipelines.end()) return static_cast<MTL::RenderPipelineState*>(it->second);
        MTL4::LibraryFunctionDescriptor* v = fn(vs);
        MTL4::LibraryFunctionDescriptor* f = fn(fs);
        MTL4::RenderPipelineDescriptor* d = MTL4::RenderPipelineDescriptor::alloc()->init();
        d->setVertexFunctionDescriptor(v);
        d->setFragmentFunctionDescriptor(f);
        if (colorFormat != MTL::PixelFormatInvalid) d->colorAttachments()->object(0)->setPixelFormat(colorFormat);
        NS::Error* err = nullptr;
        MTL::RenderPipelineState* p = compiler->newRenderPipelineState(d, nullptr, &err);
        d->release();
        f->release();
        v->release();
        if (!p) die("render pipeline " + key + ": " + (err ? err->localizedDescription()->utf8String() : "?"));
        pipelines[key] = p;
        return p;
    }
    MTL::DepthStencilState* depthAlways() {
        MTL::DepthStencilDescriptor* d = MTL::DepthStencilDescriptor::alloc()->init();
        d->setDepthCompareFunction(MTL::CompareFunctionAlways);
        d->setDepthWriteEnabled(true);
        MTL::DepthStencilState* s = dev->newDepthStencilState(d);
        d->release();
        temps.push_back(s);
        return s;
    }
    MTL4::ArgumentTable* table(u32 buffers, u32 textures) {
        MTL4::ArgumentTableDescriptor* d = MTL4::ArgumentTableDescriptor::alloc()->init();
        d->setMaxBufferBindCount(buffers);
        d->setMaxTextureBindCount(textures);
        NS::Error* err = nullptr;
        MTL4::ArgumentTable* t = dev->newArgumentTable(d, &err);
        d->release();
        if (!t) die("newArgumentTable failed");
        temps.push_back(t);
        return t;
    }

    // ---- encoders ----
    void begin() {
        allocator->reset();
        cb->beginCommandBuffer(allocator);
    }
    MTL4::ComputeCommandEncoder* computeEncoder() { return cb->computeCommandEncoder(); }

    // Colour target (may be null) and depth target (may be null).
    MTL4::RenderCommandEncoder* renderEncoder(MTL::Texture* color, MTL::Texture* depth, bool storeColor,
                                              bool storeDepth) {
        MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
        if (color) {
            auto* c = pd->colorAttachments()->object(0);
            c->setTexture(color);
            c->setLoadAction(MTL::LoadActionDontCare);
            c->setStoreAction(storeColor ? MTL::StoreActionStore : MTL::StoreActionDontCare);
        }
        if (depth) {
            auto* d = pd->depthAttachment();
            d->setTexture(depth);
            d->setLoadAction(MTL::LoadActionClear);
            d->setClearDepth(0.0);
            d->setStoreAction(storeDepth ? MTL::StoreActionStore : MTL::StoreActionDontCare);
        }
        MTL4::RenderCommandEncoder* enc = cb->renderCommandEncoder(pd);
        pd->release();
        const double dim = color ? double(color->width()) : double(depth->width());
        enc->setViewport(MTL::Viewport{0.0, 0.0, dim, dim, 0.0, 1.0});
        return enc;
    }

    // Commits, waits, returns GPU milliseconds (0 when feedback is missing).
    double run() {
        cb->endCommandBuffer();
        MTL::ResidencySetDescriptor* rd = MTL::ResidencySetDescriptor::alloc()->init();
        rd->setInitialCapacity(allocs.size() + 8);
        NS::Error* err = nullptr;
        rs = dev->newResidencySet(rd, &err);
        rd->release();
        if (!rs) die("newResidencySet failed");
        for (const MTL::Allocation* a : allocs) rs->addAllocation(a);
        rs->commit();
        queue->addResidencySet(rs);

        std::atomic<bool> got{false};
        std::atomic<double> gpuMs{0.0};
        std::string error;
        MTL4::CommitOptions* opts = MTL4::CommitOptions::alloc()->init();   // fresh per commit (see MetalContext)
        opts->addFeedbackHandler([&](MTL4::CommitFeedback* fb) {
            if (fb->error()) error = fb->error()->localizedDescription()->utf8String();
            gpuMs = (fb->GPUEndTime() - fb->GPUStartTime()) * 1000.0;
            got = true;
        });
        const MTL4::CommandBuffer* bufs[] = {cb};
        queue->commit(bufs, 1, opts);
        opts->release();
        queue->signalEvent(event, ++eventValue);
        if (!event->waitUntilSignaledValue(eventValue, 60000)) die("GPU timeout");
        for (int i = 0; i < 2000 && !got; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        gpuError = error;

        queue->removeResidencySet(rs);
        rs->release();
        rs = nullptr;
        return gpuMs;
    }

    void endRun() {
        for (NS::Object* o : temps) o->release();
        temps.clear();
        allocs.clear();
    }
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
struct Result {
    u32 wrongReps  = 0;
    u32 reps       = 0;
    u64 badElems   = 0;
    double gpuMs   = 0;
    bool haveFirst = false;
    u32 expected   = 0, got = 0;
    u32 firstIndex = 0;
};

// Compares a shared buffer with val(i, salt).
// mirrored: the consumer copied src[n - 1 - i] to out[i].
void checkBuffer(Result& r, MTL::Buffer* buf, u32 salt, u32 n = kN, bool mirrored = false) {
    const auto* p = static_cast<const u32*>(buf->contents());
    u64 bad = 0;
    for (u32 i = 0; i < n; ++i) {
        const u32 want = val(mirrored ? n - 1 - i : i, salt);
        if (p[i] != want) {
            if (!r.haveFirst) { r.haveFirst = true; r.expected = want; r.got = p[i]; r.firstIndex = i; }
            ++bad;
        }
    }
    ++r.reps;
    if (bad) { ++r.wrongReps; r.badElems += bad; }
}

// Params for every pass live in one shared buffer, 256 bytes apart.
struct ParamBlock {
    MTL::Buffer* buf = nullptr;
    u32 count = 0;
    u64 add(u32 salt, u32 iters, u32 width, u32 rev, u32 n = kN) {
        Params p{n, salt, iters, 0, width, rev, 0, 0};
        std::memcpy(static_cast<u8*>(buf->contents()) + count * kParamStride, &p, sizeof(p));
        return buf->gpuAddress() + (count++) * kParamStride;
    }
};

ParamBlock makeParams(Rig& r, u32 slots) {
    ParamBlock pb;
    pb.buf = r.shared(slots * kParamStride);
    return pb;
}

const MTL::Size kGroup1D = MTL::Size::Make(256, 1, 1);
const MTL::Size kGroup2D = MTL::Size::Make(16, 16, 1);

constexpr MTL::RenderStages kVF = MTL::RenderStageVertex | MTL::RenderStageFragment;

// ---------------------------------------------------------------------------
// Cases.  Each encodes g_cfg.reps independent repetitions in one command
// buffer, runs it and checks every repetition on the CPU.
// ---------------------------------------------------------------------------

// 1 + 2: compute writes a buffer, a render pass reads it in vertex (1) or fragment (2).
Result caseComputeToRender(Rig& r, const Barrier& b, bool fragmentReads) {
    Result res;
    const u32 reps = g_cfg.reps;
    ParamBlock pb = makeParams(r, reps);
    MTL::ComputePipelineState* fill = r.compute("k_fill");
    MTL::RenderPipelineState* pso = fragmentReads ? r.render("v_fullscreen", "f_read", MTL::PixelFormatR8Unorm)
                                                  : r.render("v_pts_read", "f_dummy", MTL::PixelFormatR8Unorm);
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Buffer* src = r.priv(kN * 4);
        MTL::Buffer* out = r.shared(kN * 4);
        MTL::Texture* dummy = r.texture(MTL::PixelFormatR8Unorm, kDim, MTL::TextureUsageRenderTarget);
        outs.push_back(out);
        const u64 prm = pb.add(k + 1, g_cfg.iters, kDim, 0);

        MTL4::ArgumentTable* t1 = r.table(4, 0);
        t1->setAddress(src->gpuAddress(), 0);
        t1->setAddress(prm, 1);
        MTL4::ComputeCommandEncoder* ce = r.computeEncoder();
        ce->setComputePipelineState(fill);
        ce->setArgumentTable(t1);
        ce->dispatchThreads(MTL::Size::Make(kN, 1, 1), kGroup1D);
        barrierAtEnd(ce, b);
        ce->endEncoding();

        MTL4::ArgumentTable* t2 = r.table(4, 0);
        t2->setAddress(src->gpuAddress(), 0);
        t2->setAddress(out->gpuAddress(), 1);
        t2->setAddress(prm, 2);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(dummy, nullptr, false, false);
        barrierBefore(re, b);
        re->setRenderPipelineState(pso);
        re->setArgumentTable(t2, kVF);
        if (fragmentReads) re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        else re->drawPrimitives(MTL::PrimitiveTypePoint, 0, kN);
        re->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1);
    return res;
}

// 3: render pass writes a colour attachment, compute reads it.
Result caseRenderToCompute(Rig& r, const Barrier& b) {
    Result res;
    const u32 reps = g_cfg.reps;
    ParamBlock pb = makeParams(r, reps * 2);
    MTL::RenderPipelineState* pso = r.render("v_fullscreen", "f_write_color", MTL::PixelFormatR32Float);
    MTL::ComputePipelineState* rd = r.compute("k_tex_to_buf");
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Texture* tex = r.texture(MTL::PixelFormatR32Float, kDim,
                                      MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        MTL::Buffer* out = r.shared(kN * 4);
        outs.push_back(out);
        const u64 p1 = pb.add(k + 1, g_cfg.iters, kDim, 0);
        const u64 p2 = pb.add(k + 1, 0, kDim, 0);

        MTL4::ArgumentTable* t1 = r.table(2, 0);
        t1->setAddress(p1, 0);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(tex, nullptr, true, false);
        re->setRenderPipelineState(pso);
        re->setArgumentTable(t1, kVF);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        barrierAtEnd(re, b);
        re->endEncoding();

        MTL4::ArgumentTable* t2 = r.table(2, 1);
        t2->setTexture(tex->gpuResourceID(), 0);
        t2->setAddress(out->gpuAddress(), 0);
        t2->setAddress(p2, 1);
        MTL4::ComputeCommandEncoder* ce = r.computeEncoder();
        barrierBefore(ce, b);
        ce->setComputePipelineState(rd);
        ce->setArgumentTable(t2);
        ce->dispatchThreads(MTL::Size::Make(kDim, kDim, 1), kGroup2D);
        ce->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1);
    return res;
}

// 4: render pass writes a Depth32Float texture, a second pass samples it.
// dim/iters are parameters so cost_render can reuse it with tiny passes.
Result caseRenderToRender(Rig& r, const Barrier& b, u32 reps, u32 dim, u32 iters) {
    Result res;
    const u32 n = dim * dim;
    ParamBlock pb = makeParams(r, reps * 2);
    MTL::RenderPipelineState* p1 = r.render("v_fullscreen", "f_write_depth", MTL::PixelFormatInvalid);
    MTL::RenderPipelineState* p2 = r.render("v_fullscreen", "f_read_depth", MTL::PixelFormatR8Unorm);
    MTL::DepthStencilState* ds = r.depthAlways();
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Texture* depth = r.texture(MTL::PixelFormatDepth32Float, dim,
                                        MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        MTL::Texture* dummy = r.texture(MTL::PixelFormatR8Unorm, dim, MTL::TextureUsageRenderTarget);
        MTL::Buffer* out = r.shared(n * 4);
        outs.push_back(out);
        const u64 a1 = pb.add(k + 1, iters, dim, 0, n);
        const u64 a2 = pb.add(k + 1, 0, dim, 0, n);

        MTL4::ArgumentTable* t1 = r.table(2, 0);
        t1->setAddress(a1, 0);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(nullptr, depth, false, true);
        re->setRenderPipelineState(p1);
        re->setDepthStencilState(ds);
        re->setArgumentTable(t1, kVF);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        barrierAtEnd(re, b);
        re->endEncoding();

        MTL4::ArgumentTable* t2 = r.table(2, 1);
        t2->setTexture(depth->gpuResourceID(), 0);
        t2->setAddress(out->gpuAddress(), 0);
        t2->setAddress(a2, 1);
        MTL4::RenderCommandEncoder* re2 = r.renderEncoder(dummy, nullptr, false, false);
        barrierBefore(re2, b);
        re2->setRenderPipelineState(p2);
        re2->setArgumentTable(t2, kVF);
        re2->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        re2->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1, n);
    return res;
}

// 5: blit (compute-encoder copy) fills a texture, a render pass samples it.
Result caseBlitToFragment(Rig& r, const Barrier& b) {
    Result res;
    const u32 reps   = g_cfg.reps;
    const u32 copies = 24;  // repeated identical copies make the producer slow
    ParamBlock pb = makeParams(r, reps);
    MTL::RenderPipelineState* pso = r.render("v_fullscreen", "f_read_tex_uint", MTL::PixelFormatR8Unorm);
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Buffer* src = r.shared(kN * 4);
        auto* sp = static_cast<u32*>(src->contents());
        for (u32 i = 0; i < kN; ++i) sp[i] = val(i, k + 1);
        MTL::Texture* tex = r.texture(MTL::PixelFormatR32Uint, kDim, MTL::TextureUsageShaderRead);
        MTL::Texture* dummy = r.texture(MTL::PixelFormatR8Unorm, kDim, MTL::TextureUsageRenderTarget);
        MTL::Buffer* out = r.shared(kN * 4);
        outs.push_back(out);
        const u64 prm = pb.add(k + 1, 0, kDim, 0);

        MTL4::ComputeCommandEncoder* ce = r.computeEncoder();
        for (u32 c = 0; c < copies; ++c)
            ce->copyFromBuffer(src, 0, kDim * 4, kN * 4, MTL::Size::Make(kDim, kDim, 1), tex, 0, 0,
                               MTL::Origin::Make(0, 0, 0));
        barrierAtEnd(ce, b);
        ce->endEncoding();

        MTL4::ArgumentTable* t = r.table(2, 1);
        t->setTexture(tex->gpuResourceID(), 0);
        t->setAddress(out->gpuAddress(), 0);
        t->setAddress(prm, 1);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(dummy, nullptr, false, false);
        barrierBefore(re, b);
        re->setRenderPipelineState(pso);
        re->setArgumentTable(t, kVF);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        re->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1);
    return res;
}

// 6: inside ONE render encoder, draw A writes a buffer (fragment or vertex
// shader), draw B reads it (fragment or vertex) from the mirrored index.
// Flow is derived from the barrier stages: after=vertex -> A writes in the
// vertex shader, else fragment; before=vertex -> B reads in the vertex shader.
// "none_<w><r>" selects the flow explicitly (f = fragment, v = vertex).
Result caseIntraRender(Rig& r, const std::string& variant) {
    Result res;
    char w = 'f', rd = 'f';
    Barrier b;
    if (variant.rfind("none_", 0) == 0) {
        w  = variant[5];
        rd = variant[6];
    } else {
        b = parseBarrier(variant);
        if (b.after == MTL::StageVertex) w = 'v';
        if (b.before == MTL::StageVertex) rd = 'v';
    }
    const u32 reps = g_cfg.reps;
    ParamBlock pb = makeParams(r, reps * 2);
    MTL::RenderPipelineState* pa = w == 'f' ? r.render("v_fullscreen", "f_write", MTL::PixelFormatR8Unorm)
                                            : r.render("v_pts_write", "f_dummy", MTL::PixelFormatR8Unorm);
    MTL::RenderPipelineState* pr = rd == 'f' ? r.render("v_fullscreen", "f_read", MTL::PixelFormatR8Unorm)
                                             : r.render("v_pts_read", "f_dummy", MTL::PixelFormatR8Unorm);
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Buffer* tmp = r.priv(kN * 4);
        MTL::Buffer* out = r.shared(kN * 4);
        MTL::Texture* dummy = r.texture(MTL::PixelFormatR8Unorm, kDim, MTL::TextureUsageRenderTarget);
        outs.push_back(out);
        const u64 a1 = pb.add(k + 1, g_cfg.iters, kDim, 0);
        const u64 a2 = pb.add(k + 1, 0, kDim, 1);

        MTL4::ArgumentTable* t1 = r.table(4, 0);
        t1->setAddress(tmp->gpuAddress(), 0);
        t1->setAddress(a1, 1);
        MTL4::ArgumentTable* t2 = r.table(4, 0);
        t2->setAddress(tmp->gpuAddress(), 0);
        t2->setAddress(out->gpuAddress(), 1);
        t2->setAddress(a2, 2);

        MTL4::RenderCommandEncoder* re = r.renderEncoder(dummy, nullptr, false, false);
        re->setRenderPipelineState(pa);
        re->setArgumentTable(t1, kVF);
        if (w == 'f') re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        else re->drawPrimitives(MTL::PrimitiveTypePoint, 0, kN);
        barrierBetween(re, b);
        re->setRenderPipelineState(pr);
        re->setArgumentTable(t2, kVF);
        if (rd == 'f') re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        else re->drawPrimitives(MTL::PrimitiveTypePoint, 0, kN);
        re->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1, kN, true);
    return res;
}

// 7: two dispatches in one compute encoder; the second reads the mirrored index.
Result caseIntraCompute(Rig& r, const Barrier& b) {
    Result res;
    const u32 reps = g_cfg.reps;
    ParamBlock pb = makeParams(r, reps * 2);
    MTL::ComputePipelineState* fill = r.compute("k_fill");
    MTL::ComputePipelineState* copy = r.compute("k_copy");
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Buffer* tmp = r.priv(kN * 4);
        MTL::Buffer* out = r.shared(kN * 4);
        outs.push_back(out);
        const u64 a1 = pb.add(k + 1, g_cfg.iters, kDim, 0);
        const u64 a2 = pb.add(k + 1, 0, kDim, 1);
        MTL4::ArgumentTable* t1 = r.table(4, 0);
        t1->setAddress(tmp->gpuAddress(), 0);
        t1->setAddress(a1, 1);
        MTL4::ArgumentTable* t2 = r.table(4, 0);
        t2->setAddress(tmp->gpuAddress(), 0);
        t2->setAddress(out->gpuAddress(), 1);
        t2->setAddress(a2, 2);
        MTL4::ComputeCommandEncoder* ce = r.computeEncoder();
        ce->setComputePipelineState(fill);
        ce->setArgumentTable(t1);
        ce->dispatchThreads(MTL::Size::Make(kN, 1, 1), kGroup1D);
        barrierBetween(ce, b);
        ce->setComputePipelineState(copy);
        ce->setArgumentTable(t2);
        ce->dispatchThreads(MTL::Size::Make(kN, 1, 1), kGroup1D);
        ce->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1, kN, true);
    return res;
}

// 8: two buffers aliased at offset 0 of a placement heap.  A: compute (slow);
// B: written by a render pass (fragment device write) and copied to a shared
// buffer.  A's late writes would corrupt B unless the barrier orders them.
Result caseAlias(Rig& r, const Barrier& b) {
    Result res;
    const u32 reps = g_cfg.reps;
    ParamBlock pb = makeParams(r, reps * 2);
    MTL::ComputePipelineState* fill = r.compute("k_fill");
    MTL::RenderPipelineState* pso = r.render("v_fullscreen", "f_write", MTL::PixelFormatR8Unorm);
    const MTL::ResourceOptions opts = MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeUntracked;
    const MTL::SizeAndAlign sa = r.dev->heapBufferSizeAndAlign(kN * 4, opts);
    std::vector<MTL::Buffer*> outs;
    r.begin();
    for (u32 k = 0; k < reps; ++k) {
        MTL::Heap* heap = r.placementHeap(sa.size);                 // residency through the heap only
        MTL::Buffer* A = heap->newBuffer(kN * 4, opts, 0);
        MTL::Buffer* B = heap->newBuffer(kN * 4, opts, 0);
        r.temps.push_back(A);
        r.temps.push_back(B);
        MTL::Buffer* out = r.shared(kN * 4);
        MTL::Texture* dummy = r.texture(MTL::PixelFormatR8Unorm, kDim, MTL::TextureUsageRenderTarget);
        outs.push_back(out);
        const u64 pa = pb.add(k + 1, g_cfg.iters, kDim, 0);
        const u64 pbAddr = pb.add(k + 1001, 1, kDim, 0);

        MTL4::ArgumentTable* t1 = r.table(2, 0);
        t1->setAddress(A->gpuAddress(), 0);
        t1->setAddress(pa, 1);
        MTL4::ComputeCommandEncoder* ce = r.computeEncoder();
        ce->setComputePipelineState(fill);
        ce->setArgumentTable(t1);
        ce->dispatchThreads(MTL::Size::Make(kN, 1, 1), kGroup1D);
        barrierAtEnd(ce, b);
        ce->endEncoding();

        MTL4::ArgumentTable* t2 = r.table(2, 0);
        t2->setAddress(B->gpuAddress(), 0);
        t2->setAddress(pbAddr, 1);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(dummy, nullptr, false, false);
        barrierBefore(re, b);
        re->setRenderPipelineState(pso);
        re->setArgumentTable(t2, kVF);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        re->endEncoding();

        // Readback with its own barrier: wait for B's writer (Fragment) and for
        // A's kernel (Dispatch) so a late A overwrite shows up in the copy.
        MTL4::ComputeCommandEncoder* cp = r.computeEncoder();
        cp->barrierAfterQueueStages(MTL::StageFragment | MTL::StageDispatch, MTL::StageBlit,
                                    MTL4::VisibilityOptionDevice);
        cp->copyFromBuffer(B, 0, out, 0, kN * 4);
        cp->endEncoding();
    }
    res.gpuMs = r.run();
    for (u32 k = 0; k < reps; ++k) checkBuffer(res, outs[k], k + 1001);
    return res;
}

// cost_queue / cost_encoder: a chain of tiny dependent dispatches (buf[i] += 1).
// Without the barrier, lost updates would show as a wrong count.
Result caseCostChain(Rig& r, const Barrier& b, bool perEncoder) {
    Result res;
    const u32 dispatches = 1000, threads = 1024;
    MTL::ComputePipelineState* inc = r.compute("k_inc");
    MTL::Buffer* buf = r.shared(threads * 4);
    MTL4::ArgumentTable* t = r.table(1, 0);
    t->setAddress(buf->gpuAddress(), 0);
    r.begin();
    MTL4::ComputeCommandEncoder* ce = nullptr;
    for (u32 d = 0; d < dispatches; ++d) {
        const bool fresh = perEncoder || !ce;
        if (fresh) ce = r.computeEncoder();
        if (perEncoder) barrierBefore(ce, b);   // queue barrier at the start of every encoder
        else if (d > 0) barrierBetween(ce, b);
        if (fresh) {
            ce->setComputePipelineState(inc);
            ce->setArgumentTable(t);
        }
        ce->dispatchThreads(MTL::Size::Make(threads, 1, 1), MTL::Size::Make(256, 1, 1));
        if (perEncoder) ce->endEncoding();
    }
    if (!perEncoder) ce->endEncoding();
    res.gpuMs = r.run();
    const auto* p = static_cast<const u32*>(buf->contents());
    u32 minv = ~0u;
    for (u32 i = 0; i < threads; ++i) minv = std::min(minv, p[i]);
    res.reps = 1;
    res.haveFirst = true;
    res.expected = dispatches;
    res.got = minv;
    if (minv != dispatches) res.wrongReps = 1;
    return res;
}

// legality: no data flow at all, only "does the API / validation layer accept
// this barrier on this encoder type?".  Variant: <render|compute>_<q|p|e>_<after>_<before>.
Result caseLegality(Rig& r, const std::string& variant) {
    Result res;
    const size_t us = variant.find('_');
    const std::string encType = variant.substr(0, us);
    const Barrier b = parseBarrier(variant.substr(us + 1));
    MTL::Buffer* buf = r.shared(4096);
    MTL4::ArgumentTable* t = r.table(1, 0);
    t->setAddress(buf->gpuAddress(), 0);
    r.begin();
    if (encType == "render") {
        MTL::Texture* dummy = r.texture(MTL::PixelFormatR8Unorm, 64, MTL::TextureUsageRenderTarget);
        MTL::RenderPipelineState* pso = r.render("v_fullscreen", "f_dummy", MTL::PixelFormatR8Unorm);
        MTL4::RenderCommandEncoder* re = r.renderEncoder(dummy, nullptr, false, false);
        barrierBefore(re, b);
        re->setRenderPipelineState(pso);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        barrierBetween(re, b);
        re->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3);
        barrierAtEnd(re, b);
        re->endEncoding();
    } else {
        MTL4::ComputeCommandEncoder* ce = r.computeEncoder();
        barrierBefore(ce, b);
        ce->setComputePipelineState(r.compute("k_inc"));
        ce->setArgumentTable(t);
        ce->dispatchThreads(MTL::Size::Make(64, 1, 1), MTL::Size::Make(64, 1, 1));
        barrierBetween(ce, b);
        ce->dispatchThreads(MTL::Size::Make(64, 1, 1), MTL::Size::Make(64, 1, 1));
        barrierAtEnd(ce, b);
        ce->endEncoding();
    }
    res.gpuMs = r.run();
    res.reps = 1;
    return res;
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------
struct CaseDef {
    const char* name;
    std::vector<const char*> variants;
    std::function<Result(Rig&, const std::string&)> run;
    bool median;  // timing case: repeat g_cfg.runs times and take the median GPU time
};

std::vector<CaseDef> makeCases() {
    std::vector<CaseDef> c;
    c.push_back({"compute_to_vertex",
                 {"none", "q_dispatch_vertex", "q_dispatch_fragment", "q_dispatch_vertex+fragment",
                  "p_dispatch_vertex"},
                 [](Rig& r, const std::string& v) { return caseComputeToRender(r, parseBarrier(v), false); }, false});
    c.push_back({"compute_to_fragment",
                 {"none", "q_dispatch_fragment", "q_dispatch_vertex", "q_dispatch_vertex+fragment",
                  "q_dispatch_tile", "p_dispatch_fragment", "q_dispatch_fragment_alias", "q_dispatch_fragment_none"},
                 [](Rig& r, const std::string& v) { return caseComputeToRender(r, parseBarrier(v), true); }, false});
    c.push_back({"render_to_compute",
                 {"none", "q_fragment_dispatch", "q_vertex_dispatch", "p_fragment_dispatch", "q_tile_dispatch"},
                 [](Rig& r, const std::string& v) { return caseRenderToCompute(r, parseBarrier(v)); }, false});
    c.push_back({"render_to_render",
                 {"none", "q_fragment_fragment", "q_fragment_vertex", "q_fragment_vertex+fragment",
                  "q_fragment_tile", "q_vertex_vertex", "q_vertex_fragment", "p_fragment_vertex"},
                 [](Rig& r, const std::string& v) {
                     return caseRenderToRender(r, parseBarrier(v), g_cfg.reps, kDim, g_cfg.iters);
                 }, false});
    c.push_back({"blit_to_fragment",
                 {"none", "q_blit_fragment", "q_blit_vertex", "p_blit_fragment", "p_blit_vertex"},
                 [](Rig& r, const std::string& v) { return caseBlitToFragment(r, parseBarrier(v)); }, false});
    c.push_back({"intra_render",
                 {"none_ff", "none_fv", "none_vf", "e_fragment_fragment", "e_fragment_vertex", "e_vertex_fragment",
                  "e_fragment_tile", "e_vertex_vertex+fragment", "e_vertex_tile"},
                 [](Rig& r, const std::string& v) { return caseIntraRender(r, v); }, false});
    c.push_back({"intra_compute",
                 {"none", "e_dispatch_dispatch"},
                 [](Rig& r, const std::string& v) { return caseIntraCompute(r, parseBarrier(v)); }, false});
    c.push_back({"alias",
                 {"none", "q_dispatch_fragment", "q_dispatch_fragment_alias", "q_dispatch_fragment_both",
                  "q_dispatch_vertex", "q_dispatch_vertex_alias", "q_dispatch_vertex_both", "q_dispatch_fragment_none",
                  "q_dispatch_vertex_none"},
                 [](Rig& r, const std::string& v) { return caseAlias(r, parseBarrier(v)); }, false});
    c.push_back({"legality",
                 {"render_q_dispatch_vertex", "render_q_dispatch_fragment", "render_q_dispatch_tile",
                  "render_q_dispatch_dispatch", "render_q_dispatch_blit", "render_q_dispatch_object",
                  "render_q_dispatch_mesh", "render_q_fragment_tile", "render_q_fragment_fragment",
                  "render_q_vertex_fragment", "render_q_tile_fragment", "render_q_tile_vertex",
                  "render_q_blit_fragment", "render_q_all_all", "render_q_fragment_all",
                  "render_p_fragment_dispatch", "render_p_fragment_fragment", "render_p_vertex_vertex",
                  "render_p_fragment_vertex", "render_p_tile_dispatch", "render_p_fragment_tile",
                  "render_p_fragment_all", "render_e_vertex_vertex", "render_e_vertex_fragment",
                  "render_e_vertex_tile", "render_e_vertex_dispatch", "render_e_vertex_blit",
                  "render_e_fragment_fragment", "render_e_tile_tile", "render_e_tile_fragment",
                  "render_e_object_vertex", "render_e_mesh_fragment", "compute_q_fragment_dispatch",
                  "compute_q_dispatch_dispatch", "compute_q_blit_dispatch", "compute_q_dispatch_blit",
                  "compute_q_tile_dispatch", "compute_q_vertex_dispatch", "compute_q_dispatch_fragment",
                  "compute_q_dispatch_vertex", "compute_q_all_all", "compute_p_dispatch_dispatch",
                  "compute_p_dispatch_fragment", "compute_p_dispatch_vertex", "compute_p_blit_dispatch",
                  "compute_e_dispatch_dispatch", "compute_e_blit_dispatch", "compute_e_dispatch_blit",
                  "compute_e_blit_blit", "compute_e_dispatch_fragment", "compute_e_fragment_dispatch",
                  "compute_e_dispatch_vertex"},
                 [](Rig& r, const std::string& v) { return caseLegality(r, v); }, false});
    c.push_back({"cost_queue",
                 {"none", "q_dispatch_dispatch"},
                 [](Rig& r, const std::string& v) { return caseCostChain(r, parseBarrier(v), true); }, true});
    c.push_back({"cost_encoder",
                 {"none", "e_dispatch_dispatch"},
                 [](Rig& r, const std::string& v) { return caseCostChain(r, parseBarrier(v), false); }, true});
    c.push_back({"cost_render",
                 {"none", "q_fragment_vertex", "q_fragment_fragment", "q_fragment_vertex+fragment"},
                 [](Rig& r, const std::string& v) {
                     return caseRenderToRender(r, parseBarrier(v), 200, 512, 8);
                 }, true});
    return c;
}

} // namespace

int main(int argc, char** argv) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--reps=", 0) == 0) g_cfg.reps = u32(std::atoi(a.c_str() + 7));
        else if (a.rfind("--iters=", 0) == 0) g_cfg.iters = u32(std::atoi(a.c_str() + 8));
        else if (a.rfind("--runs=", 0) == 0) g_cfg.runs = u32(std::atoi(a.c_str() + 7));
        else pos.push_back(a);
    }
    const std::vector<CaseDef> cases = makeCases();
    if (!pos.empty() && pos[0] == "--list") {
        for (const CaseDef& c : cases)
            for (const char* v : c.variants) std::printf("%s %s\n", c.name, v);
        return 0;
    }
    if (pos.size() != 2) {
        std::fprintf(stderr, "usage: barrier_spike [--reps=N] [--iters=N] [--runs=N] <case> <variant> | --list\n");
        return 2;
    }
    const CaseDef* def = nullptr;
    for (const CaseDef& c : cases)
        if (pos[0] == c.name) def = &c;
    if (!def) die("unknown case '" + pos[0] + "'");
    if (std::find_if(def->variants.begin(), def->variants.end(),
                     [&](const char* v) { return pos[1] == v; }) == def->variants.end())
        die("unknown variant '" + pos[1] + "' for case " + pos[0]);
    if (def->median && g_cfg.runs == 1) g_cfg.runs = 7;

    Rig rig;
    std::vector<double> times;
    Result total;
    std::string gpuError;
    for (u32 run = 0; run < g_cfg.runs; ++run) {
        Result res = def->run(rig, pos[1]);
        if (!rig.gpuError.empty()) gpuError = rig.gpuError;
        rig.endRun();
        times.push_back(res.gpuMs);
        total.wrongReps += res.wrongReps;
        total.reps += res.reps;
        total.badElems += res.badElems;
        if (res.haveFirst && (!total.haveFirst || (res.wrongReps && total.wrongReps == res.wrongReps))) {
            total.haveFirst = true;
            total.expected = res.expected;
            total.got = res.got;
            total.firstIndex = res.firstIndex;
        }
    }
    std::sort(times.begin(), times.end());
    const double ms = times[times.size() / 2];
    const char* status = !gpuError.empty() ? "ERROR" : (total.wrongReps ? "WRONG" : "ok");
    char exp[32] = "-", got[32] = "-";
    if (total.haveFirst) {
        std::snprintf(exp, sizeof exp, "0x%x", total.expected);
        std::snprintf(got, sizeof got, "0x%x", total.got);
    }
    std::printf("SPIKE %s %s result=%s gpu_ms=%.3f wrong_reps=%u/%u bad_elems=%llu expected=%s got=%s%s%s\n",
                pos[0].c_str(), pos[1].c_str(), status, ms, total.wrongReps, total.reps,
                static_cast<unsigned long long>(total.badElems), exp, got, gpuError.empty() ? "" : " gpu_error=",
                gpuError.c_str());
    std::fflush(stdout);
    pool->release();
    return 0;
}
