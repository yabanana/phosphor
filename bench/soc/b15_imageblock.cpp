// B-15: imageblock (tile memory) capacity per tile size (OPT-0.6).
// Serves S-TBDR-1 of docs/APPLE_SOC_PLAYBOOK.md.
//
// Tile pipelines (MTL4::TileRenderPipelineDescriptor) with an EXPLICIT
// imageblock of 4K bytes per pixel (K = 1 .. 512 words, one generated kernel
// per K, see shaders/b15_imageblock.metal) on render passes with tile size
// 32x32, 32x16, 16x16, 16x8 and 8x8 (MTL4::RenderPassDescriptor::setTileWidth/
// Height).  For every (tile size, K):
//   1. create the tile pipeline (failure = "creation refused", with the error text);
//   2. read the sizes the driver reports (imageblockSampleLength, maxTotalThreadsPerThreadgroup,
//      imageblockMemoryLength for the tile);
//   3. run a 2048x2048 pass (clear, dontCare, no draws, one tile dispatch): every
//      pixel writes its imageblock entry, an imageblock barrier, then reads its right
//      neighbour's entry (inside the tile) and stores the checksum in a device buffer; the
//      CPU recomputes every checksum;
//   4. time it (CommandTimer span minus the empty span).
// max_imageblock_bytes.tile_<w>x<h> = the largest 4K for which creation, execution and
// the CPU check all succeed; the K after it must fail creation or execution or be
// wrong (the negative control: the limit is a real limit) -- if no K fails up to the
// largest tried, the metric is a lower bound and the status is Partial.
//
// A failed execution (GPU error) is caught and recorded; validation layers turn
// some of these into aborts, so the suite's --validate run only exercises K up
// to the first refusal (a refused pipeline is never executed).

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace soc {
namespace {

constexpr u32 kDim = 2048;
const u32 kWords[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 32, 48, 64};

struct Tile {
    u32 w, h;
};
// 16x8 and 8x8 (and every other size tried: 8x16, 16x32, 32x8, 64x*, 16x4) abort in MTL4RenderPassDescriptor validation
// ("Invalid tile dimensions", measured on the M5 Max, macOS 27.2); they cannot be attempted in-process.
const Tile kTiles[] = {{32, 32}, {32, 16}, {16, 16}};

struct TileParams {
    u32 width, zero, pad0, pad1;
};

u32 hash(u32 x, u32 y, u32 j) {
    u32 h = x * 73856093u ^ y * 19349663u ^ (j * 83492791u + 1u);
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return h;
}

std::string errorText(NS::Error* e) {
    return e && e->localizedDescription() ? e->localizedDescription()->utf8String() : "unknown error";
}

struct Rig {
    MTL::Texture* target = nullptr;
    MTL::Buffer* out     = nullptr;
    MTL::Buffer* params  = nullptr;
};

void encodePass(Context& ctx, const Rig& r, MTL4::CommandBuffer* cmd, MTL::RenderPipelineState* pso, const Tile& tile) {
    MTL4::RenderPassDescriptor* pd = MTL4::RenderPassDescriptor::alloc()->init();
    auto* c = pd->colorAttachments()->object(0);
    c->setTexture(r.target);
    c->setLoadAction(MTL::LoadActionClear);
    c->setStoreAction(MTL::StoreActionDontCare);
    c->setClearColor(MTL::ClearColor::Make(0, 0, 0, 0));
    pd->setTileWidth(tile.w);
    pd->setTileHeight(tile.h);
    pd->setImageblockSampleLength(pso->imageblockSampleLength());
    MTL4::RenderCommandEncoder* re = cmd->renderCommandEncoder(pd);
    pd->release();
    re->setRenderPipelineState(pso);
    ctx.table()->setAddress(r.out->gpuAddress(), 0);
    ctx.table()->setAddress(r.params->gpuAddress(), 1);
    re->setArgumentTable(ctx.table(), MTL::RenderStageTile);
    re->dispatchThreadsPerTile(MTL::Size::Make(tile.w, tile.h, 1));
    re->endEncoding();
}

double timeRun(Context& ctx, const Rig& r, MTL::RenderPipelineState* pso, const Tile& tile) {
    CommandTimer t(ctx);
    encodePass(ctx, r, t.begin(), pso, tile);
    return t.finish() - ctx.emptySpanMs();
}

// Wrong checksums after one run (all 2048x2048 pixels).
u32 verify(const Rig& r, u32 words, const Tile& tile) {
    const u32* o = static_cast<const u32*>(r.out->contents());
    u32 wrong = 0;
    for (u32 py = 0; py < kDim; ++py) {
        for (u32 px = 0; px < kDim; ++px) {
            const u32 tx = px / tile.w * tile.w, lx = px - tx;
            const u32 nx = tx + (lx + 1) % tile.w; // right neighbour inside the tile
            u32 sum = 0;
            for (u32 j = 0; j < words; ++j) sum = sum * 31u + hash(nx, py, j);
            if (o[size_t(py) * kDim + px] != sum) ++wrong;
        }
    }
    return wrong;
}


struct Outcome {
    bool created = false, ran = false, correct = false, skipped = false;
    std::string error;
    u32 wrong = 0;
    u64 sampleLength = 0, maxThreads = 0, memLength = 0;
    double ms = 0;
    Stats stats;
};

Outcome tryOne(Context& ctx, const Rig& r, MTL::Library* lib, u32 words, const Tile& tile, u32 reps) {
    Outcome o;
    NS::Error* err = nullptr;
    MTL4::TileRenderPipelineDescriptor* d = MTL4::TileRenderPipelineDescriptor::alloc()->init();
    d->setTileFunctionDescriptor(ctx.function(lib, "b15_tile_" + std::to_string(words)));
    d->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    d->setThreadgroupSizeMatchesTileSize(true);
    MTL::RenderPipelineState* pso = ctx.compiler()->newRenderPipelineState(d, nullptr, &err);
    d->release();
    if (!pso) {
        o.error = errorText(err);
        return o;
    }
    ctx.keep(pso);
    o.created      = true;
    o.sampleLength = pso->imageblockSampleLength();
    o.maxThreads   = pso->maxTotalThreadsPerThreadgroup();
    o.memLength    = pso->imageblockMemoryLength(MTL::Size::Make(tile.w, tile.h, 1));
    const TileParams p{kDim, 0, 0, 0};
    std::memcpy(r.params->contents(), &p, sizeof(p));
    std::memset(r.out->contents(), 0xCD, r.out->length());
    // The validation layer ABORTS on a pass whose tile memory exceeds the device limit ("Total
    // allocated tile memory (40960) cannot be greater than (32768)"): under it (MTL_DEBUG_LAYER)
    // such a size is not executed.
    // Same for "Per sample storage (72) cannot be greater than (64)".
    if (std::getenv("MTL_DEBUG_LAYER") && (o.memLength > ctx.device()->maxThreadgroupMemoryLength() || o.sampleLength > 64)) {
        o.error   = "not executed under the validation layer (tile memory " + std::to_string(o.memLength) + " B / sample " +
                  std::to_string(o.sampleLength) + " B above its limits: the layer aborts)";
        o.skipped = true;
        return o;
    }
    try {
        ctx.keepWarm(15);
        timeRun(ctx, r, pso, tile);
        o.ran   = true;
        o.wrong = verify(r, words, tile);
        o.correct = o.wrong == 0;
        if (o.correct) {
            o.stats = ctx.measure([&] { return timeRun(ctx, r, pso, tile); }, reps);
            o.ms    = o.stats.median;
        }
    } catch (const BenchError& e) {
        o.error = e.what();
    }
    return o;
}

void benchImageblock(Context& ctx, Report& rep) {
    MTL::Library* lib = ctx.library("b15_imageblock.metal");
    Rig r;
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA8Unorm, kDim, kDim, false);
    td->setUsage(MTL::TextureUsageRenderTarget);
    td->setStorageMode(MTL::StorageModePrivate);
    r.target = ctx.texture(td);
    r.out    = ctx.buffer(size_t(kDim) * kDim * 4);
    r.params = ctx.buffer(256);
    const u32 reps = ctx.quick() ? 9 : 21;
    rep.value("device.max_threadgroup_memory", "bytes", double(ctx.device()->maxThreadgroupMemoryLength()));

    bool limitsProven = true, nonMonotonic = false;
    std::string proofText, badText;
    for (const Tile& tile : kTiles) {
        const std::string tname = "tile_" + std::to_string(tile.w) + "x" + std::to_string(tile.h);
        u32 best = 0;          // words of the largest success
        u64 bestSample = 0, bestMem = 0;
        int failedAfter = -1;  // first K that failed
        std::string failWhy;
        std::vector<double> bytesV, msV;
        for (u32 words : kWords) {
            const Outcome o = tryOne(ctx, r, lib, words, tile, reps);
            const u32 bytes = words * 4;
            const std::string tag = tname + ".bytes_" + std::to_string(bytes);
            ctx.log("B-15 %s %4u B/pixel: created %d ran %d correct %d | reported sampleLength %llu maxThreads %llu tileMemory %llu | %.3f ms %s",
                    tname.c_str(), bytes, int(o.created), int(o.ran), int(o.correct), static_cast<unsigned long long>(o.sampleLength),
                    static_cast<unsigned long long>(o.maxThreads), static_cast<unsigned long long>(o.memLength), o.ms,
                    o.error.substr(0, 120).c_str());
            const bool ok = o.created && o.ran && o.correct;
            if (ok && failedAfter >= 0) {
                nonMonotonic = true; // a size above a refused one worked: the "limit" is not a limit
                badText += tag + " worked after " + std::to_string(failedAfter * 4) + " B was refused ";
            }
            if (ok && failedAfter < 0) {
                best = words;
                bestSample = o.sampleLength;
                bestMem = o.memLength;
                rep.metric("tile_pass_ms." + tag, "ms", o.stats,
                           {{"bytes_per_pixel", double(bytes)}, {"tile_w", double(tile.w)}, {"tile_h", double(tile.h)}}, false);
                rep.value("tile_pass_ns_per_pixel." + tag, "ns", o.ms * 1e6 / (double(kDim) * kDim),
                          {{"bytes_per_pixel", double(bytes)}}, false);
                rep.value("reported_sample_length." + tag, "bytes", double(o.sampleLength), {{"bytes_per_pixel", double(bytes)}});
                rep.value("reported_tile_memory." + tag, "bytes", double(o.memLength), {{"bytes_per_pixel", double(bytes)}});
                rep.value("reported_max_threads." + tag, "threads", double(o.maxThreads), {{"bytes_per_pixel", double(bytes)}});
            } else if (failedAfter < 0) {
                failedAfter = int(words);
                failWhy     = o.created ? (o.ran ? (o.wrong == kDim * kDim ? std::string("pipeline created and pass completed without error, but the tile kernel did not run (all pixels still hold the 0xCD sentinel)") : "wrong results (" + std::to_string(o.wrong) + ")") : "execution failed: " + o.error)
                                        : "pipeline creation refused: " + o.error;
                rep.value("first_refused_bytes." + tname, "bytes", double(words * 4), {}, false);
                rep.note(tname + " first refusal at " + std::to_string(words * 4) + " B/pixel: " + failWhy.substr(0, 160));
            }
        }
        if (best == 0) {
            rep.status(Status::Partial, tname + ": no imageblock size worked");
            limitsProven = false;
            continue;
        }
        rep.value("max_sample_length." + tname, "bytes", double(bestSample), {{"tile_w", double(tile.w)}, {"tile_h", double(tile.h)}});
        rep.value("max_tile_memory." + tname, "bytes", double(bestMem), {{"tile_w", double(tile.w)}, {"tile_h", double(tile.h)}});
        rep.value("max_imageblock_bytes." + tname, "bytes", double(best * 4), {{"tile_w", double(tile.w)}, {"tile_h", double(tile.h)}});
        if (failedAfter < 0) {
            limitsProven = false;
            rep.status(Status::Partial, tname + ": no failure up to " + std::to_string(kWords[std::size(kWords) - 1] * 4) +
                                             " B/pixel: max_imageblock_bytes is a lower bound");
            proofText += tname + " no limit up to " + std::to_string(best * 4) + " B ";
        } else {
            proofText += tname + " max " + std::to_string(best * 4) + " B, " + std::to_string(failedAfter * 4) + " B refused (" +
                         failWhy.substr(0, 60) + ") ";
        }
    }
    if (std::getenv("MTL_DEBUG_LAYER"))
        rep.negative(true, "validation-layer run: sizes above the tile-memory limit are not executed (the layer aborts), the limit proof is skipped: " + proofText);
    else
        rep.negative(limitsProven, "the size after the maximum fails: " + proofText);
    rep.negative(!nonMonotonic, !nonMonotonic ? "every size up to the maximum matched the CPU checksums (imageblock round trip through neighbour reads) and no larger size worked again"
                                              : "NOT monotonic: " + badText);
    if (nonMonotonic) rep.status(Status::Failed, "a size above a refused one worked");
    rep.note("K = 4 bytes words per pixel, explicit imageblock<struct, imageblock_layout_explicit>; the sizes are the generated set 4..256 B; "
             "the time of a tile pass includes work proportional to K (write and read K words per pixel)");
}

} // namespace

SOC_BENCH("B-15", "imageblock", "Maximum imageblock (tile memory) bytes per pixel per tile size, tile shader pass time", benchImageblock);

} // namespace soc
