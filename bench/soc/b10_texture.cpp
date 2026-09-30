// B-10: texture sampling throughput per format -- RGBA8Unorm, RGBA16Float,
// RG11B10Float, RGB9E5, BC7, ASTC 4x4 LDR -- coherent vs sparse access, large
// (8192^2; 4096^2 with --quick) vs small (256^2), point vs bilinear.
//
// Serves S-TEX-1..4 of docs/APPLE_SOC_PLAYBOOK.md.
//
// Content: a 512x512 texel tile, replicated over the texture by blits into a
// private texture (uncompressed formats: random finite values; BC7/ASTC: a
// procedural image encoded by AppleTextureEncoder, loaded at run time with
// dlopen("/usr/lib/libate.dylib") so no link change is needed; without it
// random blocks are used and the compressed results are not checked; random
// ASTC blocks may be error blocks).  Texel (x, y) = tile[y % 512][x % 512],
// so the CPU replays every thread's coordinates and compares the sums.
// S-TEX-4 (sparse residency: mapped/unmapped pages, mapping cost) is NOT
// covered: "sparse" here means scattered access, not sparse textures.

#include "harness.h"

#include <AppleTextureEncoder.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace soc {
namespace {

constexpr u32 kTile = 512;
constexpr u32 kGrid = 1024; // threads per side

struct SampleParams {
    u32   mask, samples;
    float inv;
    u32   seed;
};
static_assert(sizeof(SampleParams) == 16);

Stats toRate(const Stats& s, double scale) {
    Stats r = s;
    r.median = scale / s.median;
    r.min    = scale / s.max;
    r.max    = scale / s.min;
    r.p10    = scale / s.p90;
    r.p90    = scale / s.p10;
    r.mean   = scale / s.mean;
    return r;
}

u32 hash32(u32 x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// --- AppleTextureEncoder via dlopen ------------------------------------------------
struct Ate {
    decltype(&at_encoder_create) create = nullptr;
    decltype(&at_encoder_compress_texels) compress = nullptr;
    decltype(&at_encoder_decompress_texels) decompress = nullptr;
    bool ok() const { return create && compress && decompress; }
};

Ate loadAte() {
    Ate a;
    void* h = dlopen("/usr/lib/libate.dylib", RTLD_NOW);
    if (!h) return a;
    a.create     = reinterpret_cast<decltype(a.create)>(dlsym(h, "at_encoder_create"));
    a.compress   = reinterpret_cast<decltype(a.compress)>(dlsym(h, "at_encoder_compress_texels"));
    a.decompress = reinterpret_cast<decltype(a.decompress)>(dlsym(h, "at_encoder_decompress_texels"));
    return a;
}

// --- Formats ---------------------------------------------------------------------
enum class Kind { Rgba8, Rgba16f, Rg11b10, Rgb9e5, Bc7, Astc };

struct Format {
    const char* name;
    Kind kind;
    MTL::PixelFormat pf;
    bool compressed;
    u32 bytesPerTexel; // uncompressed
    std::vector<u8> tile;         // GPU bytes of the 512x512 tile
    std::vector<float> decoded;   // 512*512*4 floats the CPU expects (empty: not checkable)
    u32 rowBytes = 0, imageBytes = 0;
};

float f11(u32 v) {
    const u32 e = (v >> 6) & 31, m = v & 63;
    return e == 0 ? std::ldexp(float(m) / 64.0f, -14) : std::ldexp(1.0f + float(m) / 64.0f, int(e) - 15);
}
float f10(u32 v) {
    const u32 e = (v >> 5) & 31, m = v & 31;
    return e == 0 ? std::ldexp(float(m) / 32.0f, -14) : std::ldexp(1.0f + float(m) / 32.0f, int(e) - 15);
}

void fillUncompressed(Format& f, u64 seed) {
    const size_t n = size_t(kTile) * kTile;
    f.rowBytes   = kTile * f.bytesPerTexel;
    f.imageBytes = f.rowBytes * kTile;
    f.tile.assign(n * f.bytesPerTexel, 0);
    f.decoded.assign(n * 4, 1.0f);
    u64 s = seed;
    for (size_t i = 0; i < n; ++i) {
        float* d = &f.decoded[i * 4];
        u8* p = &f.tile[i * f.bytesPerTexel];
        if (f.kind == Kind::Rgba8) {
            const u32 r = u32(xorshift64(s));
            std::memcpy(p, &r, 4);
            for (int c = 0; c < 4; ++c) d[c] = float((r >> (8 * c)) & 255u) / 255.0f;
        } else if (f.kind == Kind::Rgba16f) {
            _Float16 h[4];
            for (int c = 0; c < 4; ++c) {
                h[c] = _Float16(float(xorshift64(s) >> 40) / float(1 << 24)); // [0,1)
                d[c] = float(h[c]);
            }
            std::memcpy(p, h, 8);
        } else if (f.kind == Kind::Rg11b10) {
            const u32 r = u32(xorshift64(s)), g = u32(xorshift64(s)), b = u32(xorshift64(s));
            const u32 r11 = (8 + ((r >> 8) % 15)) << 6 | (r & 63); // exponent 8..22: finite, no NaN/Inf
            const u32 g11 = (8 + ((g >> 8) % 15)) << 6 | (g & 63);
            const u32 b10 = (8 + ((b >> 8) % 15)) << 5 | (b & 31);
            const u32 w = r11 | (g11 << 11) | (b10 << 22);
            std::memcpy(p, &w, 4);
            d[0] = f11(r11); d[1] = f11(g11); d[2] = f10(b10);
        } else { // Rgb9e5
            const u32 r = u32(xorshift64(s));
            const u32 mr = r & 511, mg = (r >> 9) & 511, mb = (r >> 18) & 511, e = 10 + ((r >> 27) % 11);
            const u32 w = mr | (mg << 9) | (mb << 18) | (e << 27);
            std::memcpy(p, &w, 4);
            const float sc = std::ldexp(1.0f / 512.0f, int(e) - 15);
            d[0] = float(mr) * sc; d[1] = float(mg) * sc; d[2] = float(mb) * sc;
        }
    }
}

// Procedural RGBA8 image (structure + noise) for the block encoders.
std::vector<u32> proceduralImage() {
    std::vector<u32> img(size_t(kTile) * kTile);
    u64 s = 99;
    for (u32 y = 0; y < kTile; ++y)
        for (u32 x = 0; x < kTile; ++x) {
            const float fx = float(x), fy = float(y);
            const float v[3] = {0.5f + 0.4f * std::sin(fx * 0.05f) * std::cos(fy * 0.031f),
                                0.5f + 0.4f * std::sin((fx + fy) * 0.023f),
                                0.5f + 0.4f * std::cos(fx * 0.017f - fy * 0.041f)};
            u32 px = 0xff000000u;
            for (int c = 0; c < 3; ++c) {
                const float n = float(xorshift64(s) >> 40) / float(1 << 24) - 0.5f;
                const int q = int(std::clamp(v[c] + 0.08f * n, 0.0f, 1.0f) * 255.0f + 0.5f);
                px |= u32(q) << (8 * c);
            }
            img[size_t(y) * kTile + x] = px;
        }
    return img;
}

// Returns true when the tile was made by the real encoder (and decoded for the CPU check).
bool fillCompressed(Format& f, const Ate& ate, u64 seed) {
    const u32 bw = kTile / 4;
    f.rowBytes   = bw * 16;
    f.imageBytes = f.rowBytes * bw;
    f.tile.assign(size_t(f.imageBytes), 0);
    if (ate.ok()) {
        const at_block_format_t bf = f.kind == Kind::Bc7 ? at_block_format_bc7 : at_block_format_astc_4x4_ldr;
        std::vector<u32> img = proceduralImage();
        at_encoder_t enc = ate.create(at_texel_format_rgba8_unorm, at_alpha_opaque, bf, at_alpha_opaque, nullptr);
        at_encoder_t dec = ate.create(at_texel_format_rgba16_float, at_alpha_opaque, bf, at_alpha_opaque, nullptr);
        if (enc && dec) {
            at_texel_region_t src{img.data(), {kTile, kTile, 1}, kTile * 4, size_t(kTile) * kTile * 4};
            at_block_buffer_t blocks{f.tile.data(), f.rowBytes, f.imageBytes};
            const float mse = ate.compress(enc, &src, &blocks, 0.0005f, at_flags_default);
            std::vector<_Float16> px(size_t(kTile) * kTile * 4);
            at_texel_region_t out{px.data(), {kTile, kTile, 1}, kTile * 8, size_t(kTile) * kTile * 8};
            if (mse >= 0 && ate.decompress(dec, &blocks, &out, at_flags_default) == at_error_success) {
                f.decoded.resize(px.size());
                for (size_t i = 0; i < px.size(); ++i) f.decoded[i] = float(px[i]);
                os_release(enc);
                os_release(dec);
                return true;
            }
        }
        if (enc) os_release(enc);
        if (dec) os_release(dec);
    }
    // Fallback: random blocks (BC7 mode bits and ASTC block modes may be invalid).
    u64 s = seed;
    for (size_t i = 0; i < f.tile.size(); i += 8) {
        const u64 r = xorshift64(s);
        std::memcpy(&f.tile[i], &r, 8);
    }
    return false;
}

// --- CPU replay -------------------------------------------------------------------
void coords(bool sparse, u32 gx, u32 gy, u32 s, u32 mask, u32 seed, u32& x, u32& y) {
    if (sparse) {
        const u32 h1 = hash32(gx + gy * 2048u + s * 0x9E3779B1u + seed);
        const u32 h2 = hash32(h1 ^ 0x68E31DA4u);
        x = h1 & mask;
        y = h2 & mask;
    } else {
        x = (gx + (s & 7u) * 1024u) & mask;
        y = (gy + (s >> 3) * 1024u) & mask;
    }
}

void texel(const Format& f, u32 x, u32 y, float* out) {
    const size_t i = (size_t(y % kTile) * kTile + (x % kTile)) * 4;
    for (int c = 0; c < 4; ++c) out[c] = f.decoded[i + c];
}

double replay(const Format& f, bool sparse, bool bilinear, u32 gx, u32 gy, u32 samples, u32 size, u32 seed) {
    const u32 mask = size - 1;
    double acc[4] = {0, 0, 0, 0};
    for (u32 s = 0; s < samples; ++s) {
        u32 x, y;
        coords(sparse, gx, gy, s, mask, seed, x, y);
        float t[4];
        if (!bilinear) {
            texel(f, x, y, t);
            for (int c = 0; c < 4; ++c) acc[c] += t[c];
        } else {
            const u32 x1 = std::min(x + 1, mask), y1 = std::min(y + 1, mask);
            float a[4], b[4], c[4], d[4];
            texel(f, x, y, a); texel(f, x1, y, b); texel(f, x, y1, c); texel(f, x1, y1, d);
            for (int k = 0; k < 4; ++k)
                acc[k] += 0.5625 * a[k] + 0.1875 * b[k] + 0.1875 * c[k] + 0.0625 * d[k];
        }
    }
    return acc[0] + 2 * acc[1] + 3 * acc[2] + 5 * acc[3];
}

struct Case {
    const Format* fmt;
    MTL::Texture* tex;
    u32 size;
    bool sparse, bilinear;
    MTL::ComputePipelineState* pso;
};

void benchTexture(Context& ctx, Report& rep) {
    const bool quick = ctx.quick();
    const u32 large = quick ? 4096 : 8192, small = 256;
    MTL::Library* lib = ctx.library("b10_texture.metal", /*fastMath=*/true);
    MTL::Buffer* params = ctx.buffer(256);
    MTL::Buffer* out    = ctx.buffer(size_t(kGrid) * kGrid * 4);
    const float* out32 = static_cast<const float*>(out->contents());
    const Ate ate = loadAte();

    // Pipelines: (sparse, bilinear) function constants.
    MTL::ComputePipelineState* pso[2][2];
    for (int sp = 0; sp < 2; ++sp)
        for (int bl = 0; bl < 2; ++bl) {
            MTL::FunctionConstantValues* fc = MTL::FunctionConstantValues::alloc()->init();
            const bool a = sp, b = bl;
            fc->setConstantValue(&a, MTL::DataTypeBool, NS::UInteger(0));
            fc->setConstantValue(&b, MTL::DataTypeBool, NS::UInteger(1));
            ctx.keep(fc);
            pso[sp][bl] = ctx.compute(lib, "b10_sample", fc);
        }

    std::vector<Format> fmts;
    fmts.push_back({"rgba8", Kind::Rgba8, MTL::PixelFormatRGBA8Unorm, false, 4, {}, {}});
    fmts.push_back({"rgba16f", Kind::Rgba16f, MTL::PixelFormatRGBA16Float, false, 8, {}, {}});
    fmts.push_back({"rg11b10f", Kind::Rg11b10, MTL::PixelFormatRG11B10Float, false, 4, {}, {}});
    fmts.push_back({"rgb9e5", Kind::Rgb9e5, MTL::PixelFormatRGB9E5Float, false, 4, {}, {}});
    fmts.push_back({"bc7", Kind::Bc7, MTL::PixelFormatBC7_RGBAUnorm, true, 16, {}, {}});
    fmts.push_back({"astc4x4", Kind::Astc, MTL::PixelFormatASTC_4x4_LDR, true, 16, {}, {}});
    bool encoderUsed = ate.ok(), allCompressedChecked = true;
    u64 seed = 4242;
    for (Format& f : fmts) {
        if (f.compressed) {
            if (!fillCompressed(f, ate, seed++)) allCompressedChecked = false;
        } else {
            fillUncompressed(f, seed++);
        }
    }

    bool resultsOk = true;
    std::string wrong;
    std::vector<double> ratios2x;
    double worst2x = 2, best2x = 2;
    std::map<std::string, double> rate; // key -> Gtexel/s (point)
    const double targetMs = quick ? 0.25 : 0.4;
    u32 nChecked = 0;

    for (Format& f : fmts) {
        // Texture(s) for this format: staging buffer, tile replicated by blits.
        MTL::Buffer* stage = ctx.buffer(f.tile.size());
        std::memcpy(stage->contents(), f.tile.data(), f.tile.size());
        for (u32 size : {large, small}) {
            MTL::TextureDescriptor* d = MTL::TextureDescriptor::texture2DDescriptor(f.pf, size, size, false);
            d->setUsage(MTL::TextureUsageShaderRead);
            d->setStorageMode(MTL::StorageModePrivate);
            MTL::Texture* tex = ctx.texture(d);
            {
                MTL4::CommandBuffer* cmd = ctx.beginCommands();
                MTL4::ComputeCommandEncoder* e = cmd->computeCommandEncoder();
                const u32 t = std::min(size, kTile);
                for (u32 y = 0; y < size; y += kTile)
                    for (u32 x = 0; x < size; x += kTile)
                        e->copyFromBuffer(stage, 0, f.rowBytes, f.imageBytes, MTL::Size::Make(t, t, 1), tex, 0, 0,
                                          MTL::Origin::Make(x, y, 0));
                e->endEncoding();
                ctx.submit();
            }
            const bool checkable = !f.decoded.empty();
            for (int sp = 0; sp < 2; ++sp)
                for (int bl = 0; bl < 2; ++bl) {
                    MTL::ComputePipelineState* ps = pso[sp][bl];
                    const u32 seed = 0x1234u;
                    auto once = [&](u32 samples) {
                        SampleParams p{size - 1, samples, 1.0f / float(size), seed};
                        std::memcpy(params->contents(), &p, sizeof(p));
                        ComputeTimer t(ctx);
                        MTL4::ComputeCommandEncoder* e = t.begin();
                        ctx.table()->setAddress(params->gpuAddress(), 0);
                        ctx.table()->setAddress(out->gpuAddress(), 1);
                        ctx.table()->setTexture(tex->gpuResourceID(), 0);
                        e->setComputePipelineState(ps);
                        e->setArgumentTable(ctx.table());
                        e->dispatchThreads(MTL::Size::Make(kGrid, kGrid, 1), MTL::Size::Make(16, 16, 1));
                        t.lap();
                        return t.finish()[0];
                    };
                    const u32 probe = 8;
                    const double tp = std::max(1e-4, ctx.measure([&] { return once(probe); }, 9).min); // min: contention only adds time
                    const u32 S = std::clamp<u32>(u32(targetMs / (tp / probe)), 4, 400);
                    const Stats s1 = ctx.measure([&] { return once(S); });
                    if (checkable) {
                        ++nChecked;
                        const u32 pts[8][2] = {{0, 0}, {1, 0}, {0, 1}, {17, 5}, {1023, 1023}, {512, 300}, {1000, 7}, {3, 900}};
                        for (auto& pt : pts) {
                            const double e = replay(f, sp, bl, pt[0], pt[1], S, size, seed);
                            const double g = out32[size_t(pt[1]) * kGrid + pt[0]];
                            if (!(std::fabs(g - e) <= (f.compressed ? 2e-2 : 2e-3) * std::max(1.0, std::fabs(e)))) {
                                resultsOk = false;
                                wrong += std::string(f.name) + "." + std::to_string(size) + (sp ? ".sparse" : ".coherent") + (bl ? ".bilinear" : ".point") + " ";
                                break;
                            }
                        }
                    }
                    ctx.keepWarm(10);
                    const Stats s2 = ctx.measure([&] { return once(2 * S); });
                    const double r = s2.min / s1.min; // minima: contention only adds time
                    ratios2x.push_back(r);
                    worst2x = std::max(worst2x, r);
                    best2x  = std::min(best2x, r);
                    const double samples = double(kGrid) * kGrid * S;
                    const Stats gt = toRate(s1, samples * 1e-6);
                    const std::string name = std::string("sample.") + f.name + (sp ? ".sparse." : ".coherent.") +
                                             std::to_string(size) + (bl ? ".bilinear" : ".point");
                    rep.metric(name, "Gtexel/s", gt, {{"size", double(size)}, {"sparse", double(sp)}, {"bilinear", double(bl)},
                                                      {"samples_per_thread", double(S)}, {"ms", s1.median}});
                    if (!bl) rate[std::string(f.name) + (sp ? ".sparse." : ".coherent.") + std::to_string(size)] = gt.median;
                    ctx.keepWarm(10);
                }
        }
    }

    // Sparse penalty (point) per format and size.
    u32 slowerLarge = 0;
    std::string pen;
    for (const Format& f : fmts) {
        for (u32 size : {large, small}) {
            const double c = rate[std::string(f.name) + ".coherent." + std::to_string(size)];
            const double s = rate[std::string(f.name) + ".sparse." + std::to_string(size)];
            rep.value(std::string("sparse_over_coherent.") + f.name + "." + std::to_string(size) + ".point", "ratio", s / c,
                      {{"size", double(size)}});
            if (size == large) {
                if (s < 0.9 * c) ++slowerLarge;
                pen += std::string(f.name) + "=" + std::to_string(s / c).substr(0, 4) + " ";
            }
        }
    }

    std::vector<double> sorted = ratios2x;
    std::sort(sorted.begin(), sorted.end());
    const double medianRatio = sorted[sorted.size() / 2];
    size_t outside = 0;
    for (double r : sorted) outside += (r < 1.5 || r > 2.6);
    rep.negative(medianRatio > 1.85 && medianRatio < 2.15 && outside * 4 <= sorted.size(),
                 "2x samples -> time x" + std::to_string(medianRatio).substr(0, 5) + " median (min " + std::to_string(best2x).substr(0, 4) +
                     ", max " + std::to_string(worst2x).substr(0, 4) + "; " + std::to_string(outside) + " outside 1.5..2.6, allowed 25%: contention outliers) over " + std::to_string(sorted.size()) + " cases (ratios of minima)");
    rep.negative(slowerLarge >= 5, "sparse < 0.9 x coherent at " + std::to_string(large) + "^2 (point) for " + std::to_string(slowerLarge) + "/6 formats (need >= 5): " + pen);
    rep.negative(resultsOk && nChecked > 0,
                 std::string(resultsOk ? "per-thread sums match the CPU replay" : "MISMATCH: " + wrong) + " (" + std::to_string(nChecked) + " cases x 8 threads" +
                     (allCompressedChecked ? ", BC7/ASTC checked against AppleTextureEncoder decode" : ", compressed formats NOT checked") + ")");
    rep.note("tile 512^2 replicated by blits into private textures (" + std::to_string(large) + "^2 and 256^2); uncompressed content random finite values");
    if (encoderUsed && allCompressedChecked) {
        rep.note("BC7/ASTC 4x4 encoded from a procedural image with AppleTextureEncoder (dlopen /usr/lib/libate.dylib, no CMake change); content tiled 512^2");
    } else {
        rep.status(Status::Partial, "AppleTextureEncoder unavailable: BC7/ASTC use random blocks (ASTC error blocks possible) and are not checked");
    }
    rep.note("'sparse' = hashed coordinates; S-TEX-4 sparse residency (page mapping) is not measured");
    rep.note("metric names: sample.<fmt>.<coherent|sparse>.<size>.<point|bilinear>, Gtexel/s; bilinear samples at +0.75 texel (weights 0.75/0.25)");
}

} // namespace

SOC_BENCH("B-10", "texture.sampling", "Texture sampling per format: coherent vs sparse, large vs small, point vs bilinear", benchTexture);

} // namespace soc
