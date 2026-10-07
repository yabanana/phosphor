#include "renderer/rt_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace phosphor {
namespace {

size_t byteCount(u32 width, u32 height) {
    if (!width || !height || u64(width) * height > std::numeric_limits<size_t>::max() / 4)
        throw std::invalid_argument("RT CPU texture dimensions overflow or are empty");
    return size_t(width) * height * 4;
}

const RtCpuMip& mip(const RtCpuTexture& texture, u32 level) {
    if (level >= texture.mips.size()) throw std::invalid_argument("RT CPU texture mip is missing");
    const auto& m = texture.mips[level];
    if (m.rgba8.size() != byteCount(m.width, m.height))
        throw std::invalid_argument("RT CPU texture mip byte count is invalid");
    const auto& base = texture.mips[0];
    if (m.width != std::max(1u, base.width >> std::min(level, 31u)) ||
        m.height != std::max(1u, base.height >> std::min(level, 31u)))
        throw std::invalid_argument("RT CPU texture mip dimensions are inconsistent");
    return m;
}

u32 wrap(i64 i, u32 size) {
    const auto n = i64(size);
    return u32((i % n + n) % n);
}

double bilinear(const RtCpuMip& m, double u, double v) {
    // Normalize before multiplication so very large finite UVs cannot overflow
    // the index conversion. Integer UV offsets are identical with repeat.
    const double x = (u - std::floor(u)) * m.width - 0.5;
    const double y = (v - std::floor(v)) * m.height - 0.5;
    const auto ix = i64(std::floor(x)), iy = i64(std::floor(y));
    const double fx = x - std::floor(x), fy = y - std::floor(y);
    auto alpha = [&](i64 px, i64 py) {
        return double(m.rgba8[(size_t(wrap(py, m.height)) * m.width + wrap(px, m.width)) * 4 + 3]) / 255.0;
    };
    return std::lerp(std::lerp(alpha(ix, iy), alpha(ix + 1, iy), fx),
                     std::lerp(alpha(ix, iy + 1), alpha(ix + 1, iy + 1), fx), fy);
}

// Round a normalized alpha float to IEEE binary16, ties to even, without
// depending on host _Float16 support or the floating-point rounding mode.
float halfAlpha(double alpha) {
    const float value = float(alpha);
    if (value == 0) return 0;
    int exponent = 0;
    std::frexp(value, &exponent);
    const double step = std::ldexp(1.0, std::max(-24, exponent - 11));
    const double scaled = double(value) / step;
    double rounded = std::floor(scaled);
    const double fraction = scaled - rounded;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(rounded, 2.0) != 0.0)) rounded += 1;
    return float(rounded * step);
}

using V3 = std::array<double, 3>;
V3 worldPoint(const GPUInstance& i, const GPUVertex& v) {
    const auto* m = i.modelMatrix;
    return {double(m[0]) * v.px + double(m[4]) * v.py + double(m[8]) * v.pz + m[12],
            double(m[1]) * v.px + double(m[5]) * v.py + double(m[9]) * v.pz + m[13],
            double(m[2]) * v.px + double(m[6]) * v.py + double(m[10]) * v.pz + m[14]};
}

double coneLod(const GPURtRay& ray, const RtReferenceHit& hit, const GPUInstance& instance,
               const std::array<const GPUVertex*, 3>& vertices, const RtCpuMip& texture) {
    if (ray.type != RT_PROBE_PRIMARY || !(ray.coneWidth > 0)) return 0;
    const auto a = worldPoint(instance, *vertices[0]), b = worldPoint(instance, *vertices[1]),
               c = worldPoint(instance, *vertices[2]);
    V3 ab{}, ac{};
    for (u32 k = 0; k < 3; ++k) { ab[k] = b[k] - a[k]; ac[k] = c[k] - a[k]; }
    const V3 cross{ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
    const double area = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    const auto &v0 = *vertices[0], &v1 = *vertices[1], &v2 = *vertices[2];
    const double uvArea = std::abs((double(v1.u) - v0.u) * (double(v2.v) - v0.v) -
                                   (double(v1.v) - v0.v) * (double(v2.u) - v0.u));
    if (!(area > 0 && uvArea > 0)) return 0;
    const double texelWorld = std::sqrt(area / (uvArea * double(texture.width) * texture.height));
    return std::max(0.0, std::log2(std::max(hit.t * ray.coneWidth, 1e-12) / std::max(texelWorld, 1e-20)));
}

bool skippedSecondary(const GPURtRay& ray, const GPURtHit& hit) {
    const bool secondary = ray.type == RT_PROBE_SHADOW || ray.type == RT_PROBE_AO || ray.type == RT_PROBE_DIFFUSE;
    const auto mask = ray.type == RT_PROBE_SHADOW ? RT_MASK_SHADOW : RT_MASK_INDIRECT;
    // Exact generator/rtMiss(-2) sentinel only. A malformed active ray or
    // corrupted hit must still go through the failing reference check.
    return secondary && ray.tmax == -1 && ray.tmin == 0 && ray.mask == mask &&
        ray.ox == 0 && ray.oy == 0 && ray.oz == 0 && ray.dx == 0 && ray.dy == 0 && ray.dz == 0 &&
        ray.coneWidth == 0 && ray.pad == 0 && hit.t == -2 && hit.hit == 0 && hit.u == 0 && hit.v == 0 &&
        hit.slot == ~0u && hit.primitive == ~0u && hit.generation == 0 && hit.frontFacing == 0;
}

enum class AlphaDecision { Reject, Accept, Ambiguous };

AlphaDecision alphaTest(const GPURtRay& ray, const RtReferenceHit& hit,
                        std::span<const GPUVertex> vertices, std::span<const u32> indices,
                        std::span<const GPURtMesh> meshes, std::span<const GPUInstance> instances,
                        std::span<const GPUMaterial> materials, std::span<const RtCpuTexture> textures) {
    if (hit.slot >= instances.size() || hit.material >= materials.size() || hit.mesh >= meshes.size())
        throw std::invalid_argument("RT alpha reference hit has invalid scene indices");
    const auto& material = materials[hit.material];
    if (!std::isfinite(material.alphaCutoff) || !std::isfinite(material.baseColor[3]))
        throw std::invalid_argument("RT alpha material is nonfinite");
    if (material.alphaCutoff <= 0) return AlphaDecision::Accept;
    if (material.baseColorTex == INVALID_TEXTURE_INDEX)
        return material.baseColor[3] >= material.alphaCutoff ? AlphaDecision::Accept : AlphaDecision::Reject;
    if (material.baseColorTex >= textures.size()) throw std::invalid_argument("RT CPU alpha texture is unavailable");
    const auto& texture = textures[material.baseColorTex];
    const auto& base = mip(texture, 0);
    const auto& mesh = meshes[hit.mesh];
    if (hit.primitive >= mesh.indexCount / 3) throw std::invalid_argument("RT alpha primitive is invalid");
    const u64 start = u64(mesh.indexOffset) + u64(hit.primitive) * 3;
    if (start + 3 > indices.size()) throw std::invalid_argument("RT alpha index range is invalid");
    std::array<const GPUVertex*, 3> tri;
    for (u32 k = 0; k < 3; ++k) {
        const u64 index = u64(mesh.vertexOffset) + indices[size_t(start + k)];
        if (index >= vertices.size()) throw std::invalid_argument("RT alpha vertex range is invalid");
        tri[k] = &vertices[size_t(index)];
    }
    if (!std::isfinite(ray.coneWidth) || ray.coneWidth < 0)
        throw std::invalid_argument("RT alpha cone width is invalid");
    const auto lod = coneLod(ray, hit, instances[hit.slot], tri, base);
    if (lod > 0 && (base.width > 1 || base.height > 1) && !texture.exactMips)
        throw std::invalid_argument("RT cone alpha requires GPU mip readback; CPU box mips are approximate");
    const float alpha = material.baseColor[3] * halfAlpha(rtSampleAlpha(texture, hit.texU, hit.texV, lod));
    // Sampler interpolation precision and float-vs-double barycentrics differ.
    // A band is never itself permission to ignore a bad geometric hit.
    if (std::abs(double(alpha) - material.alphaCutoff) <= 2.0 / 255.0) return AlphaDecision::Ambiguous;
    return alpha >= material.alphaCutoff ? AlphaDecision::Accept : AlphaDecision::Reject;
}

} // namespace

RtCpuTexture rtMakeCpuTexture(std::span<const u8> rgba, u32 width, u32 height, bool boxMips) {
    if (rgba.size() != byteCount(width, height)) throw std::invalid_argument("RT CPU RGBA texture size mismatch");
    RtCpuTexture texture;
    texture.mips.push_back({width, height, std::vector<u8>(rgba.begin(), rgba.end())});
    texture.exactMips = width == 1 && height == 1;
    while (boxMips && (width > 1 || height > 1)) {
        const auto& src = texture.mips.back();
        const u32 w = std::max(1u, width / 2), h = std::max(1u, height / 2);
        RtCpuMip dst{w, h, std::vector<u8>(byteCount(w, h))};
        for (u32 y = 0; y < h; ++y) for (u32 x = 0; x < w; ++x) {
            for (u32 channel = 0; channel < 4; ++channel) {
                u32 sum = 0;
                for (u32 dy = 0; dy < 2; ++dy) for (u32 dx = 0; dx < 2; ++dx) {
                    const auto sx = std::min(width - 1, 2 * x + dx), sy = std::min(height - 1, 2 * y + dy);
                    sum += src.rgba8[(size_t(sy) * width + sx) * 4 + channel];
                }
                dst.rgba8[(size_t(y) * w + x) * 4 + channel] = u8((sum + 2) / 4);
            }
        }
        texture.mips.push_back(std::move(dst));
        width = w; height = h;
    }
    return texture;
}

double rtSampleAlpha(const RtCpuTexture& texture, double u, double v, double lod) {
    if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(lod))
        throw std::invalid_argument("RT CPU alpha sampler argument is nonfinite");
    const auto& base = mip(texture, 0);
    u32 last = 0;
    for (u32 d = std::max(base.width, base.height); d > 1; d >>= 1) ++last;
    const double clamped = std::clamp(lod, 0.0, double(last));
    const auto lo = u32(std::floor(clamped)), hi = u32(std::ceil(clamped));
    return std::lerp(bilinear(mip(texture, lo), u, v), bilinear(mip(texture, hi), u, v), clamped - lo);
}

void RtChecker::setGeometry(std::span<const GPUVertex> vertices, std::span<const u32> indices,
                            std::span<const GPURtMesh> meshes) {
    reference_.setGeometry(vertices, indices, meshes);
    vertices_.assign(vertices.begin(), vertices.end());
    indices_.assign(indices.begin(), indices.end());
    meshes_.assign(meshes.begin(), meshes.end());
}

RtCheckResult RtChecker::check(std::span<const GPURtRay> rays, std::span<const GPURtHit> hits,
                              std::span<const GPUInstance> instances, std::span<const GPUMaterial> materials,
                              std::span<const RtCpuTexture> textures, std::span<const u32> masks) {
    RtCheckResult result;
    auto fail = [&](std::string error) {
        ++result.failures;
        if (result.firstError.empty()) result.firstError = std::move(error);
    };
    if (meshes_.empty()) {
        fail("RT check requires configured geometry");
        return result;
    }
    if (rays.empty() || rays.size() != hits.size()) {
        fail("RT check requires nonempty matching ray/hit arrays");
        return result;
    }
    try {
        reference_.setMaterials(materials);
        reference_.setInstances(instances, masks);
    } catch (const std::exception& e) { fail(e.what()); return result; }
    for (size_t i = 0; i < rays.size(); ++i) {
        if (skippedSecondary(rays[i], hits[i])) { ++result.skipped; continue; }
        bool uncertain = false;
        const auto filter = [&](bool acceptAmbiguous) -> RtReference::Filter {
            return [&, acceptAmbiguous](const GPURtRay& ray, const RtReferenceHit& hit) {
                const auto decision = alphaTest(ray, hit, vertices_, indices_, meshes_, instances, materials, textures);
                if (decision == AlphaDecision::Ambiguous) uncertain = true;
                return decision == AlphaDecision::Accept || (acceptAmbiguous && decision == AlphaDecision::Ambiguous);
            };
        };
        try {
            const auto strict = reference_.check(rays[i], hits[i], filter(false));
            const auto loose = uncertain ? reference_.check(rays[i], hits[i], filter(true)) : strict;
            if (strict.ok && loose.ok) {
                ++result.checked;
                result.edgeTies += strict.edgeTie || loose.edgeTie;
            } else if (uncertain && (strict.ok || loose.ok)) {
                ++result.ambiguous; // At least one complete named-hit check passed.
            } else {
                ++result.checked;
                fail("ray " + std::to_string(i) + ": " + strict.error);
            }
        } catch (const std::invalid_argument& e) {
            ++result.unsupported;
            fail("ray " + std::to_string(i) + ": " + e.what());
        }
    }
    if (result.checked == 0 && result.failures == 0) fail("RT check had no unambiguous supported rays");
    return result;
}

} // namespace phosphor
