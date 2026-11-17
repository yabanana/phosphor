#include "scene/terrain_textures.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace phosphor::terrain {
namespace {

constexpr std::array<LayerSpec, kLayerCount> kLayers{{
    {"terrain_grass", "Grass004", 16.0f, 1.0f, 0.03f, 1.0f},
    {"terrain_rock", "Rock030", 8.0f, 1.0f, 0.06f, 1.0f},
    {"terrain_ground", "Ground037", 16.0f, 1.0f, 0.04f, 1.0f},
    {"terrain_snow", "Snow006", 8.0f, 1.0f, 0.02f, 1.0f},
}};

// --- deterministic value noise -------------------------------------------------

u32 hash3(u32 x, u32 y, u32 salt) {
    u32 h = x * 0x9E3779B1u ^ (y + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (salt + kSeed) * 0xC2B2AE35u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    h *= 0x297A2D39u;
    h ^= h >> 15;
    return h;
}
float lattice(i32 x, i32 y, u32 period, u32 salt) {
    const u32 px = static_cast<u32>(((x % static_cast<i32>(period)) + static_cast<i32>(period)) % static_cast<i32>(period));
    const u32 py = static_cast<u32>(((y % static_cast<i32>(period)) + static_cast<i32>(period)) % static_cast<i32>(period));
    return static_cast<float>(hash3(px, py, salt) >> 8) * (1.0f / 16777216.0f);
}
float smooth(float t) { return t * t * (3.0f - 2.0f * t); }
// Periodic value noise in [0,1]: `period` lattice cells over uv in [0,1] (so the result tiles).
float noise(float u, float v, u32 period, u32 salt) {
    const float x = u * static_cast<float>(period), y = v * static_cast<float>(period);
    const float fx = std::floor(x), fy = std::floor(y);
    const i32 ix = static_cast<i32>(fx), iy = static_cast<i32>(fy);
    const float tx = smooth(x - fx), ty = smooth(y - fy);
    const float a = lattice(ix, iy, period, salt), b = lattice(ix + 1, iy, period, salt);
    const float c = lattice(ix, iy + 1, period, salt), d = lattice(ix + 1, iy + 1, period, salt);
    return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty;
}
float fbm(float u, float v, u32 basePeriod, u32 octaves, u32 salt) {
    float sum = 0.0f, amplitude = 0.5f, norm = 0.0f;
    u32 period = basePeriod;
    for (u32 o = 0; o < octaves; ++o) {
        sum += amplitude * noise(u, v, period, salt + o);
        norm += amplitude;
        amplitude *= 0.5f;
        period *= 2;
    }
    return sum / norm;
}
float saturate(float x) { return std::clamp(x, 0.0f, 1.0f); }
float smoothstep(float a, float b, float x) { return smooth(saturate((x - a) / (b - a))); }
u8 toByte(float x) { return static_cast<u8>(std::lround(saturate(x) * 255.0f)); }
float lerp(float a, float b, float t) { return a + (b - a) * t; }

Image makeImage(u32 size) {
    Image image;
    image.width = image.height = size;
    image.rgba.assign(static_cast<size_t>(size) * size * 4, 255);
    return image;
}
void setPixel(Image& image, u32 x, u32 y, float r, float g, float b, float a = 1.0f) {
    u8* p = &image.rgba[(static_cast<size_t>(y) * image.width + x) * 4];
    p[0] = toByte(r);
    p[1] = toByte(g);
    p[2] = toByte(b);
    p[3] = toByte(a);
}

// Slope of the heightfield (rise over run, metres per metre) from central differences in uv space.
float slopeAt(float u, float v) {
    const float e = 1.0f / 512.0f;
    const float dx = (heightAt(u + e, v) - heightAt(u - e, v)) / (2.0f * e * kPatchSize);
    const float dz = (heightAt(u, v + e) - heightAt(u, v - e)) / (2.0f * e * kPatchSize);
    return std::sqrt(dx * dx + dz * dz);
}

} // namespace

const LayerSpec& layerSpec(u32 layer) {
    if (layer >= kLayerCount) throw std::out_of_range("terrain layer");
    return kLayers[layer];
}

std::string cc0Path(const std::string& dir, u32 layer, const char* map) {
    const std::string asset = layerSpec(layer).asset;
    return dir + "/" + asset + "/" + asset + "_1K-JPG_" + map + ".jpg";
}

float heightAt(float u, float v) {
    // Rolling hills plus a ridge band; centred so that the mean is near 0.  Not periodic (the patch is one tile).
    const float hills = fbm(u * 0.5f + 0.17f, v * 0.5f + 0.31f, 3, 5, 100);
    const float ridge = 1.0f - std::abs(2.0f * fbm(u * 0.5f + 0.61f, v * 0.5f + 0.07f, 4, 4, 200) - 1.0f);
    return kHeightAmplitude * (0.62f * hills + 0.38f * ridge * ridge - 0.45f);
}

Image proceduralMask(u32 size) {
    Image mask = makeImage(size);
    const float lo = heightAt(0.0f, 0.0f);
    (void)lo;
    for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
            const float h = heightAt(u, v);
            const float slope = slopeAt(u, v);
            const float n1 = fbm(u, v, 8, 4, 300), n2 = fbm(u, v, 12, 3, 400);
            const float rock = smoothstep(0.16f, 0.40f, slope + 0.25f * (n1 - 0.5f));
            const float snow = smoothstep(0.10f, 0.22f, h + 0.12f * (n2 - 0.5f)) * (1.0f - 0.85f * rock);
            const float ground = smoothstep(0.35f, 0.65f, n1) * (1.0f - snow) * (1.0f - rock) * 0.9f;
            const float grass = std::max(0.02f, 1.0f - rock - snow - ground);
            setPixel(mask, x, y, grass, rock, ground, snow);
        }
    }
    return mask;
}

Image proceduralCavity(u32 size) {
    Image cavity = makeImage(size);
    const float radius = 0.02f; // uv: 0.16 m on the 8 m patch
    for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(size);
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
            const float h = heightAt(u, v);
            float mean = 0.0f;
            for (u32 k = 0; k < 8; ++k) {
                const float a = static_cast<float>(k) * 0.78539816f;
                mean += heightAt(u + radius * std::cos(a), v + radius * std::sin(a));
            }
            mean *= 0.125f;
            const float w = smoothstep(0.0f, 0.03f, mean - h); // below its neighbourhood by 3 cm: fully coated
            setPixel(cavity, x, y, w, w, w);
        }
    }
    return cavity;
}

Image constantImage(std::array<u8, 4> rgba, u32 size) {
    Image image = makeImage(size);
    for (size_t i = 0; i < image.rgba.size(); i += 4)
        for (size_t c = 0; c < 4; ++c) image.rgba[i + c] = rgba[c];
    return image;
}

LayerImages proceduralLayer(u32 layer, u32 size) {
    if (layer >= kLayerCount) throw std::out_of_range("terrain layer");
    // Per layer: two colours blended by a tileable fbm, the height from a second fbm with the layer's character.
    struct Look {
        std::array<float, 3> a, b;
        u32 period, octaves;
        float bump;
        float roughBase;
    };
    static constexpr std::array<Look, kLayerCount> looks{{
        {{0.12f, 0.30f, 0.05f}, {0.30f, 0.50f, 0.10f}, 16, 4, 2.5f, 0.75f}, // grass: fine and soft
        {{0.20f, 0.19f, 0.18f}, {0.42f, 0.40f, 0.37f}, 6, 5, 6.0f, 0.85f},  // rock: coarse and deep
        {{0.18f, 0.12f, 0.07f}, {0.34f, 0.25f, 0.15f}, 12, 4, 3.0f, 0.90f}, // ground
        {{0.80f, 0.82f, 0.85f}, {0.95f, 0.96f, 0.98f}, 8, 3, 0.8f, 0.55f},  // snow: bright and smooth
    }};
    const Look& look = looks[layer];
    LayerImages out{makeImage(size), makeImage(size), makeImage(size), makeImage(size), makeImage(size)};
    std::vector<float> height(static_cast<size_t>(size) * size);
    for (u32 y = 0; y < size; ++y)
        for (u32 x = 0; x < size; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(size), v = static_cast<float>(y) / static_cast<float>(size);
            height[static_cast<size_t>(y) * size + x] = fbm(u, v, look.period, look.octaves, 10 * (layer + 1));
        }
    auto at = [&](i32 x, i32 y) {
        const i32 n = static_cast<i32>(size);
        return height[static_cast<size_t>(((y % n) + n) % n) * size + static_cast<size_t>(((x % n) + n) % n)];
    };
    for (u32 y = 0; y < size; ++y) {
        for (u32 x = 0; x < size; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(size), v = static_cast<float>(y) / static_cast<float>(size);
            const float h = at(static_cast<i32>(x), static_cast<i32>(y));
            const float mixer = fbm(u, v, look.period / 2 + 1, 3, 20 * (layer + 1));
            setPixel(out.color, x, y, lerp(look.a[0], look.b[0], mixer), lerp(look.a[1], look.b[1], mixer), lerp(look.a[2], look.b[2], mixer));
            const float gx = (at(static_cast<i32>(x) + 1, static_cast<i32>(y)) - at(static_cast<i32>(x) - 1, static_cast<i32>(y))) * look.bump;
            const float gy = (at(static_cast<i32>(x), static_cast<i32>(y) + 1) - at(static_cast<i32>(x), static_cast<i32>(y) - 1)) * look.bump;
            // OpenGL convention (+Y up = towards decreasing v): n ~ (-dh/dx, +dh/dy_up, 1); image rows run towards increasing v.
            const float nx = -gx * static_cast<float>(size) * 0.25f, ny = gy * static_cast<float>(size) * 0.25f;
            const float inv = 1.0f / std::sqrt(nx * nx + ny * ny + 1.0f);
            setPixel(out.normal, x, y, nx * inv * 0.5f + 0.5f, ny * inv * 0.5f + 0.5f, inv * 0.5f + 0.5f);
            const float rough = std::clamp(look.roughBase + 0.25f * (fbm(u, v, look.period, 2, 30 * (layer + 1)) - 0.5f), 0.1f, 1.0f);
            setPixel(out.roughness, x, y, rough, rough, rough);
            const float ao = 0.65f + 0.35f * h;
            setPixel(out.occlusion, x, y, ao, ao, ao);
            setPixel(out.height, x, y, h, h, h);
        }
    }
    return out;
}

MeshData makePatch(u32 grid, float uvScale) {
    MeshData mesh = ProceduralMeshes::generatePlane(kPatchSize, kPatchSize, grid, grid);
    const float half = kPatchSize * 0.5f;
    const float e = 1.0f / static_cast<float>(grid) * 0.5f;
    for (size_t i = 0; i < mesh.positions.size(); ++i) {
        const float u = mesh.uvs[i].x, v = mesh.uvs[i].y;
        mesh.positions[i].y = heightAt(u, v);
        // Normal and tangent from the analytic neighbours (metres per metre).
        const float dhdx = (heightAt(u + e, v) - heightAt(u - e, v)) / (2.0f * e * kPatchSize);
        const float dhdz = (heightAt(u, v + e) - heightAt(u, v - e)) / (2.0f * e * kPatchSize);
        const glm::vec3 n = glm::normalize(glm::vec3(-dhdx, 1.0f, -dhdz));
        const glm::vec3 t = glm::normalize(glm::vec3(1.0f, dhdx, 0.0f));
        mesh.normals[i] = n;
        // Gram-Schmidt: the tangent must be orthogonal to the normal for the handedness rule bitangent = cross(N, T) * w.
        const glm::vec3 to = glm::normalize(t - n * glm::dot(n, t));
        mesh.tangents[i] = glm::vec4(to, 1.0f);
        mesh.uvs[i] = glm::vec2(u * uvScale, v * uvScale);
    }
    (void)half;
    return mesh;
}

} // namespace phosphor::terrain
