#include "renderer/lobe_guide_reference.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace phosphor {

LobeGuideExpectation expectedLobeParams(const GPUMaterial& material, std::span<const GPUMaterialExtRecord> extensions,
                                        u32 featureUnion, bool* extensionsValid, float instanceScale) {
    LobeGuideExpectation r{};
    if (extensionsValid) *extensionsValid = true;
    // The same expression as shadeSurface: the material's feature bits, the scene union (the value of the
    // MATERIAL_FEATURES constant) and the features the resolve resolves.
    const u32 features = (material.flags >> MATERIAL_FEATURE_SHIFT) & featureUnion & LOBE_GUIDE_RESOLVED_FEATURES;
    r.params.features = features;
    // A volume without transmission has no effect: the bit is not published (shadeSurface); transmission and volume carry the ior.
    if (!(features & MATERIAL_FEATURE_TRANSMISSION)) r.params.features &= ~MATERIAL_FEATURE_VOLUME;
    if (features & (MATERIAL_FEATURE_SPECULAR_IOR | MATERIAL_FEATURE_CLEARCOAT | MATERIAL_FEATURE_SHEEN | MATERIAL_FEATURE_TRANSMISSION)) {
        if (material.extensionIndex == MATERIAL_EXTENSION_INVALID || size_t(material.extensionIndex) >= extensions.size()) {
            if (extensionsValid) *extensionsValid = false;
            r.unverified = true;
            return r;
        }
        GPUMaterialExt0 e0;
        std::memcpy(&e0, &extensions[material.extensionIndex], sizeof(e0));
        if (features & MATERIAL_FEATURE_TRANSMISSION) r.params.ior = e0.ior; // the ior of the Fresnel split and of the medium (neutral 1.5 by default)
        if (features & MATERIAL_FEATURE_SPECULAR_IOR) {
            r.params.ior = e0.ior;
            // Without a texture the GPU multiplies by the half 1.0 fallback: exact.
            r.params.specularFactor = e0.specularFactor * 1.0f;
            for (int c = 0; c < 3; ++c) r.params.specularColor[c] = e0.specularColor[c] * 1.0f;
            if (e0.specularTex != INVALID_TEXTURE_INDEX || e0.specularColorTex != INVALID_TEXTURE_INDEX)
                r.unverified = true;
        }
        if (features & MATERIAL_FEATURE_CLEARCOAT) {
            // shadeSurface: saturate(factor * tex.r), clamp(roughness * tex.g, 0.04, 1) (the half 1.0 fallback without a texture).
            const float f = e0.clearcoatFactor * 1.0f, rough = e0.clearcoatRoughness * 1.0f;
            r.params.clearcoatFactor = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
            r.params.clearcoatRoughness = rough < 0.04f ? 0.04f : (rough > 1.0f ? 1.0f : rough);
            if (e0.clearcoatFactorTex != INVALID_TEXTURE_INDEX || e0.clearcoatRoughnessTex != INVALID_TEXTURE_INDEX)
                r.unverified = true;
            r.coatNormalChecked = true;
            r.coatNormalIsShading = e0.clearcoatNormalTex == INVALID_TEXTURE_INDEX && material.normalTex == INVALID_TEXTURE_INDEX;
        }
        if (features & MATERIAL_FEATURE_SHEEN) {
            // shadeSurface: sheenColor * tex.rgb, clamp(sheenRoughness * tex.a, 0.07, 1) (the half 1.0 fallback without a texture).
            for (int c = 0; c < 3; ++c) r.params.sheenColor[c] = e0.sheenColor[c] * 1.0f;
            const float rough = e0.sheenRoughness * 1.0f;
            r.params.sheenRoughness = rough < 0.07f ? 0.07f : (rough > 1.0f ? 1.0f : rough);
            if (e0.sheenColorTex != INVALID_TEXTURE_INDEX || e0.sheenRoughnessTex != INVALID_TEXTURE_INDEX) r.unverified = true;
        }
    }
    if (features & MATERIAL_FEATURE_ANISOTROPY) {
        // Record 1 (GPUMaterialExt1).  shadeSurface: saturate(strength * tex.b) (the half 1.0 fallback without a texture); the direction
        // is the rotated mesh tangent (or the texture direction), orthogonal to the shading normal: compared by angle in checkLobeGuide.
        if (material.extensionIndex == MATERIAL_EXTENSION_INVALID || size_t(material.extensionIndex) + 1 >= extensions.size()) {
            if (extensionsValid) *extensionsValid = false;
            r.unverified = true;
            return r;
        }
        GPUMaterialExt1 e1;
        std::memcpy(&e1, &extensions[material.extensionIndex + 1], sizeof(e1));
        const float s = e1.anisotropyStrength * 1.0f;
        r.params.anisotropyStrength = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
        r.anisotropyDirectionChecked = true;
        if (e1.anisotropyTex != INVALID_TEXTURE_INDEX) r.unverified = true;
    }
    if (features & MATERIAL_FEATURE_IRIDESCENCE) {
        // Record 1 (GPUMaterialExt1).  shadeSurface: thickness = min (1 - t) + max t with t = the thickness texture's g (the half 1.0
        // fallback without a texture: exactly the maximum); the strength is saturate(factor * tex.r), 0 where the thickness is 0 (the film is
        // disabled there).
        if (material.extensionIndex == MATERIAL_EXTENSION_INVALID || size_t(material.extensionIndex) + 1 >= extensions.size()) {
            if (extensionsValid) *extensionsValid = false;
            r.unverified = true;
            return r;
        }
        GPUMaterialExt1 e1;
        std::memcpy(&e1, &extensions[material.extensionIndex + 1], sizeof(e1));
        const float thickness = e1.iridescenceThicknessMin * (1.0f - 1.0f) + e1.iridescenceThicknessMax * 1.0f;
        const float f = e1.iridescenceFactor * 1.0f;
        r.params.iridescenceFactor = thickness > 0.0f ? (f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f)) : 0.0f;
        r.params.iridescenceIor = e1.iridescenceIor;
        r.params.iridescenceThickness = thickness;
        if (e1.iridescenceFactorTex != INVALID_TEXTURE_INDEX || e1.iridescenceThicknessTex != INVALID_TEXTURE_INDEX) r.unverified = true;
    }
    if (features & MATERIAL_FEATURE_TRANSMISSION) {
        // Record 1 (GPUMaterialExt1).  shadeSurface: t = saturate(factor * tex.r) (the half 1.0 fallback without a texture), and with the
        // volume bit the thickness (mesh-local x tex.g x the instance scale), the attenuation colour and distance.
        if (material.extensionIndex == MATERIAL_EXTENSION_INVALID || size_t(material.extensionIndex) + 1 >= extensions.size()) {
            if (extensionsValid) *extensionsValid = false;
            r.unverified = true;
            return r;
        }
        GPUMaterialExt1 e1;
        std::memcpy(&e1, &extensions[material.extensionIndex + 1], sizeof(e1));
        const float t = e1.transmissionFactor * 1.0f;
        r.params.transmissionFactor = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        if (e1.transmissionTex != INVALID_TEXTURE_INDEX) r.unverified = true;
        if (r.params.features & MATERIAL_FEATURE_VOLUME) {
            r.params.volumeThickness = e1.volumeThickness * 1.0f * instanceScale;
            for (int c = 0; c < 3; ++c) r.params.attenuationColor[c] = e1.attenuationColor[c];
            r.params.attenuationDistance = e1.attenuationDistance;
            if (e1.volumeThicknessTex != INVALID_TEXTURE_INDEX) r.unverified = true;
        }
    }
    if (features & MATERIAL_FEATURE_SUBSURFACE) {
        // Record 2 (GPUMaterialExt2).  shadeSurface publishes the bit for the thin mode (mode 0) only: weight = saturate(w * tex.a), colour =
        // color * tex.rgb (the half 1.0 fallback without a texture).  A volumetric material keeps the base model and clears the bit.
        if (material.extensionIndex == MATERIAL_EXTENSION_INVALID || size_t(material.extensionIndex) + 2 >= extensions.size()) {
            if (extensionsValid) *extensionsValid = false;
            r.unverified = true;
            return r;
        }
        GPUMaterialExt2 e2;
        std::memcpy(&e2, &extensions[material.extensionIndex + 2], sizeof(e2));
        if (e2.mode != 0u) {
            r.params.features &= ~MATERIAL_FEATURE_SUBSURFACE;
        } else {
            const float w = e2.weight * 1.0f;
            r.params.subsurfaceWeight = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);
            for (int c = 0; c < 3; ++c) r.params.subsurfaceColor[c] = e2.color[c] * 1.0f;
            if (e2.weightTex != INVALID_TEXTURE_INDEX || e2.colorTex != INVALID_TEXTURE_INDEX) r.unverified = true;
        }
    }
    return r;
}

namespace {
bool isZero(const GPULobeGuide& g) {
    const u32* w = reinterpret_cast<const u32*>(&g);
    for (size_t i = 0; i < sizeof(GPULobeGuide) / 4; ++i)
        if (w[i]) return false;
    return true;
}

void note(LobeGuideCheckResult& r, const std::string& text) {
    ++r.mismatches;
    if (r.firstMismatches.size() < 4) r.firstMismatches.push_back(text);
}

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string format(const char* fmt, ...) {
    char text[320];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    return text;
}
} // namespace

LobeGuideCheckResult checkLobeGuide(const LobeGuideCheckInput& in) {
    LobeGuideCheckResult r;
    const size_t count = size_t(in.width) * in.height;
    if (in.surfaces.size() < count || in.guides.size() < count) {
        note(r, format("readback is smaller than %u x %u pixels", in.width, in.height));
        return r;
    }
    for (size_t i = 0; i < count; ++i) {
        const GPUDISurface& s = in.surfaces[i];
        const GPULobeGuide& g = in.guides[i];
        const u32 x = u32(i % in.width), y = u32(i / in.width);
        if (!s.valid) {
            ++r.invalid;
            if (!isZero(g))
                note(r, format("pixel (%u,%u): invalid surface with a non-zero guide record (features 0x%x, ior %g)", x, y,
                               g.features, double(g.ior)));
            continue;
        }
        ++r.pixels;
        const bool slotOk = s.instanceSlot < in.instances.size() &&
                            in.instances[s.instanceSlot].generation == s.instanceGeneration;
        if (!slotOk || in.instances[s.instanceSlot].materialIndex >= in.materials.size()) {
            note(r, format("pixel (%u,%u): surface refers to a dead instance (slot %u generation %u)", x, y,
                           s.instanceSlot, s.instanceGeneration));
            continue;
        }
        const u32 materialIndex = in.instances[s.instanceSlot].materialIndex;
        bool extOk = true;
        // The thinnest axis of the instance (KHR_materials_volume thickness is mesh-local), as the shaders compute it.
        const float* m = in.instances[s.instanceSlot].modelMatrix;
        auto column = [&](int c) { return std::sqrt(m[4 * c] * m[4 * c] + m[4 * c + 1] * m[4 * c + 1] + m[4 * c + 2] * m[4 * c + 2]); };
        const float scale = std::fmin(column(0), std::fmin(column(1), column(2)));
        const LobeGuideExpectation e = expectedLobeParams(in.materials[materialIndex], in.extensions, in.featureUnion, &extOk, scale);
        if (!extOk) {
            note(r, format("pixel (%u,%u): material %u has an extension group outside the mirror", x, y, materialIndex));
            continue;
        }
        if (e.unverified) {
            ++r.unverified;
            continue;
        }
        GPULobeGuide want = packLobeGuide(e.params);
        if (e.params.features) ++r.extended; else ++r.base;
        // The oct-encoded coat normal depends on the geometry (and MSL fast-math divides): compared by angle below.
        GPULobeGuide got = g;
        if (e.coatNormalChecked) {
            const LobeVec3 cn = decodeLobeOct(g.clearcoatNormal);
            const double len = std::sqrt(double(cn.x) * cn.x + double(cn.y) * cn.y + double(cn.z) * cn.z);
            if (!(std::fabs(len - 1.0) < 1e-3)) {
                note(r, format("pixel (%u,%u) material %u: coat normal is not a unit vector (length %g)", x, y, materialIndex, len));
                continue;
            }
            if (e.coatNormalIsShading) {
                const double d = double(cn.x) * s.shadingNormal[0] + double(cn.y) * s.shadingNormal[1] + double(cn.z) * s.shadingNormal[2];
                const double sl = std::sqrt(double(s.shadingNormal[0]) * s.shadingNormal[0] + double(s.shadingNormal[1]) * s.shadingNormal[1] +
                                            double(s.shadingNormal[2]) * s.shadingNormal[2]);
                const double angle = std::acos(std::fmin(1.0, std::fmax(-1.0, d / (sl > 0.0 ? sl : 1.0))));
                if (!(angle < 2e-3)) {
                    note(r, format("pixel (%u,%u) material %u: coat normal differs from the shading normal by %g rad", x, y, materialIndex, angle));
                    continue;
                }
            }
            got.clearcoatNormal = 0;
            want.clearcoatNormal = 0;
        }
        if (e.anisotropyDirectionChecked) {
            const LobeVec3 ad = decodeLobeOct(g.anisotropyDirection);
            const double len = std::sqrt(double(ad.x) * ad.x + double(ad.y) * ad.y + double(ad.z) * ad.z);
            if (!(std::fabs(len - 1.0) < 1e-3)) {
                note(r, format("pixel (%u,%u) material %u: anisotropy direction is not a unit vector (length %g)", x, y, materialIndex, len));
                continue;
            }
            const double sl = std::sqrt(double(s.shadingNormal[0]) * s.shadingNormal[0] + double(s.shadingNormal[1]) * s.shadingNormal[1] +
                                        double(s.shadingNormal[2]) * s.shadingNormal[2]);
            const double d = (double(ad.x) * s.shadingNormal[0] + double(ad.y) * s.shadingNormal[1] + double(ad.z) * s.shadingNormal[2]) /
                             (sl > 0.0 ? sl : 1.0);
            if (!(std::fabs(d) < 2e-3)) {
                note(r, format("pixel (%u,%u) material %u: anisotropy direction is not in the tangent plane (cos to the shading normal %g)", x, y,
                               materialIndex, d));
                continue;
            }
            got.anisotropyDirection = 0;
            want.anisotropyDirection = 0;
        }
        if (std::memcmp(&got, &want, sizeof(got)) != 0) {
            u32 word = 0;
            const u32* a = reinterpret_cast<const u32*>(&got);
            const u32* b = reinterpret_cast<const u32*>(&want);
            while (word < sizeof(g) / 4 && a[word] == b[word]) ++word;
            note(r, format("pixel (%u,%u) material %u: word %u got 0x%08x want 0x%08x | ior got %g want %g", x, y,
                           materialIndex, word, a[word], b[word], double(g.ior), double(want.ior)));
        }
    }
    return r;
}

std::string formatLobeGuideCheck(u64 frame, const LobeGuideCheckResult& r) {
    std::string out = format("LOBE-GUIDE check frame %llu | pixels %u extended %u base %u unverified %u | %s",
                             static_cast<unsigned long long>(frame), r.pixels, r.extended, r.base, r.unverified,
                             r.pass() ? "PASS" : "FAIL");
    if (!r.pass()) {
        out += " (" + std::to_string(r.mismatches) + " mismatches)";
        for (const auto& m : r.firstMismatches) out += "\n  " + m;
    }
    return out;
}

} // namespace phosphor
