// F15.4 (ticket 23): seam 3 -- layer blend rules (renderer/layer_blend.h) and the layer stack's description plumbing.
// Expected values come from the rules stated in the header and from hand-worked examples, not from re-running the code.
#include <doctest/doctest.h>
#include "renderer/layer_blend.h"
#include "renderer/material_description.h"
#include "renderer/material_description_io.h"
#include "renderer/material_extension_pack.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

using namespace phosphor;

namespace {
LayerWeights W(float a, float b, float c, float d) { return LayerWeights{{a, b, c, d}}; }
LayerHeights H(float a, float b, float c, float d) { return LayerHeights{{a, b, c, d}}; }
float sum4(const LayerWeights& w) { return w.w[0] + w.w[1] + w.w[2] + w.w[3]; }
bool finite4(const LayerWeights& w) { return std::isfinite(w.w[0]) && std::isfinite(w.w[1]) && std::isfinite(w.w[2]) && std::isfinite(w.w[3]); }
u32 bits(float v) { return std::bit_cast<u32>(v); }
} // namespace

TEST_CASE("layer weights: normalised to sum 1 over the used layers") {
    const LayerWeights n = layerNormalise(W(1.0f, 3.0f, 0.0f, 4.0f), 4);
    CHECK(n.w[0] == doctest::Approx(0.125f));
    CHECK(n.w[1] == doctest::Approx(0.375f));
    CHECK(n.w[2] == 0.0f);
    CHECK(n.w[3] == doctest::Approx(0.5f));
    // Layers beyond the count do not take weight: the mask channel of an absent layer is ignored.
    const LayerWeights two = layerNormalise(W(0.25f, 0.25f, 0.9f, 0.9f), 2);
    CHECK(two.w[0] == 0.5f);
    CHECK(two.w[1] == 0.5f);
    CHECK(two.w[2] == 0.0f);
    CHECK(two.w[3] == 0.0f);
}

TEST_CASE("layer weights: all zero, negative, NaN and infinite inputs never give a NaN") {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    for (u32 count = 1; count <= 4; ++count) {
        for (const LayerWeights& in : {W(0, 0, 0, 0), W(-1, -2, 0, -0.0f), W(nan, nan, nan, nan), W(nan, 0, 0, 0), W(inf, 0.5f, 0, 0), W(inf, inf, inf, inf), W(0, 0, 0, 0.7f)}) {
            const LayerWeights n = layerNormalise(in, count);
            CHECK(finite4(n));
            CHECK(sum4(n) == doctest::Approx(1.0f));
            for (float v : n.w) CHECK((v >= 0.0f && v <= 1.0f));
            const LayerWeights h = layerHeightBlend(n, H(0.01f, -0.02f, 0.03f, 0.0f), count);
            CHECK(finite4(h));
            CHECK(sum4(h) == doctest::Approx(1.0f));
        }
    }
    // The documented rule: no coverage resolves to layer 0.
    const LayerWeights none = layerNormalise(W(0, 0, 0, 0), 4);
    CHECK(none.w[0] == 1.0f);
    // A mask that only covers a layer beyond the count also falls back to layer 0.
    CHECK(layerNormalise(W(0, 0, 1, 0), 2).w[0] == 1.0f);
}

TEST_CASE("layer weights: endpoint exactness, with and without the height contrast") {
    // A weight vector e_j gives exactly e_j (bit for bit), whatever the heights say.
    const LayerHeights heights = H(0.04f, -0.03f, 0.02f, 0.5f);
    for (u32 j = 0; j < 4; ++j) {
        LayerWeights in = W(0, 0, 0, 0);
        in.w[j] = 1.0f;
        const LayerWeights n = layerNormalise(in, 4);
        const LayerWeights h = layerHeightBlend(n, heights, 4);
        for (u32 i = 0; i < 4; ++i) {
            CHECK(bits(n.w[i]) == bits(i == j ? 1.0f : 0.0f));
            CHECK(bits(h.w[i]) == bits(i == j ? 1.0f : 0.0f));
        }
        // A non-unit single weight also normalises to exactly 1 (x / x).
        in.w[j] = 0.37f;
        CHECK(bits(layerNormalise(in, 4).w[j]) == bits(1.0f));
    }
    // A layer with weight 0 stays at 0 even when it is the tallest by far (the max(h + a) - depth formula would let it win).
    const LayerWeights h = layerHeightBlend(layerNormalise(W(0.5f, 0.5f, 0.0f, 0.0f), 4), H(0.0f, 0.0f, 1.0f, 1.0f), 4);
    CHECK(h.w[2] == 0.0f);
    CHECK(h.w[3] == 0.0f);
    CHECK(h.w[0] == doctest::Approx(0.5f));
    // Equal heights leave the weights unchanged.
    const LayerWeights n = layerNormalise(W(0.2f, 0.3f, 0.1f, 0.4f), 4);
    const LayerWeights same = layerHeightBlend(n, H(0.01f, 0.01f, 0.01f, 0.01f), 4);
    for (u32 i = 0; i < 4; ++i) CHECK(same.w[i] == doctest::Approx(n.w[i]).epsilon(1e-6));
}

TEST_CASE("layer height contrast: worked example") {
    // Two layers at weight 0.5; heights differ by S ln 3: the taller layer weighs 3x -> 0.75 / 0.25.
    const float dh = LAYER_HEIGHT_SOFTNESS_M * std::log(3.0f);
    const LayerWeights h = layerHeightBlend(W(0.5f, 0.5f, 0, 0), H(dh, 0.0f, 0, 0), 2);
    CHECK(h.w[0] == doctest::Approx(0.75f).epsilon(1e-5));
    CHECK(h.w[1] == doctest::Approx(0.25f).epsilon(1e-5));
    // The layer height itself: contrast * (texel - 0.5); a negative contrast is "no height".
    CHECK(layerHeight(0.04f, 1.0f) == doctest::Approx(0.02f));
    CHECK(layerHeight(0.04f, 0.0f) == doctest::Approx(-0.02f));
    CHECK(layerHeight(-1.0f, 0.9f) == 0.0f);
    // Huge heights do not overflow the exponent (hmax is subtracted).
    const LayerWeights big = layerHeightBlend(W(0.5f, 0.5f, 0, 0), H(1.0e6f, 0.0f, 0, 0), 2);
    CHECK(finite4(big));
    CHECK(big.w[0] == 1.0f);
}

TEST_CASE("layer transitions: monotone along a mask ramp, weights stay in [0,1] and sum to 1") {
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int trial = 0; trial < 200; ++trial) {
        const float p0 = u(rng), p1 = u(rng);
        const LayerHeights heights = H((u(rng) - 0.5f) * 0.05f, (u(rng) - 0.5f) * 0.05f, 0, 0);
        float prevW1 = -1.0f, prevH1 = -1.0f, prevP = p0;
        for (int step = 0; step <= 64; ++step) {
            const float t = float(step) / 64.0f;
            const LayerWeights n = layerNormalise(W(1.0f - t, t, 0, 0), 2);
            const LayerWeights h = layerHeightBlend(n, heights, 2);
            CHECK(sum4(n) == doctest::Approx(1.0f));
            CHECK(sum4(h) == doctest::Approx(1.0f));
            CHECK((h.w[0] >= 0.0f && h.w[0] <= 1.0f && h.w[1] >= 0.0f && h.w[1] <= 1.0f));
            CHECK(n.w[1] >= prevW1 - 1e-6f);   // plain weight rises with the mask
            CHECK(h.w[1] >= prevH1 - 1e-6f);   // so does the height-contrast weight (fixed heights)
            prevW1 = n.w[1];
            prevH1 = h.w[1];
            // A parameter blended between p0 and p1 moves monotonically from p0 to p1.
            const float p = layerBlendScalar(h, p0, p1, 0.0f, 0.0f);
            if (p1 >= p0) CHECK(p >= prevP - 1e-6f);
            else CHECK(p <= prevP + 1e-6f);
            CHECK(p >= std::min(p0, p1) - 1e-6f);
            CHECK(p <= std::max(p0, p1) + 1e-6f);
            prevP = p;
        }
    }
}

TEST_CASE("layer parameters: a weight vector e_j returns the layer's own value bit for bit") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int trial = 0; trial < 200; ++trial) {
        const float p[4] = {u(rng), u(rng) * 1.7f, u(rng), u(rng) + 0.04f};
        for (u32 j = 0; j < 4; ++j) {
            LayerWeights w = W(0, 0, 0, 0);
            w.w[j] = 1.0f;
            CHECK(bits(layerBlendScalar(w, p[0], p[1], p[2], p[3])) == bits(p[j]));
        }
    }
}

TEST_CASE("layer normals: surface gradients, exact at the endpoints, linear in the weights") {
    // Worked example: t = (0.6, 0, 0.8) is the gradient (0.75, 0).
    LayerNormals t{{0.6f, 0.0f, -0.6f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.8f, 1.0f, 0.8f, 1.0f}};
    const LayerNormalResult half = layerNormalBlend(W(0.5f, 0.5f, 0, 0), t); // layer 1 is flat
    CHECK(half.a == doctest::Approx(0.375f));
    CHECK(half.b == 0.0f);
    CHECK(half.c == 1.0f);
    // Opposite slopes cancel at equal weights (two layers' relief adds as gradients).
    const LayerNormalResult cancel = layerNormalBlend(W(0.5f, 0, 0.5f, 0), t);
    CHECK(cancel.a == doctest::Approx(0.0f).epsilon(1e-6));
    // The endpoint returns the raw tangent-space normal (the caller's arithmetic is the single material's).
    for (u32 j = 0; j < 4; ++j) {
        LayerWeights w = W(0, 0, 0, 0);
        w.w[j] = 1.0f;
        const LayerNormalResult r = layerNormalBlend(w, t);
        CHECK(bits(r.a) == bits(t.tx[j]));
        CHECK(bits(r.b) == bits(t.ty[j]));
        CHECK(bits(r.c) == bits(t.tz[j]));
    }
    // Linearity: g(w a + (1 - w) b) = w g(a) + (1 - w) g(b) for random normals.
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-0.6f, 0.6f);
    for (int trial = 0; trial < 100; ++trial) {
        LayerNormals n{};
        for (u32 i = 0; i < 4; ++i) {
            n.tx[i] = u(rng);
            n.ty[i] = u(rng);
            n.tz[i] = std::sqrt(1.0f - n.tx[i] * n.tx[i] - n.ty[i] * n.ty[i]);
        }
        const float w = 0.5f * (u(rng) + 0.6f) / 0.6f * 0.5f + 0.1f; // in [0.1, 0.6]
        const LayerNormalResult mix = layerNormalBlend(W(w, 1.0f - w, 0, 0), n);
        const LayerNormalResult a = layerNormalBlend(W(1, 0, 0, 0), n), b = layerNormalBlend(W(0, 1, 0, 0), n);
        CHECK(mix.a == doctest::Approx(w * (a.a / a.c) + (1.0f - w) * (b.a / b.c)).epsilon(1e-5));
        CHECK(mix.b == doctest::Approx(w * (a.b / a.c) + (1.0f - w) * (b.b / b.c)).epsilon(1e-5));
    }
    // A degenerate texel (tz <= 0) keeps a bounded gradient.
    LayerNormals bad{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 1, 1}};
    CHECK(std::isfinite(layerNormalBlend(W(0.5f, 0.5f, 0, 0), bad).a));
}

TEST_CASE("layer stack: the layer bit joins the feature class and the coat finish is a separate record") {
    MaterialDescription d;
    d.layers = LayerStack{};
    d.layers->mask.index = 3;
    for (u32 i = 0; i < 4; ++i) {
        MaterialLayer l;
        l.materialName = "m" + std::to_string(i);
        l.materialIndex = 10 + i;
        l.maskChannel = static_cast<TextureChannel>(i);
        l.uvScale = 4.0f + float(i);
        l.heightContrast = i == 1 ? std::optional<float>(0.04f) : std::nullopt;
        l.heightTexture.index = 20 + i;
        d.layers->layers.push_back(l);
    }
    CHECK(featureMask(d) == MATERIAL_FEATURE_LAYERS);
    const MaterialExtGroup plain = packMaterialExtensions(d);
    const GPUMaterialExt3 e3 = unpackExtRecord<GPUMaterialExt3>(plain[3]);
    CHECK(e3.layerCount == 4);
    CHECK(e3.maskTex == 3);
    for (u32 i = 0; i < 4; ++i) {
        CHECK(e3.layerMaterial[i] == 10 + i);
        CHECK(e3.uvScale[i] == 4.0f + float(i));
        CHECK(e3.heightTex[i] == 20 + i);
        CHECK(e3.heightContrast[i] == (i == 1 ? 0.04f : -1.0f));
    }
    // The coat finish: a clearcoat block on the stack's description adds its feature bit and changes record 0 only; the
    // layer record (the base the coat sits on) is byte-identical, whatever the coat factor.
    d.clearcoat = ClearcoatBlock{};
    d.clearcoat->factor = 0.0f;
    const MaterialExtGroup off = packMaterialExtensions(d);
    d.clearcoat->factor = 1.0f;
    d.clearcoat->roughness = 0.1f;
    const MaterialExtGroup on = packMaterialExtensions(d);
    CHECK(featureMask(d) == (MATERIAL_FEATURE_LAYERS | MATERIAL_FEATURE_CLEARCOAT));
    // Record 3 keeps the stack's own feature mask in its `features` word (F15.4: the resolve inherits the own blocks into the layers); every other
    // word of it is byte-identical.
    {
        GPUMaterialExt3 a = unpackExtRecord<GPUMaterialExt3>(off[3]), b = unpackExtRecord<GPUMaterialExt3>(on[3]), c = unpackExtRecord<GPUMaterialExt3>(plain[3]);
        CHECK(a.features == (MATERIAL_FEATURE_LAYERS | 0u));      // factor 0: the clearcoat block is inactive
        CHECK(b.features == (MATERIAL_FEATURE_LAYERS | MATERIAL_FEATURE_CLEARCOAT));
        CHECK(c.features == MATERIAL_FEATURE_LAYERS);
        a.features = b.features = c.features = 0;
        CHECK(std::memcmp(&a, &b, sizeof(a)) == 0);
        CHECK(std::memcmp(&c, &b, sizeof(c)) == 0);
    }
    CHECK(std::memcmp(&plain[1], &on[1], sizeof(GPUMaterialExtRecord)) == 0);
    CHECK(std::memcmp(&plain[0], &on[0], sizeof(GPUMaterialExtRecord)) != 0);
    // The base F0 of the blended parameters does not depend on the coat: the resolve gives the coat its own record, so the
    // dielectric F0 (0.04 here) stays that of the blended layers.  A neutral coat (factor 0) is inactive.
    d.clearcoat->factor = 0.0f;
    CHECK(featureMask(d) == MATERIAL_FEATURE_LAYERS);
}

TEST_CASE("layer stack: a material without a stack packs the pre-F15.4 record 3 (the new words are 0)") {
    MaterialDescription d;
    d.clearcoat = ClearcoatBlock{};
    d.clearcoat->factor = 0.5f;
    const GPUMaterialExt3 e3 = unpackExtRecord<GPUMaterialExt3>(packMaterialExtensions(d)[3]);
    CHECK(e3.layerCount == 0);
    for (u32 i = 0; i < 4; ++i) {
        CHECK(e3.uvScale[i] == 0.0f);
        CHECK(e3.heightTex[i] == 0u);
        CHECK(e3.layerMaterial[i] == MATERIAL_EXTENSION_INVALID);
        CHECK(e3.heightContrast[i] == -1.0f);
    }
    CHECK(e3.features == 0u);
    CHECK(e3.maskTex == INVALID_TEXTURE_INDEX);
}

TEST_CASE("layer stack: description file keeps uvScale and the height texture, validation rejects a bad scale") {
    NamedMaterial m;
    m.name = "terrain";
    m.description.layers = LayerStack{};
    m.description.layers->mask.path = "mask.png";
    MaterialLayer l;
    l.materialName = "grass";
    l.uvScale = 8.0f;
    l.heightContrast = 0.03f;
    l.heightTexture.path = "grass_height.png";
    m.description.layers->layers.push_back(l);
    std::string json, error;
    REQUIRE(saveMaterialDescriptions({m}, json, error));
    std::vector<NamedMaterial> back;
    REQUIRE(loadMaterialDescriptions(json, back, error));
    REQUIRE(back.size() == 1);
    REQUIRE(back[0].description.layers.has_value());
    const MaterialLayer& r = back[0].description.layers->layers.at(0);
    CHECK(r.uvScale == 8.0f);
    CHECK(r.heightTexture.path == "grass_height.png");
    CHECK(r.heightContrast == std::optional<float>(0.03f));

    MaterialDescription d = back[0].description;
    d.layers->layers[0].uvScale = 0.0f;
    bool found = false;
    for (const std::string& e : validate(d)) found = found || e.find("uvScale") != std::string::npos;
    CHECK(found);
}

TEST_CASE("layer blend: sole layer and presence-weighted secondary parameters (F15.4 features)") {
    CHECK(layerSole(W(0, 1, 0, 0)) == 1);
    CHECK(layerSole(W(0.5f, 0.5f, 0, 0)) == LAYER_STACK_MAX);
    CHECK(layerSole(W(0.999f, 0.001f, 0, 0)) == LAYER_STACK_MAX);
    // Layer 0 has no coat (presence 0): its roughness must not dilute the coat roughness of layer 1 (0.2) at any weight.
    for (float t : {0.1f, 0.5f, 0.9f}) {
        const float r = layerBlendPresence(W(1.0f - t, t, 0, 0), 0.0f, 1.0f, 0.0f, 0.0f, 0.9f, 0.2f, 0.0f, 0.0f, 0.04f);
        CHECK(r == doctest::Approx(0.2f));
    }
    // Two coats, presence 0.25 and 0.75 at equal weights: (0.25 * 0.1 + 0.75 * 0.5) / 1.0 = 0.4.
    CHECK(layerBlendPresence(W(0.5f, 0.5f, 0, 0), 0.25f, 0.75f, 0, 0, 0.1f, 0.5f, 0, 0, 0.04f) == doctest::Approx(0.4f));
    // No presence anywhere: the fallback.  The sole layer: its own value bit for bit, with or without presence.
    CHECK(layerBlendPresence(W(0.5f, 0.5f, 0, 0), 0, 0, 0, 0, 0.3f, 0.7f, 0, 0, 0.04f) == 0.04f);
    CHECK(bits(layerBlendPresence(W(0, 0, 1, 0), 0.0f, 0.0f, 0.3f, 0.0f, 0.1f, 0.2f, 0.333333f, 0.4f, 0.04f)) == bits(0.333333f));
    CHECK(layerBlendPresence(W(0, 0, 1, 0), 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.04f) == 0.04f);
}

TEST_CASE("layer features: blendable and unblendable blocks of a layer material, the stack union") {
    MaterialDescription plain;
    CHECK(layerBlendableMask(plain) == 0u);
    MaterialDescription coat;
    coat.clearcoat = ClearcoatBlock{};
    coat.clearcoat->factor = 1.0f;
    CHECK(layerBlendableMask(coat) == MATERIAL_FEATURE_CLEARCOAT);
    MaterialDescription rich = coat;
    rich.sheen = SheenBlock{};
    rich.sheen->color = {1.0f, 0.5f, 0.2f};
    rich.anisotropy = AnisotropyBlock{};
    rich.anisotropy->strength = 0.5f;
    rich.iridescence = IridescenceBlock{};
    rich.iridescence->factor = 1.0f;
    rich.specular = SpecularBlock{};
    rich.specular->ior = 1.4f;
    CHECK(layerBlendableMask(rich) == (MATERIAL_FEATURE_CLEARCOAT | MATERIAL_FEATURE_SHEEN | MATERIAL_FEATURE_ANISOTROPY | MATERIAL_FEATURE_IRIDESCENCE |
                                       MATERIAL_FEATURE_SPECULAR_IOR));
    CHECK(layerUnblendableMask(rich) == 0u);
    // Transmission and volume do not blend (the refraction / shadow transmittance passes read one record per surface).
    MaterialDescription glass = coat;
    glass.transmission = TransmissionBlock{};
    glass.transmission->factor = 1.0f;
    glass.volume = VolumeBlock{};
    glass.volume->thickness = 0.1f;
    CHECK(layerBlendableMask(glass) == MATERIAL_FEATURE_CLEARCOAT);
    CHECK(layerUnblendableMask(glass) == (MATERIAL_FEATURE_TRANSMISSION | MATERIAL_FEATURE_VOLUME));
    // Subsurface blends in its thin mode only.
    MaterialDescription thin;
    thin.subsurface = SubsurfaceBlock{};
    thin.subsurface->weight = 0.5f;
    CHECK(layerBlendableMask(thin) == MATERIAL_FEATURE_SUBSURFACE);
    thin.subsurface->mode = SubsurfaceMode::Volumetric;
    CHECK(layerBlendableMask(thin) == 0u);
    CHECK(layerUnblendableMask(thin) == MATERIAL_FEATURE_SUBSURFACE);
    // A layer that is itself a stack does not nest.
    MaterialDescription nested = coat;
    nested.layers = LayerStack{};
    CHECK(layerBlendableMask(nested) == 0u);
    // The union over a stack, looked up in the library; indices outside the library contribute nothing.
    std::vector<MaterialDescription> library{plain, coat, rich, glass};
    LayerStack stack;
    for (u32 index : {0u, 1u, 99u}) {
        MaterialLayer l;
        l.materialIndex = index;
        stack.layers.push_back(l);
    }
    CHECK(layerFeatureUnion(stack, library) == MATERIAL_FEATURE_CLEARCOAT);
    stack.layers[2].materialIndex = 2;
    CHECK(layerFeatureUnion(stack, library) == layerBlendableMask(rich));
    // The packed record carries the stack's own mask (low half) and the layers' union (high half).
    MaterialDescription owner;
    owner.layers = stack;
    owner.clearcoat = ClearcoatBlock{};
    owner.clearcoat->factor = 1.0f;
    const GPUMaterialExt3 e3 = unpackExtRecord<GPUMaterialExt3>(packMaterialExtensions(owner, layerFeatureUnion(stack, library))[3]);
    CHECK((e3.features & 0xFFFFu) == (MATERIAL_FEATURE_LAYERS | MATERIAL_FEATURE_CLEARCOAT));
    CHECK((e3.features >> 16) == layerBlendableMask(rich));
}
