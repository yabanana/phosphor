// F15.1 (ticket 13): scalar math of the ray-traced transmission transport (renderer/transmission_math.h, shared with MSL).
#include <doctest/doctest.h>
#include "renderer/transmission_math.h"

#include <array>
#include <cmath>
#include <cstdint>

using namespace phosphor;

namespace {
using V3 = std::array<double, 3>;
V3 operator*(double s, V3 a) { return {s * a[0], s * a[1], s * a[2]}; }
V3 operator+(V3 a, V3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V3 operator-(V3 a) { return {-a[0], -a[1], -a[2]}; }
double dot(V3 a, V3 b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 normalize(V3 a) { const double l = std::sqrt(dot(a, a)); return {a[0] / l, a[1] / l, a[2] / l}; }
struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    double next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return double(s >> 11) * (1.0 / 9007199254740992.0); }
};
// One interface sample in the macro frame n = +Z, incident direction `i` (pointing away from the surface, i.z > 0), eta = n_t / n_i,
// GGX alpha: the estimator of the engine's walk (D(h) n.h half vector, branch by Fresnel, Walter weight).  Returns the weight of the
// sample and whether it refracted.
struct Sample { double weight; bool refracted; V3 o; };
Sample interfaceSample(V3 i, double eta, double alpha, Rng& rng) {
    const double u = rng.next(), v = rng.next();
    const double c = std::sqrt((1.0 - u) / (1.0 + (alpha * alpha - 1.0) * u)), r = std::sqrt(std::max(0.0, 1.0 - c * c)), phi = 2.0 * M_PI * v;
    const V3 h{r * std::cos(phi), r * std::sin(phi), c};
    const double iH = dot(i, h);
    if (!(iH > 0.0)) return {0.0, false, {0, 0, 1}};
    const TransmissionFresnel fr = transmissionFresnel(float(iH), float(eta));
    if (rng.next() < fr.f) {
        const V3 o = 2.0 * iH * h + (-i);
        if (!(o[2] > 0.0)) return {0.0, false, o};
        return {transmissionMicrofacetWeight(float(i[2]), float(o[2]), float(iH), float(dot(o, h)), float(h[2]), float(alpha)), false, o};
    }
    const double ratio = 1.0 / eta;
    const V3 o = (-ratio) * i + (ratio * iH - fr.cosT) * h;
    if (!(o[2] < 0.0)) return {0.0, true, o};
    return {transmissionMicrofacetWeight(float(i[2]), float(o[2]), float(iH), float(dot(o, h)), float(h[2]), float(alpha)), true, o};
}
} // namespace

TEST_CASE("transmission Fresnel: Schlick f0 entering, the transmitted cosine leaving, total internal reflection") {
    const float f0 = 0.04f;
    CHECK(transmissionFresnel(1.0f, 1.5f).f == doctest::Approx(f0));
    CHECK(transmissionFresnel(0.0f, 1.5f).f == doctest::Approx(1.0f));
    CHECK(transmissionFresnel(0.5f, 1.5f).f == doctest::Approx(f0 + (1 - f0) * std::pow(0.5f, 5.0f)));
    // Leaving glass: normal incidence is the same f0, the critical angle asin(1 / 1.5) = 41.8 degrees is the first total reflection.
    CHECK(transmissionFresnel(1.0f, 1.0f / 1.5f).f == doctest::Approx(f0));
    const float critical = std::cos(std::asin(1.0f / 1.5f));
    CHECK_FALSE(transmissionFresnel(critical + 0.01f, 1.0f / 1.5f).tir);
    CHECK(transmissionFresnel(critical - 0.01f, 1.0f / 1.5f).tir);
    CHECK(transmissionFresnel(critical - 0.01f, 1.0f / 1.5f).f == 1.0f);
    // Just below the critical angle the transmitted cosine tends to 0 and F to 1 (continuity with TIR).
    CHECK(transmissionFresnel(critical + 1e-4f, 1.0f / 1.5f).f > 0.9f);
    // Snell: sin(theta_i) = eta sin(theta_t).
    const auto s = transmissionFresnel(0.6f, 1.5f);
    CHECK(std::sqrt(1 - 0.36f) == doctest::Approx(1.5f * std::sqrt(1 - s.cosT * s.cosT)));
}

TEST_CASE("transmission selection: ratio of the albedo weights inside the floor, 1 without transmission") {
    CHECK(transmissionSelect(0.04f, 0.96f) == doctest::Approx(0.04f));
    CHECK(transmissionSelect(0.5f, 0.5f) == doctest::Approx(0.5f));
    CHECK(transmissionSelect(0.0f, 1.0f) == doctest::Approx(kTransmissionSelectFloor)); // floored, never zero
    CHECK(transmissionSelect(1.0f, 0.0f) == 1.0f);
    CHECK(transmissionSelect(0.3f, -1.0f) == 1.0f);
    CHECK(transmissionSelect(1.0f, 1e-6f) == doctest::Approx(1.0f - kTransmissionSelectFloor));
}

TEST_CASE("Beer-Lambert: c^(length / distance), channelwise, identity without a medium") {
    CHECK(transmissionBeerLambert(0.5f, 2.0f, 2.0f) == doctest::Approx(0.5f));
    CHECK(transmissionBeerLambert(0.5f, 2.0f, 4.0f) == doctest::Approx(0.25f));
    CHECK(transmissionBeerLambert(0.5f, 2.0f, 1.0f) == doctest::Approx(std::sqrt(0.5f)));
    CHECK(transmissionBeerLambert(1.0f, 0.3f, 10.0f) == 1.0f);
    CHECK(transmissionBeerLambert(0.5f, INFINITY, 10.0f) == 1.0f); // attenuationDistance +inf
    CHECK(transmissionBeerLambert(0.5f, 0.0f, 10.0f) == 1.0f);
    CHECK(transmissionBeerLambert(0.5f, 2.0f, 0.0f) == 1.0f);
    // A black channel keeps the reference's floor (finite): sigma_t = -ln(1e-6) / d.
    CHECK(transmissionBeerLambert(0.0f, 1.0f, 1.0f) == doctest::Approx(kTransmissionMinColor).epsilon(1e-3));
    // Two segments multiply: exp(-sigma (a + b)).
    CHECK(transmissionBeerLambert(0.4f, 0.8f, 0.3f) * transmissionBeerLambert(0.4f, 0.8f, 0.5f) == doctest::Approx(transmissionBeerLambert(0.4f, 0.8f, 0.8f)));
}

TEST_CASE("microfacet weight: a smooth interface weighs 1, a back-facing facet 0") {
    CHECK(transmissionMicrofacetWeight(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.002f) == doctest::Approx(1.0f));
    CHECK(transmissionG1(0.5f, -0.1f, 0.3f) == 0.0f);
    CHECK(transmissionG1(1.0f, 1.0f, 0.3f) == doctest::Approx(1.0f));
    CHECK(transmissionG1(0.2f, 0.2f, 0.5f) < transmissionG1(0.9f, 0.9f, 0.5f)); // grazing shadows more
}

TEST_CASE("interface estimator: smooth glass reflects F and refracts 1 - F of an energy-one beam, rough glass never gains energy") {
    Rng rng;
    const double eta = 1.5;
    for (double alpha : {0.002, 0.1, 0.4, 0.8})
        for (double cosI : {1.0, 0.8, 0.5, 0.25}) {
            const V3 i = normalize(V3{std::sqrt(1.0 - cosI * cosI), 0.0, cosI});
            double sumReflect = 0, sumRefract = 0;
            const int n = 200000;
            for (int k = 0; k < n; ++k) {
                const Sample s = interfaceSample(i, eta, alpha, rng);
                (s.refracted ? sumRefract : sumReflect) += s.weight;
            }
            const double reflect = sumReflect / n, refract = sumRefract / n;
            if (alpha < 0.01) {
                // The weight is 1 and the branch probability F: the means are F and 1 - F of the macro surface.
                const double f = transmissionFresnel(float(cosI), float(eta)).f;
                CHECK(reflect == doctest::Approx(f).epsilon(0.03));
                CHECK(refract == doctest::Approx(1.0 - f).epsilon(0.01));
            }
            // Energy: reflected + refracted <= 1 (Smith loses energy, never creates it) and stays above 0.7 away from grazing.
            CHECK(reflect + refract < 1.0 + 0.01);
            if (cosI >= 0.5) CHECK(reflect + refract > 0.7);
        }
}

TEST_CASE("interface estimator: leaving glass reflects totally beyond the critical angle") {
    Rng rng;
    const V3 i = normalize(V3{std::sin(1.0), 0.0, std::cos(1.0)}); // 57 degrees > 41.8
    double refract = 0, reflect = 0;
    for (int k = 0; k < 20000; ++k) {
        const Sample s = interfaceSample(i, 1.0 / 1.5, 0.002, rng);
        (s.refracted ? refract : reflect) += s.weight;
    }
    // The GGX tail (alpha 0.002) has a few facets tilted enough to leave the total-reflection cone: well below 1 % of the energy.
    CHECK(refract / 20000 < 0.01);
    CHECK(reflect / 20000 == doctest::Approx(1.0).epsilon(0.01));
}
