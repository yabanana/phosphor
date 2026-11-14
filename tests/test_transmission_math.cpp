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

TEST_CASE("thin-walled BTDF: the directional integral equals the D(h) sampled Walter weight of the mirrored reflection") {
    // int f_t |N.L| dL over the back hemisphere (Monte Carlo, uniform directions) against E_D[|i.h| G / (|i.n||h.n|)] of the reflection
    // through the same half vectors: the thin BTDF IS the separable-Smith GGX reflection of the mirrored light.
    Rng rng;
    for (double alpha : {0.1, 0.3, 0.6})
        for (double cosV : {1.0, 0.7, 0.4}) {
            const V3 v = normalize(V3{std::sqrt(1.0 - cosV * cosV), 0.0, cosV});
            double directional = 0;
            const int n = 400000;
            for (int k = 0; k < n; ++k) {
                const double u = rng.next(), w = rng.next(), z = u, r = std::sqrt(1.0 - z * z), phi = 2.0 * M_PI * w;
                const V3 lm{r * std::cos(phi), r * std::sin(phi), z}; // mirrored light, front hemisphere (uniform: pdf 1 / 2 pi)
                const V3 h = normalize(v + lm);
                directional += transmissionThinBtdf(float(lm[2]), float(v[2]), float(h[2]), float(alpha)) * lm[2] * 2.0 * M_PI;
            }
            directional /= n;
            double walter = 0;
            for (int k = 0; k < n; ++k) {
                const Sample s = [&] {
                    // reflection only: force the branch by sampling like interfaceSample with F = 1
                    const double u = rng.next(), q = rng.next();
                    const double c = std::sqrt((1.0 - u) / (1.0 + (alpha * alpha - 1.0) * u)), rr = std::sqrt(std::max(0.0, 1.0 - c * c)), phi = 2.0 * M_PI * q;
                    const V3 h{rr * std::cos(phi), rr * std::sin(phi), c};
                    const double iH = dot(v, h);
                    if (!(iH > 0.0)) return Sample{0.0, false, {0, 0, 1}};
                    const V3 o = 2.0 * iH * h + (-v);
                    if (!(o[2] > 0.0)) return Sample{0.0, false, o};
                    return Sample{transmissionMicrofacetWeight(float(v[2]), float(o[2]), float(iH), float(dot(o, h)), float(h[2]), float(alpha)), false, o};
                }();
                walter += s.weight;
            }
            walter /= n;
            CHECK(directional == doctest::Approx(walter).epsilon(0.03));
        }
}

// F15.1 (ticket 14): the log-space shadow factors the intersection function accumulates (ADR-0004 amendment).
TEST_CASE("shadow ray transmittance: per-surface Fresnel pass, signed Beer-Lambert sum, floors") {
    // Normal incidence on ior 1.5: F = ((1.5 - 1) / (1.5 + 1))^2 = 0.04; grazing incidence reflects almost everything.
    CHECK(std::exp(transmissionShadowLnPass(1.0f, 1.5f)) == doctest::Approx(0.96).epsilon(1e-6));
    CHECK(std::exp(transmissionShadowLnPass(-1.0f, 1.5f)) == doctest::Approx(0.96).epsilon(1e-6)); // |cos|: either facing
    CHECK(std::exp(transmissionShadowLnPass(0.02f, 1.5f)) < 0.2);
    CHECK(transmissionShadowLnPass(0.7f, 0.0f) == 0.0f); // ior 0: the permanent no-Fresnel mode
    // The Schlick form of the Python declared model (tools/f15_transmission_reference.fresnel_schlick) at 25 degrees from the normal.
    const double c = std::cos(25.0 * M_PI / 180.0), f0 = 0.04, x = 1.0 - c;
    CHECK(std::exp(transmissionShadowLnPass(float(c), 1.5f)) == doctest::Approx(1.0 - (f0 + (1.0 - f0) * x * x * x * x * x)).epsilon(1e-5));
    // Two interfaces of a closed slab multiply (the log sum): (1 - F)^2, independent of the order of the two hits.
    const double pass = std::exp(transmissionShadowLnPass(float(c), 1.5f));
    CHECK(std::exp(transmissionShadowLnPass(float(c), 1.5f) + transmissionShadowLnPass(-float(c), 1.5f)) == doctest::Approx(pass * pass).epsilon(1e-5));
    // Beer-Lambert: colour 0.5 at 0.2 m, a ray crossing the slab from t = 3.0 to t = 3.3: the entry adds +sigma t_entry, the exit -sigma t_exit.
    const float sigma = transmissionShadowSigma(0.5f, 0.2f);
    CHECK(sigma == doctest::Approx(std::log(2.0) / 0.2).epsilon(1e-6));
    const double lnSum = double(sigma) * 3.0 - double(sigma) * 3.3;
    CHECK(std::exp(lnSum) == doctest::Approx(std::pow(0.5, 0.3 / 0.2)).epsilon(1e-5));
    // Either order of the two intersection-function invocations gives the same sum (order independence: no sorting).
    CHECK(double(-sigma * 3.3f + sigma * 3.0f) == doctest::Approx(lnSum).epsilon(1e-5));
    CHECK(transmissionShadowSigma(1.0f, 0.2f) == 0.0f);   // colour 1: no medium
    CHECK(transmissionShadowSigma(0.5f, 0.0f) == 0.0f);   // no attenuation distance
    CHECK(transmissionShadowSigma(0.5f, INFINITY) == 0.0f);
    CHECK(transmissionShadowSigma(0.0f, 1.0f) == doctest::Approx(-std::log(1.0e-6)).epsilon(1e-5)); // the clip of the reference
    // A zero factor (tint channel 0) floors at ln(1e-30): exp of it is 0 at float precision and the sum stays finite.
    CHECK(std::isfinite(transmissionShadowLn(0.0f)));
    CHECK(std::exp(transmissionShadowLn(0.0f)) < 1e-29);
    CHECK(transmissionShadowLn(1.0f) == 0.0f);
    // The negative control: a duplicated invocation multiplies the factor twice (the harness' `double-counted` control).
    CHECK(std::exp(2.0 * double(transmissionShadowLnPass(float(c), 1.5f))) == doctest::Approx(pass * pass).epsilon(1e-5));
}
