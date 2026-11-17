// F15.1 (ticket 09): clearcoat lobe selection and the stable denoiser key (renderer/lobe_select.h, shared with MSL).
#include <doctest/doctest.h>
#include "renderer/lobe_select.h"

#include <initializer_list>
#include <cmath>

using namespace phosphor;

TEST_CASE("coat Fresnel: F0 0.04 at normal incidence, 1 at grazing, Schlick in between") {
    CHECK(lobeCoatFresnel(1.0f) == doctest::Approx(0.04f));
    CHECK(lobeCoatFresnel(0.0f) == doctest::Approx(1.0f));
    CHECK(lobeCoatFresnel(0.5f) == doctest::Approx(0.04f + 0.96f * std::pow(0.5f, 5.0f)));
    CHECK(lobeCoatFresnel(-0.5f) == doctest::Approx(1.0f)); // clamped: callers pass |V.Nc|
    CHECK(lobeCoatFresnel(2.0f) == doctest::Approx(0.04f));
}

TEST_CASE("coat selection: weights are albedo weights, the probability is their ratio inside the floor") {
    const CoatLobeSelect s = coatLobeSelect(1.0f, 1.0f, 0.5f);
    CHECK(s.coatWeight == doctest::Approx(0.04f));
    CHECK(s.baseWeight == doctest::Approx(0.96f * 0.5f));
    CHECK(s.pCoat == doctest::Approx(0.04f / (0.04f + 0.48f)));
    CHECK(s.significant);
    // Half the coat factor halves the coat weight and attenuates the base by 1 - c F_c.
    const CoatLobeSelect h = coatLobeSelect(0.5f, 1.0f, 0.5f);
    CHECK(h.coatWeight == doctest::Approx(0.02f));
    CHECK(h.baseWeight == doctest::Approx(0.98f * 0.5f));
}

TEST_CASE("coat selection: no coat, no probability; a negligible coat is floored, never zero") {
    CHECK(coatLobeSelect(0.0f, 0.7f, 0.5f).pCoat == 0.0f);
    CHECK_FALSE(coatLobeSelect(0.0f, 0.7f, 0.5f).significant);
    const CoatLobeSelect tiny = coatLobeSelect(0.01f, 1.0f, 1.0f); // w_c = 4e-4 vs w_b ~ 1
    CHECK(tiny.pCoat == doctest::Approx(kCoatSelectFloor)); // a floor keeps the coat sampled
    CHECK_FALSE(tiny.significant);
    // A black base (no specular albedo): only the coat is left.
    const CoatLobeSelect black = coatLobeSelect(1.0f, 1.0f, 0.0f);
    CHECK(black.pCoat == doctest::Approx(1.0f - kCoatSelectFloor));
    CHECK(black.significant);
    // Factors outside [0, 1] and negative albedo behave as their clamps.
    CHECK(coatLobeSelect(3.0f, 1.0f, 0.5f).coatWeight == doctest::Approx(0.04f));
    CHECK(coatLobeSelect(1.0f, 1.0f, -2.0f).baseWeight == 0.0f);
}

TEST_CASE("coat selection: the probability stays inside [floor, 1 - floor] for any coated pixel") {
    for (float c : {0.01f, 0.1f, 0.5f, 1.0f})
        for (float nv : {0.01f, 0.2f, 0.5f, 1.0f})
            for (float e : {0.0f, 0.04f, 0.5f, 1.0f}) {
                const CoatLobeSelect s = coatLobeSelect(c, nv, e);
                CHECK(s.pCoat >= kCoatSelectFloor);
                CHECK(s.pCoat <= 1.0f - kCoatSelectFloor);
            }
}

TEST_CASE("coat significance: a pure function of the pixel (the stable denoiser key), growing towards grazing angles") {
    // A coat over a white metal (E_b ~ 0.95) at normal incidence is just significant (4.2 % >= 3 %), a quarter of it is not.
    CHECK(coatLobeSelect(1.0f, 1.0f, 0.95f).significant);
    CHECK_FALSE(coatLobeSelect(0.25f, 1.0f, 0.95f).significant);
    // At grazing angles F_c -> 1: the coat dominates whatever the factor.
    CHECK(coatLobeSelect(0.25f, 0.05f, 0.95f).significant);
    // The same inputs give the same key every time: no per-sample state.
    for (int i = 0; i < 4; ++i) CHECK(coatLobeSelect(0.5f, 0.6f, 0.3f).significant == coatLobeSelect(0.5f, 0.6f, 0.3f).significant);
}

TEST_CASE("one-sample mixture estimator is unbiased: the expectation over the lobe choice equals the sum of the lobes") {
    // Two non-negative integrands f_c, f_b over a domain with proposal densities p_c, p_b; the estimator picks the coat with
    // probability P, draws from its lobe and returns (f_c + f_b) / (P p_c + (1 - P) p_b).  Check on a discrete domain.
    const float pc[4] = {0.5f, 0.25f, 0.25f, 0.0f}, pb[4] = {0.1f, 0.2f, 0.3f, 0.4f}, fc[4] = {1.0f, 2.0f, 0.5f, 0.0f},
                fb[4] = {0.3f, 0.1f, 0.2f, 0.9f};
    for (float P : {kCoatSelectFloor, 0.3f, 0.5f, 1.0f - kCoatSelectFloor}) {
        double expect = 0.0, truth = 0.0;
        for (int i = 0; i < 4; ++i) {
            const double mix = double(P) * pc[i] + (1.0 - P) * pb[i];
            const double est = (fc[i] + fb[i]) / mix;
            expect += double(P) * pc[i] * est + (1.0 - P) * pb[i] * est;
            truth += fc[i] + fb[i];
        }
        CHECK(expect == doctest::Approx(truth));
    }
}
