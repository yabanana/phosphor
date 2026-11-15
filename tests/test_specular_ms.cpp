#include <doctest/doctest.h>

#include "renderer/ms_table.h"
#include "renderer/specular_ms.h"

#include <algorithm>
#include <cmath>

using namespace phosphor;

// F15 closure (--specular-ms): the CPU twin of the multiple-scattering lobe (renderer/specular_ms.h) against a direct quadrature of the
// white furnace.  D and Vis are written here from the glTF definitions (alpha = roughness^2), the integral is a midpoint rule in
// (cos theta_i, phi_i): no table enters the single-scattering side; the extra lobe reads the logarithmic tables of renderer/ms_table.h.
namespace {

constexpr double Pi = 3.14159265358979323846;

double ggxD(double nh, double a) {
    const double a2 = a * a, d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (Pi * d * d);
}
double smithVis(double nv, double nl, double a) {
    const double a2 = a * a;
    return 0.5 / std::max(nl * std::sqrt(nv * nv * (1.0 - a2) + a2) + nv * std::sqrt(nl * nl * (1.0 - a2) + a2), 1e-12);
}

struct Furnace {
    double single = 0.0, extra = 0.0;
};

// integral over the incident hemisphere of f cos: single = D Vis (Fresnel 1), extra = the table-driven f_ms of a lobe (f0, f90).
Furnace furnaceOf(double roughness, double mu, float f0, float f90) {
    const double a = std::max(roughness * roughness, 0.002);
    const float lossAvg = msLossAvgLookup(MsLogLossAvgTable, float(roughness));
    const float lostView = msLostLookup(MsLogLostTable, float(mu), float(roughness));
    const double so = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    constexpr int nTheta = 360, nPhi = 360;
    Furnace out;
    for (int i = 0; i < nTheta; ++i) {
        const double ci = (i + 0.5) / nTheta; // midpoint in cos(theta_i): dOmega = dcos dphi
        const double si = std::sqrt(1.0 - ci * ci);
        const float lostLight = msLostLookup(MsLogLostTable, float(ci), float(roughness));
        for (int j = 0; j < nPhi; ++j) {
            const double phi = (j + 0.5) / nPhi * 2.0 * Pi;
            const double hx = si * std::cos(phi) + so, hy = si * std::sin(phi), hz = ci + mu;
            const double nh = hz / std::sqrt(hx * hx + hy * hy + hz * hz);
            const double dOmega = (1.0 / nTheta) * (2.0 * Pi / nPhi);
            out.single += ggxD(nh, a) * smithVis(mu, ci, a) * ci * dOmega;
            out.extra += msLobeValue(f0, f90, lossAvg, lostView, lostLight) * ci * dOmega;
        }
    }
    return out;
}

} // namespace

TEST_CASE("specular ms: white furnace of a rough metal reaches one (F0 = F90 = 1)") {
    for (double r : {0.3, 0.5, 0.8, 1.0})
        for (double mu : {0.3, 0.6, 1.0}) {
            const Furnace f = furnaceOf(r, mu, 1.0f, 1.0f);
            CHECK_MESSAGE(std::fabs(f.single + f.extra - 1.0) < 0.02, "r=" << r << " mu=" << mu << " total=" << f.single + f.extra);
        }
}

TEST_CASE("specular ms: negative control, the single-scattering lobe alone loses energy at high roughness") {
    const Furnace f = furnaceOf(1.0, 0.6, 1.0f, 1.0f);
    CHECK(f.single < 0.9);                   // the loss the compensation returns
    CHECK(f.extra > 0.1);                    // and it does return it
    CHECK(std::fabs(f.single - 1.0) > 0.02); // so a gate of the same width fails without the lobe
}

TEST_CASE("specular ms: no energy gain below white (dielectric F0 0.04 and a coloured metal)") {
    for (double r : {0.5, 1.0}) {
        // the extra lobe of a lobe with Fresnel F < 1 is bounded by the energy the single-scattering lobe lacks times F_ms < F_avg
        const Furnace dielectric = furnaceOf(r, 0.6, 0.04f, 1.0f);
        CHECK(dielectric.extra < 0.002);
        const Furnace metal = furnaceOf(r, 0.6, 0.8f, 1.0f);
        CHECK(metal.extra <= 1.0 - furnaceOf(r, 0.6, 1.0f, 1.0f).single + 1e-3);
    }
}

TEST_CASE("specular ms: F_avg and F_ms closed forms") {
    CHECK(msFavg(0.04f, 1.0f) == doctest::Approx(0.04 + 0.96 / 21.0));
    CHECK(msFms(1.0f, 0.59f) == doctest::Approx(1.0)); // white: all the lost energy returns (the argument is the loss 1 - E_avg)
    CHECK(msFms(0.0f, 0.59f) == doctest::Approx(0.0)); // black: nothing scatters
    CHECK(msFms(0.5f, 0.5f) == doctest::Approx(0.25 * 0.5 / (1.0 - 0.5 * 0.5)));
    // the tables: a near mirror loses nothing, a fully rough lobe loses 59 %
    CHECK(msLossAvgLookup(MsLogLossAvgTable, 0.04f) < 1e-4f);
    CHECK(msLossAvgLookup(MsLogLossAvgTable, 1.0f) == doctest::Approx(0.5908).epsilon(0.002));
    CHECK(msLostLookup(MsLogLostTable, 1.0f, 0.04f) < 1e-3f);
}
