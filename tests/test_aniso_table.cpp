#include <doctest/doctest.h>

#include "renderer/aniso_table.h"

#include <algorithm>
#include <cmath>

using namespace phosphor;

// F15.1 (ticket 11, round 3): the C++ twin of the anisotropic albedo table lookup (renderer/aniso_table.h, the rule of the MSL resolve) against a
// direct wi-space midpoint quadrature of the lobe written here from the formulas of KHR_materials_anisotropy (Burley D, README V, Schlick weight):
// no table enters the reference side.  Criteria fixed beforehand: relative error of A + B <= 1.5 % off the nodes (the interpolation bound measured by
// tools/test_aniso_table.py), exactly the node value on a node.
namespace {

constexpr double Pi = 3.14159265358979323846;

struct AB { double a = 0.0, b = 0.0; };

AB quadrature(double mu, double phi, double r, double s) {
    const double ab = std::max(r * r, 0.002), at = ab + (1.0 - ab) * s * s;
    const double sv = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    const double wo[3] = {sv * std::cos(phi), sv * std::sin(phi), mu};
    constexpr int nc = 900, np = 1800;
    AB out;
    for (int i = 0; i < nc; ++i) {
        const double c = (i + 0.5) / nc, sn = std::sqrt(1.0 - c * c);
        for (int j = 0; j < np; ++j) {
            const double p = 2.0 * Pi * (j + 0.5) / np;
            const double wi[3] = {sn * std::cos(p), sn * std::sin(p), c};
            double h[3] = {wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]};
            const double len = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
            for (double& x : h) x /= len;
            const double vh = std::clamp(wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2], 0.0, 1.0);
            const double f = h[0] * h[0] / (at * at) + h[1] * h[1] / (ab * ab) + h[2] * h[2];
            const double d = 1.0 / (Pi * at * ab * f * f);
            const double gv = c * std::sqrt(at * at * wo[0] * wo[0] + ab * ab * wo[1] * wo[1] + mu * mu);
            const double gl = mu * std::sqrt(at * at * wi[0] * wi[0] + ab * ab * wi[1] * wi[1] + c * c);
            const double v = 0.5 / std::max(gv + gl, 1e-12);
            const double w = std::pow(1.0 - vh, 5.0), k = d * v * c * (1.0 / nc) * (2.0 * Pi / np);
            out.a += k * (1.0 - w);
            out.b += k * w;
        }
    }
    return out;
}

float lookup(double mu, double phi, double r, double s, u32 component) {
    return anisoTableLookup(AnisoTable, float(mu), float(phi / (0.5 * Pi)), float(r), float(s), component);
}

} // namespace

TEST_CASE("F15.1 ticket 11: the anisotropic albedo lookup reproduces a direct quadrature off the nodes") {
    struct Case { double mu, phi, r, s; };
    const Case cases[] = {{0.55, 0.3, 0.3, 1.0}, {0.8, 1.2, 0.45, 0.7}, {0.35, 0.0, 0.3, 0.5}, {0.9, 1.5, 0.6, 1.0}, {0.25, 0.8, 0.5, 0.9}};
    for (const Case& c : cases) {
        const AB q = quadrature(c.mu, c.phi, c.r, c.s);
        const double table = double(lookup(c.mu, c.phi, c.r, c.s, 0)) + double(lookup(c.mu, c.phi, c.r, c.s, 1));
        CHECK(std::abs(table - (q.a + q.b)) <= 0.015 * (q.a + q.b));
    }
}

TEST_CASE("F15.1 ticket 11: the lookup returns the node values on the nodes and keeps the energy bound") {
    const u32 mi = 11, pi = 3, ri = 9, si = 6;
    const double mu = double(mi) / (AnisoTableMu - 1), phi = 0.5 * Pi * double(pi) / (AnisoTableAzimuth - 1);
    const double r = double(ri) / (AnisoTableRoughness - 1), s = double(si) / (AnisoTableStrength - 1);
    const u32 base = (((si * AnisoTableRoughness + ri) * AnisoTableAzimuth + pi) * AnisoTableMu + mi) * 2;
    CHECK(lookup(mu, phi, r, s, 0) == doctest::Approx(AnisoTable[base]).epsilon(1e-5));
    CHECK(lookup(mu, phi, r, s, 1) == doctest::Approx(AnisoTable[base + 1]).epsilon(1e-5));
    float most = 0.0f, least = 1.0f;
    for (u32 i = 0; i < AnisoTableMu * AnisoTableAzimuth * AnisoTableRoughness * AnisoTableStrength; ++i) {
        most = std::max(most, AnisoTable[2 * i] + AnisoTable[2 * i + 1]);
        least = std::min(least, std::min(AnisoTable[2 * i], AnisoTable[2 * i + 1]));
    }
    CHECK(most <= 1.0f + 1e-5f);
    CHECK(least >= 0.0f);
    // out-of-range coordinates clamp (no read past the table)
    CHECK(std::isfinite(anisoTableLookup(AnisoTable, 2.0f, -1.0f, 3.0f, -2.0f, 0)));
}

TEST_CASE("F15.1 ticket 11: a stronger anisotropy lowers the albedo and the strength 0 slice has no azimuth") {
    const double mu = 0.6, r = 0.3;
    double previous = 2.0;
    for (const double s : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        const double e = double(lookup(mu, 0.4, r, s, 0)) + double(lookup(mu, 0.4, r, s, 1));
        CHECK(e < previous);
        previous = e;
    }
    CHECK(lookup(mu, 0.0, r, 0.0, 0) == doctest::Approx(lookup(mu, 1.3, r, 0.0, 0)).epsilon(1e-5));
    // the azimuth matters at strength 1 (the reason it is a dimension)
    CHECK(std::abs(lookup(mu, 0.0, r, 1.0, 0) - lookup(mu, 0.5 * Pi, r, 1.0, 0)) > 0.02f);
}
