#include <doctest/doctest.h>

#include "renderer/subsurface_walk_math.h"

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>

using namespace phosphor;

namespace {

struct Rng {
    uint64_t s;
    float next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<float>((s >> 40) & 0xFFFFFF) * (1.0f / 16777216.0f);
    }
};

// Heitz 2018 visible-normal sample (the shader's swSampleVndf): v local, z up, v.z > 0.
glm::vec3 sampleVndf(glm::vec3 v, float alpha, float u1, float u2) {
    const glm::vec3 vh = glm::normalize(glm::vec3(alpha * v.x, alpha * v.y, v.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const glm::vec3 t1 = lensq > 1e-12f ? glm::vec3(-vh.y, vh.x, 0.0f) / std::sqrt(lensq) : glm::vec3(1, 0, 0);
    const glm::vec3 t2 = glm::cross(vh, t1);
    const float r = std::sqrt(u1), phi = 2.0f * 3.14159265f * u2;
    const float a = r * std::cos(phi);
    const float s = 0.5f * (1.0f + vh.z);
    const float b = (1.0f - s) * std::sqrt(std::max(1.0f - a * a, 0.0f)) + s * r * std::sin(phi);
    const glm::vec3 nh = a * t1 + b * t2 + std::sqrt(std::max(1.0f - a * a - b * b, 0.0f)) * vh;
    return glm::normalize(glm::vec3(alpha * nh.x, alpha * nh.y, std::max(nh.z, 0.0f)));
}

} // namespace

TEST_CASE("F15 subsurface walk math: Fresnel is the dielectric one, reciprocal and total internally reflecting") {
    CHECK(subsurfaceFresnel(1.0f, 1.5f) == doctest::Approx(0.04f).epsilon(1e-4));
    CHECK(subsurfaceFresnel(0.0f, 1.5f) == doctest::Approx(1.0f));
    // the same interface seen from the other side at the refracted angle
    for (float c : {0.9f, 0.6f, 0.3f}) {
        const float cosT = std::sqrt(1.0f - (1.0f - c * c) / 2.25f);
        CHECK(subsurfaceFresnel(c, 1.5f) == doctest::Approx(subsurfaceFresnel(cosT, 1.0f / 1.5f)).epsilon(1e-4));
    }
    CHECK(subsurfaceFresnel(0.5f, 1.0f / 1.5f) == 1.0f); // beyond the critical angle (sin 60 deg > 1 / 1.5)
    CHECK(subsurfaceFresnel(0.9f, 1.0f / 1.5f) < 1.0f);
}

TEST_CASE("F15 subsurface walk math: GGX D is a normalised distribution of microfacet normals, G1 is a masking probability") {
    for (float alpha : {0.1225f, 0.25f, 0.49f}) {
        double sum = 0.0;
        const int n = 4000;
        for (int i = 0; i < n; ++i) { // integral of D(m) cos(theta_m) over the hemisphere, uniform in cos
            const float cosM = (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
            sum += subsurfaceGgxD(cosM, alpha) * cosM * 2.0 * 3.14159265358979 / n;
        }
        CHECK(sum == doctest::Approx(1.0).epsilon(2e-3));
        CHECK(subsurfaceGgxG1(1.0f, 1.0f, alpha) == doctest::Approx(1.0f));
        CHECK(subsurfaceGgxG1(0.3f, 0.5f, alpha) > 0.0f);
        CHECK(subsurfaceGgxG1(0.3f, 0.5f, alpha) <= 1.0f);
        CHECK(subsurfaceGgxG1(0.3f, -0.5f, alpha) == 0.0f); // the microfacet faces away: masked
    }
}

TEST_CASE("F15 subsurface walk math: Henyey-Greenstein sampling has mean cosine g") {
    for (float g : {0.0f, 0.3f, -0.4f, 0.8f}) {
        double sum = 0.0;
        const int n = 20000;
        for (int i = 0; i < n; ++i) sum += subsurfaceHgCos(g, (static_cast<float>(i) + 0.5f) / n);
        CHECK(sum / n == doctest::Approx(g).epsilon(1e-2).scale(1.0));
    }
}

TEST_CASE("F15 subsurface walk math: the transmission BTDF the walk evaluates agrees with its importance-sampling weight") {
    // For an inside direction o the integral of f_t |cos i| over the outside hemisphere (what the next-event estimation integrates) must equal the
    // expectation of the visible-normal estimator (1 - F) G1(i) (n_o / n_i)^2 (Walter: n_o^2 in the BTDF, the Jacobian of the refraction).
    const float eta = kSubsurfaceIor;
    for (float alpha : {0.1225f, 0.25f}) {
        for (float cosO : {0.95f, 0.8f}) { // inside the critical cone (cos 0.745)
            
            const glm::vec3 o(std::sqrt(1.0f - cosO * cosO), 0.0f, -cosO); // points into the medium
            // quadrature over i (solid angle, outside hemisphere)
            double integral = 0.0;
            const int nu = 1500, nv = 1500;
            for (int a = 0; a < nu; ++a) {
                const float cosI = (static_cast<float>(a) + 0.5f) / nu;
                const float sinI = std::sqrt(1.0f - cosI * cosI);
                for (int b = 0; b < nv; ++b) {
                    const float phi = 2.0f * 3.14159265f * (static_cast<float>(b) + 0.5f) / nv;
                    const glm::vec3 i(sinI * std::cos(phi), sinI * std::sin(phi), cosI);
                    glm::vec3 m = -(i + eta * o);
                    const float ml = glm::length(m);
                    if (!(ml > 1e-6f)) continue;
                    m /= ml;
                    if (m.z < 0.0f) m = -m;
                    const float cosIm = glm::dot(i, m), cosOm = -glm::dot(o, m);
                    if (cosIm <= 0.0f || cosOm <= 0.0f) continue;
                    const float f = subsurfaceFresnel(cosOm, 1.0f / eta);
                    const float g1i = subsurfaceGgxG1(i.z, cosIm, alpha), g1o = subsurfaceGgxG1(cosO, cosOm, alpha);
                    integral += subsurfaceBtdfCos(cosIm, cosOm, cosO, m.z, alpha, 1.0f, eta, f, g1i, g1o) * (1.0 / nv) * (2.0 * 3.14159265) / nu;
                }
            }
            // visible-normal estimator in the frame whose z points into the medium
            Rng rng{0x1234567ull + static_cast<uint64_t>(cosO * 100)};
            double estimate = 0.0;
            const int samples = 400000;
            const glm::vec3 v(o.x, o.y, -o.z);
            for (int k = 0; k < samples; ++k) {
                const glm::vec3 mLocal = sampleVndf(v, alpha, rng.next(), rng.next());
                const float cosI = glm::dot(v, mLocal);
                if (cosI <= 0.0f) continue;
                const float f = subsurfaceFresnel(cosI, 1.0f / eta);
                const float sin2 = eta * eta * (1.0f - cosI * cosI);
                if (sin2 >= 1.0f) continue; // total internal reflection: nothing transmitted
                const float cosT = std::sqrt(1.0f - sin2);
                const glm::vec3 t = -eta * v + (eta * cosI - cosT) * mLocal; // travel direction, z points out of the medium
                if (t.z >= -1e-4f) continue;
                estimate += (1.0 - f) * subsurfaceGgxG1(t.z, glm::dot(t, mLocal), alpha) * eta * eta;
            }
            estimate /= samples;
            CHECK(integral == doctest::Approx(estimate).epsilon(0.02));
            CHECK(estimate > 0.1);
        }
    }
}

TEST_CASE("F15 subsurface walk math: the albedo table inverts the observed albedo monotonically and saturates above A_max") {
    float previous = 1.0f;
    for (int i = 0; i <= 20; ++i) {
        const float c = static_cast<float>(i) / 20.0f;
        const float alpha = subsurfaceSingleScatterAlbedo(kSubsurfaceTableData, 0.35f, 0.0f, c);
        CHECK(alpha >= 0.0f);
        CHECK(alpha <= 1.0f);
        if (i > 0) CHECK(alpha >= previous - 1e-5f); // a brighter colour needs a more scattering medium
        previous = alpha;
    }
    CHECK(subsurfaceSingleScatterAlbedo(kSubsurfaceTableData, 0.35f, 0.0f, 0.0f) == doctest::Approx(0.0f).epsilon(1e-3));
    CHECK(subsurfaceSingleScatterAlbedo(kSubsurfaceTableData, 0.35f, 0.0f, 0.95f) == doctest::Approx(1.0f)); // above A_max = 0.877 at roughness 0.35
    // the boundary traps light: the medium must scatter MORE than the draft's boundary-less mapping for the same colour (0.85 -> 0.995+)
    CHECK(subsurfaceSingleScatterAlbedo(kSubsurfaceTableData, 0.35f, 0.0f, 0.5f) > 0.9f);
}
