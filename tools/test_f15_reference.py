#!/usr/bin/env python3
"""Unit tests of tools/f15_reference.py (the glTF Appendix B reference).

Expected values come from the specification or from analytic identities, or from
an independent Monte Carlo estimate written in wi-space (the reference
integrates in half-vector space), never from re-running the reference formula.

Run offline:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_reference.py
Without Mitsuba the ray-casting test is skipped; numpy alone runs the rest.
"""
from __future__ import annotations

import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import f15_reference as ref  # noqa: E402

try:
    import mitsuba  # noqa: F401
    HAVE_MITSUBA = True
except ImportError:  # pragma: no cover
    HAVE_MITSUBA = False


def unit(*v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def spec_f0(f0, vh):
    return f0 + (1 - f0) * (1 - vh) ** 5


class RoughnessConventions(unittest.TestCase):
    def test_clamp_and_alpha(self):
        self.assertAlmostEqual(float(ref.effective_alpha(0.0)), 0.002)    # clamp 0.04 -> 0.0016 -> floor 0.002
        self.assertAlmostEqual(float(ref.effective_alpha(0.04)), 0.002)
        self.assertAlmostEqual(float(ref.effective_alpha(0.1)), 0.01)
        self.assertAlmostEqual(float(ref.effective_alpha(1.0)), 1.0)
        self.assertAlmostEqual(float(ref.effective_alpha(2.0)), 1.0)
        self.assertAlmostEqual(float(ref.effective_alpha(0.5, ref.CONTROLS["alpha-roughness"])), 0.5)
        self.assertAlmostEqual(ref.DIELECTRIC_F0, 0.04)


class AppendixBMix(unittest.TestCase):
    def setUp(self):
        self.n = np.array([[0.0, 0.0, 1.0]])
        self.v = np.array([unit(0.3, 0.1, 1.0)])
        self.l = np.array([unit(-0.5, 0.2, 0.7)])
        self.base = np.array([[0.8, 0.35, 0.15]])
        self.roughness = np.array([0.5])

    def parts(self):
        n, v, l = self.n[0], self.v[0], self.l[0]
        alpha = 0.25
        h = unit(*(v + l))
        nl, nv, nh, vh = l @ n, v @ n, h @ n, v @ h
        d = alpha ** 2 / (math.pi * (nh * nh * (alpha ** 2 - 1) + 1) ** 2)
        vis = 1 / (2 * (nv * math.sqrt(alpha ** 2 + (1 - alpha ** 2) * nl ** 2) +
                        nl * math.sqrt(alpha ** 2 + (1 - alpha ** 2) * nv ** 2)))
        return d * vis, vh

    def test_dielectric_reduction(self):
        spec, vh = self.parts()
        fr = spec_f0(0.04, vh)
        expected = (1 - fr) * self.base[0] / math.pi + fr * spec
        got = ref.bsdf(self.base, np.array([0.0]), self.roughness, self.n, self.v, self.l)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_metal_reduction(self):
        spec, vh = self.parts()
        expected = spec_f0(self.base[0], vh) * spec
        got = ref.bsdf(self.base, np.array([1.0]), self.roughness, self.n, self.v, self.l)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_linear_mix_at_intermediate_metallic(self):
        diel = ref.bsdf(self.base, np.array([0.0]), self.roughness, self.n, self.v, self.l)[0]
        metal = ref.bsdf(self.base, np.array([1.0]), self.roughness, self.n, self.v, self.l)[0]
        for m in (0.25, 0.5, 0.75):
            got = ref.bsdf(self.base, np.array([m]), self.roughness, self.n, self.v, self.l)[0]
            np.testing.assert_allclose(got, (1 - m) * diel + m * metal, rtol=1e-12)

    def test_engine_blend_differs_at_intermediate_metallic_only(self):
        blend = ref.CONTROLS["engine-f0-blend"]
        for m, same in ((0.0, True), (1.0, True), (0.5, False)):
            a = ref.bsdf(self.base, np.array([m]), self.roughness, self.n, self.v, self.l)
            b = ref.bsdf(self.base, np.array([m]), self.roughness, self.n, self.v, self.l, blend)
            self.assertEqual(bool(np.allclose(a, b, rtol=1e-9)), same, f"metallic {m}")

    def test_below_horizon_is_black(self):
        l = np.array([unit(0.0, 0.8, -0.2)])
        self.assertTrue(np.all(ref.bsdf(self.base, np.array([0.0]), self.roughness, self.n, self.v, l) == 0))


class EnvironmentQuadrature(unittest.TestCase):
    def test_lambert_white_returns_albedo_times_radiance(self):
        # Quadrature of (1/pi) cos over the hemisphere is exactly 1 for any view angle.
        for mu in (1.0, 0.7, 0.3, 0.05):
            self.assertAlmostEqual(ref._diffuse_integral(mu, lambda vh: np.ones_like(vh)), 1.0, places=3)
        table = ref.environment_table(0.5)
        np.testing.assert_allclose(table["Cu"], 1.0, atol=1e-3)
        # White Lambert dielectric (diffuse only, no specular, no Fresnel weighting): L_o = albedo * L_env.
        white = np.ones((1, 3)) * 0.6
        lo = ref.environment_radiance(white, np.array([0.0]), np.array([1.0]), np.array([0.8]), 2.0,
                                      ref.CONTROLS["diffuse-unweighted"])
        # unweighted diffuse + specular: strictly more than albedo * L, by the specular albedo only (<= 0.4 L)
        self.assertGreater(float(lo[0, 0]), 0.6 * 2.0)
        self.assertLess(float(lo[0, 0]), 0.6 * 2.0 + 0.4 * 2.0)

    def independent_directional_albedo(self, roughness, mu, samples=4_000_000, seed=7):
        """E(mu) = integral of D Vis cos over wi by cosine-weighted Monte Carlo (wi space)."""
        rng = np.random.default_rng(seed)
        alpha = max(roughness ** 2, 0.002)
        u1, u2 = rng.random(samples), rng.random(samples)
        r, phi = np.sqrt(u1), 2 * math.pi * u2
        wi = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1)
        wo = np.array([math.sqrt(1 - mu * mu), 0, mu])
        h = wi + wo
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        a2 = alpha * alpha
        d = a2 / (math.pi * (h[:, 2] ** 2 * (a2 - 1) + 1) ** 2)
        vis = 1 / (2 * (mu * np.sqrt(a2 + (1 - a2) * wi[:, 2] ** 2) + wi[:, 2] * math.sqrt(a2 + (1 - a2) * mu * mu)))
        # pdf = cos / pi, integrand f cos -> estimate = mean(f * pi)
        return float(np.mean(d * vis * math.pi))

    def test_ggx_directional_albedo_matches_monte_carlo(self):
        for roughness, mu in ((0.5, 0.9), (0.5, 0.4), (0.8, 0.25), (1.0, 0.6), (0.35, 0.8)):
            table = ref.environment_table(roughness)
            mus = np.array([mu])
            albedo = float(np.interp(mus, table["mu"], table["A"] + table["B"])[0])
            estimate = self.independent_directional_albedo(roughness, mu)
            self.assertLess(abs(albedo - estimate) / estimate, 0.015, f"roughness {roughness} mu {mu}: {albedo} vs {estimate}")

    def test_white_furnace_metal_and_smooth_dielectric(self):
        white = np.ones((1, 3))
        for roughness in (0.1, 0.5, 1.0):
            for mu in (1.0, 0.5, 0.1):
                out = ref.environment_radiance(white, np.array([1.0]), np.array([roughness]), np.array([mu]), 1.0)
                self.assertLessEqual(float(out.max()), 1.0 + 1e-3, (roughness, mu))  # a white conductor never gains
            out = ref.environment_radiance(white, np.array([0.0]), np.array([roughness]), np.array([1.0]), 1.0)
            self.assertLessEqual(float(out.max()), 1.0 + 1e-3, roughness)  # head-on dielectric conserves energy
        smooth_metal = ref.environment_radiance(white, np.array([1.0]), np.array([0.1]), np.array([1.0]), 1.0)
        self.assertGreater(float(smooth_metal.min()), 0.97)
        gain = ref.environment_radiance(white, np.array([0.0]), np.array([0.2]), np.array([1.0]), 1.0,
                                        ref.CONTROLS["diffuse-unweighted"])
        self.assertGreater(float(gain.max()), 1.005)

    def test_literal_appendix_b_dielectric_gains_energy_at_grazing(self):
        # The specification itself says the fresnel_mix coupling "breaks energy conservation": diffuse is
        # weighted by 1 - F(V.H) of the *incident* direction while the specular lobe carries F -> 1 at a
        # grazing view.  The reference reports the literal model; this documents the property.
        white = np.ones((1, 3))
        out = ref.environment_radiance(white, np.array([0.0]), np.array([0.1]), np.array([0.2]), 1.0)
        self.assertGreater(float(out.max()), 1.1)


class SunDisk(unittest.TestCase):
    def test_spiral_covers_the_disk_uniformly(self):
        l = unit(0.2, 0.5, 0.8)
        radius = 0.05
        samples = ref.sun_disk_samples(l, radius, 128)
        angles = np.arccos(np.clip(samples @ l, -1, 1))
        self.assertTrue(np.all(angles <= radius + 1e-12))
        # Uniform on the disk: mean of cos(angle) = (1 + cos r) / 2 over solid angle.
        self.assertAlmostEqual(float(np.mean(samples @ l)), (1 + math.cos(radius)) / 2, places=4)
        np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-12)

    def test_tiny_radius_reduces_to_a_point_light(self):
        n = np.array([unit(0.1, 0.2, 1.0), unit(-0.3, 0.1, 1.0)])
        v = np.array([unit(0.0, 0.0, 1.0), unit(0.1, -0.1, 1.0)])
        toward = unit(-0.4, 0.3, 0.9)
        base = np.array([[0.8, 0.35, 0.15]] * 2)
        metallic = np.array([0.0, 0.5])
        for roughness in (0.1, 0.5, 1.0):
            rough = np.array([roughness] * 2)
            irradiance = np.array([3.0, 3.0, 3.0])
            got, _ = ref.sun_radiance(base, metallic, rough, n, v, toward, irradiance, 1e-7)
            expected = ref.bsdf(base, metallic, rough, n, v, np.broadcast_to(toward, n.shape)) * \
                (n @ toward)[:, None] * irradiance
            np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_narrow_rule_and_counts(self):
        # Mirror geometry (R = L) with alpha < 2 r -> 128 samples; wide lobe -> point evaluation.
        n = np.array([[0.0, 0.0, 1.0]] * 2)
        toward = unit(0.0, 0.3, 1.0)
        v = np.array([unit(0.0, -0.3, 1.0)] * 2)
        count, narrow = ref.sun_quadrature(n, v, toward, np.array([0.04, 0.7]), 0.00465)
        self.assertEqual(list(count), [128, 1])  # alpha 0.002 < 2 r vs 0.49
        self.assertEqual(list(narrow), [True, False])
        count, _ = ref.sun_quadrature(n, v, toward, np.array([0.2, 0.2]), 0.00465)
        self.assertEqual(list(count), [32, 32])  # alpha 0.04: narrow but not < 2 r = 0.0093

    def test_disk_integral_converges_to_fine_quadrature(self):
        # Mirror-aligned smooth surface: the 128-sample rule must match a 20000-sample spiral.
        n = np.array([[0.0, 0.0, 1.0]])
        toward = unit(0.0, 0.3, 1.0)
        v = np.array([unit(0.0, -0.3, 1.0)])
        base = np.array([[1.0, 1.0, 1.0]])
        radius = 0.00465
        got, _ = ref.sun_radiance(base, np.array([1.0]), np.array([0.1]), n, v, toward, np.ones(3), radius)
        samples = ref.sun_disk_samples(toward, radius, 20000)
        total = np.zeros(3)
        for s in samples:
            total += ref.bsdf(base, np.array([1.0]), np.array([0.1]), n, v, s[None, :])[0] * max(0.0, s[2])
        total /= len(samples)
        np.testing.assert_allclose(got[0], total, rtol=0.02)


class DeclaredEnvironmentModel(unittest.TestCase):
    """The engine's declared environment model (ADR-0006): DFG table + Appendix B layering."""

    def synthetic_header(self, directory):
        # 3 x 3 table with A(y, x) = 10 y + x and B(y, x) = 100 y + 7 x: bilinear lookups of a
        # function that is linear in both indices are exact, so the expectation is analytic.
        values = []
        for y in range(3):
            for x in range(3):
                values += [10 * y + x, 100 * y + 7 * x]
        path = pathlib.Path(directory) / "dfg_table.h"
        path.write_text("PHOSPHOR_GPU_CONSTANT u32 DfgTableSize = 3;\nPHOSPHOR_GPU_CONSTANT float DfgTable[18] = {\n    "
                        + ", ".join(f"{v:.8e}f" for v in values) + "\n};\n")
        return path

    def test_engine_header_is_parsed(self):
        size, table = ref.load_dfg_table()
        self.assertEqual((size, table.shape), (64, (64, 64, 2)))
        text = ref.DFG_HEADER.read_text()
        self.assertIn("DfgTable[8192]", text)
        # The first row of the header starts "4.04279739e-02f, 8.19578208e-01f": (A, B) at roughness 0, N.V 1e-3.
        np.testing.assert_allclose(table[0, 0], [4.04279739e-02, 8.19578208e-01], rtol=1e-9)
        # A perfect mirror seen head-on reflects the f0 channel unchanged: A = 1, B = 0.
        a, b = ref.dfg_lookup(np.array([1.0]), np.array([0.0]))
        self.assertAlmostEqual(float(a[0]), 1.0, delta=2e-3)
        self.assertAlmostEqual(float(b[0]), 0.0, delta=2e-3)

    def test_bilinear_rule_on_a_table_that_is_linear_in_both_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = self.synthetic_header(tmp)
            self.assertEqual(ref.load_dfg_table(header)[0], 3)
            for nv, roughness in ((0.25, 0.75), (0.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.9, 0.1)):
                a, b = ref.dfg_lookup(np.array([nv]), np.array([roughness]), header)
                x, y = nv * 2, roughness * 2
                self.assertAlmostEqual(float(a[0]), 10 * y + x, places=5, msg=(nv, roughness))
                self.assertAlmostEqual(float(b[0]), 100 * y + 7 * x, places=4, msg=(nv, roughness))
            # Out-of-range inputs clamp to the table edge.
            a, _ = ref.dfg_lookup(np.array([-0.5, 2.0]), np.array([3.0, -1.0]), header)
            self.assertAlmostEqual(float(a[0]), 10 * 2 + 0, places=5)
            self.assertAlmostEqual(float(a[1]), 10 * 0 + 2, places=5)

    def test_karis_fit_published_values(self):
        # c0 = (-1, -0.0275, -0.572, 0.022), c1 = (1, 0.0425, 1.04, -0.04): at roughness 1, N.V = 1
        # r = (0, 0.015, 0.468, -0.018), a004 = 0.015, A = -1.04 * 0.015 + 0.468, B = 1.04 * 0.015 - 0.018.
        a, b = ref.karis_ab(np.array([1.0]), np.array([1.0]))
        self.assertAlmostEqual(float(a[0]), 0.4524, places=9)
        self.assertAlmostEqual(float(b[0]), -0.0024, places=9)
        # The fit is visibly lower than the table at roughness 0.3, head-on (the reason the engine moved to the table).
        fa, _ = ref.karis_ab(np.array([1.0]), np.array([0.3]))
        ta, _ = ref.dfg_lookup(np.array([1.0]), np.array([0.3]))
        self.assertLess(float(fa[0]), float(ta[0]) - 0.1)

    def test_white_dielectric_conserves_energy_exactly(self):
        # (1 - E) + (0.04 A + B) = 1 with E = 0.04 A + B: a white dielectric returns exactly L in a furnace.
        white = np.ones((4, 3))
        nv = np.array([1.0, 0.7, 0.3, 0.05])
        for roughness in (0.04, 0.35, 0.8, 1.0):
            out = ref.declared_environment_radiance(white, np.zeros(4), np.full(4, roughness), nv, 2.5)
            np.testing.assert_allclose(out, 2.5, rtol=0, atol=1e-12, err_msg=str(roughness))

    def test_metals_and_black_bodies(self):
        nv, rough = np.array([0.8]), np.array([0.5])
        a, b = ref.dfg_lookup(nv, rough)
        black = np.zeros((1, 3))
        # Black metal: f0 = 0, only B remains (clamped at 0 from below).
        out = ref.declared_environment_radiance(black, np.array([1.0]), rough, nv, 1.0)
        np.testing.assert_allclose(out, np.clip(b[0], 0, 1), atol=1e-12)
        # White metal: f0 = 1, A + B (a directional albedo, at most 1).
        out = ref.declared_environment_radiance(np.ones((1, 3)), np.array([1.0]), rough, nv, 1.0)
        np.testing.assert_allclose(out, np.clip(a[0] + b[0], 0, 1), atol=1e-12)
        self.assertLessEqual(float(out.max()), 1.0)
        # Metallic 0.5: the diffuse term is halved, f0 is the mix of 0.04 and the base colour.
        base = np.array([[0.8, 0.35, 0.15]])
        out = ref.declared_environment_radiance(base, np.array([0.5]), rough, nv, 1.0)
        f0 = 0.5 * 0.04 + 0.5 * base
        expected = 0.5 * base * (1 - (0.04 * a[0] + b[0])) + np.clip(f0 * a[0] + b[0], 0, 1)
        np.testing.assert_allclose(out, expected, rtol=1e-12)
        # The roughness clamp: 0 behaves as 0.04.
        low = ref.declared_environment_radiance(base, np.array([0.5]), np.array([0.0]), nv, 1.0)
        clamped = ref.declared_environment_radiance(base, np.array([0.5]), np.array([0.04]), nv, 1.0)
        np.testing.assert_allclose(low, clamped, rtol=1e-12)

    def test_negative_controls_are_wrong_in_the_documented_direction(self):
        base = np.array([[0.8, 0.35, 0.15]] * 2)
        metallic = np.array([0.0, 0.5])
        rough, nv = np.array([0.4, 0.4]), np.array([0.6, 0.6])
        good = ref.declared_environment_radiance(base, metallic, rough, nv, 1.0)
        unweighted = ref.declared_environment_radiance(base, metallic, rough, nv, 1.0,
                                                       ref.DECLARED_CONTROLS["declared-diffuse-unweighted"])
        blended = ref.declared_environment_radiance(base, metallic, rough, nv, 1.0,
                                                    ref.DECLARED_CONTROLS["declared-diffuse-f0-fresnel"])
        karis = ref.declared_environment_radiance(base, metallic, rough, nv, 1.0,
                                                  ref.DECLARED_CONTROLS["declared-karis-fit"])
        self.assertTrue(np.all(unweighted > good))          # no (1 - E): energy gain
        np.testing.assert_allclose(blended[0], good[0], rtol=1e-12)   # metallic 0: f0 is 0.04 either way
        self.assertTrue(np.all(blended[1] < good[1]))       # the blended f0 removes more diffuse energy
        self.assertFalse(np.allclose(karis, good, rtol=0.01))
        self.assertEqual(sorted(ref.DECLARED_CONTROLS), ["declared-diffuse-f0-fresnel", "declared-diffuse-unweighted",
                                                        "declared-karis-fit"])


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba not installed")
class SpecularIorModel(unittest.TestCase):
    """KHR_materials_ior + KHR_materials_specular against values worked by hand from the specification text."""

    def test_ior_factor_hand_values(self):
        # ((1 - ior) / (1 + ior))^2: 0.33^2 / 2.33^2 = 0.020060; 1.5 -> 0.04; 1.0 -> 0; 2.4 -> 1.4^2 / 3.4^2 = 0.169550.
        self.assertAlmostEqual(float(ref.ior_factor(1.33)), 0.02006, places=5)
        self.assertAlmostEqual(float(ref.ior_factor(1.5)), 0.04, places=12)
        self.assertEqual(float(ref.ior_factor(1.0)), 0.0)
        self.assertAlmostEqual(float(ref.ior_factor(2.4)), 0.16955, places=5)
        # ior = 0: effective IOR +infinity, the factor is its limit 1 (Fresnel evaluates to 1).
        self.assertEqual(float(ref.ior_factor(0.0)), 1.0)
        self.assertAlmostEqual(float(ref.ior_factor(1e9)), 1.0, places=6)  # the limit is continuous

    def test_f0_clamps_before_the_weight_and_colour_scales_it(self):
        # specularColor 2 at ior 1.5: min(0.08, 1) = 0.08.
        f0 = ref.specular_f0(np.array([1.5]), np.array([[2.0, 2.0, 2.0]]))
        np.testing.assert_allclose(f0, [[0.08, 0.08, 0.08]], atol=1e-12)
        # A very large colour clamps at 1 per channel: ior 2.4, colour (30, 1, 0): (1, 0.16955, 0).
        f0 = ref.specular_f0(np.array([2.4]), np.array([[30.0, 1.0, 0.0]]))
        np.testing.assert_allclose(f0, [[1.0, 0.16955, 0.0]], atol=1e-5)

    def test_fresnel_mix_end_points_follow_the_specification(self):
        # specularColor 2, specular 0.5, ior 1.5: F(V.H = 1) = f0 * specular = 0.04, F(V.H = 0) = specular = 0.5 (f90).
        f0 = ref.specular_f0(np.array([1.5]), np.array([[2.0, 2.0, 2.0]]))
        f_normal, max_normal = ref.dielectric_fresnel(f0, np.array([0.5]), np.array([[0.0]]))
        f_grazing, max_grazing = ref.dielectric_fresnel(f0, np.array([0.5]), np.array([[1.0]]))
        np.testing.assert_allclose(f_normal, [[0.04] * 3], atol=1e-12)
        np.testing.assert_allclose(f_grazing, [[0.5] * 3], atol=1e-12)
        # The diffuse base is scaled by 1 - weight * max(fr) = 1 - max(F).
        self.assertAlmostEqual(float(max_normal[0, 0]), 0.04, places=12)
        self.assertAlmostEqual(float(max_grazing[0, 0]), 0.5, places=12)

    def test_mid_angle_matches_the_unflattened_formula(self):
        # fr = f0 + (1 - f0) (1 - V.H)^5 with V.H = 0.5 -> (0.5)^5 = 0.03125; weight 0.5, f0 = (0.08, 0.04, 0.02).
        f0 = ref.specular_f0(np.array([1.5]), np.array([[2.0, 1.0, 0.5]]))
        w = np.array([[0.03125]])
        f, max_f = ref.dielectric_fresnel(f0, np.array([0.5]), w)
        fr = np.array([[0.08, 0.04, 0.02]]) + (1 - np.array([[0.08, 0.04, 0.02]])) * 0.03125
        np.testing.assert_allclose(f, 0.5 * fr, atol=1e-12)
        self.assertAlmostEqual(float(max_f[0, 0]), 0.5 * float(fr.max()), places=12)

    def test_ior_zero_is_fresnel_one_for_weight_one_and_unit_colour(self):
        f0 = ref.specular_f0(np.array([0.0]), np.array([[1.0, 1.0, 1.0]]))
        for vh in (0.0, 0.3, 1.0):
            f, _ = ref.dielectric_fresnel(f0, np.array([1.0]), np.array([[(1 - vh) ** 5]]))
            np.testing.assert_allclose(f, [[1.0] * 3], atol=1e-12)
        # A weight below 1 scales it: the "specular-glossiness" conversion keeps F = specular.
        f, _ = ref.dielectric_fresnel(f0, np.array([0.5]), np.array([[0.2]]))
        np.testing.assert_allclose(f, [[0.5] * 3], atol=1e-12)

    def test_neutral_extension_reduces_to_the_base_dielectric(self):
        n = np.array([[0.0, 0.0, 1.0]])
        v = np.array([unit(0.3, 0.1, 1.0)])
        l = np.array([unit(-0.5, 0.2, 0.7)])
        base = np.array([[0.8, 0.35, 0.15]])
        dielectric = {"f0": ref.specular_f0(np.array([1.5]), np.ones((1, 3))), "weight": np.array([1.0]),
                      "variant": "extension"}
        for metallic in (0.0, 0.5, 1.0):
            m = np.array([metallic])
            with_ext = ref.bsdf(base, m, np.array([0.35]), n, v, l, ref.SPECULAR_EXT, dielectric)
            plain = ref.bsdf(base, m, np.array([0.35]), n, v, l)
            np.testing.assert_allclose(with_ext, plain, rtol=1e-12)

    def test_the_metal_lobe_is_untouched_by_the_extension(self):
        n = np.array([[0.0, 0.0, 1.0]])
        v = np.array([unit(0.3, 0.1, 1.0)])
        l = np.array([unit(-0.5, 0.2, 0.7)])
        base = np.array([[0.8, 0.35, 0.15]])
        dielectric = {"f0": ref.specular_f0(np.array([2.4]), np.array([[2.0, 2.0, 2.0]])), "weight": np.array([0.5]),
                      "variant": "extension"}
        with_ext = ref.bsdf(base, np.array([1.0]), np.array([0.35]), n, v, l, ref.SPECULAR_EXT, dielectric)
        plain = ref.bsdf(base, np.array([1.0]), np.array([0.35]), n, v, l)
        np.testing.assert_allclose(with_ext, plain, rtol=1e-12)

    def test_controls_differ_from_the_specification_in_the_documented_way(self):
        f0 = ref.specular_f0(np.array([1.5]), np.array([[2.0, 2.0, 2.0]]))  # 0.08
        weight, w = np.array([0.5]), np.array([[1.0]])  # grazing: V.H = 0
        good, _ = ref.dielectric_fresnel(f0, weight, w)
        f90_one, _ = ref.dielectric_fresnel(f0, weight, w, "f90-one")
        unweighted, _ = ref.dielectric_fresnel(f0, weight, w, "f0-unweighted")
        ignored, _ = ref.dielectric_fresnel(f0, weight, w, "ignored")
        np.testing.assert_allclose(good, [[0.5] * 3], atol=1e-12)
        np.testing.assert_allclose(f90_one, [[1.0] * 3], atol=1e-12)        # f90 forced to 1
        np.testing.assert_allclose(unweighted, [[0.5] * 3], atol=1e-12)     # f90 is still the weight at grazing ...
        # ... but at normal incidence the unweighted f0 is 0.08 instead of 0.04.
        g0, _ = ref.dielectric_fresnel(f0, weight, np.array([[0.0]]))
        u0, _ = ref.dielectric_fresnel(f0, weight, np.array([[0.0]]), "f0-unweighted")
        np.testing.assert_allclose(g0, [[0.04] * 3], atol=1e-12)
        np.testing.assert_allclose(u0, [[0.08] * 3], atol=1e-12)
        np.testing.assert_allclose(ignored, [[1.0] * 3], atol=1e-12)        # base: F90 = 1 at grazing

    def test_declared_environment_split_sum_per_lobe(self):
        mu, rough = np.array([0.8]), np.array([0.35])
        a, b = ref.dfg_lookup(mu, ref.clamp_roughness(rough))
        a, b = float(a[0]), float(b[0])
        base = np.array([[0.8, 0.35, 0.15]])
        f0c = ref.specular_f0(np.array([1.33]), np.array([[2.0, 1.0, 0.5]]))
        dielectric = {"f0": f0c, "weight": np.array([0.5])}
        self.assertEqual(ref.DECLARED["extension_diffuse"], "extension")  # ticket 08: the composite is guide-aware
        out = ref.declared_environment_radiance(base, np.array([0.0]), rough, mu, 0.5, ref.DECLARED, dielectric)
        # Hand-worked: E_d = f0c * 0.5 * A + 0.5 * B per channel; diffuse weight 1 - max(E_d).
        e_d = np.clip(f0c * 0.5 * a + 0.5 * b, 0.0, 1.0)
        expected = (base * (1 - e_d.max()) + e_d) * 0.5
        np.testing.assert_allclose(out, expected, rtol=1e-12)
        # The negative control: the composite weighting the environment diffuse with the base model (pre-ticket 08).
        composite = ref.declared_environment_radiance(base, np.array([0.0]), rough, mu, 0.5,
                                                      ref.SPECULAR_DECLARED_CONTROLS["declared-extension-diffuse-base"],
                                                      dielectric)
        expected_composite = (base * (1 - min(max(0.04 * a + b, 0.0), 1.0)) + e_d) * 0.5
        np.testing.assert_allclose(composite, expected_composite, rtol=1e-12)
        # The neutral extension is exactly the base declared model.
        neutral = {"f0": ref.specular_f0(np.array([1.5]), np.ones((1, 3))), "weight": np.array([1.0])}
        for metallic in (0.0, 0.5, 1.0):
            m = np.array([metallic])
            np.testing.assert_allclose(
                ref.declared_environment_radiance(base, m, rough, mu, 0.5, ref.DECLARED, neutral),
                ref.declared_environment_radiance(base, m, rough, mu, 0.5), rtol=1e-12)
        # Metals do not see the extension at all.
        np.testing.assert_allclose(
            ref.declared_environment_radiance(base, np.array([1.0]), rough, mu, 0.5, ref.DECLARED, dielectric),
            ref.declared_environment_radiance(base, np.array([1.0]), rough, mu, 0.5), rtol=1e-12)

    def test_literal_environment_of_the_extension(self):
        # F15.1 (ticket 08): the exact quadrature of the extended dielectric.  A neutral block (ior 1.5, weight 1,
        # colour 1) is the base model; the wrong variants differ; a weight below 1 lowers the specular part and the
        # exact diffuse integral follows 1 - max(F) (checked against the neutral value and the white-furnace bound).
        mu, rough = np.array([0.8, 0.5]), np.array([0.35, 0.35])
        base, metallic = np.array([[0.8, 0.35, 0.15], [1.0, 1.0, 1.0]]), np.array([0.0, 0.0])
        neutral = {"f0": ref.specular_f0(np.array([1.5, 1.5]), np.ones((2, 3))), "weight": np.array([1.0, 1.0]),
                   "variant": "extension"}
        np.testing.assert_allclose(ref.environment_radiance(base, metallic, rough, mu, 0.5, ref.SPECULAR_EXT, neutral),
                                   ref.environment_radiance(base, metallic, rough, mu, 0.5), rtol=2e-3)
        weighted = {"f0": ref.specular_f0(np.array([2.4, 2.4]), np.full((2, 3), 2.0)), "weight": np.array([0.5, 0.5]),
                    "variant": "extension"}
        good = ref.environment_radiance(base, metallic, rough, mu, 0.5, ref.SPECULAR_EXT, weighted)
        # A white albedo-1 dielectric never gains energy: radiance <= environment radiance.
        self.assertLessEqual(float(good[1].max()), 0.5 * 1.002)
        for variant in ("ignored", "f90-one", "f0-unweighted"):
            wrong = ref.environment_radiance(base, metallic, rough, mu, 0.5, ref.SPECULAR_EXT,
                                             {**weighted, "variant": variant})
            self.assertGreater(float(np.abs(wrong - good).max() / good.max()), 0.005, variant)

    def test_declared_controls_differ_from_the_model(self):
        mu, rough = np.array([0.5]), np.array([0.35])
        base = np.array([[0.8, 0.35, 0.15]])
        dielectric = {"f0": ref.specular_f0(np.array([2.4]), np.array([[2.0, 2.0, 2.0]])), "weight": np.array([0.5])}
        good = ref.declared_environment_radiance(base, np.array([0.0]), rough, mu, 0.5, ref.DECLARED, dielectric)
        for name, variant in ref.SPECULAR_DECLARED_CONTROLS.items():
            wrong = ref.declared_environment_radiance(base, np.array([0.0]), rough, mu, 0.5, variant, dielectric)
            self.assertGreater(float(np.abs(wrong - good).max() / good.max()), 0.01, name)

    def test_word_decoding_of_record_zero(self):
        data = {"materials": [{"feature_bits": 1, "extension_index": 4}, {"feature_bits": 0}],
                "material_extensions": {"version": 1, "records_per_material": 4,
                                        "records": [[0] * 20] * 4 + [[int.from_bytes(struct.pack("<f", x), "little")
                                                                      for x in (1.33, 0.5, 2.0, 1.0, 0.25)] +
                                                                     [ref.INVALID, ref.INVALID] + [0] * 13]}}
        ior, spec, color = ref.material_specular(data, data["materials"][0])
        self.assertAlmostEqual(ior, 1.33, places=6)
        self.assertEqual((spec, color), (0.5, [2.0, 1.0, 0.25]))
        self.assertIsNone(ref.material_specular(data, data["materials"][1]))
        data["materials"][0]["feature_bits"] = 4  # sheen is outside the declared model
        with self.assertRaises(ValueError):
            ref.material_specular(data, data["materials"][0])
        data["materials"][0]["feature_bits"] = 3  # ior/specular + clearcoat is inside it (ticket 09)
        self.assertEqual(ref.material_specular(data, data["materials"][0])[1], 0.5)


def coat_dict(factor, roughness, ior=1.5, variant="spec", count=1):
    """Per-pixel coat arrays of `count` identical pixels (see ref.coat_arrays)."""
    return {"factor": np.full(count, float(factor)), "roughness": np.full(count, float(roughness)),
            "f0_ior": np.full(count, float(ref.ior_factor(ior))), "variant": variant}


class ClearcoatModel(unittest.TestCase):
    """KHR_materials_clearcoat (ticket 09) against hand values of the ratified text (research/khr-materials.md
    section 2), against analytic bounds and against independent Monte Carlo; never against the shader."""

    def setUp(self):
        self.n = np.array([[0.0, 0.0, 1.0]])
        self.v = np.array([unit(0.3, 0.1, 1.0)])
        self.l = np.array([unit(-0.5, 0.2, 0.7)])
        self.base = np.array([[0.8, 0.35, 0.15]])

    @staticmethod
    def d_vis(n, v, l, alpha):
        h = unit(*(v + l))
        nl, nv, nh = l @ n, v @ n, h @ n
        d = alpha ** 2 / (math.pi * (nh * nh * (alpha ** 2 - 1) + 1) ** 2)
        vis = 1 / (2 * (nv * math.sqrt(alpha ** 2 + (1 - alpha ** 2) * nl ** 2) +
                        nl * math.sqrt(alpha ** 2 + (1 - alpha ** 2) * nv ** 2)))
        return d * vis

    def test_coat_fresnel_hand_values(self):
        # F_c = 0.04 + 0.96 (1 - |V.Nc|)^5: head-on 0.04, |V.Nc| = 0.5 -> 0.04 + 0.96 / 32 = 0.07, grazing 1.
        n = np.array([[0.0, 0.0, 1.0]] * 3)
        v = np.array([[0.0, 0.0, 1.0], list(unit(math.sqrt(0.75), 0.0, 0.5)), [1.0, 0.0, 0.0]])
        got = ref.coat_fresnel(coat_dict(1.0, 0.1, count=3), n, v)
        np.testing.assert_allclose(got, [0.04, 0.07, 1.0], atol=1e-12)
        self.assertAlmostEqual(ref.COAT_F0, 0.04, places=12)  # ((1 - 1.5) / (1 + 1.5))^2
        # The fresnel_coat function uses |V.Nc|: a back-facing view of the coat gives the same value.
        np.testing.assert_allclose(ref.coat_fresnel(coat_dict(1.0, 0.1), n[:1], -v[1:2]), [0.07], atol=1e-12)

    def test_coated_bsdf_follows_the_layering_formula(self):
        # f = (1 - c F_c) f_material + c F_c D Vis (a_c = r_c^2), checked at one geometry for each material.
        c, rc = 0.6, 0.3
        fc = 0.04 + 0.96 * (1 - float(self.v[0] @ self.n[0])) ** 5
        coat_value = self.d_vis(self.n[0], self.v[0], self.l[0], rc ** 2)
        for metallic in (0.0, 0.5, 1.0):
            m = np.array([metallic])
            material = ref.bsdf(self.base, m, np.array([0.6]), self.n, self.v, self.l)[0]
            got = ref.bsdf(self.base, m, np.array([0.6]), self.n, self.v, self.l, coat=coat_dict(c, rc))[0]
            np.testing.assert_allclose(got, (1 - c * fc) * material + c * fc * coat_value, rtol=1e-12)

    def test_the_coat_fresnel_does_not_depend_on_the_light_direction(self):
        # |V.Nc| only: two light directions give the same attenuation, unlike a Fresnel on V.H.
        other = np.array([unit(-0.8, 0.3, 0.15)])  # side-lit from behind: V.H far from V.N
        a1, v1 = ref.coat_lobe(coat_dict(1.0, 0.3), self.n, self.v, self.l)
        a2, v2 = ref.coat_lobe(coat_dict(1.0, 0.3), self.n, self.v, other)
        np.testing.assert_allclose(a1, a2, rtol=1e-14)
        self.assertNotAlmostEqual(float(v1[0]), float(v2[0]), places=6)
        b1, _ = ref.coat_lobe(coat_dict(1.0, 0.3, variant="fresnel-vh"), self.n, self.v, self.l)
        b2, _ = ref.coat_lobe(coat_dict(1.0, 0.3, variant="fresnel-vh"), self.n, self.v, other)
        self.assertGreater(abs(float(b1[0] - b2[0])), 1e-3)

    def test_zero_coat_reduces_exactly_to_the_current_model(self):
        zero = coat_dict(0.0, 0.5)
        for metallic in (0.0, 0.5, 1.0):
            m = np.array([metallic])
            np.testing.assert_allclose(ref.bsdf(self.base, m, np.array([0.4]), self.n, self.v, self.l, coat=zero),
                                       ref.bsdf(self.base, m, np.array([0.4]), self.n, self.v, self.l), rtol=1e-14)
        sun, narrow = ref.sun_radiance(self.base, np.array([0.0]), np.array([0.4]), self.n, self.v, unit(-0.5, 0.2, 0.7),
                                       np.ones(3), 0.00465, coat=coat_dict(0.0, 0.0))
        plain, plain_narrow = ref.sun_radiance(self.base, np.array([0.0]), np.array([0.4]), self.n, self.v,
                                               unit(-0.5, 0.2, 0.7), np.ones(3), 0.00465)
        np.testing.assert_allclose(sun, plain, rtol=1e-14)
        self.assertEqual(bool(narrow[0]), bool(plain_narrow[0]))
        mu, rough = np.array([0.5]), np.array([0.4])
        material = ref.environment_radiance(self.base, np.array([0.0]), rough, mu, 0.5)
        np.testing.assert_allclose(ref.environment_coat(material, mu, zero, 0.5), material, rtol=1e-14)
        declared = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 0.5)
        np.testing.assert_allclose(ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 0.5,
                                                                     coat=zero), declared, rtol=1e-14)
        # The "ignored" control is the same reduction for a non-zero coat.
        ignored = coat_dict(1.0, 0.1, variant="ignored")
        np.testing.assert_allclose(ref.bsdf(self.base, np.array([0.0]), rough, self.n, self.v, self.l, coat=ignored),
                                   ref.bsdf(self.base, np.array([0.0]), rough, self.n, self.v, self.l), rtol=1e-14)

    def test_the_coat_term_has_its_own_cosine_mask(self):
        # Light below the horizon of the (shading) normal: neither the coat term nor the base reflect.
        below = np.array([unit(0.0, 0.8, -0.2)])
        got = ref.bsdf(self.base, np.array([0.0]), np.array([0.4]), self.n, self.v, below, coat=coat_dict(1.0, 0.2))
        self.assertTrue(np.all(got == 0))

    def test_coat_only_albedo_is_bounded_and_matches_monte_carlo(self):
        # E_coat(mu, a_c) = integral of D Vis cos (Fresnel 1) <= 1 at every mu and roughness (energy conservation of
        # the single-scatter GGX lobe), against an independent wi-space cosine Monte Carlo.
        mus = np.linspace(0.05, 1.0, 20)
        for roughness in (0.0, 0.05, 0.15, 0.4, 1.0):
            albedo, weighted = ref.coat_albedo(mus, np.full(20, roughness))
            self.assertLessEqual(float(albedo.max()), 1.0 + 1e-3, roughness)
            self.assertGreater(float(albedo.min()), 0.0)
            np.testing.assert_array_less(weighted, albedo + 1e-12)  # B = the w-weighted part of A + B
        helper = EnvironmentQuadrature()
        for roughness, mu in ((0.4, 0.6), (1.0, 0.3), (0.15, 0.9)):
            albedo, _ = ref.coat_albedo(np.array([mu]), np.array([roughness]))
            estimate = helper.independent_directional_albedo(roughness, mu)
            self.assertLess(abs(float(albedo[0]) - estimate) / estimate, 0.015, (roughness, mu))

    def test_white_metal_under_a_coat_never_gains_energy_in_the_furnace(self):
        # A white metal conserves energy (<= 1); the coat mixes it with E_coat <= 1 by c F_c: still <= 1.
        white = np.ones((1, 3))
        for roughness in (0.0, 0.15, 1.0):
            for factor in (0.25, 0.5, 1.0):
                for mu in (1.0, 0.5, 0.1):
                    material = ref.environment_radiance(white, np.array([1.0]), np.array([0.6]), np.array([mu]), 1.0)
                    out = ref.environment_coat(material, np.array([mu]), coat_dict(factor, roughness), 1.0)
                    self.assertLessEqual(float(out.max()), 1.0 + 1e-3, (roughness, factor, mu))
        # The unattenuated control adds the coat ON TOP of the full base: it gains energy.
        material = ref.environment_radiance(white, np.array([1.0]), np.array([0.1]), np.array([0.3]), 1.0)
        gain = ref.environment_coat(material, np.array([0.3]), coat_dict(1.0, 0.15, variant="base-unattenuated"), 1.0)
        self.assertGreater(float(gain.max()), 1.0 + 0.01)

    def test_exact_environment_formula_and_hand_value(self):
        mu = np.array([0.5])
        rough = np.array([0.6])
        coat = coat_dict(0.5, 0.4)
        material = ref.environment_radiance(self.base, np.array([0.0]), rough, mu, 0.5)
        got = ref.environment_coat(material, mu, coat, 0.5)
        fc = 0.04 + 0.96 * 0.5 ** 5
        albedo, _ = ref.coat_albedo(mu, np.array([0.4]))
        np.testing.assert_allclose(got, (1 - 0.5 * fc) * material + 0.5 * fc * float(albedo[0]) * 0.5, rtol=1e-12)

    def test_the_controls_move_the_environment_in_the_documented_direction(self):
        mu = np.array([0.4])
        rough = np.array([0.6])
        material = ref.environment_radiance(self.base, np.array([0.0]), rough, mu, 0.5)

        def under(variant, ior=1.5):
            return ref.environment_coat(material, mu, coat_dict(1.0, 0.15, ior=ior, variant=variant), 0.5)

        spec = under("spec")
        self.assertTrue(np.all(under("ignored") < spec))                         # no coat: darker than the coated copper
        self.assertTrue(np.all(under("base-unattenuated") > spec))               # base not darkened: more energy
        self.assertTrue(np.all(under("f0-from-ior", 2.4) > under("spec", 2.4)))   # F0 0.17 instead of 0.04: more coat
        self.assertTrue(np.all(under("f0-from-ior", 1.0) < under("spec", 1.0)))   # F0 0: less coat
        np.testing.assert_allclose(under("f0-from-ior", 1.5), spec, rtol=1e-12)   # same at ior 1.5
        self.assertGreater(float(np.abs(under("fresnel-vh") - spec).max()), 1e-4)  # the V.H integral differs

    def test_declared_environment_hand_value_and_controls(self):
        mu, rough = np.array([0.5]), np.array([0.6])
        coat = coat_dict(0.5, 0.4)
        base = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0)
        got = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 2.0, coat=coat)
        a, b = ref.dfg_lookup(mu, ref.clamp_roughness(np.array([0.4])))
        fc = 0.04 + 0.96 * 0.5 ** 5
        expected = ((1 - 0.5 * fc) * base + 0.5 * fc * float(np.clip(a + b, 0, 1)[0])) * 2.0
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        full = coat_dict(1.0, 0.15, ior=2.4)
        good = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0, ref.DECLARED, coat=full)
        for name, variant in ref.COAT_DECLARED_CONTROLS.items():
            wrong = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0, variant, coat=full)
            self.assertGreater(float(np.abs(wrong - good).max() / good.max()), 0.01, name)
        ignored = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0,
                                                    ref.COAT_DECLARED_CONTROLS["declared-coat-ignored"], coat=full)
        np.testing.assert_allclose(ignored, ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0),
                                   rtol=1e-14)
        with self.assertRaises(ValueError):  # no V.H in a split-sum model: no declared fresnel-vh control
            ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0,
                                              {**ref.DECLARED, "coat": "fresnel-vh"}, coat=full)

    def test_the_material_under_the_coat_is_the_full_extended_material(self):
        dielectric = {"f0": ref.specular_f0(np.array([2.4]), np.array([[1.0, 1.0, 1.0]])), "weight": np.array([0.5]),
                      "variant": "extension"}
        m, rough = np.array([0.0]), np.array([0.35])
        c, rc = 0.7, 0.2
        material = ref.bsdf(self.base, m, rough, self.n, self.v, self.l, ref.SPECULAR_EXT, dielectric)[0]
        fc = 0.04 + 0.96 * (1 - float(self.v[0] @ self.n[0])) ** 5
        coat_value = self.d_vis(self.n[0], self.v[0], self.l[0], rc ** 2)
        got = ref.bsdf(self.base, m, rough, self.n, self.v, self.l, ref.SPECULAR_EXT, dielectric, coat_dict(c, rc))[0]
        np.testing.assert_allclose(got, (1 - c * fc) * material + c * fc * coat_value, rtol=1e-12)

    def test_controls_differ_in_the_documented_direction_for_a_light(self):
        # Direct light at one geometry: base-unattenuated > spec, f0-from-ior moves with the material ior only,
        # fresnel-vh equals the model where H = N (V.H = V.N) and differs elsewhere.
        rough = np.array([0.6])
        m = np.array([0.0])

        def f(variant, ior=1.5, l=self.l, v=self.v):
            return ref.bsdf(self.base, m, rough, self.n, v, l, ref.SPECULAR_EXT, None,
                            coat_dict(1.0, 0.05, ior=ior, variant=variant))[0]

        spec = f("spec")
        self.assertTrue(np.all(f("base-unattenuated") > spec))
        np.testing.assert_allclose(f("f0-from-ior", 1.5), spec, rtol=1e-12)
        back = np.array([unit(-0.8, 0.3, 0.15)])  # side-lit from behind: V.H differs from V.N
        self.assertGreater(abs(float(f("fresnel-vh", l=back)[0] - f("spec", l=back)[0])), 2e-4)
        mirror_l = np.array([unit(-0.3, -0.1, 1.0)])  # mirror of v about n = z: H = N
        np.testing.assert_allclose(f("fresnel-vh", l=mirror_l), f("spec", l=mirror_l), rtol=1e-12)

    def test_point_light_uses_the_coated_bsdf_at_the_light_direction(self):
        light = {"position": [-1.0, 0.4, 1.4], "intensity": 50.0, "range": 100.0, "color": [1.0, 1.0, 1.0]}
        position = np.zeros((1, 3))
        coat = coat_dict(0.8, 0.1)
        got, lit = ref.point_radiance(self.base, np.array([0.0]), np.array([0.5]), self.n, self.v, position, light,
                                      coat=coat)
        delta = np.array(light["position"])
        l = (delta / np.linalg.norm(delta))[None, :]
        incident = 50.0 * float(ref.point_window(np.linalg.norm(delta), 100.0)) / float(delta @ delta)
        expected = ref.bsdf(self.base, np.array([0.0]), np.array([0.5]), self.n, self.v, l, coat=coat)[0] * \
            float(l[0] @ self.n[0]) * incident
        np.testing.assert_allclose(got[0], expected, rtol=1e-12)
        self.assertTrue(bool(lit[0]))

    def test_sun_coat_lobe_has_its_own_alpha_axis_and_mask(self):
        # A rough base (roughness 0.6) under a smooth coat (roughness 0): the base lobe is wide (no narrow pixel), the
        # coat lobe is narrow near the mirror direction, so the union mask holds those pixels.
        n = np.array([[0.0, 0.0, 1.0]] * 3)
        toward = unit(0.0, 0.3, 1.0)
        v = np.array([unit(0.0, -0.3, 1.0), unit(0.0, -0.3, 1.0), unit(0.0, 0.9, 1.0)])
        base = np.ones((3, 3))
        metallic = np.zeros(3)
        rough = np.full(3, 0.6)
        coat = coat_dict(1.0, 0.0, count=3)
        _, plain_narrow = ref.sun_radiance(base, metallic, rough, n, v, toward, np.ones(3), 0.00465)
        radiance, narrow = ref.sun_radiance(base, metallic, rough, n, v, toward, np.ones(3), 0.00465, coat=coat)
        self.assertFalse(plain_narrow.any())
        self.assertEqual(narrow.tolist(), [True, True, False])
        # The mirror-aligned pixel matches a 20000-sample quadrature of the coated BSDF (the base part is smooth).
        samples = ref.sun_disk_samples(toward, 0.00465, 20000)
        total = np.zeros(3)
        for sample in samples:
            lk = np.broadcast_to(sample, (1, 3))
            total += ref.bsdf(base[:1], metallic[:1], rough[:1], n[:1], v[:1], lk, coat=coat_dict(1.0, 0.0))[0] * \
                max(0.0, sample[2])
        total /= len(samples)
        np.testing.assert_allclose(radiance[0], total, rtol=0.02)

    def test_the_union_mask_keeps_the_base_lobe_when_the_coat_is_wide(self):
        n = np.array([[0.0, 0.0, 1.0]])
        toward = unit(0.0, 0.3, 1.0)
        v = np.array([unit(0.0, -0.3, 1.0)])
        _, base_narrow = ref.sun_radiance(np.ones((1, 3)), np.zeros(1), np.array([0.1]), n, v, toward, np.ones(3), 0.00465)
        _, union = ref.sun_radiance(np.ones((1, 3)), np.zeros(1), np.array([0.1]), n, v, toward, np.ones(3), 0.00465,
                                    coat=coat_dict(1.0, 1.0))
        self.assertTrue(bool(base_narrow[0]) and bool(union[0]))

    def test_record_word_decoding_and_texture_refusal(self):
        def words(factor, roughness, scale=1.0, tex=(ref.INVALID, ref.INVALID, ref.INVALID)):
            w = [0] * 20
            for k, x in ((0, 1.5), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)):
                w[k] = int.from_bytes(struct.pack("<f", x), "little")
            w[5] = w[6] = ref.INVALID
            for k, x in ((7, factor), (8, roughness), (9, scale)):
                w[k] = int.from_bytes(struct.pack("<f", x), "little")
            w[10], w[11], w[12] = tex
            return w

        zero = [[0] * 20] * 3
        data = {"materials": [{"feature_bits": 2, "extension_index": 0}, {"feature_bits": 0},
                              {"feature_bits": 2, "extension_index": 4}, {"feature_bits": 2, "extension_index": 8},
                              {"feature_bits": 6, "extension_index": 0}],
                "material_extensions": {"version": 1, "records_per_material": 4,
                                        "records": [words(0.5, 0.25, 0.75)] + zero +
                                                   [words(1.0, 0.1, 1.0, (3, ref.INVALID, ref.INVALID))] + zero +
                                                   [words(1.0, 0.1, 1.0, (ref.INVALID, ref.INVALID, 7))] + zero}}
        self.assertEqual(ref.material_clearcoat(data, data["materials"][0]), (0.5, 0.25, 0.75))
        self.assertIsNone(ref.material_clearcoat(data, data["materials"][1]))
        self.assertIsNone(ref.material_specular(data, data["materials"][0]))  # coat-only: neutral specular block
        for index in (2, 3):  # a coat factor/roughness texture (word 10) or a coat normal texture (word 12)
            with self.assertRaises(ValueError):
                ref.material_clearcoat(data, data["materials"][index])
        with self.assertRaises(ValueError):  # sheen (bit 4 in feature bits 6) is outside the declared model
            ref.material_clearcoat(data, data["materials"][4])
        self.assertTrue(ref.uses_clearcoat(data) and not ref.uses_specular_extension(data))
        data["materials"][4]["feature_bits"] = 3
        self.assertEqual(ref.material_specular(data, data["materials"][4])[0], 1.5)


class RayCastedSnapshot(unittest.TestCase):
    @staticmethod
    def write_sphere_ply(path, radius=0.4, slices=48, stacks=24):
        verts, faces = [], []
        for st in range(stacks + 1):
            theta = math.pi * st / stacks
            for sl in range(slices + 1):
                phi = 2 * math.pi * sl / slices
                n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
                verts.append((*(c * radius for c in n), *n, sl / slices, st / stacks))
        for st in range(stacks):
            for sl in range(slices):
                a = st * (slices + 1) + sl
                b = a + slices + 1
                faces += [(a, b, a + 1), (a + 1, b, b + 1)]
        with open(path, "wb") as f:
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                     "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\n"
                     "property float t\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                     % (len(verts), len(faces))).encode())
            for v in verts:
                f.write(struct.pack("<8f", *v))
            for tri in faces:
                f.write(struct.pack("<B3I", 3, *tri))

    def snapshot(self, directory, sky, sun, metallic, roughness, base):
        root = pathlib.Path(directory)
        self.write_sphere_ply(root / "mesh_0.ply")
        world = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # column-major identity
        data = {"schema": 1, "state": "TEST", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
                "camera": {"position": [0, 0, 3], "direction": [0, 0, -1], "up": [0, 1, 0],
                           "fov_y_radians": math.radians(60), "near": 0.05, "far": 0, "width": 256, "height": 144,
                           "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"},
                "sky": [sky] * 3, "sun_angular_radius": 0.00465, "textures": [],
                "materials": [{"base": [*base, 1], "emissive": [0, 0, 0], "metallic": metallic, "roughness": roughness,
                               "normal_scale": 1, "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0,
                               "textures": [ref.INVALID] * 5}],
                "meshes": ["mesh_0.ply"],
                "instances": [{"slot": 0, "generation": 1, "flags": 7, "mesh": 0, "material": 0, "world": world}],
                "lights": [], "sampled_lights": []}
        if sun:
            data["lights"].append({"type": 0, "position": [0, 0, 0], "direction": list(-sun[0]),
                                   "color": [1, 1, 1], "intensity": sun[1], "range": 0, "inner": 0, "outer": 0})
        (root / "scene.json").write_text(json.dumps(data))
        return root

    def test_furnace_sphere_and_centre_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 1.0, None, 1.0, 0.5, [1.0, 1.0, 1.0])
            from f12_reference import snapshot as load
            root, data = load(root)
            image, masks, spheres, terms = ref.render_frame(root, data, threads=2)
            self.assertEqual(terms, ["environment-quadrature"])
            self.assertGreater(int(masks["roi"].sum()), 500)
            self.assertLessEqual(float(image[masks["roi"]].max()), 1.0 + 1e-3)
            # Background shows the environment itself.
            self.assertAlmostEqual(float(image[0, 0, 0]), 1.0, places=6)
            # The centre pixel looks along -z at a normal very near +z: mu ~ 1.
            table = ref.environment_table(0.5)
            expected = float(np.interp(1.0, table["mu"], table["A"] + table["B"]))  # white metal, f = 1
            self.assertAlmostEqual(float(image[72, 128, 0]), expected, delta=2e-3)

    def test_declared_render_writes_the_image_and_its_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 0.5, None, 0.5, 0.5, [0.8, 0.35, 0.15])
            out = pathlib.Path(tmp) / "out"
            code = ref.main(["render", str(root), "--out", str(out), "--declared-controls", "declared-karis-fit",
                             "declared-diffuse-unweighted"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            declared = report["declared"]
            self.assertEqual(declared["file"], "declared.pfm")
            self.assertEqual(sorted(declared["controls"]), ["declared-diffuse-unweighted", "declared-karis-fit"])
            self.assertEqual(declared["terms"], ["environment-declared"])
            self.assertEqual(len(declared["dfg_table_sha256"]), 64)
            from f12_reference import read_pfm
            self.assertEqual(read_pfm(out / "declared.pfm").shape, read_pfm(out / "reference.pfm").shape)
            for name in declared["controls"]:
                self.assertTrue((out / f"control-{name}.pfm").is_file())

    def specular_snapshot(self, directory, sky, sun, metallic, ior, weight, color):
        """A one-sphere snapshot whose material carries KHR_materials_ior/specular in record 0 (the exporter's keys)."""
        root = self.snapshot(directory, sky, sun, metallic, 0.35, [0.8, 0.35, 0.15])
        path = pathlib.Path(root) / "scene.json"
        data = json.loads(path.read_text())
        words = [int.from_bytes(struct.pack("<f", x), "little") for x in (ior, weight, *color)]
        record0 = words + [ref.INVALID, ref.INVALID] + [0] * 13
        data["materials"][0].update(feature_bits=1, extension_index=0, flags=256)
        data["material_extensions"] = {"version": 1, "records_per_material": 4,
                                       "records": [record0, [0] * 20, [0] * 20, [0] * 20]}
        path.write_text(json.dumps(data))
        return root

    def test_specular_sphere_sun_image_follows_the_extension_and_the_controls_differ(self):
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            toward = unit(-0.8, 0.35, -0.3)
            root = self.specular_snapshot(tmp, 0.0, (toward, 3.0), 0.0, 1.33, 0.5, (2.0, 1.0, 0.5))
            root, data = load(root)
            self.assertTrue(ref.uses_specular_extension(data))
            image, masks, spheres, terms = ref.render_frame(root, data, ref.SPECULAR_EXT, threads=2)
            self.assertEqual(spheres[0]["ior"], np.float32(1.33).item())
            self.assertIn("group", spheres[0])
            lit = masks["roi"] & masks["lit"]
            self.assertGreater(int(lit.sum()), 100)
            for name in ("specular-ignored", "f90-one", "specular-weight-not-on-f0"):
                other, *_ = ref.render_frame(root, data, ref.CONTROLS[name], threads=2)
                difference = np.abs(other[lit] - image[lit]).max() / image[lit].max()
                self.assertGreater(float(difference), 0.01, name)

    def test_specular_sphere_declared_environment_and_main_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.specular_snapshot(tmp, 0.5, None, 0.0, 2.4, 0.5, (2.0, 2.0, 2.0))
            out = pathlib.Path(tmp) / "out"
            code = ref.main(["render", str(root), "--out", str(out), "--declared-controls",
                             "declared-specular-ignored", "declared-f90-one", "declared-specular-weight-not-on-f0"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertEqual(report["model_variant"]["dielectric"], "extension")
            self.assertEqual(sorted(report["declared"]["controls"]),
                             ["declared-f90-one", "declared-specular-ignored", "declared-specular-weight-not-on-f0"])
            from f12_reference import read_pfm
            main = read_pfm(out / "declared.pfm")
            roi = np.load(out / "masks.npz")["roi"]
            for name in report["declared"]["controls"]:
                control = read_pfm(out / f"control-{name}.pfm")
                self.assertGreater(float(np.abs(control[roi] - main[roi]).max() / main[roi].max()), 0.01, name)

    def test_sun_lit_hemisphere_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            toward = unit(-0.8, 0.35, -0.3)
            root = self.snapshot(tmp, 0.0, (toward, 3.0), 0.0, 0.1, [0.8, 0.35, 0.15])
            from f12_reference import snapshot as load
            root, data = load(root)
            image, masks, spheres, terms = ref.render_frame(root, data, threads=2)
            self.assertEqual(terms, ["sun-disk"])
            lit = masks["roi"] & masks["lit"]
            self.assertGreater(int(lit.sum()), 100)
            unlit = masks["roi"] & ~masks["lit"]
            # No light reaches N.L <= 0 (shading normal of the facets): essentially black.
            self.assertLess(float(image[unlit].max()), 1e-6)
            # A lit diffuse-dominated pixel: rough dielectric ~ albedo/pi * E * cos (+ small specular).
            self.assertGreater(float(image[lit].max()), 0.05)

    def test_point_light_shadow_rays_follow_an_occluder(self):
        # Sphere 0 at the origin, sphere 1 between it and a point light on +x: with shadow rays the lit side of
        # sphere 0 is black, without them it is lit; the occluder itself stays lit.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 0.0, None, 0.0, 0.5, [0.8, 0.35, 0.15])
            data = json.loads((root / "scene.json").read_text())
            data["instances"].append({**data["instances"][0], "slot": 1,
                                      "world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1.2, 0, 0, 1]})
            data["lights"].append({"type": 1, "position": [3.0, 0.0, 0.0], "direction": [0, 0, -1],
                                   "color": [1, 1, 1], "intensity": 9.0, "range": 100.0, "inner": 0, "outer": 0})
            (root / "scene.json").write_text(json.dumps(data))
            from f12_reference import snapshot as load
            root, data = load(root)
            cast = ref.cast_primary_rays(root, data, 2)
            visibility = ref.point_light_visibility(root, data, cast, 2)
            self.assertEqual(len(visibility), 1)
            open_image, open_masks, _, terms = ref.render_frame(root, data, cast=cast)
            self.assertEqual(terms, ["point-light"])
            shadow_image, shadow_masks, _, terms = ref.render_frame(root, data, cast=cast, visibility=visibility)
            self.assertEqual(terms, ["point-light-shadowed"])
            facing_light = open_masks["roi"] & open_masks["lit"]
            occluded = facing_light & (open_masks["sphere"] == 0)
            occluder = facing_light & (open_masks["sphere"] == 1)
            self.assertGreater(int(occluded.sum()), 50)
            self.assertGreater(int(occluder.sum()), 50)
            self.assertGreater(float(open_image[occluded].max()), 0.01)
            self.assertLess(float(shadow_image[occluded].max()), 1e-6)  # the whole lit side is in the shadow
            self.assertFalse(bool((shadow_masks["lit"] & occluded).any()))
            # The occluder is only self-shadowed at its terminator (tessellated facets behind the interpolated
            # normal: grazing pixels, the band the runner excludes): elsewhere both images are identical.
            same = np.all(shadow_image[occluder] == open_image[occluder], axis=-1)
            self.assertGreater(float(same.mean()), 0.85)
            self.assertLess(float(open_image[occluder][~same].max()), 0.25 * float(open_image[occluder].max()))

    def test_point_light_inverse_square_and_window(self):
        light = {"type": 1, "position": [0.0, 0.0, 2.0], "color": [1, 1, 1], "intensity": 4.0, "range": 100.0}
        n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        v = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        position = np.zeros((2, 3))
        base = np.array([[0.8, 0.35, 0.15]] * 2)
        value, lit = ref.point_radiance(base, np.zeros(2), np.full(2, 0.5), n, v, position, light)
        window = (1 - (2.0 / 100.0) ** 4) ** 2
        expected = ref.bsdf(base[:1], np.zeros(1), np.full(1, 0.5), n[:1], v[:1], np.array([[0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(value[0], expected[0] * 4.0 * window / 4.0, rtol=1e-12)
        self.assertEqual(lit.tolist(), [True, False])
        np.testing.assert_array_equal(value[1], 0.0)
        self.assertAlmostEqual(float(ref.point_window(np.array(50.0), 100.0)), (1 - 0.5 ** 4) ** 2)
        self.assertEqual(float(ref.point_window(np.array(150.0), 100.0)), 0.0)

    def test_point_lights_and_their_sampled_delta_twins_are_in_the_declared_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 0.0, None, 0.0, 0.5, [0.8, 0.35, 0.15])
            data = json.loads((root / "scene.json").read_text())
            point = {"type": 1, "position": [1.0, 2.0, 3.0], "direction": [0, 0, -1], "color": [1, 1, 1],
                     "intensity": 9.0, "range": 100.0, "inner": 0, "outer": 0}
            data["lights"].append(point)
            data["sampled_lights"].append({"type": 1, "position": [1.0, 2.0, 3.0]})
            ref.require_declared_scene(data)
            data["sampled_lights"].append({"type": 3, "position": [1.0, 2.0, 3.0]})  # an area light
            with self.assertRaises(ValueError):
                ref.require_declared_scene(data)
            data["sampled_lights"].pop()
            data["lights"][0] = {**point, "color": [1, 0.5, 1]}
            with self.assertRaises(ValueError):
                ref.require_declared_scene(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
