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


def _force_ms_off():
    """ADR-0007: the engine and the reference default to the multiple-scattering lobe.  The tests of this file check the single-scattering
    lobe models (Appendix B hand values, layering formulas, furnace identities of the declared split sum), so every default model dict runs
    with the lobe OFF here; the lobe itself is tested in tools/test_f15_ms.py (which also asserts the defaults are ON)."""
    keep = set(ref.MS_CONTROLS) | set(ref.MS_DECLARED_CONTROLS)
    for name, value in vars(ref).items():
        if isinstance(value, dict) and value.get("ms") not in (None, "off") and name not in keep:
            value["ms"] = "off"
        if isinstance(value, dict) and name in ("CONTROLS", "SHEEN_CONTROLS", "ANISO_CONTROLS", "IRID_CONTROLS", "THIN_CONTROLS", "COMBO_CONTROLS",
                                                  "DECLARED_CONTROLS", "SPECULAR_DECLARED_CONTROLS", "COAT_DECLARED_CONTROLS", "COMBO_DECLARED_CONTROLS",
                                                  "SHEEN_DECLARED_CONTROLS", "ANISO_DECLARED_CONTROLS", "IRID_DECLARED_CONTROLS", "THIN_DECLARED_CONTROLS"):
            for entry in value.values():
                if isinstance(entry, dict) and entry.get("ms") not in (None, "off"):
                    entry["ms"] = "off"


_force_ms_off()

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
        data["materials"][0]["feature_bits"] = 32  # an unknown feature is outside the declared model
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
                              {"feature_bits": 18, "extension_index": 0}],
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
        # ticket 19: coat + iridescence (feature bits 18) is a composed combination; an unknown bit is still refused
        self.assertEqual(ref.material_clearcoat(data, data["materials"][4]), (0.5, 0.25, 0.75))
        data["materials"][4]["feature_bits"] = 2 | 32
        with self.assertRaises(ValueError):
            ref.material_clearcoat(data, data["materials"][4])
        data["materials"][4]["feature_bits"] = 18
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


def sheen_dict(color, roughness, variant="spec", count=1):
    """Per-pixel sheen arrays of `count` identical pixels (see ref.sheen_arrays)."""
    return {"color": np.tile(np.asarray(color, dtype=np.float64), (count, 1)), "roughness": np.full(count, roughness),
            "variant": variant}


class SheenModel(unittest.TestCase):
    """KHR_materials_sheen (ticket 10) against hand values of the notes (research/khr-materials.md section 3), analytic
    bounds, an independent wi-space grid and the committed table; never against the shader."""

    def setUp(self):
        self.n = np.array([[0.0, 0.0, 1.0]])
        self.v = np.array([unit(0.6, 0.1, 0.5)])
        self.l = np.array([unit(-0.7, 0.2, 0.3)])
        self.base = np.array([[0.1, 0.1, 0.1]])

    def test_alpha_and_roughness_clamp(self):
        self.assertAlmostEqual(float(ref.sheen_alpha(0.5)), 0.25)
        self.assertAlmostEqual(float(ref.sheen_alpha(0.0)), 0.07 ** 2)
        self.assertAlmostEqual(float(ref.sheen_alpha(2.0)), 1.0)
        self.assertAlmostEqual(float(ref.sheen_alpha(0.5, "roughness-linear")), 0.5)
        self.assertEqual(ref.SHEEN_FEATURE_BIT, 4)  # MaterialFeature::Sheen = 1 << 2

    def test_charlie_distribution_hand_values(self):
        # sin(theta_h) = 1 (N.H = 0): D = (2 + 1/alpha) / (2 pi); alpha 0.25 -> 6 / (2 pi).
        self.assertAlmostEqual(float(ref.charlie_d(0.0, 0.25)), 6.0 / (2 * math.pi), places=12)
        # N.H = 0.6: sin = 0.8, D = (2 + 4) 0.8^4 / (2 pi).
        self.assertAlmostEqual(float(ref.charlie_d(0.6, 0.25)), 6.0 * 0.8 ** 4 / (2 * math.pi), places=12)
        # Peaked at the horizon: D at N.H = 1 is 0 (a fuzz lobe), monotonic towards N.H = 0.
        values = ref.charlie_d(np.linspace(0.0, 1.0, 11), 0.25)
        self.assertTrue(np.all(np.diff(values) < 0))
        self.assertEqual(float(values[-1]), 0.0)

    def test_visibility_hand_values(self):
        # alpha_g = 1 ((1 - alpha)^2 = 0: the first coefficient column); Lambda(0.5) = exp(l(0.5)).
        l_half = 21.5473 / (1 + 3.82987 * 0.5 ** 0.19823) + -1.97760 * 0.5 + -4.32054
        self.assertAlmostEqual(float(ref.charlie_lambda(0.5, 1.0)), math.exp(l_half), places=9)
        # cos = 1 (>= 0.5): exp(2 l(0.5) - l(0)), l(0) = a + e.
        l_zero = 21.5473 + -4.32054
        self.assertAlmostEqual(float(ref.charlie_lambda(1.0, 1.0)), math.exp(2 * l_half - l_zero), places=9)
        lam_v, lam_l = float(ref.charlie_lambda(0.5, 1.0)), float(ref.charlie_lambda(0.8, 1.0))
        self.assertAlmostEqual(float(ref.charlie_visibility(0.5, 0.8, 1.0)),
                               1.0 / ((1 + lam_v + lam_l) * 4 * 0.5 * 0.8), places=12)
        # The cheaper Ashikhmin visibility of the notes: 1 / (4 (NL + NV - NL NV)).
        self.assertAlmostEqual(float(ref.charlie_visibility(0.5, 0.8, 1.0, "ashikhmin-visibility")),
                               1.0 / (4 * (0.8 + 0.5 - 0.4)), places=12)
        # Reciprocal in (N.V, N.L).
        self.assertAlmostEqual(float(ref.charlie_visibility(0.3, 0.9, 0.25)),
                               float(ref.charlie_visibility(0.9, 0.3, 0.25)), places=12)

    def test_albedo_matches_an_independent_grid_and_the_committed_table(self):
        # An independent product grid in (cos theta_i, phi_i) with the vectors wo, wi (no shared code with the
        # reference quadrature) and the engine header (tools/sheen_table.py, half-vector quadrature).
        def grid(mu, roughness, n_cos=800, n_phi=400):
            alpha = roughness ** 2
            x, w = np.polynomial.legendre.leggauss(n_cos)
            cos_i, w = 0.5 * (x + 1.0), 0.5 * w
            phi = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
            sin_i = np.sqrt(1 - cos_i ** 2)
            wo = np.array([math.sqrt(1 - mu * mu), 0.0, mu])
            wi = np.stack([sin_i[:, None] * np.cos(phi)[None, :], sin_i[:, None] * np.sin(phi)[None, :],
                           np.broadcast_to(cos_i[:, None], (n_cos, n_phi))], -1)
            h = wi + wo
            h /= np.linalg.norm(h, axis=-1, keepdims=True)
            d = (2 + 1 / alpha) * np.sqrt(1 - h[..., 2] ** 2) ** (1 / alpha) / (2 * math.pi)
            lam = ref.charlie_lambda(cos_i[:, None], alpha)
            v = 1 / ((1 + ref.charlie_lambda(mu, alpha) + lam) * 4 * mu * cos_i[:, None])
            return float(np.sum((d * v * cos_i[:, None]).mean(axis=1) * 2 * math.pi * w))

        for roughness, mu in ((0.3, 0.3), (0.5, 0.5), (0.75, 0.9), (1.0, 0.25), (0.1, 0.2)):
            got = float(ref.sheen_albedo(np.array([mu]), np.array([roughness]))[0])
            self.assertAlmostEqual(got, min(grid(mu, roughness), 1.0), delta=1.5e-3, msg=(roughness, mu))
            table = float(ref.sheen_lookup(np.array([mu]), np.array([roughness]))[0])
            self.assertAlmostEqual(got, table, delta=2.5e-3, msg=(roughness, mu))

    def test_albedo_is_accurate_at_grazing_views_against_a_dense_quadrature(self):
        # Ticket 10: the incident quadrature was off by 0.06 at N.V 0.001 (roughness 0.3).  Threshold fixed beforehand from the
        # independent dense quadrature below (Gauss-Legendre in u with cos_i = u^3 and in w with phi = pi w^4, 700 x 500 nodes;
        # it agrees with 1400 x 1000 nodes to 1e-6 and with a log-spaced midpoint grid of 6000 x 6000 to 1e-7): <= 1e-3.
        def dense(mu, alpha, n_u=700, n_w=500):
            xu, wu = np.polynomial.legendre.leggauss(n_u)
            xw, ww = np.polynomial.legendre.leggauss(n_w)
            u, wu, w, ww = 0.5 * (xu + 1), 0.5 * wu, 0.5 * (xw + 1), 0.5 * ww
            cos_i, dcos = (u ** 3)[:, None], (3 * u ** 2 * wu)[:, None]
            sin_i = np.sqrt(1 - cos_i ** 2)
            phi, dphi = math.pi * w ** 4, (4 * math.pi * w ** 3 * ww)[None, :]
            wd = sin_i * np.cos(phi)[None, :] * math.sqrt(1 - mu * mu) + cos_i * mu
            n_dot_h = np.clip((cos_i + mu) / np.sqrt(np.maximum(2 + 2 * wd, 1e-30)), 0, 1)
            vc = 1 / ((1 + ref.charlie_lambda(mu, alpha) + ref.charlie_lambda(cos_i, alpha)) * 4 * mu)
            return float(np.sum(ref.charlie_d(n_dot_h, alpha) * vc * dcos * dphi) * 2)
        worst = 0.0
        for roughness in (0.3, 0.5, 1.0):
            for mu in (1e-4, 1e-3, 3e-3, 5e-3):
                got = float(ref.sheen_albedo(np.array([mu]), np.array([roughness]))[0])
                worst = max(worst, abs(got - min(dense(mu, roughness ** 2), 1.0)))
        self.assertLess(worst, 1e-3)
        # The value the old quadrature missed (0.81 against 0.871 at N.V 0.001, roughness 0.3).
        self.assertAlmostEqual(float(ref.sheen_albedo(np.array([1e-3]), np.array([0.3]))[0]), 0.871033, delta=1e-3)

    def test_roughness_grid_matches_the_exact_tables_at_off_node_roughness(self):
        # A continuous roughness field (a texture) uses the roughness-node grid.  Threshold fixed beforehand: <= 1e-3 in E against
        # the exact table of the same roughness.
        rng = np.random.default_rng(7)
        mus = np.concatenate([[1e-4, 1e-3, 5e-3, 0.02, 0.1], rng.uniform(0.01, 1.0, 10) ** 2])
        worst = 0.0
        for r in (0.0953, 0.1337, 0.2201, 0.3913, 0.5555, 0.8123, 0.9871):
            exact = ref.sheen_albedo(mus, np.full(len(mus), r))  # one distinct value: the exact table
            field = np.concatenate([np.full(len(mus), r), np.linspace(0.5, 0.6, ref.SHEEN_EXACT_UNIQUE + 4)])
            grid = ref.sheen_albedo(np.concatenate([mus, np.full(ref.SHEEN_EXACT_UNIQUE + 4, 0.5)]), field)[:len(mus)]
            worst = max(worst, float(np.abs(grid - exact).max()))
        self.assertLess(worst, 1e-3)
        # Negative control: a grid with a quarter of the nodes (the pre-fix idea of a coarse table) breaks the bound.
        saved = ref.SHEEN_ROUGHNESS_NODES, dict(ref._SHEEN_GRID)
        try:
            ref.SHEEN_ROUGHNESS_NODES = 8
            ref._SHEEN_GRID.clear()
            field = np.concatenate([np.full(len(mus), 0.0953), np.linspace(0.5, 0.6, 20)])
            coarse = ref.sheen_albedo(np.concatenate([mus, np.full(20, 0.5)]), field)[:len(mus)]
            exact = ref.sheen_albedo(mus, np.full(len(mus), 0.0953))
            self.assertGreater(float(np.abs(coarse - exact).max()), 1e-3)
        finally:
            ref.SHEEN_ROUGHNESS_NODES = saved[0]
            ref._SHEEN_GRID.clear()
            ref._SHEEN_GRID.update(saved[1])

    def test_albedo_is_clamped_and_monotonic_in_the_expected_directions(self):
        mus = np.linspace(0.02, 1.0, 25)
        for roughness in (0.07, 0.3, 1.0):
            e = ref.sheen_albedo(mus, np.full(25, roughness))
            self.assertLessEqual(float(e.max()), ref.SHEEN_E_CLAMP)
            self.assertGreaterEqual(float(e.min()), 0.0)
            self.assertTrue(np.all(np.diff(e) <= 1e-12))  # a fuzz lobe: brighter towards grazing views
        # The table clamps to 1 where the Charlie albedo exceeds it (narrow lobe, grazing view).
        self.assertEqual(float(ref.sheen_albedo(np.array([0.01]), np.array([0.07]))[0]), 1.0)

    def test_declared_sheen_term_follows_the_probe_radiance_and_the_old_behaviour_is_a_control(self):
        # Ticket 10 (decision 06): the sheen environment term is S E L(N) with the radiance of the probe, not of the declared environment.  The control
        # `declared-radiance` lights the term with the declared radiance: here 2.5 times the probe's (0.5 / 0.2), so it must differ by 1.5 S E per unit.
        sheen = sheen_dict([0.8, 0.4, 0.2], 0.5)
        mu = np.array([0.6])
        material = np.zeros((1, 3))
        good = ref.declared_sheen_environment(material, mu, sheen, {"sheen": "spec"})
        saved = ref.SKY_RATIO_CONTROL
        try:
            ref.SKY_RATIO_CONTROL = 2.5
            old = ref.declared_sheen_environment(material, mu, sheen, {"sheen": "declared-radiance"})
        finally:
            ref.SKY_RATIO_CONTROL = saved
        np.testing.assert_allclose(old, 2.5 * good, rtol=1e-12)
        self.assertIn("declared-sheen-declared-radiance", ref.ALL_DECLARED_CONTROLS)

    def test_scaling_forms(self):
        sheen = sheen_dict([1.0, 0.5, 0.25], 0.5)
        e_v, e_l = np.array([0.3]), np.array([0.6])
        # min(1 - max3 E(V), 1 - max3 E(L)) with max3 = 1.
        self.assertAlmostEqual(float(ref.sheen_scaling(sheen, e_v, e_l)[0]), 0.4)
        self.assertAlmostEqual(float(ref.sheen_scaling(sheen, e_l, e_v)[0]), 0.4)  # symmetric in V and L
        self.assertAlmostEqual(float(ref.sheen_scaling(sheen, e_v)[0]), 0.7)         # cheap form (environment)
        dim = sheen_dict([0.2, 0.5, 0.25], 0.5)
        self.assertAlmostEqual(float(ref.sheen_scaling(dim, e_v, e_l)[0]), 1 - 0.5 * 0.6)  # max3 = 0.5
        self.assertAlmostEqual(float(ref.sheen_scaling(sheen_dict([1, 1, 1], 0.5, "unscaled-base"), e_v, e_l)[0]), 1.0)
        self.assertAlmostEqual(float(ref.sheen_scaling(sheen_dict([1, 1, 1], 0.5, "cheap-scaling"), e_v, e_l)[0]), 0.7)

    def test_zero_sheen_colour_is_the_plain_material(self):
        zero = sheen_dict([0, 0, 0], 0.5)
        for metallic in (0.0, 1.0):
            m = np.array([metallic])
            np.testing.assert_allclose(
                ref.bsdf(self.base, m, np.array([0.6]), self.n, self.v, self.l, sheen=zero),
                ref.bsdf(self.base, m, np.array([0.6]), self.n, self.v, self.l), rtol=1e-14)
        ignored = sheen_dict([1, 0.5, 0.25], 0.5, "ignored")
        np.testing.assert_allclose(
            ref.bsdf(self.base, np.array([0.0]), np.array([0.6]), self.n, self.v, self.l, sheen=ignored),
            ref.bsdf(self.base, np.array([0.0]), np.array([0.6]), self.n, self.v, self.l), rtol=1e-14)

    def test_layering_formula_at_one_geometry(self):
        # f = S D V + f_material min(1 - max3 E(V), 1 - max3 E(L)), checked with hand-evaluated D, V, E.
        sheen = sheen_dict([1.0, 0.5, 0.25], 0.5)
        n, v, l = self.n[0], self.v[0], self.l[0]
        h = unit(*(v + l))
        nv, nl, nh = float(v @ n), float(l @ n), float(h @ n)
        d = (2 + 4) * (1 - nh * nh) ** 2 / (2 * math.pi)
        vis = float(ref.charlie_visibility(nv, nl, 0.25))
        e_v = float(ref.sheen_albedo(np.array([nv]), np.array([0.5]))[0])
        e_l = float(ref.sheen_albedo(np.array([nl]), np.array([0.5]))[0])
        scale = min(1 - e_v, 1 - e_l)
        for metallic in (0.0, 1.0):
            m = np.array([metallic])
            material = ref.bsdf(self.base, m, np.array([0.6]), self.n, self.v, self.l)[0]
            got = ref.bsdf(self.base, m, np.array([0.6]), self.n, self.v, self.l, sheen=sheen)[0]
            np.testing.assert_allclose(got, np.array([1.0, 0.5, 0.25]) * d * vis + material * scale, rtol=1e-12)

    def test_light_below_the_horizon_reflects_nothing(self):
        below = np.array([unit(0.0, 0.8, -0.2)])
        got = ref.bsdf(self.base, np.array([0.0]), np.array([0.4]), self.n, self.v, below,
                       sheen=sheen_dict([1, 1, 1], 0.4))
        self.assertTrue(np.all(got == 0))

    def test_directional_albedo_of_the_full_layer_never_exceeds_one(self):
        # White furnace of the BSDF itself: integral of (f_sheen + f_metal scaling) cos over wi <= 1 for a white,
        # rough metal under a white sheen (the base is energy conserving, the scaling removes what the sheen takes).
        # The unscaled control gains energy.  Independent midpoint grid over wi.
        n_t, n_p = 300, 160
        theta = (np.arange(n_t) + 0.5) * (0.5 * math.pi / n_t)
        phi = (np.arange(n_p) + 0.5) * (2 * math.pi / n_p)
        wi = np.stack([np.sin(theta)[:, None] * np.cos(phi)[None, :], np.sin(theta)[:, None] * np.sin(phi)[None, :],
                       np.broadcast_to(np.cos(theta)[:, None], (n_t, n_p))], -1).reshape(-1, 3)
        weight = (np.sin(theta)[:, None] * np.ones((1, n_p)) * (0.5 * math.pi / n_t) * (2 * math.pi / n_p)).reshape(-1)
        white = np.ones((len(wi), 3))
        for roughness in (0.2, 0.5, 1.0):
            for mu in (1.0, 0.6, 0.25):
                v = np.tile(np.array([math.sqrt(1 - mu * mu), 0.0, mu]), (len(wi), 1))
                nn = np.tile(np.array([0.0, 0.0, 1.0]), (len(wi), 1))
                totals = {}
                for variant in ("spec", "unscaled-base"):
                    sheen = sheen_dict([1, 1, 1], roughness, variant, count=len(wi))
                    f = ref.bsdf(white, np.ones(len(wi)), np.full(len(wi), 0.5), nn, v, wi, sheen=sheen)
                    totals[variant] = float(np.sum(f[:, 0] * wi[:, 2] * weight))
                self.assertLessEqual(totals["spec"], 1.0 + 4e-3, (roughness, mu))
                self.assertGreater(totals["unscaled-base"], totals["spec"] + 0.01, (roughness, mu))

    def test_environment_furnace_bound_and_unscaled_control(self):
        # White metal under a white sheen in a uniform environment of 1: never above 1; unscaled gains energy.
        white = np.ones((1, 3))
        for roughness in (0.1, 0.5, 1.0):
            for mu in (1.0, 0.5, 0.2):
                material = ref.environment_radiance(white, np.array([1.0]), np.array([0.5]), np.array([mu]), 1.0)
                out = ref.sheen_environment(material, np.array([mu]), sheen_dict([1, 1, 1], roughness), 1.0)
                self.assertLessEqual(float(out.max()), 1.0 + 1e-3, (roughness, mu))
                gain = ref.sheen_environment(material, np.array([mu]),
                                             sheen_dict([1, 1, 1], roughness, "unscaled-base"), 1.0)
                self.assertGreaterEqual(float(gain.max()), float(out.max()))
        material = ref.environment_radiance(white, np.array([1.0]), np.array([0.5]), np.array([0.3]), 1.0)
        gain = ref.sheen_environment(material, np.array([0.3]), sheen_dict([1, 1, 1], 0.5, "unscaled-base"), 1.0)
        self.assertGreater(float(gain.max()), 1.05)

    def test_exact_environment_formula(self):
        mu, radiance = np.array([0.5]), 0.5
        material = ref.environment_radiance(self.base, np.array([0.0]), np.array([0.7]), mu, radiance)
        sheen = sheen_dict([1.0, 0.5, 0.25], 0.5)
        e = float(ref.sheen_albedo(mu, np.array([0.5]))[0])
        got = ref.sheen_environment(material, mu, sheen, radiance)
        np.testing.assert_allclose(got, material * (1 - e) + np.array([1.0, 0.5, 0.25]) * e * radiance, rtol=1e-12)

    def test_controls_move_the_images_in_the_documented_direction(self):
        # Direct light at a back-lit geometry: ignored < spec (sheen is added), unscaled > spec (base keeps energy),
        # cheap-scaling > spec (its scaling is the larger one when E(N.L) > E(N.V)), Ashikhmin and linear roughness differ.
        colour = [1.0, 0.5, 0.25]
        v = np.array([unit(0.8, 0.0, 0.6)])
        l = np.array([unit(0.9, 0.1, 0.2)])  # same side, N.L 0.2 < N.V 0.6: E(L) > E(V), forward-scatter fuzz

        def f(variant):
            return ref.bsdf(self.base, np.array([0.0]), np.array([0.7]), self.n, v, l,
                            sheen=sheen_dict(colour, 0.5, variant))[0]

        spec = f("spec")
        self.assertTrue(np.all(f("ignored") < spec))
        self.assertTrue(np.all(f("unscaled-base") > spec))
        self.assertTrue(np.all(f("cheap-scaling") > spec))
        self.assertGreater(float(np.abs(f("ashikhmin-visibility") - spec).max() / spec.max()), 0.05)
        self.assertGreater(float(np.abs(f("roughness-linear") - spec).max() / spec.max()), 0.02)
        # With N.V grazing and N.L large the min form picks the view factor: cheap-scaling equals spec.
        v2 = np.array([unit(0.9, 0.0, 0.2)])
        l2 = np.array([unit(0.1, 0.1, 1.0)])
        same = [ref.bsdf(self.base, np.array([0.0]), np.array([0.7]), self.n, v2, l2,
                         sheen=sheen_dict(colour, 0.5, k))[0] for k in ("spec", "cheap-scaling")]
        np.testing.assert_allclose(same[0], same[1], rtol=1e-12)

    def test_sheen_sits_below_the_coat(self):
        # The coat attenuates the whole sheen + material value: f = (1 - c F_c) (f_sheen + f scale) + c F_c D Vis.
        sheen = sheen_dict([1.0, 0.5, 0.25], 0.5)
        coat = coat_dict(0.6, 0.3)
        m, rough = np.array([0.0]), np.array([0.6])
        layered = ref.bsdf(self.base, m, rough, self.n, self.v, self.l, sheen=sheen)[0]
        attenuation, coat_value = ref.coat_lobe(coat, self.n, self.v, self.l)
        got = ref.bsdf(self.base, m, rough, self.n, self.v, self.l, coat=coat, sheen=sheen)[0]
        np.testing.assert_allclose(got, attenuation[0] * layered + coat_value[0], rtol=1e-12)
        # Exact environment: the coat form of the sheen-included material.
        mu = np.array([0.5])
        material = ref.sheen_environment(ref.environment_radiance(self.base, m, rough, mu, 0.5), mu, sheen, 0.5)
        fc = 0.04 + 0.96 * 0.5 ** 5
        albedo = float(ref.coat_albedo(mu, np.array([0.3]))[0][0])
        np.testing.assert_allclose(ref.environment_coat(material, mu, coat, 0.5)[0],
                                   (1 - 0.6 * fc) * material[0] + 0.6 * fc * albedo * 0.5, rtol=1e-12)

    def test_sun_disk_point_evaluation_matches_the_disk_average(self):
        # The wide sheen lobe is point-evaluated at the disk centre; the average of the full BSDF over a 4000-sample
        # disk agrees within 0.5 % (sheen roughness 0.1 is the narrowest fixture lobe, scaling and lobe vary slowly).
        radius = 0.00465
        toward = unit(-0.8, 0.35, -0.3)
        n = np.array([unit(-0.55, 0.25, 0.8)])
        v = np.array([[0.0, 0.0, 1.0]])
        for roughness in (0.1, 0.5):
            sheen = sheen_dict([1.0, 0.5, 0.25], roughness)
            got, narrow = ref.sun_radiance(self.base, np.array([0.0]), np.array([0.7]), n, v, toward, np.ones(3), radius,
                                           sheen=sheen)
            self.assertFalse(bool(narrow[0]))
            total = np.zeros(3)
            samples = ref.sun_disk_samples(toward, radius, 4000)
            for sample in samples:
                lk = np.broadcast_to(sample, (1, 3))
                total += ref.bsdf(self.base, np.array([0.0]), np.array([0.7]), n, v, lk, sheen=sheen)[0] * \
                    max(0.0, float(n[0] @ sample))
            total /= len(samples)
            np.testing.assert_allclose(got[0], total, rtol=5e-3)
            # A zero sheen colour reduces the sun to the plain material.
            plain, _ = ref.sun_radiance(self.base, np.array([0.0]), np.array([0.7]), n, v, toward, np.ones(3), radius)
            none, _ = ref.sun_radiance(self.base, np.array([0.0]), np.array([0.7]), n, v, toward, np.ones(3), radius,
                                       sheen=sheen_dict([0, 0, 0], roughness))
            np.testing.assert_allclose(plain, none, rtol=1e-14)

    def test_point_light_uses_the_sheen_bsdf_at_the_light_direction(self):
        light = {"position": [-1.5, 0.3, 0.4], "intensity": 50.0, "range": 100.0, "color": [1.0, 1.0, 1.0]}
        sheen = sheen_dict([1.0, 0.5, 0.25], 0.3)
        got, lit = ref.point_radiance(self.base, np.array([0.0]), np.array([0.7]), self.n, self.v, np.zeros((1, 3)),
                                      light, sheen=sheen)
        delta = np.array(light["position"])
        l = (delta / np.linalg.norm(delta))[None, :]
        incident = 50.0 * float(ref.point_window(np.linalg.norm(delta), 100.0)) / float(delta @ delta)
        expected = ref.bsdf(self.base, np.array([0.0]), np.array([0.7]), self.n, self.v, l, sheen=sheen)[0] * \
            float(l[0] @ self.n[0]) * incident
        np.testing.assert_allclose(got[0], expected, rtol=1e-12)
        self.assertTrue(bool(lit[0]))

    def test_declared_environment_hand_value_and_controls(self):
        mu, rough = np.array([0.5]), np.array([0.7])
        sheen = sheen_dict([1.0, 0.5, 0.25], 0.5)
        plain = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0)
        got = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 2.0, sheen=sheen)
        e = float(ref.sheen_lookup(mu, np.array([0.5]))[0])
        np.testing.assert_allclose(got, (plain * (1 - e) + np.array([1.0, 0.5, 0.25]) * e) * 2.0, rtol=1e-12)
        good = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0, sheen=sheen)
        saved = ref.SKY_RATIO_CONTROL
        try:
            ref.SKY_RATIO_CONTROL = 2.5  # the probe fixture's declared / probe radiance ratio (the declared-radiance control needs a ratio != 1)
            for name, variant in ref.SHEEN_DECLARED_CONTROLS.items():
                wrong = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0, variant, sheen=sheen)
                self.assertGreater(float(np.abs(wrong - good).max() / good.max()), 0.01, name)
        finally:
            ref.SKY_RATIO_CONTROL = saved
        ignored = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0,
                                                    ref.SHEEN_DECLARED_CONTROLS["declared-sheen-ignored"], sheen=sheen)
        np.testing.assert_allclose(ignored, plain, rtol=1e-14)
        with self.assertRaises(ValueError):  # no table exists for the Ashikhmin visibility
            ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0,
                                              {**ref.DECLARED, "sheen": "ashikhmin-visibility"}, sheen=sheen)
        # Coat above the sheen in the declared model: (1 - c F_c) declared_sheen_material + c F_c clamp(A_c + B_c).
        coat = coat_dict(0.5, 0.4)
        coated = ref.declared_environment_radiance(self.base, np.array([0.0]), rough, mu, 1.0, coat=coat, sheen=sheen)
        a, b = ref.dfg_lookup(mu, ref.clamp_roughness(np.array([0.4])))
        fc = 0.04 + 0.96 * 0.5 ** 5
        np.testing.assert_allclose(coated, (1 - 0.5 * fc) * good + 0.5 * fc * float(np.clip(a + b, 0, 1)[0]), rtol=1e-12)

    def test_declared_and_exact_sheen_terms_agree(self):
        # The same sheen layer over a black base: the declared model (engine table) equals the quadrature within 2.5e-3.
        zero = np.zeros((1, 3))
        for roughness in (0.1, 0.5, 1.0):
            for mu in (0.9, 0.5, 0.25):
                sheen = sheen_dict([1.0, 0.5, 0.25], roughness)
                declared = ref.declared_sheen_environment(zero, np.array([mu]), sheen, ref.DECLARED)
                exact = ref.sheen_environment(zero, np.array([mu]), sheen, 1.0)
                np.testing.assert_allclose(declared, exact, atol=2.5e-3, err_msg=str((roughness, mu)))

    def test_sheen_lookup_rule_on_a_synthetic_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sheen_table.h"
            values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3, 0.5]  # rows = roughness 0.07, 0.535, 1; columns = N.V 0, 0.25, 1 (quadratic axis)
            path.write_text("PHOSPHOR_GPU_CONSTANT u32 SheenTableSize = 3;\nPHOSPHOR_GPU_CONSTANT float SheenTable[9] = {\n    " +
                            ", ".join(f"{v:.8e}f" for v in values) + "\n};\n")
            self.assertEqual(ref.load_sheen_table(path)[0], 3)
            self.assertAlmostEqual(float(ref.sheen_lookup(np.array([1.0]), np.array([0.535]), path)[0]), 1.0)
            self.assertAlmostEqual(float(ref.sheen_lookup(np.array([0.25]), np.array([0.0]), path)[0]), 0.2)  # sqrt(0.25) = column 1; roughness clamps to 0.07 = row 0
            x =math.sqrt(0.5) * 2.0 - 1.0  # N.V 0.5: between columns 1 and 2
            fy = (0.30 - 0.07) / 0.93 * 2.0
            self.assertAlmostEqual(float(ref.sheen_lookup(np.array([0.5]), np.array([0.30]), path)[0]),
                                   (0.2 + 0.2 * x) * (1 - fy) + (0.8 + 0.2 * x) * fy)
            unclamped = pathlib.Path(directory) / "sheen_table_unclamped.h"  # (the loader caches by path)
            unclamped.write_text("PHOSPHOR_GPU_CONSTANT u32 SheenTableSize = 2;\nPHOSPHOR_GPU_CONSTANT float SheenTable[4] = {\n    "
                                 "1.6e+00f, 1.4e+00f, 1.2e+00f, 0.4e+00f\n};\n")  # unclamped values: the result is clamped AFTER interpolating
            self.assertAlmostEqual(float(ref.sheen_lookup(np.array([0.0]), np.array([0.0]), unclamped)[0]), 1.0)
            self.assertAlmostEqual(float(ref.sheen_lookup(np.array([1.0]), np.array([1.0]), unclamped)[0]), 0.4)
            self.assertAlmostEqual(float(ref.sheen_lookup(np.array([0.25]), np.array([0.535]), unclamped)[0]), 1.0)  # (1.6+1.4+1.2+0.4)/4 = 1.15 -> 1
            clamped = ref.sheen_lookup(np.array([-0.5, 2.0]), np.array([3.0, -1.0]), path)  # clamp like the C++ lookup
            self.assertAlmostEqual(float(clamped[0]), 0.1)   # row 2 (roughness 1), column 0
            self.assertAlmostEqual(float(clamped[1]), 0.4)   # row 0, column 2

    def test_record_word_decoding_and_texture_refusal(self):
        def words(color, roughness, textures=(ref.INVALID, ref.INVALID)):
            f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731
            w = [0] * 20
            w[5] = w[6] = ref.INVALID
            w[10:13] = [ref.INVALID] * 3
            w[13:16] = [f(c) for c in color]
            w[16] = f(roughness)
            w[17:19] = textures
            return w

        zero = [[0] * 20] * 3
        data = {"materials": [{"feature_bits": 4, "extension_index": 0}, {"feature_bits": 0},
                              {"feature_bits": 4, "extension_index": 4}, {"feature_bits": 4, "extension_index": 8},
                              {"feature_bits": 20, "extension_index": 12}, {"feature_bits": 6, "extension_index": 16}],
                "material_extensions": {"version": 1, "records_per_material": 4,
                                        "records": [words((1.0, 0.5, 0.25), 0.3)] + zero +
                                                   [words((1.0, 0.5, 0.25), 0.3, (3, ref.INVALID))] + zero +
                                                   [words((1.0, 0.5, 0.25), 0.3, (ref.INVALID, 9))] + zero +
                                                   [words((1.0, 0.5, 0.25), 0.3)] + zero +
                                                   [words((0.5, 0.5, 0.5), 0.75)] + zero}}
        color, roughness = ref.material_sheen(data, data["materials"][0])
        self.assertEqual((color, round(roughness, 6)), ([1.0, 0.5, 0.25], 0.3))
        self.assertIsNone(ref.material_sheen(data, data["materials"][1]))
        for index in (2, 3):  # sheenColorTexture (word 17) or sheenRoughnessTexture (word 18)
            with self.assertRaises(ValueError):
                ref.material_sheen(data, data["materials"][index])
        # ticket 19: sheen + iridescence (feature bits 20) is composed; an unknown bit is still refused
        self.assertEqual(ref.material_sheen(data, data["materials"][4])[0], [1.0, 0.5, 0.25])
        data["materials"][4]["feature_bits"] = 4 | 32
        with self.assertRaises(ValueError):
            ref.material_sheen(data, data["materials"][4])
        data["materials"][4]["feature_bits"] = 20
        self.assertEqual(ref.material_sheen(data, data["materials"][5])[0], [0.5, 0.5, 0.5])  # coat + sheen is inside it
        self.assertTrue(ref.uses_sheen(data))
        self.assertIsNone(ref.material_specular(data, data["materials"][0]))  # sheen-only: neutral specular block
        self.assertIsNone(ref.material_clearcoat(data, data["materials"][0]))


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba not installed")
class SheenSnapshot(unittest.TestCase):
    """A one-sphere snapshot whose material carries KHR_materials_sheen in record 0 (the exporter's keys)."""

    def sheen_snapshot(self, directory, sky, sun, colour, roughness, base=(0.1, 0.1, 0.1), base_roughness=0.7):
        root = RayCastedSnapshot("snapshot").snapshot(directory, sky, sun, 0.0, base_roughness, list(base))
        path = pathlib.Path(root) / "scene.json"
        data = json.loads(path.read_text())
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731
        record0 = [0] * 20
        for k in (5, 6, 10, 11, 12, 17, 18):
            record0[k] = ref.INVALID
        record0[13:16] = [f(c) for c in colour]
        record0[16] = f(roughness)
        data["materials"][0].update(feature_bits=4, extension_index=0, flags=256)
        data["material_extensions"] = {"version": 1, "records_per_material": 4,
                                       "records": [record0, [0] * 20, [0] * 20, [0] * 20]}
        path.write_text(json.dumps(data))
        return root

    def test_render_sun_environment_and_controls(self):
        from f12_reference import snapshot as load
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root = self.sheen_snapshot(tmp, 0.5, (toward, 3.0), (1.0, 0.5, 0.25), 0.5)
            root, data = load(root)
            self.assertTrue(ref.uses_sheen(data) and not ref.uses_specular_extension(data))
            cast = ref.cast_primary_rays(root, data, 2)
            images = {}
            for name in ("spec", "ignored", "unscaled-base", "ashikhmin-visibility", "roughness-linear", "cheap-scaling"):
                model = ref.control_model("sheen-" + name, ref.APPENDIX_B) if name != "spec" else ref.APPENDIX_B
                image, masks, spheres, terms = ref.render_frame(root, data, model, cast=cast)
                images[name] = image
            self.assertEqual(terms, ["sun-disk", "environment-quadrature"])
            self.assertEqual(spheres[0]["sheen_color"], [1.0, 0.5, 0.25])
            self.assertAlmostEqual(spheres[0]["sheen_roughness"], 0.5, places=6)
            self.assertEqual(spheres[0]["group"], "metallic 0 sheen 1,0.5,0.25")
            roi = masks["roi"]
            for name in ("ignored", "unscaled-base", "ashikhmin-visibility", "roughness-linear", "cheap-scaling"):
                difference = np.abs(images[name][roi] - images["spec"][roi]).max()
                self.assertGreater(float(difference), 1e-3, name)
            # The declared model on the same cast is a split-sum model: close to the exact image, not equal to it.
            declared, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast, environment=ref.DECLARED)
            self.assertLess(float(np.abs(declared[roi] - images["spec"][roi]).max() / images["spec"][roi].max()), 0.08)

    def test_command_writes_controls_and_declared_controls(self):
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root = self.sheen_snapshot(tmp, 0.5, (toward, 3.0), (1.0, 0.5, 0.25), 0.5)
            out = pathlib.Path(tmp) / "out"
            code = ref.main(["render", str(root), "--out", str(out), "--controls", "sheen-ignored",
                             "sheen-cheap-scaling", "--declared-controls", "declared-sheen-ignored",
                             "declared-sheen-unscaled-base"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertEqual(sorted(report["controls"]), ["sheen-cheap-scaling", "sheen-ignored"])
            self.assertEqual(report["controls"]["sheen-ignored"]["model_variant"]["sheen"], "ignored")
            self.assertEqual(sorted(report["declared"]["controls"]),
                             ["declared-sheen-ignored", "declared-sheen-unscaled-base"])
            self.assertIn("sheen", report["model"])

    def test_textured_sheen_is_refused(self):
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            root = self.sheen_snapshot(tmp, 0.5, None, (1.0, 1.0, 1.0), 0.5)
            path = pathlib.Path(root) / "scene.json"
            data = json.loads(path.read_text())
            data["material_extensions"]["records"][0][17] = 3
            path.write_text(json.dumps(data))
            root, data = load(root)
            with self.assertRaises(ValueError):
                ref.require_declared_scene(data)
            self.assertEqual(ref.main(["render", str(root), "--out", str(pathlib.Path(tmp) / "out")]), 2)


# --- KHR_materials_anisotropy (F15.3, ticket 11) -----------------------------------------------------------------------

def aniso_pixels(strength, rotation, tangent=(1.0, 0.0, 0.0), w=1.0, det=1.0, variant="spec", count=1, phi=0.3):
    """Per-pixel anisotropy arrays of `count` identical pixels (see ref.aniso_arrays); `phi` is the azimuth of the view in the lobe frame that the
    declared environment (anisotropic albedo table, ticket 11 round 3) reads (render_frame sets it from `aniso_environment_inputs`)."""
    return {"strength": np.full(count, float(strength)), "rotation": np.full(count, float(rotation)),
            "tangent": np.tile(np.asarray(tangent, dtype=np.float64), (count, 1)), "w": np.full(count, float(w)),
            "det": np.full(count, float(det)), "variant": variant, "phi": np.full(count, float(phi))}


def readme_d(n_dot_h, t_dot_h, b_dot_h, at, ab):
    """The README's GLSL D_GGX_anisotropic, written from its text (a different expression than the Burley form)."""
    a2 = at * ab
    f = np.array([ab * t_dot_h, at * b_dot_h, a2 * n_dot_h])
    w2 = a2 / float(f @ f)
    return a2 * w2 * w2 / math.pi


def readme_v(n_dot_l, n_dot_v, b_dot_v, t_dot_v, t_dot_l, b_dot_l, at, ab):
    """The README's GLSL V_GGX_anisotropic without the final clamp."""
    ggxv = n_dot_l * np.linalg.norm([at * t_dot_v, ab * b_dot_v, n_dot_v])
    ggxl = n_dot_v * np.linalg.norm([at * t_dot_l, ab * b_dot_l, n_dot_l])
    return 0.5 / (ggxv + ggxl)


class AnisotropyModel(unittest.TestCase):
    N = np.array([[0.0, 0.0, 1.0]])

    def axes(self, rotation, **kw):
        t, b = ref.aniso_axes(aniso_pixels(1.0, rotation, **kw), self.N)
        return t[0], b[0]

    def test_alpha_rule_is_the_normative_one(self):
        at, ab = ref.aniso_alphas(np.array([0.09]), np.array([0.5]))
        self.assertAlmostEqual(float(at[0]), 0.09 + 0.91 * 0.25, places=12)  # mix(alpha, 1, s^2)
        self.assertAlmostEqual(float(ab[0]), 0.09, places=12)
        at, ab = ref.aniso_alphas(np.array([0.09]), np.array([1.0]))
        self.assertAlmostEqual(float(at[0]), 1.0)
        at, ab = ref.aniso_alphas(np.array([0.09]), np.array([0.0]))
        self.assertEqual((float(at[0]), float(ab[0])), (0.09, 0.09))
        # Controls: ignored forces strength 0, swapped exchanges the two.
        at, ab = ref.aniso_alphas(np.array([0.09]), np.array([1.0]), "ignored")
        self.assertEqual((float(at[0]), float(ab[0])), (0.09, 0.09))
        at, ab = ref.aniso_alphas(np.array([0.09]), np.array([1.0]), "alpha-swapped")
        self.assertEqual((float(at[0]), float(ab[0])), (0.09, 1.0))

    def test_rotated_frame_hand_values(self):
        s = math.sqrt(0.5)
        # glTF bitangent = cross(N, T) w: with N = z, T = x, w = +1 it is +y, with w = -1 it is -y.
        t, b = self.axes(math.pi / 4, w=1.0)
        np.testing.assert_allclose(t, [s, s, 0], atol=1e-12)          # counter-clockwise from T towards B = +y
        np.testing.assert_allclose(b, [-s, s, 0], atol=1e-12)         # cross(N, T')
        t, b = self.axes(math.pi / 4, w=-1.0)
        np.testing.assert_allclose(t, [s, -s, 0], atol=1e-12)         # B = -y: the lobe turns the other way in world space
        np.testing.assert_allclose(b, [s, s, 0], atol=1e-12)
        t, _ = self.axes(math.pi / 2, w=1.0)
        np.testing.assert_allclose(t, [0, 1, 0], atol=1e-12)
        t, _ = self.axes(0.0, w=-1.0)
        np.testing.assert_allclose(t, [1, 0, 0], atol=1e-12)
        t, _ = self.axes(-math.pi / 4, w=1.0)                          # a negative rotation is clockwise
        np.testing.assert_allclose(t, [s, -s, 0], atol=1e-12)
        # A mirrored instance (det -1) carries the world image of the model-space bitangent: w = -1, det = -1 behaves like
        # w = +1, det = +1 on the world tangent/normal (the model-space frame is mirrored together with the surface).
        t, _ = self.axes(math.pi / 4, w=-1.0, det=-1.0)
        np.testing.assert_allclose(t, [s, s, 0], atol=1e-12)

    def test_controls_move_the_frame_as_documented(self):
        s = math.sqrt(0.5)
        for variant, expected in (("rotation-flipped", [s, -s, 0]),    # angle negated (w = +1)
                                  ("bitangent-flipped", [s, s, 0]),    # w ignored: with w = +1 it is the spec frame
                                  ("mirror-unaware", [s, s, 0])):      # det ignored: with det = 1 it is the spec frame
            t, _ = self.axes(math.pi / 4, w=1.0, variant=variant)
            np.testing.assert_allclose(t, [s, -s, 0] if variant == "rotation-flipped" else [s, s, 0], atol=1e-12, err_msg=variant)
        t, _ = self.axes(math.pi / 4, w=-1.0, variant="bitangent-flipped")  # w ignored: B = +y although w = -1
        np.testing.assert_allclose(t, [s, s, 0], atol=1e-12)
        t, _ = self.axes(math.pi / 4, w=-1.0, det=-1.0, variant="mirror-unaware")  # det ignored: B = cross(N, T) w = -y
        np.testing.assert_allclose(t, [s, -s, 0], atol=1e-12)
        # The lobe is even in T' and B': the rotation sign does not matter at 0 and pi/2.
        for rotation in (0.0, math.pi / 2):
            spec, _ = self.axes(rotation, w=1.0)
            flipped, _ = self.axes(rotation, w=1.0, variant="rotation-flipped")
            self.assertAlmostEqual(abs(float(spec @ flipped)), 1.0, places=12)

    def geometry(self):
        v = unit(0.3, -0.2, 0.9)
        l = unit(-0.4, 0.35, 0.8)
        return v, l

    def test_burley_d_and_the_readme_v_agree_with_the_readme_formulas(self):
        v, l = self.geometry()
        rotation = 0.6
        a = aniso_pixels(0.7, rotation, w=1.0)
        alpha = 0.2
        at, ab = ref.aniso_alphas(np.array([alpha]), np.array([0.7]))
        t, b = ref.aniso_axes(a, self.N)
        h = unit(*(v + l))
        d_readme = readme_d(h[2], float(h @ t[0]), float(h @ b[0]), float(at[0]), float(ab[0]))
        v_readme = readme_v(l[2], v[2], float(v @ b[0]), float(v @ t[0]), float(l @ t[0]), float(l @ b[0]),
                            float(at[0]), float(ab[0]))
        got = float(ref.aniso_dv(self.N, v[None], l[None], np.array([alpha]), a)[0])
        self.assertAlmostEqual(got / (d_readme * v_readme), 1.0, places=10)

    def test_isotropic_limit_equals_the_base_model_exactly(self):
        rng = np.random.default_rng(3)
        count = 300
        n = np.tile([0.0, 0.0, 1.0], (count, 1))
        v = ref.normalize(rng.normal(size=(count, 3)) * [1, 1, 0.3] + [0, 0, 2])
        l = ref.normalize(rng.normal(size=(count, 3)) * [1, 1, 0.3] + [0, 0, 2])
        for alpha in (0.002, 0.09, 0.5, 1.0):
            a = np.full(count, alpha)
            iso = ref.specular_dv(n, v, l, a)
            for rotation in (0.0, 0.7):
                got = ref.aniso_dv(n, v, l, a, aniso_pixels(0.0, rotation, count=count))
                np.testing.assert_allclose(got, iso, rtol=1e-12)
            # strength 1 with alpha = 1 is isotropic too (alpha_t = alpha_b = 1)
        got = ref.aniso_dv(n, v, l, np.full(count, 1.0), aniso_pixels(1.0, 0.4, count=count))
        np.testing.assert_allclose(got, ref.specular_dv(n, v, l, np.full(count, 1.0)), rtol=1e-12)
        # The control `ignored` is the isotropic model for any strength, and the whole BSDF follows.
        base, metallic, rough = np.tile([0.8, 0.35, 0.15], (count, 1)), np.full(count, 0.5), np.full(count, 0.3)
        plain = ref.bsdf(base, metallic, rough, n, v, l)
        blind = ref.bsdf(base, metallic, rough, n, v, l, aniso=aniso_pixels(1.0, 0.7, variant="ignored", count=count))
        np.testing.assert_allclose(blind, plain, rtol=1e-12)
        zero = ref.bsdf(base, metallic, rough, n, v, l, aniso=aniso_pixels(0.0, 0.7, count=count))
        np.testing.assert_allclose(zero, plain, rtol=1e-12)
        stretched = ref.bsdf(base, metallic, rough, n, v, l, aniso=aniso_pixels(1.0, 0.7, count=count))
        self.assertGreater(float(np.abs(stretched - plain).max()), 1e-3)

    def test_d_is_normalised_over_the_hemisphere_by_an_independent_quadrature(self):
        # integral of D(h) cos(theta_h) over the hemisphere of half vectors is 1, whatever alpha_t, alpha_b.
        n_theta, n_phi = 4000, 720
        theta = (np.arange(n_theta) + 0.5) * (0.5 * math.pi / n_theta)
        phi = (np.arange(n_phi) + 0.5) * (2 * math.pi / n_phi)
        cos_t, sin_t = np.cos(theta)[:, None], np.sin(theta)[:, None]
        for at, ab in ((0.3175, 0.09), (1.0, 0.09), (0.5, 0.2)):
            q = (sin_t * np.cos(phi)[None, :] / at) ** 2 + (sin_t * np.sin(phi)[None, :] / ab) ** 2 + cos_t ** 2
            d = 1.0 / (math.pi * at * ab * q * q)
            total = float(np.sum(d * cos_t * sin_t) * (0.5 * math.pi / n_theta) * (2 * math.pi / n_phi))
            self.assertAlmostEqual(total, 1.0, delta=2e-3, msg=(at, ab))
        # The same through the function under test (h in the frame t = x, b = y, n = z).
        a = aniso_pixels(0.5, 0.0, w=1.0)
        alpha = np.array([0.09])
        h_theta = 0.2
        h = np.array([[math.sin(h_theta) * math.cos(0.7), math.sin(h_theta) * math.sin(0.7), math.cos(h_theta)]])
        # D V with V cancelled: compare through a direction pair whose half vector is h and whose V is evaluated both ways.
        v = unit(0.1, 0.2, 0.9)
        l = 2.0 * float(v @ h[0]) * h[0] - v
        at, ab = ref.aniso_alphas(alpha, np.array([0.5]))
        got = float(ref.aniso_dv(self.N, v[None], l[None], alpha, a)[0])
        d = readme_d(h[0, 2], h[0, 0], h[0, 1], float(at[0]), float(ab[0]))
        vis = readme_v(l[2], v[2], v[1], v[0], l[0], l[1], float(at[0]), float(ab[0]))
        self.assertAlmostEqual(got / (d * vis), 1.0, places=9)

    def mc_albedo(self, at, ab, mu, phi, count=400000, seed=11):
        """Independent Monte Carlo of the integral of D V cos over wi: cosine-hemisphere samples, written in wi space
        (the reference integrates in half-vector space); returns (estimate, standard error)."""
        rng = np.random.default_rng(seed)
        r, a = np.sqrt(rng.random(count)), 2 * math.pi * rng.random(count)
        wi = np.stack([r * np.cos(a), r * np.sin(a), np.sqrt(np.maximum(0.0, 1.0 - r * r))], axis=1)
        sin_v = math.sqrt(1 - mu * mu)
        wo = np.array([sin_v * math.cos(phi), sin_v * math.sin(phi), mu])
        h = wi + wo
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        q = (h[:, 0] / at) ** 2 + (h[:, 1] / ab) ** 2 + h[:, 2] ** 2
        d = 1.0 / (math.pi * at * ab * q * q)
        ggxv = wi[:, 2] * np.sqrt((at * wo[0]) ** 2 + (ab * wo[1]) ** 2 + mu * mu)
        ggxl = mu * np.sqrt((at * wi[:, 0]) ** 2 + (ab * wi[:, 1]) ** 2 + wi[:, 2] ** 2)
        samples = math.pi * d * 0.5 / (ggxv + ggxl)  # D V cos / pdf, pdf = cos / pi
        return float(samples.mean()), float(samples.std() / math.sqrt(count))

    def test_directional_albedo_table_matches_an_independent_monte_carlo_and_never_exceeds_one(self):
        for strength in (0.5, 1.0):
            alpha = 0.09
            at, ab = float(ref.aniso_alphas(np.array([alpha]), np.array([strength]))[0][0]), alpha
            mus, phis, a, b = ref.aniso_albedo_table(at, ab)
            albedo = a + b  # Fresnel 1: the albedo of D V cos
            self.assertLessEqual(float(albedo.max()), 1.0 + 1e-3, strength)       # white furnace: no energy gain
            self.assertGreater(float(albedo.min()), 0.0)
            for mu, phi in ((0.5, 0.0), (0.5, math.pi / 2), (0.9, math.pi / 4), (0.2, math.pi / 3), (1.0, 0.3)):
                estimate, error = self.mc_albedo(at, ab, mu, phi)
                i, j = int(np.argmin(abs(mus - mu))), int(np.argmin(abs(phis - phi)))
                got = float(ref._bilinear(mus, phis, albedo, np.array([mus[i]]), np.array([phis[j]]))[0])
                self.assertAlmostEqual(got, estimate, delta=6 * error + 2e-3, msg=(strength, mu, phi))
        # Rougher everywhere: still bounded.
        mus, phis, a, b = ref.aniso_albedo_table(1.0, 0.5)
        self.assertLessEqual(float((a + b).max()), 1.0 + 1e-3)

    def test_environment_albedo_is_symmetric_and_isotropic_pixels_use_the_base_table(self):
        # Pixel with alpha_t = alpha_b: the base table of environment_table, exactly.
        env = {"at": np.array([0.09]), "ab": np.array([0.09]), "phi": np.array([0.3]), "variant": "spec"}
        a, b = ref.aniso_spec_albedo(np.array([0.5]), env)
        t = ref.environment_table(0.3)
        self.assertEqual(float(a[0]), float(np.interp(0.5, t["mu"], t["A"])))
        self.assertEqual(float(b[0]), float(np.interp(0.5, t["mu"], t["B"])))
        # The anisotropic quadrature at alpha_t = alpha_b (a direct call) reproduces the base quadrature of the same lobe.
        a1, b1 = ref._aniso_spec_integrals(0.5, 0.7, 0.09, 0.09)
        self.assertAlmostEqual(a1 + b1, float(np.interp(0.5, t["mu"], t["A"] + t["B"])), delta=2e-3)
        # Pixel azimuth: 0 and pi/2 along the lobe axes, symmetric under swapping the axes together with the azimuth.
        a_x, _ = ref._aniso_spec_integrals(0.6, 0.0, 0.5, 0.1)
        a_y, _ = ref._aniso_spec_integrals(0.6, math.pi / 2, 0.1, 0.5)
        self.assertAlmostEqual(a_x, a_y, delta=2e-3)
        # The view azimuth enters: a stretched lobe seen along vs across the stretch differs.
        a_along, b_along = ref._aniso_spec_integrals(0.4, 0.0, 1.0, 0.09)
        a_across, b_across = ref._aniso_spec_integrals(0.4, math.pi / 2, 1.0, 0.09)
        self.assertGreater(abs((a_along + b_along) - (a_across + b_across)), 0.02)

    def test_energy_gaining_control_exceeds_the_furnace_bound(self):
        at, ab = 1.0, 0.09
        a, b = ref._aniso_spec_integrals(0.8, 0.5, at, ab, "d-normalised-by-ab")
        self.assertGreater(a + b, 1.5)                       # D prefactor 1/(pi alpha_b^2): about alpha_t / alpha_b too much
        a, b = ref._aniso_spec_integrals(0.8, 0.5, at, ab)
        self.assertLess(a + b, 1.0)

    def test_sun_disk_is_point_evaluated_for_the_wide_anisotropic_lobe(self):
        # alpha_b 0.09 (>= 0.05): count 1, no quadrature mask.  The lobe at the disk centre equals the disk average within 1 %.
        radius = 0.00465
        v = unit(0.25, 0.1, 0.9)
        toward = unit(-0.45, 0.15, 0.85)  # near the mirror direction of v: the highlight region
        a = aniso_pixels(1.0, math.pi / 4, w=1.0)
        alpha = np.array([0.09])
        count, narrow = ref.sun_quadrature(self.N, v[None], toward, np.array([0.3]), radius)
        self.assertEqual((int(count[0]), bool(narrow[0])), (1, False))
        centre = float(ref.aniso_dv(self.N, v[None], toward[None], alpha, a)[0])
        samples = ref.sun_disk_samples(toward, radius, 4000)
        values = ref.aniso_dv(np.tile(self.N, (4000, 1)), np.tile(v, (4000, 1)), samples, np.full(4000, 0.09),
                              aniso_pixels(1.0, math.pi / 4, w=1.0, count=4000))
        cosines = np.clip(samples[:, 2], 0, 1)
        average = float((values * cosines).mean()) / float(max(toward[2], 1e-9))
        self.assertAlmostEqual(average / centre, 1.0, delta=0.01)

    def test_sun_radiance_matches_the_bsdf_at_the_disk_centre_for_a_wide_lobe(self):
        toward = unit(-0.45, 0.15, 0.85)
        v = unit(0.25, 0.1, 0.9)
        base, metallic, rough = np.array([[0.8, 0.35, 0.15]]), np.array([1.0]), np.array([0.3])
        a = aniso_pixels(1.0, math.pi / 4, w=1.0)
        radiance, narrow = ref.sun_radiance(base, metallic, rough, self.N, v[None], toward, np.array([3.0] * 3), 0.00465,
                                            aniso=a)
        expected = ref.bsdf(base, metallic, rough, self.N, v[None], toward[None], aniso=a) * toward[2] * 3.0
        np.testing.assert_allclose(radiance, expected, rtol=1e-12)
        self.assertFalse(bool(narrow[0]))

    def test_point_light_uses_the_same_lobe(self):
        light = {"position": [0.0, 0.0, 3.0], "color": [1, 1, 1], "intensity": 9.0, "range": 100.0}
        v = unit(0.3, 0.1, 0.9)
        base, metallic, rough = np.array([[0.8, 0.35, 0.15]]), np.array([1.0]), np.array([0.3])
        a = aniso_pixels(1.0, math.pi / 4, w=1.0)
        value, lit = ref.point_radiance(base, metallic, rough, self.N, v[None], np.array([[0.4, 0.0, 0.0]]), light, aniso=a)
        l = unit(-0.4, 0.0, 3.0)
        window = float(ref.point_window(np.linalg.norm([-0.4, 0.0, 3.0]), 100.0))
        expected = ref.bsdf(base, metallic, rough, self.N, v[None], l[None], aniso=a)[0] * l[2] * 9.0 * window / (0.16 + 9.0)
        np.testing.assert_allclose(value[0], expected, rtol=1e-10)
        self.assertTrue(bool(lit[0]))

    def test_exact_environment_is_below_the_isotropic_one_and_energy_bounded(self):
        # White metal under radiance 1: radiance = albedo <= 1; a stretched lobe loses energy against the isotropic one.
        mu = np.array([0.7, 0.7])
        base, metallic, rough = np.ones((2, 3)), np.ones(2), np.full(2, 0.3)
        env = {"at": np.array([1.0, 0.09]), "ab": np.array([0.09, 0.09]), "phi": np.array([0.4, 0.4]), "variant": "spec"}
        out = ref.environment_radiance(base, metallic, rough, mu, 1.0, aniso_env=env)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-3)
        self.assertLess(float(out[0, 0]), float(out[1, 0]) - 0.05)
        iso = ref.environment_radiance(base[1:], metallic[1:], rough[1:], mu[1:], 1.0)
        np.testing.assert_allclose(out[1], iso[0], rtol=1e-12)

    def test_declared_environment_fit_and_its_controls(self):
        at, ab = 0.3175, 0.09
        self.assertAlmostEqual(float(ref.declared_aniso_alpha(at, ab)), math.sqrt(0.5 * (at * at + ab * ab)), places=12)
        self.assertEqual(float(ref.declared_aniso_alpha(at, ab, "ignored")), ab)
        self.assertEqual(float(ref.declared_aniso_alpha(at, ab, "alpha-t")), at)
        base, metallic, rough = np.array([[0.8, 0.35, 0.15]] * 2), np.array([0.0, 1.0]), np.full(2, 0.3)
        mu = np.array([0.6, 0.6])
        plain = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5)
        # Strength 0: the declared model is the isotropic one exactly, for every variant of the fit.
        for kind in ("spec", "ignored", "alpha-t"):
            zero = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, {**ref.DECLARED, "aniso": kind},
                                                     aniso=aniso_pixels(0.0, 0.3, count=2))
            np.testing.assert_allclose(zero, plain, rtol=1e-12, err_msg=kind)
        # Strength 1: the fit lowers the metal, `ignored` is today's engine, `alpha-t` lowers it further.
        spec = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, ref.DECLARED,
                                                 aniso=aniso_pixels(1.0, 0.3, count=2))
        blind = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, ref.ANISO_DECLARED_CONTROLS["declared-aniso-ignored"],
                                                  aniso=aniso_pixels(1.0, 0.3, count=2))
        wide = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, ref.ANISO_DECLARED_CONTROLS["declared-aniso-alpha-t"],
                                                 aniso=aniso_pixels(1.0, 0.3, count=2))
        np.testing.assert_allclose(blind, plain, rtol=1e-12)
        self.assertLess(float(spec[1, 0]), float(plain[1, 0]) - 0.005)
        self.assertLess(float(wide[1, 0]), float(spec[1, 0]))
        # The declared model does not depend on the rotation (the constant environment makes the bent normal invisible).
        spin = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, ref.DECLARED,
                                                 aniso=aniso_pixels(1.0, 1.1, w=-1.0, count=2))
        np.testing.assert_allclose(spin, spec, rtol=1e-12)

    def test_declared_anisotropic_environment_reads_the_albedo_table_and_meets_decision_11(self):
        # F15.1 (ticket 11, round 3): the declared model of an anisotropic pixel is the split sum with the engine's anisotropic albedo table
        # (tools/aniso_table.py, renderer/aniso_table.h): a white metal under unit radiance returns A + B of the table at (N.V, azimuth, roughness,
        # strength).  Against the exact quadrature on the ticket grid (sphere-weighted over N.V >= 0.1 and all azimuths, r 0.2 / 0.3 / 0.5 / 0.7,
        # strength 0.25 .. 1) the table model is below the decision-11 5 % (bound fixed beforehand; the measured worst is printed) and the former rms
        # fit (control `declared-aniso-rms-fit`) is above it at strength 1 (r 0.2 / 0.3 / 0.5: 6.5 %); `azimuth-ignored` reads a fixed azimuth and
        # leaves 3-5 % where the azimuth matters.
        mus = np.linspace(0.1, 1.0, 19)
        phis = np.linspace(0.0, 0.5 * math.pi, 9)
        grid_mu, grid_phi = np.meshgrid(mus, phis, indexing="ij")
        mu, phi = grid_mu.ravel(), grid_phi.ravel()
        count = len(mu)
        weight = 2.0 * mu * np.where(np.isclose(phi, 0.0) | np.isclose(phi, 0.5 * math.pi), 0.5, 1.0)
        worst = {"spec": 0.0, "rms-fit": 0.0, "azimuth-ignored": 0.0}
        for rough in (0.2, 0.3, 0.5, 0.7):
            for strength in (0.25, 0.5, 0.75, 1.0):
                ab = max(rough * rough, 0.002)
                at = ab + (1 - ab) * strength * strength
                tmus, tphis, ta, tb = ref.aniso_albedo_table(at, ab)
                exact = ref._bilinear(tmus, tphis, ta + tb, mu, phi)
                aniso = aniso_pixels(strength, 0.0, count=count)
                aniso["phi"] = phi
                for kind in worst:
                    out = ref.declared_environment_radiance(np.ones((count, 3)), np.ones(count), np.full(count, rough), mu, 1.0, {**ref.DECLARED, "aniso": kind},
                                                            aniso=aniso)[:, 0]
                    error = float((np.abs(out - exact) / exact * weight).sum() / weight.sum())
                    worst[kind] = max(worst[kind], error)
                    if kind == "spec":
                        self.assertLess(error, 0.05, (rough, strength))
                        a, b = ref.aniso_table_lookup(mu, phi, ref.clamp_roughness(np.full(count, rough)), np.full(count, strength))
                        np.testing.assert_allclose(out, np.clip(a + b, 0, 1), rtol=1e-12)
                    if kind == "rms-fit" and strength == 1.0 and rough <= 0.5:
                        self.assertGreater(error, 0.05, (rough, strength))  # negative control: the old fit fails the bound
        print("\ndeclared anisotropic environment, worst sphere-weighted error against the exact albedo:", {k: round(v, 4) for k, v in worst.items()})
        self.assertLess(worst["spec"], 0.01)
        self.assertGreater(worst["azimuth-ignored"], 0.02)

    def test_declared_anisotropic_environment_keeps_isotropic_pixels_on_the_dfg_table(self):
        # Strength 0 pixels are not anisotropic in the engine (anisoActive false): the DFG table, bit for bit, for every kind of the declared model.
        base, metallic, rough, mu = np.array([[0.8, 0.35, 0.15]] * 2), np.array([0.0, 1.0]), np.full(2, 0.3), np.array([0.6, 0.4])
        plain = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5)
        for kind in ("spec", "azimuth-ignored", "rms-fit", "ignored"):
            out = ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, {**ref.DECLARED, "aniso": kind}, aniso=aniso_pixels(0.0, 0.3, count=2))
            np.testing.assert_allclose(out, plain, rtol=1e-12, err_msg=kind)
        with self.assertRaises(ValueError):  # the table needs the azimuth of the view
            bad = aniso_pixels(1.0, 0.3, count=2)
            del bad["phi"]
            ref.declared_environment_radiance(base, metallic, rough, mu, 0.5, aniso=bad)

    def test_declared_fit_error_against_the_exact_quadrature_is_published_and_bounded(self):
        # Sphere-weighted (pdf 2 mu over N.V >= 0.1, uniform view azimuth) mean relative error of A + B at alpha_b 0.09.
        for strength, bound in ((0.5, 0.02), (1.0, 0.09)):
            at = float(ref.aniso_alphas(np.array([0.09]), np.array([strength]))[0][0])
            mus, phis, a, b = ref.aniso_albedo_table(at, 0.09)
            exact = a + b
            alpha_eff = float(ref.declared_aniso_alpha(at, 0.09))
            table = ref.environment_table(math.sqrt(alpha_eff))
            fit = np.interp(mus, table["mu"], table["A"] + table["B"])[:, None]
            weight = (2.0 * mus * (mus >= 0.1))[:, None] * np.ones_like(exact)
            error = float((np.abs(fit - exact) / exact * weight).sum() / weight.sum())
            self.assertLess(error, bound, strength)

    def test_declared_fit_bound_over_roughness_and_no_isotropic_roughness_does_better(self):
        # Ticket 11 (F15.3 closure): the declared fit error against the exact quadrature as a TESTED bound at strength 1 (the worst
        # strength) over base roughness 0.3 / 0.7 (the roughness sweep of tools/f15_check.py measured 0.2, 0.3, 0.5, 0.7: rms fit
        # 6.5 / 6.6 / 6.5 / 4.8 %).  Bound fixed beforehand from decision 11 (5 % per sphere) plus the measured tolerance of the declared
        # fit: 7 %.  The second assertion explains why the fit cannot meet 5 %: the BEST isotropic roughness (60 candidates) still
        # leaves >= 3.5 % at strength 1, so a single effective roughness (the declared split-sum form) cannot reach below ~4 %.
        def sweep(at, ab, roughness):
            mus, phis, a, b = ref.aniso_albedo_table(at, ab)
            exact = a + b
            table = ref.environment_table(roughness)
            fit = np.interp(mus, table["mu"], table["A"] + table["B"])[:, None]
            weight = (2.0 * mus * (mus >= 0.1))[:, None] * np.ones_like(exact)
            return float((np.abs(fit - exact) / exact * weight).sum() / weight.sum())
        for rough in (0.3, 0.7):
            ab = rough * rough
            at = float(ref.aniso_alphas(np.array([ab]), np.array([1.0]))[0][0])
            fit_roughness = math.sqrt(float(ref.declared_aniso_alpha(at, ab)))
            self.assertLess(sweep(at, ab, fit_roughness), 0.07, rough)
            best = min(sweep(at, ab, c) for c in np.linspace(max(0.05, math.sqrt(ab) * 0.8), 1.0, 60))
            self.assertGreater(best, 0.035, rough)

    def test_composite_complement_never_gains_energy_on_the_sphere_average(self):
        # The composite's diffuse complement (1 - E_dielectric) uses the mean-alpha roughness sqrt((alpha_t + alpha_b) / 2) of
        # bsdfAnisoDiffuseRoughness: bound = the sphere-weighted mean of E_fit - E_exact (F0 0.04) is >= 0 (the complement never
        # exceeds the true one on average: measured +0.0007 / +0.0023 / +0.0030 / +0.0005 at (0.3, 0.5), (0.3, 1), (0.2, 1), (0.7, 1)),
        # while the root-mean-square fit of the split sum is below zero (-0.0042 at (0.3, 1): the 1.0063 furnace gain).  A per-pixel
        # bound does not hold: E_fit - E_exact reaches -0.46 at grazing views along the stretched axis (N.V < 0.1), which the sphere
        # weight (2 mu over mu >= 0.1) excludes.
        def mean_gap(at, ab, roughness):
            mus, phis, a, b = ref.aniso_albedo_table(at, ab)
            table = ref.environment_table(roughness)
            fit = np.interp(mus, table["mu"], 0.04 * table["A"] + table["B"])[:, None]
            weight = (2.0 * mus * (mus >= 0.1))[:, None] * np.ones_like(a)
            return float(((fit - (0.04 * a + b)) * weight).sum() / weight.sum())
        ab = 0.09
        at = float(ref.aniso_alphas(np.array([ab]), np.array([1.0]))[0][0])
        self.assertGreaterEqual(mean_gap(at, ab, math.sqrt(0.5 * (at + ab))), 0.0)
        self.assertLess(mean_gap(at, ab, math.sqrt(float(ref.declared_aniso_alpha(at, ab)))), -0.002)  # the control: rms gains energy

    def test_rotation_flip_is_invisible_to_the_sphere_mean_of_a_constant_environment_but_not_on_a_yawed_panel(self):
        # Ticket 19 (dropped control aniso-rotation-flipped on the exact-environment keys): on a mirror-symmetric carrier (a sphere seen
        # along its axis, combed tangents symmetric about the vertical plane) flipping the lobe rotation IS the mirror image of the carrier, so
        # the sphere mean of the exact constant-environment radiance is identical (theorem, tested to 1e-9 here); it differs per pixel.  A
        # panel yawed 50 deg about the vertical and rolled 20 deg about its normal has a uniform tangent and an asymmetric view azimuth:
        # there the flip moves the mean (> 5 %).  The fixture redesign for the env keys is therefore a yawed, rolled panel, never a sphere.
        def mean_albedo(n, tangent, rotation, strength, variant):
            m = len(n)
            v = np.tile([0.0, 0.0, 1.0], (m, 1))
            aniso = {"strength": np.full(m, strength), "rotation": np.full(m, rotation), "tangent": tangent, "w": np.ones(m),
                     "det": np.ones(m), "variant": variant}
            env = ref.aniso_environment_inputs(aniso, np.full(m, 0.3), n, v)
            mu = np.maximum(n[:, 2], ref.MIN_NDOTV)
            a, b = ref.aniso_spec_albedo(mu, env)
            return a + b
        xs = np.linspace(-0.97, 0.97, 41)
        px, py = np.meshgrid(xs, xs)
        inside = px ** 2 + py ** 2 < 0.94
        n = np.stack([px[inside], py[inside], np.sqrt(1.0 - px[inside] ** 2 - py[inside] ** 2)], axis=1)
        comb = np.cross(np.array([0.0, 1.0, 0.0]), n)
        comb = comb / np.maximum(np.linalg.norm(comb, axis=1, keepdims=True), 1e-9)
        keep = np.linalg.norm(np.cross(np.array([0.0, 1.0, 0.0]), n), axis=1) > 0.05
        n, comb = n[keep], comb[keep]
        spec = mean_albedo(n, comb, math.pi / 4, 1.0, "spec")
        flipped = mean_albedo(n, comb, math.pi / 4, 1.0, "rotation-flipped")
        self.assertAlmostEqual(float(spec.mean()), float(flipped.mean()), delta=1e-9 * float(spec.mean()))
        self.assertGreater(float(np.abs(spec - flipped).max()), 0.02)  # per pixel the flip is visible
        yaw, roll = math.radians(50.0), math.radians(20.0)
        normal = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        e1, e2 = np.array([math.cos(yaw), 0.0, -math.sin(yaw)]), np.array([0.0, 1.0, 0.0])
        axis = math.cos(roll) * e1 + math.sin(roll) * e2
        panel_n, panel_t = np.tile(normal, (16, 1)), np.tile(axis, (16, 1))
        spec = mean_albedo(panel_n, panel_t, math.pi / 4, 1.0, "spec").mean()
        flipped = mean_albedo(panel_n, panel_t, math.pi / 4, 1.0, "rotation-flipped").mean()
        self.assertGreater(abs(float(spec - flipped)) / float(spec), 0.05)

    def test_exact_disk_integration_of_a_narrow_anisotropic_lobe(self):
        # F15.3 closure (ticket 11): `--exact-aniso-disk` integrates the disk of a narrow anisotropic lobe on a polar grid (a different family from the
        # engine's golden-angle spiral) and drops the pixels from the quadrature mask.  The grid is converged (64 x 128 against 192 x 384: < 1e-6 relative),
        # and the engine's 32 / 128-sample spiral agrees with it within 1e-3 on highlight pixels (measured 3e-5), i.e. the declared rule is accurate.
        saved = (ref.EXACT_ANISO_DISK, ref.EXACT_DISK_RINGS, ref.EXACT_DISK_AZIMUTHS, ref.EXACT_DISK_COUNT)

        def radiance(offset, flag, rings=64, azimuths=128):
            ref.EXACT_ANISO_DISK, ref.EXACT_DISK_RINGS, ref.EXACT_DISK_AZIMUTHS = flag, rings, azimuths
            ref.EXACT_DISK_COUNT = rings * azimuths
            n, v = np.array([[0.0, 0.0, 1.0]]), np.array([[math.sin(0.4), 0.0, math.cos(0.4)]])
            sun = np.array([-math.sin(0.4) + math.cos(0.4) * offset, 0.0, math.cos(0.4) + math.sin(0.4) * offset])
            sun /= np.linalg.norm(sun)
            out, narrow = ref.sun_radiance(np.array([[0.8, 0.35, 0.15]]), np.array([1.0]), np.array([0.1]), n, v, sun, 3.0, 0.00465,
                                           aniso=aniso_pixels(0.5, 0.0, count=1))
            return out[0], bool(narrow[0])
        try:
            for offset in (0.0, 0.003, 0.01):
                spiral, spiral_narrow = radiance(offset, False)
                exact, exact_narrow = radiance(offset, True)
                finer, _ = radiance(offset, True, 192, 384)
                self.assertTrue(spiral_narrow)
                self.assertFalse(exact_narrow)
                self.assertLess(float(np.max(np.abs(exact - finer) / finer)), 1e-6)
                self.assertLess(float(np.max(np.abs(spiral - exact) / exact)), 1e-3)
            # the polar grid is uniform in solid angle: its mean cosine equals the analytic mean of cos(theta) over the cap
            samples = ref.sun_disk_polar(np.array([0.0, 0.0, 1.0]), 0.2)
            self.assertAlmostEqual(float(samples[:, 2].mean()), (1.0 + math.cos(0.2)) / 2.0, places=6)
        finally:
            ref.EXACT_ANISO_DISK, ref.EXACT_DISK_RINGS, ref.EXACT_DISK_AZIMUTHS, ref.EXACT_DISK_COUNT = saved

    def test_bent_normal_and_the_legacy_hemisphere_reference(self):
        # ticket 11 (F15.3 closure): the README's bent normal and the legacy two-colour hemisphere.  Hand value: N = +Z, V = (0, .6, .8), bitangent +X:
        # cross(cross(B, V), B) = V, a = ((1 - s (1 - r))^2)^2 = 0.0081 at s = 1, r = 0.3, so N' = normalize(0.9919 V + 0.0081 N).
        n, v = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.6, 0.8]])
        bent = ref.aniso_bent_normal(n, v, np.array([[1.0, 0.0, 0.0]]), np.array([0.3]), np.array([1.0]))
        expected = 0.9919 * v + 0.0081 * n
        np.testing.assert_allclose(bent, expected / np.linalg.norm(expected), atol=1e-12)
        np.testing.assert_allclose(ref.aniso_bent_normal(n, v, np.array([[1.0, 0.0, 0.0]]), np.array([0.3]), np.array([0.0])), n, atol=1e-12)  # strength 0
        # the hemisphere: y = +1 is the sky, y = -1 the ground
        np.testing.assert_allclose(ref.hemisphere_radiance(np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])),
                                   [ref.HEMISPHERE_SKY, ref.HEMISPHERE_GROUND], atol=1e-12)
        # a strongly anisotropic metal seen at a glancing angle fetches a different part of the hemisphere along the bent normal: the control must move it
        base = np.array([[0.8, 0.35, 0.15]])
        aniso = aniso_pixels(1.0, 0.0, tangent=(0.0, 1.0, 0.0), count=1)  # comb along +Y: the bitangent lies in the surface plane, perpendicular to it
        nn = np.array([[0.0, 0.6, 0.8]])
        vv = np.array([[0.0, 0.0, 1.0]])
        good = ref.legacy_hemisphere_radiance(base, np.ones(1), np.full(1, 0.3), nn, vv, aniso, ref.DECLARED)
        wrong = ref.legacy_hemisphere_radiance(base, np.ones(1), np.full(1, 0.3), nn, vv, aniso, ref.ANISO_DECLARED_CONTROLS["declared-aniso-bent-normal-ignored"])
        self.assertGreater(float(np.abs(good - wrong).max() / good.max()), 0.02)
        # strength 0 is the isotropic legacy model for both
        flat = aniso_pixels(0.0, 0.0, count=1)
        np.testing.assert_allclose(ref.legacy_hemisphere_radiance(base, np.ones(1), np.full(1, 0.3), nn, vv, flat, ref.DECLARED),
                                   ref.legacy_hemisphere_radiance(base, np.ones(1), np.full(1, 0.3), nn, vv, flat,
                                                                  ref.ANISO_DECLARED_CONTROLS["declared-aniso-bent-normal-ignored"]), rtol=1e-12)

    def test_record_word_decoding_texture_refusal_and_combinations(self):
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731

        def record0():
            w = [0] * 20
            for k in (5, 6, 10, 11, 12, 17, 18):
                w[k] = ref.INVALID
            return w

        def record1(strength, rotation, texture=ref.INVALID):
            w = [0] * 20
            w[0], w[1], w[2] = f(strength), f(rotation), texture
            return w

        zero = [[0] * 20] * 2
        data = {"materials": [{"feature_bits": 8, "extension_index": 0}, {"feature_bits": 0},
                              {"feature_bits": 8, "extension_index": 4}, {"feature_bits": 10, "extension_index": 8},
                              {"feature_bits": 12, "extension_index": 12}, {"feature_bits": 16, "extension_index": 16}, {"feature_bits": 32, "extension_index": 16}],
                "material_extensions": {"version": 1, "records_per_material": 4, "records":
                                        [record0(), record1(0.5, -0.75)] + zero +
                                        [record0(), record1(1.0, 0.0, texture=5)] + zero +
                                        [record0(), record1(1.0, 0.0)] + zero +
                                        [record0(), record1(1.0, 0.0)] + zero +
                                        [record0(), record1(1.0, 0.0)] + zero}}
        strength, rotation = ref.material_anisotropy(data, data["materials"][0])
        self.assertEqual((strength, rotation), (0.5, -0.75))
        self.assertIsNone(ref.material_anisotropy(data, data["materials"][1]))
        self.assertIsNone(ref.material_anisotropy(data, data["materials"][5]))     # a different feature
        with self.assertRaises(ValueError):                                        # anisotropyTexture is refused
            ref.material_anisotropy(data, data["materials"][2])
        for index in (3, 4):                                                       # ticket 19: coat / sheen on top are composed
            self.assertEqual(ref.material_anisotropy(data, data["materials"][index]), (1.0, 0.0))
        data["materials"][3]["feature_bits"] = 8 | 128                             # thin subsurface + anisotropy: composed (ticket 16 closure)
        self.assertEqual(ref.material_anisotropy(data, data["materials"][3]), (1.0, 0.0))
        data["materials"][3]["feature_bits"] = 10
        self.assertTrue(ref.uses_anisotropy(data))
        self.assertIsNone(ref.material_specular(data, data["materials"][0]))       # anisotropy-only: neutral specular block
        self.assertIsNone(ref.material_clearcoat(data, data["materials"][0]))
        self.assertIsNone(ref.material_sheen(data, data["materials"][0]))
        with self.assertRaises(ValueError):                                        # an unknown feature stays outside the model
            ref.material_extension_words(data, data["materials"][6])
        self.assertEqual(ref.ANISOTROPY_FEATURE_BIT, 8)  # MaterialFeature::Anisotropy

    def test_every_anisotropy_control_is_registered_with_a_declared_model_dict(self):
        for name, delta in ref.ANISO_CONTROLS.items():
            self.assertIn(delta["aniso"], ref.ANISO_VARIANTS, name)
            self.assertIn(name, ref.ALL_CONTROLS)
            self.assertEqual(ref.control_model(name, ref.APPENDIX_B)["aniso"], delta["aniso"])
        self.assertEqual(sorted(ref.ANISO_CONTROLS), ["aniso-alpha-swapped", "aniso-bitangent-flipped",
                                                      "aniso-d-normalised-by-ab", "aniso-ignored",
                                                      "aniso-mirror-unaware", "aniso-rotation-flipped"])
        self.assertEqual(sorted(ref.ANISO_DECLARED_CONTROLS), ["declared-aniso-alpha-t", "declared-aniso-azimuth-ignored", "declared-aniso-bent-normal-ignored", "declared-aniso-ignored", "declared-aniso-rms-fit"])
        self.assertIn("anisotropy", ref.MODEL)


def comb_ply(path, handedness, radius=0.4, slices=48, stacks=24):
    """A combed-sphere PLY of snapshot mesh version 2 (the generator of ProceduralMeshes::generateCombedSphere)."""
    def comb(n):
        d = n[0]
        t = [1.0 - d * n[0], -d * n[1], -d * n[2]]
        length = math.sqrt(sum(c * c for c in t))
        if length > 1e-3:
            return [c / length for c in t]
        c = [n[1] * 0 - n[2] * 1, n[2] * 0 - n[0] * 0, n[0] * 1 - n[1] * 0]  # cross(n, y)
        length = math.sqrt(sum(x * x for x in c))
        return [x / length for x in c]

    verts, faces = [], []
    for st in range(stacks + 1):
        theta = math.pi * st / stacks
        for sl in range(slices + 1):
            phi = 2 * math.pi * sl / slices
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            verts.append((*(c * radius for c in n), *n, sl / slices, st / stacks, *comb(n), handedness))
    for st in range(stacks):
        for sl in range(slices):
            a = st * (slices + 1) + sl
            b = a + slices + 1
            faces += [(a, b, a + 1), (a + 1, b, b + 1)]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\n"
                 "property float t\nproperty float tangent_x\nproperty float tangent_y\nproperty float tangent_z\n"
                 "property float handedness_x\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                 % (len(verts), len(faces))).encode())
        for v in verts:
            f.write(struct.pack("<12f", *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba not installed")
class AnisotropySnapshot(unittest.TestCase):
    """One combed sphere whose material carries KHR_materials_anisotropy in record 1 (the exporter's keys)."""

    def snapshot(self, directory, sun, strength, rotation, handedness=-1.0, mirrored=False, metallic=1.0, roughness=0.3,
                 sky=0.0, width=256, height=144, tangents=True):
        root = pathlib.Path(directory)
        if tangents:
            comb_ply(root / "mesh_0.ply", handedness)
        else:
            RayCastedSnapshot.write_sphere_ply(root / "mesh_0.ply")
        world = [-1.0 if mirrored else 1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731
        record0 = [0] * 20
        for k in (5, 6, 10, 11, 12, 17, 18):
            record0[k] = ref.INVALID
        record1 = [0] * 20
        record1[0], record1[1], record1[2] = f(strength), f(rotation), ref.INVALID
        data = {"schema": 1, "state": "TEST", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
                "camera": {"position": [0, 0, 3], "direction": [0, 0, -1], "up": [0, 1, 0],
                           "fov_y_radians": math.radians(60), "near": 0.05, "far": 0, "width": width, "height": height,
                           "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"},
                "sky": [sky] * 3, "sun_angular_radius": 0.00465, "textures": [],
                "materials": [{"base": [0.8, 0.35, 0.15, 1], "emissive": [0, 0, 0], "metallic": metallic,
                               "roughness": roughness, "normal_scale": 1, "occlusion_strength": 1, "alpha_cutoff": 0,
                               "flags": 8 << 8, "feature_bits": 8, "extension_index": 0, "textures": [ref.INVALID] * 5}],
                "material_extensions": {"version": 1, "records_per_material": 4,
                                        "records": [record0, record1, [0] * 20, [0] * 20]},
                "mesh_version": 2 if tangents else 1, "meshes": ["mesh_0.ply"],
                "instances": [{"slot": 0, "generation": 1, "flags": 7 | (8 if mirrored else 0), "mesh": 0, "material": 0,
                               "world": world}],
                "lights": [], "sampled_lights": []}
        if sun:
            data["lights"].append({"type": 0, "position": [0, 0, 0], "direction": list(-sun[0]), "color": [1, 1, 1],
                                   "intensity": sun[1], "range": 0, "inner": 0, "outer": 0})
        (root / "scene.json").write_text(json.dumps(data))
        return root

    def test_surface_tangent_at_the_centre_pixel_is_the_mesh_attribute(self):
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, None, 1.0, 0.0, handedness=-1.0))
            cast = ref.cast_primary_rays(root, data, 2)
            pick = np.flatnonzero(cast[0].reshape(-1))
            tangent, w, det = ref.surface_tangents(root, data, cast, pick)
            centre = int(np.argmin(np.abs(pick - (72 * 256 + 128))))
            self.assertEqual(int(pick[centre]), 72 * 256 + 128)
            # The interpolated attribute is only perpendicular to the interpolated normal up to the tessellation (48 x 24).
            np.testing.assert_allclose(tangent[centre], [1, 0, 0], atol=5e-2)
            n_centre = ref.normalize(cast[2][pick][centre:centre + 1])
            # (the pixel centre is half a pixel off the axis: the normal is tilted by about 0.03 rad.)
            comb = ref.normalize(np.array([[1.0, 0, 0]]) - n_centre[:, :1] * n_centre)
            np.testing.assert_allclose(ref.orthogonal_tangent(tangent[centre:centre + 1], n_centre)[0], comb[0], atol=1e-2)
            self.assertEqual((float(w[centre]), float(det[centre])), (-1.0, 1.0))
            # Everywhere: unit-ish tangent, perpendicular to the normal after orthogonalisation, handedness -1.
            n = ref.normalize(cast[2][pick])
            t = ref.orthogonal_tangent(tangent, n)
            self.assertLess(float(np.abs(ref.dot(t, n)).max()), 1e-9)
            np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-9)
            self.assertTrue(bool((w == -1.0).all()) and bool((det == 1.0).all()))
            # The orthogonalised tangent stays close to the comb direction x projected on the tangent plane (not near the poles).
            expected = ref.normalize(np.array([1.0, 0, 0]) - n[:, :1] * n)
            keep = np.abs(n[:, 0]) < 0.9
            self.assertGreater(float((t[keep] * expected[keep]).sum(axis=1).min()), 0.99)

    def test_mirrored_instance_has_the_determinant_and_the_mirror_image_of_the_surface(self):
        # A mirrored carrier lit by the mirrored sun renders as the exact horizontal mirror of the plain one (the camera
        # sits on the mirror plane): the physically correct handedness s_eff = w det.  The control that ignores the
        # determinant breaks this.
        from f12_reference import snapshot as load
        toward = unit(-0.8, 0.35, -0.3)
        mirrored_toward = np.array([-toward[0], toward[1], toward[2]])
        images = {}
        for mirrored in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                sun = (mirrored_toward if mirrored else toward, 3.0)
                root, data = load(self.snapshot(tmp, sun, 1.0, math.pi / 4, handedness=-1.0, mirrored=mirrored, sky=0.2))
                cast = ref.cast_primary_rays(root, data, 2)
                pick = np.flatnonzero(cast[0].reshape(-1))
                _, _, det = ref.surface_tangents(root, data, cast, pick)
                self.assertTrue(bool((det == (-1.0 if mirrored else 1.0)).all()))
                for name, model in (("spec", ref.APPENDIX_B), ("unaware", ref.control_model("aniso-mirror-unaware", ref.APPENDIX_B))):
                    images[(mirrored, name)] = ref.render_frame(root, data, model, cast=cast)[0]
        plain, mirrored_image = images[(False, "spec")], images[(True, "spec")]
        scale = float(plain.max())
        self.assertGreater(scale, 1.0)
        np.testing.assert_allclose(mirrored_image, plain[:, ::-1], atol=2e-3 * scale)
        unaware = images[(True, "unaware")]
        self.assertGreater(float(np.abs(unaware - plain[:, ::-1]).max()), 0.05 * scale)
        # The handedness w flips the lobe on its own: w = +1 on the plain instance equals w = -1 with the angle negated.
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, (toward, 3.0), 1.0, math.pi / 4, handedness=1.0, sky=0.2))
            cast = ref.cast_primary_rays(root, data, 2)
            plus = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, (toward, 3.0), 1.0, -math.pi / 4, handedness=-1.0, sky=0.2))
            cast = ref.cast_primary_rays(root, data, 2)
            minus = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast)[0]
        np.testing.assert_allclose(plus, minus, atol=2e-3 * scale)
        self.assertGreater(float(np.abs(plus - plain).max()), 0.05 * scale)

    def test_render_controls_declared_and_the_isotropic_limit(self):
        from f12_reference import snapshot as load
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, (toward, 3.0), 0.5, math.pi / 4, sky=0.5))
            self.assertTrue(ref.uses_anisotropy(data) and not ref.uses_specular_extension(data))
            cast = ref.cast_primary_rays(root, data, 2)
            images = {}
            for name in ("spec", "ignored", "rotation-flipped", "bitangent-flipped", "alpha-swapped", "d-normalised-by-ab"):
                model = ref.control_model("aniso-" + name, ref.APPENDIX_B) if name != "spec" else ref.APPENDIX_B
                images[name], masks, spheres, terms = ref.render_frame(root, data, model, cast=cast)
            self.assertEqual(terms, ["sun-disk", "environment-quadrature"])
            self.assertEqual(spheres[0]["anisotropy"], 0.5)
            self.assertAlmostEqual(spheres[0]["anisotropy_rotation"], math.pi / 4, places=6)
            self.assertFalse(spheres[0]["mirrored"])
            self.assertEqual(spheres[0]["group"], "metallic 1 anisotropy 0.5")
            roi = masks["roi"]
            for name in ("ignored", "rotation-flipped", "bitangent-flipped", "alpha-swapped", "d-normalised-by-ab"):
                self.assertGreater(float(np.abs(images[name][roi] - images["spec"][roi]).max()), 1e-3, name)
            # w = -1: ignoring the handedness gives exactly the rotation-flipped lobe (the combed grid cannot tell them apart).
            np.testing.assert_allclose(images["bitangent-flipped"], images["rotation-flipped"], rtol=1e-12)
            # Strength 0 is the plain model (no feature bit: the neutral material renders like Appendix B).
            with tempfile.TemporaryDirectory() as tmp0:
                root0, data0 = load(self.snapshot(tmp0, (toward, 3.0), 0.0, math.pi / 4, sky=0.5))
                zero = ref.render_frame(root0, data0, ref.APPENDIX_B, cast=ref.cast_primary_rays(root0, data0, 2))[0]
                data0["materials"][0].update(feature_bits=0, flags=0)
                plain = ref.render_frame(root0, data0, ref.APPENDIX_B, cast=ref.cast_primary_rays(root0, data0, 2))[0]
            np.testing.assert_allclose(zero, plain, rtol=1e-9, atol=1e-12)
            # The declared model on the same cast: the engine's resolve model, differing from the ignored image.
            declared, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast, environment=ref.DECLARED)
            blind, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast,
                                         environment=ref.ANISO_DECLARED_CONTROLS["declared-aniso-ignored"])
            self.assertGreater(float(np.abs(declared[roi] - blind[roi]).max()), 1e-3)

    def test_command_writes_controls_and_declared_controls(self):
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, (toward, 3.0), 1.0, math.pi / 4, sky=0.5)
            out = pathlib.Path(tmp) / "out"
            code = ref.main(["render", str(root), "--out", str(out), "--controls", "aniso-ignored", "aniso-mirror-unaware",
                             "--declared-controls", "declared-aniso-ignored", "declared-aniso-alpha-t"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertEqual(sorted(report["controls"]), ["aniso-ignored", "aniso-mirror-unaware"])
            self.assertEqual(report["controls"]["aniso-ignored"]["model_variant"]["aniso"], "ignored")
            self.assertEqual(sorted(report["declared"]["controls"]), ["declared-aniso-alpha-t", "declared-aniso-ignored"])
            self.assertIn("anisotropy", report["model"])

    def test_textured_anisotropy_and_a_mesh_without_tangents_are_refused(self):
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, None, 1.0, 0.0, sky=0.5)
            path = pathlib.Path(root) / "scene.json"
            data = json.loads(path.read_text())
            data["material_extensions"]["records"][1][2] = 3
            path.write_text(json.dumps(data))
            root, data = load(root)
            with self.assertRaises(ValueError):
                ref.require_declared_scene(data)
            self.assertEqual(ref.main(["render", str(root), "--out", str(pathlib.Path(tmp) / "out")]), 2)
        with tempfile.TemporaryDirectory() as tmp:  # mesh version 1: no tangent attribute to read
            root, data = load(self.snapshot(tmp, None, 1.0, 0.0, sky=0.5, tangents=False))
            with self.assertRaises(ValueError):
                ref.render_frame(root, data, threads=2)
            self.assertEqual(ref.main(["render", str(root), "--out", str(pathlib.Path(tmp) / "out")]), 2)

    def test_read_ply_parses_the_tangent_properties(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "m.ply"
            comb_ply(path, -1.0, slices=8, stacks=4)
            properties, faces = ref.read_ply(path)
            self.assertEqual(sorted(properties), sorted(["x", "y", "z", "nx", "ny", "nz", "s", "t", "tangent_x", "tangent_y",
                                                         "tangent_z", "handedness_x"]))
            self.assertEqual(faces.shape, (2 * 8 * 4, 3))
            self.assertTrue(bool((properties["handedness_x"] == -1.0).all()))
            self.assertAlmostEqual(float(np.sqrt(properties["tangent_x"] ** 2 + properties["tangent_y"] ** 2 +
                                                 properties["tangent_z"] ** 2).min()), 1.0, places=5)


def irid_pixels(count, strength=1.0, ior=1.3, thickness=400.0, thickness_min=100.0, nv=0.6, variant="spec"):
    """A per-pixel iridescence dict of `count` identical (or broadcast) pixels."""
    def column(x):
        return np.broadcast_to(np.asarray(x, dtype=np.float64), (count,)).copy()
    return {"strength": column(strength), "ior": column(ior), "thickness": column(thickness),
            "thickness_min": column(thickness_min), "nv": column(nv), "variant": variant}


# --- an independent SPECTRAL evaluation of a thin film (no Belcour-Barla approximation, no phase table of the README) -----
# Airy summation with the exact unpolarised Fresnel amplitudes of an air | film | base stack, integrated against the CIE 1931
# colour matching functions (Wyman-Sloan-Shirley multi-lobe fit) under an equal-energy illuminant, mapped to linear Rec709 and
# divided by the white of the equal-energy spectrum.  The base ior is the README's real ior of the base f0 (extinction 0).

def _lobe(l, mu, s1, s2):
    return np.exp(-0.5 * ((l - mu) / np.where(l < mu, s1, s2)) ** 2)


def _cie(l):
    x = 1.056 * _lobe(l, 599.8, 37.9, 31.0) + 0.362 * _lobe(l, 442.0, 16.0, 26.7) - 0.065 * _lobe(l, 501.1, 20.4, 26.2)
    y = 0.821 * _lobe(l, 568.8, 46.9, 40.5) + 0.286 * _lobe(l, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(l, 437.0, 11.8, 36.0) + 0.681 * _lobe(l, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


_SRGB_FROM_XYZ = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                           [0.0556434, -0.2040259, 1.0572252]])
_WAVELENGTHS = np.arange(360.0, 831.0, 1.0)
_CMF = _cie(_WAVELENGTHS)
_WHITE = _SRGB_FROM_XYZ @ (_CMF.sum(0) / _CMF[:, 1].sum())


def _amplitudes(n1, n2, c1):
    s2 = n1 / n2 * np.sqrt(1 - c1 ** 2)
    c2 = np.sqrt(np.maximum(1 - s2 ** 2, 0.0))
    rs = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
    rp = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
    return rs, rp, c2


def spectral_thin_film(cos1, film_ior, thickness, f0):
    base_ior = (1 + np.sqrt(np.minimum(np.asarray(f0) + 1e-4, 0.9999))) / (1 - np.sqrt(np.minimum(np.asarray(f0) + 1e-4, 0.9999)))
    r12s, r12p, c2 = _amplitudes(1.0, film_ior, cos1)
    rgb = np.empty(3)
    for channel in range(3):
        r23s, r23p, _ = _amplitudes(film_ior, base_ior[channel], c2)
        delta = 4 * math.pi * film_ior * thickness * c2 / _WAVELENGTHS
        spectrum = 0.0
        for r12, r23 in ((r12s, r23s), (r12p, r23p)):
            e = np.exp(1j * delta)
            spectrum = spectrum + 0.5 * np.abs((r12 + r23 * e) / (1 + r12 * r23 * e)) ** 2
        xyz = (spectrum[:, None] * _CMF).sum(0) / _CMF[:, 1].sum()
        rgb[channel] = (_SRGB_FROM_XYZ @ xyz)[channel] / _WHITE[channel]
    return rgb


class IridescenceModel(unittest.TestCase):
    """KHR_materials_iridescence (ticket 12): the thin-film Fresnel of the specification, its composition and the environment."""

    FILMS = [(f0, ior, d, c) for f0 in ([0.04] * 3, [0.95, 0.64, 0.54]) for ior in (1.3, 1.8)
             for d in (100.0, 200.0, 400.0, 600.0, 800.0, 1200.0) for c in (1.0, 0.7, 0.4)]

    def film(self, c, ior, d, f0, **kw):
        return ref.thin_film_fresnel(np.array([c]), np.array([float(ior)]), np.array([float(d)]), np.array([f0], dtype=float), **kw)[0]

    def test_transcription_agrees_with_an_independent_spectral_evaluation(self):
        # The README closed form is an APPROXIMATION of the spectral reflectance: the check is statistical (mean 0.016 measured,
        # max 0.14 at the red channel of a dielectric near the white-point scaling), and a sign / phase mistake breaks it.
        errors = []
        for f0, ior, d, c in self.FILMS:
            errors.append(np.abs(self.film(c, ior, d, f0) - spectral_thin_film(c, ior, d, f0)))
        errors = np.array(errors)
        self.assertLess(float(errors.mean()), 0.025)
        self.assertLess(float(errors.max()), 0.17)
        # Negative control: reflection phases all zero (a mistake in phi12/phi23) must be visibly further away.
        wrong = np.array([np.abs(self.film(c, ior, d, f0, phase_shifts=False) - spectral_thin_film(c, ior, d, f0))
                          for f0, ior, d, c in self.FILMS])
        self.assertGreater(float(wrong.mean()), 2.0 * float(errors.mean()))
        self.assertGreater(float(wrong.max()), float(errors.max()))

    def test_hand_values(self):
        # film ior equal to the base ior of the dielectric (f0 + 0.0001): R23 = 0, no interference, F = R12 = Schlick(R0, cos1).
        base_ior = (1 + math.sqrt(0.0401)) / (1 - math.sqrt(0.0401))
        for c in (1.0, 0.5):
            r0 = ((base_ior - 1) / (base_ior + 1)) ** 2
            expected = r0 + (1 - r0) * (1 - c) ** 5
            for d in (0.0, 250.0, 1000.0):
                np.testing.assert_allclose(self.film(c, base_ior, d, [0.04] * 3), expected, atol=0.01)
        # Total internal reflection returns 1 (film ior below the outside ior: outside 1.5, film 1.0, grazing).
        total = ref.thin_film_fresnel(np.array([0.3]), np.array([1.0]), np.array([300.0]), np.full((1, 3), 0.04), outside_ior=1.5)
        np.testing.assert_array_equal(total, np.ones((1, 3)))
        # White metal: base ior ~ 4e4 makes the film invisible, F = 1.
        np.testing.assert_allclose(self.film(0.7, 1.8, 400.0, [1.0] * 3), 1.0, atol=1e-3)
        # The README constants are the transcribed ones (spot check of the first lobe and of the matrix).
        self.assertAlmostEqual(float(ref._XYZ_TO_REC709[0, 0]), 3.2404542)
        self.assertAlmostEqual(float(ref._XYZ_TO_REC709[1, 0]), -0.9692660)
        self.assertAlmostEqual(float(ref._XYZ_TO_REC709[2, 2]), 1.0572252)
        # Zero-thickness evaluation of the closed form is close to (not equal to) the plain Fresnel: why the rule is a convention.
        np.testing.assert_allclose(self.film(1.0, 1.3, 0.0, [0.95, 0.64, 0.54]), [0.95, 0.64, 0.54], atol=0.02)

    def test_continuity_in_thickness_and_angle(self):
        c = np.linspace(0.02, 1.0, 400)
        for f0 in ([0.04] * 3, [0.95, 0.64, 0.54]):
            f = ref.thin_film_fresnel(c, np.full(400, 1.8), np.full(400, 600.0), np.broadcast_to(np.array(f0), (400, 3)))
            self.assertLess(float(np.abs(np.diff(f, axis=0)).max()), 0.03)
            d = np.linspace(0.0, 1500.0, 600)
            g = ref.thin_film_fresnel(np.full(600, 0.8), np.full(600, 1.3), d, np.broadcast_to(np.array(f0), (600, 3)))
            self.assertLess(float(np.abs(np.diff(g, axis=0)).max()), 0.03)
            self.assertTrue(bool(np.all(f >= 0)))

    def test_thickness_zero_and_factor_zero_reduce_exactly_to_the_base_model(self):
        n = 5
        rng = np.random.default_rng(3)
        normal = np.tile([0.0, 0.0, 1.0], (n, 1))
        v = ref.normalize(rng.normal(size=(n, 3)) + [0, 0, 3])
        l = ref.normalize(rng.normal(size=(n, 3)) + [0, 0, 2])
        base = rng.uniform(0.1, 0.9, (n, 3))
        metallic = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        rough = np.full(n, 0.3)
        plain = ref.bsdf(base, metallic, rough, normal, v, l)
        for kwargs in (dict(thickness=0.0), dict(strength=0.0)):
            film = irid_pixels(n, nv=np.clip(ref.dot(normal, v), 1e-4, 1), **kwargs)
            np.testing.assert_array_equal(ref.bsdf(base, metallic, rough, normal, v, l, irid=film), plain)
        # ... and the controls that read another thickness do see the film at thickness 0.
        film = irid_pixels(n, thickness=0.0, thickness_min=100.0, variant="thickness-min", nv=np.clip(ref.dot(normal, v), 1e-4, 1))
        self.assertGreater(float(np.abs(ref.bsdf(base, metallic, rough, normal, v, l, irid=film) - plain).max()), 1e-4)
        # The film without a thickness texture takes the MAXIMUM; `thickness-max` of the ticket cannot differ on texture-free data.
        film = irid_pixels(n, thickness=400.0, thickness_min=100.0, nv=np.clip(ref.dot(normal, v), 1e-4, 1))
        np.testing.assert_array_equal(ref.irid_thickness(film), np.full(n, 400.0))

    def test_composition_with_the_base_fresnel_follows_the_specification(self):
        # dielectric: (1 - s) F_schlick + s F_irid on the specular weight, (1 - s)(1 - F_schlick) + s (1 - max F_irid) on the diffuse.
        base = np.array([[0.8, 0.35, 0.15]])
        vh = np.array([0.7])
        film = irid_pixels(1, strength=0.5, ior=1.8, thickness=600.0)
        diffuse, specular = ref.lobe_weights(base, np.array([0.0]), vh, ref.APPENDIX_B, None, film)
        f_irid = self.film(0.7, 1.8, 600.0, [0.04] * 3)
        f_schlick = 0.04 + 0.96 * 0.3 ** 5
        np.testing.assert_allclose(specular[0], 0.5 * f_schlick + 0.5 * f_irid, rtol=1e-12)
        np.testing.assert_allclose(diffuse[0], 0.5 * (1 - f_schlick) + 0.5 * (1 - f_irid.max()), rtol=1e-12)
        # metal: (1 - s) F_schlick(base) + s F_irid(base_f0 = baseColor), no diffuse.
        diffuse, specular = ref.lobe_weights(base, np.array([1.0]), vh, ref.APPENDIX_B, None, film)
        np.testing.assert_allclose(diffuse[0], 0.0)
        np.testing.assert_allclose(specular[0], 0.5 * (base[0] + (1 - base[0]) * 0.3 ** 5) +
                                   0.5 * self.film(0.7, 1.8, 600.0, base[0]), rtol=1e-12)
        # The light-sample angle is V.H: the at-nv control reads N.V instead.
        wrong = irid_pixels(1, strength=0.5, ior=1.8, thickness=600.0, nv=0.4, variant="at-nv")
        _, spec_wrong = ref.lobe_weights(base, np.array([0.0]), vh, ref.APPENDIX_B, None, wrong)
        np.testing.assert_allclose(spec_wrong[0], 0.5 * f_schlick + 0.5 * self.film(0.4, 1.8, 600.0, [0.04] * 3), rtol=1e-12)
        diffuse, specular = ref.lobe_weights(base, np.array([0.0]), vh, ref.APPENDIX_B, None, film)
        self.assertGreater(float(np.abs(spec_wrong - specular).max()), 1e-3)

    def test_controls_move_the_model_in_the_documented_direction(self):
        base = np.array([[0.8, 0.35, 0.15]] * 2)
        metallic = np.array([0.0, 1.0])
        vh = np.array([0.7, 0.7])
        reference = ref.lobe_weights(base, metallic, vh, ref.APPENDIX_B, None, irid_pixels(2, ior=1.8, nv=0.4))
        for variant in ("ignored", "film-ior-ignored", "film-ior-default", "at-nv", "thickness-min"):
            other = ref.lobe_weights(base, metallic, vh, ref.APPENDIX_B, None, irid_pixels(2, ior=1.8, nv=0.4, variant=variant))
            self.assertGreater(float(np.abs(other[1] - reference[1]).max()), 1e-3, variant)
        # film-ior-default is the spec where the material says 1.3.
        same = ref.lobe_weights(base, metallic, vh, ref.APPENDIX_B, None, irid_pixels(2, ior=1.3, nv=0.4))
        other = ref.lobe_weights(base, metallic, vh, ref.APPENDIX_B, None, irid_pixels(2, ior=1.3, nv=0.4, variant="film-ior-default"))
        np.testing.assert_array_equal(same[1], other[1])
        # diffuse-unweighted keeps the specular and raises the dielectric diffuse weight.
        wrong = ref.lobe_weights(base, metallic, vh, ref.APPENDIX_B, None, irid_pixels(2, ior=1.8, nv=0.4, variant="diffuse-unweighted"))
        np.testing.assert_array_equal(wrong[1], reference[1])
        self.assertGreater(float(wrong[0][0, 0]), float(reference[0][0, 0]))
        self.assertTrue(set(ref.IRID_CONTROLS) <= set(ref.ALL_CONTROLS))
        self.assertTrue(set(ref.IRID_DECLARED_CONTROLS) <= set(ref.ALL_DECLARED_CONTROLS))
        for name, delta in ref.IRID_CONTROLS.items():
            self.assertEqual(ref.control_model(name, ref.APPENDIX_B)["irid"], delta["irid"])
            self.assertIn(delta["irid"], ref.IRID_VARIANTS)
        for name, model in ref.IRID_DECLARED_CONTROLS.items():
            self.assertIn(model["irid"], ref.IRID_VARIANTS)
        self.assertEqual(ref.DECLARED["irid"], "spec")

    def test_exact_environment_matches_an_independent_wi_space_monte_carlo(self):
        # Copper metal, film ior 1.8, 600 nm, factor 1: integral of D Vis F_irid(V.H) cos over the hemisphere, by uniform
        # hemisphere sampling of wi (independent of the half-vector histogram quadrature).
        rng = np.random.default_rng(11)
        count = 400000
        alpha = float(ref.effective_alpha(0.3))
        copper = np.array([0.95, 0.64, 0.54])
        for mu in (0.9, 0.5, 0.2):
            wo = np.array([math.sqrt(1 - mu * mu), 0.0, mu])
            z = rng.uniform(0.0, 1.0, count)
            phi = rng.uniform(0.0, 2 * math.pi, count)
            wi = np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], axis=1)
            h = ref.normalize(wi + wo)
            f = ref.thin_film_fresnel(np.sum(h * wo, axis=1), np.full(count, 1.8), np.full(count, 600.0),
                                      np.broadcast_to(copper, (count, 3)))
            lobe = ref.ggx_d(h[:, 2], alpha) * ref.smith_vis(mu, wi[:, 2], alpha)
            samples = f * (lobe * wi[:, 2])[:, None] * (2 * math.pi)
            estimate, sigma = samples.mean(axis=0), samples.std(axis=0) / math.sqrt(count)
            table = ref.irid_environment_table(alpha, 1.8, 600.0, copper, "spec")
            exact = np.array([np.interp(mu, table[0], table[1][:, c]) for c in range(3)])
            np.testing.assert_allclose(exact, estimate, atol=6 * float(sigma.max()) + 2e-3)

    def test_environment_value_reduces_to_the_base_without_a_film_and_is_the_spec_at_full_strength(self):
        n = 4
        base = np.tile([0.8, 0.35, 0.15], (n, 1))
        metallic = np.array([0.0, 1.0, 0.0, 1.0])
        rough = np.full(n, 0.3)
        mu = np.array([0.9, 0.6, 0.3, 0.1])
        plain = ref.environment_radiance(base, metallic, rough, mu, 1.0)
        for kwargs in (dict(thickness=0.0), dict(strength=0.0), dict(variant="ignored")):
            film = irid_pixels(n, nv=mu, **kwargs)
            np.testing.assert_array_equal(ref.environment_radiance(base, metallic, rough, mu, 1.0, irid_env=film), plain)
            np.testing.assert_array_equal(
                ref.declared_environment_radiance(base, metallic, rough, mu, 1.0, ref.DECLARED, irid=film),
                ref.declared_environment_radiance(base, metallic, rough, mu, 1.0, ref.DECLARED))
        film = irid_pixels(n, nv=mu)
        full = ref.environment_radiance(base, metallic, rough, mu, 1.0, irid_env=film)
        half = ref.environment_radiance(base, metallic, rough, mu, 1.0, irid_env=irid_pixels(n, strength=0.5, nv=mu))
        np.testing.assert_allclose(half, 0.5 * (full + plain), rtol=1e-9)  # the strength is a linear mix
        self.assertGreater(float(np.abs(full - plain).max()), 0.01)

    def test_furnace_bound_of_the_exact_quadrature_and_the_energy_control(self):
        # White base, radiance 1: the albedo may not exceed the base model's own (the literal Appendix B dielectric gains ~1 %
        # at roughness 0.3, a property of the specification, measured on the plain rows).
        mu = np.linspace(0.15, 1.0, 12)
        n = mu.size
        rough = np.full(n, 0.3)
        white = np.ones((n, 3))
        for metallic in (0.0, 1.0):
            m = np.full(n, metallic)
            plain = ref.environment_radiance(white, m, rough, mu, 1.0)
            ceiling = np.maximum(1.0, plain.max(axis=1)) + 0.05  # per N.V node; measured excess of the film over the plain model <= 0.035 (see the ticket record)
            for ior, thickness in ((1.3, 200.0), (1.3, 600.0), (1.8, 400.0), (1.8, 1200.0)):
                value = ref.environment_radiance(white, m, rough, mu, 1.0, irid_env=irid_pixels(n, ior=ior, thickness=thickness, nv=mu))
                self.assertTrue(bool((value.max(axis=1) <= ceiling).all()), (metallic, ior, thickness, value.max(axis=1) - ceiling))
            if metallic == 0.0:
                wrong = ref.environment_radiance(white, m, rough, mu, 1.0, irid_env=irid_pixels(n, ior=1.8, thickness=400.0, nv=mu,
                                                                                            variant="diffuse-unweighted"))
                self.assertGreater(float(wrong.max()), 1.05)  # gains energy: the furnace control

    def test_declared_environment_hand_value_and_its_error_against_the_exact_quadrature(self):
        n = 1
        base = np.array([[0.8, 0.35, 0.15]])
        mu = np.array([0.7])
        rough = np.array([0.3])
        film = irid_pixels(n, strength=0.5, ior=1.8, thickness=400.0, nv=mu)
        a, b = ref.dfg_lookup(mu, rough)
        carry = float(np.clip(a + b, 0, 1)[0])
        e_irid = np.clip(self.film(0.7, 1.8, 400.0, [0.04] * 3) * carry, 0, 1)
        e_schlick = float(np.clip(0.04 * a + b, 0, 1)[0])
        expected = 0.5 * (base[0] * (1 - e_schlick) + e_schlick) + 0.5 * (base[0] * (1 - e_irid.max()) + e_irid)
        got = ref.declared_environment_radiance(base, np.array([0.0]), rough, mu, 1.0, ref.DECLARED, irid=film)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        # The declared and the exact images agree to a few percent on the fixture's films (sphere-weighted mean over N.V >= 0.1);
        # the bound is a regression guard for the declared FIT of the film at N.V (`declared_irid_environment`), not a gate.
        mus = np.linspace(0.1, 1.0, 40)
        weights = mus / mus.sum()  # a sphere seen from the camera: the pixel density in N.V is proportional to N.V
        worst = 0.0
        for metallic in (0.0, 1.0):
            for ior in (1.3, 1.8):
                for thickness in (200.0, 400.0, 600.0, 800.0, 1200.0):
                    m = np.full(mus.size, metallic)
                    pixels = irid_pixels(mus.size, ior=ior, thickness=thickness, nv=mus)
                    args = (np.tile([0.8, 0.35, 0.15], (mus.size, 1)), m, np.full(mus.size, 0.3), mus, 0.5)
                    exact = ref.environment_radiance(*args, irid_env=pixels)
                    declared = ref.declared_environment_radiance(*args, ref.DECLARED, irid=pixels)
                    error = float((np.abs(declared - exact).max(axis=1) / np.maximum(exact.max(axis=1), 1e-3) * weights).sum())
                    worst = max(worst, error)
        print(f"declared vs exact iridescent environment, worst sphere-weighted mean relative error: {worst:.4f}")
        self.assertLess(worst, 0.25)

    def test_sun_radiance_at_the_disk_centre_is_the_bsdf(self):
        n = 3
        normal = np.tile([0.0, 0.0, 1.0], (n, 1))
        v = ref.normalize(np.array([[0.2, 0.1, 1.0], [-0.3, 0.2, 1.0], [0.0, 0.5, 1.0]]))
        base = np.tile([0.8, 0.35, 0.15], (n, 1))
        metallic = np.array([0.0, 1.0, 1.0])
        rough = np.full(n, 0.3)
        toward = unit(-0.4, 0.3, 0.8)
        film = irid_pixels(n, ior=1.8, thickness=600.0, nv=np.clip(ref.dot(normal, v), 1e-4, 1))
        sun, narrow = ref.sun_radiance(base, metallic, rough, normal, v, toward, np.array([3.0] * 3), 0.00465, irid=film)
        self.assertFalse(bool(narrow.any()))
        direct = ref.bsdf(base, metallic, rough, normal, v, np.tile(toward, (n, 1)), irid=film) * (toward[2] * 3.0)
        np.testing.assert_allclose(sun, direct, rtol=1e-9)
        point, _ = ref.point_radiance(base, metallic, rough, normal, v, np.zeros((n, 3)),
                                      {"position": [-4.0, 3.0, 8.0], "intensity": 100.0, "range": 100.0, "color": [1, 1, 1]}, irid=film)
        self.assertTrue(bool((point > 0).all()))

    def test_record_word_decoding_texture_refusal_and_combinations(self):
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731

        def record0():
            w = [0] * 20
            for k in (5, 6, 10, 11, 12, 17, 18):
                w[k] = ref.INVALID
            return w

        def record1(factor, ior, tmin, tmax, tex=ref.INVALID, thickness_tex=ref.INVALID):
            w = [0] * 20
            w[2] = ref.INVALID
            w[3], w[4], w[5], w[6], w[7], w[8] = f(factor), f(ior), f(tmin), f(tmax), tex, thickness_tex
            return w

        zero = [[0] * 20] * 2
        data = {"materials": [{"feature_bits": 16, "extension_index": 0}, {"feature_bits": 0},
                              {"feature_bits": 16, "extension_index": 4}, {"feature_bits": 16, "extension_index": 8},
                              {"feature_bits": 24, "extension_index": 12}, {"feature_bits": 8, "extension_index": 12}],
                "material_extensions": {"version": 1, "records_per_material": 4, "records":
                                        [record0(), record1(0.75, 1.8, 100.0, 650.0)] + zero +
                                        [record0(), record1(1.0, 1.3, 100.0, 400.0, tex=2)] + zero +
                                        [record0(), record1(1.0, 1.3, 100.0, 400.0, thickness_tex=3)] + zero +
                                        [record0(), record1(1.0, 1.3, 100.0, 400.0)] + zero}}
        np.testing.assert_allclose(ref.material_iridescence(data, data["materials"][0]), (0.75, 1.8, 100.0, 650.0), rtol=1e-7)
        self.assertIsNone(ref.material_iridescence(data, data["materials"][1]))
        self.assertIsNone(ref.material_iridescence(data, data["materials"][5]))      # a different feature
        for index in (2, 3):                                                          # factor / thickness texture: refused
            with self.assertRaises(ValueError):
                ref.material_iridescence(data, data["materials"][index])
        np.testing.assert_allclose(ref.material_iridescence(data, data["materials"][4]), (1.0, 1.3, 100.0, 400.0), rtol=1e-7)  # ticket 19: + anisotropy composed
        data["materials"][4]["feature_bits"] = 16 | 128                               # + thin subsurface: composed (ticket 16 closure)
        ref.material_iridescence(data, data["materials"][4])
        data["materials"][4]["feature_bits"] = 24
        self.assertTrue(ref.uses_iridescence(data))
        self.assertIsNone(ref.material_specular(data, data["materials"][0]))
        self.assertIsNone(ref.material_anisotropy(data, data["materials"][0]))
        self.assertEqual(ref.IRIDESCENCE_FEATURE_BIT, 16)  # MaterialFeature::Iridescence
        self.assertEqual(ref.IRIDESCENCE_RECORD, 1)


class IridescenceSnapshot(unittest.TestCase):
    """One sphere whose material carries KHR_materials_iridescence in record 1 (the exporter's keys)."""

    def snapshot(self, directory, sky, sun, metallic, factor=1.0, ior=1.8, thickness=600.0, thickness_min=100.0, roughness=0.3,
                 point=None):
        root = RayCastedSnapshot("snapshot").snapshot(directory, sky, sun, metallic, roughness, [0.8, 0.35, 0.15])
        path = pathlib.Path(root) / "scene.json"
        data = json.loads(path.read_text())
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731
        record0 = [0] * 20
        for k in (5, 6, 10, 11, 12, 17, 18):
            record0[k] = ref.INVALID
        record1 = [0] * 20
        record1[2] = ref.INVALID
        record1[3:9] = [f(factor), f(ior), f(thickness_min), f(thickness), ref.INVALID, ref.INVALID]
        data["materials"][0].update(feature_bits=16, extension_index=0, flags=16 << 8)
        data["material_extensions"] = {"version": 1, "records_per_material": 4,
                                       "records": [record0, record1, [0] * 20, [0] * 20]}
        if point:
            data["lights"].append({"type": 1, "position": point, "direction": [0, 0, -1], "color": [1, 1, 1], "intensity": 60.0,
                                   "range": 100.0, "inner": 0, "outer": 0})
        path.write_text(json.dumps(data))
        return root

    def test_render_sun_point_environment_and_controls(self):
        from f12_reference import snapshot as load
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, 0.5, (toward, 3.0), 0.5, point=[-3.0, 2.0, -3.0]))
            self.assertTrue(ref.uses_iridescence(data) and not ref.uses_specular_extension(data))
            cast = ref.cast_primary_rays(root, data, 2)
            images = {}
            for name in ("spec", *ref.IRID_CONTROLS):
                model = ref.APPENDIX_B if name == "spec" else ref.control_model(name, ref.APPENDIX_B)
                image, masks, spheres, terms = ref.render_frame(root, data, model, cast=cast)
                images[name] = image
            self.assertEqual(terms, ["point-light", "sun-disk", "environment-quadrature"])
            self.assertEqual(spheres[0]["iridescence"], 1.0)
            self.assertAlmostEqual(spheres[0]["iridescence_ior"], 1.8, places=6)
            self.assertEqual(spheres[0]["thickness"], 600.0)
            self.assertEqual(spheres[0]["group"], "metallic 0.5 iridescence 1 ior 1.8")
            roi = masks["roi"]
            for name in ref.IRID_CONTROLS:
                difference = np.abs(images[name][roi] - images["spec"][roi]).max()
                self.assertGreater(float(difference), 1e-3, name)
            declared, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast, environment=ref.DECLARED)
            blind, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast,
                                         environment=ref.IRID_DECLARED_CONTROLS["declared-iridescence-ignored"])
            self.assertGreater(float(np.abs(declared[roi] - blind[roi]).max()), 1e-3)

    def test_thickness_zero_renders_the_plain_model(self):
        from f12_reference import snapshot as load
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, 0.5, (toward, 3.0), 0.0, thickness=0.0))
            cast = ref.cast_primary_rays(root, data, 2)
            zero = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast)[0]
            data["materials"][0].update(feature_bits=0, flags=0)
            plain = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast)[0]
        np.testing.assert_allclose(zero, plain, rtol=1e-9, atol=1e-12)

    def test_command_writes_controls_and_declared_controls(self):
        toward = unit(-0.8, 0.35, -0.3)
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 0.5, (toward, 3.0), 0.0)
            out = pathlib.Path(tmp) / "out"
            code = ref.main(["render", str(root), "--out", str(out), "--controls", "iridescence-ignored", "iridescence-at-nv",
                             "--declared-controls", "declared-iridescence-ignored", "declared-iridescence-film-ior-ignored"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertEqual(sorted(report["controls"]), ["iridescence-at-nv", "iridescence-ignored"])
            self.assertEqual(report["controls"]["iridescence-at-nv"]["model_variant"]["irid"], "at-nv")
            self.assertEqual(sorted(report["declared"]["controls"]),
                             ["declared-iridescence-film-ior-ignored", "declared-iridescence-ignored"])
            self.assertIn("iridescence", report["model"])

    def test_textured_iridescence_is_refused(self):
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 0.5, None, 0.0)
            path = pathlib.Path(root) / "scene.json"
            data = json.loads(path.read_text())
            data["material_extensions"]["records"][1][8] = 3
            path.write_text(json.dumps(data))
            root, data = load(root)
            with self.assertRaises(ValueError):
                ref.require_declared_scene(data)
            self.assertEqual(ref.main(["render", str(root), "--out", str(pathlib.Path(tmp) / "out")]), 2)


def thin_pixels(count, weight=0.5, color=(1.0, 1.0, 1.0), variant="spec"):
    """A per-pixel thin-subsurface dict of `count` identical pixels."""
    return {"weight": np.full(count, float(weight)), "color": np.tile(np.asarray(color, dtype=np.float64), (count, 1)),
            "variant": variant}


def schlick_dielectric(v_dot_h):
    return 0.04 + 0.96 * (1.0 - v_dot_h) ** 5


class ThinSubsurfaceModel(unittest.TestCase):
    """KHR_materials_diffuse_transmission / subsurface thin mode (ticket 16)."""

    BASE = np.array([[0.3, 0.5, 0.8]])
    WARM = (1.0, 0.55, 0.25)

    def setUp(self):
        self.n = np.array([[0.0, 0.0, 1.0]])
        self.v = np.array([unit(0.3, 0.1, 1.0)])
        self.front = np.array([unit(-0.5, 0.2, 0.7)])
        self.back = np.array([unit(-0.5, 0.2, -0.7)])

    def transmitted(self, light, weight=0.5, color=WARM, variant="spec", metallic=0.0):
        return ref.thin_transmitted(thin_pixels(1, weight, color, variant), np.array([metallic]), self.n, self.v, light)[0]

    def test_transmitted_hand_value_of_the_literal_fresnel_mix(self):
        # f_t |N.L| = (1 - m) w (1 - F(V.H)) color / pi |N.L| with H = normalize(V + L) of the text.
        l = self.back[0]
        v = self.v[0]
        h = (v + l) / np.linalg.norm(v + l)
        expected = 0.5 * (1.0 - schlick_dielectric(float(v @ h))) * np.array(self.WARM) / math.pi * 0.7 / np.linalg.norm([0.5, 0.2, 0.7])
        np.testing.assert_allclose(self.transmitted(self.back), expected, rtol=1e-12)
        # V.H of a back light is sqrt((1 + V.L) / 2) < 0.71 (the identity the section comment relies on).
        self.assertAlmostEqual(float(v @ h), math.sqrt((1.0 + float(v @ l)) / 2.0), places=12)
        # A light straight behind the camera ray (V = -L) transmits nothing: F = 1 (the degenerate H is 0 by the engine rule).
        v_back = np.array([unit(0.0, 0.0, 1.0)])
        l_back = np.array([unit(0.0, 0.0, -1.0)])
        value = ref.thin_transmitted(thin_pixels(1), np.array([0.0]), self.n, v_back, l_back)
        np.testing.assert_array_equal(value, np.zeros((1, 3)))

    def test_hemisphere_convention_and_metals(self):
        np.testing.assert_array_equal(self.transmitted(self.front), np.zeros(3))      # a front light: no transmitted term
        self.assertGreater(float(self.transmitted(self.back).min()), 0.0)
        np.testing.assert_array_equal(self.transmitted(self.back, metallic=1.0), np.zeros(3))  # metals have no diffuse lobe
        half = self.transmitted(self.back, metallic=0.5)
        np.testing.assert_allclose(half, 0.5 * self.transmitted(self.back), rtol=1e-12)
        # The reflected value does not light a back light at all (the base BRDF is 0 for N.L <= 0).
        value = ref.bsdf(self.BASE, np.array([0.0]), np.array([0.5]), self.n, self.v, self.back, thin=thin_pixels(1, 0.5))
        np.testing.assert_array_equal(value, np.zeros((1, 3)))

    def test_reflected_part_is_reduced_by_w_and_the_specular_lobe_is_unchanged(self):
        rough = np.array([0.5])
        m = np.array([0.0])
        base_value = ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front)
        for w in (0.0, 0.25, 0.5, 1.0):
            value = ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front, thin=thin_pixels(1, w))
            h = (self.v[0] + self.front[0]) / np.linalg.norm(self.v[0] + self.front[0])
            diffuse = (1.0 - schlick_dielectric(float(self.v[0] @ h))) * self.BASE / math.pi
            np.testing.assert_allclose(base_value - value, w * diffuse, rtol=1e-10, atol=1e-14)
        # The colour of the transmission never reaches the reflected part.
        np.testing.assert_array_equal(ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front, thin=thin_pixels(1, 0.5, self.WARM)),
                                      ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front, thin=thin_pixels(1, 0.5, (0.1, 0.9, 0.2))))

    def test_weight_zero_reduces_exactly_to_the_base_model_for_every_term(self):
        n = np.tile(self.n, (3, 1))
        v = np.tile(self.v, (3, 1))
        l = np.concatenate([self.front, self.back, np.array([unit(0.9, 0.0, 0.1)])])
        base = np.tile(self.BASE, (3, 1))
        m = np.array([0.0, 0.3, 1.0])
        rough = np.array([0.2, 0.5, 0.9])
        zero = thin_pixels(3, 0.0, self.WARM)
        np.testing.assert_array_equal(ref.bsdf(base, m, rough, n, v, l), ref.bsdf(base, m, rough, n, v, l, thin=zero))
        light = {"position": [0.5, 1.0, 3.0], "range": 100.0, "intensity": 50.0, "color": [1, 1, 1]}
        pos = np.zeros((3, 3))
        np.testing.assert_array_equal(ref.point_radiance(base, m, rough, n, v, pos, light)[0],
                                      ref.point_radiance(base, m, rough, n, v, pos, light, thin=zero)[0])
        toward = unit(-0.5, 0.4, -0.7)
        np.testing.assert_array_equal(ref.sun_radiance(base, m, rough, n, v, toward, 3.0, 0.00465)[0],
                                      ref.sun_radiance(base, m, rough, n, v, toward, 3.0, 0.00465, thin=zero)[0])
        mu = np.array([0.2, 0.6, 1.0])
        np.testing.assert_array_equal(ref.environment_radiance(base, m, rough, mu, 0.5),
                                      ref.environment_radiance(base, m, rough, mu, 0.5, thin_env=zero))
        np.testing.assert_array_equal(ref.declared_environment_radiance(base, m, rough, mu, 0.5, ref.DECLARED),
                                      ref.declared_environment_radiance(base, m, rough, mu, 0.5, ref.DECLARED, thin=zero))
        # `lit` of a w = 0 material is the base one (nothing reaches it from behind).
        self.assertFalse(bool(ref.thin_lit(thin_pixels(1, 0.0), self.n, self.back)[0]))
        self.assertTrue(bool(ref.thin_lit(thin_pixels(1, 0.25), self.n, self.back)[0]))
        self.assertTrue(bool(ref.thin_lit(thin_pixels(1, 0.0), self.n, self.front)[0]))

    def test_direct_sun_and_point_terms_add_the_reflected_and_the_transmitted_lobe(self):
        n, v = np.tile(self.n, (2, 1)), np.tile(self.v, (2, 1))
        base = np.tile(self.BASE, (2, 1))
        m, rough = np.zeros(2), np.full(2, 0.7)
        thin = thin_pixels(2, 0.5, self.WARM)
        l = np.concatenate([self.front, self.back])
        # point light: value = bsdf cos + transmitted, at both hemispheres, times the incident radiance
        light = {"position": [0.0, 0.0, 4.0], "range": 100.0, "intensity": 40.0, "color": [1, 1, 1]}
        pos = np.zeros((2, 3))
        got, lit = ref.point_radiance(base, m, rough, n, v, pos, light, thin=thin_pixels(2, 0.5, self.WARM))
        self.assertTrue(bool(lit.all()))
        direction = np.array([0.0, 0.0, 1.0])
        l_front = np.tile(direction, (2, 1))
        expected = (ref.bsdf(base, m, rough, n, v, l_front, thin=thin) * 1.0 +
                    ref.thin_transmitted(thin, m, n, v, l_front)) * 40.0 / 16.0 * ref.point_window(4.0, 100.0)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        # the same light behind the surface: the reflected term vanishes, the transmitted one stays
        behind = {**light, "position": [0.0, 0.0, -4.0]}
        got, lit = ref.point_radiance(base, m, rough, n, v, pos, behind, thin=thin)
        expected = ref.thin_transmitted(thin, m, n, v, -l_front) * 40.0 / 16.0 * ref.point_window(4.0, 100.0)
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        self.assertTrue(bool(lit.all()))
        # sun from behind: the disk centre transmitted term times the irradiance
        toward = unit(-0.5, 0.3, -0.8)
        sun, _ = ref.sun_radiance(base, m, rough, n, v, toward, 3.0, 0.00465, thin=thin)
        np.testing.assert_allclose(sun, ref.thin_transmitted(thin, m, n, v, np.broadcast_to(toward, n.shape)) * 3.0, rtol=1e-12)

    def test_controls_move_the_model_in_the_documented_direction(self):
        w = self.transmitted(self.back, 0.5)
        np.testing.assert_array_equal(self.transmitted(self.back, variant="ignored"), np.zeros(3))
        untinted = self.transmitted(self.back, variant="untinted")
        np.testing.assert_allclose(untinted / untinted[0], np.ones(3), rtol=1e-12)
        self.assertTrue(bool((untinted >= w - 1e-15).all()))
        self.assertGreater(float(np.abs(untinted - w).max()), 1e-4)
        np.testing.assert_array_equal(self.transmitted(self.back, variant="front-only"), np.zeros(3))      # back lights ignored
        extra = self.transmitted(self.front, variant="front-only")                                          # ... front lights gain it
        self.assertGreater(float(extra.min()), 0.0)
        unweighted = self.transmitted(self.back, variant="fresnel-ignored")
        self.assertTrue(bool((unweighted > w).all()))
        h = (self.v[0] + self.back[0]) / np.linalg.norm(self.v[0] + self.back[0])
        np.testing.assert_allclose(unweighted / w, np.full(3, 1.0 / (1.0 - schlick_dielectric(float(self.v[0] @ h)))), rtol=1e-12)
        # double counted: the reflected diffuse keeps its full weight while the transmitted lobe is added (energy gain)
        rough, m = np.array([0.5]), np.array([0.0])
        spec = ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front, thin=thin_pixels(1, 0.5))
        doubled = ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front, thin=thin_pixels(1, 0.5, variant="double-counted"))
        plain = ref.bsdf(self.BASE, m, rough, self.n, self.v, self.front)
        np.testing.assert_array_equal(doubled, plain)
        self.assertTrue(bool((doubled > spec).all()))
        # every control and its registration
        for name, delta in ref.THIN_CONTROLS.items():
            self.assertEqual(ref.control_model(name, ref.APPENDIX_B), {**ref.APPENDIX_B, **delta})
            self.assertIn(delta["thin"], ref.THIN_VARIANTS)
            self.assertIn(name, ref.ALL_CONTROLS)
        for name, model in ref.THIN_DECLARED_CONTROLS.items():
            self.assertIn(model["thin"], ref.THIN_VARIANTS)
            self.assertIn(name, ref.ALL_DECLARED_CONTROLS)
        self.assertEqual(ref.DECLARED["thin"], "spec")

    def test_exact_transmitted_albedo_matches_an_independent_monte_carlo_and_is_bounded_by_the_front_diffuse_weight(self):
        mus, table = ref.thin_transmission_table()
        self.assertAlmostEqual(float(np.interp(1.0, mus, table)), 0.8075, delta=2e-3)  # hand-run value, see the section comment
        self.assertAlmostEqual(float(np.interp(1e-4, mus, table)), 0.933, delta=2e-3)
        rng = np.random.default_rng(7)
        count = 300_000
        for mu in (0.2, 0.55, 0.9):
            # cosine-weighted directions over the BACK hemisphere (z < 0): the albedo is E[(1 - F(V.H))]
            u1, u2 = rng.random(count), rng.random(count)
            r, phi = np.sqrt(u1), 2.0 * math.pi * u2
            l = np.stack([r * np.cos(phi), r * np.sin(phi), -np.sqrt(1.0 - u1)], axis=1)
            v = np.array([math.sqrt(1 - mu * mu), 0.0, mu])
            h = v + l
            h /= np.linalg.norm(h, axis=1, keepdims=True)
            estimate = float((1.0 - schlick_dielectric(np.clip(h @ v, 0.0, 1.0))).mean())
            self.assertAlmostEqual(estimate, float(np.interp(mu, mus, table)), delta=4e-3, msg=mu)
        cd = ref.environment_table(0.5)["Cd"]
        self.assertTrue(bool((table <= cd + 1e-9).all()))  # T <= Cd at every N.V: the thin lobe never adds energy
        # the declared ratio 1 - 0.16 mu is T / Cd within the published fit error
        error = np.abs(ref.thin_ratio(mus) - table / cd)
        self.assertLess(float(error.max()), 2e-3)
        self.assertEqual(ref.THIN_RATIO_SLOPE, 0.16)

    def test_environment_exact_value_albedo_energy_and_the_control_directions(self):
        mu = np.array([0.15, 0.5, 1.0])
        white = np.ones((3, 3))
        m, rough = np.zeros(3), np.full(3, 0.5)
        plain = ref.environment_radiance(white, m, rough, mu, 1.0)
        previous = plain
        mus, table = ref.thin_transmission_table()
        for w in (0.25, 0.5, 1.0):
            thin = thin_pixels(3, w)
            value = ref.environment_radiance(white, m, rough, mu, 1.0, thin_env=thin)
            # energy moved, not added: the white albedo never exceeds the unextended model, and falls with w
            self.assertTrue(bool((value <= previous + 1e-12).all()), w)
            previous = value
            cd = ref.environment_table(0.5)["Cd"]
            expected = plain - w * (np.interp(mu, mus, cd) - np.interp(mu, mus, table))[:, None]
            np.testing.assert_allclose(value, expected, rtol=1e-6)
        # colour: only the transmitted share is tinted
        warm = ref.environment_radiance(white, m, rough, mu, 1.0, thin_env=thin_pixels(3, 1.0, self.WARM))
        self.assertTrue(bool((warm[:, 0] > warm[:, 1]).all()) and bool((warm[:, 1] > warm[:, 2]).all()))
        # controls: double counted gains energy (furnace bound exceeded), front-only changes T into Cd, untinted brightens
        doubled = ref.environment_radiance(white, m, rough, mu, 1.0, thin_env=thin_pixels(3, 0.5, variant="double-counted"))
        self.assertGreater(float(doubled.max()), 1.05)
        self.assertTrue(bool((doubled > plain).all()))
        for variant in ("front-only", "fresnel-ignored", "untinted"):
            other = ref.environment_radiance(white, m, rough, mu, 1.0, thin_env=thin_pixels(3, 1.0, self.WARM, variant))
            self.assertGreater(float(np.abs(other - warm).max()), 1e-2, variant)
        # metals have no diffuse lobe: the metallic 1 rows are unchanged
        metal = ref.environment_radiance(white, np.ones(3), rough, mu, 1.0, thin_env=thin_pixels(3, 1.0, self.WARM))
        np.testing.assert_allclose(metal, ref.environment_radiance(white, np.ones(3), rough, mu, 1.0), rtol=1e-12)

    def test_furnace_bound_of_the_declared_model_by_construction_and_the_energy_control(self):
        mu = np.linspace(0.05, 1.0, 20)
        n = mu.size
        white = np.ones((n, 3))
        for rough in (0.2, 0.5, 0.9):
            r = np.full(n, rough)
            plain = ref.declared_environment_radiance(white, np.zeros(n), r, mu, 1.0, ref.DECLARED)
            for w in (0.25, 0.5, 1.0):
                value = ref.declared_environment_radiance(white, np.zeros(n), r, mu, 1.0, ref.DECLARED, thin=thin_pixels(n, w))
                self.assertTrue(bool((value <= plain + 1e-12).all()), (rough, w))
                self.assertLessEqual(float(value.max()), 1.0 + 1e-9)
            wrong = ref.declared_environment_radiance(white, np.zeros(n), r, mu, 1.0, ref.THIN_DECLARED_CONTROLS["declared-transmission-double-counted"],
                                                      thin=thin_pixels(n, 0.5, variant="double-counted"))
            self.assertGreater(float(wrong.max()), 1.0 + 0.005)

    def test_declared_environment_hand_value_and_its_error_against_the_exact_quadrature(self):
        base = np.array([[0.3, 0.5, 0.8]])
        mu, rough, w = np.array([0.7]), np.array([0.5]), 0.5
        a, b = ref.dfg_lookup(mu, rough)
        e_d = float(np.clip(0.04 * a + b, 0, 1)[0])
        expected = (1 - w) * base[0] * (1 - e_d) + w * np.array(self.WARM) * (1 - e_d) * (1 - 0.16 * 0.7) + e_d
        got = ref.declared_environment_radiance(base, np.array([0.0]), rough, mu, 1.0, ref.DECLARED,
                                                thin=thin_pixels(1, w, self.WARM))[0]
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        # declared against exact, sphere-weighted mean relative error over N.V >= 0.1 (a fit, bounded like the base environment, 5 %)
        mus = np.linspace(0.1, 1.0, 40)
        weights = mus / mus.sum()
        worst = 0.0
        for rough_value in (0.2, 0.5, 0.9):
            for weight in (0.25, 0.5, 1.0):
                args = (np.tile(self.BASE, (mus.size, 1)), np.zeros(mus.size), np.full(mus.size, rough_value), mus, 1.0)
                thin = thin_pixels(mus.size, weight, self.WARM)
                exact = ref.environment_radiance(*args, thin_env=thin)
                declared = ref.declared_environment_radiance(*args, ref.DECLARED, thin=thin)
                error = float((np.abs(declared - exact).max(axis=1) / np.maximum(exact.max(axis=1), 1e-3) * weights).sum())
                worst = max(worst, error)
        print(f"declared vs exact thin-subsurface environment, worst sphere-weighted mean relative error: {worst:.4f}")
        self.assertLess(worst, 0.05)

    def test_record_word_decoding_refusals_and_uses(self):
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731

        def record(mode=0, weight=0.5, color=(1.0, 0.5, 0.25), anisotropy=0.0, textures=(ref.INVALID,) * 3):
            w = [0] * 20
            w[0], w[1] = mode, f(weight)
            w[2:5] = [f(c) for c in color]
            w[5] = f(1.0)
            w[6:9] = [f(1.0), f(0.5), f(0.25)]
            w[9] = f(anisotropy)
            w[10:13] = list(textures)
            return w

        record0 = [0] * 20
        for k in (5, 6, 10, 11, 12, 17, 18):
            record0[k] = ref.INVALID
        zero = [record0, [0] * 20]
        groups = [record(),record(mode=1), record(weight=0.9, color=(2.0, 0.0, 0.1)), record(anisotropy=0.4),
                  record(textures=(3, ref.INVALID, ref.INVALID)), record(textures=(ref.INVALID, ref.INVALID, 4)),
                  record(weight=1.5), record(color=(-0.1, 0.0, 0.0))]
        records = []
        for g in groups:
            records += zero + [g, [0] * 20]
        materials = [{"feature_bits": 128, "extension_index": 4 * k} for k in range(len(groups))]
        materials += [{"feature_bits": 0}, {"feature_bits": 128 | 16, "extension_index": 0}]
        data = {"materials": materials, "material_extensions": {"version": 1, "records_per_material": 4, "records": records}}
        weight, color = ref.material_subsurface(data, materials[0])
        self.assertEqual((weight, color), (0.5, [1.0, 0.5, 0.25]))
        weight2, color2 = ref.material_subsurface(data, materials[2])                  # a colour above 1 is legal
        self.assertAlmostEqual(weight2, 0.9, places=6)
        np.testing.assert_allclose(color2, [2.0, 0.0, 0.1], rtol=1e-7)
        self.assertIsNone(ref.material_subsurface(data, materials[len(groups)]))
        for index in range(1, len(groups)):
            if index == 2:
                continue
            with self.assertRaises(ValueError, msg=index):
                ref.material_subsurface(data, materials[index])
        ref.material_subsurface(data, materials[len(groups) + 1])  # a combination with iridescence is composed (ticket 16 closure)
        self.assertTrue(ref.uses_subsurface(data))
        self.assertEqual((ref.SUBSURFACE_FEATURE_BIT, ref.SUBSURFACE_RECORD), (128, 2))  # MaterialFeature::Subsurface, GPUMaterialExt2
        # a subsurface-only material is not mistaken for the other features
        self.assertIsNone(ref.material_specular(data, materials[0]))
        self.assertIsNone(ref.material_iridescence(data, materials[0]))
        self.assertIsNone(ref.material_anisotropy(data, materials[0]))


def write_quad_ply(path, half=0.4):
    """A quad in the z = 0 plane facing +z (two triangles, vertex normals +z)."""
    verts = [(-half, -half, 0, 0, 0, 1, 0, 0), (half, -half, 0, 0, 0, 1, 1, 0), (half, half, 0, 0, 0, 1, 1, 1),
             (-half, half, 0, 0, 0, 1, 0, 1)]
    faces = [(0, 1, 2), (0, 2, 3)]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\n"
                 "element face 2\nproperty list uchar uint vertex_indices\nend_header\n").encode())
        for v in verts:
            f.write(struct.pack("<8f", *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


@unittest.skipUnless(HAVE_MITSUBA, "Mitsuba is required to ray cast the snapshot")
class ThinSubsurfaceSnapshot(unittest.TestCase):
    """One sphere (or quad) whose material carries the thin subsurface block in record 2 (the exporter's keys)."""

    def snapshot(self, directory, sky, sun, metallic=0.0, weight=0.5, color=(1.0, 0.55, 0.25), roughness=0.5, point=None, quad=False):
        root = RayCastedSnapshot("snapshot").snapshot(directory, sky, sun, metallic, roughness, [0.3, 0.5, 0.8])
        path = pathlib.Path(root) / "scene.json"
        data = json.loads(path.read_text())
        f = lambda x: int.from_bytes(struct.pack("<f", x), "little")  # noqa: E731
        record0 = [0] * 20
        for k in (5, 6, 10, 11, 12, 17, 18):
            record0[k] = ref.INVALID
        record2 = [0] * 20
        record2[1] = f(weight)
        record2[2:5] = [f(c) for c in color]
        record2[10:13] = [ref.INVALID] * 3
        data["materials"][0].update(feature_bits=128, extension_index=0, flags=128 << 8)
        data["material_extensions"] = {"version": 1, "records_per_material": 4,
                                       "records": [record0, [0] * 20, record2, [0] * 20]}
        if quad:
            write_quad_ply(pathlib.Path(root) / "mesh_0.ply")
        if point:
            data["lights"].append({"type": 1, "position": point, "direction": [0, 0, -1], "color": [1, 1, 1], "intensity": 60.0,
                                   "range": 100.0, "inner": 0, "outer": 0})
        path.write_text(json.dumps(data))
        return root

    def test_sun_behind_lights_the_visible_face_through_the_transmitted_lobe(self):
        from f12_reference import snapshot as load
        toward = unit(-0.4, 0.3, -0.85)
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, 0.0, (toward, 3.0)))
            cast = ref.cast_primary_rays(root, data, 2)
            images = {}
            for name in ("spec", *ref.THIN_CONTROLS):
                model = ref.APPENDIX_B if name == "spec" else ref.control_model(name, ref.APPENDIX_B)
                image, masks, spheres, terms = ref.render_frame(root, data, model, cast=cast)
                images[name] = image
            self.assertEqual(terms, ["sun-disk"])
            self.assertEqual(spheres[0]["thin_weight"], 0.5)
            self.assertEqual(spheres[0]["group"], "metallic 0 thin weight 0.5")
            roi = masks["roi"]
            # the whole visible face of the sphere is lit through the transmitted lobe (w > 0)
            self.assertGreater(int(masks["lit"][roi].sum()), int(roi.sum()) * 0.95)
            for name in ref.THIN_CONTROLS:
                self.assertGreater(float(np.abs(images[name][roi] - images["spec"][roi]).max()), 1e-4, name)
            # centre pixel: normal +z, V ~ +z: the closed form of the transmitted lobe plus the (unlit) reflected one
            centre = images["spec"][72, 128]
            n, v = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0]])
            expected = ref.thin_transmitted(thin_pixels(1, 0.5, (1.0, 0.55, 0.25)), np.array([0.0]), n, v,
                                            np.array([toward]))[0] * 3.0
            np.testing.assert_allclose(centre, expected, rtol=5e-2)
            # the control dropping the weight keeps only the rim that the sun reaches from the front: the rest is black
            self.assertEqual(float(images["transmission-ignored"][roi].min()), 0.0)
            self.assertGreater(float(images["spec"][roi].min()), 0.0)

    def test_render_environment_declared_and_the_point_light(self):
        from f12_reference import snapshot as load
        toward = unit(-0.5, 0.4, -0.7)
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, 0.5, (toward, 3.0), point=[-2.0, 1.0, -3.0]))
            self.assertTrue(ref.uses_subsurface(data) and not ref.uses_specular_extension(data))
            cast = ref.cast_primary_rays(root, data, 2)
            image, masks, _, terms = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast)
            self.assertEqual(terms, ["point-light", "sun-disk", "environment-quadrature"])
            declared, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast, environment=ref.DECLARED)
            blind, *_ = ref.render_frame(root, data, ref.APPENDIX_B, cast=cast,
                                         environment=ref.THIN_DECLARED_CONTROLS["declared-transmission-ignored"])
            roi = masks["roi"]
            self.assertGreater(float(np.abs(declared[roi] - blind[roi]).max()), 1e-3)

    def test_shadow_rays_of_a_thin_carrier_start_on_the_side_of_the_light(self):
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(self.snapshot(tmp, 0.0, None, quad=True, point=[0.0, 0.0, -5.0]))
            data["camera"]["position"] = [0, 0, 3]
            cast = ref.cast_primary_rays(root, data, 2)
            valid = cast[0]
            self.assertGreater(int(valid.sum()), 500)
            behind = ref.point_light_visibility(root, data, cast, 2)[0]
            self.assertTrue(bool(behind.all()))                  # thin: the quad does not occlude its own transmitted light
            data["materials"][0].update(feature_bits=0, flags=0)  # a plain (opaque) quad blocks a light behind it
            plain = ref.point_light_visibility(root, data, cast, 2)[0]
            self.assertFalse(bool(plain.any()))

    def test_command_writes_controls_and_declared_controls(self):
        toward = unit(-0.4, 0.3, -0.85)
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 0.5, (toward, 3.0))
            out = pathlib.Path(tmp) / "out"
            code = ref.main(["render", str(root), "--out", str(out), "--controls", "transmission-ignored", "transmission-front-only",
                             "--declared-controls", "declared-transmission-ignored", "declared-transmission-double-counted"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertEqual(sorted(report["controls"]), ["transmission-front-only", "transmission-ignored"])
            self.assertEqual(report["controls"]["transmission-front-only"]["model_variant"]["thin"], "front-only")
            self.assertEqual(sorted(report["declared"]["controls"]),
                             ["declared-transmission-double-counted", "declared-transmission-ignored"])
            self.assertIn("thin-subsurface", report["model"])

    def test_textured_volumetric_and_anisotropic_blocks_are_refused(self):
        from f12_reference import snapshot as load
        for word, value in ((10, 3), (0, 1), (9, int.from_bytes(struct.pack("<f", 0.3), "little"))):
            with tempfile.TemporaryDirectory() as tmp:
                root = self.snapshot(tmp, 0.5, None)
                path = pathlib.Path(root) / "scene.json"
                data = json.loads(path.read_text())
                data["material_extensions"]["records"][2][word] = value
                path.write_text(json.dumps(data))
                root, data = load(root)
                with self.assertRaises(ValueError, msg=word):
                    ref.require_declared_scene(data)
                self.assertEqual(ref.main(["render", str(root), "--out", str(pathlib.Path(tmp) / "out")]), 2)


# --- F15.3 (ticket 19): critical lobe combinations ---------------------------------------------------------------------------

def dielectric_dict(count, ior=1.5, weight=1.0, color=(1.0, 1.0, 1.0), variant="extension"):
    """Per-pixel ior/specular arrays of `count` identical pixels (see ref.dielectric_arrays)."""
    f0 = ref.specular_f0(ior, np.asarray(color, dtype=np.float64))
    return {"f0": np.tile(f0, (count, 1)), "weight": np.full(count, float(weight)), "variant": variant}


class CombinationModel(unittest.TestCase):
    """Combinations of the lobes (ticket 19), from the layering text of research/khr-materials.md section 12: clearcoat over
    sheen over the base; iridescence replaces the base specular Fresnel (with the ior/specular f0 under the film); anisotropy
    changes D and V of the base specular lobe; the thin subsurface transmitted lobe is part of the base under the coat.  Hand
    values assembled in the test, identities when one lobe is neutral, an independent wi-space Monte Carlo for the exact
    environment, furnace bounds and the discrimination of every control."""

    BASE = np.array([[0.4, 0.25, 0.1]])

    def setUp(self):
        self.n = np.array([[0.0, 0.0, 1.0]])
        self.v = np.array([unit(0.6, 0.1, 0.5)])
        self.l = np.array([unit(-0.7, 0.2, 0.3)])
        self.rough = np.array([0.45])

    def f(self, metallic=0.0, base=None, **kw):
        base = self.BASE if base is None else base
        return ref.bsdf(base, np.array([metallic]), self.rough, self.n, self.v, self.l, **kw)[0]

    # -- identities: a neutral lobe leaves the other exactly as it is ----------------------------------------------------------
    def test_a_neutral_lobe_reduces_every_pair_to_the_single_lobe(self):
        lobes = {"coat": coat_dict(0.7, 0.3), "sheen": sheen_dict((0.9, 0.6, 0.3), 0.4), "aniso": aniso_pixels(0.8, 0.6),
                 "irid": irid_pixels(1, 1.0, 1.4, 500.0, nv=float(self.v[0] @ self.n[0])), "dielectric": dielectric_dict(1, 1.8, 0.7, (1.0, 0.8, 0.6))}
        neutral = {"coat": coat_dict(0.0, 0.3), "sheen": sheen_dict((0.0, 0.0, 0.0), 0.4), "aniso": aniso_pixels(0.0, 0.6),
                   "irid": irid_pixels(1, 0.0, 1.4, 500.0, nv=float(self.v[0] @ self.n[0])),
                   "dielectric": dielectric_dict(1)}
        for metallic in (0.0, 1.0):
            for a in lobes:
                for b in lobes:
                    if a >= b:
                        continue
                    only_a = self.f(metallic, **{a: lobes[a]})
                    # lobe b neutral next to lobe a == lobe a alone (the neutral ior/specular block is the base f0 0.04 weight 1)
                    both = self.f(metallic, **{a: lobes[a], b: neutral[b]})
                    np.testing.assert_allclose(both, only_a, rtol=1e-11, err_msg=f"{a} + neutral {b}, metallic {metallic}")
                    both = self.f(metallic, **{b: lobes[b], a: neutral[a]})
                    np.testing.assert_allclose(both, self.f(metallic, **{b: lobes[b]}), rtol=1e-11,
                                               err_msg=f"neutral {a} + {b}, metallic {metallic}")

    def test_every_pair_of_active_lobes_differs_from_each_lobe_alone(self):
        # The identities above must not hold vacuously: with both lobes active the value moves away from both single lobes.
        lobes = {"coat": coat_dict(0.7, 0.3), "sheen": sheen_dict((0.9, 0.6, 0.3), 0.4), "aniso": aniso_pixels(0.8, 0.6),
                 "irid": irid_pixels(1, 1.0, 1.4, 500.0, nv=float(self.v[0] @ self.n[0])), "dielectric": dielectric_dict(1, 1.8, 0.7, (1.0, 0.8, 0.6))}
        for a in lobes:
            for b in lobes:
                if a >= b:
                    continue
                both = self.f(0.0, **{a: lobes[a], b: lobes[b]})
                for single in (a, b):
                    alone = self.f(0.0, **{single: lobes[single]})
                    self.assertGreater(float(np.abs(both - alone).max() / alone.max()), 1e-3, (a, b, single))

    def test_thin_with_coat_reduces_to_each_single_lobe(self):
        thin = thin_pixels(1, 0.6, (1.0, 0.55, 0.25))
        no_thin = thin_pixels(1, 0.0, (1.0, 0.55, 0.25))
        for light in (self.l, np.array([unit(-0.5, 0.2, -0.7)])):
            def radiance(coat=None, thin=None):
                position = np.array([[0.0, 0.0, 0.0]])
                lamp = {"position": (position + 3.0 * light)[0], "intensity": 10.0, "range": 100.0, "color": (1.0, 1.0, 1.0)}
                return ref.point_radiance(self.BASE, np.array([0.0]), self.rough, self.n, self.v, position, lamp, coat=coat, thin=thin)[0][0]
            np.testing.assert_allclose(radiance(coat_dict(0.0, 0.3), thin), radiance(None, thin), rtol=1e-12)
            np.testing.assert_allclose(radiance(coat_dict(0.7, 0.3), no_thin), radiance(coat_dict(0.7, 0.3)), rtol=1e-12)

    # -- hand values of the layering ---------------------------------------------------------------------------------------
    def test_coat_over_sheen_hand_value(self):
        # f = (1 - c F_c) (f_sheen + f_material scaling) + c F_c D Vis_coat: assembled here from the parts.
        c, rc, color, rs = 0.8, 0.3, np.array([0.9, 0.6, 0.3]), 0.5
        sheen = sheen_dict(color, rs)
        n, v, l = self.n, self.v, self.l
        nv, nl = float(v[0] @ n[0]), float(l[0] @ n[0])
        h = unit(*(v[0] + l[0]))
        nh = float(h @ n[0])
        material = self.f(0.0)
        e_v = float(ref.sheen_albedo(np.array([nv]), np.array([rs]))[0])
        e_l = float(ref.sheen_albedo(np.array([nl]), np.array([rs]))[0])
        scaling = min(1 - color.max() * e_v, 1 - color.max() * e_l)
        alpha_g = rs ** 2
        lobe = color * float(ref.charlie_d(nh, alpha_g) * ref.charlie_visibility(nv, nl, alpha_g))
        fc = 0.04 + 0.96 * (1 - nv) ** 5
        coat_value = ClearcoatModel.d_vis(n[0], v[0], l[0], rc ** 2)
        expected = (1 - c * fc) * (lobe + material * scaling) + c * fc * coat_value
        np.testing.assert_allclose(self.f(0.0, coat=coat_dict(c, rc), sheen=sheen), expected, rtol=1e-11)

    def test_swapped_order_differs_and_is_the_sheen_over_coat_formula(self):
        c, rc, color, rs = 1.0, 0.3, np.array([1.0, 1.0, 1.0]), 0.4
        sheen, coat = sheen_dict(color, rs), coat_dict(c, rc)
        normal = self.f(0.0, coat=coat, sheen=sheen)
        swapped = self.f(0.0, coat=coat, sheen=sheen, model={**ref.SPECULAR_EXT, "order": "swapped"})
        self.assertGreater(float(np.abs(normal - swapped).max() / normal.max()), 0.01)
        # the swapped value is f_sheen + (coated material) * scaling: assembled from the single-layer values
        alone_sheen = self.f(0.0, sheen=sheen)
        alone_material = self.f(0.0)
        coated = self.f(0.0, coat=coat)
        nv, nl = float(self.v[0] @ self.n[0]), float(self.l[0] @ self.n[0])
        e_v = float(ref.sheen_albedo(np.array([nv]), np.array([rs]))[0])
        e_l = float(ref.sheen_albedo(np.array([nl]), np.array([rs]))[0])
        scaling = min(1 - e_v, 1 - e_l)
        lobe = alone_sheen - alone_material * scaling
        np.testing.assert_allclose(swapped, lobe + coated * scaling, rtol=1e-10)

    def test_swapped_order_differs_in_the_sun_and_in_both_environments(self):
        coat, sheen = coat_dict(1.0, 0.3), sheen_dict((1.0, 0.5, 0.25), 0.4)
        toward = unit(-0.7, 0.2, 0.3)
        args = (self.BASE, np.array([0.0]), self.rough, self.n, self.v, toward, np.ones(3), 0.00465)
        normal, _ = ref.sun_radiance(*args, coat=coat, sheen=sheen)
        swapped, _ = ref.sun_radiance(*args, model={**ref.SPECULAR_EXT, "order": "swapped"}, coat=coat, sheen=sheen)
        self.assertGreater(float(np.abs(normal - swapped).max() / normal.max()), 0.005)
        mu = np.array([0.3])
        material = ref.environment_radiance(self.BASE, np.array([0.0]), self.rough, mu, 0.5)
        a = ref.layered_environment(material, mu, coat, sheen, 0.5)
        b = ref.layered_environment(material, mu, coat, sheen, 0.5, swapped=True)
        self.assertGreater(float(np.abs(a - b).max() / a.max()), 0.005)
        decl = ref.declared_environment_radiance(self.BASE, np.array([0.0]), self.rough, mu, 0.5, ref.DECLARED, None, coat, sheen)
        decl_s = ref.declared_environment_radiance(self.BASE, np.array([0.0]), self.rough, mu, 0.5, ref.COMBO_DECLARED_CONTROLS["declared-order-swapped"],
                                                   None, coat, sheen)
        self.assertGreater(float(np.abs(decl - decl_s).max() / decl.max()), 0.005)

    def test_the_sun_term_of_the_combination_matches_the_point_evaluation_for_a_wide_lobe(self):
        # The sun disk of every combination at a wide lobe is the point evaluation of the same BSDF at the disk centre (the
        # disk rule only changes narrow lobes): ties sun_radiance to bsdf for coat + sheen + anisotropy + iridescence together.
        toward = unit(-0.7, 0.2, 0.3)
        lobes = dict(coat=coat_dict(0.6, 0.5), sheen=sheen_dict((0.8, 0.4, 0.2), 0.5), aniso=aniso_pixels(0.7, 0.5),
                     irid=irid_pixels(1, 1.0, 1.3, 400.0, nv=float(self.v[0] @ self.n[0])), dielectric=dielectric_dict(1, 1.8, 0.7))
        irid_less = dict(lobes)
        del irid_less["irid"]
        for metallic in (0.0, 1.0):
            for use in (lobes, irid_less):
                sun, narrow = ref.sun_radiance(self.BASE, np.array([metallic]), self.rough, self.n, self.v, toward, np.ones(3), 0.00465,
                                               **use)
                point = ref.bsdf(self.BASE, np.array([metallic]), self.rough, self.n, self.v, np.array([toward]), **use)[0] * float(toward @ self.n[0])
                self.assertFalse(bool(narrow[0]))
                np.testing.assert_allclose(sun[0], point, rtol=1e-9)  # wide lobes: the disk rule is the point evaluation

    # -- iridescence with ior/specular ------------------------------------------------------------------------------------------
    def test_iridescence_over_the_specular_extension_hand_value(self):
        # README: rgb_mix(base, spec, mix(w fr, F_irid(base_f0 = w f0), s)); with s = 1: spec weight F_irid(w f0, V.H), diffuse 1 - max.
        ior, weight, color = 1.8, 0.7, (1.0, 0.8, 0.6)
        f0 = ref.specular_f0(ior, np.array(color))
        h = unit(*(self.v[0] + self.l[0]))
        vh = float(self.v[0] @ h)
        film = ref.thin_film_fresnel(np.array([vh]), np.array([1.4]), np.array([500.0]), (weight * f0)[None, :])
        diffuse_w, spec_w = ref.lobe_weights(self.BASE, np.array([0.0]), np.array([vh]), ref.SPECULAR_EXT,
                                             dielectric_dict(1, ior, weight, color), irid_pixels(1, 1.0, 1.4, 500.0))
        np.testing.assert_allclose(spec_w, film, rtol=1e-12)
        np.testing.assert_allclose(diffuse_w, 1 - film.max(axis=-1, keepdims=True), rtol=1e-12)
        # strength 0: the extended Fresnel alone (w fr, diffuse 1 - w max fr)
        diffuse_0, spec_0 = ref.lobe_weights(self.BASE, np.array([0.0]), np.array([vh]), ref.SPECULAR_EXT,
                                             dielectric_dict(1, ior, weight, color), irid_pixels(1, 0.0, 1.4, 500.0))
        fr = weight * (f0 + (1 - f0) * (1 - vh) ** 5)
        np.testing.assert_allclose(spec_0, fr[None, :], rtol=1e-12)
        np.testing.assert_allclose(diffuse_0, 1 - fr.max(), rtol=1e-12)

    def test_the_film_takes_the_scaled_f0_and_a_metal_ignores_the_extension(self):
        weak = dielectric_dict(1, 1.5, 0.3)
        strong = dielectric_dict(1, 1.5, 1.0)
        irid = irid_pixels(1, 1.0, 1.3, 300.0)
        _, w_weak = ref.lobe_weights(self.BASE, np.array([0.0]), np.array([0.6]), ref.SPECULAR_EXT, weak, irid)
        _, w_strong = ref.lobe_weights(self.BASE, np.array([0.0]), np.array([0.6]), ref.SPECULAR_EXT, strong, irid)
        self.assertGreater(float(np.abs(w_weak - w_strong).max()), 1e-3)  # base_f0 = weight * f0 reaches the film
        _, m_weak = ref.lobe_weights(self.BASE, np.array([1.0]), np.array([0.6]), ref.SPECULAR_EXT, weak, irid)
        _, m_strong = ref.lobe_weights(self.BASE, np.array([1.0]), np.array([0.6]), ref.SPECULAR_EXT, strong, irid)
        np.testing.assert_allclose(m_weak, m_strong, rtol=1e-13)

    def test_the_exact_environment_of_iridescence_with_a_neutral_extension_is_the_iridescence_only_value(self):
        mu = np.array([0.35, 0.8])
        irid = irid_pixels(2, 1.0, 1.3, 400.0, nv=mu)
        n = 2
        plain = ref.environment_radiance(np.tile(self.BASE, (n, 1)), np.zeros(n), np.full(n, 0.3), mu, 1.0, irid_env=irid)
        ext = ref.environment_radiance(np.tile(self.BASE, (n, 1)), np.zeros(n), np.full(n, 0.3), mu, 1.0, ref.SPECULAR_EXT,
                                       dielectric_dict(n), irid_env=irid)
        np.testing.assert_allclose(ext, plain, rtol=1e-9)
        decl = ref.declared_environment_radiance(np.tile(self.BASE, (n, 1)), np.zeros(n), np.full(n, 0.3), mu, 1.0, ref.DECLARED,
                                                 None, irid=irid)
        decl_ext = ref.declared_environment_radiance(np.tile(self.BASE, (n, 1)), np.zeros(n), np.full(n, 0.3), mu, 1.0, ref.DECLARED,
                                                     dielectric_dict(n), irid=irid)
        np.testing.assert_allclose(decl_ext, decl, rtol=1e-12)

    def test_anisotropic_vh_histograms_reproduce_the_a_and_b_tables(self):
        # hist . (1 - w) = A and hist . w = B of the anisotropic quadrature (the film replaces w by F_irid(V.H))
        at, ab = ref.aniso_alphas(float(ref.effective_alpha(0.3)), 0.8)
        mus, phis, hist = ref.aniso_vh_histograms(float(at), float(ab))
        bins = np.linspace(0.0, 1.0, ref.IRID_BINS)
        w = ref.schlick_weight(bins)
        for i, j in ((10, 4), (40, 20), (64, 32)):
            a, b = ref._aniso_spec_integrals(float(mus[i]), float(phis[j]), float(at), float(ab))
            self.assertAlmostEqual(float(hist[i, j] @ (1 - w)), a, delta=2e-4)
            self.assertAlmostEqual(float(hist[i, j] @ w), b, delta=2e-4)

    def test_exact_environment_of_iridescence_with_anisotropy_agrees_with_a_monte_carlo(self):
        aniso = aniso_pixels(0.8, 0.6)
        for metallic, mu in ((0.0, 0.5), (1.0, 0.85)):
            v = np.array([[math.sqrt(1 - mu * mu), 0.0, mu]])
            n = np.array([[0.0, 0.0, 1.0]])
            irid = irid_pixels(1, 1.0, 1.4, 500.0, nv=mu)
            ai = ref.aniso_environment_inputs(aniso, np.array([0.3]), n, v)
            exact = ref.environment_radiance(self.BASE, np.array([metallic]), np.array([0.3]), np.array([mu]), 1.0,
                                             aniso_env=ai, irid_env=irid)[0]
            mean, sigma = self.mc_environment(mu, base=self.BASE, metallic=metallic, roughness=0.3, aniso=aniso, irid=irid)
            np.testing.assert_allclose(exact, mean, atol=6 * sigma.max() + 2e-3, err_msg=f"metallic {metallic} mu {mu}")
        # the film acts: the same pixel without the film differs
        plain = ref.environment_radiance(self.BASE, np.array([0.0]), np.array([0.3]), np.array([0.5]), 1.0, aniso_env=ai)[0]
        self.assertGreater(float(np.abs(plain - exact).max() / plain.max()), 0.01)

    def mc_environment(self, mu, count=300000, seed=5, **kw):
        """Independent wi-space estimate of the unit-radiance environment value: uniform sphere, bsdf cos on the front hemisphere."""
        rng = np.random.default_rng(seed)
        w = rng.normal(size=(count, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        v = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
        n = np.tile([0.0, 0.0, 1.0], (count, 1))
        vv = np.tile(v, (count, 1))
        base = np.tile(kw.pop("base"), (count, 1))
        metallic = np.full(count, kw.pop("metallic"))
        rough = np.full(count, kw.pop("roughness"))
        take_n = lambda d: None if d is None else {k: (np.repeat(x[:1], count, axis=0) if isinstance(x, np.ndarray) else x) for k, x in d.items()}  # noqa: E731
        kw = {k: take_n(x) for k, x in kw.items()}
        f = ref.bsdf(base, metallic, rough, n, vv, w, **kw) * np.clip(w[:, 2:3], 0.0, None)
        value = 4 * math.pi * f
        return value.mean(axis=0), value.std(axis=0) / math.sqrt(count)

    def test_exact_environment_of_iridescence_with_specular_agrees_with_a_monte_carlo(self):
        ior, weight, color = 1.8, 0.7, (1.0, 0.8, 0.6)
        for mu in (0.4, 0.9):
            mean, sigma = self.mc_environment(mu, base=self.BASE, metallic=0.0, roughness=0.45, model=ref.SPECULAR_EXT,
                                              dielectric=dielectric_dict(1, ior, weight, color), irid=irid_pixels(1, 1.0, 1.4, 500.0))
            exact = ref.environment_radiance(self.BASE, np.array([0.0]), np.array([0.45]), np.array([mu]), 1.0, ref.SPECULAR_EXT,
                                             dielectric_dict(1, ior, weight, color), irid_env=irid_pixels(1, 1.0, 1.4, 500.0, nv=mu))[0]
            np.testing.assert_allclose(exact, mean, atol=6 * sigma.max() + 2e-3, err_msg=f"mu {mu}")

    def test_exact_environment_of_the_coat_over_the_thin_lobe_agrees_with_a_monte_carlo(self):
        coat, thin = coat_dict(0.8, 0.35), thin_pixels(1, 0.6, (1.0, 0.55, 0.25))
        for mu in (0.4, 0.9):
            n = 1
            material = ref.environment_radiance(self.BASE, np.array([0.0]), np.array([0.45]), np.array([mu]), 1.0, thin_env=thin)
            exact = ref.layered_environment(material, np.array([mu]), coat, None, 1.0)[0]
            # the coat attenuates the whole base value including the transmitted lobe: the estimate adds the attenuated transmitted lobe
            mean, sigma = self.mc_environment_coated_thin(mu, coat, thin)
            np.testing.assert_allclose(exact, mean, atol=6 * sigma.max() + 3e-3, err_msg=f"mu {mu}")
            del n

    def mc_environment_coated_thin(self, mu, coat, thin, count=300000, seed=9):
        rng = np.random.default_rng(seed)
        w = rng.normal(size=(count, 3))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        v = np.tile([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu], (count, 1))
        n = np.tile([0.0, 0.0, 1.0], (count, 1))
        base = np.tile(self.BASE, (count, 1))
        metallic, rough = np.zeros(count), np.full(count, 0.45)
        rep = lambda d: {k: (np.repeat(x[:1], count, axis=0) if isinstance(x, np.ndarray) else x) for k, x in d.items()}  # noqa: E731
        c, t = rep(coat), rep(thin)
        front = ref.bsdf(base, metallic, rough, n, v, w, coat=c, thin=t) * np.clip(w[:, 2:3], 0.0, None)
        attenuation = ref.coat_lobe(c, n, v, w)[0][:, None]
        back = ref.thin_transmitted(t, metallic, n, v, w) * attenuation
        value = 4 * math.pi * (front + back)
        return value.mean(axis=0), value.std(axis=0) / math.sqrt(count)

    # -- thin subsurface under the coat ---------------------------------------------------------------------------------------
    def test_the_coat_attenuates_the_transmitted_lobe_and_the_uncoated_control_does_not(self):
        thin, coat = thin_pixels(1, 0.6, (1.0, 0.55, 0.25)), coat_dict(1.0, 0.3)
        back = np.array([unit(-0.5, 0.2, -0.7)])
        position = np.array([[0.0, 0.0, 0.0]])
        lamp = {"position": (position + 3.0 * back)[0], "intensity": 10.0, "range": 100.0, "color": (1.0, 1.0, 1.0)}

        def radiance(thin, coat):
            return ref.point_radiance(self.BASE, np.array([0.0]), self.rough, self.n, self.v, position, lamp, coat=coat, thin=thin)[0][0]
        plain = radiance(thin, None)
        fc = 0.04 + 0.96 * (1 - float(self.v[0] @ self.n[0])) ** 5
        np.testing.assert_allclose(radiance(thin, coat), (1 - fc) * plain, rtol=1e-12)  # a back light: no coat lobe, only the attenuation
        np.testing.assert_allclose(radiance({**thin, "variant": "uncoated"}, coat), plain, rtol=1e-12)
        # sun: same rule (the transmitted term is inside the base the coat attenuates)
        toward = unit(-0.5, 0.2, -0.7)
        args = (self.BASE, np.array([0.0]), self.rough, self.n, self.v, toward, np.ones(3), 0.00465)
        sun_plain, _ = ref.sun_radiance(*args, thin=thin)
        sun_coated, _ = ref.sun_radiance(*args, coat=coat, thin=thin)
        sun_uncoated, _ = ref.sun_radiance(*args, coat=coat, thin={**thin, "variant": "uncoated"})
        np.testing.assert_allclose(sun_coated, (1 - fc) * sun_plain, rtol=1e-10)
        np.testing.assert_allclose(sun_uncoated, sun_plain, rtol=1e-10)

    def test_thin_subsurface_with_every_other_lobe_is_composed(self):
        for other in (4, 8, 16, 1, 2):
            ref.check_combination(128 | other)
        ref.check_combination(128 | 1 | 2 | 4 | 8 | 16)
        with self.assertRaises(ValueError):   # an unknown bit (transmission, 32) is refused by material_extension_words
            ref.material_extension_words({"material_extensions": {"records": [[0] * 20]}}, {"feature_bits": 2 | 32, "extension_index": 0})

    def test_thin_transmitted_lobe_takes_the_diffuse_weight_of_the_extended_dielectric(self):
        # ticket 16 closure: under ior/specular the transmitted weight is (1 - metallic) (1 - max3 F_spec(V.H)) (the weight of the reflected diffuse),
        # not the default 0.04 Fresnel; without a dielectric the original expression stands.
        n = np.array([[0.0, 0.0, 1.0]] * 2)
        v = np.array([[0.0, 0.6, 0.8]] * 2)
        l = np.array([[0.3, -0.4, -0.8]] * 2) / np.linalg.norm([0.3, -0.4, -0.8])
        thin = {"weight": np.array([1.0, 1.0]), "color": np.ones((2, 3)), "variant": "spec"}
        plain = ref.thin_transmitted(thin, np.zeros(2), n, v, l)
        high = {"f0": np.tile(ref.specular_f0(2.4, [1, 1, 1]), (2, 1)), "weight": np.ones(2), "variant": "spec"}
        same = {"f0": np.full((2, 3), ref.DIELECTRIC_F0), "weight": np.ones(2), "variant": "spec"}
        np.testing.assert_allclose(ref.thin_transmitted(thin, np.zeros(2), n, v, l, ref.SPECULAR_EXT, same), plain, rtol=1e-12)
        self.assertTrue(np.all(ref.thin_transmitted(thin, np.zeros(2), n, v, l, ref.SPECULAR_EXT, high) < plain - 1e-3))
        # a metal transmits nothing
        self.assertEqual(float(ref.thin_transmitted(thin, np.ones(2), n, v, l, ref.SPECULAR_EXT, high).max()), 0.0)

    def test_exact_transmitted_albedo_tables_per_ior_and_per_film(self):
        # F15.3 closure (ticket 16): the exact environment of a thin material under ior/specular or a film uses its own transmitted albedo.
        # The generic quadrature reproduces the default table (f0 0.04, weight 1 + 1e-7 so that the shortcut is not taken) within 3e-4, a high
        # index lowers T (more reflected: 1 - max F smaller), a film changes it and strength 0 is the plain dielectric.
        mus, default = ref.thin_transmission_table()
        mus2, near = ref.thin_transmission_table_for([0.04] * 3, 1.0 + 1e-7)
        np.testing.assert_allclose(np.interp(mus, mus2, near), default, atol=3e-4)
        _, high = ref.thin_transmission_table_for(ref.specular_f0(2.4, [1, 1, 1]), 1.0)
        self.assertTrue(np.all(np.interp(mus, mus2, high) < default - 1e-3))
        film = (1.3, 400.0, 1.0, [0.04] * 3)
        _, filmed = ref.thin_transmission_table_for([0.04] * 3, 1.0, film)
        self.assertGreater(float(np.abs(np.interp(mus, mus2, filmed) - default).max()), 5e-3)
        _, off = ref.thin_transmission_table_for([0.04] * 3, 1.0, (1.3, 400.0, 0.0, [0.04] * 3))
        np.testing.assert_allclose(off, near if False else ref.thin_transmission_table_for([0.04] * 3, 1.0 + 1e-7)[1], atol=3e-4)
        # the exact environment takes it: a high-index dielectric under a thin lobe transmits less than the default one
        mu = np.array([0.5])
        thin = {"weight": np.ones(1), "color": np.ones((1, 3)), "variant": "spec"}
        dielectric = {"f0": np.array([ref.specular_f0(2.4, [1, 1, 1])]), "weight": np.ones(1), "variant": "extension"}
        high_env = ref.environment_radiance(np.ones((1, 3)), np.zeros(1), np.full(1, 0.5), mu, 1.0, ref.SPECULAR_EXT, dielectric, thin_env=thin)
        plain_dielectric = {"f0": np.full((1, 3), ref.DIELECTRIC_F0), "weight": np.ones(1), "variant": "extension"}
        default_env = ref.environment_radiance(np.ones((1, 3)), np.zeros(1), np.full(1, 0.5), mu, 1.0, ref.SPECULAR_EXT, plain_dielectric, thin_env=thin)
        self.assertLess(float(high_env[0, 0]), float(default_env[0, 0]) + 0.5)  # both are finite and ordered with the extra specular share
        controlled = ref.environment_radiance(np.ones((1, 3)), np.zeros(1), np.full(1, 0.5), mu, 1.0, ref.SPECULAR_EXT, dielectric,
                                              thin_env={**thin, "variant": "default-fresnel"})
        self.assertGreater(abs(float(controlled[0, 0] - high_env[0, 0])), 1e-3)  # the control (default 0.04 weight) differs

    def test_declared_thin_albedo_fit_tracks_the_exact_table_and_keeps_the_radiance_within_the_bound(self):
        # ticket 16 closure: the 4-node declared transmitted albedo (THIN_ENV_*) against the exact table per ior / per film, and the declared radiance against the exact
        # one (sphere-weighted mean relative error, the 5 % bound of decision 11); the control `default-weight` (the old ratio fit) exceeds it at film 1.3 / 400 nm.
        mus = np.linspace(0.05, 1.0, 30)
        weight = 2 * mus
        n = len(mus)
        base = np.tile([0.3, 0.5, 0.8], (n, 1))
        rough = np.full(n, 0.3)
        f0 = np.array(ref.specular_f0(1.8, [1.0, 0.8, 0.6]))
        cases = {
            "film 1.3 / 400 nm": ({"strength": np.ones(n), "ior": np.full(n, 1.3), "thickness": np.full(n, 400.0), "thickness_min": np.full(n, 100.0), "nv": mus,
                                   "variant": "spec"}, None),
            "ior 1.8 / 0.7": (None, {"f0": np.tile(f0, (n, 1)), "weight": np.full(n, 0.7), "variant": "extension"}),
        }
        for name, (irid, dielectric) in cases.items():
            thin = {"weight": np.ones(n), "color": np.ones((n, 3)), "variant": "spec"}
            self.assertFalse(bool(np.any(ref.thin_env_is_default(dielectric, irid))), name)
            film = None if irid is None else (1.3, 400.0, 1.0, ref.irid_base_f0(base, dielectric)[0])
            mt, table = ref.thin_transmission_table_for(f0 if dielectric is not None else [0.04] * 3, 0.7 if dielectric is not None else 1.0, film)
            fit = ref.thin_env_albedo(base, mus, dielectric, irid)
            np.testing.assert_allclose(fit, np.interp(mus, mt, table), rtol=0.012, atol=2e-3, err_msg=name)
            exact = ref.environment_radiance(base, np.zeros(n), rough, mus, 1.0, ref.SPECULAR_EXT, dielectric, None, irid, thin)

            def gap(variant, kind="spec"):
                declared = ref.declared_environment_radiance(base, np.zeros(n), rough, mus, 1.0, variant, dielectric, irid=irid, thin={**thin, "variant": kind})
                err = np.abs(declared - exact).max(axis=1) / np.maximum(exact.max(axis=1), 1e-9)
                return float((err * weight).sum() / weight.sum())
            self.assertLess(gap(ref.DECLARED), 0.02, name)
            self.assertGreater(gap(ref.COMBO_DECLARED_CONTROLS["declared-transmission-default-weight"], "default-weight"), 0.04, name)
        self.assertTrue(bool(ref.thin_env_is_default({"f0": np.full((1, 3), 0.04), "weight": np.ones(1), "variant": "extension"}, None)[0]))
        # per pixel: a default pixel next to a film pixel keeps the old ratio fit
        mixed = {"strength": np.array([0.0, 1.0]), "ior": np.full(2, 1.3), "thickness": np.full(2, 400.0), "thickness_min": np.full(2, 100.0), "nv": np.array([0.5, 0.5]), "variant": "spec"}
        self.assertEqual(ref.thin_env_is_default(None, mixed).tolist(), [True, False])

    @staticmethod
    def shader_thin_env_constants(text):
        """(nodes, polynomials) parsed from the bsdfThinEnvNode calls of bsdfThinEnvAlbedo in shaders/materials/bsdf.h."""
        import re
        body = text[text.index("inline float bsdfThinEnvAlbedo"):]
        body = body[:body.index("return max(t")]
        calls = re.findall(r"bsdfThinEnvNode\(s, irid, ([-0-9.]+)f, ([-0-9.]+)f, ([-0-9.]+)f, ([-0-9.]+)f, ([-0-9.]+)f, mu\)", body)
        return tuple(float(c[0]) for c in calls), tuple(tuple(float(v) for v in c[1:]) for c in calls)

    def test_thin_env_fit_constants_of_the_shader_equal_the_reference(self):
        # The declared thin transmitted albedo (ticket 16 closure) lives twice: THIN_ENV_NODES / THIN_ENV_POLY here and as literals in bsdf.h.
        text = (ref.REPO / "shaders" / "materials" / "bsdf.h").read_text()
        nodes, poly = self.shader_thin_env_constants(text)
        self.assertEqual(len(nodes), len(ref.THIN_ENV_NODES))
        np.testing.assert_allclose(nodes, ref.THIN_ENV_NODES, rtol=0, atol=1e-6)
        np.testing.assert_allclose(poly, ref.THIN_ENV_POLY, rtol=0, atol=1e-8)  # the shader keeps 9 decimals
        # negative control: one perturbed literal must fail the comparison
        bad = text.replace("-0.093795337f", "-0.093795437f", 1)
        self.assertNotEqual(bad, text)
        bad_nodes, bad_poly = self.shader_thin_env_constants(bad)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(bad_poly, ref.THIN_ENV_POLY, rtol=0, atol=1e-8)
        bad_node = text.replace("bsdfThinEnvNode(s, irid, 0.4f,", "bsdfThinEnvNode(s, irid, 0.41f,", 1)
        self.assertNotEqual(bad_node, text)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(self.shader_thin_env_constants(bad_node)[0], ref.THIN_ENV_NODES, rtol=0, atol=1e-6)

    def test_sheen_scales_the_transmitted_lobe_of_a_back_light_with_the_smallest_n_dot_l(self):
        # the glTF Sample Renderer scales the whole base by min(1 - m E(N.V), 1 - m E(N.L)) with N.L = 0 behind the surface: E at the table minimum
        base, metallic, rough = np.ones((1, 3)), np.zeros(1), np.full(1, 0.5)
        n, v = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.6, 0.8]])
        position, light = np.zeros((1, 3)), {"position": [0.6, -0.8, -1.6], "color": [1, 1, 1], "intensity": 10.0, "range": 100.0}
        thin = {"weight": np.ones(1), "color": np.ones((1, 3)), "variant": "spec"}
        sheen = {"color": np.full((1, 3), 0.8), "roughness": np.full(1, 0.5), "variant": "spec"}
        plain, _ = ref.point_radiance(base, metallic, rough, n, v, position, light, thin=thin)
        layered, _ = ref.point_radiance(base, metallic, rough, n, v, position, light, sheen=sheen, thin=thin)
        e_v = float(ref.sheen_albedo(np.array([0.8]), np.array([0.5]))[0])
        e_l = float(ref.sheen_albedo(np.array([ref.THIN_BACK_NDOTL]), np.array([0.5]))[0])
        np.testing.assert_allclose(layered, plain * min(1 - 0.8 * e_v, 1 - 0.8 * e_l), rtol=1e-10)
        self.assertTrue(float(layered.max()) < float(plain.max()))

    # -- energy ---------------------------------------------------------------------------------------------------------------
    def white_furnace(self, mu, metallic, coat=None, sheen=None, irid=None, thin=None, variant=None, rough=0.4):
        """Exact-quadrature furnace value (radiance 1, albedo 1) of one pixel and the same of the base material alone."""
        base = np.ones((1, 3))
        args = (base, np.array([metallic]), np.array([rough]), np.array([mu]), 1.0)
        material = ref.environment_radiance(*args, irid_env=irid, thin_env=thin)
        model = {**ref.SPECULAR_EXT, **(variant or {})}
        coat_v = None if coat is None else {**coat, "variant": model.get("coat", "spec")}
        sheen_v = None if sheen is None else {**sheen, "variant": model.get("sheen", "spec")}
        return ref.layered_environment(material, np.array([mu]), coat_v, sheen_v, 1.0, model.get("order") == "swapped")[0], \
            ref.environment_radiance(*args)[0]

    def test_layering_never_adds_energy_to_the_exact_furnace_and_the_additive_control_does(self):
        coat, sheen = coat_dict(1.0, 0.3), sheen_dict((1.0, 1.0, 1.0), 0.4)
        additive = ref.COMBO_CONTROLS["additive-layering"]
        for metallic in (0.0, 1.0):
            for mu in (0.1, 0.3, 0.6, 1.0):
                layered, base = self.white_furnace(mu, metallic, coat, sheen)
                bound = max(1.0, float(base.max())) + 1e-9
                self.assertLessEqual(float(layered.max()), bound, (metallic, mu))
                swapped, _ = self.white_furnace(mu, metallic, coat, sheen, variant=ref.COMBO_CONTROLS["order-swapped"])
                self.assertLessEqual(float(swapped.max()), bound, (metallic, mu))
        summed, base = self.white_furnace(1.0, 0.0, coat, sheen, variant=additive)
        self.assertGreater(float(summed.max()), 1.04)  # lobes summed without the attenuation: far above the furnace
        layered, _ = self.white_furnace(1.0, 0.0, coat, sheen)
        self.assertLess(float(layered.max()), 1.0 + 1e-9)

    def test_declared_furnace_bound_of_the_combinations_and_their_additive_control(self):
        mu = np.linspace(0.05, 1.0, 20)
        n = len(mu)
        base = np.ones((n, 3))
        coat = {**coat_dict(1.0, 0.3, count=n)}
        sheen = {**sheen_dict((1.0, 1.0, 1.0), 0.4, count=n)}
        irid = irid_pixels(n, 1.0, 1.3, 400.0, nv=mu)
        for metallic in (0.0, 1.0):
            kwargs = dict(coat=coat, sheen=sheen, irid=irid)
            value = ref.declared_environment_radiance(base, np.full(n, metallic), np.full(n, 0.4), mu, 1.0, ref.DECLARED, None, **kwargs)
            self.assertLessEqual(float(value.max()), 1.0 + 1e-9, metallic)
            additive = ref.declared_environment_radiance(base, np.full(n, metallic), np.full(n, 0.4), mu, 1.0,
                                                         ref.COMBO_DECLARED_CONTROLS["declared-additive-layering"], None, **kwargs)
            self.assertGreater(float(additive.max()), 1.05, metallic)
            swapped = ref.declared_environment_radiance(base, np.full(n, metallic), np.full(n, 0.4), mu, 1.0,
                                                        ref.COMBO_DECLARED_CONTROLS["declared-order-swapped"], None, **kwargs)
            self.assertLessEqual(float(swapped.max()), 1.0 + 1e-9, metallic)

    def test_the_declared_furnace_of_thin_subsurface_with_coat_holds_and_its_additive_control_gains_energy(self):
        mu = np.linspace(0.05, 1.0, 20)
        n = len(mu)
        thin = thin_pixels(n, 0.7, (1.0, 1.0, 1.0))
        coat = coat_dict(1.0, 0.3, count=n)
        value = ref.declared_environment_radiance(np.ones((n, 3)), np.zeros(n), np.full(n, 0.4), mu, 1.0, ref.DECLARED, None, coat=coat, thin=thin)
        self.assertLessEqual(float(value.max()), 1.0 + 1e-9)
        additive = ref.declared_environment_radiance(np.ones((n, 3)), np.zeros(n), np.full(n, 0.4), mu, 1.0,
                                                     ref.COMBO_DECLARED_CONTROLS["declared-additive-layering"], None, coat=coat, thin=thin)
        self.assertGreater(float(additive.max()), 1.05)

    def test_the_exact_environment_keeps_the_thin_lobe_next_to_the_extended_dielectric(self):
        # regression: with a coat the main model carries the ior/specular dielectric arrays; the thin transmitted term must survive there
        mu = np.array([0.4, 0.9])
        thin = thin_pixels(2, 0.6, (1.0, 0.55, 0.25))
        base = np.tile(self.BASE, (2, 1))
        plain = ref.environment_radiance(base, np.zeros(2), np.full(2, 0.5), mu, 1.0, thin_env=thin)
        ext = ref.environment_radiance(base, np.zeros(2), np.full(2, 0.5), mu, 1.0, ref.SPECULAR_EXT, dielectric_dict(2), thin_env=thin)
        np.testing.assert_allclose(ext, plain, rtol=1e-9)
        ignored = ref.environment_radiance(base, np.zeros(2), np.full(2, 0.5), mu, 1.0, ref.SPECULAR_EXT, dielectric_dict(2),
                                           thin_env=thin_pixels(2, 0.6, (1.0, 0.55, 0.25), "ignored"))
        self.assertGreater(float(np.abs(ext - ignored).max() / ext.max()), 0.05)

    def test_the_exact_furnace_of_the_thin_lobe_under_the_coat_does_not_exceed_the_unextended_model(self):
        # reflected + transmitted never exceeds the unextended material (ticket 16), and the coat only attenuates it
        coat, thin = coat_dict(1.0, 0.3), thin_pixels(1, 1.0, (1.0, 1.0, 1.0))
        for mu in (0.2, 0.6, 1.0):
            layered, base = self.white_furnace(mu, 0.0, coat=coat, thin=thin)
            self.assertLessEqual(float(layered.max()), max(1.0, float(base.max())) + 1e-9, mu)

    # -- the controls discriminate ---------------------------------------------------------------------------------------------
    def test_the_per_lobe_controls_move_the_combination_value(self):
        base = np.ones((1, 3)) * 0.5
        lobes = dict(coat=coat_dict(1.0, 0.3), sheen=sheen_dict((1.0, 0.6, 0.3), 0.4), aniso=aniso_pixels(0.9, 0.7),
                     irid=irid_pixels(1, 1.0, 1.4, 500.0, nv=float(self.v[0] @ self.n[0])))
        reference = ref.bsdf(base, np.array([0.0]), self.rough, self.n, self.v, self.l, ref.SPECULAR_EXT, **lobes)
        for key, variant in (("coat", "ignored"), ("sheen", "ignored"), ("aniso", "ignored"), ("irid", "ignored")):
            changed = dict(lobes)
            changed[key] = {**lobes[key], "variant": variant}
            other = ref.bsdf(base, np.array([0.0]), self.rough, self.n, self.v, self.l, ref.SPECULAR_EXT, **changed)
            self.assertGreater(float(np.abs(other - reference).max() / reference.max()), 1e-3, key)
        additive = ref.bsdf(base, np.array([0.0]), self.rough, self.n, self.v, self.l,
                            {**ref.SPECULAR_EXT, **ref.COMBO_CONTROLS["additive-layering"]},
                            coat={**lobes["coat"], "variant": "base-unattenuated"}, sheen={**lobes["sheen"], "variant": "unscaled-base"},
                            aniso=lobes["aniso"], irid={**lobes["irid"], "variant": "diffuse-unweighted"})
        self.assertGreater(float(additive.max()), float(reference.max()))

    def test_control_models_and_declared_twins_are_registered(self):
        for name in ("additive-layering", "order-swapped", "transmission-uncoated"):
            self.assertIn(name, ref.ALL_CONTROLS)
            self.assertIn(name, ref.COMBO_CONTROLS)
        self.assertEqual(ref.control_model("additive-layering", ref.SPECULAR_EXT)["coat"], "base-unattenuated")
        self.assertEqual(ref.control_model("transmission-uncoated", ref.SPECULAR_EXT)["thin"], "uncoated")
        self.assertIn("declared-additive-layering", ref.ALL_DECLARED_CONTROLS)
        self.assertIn("declared-order-swapped", ref.ALL_DECLARED_CONTROLS)
        self.assertNotIn("declared-transmission-uncoated", ref.ALL_DECLARED_CONTROLS)


class TextureFixtures(unittest.TestCase):
    """The KHR texture fixtures section: sampler hand values, channel semantics and the control directions (F15.3 texture closure)."""

    @staticmethod
    def field(rgb, alpha, uv, foot=None):
        field = object.__new__(ref.TextureField)
        field.uv = np.asarray(uv, dtype=np.float64)
        field.foot = np.zeros_like(field.uv) if foot is None else np.asarray(foot, dtype=np.float64)
        field.images = {0: (np.asarray(rgb, dtype=np.float64), np.asarray(alpha, dtype=np.float64))}
        field.ambiguous = np.zeros(len(field.uv), dtype=bool)
        field._cache = {}
        return field

    def test_sampler_is_bilinear_with_repeat_and_half_rounding(self):
        rgb = np.zeros((2, 4, 3))
        rgb[:, :, 0] = [[0.0, 0.25, 0.5, 0.75], [1.0, 0.0, 0.0, 0.0]]
        alpha = np.zeros((2, 4))
        alpha[0, 1] = 1.0
        # texel centres are (i + 0.5) / size: u = 0.375 is the centre of texel 1, v = 0.25 the centre of row 0.
        f = self.field(rgb, alpha, [[0.375, 0.25], [0.5, 0.25], [0.5, 0.5], [0.0, 0.25], [0.2, 0.2]])
        got = f.sample_all(0)
        self.assertAlmostEqual(got[0, 0], 0.25, places=12)
        self.assertAlmostEqual(got[0, 3], 1.0, places=12)
        self.assertAlmostEqual(got[1, 0], 0.375, places=12)  # halfway between texel 1 (0.25) and 2 (0.5)
        self.assertAlmostEqual(got[2, 0], 0.5 * 0.375, places=12)  # halfway between rows 0 (0.375) and 1 (0) at u = 0.5
        self.assertAlmostEqual(got[3, 0], 0.5 * (0.75 + 0.0), places=12)  # u = 0 wraps between texel 3 and texel 0 (repeat)
        # every value is a binary16 number
        self.assertTrue(np.array_equal(got, got.astype(np.float16).astype(np.float64)))
        flipped = f.sample_all(0, flip_v=True)
        self.assertAlmostEqual(flipped[0, 0], 0.0, places=12)  # v = 0.75 is the centre of row 1, texel 1 there is 0
        self.assertFalse(np.array_equal(flipped, got))  # the v-flip control moves the lookup

    def test_footprint_beyond_half_a_texel_marks_the_pixel_ambiguous(self):
        rgb, alpha = np.zeros((16, 32, 3)), np.zeros((16, 32))
        f = self.field(rgb, alpha, [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]],
                       foot=[[0.4 / 32, 0.0], [0.6 / 32, 0.0], [0.0, 0.6 / 16]])
        f.use(0, np.arange(3))
        self.assertEqual(f.ambiguous.tolist(), [False, True, True])

    def test_srgb_encode_inverts_the_hardware_decode(self):
        x = np.array([0.0, 0.0031308, 0.2, 0.5, 1.0])
        decoded = np.where(ref.srgb_encode(x) <= 0.04045, ref.srgb_encode(x) / 12.92, ((ref.srgb_encode(x) + 0.055) / 1.055) ** 2.4)
        np.testing.assert_allclose(decoded, x, atol=1e-12)
        self.assertAlmostEqual(float(ref.srgb_encode(np.array([0.5]))[0]), 0.7353569830524495, places=9)

    def scene(self, feature_bit, record, word, **extra):
        words = [[0] * 32 for _ in range(3)]
        for r in range(3):
            words[r] = [int(ref.INVALID)] * 32
        words[record][word] = 0
        material = {"feature_bits": feature_bit, "extension_index": 0}
        return {"materials": [material], "material_extensions": {"records": words}}, material

    def test_texture_slots_read_only_the_records_of_the_feature_bits(self):
        data, material = self.scene(4, 0, 17)
        self.assertEqual(ref.texture_slots(data, material), {"sheen_color": 0})
        material["feature_bits"] = 8  # anisotropy only: the sheen word is not read
        self.assertEqual(ref.texture_slots(data, material), {})
        material["feature_bits"] = 0
        self.assertEqual(ref.texture_slot_bits(data, material), 0)

    def test_sheen_textures_substitute_colour_and_roughness_and_controls_move_them(self):
        data, _ = self.scene(4, 0, 17)
        data["material_extensions"]["records"][0][18] = 0
        rgb = np.zeros((1, 1, 3))
        rgb[0, 0] = [0.5, 0.25, 1.0]
        alpha = np.full((1, 1), 0.5)
        n = 3
        sheen = {"color": np.ones((n, 3)) * 0.8, "roughness": np.ones(n) * 0.5, "variant": "spec"}
        material_of_pixel = np.zeros(n, dtype=np.int64)
        v = np.tile([0.0, 0.0, 1.0], (n, 1))
        for mode, colour, roughness in (("spec", [0.4, 0.2, 0.8], 0.25), ("sheen-color-ignored", [0.8] * 3, 0.25),
                                        ("sheen-roughness-ignored", [0.4, 0.2, 0.8], 0.5)):
            f = self.field(rgb, alpha, [[0.5, 0.5]] * n)
            _, got, _, _, _ = ref.apply_textures(f, data, material_of_pixel, v, v, None, None, sheen, None, None, None, mode)
            np.testing.assert_allclose(got["color"][0], colour, atol=1e-3, err_msg=mode)
            self.assertAlmostEqual(float(got["roughness"][0]), roughness, delta=1e-3, msg=mode)
        encoded = ref.apply_textures(self.field(rgb, alpha, [[0.5, 0.5]] * n), data, material_of_pixel, v, v, None, None, sheen, None,
                                     None, None, "sheen-color-srgb-encoded")[1]
        self.assertGreater(float(encoded["color"][0, 0]), 0.5)  # 0.8 * srgb(0.5) = 0.588 > 0.4

    def test_aniso_direction_adds_the_texel_angle_to_the_rotation(self):
        data, _ = self.scene(8, 1, 2)
        rgb = np.zeros((1, 1, 3))
        rgb[0, 0] = [0.5, 1.0, 0.5]  # direction (0, 1): 90 degrees, strength 0.5
        n = 2
        aniso = {"strength": np.ones(n), "rotation": np.full(n, 0.25), "variant": "spec"}
        zeros = np.zeros((n, 3))
        for mode, rotation, strength in (("spec", 0.25 + math.pi / 2, 0.5), ("aniso-direction-ignored", 0.25, 0.5),
                                         ("aniso-strength-ignored", 0.25 + math.pi / 2, 1.0),
                                         ("aniso-direction-y-flipped", 0.25 - math.pi / 2, 0.5)):
            f = self.field(rgb, np.ones((1, 1)), [[0.5, 0.5]] * n)
            got = ref.apply_textures(f, data, np.zeros(n, dtype=np.int64), zeros, zeros, None, None, None, aniso, None, None, mode)[2]
            self.assertAlmostEqual(float(got["rotation"][0]), rotation, delta=2e-3, msg=mode)
            self.assertAlmostEqual(float(got["strength"][0]), strength, delta=2e-3, msg=mode)

    def test_every_texture_control_has_a_declared_twin(self):
        for name in ref.TEXTURE_CONTROLS:
            self.assertIn("declared-" + name, ref.TEXTURE_DECLARED_CONTROLS)
            self.assertIn(name, ref.ALL_CONTROLS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
