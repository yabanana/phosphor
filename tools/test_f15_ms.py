#!/usr/bin/env python3
"""Unit tests of tools/f15_ms.py (the independent reference of the multiple-scattering compensation, --specular-ms).

Criteria fixed BEFORE any comparison with the engine (decision 11), from independent arguments:
  * white furnace of a rough metal (f0 = f90 = 1, constant environment 1): |radiance - 1| <= 0.005.  The identity E + F_ms (1 - E) = 1
    holds for any E; the budget is the 0.2 % quadrature error of tools/ms_table.py --check and the engine's table interpolation.
  * no energy gain: every furnace value <= 1 + 0.005 (decision 11's furnace bound), for the dielectric (F0 0.04) at albedo 1 too.
  * declared (table) model against the literal quadrature: |difference of the extra term| <= 0.01 (1 % of the furnace radiance).
Negative controls: compensation off must FAIL the furnace gate (loss 59 % at roughness 1); F_ms doubled must FAIL the gain gate.

Run offline:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_ms.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import f15_reference as ref  # noqa: E402
import f15_ms as ms  # noqa: E402

FURNACE_TOLERANCE = 0.005
GAIN_BOUND = 1.005
DECLARED_TOLERANCE = 0.01
ROUGHNESS = (0.04, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
COSINES = (0.05, 0.3, 0.7, 1.0)


def furnace(base, metallic, r, mu, kind):
    """Exact-quadrature furnace radiance (material + extra lobe) of one (roughness, N.V) pair under a radiance-1 environment.  The
    reference's older environment quadrature (f15_reference.environment_table, 512 half-vector nodes) is 0.5 % off at grazing incidence
    (measured: E 0.9151 against 0.9201 at roughness 0.35, N.V 0.05) and is the single-scattering side of the gates against the engine
    (5 % bound); the furnace IDENTITY is therefore tested with the quadrature of the lost-energy reference itself for a white metal."""
    model = {**ref.APPENDIX_B, "ms": kind}
    b = np.full((1, 3), base)
    material = ref.environment_radiance(b, np.array([metallic]), np.array([r]), np.array([mu]), 1.0, model)
    if base == 1.0 and metallic == 1.0:  # white metal: F = 1, the single-scattering albedo is E itself
        grid, lost, _ = ms.exact_lost(r, model)
        material = np.full((1, 3), 1.0 - float(np.interp(mu, grid, lost)))
    extra = ms.environment_value(b, np.array([metallic]), np.array([r]), np.array([mu]), 1.0, model, None, kind) if kind != "off" else 0.0
    return float(np.max(material + extra))


class Furnace(unittest.TestCase):
    def test_rough_metal_reaches_one(self):
        for r in ROUGHNESS:
            for mu in COSINES:
                self.assertAlmostEqual(furnace(1.0, 1.0, r, mu, "kulla-conty"), 1.0, delta=FURNACE_TOLERANCE, msg=(r, mu))

    def test_negative_control_off_fails_the_furnace(self):
        loss = 1.0 - furnace(1.0, 1.0, 1.0, 0.7, "off")
        self.assertGreater(loss, FURNACE_TOLERANCE * 10)  # 59 % at roughness 1 (E_avg 0.41)
        self.assertTrue(any(abs(furnace(1.0, 1.0, r, 0.7, "off") - 1.0) > FURNACE_TOLERANCE for r in ROUGHNESS))

    def test_no_gain_metal_literal_and_every_material_declared(self):
        # the literal (Appendix B) albedo-1 dielectric already exceeds 1 without the extra lobe (diffuse weighted per light sample), so the
        # dielectric is gated on the declared model the engine's furnace uses; the metal on the exact quadrature
        for r in ROUGHNESS:
            for mu in COSINES:
                self.assertLessEqual(furnace(1.0, 1.0, r, mu, "kulla-conty"), GAIN_BOUND, msg=("literal metal", r, mu))
                for metallic in (0.0, 0.5, 1.0):
                    b, m, rr, v = np.ones((1, 3)), np.array([metallic]), np.array([r]), np.array([mu])
                    declared = ref.declared_environment_radiance(b, m, rr, v, 1.0, ref.DECLARED)  # the declared model carries the lobe (ADR-0007)
                    self.assertLessEqual(float(declared.max()), GAIN_BOUND, msg=("declared", metallic, r, mu))

    def test_a_fresnel_one_dielectric_does_not_reflect_the_lost_energy_twice(self):
        # ior 0 / saturated specular colour: F = 1 on a dielectric (metallic 0).  1 - E_d already refilled the single-scattering loss into the
        # diffuse base, so the lobe's F_ms L must come out of that complement (ADR-0007): the furnace stays at 1, not 1 + L.
        n = 1
        for r in (0.35, 0.7, 1.0):
            for mu in (0.3, 0.7, 1.0):
                b, m, rr, v = np.ones((n, 3)), np.zeros(n), np.array([r]), np.array([mu])
                dielectric = {"f0": np.ones((n, 3)), "weight": np.ones(n), "variant": "extension"}
                with_ms = ref.declared_environment_radiance(b, m, rr, v, 1.0, ref.DECLARED, dielectric)
                self.assertLessEqual(float(with_ms.max()), 1.0 + 1e-6, (r, mu))
                self.assertGreater(float(with_ms.min()), 1.0 - 1e-6, (r, mu))
                # the same lobe added without taking it from the complement would gain L: the control
                naive = ref.declared_environment_radiance(b, m, rr, v, 1.0, {**ref.DECLARED, "ms": "off"}, dielectric) + \
                    ms.declared_environment_value(b, m, rr, v, 1.0, "table", dielectric)
                self.assertGreater(float(naive.max()), 1.0 + 0.01 * (r > 0.5), (r, mu))

    def test_negative_control_double_gains_energy(self):
        self.assertGreater(furnace(1.0, 1.0, 1.0, 0.7, "double"), GAIN_BOUND)

    def test_dielectric_gain_is_below_two_tenths_of_a_percent(self):
        # Kulla-Conty: F_ms <= F_avg^2 for F0 = 0.04: the extra lobe of a dielectric adds at most ~0.16 %
        for r in ROUGHNESS:
            gain = furnace(0.0, 0.0, r, 0.7, "kulla-conty") - furnace(0.0, 0.0, r, 0.7, "off")
            self.assertLess(gain, 0.002, r)


class Lobe(unittest.TestCase):
    def test_hemisphere_integral_of_the_lobe_equals_the_environment_form(self):
        # independent midpoint quadrature of f_ms cos over the hemisphere of incident directions (wi-space), against the closed form
        model = {**ref.APPENDIX_B, "ms": "kulla-conty"}
        for r in (0.4, 1.0):
            for mu in (0.2, 0.9):
                n_theta, n_phi = 256, 8
                theta = (np.arange(n_theta) + 0.5) / n_theta * (math.pi / 2)
                mu_i = np.cos(theta)
                weight = np.sin(theta) * (math.pi / 2) / n_theta * 2 * math.pi  # dOmega integrated over phi (f_ms does not depend on phi)
                base = np.ones((n_theta, 3))
                value = ms.lobe_value(base, np.ones(n_theta), np.full(n_theta, r), np.full(n_theta, mu), mu_i, model, None, "kulla-conty")
                integral = np.sum(value * (mu_i * weight)[:, None], axis=0)
                closed = ms.environment_value(np.ones((1, 3)), np.ones(1), np.array([r]), np.array([mu]), 1.0, model, None, "kulla-conty")[0]
                np.testing.assert_allclose(integral, closed, atol=2e-3, err_msg=(r, mu))

    def test_reciprocity(self):
        model = {**ref.APPENDIX_B, "ms": "kulla-conty"}
        a = ms.lobe_value(np.ones((1, 3)) * 0.8, np.ones(1), np.array([0.7]), np.array([0.3]), np.array([0.8]), model, None, "kulla-conty")
        b = ms.lobe_value(np.ones((1, 3)) * 0.8, np.ones(1), np.array([0.7]), np.array([0.8]), np.array([0.3]), model, None, "kulla-conty")
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_endpoints_and_fms(self):
        f0, f90 = ms.lobe_endpoints(np.array([[0.8, 0.35, 0.15]]), np.array([1.0]), None)
        np.testing.assert_allclose(f0, [[0.8, 0.35, 0.15]])
        np.testing.assert_allclose(f90, [[1.0, 1.0, 1.0]])
        f0, f90 = ms.lobe_endpoints(np.array([[0.8, 0.35, 0.15]]), np.array([0.0]), None)
        np.testing.assert_allclose(f0, ref.DIELECTRIC_F0)
        # F_ms of a white lobe is E_avg-independent 1 (F_avg = 1): E_avg / (1 - (1 - E_avg)) = 1 (the argument is the loss 1 - E_avg)
        np.testing.assert_allclose(ms.fms(np.ones((1, 3)), np.ones((1, 3)), np.array([0.59])), 1.0)
        # F_avg of Schlick: f0 + (f90 - f0) / 21 (2 int mu (1 - mu)^5 dmu), checked by quadrature
        mu = (np.arange(200000) + 0.5) / 200000
        self.assertAlmostEqual(float(2 * np.mean(mu * (1 - mu) ** 5)), 1 / 21, places=6)

    def test_declared_matches_literal_over_the_sweep(self):
        model = {**ref.APPENDIX_B, "ms": "kulla-conty"}
        worst = 0.0
        for metallic in (0.0, 1.0):
            for r in ROUGHNESS:
                for mu in COSINES:
                    base = np.full((1, 3), 0.8)
                    literal = ms.environment_value(base, np.array([metallic]), np.array([r]), np.array([mu]), 1.0, model, None, "kulla-conty")
                    declared = ms.declared_environment_value(base, np.array([metallic]), np.array([r]), np.array([mu]), 1.0, "table", None)
                    worst = max(worst, float(np.max(np.abs(literal - declared))))
        self.assertLess(worst, DECLARED_TOLERANCE)

    def test_engine_table_header_parses_and_follows_the_lookup_rule(self):
        loss, lost = ms.load_ms_tables()  # ln values
        self.assertEqual((len(loss), lost.shape), (64, (64, 64)))
        self.assertAlmostEqual(float(ms.table_loss_avg(1.0)), float(np.exp(loss[-1])))
        self.assertAlmostEqual(float(ms.table_loss_avg(0.5 / 63)), float(np.exp(0.5 * (loss[0] + loss[1]))))
        self.assertAlmostEqual(float(ms.table_lost(5.0 / 63, 7.0 / 63)), float(np.exp(lost[7, 5])))  # a node
        self.assertAlmostEqual(float(ms.table_lost(0.5 / 63, 0.5 / 63)), float(np.exp(lost[:2, :2].mean())), places=12)  # a cell centre

    def test_engine_table_agrees_with_the_quadrature_between_nodes(self):
        # the table is the quadrature on a 64 x 64 grid, interpolated in ln L: off-node, relative error of the lost energy <= 1 % for
        # roughness >= 0.27 (fixed from the 1 % bound of the direct-light gates; mu >= 0.1: the grazing cells are weighted by cos and excluded
        # from the ROI), absolute error <= 1e-3 below (where L itself is < 1e-2: the lobe is a few 1e-3 of the pixel at most)
        model = {**ref.APPENDIX_B}
        for r in (0.1, 0.15, 0.2, 0.27, 0.35, 0.5, 0.7, 1.0):
            mu, lost, _ = ms.exact_lost(r, model)
            pick = mu >= 0.1
            table = ms.table_lost(mu[pick], np.full(int(pick.sum()), r))
            if r >= 0.27:
                self.assertLess(float(np.max(np.abs(table - lost[pick]) / lost[pick])), 0.01, r)
            else:
                self.assertLess(float(np.max(np.abs(table - lost[pick]))), 1e-3, r)

    def test_exact_environment_uses_a_more_accurate_albedo_than_the_dfg_table(self):
        # the engine's DFG table is a 16384-sample Monte Carlo: its lost energy differs by percent where the quadrature is exact
        model = {**ref.APPENDIX_B}
        mu, lost, _ = ms.exact_lost(0.35, model)
        a, b = ref.dfg_lookup(mu[mu >= 0.1], np.full(int((mu >= 0.1).sum()), 0.35))
        dfg_lost = 1.0 - np.clip(a + b, 0.0, 1.0)
        self.assertGreater(float(np.max(np.abs(dfg_lost - lost[mu >= 0.1]) / lost[mu >= 0.1])), 0.005)


class Layers(unittest.TestCase):
    def test_defaults_are_on(self):  # ADR-0007
        self.assertEqual(ref.APPENDIX_B["ms"], "kulla-conty")
        self.assertEqual(ref.DECLARED["ms"], "table")
        self.assertEqual(ref.SPECULAR_EXT["ms"], "kulla-conty")

    def test_the_lobe_sits_under_the_coat_and_the_coat_adds_its_own(self):
        n_ = 1
        n = np.array([[0.0, 0.0, 1.0]])
        v = ref.normalize(np.array([[0.5, 0.0, 0.8]]))
        l = ref.normalize(np.array([[-0.4, 0.3, 0.7]]))
        base, metallic, roughness = np.ones((n_, 3)), np.ones(n_), np.array([0.8])
        coat = {"factor": np.array([0.7]), "roughness": np.array([0.5]), "f0_ior": np.array([0.04]), "variant": "spec"}
        on, off = {**ref.APPENDIX_B}, {**ref.APPENDIX_B, "ms": "off"}
        with_coat = ref.bsdf(base, metallic, roughness, n, v, l, on, None, coat) - ref.bsdf(base, metallic, roughness, n, v, l, off, None, coat)
        plain = ref.bsdf(base, metallic, roughness, n, v, l, on) - ref.bsdf(base, metallic, roughness, n, v, l, off)
        attenuation = ref.coat_lobe(coat, n, v, l)[0][0]
        nv, nl = float(np.abs(ref.dot(n, v))[0]), float(ref.dot(n, l)[0])
        own = float(ms.coat_lobe_value(coat, np.array([0.7]), np.array([nv]), np.array([nl]), on)[0])
        np.testing.assert_allclose(with_coat[0], attenuation * plain[0] + own, rtol=1e-12)
        self.assertGreater(own, 0.0)
        self.assertGreater(float(plain[0, 0]), 0.0)

    def test_an_anisotropic_lobe_carries_no_multiple_scattering_lobe(self):
        # ADR-0007: measured +3.3 % in the white furnace of a pure metal at strength 1 with the effective isotropic roughness (the lost energy of an
        # anisotropic lobe depends on the azimuth of the view against the lobe axis, which an isotropic table cannot follow): none is applied
        aniso = {"strength": np.array([0.0, 0.5, 1.0])}
        np.testing.assert_array_equal(ms.isotropic(aniso, 3), [True, False, False])
        np.testing.assert_array_equal(ms.isotropic(None, 3), [True, True, True])
        n = np.array([[0.0, 0.0, 1.0]] * 3)
        v = ref.normalize(np.array([[0.5, 0.0, 0.8]] * 3))
        l = ref.normalize(np.array([[-0.4, 0.3, 0.7]] * 3))
        base, metallic, rough = np.ones((3, 3)), np.ones(3), np.full(3, 0.3)
        model = {**ref.APPENDIX_B}
        value = ref.bsdf(base, metallic, rough, n, v, l, model, None, aniso=None) - ref.bsdf(base, metallic, rough, n, v, l, {**model, "ms": "off"}, None, aniso=None)
        self.assertGreater(float(value[0, 0]), 0.0)  # the isotropic lobe of the same pixel carries it
        # the declared environment of the anisotropic pixels equals the one without the lobe (round 3: the anisotropic albedo table needs the azimuth of V
        # in the lobe frame, any fixed value here)
        env = ref.declared_environment_radiance(base, metallic, rough, np.full(3, 0.7), 1.0, ref.DECLARED, None, None, None, {"strength": aniso["strength"], "rotation": np.zeros(3), "phi": np.full(3, 0.3)})
        off = ref.declared_environment_radiance(base, metallic, rough, np.full(3, 0.7), 1.0, {**ref.DECLARED, "ms": "off"}, None, None, None,
                                                {"strength": aniso["strength"], "rotation": np.zeros(3), "phi": np.full(3, 0.3)})
        np.testing.assert_allclose(env[1:], off[1:], rtol=1e-12)
        self.assertGreater(float(env[0, 0] - off[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
