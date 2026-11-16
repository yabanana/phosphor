#!/usr/bin/env python3
"""Unit tests of tools/aniso_table.py (the anisotropic directional albedo table of the declared environment, F15.1 ticket 11 round 3).

Criteria fixed BEFORE the comparisons (independent arguments, never loosened):
  * the half-vector quadrature agrees with a wi-space midpoint rule (cosine of the incident polar angle uniform, no half-vector substitution:
    shares nothing with it) within 3e-3 absolute in A and B on lobes with alpha_b >= 0.09 (that rule's own error is ~2e-4 at 1400 x 1400 nodes);
  * the committed header equals the generator at sampled nodes within the printed precision (1e-6 relative);
  * the four-linear lookup reproduces the quadrature of the reference (tools/f15_reference.py `_aniso_spec_integrals`, its own integrator and node
    layout) at random OFF-node points over N.V >= 0.1: worst relative error of A + B <= 1.5 %, 99th percentile <= 0.6 %; the sphere-weighted mean
    relative error on the ticket grid (r 0.2 / 0.3 / 0.5 / 0.7, strength 0.25 .. 1, N.V >= 0.1, pdf 2 mu, uniform azimuth) <= 0.2 %;
  * decision 11: the table model against the exact albedo is below 5 % at every strength / roughness of that grid, while the former fit (the
    isotropic split sum at the rms effective roughness) is above 5 % at r 0.2 / 0.3 / 0.5 strength 1 (negative control: it must be able to fail);
  * white-furnace bound: A + B <= 1 at every node (a directional albedo of a Fresnel-1 lobe cannot exceed 1).

Run offline:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_aniso_table.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import aniso_table as at  # noqa: E402
import f15_reference as ref  # noqa: E402

GATE_WI = 3e-3
GATE_HEADER = 1e-6
GATE_WORST = 0.015
GATE_P99 = 0.006
GATE_GRID_MEAN = 0.002
DECISION_11 = 0.05


def wi_quadrature(mu, phi, alpha_t, alpha_b, n=1400):
    """(A, B) by a midpoint rule over the incident hemisphere (cos theta_i uniform, azimuth uniform): D V cos_i (1 - w) and D V cos_i w with the
    Burley D and the README V written here from their formulas."""
    ci = (np.arange(n) + 0.5) / n
    pj = (np.arange(2 * n) + 0.5) / (2 * n) * 2 * math.pi
    c, p = np.meshgrid(ci, pj, indexing="ij")
    s = np.sqrt(1 - c * c)
    wo = np.array([math.sqrt(1 - mu * mu) * math.cos(phi), math.sqrt(1 - mu * mu) * math.sin(phi), mu])
    wi = np.stack([s * np.cos(p), s * np.sin(p), c], axis=-1)
    h = wi + wo
    h /= np.linalg.norm(h, axis=-1, keepdims=True)
    nh = np.maximum(h[..., 2], 1e-9)
    d = 1.0 / (math.pi * alpha_t * alpha_b * ((h[..., 0] / alpha_t) ** 2 + (h[..., 1] / alpha_b) ** 2 + nh ** 2) ** 2)
    gv = c * np.sqrt((alpha_t * wo[0]) ** 2 + (alpha_b * wo[1]) ** 2 + mu * mu)
    gl = mu * np.sqrt((alpha_t * wi[..., 0]) ** 2 + (alpha_b * wi[..., 1]) ** 2 + c * c)
    vis = 0.5 / np.maximum(gv + gl, 1e-12)
    vh = np.clip(np.sum(wo * h, axis=-1), 0.0, 1.0)
    w = (1 - vh) ** 5
    integrand = d * vis * c
    cell = (1.0 / n) * (2 * math.pi / (2 * n))
    return float(np.sum(integrand * (1 - w)) * cell), float(np.sum(integrand * w) * cell)


def sphere_error(model, at_, ab, rough, strength):
    """Sphere-weighted mean relative error of A + B of `model(mu, phi)` against the reference quadrature table (ticket 11 grid metric)."""
    mus, phis, a, b = ref.aniso_albedo_table(at_, ab)
    exact = a + b
    fit = np.array([[model(mu, phi) for phi in phis] for mu in mus])
    weight = (2.0 * mus * (mus >= 0.1))[:, None] * np.ones_like(exact)
    return float((np.abs(fit - exact) / exact * weight).sum() / weight.sum())


def rms_fit_model(at_, ab):
    """The former declared model: the isotropic split sum (DFG table of the reference) at the rms effective roughness."""
    table = ref.environment_table(math.sqrt(float(ref.declared_aniso_alpha(at_, ab))))
    return lambda mu, phi: float(np.interp(mu, table["mu"], table["A"] + table["B"]))


def table_model(rough, strength):
    def model(mu, phi):
        a, b = ref.aniso_table_lookup(np.array([mu]), np.array([phi]), np.array([rough]), np.array([strength]))
        return float(a[0] + b[0])
    return model


class Quadrature(unittest.TestCase):
    def test_half_vector_quadrature_matches_the_wi_space_rule(self):
        for rough, strength, mu, phi in ((0.3, 0.0, 0.8, 0.3), (0.3, 0.5, 0.8, 0.4), (0.3, 1.0, 0.5, 0.9), (0.5, 1.0, 0.3, 1.4), (0.5, 0.75, 0.9, 0.0), (0.3, 1.0, 0.7, math.pi / 2)):
            t, b = at.alphas(rough, strength)
            a1, b1 = at.half_vector_integrals(mu, phi, t, b)
            a2, b2 = wi_quadrature(mu, phi, t, b)
            self.assertAlmostEqual(a1, a2, delta=GATE_WI, msg=(rough, strength, mu, phi))
            self.assertAlmostEqual(b1, b2, delta=GATE_WI, msg=(rough, strength, mu, phi))

    def test_the_wi_rule_can_fail(self):  # negative control: the rule is sensitive to the lobe (the isotropic lobe is not the stretched one)
        t, b = at.alphas(0.3, 1.0)
        a1, _ = at.half_vector_integrals(0.5, 0.9, t, b)
        a2, _ = wi_quadrature(0.5, 0.9, b, b)
        self.assertGreater(abs(a1 - a2), 10 * GATE_WI)

    def test_quadrature_is_converged(self):
        t, b = at.alphas(0.3, 1.0)
        coarse = at.half_vector_integrals(0.4, 0.7, t, b)
        fine = at.half_vector_integrals(0.4, 0.7, t, b, (768, 768))
        self.assertAlmostEqual(coarse[0], fine[0], delta=2e-4)
        self.assertAlmostEqual(coarse[1], fine[1], delta=2e-4)

    def test_the_strength_zero_slice_is_the_isotropic_lobe(self):
        size, dfg = ref.load_dfg_table()  # Monte Carlo table, ~1e-3 in E
        table = at.parse()
        for ri in (4, 8, 12):
            r = ri / (at.R - 1)
            row = int(round(r * (size - 1)))
            for mi in (6, 14, 23):
                mu = mi / (at.M - 1)
                col = int(round(mu * (size - 1)))
                e_table = table[0, ri, 0, mi].sum()
                e_dfg = dfg[row, col].sum()
                self.assertAlmostEqual(e_table, e_dfg, delta=0.02, msg=(ri, mi))
        self.assertTrue(np.allclose(table[0, :, 0, :], table[0, :, at.P - 1, :], atol=1e-9))  # no azimuth dependence at strength 0


class Header(unittest.TestCase):
    def test_header_equals_the_generator_at_nodes(self):
        table = at.parse()
        self.assertEqual(table.shape, (at.S, at.R, at.P, at.M, 2))
        rng = np.random.default_rng(3)
        for _ in range(24):
            si, ri, pi_, mi = (int(rng.integers(0, n)) for n in (at.S, at.R, at.P, at.M))
            t, b = at.alphas(ri / (at.R - 1), si / (at.S - 1))
            mu = at.MU_FLOOR if mi == 0 else mi / (at.M - 1)
            a, bb = at.half_vector_integrals(mu, 0.5 * math.pi * pi_ / (at.P - 1), t, b)
            for got, want in ((table[si, ri, pi_, mi, 0], a), (table[si, ri, pi_, mi, 1], bb)):
                self.assertAlmostEqual(got, want, delta=GATE_HEADER * abs(want) + 1e-12, msg=(si, ri, pi_, mi))

    def test_a_perturbed_header_fails(self):  # negative control of the previous check
        table = at.parse()
        si, ri, mi = 4, 8, int(round(0.5 * (at.M - 1)))
        mu = mi / (at.M - 1)
        want = at.half_vector_integrals(mu, 0.0, *at.alphas(ri / (at.R - 1), si / (at.S - 1)))[0]
        self.assertGreater(abs(table[si, ri, 0, mi, 0] * 1.01 - want), GATE_HEADER * want)

    def test_albedo_bound(self):
        table = at.parse()
        e = table.sum(axis=-1)
        self.assertLessEqual(float(e.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(table.min()), 0.0)

    def test_python_lookup_equals_the_reference_lookup(self):
        table = at.parse()
        rng = np.random.default_rng(5)
        for _ in range(40):
            mu, phi, r, s = rng.uniform(0, 1), rng.uniform(0, math.pi / 2), rng.uniform(0, 1), rng.uniform(0, 1)
            a1, b1 = at.lookup(table, mu, phi, r, s)
            a2, b2 = ref.aniso_table_lookup(np.array([mu]), np.array([phi]), np.array([r]), np.array([s]))
            self.assertAlmostEqual(a1, float(a2[0]), places=12)
            self.assertAlmostEqual(b1, float(b2[0]), places=12)


class Fit(unittest.TestCase):
    def test_lookup_reproduces_the_reference_quadrature_off_node(self):
        table = at.parse()
        rng = np.random.default_rng(7)
        errors = []
        for _ in range(240):
            mu, phi, r, s = rng.uniform(0.1, 1), rng.uniform(0, math.pi / 2), rng.uniform(0.06, 1), rng.uniform(0, 1)
            t, b = at.alphas(r, s)
            a, bb = ref._aniso_spec_integrals(mu, phi, t, b)
            got = at.lookup(table, mu, phi, r, s)
            errors.append(abs(got[0] + got[1] - a - bb) / (a + bb))
        self.assertLess(max(errors), GATE_WORST)
        self.assertLess(float(np.quantile(errors, 0.99)), GATE_P99)

    def test_ticket_grid_error_of_the_table_is_below_decision_11_and_the_old_fit_is_not(self):
        worst_table, worst_old = 0.0, 0.0
        report = []
        for rough in (0.2, 0.3, 0.5, 0.7):
            for strength in (0.25, 0.5, 0.75, 1.0):
                ab = max(rough * rough, 0.002)
                t = ab + (1 - ab) * strength * strength
                e_table = sphere_error(table_model(rough, strength), t, ab, rough, strength)
                e_old = sphere_error(rms_fit_model(t, ab), t, ab, rough, strength)
                report.append((rough, strength, e_table, e_old))
                self.assertLess(e_table, GATE_GRID_MEAN, (rough, strength))
                self.assertLess(e_table, DECISION_11, (rough, strength))
                worst_table, worst_old = max(worst_table, e_table), max(worst_old, e_old)
                if strength == 1.0 and rough <= 0.5:  # negative control: the former fit is above the decision-11 bound where the ticket measured it
                    self.assertGreater(e_old, DECISION_11, (rough, strength))
        print("\nticket grid (r, s, table error, old rms-fit error):")
        for row in report:
            print("  r %.1f s %.2f  table %.4f  old %.4f" % row)
        print("  worst table %.4f, worst old %.4f" % (worst_table, worst_old))

    def test_an_azimuth_free_table_misses_the_bound_margin(self):
        # Why the azimuth is a dimension: collapsing it to the azimuth average leaves 3.6-4.5 % at strength 1 (decision-11 bound 5 %).
        rough, strength = 0.5, 1.0
        ab = rough * rough
        t = ab + (1 - ab) * strength * strength
        mus, phis, a, b = ref.aniso_albedo_table(t, ab)
        mean = (a + b).mean(axis=1)
        exact = a + b
        weight = (2.0 * mus * (mus >= 0.1))[:, None] * np.ones_like(exact)
        error = float((np.abs(mean[:, None] - exact) / exact * weight).sum() / weight.sum())
        self.assertGreater(error, 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=1)
