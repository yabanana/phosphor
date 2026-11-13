#!/usr/bin/env python3
"""Unit tests of tools/sheen_table.py (the sheen albedo table tool and its independent check).

Expected values come from the specification text of the Charlie notes (research/khr-materials.md section 3), from
analytic identities, or from the committed header; the negative controls prove that the independent check can fail.

Run offline:
  uv run --offline --no-project --python 3.13 --with numpy python -I tools/test_sheen_table.py
"""
from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import sheen_table as st  # noqa: E402

_COMMITTED = None


def committed():
    global _COMMITTED
    if _COMMITTED is None:
        _COMMITTED = st.parse_header(st.HEADER)
    return _COMMITTED


class CharlieTerms(unittest.TestCase):
    def test_lambda_hand_values(self):
        # alpha_g = 1: (1 - alpha)^2 = 0, the first column of the fit; Lambda(0.5) = exp(l(0.5)).
        x = 0.5
        l = 21.5473 / (1.0 + 3.82987 * x ** 0.19823) + -1.97760 * x + -4.32054
        self.assertAlmostEqual(float(st._lambda(0.5, 1.0)), math.exp(l), places=9)
        # alpha_g = 0 (never reached through the clamp, but the formula's other end): the second column.
        l0 = 25.3245 / (1.0 + 3.32435 * x ** 0.16801) + -1.27393 * x + -4.85967
        self.assertAlmostEqual(float(st._lambda(0.5, 0.0)), math.exp(l0), places=9)
        # cos >= 0.5: exp(2 l(0.5) - l(1 - cos)); at cos = 1 the argument of l is 0.
        l_half = l
        l_zero = 21.5473 / (1.0 + 0.0) + -4.32054
        self.assertAlmostEqual(float(st._lambda(1.0, 1.0)), math.exp(2 * l_half - l_zero), places=9)

    def test_lambda_branches_agree_between_implementations(self):
        for r in (0.07, 0.3, 0.8, 1.0):
            for cosine in (0.0, 0.01, 0.3, 0.49, 0.5, 0.51, 0.9, 1.0, -0.4):
                a = float(st._lambda(cosine, st.sheen_alpha(r)))
                b = float(st._independent_lambda(cosine, r))
                self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(a)), msg=(r, cosine))

    def test_alpha_and_clamp(self):
        self.assertAlmostEqual(st.sheen_alpha(0.5), 0.25)
        self.assertAlmostEqual(st.sheen_alpha(0.0), st.MIN_ROUGHNESS ** 2)
        self.assertAlmostEqual(st.sheen_alpha(3.0), 1.0)


class TableValues(unittest.TestCase):
    def test_normal_view_is_dark_and_grazing_view_is_bright(self):
        # Sheen is a fuzz lobe: dark at normal incidence, brighter towards grazing views.
        for r in (0.3, 0.5, 1.0):
            normal = st.albedo_half_vector(1.0, r)
            grazing = st.albedo_half_vector(0.1, r)
            self.assertLess(normal, grazing)
            self.assertLess(normal, 0.2)

    def test_hand_value_rough_normal_view(self):
        # r = 1 (alpha 1, inv_r 1): D = 3 sin(theta_h) / (2 pi); at N.V = 1, N.H = cos(theta_i / 2).
        mu = 1.0
        n, total = 2000, 0.0
        for k in range(n):  # one-dimensional midpoint rule over theta_i (the integrand is axisymmetric at N.V = 1)
            theta = (k + 0.5) * (0.5 * math.pi / n)
            cos_i = math.cos(theta)
            h_theta = theta / 2.0
            d = 3.0 * math.sin(h_theta) / (2.0 * math.pi)
            visibility_times_cos = 1.0 / ((1.0 + float(st._lambda(mu, 1.0)) + float(st._lambda(cos_i, 1.0))) * 4.0 * mu)
            total += d * visibility_times_cos * 2.0 * math.pi * math.sin(theta) * (0.5 * math.pi / n)
        self.assertAlmostEqual(st.albedo_half_vector(1.0, 1.0), total, delta=2e-4)
        self.assertAlmostEqual(st.albedo_incident(1.0, 1.0), total, delta=2e-4)

    def test_two_methods_agree(self):
        for r, mu in ((0.1, 0.1), (0.3, 0.3), (0.5, 0.9), (0.75, 0.05), (1.0, 0.5)):
            self.assertAlmostEqual(st.albedo_half_vector(mu, r), st.albedo_incident(mu, r), delta=1e-3, msg=(r, mu))

    def test_fixed_seed_monte_carlo_agrees(self):
        # A third method: cosine-weighted hemisphere sampling with a fixed seed, estimator D V pi.
        rng = np.random.default_rng(20261113)
        n = 400000
        for r, mu in ((0.3, 0.3), (0.6, 0.7), (1.0, 0.2)):
            u1, u2 = rng.random(n), rng.random(n)
            cos_i, phi = np.sqrt(1.0 - u1), 2.0 * math.pi * u2
            sin_i = np.sqrt(u1)
            wo = np.array([math.sqrt(1.0 - mu * mu), 0.0, mu])
            wi = np.stack([sin_i * np.cos(phi), sin_i * np.sin(phi), cos_i], axis=-1)
            h = wi + wo
            h /= np.linalg.norm(h, axis=-1, keepdims=True)
            inv = 1.0 / (r * r)
            d = (2.0 + inv) * np.sqrt(np.maximum(1.0 - h[:, 2] ** 2, 0.0)) ** inv / (2.0 * math.pi)
            v = 1.0 / ((1.0 + st._independent_lambda(mu, r) + st._independent_lambda(cos_i, r)) * 4.0 * mu * cos_i)
            samples = d * v * math.pi  # integrand D V N.L over the pdf N.L / pi
            estimate = float(samples.mean())
            sigma = float(samples.std() / math.sqrt(n))
            self.assertAlmostEqual(estimate, min(st.albedo_half_vector(mu, r), 9.0), delta=max(5 * sigma, 1e-3), msg=(r, mu))

    def test_energy_is_not_conserved_unclamped_at_grazing_narrow_lobes(self):
        # The reason for the clamp: the unclamped Charlie albedo exceeds 1 at N.V = 1e-3, roughness 0.07.
        self.assertGreater(st.albedo_half_vector(1e-3, 0.07), 1.5)
        self.assertEqual(committed()[1][0, 0], 1.0)


class CommittedHeader(unittest.TestCase):
    def test_header_shape_and_macros(self):
        size, values, nodes = committed()
        self.assertEqual(size, 64)
        self.assertEqual(values.shape, (64, 64))
        self.assertEqual(nodes, st.GENERATOR_NODES)
        text = st.HEADER.read_text()
        for needle in ("PHOSPHOR_GPU_CONSTANT u32 SheenTableSize", "PHOSPHOR_GPU_CONSTANT float SheenTable[4096]",
                       "#define PHOSPHOR_SHEEN_TABLE const constant float*", "#define PHOSPHOR_SHEEN_TABLE const float*",
                       "inline float sheenTableLookup(PHOSPHOR_SHEEN_TABLE table, float nv, float roughness)",
                       "GENERATED by tools/sheen_table.py"):
            self.assertIn(needle, text)

    def test_values_in_unit_range_and_row_clamp(self):
        _, values, _ = committed()
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertGreaterEqual(values.min(), 0.0)
        self.assertLessEqual(values.max(), 1.0)
        # Rows below the roughness clamp repeat row 0.07 (row 4 of 63 = 0.0635 < 0.07: identical to row 0).
        np.testing.assert_allclose(values[0], values[3], atol=0)
        # More roughness raises the albedo at normal incidence (the lobe widens toward the normal).
        self.assertLess(values[8, -1], values[40, -1])

    def test_regeneration_of_a_row_matches_the_header(self):
        size, values, nodes = committed()
        for j, i in ((40, 10), (25, 33), (63, 0)):
            fresh = min(st.albedo_half_vector(i / (size - 1), j / (size - 1), nodes), st.E_CLAMP)
            self.assertAlmostEqual(fresh, values[j, i], delta=st.REGENERATION_TOLERANCE)

    def test_independent_check_passes_on_the_committed_table(self):
        _, values, _ = committed()
        self.assertEqual(st.independent_check(values), [])

    def test_rendered_header_roundtrips(self):
        rows = [0.25 * ((k % 7) / 7.0) for k in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sheen_table.h"
            path.write_text(st.render_header(3, (8, 8), rows))
            size, values, nodes = st.parse_header(path)
        self.assertEqual((size, nodes), (3, (8, 8)))
        np.testing.assert_allclose(values.reshape(-1), rows, rtol=1e-7)


class NegativeControl(unittest.TestCase):
    """A corrupted table must FAIL the independent check, at every verification node, in both directions."""

    @classmethod
    def setUpClass(cls):
        _, cls.values, _ = committed()
        size = cls.values.shape[0]
        cls.cache = {(j, i): min(st.albedo_incident(i / (size - 1), j / (size - 1)), st.E_CLAMP)
                     for j, i in st.check_nodes(size)}

    def run_check(self, values):
        size = values.shape[0]
        lookup = {(round(i / (size - 1), 9), round(j / (size - 1), 9)): v for (j, i), v in self.cache.items()}
        return st.independent_check(values, integrate=lambda mu, r: lookup[(round(mu, 9), round(r, 9))])

    def test_pristine_table_passes_with_the_cache(self):
        self.assertEqual(self.run_check(self.values), [])

    def test_every_checked_node_detects_a_perturbation(self):
        for j, i in st.check_nodes(64):
            for delta in (0.05, -0.05):
                bad = self.values.copy()
                bad[j, i] = bad[j, i] + delta
                self.assertTrue(self.run_check(bad), msg=(j, i, delta))

    def test_default_corruption_helper_fails(self):
        self.assertTrue(st.independent_check(st.corrupt(self.values)))

    def test_small_perturbation_inside_tolerance_is_accepted(self):
        bad = self.values.copy()
        bad[30, 30] += 0.5 * st.INDEPENDENT_TOLERANCE
        # not a verification node: the check samples nodes, it does not claim to cover every entry
        self.assertNotIn((30, 30), st.check_nodes(64))
        self.assertEqual(self.run_check(bad), [])

    def test_range_violation_is_reported(self):
        bad = self.values.copy()
        bad[10, 10] = 1.5
        self.assertTrue(any("outside [0, 1]" in line for line in self.run_check(bad)))
        bad[10, 10] = float("nan")
        self.assertTrue(any("outside [0, 1]" in line for line in self.run_check(bad)))

    def test_check_command_fails_on_a_corrupted_header(self):
        size, values, nodes = committed()
        bad = values.copy()
        bad[63, 32] += 0.02  # a verification node
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sheen_table.h"
            path.write_text(st.render_header(size, nodes, list(bad.reshape(-1))))
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = st.command_check(path)
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=1)
