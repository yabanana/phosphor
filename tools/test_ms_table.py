#!/usr/bin/env python3
"""Unit tests of tools/ms_table.py (the lost-energy and 1 - E_avg tables of the multiple-scattering compensation and their checks).

Criteria fixed from independent arguments BEFORE the comparisons: the quadrature `albedo` must agree with a wi-space midpoint rule (shares
nothing with it) within 5e-4 at roughness >= 0.5 and mid angles (that rule's own error is ~1e-4) and with a 16x finer self-convergence run
within 2e-4; E_avg must agree with the independent 3D estimate within 0.5 % at the check roughnesses (its own error 0.16 %).  The negative
controls prove the checks can fail: a loss table scaled by 1.1 and a table built without the cosine weight.

Run offline:
  uv run --offline --no-project --python 3.13 --with numpy python -I tools/test_ms_table.py
"""
from __future__ import annotations

import pathlib
import re
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import ms_table as mt  # noqa: E402

GATE_E_AVG = 0.005
GATE_ALBEDO = 5e-4
GATE_CONVERGENCE = 2e-4


def committed():
    text = mt.DEFAULT_OUT.read_text()
    size = int(re.search(r"MsTableSize\s*=\s*(\d+)\s*;", text).group(1))

    def block(name):
        start = text.index(f"float {name}[")
        body = text[text.index("{", start) + 1:text.index("};", start)]
        return np.asarray([float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)])

    loss, lost = np.exp(block("MsLogLossAvgTable")), np.exp(block("MsLogLostTable"))  # the header stores natural logarithms
    assert len(loss) == size and len(lost) == size * size
    return loss, lost.reshape(size, size)


class Tables(unittest.TestCase):
    def test_nodes_are_the_quadrature(self):
        _, lost = committed()
        size = lost.shape[0]
        for j, i in ((10, 30), (25, 5), (40, 63), (63, 1), (63, 63), (50, 20), (8, 12)):
            mu = 1e-3 if i == 0 else i / (size - 1)
            expected = max(mt.LOG_FLOOR, 1.0 - mt.albedo(mu, j / (size - 1)))
            self.assertAlmostEqual(lost[j, i], expected, delta=1e-6 * expected + 1e-12, msg=(j, i))  # the header prints 9 digits of ln L

    def test_loss_average_is_the_integral_of_the_shader_interpolation(self):
        loss, lost = committed()
        for j in range(lost.shape[0]):
            self.assertAlmostEqual(loss[j], mt.e_avg_from_log_nodes(np.log(lost[j])), delta=1e-5 * loss[j], msg=j)

    def test_independent_e_avg_check_passes(self):
        loss, _ = committed()
        for r, tabulated, reference, relative in mt.check(list(loss)):
            self.assertLess(relative, GATE_E_AVG, (r, tabulated, reference))

    def test_negative_control_scaled_loss_fails(self):
        loss, _ = committed()
        self.assertTrue(any(relative > GATE_E_AVG for *_, relative in mt.check(list(loss * 1.1))))

    def test_negative_control_without_cosine_weight_fails(self):
        _, lost = committed()
        size = lost.shape[0]
        unweighted = [float(np.trapezoid(lost[j], dx=1.0 / (size - 1))) for j in range(size)]  # int L dmu instead of 2 int L mu dmu
        self.assertTrue(any(relative > GATE_E_AVG for *_, relative in mt.check(unweighted)))

    def test_mirror_limit_and_monotonic(self):
        loss, lost = committed()
        self.assertLess(loss[0], 1e-4)  # a near mirror loses nothing
        self.assertTrue(np.all(np.diff(loss) >= -1e-9), "1 - E_avg must not decrease with roughness")
        self.assertGreater(1.0 - loss[-1], 0.35)
        self.assertLess(1.0 - loss[-1], 0.5)
        self.assertTrue(np.all(lost >= 0.0) and np.all(lost <= 1.0))


class Quadrature(unittest.TestCase):
    def test_against_the_wi_space_midpoint_rule(self):
        for r in (0.5, 1.0):
            for mu in (0.3, 0.8):
                self.assertAlmostEqual(mt.albedo(mu, r), mt.independent_albedo(mu, r), delta=GATE_ALBEDO, msg=(r, mu))

    def test_self_convergence(self):
        for r, mu in ((0.35, 0.3), (1.0, 0.5), (0.2, 0.8)):
            self.assertAlmostEqual(mt.albedo(mu, r), mt.albedo(mu, r, 65536, 1024), delta=GATE_CONVERGENCE, msg=(r, mu))

    def test_energy_bounds_and_mirror(self):
        self.assertLess(mt.albedo(0.5, 0.04), 1.0 + 1e-6)
        self.assertGreater(mt.albedo(1.0, 0.04), 0.99999)  # a near mirror at normal incidence returns everything
        self.assertGreater(mt.albedo(0.7, 1.0), mt.albedo(0.3, 1.0) - 1.0)  # finite and positive
        self.assertLess(mt.albedo(0.5, 1.0), 0.6)  # a fully rough GGX lobe loses a large share


class Helpers(unittest.TestCase):
    def test_e_avg_integrators_on_known_functions(self):
        n = 64
        mu = np.arange(n) / (n - 1)
        self.assertAlmostEqual(mt.e_avg_from_nodes(np.ones(n)), 1.0, places=12)  # 2 int mu dmu
        self.assertAlmostEqual(mt.e_avg_from_nodes(mu), 2.0 / 3.0, places=12)  # 2 int mu^2 dmu (linear E is exact)
        self.assertAlmostEqual(mt.e_avg_from_log_nodes(np.zeros(n)), 1.0, places=12)  # L = 1
        self.assertAlmostEqual(mt.e_avg_from_log_nodes(np.full(n, np.log(0.25))), 0.25, places=12)
        # ln L = 2 ln(mu_node): not exactly linear, but L = (1 + ...) mu^2 is smooth: 2 int mu^3 dmu = 1/2 within the interpolation error
        self.assertAlmostEqual(mt.e_avg_from_log_nodes(2.0 * np.log(np.maximum(mu, 1e-3))), 0.5, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
