#!/usr/bin/env python3
"""Tests of tools/f15_subsurface_profile.py (numpy only)."""
import json
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_subsurface_profile as p  # noqa: E402


class Profile(unittest.TestCase):
    def test_the_walk_reproduces_the_mitsuba_albedo_table(self):
        # The exit flux of a normal-incidence searchlight is (1 - R_spec(0)) A(0 degrees) of the table (same boundary, same medium).
        table = json.loads(p.ALBEDO_PATH.read_text())
        curve = [c for c in table["curves"] if c["g"] == 0.0 and abs(c["alpha_b"] - p.ALPHA_B) < 1e-4 and c["ior"] == p.IOR][0]
        for point in curve["points"]:
            if point["albedo"] in (0.5, 0.95, 0.99):
                _, flux = p.profile_cdf(point["albedo"], 0.0, 30000, 3)
                expected = (1.0 - curve["R_spec"]["0"]) * point["A_angle"]["0"]
                self.assertAlmostEqual(flux, expected, delta=0.04, msg=str(point["albedo"]))

    def test_an_absorbing_medium_returns_almost_nothing_and_the_cdf_is_a_cdf(self):
        _, flux = p.profile_cdf(0.0, 0.0, 5000, 1)
        self.assertLess(flux, 0.01)
        cdf, _ = p.profile_cdf(0.9, 0.0, 5000, 2)
        self.assertTrue(np.all(np.diff(cdf) >= 0.0))
        self.assertAlmostEqual(float(cdf[-1]), 1.0, places=6)

    def test_a_larger_single_scatter_albedo_spreads_wider(self):
        e = p.edges()
        low, _ = p.profile_cdf(0.7, 0.0, 20000, 4)
        high, _ = p.profile_cdf(0.999, 0.0, 20000, 4)
        k = int(np.searchsorted(e, 2.0))
        self.assertGreater(low[k], high[k] + 0.1)

    def test_the_committed_lut_matches_its_grid(self):
        data = json.loads(p.LUT_PATH.read_text())
        self.assertEqual(len(data["profiles"]), len(data["g_grid"]) * len(data["c_grid"]))
        for row in data["profiles"]:
            self.assertEqual(len(row["cdf"]), data["bins"] + 1)
            self.assertAlmostEqual(row["cdf"][-1], 1.0, places=6)
            self.assertAlmostEqual(row["alpha_ss"], p.inverse_albedo(row["C"], row["g"]), places=9)
        self.assertEqual(data["bins"], p.BINS)

    def test_the_header_lists_every_cdf_value(self):
        data = json.loads(p.LUT_PATH.read_text())
        text = p.header_text(data)
        self.assertEqual(text.count("f,") + 1 >= len(data["profiles"]) * (data["bins"] + 1), True)
        self.assertIn("SSS_PROFILE_BINS = 64u", text)


if __name__ == "__main__":
    unittest.main()
