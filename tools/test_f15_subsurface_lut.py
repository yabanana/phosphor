#!/usr/bin/env python3
"""Tests of tools/f15_subsurface_lut.py (numpy only): the committed header equals the generator output, the table inverts the measured
observed-albedo curves, saturates above A_max and is monotone."""
import math
import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_subsurface_lut as lut  # noqa: E402
import f15_subsurface_reference as ref  # noqa: E402


class Table(unittest.TestCase):
    def test_the_committed_header_is_not_stale(self):
        self.assertEqual(lut.HEADER_PATH.read_text(), lut.header_text())

    def test_the_table_inverts_every_measured_curve(self):
        roughness, data, a_max = lut.table()
        curves = lut.collect_curves()
        index = 0
        for r in roughness:
            for g in lut.G_KNOTS:
                curve = curves[(r, g)]
                for i in range(lut.COLOUR_COUNT):
                    c = i / (lut.COLOUR_COUNT - 1)
                    s = data[index]
                    index += 1
                    alpha = 1.0 - s * s
                    reached = ref.observed_albedo(alpha, curve)
                    top = ref.observed_albedo(1.0, curve)
                    if c < top - 1e-3:
                        self.assertAlmostEqual(reached, c, delta=2e-3, msg=f"r {r} g {g} colour {c}")
                    else:
                        self.assertAlmostEqual(s, 0.0, delta=0.05, msg=f"saturated r {r} g {g} colour {c}")

    def test_monotone_in_colour_and_bounded(self):
        roughness, data, _ = lut.table()
        n = lut.COLOUR_COUNT
        for block in range(len(data) // n):
            row = data[block * n:(block + 1) * n]
            self.assertAlmostEqual(row[0], 1.0, places=6)  # colour 0: albedo 0
            for a, b in zip(row, row[1:]):
                self.assertLessEqual(b, a + 1e-9)  # brighter colour = more scattering = smaller sqrt(1 - albedo)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in row))

    def test_roughness_knots_cover_the_fixture_roughness(self):
        roughness, _, a_max = lut.table()
        self.assertIn(0.35, roughness)
        self.assertGreaterEqual(len(roughness), 3)
        # a rougher boundary traps and loses more light: the largest reachable colour falls with the roughness (per g)
        per_g = [a_max[k::len(lut.G_KNOTS)] for k in range(len(lut.G_KNOTS))]
        for series in per_g:
            for a, b in zip(series, series[1:]):
                self.assertGreater(a, b - 0.01)

    def test_the_draft_mapping_ignores_the_boundary(self):
        # negative control of the table: the draft's Kulla-Conty single-scatter albedo for colour 0.5 is far below the boundary-aware one
        draft = float(ref.single_scatter_albedo(0.5, 0.0))
        curve = lut.collect_curves()[(0.35, 0.0)]
        aware, _, _ = ref.invert_observed_albedo(0.5, curve)
        self.assertGreater(aware, draft + 0.05)


if __name__ == "__main__":
    unittest.main()
