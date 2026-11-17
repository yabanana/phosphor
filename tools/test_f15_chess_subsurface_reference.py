#!/usr/bin/env python3
"""Tests of the non-finite sample repair of tools/f15_chess_subsurface_reference.py (no GPU, no Mitsuba render).

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/test_f15_chess_subsurface_reference.py
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import f15_chess_subsurface_reference as ref  # noqa: E402


class RepairNonfinite(unittest.TestCase):
    def test_a_finite_image_is_returned_untouched(self):
        image = np.random.default_rng(1).random((8, 9, 3)).astype(np.float32)
        out, count = ref.repair_nonfinite(image)
        self.assertEqual(count, 0)
        self.assertTrue(np.array_equal(out, image))

    def test_nan_and_inf_take_the_mean_of_the_finite_5x5_neighbours_of_their_channel(self):
        image = np.ones((9, 9, 3), dtype=np.float32)
        image[..., 1] = 2.0
        image[4, 4, 0] = np.nan
        image[2, 7, 2] = np.inf
        image[0, 0, 1] = -np.inf
        out, count = ref.repair_nonfinite(image)
        self.assertEqual(count, 3)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(float(out[4, 4, 0]), 1.0)   # channel 0 neighbours are 1
        self.assertEqual(float(out[2, 7, 2]), 1.0)
        self.assertEqual(float(out[0, 0, 1]), 2.0)   # corner window clips to the image; channel 1 neighbours are 2
        self.assertTrue(np.isnan(image[4, 4, 0]))    # the input is not modified

    def test_the_result_does_not_depend_on_the_scan_order(self):
        image = np.arange(7 * 7 * 3, dtype=np.float32).reshape(7, 7, 3)
        image[3, 3, 0] = np.nan
        image[3, 4, 0] = np.nan
        out, count = ref.repair_nonfinite(image)
        flipped, _ = ref.repair_nonfinite(image[::-1, ::-1].copy())
        self.assertEqual(count, 2)
        self.assertTrue(np.array_equal(out, flipped[::-1, ::-1]))
        window = image[1:6, 1:6, 0]
        self.assertAlmostEqual(float(out[3, 3, 0]), float(window[np.isfinite(window)].mean()), places=4)   # neighbours read from the UNREPAIRED image

    def test_a_fully_nonfinite_neighbourhood_gives_zero(self):
        image = np.full((5, 5, 3), np.nan, dtype=np.float32)
        out, count = ref.repair_nonfinite(image)
        self.assertEqual(count, 75)
        self.assertTrue(np.array_equal(out, np.zeros_like(image)))


if __name__ == "__main__":
    unittest.main()
