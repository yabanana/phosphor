#!/usr/bin/env python3
"""Tests of tools/f15_walk_clip_check.py (numpy only): the metric, the gates and the negative control's discrimination on synthetic clips."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import f15_walk_clip_check as wc  # noqa: E402


def clip(frames=40, h=64, w=64, shift=0):
    """A bright disc moving right by `shift` pixels per frame on a dark background (luminance > the ROI threshold)."""
    out = np.zeros((frames, h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for f in range(frames):
        cx = 20 + shift * f
        out[f] = np.where((yy - 32) ** 2 + (xx - cx) ** 2 < 14 ** 2, 0.5, 0.0)
    return out


class Metric(unittest.TestCase):
    def test_a_candidate_equal_to_the_control_passes(self):
        control = clip(shift=0.3)
        rows = wc.frame_errors(np, control, control)
        self.assertTrue(wc.judge(rows)["pass"])

    def test_noise_inside_the_bound_passes_and_beyond_it_fails(self):
        control = clip()
        rng = np.random.default_rng(1)
        mild = control * (1.0 + 0.05 * rng.standard_normal(control.shape).astype(np.float32))
        wild = control * (1.0 + 0.6 * rng.standard_normal(control.shape).astype(np.float32))
        self.assertTrue(wc.judge(wc.frame_errors(np, mild, control))["pass"])
        self.assertFalse(wc.judge(wc.frame_errors(np, wild, control))["pass"])

    def test_a_ghost_history_that_never_follows_the_motion_fails_the_gate(self):
        control = clip(shift=0.8)
        ghost = np.repeat(clip(shift=0.8)[:1], len(control), axis=0)  # the first frame forever: history read at the same pixel, never rejected
        verdict = wc.judge(wc.frame_errors(np, ghost, control))
        self.assertFalse(verdict["pass"])
        self.assertGreaterEqual(verdict["ghost_ratio"], 1.0)

    def test_the_warm_up_frames_are_not_judged(self):
        control = clip(frames=wc.WARMUP + 4)
        bad = control.copy()
        bad[: wc.WARMUP] = 0.0
        self.assertTrue(wc.judge(wc.frame_errors(np, bad, control))["pass"])

    def test_box_mean_is_the_window_average(self):
        image = np.zeros((9, 9, 3), dtype=np.float32)
        image[4, 4] = 9.0
        pooled = wc.box_mean(np, image, 3)
        self.assertAlmostEqual(float(pooled[4, 4, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(pooled[3, 3, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(pooled[2, 2, 0]), 0.0, places=5)

    def test_the_gate_is_the_side_lit_bound(self):
        import f15_check as fc
        self.assertEqual((wc.GATE["mean"], wc.GATE["p95"]), (fc.THRESHOLDS["subsurface"]["mean"], fc.THRESHOLDS["subsurface"]["p95"]))
        self.assertEqual(wc.GATE["floor_fraction"], fc.THRESHOLDS["subsurface"]["floor_fraction"])
        self.assertIn("negative-reject-never", wc.CASES)


if __name__ == "__main__":
    unittest.main()
