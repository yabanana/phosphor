#!/usr/bin/env python3
"""Unit tests of tools/f15_clip_bias.py (no GPU): the control-noise-corrected clip metrics on synthetic clips of known noise and bias.

Needs numpy (skipped without it, like the other numpy-only tools)."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

GATES = {"linear_rmse_p95_max": 0.06, "linear_rmse_max": 0.15, "residual_flicker_mean_max": 0.02}
SCALE = 2.0
FRAMES, SAMPLES, SEEDS = 120, 1200, 16  # the control mean of SEEDS seeds, in two halves


def make(bias, candidate_sigma, sigma, seed=1):
    """(candidate, control, half A, half B) of a moving truth: control = truth + N(sigma^2 / SEEDS), halves = truth + N(2 sigma^2 / SEEDS)."""
    rng = np.random.default_rng(seed)
    truth = 0.5 + 0.2 * np.sin(np.linspace(0, 6, FRAMES))[:, None] * np.ones((1, SAMPLES))
    seeds = truth[None] + rng.normal(0.0, sigma, (SEEDS, FRAMES, SAMPLES))
    half_a, half_b = seeds[0::2].mean(axis=0), seeds[1::2].mean(axis=0)
    control = seeds.mean(axis=0)
    candidate = truth + bias + rng.normal(0.0, candidate_sigma, (FRAMES, SAMPLES))
    return candidate, control, half_a, half_b


@unittest.skipIf(np is None, "numpy not available")
class CorrectedMetrics(unittest.TestCase):
    def setUp(self):
        import f15_clip_bias as bias
        self.bias = bias

    def test_unbiased_candidate_is_zero_while_the_raw_value_reads_the_control_noise(self):
        # control noise per sample: sigma / sqrt(16) = 0.5 * 2 = ... choose sigma so the raw RMSE / scale is well above the gate
        candidate, control, a, b = make(0.0, 0.0, sigma=0.6)
        m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
        self.assertGreater(m["raw_p95"], 0.06)  # the raw judge would fail an exact candidate
        self.assertLess(m["p95"], 0.03)  # resolution limit: the control noise energy is itself estimated (about 4 percent each way)
        self.assertLess(m["max"], 0.04)
        self.assertLess(m["flicker"], 0.01)
        self.assertGreater(m["control_floor_p95"], 0.06)

    def test_a_known_bias_is_recovered(self):
        for offset in (0.06, 0.2):
            candidate, control, a, b = make(offset, 0.0, sigma=0.6, seed=3)
            m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
            self.assertAlmostEqual(m["p95"], offset / SCALE, delta=0.02)  # RMSE of a constant offset / scale

    def test_candidate_noise_is_kept_and_control_noise_is_not(self):
        candidate, control, a, b = make(0.0, 0.2, sigma=0.6, seed=5)
        m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
        self.assertAlmostEqual(m["p95"], 0.2 / SCALE, delta=0.025)  # the candidate's own noise, sqrt(3 * 0.2^2) RMS over RGB-free samples
        self.assertGreater(m["raw_p95"], m["p95"])

    def test_negative_control_still_fails_after_the_correction(self):
        candidate, control, a, b = make(0.3, 0.1, sigma=0.6, seed=7)  # a bad candidate (bias 0.3)
        m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
        verdict = self.bias.judge(m, GATES)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["checks"]["spatial_p95"])
        self.assertFalse(verdict["checks"]["worst_frame"])

    def test_the_flicker_correction_removes_control_flicker_only(self):
        rng = np.random.default_rng(11)
        candidate, control, a, b = make(0.0, 0.0, sigma=0.6, seed=9)
        m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
        self.assertGreater(m["raw_flicker"], 0.05)
        self.assertLess(m["flicker"], 0.2 * m["raw_flicker"])
        flickering = candidate + rng.normal(0.0, 0.3, candidate.shape)
        m2 = self.bias.corrected_metrics(flickering, control, a, b, SCALE)
        self.assertGreater(m2["flicker"], 0.1)

    def test_not_judgeable_when_the_control_floor_exceeds_a_gate(self):
        candidate, control, a, b = make(0.0, 0.0, sigma=0.6, seed=13)
        m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
        self.assertFalse(self.bias.judge(m, GATES)["judgeable"])
        candidate, control, a, b = make(0.0, 0.0, sigma=0.01, seed=13)
        m = self.bias.corrected_metrics(candidate, control, a, b, SCALE)
        self.assertTrue(self.bias.judge(m, GATES)["judgeable"])


if __name__ == "__main__":
    unittest.main()
