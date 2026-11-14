#!/usr/bin/env python3
"""Unit tests of tools/f15_layers.py on synthetic data (no GPU).  Run with numpy (uv ... python -I tools/test_f15_layers.py)."""
import pathlib
import re
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_layers as layers  # noqa: E402


def noise_image(seed=1, shape=(6, 8, 3)):
    return np.random.default_rng(seed).random(shape).astype(np.float32) + 0.1


class CompareTests(unittest.TestCase):
    def test_identical_images_are_exact(self):
        a = noise_image()
        r = layers.evaluate_equal(np, a, a.copy())
        self.assertTrue(r["pass"])
        self.assertEqual((r["bitwise_differing"], r["max_ulp"], r["eight_bit_differing"]), (0, 0, 0))

    def test_one_ulp_fails_the_exact_gate_but_not_the_envelope(self):
        a = noise_image()
        b = a.copy()
        b[2, 3, 1] = np.nextafter(b[2, 3, 1], np.float32(10))
        exact = layers.evaluate_equal(np, a, b)
        self.assertFalse(exact["pass"])
        self.assertEqual((exact["bitwise_differing"], exact["max_ulp"]), (1, 1))
        self.assertTrue(layers.evaluate_equal(np, a, b, envelope=True)["pass"])

    def test_envelope_bound(self):
        a = noise_image()
        b = a * np.float32(1.0 + 2e-5)
        self.assertFalse(layers.evaluate_equal(np, a, b, envelope=True)["pass"])
        self.assertTrue(layers.evaluate_equal(np, a, a * np.float32(1.0 + 2e-6), envelope=True)["pass"])

    def test_empty_images_do_not_pass(self):
        z = np.zeros((4, 4, 3), dtype=np.float32)
        self.assertFalse(layers.evaluate_equal(np, z, z)["pass"])

    def test_ulp_distance_across_zero_and_sizes(self):
        a = np.array([[[0.0, 1.0, -1.0]]], dtype=np.float32)
        b = np.array([[[-0.0, np.nextafter(np.float32(1.0), np.float32(2.0)), np.nextafter(np.float32(-1.0), np.float32(-2.0))]]], dtype=np.float32)
        r = layers.compare_images(np, a, b)
        self.assertEqual(r["max_ulp"], 1)
        with self.assertRaises(ValueError):
            layers.compare_images(np, a, np.zeros((2, 2, 3), dtype=np.float32))

    def test_eight_bit_counts_visible_differences_only(self):
        a = np.full((2, 2, 3), 0.5, dtype=np.float32)
        b = a.copy()
        b[0, 0, 0] += 0.01
        self.assertEqual(layers.compare_images(np, a, b)["eight_bit_differing"], 1)
        c = a.copy()
        c[1, 1, 0] = np.nextafter(c[1, 1, 0], np.float32(1))
        self.assertEqual(layers.compare_images(np, a, c)["eight_bit_differing"], 0)


def ramp_scene(reverse=False, width=64, rows=6):
    """Synthetic flat-quad captures: endpoints dark/light, the candidate blends them with the mask u (affine in albedo)."""
    u = np.tile((np.arange(width) + 0.5) / width, (rows, 1))
    base = np.linspace(0.8, 1.2, width)[None, :] * np.ones((rows, 1))  # shading factor varying with position
    dark, light = 0.05 * base, 0.9 * base
    t = (1.0 - u) if reverse else u
    mix = dark + (light - dark) * t
    stack = lambda l: np.repeat(l[..., None], 3, axis=2).astype(np.float32)  # noqa: E731
    return stack(mix), stack(dark), stack(light), np.ones((rows, width), dtype=bool), u


class RampTests(unittest.TestCase):
    def test_monotonic_ramp_passes_and_reverse_fails(self):
        cand, e0, e1, roi, u = ramp_scene()
        result = layers.evaluate_ramp(np, cand, e0, e1, roi, u)
        self.assertTrue(result["pass"], result)
        self.assertAlmostEqual(result["fitted_slope"], 1.0, places=2)
        rev, e0, e1, roi, u = ramp_scene(reverse=True)
        self.assertFalse(layers.evaluate_ramp(np, rev, e0, e1, roi, u)["pass"])

    def test_a_dip_in_the_ramp_is_detected(self):
        cand, e0, e1, roi, u = ramp_scene()
        cand[:, 30:34, :] *= np.float32(0.5)
        result = layers.evaluate_ramp(np, cand, e0, e1, roi, u)
        self.assertFalse(result["pass"])
        self.assertGreater(result["rows_with_reversal"], 0)

    def test_witness_positions_and_uv(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pixels = [None, {"position": [-4.0, 0.0, -4.0]}, {"position": [0.0, 0.0, 2.0]}, None]
            (pathlib.Path(tmp) / "primary-surfaces.json").write_text(json.dumps({"width": 2, "height": 2, "pixels": pixels}))
            roi, x, z = layers.witness_positions(np, tmp)
            self.assertEqual(roi.tolist(), [[False, True], [True, False]])
            u, v = layers.plane_uv(np, x, z)
            self.assertAlmostEqual(float(u[0, 1]), 0.0)
            self.assertAlmostEqual(float(v[1, 0]), 0.75)
            self.assertAlmostEqual(float(u[1, 0]), 0.5)


class FurnaceTests(unittest.TestCase):
    def test_bilinear_weights_sum_to_one_and_hit_the_corners(self):
        u = np.array([0.0, 1.0, 0.0, 1.0, 0.3])
        v = np.array([0.0, 0.0, 1.0, 1.0, 0.6])
        w = layers.bilinear_weights(np, u, v)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)
        np.testing.assert_array_equal(w[:4], np.eye(4))

    def test_reference_blend_mix_and_additive(self):
        u = np.array([0.0, 1.0, 0.0, 1.0, 0.5, 0.25])
        v = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.75])
        mu = np.full(6, 0.7)
        ref = layers.furnace_reference(np, u, v, mu, declared=True)
        # At a corner (one layer, weight 1) the blend, the BSDF mix and the layer coincide; the additive blend too.
        for i in range(4):
            np.testing.assert_allclose(ref["blended"][i], ref["layers"][i][i], rtol=1e-9)
            np.testing.assert_allclose(ref["mixed"][i], ref["layers"][i][i], rtol=1e-9)
            np.testing.assert_allclose(ref["additive"][i], ref["layers"][i][i], rtol=1e-9)
        # Inside, the additive blend exceeds every single layer (it sums energies), the weighted mix does not.
        self.assertGreater(ref["additive"][4].max(), 1.0 + layers.FURNACE_LIMIT - 1.0)
        self.assertLessEqual(ref["mixed"][4].max(), max(l[4].max() for l in ref["layers"]) + 1e-9)
        # Every blended value stays below the furnace bound (the property under test).
        self.assertLessEqual(float(ref["blended"].max()), layers.FURNACE_LIMIT)

    def test_evaluate_furnace_gate_and_controls(self):
        u = np.array([0.2, 0.5, 0.8])
        v = np.array([0.3, 0.5, 0.7])
        ref = layers.furnace_reference(np, u, v, np.full(3, 0.6), declared=True)
        image = np.zeros((1, 3, 3), dtype=np.float32)
        image[0] = ref["blended"].astype(np.float32)
        roi = np.ones((1, 3), dtype=bool)
        result = layers.evaluate_furnace(np, image, roi, ref, {"additive-blend": ref["additive"]})
        self.assertTrue(result["pass"], result)
        self.assertTrue(result["controls"]["additive-blend"]["fails_gate"], result["controls"])
        self.assertIn("max_abs", result["deviation_from_bsdf_mix"])
        hot = image.copy()
        hot[0, 1, 0] = 1.01
        self.assertFalse(layers.evaluate_furnace(np, hot, roi, ref, {})["pass"])


class MirrorTests(unittest.TestCase):
    """The constants declared in C++ and here must agree (the fixture is the source, this tool mirrors it)."""

    def test_constants_match_the_fixture_header(self):
        text = (REPO / "src/testbench/terrain_fixture.h").read_text()
        arrays = {m.group(1): [float(x.rstrip("f")) for x in m.group(2).split(",")]
                  for m in re.finditer(r"(Furnace\w+)\{([^}]*)\}", text)}
        self.assertEqual(tuple(arrays["FurnaceMetallic"]), layers.FURNACE_METALLIC)
        self.assertEqual(tuple(arrays["FurnaceRoughness"]), layers.FURNACE_ROUGHNESS)
        ramp = dict((m.group(1), float(m.group(2))) for m in re.finditer(r"(Ramp\w+) = ([0-9.]+)f", text))
        self.assertEqual((ramp["RampDark"], ramp["RampLight"], ramp["RampRoughness"]), (layers.RAMP_DARK, layers.RAMP_LIGHT, layers.RAMP_ROUGHNESS))
        self.assertIn(f"FlatMaskWidth = {layers.MASK_SIZE}", text)
        textures = (REPO / "src/scene/terrain_textures.h").read_text()
        self.assertIn(f"kPatchSize = {layers.PATCH_SIZE:.1f}f", textures)


if __name__ == "__main__":
    unittest.main()
