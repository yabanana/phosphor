#!/usr/bin/env python3
"""Tests of tools/f15_glass_receiver_check.py (numpy only, no GPU): the analytic thin-walled BTDF, the pixel regions, the gate logic and its
negative controls on synthetic captures.   python -I tools/test_f15_glass_receiver_check.py"""
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f15_glass_receiver_check as g  # noqa: E402
from f12_reference import write_pfm  # noqa: E402


class Btdf(unittest.TestCase):
    def test_front_lights_and_grazing_views_give_zero(self):
        n = np.array([[0.0, 0.0, 1.0]])
        v = np.array([[0.0, 0.0, 1.0]])
        front = np.array([[0.0, 0.6, 0.8]])  # on the camera side: not a transmitted light
        self.assertEqual(float(g.thin_btdf(np, n, v, front, 0.09)[0]), 0.0)
        graze = np.array([[0.0, 1.0, 0.0]])
        self.assertEqual(float(g.thin_btdf(np, n, graze, np.array([[0.0, 0.0, -1.0]]), 0.09)[0]), 0.0)

    def test_the_lobe_peaks_straight_through_and_is_symmetric_about_it(self):
        n = np.array([[0.0, 0.0, 1.0]])
        v = np.array([[0.0, 0.0, 1.0]])
        straight = np.array([[0.0, 0.0, -1.0]])
        s = math.sin(0.1)
        a = np.array([[s, 0.0, -math.cos(0.1)]])
        b = np.array([[-s, 0.0, -math.cos(0.1)]])
        peak = float(g.thin_btdf(np, n, v, straight, 0.09)[0])
        self.assertGreater(peak, float(g.thin_btdf(np, n, v, a, 0.09)[0]))
        self.assertAlmostEqual(float(g.thin_btdf(np, n, v, a, 0.09)[0]), float(g.thin_btdf(np, n, v, b, 0.09)[0]), places=9)

    def test_albedo_is_one_minus_fresnel_times_a_ggx_albedo_below_one(self):
        # integral of f_t |N.L| over the back hemisphere at normal incidence: (1 - F) x the directional albedo of the mirrored GGX lobe (< 1)
        alpha = 0.3
        n_theta, n_phi = 400, 64
        theta = (np.arange(n_theta) + 0.5) / n_theta * (math.pi / 2)
        phi = (np.arange(n_phi) + 0.5) / n_phi * 2 * math.pi
        t, p = np.meshgrid(theta, phi, indexing="ij")
        l = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), -np.cos(t)], axis=-1).reshape(-1, 3)
        v = np.broadcast_to(np.array([0.0, 0.0, 1.0]), l.shape)
        n = v
        value = g.thin_btdf(np, n, v, l, alpha)
        # f_t |N.L| integrated with d omega = sin(theta) dtheta dphi (the |N.L| is inside thin_btdf; f_t |N.L| E: radiance per unit irradiance
        # along L is f_t |N.L|, so the albedo is the integral of f_t |N.L| d omega)
        weight = np.sin(t).reshape(-1) * (math.pi / 2 / n_theta) * (2 * math.pi / n_phi)
        albedo = float(np.sum(value * weight))
        self.assertGreater(albedo, 0.75)  # single-scattering GGX loses energy at alpha 0.3; (1 - F) <= 0.96
        self.assertLess(albedo, 1.0)

    def test_controls_differ_where_they_claim_to(self):
        position, directions = g.camera_rays(np, g.BTDF_CAMERA)
        key = "btdf-point@legacy"
        index = 4  # tinted, roughness 0.3
        inside, hits = g.plane_hits(np, position, directions, g.BTDF_PANES[index][0], (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), g.REGION_MARGIN)
        self.assertGreater(int(inside.sum()), 2000)
        exact = g.btdf_expected(np, key, index, position, directions, inside, hits)
        for variant in ("tint-ignored", "not-mirrored", "no-lobe"):
            other = g.btdf_expected(np, key, index, position, directions, inside, hits, variant)
            mean, p95 = g.relative_stats(np, other, exact)
            self.assertTrue(np.isfinite(mean) and np.isfinite(p95), variant)
            mean2, p952 = g.relative_stats(np, exact, other, floor_from=exact)  # a control under the exact floor stays finite
            self.assertTrue(np.isfinite(mean2) and np.isfinite(p952), variant)
            self.assertTrue(mean > g.MEAN_BOUND or p95 > g.P95_BOUND, variant)
        half = g.btdf_expected(np, key, 6, position, directions, *g.plane_hits(np, position, directions, g.BTDF_PANES[6][0], (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), g.REGION_MARGIN)[:1],
                               g.plane_hits(np, position, directions, g.BTDF_PANES[6][0], (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), g.REGION_MARGIN)[1])
        ignored = g.btdf_expected(np, key, 6, position, directions, g.plane_hits(np, position, directions, g.BTDF_PANES[6][0], (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), g.REGION_MARGIN)[0],
                                  g.plane_hits(np, position, directions, g.BTDF_PANES[6][0], (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), g.REGION_MARGIN)[1], "factor-ignored")
        self.assertAlmostEqual(float(np.mean(ignored / np.maximum(half, 1e-12))), 2.0, places=6)

    def test_sun_disk_samples_stay_inside_the_disk(self):
        directions = g.sun_disk_directions(np, g.BTDF_SUN["toward"])
        centre = np.asarray(g.BTDF_SUN["toward"], dtype=float)
        centre /= np.linalg.norm(centre)
        angles = np.arccos(np.clip(directions @ centre, -1, 1))
        self.assertLessEqual(float(angles.max()), g.SUN_ANGULAR_RADIUS * (1 + 1e-6))
        self.assertEqual(len(directions), g.SUN_SAMPLES)


class Regions(unittest.TestCase):
    def test_pane_regions_are_disjoint_and_cover_the_declared_fraction(self):
        position, directions = g.camera_rays(np, g.BTDF_CAMERA)
        masks = [g.plane_hits(np, position, directions, c, (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), g.REGION_MARGIN)[0] for c, *_ in g.BTDF_PANES]
        total = np.zeros_like(masks[0], dtype=int)
        for m in masks:
            total += m
        self.assertEqual(int(total.max()), 1)
        for m in masks:
            self.assertGreater(int(m.sum()), 2000)

    def test_b_region_is_found_by_the_camera_model(self):
        position, directions = g.camera_rays(np, g.STACK_CAMERA)
        c, s = g.STACK_B["centre"], g.STACK_B["size"]
        inside, _ = g.plane_hits(np, position, directions, c, (1, 0, 0), (0, 0, 1), 0.5 * s[0], 0.5 * s[1], (0, 1, 0), g.REGION_MARGIN)
        self.assertGreater(int(inside.sum()), 1500)


class StackGate(unittest.TestCase):
    def synthetic(self, factor):
        """single = a smooth highlight on the B region; stack = factor (per channel) times it."""
        position, directions = g.camera_rays(np, g.STACK_CAMERA)
        c, s = g.STACK_B["centre"], g.STACK_B["size"]
        inside, _ = g.plane_hits(np, position, directions, c, (1, 0, 0), (0, 0, 1), 0.5 * s[0], 0.5 * s[1], (0, 1, 0), 1.0)
        width, height = g.RESOLUTION
        ys, xs = np.mgrid[0:height, 0:width]
        highlight = 0.5 + 0.5 * np.cos((xs - width / 2) / 40.0)
        base = np.zeros((height, width, 3), dtype=np.float32)
        base[inside] = (highlight[inside, None] * np.asarray(g.STACK_B["tint"])[None, :] * 0.2 + 0.001).astype(np.float32)
        return base, (base * np.asarray(factor, dtype=np.float32)[None, None, :]).astype(np.float32)

    def run_gate(self, factor):
        t_a = g.schlick_pass(math.sin(math.radians(g.STACK_CAMERA["elevation"]))) * np.asarray(g.STACK_A_TINT)
        base, stack = self.synthetic(np.asarray(factor) * t_a)
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            for key, image in (("stack@csm", stack), ("single@csm", base)):
                (root / key).mkdir()
                write_pfm(root / key / "engine.pfm", image)
            return g.evaluate_stack(np, root, ["csm"])["csm"]

    def test_the_corrected_engine_passes_and_both_wrong_expectations_fail(self):
        result = self.run_gate([1.0, 1.0, 1.0])
        self.assertTrue(result["gate_pass"], result)
        self.assertTrue(result["controls_fail"], result)

    def test_an_engine_that_counts_the_receiver_twice_fails_the_gate(self):
        t = g.schlick_pass(math.sin(math.radians(g.STACK_CAMERA["elevation"])))
        result = self.run_gate(t * np.asarray(g.STACK_B["tint"]))
        self.assertFalse(result["gate_pass"], result)

    def test_an_engine_without_attenuation_fails_the_gate(self):
        t = g.schlick_pass(math.sin(math.radians(g.STACK_CAMERA["elevation"]))) * np.asarray(g.STACK_A_TINT)
        result = self.run_gate(1.0 / t)
        self.assertFalse(result["gate_pass"], result)


class Cameras(unittest.TestCase):
    def test_camera_rays_are_unit_and_the_centre_ray_hits_the_target(self):
        for camera in (g.BTDF_CAMERA, g.STACK_CAMERA):
            position, directions = g.camera_rays(np, camera)
            self.assertTrue(np.allclose(np.linalg.norm(directions, axis=2), 1.0))
            width, height = g.RESOLUTION
            centre = directions[height // 2, width // 2]
            target = np.asarray(camera["target"], dtype=float)
            toward = (target - position) / np.linalg.norm(target - position)
            self.assertGreater(float(centre @ toward), 0.9999)


class EdgesAndCache(unittest.TestCase):
    def floor(self):
        position, directions = g.camera_rays(np, g.EDGE_CAMERA)
        return g.floor_points(np, position, directions)

    def step_image(self, wx, wz, x0=-1.15, x1=-0.25, z0=-0.077, z1=0.823, shift=0.0, level=0.4):
        red = np.ones(wx.shape, dtype=np.float32)
        inside = (wx > x0 + shift) & (wx < x1 + shift) & (wz > z0) & (wz < z1)
        red[inside] = level
        image = np.zeros(wx.shape + (3,), dtype=np.float32)
        image[..., 0] = red
        image[..., 1] = image[..., 2] = 1.0
        return image

    def test_crossing_is_subpixel_and_exact_on_a_ramp(self):
        coord = np.linspace(0.0, 1.0, 11)
        profile = coord * 2.0  # a linear ramp: the level 0.8 is crossed at 0.4, between the samples 0.4 and 0.5 (exactly on one)
        self.assertAlmostEqual(g.crossing(np, profile, coord, 0.83), 0.415, places=6)
        self.assertTrue(math.isnan(g.crossing(np, np.ones(11), coord, 0.5)))

    def test_contours_of_a_synthetic_shadow_have_one_pixel_accuracy_and_a_shift_is_detected(self):
        wx, wz = self.floor()
        truth = g.edge_contours(np, self.step_image(wx, wz), wx, wz)
        for edge in ("x0", "x1", "z0", "z1"):
            self.assertGreater(len(truth[edge]), 20, edge)
        moved = g.edge_contours(np, self.step_image(wx, wz, shift=0.03), wx, wz)
        shared = sorted(set(truth["x0"]) & set(moved["x0"]))
        delta = np.array([moved["x0"][k][0] - truth["x0"][k][0] for k in shared])
        self.assertAlmostEqual(float(np.median(delta)), 0.03, delta=0.006)  # within a pixel footprint (5.5 mm) of the true shift

    def test_the_edge_gate_passes_identical_images_and_fails_the_shift_control(self):
        wx, wz = self.floor()
        image = self.step_image(wx, wz)
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            for key in ("edge@csm", "edge@rt"):
                (root / key).mkdir()
                write_pfm(root / key / "engine.pfm", image)
            result = g.evaluate_edges(np, root)
        self.assertTrue(result["gate_pass"], result)
        self.assertTrue(result["control_fails"], result)

    def test_the_cache_gate_follows_the_closed_form_and_the_opaque_engine_fails_it(self):
        wx, wz = self.floor()
        cos_sun = math.cos(math.radians(25.0))
        t = g.schlick_pass(cos_sun) * np.asarray(g.TINT)
        (sx0, sx1), (sz0, sz1) = g.CACHE_REGION["shadow"]
        shadow = np.isfinite(wx) & (wx > sx0) & (wx < sx1) & (wz > sz0) & (wz < sz1)

        def image(factor):
            im = np.full(wx.shape + (3,), 0.5, dtype=np.float32)
            im[shadow] = (0.5 * factor).astype(np.float32)
            return im

        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            for key, factor in (("edge@csm", t), ("cache@csm", t), ("cachebefore@csm", np.zeros(3))):
                (root / key).mkdir()
                write_pfm(root / key / "engine.pfm", image(factor))
            result = g.evaluate_cache(np, root)
        self.assertTrue(result["runs"]["cache@csm"]["passed"], result)
        self.assertTrue(result["runs"]["cachebefore@csm"]["passed"], result)  # passes by failing the gate
        self.assertFalse(result["runs"]["cachebefore@csm"]["gate_pass"])
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
