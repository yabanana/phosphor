#!/usr/bin/env python3
"""Unit tests of tools/f15_gi_environment_check.py (no GPU): the CPU sky-visibility reference, the fixture geometry guarantees and the gates."""
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_gi_environment_check as check  # noqa: E402


class Reference(unittest.TestCase):
    def test_open_sky_sees_everything(self):
        self.assertEqual(check.sky_visibility((0, 0, 0), {}), 1.0)

    def test_hemisphere_sealing_plane_sees_nothing(self):
        roof = {"roof": ("z", 0.05, (-100, 100), (-100, 100))}
        self.assertEqual(check.sky_visibility((0, 0, 0), roof), 0.0)

    def test_floor_blocks_half_the_cosine_weight(self):
        # a floor plane y = -1e-6 below a +z wall point blocks the directions with y < 0: half of the cosine-weighted hemisphere
        floor = {"floor": ("y", -1e-6, (-100, 100), (0, 100))}
        self.assertAlmostEqual(check.sky_visibility((0, 0, 0), floor), 0.5, delta=0.02)

    def test_plate_overhead_matches_the_analytic_cosine_law(self):
        # a small square plate of half-size a at height h on the axis blocks (2a)^2 / (pi h^2) of the cosine-weighted hemisphere
        a, h = 0.05, 1.0
        plate = {"plate": ("z", h, (-a, a), (-a, a))}
        self.assertAlmostEqual(1.0 - check.sky_visibility((0, 0, 0), plate, n=192), (2 * a) ** 2 / (math.pi * h * h), delta=2e-3)

    def test_ray_hits_respect_tmax_and_extent(self):
        o = ("z", 1.0, (0, 1), (0, 1))
        self.assertTrue(check.hit_rectangle((0.5, 0.5, 0), (0, 0, 1), o))
        self.assertFalse(check.hit_rectangle((0.5, 0.5, 0), (0, 0, 1), o, tmax=0.5))
        self.assertFalse(check.hit_rectangle((2.5, 0.5, 0), (0, 0, 1), o))
        self.assertFalse(check.hit_rectangle((0.5, 0.5, 0), (0, 0, -1), o))


class Fixture(unittest.TestCase):
    def test_every_region_is_in_the_umbra_and_visible(self):
        for name in check.REGIONS:
            ref = check.region_reference(name)
            self.assertTrue(ref["sun_blocked"], name)
            self.assertTrue(ref["camera_sees_wall"], name)

    def test_open_region_is_nearly_unoccluded_and_slot_region_is_occluded(self):
        open_ref = check.region_reference("open")
        slot = check.region_reference("occluded")
        self.assertGreater(open_ref["visibility"], 0.93)
        self.assertLess(slot["visibility"], 0.35)
        self.assertAlmostEqual(slot["reference"], check.ALBEDO * check.L_ENV * slot["visibility"])

    def test_an_unoccluded_term_would_fail_the_occluded_gate(self):
        slot = check.region_reference("occluded")
        unoccluded_ratio = 1.0 / slot["visibility"]  # what an additive environment x albedo term reads
        self.assertGreater(unoccluded_ratio, check.OCCLUDED_GATE[1])

    def test_sun_check_discriminates(self):
        point = (0.55, -0.01, 0.0)
        self.assertTrue(check.sun_blocked(point))
        self.assertFalse(check.sun_blocked(point, {}))  # without the shelf the same wall point is lit by the sun

    def test_scene_writes_wall_and_occluders(self):
        with tempfile.TemporaryDirectory() as folder:
            path = check.build_scene().write(folder)
            document = json.loads(path.read_text())
            self.assertEqual(len(document["nodes"]), 1 + len(check.OCCLUDERS))
            self.assertTrue(all(m.get("doubleSided") for m in document["materials"][1:]))
            self.assertFalse(document["materials"][0].get("doubleSided", False))

    def test_region_boxes_are_inside_the_frame_and_not_empty(self):
        for name, (x0, y0, x1, y1) in check.region_boxes().items():
            self.assertTrue(0 <= x0 < x1 <= check.fx.WIDTH and 0 <= y0 < y1 <= check.fx.HEIGHT, name)


class Gates(unittest.TestCase):
    def test_correct_ratios_pass_every_mode(self):
        for mode in ("ddgi", "cache", "restir"):
            self.assertEqual(check.gate_failures(mode, {"open": 1.02, "occluded": 1.1}), [])

    def test_unoccluded_term_fails_the_occluded_gate_only(self):
        self.assertEqual(check.gate_failures("ddgi", {"open": 1.0, "occluded": 3.9}), ["occluded"])

    def test_missing_environment_fails_both(self):
        self.assertEqual(check.gate_failures("restir", {"open": 0.02, "occluded": 0.06}), ["open", "occluded"])

    def test_gi_off_does_not_gate_the_occluded_region(self):
        self.assertEqual(check.gate_failures("off", {"open": 1.05, "occluded": 3.94}), [])
        self.assertEqual(check.gate_failures("ddgi-default-grid", {"open": 1.0, "occluded": 2.5}), [])
        self.assertEqual(check.gate_failures("ddgi", {"open": 1.0, "occluded": 2.5}), ["occluded"])

    def test_every_mode_has_a_setup_and_the_per_pixel_modes_are_gated(self):
        self.assertEqual(set(check.MODES), set(check.MODE_SETUP))
        for mode in ("ddgi", "cache", "restir"):
            self.assertTrue(check.MODE_SETUP[mode][2])

    def test_control_must_lose_the_environment_in_every_gi_mode(self):
        self.assertTrue(check.control_fails("ddgi", {"open": 0.037, "occluded": 0.09}))
        self.assertFalse(check.control_fails("ddgi", {"open": 1.0, "occluded": 1.0}))

    def test_ratios_against_the_reference_are_unity(self):
        refs = {n: check.region_reference(n) for n in check.REGIONS}
        ratios = check.ratios({n: refs[n]["reference"] for n in refs}, refs)
        self.assertEqual(ratios, {"open": 1.0, "occluded": 1.0})


class Measurement(unittest.TestCase):
    def test_pfm_region_mean(self):
        width, height = 8, 6
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "a.pfm"
            body = b"".join(struct.pack("<3f", 0.25, 0.25, 0.25) for _ in range(width * height))
            path.write_bytes(b"PF\n8 6\n-1.0\n" + body)
            means = check.region_means(path, {"r": (1, 1, 5, 4)})
            self.assertAlmostEqual(means["r"], 0.25, places=5)


if __name__ == "__main__":
    unittest.main()
