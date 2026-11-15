#!/usr/bin/env python3
"""Unit tests of tools/f15_transport_check.py (no GPU): the generated scenes, the rig geometry, the analytic expectation and the gates."""
import json
import math
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_transport_check as check  # noqa: E402


class TransportCheck(unittest.TestCase):
    def test_volume_scene_is_closed_outward_and_the_emitter_sits_inside(self):
        scene = check.volume_scene(True, True)
        lo, hi = scene.bounds()
        self.assertEqual(lo[2], 0.0)
        self.assertAlmostEqual(hi[2], check.CUBE_HI[2])
        cube = [p for p in scene.primitives if p[4] == 1]
        self.assertEqual(len(cube), 6)
        centre = [0.5 * (a + b) for a, b in zip(check.CUBE_LO, check.CUBE_HI)]
        for positions, normals, _, _, _ in cube:  # every face normal points away from the cube centre
            face = [sum(c[i] for c in positions) / 4.0 for i in range(3)]
            self.assertGreater(sum((face[i] - centre[i]) * normals[0][i] for i in range(3)), 0.0)
        emitter = [p for p in scene.primitives if p[4] == 2][0]
        self.assertEqual(emitter[1][0], (0.0, 0.0, -1.0))  # faces the wall
        for i in range(3):
            self.assertTrue(check.CUBE_LO[i] < check.EMITTER_CENTRE[i] < check.CUBE_HI[i])

    def test_arms_differ_only_in_attenuation_and_emission(self):
        a, b, c = (check.volume_scene(False, True), check.volume_scene(True, True), check.volume_scene(True, False))
        self.assertEqual(a.primitives, b.primitives)
        self.assertNotIn("attenuationDistance", a.materials[1]["extensions"]["KHR_materials_volume"])
        self.assertEqual(b.materials[1]["extensions"]["KHR_materials_volume"]["attenuationDistance"], check.ATTENUATION[1])
        self.assertEqual(c.materials[2]["emissiveFactor"], [0, 0, 0])
        self.assertEqual(b.materials[2]["emissiveFactor"], [1, 1, 1])

    def test_rois_see_the_wall_beside_the_cube_and_the_segment_enters_the_cube(self):
        scene = check.volume_scene(True, True)
        centre, distance = check.frame_rig(*scene.bounds())
        for name, rect in check.VOLUME_ROIS.items():
            box = check.pixel_box(rect, centre, distance)
            self.assertTrue(0 <= box[0] < box[2] <= check.WIDTH and 0 <= box[1] < box[3] <= check.HEIGHT, name)
            for corner in ((box[0], box[1]), (box[2] - 1, box[3] - 1)):
                p = check.wall_point(*corner, centre, distance)
                self.assertGreater(abs(p[0]), 0.4)
                s = check.segment_entry(p, check.EMITTER_CENTRE, check.CUBE_LO, check.CUBE_HI)
                self.assertIsNotNone(s, name)
                # the wall pixel is not seen THROUGH the cube: the camera ray to the wall point misses the cube
                cam = (centre[0], centre[1], centre[2] + distance)
                self.assertIsNone(check.segment_entry(cam, p, check.CUBE_LO, check.CUBE_HI), name)

    def test_expectation_is_exp_of_the_open_length(self):
        centre, distance = check.frame_rig(*check.volume_scene(True, True).bounds())
        box = check.pixel_box(check.VOLUME_ROIS["right"], centre, distance)
        result = check.volume_expectation({"right": box}, centre, distance, lambda px, py: [1.0, 1.0, 1.0])["right"]
        self.assertEqual(result["skipped"], 0)
        sigma = [-math.log(c) / check.ATTENUATION[1] for c in check.ATTENUATION[0]]
        for c in range(3):
            e = result["expected"][c]
            self.assertTrue(0.0 < e < 1.0)
            # between the shortest and longest open length of the ROI (0.1 m .. 0.5 m)
            self.assertTrue(math.exp(-sigma[c] * 0.5) < e < math.exp(-sigma[c] * 0.1))
        # the expected ratio of the weakest channel must be far from the clamped engine's 1 (the fixture discriminates)
        self.assertLess(result["expected"][0], 1.0 - 4.0 * check.BOUND)

    def test_segment_entry_boundaries(self):
        lo, hi = (0, 0, 0), (1, 1, 1)
        self.assertAlmostEqual(check.segment_entry((-1, 0.5, 0.5), (3, 0.5, 0.5), lo, hi), 0.25)
        self.assertIsNone(check.segment_entry((-1, 2.0, 0.5), (3, 2.0, 0.5), lo, hi))
        self.assertIsNone(check.segment_entry((0.5, 0.5, 0.5), (3, 0.5, 0.5), lo, hi))  # starts inside: no entry

    def test_masked_scene_texture_halves_and_regions(self):
        m, o = check.masked_scene("M"), check.masked_scene("O")
        self.assertEqual(m.materials[1]["alphaMode"], "MASK")
        self.assertEqual(m.materials[1]["alphaCutoff"], 0.5)
        self.assertEqual(len(m.images), 1)
        self.assertEqual(len(o.images), 0)
        self.assertNotIn("alphaMode", o.materials[1])
        with tempfile.TemporaryDirectory() as tmp:
            path = m.write(tmp)
            document = json.loads(path.read_text())
            self.assertEqual(document["images"], [{"uri": "texture0.png"}])
            self.assertTrue((pathlib.Path(tmp) / "texture0.png").read_bytes().startswith(b"\x89PNG"))
        # the sun shadow of each pane half contains its ROI (and the lit ROI is outside every shadow)
        dx, dy = -check.SUN_OVER_Z[0] * check.PANE_Z, -check.SUN_OVER_Z[1] * check.PANE_Z
        x0, y0, x1, y1 = check.PANE
        half = 0.5 * (x0 + x1)
        left, right, lit = check.MASK_ROIS["left"], check.MASK_ROIS["right"], check.MASK_ROIS["lit"]
        shadow_left = (x0 + dx, y0 + dy, half + dx, y1 + dy)
        shadow_right = (half + dx, y0 + dy, x1 + dx, y1 + dy)
        margin = 0.03  # texel transition of the alpha texture (16 texels over 1 m, linear filter) + the sun penumbra
        for roi, shadow in ((left, shadow_left), (right, shadow_right)):
            self.assertTrue(shadow[0] + margin <= roi[0] and roi[2] <= shadow[2] - margin and shadow[1] <= roi[1] and roi[3] <= shadow[3])
        self.assertTrue(lit[1] > shadow_right[3] + 0.05)
        # no ROI is seen THROUGH the pane: the camera ray to the wall point misses the pane quad (plane z = PANE_Z)
        centre, distance = check.frame_rig(*m.bounds())
        cam_z = centre[2] + distance
        for roi in (left, right, lit):
            for cx, cy in ((roi[0], roi[1]), (roi[2], roi[3])):
                t = (cam_z - check.PANE_Z) / cam_z  # fraction along the ray from the camera to the wall point at z = PANE_Z
                px, py = centre[0] + (cx - centre[0]) * t, centre[1] + (cy - centre[1]) * t
                self.assertFalse(x0 <= px <= x1 and y0 <= py <= y1)

    def test_gates_and_controls_discriminate(self):
        good = {"transmittance": {"left": [1.0, 1.0, 1.0], "right": list(check.TINT)}}
        gates, controls = check.gate_masked(good)
        self.assertTrue(all(g["pass"] for g in gates.values()))
        self.assertTrue(controls["opaque-model"]["right"]["fails"] and controls["ignored-model"]["right"]["fails"])
        before = {"transmittance": {"left": [1.0, 1.0, 1.0], "right": [0.0, 0.0, 0.0]}}  # the masked-opaque rule
        gates, _ = check.gate_masked(before)
        self.assertTrue(gates["left"]["pass"] and not gates["right"]["pass"])
        volume = {"rois": {"right": {"measured_ratio": [0.62, 0.8, 0.95], "expected_ratio": [0.63, 0.81, 0.95]}}}
        gates, controls = check.gate_volume(volume)
        self.assertTrue(gates["right"]["pass"] and controls["clamp-model"]["right"]["fails"])
        clamped = {"rois": {"right": {"measured_ratio": [1.04, 1.04, 1.04], "expected_ratio": [0.63, 0.81, 0.95]}}}
        self.assertFalse(check.gate_volume(clamped)[0]["right"]["pass"])

    def test_png_writer_header(self):
        png = check.png_rgba(2, 1, [[(1, 2, 3, 4), (5, 6, 7, 8)]])
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", png)


if __name__ == "__main__":
    unittest.main()
