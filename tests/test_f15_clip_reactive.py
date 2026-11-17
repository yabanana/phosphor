#!/usr/bin/env python3
"""Runner logic of F15.1 ticket 13 (no GPU, no numpy): the transmission clip cases and ROIs of tools/f15_clip_check.py and the
reactive-mask predicates of tools/f15_reactive_check.py."""
import math
import pathlib
import sys
import unittest
from array import array

TOOLS = pathlib.Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import f15_clip_check as clip  # noqa: E402
import f15_reactive_check as reactive  # noqa: E402
from f13_f14_check import THRESHOLDS  # noqa: E402
from temporal_light_metrics import Image  # noqa: E402


class TransmissionClipCases(unittest.TestCase):
    def setUp(self):
        self.cases = {c["name"]: c for c in clip.case_table(pathlib.Path("/tmp/x"), pathlib.Path("/tmp/phosphor"))}

    def test_cases_and_controls(self):
        for name in ("transmission-raw", "transmission-custom", "transmission-rt8", "negative-refract-axis"):
            self.assertIn(name, self.cases)
            command = self.cases[name]["command"]
            self.assertEqual(command[command.index("--material-scenario") + 1], "transmission-roughness-ior")
            self.assertIn("--transmission-motion", command)
            self.assertEqual(command[command.index("--reflections") + 1], "rt")  # glass is traced whatever the band, RT on
            self.assertEqual(command[command.index("--frames") + 1], str(clip.CLIP_FRAMES))
        spp = lambda n: self.cases[n]["command"][self.cases[n]["command"].index("--reflection-samples") + 1]
        denoise = lambda n: self.cases[n]["command"][self.cases[n]["command"].index("--lighting-denoise") + 1]
        self.assertEqual((spp("transmission-raw"), denoise("transmission-raw")), ("1", "off"))
        self.assertEqual((spp("transmission-custom"), denoise("transmission-custom")), ("1", "custom"))
        self.assertEqual((spp("transmission-rt8"), denoise("transmission-rt8")), ("8", "off"))
        self.assertEqual((spp("negative-refract-axis"), denoise("negative-refract-axis")), ("1", "custom"))
        self.assertEqual(self.cases["negative-refract-axis"]["env"], {"PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE": "reflect-axis"})
        for name in ("transmission-raw", "transmission-custom", "transmission-rt8"):
            self.assertEqual(self.cases[name]["env"], {})

    def test_converged_control_uses_the_both_branch_estimator_only_for_the_control(self):
        seeds = [c for c in clip.case_table(pathlib.Path("/tmp/x"), pathlib.Path("/tmp/phosphor"), 4) if c["kind"] == "control-seed"]
        self.assertEqual([c["seed"] for c in seeds], [1000, 1001, 1002, 1003])
        for case in seeds:
            self.assertIn("--debug-transmission-both-branches", case["command"])
            self.assertIn("--debug-lighting", case["command"])  # the switch is refused without it
            self.assertEqual(case["command"][case["command"].index("--reflection-samples") + 1], "8")
        for name in ("transmission-raw", "transmission-custom", "transmission-rt8", "negative-refract-axis"):
            self.assertNotIn("--debug-transmission-both-branches", self.cases[name]["command"])  # never the product path

    def test_gates_are_the_f13_thresholds_on_glass_pixels(self):
        config = clip.transmission_roi_config()
        self.assertEqual([r["name"] for r in config["rois"]], [n for n, _, _ in clip.GLASS_GATED])
        for roi in config["rois"]:
            self.assertEqual(roi["thresholds"], THRESHOLDS)  # never loosened
            self.assertEqual(roi["scale"], clip.TRANSMISSION_ROI_SCALE)
        diagnostic = clip.transmission_roi_config(diagnostic=True)
        self.assertEqual([r["name"] for r in diagnostic["rois"]], [n for n, _, _ in clip.GLASS_DIAGNOSTIC] + ["glass-grid"])
        self.assertEqual([n for n, _, _ in clip.GLASS_GATED], ["glass-smooth-ior1.33", "glass-smooth-ior1.5", "glass-smooth-ior2.0"])

    def test_rois_lie_on_the_spheres_with_a_motion_margin(self):
        w, h = 640, 360
        for name, row, column in clip.GLASS_GATED + clip.GLASS_DIAGNOSTIC:
            cx, cy, rx, ry = clip.glass_sphere_screen(column, row, w, h)
            x0, y0, x1, y1 = clip.glass_roi(column, row, w, h)
            for x in (x0, x1):
                for y in (y0, y1):  # every corner of the ROI is inside 0.72 of the sphere radius
                    self.assertLessEqual(math.hypot((x - cx) / rx, (y - cy) / ry), 0.72, name)
            self.assertGreater(x1 - x0, 0.0)
        # disjoint gated ROIs
        boxes = [clip.glass_roi(c, r, w, h) for _, r, c in clip.GLASS_GATED]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                self.assertTrue(a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    def test_default_group_runs_old_and_new_cases(self):
        names = {c["name"] for c in self.cases.values()}
        self.assertTrue({"clip-custom", "coat-custom", "transmission-custom", "negative-refract-axis", "adapt-on"} <= names)


class CaptureRules(unittest.TestCase):
    def test_every_clip_case_is_synchronous(self):
        for case in clip.case_table(pathlib.Path("/tmp/x"), pathlib.Path("/tmp/phosphor"), 2):
            self.assertEqual(case["command"].count("--pipeline-sync"), 1, case["name"])

    def test_fallback_draws_rule(self):
        self.assertEqual(clip.fallback_draws("x\nPIPELINES requests 1 | fallbacks 2 (draws 13) | failures 0\nEXIT 0"), 13)
        self.assertEqual(clip.fallback_draws("PIPELINES a fallbacks 2 (draws 9)\nPIPELINES b fallbacks 0 (draws 0)"), 0)  # the LAST line decides
        self.assertIsNone(clip.fallback_draws("no summary\nEXIT 0"))
        import tempfile
        for log, ok in (("PIPELINES a fallbacks 0 (draws 0)\nEXIT 0\n", True), ("PIPELINES a fallbacks 2 (draws 13)\nEXIT 0\n", False), ("EXIT 0\n", False)):
            with tempfile.TemporaryDirectory() as folder:
                pathlib.Path(folder, "log.txt").write_text(log)
                errors = clip.functional({"folder": folder, "kind": "adapt", "denoise": "off"})
                self.assertEqual(not any("fallback" in e for e in errors), ok, log)


class ReactivePredicates(unittest.TestCase):
    def image(self, fixture, fill):
        w, h = reactive.WIDTH, reactive.HEIGHT
        values = array("f", [fill(x, y) for y in range(h) for x in range(w) for _ in range(3)])
        return Image(w, h, values)

    def test_glass_pixels_marked(self):
        fixture = reactive.thin_volume()
        discs = [reactive.disc(fixture, c, 1.0) for c in fixture["spheres"]]
        marked = self.image(fixture, lambda x, y: 0.749 if any(reactive.inside(d, x, y) for d in discs) else 1.0)
        row = reactive.evaluate_mask(marked, fixture)
        self.assertTrue(reactive.glass_ok(row))
        self.assertFalse(reactive.opaque_ok(row))

    def test_negative_controls_fail_the_glass_predicate(self):
        fixture = reactive.thin_volume()
        opaque = self.image(fixture, lambda x, y: 0.0)  # a path that did not set the bit
        row = reactive.evaluate_mask(opaque, fixture)
        self.assertFalse(reactive.glass_ok(row))
        self.assertTrue(reactive.opaque_ok(row))
        # a mask that marks one core pixel short, or a marked pixel clear of every sphere, also fails
        discs = [reactive.disc(fixture, c, 1.0) for c in fixture["spheres"]]
        short = self.image(fixture, lambda x, y: 0.749 if any(reactive.inside(d, x, y) for d in discs) and (x, y) != (int(discs[1][0]), int(discs[1][1])) else 0.0)
        self.assertFalse(reactive.glass_ok(reactive.evaluate_mask(short, fixture)))
        stray = self.image(fixture, lambda x, y: 0.749 if (any(reactive.inside(d, x, y) for d in discs) or (x, y) == (5, 5)) else 0.0)
        self.assertFalse(reactive.glass_ok(reactive.evaluate_mask(stray, fixture)))

    def test_every_fixture_disc_is_inside_the_frame_and_discs_do_not_overlap(self):
        for fixture in (reactive.thin_volume(), reactive.roughness_ior(), reactive.base_grid()):
            discs = [reactive.disc(fixture, c, reactive.DILATED) for c in fixture["spheres"]]
            for d in discs:
                self.assertGreater(d[0] - d[2], 0)
                self.assertLess(d[0] + d[2], reactive.WIDTH)
                self.assertGreater(d[1] - d[3], 0)
                self.assertLess(d[1] + d[3], reactive.HEIGHT)
            for i, a in enumerate(discs):
                for b in discs[i + 1:]:
                    self.assertGreater(math.hypot(a[0] - b[0], a[1] - b[1]), max(a[2], a[3]) + max(b[2], b[3]))


if __name__ == "__main__":
    unittest.main()
