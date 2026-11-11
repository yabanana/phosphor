#!/usr/bin/env python3
"""Reference roles of the F13/F14 runner cases and the gating rule."""
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import f13_f14_check as runner  # noqa: E402


class Roles(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.cases = {c["name"]: c for c in runner.make_cases(pathlib.Path(tmp), 300, "fallback")}

    def test_every_case_has_a_declared_role(self):
        allowed = {"independent-clip", "numerical", "full-rate-pair", "control", "functional"}
        for name, case in self.cases.items():
            self.assertIn(case["reference_role"], allowed, name)

    def test_f13_product_outputs_need_independent_clips(self):
        for scene in ("mirror", "roughness", "moving-light", "disocclusion"):
            self.assertEqual(self.cases[scene + "-custom"]["reference_role"], "independent-clip")
            self.assertEqual(self.cases[scene + "-raw"]["reference_role"], "control")
            self.assertEqual(self.cases[scene + "-rt8-control"]["reference_role"], "control")
        self.assertEqual(self.cases["mirror-ssr"]["reference_role"], "control")
        self.assertEqual(self.cases["ao-cavity-gtao"]["reference_role"], "independent-clip")

    def test_f14_roles_follow_the_plan(self):
        self.assertEqual(self.cases["atmo-zenith"]["reference_role"], "numerical")
        self.assertEqual(self.cases["fog-homogeneous-oracle"]["reference_role"], "numerical")
        self.assertEqual(self.cases["clouds-reconstructed"]["reference_role"], "full-rate-pair")
        self.assertEqual(self.cases["clouds-full"]["reference_role"], "control")
        self.assertEqual(self.cases["negative-f14-lut"]["reference_role"], "functional")

    def test_gating(self):
        ok = {"functional_passed": True}
        self.assertFalse(runner.gating_passed(dict(ok), "independent-clip"))  # missing reference never passes
        self.assertTrue(runner.gating_passed(dict(ok, independent_metrics={"passed": True}), "independent-clip"))
        self.assertFalse(runner.gating_passed(dict(ok, independent_metrics={"passed": False}), "independent-clip"))
        self.assertTrue(runner.gating_passed(dict(ok, independent_metrics={"passed": False}), "control"))
        self.assertFalse(runner.gating_passed(dict(ok, control_metrics={"passed": False}), "full-rate-pair"))
        # A noisy higher-sample control never vetoes a product output that matches the independent clip.
        self.assertTrue(runner.gating_passed(dict(ok, independent_metrics={"passed": True}, control_metrics={"passed": False}), "independent-clip"))
        self.assertFalse(runner.gating_passed(dict(ok, numerical={"passed": False}), "numerical"))
        self.assertTrue(runner.gating_passed(dict(ok, numerical={"passed": True}), "numerical"))
        self.assertFalse(runner.gating_passed({"functional_passed": False}, "control"))


if __name__ == "__main__":
    unittest.main()
