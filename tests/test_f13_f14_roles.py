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
        allowed = {"independent-clip", "numerical", "full-rate-pair", "repeatability-pair", "control", "functional"}
        for name, case in self.cases.items():
            self.assertIn(case["reference_role"], allowed, name)

    def test_f13_product_outputs_need_independent_clips(self):
        for scene in ("mirror", "roughness", "moving-light", "disocclusion"):
            self.assertEqual(self.cases[scene + "-custom"]["reference_role"], "independent-clip")
            self.assertEqual(self.cases[scene + "-raw"]["reference_role"], "control")
            self.assertEqual(self.cases[scene + "-rt8-control"]["reference_role"], "control")
        self.assertEqual(self.cases["mirror-ssr"]["reference_role"], "control")
        self.assertEqual(self.cases["ao-cavity-gtao"]["reference_role"], "independent-clip")

    def test_ghost_definition_per_phase(self):
        self.assertEqual(self.cases["moving-light-custom"]["roi"]["rois"][0]["ghost_metric"], "error-change")
        for name in ("clouds-reconstructed", "clouds-flight-top-reconstructed", "fog-moving-light", "fog-disocclusion"):
            for roi in self.cases[name]["roi"]["rois"]:
                self.assertEqual(roi["ghost_metric"], "error-change", name)

    def test_f14_roles_follow_the_plan(self):
        self.assertEqual(self.cases["atmo-zenith"]["reference_role"], "numerical")
        self.assertEqual(self.cases["fog-homogeneous-oracle"]["reference_role"], "numerical")
        self.assertEqual(self.cases["clouds-reconstructed"]["reference_role"], "full-rate-pair")
        self.assertEqual(self.cases["clouds-full"]["reference_role"], "control")
        self.assertEqual(self.cases["negative-f14-lut"]["reference_role"], "functional")
        self.assertEqual(self.cases["daynight-a"]["reference_role"], "numerical")
        self.assertEqual(self.cases["daynight-r1"]["reference_role"], "numerical")
        self.assertEqual(self.cases["daynight-r2"]["reference_role"], "repeatability-pair")
        self.assertEqual(self.cases["daynight-r2"]["validation"], "api")
        self.assertNotIn("validation", self.cases["daynight-a"])
        self.assertEqual(self.cases["atmo-extreme-dense"]["reference_role"], "numerical")

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
        self.assertFalse(runner.gating_passed(dict(ok), "repeatability-pair"))
        self.assertTrue(runner.gating_passed(dict(ok, repeatability={"passed": True}), "repeatability-pair"))

    def test_f14_2_and_f14_3_cases_follow_the_plan(self):
        c = self.cases
        def core(flags):  # every F14 case appends its own oracle directory last
            i = flags.index("--volume-oracle"); return flags[:i] + flags[i + 2:]
        for name in ("fog-slicing", "fog-slicing-height"):
            self.assertEqual(c[name]["reference_role"], "numerical")
            self.assertEqual(c[name]["scene"], "fog-depths")
        self.assertIn("--fog-homogeneous", c["fog-slicing"]["flags"])
        self.assertEqual(c["fog-slicing-height"]["fog_fixture_falloff"], 0.05)
        for name in ("fog-moving-light", "fog-disocclusion"):
            self.assertEqual(c[name]["reference_role"], "full-rate-pair")
            self.assertEqual(c[name + "-control"]["reference_role"], "control")
            control = c[name + "-control"]["flags"]
            self.assertEqual(control[control.index("--fog-history") + 1], "off")
            self.assertEqual(core(c[name + "-control"]["flags"]), core(c[name]["flags"]) + ["--fog-history", "off", "--fog-local-samples", "16"])
        self.assertTrue(c["fog-disocclusion"]["require_history_reuse_frames"])
        events = {e["name"]: e["frame"] for e in c["fog-moving-light"]["roi"]["events"]}
        self.assertEqual(events, {"emissive-step": 119, "time-jump-60": 60, "time-jump-120": 120, "time-jump-180": 180, "time-jump-240": 240})
        for flight in ("flight-base", "flight-top", "backlit", "screen-edge", "screen-edge-deck"):
            for suffix in ("", "-t"):
                recon, full = c[f"clouds-{flight}{suffix}-reconstructed"], c[f"clouds-{flight}{suffix}-full"]
                self.assertEqual(recon["reference_role"], "full-rate-pair")
                self.assertEqual(full["reference_role"], "control")
                self.assertEqual(core(recon["flags"]) + ["--cloud-full-rate"], core(full["flags"]))
                self.assertTrue(recon["require_history_reuse_frames"])
                self.assertEqual(recon["coverage_check"], suffix == "-t")
                self.assertEqual(recon["signal"], "cloud-transmittance" if suffix else "hdr")
                self.assertEqual(recon["roi"]["rois"][0]["scale"], 1 if suffix else 25)
        self.assertEqual([r["name"] for r in c["clouds-backlit-reconstructed"]["roi"]["rois"]], ["above-sun", "beside-sun-left", "beside-sun-right"])
        for flight in ("flight-base", "flight-top"):
            flags = c[f"clouds-{flight}-reconstructed"]["flags"]
            self.assertLess(float(flags[flags.index("--camera-climb") + 1]) / 60, 4.0)  # below the camera-cut distance
        self.assertNotIn("clouds-flight-reconstructed", c)
        self.assertEqual([r["name"] for r in c["clouds-screen-edge-t-reconstructed"]["roi"]["rois"]], ["full-frame", "entering-band", "entering-edge", "leaving-edge"])
        for negative, pair in (("negative-f14-fog-stale", "fog-moving-light-control"), ("negative-f14-cloud-edge", "clouds-screen-edge-deck-t-full")):
            self.assertTrue(c[negative]["expected_pair_failure"])
            self.assertEqual(c[negative]["quality_pair"], pair)
            self.assertEqual(c[negative]["expected_exit"], 0)
        self.assertNotIn("negative-f14-cloud-motion", c)
        edge = c["negative-f14-cloud-edge"]
        self.assertEqual(edge["signal"], "cloud-transmittance")
        self.assertEqual(core(edge["flags"]), core(c["clouds-screen-edge-deck-t-reconstructed"]["flags"]) + ["--debug-volume-corrupt", "cloud-edge"])
        self.assertEqual(edge["roi"], c["clouds-screen-edge-deck-t-reconstructed"]["roi"])

    def test_pair_negative_and_coverage_gating(self):
        ok = {"functional_passed": True}
        negative = {"expected_pair_failure": True}
        self.assertTrue(runner.gating_passed(dict(ok, control_metrics={"passed": False, "regions": {}}), "full-rate-pair", negative))
        self.assertFalse(runner.gating_passed(dict(ok, control_metrics={"passed": True, "regions": {}}), "full-rate-pair", negative))
        self.assertFalse(runner.gating_passed(dict(ok, control_metrics={"passed": False, "error": "missing"}), "full-rate-pair", negative))
        coverage = {"coverage_check": True}
        self.assertFalse(runner.gating_passed(dict(ok, control_metrics={"passed": True}), "full-rate-pair", coverage))
        self.assertTrue(runner.gating_passed(dict(ok, control_metrics={"passed": True}, coverage={"passed": True}), "full-rate-pair", coverage))

    def test_coverage_counts_covered_fraction_per_frame(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            a, b = pathlib.Path(tmp, "a"), pathlib.Path(tmp, "b"); a.mkdir(); b.mkdir()
            def write(folder, name, values):
                payload = b"".join(struct.pack("<fff", v, v, v) for v in values)
                (folder / name).write_bytes(b"PF\n10 1\n-1\n" + payload)
            config = {"rois": [{"name": "all", "normalized": [0, 0, 1, 1]}]}
            write(a, "frame-000000.pfm", [0.2] * 5 + [1.0] * 5); write(b, "frame-000000.pfm", [0.2] * 5 + [1.0] * 5)
            r = runner.coverage_evidence(a, b, config)
            self.assertTrue(r["passed"]); self.assertEqual(r["max_delta"], 0)
            write(b, "frame-000000.pfm", [0.2] * 6 + [1.0] * 4)
            r = runner.coverage_evidence(a, b, config)
            self.assertFalse(r["passed"]); self.assertAlmostEqual(r["max_delta"], 0.1)
            write(a, "frame-000000.pfm", [1.0] * 10); write(b, "frame-000000.pfm", [1.0] * 10)
            self.assertFalse(runner.coverage_evidence(a, b, config)["exercised"])  # no cloud at all never passes

    def test_repeatability_compares_frames_bitwise(self):
        import struct
        with tempfile.TemporaryDirectory() as tmp:
            a, b = pathlib.Path(tmp, "a"), pathlib.Path(tmp, "b"); a.mkdir(); b.mkdir()
            def write(folder, name, value):
                (folder / name).write_bytes(b"PF\n1 1\n-1\n" + struct.pack("<fff", value, value, value))
            for i in range(3):
                write(a, f"frame-{i:06d}.pfm", 1.0); write(b, f"frame-{i:06d}.pfm", 1.0)
            self.assertTrue(runner.repeatability(a, b, 3)["passed"])
            write(b, "frame-000002.pfm", 1.5)
            r = runner.repeatability(a, b, 3)
            self.assertFalse(r["passed"]); self.assertEqual(r["differing_frames"], ["frame-000002.pfm"])
            self.assertAlmostEqual(r["max_abs_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
