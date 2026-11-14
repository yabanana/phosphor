#!/usr/bin/env python3
"""tools/f15_permutations.py helpers: case selection, command retargeting, capture comparison (no GPU)."""
import pathlib
import struct
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_permutations as fp  # noqa: E402


def write_pfm(path, pixels, width, height):
    flat = [v for p in pixels for v in p]
    path.write_bytes(b"PF\n%d %d\n-1.0\n" % (width, height) + struct.pack(f"<{len(flat)}f", *flat))


class Permutations(unittest.TestCase):
    def test_every_manifest_scene_has_a_case_and_every_f15_key_of_a_fixture_is_a_case(self):
        scenes = fp.manifest_scenes()
        self.assertGreaterEqual(len(scenes), 81)
        cs = fp.cases(scenes)
        covered = {c["scene"]["name"] for c in cs}
        self.assertEqual(covered, {s["name"] for s in scenes})
        ids = [c["id"] for c in cs]
        self.assertEqual(len(ids), len(set(ids)))
        keyed = {c["id"] for c in cs if "#" not in c["id"]}
        for key in fp.fc.SCENARIOS:
            if fp.fc.SCENARIOS[key].get("capture_only") or fp.fc.bench_scenario(key) not in {s["scenario"] for s in scenes}:
                continue
            self.assertIn(key, keyed)  # the lit variants (restir, clustered) are cases too: both resolve kernels are exercised

    def test_retargeted_command(self):
        scene = next(s for s in fp.manifest_scenes() if s["scenario"] == "clearcoat-grid-sun")
        case = {"id": "x", "template": "base-grid-sun", "scene": scene}
        folder = pathlib.Path("/tmp/x")
        cmd = fp.base_command(pathlib.Path("/b/phosphor"), case, folder, 8, "640x360")
        self.assertEqual(cmd[cmd.index("--material-scenario") + 1], "clearcoat-grid-sun")
        self.assertEqual(cmd[cmd.index("--frames") + 1], "8")
        self.assertEqual(cmd[cmd.index("--capture-linear-frame") + 1], "7")
        self.assertEqual(cmd[cmd.index("--resolution") + 1], "640x360")
        self.assertEqual(cmd[cmd.index("--capture") + 1], str(folder / "final.png"))
        for flag in ("--capture-linear-accumulate", "--export-reference", "--export-reference-frame"):
            self.assertNotIn(flag, cmd)
        no_capture = fp.base_command(pathlib.Path("/b/phosphor"), case, folder, 8, "640x360", capture=False)
        for flag in ("--capture-linear", "--capture", "--capture-linear-frame"):
            self.assertNotIn(flag, no_capture)

    def test_flag_helpers(self):
        self.assertEqual(fp.set_flag(["a", "--x", "1"], "--x", "2"), ["a", "--x", "2"])
        self.assertEqual(fp.set_flag(["a"], "--x", "2"), ["a", "--x", "2"])
        self.assertEqual(fp.drop_flag(["a", "--x", "1", "b"], "--x"), ["a", "b"])
        self.assertEqual(fp.drop_flag(["a"], "--x"), ["a"])

    def test_compare_pfm(self):
        with tempfile.TemporaryDirectory() as t:
            t = pathlib.Path(t)
            a = [(1.0, 2.0, 3.0), (0.5, 0.25, 0.125), (4.0, 4.0, 4.0)]
            write_pfm(t / "a.pfm", a, 3, 1)
            write_pfm(t / "same.pfm", a, 3, 1)
            self.assertEqual(fp.compare_pfm(t / "a.pfm", t / "same.pfm"), {"pixels": 3, "bitwise": 0, "max_ulp": 0, "max_rel": 0.0})
            one_ulp = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", 4.0))[0] + 1))[0]
            b = [a[0], a[1], (4.0, one_ulp, 4.0)]
            write_pfm(t / "b.pfm", b, 3, 1)
            c = fp.compare_pfm(t / "a.pfm", t / "b.pfm")
            self.assertEqual((c["bitwise"], c["max_ulp"]), (1, 1))
            self.assertLess(c["max_rel"], 2e-7)
            self.assertTrue(fp.within_envelope(c))
            big = [a[0], (0.5, 0.26, 0.125), a[2]]
            write_pfm(t / "c.pfm", big, 3, 1)
            c = fp.compare_pfm(t / "a.pfm", t / "c.pfm")
            self.assertFalse(fp.within_envelope(c))  # a missing lobe is far outside the codegen envelope
            write_pfm(t / "d.pfm", a[:2], 2, 1)
            self.assertEqual(fp.compare_pfm(t / "a.pfm", t / "d.pfm")["bitwise"], 3)  # size mismatch: everything differs

    def test_ordered_is_monotonic_across_zero(self):
        bits = lambda f: struct.unpack("<i", struct.pack("<f", f))[0]  # noqa: E731
        values = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        self.assertEqual([fp.ordered(bits(v)) for v in values], sorted(fp.ordered(bits(v)) for v in values))


if __name__ == "__main__":
    unittest.main()
