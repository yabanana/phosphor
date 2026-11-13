#!/usr/bin/env python3
"""Tests of tools/f15_check.py (runner logic) and of the fixture design.

The bench-mirror tests rebuild the four material scenarios as snapshots (the
numbers mirror src/testbench/material_validation.cpp, which tests/test_material_
validation.cpp pins), ray cast them with Mitsuba, emulate the CURRENT engine
model on the reference side (diffuse weighted by the Fresnel of the
metallic-blended f0; environment diffuse without Fresnel weighting) and check
that the runner reports the expected red state, that the corrected model turns
it green, and that every negative control is discriminating.

  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_check.py
"""
from __future__ import annotations

import io
import contextlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import f15_check as check  # noqa: E402
import f15_reference as ref  # noqa: E402
from f12_reference import write_pfm  # noqa: E402

try:
    import mitsuba  # noqa: F401
    HAVE_MITSUBA = True
except ImportError:  # pragma: no cover
    HAVE_MITSUBA = False

ROUGHNESS = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
METALLIC = [0.0, 0.25, 0.5, 0.75, 1.0]
COPPER = [0.8, 0.35, 0.15]
TOWARD_SUN = np.array([-0.8, 0.35, -0.3]) / np.linalg.norm([-0.8, 0.35, -0.3])
SPECS = {
    "base-grid-sun": dict(roughness=ROUGHNESS, metallic=METALLIC, base=COPPER, sun=3.0, env=0.0),
    "base-grid-env": dict(roughness=ROUGHNESS, metallic=METALLIC, base=COPPER, sun=0.0, env=0.5),
    "furnace-base": dict(roughness=ROUGHNESS, metallic=METALLIC, base=[1, 1, 1], sun=0.0, env=1.0),
    "roughness-extremes": dict(roughness=[0.0, 1.0], metallic=[0.0, 0.5, 1.0], base=COPPER, sun=3.0, env=0.3),
}


def write_sphere_ply(path, radius=0.4, slices=64, stacks=32):
    import struct
    verts, faces = [], []
    for st in range(stacks + 1):
        theta = math.pi * st / stacks
        for sl in range(slices + 1):
            phi = 2 * math.pi * sl / slices
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            verts.append((*(c * radius for c in n), *n, sl / slices, st / stacks))
    for st in range(stacks):
        for sl in range(slices):
            a = st * (slices + 1) + sl
            b = a + slices + 1
            faces += [(a, b, a + 1), (a + 1, b, b + 1)]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\n"
                 "property float t\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                 % (len(verts), len(faces))).encode())
        for v in verts:
            f.write(struct.pack("<8f", *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


def make_bench_snapshot(folder, name, width, height):
    spec = SPECS[name]
    folder = pathlib.Path(folder)
    (folder / "snapshot").mkdir(parents=True)
    write_sphere_ply(folder / "snapshot" / "mesh_0.ply")
    cols, rows = len(spec["roughness"]), len(spec["metallic"])
    instances, materials = [], []
    for row in range(rows):
        for col in range(cols):
            x, y = -0.5 * (cols - 1) + col, float(row)
            world = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, 0, 1]
            materials.append({"base": [*spec["base"], 1], "emissive": [0, 0, 0], "metallic": spec["metallic"][row],
                              "roughness": spec["roughness"][col], "normal_scale": 1, "occlusion_strength": 1,
                              "alpha_cutoff": 0, "flags": 0, "textures": [ref.INVALID] * 5})
            instances.append({"slot": len(instances), "generation": 1, "flags": 7, "mesh": 0,
                              "material": len(materials) - 1, "world": world})
    distance = 5.5 if cols > 2 else 4.0
    data = {"schema": 1, "state": "TEST", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
            "camera": {"position": [0, 0.5 * (rows - 1), distance], "direction": [0, 0, -1], "up": [0, 1, 0],
                       "fov_y_radians": math.radians(60), "near": 0.05, "far": 0, "width": width, "height": height,
                       "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"},
            "sky": [spec["env"]] * 3, "sun_angular_radius": 0.00465, "textures": [], "materials": materials,
            "meshes": ["mesh_0.ply"], "instances": instances, "lights": [], "sampled_lights": []}
    if spec["sun"] > 0:
        data["lights"].append({"type": 0, "position": [0, 0, 0], "direction": list(-TOWARD_SUN), "color": [1, 1, 1],
                               "intensity": spec["sun"], "range": 0, "inner": 0, "outer": 0})
    (folder / "snapshot" / "scene.json").write_text(json.dumps(data))
    return folder


class RunnerLogic(unittest.TestCase):
    def test_terminator_band_excludes_only_lit_pixels_next_to_unlit_ones_of_the_same_sphere(self):
        # Two spheres side by side; sphere 1 lit on its left half, sphere 2 fully lit.
        lit = np.array([[1, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1, 1]], dtype=bool)
        sphere = np.array([[1, 1, 1, 1, 2, 2],
                           [1, 1, 1, 1, 2, 2]], dtype=np.int32)
        band = check.terminator_band(np, lit, sphere)
        expected = np.array([[0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0]], dtype=bool)
        np.testing.assert_array_equal(band, expected)

    def test_plan_mode_prints_exact_capture_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = check.main(["--out", tmp, "--binary", "build/release/phosphor"])
            text = buffer.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("NOT_EXECUTED", text)
            plan = json.loads((pathlib.Path(tmp) / "plan.json").read_text())
            self.assertEqual(sorted(plan["commands"]), sorted(check.SCENARIOS))
            for name, commands in plan["commands"].items():
                capture = commands["capture"]
                self.assertEqual(capture[capture.index("--material-scenario") + 1], name)
                self.assertEqual(capture[capture.index("--bench") + 1], "2")
                for flag in ("--capture-linear", "--export-reference", "--reflections", "--resolution"):
                    self.assertIn(flag, capture)
                self.assertIn(" ".join(capture), text)
                self.assertIn("f15_reference.py", " ".join(commands["reference"]))
            for key in ("commit", "os", "sdk_version", "power", "tool_sha256", "seed", "thresholds"):
                self.assertIn(key, plan)

    def test_process_guard_and_lock(self):
        ps = "  1 /sbin/launchd\n 4242 /Users/x/build/release/phosphor\n 77 /usr/bin/python3\n 88 f13_spike\n 99 soc_bench\n"
        self.assertEqual(check.running_gpu_processes(ps), ["4242:phosphor", "88:f13_spike", "99:soc_bench"])
        self.assertEqual(check.running_gpu_processes("1 launchd\n2 phosphor_tests\n"), [])
        with tempfile.TemporaryDirectory() as tmp:
            lock = pathlib.Path(tmp) / "gpu.lock"
            with check.gpu_lock(lock):
                with self.assertRaises(RuntimeError):
                    with check.gpu_lock(lock):
                        pass
            with check.gpu_lock(lock):  # released afterwards
                pass

    def test_run_refuses_while_another_gpu_job_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = pathlib.Path(tmp) / "phosphor"
            binary.write_bytes(b"x")
            original = check.running_gpu_processes
            check.running_gpu_processes = lambda *a, **k: ["31337:phosphor"]
            try:
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = check.main(["--run", "--binary", str(binary), "--out", str(pathlib.Path(tmp) / "o"),
                                       "--gpu-lock", str(pathlib.Path(tmp) / "l")])
            finally:
                check.running_gpu_processes = original
            self.assertEqual(code, 4)
            self.assertIn("refusing to start", err.getvalue())

    def synthetic_masks(self, n=40):
        sphere = np.zeros((n, n), dtype=np.int32)
        sphere[:, n // 2:] = 1
        roi = np.ones((n, n), dtype=bool)
        return {"roi": roi, "sphere": sphere, "narrow": np.zeros((n, n), dtype=bool), "lit": roi.copy()}, \
            [{"instance": 0, "metallic": 0.0, "roughness": 0.5}, {"instance": 1, "metallic": 0.5, "roughness": 0.5}]

    def test_direct_gate_floor_mask_and_rows(self):
        masks, spheres = self.synthetic_masks()
        reference = np.full((40, 40, 3), 1.0, dtype=np.float32)
        candidate = reference.copy()
        candidate[:, 20:] *= 1.02  # row metallic 0.5: 2 % everywhere -> p99 2 % fails
        result = check.evaluate_direct(np, candidate, reference, masks, spheres)
        self.assertTrue(result["rows"]["0.0"]["pass"])
        self.assertFalse(result["rows"]["0.5"]["pass"])
        # One 20 % outlier in 1600 pixels inside the mask only: p99 unaffected, max exempt.
        candidate = reference.copy()
        candidate[3, 3] = 1.2
        masks["narrow"][3, 3] = True
        self.assertTrue(check.evaluate_direct(np, candidate, reference, masks, spheres)["rows"]["0.0"]["pass"])
        masks["narrow"][3, 3] = False  # same outlier outside the mask: max > 3 % fails
        self.assertFalse(check.evaluate_direct(np, candidate, reference, masks, spheres)["rows"]["0.0"]["pass"])
        # Absolute floor: tiny reference values do not blow the ratio up.
        reference = np.full((40, 40, 3), 1.0, dtype=np.float32)
        reference[:2] = 1e-6
        candidate = reference.copy()
        candidate[:2] = 4e-4  # 0.84 % of the 0.05 * mean floor, but 400 times the reference itself
        self.assertTrue(check.evaluate_direct(np, candidate, reference, masks, spheres)["pass"])

    def test_environment_and_furnace_gates(self):
        masks, spheres = self.synthetic_masks()
        reference = np.full((40, 40, 3), 0.5, dtype=np.float32)
        candidate = reference.copy()
        candidate[:, 20:] *= 1.06
        env = check.evaluate_environment(np, candidate, reference, masks, spheres)
        self.assertTrue(env["spheres"][0]["pass"])
        self.assertFalse(env["spheres"][1]["pass"])
        self.assertFalse(env["pass"])
        furnace_reference = np.full((40, 40, 3), 0.9, dtype=np.float32)
        ok = check.evaluate_furnace(np, np.full((40, 40, 3), 1.004, dtype=np.float32), furnace_reference, masks, spheres)
        bad = check.evaluate_furnace(np, np.full((40, 40, 3), 1.006, dtype=np.float32), furnace_reference, masks, spheres)
        self.assertTrue(ok["pass"])
        self.assertFalse(bad["pass"])
        self.assertAlmostEqual(ok["spheres"][0]["energy_loss"], 1 - 1.004, places=6)

    def test_verdict_blocks_non_discriminating_control(self):
        def entry(discriminating, met=True, passed=False):
            return {"discriminating": discriminating, "expectation": {"met": met}, "result": {"pass": passed}}
        self.assertEqual(check.verdict([entry(True), entry(False)], "red")[1], 3)
        self.assertEqual(check.verdict([entry(True), entry(True)], "red"), ("EXPECTED_RED_CONFIRMED", 0))
        self.assertEqual(check.verdict([entry(True, met=False)], "red")[1], 1)
        self.assertEqual(check.verdict([entry(True, passed=True)], "green"), ("GREEN", 0))
        self.assertEqual(check.verdict([entry(True, passed=False)], "green")[1], 1)


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba not installed")
class BenchMirror(unittest.TestCase):
    """Red state of the current engine model, green state of the Appendix B model, control discrimination."""

    WIDTH, HEIGHT = 480, 270

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scenario(self, name, engine_model, mode):
        folder = pathlib.Path(self.tmp.name) / f"{name}-{mode}"
        make_bench_snapshot(folder, name, self.WIDTH, self.HEIGHT)
        from f12_reference import snapshot as load
        root, data = load(folder / "snapshot")
        image, *_ = ref.render_frame(root, data, engine_model)
        write_pfm(folder / "engine.pfm", image)
        with contextlib.redirect_stdout(io.StringIO()):
            return check.evaluate_scenario(np, name, folder, mode)

    def test_current_engine_model_is_red_where_expected(self):
        engine = {"base-grid-sun": ref.CONTROLS["engine-f0-blend"],
                  "furnace-base": ref.CONTROLS["diffuse-unweighted"]}
        entries = []
        for name, model in engine.items():
            entry = self.scenario(name, model, "red")
            entries.append(entry)
            self.assertTrue(entry["discriminating"], (name, entry["controls"]))
            self.assertTrue(entry["expectation"]["met"], (name, json.dumps(entry["expectation"], indent=1)))
        direct = entries[0]["result"]["rows"]
        self.assertTrue(direct["0.0"]["pass"] and direct["1.0"]["pass"])
        for middle in ("0.25", "0.5", "0.75"):
            self.assertFalse(direct[middle]["pass"], middle)
        self.assertFalse(entries[1]["result"]["pass"])
        self.assertEqual(check.verdict(entries, "red"), ("EXPECTED_RED_CONFIRMED", 0))

    def test_corrected_model_is_green_and_controls_discriminate(self):
        entries = [self.scenario(name, ref.APPENDIX_B, "green") for name in ("base-grid-sun", "base-grid-env",
                                                                           "roughness-extremes")]
        for entry in entries:
            self.assertTrue(entry["result"]["pass"], entry["scenario"])
            self.assertTrue(entry["discriminating"], (entry["scenario"], entry["controls"]))
        self.assertEqual(check.verdict(entries, "green"), ("GREEN", 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
