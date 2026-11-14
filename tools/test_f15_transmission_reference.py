#!/usr/bin/env python3
"""Tests of tools/f15_transmission_reference.py (the Mitsuba ground truth of ticket 13), of the transmission gate and keys of
tools/f15_check.py and of the fixtures' discrimination.

The Mitsuba tests path-trace tiny glass-mask crops (320 x 180 snapshots of the bench's numbers, tools/f15_transmission_fixtures.py)
and compare them with the ANALYTIC sphere tracer of the module (an exact expectation of the Fresnel-chosen reflect/refract tree:
no Mitsuba), which validates the whole chain (pixel-centre rays, medium units, tint, thin-walled plugin, Beer-Lambert).

  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_transmission_reference.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import f15_check as check  # noqa: E402
import f15_reference as ref  # noqa: E402
import f15_transmission_fixtures as fx  # noqa: E402
import f15_transmission_reference as tr  # noqa: E402
from f12_reference import read_pfm, snapshot, write_pfm  # noqa: E402

SKY = [0.5] * 3
SPP = 512  # tiny crops: the tests compare against analytic values with a tolerance that covers the sampling noise


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return tr.main(argv)


class AnalyticModel(unittest.TestCase):
    def test_fresnel_values(self):
        for function in (tr.fresnel_exact, tr.fresnel_schlick):
            self.assertAlmostEqual(float(function(1.0, 1.5)), 0.04, places=6)
            self.assertAlmostEqual(float(function(1.0, 2.0)), ((2.0 - 1.0) / 3.0) ** 2, places=6)
            self.assertEqual(float(function(0.1, 1.0 / 1.5)), 1.0)  # leaving glass beyond the critical angle: TIR
        # Schlick is the glTF approximation of the exact curve: the largest gap entering glass over all angles (measured:
        # 0.060 at ior 1.33, 0.036 at 1.5, 0.044 at 2.0, all at N.V about 0.1, i.e. near the silhouette) bounds the Fresnel
        # part of the glTF-vs-Mitsuba model difference
        cos = np.linspace(0.02, 1.0, 400)
        for eta, bound in ((1.33, 0.065), (1.5, 0.04), (2.0, 0.05)):
            gap = float(np.max(np.abs(tr.fresnel_exact(cos, eta) - tr.fresnel_schlick(cos, eta))))
            self.assertLess(gap, bound, (eta, gap))
        # ... and at grazing incidence both go to 1
        self.assertGreater(float(tr.fresnel_exact(0.001, 1.5)), 0.99)
        self.assertGreater(float(tr.fresnel_schlick(0.001, 1.5)), 0.99)

    def test_beer_lambert_coefficients(self):
        sigma = tr.sigma_t([0.5, 0.25, 1.0], 2.0)
        np.testing.assert_allclose(sigma, [math.log(2) / 2, math.log(4) / 2, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.exp(-sigma * 2.0), [0.5, 0.25, 1.0], atol=1e-12)  # T(d) = attenuationColor
        self.assertTrue((tr.sigma_t([0.1, 0.1, 0.1], math.inf) == 0).all())
        self.assertTrue((tr.sigma_t([0.0, 1.0, 1.0], 1.0) >= 0).all())

    def test_roughness_clamp_and_alpha(self):
        self.assertAlmostEqual(tr.mitsuba_alpha(0.0), 0.002)   # clamp 0.04 -> 0.0016 -> floor 0.002
        self.assertAlmostEqual(tr.mitsuba_alpha(0.1), 0.01)
        self.assertAlmostEqual(tr.mitsuba_alpha(0.6), 0.36)
        self.assertAlmostEqual(tr.mitsuba_alpha(2.0), 1.0)
        self.assertAlmostEqual(tr.ior_f0(1.5), 0.04)

    def pixels(self, radius=0.4, n=200):
        x = np.linspace(-0.85, 0.85, n) * radius
        origin = np.array([0.0, 0.0, 5.5])
        target = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
        direction = target - origin
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return np.broadcast_to(origin, (n, 3)).copy(), direction

    glass = lambda self, **k: {"ior": 1.5, "base": [1, 1, 1], "sigma_t": np.zeros(3), "thin": False, **k}  # noqa: E731

    def test_a_lossless_sphere_in_a_uniform_environment_conserves_energy(self):
        o, d = self.pixels()
        sphere = {"centre": np.zeros(3), "radius": 0.4}
        for thin in (False, True):
            for fresnel in ("exact", "schlick"):
                for ior in (1.33, 1.5, 2.0):
                    image = tr.trace_sphere(o, d, sphere, self.glass(thin=thin, ior=ior), np.zeros((0, 8)), SKY, fresnel)
                    np.testing.assert_allclose(image, 0.5, atol=2e-3, err_msg=f"thin={thin} {fresnel} ior={ior}")

    def test_the_crossing_bound_is_visible_only_when_a_path_is_cut_to_black(self):
        o, d = self.pixels()
        sphere = {"centre": np.zeros(3), "radius": 0.4}
        g = self.glass(ior=2.0)
        env = tr.trace_sphere(o, d, sphere, g, np.zeros((0, 8)), SKY, "exact", 4, "env")
        zero = tr.trace_sphere(o, d, sphere, g, np.zeros((0, 8)), SKY, "exact", 4, "zero")
        np.testing.assert_allclose(env, 0.5, atol=2e-3)  # a uniform environment hides a cut that ends in the environment
        self.assertTrue((zero <= 0.5 + 1e-9).all())
        self.assertLess(float(zero.mean()), 0.5)
        self.assertGreater(float(zero.mean()), 0.45)     # only the multiply-reflected tail is lost: bounded, small

    def test_absorption_tint_and_monotonicity(self):
        o, d = self.pixels(n=50)
        sphere = {"centre": np.zeros(3), "radius": 0.4}
        none = tr.trace_sphere(o, d, sphere, self.glass(), np.zeros((0, 8)), SKY)
        weak = tr.trace_sphere(o, d, sphere, self.glass(sigma_t=tr.sigma_t([0.8, 0.8, 0.8], 1.0)), np.zeros((0, 8)), SKY)
        strong = tr.trace_sphere(o, d, sphere, self.glass(sigma_t=tr.sigma_t([0.2, 0.5, 0.9], 0.2)), np.zeros((0, 8)), SKY)
        self.assertTrue((weak < none).all())
        self.assertTrue((strong < weak).all() or float(strong.mean()) < float(weak.mean()))
        self.assertTrue((strong[:, 0] < strong[:, 1]).all() and (strong[:, 1] < strong[:, 2]).all())  # red absorbed most
        tinted = tr.trace_sphere(o, d, sphere, self.glass(base=[0.9, 0.5, 0.25]), np.zeros((0, 8)), SKY)
        self.assertTrue((tinted[:, 0] > tinted[:, 1]).all() and (tinted[:, 1] > tinted[:, 2]).all())
        self.assertTrue((tinted < none).all())

    def test_central_ray_through_a_lossless_sphere_sees_the_emitter_behind_it(self):
        # normal incidence: F = 0.04 per interface; the straight central ray exits towards the emitter at z = -3
        rect = np.array([[-3.0, 0.0, 0.0, 5.0, 5.0, 2.0, 2.0, 2.0]])  # emits 2.0 over the whole screen region behind
        origin = np.array([[0.0, 0.0, 5.5]])
        direction = np.array([[0.0, 0.0, -1.0]])
        sphere = {"centre": np.zeros(3), "radius": 0.4}
        image = tr.trace_sphere(origin, direction, sphere, self.glass(), rect, SKY)
        # all paths end at the emitter (2.0) except those reflected back to the camera side (environment 0.5)
        f = 0.04
        # series: the central ray bounces between the two surfaces; each time it reaches the back surface it leaves with (1 - f)
        # towards the emitter and each time it reaches the front surface from inside it leaves with (1 - f) towards the camera
        to_emitter = (1 - f) * (1 - f) / (1 - f * f)
        to_camera = f + (1 - f) * (1 - f) * f / (1 - f * f)
        self.assertAlmostEqual(to_emitter + to_camera, 1.0, places=9)
        self.assertAlmostEqual(float(image[0, 0]), 2.0 * to_emitter + 0.5 * to_camera, places=4)

    def test_emitters_face_plus_z_only(self):
        rect = np.array([[-3.0, 0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 4.0]])
        o = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]])
        d = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        out = tr.exit_radiance(o, d, rect, SKY)
        np.testing.assert_allclose(out[0], 0.5)  # approaching the rectangle from behind: not emitted
        np.testing.assert_allclose(out[1], 4.0)


class SnapshotParsing(unittest.TestCase):
    def snapshot(self, objects, emitters=(), **kw):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        fx.write_snapshot(tmp, objects, list(emitters), **kw)
        return snapshot(tmp)

    def test_glass_table_reads_the_extension_records(self):
        root, data = self.snapshot([fx.glass(thickness=0.0, base=fx.TINT, ior=1.33, transmission=0.5),
                                    fx.glass(attenuation_color=(0.4, 0.8, 1.0), attenuation_distance=0.8, roughness=0.3)],
                                   fx.checker_cells()[:3])
        glass = tr.glass_table(data)
        self.assertEqual(sorted(glass), [0, 1])
        thin, solid = glass[0], glass[1]
        self.assertTrue(thin["thin"] and not solid["thin"])
        self.assertAlmostEqual(thin["ior"], 1.33, places=6)
        self.assertAlmostEqual(thin["transmission"], 0.5)
        self.assertEqual(thin["base"], [0.9, 0.55, 0.25])
        self.assertTrue(solid["has_medium"] and not thin["has_medium"])
        np.testing.assert_allclose(solid["sigma_t"], tr.sigma_t([0.4, 0.8, 1.0], 0.8), rtol=1e-6)
        self.assertEqual(sorted(tr.emitters(data)), [2, 3, 4])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "rough thin"):
            tr.glass_table(self.snapshot([fx.glass(thickness=0.0, roughness=0.2)])[1])
        tr.glass_table(self.snapshot([fx.glass(thickness=0.0, roughness=0.2)])[1], allow_rough_thin=True)  # the declared model accepts it
        with self.assertRaisesRegex(ValueError, "no transmissive"):
            tr.glass_table(self.snapshot([fx.glass(transmission=0.0, thickness=0.0)])[1])
        root, data = self.snapshot([fx.glass()])
        data["material_extensions"]["records"][1][10] = 3  # a transmission texture
        with self.assertRaisesRegex(ValueError, "texture"):
            tr.glass_table(data)
        root, data = self.snapshot([fx.glass()])
        data["materials"][0]["feature_bits"] |= 128
        with self.assertRaisesRegex(ValueError, "outside"):
            tr.glass_table(data)
        root, data = self.snapshot([fx.glass(ior=0.0)])
        with self.assertRaisesRegex(ValueError, "ior"):
            tr.glass_table(data)
        root, data = self.snapshot([fx.glass()])
        data["lights"] = [{"type": 0, "intensity": 3.0}]
        with self.assertRaisesRegex(ValueError, "analytic lights"):
            tr.check_scene(data)

    def test_camera_rays_are_the_pixel_centre_rays_of_the_reference_cast(self):
        root, data = self.snapshot([fx.glass()], fx.checker_cells()[:2])
        origin, directions = tr.camera_directions(data)
        cast = ref.cast_primary_rays(root, data, 2)
        valid, position = cast[0].reshape(-1), cast[1]
        pick = np.flatnonzero(valid)[::37]
        t = np.linalg.norm(position[pick] - origin, axis=1)
        np.testing.assert_allclose(origin + t[:, None] * directions[pick], position[pick], atol=2e-5)

    def test_glass_mask_excludes_a_two_pixel_silhouette_band(self):
        root, data = self.snapshot([fx.glass(), fx.glass(center=(1.2, 1, 0))], fx.checker_cells()[:2])
        glass = tr.glass_table(data)
        cast = ref.cast_primary_rays(root, data, 2)
        instance_map, roi = tr.glass_masks(cast, glass)
        self.assertTrue(roi.any() and (instance_map >= 0).sum() > roi.sum())
        ys, xs = np.nonzero(roi)
        for y, x in list(zip(ys, xs))[::11]:
            window = instance_map[y - 2:y + 3, x - 2:x + 3]
            self.assertTrue((window == instance_map[y, x]).all())

    def test_backface_removal_keeps_the_camera_facing_half(self):
        root, data = self.snapshot([fx.glass()])
        with tempfile.TemporaryDirectory() as tmp:
            path = tr.front_facing_mesh(root, data, 0, np.array(data["camera"]["position"]), pathlib.Path(tmp))
            properties, faces = ref.read_ply(path)
            _, all_faces = ref.read_ply(root / "mesh_0.ply")
            self.assertLess(len(faces), len(all_faces))
            # a camera at distance d sees (1 - R / d) / 2 of a sphere: 0.464 at d = 5.5, R = 0.4 (a tessellation keeps a bit less)
            self.assertAlmostEqual(len(faces) / len(all_faces), 0.464, delta=0.06)
            # the kept triangles face the camera: their normals have positive z (the camera looks down -Z from z = 5.5)
            p = np.stack([properties["x"], properties["y"], properties["z"]], axis=1)
            normal = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
            centroid = p[faces].mean(axis=1) + np.array(data["instances"][0]["world"][12:15])
            camera = np.array(data["camera"]["position"])
            self.assertTrue((np.einsum("ij,ij->i", normal, centroid - camera) < 0).all())


class Blend(unittest.TestCase):
    def test_transmission_factor_blend(self):
        i0, i1 = np.full((3, 3), 0.2), np.full((3, 3), 1.0)
        out = tr.blend_transmission_factor(i0, i1, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(out[:, 0], [0.2, 0.6, 1.0])


class DeclaredModel(unittest.TestCase):
    def arrays(self, n=5):
        return dict(base=np.tile([0.9, 0.55, 0.25], (n, 1)), roughness=np.full(n, 0.1), ior=np.full(n, 1.5), t=np.ones(n),
                    thickness=np.zeros(n), att_color=np.ones((n, 3)), att_distance=np.full(n, math.inf),
                    n_dot_v=np.linspace(0.2, 1.0, n))

    def run_model(self, variant="spec", **override):
        a = {**self.arrays(), **override}
        return tr.declared_pixel(a["base"], a["roughness"], a["ior"], a["t"], a["thickness"], a["att_color"], a["att_distance"],
                                 a["n_dot_v"], 0.5, variant)

    def test_opaque_limit_equals_the_engines_declared_environment_model(self):
        a = self.arrays()
        opaque = self.run_model("declared-transmission-ignored")
        expected = ref.declared_environment_radiance(a["base"], np.zeros(5), a["roughness"], a["n_dot_v"], 0.5, ref.DECLARED)
        np.testing.assert_allclose(opaque, expected, rtol=2e-4)

    def test_thin_white_glass_and_opaque_white_are_the_same_in_a_uniform_environment(self):
        white = dict(base=np.ones((5, 3)))
        np.testing.assert_allclose(self.run_model(**white), self.run_model("declared-transmission-ignored", **white), rtol=1e-9)

    def test_volume_attenuation_tint_and_factor(self):
        colour, distance = np.tile([0.4, 0.8, 1.0], (5, 1)), np.full(5, 0.8)
        volume = dict(thickness=np.full(5, 0.8), att_color=colour, att_distance=distance, base=np.ones((5, 3)))
        attenuated = self.run_model(**volume)
        clear = self.run_model("declared-volume-ignored", **volume)
        self.assertTrue((attenuated[:, 0] < clear[:, 0]).all())
        np.testing.assert_allclose(attenuated[:, 2], clear[:, 2], rtol=1e-6)  # attenuation colour 1 in blue: untouched
        half = self.run_model(t=np.full(5, 0.5), **volume)
        self.assertTrue((half[:, 0] > attenuated[:, 0]).all() and (half[:, 0] < clear[:, 0] + 1e-12).all())
        untinted = self.run_model("declared-not-tinted")
        self.assertTrue((untinted[:, 2] > self.run_model()[:, 2]).all())


class Runner(unittest.TestCase):
    def test_every_transmission_key_is_no_longer_pending_and_keeps_the_ignored_control(self):
        # The engine side landed with F15.1 ticket 13: every gate must pass; the opaque (transmission-ignored) control stays as a
        # negative control that must FAIL the gate.
        keys = [k for k in check.SCENARIOS if k.startswith("transmission-")]
        self.assertEqual(len(keys), 8)
        for key in keys:
            spec = check.SCENARIOS[key]
            for field in ("pending", "pending_matches", "pending_gate"):
                self.assertNotIn(field, spec, key)
            declared = spec["kind"] == "transmission-declared"
            self.assertIn("declared-transmission-ignored" if declared else "transmission-ignored", spec["controls"])
            allowed = tr.DECLARED_CONTROLS if declared else tr.CONTROLS
            self.assertTrue(set(spec["controls"]) <= set(allowed), key)
            self.assertGreaterEqual(len(spec["controls"]), 2, key)
            self.assertEqual(spec["kind"] in check.TRANSMISSION_KINDS, True)
        self.assertEqual(sum(1 for k in keys if check.SCENARIOS[k]["kind"] == "transmission-declared"), 1)

    def test_capture_commands_accumulate_and_trace_the_glass(self):
        binary = pathlib.Path("build/release/phosphor")
        for key in (k for k in check.SCENARIOS if k.startswith("transmission-")):
            spec = check.SCENARIOS[key]
            c = check.capture_command(binary, key, pathlib.Path("out"))
            self.assertEqual(c[c.index("--material-scenario") + 1], key)
            if spec["kind"] == "transmission":
                self.assertEqual(c[c.index("--capture-linear-accumulate") + 1], str(check.ACCUMULATE))
                self.assertLessEqual(check.ACCUMULATE, int(c[c.index("--capture-linear-frame") + 1]) + 1)
                self.assertEqual(c[c.index("--rt") + 1], "on", key)  # the glass is always traced
                self.assertEqual(c[c.index("--reflections") + 1], "rt", key)
            else:  # --rt off, probes
                self.assertEqual(c[c.index("--rt") + 1], "off")
                self.assertEqual(c[c.index("--reflections") + 1], "probes")
                self.assertNotIn("--capture-linear-accumulate", c)
        emissive = check.capture_command(binary, "transmission-emissive-panel", pathlib.Path("out"))
        self.assertEqual(emissive[emissive.index("--lighting") + 1], "restir")

    def test_reference_commands_name_the_transmission_module(self):
        for key in (k for k in check.SCENARIOS if k.startswith("transmission-")):
            command = check.reference_command(key, pathlib.Path("out"))
            joined = " ".join(command)
            self.assertIn("f15_transmission_reference.py", joined)
            self.assertIn("declared" if check.SCENARIOS[key]["kind"] == "transmission-declared" else "render", command)
            for control in check.SCENARIOS[key]["controls"]:
                self.assertIn(control, command)

    def objects(self, per=12):
        return [{"instance": k, "label": f"o{k}", "roughness": 0.0} for k in range(2)]

    def masks(self):
        sphere = np.full((20, 40), -1, dtype=np.int32)
        sphere[2:18, 2:18] = 0
        sphere[2:18, 22:38] = 1
        return {"roi": sphere >= 0, "sphere": sphere}

    def test_transmission_gate_bounds(self):
        masks = self.masks()
        rng = np.random.default_rng(3)
        reference = np.full((20, 40, 3), 1.0, dtype=np.float32) + rng.random((20, 40, 3)).astype(np.float32)
        gate = check.THRESHOLDS["transmission"]
        self.assertEqual((gate["mean_per_object"], gate["p95_per_object"]), (0.05, 0.15))
        self.assertTrue(check.evaluate(np, "transmission", reference.copy(), reference, masks, self.objects())["pass"])
        # a constant 4.5 % shortfall passes both bounds, 5.5 % fails the mean
        floor = 0.05 * reference[masks["roi"]].mean()
        for scale, passes in ((0.955, True), (0.94, False)):
            candidate = reference * scale
            result = check.evaluate(np, "transmission", candidate, reference, masks, self.objects())
            expected_error = (reference * (1 - scale) / np.maximum(reference, floor)).max(axis=-1)[masks["roi"]].mean()
            self.assertEqual(result["pass"], passes, (scale, expected_error))
        # p95: five per cent of the pixels of one object 20 % off fails the p95 bound while the mean stays below 5 %
        candidate = reference.copy()
        pixels = np.argwhere(masks["sphere"] == 1)
        for y, x in pixels[: int(len(pixels) * 0.10)]:
            candidate[y, x] *= 0.8
        result = check.evaluate(np, "transmission", candidate, reference, masks, self.objects())
        worst = result["objects"][1]
        self.assertLess(worst["mean"], 0.05)
        self.assertGreater(worst["p95"], 0.15)
        self.assertFalse(result["pass"])
        # an object without pixels cannot pass
        self.assertFalse(check.evaluate(np, "transmission", reference, reference, masks, [{"instance": 7, "label": "x"}])["pass"])
        # the same gate serves the declared kind
        self.assertTrue(check.evaluate(np, "transmission-declared", reference.copy(), reference, masks, self.objects())["pass"])

    def test_summary_lists_the_failing_objects(self):
        masks = self.masks()
        reference = np.ones((20, 40, 3), dtype=np.float32)
        candidate = reference.copy()
        candidate[masks["sphere"] == 1] *= 0.5
        summary = check.summarize(check.evaluate(np, "transmission", candidate, reference, masks, self.objects()))
        self.assertEqual(summary["objects_failing"], ["o1"])
        self.assertNotIn("objects", summary)
        lines = "\n".join(check.result_lines(check.evaluate(np, "transmission", candidate, reference, masks, self.objects())))
        self.assertIn("| o1 |", lines)

    def test_pending_semantics_match_the_other_tickets(self):
        failing = {"kind": "transmission", "pass": False, "objects": []}
        passing = {"kind": "transmission", "pass": True, "objects": []}
        for mode in ("red", "green"):
            self.assertTrue(check.expectation("transmission", mode, failing, pending="ticket 13")["met"])
            self.assertFalse(check.expectation("transmission", mode, passing, pending="ticket 13")["met"])


class EndToEnd(unittest.TestCase):
    """A tiny run folder: the engine of today = the transmission-ignored control; the pending gate must FAIL against the Mitsuba
    reference, PASS against the control, and every control of the key must be discriminating."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.folder = pathlib.Path(cls.tmp.name) / "key"
        objects = [fx.glass(center=(-1.3, 1, 0), thickness=0.0, base=fx.TINT, transmission=1.0),
                   fx.glass(center=(0.0, 1, 0), thickness=0.8),
                   fx.glass(center=(1.3, 1, 0), thickness=0.8, attenuation_color=(0.4, 0.8, 1.0), attenuation_distance=0.8)]
        fx.write_snapshot(cls.folder / "snapshot", objects, fx.checker_cells())
        cls.controls = ["transmission-ignored", "transmission-not-tinted", "volume-ignored", "ior-ignored", "thin-as-volume"]
        cls.key = "transmission-test-key"
        check.SCENARIOS[cls.key] = {"kind": "transmission", "pending": "ticket 13", "pending_matches": "transmission-ignored",
                                    "controls": cls.controls, "spp": SPP, "terms": "test"}
        with contextlib.redirect_stderr(io.StringIO()):
            code = quiet_main(["render", str(cls.folder / "snapshot"), "--out", str(cls.folder / "reference"), "--spp", str(SPP),
                               "--controls", *cls.controls])
        assert code == 0
        control = read_pfm(cls.folder / "reference" / "control-transmission-ignored.pfm")
        write_pfm(cls.folder / "engine.pfm", control)  # "today's engine"
        cls.entry = check.evaluate_transmission_scenario(np, cls.key, cls.folder, "green")

    @classmethod
    def tearDownClass(cls):
        check.SCENARIOS.pop(cls.key, None)
        cls.tmp.cleanup()

    def test_the_engine_of_today_fails_the_gate_and_matches_the_ignored_control(self):
        self.assertFalse(self.entry["gates_pass"])
        self.assertTrue(self.entry["pending_matches_met"])
        self.assertTrue(self.entry["expectation_met"])
        self.assertEqual(check.verdict([self.entry], "green"), ("GREEN", check.EXIT_OK))

    def test_every_control_is_discriminating(self):
        for name, verdict in self.entry["controls"].items():
            self.assertTrue(verdict["fails_gate"], (name, verdict))
        self.assertTrue(self.entry["discriminating"])

    def test_reference_matches_the_independent_analytic_tree(self):
        root, data = snapshot(self.folder / "snapshot")
        glass = tr.glass_table(data)
        image = read_pfm(self.folder / "reference" / "reference.pfm")
        masks = np.load(self.folder / "reference" / "masks.npz")
        origin, directions = tr.camera_directions(data)
        rectangles = tr.backdrop_rectangles(data)
        # the thin sphere follows the glTF (Schlick) rule, the solids Mitsuba's exact Fresnel
        for k, g in sorted(glass.items()):
            pixels = np.flatnonzero(masks["roi"].reshape(-1) & (masks["sphere"].reshape(-1) == k))
            self.assertGreater(len(pixels), 100)
            sphere = tr.sphere_of(root, data, k)
            expected = tr.trace_sphere(np.broadcast_to(origin, (len(pixels), 3)).copy(), directions[pixels], sphere, g, rectangles,
                                       SKY, "schlick" if g["thin"] else "exact")
            seen = image.reshape(-1, 3)[pixels]
            floor = 0.05 * float(expected.mean())
            error = (np.abs(seen - expected) / np.maximum(expected, floor)).max(axis=1)
            # the analytic tree is an exact sphere, the snapshot a 64 x 32 tessellation with interpolated normals: a handful of
            # pixels on the sharp checkerboard edges move by a fraction of a pixel (measured max 270 % on 3 of 250 pixels), so the
            # robust statistics gate: the channel means agree within 1.5 %, the median and the 90th percentile are small
            np.testing.assert_allclose(seen.mean(axis=0), expected.mean(axis=0), rtol=0.015, err_msg=g["label"])
            self.assertLess(float(np.median(error)), 0.04, g["label"])
            self.assertLess(float(np.percentile(error, 90)), 0.10, g["label"])

    def test_reference_noise_is_recorded(self):
        report = json.loads((self.folder / "reference" / "reference.json").read_text())
        noise = report["reference"]["noise"]
        self.assertLess(noise["mean_relative_standard_error"], 0.04)
        self.assertIn("per_object", noise)
        self.assertEqual(report["spp"], SPP)
        self.assertEqual(report["reference"]["non_finite_samples"], 0)
        self.assertEqual([o["thin"] for o in report["objects"]], [True, False, False])

    def test_the_engine_matching_the_reference_would_pass(self):
        reference = read_pfm(self.folder / "reference" / "reference.pfm")
        masks = dict(np.load(self.folder / "reference" / "masks.npz"))
        report = json.loads((self.folder / "reference" / "reference.json").read_text())
        self.assertTrue(check.evaluate_transmission(np, reference.copy(), reference, masks, report["objects"])["pass"])

    def test_cached_reference_is_reused_only_for_the_same_snapshot_tool_and_spp(self):
        reference_dir, snapshot_dir = self.folder / "reference", self.folder / "snapshot"
        self.assertTrue(check.cached_transmission_reference_usable(reference_dir, snapshot_dir, self.controls, SPP))
        self.assertFalse(check.cached_transmission_reference_usable(reference_dir, snapshot_dir, self.controls, SPP * 2))
        self.assertFalse(check.cached_transmission_reference_usable(reference_dir, snapshot_dir, self.controls + ["backface-rejected"], SPP))


class DeclaredEndToEnd(unittest.TestCase):
    def test_declared_run_fails_for_todays_engine_and_the_controls_discriminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp) / "key"
            objects = [fx.glass(center=(-1.3, 1, 0), thickness=0.0, roughness=0.1, base=fx.TINT),
                       fx.glass(center=(0.0, 1, 0), thickness=0.8, roughness=0.1, attenuation_color=(0.4, 0.8, 1.0),
                                attenuation_distance=0.8)]
            fx.write_snapshot(folder / "snapshot", objects, [], width=320, height=180)
            controls = ["declared-transmission-ignored", "declared-not-tinted", "declared-volume-ignored"]
            check.SCENARIOS["transmission-test-declared"] = {
                "kind": "transmission-declared", "pending": "ticket 13", "pending_matches": "declared-transmission-ignored",
                "controls": controls, "terms": "test"}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = tr.main(["declared", str(folder / "snapshot"), "--out", str(folder / "reference"), "--controls", *controls])
                self.assertEqual(code, 0)
                write_pfm(folder / "engine.pfm", read_pfm(folder / "reference" / "control-declared-transmission-ignored.pfm"))
                entry = check.evaluate_transmission_scenario(np, "transmission-test-declared", folder, "green")
            finally:
                check.SCENARIOS.pop("transmission-test-declared", None)
            self.assertFalse(entry["gates_pass"])
            self.assertTrue(entry["pending_matches_met"])
            self.assertTrue(entry["discriminating"], entry["controls"])
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", check.EXIT_OK))


class ControlsOnEveryFixtureShape(unittest.TestCase):
    """The controls that need their own geometry (back faces, emitter doubling, glass behind glass, slabs, panes)."""

    def run_controls(self, objects, emitters, controls, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        snap = fx.write_snapshot(pathlib.Path(tmp.name) / "snapshot", objects, emitters, **kw)
        out = pathlib.Path(tmp.name) / "ref"
        self.assertEqual(quiet_main(["render", str(snap), "--out", str(out), "--spp", "64", "--controls", *controls]), 0)
        report = json.loads((out / "reference.json").read_text())
        reference = read_pfm(out / "reference.pfm")
        masks = dict(np.load(out / "masks.npz"))
        verdicts = {}
        for name in controls:
            result = check.evaluate_transmission(np, read_pfm(out / f"control-{name}.pfm"), reference, masks, report["objects"])
            verdicts[name] = result
        return verdicts

    def test_single_sided_cube_and_lens_backface_control(self):
        objects = [fx.glass(center=(-0.9, 1, 0), thickness=0.8),
                   fx.glass(shape="cube", center=(0.9, 1, 0), scale=(0.8, 0.8, 0.8), yaw=25, pitch=15, thickness=1.0)]
        verdicts = self.run_controls(objects, fx.checker_cells(), ["backface-rejected", "ior-ignored"], width=480, height=270,
                                     distance=4.5)
        for name, result in verdicts.items():
            self.assertFalse(result["pass"], (name, result["worst_mean"]))
            self.assertTrue(all(not o["pass"] for o in result["objects"]), name)

    def test_emitter_double_count_and_tint_controls_on_an_emissive_panel_pane(self):
        pane = fx.glass(shape="pane", center=(0, 1.2, 0.2), scale=(1.2, 1, 1.2), thickness=0.0, base=fx.TINT, double_sided=True)
        panel = [(0.0, 1.2, -0.6, 1.0, 1.0)]
        verdicts = self.run_controls([pane], panel, ["emitter-double-counted", "transmission-not-tinted", "transmission-ignored"],
                                     emission=3.0, distance=4.5, camera_target=(0, 1.2, 0))
        for name, result in verdicts.items():
            self.assertFalse(result["pass"], (name, result["worst_mean"]))

    def test_glass_behind_glass_is_not_black_and_the_black_control_is_discriminated(self):
        objects = [fx.glass(center=(-0.25, 1, 0.7)),
                   fx.glass(center=(0.25, 1, -0.7), attenuation_color=(0.5, 0.8, 1.0), attenuation_distance=0.6)]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        snap = fx.write_snapshot(pathlib.Path(tmp.name) / "snapshot", objects, fx.checker_cells(), width=480, height=270)
        out = pathlib.Path(tmp.name) / "ref"
        self.assertEqual(quiet_main(["render", str(snap), "--out", str(out), "--spp", "64", "--controls", "back-glass-black"]), 0)
        reference, black = read_pfm(out / "reference.pfm"), read_pfm(out / "control-back-glass-black.pfm")
        masks = np.load(out / "masks.npz")
        back = masks["roi"] & (masks["sphere"] == 1)
        self.assertTrue(back.any())
        self.assertGreater(float(reference[back].mean()), 0.3)  # not black
        self.assertLess(float(black[back].mean()), 0.6 * float(reference[back].mean()))  # the control is

    def test_pane_with_a_transmission_factor_blends_with_the_opaque_image(self):
        pane = lambda t: fx.glass(shape="pane", center=(-1.0, 1, 0), scale=(0.9, 1, 0.9), thickness=0.0, base=fx.TINT, transmission=t,  # noqa: E731
                                  double_sided=True)
        images = {}
        for t in (1.0, 0.5):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            snap = fx.write_snapshot(pathlib.Path(tmp.name) / "snapshot", [pane(t)], fx.checker_cells())
            out = pathlib.Path(tmp.name) / "ref"
            self.assertEqual(quiet_main(["render", str(snap), "--out", str(out), "--spp", "32", "--controls", "transmission-ignored"]), 0)
            masks = np.load(out / "masks.npz")
            images[t] = (read_pfm(out / "reference.pfm"), read_pfm(out / "control-transmission-ignored.pfm"), masks["roi"])
        roi = images[1.0][2]
        full, opaque = images[1.0][0][roi], images[1.0][1][roi]
        half = images[0.5][0][roi]
        # glTF mix(diffuse, btdf, t) is linear in t: the t = 0.5 image is the mean of the t = 1 image and the opaque one
        np.testing.assert_allclose(half, 0.5 * full + 0.5 * opaque, rtol=0.08, atol=0.03)


if __name__ == "__main__":
    unittest.main(verbosity=2)
