#!/usr/bin/env python3
"""Tests of tools/f15_shadow_reference.py (the Mitsuba ground truth of tickets 14/15), of the transmissive-shadow keys and gate of
tools/f15_check.py and of the fixtures' discrimination.

The Mitsuba tests render 320 x 180 crops of the bench's world-space numbers (tools/f15_shadow_fixtures.py: the frame is smaller, the
geometry identical) and compare them with CLOSED FORMS that do not use Mitsuba: the Schlick interface of a thin-walled pane, the
exact slab series of a thick pane ((1 - F)^2 / (1 - F^2) with exact Fresnel, with Beer-Lambert along the refracted path).

  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_shadow_reference.py
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
import f15_shadow_fixtures as fx  # noqa: E402
import f15_shadow_reference as sr  # noqa: E402
import f15_transmission_reference as tr  # noqa: E402
from f12_reference import read_pfm, snapshot, write_pfm  # noqa: E402

KEYS = [k for k in check.SCENARIOS if k.startswith("transmissive-shadow-")]
GATED = list(KEYS)  # every key has a gate (the curved one a qualitative gate)
PENDING = [k for k in KEYS if k.endswith("@csm")]  # ticket 15; the RT, DI and curved keys landed with ticket 14
CURVED = "transmissive-shadow-curved@rt"
TRUTH_KEYS = [k for k in KEYS if k != CURVED]


def quiet_main(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return sr.main(argv)


def render(folder, bench, margin, controls=(), extra=()):
    fx.write_snapshot(folder / "snapshot", bench)
    code = quiet_main(["render", str(folder / "snapshot"), "--out", str(folder / "reference"), "--margin", str(margin), *extra,
                       "--controls", *controls])
    assert code == 0
    report = json.loads((folder / "reference" / "reference.json").read_text())
    return report, read_pfm(folder / "reference" / "reference.pfm"), dict(np.load(folder / "reference" / "masks.npz"))


class Pins(unittest.TestCase):
    """The numbers of the C++ bench (tests/test_transmissive_shadow_validation.cpp pins the same ones)."""

    def test_fixture_numbers(self):
        s = fx.SCENARIOS
        self.assertEqual(sorted(s), ["transmissive-shadow-curved", "transmissive-shadow-pane-sun", "transmissive-shadow-point",
                                     "transmissive-shadow-volume-pane", "transmissive-shadow-water-sun"])
        pane = s["transmissive-shadow-pane-sun"]["casters"]
        self.assertEqual([c["center"][0] for c in pane], [-2.1, -0.7, 0.7, 2.1])
        self.assertEqual(pane[1]["base"], (0.4, 0.9, 0.45))
        self.assertEqual(pane[2]["scale"], (0.9, 0.06, 0.9))
        self.assertTrue(pane[3]["opaque"])
        self.assertAlmostEqual(s["transmissive-shadow-pane-sun"]["sun"][0][1], math.cos(math.radians(25)))
        water = s["transmissive-shadow-water-sun"]["casters"]
        self.assertEqual([c["ior"] for c in water], [1.33, 1.33, 1.5])
        self.assertEqual((water[1]["attenuation_color"], water[1]["attenuation_distance"]), ((0.45, 0.75, 0.9), 0.3))
        volume = s["transmissive-shadow-volume-pane"]["casters"]
        self.assertEqual([c["attenuation_distance"] for c in volume], [0.2, 0.2, 0.15])
        point = s["transmissive-shadow-point"]
        self.assertEqual(point["points"], [((0.0, 4.5, -0.3), 60.0, 100.0)])
        self.assertEqual([c["scale"][0] for c in point["casters"]], [0.7] * 4)
        self.assertEqual(s["transmissive-shadow-curved"]["casters"][1]["attenuation_distance"], 0.6)

    def test_every_sun_is_within_thirty_degrees_of_the_normal(self):
        for name, spec in fx.SCENARIOS.items():
            if spec["sun"]:
                self.assertGreaterEqual(spec["sun"][0][1], math.cos(math.radians(30)) - 1e-9, name)


class ClosedForms(unittest.TestCase):
    def test_declared_model_values_on_a_synthetic_hit_list(self):
        casters = {1: {"opaque": False, "ior": 1.5, "base": [1.0, 1.0, 1.0], "transmission": 1.0, "thin": True, "has_medium": False,
                       "sigma_t": np.zeros(3)},
                   2: {"opaque": False, "ior": 1.5, "base": [0.5, 1.0, 0.25], "transmission": 0.5, "thin": True, "has_medium": False,
                       "sigma_t": np.zeros(3)},
                   3: {"opaque": False, "ior": 1.5, "base": [1.0, 1.0, 1.0], "transmission": 1.0, "thin": False, "has_medium": True,
                       "sigma_t": np.array([0.0, 1.0, 2.0])},
                   4: {"opaque": True, "ior": 1.5, "base": [0.5] * 3, "transmission": 0.0, "thin": False, "has_medium": False,
                       "sigma_t": np.zeros(3)}}
        # one ray per row: a thin pane, a half-transmissive tinted pane, a slab (entry -0.1, exit +0.3 along the ray), an opaque hit, none
        instance = np.array([[1, -1, -1], [2, -1, -1], [3, 3, -1], [4, -1, -1], [-1, -1, -1]])
        distance = np.array([[1.0, 0, 0], [1.0, 0, 0], [2.0, 2.2, 0], [1.0, 0, 0], [0, 0, 0]])
        cosine = np.array([[-1.0, 0, 0], [-1.0, 0, 0], [-1.0, 1.0, 0], [-1.0, 0, 0], [0, 0, 0]])  # head-on: F = f0 = 0.04
        t = sr.declared_transmittance((instance, distance, cosine), casters)
        np.testing.assert_allclose(t[0], [0.96] * 3, rtol=1e-12)
        np.testing.assert_allclose(t[1], 0.5 * 0.96 * np.array([0.5, 1.0, 0.25]), rtol=1e-12)
        np.testing.assert_allclose(t[2], 0.96 ** 2 * np.exp(-0.2 * np.array([0.0, 1.0, 2.0])), rtol=1e-12)  # 0.2 inside
        np.testing.assert_allclose(t[3], 0.0)
        np.testing.assert_allclose(t[4], 1.0)
        hits = (instance, distance, cosine)
        np.testing.assert_allclose(sr.declared_transmittance(hits, casters, "tint-ignored")[1], 0.5 * 0.96, rtol=1e-12)
        np.testing.assert_allclose(sr.declared_transmittance(hits, casters, "fresnel-ignored")[0], 1.0)
        np.testing.assert_allclose(sr.declared_transmittance(hits, casters, "beer-lambert-ignored")[2], 0.96 ** 2, rtol=1e-12)
        np.testing.assert_allclose(sr.declared_transmittance(hits, casters, "double-counted")[0], 0.96 ** 2, rtol=1e-12)
        np.testing.assert_allclose(sr.declared_transmittance(hits, casters, "transmittance-ignored"), [[0.0] * 3] * 4 + [[1.0] * 3])

    def test_radical_inverse_is_a_low_discrepancy_prefix(self):
        values = sr.radical_inverse(np.arange(8), 2)
        np.testing.assert_allclose(values, [0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875])
        # the first n points of the sequence put n F +- 2 of them below any threshold F (discrepancy O(log n))
        for n in (57, 150, 256):
            sample = sr.radical_inverse(np.arange(n), 2)
            for f in (0.04, 0.2):
                self.assertLessEqual(abs(int((sample < f).sum()) - n * f), 2.0)

    def test_cone_directions_are_uniform_in_solid_angle_inside_the_cone(self):
        axis = np.array([[0.0, 0.6, 0.8]])
        d = sr.cone_directions(axis, np.array([0.05]), 400, 0.0)[0]
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
        angle = np.arccos(np.clip(d @ axis[0], -1, 1))
        self.assertLessEqual(float(angle.max()), 0.05)
        inner = float((angle <= 0.05 / math.sqrt(2)).mean())  # half the solid angle lies inside radius / sqrt 2
        self.assertAlmostEqual(inner, 0.5, delta=0.01)


class MitsubaTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def transmittance(self, folder, report, image, masks, instance):
        """Mean RGB transmittance of one region: the reference over the unshadowed floor radiance (lit.pfm), per pixel."""
        lit = read_pfm(folder / "reference" / "lit.pfm")
        pick = masks["sphere"] == instance
        self.assertGreater(int(pick.sum()), 150, instance)
        return (image[pick] / lit[pick]).mean(axis=0)

    def test_pane_sun_against_closed_forms(self):
        folder = self.root / "pane"
        report, image, masks = render(folder, "transmissive-shadow-pane-sun", 0.06)
        n_dot_l = fx.SCENARIOS["transmissive-shadow-pane-sun"]["sun"][0][1]
        self.assertEqual([o["pixels"] > 150 for o in report["objects"]], [True] * 4)
        f = 0.04 + 0.96 * (1 - n_dot_l) ** 5  # Schlick at the pane normal, the glTF interface (thin-walled: one surface)
        mean = {o["instance"]: self.transmittance(folder, report, image, masks, o["instance"]) for o in report["objects"]}
        np.testing.assert_allclose(mean[1], 1 - f, atol=0.006)                                 # thin clear: (1 - F)
        np.testing.assert_allclose(mean[2], (1 - f) * np.array(fx.GREEN), atol=0.006)          # thin green: (1 - F) tint
        f_exact = float(tr.fresnel_exact(n_dot_l, 1.5))
        np.testing.assert_allclose(mean[3], (1 - f_exact) / (1 + f_exact), atol=0.008)         # thick clear: the slab series
        np.testing.assert_allclose(mean[4], 0.0, atol=1e-9)                                    # opaque
        self.assertLess(report["reference"]["noise"]["mean_relative_standard_error"], 0.01)

    def test_attenuating_slab_follows_the_refracted_beer_lambert_path(self):
        folder = self.root / "volume"
        report, image, masks = render(folder, "transmissive-shadow-volume-pane", 0.06)
        cos_a = fx.SCENARIOS["transmissive-shadow-volume-pane"]["sun"][0][1]
        cos_t = math.sqrt(1 - (1 - cos_a ** 2) / 1.5 ** 2)
        f = float(tr.fresnel_exact(cos_a, 1.5))
        for entry, c in zip(report["objects"], fx.SCENARIOS["transmissive-shadow-volume-pane"]["casters"]):
            sigma = tr.sigma_t(c["attenuation_color"], c["attenuation_distance"])
            a = np.exp(-sigma * c["scale"][1] / cos_t)
            expected = (1 - f) ** 2 * a / (1 - f ** 2 * a ** 2)  # slab series with absorption
            seen = self.transmittance(folder, report, image, masks, entry["instance"])
            np.testing.assert_allclose(seen, expected, rtol=0.012, err_msg=str(entry["instance"]))

    def test_declared_model_agrees_with_the_truth_on_every_gated_fixture(self):
        # the engine implementing the declaration must be able to meet the bound: the model passes the gate with room to spare
        for bench in ("transmissive-shadow-pane-sun", "transmissive-shadow-water-sun", "transmissive-shadow-volume-pane",
                      "transmissive-shadow-point"):
            folder = self.root / ("model-" + bench)
            report, _, _ = render(folder, bench, 0.06)
            for instance, stats in report["model"]["vs_truth"].items():
                self.assertLess(stats["mean"], 0.025, (bench, instance, stats))
                self.assertLess(stats["p95"], 0.05, (bench, instance, stats))

    def test_receiver_regions_are_floor_pixels_inside_one_shadow_and_wider_margins_shrink_them(self):
        folder = self.root / "regions"
        fx.write_snapshot(folder / "snapshot", "transmissive-shadow-pane-sun")
        counts = []
        for margin in (0.03, 0.06, 0.12):
            code = quiet_main(["render", str(folder / "snapshot"), "--out", str(folder / f"m{margin}"), "--margin", str(margin)])
            self.assertEqual(code, 0)
            masks = np.load(folder / f"m{margin}" / "masks.npz")
            counts.append(int(masks["roi"].sum()))
            self.assertTrue(((masks["sphere"] >= 0) == masks["roi"]).all())
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        self.assertGreater(counts[2], 0)
        # the primary hit of every region pixel is the floor (instance 0), never a caster
        root, data = snapshot(folder / "snapshot")
        import f15_reference as ref
        cast = ref.cast_primary_rays(root, data, 2)
        masks = np.load(folder / "m0.06" / "masks.npz")
        self.assertTrue((cast[3][masks["roi"]] == 0).all())

    def test_curved_key_reference_is_the_declared_model(self):
        folder = self.root / "curved"
        report, image, masks = render(folder, "transmissive-shadow-curved", 0.06, extra=["--no-ground-truth"])
        self.assertEqual(report["kind"], "declared-model")
        for entry in report["objects"]:
            self.assertGreater(entry["pixels"], 150)
        clear, murky = (image[masks["sphere"] == o["instance"]].mean(axis=0) for o in report["objects"])
        self.assertTrue((murky <= clear * (1 + 1e-5)).all() and (murky < clear).any())  # the absorbing sphere casts the darker (red, green) shadow
        self.assertTrue((clear > 0).all())


class Runner(unittest.TestCase):
    def test_every_gated_key_is_pending_and_models_todays_engine(self):
        self.assertEqual(sorted(KEYS), sorted([
            "transmissive-shadow-pane-sun@rt", "transmissive-shadow-pane-sun@csm", "transmissive-shadow-water-sun@rt",
            "transmissive-shadow-water-sun@csm", "transmissive-shadow-volume-pane@rt", "transmissive-shadow-volume-pane@csm",
            "transmissive-shadow-point@restir", "transmissive-shadow-curved@rt"]))
        for key in TRUTH_KEYS:
            spec = check.SCENARIOS[key]
            if key.endswith("@csm"):
                self.assertEqual(spec["pending"], "ticket 15", key)
                self.assertEqual(spec["pending_matches"], "transmittance-ignored")
            else:
                self.assertNotIn("pending", spec, key)  # ticket 14 landed: the RT and DI keys are gated green
                self.assertNotIn("pending_matches", spec, key)
            self.assertIn("transmittance-ignored", spec["controls"])
            self.assertTrue(set(spec["controls"]) <= set(sr.CONTROLS), key)
            self.assertEqual(spec["envelope"], "csm" if key.endswith("@csm") else "rt", key)
            self.assertEqual(spec["reflections"], "off")
        curved = check.SCENARIOS[CURVED]
        self.assertEqual(curved["kind"], "transmissive-shadow-curved")
        self.assertNotIn("pending", curved)
        self.assertNotIn("always_green", curved)
        self.assertEqual(curved["controls"], ["shadow-ignored", "transmittance-ignored"])
        self.assertEqual(curved["reference_args"], ["--no-ground-truth"])
        for key in KEYS:  # orchestrator decision: the CSM keys gate the transmittance ratio, the others the absolute bound
            self.assertEqual(check.SCENARIOS[key]["kind"], "transmissive-shadow-ratio" if key.endswith("@csm") else
                             "transmissive-shadow-curved" if key == CURVED else "transmissive-shadow", key)

    def test_dropped_controls_are_the_ones_equal_to_the_reference_within_the_bound(self):
        # Fresnel-ignored on water (2 % per surface, 4.6 %) and every control that needs an ingredient the fixture lacks
        water = check.SCENARIOS["transmissive-shadow-water-sun@rt"]["controls"]
        self.assertNotIn("fresnel-ignored", water)
        self.assertNotIn("tint-ignored", water)
        for key in ("transmissive-shadow-pane-sun@rt", "transmissive-shadow-point@restir"):
            self.assertNotIn("beer-lambert-ignored", check.SCENARIOS[key]["controls"])  # no attenuation on those fixtures
            self.assertIn("tint-ignored", check.SCENARIOS[key]["controls"])
        self.assertNotIn("tint-ignored", check.SCENARIOS["transmissive-shadow-volume-pane@rt"]["controls"])  # white base

    def test_capture_commands(self):
        binary = pathlib.Path("build/release/phosphor")
        for key in KEYS:
            spec = check.SCENARIOS[key]
            c = check.capture_command(binary, key, pathlib.Path("out"))
            self.assertEqual(c[c.index("--material-scenario") + 1], check.bench_scenario(key))
            self.assertEqual(c[c.index("--reflections") + 1], "off")
            self.assertEqual(c[c.index("--capture-linear-accumulate") + 1], str(check.SHADOW_ACCUMULATE))
            self.assertLessEqual(check.SHADOW_ACCUMULATE, int(c[c.index("--capture-linear-frame") + 1]) + 1)
            if key.endswith("@csm"):
                self.assertEqual(c[c.index("--shadows") + 1], "csm")
                self.assertEqual(c[c.index("--rt") + 1], "off")
            elif key.endswith("@restir"):
                self.assertEqual(c[c.index("--lighting") + 1], "restir")
                self.assertEqual(c[c.index("--rt") + 1], "on")
                self.assertEqual(c[c.index("--lighting-denoise") + 1], "off")
            else:
                self.assertEqual(c[c.index("--shadows") + 1], "rt")
                self.assertEqual(c[c.index("--rt") + 1], "on")
            del spec

    def test_reference_commands_name_the_shadow_module_and_the_declared_envelope(self):
        for key in KEYS:
            command = check.reference_command(key, pathlib.Path("out"))
            self.assertIn("f15_shadow_reference.py", " ".join(command))
            self.assertEqual(command[command.index("--margin") + 1], str(check.SHADOW_ENVELOPE[check.SCENARIOS[key]["envelope"]]))
            for control in check.SCENARIOS[key]["controls"]:
                self.assertIn(control, command)
        self.assertIn("--no-ground-truth", check.reference_command("transmissive-shadow-curved@rt", pathlib.Path("out")))
        self.assertEqual(check.SHADOW_ENVELOPE, {"rt": 0.06, "csm": 0.08})  # declared before any measurement

    def test_csm_and_rt_keys_share_the_bench_scene(self):
        for bench in ("pane-sun", "water-sun", "volume-pane"):
            rt, csm = (check.SCENARIOS[f"transmissive-shadow-{bench}@{v}"] for v in ("rt", "csm"))
            self.assertEqual(rt["controls"], csm["controls"])
            self.assertEqual((rt["kind"], csm["kind"]), ("transmissive-shadow", "transmissive-shadow-ratio"))


class EndToEnd(unittest.TestCase):
    """The runner's own evaluation of every gated key on a 320 x 180 crop: an engine with opaque shadows (the control of today's
    engine) is red for the right reason, the declared model is green, and every control discriminates."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.entries = {}
        cls.folders = {}
        for key in KEYS:
            spec = check.SCENARIOS[key]
            folder = pathlib.Path(cls.tmp.name) / key.replace("@", "_")
            cls.folders[key] = folder
            fx.write_snapshot(folder / "snapshot", check.bench_scenario(key))
            margin = check.SHADOW_ENVELOPE[spec["envelope"]]
            code = quiet_main(["render", str(folder / "snapshot"), "--out", str(folder / "reference"), "--margin", str(margin),
                               *spec.get("reference_args", []), "--controls", *spec["controls"]])
            assert code == 0, key
            # The engine of the red state shadows opaquely (transmittance-ignored): the pending CSM keys; the keys that landed take the
            # declared model as their engine (an engine implementing the declaration), which every gate must pass.
            today = "transmittance-ignored" if key in PENDING else None
            source = folder / "reference" / (f"control-{today}.pfm" if today else "model.pfm")
            write_pfm(folder / "engine.pfm", read_pfm(source))
            cls.entries[key] = check.evaluate_shadow_scenario(np, key, folder, "green")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pending_keys_fail_with_the_ignored_control_and_landed_keys_pass_with_the_model(self):
        for key in GATED:
            e = self.entries[key]
            self.assertEqual(e["gates_pass"], key not in PENDING, key)
            self.assertTrue(e["expectation_met"], key)
            if key in PENDING:
                self.assertTrue(e["pending_matches_met"], key)
                self.assertEqual(e["pending"], check.SCENARIOS[key]["pending"])
            else:
                self.assertIsNone(e["pending"], key)
        self.assertEqual(check.verdict([self.entries[k] for k in KEYS], "green"), ("GREEN", check.EXIT_OK))

    def test_every_control_is_discriminating_and_the_declared_model_is_feasible(self):
        for key in GATED:
            e = self.entries[key]
            for name, verdict in e["controls"].items():
                self.assertTrue(verdict["fails_gate"], (key, name, verdict["gate_result"]))
            self.assertTrue(e["model_feasible"], (key, e["model_gate"]))
            self.assertTrue(e["discriminating"], key)

    def test_the_curved_key_gates_only_the_qualitative_properties(self):
        e = self.entries[CURVED]
        self.assertTrue(e["gates_pass"])            # an engine implementing the declared model passes the qualitative gate
        self.assertTrue(e["expectation_met"])
        self.assertTrue(e["model_feasible"])        # the declared model is lighter than opaque, darker than unshadowed, tinted
        self.assertTrue(e["discriminating"], e["controls"])
        self.assertFalse(e["result"]["numbers_gated"])
        self.assertEqual(len(e["result"]["objects"]), 2)
        folder = self.folders[CURVED]
        masks = dict(np.load(folder / "reference" / "masks.npz"))
        report = json.loads((folder / "reference" / "reference.json").read_text())
        model = read_pfm(folder / "reference" / "model.pfm")
        result = check.evaluate(np, "transmissive-shadow-curved", model, model, masks, report["objects"])
        clear, tinted = result["objects"]
        self.assertTrue(result["pass"])
        self.assertTrue(0.001 < clear["transmittance"] < 0.995 and 0.001 < tinted["transmittance"] < 0.995)
        self.assertLess(abs(math.log(clear["chroma_r_over_b"])), 0.02)
        self.assertLess(tinted["chroma_r_over_b"], 0.9)                       # attenuation (0.5, 0.8, 1.0): red absorbed more than blue
        self.assertAlmostEqual(tinted["expected_chroma_r_over_b"], 0.5)
        # a grey shadow of the tinted caster (chroma ignored) and an unshadowed one both fail
        grey = model.copy()
        for o in report["objects"]:
            pick = masks["roi"] & (masks["sphere"] == o["instance"])
            grey[pick] = grey[pick].mean() * np.ones(3)
        self.assertFalse(check.evaluate(np, "transmissive-shadow-curved", grey, model, masks, report["objects"])["pass"])
        lit = read_pfm(folder / "reference" / "lit.pfm")
        self.assertFalse(check.evaluate(np, "transmissive-shadow-curved", lit, model, masks, report["objects"])["pass"])
        self.assertFalse(check.evaluate(np, "transmissive-shadow-curved", model * 0.0, model, masks, report["objects"])["pass"])

    def test_the_lit_reference_region_excludes_shadows_and_envelope_and_sits_next_to_them(self):
        key = "transmissive-shadow-pane-sun@csm"
        masks = dict(np.load(self.folders[key] / "reference" / "masks.npz"))
        lit, roi = masks["lit"], masks["roi"]
        self.assertGreater(int(lit.sum()), 1000)
        self.assertFalse((lit & roi).any())
        ys, xs = np.nonzero(roi)
        ly, lx = np.nonzero(lit)
        self.assertGreater(ly.min(), ys.min() - 200)   # a band next to the shadows, not the whole floor
        self.assertLess(lx.max(), xs.max() + 400)

    def test_csm_ratio_gate_ignores_the_lit_floor_bias_but_not_a_wrong_transmittance(self):
        key = "transmissive-shadow-pane-sun@csm"
        folder = self.folders[key]
        reference = read_pfm(folder / "reference" / "reference.pfm")
        masks = dict(np.load(folder / "reference" / "masks.npz"))
        objects = json.loads((folder / "reference" / "reference.json").read_text())["objects"]
        # a CSM-like bias: the whole image 5.2 % too dark (lit floor and shadow alike): the ratio is untouched
        biased = reference * 0.948
        ratio = check.evaluate(np, "transmissive-shadow-ratio", biased, reference, masks, objects)
        self.assertTrue(ratio["pass"], ratio["objects"])
        self.assertAlmostEqual(ratio["lit_floor_diagnostic"]["lit_level_ratio"], 0.948, places=3)
        self.assertFalse(check.evaluate(np, "transmissive-shadow", biased, reference, masks, objects)["pass"])  # the absolute gate sees the bias
        # a lit floor 5 % dark with the right shadows relative to it passes; a shadow 8 % too light relative to the lit floor fails
        wrong = reference.copy()
        pick = masks["roi"] & (masks["sphere"] == objects[0]["instance"])
        wrong[pick] *= 1.08
        self.assertFalse(check.evaluate(np, "transmissive-shadow-ratio", wrong, reference, masks, objects)["pass"])
        # the opaque engine of today: T = 0 against the truth's T fails; against the ignored twin it passes with the truth's floor
        control = read_pfm(folder / "reference" / "control-transmittance-ignored.pfm")
        self.assertFalse(check.evaluate(np, "transmissive-shadow-ratio", control, reference, masks, objects)["pass"])
        self.assertTrue(check.evaluate_shadow_ratio(np, control, control, masks, objects, floor_from=reference)["pass"])

    def test_the_declared_model_engine_would_be_green(self):
        # an engine that renders the declared straight-line model passes every gated gate (it is not pending then)
        for key in GATED:
            folder = self.folders[key]
            reference = read_pfm(folder / "reference" / "reference.pfm")
            model = read_pfm(folder / "reference" / "model.pfm")
            masks = dict(np.load(folder / "reference" / "masks.npz"))
            objects = json.loads((folder / "reference" / "reference.json").read_text())["objects"]
            result = check.evaluate(np, check.SCENARIOS[key]["kind"], model, reference, masks, objects)
            self.assertTrue(result["pass"], (key, result["worst_mean"], result["worst_p95"]))

    def test_the_floor_comes_from_the_truth_not_from_the_candidate(self):
        # the all-black twin has floor 0 of its own: the opaque engine must still be judged against the truth's floor
        key = "transmissive-shadow-pane-sun@rt"
        folder = self.folders[key]
        masks = dict(np.load(folder / "reference" / "masks.npz"))
        objects = json.loads((folder / "reference" / "reference.json").read_text())["objects"]
        black = read_pfm(folder / "reference" / "control-transmittance-ignored.pfm")
        reference = read_pfm(folder / "reference" / "reference.pfm")
        twin = check.evaluate_transmission(np, black, black, masks, objects, "transmissive_shadow", "transmissive-shadow", reference)
        self.assertTrue(twin["pass"])
        self.assertGreater(twin["floor"], 0.0)
        noisy = black.copy()
        noisy[masks["roi"]] += 0.5 * twin["floor"]  # half the floor everywhere: 50 % of the floor per pixel
        self.assertFalse(check.evaluate_transmission(np, noisy, black, masks, objects, "transmissive_shadow", "transmissive-shadow",
                                                     reference)["pass"])

    def test_noise_and_settings_are_recorded(self):
        for key in TRUTH_KEYS:
            report = json.loads((self.folders[key] / "reference" / "reference.json").read_text())
            self.assertEqual(report["kind"], "mitsuba")
            self.assertEqual(report["samples"], sr.SAMPLES)
            self.assertLess(report["reference"]["noise"]["mean_relative_standard_error"], 0.01, key)
            self.assertEqual(report["margin_m"], check.SHADOW_ENVELOPE[check.SCENARIOS[key]["envelope"]])

    def test_cached_reference_is_reused_only_for_the_same_snapshot_tool_samples_and_margin(self):
        key = "transmissive-shadow-pane-sun@rt"
        folder, controls = self.folders[key], check.SCENARIOS[key]["controls"]
        usable = lambda **kw: check.cached_transmission_reference_usable(  # noqa: E731
            folder / "reference", folder / "snapshot", kw.get("controls", controls), kw.get("samples", sr.SAMPLES),
            "f15_shadow_reference.py", "samples", kw.get("margin", 0.06))
        self.assertTrue(usable())
        self.assertFalse(usable(samples=sr.SAMPLES * 2))
        self.assertFalse(usable(margin=0.08))
        self.assertFalse(usable(controls=controls + ["beer-lambert-ignored"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
