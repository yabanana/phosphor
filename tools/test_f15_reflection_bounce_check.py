#!/usr/bin/env python3
"""Unit tests of tools/f15_reflection_bounce_check.py and its CPU model (no GPU): fixture population, the glTF writer, the PFM reader, and the
gates against the model itself (positive), against the single-bounce model (the negative control) and against perturbed images."""
import json
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_reflection_bounce_check as check  # noqa: E402
import f15_reflection_bounce_model as model  # noqa: E402

REFERENCES = {}


def reference(name):
    if name not in REFERENCES:
        REFERENCES[name] = check.reference(name)
    return REFERENCES[name]


def single_bounce_rows(name):
    """What the F13 single-bounce engine shows: the model with 0 extra bounces (a mirror hit that is not followed returns no light)."""
    description = model.SCENES[name]
    low, high = model.scene_bounds(description)
    rad, _, _ = model.render(model.build_scene(description), check.WIDTH, check.HEIGHT, low, high, bounces=0, f0=model.SCENE_F0.get(name, model.MIRROR_F0))
    return rad


class ModelFixtures(unittest.TestCase):
    def test_every_expected_class_is_populated(self):
        for name, classes in check.EXPECTED_CLASSES.items():
            _, cls, stable, _ = reference(name)
            counts = model.class_counts(cls, stable)
            for k in classes:
                self.assertGreaterEqual(counts.get(k, 0), check.MIN_PIXELS, f"{name} class {k}: {counts}")

    def test_class_radiance_is_environment_times_fresnel_products(self):
        rad, cls, stable, _ = reference("corridor")
        by_class = {}
        for y, row in enumerate(cls):
            for x, k in enumerate(row):
                if stable[y][x]:
                    by_class.setdefault(k, []).append(rad[y][x])
        self.assertAlmostEqual(min(by_class[0]), 0.5)
        for k in (1, 2, 3, 4):  # each mirror event multiplies by a Fresnel factor in [F0, 1]
            self.assertTrue(0.5 * 0.8 ** k - 1e-9 <= min(by_class[k]) and max(by_class[k]) <= 0.5 + 1e-9, k)
        self.assertGreater(sum(by_class[1]) / len(by_class[1]), sum(by_class[4]) / len(by_class[4]))

    def test_model_camera_matches_the_rig_formula(self):
        origin, tan_half, distance = model.frame((-1, -0.5, -4), (1, 0.5, 0), 16 / 9)
        self.assertAlmostEqual(origin[2] - -2.0, distance)
        self.assertAlmostEqual(tan_half, 0.5773502691896257)


class Gates(unittest.TestCase):
    def test_the_model_itself_passes_every_gate(self):
        for name, classes in check.EXPECTED_CLASSES.items():
            rad, cls, stable, _ = reference(name)
            result = check.evaluate(rad, cls, stable, rad, classes)
            ok, reasons = check.verdict(result, classes)
            self.assertTrue(ok, (name, reasons))
            for k in classes:
                self.assertAlmostEqual(result["classes"][k]["mean_relative_error"], 0.0, places=9)

    def test_single_bounce_image_fails_the_control_semantics(self):
        for name, classes in check.EXPECTED_CLASSES.items():
            rad, cls, stable, _ = reference(name)
            single = single_bounce_rows(name)
            result = check.evaluate(rad, cls, stable, single, classes)
            ok, reasons = check.verdict(result, classes)
            self.assertFalse(ok, name)  # the normal gates fail (the engine would be dark in mirrors)
            for k in classes:
                if k >= 2:
                    self.assertGreater(result["classes"][k]["mean_relative_error"], check.GATE, (name, k))
                else:
                    self.assertLessEqual(result["classes"][k]["mean_relative_error"], check.GATE, (name, k))
            ok, reasons = check.verdict(result, classes, control=True)
            self.assertTrue(ok, (name, reasons))  # and it fails exactly where the control must

    def test_a_ten_percent_error_in_one_class_is_caught(self):
        rad, cls, stable, _ = reference("corridor")
        scaled = [[v * (1.10 if cls[y][x] == 3 else 1.0) for x, v in enumerate(row)] for y, row in enumerate(rad)]
        result = check.evaluate(rad, cls, stable, scaled, (1, 2, 3, 4))
        ok, reasons = check.verdict(result, (1, 2, 3, 4))
        self.assertFalse(ok)
        self.assertTrue(any("class 3" in r for r in reasons), reasons)
        self.assertLessEqual(result["classes"][2]["mean_relative_error"], 0.001)

    def test_a_three_percent_error_passes_and_a_dark_background_fails(self):
        rad, cls, stable, _ = reference("ball")
        scaled = [[v * (1.03 if cls[y][x] == 2 else 1.0) for x, v in enumerate(row)] for y, row in enumerate(rad)]
        ok, reasons = check.verdict(check.evaluate(rad, cls, stable, scaled, (1, 2, 3)), (1, 2, 3))
        self.assertTrue(ok, reasons)
        dark = [[v * 0.8 for v in row] for row in rad]  # exposure error: the background is off, classes are normalised by it
        ok, reasons = check.verdict(check.evaluate(rad, cls, stable, dark, (1, 2, 3)), (1, 2, 3))
        self.assertFalse(ok)
        self.assertTrue(any("background" in r for r in reasons), reasons)

    def test_too_few_pixels_fail_the_fixture_gate(self):
        result = {"background": {"mean": 0.5, "relative_error": 0.0, "pixels": 1000},
                  "classes": {1: {"pixels": check.MIN_PIXELS - 1, "mean_relative_error": 0.0}}}
        ok, reasons = check.verdict(result, (1,))
        self.assertFalse(ok)


class Writers(unittest.TestCase):
    def test_gltf_is_valid_and_aligned(self):
        for name in check.EXPECTED_CLASSES:
            with tempfile.TemporaryDirectory() as folder:
                path = check.write_scene(pathlib.Path(folder), model.SCENES[name])
                document = json.loads(path.read_text())
                binary = (path.parent / "scene.bin").read_bytes()
                self.assertEqual(len(binary), document["buffers"][0]["byteLength"])
                for view in document["bufferViews"]:
                    self.assertEqual(view["byteOffset"] % 4, 0)
                    self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(binary))
                self.assertEqual(len(document["meshes"]), len(model.SCENES[name]["quads"]) + len(model.SCENES[name]["spheres"]))
                material = document["materials"][0]
                self.assertEqual(material["pbrMetallicRoughness"]["metallicFactor"], 1)
                self.assertEqual(material["pbrMetallicRoughness"]["roughnessFactor"], 0)
                self.assertNotIn("extensions", material)

    def test_quad_and_sphere_faces_point_outward(self):
        for name in check.EXPECTED_CLASSES:
            for quad in model.SCENES[name]["quads"]:
                positions, normals, _, indices = check.quad_mesh(*quad)
                a, b, c = (positions[i] for i in indices[:3])
                face = model.cross(model.sub(b, a), model.sub(c, a))
                self.assertGreater(model.dot(face, normals[0]), 0)
        positions, normals, _, indices = check.sphere_mesh((0.0, 0.0, 0.0), 1.0)
        for t in range(0, len(indices), 99):
            a, b, c = (positions[i] for i in indices[t:t + 3])
            face = model.cross(model.sub(b, a), model.sub(c, a))
            if model.dot(face, face) > 1e-12:
                self.assertGreater(model.dot(face, a), 0)

    def test_pfm_reader_flips_rows_and_weights_luminance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "t.pfm"
            bottom, top = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)  # file order: bottom row first
            path.write_bytes(b"PF\n1 2\n-1.0\n" + struct.pack("<6f", *bottom, *top))
            width, height, rows = check.read_pfm_luminance(path)
            self.assertEqual((width, height), (1, 2))
            self.assertAlmostEqual(rows[0][0], 0.7152, places=5)
            self.assertAlmostEqual(rows[1][0], 0.2126, places=5)


if __name__ == "__main__":
    unittest.main()
