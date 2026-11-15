#!/usr/bin/env python3
"""F15 ticket 22 (C22-A1): unit test of the ideal-mirror continuation of tools/f13_reference.py against an analytic two-mirror case.

Scene (written here as a snapshot, no engine involved): two facing mirror walls 1.0 apart and 8.0 long (metal, albedo 0.8, roughness at the
reference clamp 0.04), constant sky radiance 0.5, no lights.  The expected radiance of a pixel comes from tools/f15_reflection_bounce_model.py
(analytic ray casting, Schlick F0 0.8): environment times the product of the Fresnel factors of the k mirror events of the camera ray.
The reference with --mirror-bounces 3 must match it per class k = 1..4 (mean relative error <= 5 %, the fixture gate); the negative control
(`mirror_bounces=0`, the declared F13 model without specular continuation) must FAIL for every class >= 2 (those pixels are dark there).

Needs Mitsuba: run it with
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f13_reference_mirror.py
Without Mitsuba the Mitsuba cases are skipped; the pure-Python cases (argument plumbing) still run.
"""
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f15_reflection_bounce_model as model  # noqa: E402

try:
    import mitsuba  # noqa: F401
    import numpy  # noqa: F401
    HAVE_MITSUBA = True
except ImportError:
    HAVE_MITSUBA = False

WIDTH, HEIGHT = 160, 90
# Two facing walls 0.6 apart, 6 long: classes 1..3 within a camera view that is not so grazing that the reference's own near-delta camera lobe
# (alpha 0.002 at N.V ~ 0.01) throws rare very large weights; the engine fixture (tools/f15_reflection_bounce_check.py) uses the 1.0 x 8.0 corridor.
CORRIDOR = {"quads": [((-0.3, -0.5, 0.0), (0.0, 0.0, -6.0), (0.0, 1.0, 0.0)), ((0.3, -0.5, -6.0), (0.0, 0.0, 6.0), (0.0, 1.0, 0.0))], "spheres": []}
CLASSES = (1, 2, 3)
GATE = 0.05
FLOOR = 0.05
MIN_PIXELS = 30


def write_ply(path, corners, normal):
    """Binary little-endian PLY quad (two triangles) with normals and uvs, the layout of the engine's exports."""
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\n"
              "element face 2\nproperty list uchar uint vertex_indices\nend_header\n")
    body = b"".join(struct.pack("<8f", *c, *normal, u, v) for c, (u, v) in zip(corners, ((0, 0), (1, 0), (1, 1), (0, 1))))
    body += b"".join(struct.pack("<BIII", 3, *tri) for tri in ((0, 1, 2), (0, 2, 3)))
    path.write_bytes(header.encode() + body)


def write_snapshot(folder, description):
    """scene.json + PLY quads of a model scene (quads only), camera framed like the corpus rig (yaw 0, pitch 0)."""
    meshes, instances = [], []
    for k, (corner, u, v) in enumerate(description["quads"]):
        corners = [corner, tuple(corner[i] + u[i] for i in range(3)), tuple(corner[i] + u[i] + v[i] for i in range(3)),
                   tuple(corner[i] + v[i] for i in range(3))]
        write_ply(folder / f"mesh_{k}.ply", corners, model.normalize(model.cross(u, v)))
        meshes.append(f"mesh_{k}.ply")
        instances.append({"slot": k, "generation": 1, "flags": 23, "mesh": k, "material": 0,
                          "world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]})
    low, high = model.scene_bounds(description)
    origin, _, _ = model.frame(low, high, WIDTH / HEIGHT)
    data = {"schema": 1, "state": "UNVERIFIED_REFERENCE_EXPORT", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
            "camera": {"position": list(origin), "direction": [0, 0, -1], "up": [0, 1, 0], "fov_y_radians": math.radians(60.0), "near": 0.05,
                       "far": 0, "width": WIDTH, "height": HEIGHT, "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"},
            "sky": [model.ENVIRONMENT] * 3, "sun_angular_radius": 0.0, "textures": [],
            "materials": [{"base": [0.8, 0.8, 0.8, 1], "emissive": [0, 0, 0], "metallic": 1, "roughness": 0.04, "normal_scale": 1,
                           "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0, "textures": [0xFFFFFFFF] * 5}],
            "meshes": meshes, "instances": instances, "lights": [], "sampled_lights": []}
    (folder / "scene.json").write_text(json.dumps(data))
    return data


def class_errors(reference_image, description, spp):
    """Per class k: (stable pixels, mean relative error of reference/environment vs model/environment), model classes at the declared bound."""
    low, high = model.scene_bounds(description)
    rad, cls, mask = model.render(model.build_scene(description), WIDTH, HEIGHT, low, high)
    stable = model.stable_mask(cls, mask, 2)
    result = {}
    for k in CLASSES:
        pixels = [(y, x) for y in range(HEIGHT) for x in range(WIDTH) if stable[y][x] and cls[y][x] == k]
        expected = [rad[y][x] / model.ENVIRONMENT for y, x in pixels]
        floor = FLOOR * (sum(expected) / max(len(expected), 1))
        got = [float(reference_image[y, x, :].mean()) / spp / model.ENVIRONMENT for y, x in pixels]
        error = sum(abs(g - e) / max(e, floor) for g, e in zip(got, expected)) / max(len(pixels), 1)
        result[k] = (len(pixels), error)
    return result


@unittest.skipUnless(HAVE_MITSUBA, "Mitsuba is not installed: run through uv with --with mitsuba==3.9.1 --with numpy")
class MirrorContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import f13_reference as reference
        cls.reference = reference
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.folder.name)
        cls.description = CORRIDOR
        cls.data = write_snapshot(cls.root, cls.description)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def render(self, bounces, spp=8):
        context = self.reference.Context(self.root, self.data, 2, "regularized", mirror_bounces=bounces)
        return context.render_pass(1, spp), spp

    def test_three_bounces_match_the_analytic_two_mirror_model(self):
        image, spp = self.render(3)
        errors = class_errors(image, self.description, spp)
        for k, (count, error) in errors.items():
            self.assertGreaterEqual(count, MIN_PIXELS, errors)
            self.assertLessEqual(error, GATE, f"class {k}: {errors}")

    def test_negative_control_without_continuation_fails_every_class_from_two(self):
        image, spp = self.render(0)
        errors = class_errors(image, self.description, spp)
        self.assertLessEqual(errors[1][1], GATE, errors)  # the camera-visible mirror is unaffected
        for k in (2, 3):
            self.assertGreater(errors[k][1], 0.5, f"class {k}: {errors}")  # dark: the reference without continuation cannot show a mirror in a mirror

    def test_bound_is_respected(self):
        image, spp = self.render(1)
        errors = class_errors(image, self.description, spp)
        self.assertLessEqual(errors[2][1], GATE, errors)  # one extra hit reaches class 2
        self.assertGreater(errors[3][1], 0.5, errors)  # but not class 3

    def test_argument_is_validated(self):
        with self.assertRaises(ValueError):
            self.reference.Context(self.root, self.data, 2, "regularized", mirror_bounces=9)


class Plumbing(unittest.TestCase):
    def test_reference_clip_cli_accepts_the_option(self):
        import f13_reference_clip as clip
        text = pathlib.Path(clip.__file__).read_text()
        self.assertIn("--mirror-bounces", text)
        self.assertIn("mirror_bounces=getattr(args, \"mirror_bounces\", 0)", text)


if __name__ == "__main__":
    unittest.main()
