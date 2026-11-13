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
import struct
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
    "base-grid-env-resolve": dict(roughness=ROUGHNESS, metallic=METALLIC, base=COPPER, sun=0.0, env=0.5),
    "furnace-base": dict(roughness=ROUGHNESS, metallic=METALLIC, base=[1, 1, 1], sun=0.0, env=1.0),
    "furnace-resolve": dict(roughness=ROUGHNESS, metallic=METALLIC, base=[1, 1, 1], sun=0.0, env=1.0),
    "roughness-extremes": dict(roughness=[0.0, 1.0], metallic=[0.0, 0.5, 1.0], base=COPPER, sun=3.0, env=0.3),
    "roughness-extremes-sun": dict(roughness=[0.0, 1.0], metallic=[0.0, 0.5, 1.0], base=COPPER, sun=3.0, env=0.0),
}
# F15.1: the ior x specular grids (mirrors MaterialValidation: columns = ior, rows = (factor, colour, metallic)).
SPEC_IOR = [0.0, 1.0, 1.33, 1.5, 2.4]
SPEC_ROWS = [(1.0, [1, 1, 1], 0.0), (0.5, [1, 1, 1], 0.0), (1.0, [2, 2, 2], 0.0), (0.5, [2, 1, 0.5], 0.0),
             (0.5, [2, 2, 2], 1.0)]
for _name, _sun, _env, _base in (("specular-grid-sun", 3.0, 0.0, COPPER), ("specular-grid-env-resolve", 0.0, 0.5, COPPER),
                                 ("specular-grid-env", 0.0, 0.5, COPPER), ("furnace-specular", 0.0, 1.0, [1, 1, 1])):
    SPECS[_name] = dict(roughness=[0.35] * 5, metallic=[row[2] for row in SPEC_ROWS], base=_base, sun=_sun, env=_env,
                        ior=SPEC_IOR, specular=[(row[0], row[1]) for row in SPEC_ROWS])
# F15.3 (ticket 09): the clearcoat grids (mirrors MaterialValidation: rows = coat factor, then coat 1 over ior 2.4, then a
# metal under a full coat; columns = coat roughness over a rough dielectric base).
COAT = dict(factor=[0.0, 0.25, 0.5, 1.0, 1.0, 1.0], roughness=[0.0, 0.05, 0.15, 0.4, 1.0], ior=[0, 0, 0, 0, 2.4, 0])
POINT_LIGHT = {"type": 1, "position": [-5.85, 7.45, -4.68], "direction": [0, 0, -1], "color": [1, 1, 1],
               "intensity": 250.0, "range": 100.0, "inner": 0, "outer": 0}
for _name, _sun, _env, _base, _point in (("clearcoat-grid-sun", 3.0, 0.0, COPPER, False),
                                         ("clearcoat-grid-env-resolve", 0.0, 0.5, COPPER, False),
                                         ("clearcoat-grid-env", 0.0, 0.5, COPPER, False),
                                         ("clearcoat-grid-point", 0.0, 0.0, COPPER, True),
                                         ("furnace-clearcoat", 0.0, 1.0, [1, 1, 1], False)):
    SPECS[_name] = dict(roughness=[0.6] * 5, metallic=[0.0] * 5 + [1.0], base=_base, sun=_sun, env=_env, coat=COAT,
                        point=_point)


# F15.3 (ticket 10): the sheen grids (mirrors MaterialValidation: rows = sheen colour, columns = sheen roughness, over a
# dark rough dielectric base: grey 0.1, roughness 0.7).
SHEEN = dict(color=[[0, 0, 0], [1, 1, 1], [1, 0.5, 0.25], [0.25, 0.5, 1]], roughness=[0.1, 0.3, 0.5, 0.75, 1.0])
SHEEN_POINT_LIGHT = {**POINT_LIGHT, "position": [-5.85, 6.45, -4.68]}  # grid centre y = 1.5
for _name, _sun, _env, _base, _point in (("sheen-grid-sun", 3.0, 0.0, [0.1] * 3, False),
                                         ("sheen-grid-env-resolve", 0.0, 0.5, [0.1] * 3, False),
                                         ("sheen-grid-env", 0.0, 0.5, [0.1] * 3, False),
                                         ("sheen-grid-point", 0.0, 0.0, [0.1] * 3, True),
                                         ("furnace-sheen", 0.0, 1.0, [1, 1, 1], False)):
    SPECS[_name] = dict(roughness=[0.7] * 5, metallic=[0.0] * 4, base=_base, sun=_sun, env=_env, sheen=SHEEN,
                        point=SHEEN_POINT_LIGHT if _point else False)


# F15.3 (ticket 11): the anisotropy grids (mirrors MaterialValidation: rows = strength over a dielectric then a metal,
# columns = rotation, combed spheres of handedness -1) and the mirrored carrier fixture (columns = plain / mirrored over
# handedness -1 / +1 meshes, every rotation pi/4, rows = dielectric, metal at strength 1).
ANISO = dict(strength=[0.0, 0.5, 1.0, 0.0, 0.5, 1.0], rotation=[0.0, math.pi / 4, math.pi / 2, -math.pi / 4],
             mirrored=[False] * 4, handedness=[-1.0] * 4)
ANISO_POINT_LIGHT = POINT_LIGHT  # grid centre y = 2.5, like the coat grid
for _name, _sun, _env, _base, _point in (("aniso-grid-sun", 3.0, 0.0, COPPER, False),
                                         ("aniso-grid-env-resolve", 0.0, 0.5, COPPER, False),
                                         ("aniso-grid-env", 0.0, 0.5, COPPER, False),
                                         ("aniso-grid-point", 0.0, 0.0, COPPER, True),
                                         ("furnace-aniso", 0.0, 1.0, [1, 1, 1], False)):
    SPECS[_name] = dict(roughness=[0.3] * 4, metallic=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], base=_base, sun=_sun, env=_env,
                        aniso=ANISO, point=ANISO_POINT_LIGHT if _point else False)
SPECS["aniso-mirrored-sun"] = dict(
    roughness=[0.3] * 4, metallic=[0.0, 1.0], base=COPPER, sun=3.0, env=0.0, point=False,
    aniso=dict(strength=[1.0, 1.0], rotation=[math.pi / 4] * 4, mirrored=[False, True, False, True],
               handedness=[-1.0, -1.0, 1.0, 1.0]))


# F15.3 (ticket 12): the iridescence grids (mirrors MaterialValidation: rows = (factor, film ior) over a dielectric then a metal,
# columns = thickness in nm, base roughness 0.3).
IRID = dict(factor=[0.5, 1.0, 1.0, 0.5, 1.0, 1.0], ior=[1.3, 1.3, 1.8, 1.3, 1.3, 1.8],
            thickness=[0.0, 200.0, 400.0, 600.0, 800.0, 1200.0])
for _name, _sun, _env, _base, _point in (("iridescence-grid-sun", 3.0, 0.0, COPPER, False),
                                         ("iridescence-grid-env-resolve", 0.0, 0.5, COPPER, False),
                                         ("iridescence-grid-env", 0.0, 0.5, COPPER, False),
                                         ("iridescence-grid-point", 0.0, 0.0, COPPER, True),
                                         ("furnace-iridescence", 0.0, 1.0, [1, 1, 1], False)):
    SPECS[_name] = dict(roughness=[0.3] * 6, metallic=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], base=_base, sun=_sun, env=_env,
                        irid=IRID, point=POINT_LIGHT if _point else False)


# F15.3 (ticket 16): the thin subsurface scenarios (mirrors MaterialValidation: rows = weight, columns = transmission colour, blue-grey
# base of roughness 0.5; spheres for the sun / environment scenarios, single-sided quads and one back point light for the point one).
THIN_COLORS = [[1.0, 1.0, 1.0], [1.0, 0.55, 0.25], [0.1, 0.7, 0.6]]
THIN_BASE = [0.3, 0.5, 0.8]
THIN_TOWARD = np.array([-0.55, 0.4, -0.73]) / np.linalg.norm([-0.55, 0.4, -0.73])
THIN_POINT_LIGHT = {**POINT_LIGHT, "position": [float(c) for c in np.array([0.0, 1.0, 0.0]) + 9.0 * THIN_TOWARD]}
THIN_WEIGHTS = [0.0, 0.25, 0.5, 1.0]
for _name, _sun, _env, _base, _weights, _colors, _rough, _point in (
        ("subsurface-thin-sun", 3.0, 0.0, THIN_BASE, THIN_WEIGHTS, THIN_COLORS, [0.5] * 3, False),
        ("subsurface-thin-env-resolve", 0.0, 0.5, THIN_BASE, THIN_WEIGHTS, THIN_COLORS, [0.5] * 3, False),
        ("subsurface-thin-env", 0.0, 0.5, THIN_BASE, THIN_WEIGHTS, THIN_COLORS, [0.5] * 3, False),
        ("subsurface-thin-point", 0.0, 0.0, THIN_BASE, [0.25, 0.5, 1.0], THIN_COLORS, [0.5] * 3, True),
        ("furnace-thin-subsurface", 0.0, 1.0, [1, 1, 1], THIN_WEIGHTS, [[1.0, 1.0, 1.0]] * 3, [0.2, 0.5, 0.9], False)):
    SPECS[_name] = dict(roughness=_rough, metallic=[0.0] * len(_weights), base=_base, sun=_sun, env=_env,
                        thin=dict(weight=_weights, color=_colors, quad=_point), toward=THIN_TOWARD,
                        point=THIN_POINT_LIGHT if _point else False)


def write_quad_ply(path, half=0.4):
    """The thin point-light carrier: a quad in the z = 0 plane facing +z (what the bench's rotated generatePlane exports)."""
    verts = [(-half, -half, 0, 0, 0, 1, 0, 0), (half, -half, 0, 0, 0, 1, 1, 0), (half, half, 0, 0, 0, 1, 1, 1),
             (-half, half, 0, 0, 0, 1, 0, 1)]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\n"
                 "element face 2\nproperty list uchar uint vertex_indices\nend_header\n").encode())
        for v in verts:
            f.write(struct.pack("<8f", *v))
        for tri in ((0, 1, 2), (0, 2, 3)):
            f.write(struct.pack("<B3I", 3, *tri))


def comb_tangent(n, comb=(1.0, 0.0, 0.0)):
    """The tangent of ProceduralMeshes::generateCombedSphere at a vertex of normal n: `comb` projected onto the tangent
    plane; where nothing is left, a unit perpendicular."""
    d = sum(a * b for a, b in zip(comb, n))
    t = [comb[i] - d * n[i] for i in range(3)]
    length = math.sqrt(sum(c * c for c in t))
    if length > 1e-3:
        return [c / length for c in t]
    helper = (0.0, 1.0, 0.0)  # |comb.y| < 0.9
    c = [n[1] * helper[2] - n[2] * helper[1], n[2] * helper[0] - n[0] * helper[2], n[0] * helper[1] - n[1] * helper[0]]
    length = math.sqrt(sum(x * x for x in c))
    return [x / length for x in c]


def write_sphere_ply(path, radius=0.4, slices=64, stacks=32, handedness=None):
    """The material-validation sphere; with `handedness` (+1 / -1) the combed tangent attribute of the snapshot (mesh
    version 2: properties tangent_x/y/z and handedness_x after s, t)."""
    verts, faces = [], []
    for st in range(stacks + 1):
        theta = math.pi * st / stacks
        for sl in range(slices + 1):
            phi = 2 * math.pi * sl / slices
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            extra = (*comb_tangent(n), handedness) if handedness is not None else ()
            verts.append((*(c * radius for c in n), *n, sl / slices, st / stacks, *extra))
    for st in range(stacks):
        for sl in range(slices):
            a = st * (slices + 1) + sl
            b = a + slices + 1
            faces += [(a, b, a + 1), (a + 1, b, b + 1)]
    tangent = ("property float tangent_x\nproperty float tangent_y\nproperty float tangent_z\n"
               "property float handedness_x\n") if handedness is not None else ""
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\n"
                 "property float t\n%selement face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                 % (len(verts), tangent, len(faces))).encode())
        for v in verts:
            f.write(struct.pack("<%df" % len(v), *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


def make_bench_snapshot(folder, name, width, height):
    spec = SPECS[name]
    folder = pathlib.Path(folder)
    (folder / "snapshot").mkdir(parents=True)
    handed = spec["aniso"]["handedness"] if "aniso" in spec else None
    if spec.get("thin", {}).get("quad"):
        write_quad_ply(folder / "snapshot" / "mesh_0.ply")
    else:
        write_sphere_ply(folder / "snapshot" / "mesh_0.ply", handedness=-1.0 if handed else None)
    meshes = ["mesh_0.ply"]
    if handed and 1.0 in handed:
        write_sphere_ply(folder / "snapshot" / "mesh_1.ply", handedness=1.0)
        meshes.append("mesh_1.ply")
    cols, rows = len(spec["roughness"]), len(spec["metallic"])
    instances, materials, records = [], [], []
    for row in range(rows):
        for col in range(cols):
            x, y = -0.5 * (cols - 1) + col, float(row)
            world = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, 0, 1]
            materials.append({"base": [*spec["base"], 1], "emissive": [0, 0, 0], "metallic": spec["metallic"][row],
                              "roughness": spec["roughness"][col], "normal_scale": 1, "occlusion_strength": 1,
                              "alpha_cutoff": 0, "flags": 0, "textures": [ref.INVALID] * 5,
                              "feature_bits": 0, "extension_index": -1})
            if "ior" in spec:
                ior, (factor, color) = spec["ior"][col], spec["specular"][row]
                if not (ior == 1.5 and factor == 1.0 and color == [1, 1, 1]):  # the neutral block is not a feature
                    words = [int.from_bytes(struct.pack("<f", v), "little") for v in (ior, factor, *color)]
                    materials[-1].update(feature_bits=1, flags=256, extension_index=len(records))
                    records += [words + [ref.INVALID, ref.INVALID] + [0] * 13, [0] * 20, [0] * 20, [0] * 20]
            if "coat" in spec:
                factor, rough, ior = spec["coat"]["factor"][row], spec["coat"]["roughness"][col], spec["coat"]["ior"][row]
                bits = (2 if factor > 0 else 0) | (1 if ior > 0 else 0)
                if bits:
                    bit = lambda v: int.from_bytes(struct.pack("<f", v), "little")  # noqa: E731
                    words = ([bit(ior if ior > 0 else 1.5), bit(1.0), bit(1.0), bit(1.0), bit(1.0), ref.INVALID, ref.INVALID,
                              bit(factor), bit(rough), bit(1.0), ref.INVALID, ref.INVALID, ref.INVALID] + [0] * 7)
                    materials[-1].update(feature_bits=bits, flags=bits << 8, extension_index=len(records))
                    records += [words, [0] * 20, [0] * 20, [0] * 20]
            if "sheen" in spec:
                color, rough = spec["sheen"]["color"][row], spec["sheen"]["roughness"][col]
                if any(color):  # a zero sheen colour disables the layer: no feature bit, no extension
                    bit = lambda v: int.from_bytes(struct.pack("<f", v), "little")  # noqa: E731
                    words = [0] * 20
                    for k in (5, 6, 10, 11, 12, 17, 18):
                        words[k] = ref.INVALID
                    words[13:16] = [bit(c) for c in color]
                    words[16] = bit(rough)
                    materials[-1].update(feature_bits=4, flags=4 << 8, extension_index=len(records))
                    records += [words, [0] * 20, [0] * 20, [0] * 20]
            if "irid" in spec:  # every sphere carries the block (factor > 0 sets the bit, thickness 0 included)
                bit = lambda v: int.from_bytes(struct.pack("<f", v), "little")  # noqa: E731
                record0 = [0] * 20
                for k in (5, 6, 10, 11, 12, 17, 18):
                    record0[k] = ref.INVALID
                record1 = [0] * 20
                record1[2] = ref.INVALID
                record1[3:9] = [bit(spec["irid"]["factor"][row]), bit(spec["irid"]["ior"][row]), bit(100.0),
                                bit(spec["irid"]["thickness"][col]), ref.INVALID, ref.INVALID]
                materials[-1].update(feature_bits=16, flags=16 << 8, extension_index=len(records))
                records += [record0, record1, [0] * 20, [0] * 20]
            if "thin" in spec:  # weight 0 carries no feature (the bench adds no extension to that row)
                weight = spec["thin"]["weight"][row]
                if weight > 0:
                    bit = lambda v: int.from_bytes(struct.pack("<f", v), "little")  # noqa: E731
                    record0 = [0] * 20
                    for k in (5, 6, 10, 11, 12, 17, 18):
                        record0[k] = ref.INVALID
                    record2 = [0] * 20
                    record2[1] = bit(weight)
                    record2[2:5] = [bit(c) for c in spec["thin"]["color"][col]]
                    record2[5] = bit(1.0)
                    record2[6:9] = [bit(1.0), bit(0.5), bit(0.25)]
                    record2[10:13] = [ref.INVALID] * 3
                    materials[-1].update(feature_bits=128, flags=128 << 8, extension_index=len(records))
                    records += [record0, [0] * 20, record2, [0] * 20]
            mesh = 0
            if "aniso" in spec:
                a = spec["aniso"]
                strength, rotation = a["strength"][row], a["rotation"][col]
                if strength > 0:  # strength 0: the feature is inactive, no feature bit and no extension
                    bit = lambda v: int.from_bytes(struct.pack("<f", v), "little")  # noqa: E731
                    record0 = [0] * 20
                    for k in (5, 6, 10, 11, 12, 17, 18):
                        record0[k] = ref.INVALID
                    record1 = [0] * 20
                    record1[0], record1[1], record1[2] = bit(strength), bit(rotation), ref.INVALID
                    materials[-1].update(feature_bits=8, flags=8 << 8, extension_index=len(records))
                    records += [record0, record1, [0] * 20, [0] * 20]
                if a["mirrored"][col]:
                    world[0] = -1.0
                mesh = 1 if a["handedness"][col] > 0 else 0
            instances.append({"slot": len(instances), "generation": 1, "flags": 7 | (8 if world[0] < 0 else 0), "mesh": mesh,
                              "material": len(materials) - 1, "world": world})
    distance = 5.5 if cols > 2 else 4.0
    data = {"schema": 1, "state": "TEST", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
            "camera": {"position": [0, 0.5 * (rows - 1), distance], "direction": [0, 0, -1], "up": [0, 1, 0],
                       "fov_y_radians": math.radians(60), "near": 0.05, "far": 0, "width": width, "height": height,
                       "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"},
            "sky": [spec["env"]] * 3, "sun_angular_radius": 0.00465, "textures": [], "materials": materials,
            "meshes": meshes, "mesh_version": 2 if handed else 1, "instances": instances, "lights": [], "sampled_lights": [],
            "material_extensions": {"version": 1, "records_per_material": 4, "records": records}}
    if spec["sun"] > 0:
        data["lights"].append({"type": 0, "position": [0, 0, 0], "direction": list(-spec.get("toward", TOWARD_SUN)), "color": [1, 1, 1],
                               "intensity": spec["sun"], "range": 0, "inner": 0, "outer": 0})
    if spec.get("point"):
        data["lights"].append(POINT_LIGHT if spec["point"] is True else spec["point"])
    (folder / "snapshot" / "scene.json").write_text(json.dumps(data))
    return folder


class RunnerLogic(unittest.TestCase):
    def test_transport_furnace_gates_the_sphere_mean_and_the_resolve_furnace_every_pixel(self):
        # One sphere of 4 pixels: a single noisy pixel above 1.005, mean below it.
        candidate = np.array([[[1.02] * 3, [0.99] * 3], [[0.995] * 3, [0.99] * 3]], dtype=np.float32)
        masks = {"roi": np.ones((2, 2), dtype=bool), "sphere": np.full((2, 2), 7, dtype=np.int32)}
        spheres = [{"instance": 7, "metallic": 1.0, "roughness": 0.1}]
        reference = np.ones_like(candidate)
        transport = check.evaluate(np, "furnace-transport", candidate, reference, masks, spheres)
        resolve = check.evaluate(np, "furnace", candidate, reference, masks, spheres)
        self.assertTrue(transport["pass"])           # mean 0.99875 <= 1.005
        self.assertFalse(resolve["pass"])            # max pixel 1.02 > 1.005
        hot = candidate + 0.01                       # mean 1.00875 > 1.005: energy gain
        self.assertFalse(check.evaluate(np, "furnace-transport", hot, reference, masks, spheres)["pass"])

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
                self.assertEqual(capture[capture.index("--material-scenario") + 1], check.bench_scenario(name))
                self.assertEqual(capture[capture.index("--bench") + 1], "2")
                for flag in ("--capture-linear", "--export-reference", "--reflections", "--resolution"):
                    self.assertIn(flag, capture)
                self.assertIn(" ".join(capture), text)
                self.assertIn(check.reference_script(name), " ".join(commands["reference"]))
            for key in ("commit", "os", "sdk_version", "power", "tool_sha256", "seed", "thresholds"):
                self.assertIn(key, plan)
            self.assertIn("f15_common.py", plan["tool_sha256"])

    def test_plan_prints_the_specular_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = check.main(["--out", tmp, "--binary", "build/release/phosphor", "--only", "specular-grid-sun",
                                   "--only", "specular-grid-env-resolve"])
            self.assertEqual(code, 0)
            text = buffer.getvalue()
            self.assertIn("# specular-grid-sun:", text)
            self.assertIn("# specular-grid-env-resolve:", text)
            plan = json.loads((pathlib.Path(tmp) / "plan.json").read_text())
            sun = plan["commands"]["specular-grid-sun"]
            env = plan["commands"]["specular-grid-env-resolve"]
            self.assertEqual(sun["capture"][sun["capture"].index("--material-scenario") + 1], "specular-grid-sun")
            self.assertEqual(sun["capture"][sun["capture"].index("--reflections") + 1], "ssr")
            self.assertEqual(env["capture"][env["capture"].index("--reflections") + 1], "off")
            self.assertEqual(env["capture"][env["capture"].index("--lighting-denoise") + 1], "custom")
            for name in ("specular-ignored", "f90-one", "specular-weight-not-on-f0"):
                self.assertIn(name, sun["reference"])
            for name in ("declared-specular-ignored", "declared-f90-one", "declared-specular-weight-not-on-f0"):
                self.assertIn(name, env["reference"])
            self.assertTrue(check.SCENARIOS["specular-grid-sun"]["always_green"])

    def test_point_light_variants_share_the_bench_scene_and_set_lighting_rt_and_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                code = check.main(["--out", tmp, "--binary", "build/release/phosphor", "--only", "specular-grid-point",
                                   "--only", "specular-grid-point@restir", "--only", "specular-grid-point@clustered"])
            self.assertEqual(code, 0)
            plan = json.loads((pathlib.Path(tmp) / "plan.json").read_text())
            for name, lighting, rt, visibility in (("specular-grid-point", "legacy", "off", "none"),
                                                   ("specular-grid-point@restir", "restir", "on", "rays"),
                                                   ("specular-grid-point@clustered", "clustered", "on", "rays")):
                capture = plan["commands"][name]["capture"]
                reference = plan["commands"][name]["reference"]
                self.assertEqual(capture[capture.index("--material-scenario") + 1], "specular-grid-point")
                self.assertEqual(capture[capture.index("--lighting") + 1], lighting)
                self.assertEqual(capture[capture.index("--rt") + 1], rt)
                self.assertEqual(capture[capture.index("--shadows") + 1], "off")
                self.assertEqual(reference[reference.index("--visibility") + 1], visibility)
                self.assertEqual(capture[capture.index("--lighting-denoise") + 1],
                                 "custom" if lighting == "legacy" else "off")
                for control in ("specular-ignored", "f90-one", "specular-weight-not-on-f0"):
                    self.assertIn(control, reference)
        self.assertTrue(check.SCENARIOS["specular-grid-point"]["always_green"])
        for key in ("specular-grid-point@restir", "specular-grid-point@clustered"):
            self.assertNotIn("pending", check.SCENARIOS[key])
            self.assertNotIn("pending_matches", check.SCENARIOS[key])

    def test_transport_variants_of_the_specular_grid_set_reflections_rt_and_denoise(self):
        # F15.1 (ticket 08): SSR (default), RT reflections, the probe path and the extended furnace.
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                code = check.main(["--out", tmp, "--binary", "build/release/phosphor", "--only", "specular-grid-env",
                                   "--only", "specular-grid-env@rt", "--only", "specular-grid-env@probes",
                                   "--only", "furnace-specular", "--only", "furnace-specular@resolve"])
            self.assertEqual(code, 0)
            commands = json.loads((pathlib.Path(tmp) / "plan.json").read_text())["commands"]
            for name, bench, reflections, rt, denoise in (
                    ("specular-grid-env", "specular-grid-env", "ssr", "off", "custom"),
                    ("specular-grid-env@rt", "specular-grid-env", "rt", "on", "custom"),
                    ("specular-grid-env@probes", "specular-grid-env", "probes", "off", "off"),
                    ("furnace-specular", "furnace-specular", "ssr", "off", "custom"),
                    ("furnace-specular@resolve", "furnace-specular", "off", "off", "custom")):
                capture = commands[name]["capture"]
                self.assertEqual(capture[capture.index("--material-scenario") + 1], bench, name)
                self.assertEqual(capture[capture.index("--reflections") + 1], reflections, name)
                self.assertEqual(capture[capture.index("--rt") + 1], rt, name)
                self.assertEqual(capture[capture.index("--lighting-denoise") + 1], denoise, name)
        self.assertEqual(check.SCENARIOS["specular-grid-env"]["kind"], "environment")
        self.assertEqual(check.SCENARIOS["specular-grid-env@probes"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["furnace-specular"]["kind"], "furnace-transport")
        self.assertEqual(check.SCENARIOS["furnace-specular@resolve"]["kind"], "furnace")
        for key in ("specular-grid-env", "specular-grid-env@rt", "specular-grid-env@probes", "furnace-specular",
                    "furnace-specular@resolve"):
            self.assertTrue(check.SCENARIOS[key]["always_green"], key)

    def test_clearcoat_keys_are_planned_with_their_controls_and_no_longer_pending(self):
        keys = ["clearcoat-grid-sun", "clearcoat-grid-env-resolve", "clearcoat-grid-env", "clearcoat-grid-point",
                "clearcoat-grid-point@restir", "clearcoat-grid-point@clustered", "furnace-clearcoat",
                "furnace-clearcoat@resolve"]
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                args = ["--out", tmp, "--binary", "build/release/phosphor"]
                for key in keys:
                    args += ["--only", key]
                self.assertEqual(check.main(args), 0)
            commands = json.loads((pathlib.Path(tmp) / "plan.json").read_text())["commands"]
        expected = {"clearcoat-grid-sun": ("clearcoat-grid-sun", "legacy", "off", "ssr", "custom", "none"),
                    "clearcoat-grid-env-resolve": ("clearcoat-grid-env-resolve", "legacy", "off", "off", "custom", "none"),
                    "clearcoat-grid-env": ("clearcoat-grid-env", "legacy", "off", "ssr", "custom", "none"),
                    "clearcoat-grid-point": ("clearcoat-grid-point", "legacy", "off", "ssr", "custom", "none"),
                    "clearcoat-grid-point@restir": ("clearcoat-grid-point", "restir", "on", "ssr", "off", "rays"),
                    "clearcoat-grid-point@clustered": ("clearcoat-grid-point", "clustered", "on", "ssr", "off", "rays"),
                    "furnace-clearcoat": ("furnace-clearcoat", "legacy", "off", "ssr", "custom", "none"),
                    "furnace-clearcoat@resolve": ("furnace-clearcoat", "legacy", "off", "off", "custom", "none")}
        for key, (bench, lighting, rt, reflections, denoise, visibility) in expected.items():
            capture, reference = commands[key]["capture"], commands[key]["reference"]
            self.assertEqual(capture[capture.index("--material-scenario") + 1], bench, key)
            self.assertEqual(capture[capture.index("--lighting") + 1], lighting, key)
            self.assertEqual(capture[capture.index("--rt") + 1], rt, key)
            self.assertEqual(capture[capture.index("--reflections") + 1], reflections, key)
            self.assertEqual(capture[capture.index("--lighting-denoise") + 1], denoise, key)
            self.assertEqual(reference[reference.index("--visibility") + 1], visibility, key)
            spec = check.SCENARIOS[key]
            self.assertNotIn("pending", spec, key)
            self.assertNotIn("pending_matches", spec, key)
            self.assertNotIn("pending_gate", spec, key)
            literal, declared = check.scenario_controls(key)
            for name in literal + declared:
                self.assertIn(name, reference, key)
        self.assertTrue(check.SCENARIOS["furnace-clearcoat"]["always_green"])
        self.assertTrue(check.SCENARIOS["furnace-clearcoat@resolve"]["always_green"])
        self.assertEqual(check.SCENARIOS["clearcoat-grid-env-resolve"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["clearcoat-grid-env"]["kind"], "environment")
        for key in ("clearcoat-grid-sun", "clearcoat-grid-point"):
            self.assertEqual(check.SCENARIOS[key]["kind"], "direct")
            for name in ("coat-ignored", "coat-f0-from-ior", "coat-fresnel-vh", "coat-base-unattenuated"):
                self.assertIn(name, check.SCENARIOS[key]["controls"])

    def test_sheen_keys_are_planned_with_their_controls_and_no_longer_pending(self):
        keys = ["sheen-grid-sun", "sheen-grid-env-resolve", "sheen-grid-env", "sheen-grid-env@probes", "sheen-grid-point",
                "sheen-grid-point@restir", "sheen-grid-point@clustered", "furnace-sheen", "furnace-sheen@resolve"]
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                args = ["--out", tmp, "--binary", "build/release/phosphor"]
                for key in keys:
                    args += ["--only", key]
                self.assertEqual(check.main(args), 0)
            commands = json.loads((pathlib.Path(tmp) / "plan.json").read_text())["commands"]
        expected = {"sheen-grid-sun": ("sheen-grid-sun", "legacy", "off", "ssr", "custom", "none"),
                    "sheen-grid-env-resolve": ("sheen-grid-env-resolve", "legacy", "off", "off", "custom", "none"),
                    "sheen-grid-env": ("sheen-grid-env", "legacy", "off", "ssr", "custom", "none"),
                    "sheen-grid-env@probes": ("sheen-grid-env", "legacy", "off", "probes", "off", "none"),
                    "sheen-grid-point": ("sheen-grid-point", "legacy", "off", "ssr", "custom", "none"),
                    "sheen-grid-point@restir": ("sheen-grid-point", "restir", "on", "ssr", "off", "rays"),
                    "sheen-grid-point@clustered": ("sheen-grid-point", "clustered", "on", "ssr", "off", "rays"),
                    "furnace-sheen": ("furnace-sheen", "legacy", "off", "ssr", "custom", "none"),
                    "furnace-sheen@resolve": ("furnace-sheen", "legacy", "off", "off", "custom", "none")}
        for key, (bench, lighting, rt, reflections, denoise, visibility) in expected.items():
            capture, reference = commands[key]["capture"], commands[key]["reference"]
            self.assertEqual(capture[capture.index("--material-scenario") + 1], bench, key)
            self.assertEqual(capture[capture.index("--lighting") + 1], lighting, key)
            self.assertEqual(capture[capture.index("--rt") + 1], rt, key)
            self.assertEqual(capture[capture.index("--reflections") + 1], reflections, key)
            self.assertEqual(capture[capture.index("--lighting-denoise") + 1], denoise, key)
            self.assertEqual(reference[reference.index("--visibility") + 1], visibility, key)
            spec = check.SCENARIOS[key]
            self.assertNotIn("pending", spec, key)
            self.assertNotIn("pending_matches", spec, key)
            self.assertNotIn("pending_gate", spec, key)
            literal, declared = check.scenario_controls(key)
            for name in literal + declared:
                self.assertIn(name, reference, key)
        # Every sheen control the reference offers is a known control name.
        for key in keys:
            literal, declared = check.scenario_controls(key)
            for name in literal:
                self.assertIn(name, ref.SHEEN_CONTROLS, key)
            for name in declared:
                self.assertIn(name, ref.SHEEN_DECLARED_CONTROLS, key)
        self.assertTrue(check.SCENARIOS["furnace-sheen"]["always_green"])
        self.assertTrue(check.SCENARIOS["furnace-sheen@resolve"]["always_green"])
        self.assertEqual(check.SCENARIOS["sheen-grid-env-resolve"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["sheen-grid-env@probes"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["sheen-grid-env"]["kind"], "environment")
        # The five direct controls (the environment forms share the cheap scaling: no cheap-scaling control there).
        for key in ("sheen-grid-sun", "sheen-grid-point", "sheen-grid-point@restir", "sheen-grid-point@clustered"):
            self.assertEqual(check.SCENARIOS[key]["kind"], "direct")
            self.assertEqual(sorted(check.SCENARIOS[key]["controls"]), sorted(ref.SHEEN_CONTROLS), key)
        self.assertNotIn("sheen-cheap-scaling", check.SCENARIOS["sheen-grid-env"]["controls"])

    def test_aniso_keys_are_planned_with_their_controls_and_no_longer_pending(self):
        keys = ["aniso-grid-sun", "aniso-grid-env-resolve", "aniso-grid-env", "aniso-grid-point", "aniso-grid-point@restir",
                "aniso-grid-point@clustered", "furnace-aniso", "furnace-aniso@resolve", "aniso-mirrored-sun"]
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                args = ["--out", tmp, "--binary", "build/release/phosphor"]
                for key in keys:
                    args += ["--only", key]
                self.assertEqual(check.main(args), 0)
            commands = json.loads((pathlib.Path(tmp) / "plan.json").read_text())["commands"]
        expected = {"aniso-grid-sun": ("aniso-grid-sun", "legacy", "off", "ssr", "custom", "none"),
                    "aniso-grid-env-resolve": ("aniso-grid-env-resolve", "legacy", "off", "off", "custom", "none"),
                    "aniso-grid-env": ("aniso-grid-env", "legacy", "off", "ssr", "custom", "none"),
                    "aniso-grid-point": ("aniso-grid-point", "legacy", "off", "ssr", "custom", "none"),
                    "aniso-grid-point@restir": ("aniso-grid-point", "restir", "on", "ssr", "off", "rays"),
                    "aniso-grid-point@clustered": ("aniso-grid-point", "clustered", "on", "ssr", "off", "rays"),
                    "furnace-aniso": ("furnace-aniso", "legacy", "off", "ssr", "custom", "none"),
                    "furnace-aniso@resolve": ("furnace-aniso", "legacy", "off", "off", "custom", "none"),
                    "aniso-mirrored-sun": ("aniso-mirrored-sun", "legacy", "off", "ssr", "custom", "none")}
        for key, (bench, lighting, rt, reflections, denoise, visibility) in expected.items():
            capture, reference = commands[key]["capture"], commands[key]["reference"]
            self.assertEqual(capture[capture.index("--material-scenario") + 1], bench, key)
            self.assertEqual(capture[capture.index("--lighting") + 1], lighting, key)
            self.assertEqual(capture[capture.index("--rt") + 1], rt, key)
            self.assertEqual(capture[capture.index("--reflections") + 1], reflections, key)
            self.assertEqual(capture[capture.index("--lighting-denoise") + 1], denoise, key)
            self.assertEqual(reference[reference.index("--visibility") + 1], visibility, key)
            literal, declared = check.scenario_controls(key)
            for name in literal + declared:
                self.assertIn(name, reference, key)
            spec = check.SCENARIOS[key]
            for name in literal:
                self.assertIn(name, ref.ANISO_CONTROLS, key)
            for name in declared:
                self.assertIn(name, ref.ANISO_DECLARED_CONTROLS, key)
            for field in ("pending", "pending_matches", "pending_gate"):
                self.assertNotIn(field, spec, key)
        self.assertTrue(check.SCENARIOS["furnace-aniso"]["always_green"])
        self.assertNotIn("pending", check.SCENARIOS["furnace-aniso"])
        self.assertTrue(check.SCENARIOS["furnace-aniso@resolve"]["always_green"])
        self.assertEqual(check.SCENARIOS["aniso-grid-env-resolve"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["aniso-grid-env"]["kind"], "environment")
        for key in ("aniso-grid-sun", "aniso-grid-point", "aniso-grid-point@restir", "aniso-grid-point@clustered"):
            self.assertEqual(check.SCENARIOS[key]["kind"], "direct")
            self.assertEqual(sorted(check.SCENARIOS[key]["controls"]), sorted(
                set(ref.ANISO_CONTROLS) - {"aniso-mirror-unaware"}), key)
        # The mirrored fixture alone has the determinant control, and every anisotropy control.
        self.assertEqual(sorted(check.SCENARIOS["aniso-mirrored-sun"]["controls"]), sorted(ref.ANISO_CONTROLS))

    def test_iridescence_keys_are_planned_with_their_controls_and_no_longer_pending(self):
        keys = ["iridescence-grid-sun", "iridescence-grid-env-resolve", "iridescence-grid-env", "iridescence-grid-point",
                "iridescence-grid-point@restir", "iridescence-grid-point@clustered", "furnace-iridescence",
                "furnace-iridescence@resolve"]
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                args = ["--out", tmp, "--binary", "build/release/phosphor"]
                for key in keys:
                    args += ["--only", key]
                self.assertEqual(check.main(args), 0)
            commands = json.loads((pathlib.Path(tmp) / "plan.json").read_text())["commands"]
        expected = {"iridescence-grid-sun": ("iridescence-grid-sun", "legacy", "off", "ssr", "custom", "none"),
                    "iridescence-grid-env-resolve": ("iridescence-grid-env-resolve", "legacy", "off", "off", "custom", "none"),
                    "iridescence-grid-env": ("iridescence-grid-env", "legacy", "off", "ssr", "custom", "none"),
                    "iridescence-grid-point": ("iridescence-grid-point", "legacy", "off", "ssr", "custom", "none"),
                    "iridescence-grid-point@restir": ("iridescence-grid-point", "restir", "on", "ssr", "off", "rays"),
                    "iridescence-grid-point@clustered": ("iridescence-grid-point", "clustered", "on", "ssr", "off", "rays"),
                    "furnace-iridescence": ("furnace-iridescence", "legacy", "off", "ssr", "custom", "none"),
                    "furnace-iridescence@resolve": ("furnace-iridescence", "legacy", "off", "off", "custom", "none")}
        for key, (bench, lighting, rt, reflections, denoise, visibility) in expected.items():
            capture, reference = commands[key]["capture"], commands[key]["reference"]
            self.assertEqual(capture[capture.index("--material-scenario") + 1], bench, key)
            self.assertEqual(capture[capture.index("--lighting") + 1], lighting, key)
            self.assertEqual(capture[capture.index("--rt") + 1], rt, key)
            self.assertEqual(capture[capture.index("--reflections") + 1], reflections, key)
            self.assertEqual(capture[capture.index("--lighting-denoise") + 1], denoise, key)
            self.assertEqual(reference[reference.index("--visibility") + 1], visibility, key)
            literal, declared = check.scenario_controls(key)
            for name in literal + declared:
                self.assertIn(name, reference, key)
            spec = check.SCENARIOS[key]
            for name in literal:
                self.assertIn(name, ref.IRID_CONTROLS, key)
            for name in declared:
                self.assertIn(name, ref.IRID_DECLARED_CONTROLS, key)
            for field in ("pending", "pending_matches", "pending_gate"):
                self.assertNotIn(field, spec, key)
        self.assertTrue(check.SCENARIOS["furnace-iridescence"]["always_green"])
        self.assertTrue(check.SCENARIOS["furnace-iridescence@resolve"]["always_green"])
        self.assertEqual(check.SCENARIOS["iridescence-grid-env-resolve"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["iridescence-grid-env"]["kind"], "environment")
        for key in ("iridescence-grid-sun", "iridescence-grid-env", "iridescence-grid-point", "iridescence-grid-point@restir",
                    "iridescence-grid-point@clustered"):
            self.assertEqual(sorted(check.SCENARIOS[key]["controls"]), sorted(ref.IRID_CONTROLS), key)
        self.assertEqual(sorted(check.SCENARIOS["iridescence-grid-env-resolve"]["controls"]), sorted(ref.IRID_DECLARED_CONTROLS))

    def test_thin_subsurface_keys_are_planned_with_their_controls_and_no_longer_pending(self):
        keys = ["subsurface-thin-sun", "subsurface-thin-env-resolve", "subsurface-thin-env", "subsurface-thin-point",
                "subsurface-thin-point@restir", "subsurface-thin-point@clustered", "furnace-thin-subsurface",
                "furnace-thin-subsurface@resolve"]
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                args = ["--out", tmp, "--binary", "build/release/phosphor"]
                for key in keys:
                    args += ["--only", key]
                self.assertEqual(check.main(args), 0)
            commands = json.loads((pathlib.Path(tmp) / "plan.json").read_text())["commands"]
        expected = {"subsurface-thin-sun": ("subsurface-thin-sun", "legacy", "off", "ssr", "custom", "none"),
                    "subsurface-thin-env-resolve": ("subsurface-thin-env-resolve", "legacy", "off", "off", "custom", "none"),
                    "subsurface-thin-env": ("subsurface-thin-env", "legacy", "off", "ssr", "custom", "none"),
                    "subsurface-thin-point": ("subsurface-thin-point", "legacy", "off", "ssr", "custom", "none"),
                    "subsurface-thin-point@restir": ("subsurface-thin-point", "restir", "on", "ssr", "off", "rays"),
                    "subsurface-thin-point@clustered": ("subsurface-thin-point", "clustered", "on", "ssr", "off", "rays"),
                    "furnace-thin-subsurface": ("furnace-thin-subsurface", "legacy", "off", "ssr", "custom", "none"),
                    "furnace-thin-subsurface@resolve": ("furnace-thin-subsurface", "legacy", "off", "off", "custom", "none")}
        for key, (bench, lighting, rt, reflections, denoise, visibility) in expected.items():
            capture, reference = commands[key]["capture"], commands[key]["reference"]
            self.assertEqual(capture[capture.index("--material-scenario") + 1], bench, key)
            self.assertEqual(capture[capture.index("--lighting") + 1], lighting, key)
            self.assertEqual(capture[capture.index("--rt") + 1], rt, key)
            self.assertEqual(capture[capture.index("--reflections") + 1], reflections, key)
            self.assertEqual(capture[capture.index("--lighting-denoise") + 1], denoise, key)
            self.assertEqual(reference[reference.index("--visibility") + 1], visibility, key)
            literal, declared = check.scenario_controls(key)
            for name in literal + declared:
                self.assertIn(name, reference, key)
            spec = check.SCENARIOS[key]
            for name in literal:
                self.assertIn(name, ref.THIN_CONTROLS, key)
            for name in declared:
                self.assertIn(name, ref.THIN_DECLARED_CONTROLS, key)
            for field in ("pending", "pending_matches", "pending_gate"):
                self.assertNotIn(field, spec, key)
        self.assertTrue(check.SCENARIOS["furnace-thin-subsurface"]["always_green"])
        self.assertTrue(check.SCENARIOS["furnace-thin-subsurface@resolve"]["always_green"])
        self.assertEqual(check.SCENARIOS["subsurface-thin-env-resolve"]["kind"], "declared")
        self.assertEqual(check.SCENARIOS["subsurface-thin-env"]["kind"], "environment")
        for key in ("subsurface-thin-sun", "subsurface-thin-env"):
            self.assertEqual(sorted(check.SCENARIOS[key]["controls"]), sorted(ref.THIN_CONTROLS), key)
        # the point-light carriers have no front-lit pixel: the double-counted control cannot differ there
        for key in ("subsurface-thin-point", "subsurface-thin-point@restir", "subsurface-thin-point@clustered"):
            self.assertEqual(sorted(check.SCENARIOS[key]["controls"]),
                             sorted(set(ref.THIN_CONTROLS) - {"transmission-double-counted"}), key)
        self.assertEqual(sorted(check.SCENARIOS["subsurface-thin-env-resolve"]["controls"]), sorted(ref.THIN_DECLARED_CONTROLS))

    def test_a_reference_that_is_black_over_the_whole_roi_compares_equal_to_a_black_image_and_not_to_a_lit_one(self):
        roi = np.ones((2, 2), dtype=bool)
        black = np.zeros((2, 2, 3), dtype=np.float32)
        error, floor = check.relative_error(np, black, black, roi)
        self.assertEqual(floor, 0.0)
        np.testing.assert_array_equal(error, np.zeros(4))      # 0 / 0 is no error, never NaN
        lit = np.full((2, 2, 3), 0.01, dtype=np.float32)
        error, _ = check.relative_error(np, lit, black, roi)
        self.assertTrue(bool(np.all(error > 1e30)))            # a lit pixel where the reference is black is infinitely wrong
        # the ordinary floor rule is unchanged
        reference = np.full((2, 2, 3), 0.5, dtype=np.float32)
        error, floor = check.relative_error(np, reference * 1.1, reference, roi)
        self.assertAlmostEqual(floor, 0.025)
        np.testing.assert_allclose(error, np.full(4, 0.1), rtol=1e-5)

    def test_pending_matches_must_also_hold_in_green_mode(self):
        failing = {"kind": "direct", "pass": False, "rows": {}}
        base = {"discriminating": True, "gates_pass": False, "pending": "ticket 09", "result": failing,
                "expectation": {"met": True}}
        self.assertEqual(check.verdict([{**base, "pending_matches_met": True}], "green"), ("GREEN", check.EXIT_OK))
        self.assertEqual(check.verdict([{**base, "pending_matches_met": False}], "green")[1], check.EXIT_UNEXPECTED)

    def test_terminator_band_width_two_reaches_two_pixels_into_the_lit_side(self):
        lit = np.zeros((7, 9), dtype=bool)
        lit[:, 4:] = True  # columns 4.. lit, 0..3 unlit, one sphere
        sphere = np.zeros((7, 9), dtype=np.int64)
        one = check.terminator_band(np, lit, sphere)
        two = check.terminator_band(np, lit, sphere, 2)
        self.assertEqual(sorted(set(np.nonzero(one)[1].tolist())), [4])
        self.assertEqual(sorted(set(np.nonzero(two)[1].tolist())), [4, 5])
        other = np.ones((7, 9), dtype=np.int64)
        other[:, 4:] = 2  # the unlit pixels belong to another sphere: no band
        self.assertFalse(check.terminator_band(np, lit, other, 2).any())

    def test_pending_scenario_must_fail_in_every_mode_and_green_requires_it(self):
        failing = {"kind": "direct", "pass": False, "rows": {}}
        passing = {"kind": "direct", "pass": True, "rows": {}}
        for mode in ("red", "green"):
            self.assertTrue(check.expectation("direct", mode, failing, pending="ticket 07")["met"])
            self.assertFalse(check.expectation("direct", mode, passing, pending="ticket 07")["met"])
        ok = {"discriminating": True, "gates_pass": True, "pending": None, "result": passing,
              "expectation": {"met": True}}
        pending_failing = {"discriminating": True, "gates_pass": False, "pending": "ticket 07", "result": failing,
                           "expectation": {"met": True}}
        pending_passing = {**pending_failing, "gates_pass": True, "result": passing}
        self.assertEqual(check.verdict([ok, pending_failing], "green"), ("GREEN", check.EXIT_OK))
        self.assertEqual(check.verdict([ok, pending_passing], "green")[1], check.EXIT_UNEXPECTED)

    def test_plan_prints_the_exact_commands_of_the_new_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = check.main(["--out", tmp, "--binary", "build/release/phosphor", "--only", "base-grid-env-resolve",
                                   "--only", "roughness-extremes-sun"])
            self.assertEqual(code, 0)
            plan = json.loads((pathlib.Path(tmp) / "plan.json").read_text())
            resolve = plan["commands"]["base-grid-env-resolve"]["capture"]
            self.assertEqual(resolve[resolve.index("--reflections") + 1], "off")
            self.assertEqual(resolve[resolve.index("--lighting-denoise") + 1], "custom")
            self.assertEqual(resolve[resolve.index("--material-scenario") + 1], "base-grid-env-resolve")
            sun = plan["commands"]["roughness-extremes-sun"]["capture"]
            self.assertEqual(sun[sun.index("--material-scenario") + 1], "roughness-extremes-sun")
            self.assertEqual(sun[sun.index("--reflections") + 1], "ssr")
            reference = plan["commands"]["base-grid-env-resolve"]["reference"]
            self.assertIn("--declared-controls", reference)
            for name in ref.DECLARED_CONTROLS:
                self.assertIn(name, reference)
            self.assertNotIn("--declared-controls", plan["commands"]["roughness-extremes-sun"]["reference"])
            self.assertIn("furnace-resolve", check.SCENARIOS)
            self.assertIn("--declared-controls", check.reference_command("furnace-resolve", pathlib.Path("x")))
            for line in buffer.getvalue().splitlines():
                if line.startswith("#"):
                    continue
                self.assertTrue(line.startswith(("build/release/phosphor", str(pathlib.Path("build/release/phosphor").resolve())))
                                or line.startswith("uv ") or line.startswith("NOT_EXECUTED"), line)

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
            self.assertEqual(code, check.EXIT_GPU_BUSY)
            self.assertIn("refusing to start", err.getvalue())

    def test_exit_codes_are_distinct_and_documented(self):
        codes = [check.EXIT_OK, check.EXIT_UNEXPECTED, check.EXIT_USAGE, check.EXIT_NON_DISCRIMINATING,
                 check.EXIT_GPU_BUSY, check.EXIT_INTERNAL, check.EXIT_CAPTURE_FAILED]
        self.assertEqual(codes, [0, 1, 2, 3, 4, 5, 6])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as raised:
            check.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = out.getvalue()
        for code, words in ((0, "expected state confirmed"), (1, "unexpected gate result"), (2, "usage error"),
                            (3, "non-discriminating"), (4, "GPU busy"), (5, "internal error"), (6, "capture failed")):
            self.assertRegex(help_text, rf"\n\s+{code}\s+.*{words}", (code, words))

    def test_usage_errors_exit_2(self):
        for argv in (["--only", "nope"], ["--expect", "blue"], ["--run", "--evaluate"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                check.main(argv)
            self.assertEqual(raised.exception.code, check.EXIT_USAGE, argv)
        with tempfile.TemporaryDirectory() as tmp:  # --run without a renderer
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                check.main(["--run", "--binary", str(pathlib.Path(tmp) / "absent"), "--out", tmp])
            self.assertEqual(raised.exception.code, check.EXIT_USAGE)

    def run_with(self, tmp, argv, busy_sequence, captures):
        """Run --run with the GPU guard and the capture replaced; returns (exit code, guard calls, capture names)."""
        binary = pathlib.Path(tmp) / "phosphor"
        binary.write_bytes(b"x")
        guard_calls = []
        captured = []

        def guard(*_a, **_k):
            guard_calls.append(1)
            return busy_sequence[min(len(guard_calls) - 1, len(busy_sequence) - 1)]

        def capture(command, log, timeout):
            captured.append(command[command.index("--material-scenario") + 1])
            return captures(len(captured))

        saved = (check.running_gpu_processes, check.capture, check.time.sleep)
        check.running_gpu_processes, check.capture, check.time.sleep = guard, capture, lambda *_: None
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                code = check.main(["--run", "--binary", str(binary), "--out", str(pathlib.Path(tmp) / "o"),
                                   "--gpu-lock", str(pathlib.Path(tmp) / "l"), *argv])
        finally:
            check.running_gpu_processes, check.capture, check.time.sleep = saved
        return code, guard_calls, captured

    def test_gpu_guard_runs_before_every_capture(self):
        ok = lambda _n: {"passed": True, "failures": []}  # noqa: E731
        with tempfile.TemporaryDirectory() as tmp:
            # Free for the first scenario, busy for the second: exit 4 after exactly one capture.
            code, guards, captured = self.run_with(tmp, ["--only", "base-grid-sun", "--only", "base-grid-env"],
                                                   [[], ["999:phosphor"]], ok)
            self.assertEqual(code, check.EXIT_GPU_BUSY)
            self.assertEqual(captured, ["base-grid-sun"])
            self.assertEqual(len(guards), 2)  # one guard call before each capture attempt
        with tempfile.TemporaryDirectory() as tmp:
            code, guards, captured = self.run_with(tmp, ["--only", "base-grid-sun"], [["999:phosphor"]], ok)
            self.assertEqual((code, captured, len(guards)), (check.EXIT_GPU_BUSY, [], 1))  # busy before the first

    def test_capture_failure_exits_6_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, captured = self.run_with(tmp, ["--only", "base-grid-sun", "--only", "base-grid-env"], [[]],
                                              lambda _n: {"passed": False, "failures": ["boom"]})
            self.assertEqual(code, check.EXIT_CAPTURE_FAILED)
            self.assertEqual(captured, ["base-grid-sun"])

    def test_output_folders_are_checked_before_the_capture_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = pathlib.Path(tmp) / "o" / "base-grid-env"
            stale.mkdir(parents=True)
            (stale / "leftover.txt").write_text("x")
            with self.assertRaises(SystemExit) as raised:
                self.run_with(tmp, ["--only", "base-grid-sun", "--only", "base-grid-env"], [[]],
                              lambda _n: {"passed": True, "failures": []})
            self.assertEqual(raised.exception.code, check.EXIT_USAGE)
            self.assertFalse((pathlib.Path(tmp) / "o" / "base-grid-sun").exists())  # nothing was started

    def test_any_exception_is_an_internal_error_not_a_gate_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = check.evaluate_scenario
            check.evaluate_scenario = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    code = check.main(["--evaluate", "--out", tmp, "--only", "base-grid-sun"])
            finally:
                check.evaluate_scenario = original
            self.assertEqual(code, check.EXIT_INTERNAL)

    def test_lock_held_by_another_verification_exits_4(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = pathlib.Path(tmp) / "l"
            binary = pathlib.Path(tmp) / "phosphor"
            binary.write_bytes(b"x")
            with check.gpu_lock(lock), contextlib.redirect_stderr(io.StringIO()):
                code = check.main(["--run", "--binary", str(binary), "--out", str(pathlib.Path(tmp) / "o"),
                                   "--gpu-lock", str(lock)])
            self.assertEqual(code, check.EXIT_GPU_BUSY)

    def test_verdict_exit_codes(self):
        def entry(discriminating=True, met=True, passed=False):
            return {"discriminating": discriminating, "expectation": {"met": met}, "result": {"pass": passed}}
        self.assertEqual(check.verdict([entry(False)], "red")[1], check.EXIT_NON_DISCRIMINATING)
        self.assertEqual(check.verdict([entry(met=False)], "red")[1], check.EXIT_UNEXPECTED)
        self.assertEqual(check.verdict([entry()], "red")[1], check.EXIT_OK)
        # A declared-model gate failing next to a passing literal gate is a RED green-mode verdict.
        both = {"discriminating": True, "expectation": {"met": True}, "result": {"pass": True}, "gates_pass": False}
        self.assertEqual(check.verdict([both], "green"), ("RED", check.EXIT_UNEXPECTED))

    def test_sw_vers_parsing_is_robust(self):
        full = "ProductName:\t\tmacOS\nProductVersion:\t\t27.3\nBuildVersion:\t\t26D1\n"
        self.assertEqual(check.parse_sw_vers(full), {"ProductName": "macOS", "ProductVersion": "27.3", "BuildVersion": "26D1"})
        self.assertEqual(check.os_description(check.parse_sw_vers(full)), "macOS 27.3 (26D1)")
        for odd in (None, "", "\n\n", "single line without a colon", "ProductName: macOS"):
            self.assertIsInstance(check.parse_sw_vers(odd), dict)
        self.assertEqual(check.os_description(check.parse_sw_vers(None)), "")
        self.assertEqual(check.os_description(check.parse_sw_vers("ProductName: macOS")), "macOS")
        self.assertEqual(check.parse_sw_vers("Key: a: b")["Key"], "a: b")
        # The markdown header tolerates a manifest without OS information.
        text = check.markdown({"commit": "c", "binary_sha256": "b", "os": None}, [], "S")
        self.assertIn("# F15.3 material validation: S", text)

    def test_one_sha256_helper_serves_both_tools(self):
        import f15_common
        self.assertIs(check.sha256_file, f15_common.sha256_file)
        self.assertIs(ref.sha256_file, f15_common.sha256_file)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "x"
            path.write_bytes(b"abc")
            self.assertEqual(f15_common.sha256_file(path),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")  # FIPS 180-2 vector

    def test_scenario_names_match_the_bench(self):
        source = (HERE.parent / "src" / "testbench" / "material_validation.cpp").read_text()
        import re
        block = source[source.index("kScenarios{"):source.index("};", source.index("kScenarios{"))]
        # the two clip fixtures of ticket 08 belong to tools/f15_clip_check.py, not to this reference runner
        clip_only = {"specular-grid-edit", "specular-grid-rough"}
        # ticket 13: the transmission scenarios are the separate bench class TransmissionValidation
        transmission = (HERE.parent / "src" / "testbench" / "transmission_validation.cpp").read_text()
        names = transmission[transmission.index("kAll{"):transmission.index("};", transmission.index("kAll{"))]
        self.assertEqual(sorted((set(re.findall(r'"([a-z0-9-]+)"', block)) - clip_only) | set(re.findall(r'"([a-z0-9-]+)"', names))),
                         sorted({check.bench_scenario(name) for name in check.SCENARIOS}))

    def test_declared_gate_bounds(self):
        masks, spheres = self.synthetic_masks()
        reference = np.full((40, 40, 3), 0.5, dtype=np.float32)
        ok = reference * 1.004  # 0.4 % everywhere: mean <= 0.5 %, p99 <= 1 %
        self.assertTrue(check.evaluate_declared(np, ok, reference, masks, spheres)["pass"])
        high_mean = reference * 1.006  # mean 0.6 % > 0.5 %
        self.assertFalse(check.evaluate_declared(np, high_mean, reference, masks, spheres)["pass"])
        outliers = reference.copy()
        outliers[:2, :20] *= 1.03  # 40 of 800 pixels of sphere 0 at 3 %: mean 0.15 %, p99 3 %
        result = check.evaluate_declared(np, outliers, reference, masks, spheres)
        self.assertFalse(result["spheres"][0]["pass"])
        self.assertLessEqual(result["spheres"][0]["mean"], 0.005)
        self.assertGreater(result["spheres"][0]["p99"], 0.01)
        self.assertTrue(result["spheres"][1]["pass"])
        self.assertEqual(check.THRESHOLDS["declared"]["mean_per_sphere"], 0.005)
        self.assertEqual(check.THRESHOLDS["declared"]["p99_per_sphere"], 0.01)

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

    def scenario(self, name, engine_model, mode, environment=None, tag="", key=None):
        """Evaluate one scenario against a synthetic engine image: `engine_model` for the sun and the literal
        environment, or `environment` = a declared variant for the engine's declared environment model."""
        folder = pathlib.Path(self.tmp.name) / f"{name}-{mode}{tag}"
        make_bench_snapshot(folder, name, self.WIDTH, self.HEIGHT)
        from f12_reference import snapshot as load
        root, data = load(folder / "snapshot")
        image, *_ = ref.render_frame(root, data, engine_model, environment=environment)
        write_pfm(folder / "engine.pfm", image)
        with contextlib.redirect_stdout(io.StringIO()):
            return check.evaluate_scenario(np, key or name, folder, mode)

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

    def test_declared_environment_gate_passes_the_declared_model_and_rejects_every_control(self):
        entries = [self.scenario(name, ref.APPENDIX_B, "green", environment=ref.DECLARED)
                   for name in ("base-grid-env-resolve", "furnace-resolve")]
        for entry in entries:
            gate = entry["result"] if entry["kind"] == "declared" else entry["declared"]["result"]
            self.assertTrue(gate["pass"], (entry["scenario"], json.dumps(check.summarize(gate))))
            self.assertLess(gate["worst_mean"], 1e-4)  # the same model on the same rays
            controls = entry["controls"] if entry["kind"] == "declared" else entry["declared"]["controls"]
            self.assertEqual(sorted(controls), sorted(ref.DECLARED_CONTROLS))
            for name, verdict in controls.items():
                self.assertTrue(verdict["fails_gate"], (entry["scenario"], name, verdict))
            self.assertTrue(entry["discriminating"], entry["scenario"])
            self.assertTrue(entry["gates_pass"], entry["scenario"])
        # The deterministic furnace keeps its literal gate: the declared model never exceeds 1.
        self.assertTrue(entries[1]["result"]["pass"], entries[1]["result"]["max_pixel"])
        self.assertEqual(check.verdict(entries, "green"), ("GREEN", 0))

    def test_each_declared_control_alone_fails_the_declared_gate_as_an_engine_image(self):
        # An engine that implemented a wrong model must not pass the declared gate.
        for control, variant in ref.DECLARED_CONTROLS.items():
            entry = self.scenario("base-grid-env-resolve", ref.APPENDIX_B, "green", environment=variant, tag="-" + control)
            self.assertFalse(entry["result"]["pass"], control)
            self.assertEqual(check.verdict([entry], "green")[1], check.EXIT_UNEXPECTED)

    def test_new_direct_scenario_is_green_and_its_controls_discriminate(self):
        entry = self.scenario("roughness-extremes-sun", ref.APPENDIX_B, "green")
        self.assertTrue(entry["result"]["pass"], json.dumps(check.summarize(entry["result"])))
        for name, verdict in entry["controls"].items():
            self.assertTrue(verdict["fails_gate"], (name, verdict))
        self.assertTrue(entry["discriminating"])
        # The current engine model (F0-blended diffuse weight) is red on the intermediate metallic row only.
        red = self.scenario("roughness-extremes-sun", ref.CONTROLS["engine-f0-blend"], "green", tag="-red")
        self.assertTrue(red["result"]["rows"]["0.0"]["pass"] and red["result"]["rows"]["1.0"]["pass"])
        self.assertFalse(red["result"]["rows"]["0.5"]["pass"])

    def test_specular_grid_sun_is_green_for_the_extension_and_every_control_fails(self):
        entry = self.scenario("specular-grid-sun", ref.SPECULAR_EXT, "red")  # always asserted green, even in red mode
        self.assertTrue(entry["result"]["pass"], json.dumps(check.summarize(entry["result"])))
        self.assertTrue(entry["expectation"]["asserted"] and entry["expectation"]["met"])
        self.assertEqual(sorted(entry["controls"]), ["f90-one", "specular-ignored", "specular-weight-not-on-f0"])
        for name, verdict in entry["controls"].items():
            self.assertTrue(verdict["fails_gate"], (name, verdict))
        self.assertTrue(entry["discriminating"])
        # Rows are grouped by (metallic, weight, colour): the four dielectric rows and the metal row.
        self.assertEqual(len(entry["result"]["rows"]), 5)
        self.assertEqual(check.verdict([entry], "red"), ("EXPECTED_RED_CONFIRMED", 0))

    def test_specular_grid_sun_rejects_an_engine_that_ignores_or_mis_models_the_extension(self):
        for control in ("specular-ignored", "f90-one", "specular-weight-not-on-f0"):
            entry = self.scenario("specular-grid-sun", ref.CONTROLS[control], "red", tag="-" + control)
            self.assertFalse(entry["result"]["pass"], control)
            self.assertFalse(entry["expectation"]["met"], control)  # asserted green: red is an unexpected result
            self.assertEqual(check.verdict([entry], "red")[1], check.EXIT_UNEXPECTED)

    def test_specular_grid_env_resolve_declared_gate_and_controls(self):
        entry = self.scenario("specular-grid-env-resolve", ref.APPENDIX_B, "red", environment=ref.DECLARED)
        self.assertTrue(entry["result"]["pass"], json.dumps(check.summarize(entry["result"])))
        self.assertLess(entry["result"]["worst_mean"], 1e-4)
        self.assertEqual(sorted(entry["controls"]), sorted(ref.SPECULAR_DECLARED_CONTROLS))
        for name, verdict in entry["controls"].items():
            self.assertTrue(verdict["fails_gate"], (name, verdict))
        self.assertTrue(entry["discriminating"] and entry["gates_pass"])
        for control, variant in ref.SPECULAR_DECLARED_CONTROLS.items():
            wrong = self.scenario("specular-grid-env-resolve", ref.APPENDIX_B, "red", environment=variant, tag="-" + control)
            self.assertFalse(wrong["result"]["pass"], control)
            self.assertEqual(check.verdict([wrong], "red")[1], check.EXIT_UNEXPECTED)

    # --- F15.3 (ticket 09): the clearcoat harness.  The "engine" images below are reference renders of the model the
    # real engine will have (coat) or has today (coat-ignored); the runner must call the second one red-for-the-
    # right-reason (pending key GREEN) and the first one a pending violation.

    COAT_KEYS = (("clearcoat-grid-sun", False), ("clearcoat-grid-env-resolve", True), ("clearcoat-grid-env", False),
                 ("clearcoat-grid-point", False), ("furnace-clearcoat@resolve", True))

    def coat_engine(self, key, ignored):
        """The engine image of the model (coat) or of today's engine (coat ignored), for one runner key."""
        declared = key in ("clearcoat-grid-env-resolve", "furnace-clearcoat@resolve")
        model = ref.CONTROLS["coat-ignored"] if ignored else ref.SPECULAR_EXT
        environment = None
        if declared:
            environment = ref.COAT_DECLARED_CONTROLS["declared-coat-ignored"] if ignored else ref.DECLARED
        return model, environment

    def test_clearcoat_keys_pass_with_an_engine_that_implements_the_coat(self):
        for key, _ in self.COAT_KEYS:
            model, environment = self.coat_engine(key, ignored=False)
            entry = self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + "-coat",
                                  key=key)
            self.assertTrue(entry["discriminating"], (key, json.dumps(entry["controls"], indent=1)))
            self.assertTrue(entry["gates_pass"], (key, json.dumps(check.summarize(entry["result"]))))
            self.assertFalse(entry.get("pending"), key)
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", check.EXIT_OK), key)

    def test_clearcoat_keys_fail_with_a_coat_blind_engine(self):
        # The old engine (coat ignored) fails the gate of every key but the furnace upper bound of the transport.
        for key, _ in self.COAT_KEYS:
            model, environment = self.coat_engine(key, ignored=True)
            entry = self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + "-blind",
                                  key=key)
            self.assertFalse(entry["gates_pass"], key)
            self.assertEqual(check.verdict([entry], "green")[1], check.EXIT_UNEXPECTED, key)

    def test_clearcoat_controls_each_fail_their_gate(self):
        for key, _ in self.COAT_KEYS:
            model, environment = self.coat_engine(key, ignored=False)
            entry = self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + "-ctl",
                                  key=key)
            controls = entry["declared"]["controls"] if entry["kind"] == "furnace" else entry["controls"]
            if entry["kind"] == "furnace":
                controls = {**entry["controls"], **controls}
            wanted = set(check.SCENARIOS[key]["controls"]) | set(check.SCENARIOS[key].get("declared", []))
            self.assertEqual(set(controls), wanted, key)
            for name, verdict in controls.items():
                self.assertTrue(verdict["fails_gate"], (key, name, verdict))

    def test_furnace_clearcoat_bound_holds_for_the_model_and_the_unattenuated_base_breaks_it(self):
        key = "furnace-clearcoat"
        for ignored in (False, True):
            model, _ = self.coat_engine(key, ignored)
            entry = self.scenario(key, model, "red", tag=f"-{ignored}")
            self.assertTrue(entry["result"]["pass"], ignored)  # the upper bound holds, also for the coat-blind engine
            self.assertTrue(entry["expectation"]["asserted"] and entry["expectation"]["met"])
            self.assertTrue(entry["controls"]["coat-base-unattenuated"]["fails_gate"])
            self.assertEqual(check.verdict([entry], "red"), ("EXPECTED_RED_CONFIRMED", 0))
        wrong = self.scenario(key, ref.CONTROLS["coat-base-unattenuated"], "red", tag="-wrong")
        self.assertFalse(wrong["result"]["pass"])  # an engine that adds the coat on top of the full base gains energy
        self.assertEqual(check.verdict([wrong], "red")[1], check.EXIT_UNEXPECTED)

    # --- F15.3 (ticket 10): the sheen harness, same method as the clearcoat one (the "engine" images are reference
    # renders of the model the engine will have (sheen) or has today (sheen-ignored)).

    SHEEN_KEYS = ("sheen-grid-sun", "sheen-grid-env-resolve", "sheen-grid-env", "sheen-grid-env@probes",
                  "sheen-grid-point", "furnace-sheen@resolve")
    SHEEN_DECLARED_KEYS = ("sheen-grid-env-resolve", "sheen-grid-env@probes", "furnace-sheen@resolve")

    def sheen_engine(self, key, ignored):
        declared = key in self.SHEEN_DECLARED_KEYS
        model = ref.control_model("sheen-ignored", ref.APPENDIX_B) if ignored else ref.APPENDIX_B
        environment = None
        if declared:
            environment = ref.SHEEN_DECLARED_CONTROLS["declared-sheen-ignored"] if ignored else ref.DECLARED
        return model, environment

    def sheen_entry(self, key, ignored, tag):
        model, environment = self.sheen_engine(key, ignored)
        return self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + tag,
                             key=key)

    def test_sheen_keys_pass_with_an_engine_that_implements_the_sheen(self):
        for key in self.SHEEN_KEYS:
            entry = self.sheen_entry(key, False, "-sheen")
            self.assertTrue(entry["discriminating"], (key, json.dumps(entry["controls"], indent=1)))
            self.assertTrue(entry["gates_pass"], (key, json.dumps(check.summarize(entry["result"]))))
            self.assertFalse(entry.get("pending"), key)
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", check.EXIT_OK), key)

    def test_sheen_keys_fail_with_a_sheen_blind_engine(self):
        # The old engine (sheen ignored) fails the gate of every key (the furnace bound is an upper bound: it holds).
        for key in self.SHEEN_KEYS:
            entry = self.sheen_entry(key, True, "-blind")
            self.assertFalse(entry["gates_pass"], key)

    def test_sheen_controls_each_fail_their_gate(self):
        for key in self.SHEEN_KEYS:
            entry = self.sheen_entry(key, False, "-ctl")
            controls = entry["declared"]["controls"] if entry["kind"] == "furnace" else entry["controls"]
            if entry["kind"] == "furnace":
                controls = {**entry["controls"], **controls}
            wanted = set(check.SCENARIOS[key]["controls"]) | set(check.SCENARIOS[key].get("declared", []))
            self.assertEqual(set(controls), wanted, key)
            for name, verdict in controls.items():
                self.assertTrue(verdict["fails_gate"], (key, name, verdict))

    def test_every_direct_sheen_control_alone_fails_the_direct_gate_as_an_engine_image(self):
        # An engine that implemented one of the wrong models (Ashikhmin visibility, linear roughness, view-only
        # scaling, no scaling) must not pass the sheen gate.
        for control in ref.SHEEN_CONTROLS:
            if control == "sheen-ignored":
                continue
            entry = self.scenario("sheen-grid-sun", ref.control_model(control, ref.APPENDIX_B), "green",
                                  tag="-" + control, key="sheen-grid-sun")
            self.assertFalse(entry["result"]["pass"], control)

    def test_furnace_sheen_bound_holds_for_the_model_and_the_unscaled_base_breaks_it(self):
        key = "furnace-sheen"
        for ignored in (False, True):
            model, _ = self.sheen_engine(key, ignored)
            entry = self.scenario(key, model, "red", tag=f"-{ignored}")
            self.assertTrue(entry["result"]["pass"], (ignored, entry["result"]["max_sphere_mean"]))
            self.assertTrue(entry["expectation"]["asserted"] and entry["expectation"]["met"])
            self.assertTrue(entry["controls"]["sheen-unscaled-base"]["fails_gate"])
            self.assertEqual(check.verdict([entry], "red"), ("EXPECTED_RED_CONFIRMED", 0))
        wrong = self.scenario(key, ref.control_model("sheen-unscaled-base", ref.APPENDIX_B), "red", tag="-wrong")
        self.assertFalse(wrong["result"]["pass"])  # an engine that adds the sheen on top of the full base gains energy
        self.assertEqual(check.verdict([wrong], "red")[1], check.EXIT_UNEXPECTED)

    # --- F15.3 (ticket 11): the anisotropy harness, same method as the sheen one (the "engine" images are reference
    # renders of the model the engine will have (anisotropy) or has today (aniso-ignored)).

    ANISO_KEYS = ("aniso-grid-sun", "aniso-grid-env-resolve", "aniso-grid-env", "aniso-grid-point", "furnace-aniso@resolve",
                  "aniso-mirrored-sun")
    ANISO_DECLARED_KEYS = ("aniso-grid-env-resolve", "furnace-aniso@resolve")

    def aniso_engine(self, key, ignored):
        declared = key in self.ANISO_DECLARED_KEYS
        model = ref.control_model("aniso-ignored", ref.APPENDIX_B) if ignored else ref.APPENDIX_B
        environment = None
        if declared:
            environment = ref.ANISO_DECLARED_CONTROLS["declared-aniso-ignored"] if ignored else ref.DECLARED
        return model, environment

    def aniso_entry(self, key, ignored, tag):
        model, environment = self.aniso_engine(key, ignored)
        return self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + tag,
                             key=key)

    def test_aniso_keys_pass_with_an_engine_that_implements_the_anisotropy(self):
        for key in self.ANISO_KEYS:
            entry = self.aniso_entry(key, False, "-aniso")
            self.assertTrue(entry["discriminating"], (key, json.dumps(entry["controls"], indent=1)))
            self.assertTrue(entry["gates_pass"], (key, json.dumps(check.summarize(entry["result"]))))
            self.assertFalse(entry.get("pending"), key)
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", check.EXIT_OK), key)

    def test_aniso_keys_fail_with_an_anisotropy_blind_engine(self):
        # The old engine (anisotropy ignored) fails the gate of every key except the furnace bound, an upper bound that it holds.
        for key in self.ANISO_KEYS:
            entry = self.aniso_entry(key, True, "-blind")
            self.assertFalse(entry["gates_pass"], key)

    def test_aniso_controls_each_fail_their_gate(self):
        for key in self.ANISO_KEYS:
            entry = self.aniso_entry(key, False, "-ctl")
            controls = entry["declared"]["controls"] if entry["kind"] == "furnace" else entry["controls"]
            if entry["kind"] == "furnace":
                controls = {**entry["controls"], **controls}
            wanted = set(check.SCENARIOS[key]["controls"]) | set(check.SCENARIOS[key].get("declared", []))
            self.assertEqual(set(controls), wanted, key)
            for name, verdict in controls.items():
                self.assertTrue(verdict["fails_gate"], (key, name, verdict))

    def test_every_direct_aniso_control_alone_fails_the_direct_gate_as_an_engine_image(self):
        # An engine that implemented one of the wrong models must not pass the anisotropy gates (grid and mirrored fixture).
        for control in ref.ANISO_CONTROLS:
            if control == "aniso-ignored":
                continue
            for key in ("aniso-grid-sun", "aniso-mirrored-sun"):
                if control == "aniso-mirror-unaware" and key == "aniso-grid-sun":
                    continue  # no mirrored instance in the grid: the control is the spec there
                entry = self.scenario(key, ref.control_model(control, ref.APPENDIX_B), "green", tag="-" + control, key=key)
                self.assertFalse(entry["result"]["pass"], (key, control))

    def test_mirror_unaware_engine_is_the_spec_on_the_grid_and_wrong_only_on_mirrored_columns(self):
        grid = self.scenario("aniso-grid-sun", ref.control_model("aniso-mirror-unaware", ref.APPENDIX_B), "green",
                             tag="-mirror-grid", key="aniso-grid-sun")
        self.assertTrue(grid["result"]["pass"])
        mirrored = self.scenario("aniso-mirrored-sun", ref.control_model("aniso-mirror-unaware", ref.APPENDIX_B), "green",
                                 tag="-mirror", key="aniso-mirrored-sun")
        self.assertFalse(mirrored["result"]["pass"])

    def test_env_aniso_exact_gate_sees_every_wrong_model_the_direct_keys_see(self):
        # The per-sphere mean over the combed sphere sees the lobe orientation (the view azimuth against the lobe axes is not
        # uniform on a combed sphere), so the exact environment key uses the same five controls as the direct keys.
        for control in check.SCENARIOS["aniso-grid-env"]["controls"]:
            entry = self.scenario("aniso-grid-env", ref.control_model(control, ref.APPENDIX_B), "green", tag="-" + control,
                                  key="aniso-grid-env")
            self.assertEqual(entry["result"]["pass"], False, control)
        self.assertEqual(sorted(check.SCENARIOS["aniso-grid-env"]["controls"]), sorted(check.ANISO_DIRECT_CONTROLS))

    def test_furnace_aniso_bound_holds_for_the_model_and_the_energy_control_breaks_it(self):
        key = "furnace-aniso"
        # The literal Appendix B dielectric gains 1.2 % at roughness 0.3 (the specification says so, measured on the base
        # rows), so the engine emulation is the energy-conserving declared model (with and without the anisotropy fit).
        for ignored in (False, True):
            environment = ref.ANISO_DECLARED_CONTROLS["declared-aniso-ignored"] if ignored else ref.DECLARED
            entry = self.scenario(key, ref.APPENDIX_B, "red", environment=environment, tag=f"-{ignored}")
            self.assertTrue(entry["result"]["pass"], (ignored, entry["result"]["max_sphere_mean"]))
            self.assertTrue(entry["expectation"]["asserted"] and entry["expectation"]["met"])
            self.assertTrue(entry["controls"]["aniso-d-normalised-by-ab"]["fails_gate"])
            self.assertEqual(check.verdict([entry], "red"), ("EXPECTED_RED_CONFIRMED", 0))
        wrong = self.scenario(key, ref.control_model("aniso-d-normalised-by-ab", ref.APPENDIX_B), "red", tag="-wrong")
        self.assertFalse(wrong["result"]["pass"])  # an engine whose D is normalised by alpha_b^2 gains energy
        self.assertEqual(check.verdict([wrong], "red")[1], check.EXIT_UNEXPECTED)

    # --- F15.3 (ticket 12): the iridescence harness, same method as the anisotropy one.

    IRID_KEYS = ("iridescence-grid-sun", "iridescence-grid-env-resolve", "iridescence-grid-env", "iridescence-grid-point",
                 "furnace-iridescence@resolve")
    IRID_DECLARED_KEYS = ("iridescence-grid-env-resolve", "furnace-iridescence@resolve")

    def irid_entry(self, key, ignored, tag):
        declared = key in self.IRID_DECLARED_KEYS
        model = ref.control_model("iridescence-ignored", ref.APPENDIX_B) if ignored else ref.APPENDIX_B
        environment = None
        if declared:
            environment = ref.IRID_DECLARED_CONTROLS["declared-iridescence-ignored"] if ignored else ref.DECLARED
        return self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + tag, key=key)

    def test_iridescence_keys_pass_with_an_engine_that_implements_the_film(self):
        for key in self.IRID_KEYS:
            entry = self.irid_entry(key, False, "-film")
            self.assertTrue(entry["discriminating"], (key, json.dumps(entry["controls"], indent=1)))
            self.assertTrue(entry["gates_pass"], (key, json.dumps(check.summarize(entry["result"]))))
            self.assertFalse(entry.get("pending"), key)
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", check.EXIT_OK), key)

    def test_iridescence_keys_fail_with_a_film_blind_engine(self):
        # The old engine (film ignored) fails the gate of every key listed here (the furnace bound of the transport is not).
        for key in self.IRID_KEYS:
            entry = self.irid_entry(key, True, "-blind")
            self.assertFalse(entry["gates_pass"], key)

    def test_iridescence_controls_each_fail_their_gate(self):
        for key in self.IRID_KEYS:
            entry = self.irid_entry(key, False, "-ctl")
            controls = entry["declared"]["controls"] if entry["kind"] == "furnace" else entry["controls"]
            if entry["kind"] == "furnace":
                controls = {**entry["controls"], **controls}
            wanted = set(check.SCENARIOS[key]["controls"]) | set(check.SCENARIOS[key].get("declared", []))
            self.assertEqual(set(controls), wanted, key)
            for name, verdict in controls.items():
                self.assertTrue(verdict["fails_gate"], (key, name, verdict))

    def test_every_iridescence_control_alone_fails_the_direct_gate_as_an_engine_image(self):
        for control in ref.IRID_CONTROLS:
            if control == "iridescence-ignored":
                continue
            entry = self.scenario("iridescence-grid-sun", ref.control_model(control, ref.APPENDIX_B), "green",
                                  tag="-" + control, key="iridescence-grid-sun")
            self.assertFalse(entry["result"]["pass"], control)

    def test_env_iridescence_exact_gate_sees_every_wrong_model_the_direct_keys_see(self):
        for control in check.SCENARIOS["iridescence-grid-env"]["controls"]:
            entry = self.scenario("iridescence-grid-env", ref.control_model(control, ref.APPENDIX_B), "green",
                                  tag="-" + control, key="iridescence-grid-env")
            self.assertEqual(entry["result"]["pass"], False, control)

    def test_furnace_iridescence_bound_holds_for_the_model_and_the_energy_control_breaks_it(self):
        key = "furnace-iridescence"
        # As for the other furnaces the engine emulation is the energy-conserving declared model (the literal Appendix B
        # dielectric gains ~1.2 % at roughness 0.3, and the exact film gains up to ~3 % more at grazing angles).
        for ignored in (False, True):
            environment = ref.IRID_DECLARED_CONTROLS["declared-iridescence-ignored"] if ignored else ref.DECLARED
            entry = self.scenario(key, ref.APPENDIX_B, "red", environment=environment, tag=f"-{ignored}")
            self.assertTrue(entry["result"]["pass"], (ignored, entry["result"]["max_sphere_mean"]))
            self.assertTrue(entry["expectation"]["asserted"] and entry["expectation"]["met"])
            self.assertTrue(entry["controls"]["iridescence-diffuse-unweighted"]["fails_gate"])
            self.assertEqual(check.verdict([entry], "red"), ("EXPECTED_RED_CONFIRMED", 0))
        wrong = self.scenario(key, ref.control_model("iridescence-diffuse-unweighted", ref.APPENDIX_B), "red", tag="-wrong")
        self.assertFalse(wrong["result"]["pass"])  # an engine that does not scale the diffuse by 1 - max F_irid gains energy
        self.assertEqual(check.verdict([wrong], "red")[1], check.EXIT_UNEXPECTED)

    # --- F15.3 (ticket 16): the thin subsurface harness, same method as the iridescence one.

    THIN_KEYS = ("subsurface-thin-sun", "subsurface-thin-env-resolve", "subsurface-thin-env", "subsurface-thin-point",
                 "furnace-thin-subsurface@resolve")
    THIN_DECLARED_KEYS = ("subsurface-thin-env-resolve", "furnace-thin-subsurface@resolve")

    def thin_entry(self, key, ignored, tag):
        declared = key in self.THIN_DECLARED_KEYS
        model = ref.control_model("transmission-ignored", ref.APPENDIX_B) if ignored else ref.APPENDIX_B
        environment = None
        if declared:
            environment = ref.THIN_DECLARED_CONTROLS["declared-transmission-ignored"] if ignored else ref.DECLARED
        return self.scenario(check.bench_scenario(key), model, "green", environment=environment, tag="-" + key + tag, key=key)

    def test_thin_subsurface_keys_pass_with_an_engine_that_implements_the_lobe(self):
        for key in self.THIN_KEYS:
            entry = self.thin_entry(key, False, "-lobe")
            self.assertTrue(entry["discriminating"], (key, json.dumps(entry["controls"], indent=1)))
            self.assertTrue(entry["gates_pass"], (key, json.dumps(check.summarize(entry["result"]))))
            self.assertFalse(entry.get("pending"), key)
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", check.EXIT_OK), key)

    def test_thin_subsurface_keys_fail_with_a_transmission_blind_engine(self):
        # The old engine (transmission ignored) fails the gate of every key listed here (the furnace bound of the transport is not).
        for key in self.THIN_KEYS:
            entry = self.thin_entry(key, True, "-blind")
            self.assertFalse(entry["gates_pass"], key)

    def test_thin_subsurface_controls_each_fail_their_gate(self):
        for key in self.THIN_KEYS:
            entry = self.thin_entry(key, False, "-ctl")
            controls = entry["declared"]["controls"] if entry["kind"] == "furnace" else entry["controls"]
            if entry["kind"] == "furnace":
                controls = {**entry["controls"], **controls}
            wanted = set(check.SCENARIOS[key]["controls"]) | set(check.SCENARIOS[key].get("declared", []))
            self.assertEqual(set(controls), wanted, key)
            for name, verdict in controls.items():
                self.assertTrue(verdict["fails_gate"], (key, name, verdict))

    def test_every_thin_subsurface_control_alone_fails_the_direct_gates_it_is_listed_for_as_an_engine_image(self):
        for key, controls in (("subsurface-thin-sun", check.THIN_SUN_CONTROLS), ("subsurface-thin-env", check.THIN_SUN_CONTROLS),
                              ("subsurface-thin-point", check.THIN_POINT_CONTROLS)):
            for control in controls:
                entry = self.scenario(check.bench_scenario(key), ref.control_model(control, ref.APPENDIX_B), "green",
                                      tag="-" + key + "-" + control, key=key)
                self.assertFalse(entry["result"]["pass"], (key, control))

    def test_every_declared_thin_subsurface_control_alone_fails_the_declared_gate_as_an_engine_image(self):
        for control, variant in ref.THIN_DECLARED_CONTROLS.items():
            entry = self.scenario("subsurface-thin-env-resolve", ref.APPENDIX_B, "green", environment=variant, tag="-" + control,
                                  key="subsurface-thin-env-resolve")
            self.assertFalse(entry["result"]["pass"], control)

    def test_furnace_thin_subsurface_bound_holds_for_the_declared_model_and_the_energy_control_breaks_it(self):
        key = "furnace-thin-subsurface"
        # As for the other furnaces the engine emulation is the energy-conserving declared model: the transmitted weight never
        # exceeds the diffuse weight it replaces, so the white albedo stays at the base model's (<= 1).
        for ignored in (False, True):
            environment = ref.THIN_DECLARED_CONTROLS["declared-transmission-ignored"] if ignored else ref.DECLARED
            entry = self.scenario(key, ref.APPENDIX_B, "red", environment=environment, tag=f"-{ignored}")
            self.assertTrue(entry["result"]["pass"], (ignored, entry["result"]["max_sphere_mean"]))
            self.assertTrue(entry["expectation"]["asserted"] and entry["expectation"]["met"])
            self.assertTrue(entry["controls"]["transmission-double-counted"]["fails_gate"])
            self.assertEqual(check.verdict([entry], "red"), ("EXPECTED_RED_CONFIRMED", 0))
        wrong = self.scenario(key, ref.control_model("transmission-double-counted", ref.APPENDIX_B), "red", tag="-wrong")
        self.assertFalse(wrong["result"]["pass"])  # an engine that does not reduce the diffuse by w gains energy
        self.assertEqual(check.verdict([wrong], "red")[1], check.EXIT_UNEXPECTED)

    def test_cached_reference_is_reused_only_for_the_same_snapshot(self):
        entry = self.scenario("base-grid-env-resolve", ref.APPENDIX_B, "green", environment=ref.DECLARED, tag="-cache")
        folder = pathlib.Path(self.tmp.name) / "base-grid-env-resolve-green-cache"
        literal, declared = check.scenario_controls("base-grid-env-resolve")
        usable = lambda: check.cached_reference_usable(folder / "reference", folder / "snapshot", literal, declared)  # noqa: E731
        self.assertTrue(usable())
        self.assertEqual(json.loads((folder / "reference" / "reference.json").read_text())["snapshot_sha256"],
                         entry["snapshot_sha256"])
        # A re-evaluation reuses it: the reference tool is not called.
        calls = []
        original = ref.main
        ref.main = lambda *a, **k: calls.append(a) or 0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                check.evaluate_scenario(np, "base-grid-env-resolve", folder, "green")
            self.assertEqual(calls, [])
            # Change the snapshot: the recorded SHA-256 no longer matches, the reference is not reused.
            scene = folder / "snapshot" / "scene.json"
            data = json.loads(scene.read_text())
            data["sky"] = [0.25] * 3
            scene.write_text(json.dumps(data))
            self.assertFalse(usable())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                check.evaluate_scenario(np, "base-grid-env-resolve", folder, "green")
            self.assertEqual(len(calls), 1)  # re-rendered (the stub returned 0 and the old files were read)
        finally:
            ref.main = original
        # A missing control file or an unrequested control also defeats the cache.
        data["sky"] = [0.5] * 3
        scene.write_text(json.dumps(data))
        self.assertTrue(usable())
        self.assertFalse(check.cached_reference_usable(folder / "reference", folder / "snapshot", ["fresnel-off"], declared))
        (folder / "reference" / "control-declared-karis-fit.pfm").unlink()
        self.assertFalse(usable())


class VolumetricSubsurfaceRunner(unittest.TestCase):
    """Ticket 17: the gate kinds `subsurface`, `radius-monotonic`, `equal`, `subsurface-report` and the pending machinery of the keys, on
    synthetic discs (numpy only): lit half, a bleed into the unlit half that grows with the radius.  The ground truth of the real keys is
    the Mitsuba random walk of tools/f15_subsurface_reference.py (tested there)."""

    RADII = [0.02, 0.04, 0.08, 0.16, 0.24]
    SERIES = ["skin", "jade"]
    SIZE = 24

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def layout(self):
        """masks, spheres of 2 series x 5 radii: discs on a grid, `lit` = the left 70 % of every disc."""
        r, n = self.SIZE, 10
        height, width = 3 * r, (n + 1) * r
        sphere = np.full((height, width), -1, dtype=np.int32)
        lit = np.zeros((height, width), dtype=bool)
        yy, xx = np.mgrid[0:height, 0:width]
        spheres = []
        for k in range(n):
            cx, cy = (k + 1) * r - r // 2, height // 2
            inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (r * 0.45) ** 2
            sphere[inside] = k
            lit |= inside & (xx < cx + 0.2 * r * 0.45)
            series, radius = self.SERIES[k // 5], self.RADII[k % 5]
            spheres.append({"instance": k, "slot": k, "base": [0.8] * 3, "metallic": 0.0, "roughness": 0.7, "weight": 1.0,
                            "radius": radius, "series": series, "group": "weight 1"})
        roi = sphere >= 0
        inner = roi.copy()
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            inner &= np.roll(sphere, shift, axis=axis) == sphere
        return {"roi": inner, "sphere": sphere, "narrow": np.zeros_like(inner), "lit": lit}, spheres

    def image(self, masks, lit_value, bleed):
        """Luminance image: lit pixels `lit_value`, unlit ones the callable bleed(piece index)."""
        image = np.zeros(masks["roi"].shape + (3,), dtype=np.float32)
        for k in range(10):
            piece = masks["sphere"] == k
            image[piece & masks["lit"]] = lit_value
            image[piece & ~masks["lit"]] = bleed(k)
        return image

    def truth(self, masks):
        return self.image(masks, 0.6, lambda k: 0.04 * (k % 5 + 1))  # bleed 0.04 ... 0.20: rises with the radius

    def off(self, masks):
        return self.image(masks, 0.7, lambda k: 0.0)

    def test_bound_gate_passes_the_truth_and_its_noise_and_fails_the_base_model_and_every_control(self):
        masks, spheres = self.layout()
        truth = self.truth(masks)
        self.assertTrue(check.evaluate(np, "subsurface", truth, truth, masks, spheres)["pass"])
        noisy = truth * np.float32(1.04)  # a uniform 4 % error: inside the bound
        self.assertTrue(check.evaluate(np, "subsurface", noisy, truth, masks, spheres)["pass"])
        off = check.evaluate(np, "subsurface", self.off(masks), truth, masks, spheres)
        self.assertFalse(off["pass"])
        self.assertEqual(off["pieces_failing"] if "pieces_failing" in off else None, None)  # summarize adds it, evaluate does not
        self.assertTrue(all(not piece["pass"] for piece in off["pieces"]))
        self.assertGreater(off["worst_mean"], 0.2)
        # One piece at 12 % mean error fails the whole gate (every piece must pass); the p95 bound alone also fails.
        bad = truth.copy()
        piece = masks["roi"] & (masks["sphere"] == 3)
        bad[piece] = truth[piece] * np.float32(1.12)
        result = check.evaluate(np, "subsurface", bad, truth, masks, spheres)
        self.assertFalse(result["pass"])
        self.assertEqual(check.summarize(result)["pieces_failing"], [3])
        spiky = truth.copy()
        pixels = np.flatnonzero(piece.reshape(-1))
        spiky.reshape(-1, 3)[pixels[: len(pixels) // 10]] *= 1.5  # 10 % of the pixels 50 % off: mean 5 %, p95 50 %
        spiky_result = check.evaluate(np, "subsurface", spiky, truth, masks, spheres)
        self.assertFalse(spiky_result["pass"])
        self.assertLess(spiky_result["pieces"][3]["mean"], 0.10)
        self.assertGreater(spiky_result["pieces"][3]["p95"], 0.25)

    def test_the_absolute_floor_protects_dark_pixels_but_not_a_missing_bleed(self):
        masks, spheres = self.layout()
        truth = self.truth(masks)
        # The error floor is 5 % of the piece's mean luminance: 0.001 on a 0.02 bleed (a quarter of the floor 0.0..) is harmless...
        nearly = truth.copy()
        nearly[(masks["sphere"] == 0) & ~masks["lit"]] += 0.001
        self.assertTrue(check.evaluate(np, "subsurface", nearly, truth, masks, spheres)["pass"])
        # ... but a bleed of 0 against 0.2 is an error of 1 on those pixels.
        gone = truth.copy()
        gone[(masks["sphere"] == 4) & ~masks["lit"]] = 0.0
        self.assertFalse(check.evaluate(np, "subsurface", gone, truth, masks, spheres)["pass"])

    def test_report_kind_measures_the_same_numbers_and_never_fails(self):
        masks, spheres = self.layout()
        truth = self.truth(masks)
        gated = check.evaluate(np, "subsurface", self.off(masks), truth, masks, spheres)
        report = check.evaluate(np, "subsurface-report", self.off(masks), truth, masks, spheres)
        self.assertTrue(report["pass"] and not report["gated"] and not report["bound_met"])
        self.assertEqual([p["mean"] for p in report["pieces"]], [p["mean"] for p in gated["pieces"]])

    def test_terminator_statistic_and_the_monotonic_check(self):
        masks, spheres = self.layout()
        truth = self.truth(masks)
        bleed = [check.terminator_bleed(np, truth, masks, k) for k in range(5)]
        np.testing.assert_allclose(bleed, [0.04 * (k + 1) / 0.6 for k in range(5)], rtol=1e-5)  # unlit mean / lit mean
        self.assertEqual(check.terminator_bleed(np, self.off(masks), masks, 0), 0.0)
        self.assertTrue(check.evaluate(np, "radius-monotonic", truth, truth, masks, spheres)["pass"])
        # An engine whose bleed rises like the truth's but smaller still passes (the check is about the response, not the magnitude).
        weaker = self.image(masks, 0.6, lambda k: 0.02 * (k % 5 + 1))
        self.assertTrue(check.evaluate(np, "radius-monotonic", weaker, truth, masks, spheres)["pass"])
        # Flat (radius ignored) against the rising truth fails; flat against flat passes (the pending twin).
        flat = self.image(masks, 0.6, lambda k: 0.1)
        self.assertFalse(check.evaluate(np, "radius-monotonic", flat, truth, masks, spheres)["pass"])
        self.assertTrue(check.evaluate(np, "radius-monotonic", self.off(masks), self.off(masks), masks, spheres)["pass"])
        # A reversal beyond the tolerance fails even where the truth is flat.
        reversed_ = self.image(masks, 0.6, lambda k: 0.04 * (5 - k % 5))
        self.assertFalse(check.evaluate(np, "radius-monotonic", reversed_, truth, masks, spheres)["pass"])
        # A series whose ground truth does not rise at every step is reported as a non-discriminating fixture.
        self.assertTrue(check.evaluate(np, "radius-monotonic", truth, truth, masks, spheres)["reference_monotonic"])
        self.assertFalse(check.evaluate(np, "radius-monotonic", flat, flat, masks, spheres)["reference_monotonic"])
        result = check.evaluate(np, "radius-monotonic", flat, truth, masks, spheres)
        self.assertEqual(sorted(check.summarize(result)["series_failing"]), ["jade", "skin"])
        self.assertEqual(len(result["series"][0]["steps"]), 4)

    def test_equal_kind_accepts_floating_point_noise_and_rejects_diffusion_or_an_empty_image(self):
        masks, spheres = self.layout()
        pair = self.truth(masks)
        self.assertTrue(check.evaluate(np, "equal", pair, pair, masks, spheres)["pass"])
        self.assertTrue(check.evaluate(np, "equal", pair * np.float32(1.0 + 5e-4), pair, masks, spheres)["pass"])
        self.assertFalse(check.evaluate(np, "equal", pair * np.float32(1.01), pair, masks, spheres)["pass"])
        self.assertFalse(check.evaluate(np, "equal", self.off(masks), pair, masks, spheres)["pass"])
        black = np.zeros_like(pair)
        self.assertFalse(check.evaluate(np, "equal", black, black, masks, spheres)["pass"])  # equal but empty

    # --- the keys ---------------------------------------------------------------------------------------------------------------

    def write_folder(self, key, engine, reference, controls, masks, spheres):
        from f12_reference import scene_digest
        folder = pathlib.Path(self.tmp.name) / key
        (folder / "snapshot").mkdir(parents=True)
        (folder / "snapshot" / "scene.json").write_text("{}")
        out = folder / "reference"
        out.mkdir()
        write_pfm(folder / "engine.pfm", engine)
        write_pfm(out / "reference.pfm", reference)
        np.savez_compressed(out / "masks.npz", **masks)
        meta = {"schema": 1, "terms": ["sun-delta-mitsuba-volpath"], "maximum_radiance": 1.0, "quadrature_mask_pixels": 0,
                "roi_pixels": int(masks["roi"].sum()), "spheres": spheres, "visibility": "none", "controls": {},
                "snapshot_sha256": scene_digest(folder / "snapshot")}
        for name, image in controls.items():
            write_pfm(out / f"control-{name}.pfm", image)
            meta["controls"][name] = {"model_variant": name, "file": f"control-{name}.pfm"}
        (out / "reference.json").write_text(json.dumps(meta))
        return folder

    def controls_of(self, masks):
        return {"subsurface-off": self.off(masks),
                "diffusion-off": self.image(masks, 0.5, lambda k: 0.0),
                "radius-ignored": self.image(masks, 0.6, lambda k: 0.12)}

    def test_the_bound_and_sweep_keys_are_red_for_the_right_reason_until_the_engine_lands(self):
        masks, spheres = self.layout()
        truth = self.truth(masks)
        controls = self.controls_of(masks)
        for key in ("subsurface-volumetric-sun", "subsurface-volumetric-radius"):
            spec = check.SCENARIOS[key]
            self.assertEqual((spec["pending"], spec["pending_matches"], spec["reference"]), ("ticket 17", "subsurface-off", "subsurface"))
            # Today's engine = the subsurface-off control: every gate fails, the twin passes, every control discriminates -> GREEN.
            today = self.write_folder(key + "-today", controls["subsurface-off"], truth, controls, masks, spheres)
            entry = check.evaluate_scenario(np, key, today, "green")
            self.assertFalse(entry["gates_pass"], key)
            self.assertTrue(entry["pending_matches_met"], key)
            self.assertTrue(entry["discriminating"], key)
            self.assertEqual(entry["expectation"]["matches_control"], "subsurface-off")
            self.assertEqual(check.verdict([entry], "green"), ("GREEN", 0), key)
            self.assertTrue(all(v["fails_gate"] for v in entry["controls"].values()), key)
            # The engine that implements the mode passes the gate: a pending key must FAIL, so the verdict turns red (the runner keys
            # lose "pending" in the ticket that implements it).
            done = self.write_folder(key + "-done", truth, truth, controls, masks, spheres)
            entry = check.evaluate_scenario(np, key, done, "green")
            self.assertTrue(entry["gates_pass"], key)
            self.assertEqual(check.verdict([entry], "green")[1], check.EXIT_UNEXPECTED, key)
            # An engine that is neither the base model nor the truth is not "red for the right reason".
            other = self.write_folder(key + "-other", controls["radius-ignored"], truth, controls, masks, spheres)
            entry = check.evaluate_scenario(np, key, other, "green")
            self.assertFalse(entry["pending_matches_met"], key)
            self.assertEqual(check.verdict([entry], "green")[1], check.EXIT_UNEXPECTED, key)

    def test_statistic_match_twin_distinguishes_the_base_model_from_a_flat_bleed(self):
        masks, spheres = self.layout()
        off = self.off(masks)
        self.assertTrue(check.evaluate(np, "radius-statistic-match", off, off, masks, spheres)["pass"])
        self.assertFalse(check.evaluate(np, "radius-statistic-match", self.image(masks, 0.6, lambda k: 0.12), off, masks, spheres)["pass"])
        self.assertEqual(check.SCENARIOS["subsurface-volumetric-radius"]["twin_kind"], "radius-statistic-match")

    def test_a_sweep_fixture_whose_truth_does_not_rise_is_non_discriminating(self):
        masks, spheres = self.layout()
        flat = self.image(masks, 0.6, lambda k: 0.1)
        controls = self.controls_of(masks)
        folder = self.write_folder("flat-truth", controls["subsurface-off"], flat, controls, masks, spheres)
        entry = check.evaluate_scenario(np, "subsurface-volumetric-radius", folder, "green")
        self.assertFalse(entry["discriminating"])
        self.assertEqual(check.verdict([entry], "green")[1], check.EXIT_NON_DISCRIMINATING)

    def test_subpixel_pair_key_compares_two_engine_captures(self):
        masks, spheres = self.layout()
        pair_image = self.truth(masks)
        spec = check.SCENARIOS["subsurface-volumetric-subpixel"]
        self.assertEqual(spec["pair"], "subsurface-volumetric-subpixel-off")
        self.assertTrue(check.SCENARIOS[spec["pair"]]["capture_only"])
        self.assertNotIn("pending", spec)
        controls = {"subsurface-leak": self.image(masks, 0.6, lambda k: 0.05)}
        for name, engine, expect in (("same", pair_image, True), ("fp", pair_image * np.float32(1.0 + 1e-4), True),
                                     ("diffused", self.image(masks, 0.6, lambda k: 0.06), False)):
            root = pathlib.Path(self.tmp.name) / f"pair-{name}"
            root.mkdir()
            self.write_folder_in(root, "subsurface-volumetric-subpixel", engine, pair_image, controls, masks, spheres)
            (root / "subsurface-volumetric-subpixel-off").mkdir()
            write_pfm(root / "subsurface-volumetric-subpixel-off" / "engine.pfm", pair_image)
            entry = check.evaluate_scenario(np, "subsurface-volumetric-subpixel", root / "subsurface-volumetric-subpixel", "green")
            self.assertEqual(entry["gates_pass"], expect, name)
            self.assertTrue(entry["discriminating"], name)  # the leak control (a different image) fails the equality gate
            self.assertTrue(entry["controls"]["subsurface-leak"]["fails_gate"], name)

    def write_folder_in(self, root, key, engine, reference, controls, masks, spheres):
        from f12_reference import scene_digest
        folder = root / key
        (folder / "snapshot").mkdir(parents=True)
        (folder / "snapshot" / "scene.json").write_text("{}")
        out = folder / "reference"
        out.mkdir()
        write_pfm(folder / "engine.pfm", engine)
        write_pfm(out / "reference.pfm", reference)
        np.savez_compressed(out / "masks.npz", **masks)
        meta = {"schema": 1, "terms": ["sun-delta-mitsuba-volpath"], "maximum_radiance": 1.0, "quadrature_mask_pixels": 0,
                "roi_pixels": int(masks["roi"].sum()), "spheres": spheres, "visibility": "none", "controls": {},
                "snapshot_sha256": scene_digest(folder / "snapshot")}
        for name, image in controls.items():
            write_pfm(out / f"control-{name}.pfm", image)
            meta["controls"][name] = {"model_variant": name, "file": f"control-{name}.pfm"}
        (out / "reference.json").write_text(json.dumps(meta))

    def test_backlit_report_key_is_never_gated_or_pending(self):
        masks, spheres = self.layout()
        spec = check.SCENARIOS["subsurface-volumetric-backlit"]
        self.assertEqual(spec["kind"], "subsurface-report")
        self.assertNotIn("pending", spec)
        self.assertEqual(spec["controls"], [])
        truth = self.truth(masks)
        folder = self.write_folder("backlit", self.off(masks), truth, {}, masks, spheres)
        entry = check.evaluate_scenario(np, "subsurface-volumetric-backlit", folder, "green")
        self.assertTrue(entry["gates_pass"])
        self.assertFalse(entry["result"]["bound_met"])  # measured: the base model is far from the thick back-lit truth
        self.assertEqual(check.verdict([entry], "green"), ("GREEN", 0))

    def test_plan_lists_the_pair_partner_and_the_subsurface_reference_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                code = check.main(["--out", tmp, "--binary", "build/release/phosphor", "--only", "subsurface-volumetric-subpixel"])
            self.assertEqual(code, 0)
            plan = json.loads((pathlib.Path(tmp) / "plan.json").read_text())
            self.assertEqual(sorted(plan["commands"]), ["subsurface-volumetric-subpixel", "subsurface-volumetric-subpixel-off"])
            command = " ".join(plan["commands"]["subsurface-volumetric-subpixel"]["reference"])
            self.assertIn("f15_subsurface_reference.py", command)
            self.assertIn("subsurface-leak", command)
            self.assertIn("--reflections", plan["commands"]["subsurface-volumetric-sun" if False else "subsurface-volumetric-subpixel"]["capture"])

    def test_every_volumetric_key_and_its_declared_controls(self):
        keys = [k for k in check.SCENARIOS if k.startswith("subsurface-volumetric")]
        self.assertEqual(sorted(keys), ["subsurface-volumetric-backlit", "subsurface-volumetric-radius", "subsurface-volumetric-subpixel",
                                        "subsurface-volumetric-subpixel-off", "subsurface-volumetric-sun"])
        import f15_subsurface_reference as tool
        for key in keys:
            spec = check.SCENARIOS[key]
            for control in spec["controls"]:
                self.assertIn(control, tool.CONTROLS, key)
            self.assertEqual(spec.get("reflections"), "off", key)
        self.assertEqual(check.SCENARIOS["subsurface-volumetric-sun"]["kind"], "subsurface")
        self.assertEqual(check.SCENARIOS["subsurface-volumetric-radius"]["kind"], "radius-monotonic")
        self.assertEqual(check.THRESHOLDS["subsurface"]["mean"], 0.10)
        self.assertEqual(check.THRESHOLDS["subsurface"]["p95"], 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
