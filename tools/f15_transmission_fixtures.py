#!/usr/bin/env python3
"""Snapshot builders that mirror the transmission bench (src/testbench/transmission_validation.cpp) for the Python tests.

The numbers (positions, sizes, factors) are the ones tests/test_transmission_validation.cpp pins.  `write_snapshot` writes
the files `exportOfflineReference` would (scene.json, mesh_N.ply with the mesh-version-2 vertex layout) so that
tools/f15_transmission_reference.py and tools/f15_check.py can run without a GPU.  Stdlib + numpy only.
"""
from __future__ import annotations

import json
import math
import pathlib
import struct

INVALID = 0xFFFFFFFF
FEATURE_SHIFT = 8  # MATERIAL_FEATURE_SHIFT
TINT = [0.9, 0.55, 0.25]
SPHERE_RADIUS = 0.4


def bits(v):
    return int.from_bytes(struct.pack("<f", float(v)), "little")


def sphere_mesh(radius=SPHERE_RADIUS, slices=64, stacks=32):
    verts, faces = [], []
    for st in range(stacks + 1):
        theta = math.pi * st / stacks
        for sl in range(slices + 1):
            phi = 2 * math.pi * sl / slices
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            verts.append((*(c * radius for c in n), *n, sl / slices, st / stacks, 1, 0, 0, -1))
    for st in range(stacks):
        for sl in range(slices):
            a = st * (slices + 1) + sl
            b = a + slices + 1
            faces += [(a, a + 1, b), (a + 1, b + 1, b)]
    return verts, faces


def cube_mesh(h=0.5):
    faces_def = [((1, 0, 0), [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)]),
                 ((-1, 0, 0), [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)]),
                 ((0, 1, 0), [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)]),
                 ((0, -1, 0), [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),
                 ((0, 0, 1), [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),
                 ((0, 0, -1), [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)])]
    verts, faces = [], []
    for normal, corners in faces_def:
        base = len(verts)
        for k, c in enumerate(corners):
            verts.append((*c, *normal, k % 2, k // 2, 1, 0, 0, 1))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return verts, faces


def plane_mesh():
    """The unit XZ plane (normal +Y), counter-clockwise about +Y, like ProceduralMeshes::generatePlane(1, 1, 1, 1)."""
    verts = [(-0.5, 0, -0.5, 0, 1, 0, 0, 0, 1, 0, 0, 1), (0.5, 0, -0.5, 0, 1, 0, 1, 0, 1, 0, 0, 1),
             (-0.5, 0, 0.5, 0, 1, 0, 0, 1, 1, 0, 0, 1), (0.5, 0, 0.5, 0, 1, 0, 1, 1, 1, 0, 0, 1)]
    return verts, [(0, 2, 1), (1, 2, 3)]


def write_ply(path, mesh):
    verts, faces = mesh
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\n"
                 "property float nx\nproperty float ny\nproperty float nz\nproperty float s\nproperty float t\n"
                 "property float tangent_x\nproperty float tangent_y\nproperty float tangent_z\nproperty float handedness_x\n"
                 "element face %d\nproperty list uchar uint vertex_indices\nend_header\n" % (len(verts), len(faces))).encode())
        for v in verts:
            f.write(struct.pack("<12f", *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


def rotation_matrix(yaw_deg=0.0, pitch_deg=0.0, face_plus_z=False):
    """yaw about Y, then pitch about X (the bench's orientation); a pane additionally turns +Y to +Z (+90 about X)."""
    def rot_y(a):
        c, s = math.cos(a), math.sin(a)
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

    def rot_x(a):
        c, s = math.cos(a), math.sin(a)
        return [[1, 0, 0], [0, c, -s], [0, s, c]]

    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    m = mul(rot_y(math.radians(yaw_deg)), rot_x(math.radians(pitch_deg)))
    if face_plus_z:
        m = mul(m, rot_x(math.pi / 2))
    return m


def world_words(position, scale, rotation):
    """Column-major 4x4 words: translation * rotation * scale."""
    cols = []
    for j in range(3):
        cols.append([rotation[i][j] * scale[j] for i in range(3)] + [0.0])
    cols.append(list(position) + [1.0])
    return [v for col in cols for v in col]


def glass(shape="sphere", center=(0, 1, 0), scale=(1, 1, 1), yaw=0.0, pitch=0.0, base=(1, 1, 1), roughness=0.0,
          transmission=1.0, ior=1.5, thickness=0.8, attenuation_color=(1, 1, 1), attenuation_distance=math.inf,
          double_sided=False):
    return dict(shape=shape, center=center, scale=scale, yaw=yaw, pitch=pitch, base=base, roughness=roughness,
                transmission=transmission, ior=ior, thickness=thickness, attenuation_color=attenuation_color,
                attenuation_distance=attenuation_distance, double_sided=double_sided)


def checker_cells(columns=24, rows=16, cell=0.4, origin=(-4.8, -1.2), z=-3.0):
    return [(origin[0] + (c + 0.5) * cell, origin[1] + (r + 0.5) * cell, z, cell, cell)
            for r in range(rows) for c in range(columns) if (r + c) % 2 == 0]


def write_snapshot(folder, objects, emitters, sky=0.5, emission=2.0, width=320, height=180, camera_target=(0, 1, 0),
                   distance=5.5):
    """scene.json + PLYs of the glass `objects` (dicts of `glass()`) and emissive rectangles `emitters`
    ((cx, cy, z, width, height), facing +Z, radiance `emission`)."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_ply(folder / "mesh_0.ply", sphere_mesh())
    write_ply(folder / "mesh_1.ply", cube_mesh())
    write_ply(folder / "mesh_2.ply", plane_mesh())
    mesh_of = {"sphere": 0, "cube": 1, "pane": 2}
    materials, records, instances = [], [], []
    for g in objects:
        feature_bits = (32 if g["transmission"] > 0 else 0) | (64 if g["thickness"] > 0 else 0) | (1 if g["ior"] != 1.5 else 0)
        record0 = [bits(g["ior"]), bits(1.0), bits(1), bits(1), bits(1), INVALID, INVALID] + [0] * 13
        record1 = ([bits(0), bits(0), INVALID, bits(0), bits(1.3), bits(100), bits(400), INVALID, INVALID,
                    bits(g["transmission"]), INVALID, bits(g["thickness"]), INVALID] +
                   [bits(c) for c in g["attenuation_color"]] + [bits(g["attenuation_distance"]), 0, 0, 0])
        materials.append({"base": [*g["base"], 1], "emissive": [0, 0, 0], "metallic": 0.0, "roughness": g["roughness"],
                          "normal_scale": 1, "occlusion_strength": 1, "alpha_cutoff": 0,
                          "flags": (feature_bits << FEATURE_SHIFT) | (1 if g["double_sided"] else 0), "textures": [INVALID] * 5,
                          "feature_bits": feature_bits, "extension_index": len(records)})
        records += [record0, record1, [0] * 20, [0] * 20]
        rotation = rotation_matrix(g["yaw"], g["pitch"], g["shape"] == "pane")
        instances.append({"slot": len(instances), "generation": 1, "flags": 3, "mesh": mesh_of[g["shape"]],
                          "material": len(materials) - 1, "world": world_words(g["center"], g["scale"], rotation)})
    emitter_material = len(materials)
    if emitters:
        materials.append({"base": [0, 0, 0, 1], "emissive": [emission] * 3, "metallic": 0.0, "roughness": 1.0, "normal_scale": 1,
                          "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0, "textures": [INVALID] * 5, "feature_bits": 0,
                          "extension_index": -1})
    rotation = rotation_matrix(0, 0, True)
    for cx, cy, z, w, h in emitters:
        instances.append({"slot": len(instances), "generation": 1, "flags": 3, "mesh": 2, "material": emitter_material,
                          "world": world_words((cx, cy, z), (w, 1.0, h), rotation)})
    camera = {"position": [camera_target[0], camera_target[1], camera_target[2] + distance], "direction": [0, 0, -1],
              "up": [0, 1, 0], "fov_y_radians": math.radians(60), "near": 0.1, "far": 0, "width": width, "height": height,
              "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"}
    scene = {"schema": 1, "state": "UNVERIFIED_REFERENCE_EXPORT", "linear": True, "units": "metres", "frame": 0,
             "revisions": [0, 0, 0], "camera": camera, "sky": [sky] * 3, "sun_angular_radius": 0.00465, "textures": [],
             "materials": materials, "material_extensions": {"version": 1, "records_per_material": 4, "records": records},
             "mesh_version": 2, "meshes": ["mesh_0.ply", "mesh_1.ply", "mesh_2.ply"], "instances": instances, "lights": [],
             "sampled_lights": []}
    (folder / "scene.json").write_text(json.dumps(scene))
    return folder
