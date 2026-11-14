#!/usr/bin/env python3
"""Snapshot builders that mirror the transmissive-shadow bench (src/testbench/transmissive_shadow_validation.cpp) for the Python tests.

The numbers (positions, sizes, factors, lights, camera) are the ones tests/test_transmissive_shadow_validation.cpp pins.
`write_snapshot` writes the files `exportOfflineReference` would (scene.json, mesh_N.ply) so that tools/f15_shadow_reference.py
and tools/f15_check.py can run without a GPU.  Stdlib + numpy only (the mesh builders are those of f15_transmission_fixtures).
"""
from __future__ import annotations

import json
import math
import pathlib

import f15_transmission_fixtures as tf

INVALID = tf.INVALID
GREEN = (0.4, 0.9, 0.45)
CASTER_HEIGHT = 0.8
FLOOR_SIZE = 16.0
FLOOR_BASE = (0.8, 0.8, 0.8)
SUN_RADIUS = 0.00465


def to_sun(degrees_from_vertical):
    a = math.radians(degrees_from_vertical)
    return (0.0, math.cos(a), -math.sin(a))


def caster(shape, x, y, scale, base=(1, 1, 1), ior=1.5, thickness=0.0, attenuation_color=(1, 1, 1),
           attenuation_distance=math.inf, double_sided=False, opaque=False):
    return dict(shape=shape, center=(x, y, 0.0), scale=scale, base=base, ior=ior, thickness=thickness,
                attenuation_color=attenuation_color, attenuation_distance=attenuation_distance,
                double_sided=double_sided, opaque=opaque)


def thin_pane(x, side, base=(1, 1, 1)):
    return caster("pane", x, CASTER_HEIGHT, (side, 1.0, side), base=base, thickness=0.0, double_sided=True)


def slab(x, y, side, thickness, ior, attenuation_color=(1, 1, 1), attenuation_distance=math.inf):
    return caster("cube", x, y, (side, thickness, side), ior=ior, thickness=1.0, attenuation_color=attenuation_color,
                  attenuation_distance=attenuation_distance)


def opaque_slab(x, y, side, thickness):
    return caster("cube", x, y, (side, thickness, side), thickness=0.0, opaque=True)


def glass_sphere(x, y, ior, attenuation_color=(1, 1, 1), attenuation_distance=math.inf):
    return caster("sphere", x, y, (1.0, 1.0, 1.0), ior=ior, thickness=0.8, attenuation_color=attenuation_color,
                  attenuation_distance=attenuation_distance)


H = CASTER_HEIGHT
SCENARIOS = {
    "transmissive-shadow-pane-sun": dict(
        casters=[thin_pane(-2.1, 0.9), thin_pane(-0.7, 0.9, GREEN), slab(0.7, H, 0.9, 0.06, 1.5), opaque_slab(2.1, H, 0.9, 0.06)],
        sun=(to_sun(25.0), 3.0), points=[]),
    "transmissive-shadow-water-sun": dict(
        casters=[slab(-1.4, 0.5, 0.9, 0.12, 1.33), slab(0.0, 0.5, 0.9, 0.12, 1.33, (0.45, 0.75, 0.9), 0.3),
                 opaque_slab(1.4, 0.5, 0.9, 0.12)],
        sun=(to_sun(20.0), 3.0), points=[]),
    "transmissive-shadow-volume-pane": dict(
        casters=[slab(-1.4, H, 0.9, 0.1, 1.5, (0.5, 0.8, 1.0), 0.2), slab(0.0, H, 0.9, 0.1, 1.5, (1.0, 0.6, 0.35), 0.2),
                 slab(1.4, H, 0.9, 0.1, 1.5, (0.65, 0.65, 0.65), 0.15)],
        sun=(to_sun(20.0), 3.0), points=[]),
    "transmissive-shadow-point": dict(
        casters=[thin_pane(-1.8, 0.7), thin_pane(-0.6, 0.7, GREEN), slab(0.6, H, 0.7, 0.06, 1.5), opaque_slab(1.8, H, 0.7, 0.06)],
        sun=None, points=[((0.0, 4.5, -0.3), 60.0, 100.0)]),
    "transmissive-shadow-curved": dict(
        casters=[glass_sphere(-1.0, 0.9, 1.5), glass_sphere(1.0, 0.9, 1.5, (0.5, 0.8, 1.0), 0.6)],
        sun=(to_sun(25.0), 3.0), points=[]),
}


def write_snapshot(folder, name, width=320, height=180, elevation=38.0, distance=6.2, target=(0.0, 0.0, 0.3)):
    """scene.json + PLYs of one scenario (a floor, the casters, the light); a smaller frame than the bench's 1280x720 keeps the
    Mitsuba tests quick while the world-space geometry stays identical."""
    spec = SCENARIOS[name]
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tf.write_ply(folder / "mesh_0.ply", tf.sphere_mesh())
    tf.write_ply(folder / "mesh_1.ply", tf.cube_mesh())
    tf.write_ply(folder / "mesh_2.ply", tf.plane_mesh())
    mesh_of = {"sphere": 0, "cube": 1, "pane": 2}
    materials, records, instances = [], [], []
    # instance 0: the floor (opaque, diffuse, no extension)
    materials.append({"base": [*FLOOR_BASE, 1], "emissive": [0, 0, 0], "metallic": 0.0, "roughness": 1.0, "normal_scale": 1,
                      "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0, "textures": [INVALID] * 5, "feature_bits": 0,
                      "extension_index": -1})
    identity = tf.rotation_matrix()
    instances.append({"slot": 0, "generation": 1, "flags": 3, "mesh": 2, "material": 0,
                      "world": tf.world_words((0, 0, 0), (FLOOR_SIZE, 1.0, FLOOR_SIZE), identity)})
    for g in spec["casters"]:
        if g["opaque"]:
            materials.append({"base": [0.5, 0.5, 0.5, 1], "emissive": [0, 0, 0], "metallic": 0.0, "roughness": 1.0,
                              "normal_scale": 1, "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0,
                              "textures": [INVALID] * 5, "feature_bits": 0, "extension_index": -1})
        else:
            feature_bits = 32 | (64 if g["thickness"] > 0 else 0) | (1 if g["ior"] != 1.5 else 0)
            record0 = [tf.bits(g["ior"]), tf.bits(1.0), tf.bits(1), tf.bits(1), tf.bits(1), INVALID, INVALID] + [0] * 13
            record1 = ([tf.bits(0), tf.bits(0), INVALID, tf.bits(0), tf.bits(1.3), tf.bits(100), tf.bits(400), INVALID, INVALID,
                        tf.bits(1.0), INVALID, tf.bits(g["thickness"]), INVALID] +
                       [tf.bits(c) for c in g["attenuation_color"]] + [tf.bits(g["attenuation_distance"]), 0, 0, 0])
            materials.append({"base": [*g["base"], 1], "emissive": [0, 0, 0], "metallic": 0.0, "roughness": 0.0, "normal_scale": 1,
                              "occlusion_strength": 1, "alpha_cutoff": 0,
                              "flags": (feature_bits << tf.FEATURE_SHIFT) | (1 if g["double_sided"] else 0),
                              "textures": [INVALID] * 5, "feature_bits": feature_bits, "extension_index": len(records)})
            records += [record0, record1, [0] * 20, [0] * 20]
        instances.append({"slot": len(instances), "generation": 1, "flags": 3, "mesh": mesh_of[g["shape"]],
                          "material": len(materials) - 1, "world": tf.world_words(g["center"], g["scale"], identity)})
    e = math.radians(elevation)
    position = [target[0], target[1] + distance * math.sin(e), target[2] + distance * math.cos(e)]
    direction = [target[i] - position[i] for i in range(3)]
    norm = math.sqrt(sum(c * c for c in direction))
    camera = {"position": position, "direction": [c / norm for c in direction], "up": [0, 1, 0],
              "fov_y_radians": math.radians(60), "near": 0.1, "far": 0, "width": width, "height": height,
              "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"}
    lights, sampled = [], []
    if spec["sun"]:
        toward, irradiance = spec["sun"]
        lights.append({"type": 0, "position": [0, 0, 0], "direction": [-c for c in toward], "color": [1, 1, 1],
                       "intensity": irradiance, "range": 0, "inner": 0, "outer": 0})
    for position_, intensity, light_range in spec["points"]:
        lights.append({"type": 1, "position": list(position_), "direction": [0, 0, -1], "color": [1, 1, 1],
                       "intensity": intensity, "range": light_range, "inner": 0, "outer": 0})
        sampled.append({"type": 1, "position": list(position_)})
    scene = {"schema": 1, "state": "UNVERIFIED_REFERENCE_EXPORT", "linear": True, "units": "metres", "frame": 0,
             "revisions": [0, 0, 0], "camera": camera, "sky": [0.0] * 3, "sun_angular_radius": SUN_RADIUS, "textures": [],
             "materials": materials, "material_extensions": {"version": 1, "records_per_material": 4, "records": records},
             "mesh_version": 2, "meshes": ["mesh_0.ply", "mesh_1.ply", "mesh_2.ply"], "instances": instances,
             "lights": lights, "sampled_lights": sampled}
    (folder / "scene.json").write_text(json.dumps(scene))
    return folder
