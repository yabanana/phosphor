#!/usr/bin/env python3
"""F15 closure (loader-misc): KHR_materials_unlit seen by the GI rays (shaders/gi_common.h `giMaterial`: an unlit surface has no hit emission).

Scene: a diffuse wall (albedo 0.8, facing +Z) and a floor panel (6 x 6 m, facing up, in front of the wall) of one of three kinds, under the declared
corpus rig (sun 3 W/m2, constant environment 0.5) with DDGI (`--gi ddgi`): `unlit` (KHR_materials_unlit, colour 1), `emissive` (a lit black material
with emissiveFactor 1: the same visible colour, but a light source for the GI rays), `none` (the same unlit panel, black: no emission, no reflection, the dark twin with identical bounds and probes).  The wall patch just above the floor receives
the panel's bounce only when the panel emits into the GI.  Criteria, fixed before any capture: the unlit panel itself shows its colour (centre
within 0.002 of 1.0 in every channel); delta(unlit) = mean(wall patch, unlit) - mean(wall patch, none) must be within 0.005 of 0 (the unlit panel
lights nothing: the settled DDGI of two runs of the same scene differs by less, measured as |none - none'| in the same run set); delta(emissive) must
be above 0.05 (the control: the emissive twin does light the wall, so the patch can see a panel emission at all).  `none'` is a second capture of
the dark twin (the noise floor of the settled GI).

  python3 "$H/gpulock.py" python3 -I tools/f15_unlit_gi_check.py run OUTDIR [BINARY]
  python3 "$H/gpulock.py" python3 -I tools/f15_unlit_gi_check.py validate OUTDIR [BINARY]   (the unlit scene under API + shader validation: no message)
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_unlit_lights_check as base  # noqa: E402

RES = (1280, 720)
FLOOR_Y, WALL_Z = -1.0, -2.5


def quad(positions, normal, indices):
    uv = [0, 1, 1, 1, 1, 0, 0, 0]
    blob = struct.pack("<12f", *positions) + struct.pack("<12f", *(list(normal) * 4)) + struct.pack("<8f", *uv) + struct.pack("<6H", *indices)
    return blob


def scene(kind):
    """kind: unlit | emissive | none."""
    wall = quad([-5, -1, WALL_Z, 5, -1, WALL_Z, 5, 5, WALL_Z, -5, 5, WALL_Z], (0, 0, 1), [0, 1, 2, 0, 2, 3])
    floor = quad([-3, FLOOR_Y, -2.4, 3, FLOOR_Y, -2.4, 3, FLOOR_Y, 3.6, -3, FLOOR_Y, 3.6], (0, 1, 0), [0, 3, 2, 0, 2, 1])
    blob = wall + floor
    materials = [{"name": "wall", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": 1}}]
    nodes = [{"mesh": 0}]
    meshes = [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}]
    extras = {}
    if kind in ("unlit", "none"):
        colour = 1 if kind == "unlit" else 0  # `none`: the same unlit panel, black (no emission, no reflection): the dark twin
        materials.append({"name": "panel", "pbrMetallicRoughness": {"baseColorFactor": [colour, colour, colour, 1]}, "extensions": {"KHR_materials_unlit": {}}})
        extras = {"extensionsUsed": ["KHR_materials_unlit"]}
    else:
        materials.append({"name": "panel", "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1], "metallicFactor": 0, "roughnessFactor": 1},
                          "emissiveFactor": [1, 1, 1]})
    nodes.append({"mesh": 1})
    meshes.append({"primitives": [{"attributes": {"POSITION": 4, "NORMAL": 5, "TEXCOORD_0": 6}, "indices": 7, "material": 1}]})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": materials,
           "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
           "bufferViews": [], "accessors": []}
    stride = 140
    for m in range(2):
        o = m * stride
        doc["bufferViews"] += [{"buffer": 0, "byteOffset": o, "byteLength": 48}, {"buffer": 0, "byteOffset": o + 48, "byteLength": 48},
                               {"buffer": 0, "byteOffset": o + 96, "byteLength": 32}, {"buffer": 0, "byteOffset": o + 128, "byteLength": 12}]
        lo, hi = ([-5, -1, WALL_Z], [5, 5, WALL_Z]) if m == 0 else ([-3, FLOOR_Y, -2.4], [3, FLOOR_Y, 3.6])
        v = m * 4
        doc["accessors"] += [{"bufferView": v, "componentType": 5126, "count": 4, "type": "VEC3", "min": lo, "max": hi},
                             {"bufferView": v + 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                             {"bufferView": v + 2, "componentType": 5126, "count": 4, "type": "VEC2"},
                             {"bufferView": v + 3, "componentType": 5123, "count": 6, "type": "SCALAR"}]
    doc.update(extras)
    return doc


def capture(binary, scene_path, folder, name, validation=False):
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--bench", "4", "--scene", str(scene_path), "--corpus-rig", "0,0", "--rt", "on", "--shadows", "rt", "--lighting", "restir",
           "--gi", "ddgi", "--gi-grid", "16x8x16", "--gi-rays", "64", "--reflections", "rt", "--ao", "off", "--lighting-denoise", "custom",
           "--tonemap", "aces", "--post", "--upscaler", "native", "--resolution", f"{RES[0]}x{RES[1]}", "--warmup", "96", "--frames", "1",
           "--fixed-timestep", "--offscreen", "--no-ui", "--no-vsync", "--no-gpu-timing", "--render-path", "visibility", "--geometry-path", "mesh",
           "--pipeline-sync", "--capture-linear", str(folder / f"{name}.pfm"), "--capture-linear-signal", "hdr"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_")}
    if validation:
        env.update({"MTL_DEBUG_LAYER": "1", "MTL_SHADER_VALIDATION": "1", "MTL_DEBUG_LAYER_WARNING_MODE": "nslog"})
        cmd.append("--no-pipeline-archive")
    log = folder / f"{name}.log"
    with log.open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=900).returncode
    return code, log.read_text(errors="replace")


def rig_geometry(text):
    m = re.search(r"bounds min \(([-\d. e+]+)\) max \(([-\d. e+]+)\), camera distance ([\d.]+)", text)
    lo = [float(v) for v in m.group(1).split()]
    hi = [float(v) for v in m.group(2).split()]
    return lo, hi, float(m.group(3))


def project(lo, hi, distance, size, point):
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    depth = distance - (point[2] - centre[2])
    half_h = depth * math.tan(math.radians(30))
    half_w = half_h * size[0] / size[1]
    return (int(round(((point[0] - centre[0]) / half_w + 1) / 2 * size[0])), int(round((1 - (point[1] - centre[1]) / half_h) / 2 * size[1])))


def verify(out):
    imgs = {k: base.read_pfm(out / f"{k}.pfm") for k in ("unlit", "emissive", "none", "none2")}
    lo, hi, d = rig_geometry((out / "none.log").read_text(errors="replace"))
    # wall patch 0.5 m above the floor line at x = 0, the floor panel's centre is the unlit-colour probe
    wx, wy = project(lo, hi, d, RES, (0.0, FLOOR_Y + 0.5, WALL_Z))
    fx, fy = project(lo, hi, d, RES, (0.0, FLOOR_Y, 0.6))
    means = {k: base.mean_and_dev(base.patch(v, wx, wy))[0] for k, v in imgs.items()}
    lum = {k: sum(v) / 3 for k, v in means.items()}
    floor_unlit = base.mean_and_dev(base.patch(imgs["unlit"], fx, fy))
    floor_err = max(abs(floor_unlit[0][c] - 1.0) for c in range(3))
    delta_u = lum["unlit"] - lum["none"]
    delta_e = lum["emissive"] - lum["none"]
    noise = abs(lum["none2"] - lum["none"])
    lines = [f"wall patch {wx},{wy} luminance: none {lum['none']:.5f}, none' {lum['none2']:.5f}, unlit {lum['unlit']:.5f}, emissive {lum['emissive']:.5f}",
             f"floor panel centre {fx},{fy} unlit: error from 1.0 = {floor_err:.5f} (<= 0.002 required; the visible panel is its colour)",
             f"noise floor |none' - none| = {noise:.5f}; delta(unlit) = {delta_u:+.5f} (|.| <= max(0.005, 2 x noise) required); delta(emissive) = {delta_e:+.5f} (> 0.05 required, the control)"]
    ok = floor_err <= 0.002 and abs(delta_u) <= max(0.005, 2 * noise) and delta_e > 0.05
    lines.append("PASS" if ok else "FAIL")
    return ok, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "validate", "verify"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "verify":
        ok, lines = verify(out)
        print("\n".join(lines))
        return 0 if ok else 1
    for kind in ("unlit", "emissive", "none"):
        (out / f"{kind}.gltf").write_text(json.dumps(scene(kind)))
    if args.mode == "validate":
        code, text = capture(args.binary, out / "unlit.gltf", out, "unlit-validation", validation=True)
        messages = [l for l in text.splitlines() if re.search(r"failed assertion|Shader Validation|Validation Error|GPU timeout|MTLDebug", l)]
        print(f"unlit under API + shader validation: exit {code}, {len(messages)} validation messages {messages[:3]}")
        return 0 if code == 0 and not messages else 1
    for name, kind in (("unlit", "unlit"), ("emissive", "emissive"), ("none", "none"), ("none2", "none")):
        scene_path = out / f"{kind}.gltf"
        code, text = capture(args.binary, scene_path, out, name)
        print(f"{name}: exit {code}", flush=True)
        if code != 0:
            print(text[-1200:])
            return 1
    ok, lines = verify(out)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
