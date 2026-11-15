#!/usr/bin/env python3
"""F15.1 (ticket 22, C22-A1) fixture check: a mirror seen in a mirror is no longer dark (multi-bounce RT reflections).

Two generated scenes (nothing external), every surface a perfectly smooth metal (baseColor 0.8, metallic 1, roughness 0, no extensions) under the
declared corpus rig (`--corpus-rig 0,0`: constant environment 0.5, GI off, RT reflections):
  corridor  two facing mirror walls 1.0 apart and 8.0 long: a camera ray bounces up to 4 times between them (classes 1..4)
  ball      a mirror ball next to a mirror wall turned by 65 degrees: the ball sees the wall (2), the wall sees the ball (2), the ball sees the
            wall seeing the ball (3)
`tools/f15_reflection_bounce_model.py` is the independent CPU model (analytic ray casting, Schlick Fresnel F0 0.8, environment 0.5): per pixel the
number of mirror events (the "class") and the radiance the declared bounded estimator (3 extra bounces) must show.  Pixels are compared only where
the class is constant over a 7x7 window (silhouettes, mirror edges and the denoiser's reach excluded) and no event is near the sun.

Gates (fixed here BEFORE any engine comparison, decision 11; never loosened):
  fixture   every expected class has at least MIN_PIXELS stable pixels in the model
  class k   mean over the class of |engine/background - model/0.5| / max(model/0.5, FLOOR * class mean of model/0.5)  <=  GATE
  background  mean of the class-0 pixels within BACKGROUND_TOLERANCE of the declared environment 0.5
Negative control `--reflection-bounces 0` (the F13 single bounce): classes >= 2 must EXCEED the gate (they are dark), class 1 still passes.

Run under the GPU lock (this script does not take it): python3 gpulock.py python3 tools/f15_reflection_bounce_check.py --binary build/release/phosphor
Exit codes: 0 gates hold (and the control fails as it must), 1 a gate failed, 3 the control did not fail, 5 fixture/model problem, 6 capture failed.
"""
from __future__ import annotations

import argparse
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
import f15_corpus as corpus  # noqa: E402  (the declared capture command)
import f15_reflection_bounce_model as model  # noqa: E402

WIDTH, HEIGHT = 640, 360
WARMUP, FRAMES = 48, 48
GATE = 0.05
FLOOR = 0.05
MIN_PIXELS = 400
BACKGROUND_TOLERANCE = 0.05
EXPECTED_CLASSES = {"corridor": (1, 2, 3, 4), "ball": (1, 2, 3)}
LUMA = (0.2126, 0.7152, 0.0722)
SPHERE_SEGMENTS, SPHERE_RINGS = 96, 48


# --- fixture writer -----------------------------------------------------------------------------------------

def quad_mesh(corner, u, v):
    """Positions, normals, uvs, indices of a parallelogram whose front face is cross(u, v) (counter-clockwise)."""
    corners = [corner, tuple(corner[i] + u[i] for i in range(3)), tuple(corner[i] + u[i] + v[i] for i in range(3)), tuple(corner[i] + v[i] for i in range(3))]
    n = model.normalize(model.cross(u, v))
    return corners, [n] * 4, [(0, 1), (1, 1), (1, 0), (0, 0)], [0, 1, 2, 0, 2, 3]


def sphere_mesh(centre, radius):
    """UV sphere with smooth outward normals (counter-clockwise seen from outside)."""
    positions, normals, uvs, indices = [], [], [], []
    for ring in range(SPHERE_RINGS + 1):
        theta = math.pi * ring / SPHERE_RINGS
        for seg in range(SPHERE_SEGMENTS + 1):
            phi = 2.0 * math.pi * seg / SPHERE_SEGMENTS
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            positions.append(tuple(centre[i] + radius * n[i] for i in range(3)))
            normals.append(n)
            uvs.append((seg / SPHERE_SEGMENTS, ring / SPHERE_RINGS))
    stride = SPHERE_SEGMENTS + 1
    for ring in range(SPHERE_RINGS):
        for seg in range(SPHERE_SEGMENTS):
            a, b, c, d = ring * stride + seg, ring * stride + seg + 1, (ring + 1) * stride + seg, (ring + 1) * stride + seg + 1
            indices += [a, b, c, b, d, c]
    return positions, normals, uvs, indices


def write_scene(folder, description):
    """glTF (one mesh per quad / sphere, one metal material); returns the .gltf path."""
    folder.mkdir(parents=True, exist_ok=True)
    meshes_data = [quad_mesh(*q) for q in description["quads"]] + [sphere_mesh(c, r) for c, r in description["spheres"]]
    blob = bytearray()
    views, accessors, meshes, nodes = [], [], [], []
    for k, (positions, normals, uvs, indices) in enumerate(meshes_data):
        accessor_base = len(accessors)
        for data, kind, count_type, fmt in ((positions, "VEC3", 5126, "<3f"), (normals, "VEC3", 5126, "<3f"), (uvs, "VEC2", 5126, "<2f"), (indices, "SCALAR", 5125, "<I")):
            offset = len(blob)
            for item in data:
                blob += struct.pack(fmt, *item) if kind != "SCALAR" else struct.pack(fmt, item)
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob) - offset})
            accessor = {"bufferView": len(views) - 1, "componentType": count_type, "count": len(data), "type": kind}
            if data is positions:
                accessor["min"] = [min(p[i] for p in positions) for i in range(3)]
                accessor["max"] = [max(p[i] for p in positions) for i in range(3)]
            accessors.append(accessor)
        meshes.append({"primitives": [{"attributes": {"POSITION": accessor_base, "NORMAL": accessor_base + 1, "TEXCOORD_0": accessor_base + 2},
                                       "indices": accessor_base + 3, "material": 0}]})
        nodes.append({"mesh": k})
    document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
                "materials": [{"name": "mirror", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 1, "roughnessFactor": 0}}],
                "accessors": accessors, "bufferViews": views, "buffers": [{"uri": "scene.bin", "byteLength": len(blob)}]}
    (folder / "scene.bin").write_bytes(bytes(blob))
    path = folder / "scene.gltf"
    path.write_text(json.dumps(document))
    return path


# --- engine capture -----------------------------------------------------------------------------------------

class CaptureFailed(Exception):
    pass


def capture(binary, scene, folder, bounces):
    """One capture (the caller holds the GPU lock); returns (pfm path, log text)."""
    folder.mkdir(parents=True, exist_ok=True)
    corpus.MODELS["fixture"] = {"tier": "A", "view": (0, 0), "shows": "reflection bounce fixture", "resolution": f"{WIDTH}x{HEIGHT}"}
    corpus.gltf_path = lambda corpus_dir, name: scene
    command = corpus.capture_command(binary, None, "fixture", folder)
    command[command.index("--warmup") + 1] = str(WARMUP)
    command[command.index("--frames") + 1] = str(FRAMES)
    command += ["--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", str(WARMUP + FRAMES - 1), "--capture-linear-signal", "hdr",
                "--reflection-bounces", str(bounces), "--debug-lighting", "1"]
    env = {k: v for k, v in os.environ.items() if k not in corpus.VALIDATION_ENV}
    with (folder / "engine.log").open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if code != 0 or not (folder / "engine.pfm").is_file():
        raise CaptureFailed(f"capture failed with exit {code}: {folder / 'engine.log'}")
    return folder / "engine.pfm", (folder / "engine.log").read_text()


def read_pfm_luminance(path):
    """(width, height, rows of linear luminance, top row first) of an RGB float32 PFM."""
    data = pathlib.Path(path).read_bytes()
    position, fields = 0, []
    while len(fields) < 3:
        end = data.index(b"\n", position)
        fields.append(data[position:end].decode())
        position = end + 1
    if fields[0] != "PF":
        raise ValueError(f"{path}: expected an RGB PFM")
    width, height = map(int, fields[1].split())
    little = float(fields[2]) < 0
    values = struct.unpack(("<" if little else ">") + f"{3 * width * height}f", data[position:position + 12 * width * height])
    rows = []
    for y in range(height):
        base = 3 * width * y
        rows.append([LUMA[0] * values[base + 3 * x] + LUMA[1] * values[base + 3 * x + 1] + LUMA[2] * values[base + 3 * x + 2] for x in range(width)])
    rows.reverse()  # PFM stores the bottom row first
    return width, height, rows


# --- comparison ---------------------------------------------------------------------------------------------

def reference(scene_name, width=WIDTH, height=HEIGHT):
    """(radiance grid, class grid, stable mask, model camera distance) of a fixture scene at the declared bound."""
    description = model.SCENES[scene_name]
    low, high = model.scene_bounds(description)
    rad, cls, mask = model.render(model.build_scene(description), width, height, low, high)
    return rad, cls, model.stable_mask(cls, mask), model.frame(low, high, width / height)[2]


def evaluate(rad, cls, stable, engine_rows, expected):
    """Per-class statistics of an engine luminance image against the model.  Returns {"background": ..., "classes": {k: {...}}}."""
    height, width = len(cls), len(cls[0])
    background = [engine_rows[y][x] for y in range(height) for x in range(width) if stable[y][x] and cls[y][x] == 0]
    if not background:
        raise ValueError("no stable background pixels")
    bg = sum(background) / len(background)
    result = {"background": {"mean": bg, "relative_error": abs(bg - model.ENVIRONMENT) / model.ENVIRONMENT, "pixels": len(background)}, "classes": {}}
    for k in expected:
        pixels = [(y, x) for y in range(height) for x in range(width) if stable[y][x] and cls[y][x] == k]
        if not pixels:
            result["classes"][k] = {"pixels": 0, "mean_relative_error": float("inf"), "model_mean": 0.0, "engine_mean": 0.0}
            continue
        expected_rel = [rad[y][x] / model.ENVIRONMENT for y, x in pixels]
        floor = FLOOR * (sum(expected_rel) / len(expected_rel))
        error = sum(abs(engine_rows[y][x] / bg - e) / max(e, floor) for (y, x), e in zip(pixels, expected_rel)) / len(pixels)
        result["classes"][k] = {"pixels": len(pixels), "mean_relative_error": error, "model_mean": sum(expected_rel) / len(expected_rel),
                                "engine_mean": sum(engine_rows[y][x] / bg for y, x in pixels) / len(pixels)}
    return result


def verdict(result, expected, control=False):
    """Fixed-gate decision.  Normal run: every gate holds.  Control run (single bounce): classes >= 2 must exceed the gate, class 1 and the
    background still hold.  Returns (ok, reasons)."""
    reasons = []
    if result["background"]["relative_error"] > BACKGROUND_TOLERANCE:
        reasons.append(f"background mean {result['background']['mean']:.4f} off the declared 0.5 by {result['background']['relative_error']:.3f}")
    for k in expected:
        stats = result["classes"][k]
        if stats["pixels"] < MIN_PIXELS:
            reasons.append(f"class {k}: {stats['pixels']} stable pixels < {MIN_PIXELS}")
            continue
        holds = stats["mean_relative_error"] <= GATE
        should_hold = (not control) or k == 1
        if holds != should_hold:
            reasons.append(f"class {k}: error {stats['mean_relative_error']:.4f} {'exceeds' if should_hold else 'does not exceed'} {GATE}")
    return not reasons, reasons


def run_scene(name, args):
    folder = args.out / name
    scene = write_scene(folder / "scene", model.SCENES[name])
    rad, cls, stable, distance = reference(name)
    expected = EXPECTED_CLASSES[name]
    report = {"scene": str(scene), "model_distance": distance}
    runs = {"fixed": args.bounces}
    if args.control:
        runs["control"] = 0
    for label, bounces in runs.items():
        pfm, log = capture(args.binary.resolve(), scene, folder / label, bounces)
        rig = corpus.RIG_LINE.search(log)
        if not rig:
            raise CaptureFailed(f"no CORPUS_RIG line in {folder / label / 'engine.log'}")
        if abs(float(rig.group(4)) - distance) > 1e-3 * distance:
            raise ValueError(f"engine camera distance {rig.group(4)} differs from the model's {distance:.5f}")
        width, height, rows = read_pfm_luminance(pfm)
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"capture is {width}x{height}")
        result = evaluate(rad, cls, stable, rows, expected)
        result["bounces"] = bounces
        counters = re.search(r"REFLECTION mirror bounces slot \d+ \| extra specular bounces traced (\d+) \| chains ended by the bound (\d+)", log)
        result["counters"] = {"extra_traces": int(counters.group(1)), "bound_ended": int(counters.group(2))} if counters else None
        ok, reasons = verdict(result, expected, control=(label == "control"))
        result["ok"], result["reasons"] = ok, reasons
        report[label] = result
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=list(model.SCENES), choices=list(model.SCENES))
    parser.add_argument("--bounces", type=int, default=3, help="--reflection-bounces of the main run (default 3, the bound the model declares)")
    parser.add_argument("--control", action=argparse.BooleanOptionalAction, default=True, help="also run --reflection-bounces 0, which must fail")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-reflection-bounce")
    args = parser.parse_args(argv)
    code = 0
    try:
        for name in args.scenes:
            report = run_scene(name, args)
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"result-{name}.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({name: report}, indent=2))
            if not report["fixed"]["ok"]:
                code = max(code, 1)
            elif args.control and not report["control"]["ok"]:
                code = 3 if code == 0 else code
    except CaptureFailed as error:
        print(error)
        return 6
    except ValueError as error:
        print(error)
        return 5
    return code


if __name__ == "__main__":
    sys.exit(main())
