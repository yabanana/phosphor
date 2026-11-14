#!/usr/bin/env python3
"""F15.1 (ticket 22, C22-D4) regression check: a surface seen THROUGH glass that the sun does not reach keeps its ambient light.

Fixture (generated here, nothing external): a white wall, an opaque panel in front of it whose sun shadow falls on the wall, and two thin-walled
glass panes (transmission 1, IOR 1.0: no bend, no Fresnel) in front of the wall, one over a shadowed part and one over a lit part.  Rendered
with the declared corpus rig (`--corpus-rig 0,0`: sun 3 W/m2 from the upper left behind the camera, constant environment 0.5, GI off).  The
transmitted ray of a glass pixel ends at the wall, whose sun shadow ray is blocked by the panel: before the fix the walk lit that hit with the
sun only (black, the corpus TransmissionRoughnessTest defect), the camera-visible wall is ambient-lit.  Gates (mean of the 8-bit capture over a
region of the wall seen through the glass / the same wall region seen directly, at equal lighting):
  shadowed ratio in [0.90, 1.10]   lit ratio in [0.90, 1.10]
Negative control (`--control-binary`, a build WITHOUT the fix) must fail the shadowed gate (ratio < 0.5) and keep the lit gate.

Exit codes: 0 gates hold (and the control fails as it must), 1 a gate failed, 2 usage, 3 the control did not fail (non-discriminating),
4 GPU busy, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import time
import zlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_corpus as corpus  # noqa: E402  (the runner's GPU guard and the declared capture command)

WIDTH, HEIGHT = 1024, 768
FOV_Y = math.radians(60.0)
MARGIN = 1.12  # CorpusRig::margin (src/testbench/corpus_rig.h)
SUN_OVER_Z = (-0.45 / 0.70, 0.55 / 0.70)  # towardSun.xy / towardSun.z of the rig at yaw 0, pitch 0

# World geometry (z = 0 is the wall, the camera looks along -z).
WALL = (-1.0, -0.6, 1.0, 0.6, 0.0)
PANEL = (-0.5, 0.0, 0.0, 0.4, 0.30)
PANE_SHADOW = (-0.34, -0.26, -0.02, 0.02, 0.15)
PANE_LIT = (0.40, -0.26, 0.74, 0.02, 0.15)
REGIONS = {  # wall rectangles (x0, y0, x1, y1) at z = 0
    "shadow_glass": (-0.25, -0.17, -0.08, -0.06), "shadow_direct": (0.02, -0.17, 0.15, -0.06),
    "lit_glass": (0.50, -0.17, 0.65, -0.06), "lit_direct": (0.78, -0.17, 0.92, -0.06),
}
GATE = (0.90, 1.10)
CONTROL_MAX = 0.5


def shadow_rect(panel, shadow_z=0.0):
    """Where the sun shadow of a panel at height h falls on the wall: the panel shifted by -h * towardSun.xy / towardSun.z."""
    x0, y0, x1, y1, z = panel
    dx, dy = -SUN_OVER_Z[0] * z, -SUN_OVER_Z[1] * z
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def frame_rig(bounds_min, bounds_max, aspect=WIDTH / HEIGHT):
    """Mirror of frameCorpusRig for yaw 0, pitch 0: camera distance and target."""
    centre = [0.5 * (a + b) for a, b in zip(bounds_min, bounds_max)]
    half = [0.5 * (b - a) for a, b in zip(bounds_min, bounds_max)]
    tan_half = math.tan(0.5 * FOV_Y)
    distance = 1e-3
    for corner in range(8):
        r = half[0] if corner & 1 else -half[0]
        u = half[1] if corner & 2 else -half[1]
        b = half[2] if corner & 4 else -half[2]
        distance = max(distance, MARGIN * max(abs(r) / (tan_half * aspect), abs(u) / tan_half) + b)
    return centre, distance


def project(point, centre, distance, aspect=WIDTH / HEIGHT):
    tan_half = math.tan(0.5 * FOV_Y)
    depth = distance - (point[2] - centre[2])
    ndc_x = (point[0] - centre[0]) / (depth * tan_half * aspect)
    ndc_y = (point[1] - centre[1]) / (depth * tan_half)
    return (0.5 + 0.5 * ndc_x) * WIDTH, (0.5 - 0.5 * ndc_y) * HEIGHT


def region_pixels(rect, centre, distance):
    x0, y0 = project((rect[0], rect[3], 0.0), centre, distance)
    x1, y1 = project((rect[2], rect[1], 0.0), centre, distance)
    return int(math.ceil(x0)), int(math.ceil(y0)), int(math.floor(x1)), int(math.floor(y1))


def write_scene(folder):
    """glTF with four quads (wall, panel, two glass panes); returns the .gltf path."""
    folder.mkdir(parents=True, exist_ok=True)
    quads = [(WALL, 0), (PANEL, 1), (PANE_SHADOW, 2), (PANE_LIT, 2)]
    positions, indices, nodes, meshes = [], [], [], []
    blob = bytearray()
    for k, ((x0, y0, x1, y1, z), material) in enumerate(quads):
        corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        base = len(blob)
        for c in corners:
            blob += struct.pack("<3f", *c)
        for n in range(4):
            blob += struct.pack("<3f", 0.0, 0.0, 1.0)
        for uv in ((0, 1), (1, 1), (1, 0), (0, 0)):
            blob += struct.pack("<2f", *uv)
        blob += struct.pack("<6H", 0, 1, 2, 0, 2, 3)
        meshes.append({"primitives": [{"attributes": {"POSITION": 3 * k, "NORMAL": 3 * k + 1, "TEXCOORD_0": 3 * k + 2}, "indices": 3 * len(quads) + k, "material": material}]})
        nodes.append({"mesh": k})
        positions.append((base, corners))
    accessors, views = [], []
    for k, (base, corners) in enumerate(positions):
        views += [{"buffer": 0, "byteOffset": base, "byteLength": 48}, {"buffer": 0, "byteOffset": base + 48, "byteLength": 48},
                  {"buffer": 0, "byteOffset": base + 96, "byteLength": 32}, {"buffer": 0, "byteOffset": base + 128, "byteLength": 12}]
        lo = [min(c[i] for c in corners) for i in range(3)]
        hi = [max(c[i] for c in corners) for i in range(3)]
        accessors += [{"bufferView": 4 * k, "componentType": 5126, "count": 4, "type": "VEC3", "min": lo, "max": hi},
                      {"bufferView": 4 * k + 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                      {"bufferView": 4 * k + 2, "componentType": 5126, "count": 4, "type": "VEC2"}]
    index_accessors = [{"bufferView": 4 * k + 3, "componentType": 5123, "count": 6, "type": "SCALAR"} for k in range(len(quads))]
    # accessor order: per quad position/normal/uv, then the indices (matches the mesh references above)
    document = {
        "asset": {"version": "2.0"}, "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior"],
        "scene": 0, "scenes": [{"nodes": list(range(len(quads)))}], "nodes": nodes, "meshes": meshes,
        "materials": [
            {"name": "wall", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": 1}},
            {"name": "panel", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1], "metallicFactor": 0, "roughnessFactor": 1}},
            {"name": "thin glass", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0},
             "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0}}}],
        "accessors": accessors + index_accessors, "bufferViews": views, "buffers": [{"uri": "scene.bin", "byteLength": len(blob)}]}
    (folder / "scene.bin").write_bytes(bytes(blob))
    path = folder / "scene.gltf"
    path.write_text(json.dumps(document))
    return path


def read_png_gray(path):
    """Mean-friendly decoder for the 8-bit RGB(A) PNGs the engine writes (no external dependency)."""
    data = pathlib.Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, width, height, channels = 8, b"", 0, 0, 0
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            assert depth == 8 and colour in (2, 6), "expected 8-bit RGB or RGBA"
            channels = 3 if colour == 2 else 4
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * channels
    rows, previous = [], bytearray(stride)
    for y in range(height):
        base = y * (stride + 1)
        filter_type, line = raw[base], bytearray(raw[base + 1:base + 1 + stride])
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b, c = previous[i], (previous[i - channels] if i >= channels else 0)
            if filter_type == 1:
                line[i] = (line[i] + a) & 255
            elif filter_type == 2:
                line[i] = (line[i] + b) & 255
            elif filter_type == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        previous = line
    return width, height, channels, rows


def region_mean(image, box):
    width, height, channels, rows = image
    x0, y0, x1, y1 = box
    total, count = 0.0, 0
    for y in range(max(y0, 0), min(y1, height)):
        for x in range(max(x0, 0), min(x1, width)):
            r, g, b = rows[y][x * channels:x * channels + 3]
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
    return total / max(count, 1)


def capture(binary, scene, folder):
    """One serial capture under the runner's GPU guard; returns the PNG path."""
    corpus.MODELS["fixture"] = {"tier": "A", "view": (0, 0), "shows": "shadowed hit seen through glass"}
    corpus.gltf_path = lambda corpus_dir, name: scene
    folder.mkdir(parents=True, exist_ok=True)
    command = corpus.capture_command(binary, None, "fixture", folder)
    deadline = time.monotonic() + 3600
    while True:
        try:
            if corpus.running_gpu_processes():
                raise corpus.GpuBusy("a GPU job runs")
            with corpus.gpu_lock("/tmp/phosphor-gpu-verification.lock"):
                if corpus.running_gpu_processes():
                    raise corpus.GpuBusy("a GPU job runs")
                env = {k: v for k, v in os.environ.items() if k not in corpus.VALIDATION_ENV}
                with (folder / "engine.log").open("w") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                break
        except corpus.GpuBusy:
            if time.monotonic() > deadline:
                raise
            time.sleep(5)
    if code != 0 or not (folder / "engine.png").is_file():
        raise corpus.CaptureFailed(f"capture failed with exit {code}: {folder / 'engine.log'}")
    return folder / "engine.png"


def measure(png):
    bounds_min = (WALL[0], WALL[1], 0.0)
    bounds_max = (WALL[2], WALL[3], PANEL[4])
    centre, distance = frame_rig(bounds_min, bounds_max)
    image = read_png_gray(png)
    means = {name: region_mean(image, region_pixels(rect, centre, distance)) for name, rect in REGIONS.items()}
    return {"means": means, "shadowed_ratio": means["shadow_glass"] / max(means["shadow_direct"], 1e-6),
            "lit_ratio": means["lit_glass"] / max(means["lit_direct"], 1e-6)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--control-binary", type=pathlib.Path, help="a build WITHOUT the fix: the shadowed gate must fail with it")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-shadowed-hit")
    args = parser.parse_args(argv)
    try:
        scene = write_scene(args.out / "scene")
        result = {"fixture": str(scene)}
        result["fixed"] = measure(capture(args.binary.resolve(), scene, args.out / "fixed"))
        ok = all(GATE[0] <= result["fixed"][k] <= GATE[1] for k in ("shadowed_ratio", "lit_ratio"))
        result["gates_hold"] = ok
        control_failed = None
        if args.control_binary:
            result["control"] = measure(capture(args.control_binary.resolve(), scene, args.out / "control"))
            control_failed = result["control"]["shadowed_ratio"] < CONTROL_MAX and GATE[0] <= result["control"]["lit_ratio"] <= GATE[1]
            result["control_fails_as_required"] = control_failed
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if not ok:
            return 1
        if args.control_binary and not control_failed:
            return 3
        return 0
    except corpus.GpuBusy:
        print("GPU busy")
        return 4
    except corpus.CaptureFailed as error:
        print(error)
        return 6


if __name__ == "__main__":
    sys.exit(main())
