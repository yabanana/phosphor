#!/usr/bin/env python3
"""Shared helpers of the F15 residue fixtures (stdlib only): a tiny glTF writer, the declared-rig capture and PNG measurement.

Used by tools/f15_residue_check.py (F13 residues found in F15: silhouette grazing, mirror environment term, DDGI behind glass, single-sided
back faces) and available to any check that needs a generated scene instead of an external model.  Nothing external is loaded or executed.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_corpus as corpus  # noqa: E402  (the runner's GPU guard and the declared capture command)
from f15_shadowed_hit_check import read_png_gray, region_mean  # noqa: E402,F401

WIDTH, HEIGHT = 1024, 768
FOV_Y = math.radians(60.0)
MARGIN = 1.12  # CorpusRig::margin (src/testbench/corpus_rig.h)
# towardSun.xy / towardSun.z of the rig at yaw 0, pitch 0 (the sun is up and to the left behind the camera)
SUN_OVER_Z = (-0.45 / 0.70, 0.55 / 0.70)


class Scene:
    """Indexed triangle meshes with PBR materials; every primitive is its own node."""

    def __init__(self):
        self.materials, self.primitives, self.extensions = [], [], set()

    def material(self, name, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=1.0, extensions=None, double_sided=False):
        entry = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [*color, 1], "metallicFactor": metallic, "roughnessFactor": roughness}}
        if extensions:
            entry["extensions"] = extensions
            self.extensions.update(extensions)
        if double_sided:
            entry["doubleSided"] = True
        self.materials.append(entry)
        return len(self.materials) - 1

    def add(self, positions, normals, indices, material):
        self.primitives.append((positions, normals, indices, material))

    def quad(self, corners, material):
        """Four corners counter-clockwise seen from the FRONT side (the normal is the right-handed one of the first three)."""
        a, b, c = corners[:3]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        length = math.sqrt(sum(x * x for x in n))
        n = [x / length for x in n]
        self.add(list(corners), [tuple(n)] * 4, [0, 1, 2, 0, 2, 3], material)

    def sphere(self, centre, radius, segments, material):
        """UV sphere with smooth (analytic) normals; segments = longitudes (latitudes are half of them)."""
        lon, lat = segments, max(segments // 2, 4)
        positions, normals, indices = [], [], []
        for j in range(lat + 1):
            theta = math.pi * j / lat
            for i in range(lon + 1):
                phi = 2.0 * math.pi * i / lon
                n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
                positions.append(tuple(centre[k] + radius * n[k] for k in range(3)))
                normals.append(n)
        for j in range(lat):
            for i in range(lon):
                a, b = j * (lon + 1) + i, (j + 1) * (lon + 1) + i
                indices += [a, a + 1, b, a + 1, b + 1, b]  # counter-clockwise seen from outside
        self.add(positions, normals, indices, material)

    def bounds(self):
        points = [p for positions, _, _, _ in self.primitives for p in positions]
        return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))

    def write(self, folder):
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        blob, views, accessors, meshes, nodes = bytearray(), [], [], [], []
        for k, (positions, normals, indices, material) in enumerate(self.primitives):
            def view(data):
                while len(blob) % 4:
                    blob.append(0)
                views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
                blob.extend(data)
                return len(views) - 1
            pos = view(b"".join(struct.pack("<3f", *p) for p in positions))
            nor = view(b"".join(struct.pack("<3f", *n) for n in normals))
            tex = view(b"".join(struct.pack("<2f", 0.0, 0.0) for _ in positions))
            idx = view(b"".join(struct.pack("<I", i) for i in indices))
            lo = [min(p[i] for p in positions) for i in range(3)]
            hi = [max(p[i] for p in positions) for i in range(3)]
            base = len(accessors)
            accessors += [{"bufferView": pos, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": lo, "max": hi},
                          {"bufferView": nor, "componentType": 5126, "count": len(positions), "type": "VEC3"},
                          {"bufferView": tex, "componentType": 5126, "count": len(positions), "type": "VEC2"},
                          {"bufferView": idx, "componentType": 5125, "count": len(indices), "type": "SCALAR"}]
            meshes.append({"primitives": [{"attributes": {"POSITION": base, "NORMAL": base + 1, "TEXCOORD_0": base + 2}, "indices": base + 3, "material": material}]})
            nodes.append({"mesh": k})
        document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
                    "materials": self.materials, "accessors": accessors, "bufferViews": views, "buffers": [{"uri": "scene.bin", "byteLength": len(blob)}]}
        if self.extensions:
            document["extensionsUsed"] = sorted(self.extensions)
        (folder / "scene.bin").write_bytes(bytes(blob))
        path = folder / "scene.gltf"
        path.write_text(json.dumps(document))
        return path


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
    return ((0.5 + 0.5 * (point[0] - centre[0]) / (depth * tan_half * aspect)) * WIDTH,
            (0.5 - 0.5 * (point[1] - centre[1]) / (depth * tan_half)) * HEIGHT)


def region_pixels(rect, z, centre, distance):
    """Pixel box of a world rectangle (x0, y0, x1, y1) lying in the plane z."""
    x0, y0 = project((rect[0], rect[3], z), centre, distance)
    x1, y1 = project((rect[2], rect[1], z), centre, distance)
    return int(math.ceil(x0)), int(math.ceil(y0)), int(math.floor(x1)), int(math.floor(y1))


def set_flag(command, flag, value):
    """Replace the value of `flag` (or append the pair)."""
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command += [flag, value]
    return command


def capture(binary, scene, folder, flags=None, extra=None, name="fixture"):
    """One serial capture under the runner's GPU guard; returns the PNG path.  `flags` replaces declared flag values ({flag: value}),
    `extra` appends further arguments."""
    corpus.MODELS[name] = {"tier": "A", "view": (0, 0), "shows": "generated fixture"}
    corpus.gltf_path = lambda corpus_dir, n: scene
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    command = corpus.capture_command(binary, None, name, folder)
    for flag, value in (flags or {}).items():
        set_flag(command, flag, value)
    command += list(extra or [])
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


def luminance(image, x, y):
    _, _, channels, rows = image
    r, g, b = rows[y][x * channels:x * channels + 3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
