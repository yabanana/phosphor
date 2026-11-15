#!/usr/bin/env python3
"""F15.1 (ticket 22) per-slot KHR_texture_transform at the SECONDARY sites: GPU image against a CPU geometric reference.

tools/f15_texture_transform_check.py measures the primary paths.  This runner measures the sites that sample a material texture away from
the camera-visible surface, each with a fixture that FAILS on the build without the site change and passes with it (and a negative control
that must fail):

  rt-shadow     an alpha-masked quad (MASK, cut-out pattern in the alpha channel of the base colour texture, a per-slot transform on that slot and
                DIFFERENT transforms on two decoy slots) casts the sun's shadow on a floor behind it with --shadows rt (the RT any-hit alpha test);
  csm-shadow    the same scene with --shadows csm (the shadow-map alpha test, shadow_depth_fragment);
  cache-shadow  the same with --shadows csm --shadow-cache on (shadow_cache_depth_fragment);
  reflection    a palette-textured quad (base colour slot transformed, decoys on two other slots) seen in a mirror floor with --reflections rt
                (the RT hit surface, reflectionMaterial).

Reference (independent of the renderer): the planar geometry of the scene and the KHR_texture_transform README formula.  A floor pixel is
shadowed when the ray towards the sun (src/testbench/corpus_rig.h direction for yaw 0) crosses the caster at a cell whose alpha is 255; a mirror
pixel shows the palette cell the reflected camera ray hits.  Cells encode a 3-bit code (reflection: colour bits per channel against the
midpoint of that channel's range over the valid pixels; shadows: dark / lit against the midpoint of the luminance range).

Thresholds, fixed here before any comparison (a wrong transform maps a pixel to a random cell: agreement about 0.5 for a bit, 0.1-0.2 for a
3-bit code): agreement of the correct reference >= 0.99 (rt-shadow), >= 0.97 (csm-shadow, cache-shadow: the shadow-map texel and the PCF
footprint are wider than the edge margin of the other cases), >= 0.95 (reflection: the RT reflection is denoised); every negative control
(the same pixels against the transform of another slot = the uniform-bake result, against the identity) must stay below 0.90, otherwise the
fixture does not discriminate and the run exits 3.  Valid pixels keep >= 0.15 cell from a cell edge and the caster / reflection borders; the shadow cases count only floor pixels whose sun ray
crosses the caster (the others do not depend on the alpha test).

Modes: --selftest (CPU), --run (GPU lock: wrap with gpulock.py), --evaluate.  Exit codes: 0 ok, 1 gate failed, 2 usage, 3 non-discriminating
control, 6 capture failed.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import re
import struct
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_texture_transform_check as base  # noqa: E402

CASES = {
    "rt-shadow": {"shadows": ["--shadows", "rt"], "agreement": 0.99, "kind": "shadow"},
    "csm-shadow": {"shadows": ["--shadows", "csm"], "agreement": 0.97, "kind": "shadow"},
    "cache-shadow": {"shadows": ["--shadows", "csm", "--shadow-cache", "on"], "agreement": 0.97, "kind": "shadow"},
    "reflection": {"shadows": ["--shadows", "rt"], "agreement": 0.95, "kind": "reflection", "pitch": 30.0},
}
EDGE = 0.15
SIZE = 1.2  # caster / object size
FLOOR = (4.0, 3.0)  # shadow floor extent, centred, at z = FLOOR_Z
FLOOR_Z = -1.0
OBJECT_Z = -0.4
MIRROR_Y = -0.5
MIRROR_HALF = 1.5
MARGIN_PX = 6


def decorated(slot_transforms):
    return {slot: {"offset": t["offset"], "rotation": t["rotation"], "scale": t["scale"]} for slot, t in slot_transforms.items()}


# --- scene ----------------------------------------------------------------------------------------------------------------

def build_scene(path, case):
    kind = CASES[case]["kind"]
    images = {"alphatex": base.alpha_png(), "palette": base.palette_png("color"), "black": base.flat_png(0), "white": base.flat_png(255),
              "flatnormal": base.png_bytes(8, 8, lambda x, y: (128, 128, 255))}
    names = list(images)
    index = {n: i for i, n in enumerate(names)}

    def tex(name, transform):
        return {"index": index[name], "extensions": {"KHR_texture_transform": {"offset": transform["offset"], "rotation": transform["rotation"], "scale": transform["scale"]}}}

    T = base.TRANSFORMS
    caster_alpha = {
        "name": "caster-alpha", "alphaMode": "MASK", "alphaCutoff": 0.5,
        "pbrMetallicRoughness": {"baseColorTexture": tex("alphatex", T["alpha"]), "metallicFactor": 0.0, "roughnessFactor": 1.0},
        "emissiveTexture": tex("black", T["emissive"]), "emissiveFactor": [1, 1, 1], "normalTexture": tex("flatnormal", T["normal"]),
    }
    palette_object = {
        "name": "palette-object",
        "pbrMetallicRoughness": {"baseColorTexture": tex("palette", T["baseColor"]), "metallicFactor": 0.0, "roughnessFactor": 1.0},
        "emissiveTexture": tex("black", T["emissive"]), "emissiveFactor": [1, 1, 1], "normalTexture": tex("flatnormal", T["normal"]),
    }
    floor = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.9, 0.9, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    mirror = {"name": "mirror", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0, "roughnessFactor": 0.0}}
    h = SIZE / 2
    if kind == "shadow":
        # (material, origin = corner of (u, v) = (0, 0), U, V)
        quads = [(floor, (-FLOOR[0] / 2, FLOOR[1] / 2, FLOOR_Z), (FLOOR[0], 0, 0), (0, -FLOOR[1], 0)),
                 (caster_alpha, (-h, h, 0.0), (SIZE, 0, 0), (0, -SIZE, 0))]
    else:
        quads = [(mirror, (-MIRROR_HALF, MIRROR_Y, -MIRROR_HALF), (2 * MIRROR_HALF, 0, 0), (0, 0, 2 * MIRROR_HALF)),
                 (palette_object, (-h, MIRROR_Y + SIZE, OBJECT_Z), (SIZE, 0, 0), (0, -SIZE, 0))]
    blob = bytearray()
    views, accessors, meshes, nodes, materials = [], [], [], [], []

    def add(data, target_type, count, component, mins=None, maxs=None):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": count, "type": target_type}
        if mins:
            acc["min"], acc["max"] = mins, maxs
        accessors.append(acc)
        return len(accessors) - 1

    for qi, (material, o, U, V) in enumerate(quads):
        corners = [(0, 0), (0, 1), (1, 1), (1, 0)]
        pos, nor, uvs, tan = [], [], [], []
        N = (U[1] * -V[2] - U[2] * -V[1], U[2] * -V[0] - U[0] * -V[2], U[0] * -V[1] - U[1] * -V[0])  # cross(U, -V)
        ln = math.sqrt(sum(c * c for c in N))
        N = tuple(c / ln for c in N)
        lu = math.sqrt(sum(c * c for c in U))
        for (u, v) in corners:
            p = tuple(o[i] + U[i] * u + V[i] * v for i in range(3))
            pos += list(p)
            nor += list(N)
            uvs += [float(u), float(v)]
            tan += [U[0] / lu, U[1] / lu, U[2] / lu, 1.0]
        mins = [min(pos[i::3]) for i in range(3)]
        maxs = [max(pos[i::3]) for i in range(3)]
        a_pos = add(struct.pack("<12f", *pos), "VEC3", 4, 5126, mins, maxs)
        a_nor = add(struct.pack("<12f", *nor), "VEC3", 4, 5126)
        a_uv = add(struct.pack("<8f", *uvs), "VEC2", 4, 5126)
        a_tan = add(struct.pack("<16f", *tan), "VEC4", 4, 5126)
        a_idx = add(struct.pack("<6H", 0, 1, 2, 0, 2, 3), "SCALAR", 6, 5123)
        materials.append(material)
        meshes.append({"primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nor, "TEXCOORD_0": a_uv, "TANGENT": a_tan}, "indices": a_idx, "material": qi}]})
        nodes.append({"mesh": qi, "name": material["name"]})
    doc = {
        "asset": {"version": "2.0", "generator": "tools/f15_texture_transform_secondary.py"}, "extensionsUsed": ["KHR_texture_transform"],
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": materials,
        "textures": [{"source": i, "sampler": 0} for i in range(len(names))], "images": [{"uri": base.data_uri(images[n])} for n in names],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode(), "byteLength": len(blob)}],
        "bufferViews": views, "accessors": accessors,
    }
    pathlib.Path(path).write_text(json.dumps(doc))


# --- camera ---------------------------------------------------------------------------------------------------------------

class Camera:
    """The corpus rig for yaw 0 and a pitch: the target is the centre of the world bounds, the camera sits `distance` along `back`."""

    def __init__(self, bounds_min, bounds_max, distance, pitch_degrees, width, height):
        self.width, self.height = width, height
        self.aspect = width / height
        p = math.radians(pitch_degrees)
        self.back = (0.0, math.sin(p), math.cos(p))
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, math.cos(p), -math.sin(p))  # cross(back, right)
        centre = tuple(0.5 * (bounds_min[i] + bounds_max[i]) for i in range(3))
        self.position = tuple(centre[i] + self.back[i] * distance for i in range(3))

    def ray(self, px, py):
        nx = (px + 0.5) / self.width * 2 - 1
        ny = 1 - (py + 0.5) / self.height * 2
        d = tuple(-self.back[i] + nx * base.FOV_HALF_TAN * self.aspect * self.right[i] + ny * base.FOV_HALF_TAN * self.up[i] for i in range(3))
        n = math.sqrt(sum(c * c for c in d))
        return tuple(c / n for c in d)

    def project(self, point):
        rel = tuple(point[i] - self.position[i] for i in range(3))
        depth = -sum(rel[i] * self.back[i] for i in range(3))
        x = sum(rel[i] * self.right[i] for i in range(3)) / (depth * base.FOV_HALF_TAN * self.aspect)
        y = sum(rel[i] * self.up[i] for i in range(3)) / (depth * base.FOV_HALF_TAN)
        return (x * 0.5 + 0.5) * self.width, (0.5 - y * 0.5) * self.height


def parse_rig(log_text):
    m = re.search(r"CORPUS_RIG: \d+ instances, bounds min \(([-\d. e+]+)\) max \(([-\d. e+]+)\), camera distance ([\d.e+-]+)", log_text)
    if not m:
        raise RuntimeError("no CORPUS_RIG line in the log")
    lo = tuple(float(v) for v in m.group(1).split())
    hi = tuple(float(v) for v in m.group(2).split())
    return lo, hi, float(m.group(3))


# --- reference -------------------------------------------------------------------------------------------------------------

def shadow_expectation(cam, px, py, transform):
    """None (not a valid pixel) or True when the floor pixel is shadowed by the caster (alpha 255 at the cell the sun ray crosses)."""
    d = cam.ray(px, py)
    t = (FLOOR_Z - cam.position[2]) / d[2]
    if not t > 0:
        return None
    P = tuple(cam.position[i] + d[i] * t for i in range(3))
    if abs(P[0]) > FLOOR[0] / 2 - 0.05 or abs(P[1]) > FLOOR[1] / 2 - 0.05:
        return None
    s = (0.0 - P[2]) / base.SUN[2]
    Q = tuple(P[i] + base.SUN[i] * s for i in range(3))
    u = (Q[0] + SIZE / 2) / SIZE
    v = (SIZE / 2 - Q[1]) / SIZE
    return u, v


def cell_bit(u, v, transform):
    tu, tv = base.transform_uv(transform, u, v)
    if base.edge_distance(tu * base.CELLS, tv * base.CELLS) < EDGE:
        return None
    col, row, _, _ = base.cell_of(tu, tv)
    return base.code_at(col, row)


def evaluate_shadow(img, cam, candidates):
    width, height = img.size
    pix = img.load()
    # the caster's screen footprint (excluded: the camera sees the caster or the floor through its holes there)
    xs, ys = zip(*[cam.project((sx * SIZE / 2, sy * SIZE / 2, 0.0)) for sx in (-1, 1) for sy in (-1, 1)])
    box = (min(xs) - MARGIN_PX, min(ys) - MARGIN_PX, max(xs) + MARGIN_PX, max(ys) + MARGIN_PX)
    samples = []
    for py in range(0, height, 2):
        for px in range(0, width, 2):
            if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                continue
            e = shadow_expectation(cam, px, py, None)
            if e is None:
                continue
            u, v = e
            # only floor pixels whose sun ray crosses the caster count: the lit pixels outside it do not depend on the alpha test and would
            # make every control agree with the image (measured on the build without the site change: 0.98 for any transform)
            inside = True
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0) or min(u, 1 - u, v, 1 - v) < EDGE / 2:
                continue
            if cell_bit(u, v, candidates["reference"]) is None:
                continue
            samples.append((px, py, u, v, inside))
    if not samples:
        raise RuntimeError("no valid pixel")
    lumas = [sum(pix[px, py][:3]) / 3.0 for px, py, _, _, _ in samples]
    mid = (min(lumas) + max(lumas)) / 2.0
    agree = {name: 0 for name in candidates}
    counted = {name: 0 for name in candidates}
    for (px, py, u, v, inside), luma in zip(samples, lumas):
        dark = luma < mid
        for name, t in candidates.items():
            code = cell_bit(u, v, t)
            want = bool(code & 1) if code is not None else False  # a cell-edge pixel of a CONTROL transform counts as lit
            counted[name] += 1
            agree[name] += int(want == dark)
    return {"valid_pixels": len(samples), "luma_range": [min(lumas), max(lumas)],
            "shadowed_expected_fraction": round(sum(1 for s in samples if s[4] and (cell_bit(s[2], s[3], candidates["reference"]) or 0) & 1) / len(samples), 3),
            "agreement": {n: round(agree[n] / counted[n], 4) for n in candidates}}


def evaluate_reflection(img, cam, candidates):
    width, height = img.size
    pix = img.load()
    samples = []
    for py in range(0, height, 2):
        for px in range(0, width, 2):
            d = cam.ray(px, py)
            if d[1] >= 0:
                continue
            t = (MIRROR_Y - cam.position[1]) / d[1]
            if not t > 0:
                continue
            M = tuple(cam.position[i] + d[i] * t for i in range(3))
            if abs(M[0]) > MIRROR_HALF - 0.05 or abs(M[2]) > MIRROR_HALF - 0.05 or M[2] < OBJECT_Z + 0.08:
                continue
            # the camera ray must reach the mirror before the object quad
            tc = (OBJECT_Z - cam.position[2]) / d[2]
            if tc > 0 and tc < t:
                H = tuple(cam.position[i] + d[i] * tc for i in range(3))
                if abs(H[0]) <= SIZE / 2 + 0.05 and MIRROR_Y - 0.05 <= H[1] <= MIRROR_Y + SIZE + 0.05:
                    continue
            r = (d[0], -d[1], d[2])
            if not r[2] < -1e-6:
                continue
            s = (OBJECT_Z - M[2]) / r[2]
            H = tuple(M[i] + r[i] * s for i in range(3))
            u = (H[0] + SIZE / 2) / SIZE
            v = (MIRROR_Y + SIZE - H[1]) / SIZE
            if not (EDGE / 2 < u < 1 - EDGE / 2 and EDGE / 2 < v < 1 - EDGE / 2):
                continue
            if cell_bit(u, v, candidates["reference"]) is None:
                continue
            samples.append((px, py, u, v, pix[px, py]))
    if not samples:
        raise RuntimeError("no valid pixel")
    lo, hi = [255] * 3, [0] * 3
    for *_, rgb in samples:
        for c in range(3):
            lo[c], hi[c] = min(lo[c], rgb[c]), max(hi[c], rgb[c])
    agree = {name: 0 for name in candidates}
    for _, _, u, v, rgb in samples:
        seen = sum(int(rgb[c] > (lo[c] + hi[c]) / 2.0) << (2 - c) for c in range(3))
        for name, t in candidates.items():
            code = cell_bit(u, v, t)
            agree[name] += int(code is not None and code == seen) if code is not None else 0
    return {"valid_pixels": len(samples), "range": {"lo": lo, "hi": hi}, "agreement": {n: round(agree[n] / len(samples), 4) for n in candidates}}


def candidates_for(case):
    T = base.TRANSFORMS
    if CASES[case]["kind"] == "shadow":
        ref = T["alpha"]
        others = {"uniform-bake:baseColor-transform": T["baseColor"], "uniform-bake:emissive-transform": T["emissive"], "uniform-bake:normal-transform": T["normal"]}
    else:
        ref = T["baseColor"]
        others = {"uniform-bake:emissive-transform": T["emissive"], "uniform-bake:normal-transform": T["normal"], "uniform-bake:alpha-transform": T["alpha"]}
    out = {"reference": ref, "identity": base.IDENTITY}
    out.update(others)
    return out


def evaluate(case, capture, log_text):
    from PIL import Image
    img = Image.open(capture).convert("RGB")
    lo, hi, distance = parse_rig(log_text)
    cam = Camera(lo, hi, distance, CASES[case].get("pitch", 0.0), *img.size)
    cands = candidates_for(case)
    result = evaluate_shadow(img, cam, cands) if CASES[case]["kind"] == "shadow" else evaluate_reflection(img, cam, cands)
    result["case"] = case
    result["threshold"] = CASES[case]["agreement"]
    result["gate_ok"] = result["agreement"]["reference"] >= CASES[case]["agreement"]
    result["controls_discriminate"] = all(v < base.CONTROL_MAX for n, v in result["agreement"].items() if n != "reference")
    return result


# --- self test -------------------------------------------------------------------------------------------------------------

def selftest():
    import tempfile
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "scene.gltf"
            build_scene(path, case)
            doc = json.loads(path.read_text())
            assert len(doc["meshes"]) == 2 and len(doc["materials"]) == 2
            for acc in doc["accessors"]:
                assert doc["bufferViews"][acc["bufferView"]]["byteOffset"] % 4 == 0
    # projection round trip of the camera (pitch 30)
    cam = Camera((-1.5, -0.5, -1.5), (1.5, 0.7, 1.5), 4.0, 30.0, 1024, 768)
    point = (0.2, -0.1, 0.3)
    px, py = cam.project(point)
    d = cam.ray(px - 0.5, py - 0.5)
    rel = tuple(point[i] - cam.position[i] for i in range(3))
    n = math.sqrt(sum(c * c for c in rel))
    assert all(abs(d[i] - rel[i] / n) < 1e-9 for i in range(3)), (d, rel)
    # the shadow geometry: the sun ray from a floor point reaches the caster plane
    u, v = shadow_expectation(Camera((-2, -1.5, -1), (2, 1.5, 0), 5.0, 0.0, 1024, 768), 512, 384, None)
    assert u is not None
    # every control transform differs from the reference on at least 60 % of a sample grid (the fixture discriminates by construction)
    for case in CASES:
        c = candidates_for(case)
        ref = c["reference"]
        for name, t in c.items():
            if name == "reference":
                continue
            moved = total = 0
            for i in range(20):
                for j in range(20):
                    a = base.cell_of(*base.transform_uv(ref, (i + .5) / 20, (j + .5) / 20))[:2]
                    b = base.cell_of(*base.transform_uv(t, (i + .5) / 20, (j + .5) / 20))[:2]
                    total += 1
                    moved += a != b
            assert moved / total > 0.6, (case, name, moved / total)
    print("selftest ok")


def capture_command(binary, gltf, out, case, no_archive=False, harvest=None):
    cfg = CASES[case]
    cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--corpus-rig", f"0,{cfg.get('pitch', 0.0)}",
           "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", *cfg["shadows"], "--lighting", "restir",
           "--gi", "off", "--reflections", cfg.get("reflections", "rt"), "--ao", "off", "--lighting-denoise", "custom",
           "--post", "--tonemap", "aces", "--upscaler", "native", "--resolution", f"{base.RESOLUTION[0]}x{base.RESOLUTION[1]}",
           "--warmup", "48", "--frames", "48", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--pipeline-sync",
           "--capture", str(out / "engine.png"), "--report", str(out / "report.json")]
    if no_archive:
        cmd.append("--no-pipeline-archive")
    if harvest:
        cmd += ["--harvest-pipelines", harvest]
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--case", choices=sorted(CASES), action="append")
    p.add_argument("--run", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--binary")
    p.add_argument("--out", default="build/f15-texxform-secondary")
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--harvest", metavar="DIR", help="record the pipelines of every case as DIR/<case>.raw.json")
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not (a.run or a.evaluate):
        p.print_help()
        return 2
    status = 0
    for case in a.case or sorted(CASES):
        out = pathlib.Path(a.out) / case
        out.mkdir(parents=True, exist_ok=True)
        gltf = out / "scene.gltf"
        log = out / "engine.log"
        if a.run:
            if not a.binary:
                print("--run needs --binary", file=sys.stderr)
                return 2
            build_scene(gltf, case)
            harvest = None
            if a.harvest:
                pathlib.Path(a.harvest).mkdir(parents=True, exist_ok=True)
                harvest = str(pathlib.Path(a.harvest) / f"{case}.raw.json")
            with log.open("w") as stream:
                code = subprocess.run(capture_command(a.binary, gltf, out, case, a.no_archive, harvest), stdout=stream, stderr=subprocess.STDOUT).returncode
            text = log.read_text(errors="replace")
            if code != 0 or not (out / "engine.png").is_file() or not re.search(r"^EXIT 0$", text, re.M):
                print(f"{case}: capture failed (exit {code}); see {log}", file=sys.stderr)
                return 6
        result = evaluate(case, out / "engine.png", log.read_text(errors="replace"))
        (out / "result.json").write_text(json.dumps(result, indent=1))
        print(f"{case}: valid {result['valid_pixels']} agreement {result['agreement']} threshold {result['threshold']}")
        if not result["controls_discriminate"]:
            status = max(status, 3)
        elif not result["gate_ok"]:
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
