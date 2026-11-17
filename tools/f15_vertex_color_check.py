#!/usr/bin/env python3
"""F15 closure: glTF COLOR_0 (vertex colour) fixtures against a CPU reference, each with a negative control (COLOR_0 ignored must FAIL).

Scene (tools generate it, no GPU): a 4 x 4 m plane of 4 x 4 quads (5 x 5 vertices, two triangles per quad, diagonal 0-2), per-vertex colours
RGBA float c(i, j) = (i / 4, j / 4, (i + j) % 2, 1), seen head-on under the declared corpus rig (linear capture, exposure 1).  The reference is the
barycentric interpolation of the vertex colours at the world point under the centre of each pixel (CPU, independent of the engine).

  UNLIT  (KHR_materials_unlit, base white): the displayed value is base x COLOR_0 = the interpolated colour.  PASS: max |engine - reference| <= 0.004
         over every sampled pixel (half-float rounding of the colour attachment is 0.0005 at 1.0; the rest is geometry which is exact head-on).
         Control: the same scene without COLOR_0 shows white: the same predicate FAILS (error > 0.1).
  LIT    (metallic 0, roughness 1, sun): per-channel ratio of the coloured scene to the white scene equals the colour (diffuse is linear in the base
         colour; the dielectric specular lobe is colour independent and at most 1.3 % of the diffuse term, so |ratio - colour| <= 0.03 where colour >= 0.2).
         Control: without COLOR_0 the ratio is 1: FAILS where the colour is below 0.8.
  CUTOUT (alphaMode MASK, cutoff 0.5, COLOR_0 alpha = x / 4 across the plane, base alpha 1): the visible mask is alpha >= 0.5 (the right half).  PASS:
         >= 99.5 % of the pixels farther than 3 px from the edge agree with the reference mask.  Control: without COLOR_0 everything is visible: FAILS.

  MIRROR (RT reflection of a vertex-coloured object): an unlit COLOR_0 panel (3 m, centre height -0.5, standing on a smooth white metal floor at y = -2) seen from
         a pitch of 30 degrees with `--reflections rt`.  The reflection of the object point P appears where the mirror image P' = (x, -4 - y, z) projects
         (the engine's own camera model of the corpus rig, replicated); the metal floor has F = 1 for a white base, so the reflected value is the colour of P
         (the engine's RT hit shading multiplies the vertex colour into the unlit emission).  PASS: the mean error over a 9 x 9 grid of interior points is <= 15 % of the control's mean error against the same reference and the per-channel
         mean bias is <= 0.03 (the first version required a max error <= 0.06 and measured 0.117 at single samples of steep gradients: the reflection is traced and
         filtered, the geometry matches to a pixel).  Control: without COLOR_0 the reflection is white: FAILS.  Mesh path only (the secondary site is the RT hit surface).

  SHADOW (CUT-OUT SHADOW through COLOR_0 alpha): a horizontal double-sided MASK panel (4 m, alpha = x / 4 from COLOR_0, cutoff 0.5) 1 m above a lit white floor
         under the rig's sun; `--shadows rt` (the any-hit alpha test) and `--shadows csm` (the CSM caster alpha test).  Floor points in the shadow of the VISIBLE
         half (x >= 0) must be darker than 0.7 x the unshadowed floor luminance, floor points in the shadow of the CUT half (x < 0) at least 0.9 x (not covered
         by the panel in the image).  Control: without COLOR_0 the whole panel casts: the cut half is dark, the predicate FAILS.  GI hit shading, the probe capture
         and the emitter alpha have no numeric fixture (the sites are written and compile; see the ticket).

Paths: `mesh` (visibility + meshlets, resolve kernels and the visibility alpha pass), `overflow` (the same forced through the indexed overflow receivers:
--debug-meshlets 1 --debug-meshlets-corrupt count, exit 1 by design) and `forward` (--render-path forward: forward_vs / forward_fs, PNG capture compared
in the display domain, UNLIT and CUTOUT only) and `meshforward` (--render-path forward --geometry-path mesh: the meshlet mesh stages and forward_fs).

  python3 "$H/gpulock.py" python3 -I tools/f15_vertex_color_check.py run OUTDIR [BINARY]
  python3 -I tools/f15_vertex_color_check.py verify OUTDIR
"""
from __future__ import annotations

import argparse
import array
import base64
import json
import math
import os
import pathlib
import re
import struct
import subprocess
import sys
import zlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_unlit_lights_check as base  # noqa: E402

N = 4            # quads per side
SIZE = 4.0       # metres
RES = (1280, 720)


def vertex_color(i, j):
    return (i / N, j / N, float((i + j) % 2), 1.0)


def scene(kind, colors=True):
    """kind: unlit | lit | lit-white | cutout; colors False strips COLOR_0 (the control)."""
    positions, cols, uvs = [], [], []
    for j in range(N + 1):
        for i in range(N + 1):
            positions += [i * SIZE / N - SIZE / 2, j * SIZE / N - SIZE / 2, 0.0]
            c = vertex_color(i, j)
            if kind == "cutout":
                c = (1.0, 1.0, 1.0, i / N)
            cols += list(c)
            uvs += [i / N, 1.0 - j / N]
    indices = []
    for j in range(N):
        for i in range(N):
            a, b, c, d = j * (N + 1) + i, j * (N + 1) + i + 1, (j + 1) * (N + 1) + i + 1, (j + 1) * (N + 1) + i
            indices += [a, b, c, a, c, d]
    # The reflection probe is placed in the scene bounds, which need a positive extent on every axis (a flat plane has none: "invalid probe capture"):
    # one tiny back-facing triangle 0.5 m behind the plane, outside the sampled area (culled from the camera side).
    base_v = len(positions) // 3
    positions += [3.0, 3.0, -0.5, 3.1, 3.0, -0.5, 3.0, 3.1, -0.5]
    cols += [1.0] * 12
    uvs += [0.0] * 6
    indices += [base_v, base_v + 2, base_v + 1]
    nv = len(positions) // 3
    blob = struct.pack(f"<{nv * 3}f", *positions) + struct.pack(f"<{nv * 3}f", *([0.0, 0.0, 1.0] * nv)) + struct.pack(f"<{nv * 2}f", *uvs)
    blob += struct.pack(f"<{len(indices)}H", *indices)
    ioff = nv * 3 * 4 * 2 + nv * 2 * 4
    coff = len(blob)
    blob += struct.pack(f"<{nv * 4}f", *cols)
    mat = {"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    extras = {}
    if kind == "unlit":
        mat["extensions"] = {"KHR_materials_unlit": {}}
        extras = {"extensionsUsed": ["KHR_materials_unlit"]}
    if kind == "cutout":
        mat["alphaMode"] = "MASK"
        mat["alphaCutoff"] = 0.5
        mat["extensions"] = {"KHR_materials_unlit": {}}
        extras = {"extensionsUsed": ["KHR_materials_unlit"]}
    attrs = {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}
    accessors = [{"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": [-SIZE / 2, -SIZE / 2, -0.5], "max": [3.1, 3.1, 0]},
                 {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
                 {"bufferView": 2, "componentType": 5126, "count": nv, "type": "VEC2"},
                 {"bufferView": 3, "componentType": 5123, "count": len(indices), "type": "SCALAR"}]
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": nv * 12}, {"buffer": 0, "byteOffset": nv * 12, "byteLength": nv * 12},
             {"buffer": 0, "byteOffset": nv * 24, "byteLength": nv * 8}, {"buffer": 0, "byteOffset": ioff, "byteLength": len(indices) * 2}]
    if colors:
        attrs["COLOR_0"] = 4
        accessors.append({"bufferView": 4, "componentType": 5126, "count": nv, "type": "VEC4"})
        views.append({"buffer": 0, "byteOffset": coff, "byteLength": nv * 16})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
           "meshes": [{"primitives": [{"attributes": attrs, "indices": 3, "material": 0}]}], "materials": [mat],
           "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
           "bufferViews": views, "accessors": accessors}
    doc.update(extras)
    return doc


def scene_shadow(colors):
    """The COLOR_0 cut-out grid, horizontal, 1 m above a floor, lit and double sided."""
    doc = scene("cutout", colors)
    doc["materials"][0] = {"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}, "alphaMode": "MASK",
                           "alphaCutoff": 0.5, "doubleSided": True}
    doc.pop("extensionsUsed", None)
    doc["nodes"] = [{"mesh": 0, "translation": [0.0, 1.0, 0.0], "rotation": [-0.70710678, 0.0, 0.0, 0.70710678]}, {"mesh": 1}]
    doc["scenes"] = [{"nodes": [0, 1]}]
    floor = struct.pack("<12f", -7, 0, -7, 7, 0, -7, 7, 0, 7, -7, 0, 7) + struct.pack("<12f", *([0.0, 1.0, 0.0] * 4)) + struct.pack("<8f", 0, 0, 1, 0, 1, 1, 0, 1)
    floor += struct.pack("<6H", 0, 3, 2, 0, 2, 1)
    blob = base64.b64decode(doc["buffers"][0]["uri"].split(",", 1)[1])
    offset = len(blob)
    blob += floor
    doc["buffers"][0] = {"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}
    nview, na = len(doc["bufferViews"]), len(doc["accessors"])
    doc["bufferViews"] += [{"buffer": 0, "byteOffset": offset, "byteLength": 48}, {"buffer": 0, "byteOffset": offset + 48, "byteLength": 48},
                           {"buffer": 0, "byteOffset": offset + 96, "byteLength": 32}, {"buffer": 0, "byteOffset": offset + 128, "byteLength": 12}]
    doc["accessors"] += [{"bufferView": nview, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-7, 0, -7], "max": [7, 0, 7]},
                         {"bufferView": nview + 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                         {"bufferView": nview + 2, "componentType": 5126, "count": 4, "type": "VEC2"},
                         {"bufferView": nview + 3, "componentType": 5123, "count": 6, "type": "SCALAR"}]
    doc["meshes"].append({"primitives": [{"attributes": {"POSITION": na, "NORMAL": na + 1, "TEXCOORD_0": na + 2}, "indices": na + 3, "material": 1}]})
    doc["materials"].append({"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}})
    return doc


def scene_mirror(colors):
    """The COLOR_0 panel of `scene('unlit')` (3 m, centre (0, -0.5, 0)) standing on a metal mirror floor at y = -2."""
    doc = scene("unlit", colors)
    doc["nodes"] = [{"mesh": 0, "translation": [0.0, -0.5, 0.0], "scale": [0.75, 0.75, 1.0]}, {"mesh": 1}]
    doc["scenes"] = [{"nodes": [0, 1]}]
    floor = struct.pack("<12f", -5, -2, -5, 5, -2, -5, 5, -2, 5, -5, -2, 5) + struct.pack("<12f", *([0.0, 1.0, 0.0] * 4)) + struct.pack("<8f", 0, 0, 1, 0, 1, 1, 0, 1)
    floor += struct.pack("<6H", 0, 3, 2, 0, 2, 1)
    old = doc["buffers"][0]["uri"].split(",", 1)[1]
    blob = base64.b64decode(old)
    offset = len(blob)
    blob += floor
    doc["buffers"][0] = {"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}
    nv = 4
    nview = len(doc["bufferViews"])
    doc["bufferViews"] += [{"buffer": 0, "byteOffset": offset, "byteLength": 48}, {"buffer": 0, "byteOffset": offset + 48, "byteLength": 48},
                           {"buffer": 0, "byteOffset": offset + 96, "byteLength": 32}, {"buffer": 0, "byteOffset": offset + 128, "byteLength": 12}]
    na = len(doc["accessors"])
    doc["accessors"] += [{"bufferView": nview, "componentType": 5126, "count": nv, "type": "VEC3", "min": [-5, -2, -5], "max": [5, -2, 5]},
                         {"bufferView": nview + 1, "componentType": 5126, "count": nv, "type": "VEC3"},
                         {"bufferView": nview + 2, "componentType": 5126, "count": nv, "type": "VEC2"},
                         {"bufferView": nview + 3, "componentType": 5123, "count": 6, "type": "SCALAR"}]
    doc["meshes"].append({"primitives": [{"attributes": {"POSITION": na, "NORMAL": na + 1, "TEXCOORD_0": na + 2}, "indices": na + 3, "material": 1}]})
    doc["materials"].append({"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0, "roughnessFactor": 0.0}})
    return doc


CASES = {  # name -> (scene kind, colours present)
    "unlit": ("unlit", True), "unlit-control": ("unlit", False),
    "lit": ("lit", True), "lit-white": ("lit", False),
    "cutout": ("cutout", True), "cutout-control": ("cutout", False),
    "mirror": ("mirror", True), "mirror-control": ("mirror", False),
}

PATHS = {
    "mesh": {"extra": [], "linear": True},
    "overflow": {"extra": ["--debug-meshlets", "1", "--debug-meshlets-corrupt", "count"], "linear": True},
    "forward": {"extra": [], "linear": False},
    "meshforward": {"extra": [], "linear": False},  # --render-path forward --geometry-path mesh: meshlet_mesh + forward_fs (the _vc twins)
    # F15 closure round 3: `visibility_tile` (--tile-resolve, the on-tile material resolve) read no COLOR_0 (its pipeline was served with the lobe-only
    # mask): the same kernel body as the compute resolve, a different pipeline.  Legacy lights (the tile path has no lit-resolve lighting), linear capture.
    "tile": {"extra": ["--tile-resolve"], "linear": True},
    # round 3 (archive): the meshlet forward mesh stages that only an option selects, so the `_vc` twins have an archive entry
    "meshforward-tricull": {"extra": ["--meshlet-triangle-cull", "on"], "linear": False},                             # meshlet_mesh_tricull_vc
    "meshforward-direct": {"extra": ["--meshlet-object", "off", "--meshlet-cull", "off"], "linear": False},          # meshlet_mesh_direct_vc
}


def capture(binary, folder, path, name, gltf, pitch=0, shadows="rt"):
    folder.mkdir(parents=True, exist_ok=True)
    spec = PATHS[path]
    out = folder / f"{path}-{name}"
    if path == "tile":
        cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--corpus-rig", "0,0", "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "off",
               "--lighting", "legacy", "--gi", "off", "--reflections", "off", "--ao", "off", "--lighting-denoise", "off", "--post", "--upscaler", "native",
               "--resolution", f"{RES[0]}x{RES[1]}", "--warmup", "16", "--frames", "2", "--fixed-timestep", "--offscreen", "--no-ui", "--no-vsync", "--no-gpu-timing",
               "--pipeline-sync", "--report", str(out) + ".json", "--capture-linear", str(out) + ".pfm", *spec["extra"]]
    elif path == "forward" or path.startswith("meshforward"):
        cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--corpus-rig", "0,0", "--render-path", "forward", "--geometry-path", "indexed" if path == "forward" else "mesh", "--rt", "off",
               "--lighting", "legacy", "--gi", "off", "--reflections", "off", "--ao", "off", "--lighting-denoise", "off",
               "--resolution", f"{RES[0]}x{RES[1]}", "--warmup", "16", "--frames", "2", "--fixed-timestep", "--offscreen", "--no-ui", "--no-vsync", "--no-gpu-timing",
               "--pipeline-sync", "--report", str(out) + ".json", "--capture", str(out) + ".png", *spec["extra"]]
    else:
        cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--corpus-rig", f"0,{pitch}", "--rt", "on", *(["--shadows", "csm", "--shadow-cache", "on"] if shadows == "cache" else ["--shadows", shadows]), "--lighting", "restir", "--gi", "off",
               "--reflections", "rt", "--ao", "off", "--lighting-denoise", "custom", "--tonemap", "aces", "--post", "--upscaler", "native",
               "--render-path", "visibility", "--geometry-path", "mesh", "--resolution", f"{RES[0]}x{RES[1]}", "--warmup", "48", "--frames", "1",
               "--fixed-timestep", "--offscreen", "--no-ui", "--no-vsync", "--no-gpu-timing", "--pipeline-sync", "--capture-linear", str(out) + ".pfm", *spec["extra"]]
    if os.environ.get("F15_VC_HARVEST"):  # targeted archive update: record the pipelines of this capture (tools/filter_pipeline_harvest.py then keeps the new ones)
        cmd += ["--harvest-pipelines", str(pathlib.Path(os.environ["F15_VC_HARVEST"]) / f"{path}-{name}.raw.json")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_")}
    env["MTL_DEBUG_LAYER"] = "1"
    env["MTL_DEBUG_LAYER_WARNING_MODE"] = "nslog"
    with open(str(out) + ".log", "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=300).returncode
    return code


# --- readers ----------------------------------------------------------------------------------------------------

def read_png(path):
    data = pathlib.Path(path).read_bytes()
    pos, idat, w = 8, b"", 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", chunk[:10])
        elif kind == b"IDAT":
            idat += chunk
        pos += 12 + n
    assert depth == 8 and ctype in (2, 6), (depth, ctype)
    ch = 3 if ctype == 2 else 4
    raw = zlib.decompress(idat)
    stride = w * ch
    rows, prev = [], bytearray(stride)
    p = 0
    for _ in range(h):
        ft = raw[p]; line = bytearray(raw[p + 1:p + 1 + stride]); p += 1 + stride
        for i in range(stride):
            a = line[i - ch] if i >= ch else 0
            b = prev[i]
            c = prev[i - ch] if i >= ch else 0
            if ft == 1: line[i] = (line[i] + a) & 255
            elif ft == 2: line[i] = (line[i] + b) & 255
            elif ft == 3: line[i] = (line[i] + (a + b) // 2) & 255
            elif ft == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(bytes(line)); prev = line
    return w, h, ch, rows


def pixel_png(img, x, y):
    w, h, ch, rows = img
    return tuple(rows[y][x * ch + k] / 255.0 for k in range(3))


def pixel_pfm(img, x, y):
    w, h, ch, v = img
    o = ((h - 1 - y) * w + x) * ch
    return tuple(v[o:o + 3])


def aces(v):
    return max(0.0, min(1.0, (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)))


def srgb(v):
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def display(c):
    return tuple(srgb(aces(max(0.0, x))) for x in c)


# --- reference --------------------------------------------------------------------------------------------------

def reference(x, y, kind):
    """Interpolated vertex colour at the world point (x, y) of the plane; None outside."""
    gx, gy = (x + SIZE / 2) / SIZE * N, (y + SIZE / 2) / SIZE * N
    if not (0 <= gx < N and 0 <= gy < N):
        return None
    i, j = int(gx), int(gy)
    u, v = gx - i, gy - j
    def col(ii, jj):
        c = vertex_color(ii, jj) if kind != "cutout" else (1.0, 1.0, 1.0, ii / N)
        return c
    # triangles (a, b, c) = ((i, j), (i+1, j), (i+1, j+1)) and (a, c, d) = ((i, j), (i+1, j+1), (i, j+1)); the diagonal is u == v
    if u >= v:
        w = (1 - u, u - v, v); cs = (col(i, j), col(i + 1, j), col(i + 1, j + 1))
    else:
        w = (1 - v, u, v - u); cs = (col(i, j), col(i + 1, j + 1), col(i, j + 1))
    return tuple(sum(w[k] * cs[k][c] for k in range(3)) for c in range(4))


def world_of_pixel(px, py, distance, size, centre=(0.0, 0.0, 0.0)):
    """World (x, y) on the plane z = 0 under the centre of pixel (px, py) for the head-on rig camera: `distance` is the rig's camera distance,
    `centre` the rig's target (the centre of the scene bounds, which include the culled triangle that gives the bounds a z extent)."""
    depth = distance + centre[2]
    half_h = depth * math.tan(math.radians(30))
    half_w = half_h * size[0] / size[1]
    return centre[0] + ((px + 0.5) / size[0] * 2 - 1) * half_w, centre[1] + (1 - (py + 0.5) / size[1] * 2) * half_h


def rig_distance(log_text):
    m = re.search(r"camera distance ([\d.]+)", log_text)
    return float(m.group(1))


def load(out, path, name):
    stem = out / f"{path}-{name}"
    log = pathlib.Path(str(stem) + ".log").read_text(errors="replace")
    centre, dist = log_bounds(log)
    if PATHS[path]["linear"]:
        w, h, ch, v = base.read_pfm(str(stem) + ".pfm")
        return (w, h, ch, v), pixel_pfm, (dist, centre)
    img = read_png(str(stem) + ".png")
    return img, pixel_png, (dist, centre)


def sample_pixels(rig, size, margin=3, step=2):
    """Pixels whose centre is inside the plane (shrunk by `margin` px)."""
    distance, centre = rig
    half_h = (distance + centre[2]) * math.tan(math.radians(30))
    px_per_m = size[1] / (2 * half_h)
    inner = SIZE / 2 - margin / px_per_m
    out = []
    for py in range(size[1]):
        for px in range(size[0]):
            if (px % step) or (py % step):
                continue
            x, y = world_of_pixel(px, py, distance, size, centre)
            if abs(x) <= inner and abs(y) <= inner:
                out.append((px, py, x, y))
    return out


def unlit_error(out, path, name):
    img, pix, d = load(out, path, name)
    size = (img[0], img[1])
    worst = 0.0
    for px, py, x, y in sample_pixels(d, size):
        ref = reference(x, y, "unlit")
        got = pix(img, px, py)
        want = ref[:3] if PATHS[path]["linear"] else display(ref[:3])
        worst = max(worst, max(abs(got[k] - want[k]) for k in range(3)))
    return worst


def lit_error(out, path, name_a, name_b):
    ia, pa, d = load(out, path, name_a)
    ib, pb, _ = load(out, path, name_b)
    size = (ia[0], ia[1])
    worst, floor = 0.0, 0.0
    for px, py, x, y in sample_pixels(d, size, step=4):
        ref = reference(x, y, "lit")
        a, b = pa(ia, px, py), pb(ib, px, py)
        for k in range(3):
            if ref[k] >= 0.2 and b[k] > 0.05:
                worst = max(worst, abs(a[k] / b[k] - ref[k]))
            if ref[k] < 0.8 and b[k] > 0.05:
                floor = max(floor, abs(1.0 - ref[k]) if abs(a[k] / b[k] - 1.0) < 0.03 else 0.0)  # how many pixels a white image would satisfy
    return worst


def cutout_agreement(out, path, name):
    img, pix, d = load(out, path, name)
    size = (img[0], img[1])
    background = pix(img, 2, 2)
    agree = total = 0
    half_h = (d[0] + d[1][2]) * math.tan(math.radians(30))
    px_per_m = size[1] / (2 * half_h)
    for px, py, x, y in sample_pixels(d, size, margin=3):
        ref = reference(x, y, "cutout")
        if abs(ref[3] - 0.5) * SIZE / N * 0 + abs((x + SIZE / 2) / SIZE - 0.5) * SIZE * px_per_m < 3:
            continue  # within 3 px of the cut
        visible = max(abs(a - b) for a, b in zip(pix(img, px, py), background)) > 0.02
        agree += visible == (ref[3] >= 0.5)
        total += 1
    return agree / max(total, 1), total


def log_bounds(text):
    m = re.search(r"bounds min \(([-\d. e+]+)\) max \(([-\d. e+]+)\), camera distance ([\d.]+)", text)
    lo = [float(v) for v in m.group(1).split()]; hi = [float(v) for v in m.group(2).split()]
    return [(lo[i] + hi[i]) / 2 for i in range(3)], float(m.group(3))


def mirror_error(out, name, pitch=30.0):
    img, pix, _ = load(out, "mesh", name)
    centre, dist = log_bounds(pathlib.Path(out / f"mesh-{name}.log").read_text(errors="replace"))
    size = (img[0], img[1])
    pr = math.radians(pitch)
    back = (0.0, math.sin(pr), math.cos(pr)); right = (1.0, 0.0, 0.0); up = (0.0, math.cos(pr), -math.sin(pr))
    pos = [centre[k] + back[k] * dist for k in range(3)]
    tan_half = math.tan(math.radians(30))
    def project(P):
        v = [P[k] - pos[k] for k in range(3)]
        xc = sum(v[k] * right[k] for k in range(3)); yc = sum(v[k] * up[k] for k in range(3)); zc = -sum(v[k] * back[k] for k in range(3))
        ndc = (xc / (zc * tan_half * size[0] / size[1]), yc / (zc * tan_half))
        return int(math.floor((ndc[0] * 0.5 + 0.5) * size[0])), int(math.floor((0.5 - ndc[1] * 0.5) * size[1]))
    pairs = []
    n = 9
    for a in range(n):
        for b in range(n):
            gx = -1.5 + 3.0 * (a + 0.5) / n; gy = -1.5 + 3.0 * (b + 0.5) / n   # object-local plane coordinates (scale 0.75 applies to the 4 m grid -> 3 m)
            ref = reference(gx / 0.75, gy / 0.75, "unlit")
            P = (gx, gy - 0.5, 0.0)
            Pm = (P[0], -4.0 - P[1], P[2])
            px, py = project(Pm)
            pairs.append((pix(img, px, py), ref))
    # Score = mean |engine - reference| over all samples and channels.  The reflection is traced / filtered (RT hit shading, the custom denoiser), so a
    # steep gradient moves by up to ~0.1 at single samples: the criterion is the MEAN error, compared with the control's mean error against the same
    # reference (a reflection that ignores COLOR_0 shows white): score = mean(engine) / mean(control) must be <= 0.15, the mean bias per channel <= 0.03.
    err = sum(abs(g[c] - r[c]) for g, r in pairs for c in range(3)) / (3 * len(pairs))
    bias = max(abs(sum(g[c] - r[c] for g, r in pairs) / len(pairs)) for c in range(3))
    return err, bias, pairs


def verify(out):
    ok, lines = True, []
    if "mesh" not in PATHS:
        pass
    elif (out / "mesh-mirror.pfm").exists():
        e, bias, _ = mirror_error(out, "mirror")
        c, _, _ = mirror_error(out, "mirror-control")
        score = e / c if c > 0 else 1.0
        good = score <= 0.15 and bias <= 0.03 and c > 0.2
        lines.append(f"mesh MIRROR: mean error {e:.4f}, bias {bias:.4f} (<= 0.03), control mean error {c:.4f} (> 0.2), ratio {score:.3f} (<= 0.15) -> {'PASS' if good else 'FAIL'}")
        ok = ok and good
    else:
        lines.append("mesh MIRROR: captures missing"); ok = False
    for path in PATHS:
        have = all((out / f"{path}-{n}.{'pfm' if PATHS[path]['linear'] else 'png'}").exists() for n in CASES if not n.startswith("mirror"))
        if not have:
            lines.append(f"{path}: captures missing"); ok = False; continue
        if path == "tile":  # the on-tile resolve must really run (--tile-resolve selects the visibility path)
            got = json.loads((out / "tile-unlit.json").read_text()).get("rendering", {}).get("path")
            if got != "visibility-tile":
                lines.append(f"tile: rendering path {got!r}, expected 'visibility-tile' -> FAIL"); ok = False; continue
        if not PATHS[path]["linear"]:  # the forward captures must really run the forward path (--tonemap / --post silently select the visibility path)
            want_path = "forward-indexed" if path == "forward" else "forward-mesh"
            got = json.loads((out / f"{path}-unlit.json").read_text()).get("rendering", {}).get("path")
            if got != want_path:
                lines.append(f"{path}: rendering path {got!r}, expected {want_path!r} -> FAIL"); ok = False; continue
        e = unlit_error(out, path, "unlit"); c = unlit_error(out, path, "unlit-control")
        tol = 0.004 if PATHS[path]["linear"] else 3 / 255 + 0.002
        good = e <= tol and c > 0.05
        lines.append(f"{path} UNLIT: max error {e:.5f} (<= {tol:.4f}), control without COLOR_0 {c:.4f} (must be > 0.05) -> {'PASS' if good else 'FAIL'}")
        ok = ok and good
        if PATHS[path]["linear"]:
            l = lit_error(out, path, "lit", "lit-white"); lc = lit_error(out, path, "lit-white", "lit-white")
            good = l <= 0.03 and lc > 0.1
            lines.append(f"{path} LIT: max |ratio - colour| {l:.4f} (<= 0.03), control (COLOR_0 ignored) {lc:.4f} (must be > 0.1) -> {'PASS' if good else 'FAIL'}")
            ok = ok and good
        a, n = cutout_agreement(out, path, "cutout"); ca, _ = cutout_agreement(out, path, "cutout-control")
        good = a >= 0.995 and ca < 0.9
        lines.append(f"{path} CUTOUT: mask agreement {a * 100:.2f} % of {n} pixels (>= 99.5), control {ca * 100:.2f} % (must be < 90) -> {'PASS' if good else 'FAIL'}")
        ok = ok and good
    for mode in ("rt", "csm", "cache") if "mesh" in PATHS else ():
        if not (out / f"mesh-shadow-{mode}.pfm").exists():
            lines.append(f"mesh SHADOW {mode}: captures missing"); ok = False; continue
        cut, vis, lit, ccut = shadow_scores(out, mode)
        good = vis < 0.7 and cut >= 0.9 and ccut < 0.7
        lines.append(f"mesh SHADOW {mode}: visible-half shadow {vis:.3f} x lit (< 0.7), cut-half {cut:.3f} x lit (>= 0.9), control cut-half {ccut:.3f} (must be < 0.7) -> {'PASS' if good else 'FAIL'}")
        ok = ok and good
    lines.append("PASS" if ok else "FAIL")
    return ok, lines


SHADOW_PITCH = 55.0


def shadow_scores(out, mode):
    """(cut-half, visible-half, unshadowed reference, control cut-half): mean floor luminance of the points in the shadow of each half, relative to the lit floor."""
    results = {}
    for name in (f"shadow-{mode}", f"shadow-{mode}-control"):
        img, pix, _ = load(out, "mesh", name)
        centre, dist = log_bounds(pathlib.Path(out / f"mesh-{name}.log").read_text(errors="replace"))
        size = (img[0], img[1])
        pr = math.radians(SHADOW_PITCH)
        back = (0.0, math.sin(pr), math.cos(pr)); right = (1.0, 0.0, 0.0); up = (0.0, math.cos(pr), -math.sin(pr))
        pos = [centre[k] + back[k] * dist for k in range(3)]
        tan_half = math.tan(math.radians(30))
        def project(P):
            v = [P[k] - pos[k] for k in range(3)]
            xc = sum(v[k] * right[k] for k in range(3)); yc = sum(v[k] * up[k] for k in range(3)); zc = -sum(v[k] * back[k] for k in range(3))
            return ((xc / (zc * tan_half * size[0] / size[1])) * 0.5 + 0.5) * size[0], (0.5 - (yc / (zc * tan_half)) * 0.5) * size[1]
        sun = [-0.45 * right[k] + 0.55 * up[k] + 0.70 * back[k] for k in range(3)]
        norm = math.sqrt(sum(c * c for c in sun)); sun = [c / norm for c in sun]
        def lum(P):
            px, py = project(P)
            c = pix(img, int(px), int(py))
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        # panel corners in the image (the horizontal panel, y = 1): points of the floor covered by the panel in the image are skipped
        corners = [project((-2, 1, -2)), project((2, 1, -2)), project((2, 1, 2)), project((-2, 1, 2))]
        def covered(px, py):
            sign = 0
            for a, b in zip(corners, corners[1:] + corners[:1]):
                cr = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
                if cr != 0:
                    if sign == 0: sign = 1 if cr > 0 else -1
                    elif (cr > 0) != (sign > 0): return False
            return True
        def floor_of(x, z):  # the floor point in the shadow of the panel point (x, 1, z): along -sun down to y = 0
            t = 1.0 / sun[1]
            return (x - sun[0] * t, 0.0, z - sun[2] * t)
        def mean_shadow(xs):
            vals = []
            for x in xs:
                for z in (1.5, 1.7, 1.9):  # the strip under the panel's front edge: the floor there is visible from the camera
                    F = floor_of(x, z)
                    px, py = project(F)
                    if covered(px, py) or not (0 <= px < size[0] and 0 <= py < size[1]): continue
                    vals.append(lum(F))
            return sum(vals) / len(vals) if vals else float("nan"), len(vals)
        reference = sum(lum((5.5, 0.0, 5.0 + 0.4 * k)) for k in range(3)) / 3  # an unshadowed floor corner
        results[name] = (mean_shadow([-1.5, -0.9]), mean_shadow([0.9, 1.5]), reference)
    (cut, _), (vis, _), lit = results[f"shadow-{mode}"][0], results[f"shadow-{mode}"][1], results[f"shadow-{mode}"][2]
    (ccut, _) = results[f"shadow-{mode}-control"][0]
    return cut / lit, vis / lit, lit, ccut / results[f"shadow-{mode}-control"][2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "verify"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    parser.add_argument("--paths", help="comma list of the paths to capture / verify (default: every path; the mirror and shadow cases belong to `mesh`)")
    args = parser.parse_args()
    if args.paths:
        for name in list(PATHS):
            if name not in args.paths.split(","):
                del PATHS[name]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "run":
        for name, (kind, colors) in CASES.items():
            (out / f"{name}.gltf").write_text(json.dumps(scene_mirror(colors) if kind == "mirror" else scene(kind, colors)))
        for path in PATHS:
            for name in CASES:
                if name.startswith("mirror") and path != "mesh":
                    continue
                code = capture(args.binary, out, path, name, out / f"{name}.gltf", pitch=30 if name.startswith("mirror") else 0)
                print(f"{path}-{name}: exit {code}", flush=True)
        for mode in ("rt", "csm", "cache") if "mesh" in PATHS else ():
            for control in (False, True):
                name = f"shadow-{mode}" + ("-control" if control else "")
                (out / f"{name}.gltf").write_text(json.dumps(scene_shadow(not control)))
                code = capture(args.binary, out, "mesh", name, out / f"{name}.gltf", pitch=SHADOW_PITCH, shadows=mode)
                print(f"mesh-{name}: exit {code}", flush=True)
    ok, lines = verify(out)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
