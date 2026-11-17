#!/usr/bin/env python3
"""F15.1 (ticket 22) per-slot KHR_texture_transform fixture: GPU image against a CPU texel-lookup reference.

Generates a synthetic glTF (three quads, one material each, textures embedded; stdlib + PIL for the decode only), renders it under the
declared corpus rig (src/testbench/corpus_rig.h) and checks, for every measured slot, that the colour the engine shows at a pixel is the
texel the KHR_texture_transform README formula selects for that pixel:

  uv' = T R S uv   (offset, rotation counter-clockwise, scale), repeat wrap, 4 x 4 cells of 64 texels per texture.

Every material carries THREE transformed texture slots with DIFFERENT transforms (baseColor, emissive, occlusion): each quad measures one
slot and the other two are decoys, so a material-wide ("uniform bake") transform cannot pass.  Cells encode a 3-bit code in the three channels
(baseColor, emissive: sRGB 64 / 255 per channel) or one bit in the occlusion texture (1.0 / 0.15).  The engine pixel is decoded per channel
against the midpoint of that channel's range over the quad (the tonemap and the lighting are monotone, the decode needs no radiometry).

Thresholds (fixed here, before any comparison; the criterion is independent of the renderer: a wrong transform maps a pixel to a random cell,
two random 3-bit codes agree with probability 1/8 for a uniform draw, up to 0.4 for this palette):
  - agreement of the correct reference >= 0.995 of the valid pixels of every measured slot;
  - every negative control (the same pixels against the transform of ANOTHER slot of the material, i.e. the uniform-bake result, and against
    the identity) must stay below 0.90, otherwise the fixture does not discriminate and the run exits 3.
Valid pixels: inside the quad (6 px inset) and at least 0.08 cell away from a cell edge in the transformed UV (filtering and the edge
position error of one pixel excluded).

A fourth quad measures the NORMAL slot, including the rotation of the tangent frame with the transform (a normal map sampled through a
rotated UV must tilt the normal along the rotated texel axes): its 4 x 4 cells tilt the normal by 0.6 towards +x, -x, +y or -y of the
texture, the sun of the rig (direction fixed by src/testbench/corpus_rig.h) lights the quad, and a cell is "bright" when N'.L exceeds the
flat N.L.  Extra negative controls: the frame NOT rotated with the transform and the frame rotated the wrong way.

A fifth quad measures the ALPHA test (alphaMode MASK, cutoff 0.5) through the base colour slot's transform: the alpha channel of its
texture is 255 / 0 per cell, a cell is "visible" when the quad covers the pixel (differs from the background) and "cut out" otherwise; it
runs through the alpha raster pipeline of the visibility path (visibility_alpha_lit_tt_fs).

Modes: --selftest (CPU: reference maths), --generate DIR, --run (needs the GPU lock: wrap with gpulock.py), --evaluate (existing capture).
Exit codes: 0 ok, 1 gate failed, 2 usage, 3 non-discriminating control, 6 capture failed.
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
import zlib

CELLS = 4
TEXELS = 256
RESOLUTION = (1024, 768)
FOV_HALF_TAN = math.tan(math.radians(30.0))
RIG_MARGIN = 1.12
AGREEMENT_MIN = 0.995
CONTROL_MAX = 0.90
EDGE_MARGIN = 0.08  # cells
INSET_PX = 6

# 16 cell codes (3 bits), irregular so that no rotation / mirror / shift of the pattern maps it onto itself.
CODES = [5, 3, 6, 1,
         2, 7, 4, 0,
         3, 6, 1, 5,
         7, 2, 0, 4]

TILT = 0.6  # tangent-space tilt of a normal-map cell (sin of the angle)
TRANSFORMS = {
    "baseColor": {"offset": [0.2, 0.1], "rotation": 0.3, "scale": [2.0, 1.5]},
    "emissive": {"offset": [0.5, 0.25], "rotation": 1.5707963267948966, "scale": [1.25, 1.25]},
    "occlusion": {"offset": [0.05, 0.6], "rotation": -0.7, "scale": [3.0, 3.0]},
    "normal": {"offset": [0.3, 0.15], "rotation": 1.5707963267948966, "scale": [2.0, 2.0]},
    "alpha": {"offset": [0.1, 0.35], "rotation": 2.2, "scale": [1.75, 2.5]},
}
# towardSun of corpus_rig.h for yaw 0, pitch 0 (right = +X, up = +Y, back = +Z), normalised
_L = (-0.45, 0.55, 0.70)
_LN = math.sqrt(sum(c * c for c in _L))
SUN = tuple(c / _LN for c in _L)
IDENTITY = {"offset": [0.0, 0.0], "rotation": 0.0, "scale": [1.0, 1.0]}
# quad index, measured slot, X centre
# (slot, X centre, Z): the quads sit at +-0.1 in Z so that the scene bounds have a depth (the probe capture of the renderer needs a positive extent)
QUADS = [("baseColor", -2.6, 0.1), ("emissive", -1.3, -0.1), ("occlusion", 0.0, 0.1), ("normal", 1.3, -0.1), ("alpha", 2.6, 0.1)]
QUAD_SIZE = 1.0


def transform_uv(t, u, v):
    """The README's matrices: scale, then rotate counter-clockwise, then translate."""
    sx, sy = u * t["scale"][0], v * t["scale"][1]
    c, s = math.cos(t["rotation"]), math.sin(t["rotation"])
    return c * sx - s * sy + t["offset"][0], s * sx + c * sy + t["offset"][1]


def cell_of(u, v):
    """Cell (column, row) of a repeat-wrapped UV, row 0 at the top of the image (glTF v origin top-left)."""
    fu, fv = u - math.floor(u), v - math.floor(v)
    return min(CELLS - 1, int(fu * CELLS)), min(CELLS - 1, int(fv * CELLS)), fu * CELLS, fv * CELLS


def code_at(col, row):
    return CODES[row * CELLS + col]


def edge_distance(cu, cv):
    """Distance in cells to the nearest cell edge of the wrapped coordinates cu, cv (in cell units)."""
    fu, fv = cu - math.floor(cu), cv - math.floor(cv)
    return min(fu, 1 - fu, fv, 1 - fv)


# --- PNG ---------------------------------------------------------------------------------------------------------

def png_bytes(width, height, pixel):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw += bytes(pixel(x, y))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def palette_png(kind):
    """kind 'color': 3-bit code -> RGB in {64, 255}; 'ao': bit 0 of the code -> 255 / 38 (1.0 / 0.15 linear) grey."""
    cell = TEXELS // CELLS

    def pixel(x, y):
        code = code_at(x // cell, y // cell)
        if kind == "color":
            return tuple(255 if (code >> bit) & 1 else 64 for bit in (2, 1, 0))
        value = 255 if code & 1 else 38
        return (value, value, value)

    return png_bytes(TEXELS, TEXELS, pixel)


def normal_png():
    """Tilt codes 0..3 (+x, -x, +y, -y) of the cell code & 3 as a tangent-space normal map (linear RGB)."""
    cell = TEXELS // CELLS
    tz = math.sqrt(1.0 - TILT * TILT)
    vectors = [(TILT, 0.0), (-TILT, 0.0), (0.0, TILT), (0.0, -TILT)]

    def pixel(x, y):
        tx, ty = vectors[code_at(x // cell, y // cell) & 3]
        return tuple(int(round((v * 0.5 + 0.5) * 255)) for v in (tx, ty, tz))

    return png_bytes(TEXELS, TEXELS, pixel)


def alpha_png():
    """Red cells whose alpha is 255 / 0 by bit 0 of the cell code (RGBA)."""
    cell = TEXELS // CELLS
    raw = bytearray()
    for y in range(TEXELS):
        raw.append(0)
        for x in range(TEXELS):
            raw += bytes((230, 40, 40, 255 if code_at(x // cell, y // cell) & 1 else 0))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", TEXELS, TEXELS, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def flat_png(value):
    return png_bytes(8, 8, lambda x, y: (value, value, value))


def data_uri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


# --- glTF ----------------------------------------------------------------------------------------------------------

def build_gltf(path):
    images = {  # name -> png
        "palette": palette_png("color"), "ao": palette_png("ao"),
        "black": flat_png(0), "grey": flat_png(60), "white": flat_png(255), "normal": normal_png(), "alphatex": alpha_png(), "flatnormal": png_bytes(8, 8, lambda x, y: (128, 128, 255)),
    }
    names = list(images)
    textures = [{"source": i, "sampler": 0} for i in range(len(names))]
    index = {n: i for i, n in enumerate(names)}

    def tex(name, slot):
        t = TRANSFORMS[slot]
        return {"index": index[name], "extensions": {"KHR_texture_transform": {"offset": t["offset"], "rotation": t["rotation"], "scale": t["scale"]}}}

    # (base texture, emissive texture, occlusion texture, emissive factor)
    recipes = {
        "baseColor": ("palette", "black", "white", [1, 1, 1]),
        "emissive": ("grey", "palette", "white", [1, 1, 1]),
        "occlusion": ("white", "black", "ao", [1, 1, 1]),
        "normal": ("white", "black", "white", [1, 1, 1]),
        "alpha": ("alphatex", "black", "white", [1, 1, 1]),
    }
    materials = []
    positions, normals, uvs, indices, tangents = [], [], [], [], []
    for qi, (slot, cx, cz) in enumerate(QUADS):
        base, emi, occ, ef = recipes[slot]
        materials.append({
            "name": f"measure-{slot}",
            **({"alphaMode": "MASK", "alphaCutoff": 0.5} if slot == "alpha" else {}),
            "pbrMetallicRoughness": {"baseColorTexture": tex(base, "alpha" if slot == "alpha" else "baseColor"), "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "emissiveTexture": tex(emi, "emissive"), "emissiveFactor": ef,
            "occlusionTexture": tex(occ, "occlusion"),
            "normalTexture": tex("normal" if slot == "normal" else "flatnormal", "normal"),
        })
        h = QUAD_SIZE / 2
        corners = [(cx - h, h, 0, 0), (cx - h, -h, 0, 1), (cx + h, -h, 1, 1), (cx + h, h, 1, 0)]
        for x, y, u, v in corners:
            positions += [x, y, cz]
            normals += [0.0, 0.0, 1.0]
            tangents += [1.0, 0.0, 0.0, 1.0]
            uvs += [u, v]
        b = qi * 4
        indices += [b, b + 1, b + 2, b, b + 2, b + 3]
    # one mesh per quad below, sharing the vertex accessors
    pos_b = struct.pack(f"<{len(positions)}f", *positions)
    nor_b = struct.pack(f"<{len(normals)}f", *normals)
    uv_b = struct.pack(f"<{len(uvs)}f", *uvs)
    tan_b = struct.pack(f"<{len(tangents)}f", *tangents)
    idx_b = struct.pack(f"<{len(indices)}H", *indices)
    blob = pos_b + nor_b + uv_b + tan_b + idx_b
    views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b)},
        {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(nor_b)},
        {"buffer": 0, "byteOffset": len(pos_b) + len(nor_b), "byteLength": len(uv_b)},
        {"buffer": 0, "byteOffset": len(pos_b) + len(nor_b) + len(uv_b), "byteLength": len(tan_b)},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": len(positions) // 3, "type": "VEC3",
         "min": [min(positions[0::3]), min(positions[1::3]), min(positions[2::3])], "max": [max(positions[0::3]), max(positions[1::3]), max(positions[2::3])]},
        {"bufferView": 1, "componentType": 5126, "count": len(normals) // 3, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5126, "count": len(uvs) // 2, "type": "VEC2"},
        {"bufferView": 3, "componentType": 5126, "count": len(tangents) // 4, "type": "VEC4"},
    ]
    # one mesh per quad, each with its own index buffer view
    meshes, nodes = [], []
    offset = len(pos_b) + len(nor_b) + len(uv_b) + len(tan_b)
    for qi in range(len(QUADS)):
        views.append({"buffer": 0, "byteOffset": offset + qi * 12, "byteLength": 12})
        accessors.append({"bufferView": len(views) - 1, "componentType": 5123, "count": 6, "type": "SCALAR"})
        meshes.append({"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2, "TANGENT": 3}, "indices": len(accessors) - 1, "material": qi}]})
        nodes.append({"mesh": qi, "name": f"quad-{QUADS[qi][0]}"})
    doc = {
        "asset": {"version": "2.0", "generator": "tools/f15_texture_transform_check.py"},
        "extensionsUsed": ["KHR_texture_transform"],
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": materials,
        "textures": textures, "images": [{"uri": data_uri(images[n])} for n in names],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode(), "byteLength": len(blob)}],
        "bufferViews": views, "accessors": accessors,
    }
    pathlib.Path(path).write_text(json.dumps(doc))


# --- rig / projection ----------------------------------------------------------------------------------------------

def rig_distance(aspect):
    half_x = max(abs(cx) for _, cx, _ in QUADS) + QUAD_SIZE / 2
    half_y = QUAD_SIZE / 2
    half_z = max(abs(cz) for _, _, cz in QUADS)
    return RIG_MARGIN * max(half_x / (FOV_HALF_TAN * aspect), half_y / FOV_HALF_TAN) + half_z


def project(x, y, z, distance, aspect, width, height):
    depth = distance - z
    ndc_x = x / (depth * FOV_HALF_TAN * aspect)
    ndc_y = y / (depth * FOV_HALF_TAN)
    return (ndc_x * 0.5 + 0.5) * width, (0.5 - ndc_y * 0.5) * height


def quad_pixel_to_uv(px, py, cx, cz, distance, aspect, width, height):
    """Quad (u, v) of the pixel centre (px + 0.5, py + 0.5): the inverse of project() for the plane z = cz."""
    ndc_x = (px + 0.5) / width * 2 - 1
    ndc_y = 1 - (py + 0.5) / height * 2
    depth = distance - cz
    x = ndc_x * depth * FOV_HALF_TAN * aspect
    y = ndc_y * depth * FOV_HALF_TAN
    return (x - (cx - QUAD_SIZE / 2)) / QUAD_SIZE, (QUAD_SIZE / 2 - y) / QUAD_SIZE


# --- evaluation ----------------------------------------------------------------------------------------------------

def normal_bright(code, transform, frame):
    """Is the cell's normal turned towards the sun more than the flat one?  frame: 'rotated' (the tangent frame follows the UV
    transform: T' = inv00 T - inv10 B, B' = -inv01 T + inv11 B), 'plain' (not rotated), 'inverse' (rotated the wrong way)."""
    tz = math.sqrt(1.0 - TILT * TILT)
    tx, ty = [(TILT, 0.0), (-TILT, 0.0), (0.0, TILT), (0.0, -TILT)][code & 3]
    c, s_ = math.cos(transform["rotation"]), math.sin(transform["rotation"])
    sx, sy = transform["scale"]
    m00, m01, m10, m11 = c * sx, -s_ * sy, s_ * sx, c * sy
    det = m00 * m11 - m01 * m10
    i00, i01, i10, i11 = m11 / det, -m01 / det, -m10 / det, m00 / det
    if frame == "plain":
        t_, b_ = (1.0, 0.0), (0.0, 1.0)
    else:
        sign = 1.0 if frame == "rotated" else -1.0
        # T = (1, 0), B = (0, 1) in the plane of the quad; the inverse rotation negates the off-diagonal terms
        t_ = (i00, -sign * i10)
        b_ = (-sign * i01, i11)
        lt, lb = math.hypot(*t_), math.hypot(*b_)
        t_, b_ = (t_[0] / lt, t_[1] / lt), (b_[0] / lb, b_[1] / lb)
    n = (t_[0] * tx + b_[0] * ty, t_[1] * tx + b_[1] * ty, tz)
    ln = math.sqrt(sum(v * v for v in n))
    return sum(n[i] * SUN[i] for i in range(3)) / ln > SUN[2]


def decode_pixel(rgb, lo, hi, slot, background=None):
    if slot == "alpha":
        return int(max(abs(rgb[c] - background[c]) for c in range(3)) > 8)
    if slot in ("occlusion", "normal"):
        luma = (rgb[0] + rgb[1] + rgb[2]) / 3.0
        return int(luma > (lo[3] + hi[3]) / 2.0)
    return sum(int(rgb[c] > (lo[c] + hi[c]) / 2.0) << (2 - c) for c in range(3))


def evaluate_slot(img, slot, cx, cz, distance, aspect):
    width, height = img.size
    pix = img.load()
    background = pix[4, 4]
    x0, y0 = project(cx - QUAD_SIZE / 2, QUAD_SIZE / 2, cz, distance, aspect, width, height)
    x1, y1 = project(cx + QUAD_SIZE / 2, -QUAD_SIZE / 2, cz, distance, aspect, width, height)
    samples = []
    for py in range(int(y0) + INSET_PX, int(y1) - INSET_PX, 2):
        for px in range(int(x0) + INSET_PX, int(x1) - INSET_PX, 2):
            u, v = quad_pixel_to_uv(px, py, cx, cz, distance, aspect, width, height)
            tu, tv = transform_uv(TRANSFORMS[slot], u, v)
            if edge_distance(tu * CELLS, tv * CELLS) < EDGE_MARGIN:
                continue
            samples.append((px, py, u, v, pix[px, py]))
    if not samples:
        raise RuntimeError(f"{slot}: no valid pixel")
    # channel ranges over the quad (the occlusion bit uses the luminance range stored in index 3)
    lo, hi = [255] * 4, [0] * 4
    for _, _, _, _, rgb in samples:
        for c in range(3):
            lo[c], hi[c] = min(lo[c], rgb[c]), max(hi[c], rgb[c])
        luma = sum(rgb[:3]) / 3.0
        lo[3], hi[3] = min(lo[3], luma), max(hi[3], luma)
    result = {"valid_pixels": len(samples), "range": {"lo": lo, "hi": hi}}
    # name -> (transform of the lookup, frame mode of the normal slot)
    candidates = {"reference": (TRANSFORMS[slot], "rotated"), "identity": (IDENTITY, "rotated")}
    for other in TRANSFORMS:
        if other != slot:
            candidates[f"uniform-bake:{other}-transform"] = (TRANSFORMS[other], "rotated")
    if slot == "normal":
        candidates["frame-not-rotated"] = (TRANSFORMS[slot], "plain")
        candidates["frame-rotated-the-wrong-way"] = (TRANSFORMS[slot], "inverse")
    agree = {name: 0 for name in candidates}
    for _, _, u, v, rgb in samples:
        seen = decode_pixel(rgb, lo, hi, slot, background)
        for name, (t, frame) in candidates.items():
            col, row, _, _ = cell_of(*transform_uv(t, u, v))
            want = code_at(col, row)
            if slot in ("occlusion", "alpha"):
                want &= 1
            elif slot == "normal":
                want = int(normal_bright(want, t, frame))
            if seen == want:
                agree[name] += 1
    result["agreement"] = {name: round(n / len(samples), 4) for name, n in agree.items()}
    return result


def evaluate(capture, log_text):
    from PIL import Image
    img = Image.open(capture).convert("RGB")
    width, height = img.size
    aspect = width / height
    match = re.search(r"CORPUS_RIG: .*camera distance ([\d.e+-]+)", log_text) if log_text else None
    distance = float(match.group(1)) if match else rig_distance(aspect)
    out = {"capture": str(capture), "size": [width, height], "distance": distance, "distance_expected": rig_distance(aspect), "slots": {}}
    ok, discriminates = True, True
    for slot, cx, cz in QUADS:
        r = evaluate_slot(img, slot, cx, cz, distance, aspect)
        out["slots"][slot] = r
        if r["agreement"]["reference"] < AGREEMENT_MIN:
            ok = False
        for name, value in r["agreement"].items():
            if name != "reference" and value >= CONTROL_MAX:
                discriminates = False
    out["gate_ok"], out["controls_discriminate"] = ok, discriminates
    return out


# --- self test (CPU) -------------------------------------------------------------------------------------------------

def selftest():
    # The README example used by tests/test_material_description.cpp: scale (2, 3), rotation pi/2, uv (0.25, 0.5).
    t = {"offset": [0.0, 0.0], "rotation": math.pi / 2, "scale": [2.0, 3.0]}
    u, v = transform_uv(t, 0.25, 0.5)
    assert abs(u - (-1.5)) < 1e-9 and abs(v - 0.5) < 1e-9, (u, v)
    u, v = transform_uv({"offset": [0.5, 0.25], "rotation": 0.0, "scale": [1.0, 1.0]}, 0.25, 0.5)
    assert abs(u - 0.75) < 1e-12 and abs(v - 0.75) < 1e-12
    # cells: a transform must change the cell of at least 60 % of the sample grid for each slot (the fixture discriminates by construction).
    for slot, t in TRANSFORMS.items():
        moved = total = 0
        for i in range(20):
            for j in range(20):
                u0, v0 = (i + 0.5) / 20, (j + 0.5) / 20
                a = cell_of(*transform_uv(t, u0, v0))[:2]
                b = cell_of(u0, v0)[:2]
                total += 1
                moved += a != b
        assert moved / total > 0.6, (slot, moved / total)
    # the palette: every 3-bit code appears and the pattern is not invariant under the mirrors
    assert sorted(set(CODES)) == list(range(8)), CODES
    grid = [CODES[r * CELLS:(r + 1) * CELLS] for r in range(CELLS)]
    assert grid != [row[::-1] for row in grid] and grid != grid[::-1]
    # the projection round trip
    d = rig_distance(RESOLUTION[0] / RESOLUTION[1])
    px, py = project(-1.3, 0.25, 0.1, d, RESOLUTION[0] / RESOLUTION[1], *RESOLUTION)
    u, v = quad_pixel_to_uv(px - 0.5, py - 0.5, -1.3, 0.1, d, RESOLUTION[0] / RESOLUTION[1], *RESOLUTION)
    assert abs(u - 0.5) < 1e-9 and abs(v - 0.25) < 1e-9, (u, v)
    # the fixture file: one material per quad, every material carries four different transformed slots plus the measured one
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "fixture.gltf"
        build_gltf(path)
        doc = json.loads(path.read_text())
        assert len(doc["materials"]) == len(QUADS) == len(doc["meshes"]) == len(doc["nodes"])
        for material in doc["materials"]:
            assert material["emissiveTexture"]["extensions"]["KHR_texture_transform"]["rotation"] == TRANSFORMS["emissive"]["rotation"]
            assert material["occlusionTexture"]["extensions"]["KHR_texture_transform"]["scale"] == TRANSFORMS["occlusion"]["scale"]
            assert material["normalTexture"]["extensions"]["KHR_texture_transform"]["offset"] == TRANSFORMS["normal"]["offset"]
        assert doc["materials"][-1]["alphaMode"] == "MASK"
    print("selftest ok")


def capture_command(binary, gltf, out):
    return [str(binary), "--bench", "4", "--scene", str(gltf), "--corpus-rig", "0,0",
            "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", "--shadows", "rt", "--lighting", "restir",
            "--gi", "off", "--reflections", "rt", "--ao", "off", "--lighting-denoise", "custom",
            "--post", "--tonemap", "aces", "--upscaler", "native", "--resolution", f"{RESOLUTION[0]}x{RESOLUTION[1]}",
            "--warmup", "48", "--frames", "48", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--pipeline-sync",
            "--capture", str(out / "engine.png"), "--report", str(out / "report.json")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--generate", metavar="DIR")
    p.add_argument("--run", action="store_true", help="capture with --binary (wrap with gpulock.py) and evaluate")
    p.add_argument("--evaluate", action="store_true", help="evaluate OUT/engine.png")
    p.add_argument("--binary")
    p.add_argument("--out", default="build/f15-texxform")
    p.add_argument("--no-archive", action="store_true", help="run with --no-pipeline-archive (needed under MTL_SHADER_VALIDATION)")
    p.add_argument("--harvest", metavar="FILE", help="record the pipelines of the run with --harvest-pipelines FILE")
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    out = pathlib.Path(a.generate or a.out)
    out.mkdir(parents=True, exist_ok=True)
    gltf = out / "texture_transform_fixture.gltf"
    if a.generate or a.run:
        build_gltf(gltf)
        if a.generate:
            print(gltf)
            return 0
    log = out / "engine.log"
    if a.run:
        if not a.binary:
            print("--run needs --binary", file=sys.stderr)
            return 2
        cmd = capture_command(a.binary, gltf, out)
        if a.no_archive:
            cmd.append("--no-pipeline-archive")
        if a.harvest:
            cmd += ["--harvest-pipelines", a.harvest]
        with log.open("w") as stream:
            code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
        text = log.read_text(errors="replace")
        if code != 0 or not (out / "engine.png").is_file() or not re.search(r"^EXIT 0$", text, re.M):
            print(f"capture failed (exit {code}); see {log}", file=sys.stderr)
            return 6
    if a.run or a.evaluate:
        result = evaluate(out / "engine.png", log.read_text(errors="replace") if log.is_file() else "")
        (out / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k != "slots"}, indent=1))
        for slot, r in result["slots"].items():
            print(f"{slot}: valid {r['valid_pixels']} agreement {r['agreement']}")
        if not result["controls_discriminate"]:
            return 3
        return 0 if result["gate_ok"] else 1
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
