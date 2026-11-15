#!/usr/bin/env python3
"""F15.4 (ticket 23) + F15.1 (ticket 22): layer-stack coverage alpha and per-slot texture transforms inside layer stacks, GPU image against a CPU model.

Scenes are glTF (a stack material "stack" + two layer materials, tiny quads keep the layers in the library) plus a material-overrides file that
makes "stack" a layer stack (mask / heights as PNG files next to it); rendered under the corpus rig (src/testbench/corpus_rig.h).

Coverage cases (the stack quad has alphaMode MASK, cutoff 0.5; layer 0 alpha 1, layer 1 alpha 0; a horizontal ramp mask R = u, G = 1 - u):
  stack-visibility   the quad seen in front of a floor: pixel visible iff the blended coverage w0 = R(u) >= 0.5, i.e. u >= 0.5
  stack-rt-shadow    the sun shadow of the quad on the floor behind it (--shadows rt: the any-hit alpha test) blocked iff u >= 0.5
  stack-csm-shadow   the same with --shadows csm (shadow_depth_fragment)
  stack-reflection   the quad seen in a mirror floor (--reflections rt: the any-hit alpha test of the reflection ray)
  stack-cache-shadow --shadows csm --shadow-cache on (shadow_cache_depth_fragment)
  Negative controls (each must FAIL): the stack's OWN record alpha (coverage 1 everywhere: visible / blocking everywhere) and the inverted ramp.

Emitter case:
  stack-emitter      an EMISSIVE layer stack (MASK cutoff 0.5, emissive strength 40, the same ramp mask) cuts the emitter out where the blended coverage is
                     below the cutoff, so only the u >= 0.5 half of the quad lights the floor in front of it: the luminance of the floor next to the right half
                     minus the left half is the gate (threshold 0.08 of the display range, as the emitter-texture case of f15_texture_transform_secondary.py);
                     the CPU model is a form-factor sum over the emitting half; controls: the stack's OWN alpha (the whole quad emits: no left-right
                     difference) and the inverted ramp (the other half emits).

Dual-use case:
  direct-and-layer   ONE material (palette base colour, a UNIFORM transform: the loader bakes it into the vertex uvs of the primitives that use it
                     directly) is used by a quad directly AND as layer 1 of a stack (weight 1) on a second quad.  Both quads must show the palette
                     through the transform exactly ONCE (cells against the CPU lookup, 0.995 as layer-tex); the control "applied twice" (the transform
                     composed with itself: what a block applied on top of the baked uvs gives) and identity / other transforms must fail.

Transform cases (layer 1 carries a per-slot transform on its base colour slot and a DIFFERENT transform on a decoy slot):
  layer-tex     weight 1 on layer 1 (constant mask), layer 1 base colour = the 4 x 4 palette: cell codes through the layer's transform against the
                CPU README lookup; controls: identity and the transforms of other slots (the uniform-bake result)
  layer-mask    red / green layers, a halves mask (R left, G right) whose own transform swaps the halves: region mean R - G, left minus right
  layer-height  two layers at equal weight, layer 0's height texture halves (high left, low right) whose transform swaps them: the higher layer wins

Thresholds, fixed here before any comparison: coverage cases agreement >= 0.97 (shadow-map footprint, filtered ramp: pixels within 0.06 of the
boundary u = 0.5 and 0.12 cell-widths of its borders are excluded); every control must stay below 0.90 (exit 3 otherwise); layer-tex cell agreement
>= 0.995 (the primary fixture's value); layer-mask / layer-height: the measured R - G difference must be >= 0.05 in magnitude with the sign of the
reference model, and every control predicts the other sign or none (the self-test asserts it).

Modes: --selftest (CPU), --run (GPU lock: wrap with gpulock.py), --evaluate.  Exit codes: 0 ok, 1 gate failed, 2 usage, 3 non-discriminating control, 6 capture failed.
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
import f15_texture_transform_secondary as sec  # noqa: E402

RED, GREEN = (230, 40, 40), (40, 230, 40)
SWAP = {"offset": [0.5, 0.0], "rotation": 0.0, "scale": [1.0, 1.0]}
# the mask is sampled clamp-to-edge (no repeat): its halves are exchanged by a mirror (u' = 1 - u), the height (repeat) by an offset of one half
FLIP = {"offset": [1.0, 0.0], "rotation": 0.0, "scale": [-1.0, 1.0]}
SIZE = sec.SIZE
CUT = 0.5
RAMP_EDGE = 0.06
AGREE = 0.97
CONTROL_SLOTS = {"mask": ("baseColor", "emissive", "occlusion", "normal"), "height": ("baseColor", "emissive", "occlusion", "normal", "alpha")}
CASES = {
    "stack-visibility": {"kind": "cover", "view": "visibility", "shadows": ["--shadows", "rt"], "pitch": 0.0},
    "stack-rt-shadow": {"kind": "cover", "view": "shadow", "shadows": ["--shadows", "rt"], "pitch": 0.0},
    "stack-csm-shadow": {"kind": "cover", "view": "shadow", "shadows": ["--shadows", "csm"], "pitch": 0.0},
    "stack-cache-shadow": {"kind": "cover", "view": "shadow", "shadows": ["--shadows", "csm", "--shadow-cache", "on"], "pitch": 0.0},
    "stack-reflection": {"kind": "cover", "view": "reflection", "shadows": ["--shadows", "rt"], "pitch": 30.0},
    "stack-emitter": {"kind": "emitter", "shadows": ["--shadows", "rt"], "pitch": 40.0},
    "direct-and-layer": {"kind": "dual", "shadows": ["--shadows", "rt"], "pitch": 0.0},
    "layer-tex": {"kind": "tex", "shadows": ["--shadows", "rt"], "pitch": 0.0},
    "layer-mask": {"kind": "mask", "shadows": ["--shadows", "rt"], "pitch": 0.0},
    "layer-height": {"kind": "height", "shadows": ["--shadows", "rt"], "pitch": 0.0},
}


# --- textures ---------------------------------------------------------------------------------------------------------------

def ramp_png():
    """256 x 4 mask: R = x (weight of layer 0 grows with u), G = 255 - x."""
    return base.png_bytes(256, 4, lambda x, y: (x, 255 - x, 0))


def const_png(rgb):
    return base.png_bytes(8, 8, lambda x, y: rgb)


def halves(left, right):
    return sec.halves_png(left, right)


# --- scene ----------------------------------------------------------------------------------------------------------------

def tex_json(index, transform=None):
    j = {"index": index}
    if transform:
        j["extensions"] = {"KHR_texture_transform": {"offset": transform["offset"], "rotation": transform["rotation"], "scale": transform["scale"]}}
    return j


def build_scene(out: pathlib.Path, case):
    cfg = CASES[case]
    T = base.TRANSFORMS
    images = {"palette": base.palette_png("color"), "white": base.flat_png(255), "black": base.flat_png(0)}
    names = list(images)
    index = {n: i for i, n in enumerate(names)}
    files = {}  # mask / height PNGs written next to the overrides
    override = {"name": "stack", "layers": {"layers": []}}
    L = override["layers"]
    stack_alpha = {"alphaMode": "MASK", "alphaCutoff": 0.5}
    layer0 = {"name": "layerA", "pbrMetallicRoughness": {"baseColorFactor": [RED[0] / 255, RED[1] / 255, RED[2] / 255, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    layer1 = {"name": "layerB", "pbrMetallicRoughness": {"baseColorFactor": [RED[0] / 255, RED[1] / 255, RED[2] / 255, 0.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    stack = {"name": "stack", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    if cfg["kind"] in ("cover", "emitter"):
        layer0.update({"alphaMode": "MASK", "alphaCutoff": 0.5})
        layer1.update({"alphaMode": "MASK", "alphaCutoff": 0.5})
        stack.update(stack_alpha)
        if cfg["kind"] == "emitter":
            stack["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 0, 1]
            stack["emissiveFactor"] = [1, 1, 1]
            stack["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": 40.0}}
        files["mask.png"] = ramp_png()
        L["mask"] = {"path": "mask.png"}
        L["layers"] = [{"material": "layerA", "maskChannel": "R"}, {"material": "layerB", "maskChannel": "G"}]
    elif cfg["kind"] == "tex":
        # layer 1 weight 1; its base colour is the palette through a per-slot transform, the emissive slot (black) carries another transform
        layer0["pbrMetallicRoughness"]["baseColorFactor"] = [0.2, 0.2, 0.2, 1.0]
        layer1["pbrMetallicRoughness"] = {"baseColorTexture": tex_json(index["palette"], T["alpha"]), "metallicFactor": 0.0, "roughnessFactor": 1.0}
        layer1["emissiveTexture"] = tex_json(index["black"], T["emissive"])
        layer1["emissiveFactor"] = [1, 1, 1]
        files["mask.png"] = const_png((0, 255, 0))
        L["mask"] = {"path": "mask.png"}
        L["layers"] = [{"material": "layerA", "maskChannel": "R"}, {"material": "layerB", "maskChannel": "G"}]
        # a second slot with a transform makes the sub-material PerSlot (the base slot alone would be Uniform: a layer is never baked either way)
    elif cfg["kind"] == "mask":
        layer0["pbrMetallicRoughness"]["baseColorFactor"] = [RED[0] / 255, RED[1] / 255, RED[2] / 255, 1.0]
        layer1["pbrMetallicRoughness"]["baseColorFactor"] = [GREEN[0] / 255, GREEN[1] / 255, GREEN[2] / 255, 1.0]
        files["mask.png"] = halves((255, 0, 0), (0, 255, 0))
        L["mask"] = {"path": "mask.png", "transform": {"offset": FLIP["offset"], "rotation": 0.0, "scale": FLIP["scale"]}}
        L["layers"] = [{"material": "layerA", "maskChannel": "R"}, {"material": "layerB", "maskChannel": "G"}]
    else:  # height
        layer0["pbrMetallicRoughness"]["baseColorFactor"] = [RED[0] / 255, RED[1] / 255, RED[2] / 255, 1.0]
        layer1["pbrMetallicRoughness"]["baseColorFactor"] = [GREEN[0] / 255, GREEN[1] / 255, GREEN[2] / 255, 1.0]
        files["mask.png"] = const_png((255, 255, 0))
        files["height0.png"] = halves((255, 255, 255), (0, 0, 0))
        L["mask"] = {"path": "mask.png"}
        L["layers"] = [{"material": "layerA", "maskChannel": "R", "heightContrast": 0.1,
                        "heightTexture": {"path": "height0.png", "transform": {"offset": SWAP["offset"], "rotation": 0.0, "scale": SWAP["scale"]}}},
                       {"material": "layerB", "maskChannel": "G", "heightContrast": 0.1}]
    # an emissive decoy slot on the stack with its own transform makes the stack PerSlot together with the mask / height transforms
    h = SIZE / 2
    view = cfg.get("view")
    quads = []
    if cfg["kind"] == "dual":
        shared = {"name": "shared", "pbrMetallicRoughness": {"baseColorTexture": tex_json(index["palette"], T["alpha"]), "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        floor = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.9, 0.9, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        files["mask.png"] = const_png((0, 255, 0))
        L["mask"] = {"path": "mask.png"}
        L["layers"] = [{"material": "layerA", "maskChannel": "R"}, {"material": "shared", "maskChannel": "G"}]
        layer0["pbrMetallicRoughness"]["baseColorFactor"] = [0.2, 0.2, 0.2, 1.0]
        layer1 = shared  # the tiny keeper quad below references it
        quads.append((floor, (-2.0, 1.5, sec.FLOOR_Z), (4.0, 0, 0), (0, -3.0, 0)))
        quads.append((shared, (-1.3, 0.6, 0.0), (1.2, 0, 0), (0, -1.2, 0)))      # the direct user (baked)
        quads.append((stack, (0.1, 0.6, 0.0), (1.2, 0, 0), (0, -1.2, 0)))        # the stack, layer 1 = the same material
    elif cfg["kind"] == "emitter":
        floor = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.15, 0.15, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        quads.append((floor, (-2.0, 0.0, 0.0), (4.0, 0, 0), (0, 0, 3.0)))
        quads.append((stack, (-0.6, 1.2, 0.0), (1.2, 0, 0), (0, -1.2, 0)))
    elif view in ("visibility", "shadow") or cfg["kind"] in ("tex", "mask", "height"):
        floor = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.9, 0.9, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
        quads.append((floor, (-sec.FLOOR[0] / 2, sec.FLOOR[1] / 2, sec.FLOOR_Z), (sec.FLOOR[0], 0, 0), (0, -sec.FLOOR[1], 0)))
        quads.append((stack, (-h, h, 0.0), (SIZE, 0, 0), (0, -SIZE, 0)))
    else:  # reflection
        mirror = {"name": "mirror", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0, "roughnessFactor": 0.0}}
        quads.append((mirror, (-sec.MIRROR_HALF, sec.MIRROR_Y, -sec.MIRROR_HALF), (2 * sec.MIRROR_HALF, 0, 0), (0, 0, 2 * sec.MIRROR_HALF)))
        quads.append((stack, (-h, sec.MIRROR_Y + SIZE, sec.OBJECT_Z), (SIZE, 0, 0), (0, -SIZE, 0)))
    # tiny quads keep the layer materials in the library (outside every measured region, near the floor corner)
    for k, lm in enumerate((layer0, layer1)):
        quads.append((lm, (-1.9 + 0.05 * k, -1.4, sec.FLOOR_Z + 0.05 if view != "reflection" else -1.2), (0.02, 0, 0), (0, -0.02, 0)))
    blob = bytearray()
    views, accessors, meshes, nodes, materials = [], [], [], [], []

    def add(data, ttype, count, component, mins=None, maxs=None):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": count, "type": ttype}
        if mins:
            acc["min"], acc["max"] = mins, maxs
        accessors.append(acc)
        return len(accessors) - 1

    for qi, (material, o, U, V) in enumerate(quads):
        pos, nor, uvs, tan = [], [], [], []
        N = (U[1] * -V[2] - U[2] * -V[1], U[2] * -V[0] - U[0] * -V[2], U[0] * -V[1] - U[1] * -V[0])
        ln = math.sqrt(sum(c * c for c in N))
        N = tuple(c / ln for c in N)
        lu = math.sqrt(sum(c * c for c in U))
        for (u, v) in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            pos += [o[i] + U[i] * u + V[i] * v for i in range(3)]
            nor += list(N)
            uvs += [float(u), float(v)]
            tan += [U[0] / lu, U[1] / lu, U[2] / lu, 1.0]
        a = [add(struct.pack("<12f", *pos), "VEC3", 4, 5126, [min(pos[i::3]) for i in range(3)], [max(pos[i::3]) for i in range(3)]),
             add(struct.pack("<12f", *nor), "VEC3", 4, 5126), add(struct.pack("<8f", *uvs), "VEC2", 4, 5126),
             add(struct.pack("<16f", *tan), "VEC4", 4, 5126), add(struct.pack("<6H", 0, 1, 2, 0, 2, 3), "SCALAR", 6, 5123)]
        materials.append(material)
        meshes.append({"primitives": [{"attributes": {"POSITION": a[0], "NORMAL": a[1], "TEXCOORD_0": a[2], "TANGENT": a[3]}, "indices": a[4], "material": qi}]})
        nodes.append({"mesh": qi, "name": material["name"] + f"-{qi}"})
    doc = {
        "asset": {"version": "2.0", "generator": "tools/f15_texture_transform_layers.py"},
        "extensionsUsed": ["KHR_texture_transform"] + sorted({e for m in materials for e in m.get("extensions", {})}),
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": materials,
        "textures": [{"source": i, "sampler": 0} for i in range(len(names))], "images": [{"uri": base.data_uri(images[n])} for n in names],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode(), "byteLength": len(blob)}],
        "bufferViews": views, "accessors": accessors,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.gltf").write_text(json.dumps(doc))
    for name, data in files.items():
        (out / name).write_bytes(data)
    (out / "overrides.json").write_text(json.dumps({"schema": "phosphor-material-overrides", "version": 1, "overrides": [override]}, indent=1))


# --- model ----------------------------------------------------------------------------------------------------------------

def coverage_candidates():
    """name -> (u, v) -> expected coverage bit (1 blocks / visible) or None (excluded)."""
    def ref(u, v):
        return None if abs(u - CUT) < RAMP_EDGE else int(u >= CUT)
    return {"reference": ref, "own-alpha": lambda u, v: 1, "inverted-ramp": lambda u, v: None if abs(u - CUT) < RAMP_EDGE else int(u < CUT)}


def evaluate_cover(case, img, cam, log_text):
    cfg = CASES[case]
    cands = coverage_candidates()
    pix = img.load()
    width, height = img.size
    ref = cands["reference"]
    samples = []  # (px, py, u, v)
    xs, ys = zip(*[cam.project((sx * SIZE / 2, sy * SIZE / 2, 0.0)) for sx in (-1, 1) for sy in (-1, 1)])
    if cfg["view"] == "visibility":
        background = pix[4, 4]
        floorlike = None
        for py in range(int(min(ys)) + 6, int(max(ys)) - 6, 2):
            for px in range(int(min(xs)) + 6, int(max(xs)) - 6, 2):
                d = cam.ray(px, py)
                t = (0.0 - cam.position[2]) / d[2]
                P = tuple(cam.position[i] + d[i] * t for i in range(3))
                u, v = (P[0] + SIZE / 2) / SIZE, (SIZE / 2 - P[1]) / SIZE
                if 0.03 < v < 0.97 and ref(u, v) is not None:
                    samples.append((px, py, u, v))
        def seen(px, py):
            r, g, b = pix[px, py][:3]
            return int(r > g + 60)  # the stack is red, the floor light grey
    elif cfg["view"] == "shadow":
        box = (min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6)
        for py in range(0, height, 2):
            for px in range(0, width, 2):
                if box[0] <= px <= box[2] and box[1] <= py <= box[3]:
                    continue
                e = sec.shadow_expectation(cam, px, py, None)
                if e is None:
                    continue
                u, v = e
                if 0.03 < v < 0.97 and 0.0 <= u <= 1.0 and ref(u, v) is not None and min(u, 1 - u) > 0.06:
                    samples.append((px, py, u, v))
        lumas = [sum(pix[px, py][:3]) / 3.0 for px, py, _, _ in samples]
        mid = (min(lumas) + max(lumas)) / 2.0
        def seen(px, py):
            return int(sum(pix[px, py][:3]) / 3.0 < mid)  # dark = blocked
    else:  # reflection
        for py in range(0, height, 2):
            for px in range(0, width, 2):
                d = cam.ray(px, py)
                if d[1] >= 0:
                    continue
                t = (sec.MIRROR_Y - cam.position[1]) / d[1]
                M = tuple(cam.position[i] + d[i] * t for i in range(3))
                if not t > 0 or abs(M[0]) > sec.MIRROR_HALF - 0.05 or abs(M[2]) > sec.MIRROR_HALF - 0.05 or M[2] < sec.OBJECT_Z + 0.08:
                    continue
                tc = (sec.OBJECT_Z - cam.position[2]) / d[2]
                if tc > 0 and tc < t:
                    H = tuple(cam.position[i] + d[i] * tc for i in range(3))
                    if abs(H[0]) <= SIZE / 2 + 0.05 and sec.MIRROR_Y - 0.05 <= H[1] <= sec.MIRROR_Y + SIZE + 0.05:
                        continue
                r = (d[0], -d[1], d[2])
                if not r[2] < -1e-6:
                    continue
                s = (sec.OBJECT_Z - M[2]) / r[2]
                H = tuple(M[i] + r[i] * s for i in range(3))
                u, v = (H[0] + SIZE / 2) / SIZE, (sec.MIRROR_Y + SIZE - H[1]) / SIZE
                if 0.03 < v < 0.97 and 0.0 < u < 1.0 and ref(u, v) is not None:
                    samples.append((px, py, u, v))
        def seen(px, py):
            r, g, b = pix[px, py][:3]
            return int(r > g + 25)  # red object reflected vs the grey environment
    if not samples:
        raise RuntimeError(f"{case}: no valid pixel")
    agree = {n: 0 for n in cands}
    counted = {n: 0 for n in cands}
    for px, py, u, v in samples:
        s_ = seen(px, py)
        for n, f in cands.items():
            want = f(u, v)
            if want is None:
                continue
            counted[n] += 1
            agree[n] += int(want == s_)
    res = {"valid_pixels": len(samples), "agreement": {n: round(agree[n] / max(1, counted[n]), 4) for n in cands}, "threshold": AGREE}
    res["gate_ok"] = res["agreement"]["reference"] >= AGREE
    res["controls_discriminate"] = all(v < base.CONTROL_MAX for n, v in res["agreement"].items() if n != "reference")
    return res


def emitter_difference(emits, n=40):
    """Normalised model irradiance difference left - right of the floor regions next to the quad (z = 0, facing +Z, y in [0, 1.2]); `emits(u, v)` says
    whether the quad point emits (a form-factor sum, as f15_texture_transform_secondary.emitter_irradiance)."""
    def irradiance(region):
        x0, x1, z0, z1 = region
        total = 0.0
        for a in range(5):
            for b in range(5):
                P = (x0 + (x1 - x0) * (a + .5) / 5, 0.0, z0 + (z1 - z0) * (b + .5) / 5)
                for i in range(n):
                    for j in range(n):
                        u, v = (i + .5) / n, (j + .5) / n
                        if not emits(u, v):
                            continue
                        S = (-0.6 + 1.2 * u, 1.2 - 1.2 * v, 0.0)
                        w = (P[0] - S[0], P[1] - S[1], P[2] - S[2])
                        r2 = sum(c * c for c in w)
                        r = math.sqrt(r2)
                        total += (w[2] / r) * (S[1] / r) / r2
        return total
    l, r = irradiance(sec.EMITTER_REGIONS["left"]), irradiance(sec.EMITTER_REGIONS["right"])
    return (l - r) / (l + r) if l + r > 0 else 0.0


def evaluate_emitter(img, cam):
    (cl, nl), (cr, nr) = sec.region_pixels(cam, img, "emitter", "left"), sec.region_pixels(cam, img, "emitter", "right")
    measured = (sum(cl) - sum(cr)) / 3.0
    th, mm = 0.08, 0.3
    mclass = sec.klass(measured, th)
    cands = {"reference": lambda u, v: u >= CUT, "own-alpha": lambda u, v: True, "inverted-ramp": lambda u, v: u < CUT}
    model = {n: round(emitter_difference(f), 3) for n, f in cands.items()}
    agree = {n: 1.0 if sec.klass(m, mm) == mclass else 0.0 for n, m in model.items()}
    return {"valid_pixels": nl + nr, "measured_difference": round(measured, 4), "threshold": th, "model_difference": model, "agreement": agree,
            "measured_class": mclass, "gate_ok": agree["reference"] == 1.0 and mclass != 0,
            "controls_discriminate": all(v == 0.0 for n, v in agree.items() if n != "reference")}


def evaluate_tex(img, cam, cx=0.0):
    """Palette cells of a quad centred at x = cx (z = 0) through the per-slot transform against the CPU lookup, the primary fixture's decode."""
    pix = img.load()
    xs, ys = zip(*[cam.project((cx + sx * SIZE / 2, sy * SIZE / 2 + (0.0 if cx == 0.0 else 0.0), 0.0)) for sx in (-1, 1) for sy in (-1, 1)])
    samples = []
    T = base.TRANSFORMS["alpha"]
    for py in range(int(min(ys)) + 6, int(max(ys)) - 6, 2):
        for px in range(int(min(xs)) + 6, int(max(xs)) - 6, 2):
            d = cam.ray(px, py)
            t = (0.0 - cam.position[2]) / d[2]
            P = tuple(cam.position[i] + d[i] * t for i in range(3))
            u, v = (P[0] - cx + SIZE / 2) / SIZE, (SIZE / 2 - P[1]) / SIZE
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                continue
            tu, tv = base.transform_uv(T, u, v)
            if base.edge_distance(tu * base.CELLS, tv * base.CELLS) < base.EDGE_MARGIN:
                continue
            samples.append((u, v, pix[px, py]))
    lo, hi = [255] * 3, [0] * 3
    for _, _, rgb in samples:
        for c in range(3):
            lo[c], hi[c] = min(lo[c], rgb[c]), max(hi[c], rgb[c])
    once = lambda u, v: base.transform_uv(T, u, v)
    cands = {"reference": once, "applied-twice": lambda u, v: base.transform_uv(T, *base.transform_uv(T, u, v)),
             "identity": lambda u, v: (u, v)}
    for k in ("baseColor", "emissive", "occlusion", "normal"):
        cands[f"uniform-bake:{k}-transform"] = (lambda tk: (lambda u, v: base.transform_uv(base.TRANSFORMS[tk], u, v)))(k)
    agree = {n: 0 for n in cands}
    for u, v, rgb in samples:
        seen = sum(int(rgb[c] > (lo[c] + hi[c]) / 2.0) << (2 - c) for c in range(3))
        for n, f in cands.items():
            col, row, _, _ = base.cell_of(*f(u, v))
            agree[n] += int(base.code_at(col, row) == seen)
    res = {"valid_pixels": len(samples), "agreement": {n: round(a / len(samples), 4) for n, a in agree.items()}, "threshold": base.AGREEMENT_MIN}
    res["gate_ok"] = res["agreement"]["reference"] >= base.AGREEMENT_MIN
    res["controls_discriminate"] = all(v < base.CONTROL_MAX for n, v in res["agreement"].items() if n != "reference")
    return res


def evaluate_dual(img, cam):
    direct, layered = evaluate_tex(img, cam, -0.7), evaluate_tex(img, cam, 0.7)
    out = {"valid_pixels": direct["valid_pixels"] + layered["valid_pixels"], "direct": direct, "layered": layered,
           "agreement": {"direct:" + k: v for k, v in direct["agreement"].items()} | {"layered:" + k: v for k, v in layered["agreement"].items()}, "threshold": base.AGREEMENT_MIN}
    out["gate_ok"] = direct["gate_ok"] and layered["gate_ok"]
    out["controls_discriminate"] = direct["controls_discriminate"] and layered["controls_discriminate"]
    return out


def quad_halves_rg(img, cam):
    """(R - G) mean of the left and of the right half of the quad's footprint (display units 0..1)."""
    pix = img.load()
    xs, ys = zip(*[cam.project((sx * SIZE / 2, sy * SIZE / 2, 0.0)) for sx in (-1, 1) for sy in (-1, 1)])
    acc = {"left": [0.0, 0], "right": [0.0, 0]}
    for py in range(int(min(ys)) + 8, int(max(ys)) - 8, 2):
        for px in range(int(min(xs)) + 8, int(max(xs)) - 8, 2):
            d = cam.ray(px, py)
            t = (0.0 - cam.position[2]) / d[2]
            x = cam.position[0] + d[0] * t
            u = (x + SIZE / 2) / SIZE
            if abs(u - 0.5) < 0.1 or not (0.08 < u < 0.92):
                continue
            r, g, b = pix[px, py][:3]
            side = "left" if u < 0.5 else "right"
            acc[side][0] += (r - g) / 255.0
            acc[side][1] += 1
    return {k: v[0] / v[1] for k, v in acc.items()}, acc["left"][1] + acc["right"][1]


def region_model(kind, transform):
    """Model R - G sign difference left - right (normalised to [-1, 1]) for the mask / height swap cases, averaged over the quad."""
    def sign_at(u, v):
        tu, _ = base.transform_uv(transform, u, v)
        if kind == "mask":
            tu = min(max(tu, 0.0), 1.0 - 1e-6)  # the mask sampler clamps to the edge
        # mask: R = 255 on the texture's left half (layer 0, red), G on the right (layer 1, green); height: high on the left half: layer 0 (red) wins
        return 1.0 if (tu - math.floor(tu)) < 0.5 else -1.0
    n = 24
    left = sum(sign_at((i + .5) / n * 0.5, (j + .5) / n) for i in range(n) for j in range(n)) / (n * n)
    right = sum(sign_at(0.5 + (i + .5) / n * 0.5, (j + .5) / n) for i in range(n) for j in range(n)) / (n * n)
    return (left - right) / 2.0


def evaluate_region_case(case, img, cam):
    kind = CASES[case]["kind"]
    means, n = quad_halves_rg(img, cam)
    measured = means["left"] - means["right"]
    cands = {"reference": FLIP if kind == "mask" else SWAP, "identity": base.IDENTITY}
    refclass = sec.klass(region_model(kind, cands["reference"]), 0.3)
    for k in CONTROL_SLOTS[kind]:  # only transforms whose model does NOT predict the reference's difference are controls
        if sec.klass(region_model(kind, base.TRANSFORMS[k]), 0.3) != refclass:
            cands[f"uniform-bake:{k}-transform"] = base.TRANSFORMS[k]
    mclass = sec.klass(measured, 0.05)
    model = {nme: round(region_model(kind, t), 3) for nme, t in cands.items()}
    agree = {nme: 1.0 if sec.klass(m, 0.3) == mclass else 0.0 for nme, m in model.items()}
    return {"valid_pixels": n, "measured_difference": round(measured, 4), "threshold": 0.05, "model_difference": model, "agreement": agree, "measured_class": mclass,
            "gate_ok": agree["reference"] == 1.0 and mclass != 0, "controls_discriminate": all(v == 0.0 for nme, v in agree.items() if nme != "reference")}


def evaluate(case, capture, log_text):
    from PIL import Image
    img = Image.open(capture).convert("RGB")
    lo, hi, distance = sec.parse_rig(log_text)
    cam = sec.Camera(lo, hi, distance, CASES[case]["pitch"], *img.size)
    kind = CASES[case]["kind"]
    if kind == "cover":
        res = evaluate_cover(case, img, cam, log_text)
    elif kind == "emitter":
        res = evaluate_emitter(img, cam)
    elif kind == "tex":
        res = evaluate_tex(img, cam)
    elif kind == "dual":
        res = evaluate_dual(img, cam)
    else:
        res = evaluate_region_case(case, img, cam)
    res["case"] = case
    return res


def selftest():
    import tempfile
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            build_scene(pathlib.Path(tmp), case)
            doc = json.loads((pathlib.Path(tmp) / "scene.gltf").read_text())
            assert {m["name"] for m in doc["materials"]} >= ({"stack", "layerA", "shared"} if CASES[case]["kind"] == "dual" else {"stack", "layerA", "layerB"})
            ov = json.loads((pathlib.Path(tmp) / "overrides.json").read_text())["overrides"][0]
            assert ov["name"] == "stack" and len(ov["layers"]["layers"]) == 2
            assert (pathlib.Path(tmp) / "mask.png").is_file()
    # the swap predicts one sign, the identity the opposite, every other transform no difference
    for kind in ("mask", "height"):
        ref = sec.klass(region_model(kind, FLIP if kind == "mask" else SWAP), 0.3)
        assert ref == -1 and sec.klass(region_model(kind, base.IDENTITY), 0.3) == 1
        usable = [k for k in CONTROL_SLOTS[kind] if sec.klass(region_model(kind, base.TRANSFORMS[k]), 0.3) != ref]
        assert len(usable) >= 2, (kind, usable)
    m_ref = emitter_difference(lambda u, v: u >= CUT)
    assert sec.klass(m_ref, 0.3) == -1 and sec.klass(emitter_difference(lambda u, v: True), 0.3) == 0 and sec.klass(emitter_difference(lambda u, v: u < CUT), 0.3) == 1
    c = coverage_candidates()
    assert c["reference"](0.2, 0.5) == 0 and c["reference"](0.8, 0.5) == 1 and c["reference"](0.5, 0.5) is None
    assert c["own-alpha"](0.2, 0.5) == 1
    print("selftest ok")


def capture_command(binary, out, case, no_archive=False, harvest=None):
    cfg = CASES[case]
    cmd = [str(binary), "--bench", "4", "--scene", str(out / "scene.gltf"), "--material-overrides", str(out / "overrides.json"),
           "--corpus-rig", f"0,{cfg['pitch']}", "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", *cfg["shadows"],
           "--lighting", "restir", "--gi", "off", "--reflections", "rt", "--ao", "off", "--lighting-denoise", "custom",
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
    p.add_argument("--out", default="build/f15-texxform-layers")
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--harvest", metavar="DIR")
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
        log = out / "engine.log"
        if a.run:
            if not a.binary:
                print("--run needs --binary", file=sys.stderr)
                return 2
            build_scene(out, case)
            harvest = None
            if a.harvest:
                pathlib.Path(a.harvest).mkdir(parents=True, exist_ok=True)
                harvest = str(pathlib.Path(a.harvest) / f"{case}.raw.json")
            with log.open("w") as stream:
                code = subprocess.run(capture_command(a.binary, out, case, a.no_archive, harvest), stdout=stream, stderr=subprocess.STDOUT).returncode
            text = log.read_text(errors="replace")
            if code != 0 or not (out / "engine.png").is_file() or not re.search(r"^EXIT 0$", text, re.M):
                print(f"{case}: capture failed (exit {code}); see {log}", file=sys.stderr)
                return 6
        result = evaluate(case, out / "engine.png", log.read_text(errors="replace"))
        (out / "result.json").write_text(json.dumps(result, indent=1))
        extra = f" measured {result['measured_difference']} model {result['model_difference']}" if "measured_difference" in result else ""
        print(f"{case}: valid {result['valid_pixels']}{extra} agreement {result['agreement']} threshold {result['threshold']}")
        if not result["controls_discriminate"]:
            status = max(status, 3)
        elif not result["gate_ok"]:
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
