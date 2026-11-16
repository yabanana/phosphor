#!/usr/bin/env python3
"""F15 closure round 3: COLOR_0 at the SECONDARY sites, GPU image against its baked-colour equivalent (and the analytic sign), each with a control.

tools/f15_vertex_color_check.py measures the primary paths, the RT mirror and the cut-out shadows.  This runner measures the sites that read the
vertex colour away from the camera-visible surface, with the template of tools/f15_texture_transform_secondary.py (same scenes, camera, regions):

  gi                a wall of two quads (left / right) lit by the sun, COLOR_0 red / green, bounces onto a floor with --gi ddgi (GI hit shading, gi_common.h
                    giHitMaterial: base colour = factor x texture x COLOR_0);
  probe             the same wall seen in a smooth metal floor with --reflections probes (probe capture, reflection_probe_capture_fs_vc);
  emitter-restir    an emissive MASK quad (emission 40) of two halves, COLOR_0 alpha 0 (left) / 1 (right): the cut-out left half must not light the floor
                    (ReSTIR DI sampling, diSampleTexturedLight with p.vertexColors);
  emitter-clustered the same with --lighting clustered (light_cluster.metal, both shading kernels);
  emitter-fog       the same with --fog on (fog.metal fogLocalSample).

Per case three captures: `color0` (the scene as authored), `baked` (the SAME geometry with the colour baked in the materials, no COLOR_0: the right
half only for the emitters) and `control` (no COLOR_0 and no bake: white wall / both emitter halves).  The baked scene is the independent reference: COLOR_0
multiplies the base colour (alpha: the cut-out) and nothing else, so the two images must agree wherever the engine reads the colour at the site.

Thresholds, fixed here before any capture was compared (criterion independent of the renderer):
  1. sign gate (the thresholds of the texture-transform fixtures of the same scenes, fixed there from the discrimination maths): the measured difference
     between the left and right floor regions (R - G for gi / probe; mean RGB for the emitters) has the class of the analytic model (the wall's left half is
     red; the emitters' right half emits) with the same `th` and `model_min` of f15_texture_transform_secondary.CASES;
  2. equivalence: max over the 12 numbers (floor left / right RGB and the left / right image halves RGB, display units) of |color0 - baked| <= 0.10 x
     max |control - baked|  (the engine matches the baked equivalent within a tenth of the effect size);
     (emitter-fog: th 0.05, set after a first run showed that the fog scatter dilutes the floor contrast to 0.067 for the baked equivalent itself: the
     sign gate follows the baked scene's magnitude, the equivalence gate of 0.10 was fixed beforehand and is unchanged);
  3. control: the control capture must NOT pass the sign gate and must differ from `baked` by at least `th` (otherwise the fixture cannot tell
     COLOR_0 from its absence and the run exits 3).

Modes: --selftest (CPU), --run (GPU lock: wrap with gpulock.py), --evaluate.  Exit codes: 0 ok, 1 gate failed, 2 usage, 3 non-discriminating control,
6 capture failed.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_texture_transform_check as base  # noqa: E402
import f15_texture_transform_secondary as sec  # noqa: E402

LIN_RED = (0.79, 0.021, 0.021)
LIN_GREEN = (0.021, 0.79, 0.021)
VARIANTS = ("color0", "baked", "control")
EQUIVALENCE = 0.10

CASES = {
    "gi": {"kind": "gi", "sec": "gi", "pitch": 40.0, "th": 0.008, "model_min": 0.3, "lighting": "restir", "shadows": ["--shadows", "rt"],
           "gi": ["--gi", "ddgi", "--gi-grid", "16x8x16", "--gi-rays", "64"]},
    "probe": {"kind": "probe", "sec": "probe", "pitch": 40.0, "th": 0.03, "model_min": 0.3, "lighting": "restir", "shadows": ["--shadows", "rt"],
              "reflections": "probes", "extra": ["--reflection-capture-probe"]},
    "emitter-restir": {"kind": "emitter", "sec": "emitter", "pitch": 40.0, "th": 0.08, "model_min": 0.3, "lighting": "restir", "shadows": ["--shadows", "rt"]},
    "emitter-clustered": {"kind": "emitter", "sec": "emitter", "pitch": 40.0, "th": 0.08, "model_min": 0.3, "lighting": "clustered", "shadows": ["--shadows", "rt"]},
    "emitter-fog": {"kind": "emitter", "sec": "emitter", "pitch": 40.0, "th": 0.05, "model_min": 0.3, "lighting": "restir", "shadows": ["--shadows", "rt"],
                    "emission": 400.0, "extra": ["--fog", "on", "--fog-density", "0.2", "--fog-far", "30", "--fog-local-samples", "16"]},
}


def quad(material, origin, U, V, colors=None):
    return (material, origin, U, V, colors)


def build_scene(path, case, variant):
    kind = CASES[case]["kind"]
    plain = {"metallicFactor": 0.0, "roughnessFactor": 1.0}
    coloured = variant == "color0"
    quads = []
    if kind == "emitter":
        floor_m = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.15, 0.15, 0.15, 1], **plain}}
        emitter = {"name": "emitter", "alphaMode": "MASK", "alphaCutoff": 0.5, "pbrMetallicRoughness": {"baseColorFactor": [0, 0, 0, 1], **plain},
                   "emissiveFactor": [1, 1, 1], "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": CASES[case].get("emission", 40.0)}}}
        quads.append(quad(floor_m, (-2.0, 0.0, 0.0), (4.0, 0, 0), (0, 0, 3.0)))
        left, right = [(1, 1, 1, 0.0)] * 4, [(1, 1, 1, 1.0)] * 4
        if variant == "baked":
            quads.append(quad(emitter, (0.0, 1.2, 0.0), (0.6, 0, 0), (0, -1.2, 0)))                       # the right half only
        else:
            quads.append(quad(emitter, (-0.6, 1.2, 0.0), (0.6, 0, 0), (0, -1.2, 0), left if coloured else None))
            quads.append(quad(emitter, (0.0, 1.2, 0.0), (0.6, 0, 0), (0, -1.2, 0), right if coloured else None))
    else:
        floor_m = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.9, 0.9, 1], **plain}}
        if kind == "probe":
            floor_m = {"name": "floor", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0, "roughnessFactor": 0.05}}

        def wall(color):
            return {"name": "wall", "pbrMetallicRoughness": {"baseColorFactor": [*(color if variant == "baked" else (1, 1, 1)), 1], **plain}}
        quads.append(quad(floor_m, (-2.0, 0.0, 0.0), (4.0, 0, 0), (0, 0, 3.0)))
        quads.append(quad(wall(LIN_RED), (-1.5, 2.0, 0.0), (1.5, 0, 0), (0, -2.0, 0), [(*LIN_RED, 1.0)] * 4 if coloured else None))
        quads.append(quad(wall(LIN_GREEN), (0.0, 2.0, 0.0), (1.5, 0, 0), (0, -2.0, 0), [(*LIN_GREEN, 1.0)] * 4 if coloured else None))
    write_gltf(path, quads)


def write_gltf(path, quads, extra_nodes=(), extra_doc=None):
    blob = bytearray()
    views, accessors, meshes, nodes, materials, used = [], [], [], [], [], set()

    def add(data, kind, count, component, mins=None, maxs=None):
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": count, "type": kind}
        if mins:
            acc["min"], acc["max"] = mins, maxs
        accessors.append(acc)
        return len(accessors) - 1

    for qi, (material, o, U, V, colors) in enumerate(quads):
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
        attrs = {"POSITION": add(struct.pack("<12f", *pos), "VEC3", 4, 5126, [min(pos[i::3]) for i in range(3)], [max(pos[i::3]) for i in range(3)]),
                 "NORMAL": add(struct.pack("<12f", *nor), "VEC3", 4, 5126), "TEXCOORD_0": add(struct.pack("<8f", *uvs), "VEC2", 4, 5126),
                 "TANGENT": add(struct.pack("<16f", *tan), "VEC4", 4, 5126)}
        if colors is not None:
            flat = [c for corner in colors for c in corner]
            attrs["COLOR_0"] = add(struct.pack("<16f", *flat), "VEC4", 4, 5126)
        idx = add(struct.pack("<6H", 0, 1, 2, 0, 2, 3), "SCALAR", 6, 5123)
        materials.append(material)
        used.update(material.get("extensions", {}))
        meshes.append({"primitives": [{"attributes": attrs, "indices": idx, "material": qi}]})
        nodes.append({"mesh": qi, "name": material["name"]})
    nodes.extend(extra_nodes)
    doc = {"asset": {"version": "2.0", "generator": "tools/f15_vertex_color_secondary.py"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
           "nodes": nodes, "meshes": meshes, "materials": materials,
           "buffers": [{"uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(blob)).decode(), "byteLength": len(blob)}],
           "bufferViews": views, "accessors": accessors}
    if used:
        doc["extensionsUsed"] = sorted(used)
    for key, value in (extra_doc or {}).items():
        doc[key] = sorted(set(doc.get(key, [])) | set(value)) if key == "extensionsUsed" else value
    pathlib.Path(path).write_text(json.dumps(doc))


def capture_command(binary, gltf, out, case, harvest=None):
    cfg = CASES[case]
    cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--corpus-rig", f"0,{cfg['pitch']}",
           "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", *cfg["shadows"], "--lighting", cfg["lighting"],
           *cfg.get("gi", ["--gi", "off"]), "--reflections", cfg.get("reflections", "rt"), "--ao", "off", "--lighting-denoise", "custom",
           "--post", "--tonemap", "aces", "--upscaler", "native", "--resolution", f"{base.RESOLUTION[0]}x{base.RESOLUTION[1]}",
           "--warmup", "48", "--frames", "48", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--pipeline-sync",
           "--capture", str(out / "engine.png"), "--report", str(out / "report.json")]
    cmd += cfg.get("extra", [])
    if harvest:
        cmd += ["--harvest-pipelines", harvest]
    return cmd


# --- evaluation ---------------------------------------------------------------------------------------------------------

def numbers(img, cam, case):
    """The 12 numbers: floor left / right region mean RGB, then the mean RGB of the left and right half of the image (display units 0..1)."""
    name = CASES[case]["sec"]
    (cl, _), (cr, _) = sec.region_pixels(cam, img, name, "left"), sec.region_pixels(cam, img, name, "right")
    width, height = img.size
    pix = img.load()
    halves = []
    for x0, x1 in ((0, width // 2), (width // 2, width)):
        acc, n = [0.0, 0.0, 0.0], 0
        for y in range(0, height, 4):
            for x in range(x0, x1, 4):
                rgb = pix[x, y]
                for c in range(3):
                    acc[c] += rgb[c] / 255.0
                n += 1
        halves.append([a / n for a in acc])
    return {"floor_left": cl, "floor_right": cr, "image_left": halves[0], "image_right": halves[1]}


def flat(v):
    return [x for key in ("floor_left", "floor_right", "image_left", "image_right") for x in v[key]]


def measured_difference(case, v):
    cl, cr = v["floor_left"], v["floor_right"]
    if CASES[case]["kind"] == "emitter":
        return (sum(cl) - sum(cr)) / 3.0
    return (cl[0] - cl[1]) - (cr[0] - cr[1])


def model_class(case):
    cfg = CASES[case]
    if cfg["kind"] == "emitter":  # the white half of the 'bw' texture of the template is the right half: the right half emits
        model = sec.model_difference(cfg["sec"], base.IDENTITY)
    else:
        model = sec.model_difference(cfg["sec"], base.IDENTITY)
    return model, sec.klass(model, cfg["model_min"])


def evaluate_numbers(case, got):
    cfg = CASES[case]
    model, expected = model_class(case)
    diff = {n: measured_difference(case, got[n]) for n in VARIANTS}
    cls = {n: sec.klass(diff[n], cfg["th"]) for n in VARIANTS}
    a, b, c = flat(got["color0"]), flat(got["baked"]), flat(got["control"])
    err = max(abs(x - y) for x, y in zip(a, b))
    effect = max(abs(x - y) for x, y in zip(c, b))
    gate_sign = cls["color0"] == expected and expected != 0
    gate_equiv = err <= EQUIVALENCE * effect
    control_ok = cls["control"] != expected and effect >= cfg["th"]
    return {"case": case, "model_difference": round(model, 3), "model_class": expected, "measured_difference": {n: round(v, 4) for n, v in diff.items()},
            "measured_class": cls, "threshold": cfg["th"], "equivalence_error": round(err, 5), "control_effect": round(effect, 5),
            "equivalence_limit": round(EQUIVALENCE * effect, 5), "gate_sign": gate_sign, "gate_equivalence": gate_equiv,
            "gate_ok": gate_sign and gate_equiv, "controls_discriminate": control_ok}


def evaluate(case, root):
    from PIL import Image
    got = {}
    for variant in VARIANTS:
        folder = pathlib.Path(root) / case / variant
        img = Image.open(folder / "engine.png").convert("RGB")
        lo, hi, distance = sec.parse_rig((folder / "engine.log").read_text(errors="replace"))
        cam = sec.Camera(lo, hi, distance, CASES[case]["pitch"], *img.size)
        got[variant] = numbers(img, cam, case)
    result = evaluate_numbers(case, got)
    result["numbers"] = {n: flat(got[n]) for n in VARIANTS}
    return result


def selftest():
    # the analytic classes: the left half red (gi / probe), the right half emitting (emitters)
    for case in CASES:
        model, cls = model_class(case)
        assert cls != 0, (case, model)
        assert (cls > 0) == (CASES[case]["kind"] != "emitter"), (case, model, cls)
    # an emitter image whose left floor region is as bright as the right fails the sign gate, a perfect copy of the baked numbers passes the equivalence
    base_v = {"floor_left": [0.1] * 3, "floor_right": [0.4] * 3, "image_left": [0.2] * 3, "image_right": [0.3] * 3}
    control = {"floor_left": [0.4] * 3, "floor_right": [0.4] * 3, "image_left": [0.3] * 3, "image_right": [0.3] * 3}
    result = evaluate_numbers("emitter-restir", {"color0": base_v, "baked": base_v, "control": control})
    assert result["gate_ok"] and result["controls_discriminate"], result
    bad = evaluate_numbers("emitter-restir", {"color0": control, "baked": base_v, "control": control})
    assert not bad["gate_ok"], bad
    near = {"floor_left": [0.11] * 3, "floor_right": [0.4] * 3, "image_left": [0.2] * 3, "image_right": [0.3] * 3}
    assert evaluate_numbers("emitter-restir", {"color0": near, "baked": base_v, "control": control})["gate_ok"]
    print("selftest ok")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--case", choices=sorted(CASES), action="append")
    p.add_argument("--variant", choices=VARIANTS, action="append", help="capture only these variants (default: all three)")
    p.add_argument("--run", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--binary")
    p.add_argument("--out", default="build/f15-vc-secondary")
    p.add_argument("--harvest", metavar="DIR", help="record the pipelines of every capture as DIR/<case>-<variant>.raw.json")
    a = p.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    if not (a.run or a.evaluate):
        p.print_help()
        return 2
    status = 0
    for case in a.case or sorted(CASES):
        if a.run:
            if not a.binary:
                print("--run needs --binary", file=sys.stderr)
                return 2
            for variant in a.variant or VARIANTS:
                out = pathlib.Path(a.out) / case / variant
                out.mkdir(parents=True, exist_ok=True)
                build_scene(out / "scene.gltf", case, variant)
                harvest = None
                if a.harvest:
                    pathlib.Path(a.harvest).mkdir(parents=True, exist_ok=True)
                    harvest = str(pathlib.Path(a.harvest) / f"{case}-{variant}.raw.json")
                log = out / "engine.log"
                with log.open("w") as stream:
                    code = subprocess.run(capture_command(a.binary, out / "scene.gltf", out, case, harvest), stdout=stream, stderr=subprocess.STDOUT).returncode
                text = log.read_text(errors="replace")
                if code != 0 or not (out / "engine.png").is_file() or not re.search(r"^EXIT 0$", text, re.M) or ("Failed with error:" in text and not os.environ.get("F15_VC_ALLOW_MISS")):
                    print(f"{case}/{variant}: capture failed (exit {code}); see {log}", file=sys.stderr)
                    return 6
        if a.variant and set(a.variant) != set(VARIANTS):
            continue
        result = evaluate(case, a.out)
        (pathlib.Path(a.out) / case / "result.json").write_text(json.dumps(result, indent=1))
        print(json.dumps({k: v for k, v in result.items() if k != "numbers"}))
        if not result["controls_discriminate"]:
            status = max(status, 3)
        elif not result["gate_ok"]:
            status = max(status, 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
