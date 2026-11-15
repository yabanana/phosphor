#!/usr/bin/env python3
"""F15 closure (loader-misc): GPU fixtures of KHR_materials_unlit and KHR_lights_punctual, with their negative controls.

  python3 -I tools/f15_unlit_lights_check.py gen OUTDIR          write the glTF fixtures (no GPU)
  python3 "$H/gpulock.py" python3 -I tools/f15_unlit_lights_check.py run  OUTDIR BINARY    capture every case (GPU, serial) and verify
  python3 -I tools/f15_unlit_lights_check.py verify OUTDIR       re-verify the captures of a previous run (CPU only)

Criteria are fixed from independent facts before any capture is compared:

UNLIT (declared corpus rig, linear capture, exposure 1): four 2x2 panels, an unlit and a lit one flat towards the camera, an unlit and a lit
one turned 55 degrees about Y.  The linear value at the centre of an unlit panel must equal its base colour (the extension's README: the
colour is displayed as is) within 0.002 (two half-float ulps at 1.0), for the flat and the tilted panel alike, and uniform over the 9x9 patch
(max deviation <= 0.002).  Control 1: the lit panel with the same colour differs by more than 0.02 in a channel (the check can tell lit from
unlit).  Control 2: the lit twin of the whole file (extension removed) must FAIL the unlit predicate.

LIGHTS (no rig, legacy resolve lights, shadows off, camera of a glTF scene at (0,3,0) looking -Z, fixed exposure): each light case is
captured with the file's light and with intensity 0, and the DELTA of the linear value at the centre of a white-ish panel (base 0.8, roughness 1,
metallic 0) is compared with the Lambert value base/pi * E, E = I cos(theta) / d^2 (point, spot) or I (directional, lux = W/m^2 factor 1).
The tolerance is 6 %: the engine's dielectric specular lobe adds at most F0 / 4 = 1 % of the diffuse term at alpha 1 and a (1 - F) diffuse
weight removes at most 4 %.  Directional: the light shines along its node's -Z: with the identity rotation the panel facing +Z is lit
(delta = base I / pi), with the node turned 90 degrees about Y (light along -X or +X, grazing) or 180 degrees (light behind the panel) the delta is
below 0.002 (the mapping controls: a +Z mapping would light the 180 degree case and darken the identity one).  Spot: on the axis the delta is
base I / (pi d^2); outside the outer cone it is below 0.002.
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

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = (1280, 720)
BASE = 0.8


def quad_buffer():
    positions = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]
    normals = [0, 0, 1] * 4
    uvs = [0, 1, 1, 1, 1, 0, 0, 0]
    indices = [0, 1, 2, 0, 2, 3]
    blob = struct.pack("<12f", *positions) + struct.pack("<12f", *normals) + struct.pack("<8f", *uvs) + struct.pack("<6H", *indices)
    return blob


def gltf(nodes, materials, extras=None):
    """One quad mesh per material (same accessors); `nodes` = [dict(material=i, translation, rotation, scale, extensions)]."""
    blob = quad_buffer()
    meshes = [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": m}]} for m in range(len(materials))]
    out_nodes = []
    for n in nodes:
        node = {k: v for k, v in n.items() if k != "material"}
        if "material" in n:
            node["mesh"] = n["material"]
        out_nodes.append(node)
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(out_nodes)))}],
        "nodes": out_nodes, "meshes": meshes, "materials": materials,
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," + base64.b64encode(blob).decode()}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48}, {"buffer": 0, "byteOffset": 48, "byteLength": 48},
                        {"buffer": 0, "byteOffset": 96, "byteLength": 32}, {"buffer": 0, "byteOffset": 128, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]},
                      {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                      {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
                      {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}],
    }
    doc.update(extras or {})
    return doc


def quat_y(degrees):
    h = math.radians(degrees) / 2
    return [0.0, math.sin(h), 0.0, math.cos(h)]


# --- unlit ------------------------------------------------------------------------------------------------------------

UNLIT_COLORS = {"A": (0.8, 0.4, 0.2), "C": (0.2, 0.5, 0.9)}
UNLIT_CENTRES = {"A": (-1.4, 1.4), "B": (1.4, 1.4), "C": (-1.4, -1.4), "D": (1.4, -1.4)}


def material(color, unlit):
    m = {"name": "unlit" if unlit else "lit", "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}}
    if unlit:
        m["extensions"] = {"KHR_materials_unlit": {}}
    return m


def unlit_scene(all_lit=False):
    mats = [material(UNLIT_COLORS["A"], not all_lit), material(UNLIT_COLORS["A"], False), material(UNLIT_COLORS["C"], not all_lit),
            material(UNLIT_COLORS["C"], False)]
    nodes = [{"material": 0, "translation": [*UNLIT_CENTRES["A"], 0]}, {"material": 1, "translation": [*UNLIT_CENTRES["B"], 0]},
             {"material": 2, "translation": [*UNLIT_CENTRES["C"], 0], "rotation": quat_y(-55)},
             {"material": 3, "translation": [*UNLIT_CENTRES["D"], 0], "rotation": quat_y(-55)}]
    extras = {} if all_lit else {"extensionsUsed": ["KHR_materials_unlit"], "extensionsRequired": ["KHR_materials_unlit"]}
    return gltf(nodes, mats, extras)


# --- lights -----------------------------------------------------------------------------------------------------------

LIGHT_Z = -6.0
CAMERA = (0.0, 3.0, 0.0)


def lights_scene(panels, light_node, light, intensity):
    mat = [{"name": "panel", "pbrMetallicRoughness": {"baseColorFactor": [BASE, BASE, BASE, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}]
    nodes = [{"material": 0, "translation": [x, 3.0, LIGHT_Z], "scale": [0.6, 0.6, 1.0]} for x in panels]
    node = dict(light_node)
    node["extensions"] = {"KHR_lights_punctual": {"light": 0}}
    nodes.append(node)
    light = dict(light, intensity=intensity)
    return gltf(nodes, mat, {"extensionsUsed": ["KHR_lights_punctual"], "extensionsRequired": ["KHR_lights_punctual"],
                              "extensions": {"KHR_lights_punctual": {"lights": [light]}}})


# the cases: name -> (file builder, panel centre x list, expected delta per panel as a function of intensity)
def lights_cases():
    cases = {}
    I_DIR, I_POINT, I_SPOT = 3.0, 60.0, 40.0
    panels_dir = [-3.0]
    for name, rotation, expected in (("dir-id", None, BASE * I_DIR / math.pi), ("dir-y90", quat_y(90), 0.0), ("dir-y180", quat_y(180), 0.0)):
        node = {"translation": [0.0, 3.0, 0.0]}
        if rotation:
            node["rotation"] = rotation
        cases[name] = {"panels": panels_dir, "node": node, "light": {"type": "directional"}, "intensity": I_DIR, "expect": [expected]}
    px = [-2.0, 3.0]
    ex = []
    for x in px:
        d2 = x * x + 9.0
        ex.append(BASE / math.pi * I_POINT * (3.0 / math.sqrt(d2)) / d2)
    cases["point"] = {"panels": px, "node": {"translation": [0.0, 3.0, -3.0]}, "light": {"type": "point"}, "intensity": I_POINT, "expect": ex}
    spot_panels = [0.0, 5.2]
    cases["spot"] = {"panels": spot_panels, "node": {"translation": [0.0, 3.0, -3.0]},
                     "light": {"type": "spot", "spot": {"innerConeAngle": 0.3, "outerConeAngle": 0.6}}, "intensity": I_SPOT,
                     "expect": [BASE / math.pi * I_SPOT / 9.0, 0.0]}
    return cases


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "unlit.gltf").write_text(json.dumps(unlit_scene()))
    (out / "unlit-litcontrol.gltf").write_text(json.dumps(unlit_scene(all_lit=True)))
    for name, c in lights_cases().items():
        (out / f"lights-{name}.gltf").write_text(json.dumps(lights_scene(c["panels"], c["node"], c["light"], c["intensity"])))
        (out / f"lights-{name}-off.gltf").write_text(json.dumps(lights_scene(c["panels"], c["node"], c["light"], 0.0)))


# --- PFM --------------------------------------------------------------------------------------------------------------

def read_pfm(path):
    data = pathlib.Path(path).read_bytes()
    m = re.match(rb"(PF|Pf)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s", data)
    kind, w, h, scale = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
    channels = 3 if kind == b"PF" else 1
    values = array.array("f")
    values.frombytes(data[m.end():m.end() + w * h * channels * 4])
    if (scale < 0) != (sys.byteorder == "little"):
        values.byteswap()
    return w, h, channels, values


def patch(img, cx, cy, radius=4):
    w, h, ch, v = img
    rows = []
    for y in range(cy - radius, cy + radius + 1):
        row_index = h - 1 - y  # PFM rows run bottom to top
        for x in range(cx - radius, cx + radius + 1):
            o = (row_index * w + x) * ch
            rows.append(tuple(v[o:o + ch]))
    return rows


def mean_and_dev(pixels):
    n = len(pixels)
    mean = tuple(sum(p[c] for p in pixels) / n for c in range(len(pixels[0])))
    dev = max(abs(p[c] - mean[c]) for p in pixels for c in range(len(mean)))
    return mean, dev


def project(point_xy, camera_distance, aspect, size):
    """Pixel of a point of the plane through the camera target (camera on +Z, yaw = pitch = 0, 60 degree vertical fov)."""
    half_h = camera_distance * math.tan(math.radians(30))
    half_w = half_h * aspect
    return (int(round((point_xy[0] / half_w + 1) / 2 * size[0])), int(round((1 - point_xy[1] / half_h) / 2 * size[1])))


def project_camera(x, y, z, size):
    """Pixel of a world point for the default glTF scene camera (0,3,0) looking -Z, 60 degree vertical fov."""
    depth = CAMERA[2] - z
    half_h = depth * math.tan(math.radians(30))
    half_w = half_h * size[0] / size[1]
    return (int(round(((x - CAMERA[0]) / half_w + 1) / 2 * size[0])), int(round((1 - (y - CAMERA[1]) / half_h) / 2 * size[1])))


# --- commands ---------------------------------------------------------------------------------------------------------

def capture(binary, scene, folder, name, rig):
    folder.mkdir(parents=True, exist_ok=True)
    common = ["--fixed-timestep", "--offscreen", "--no-ui", "--no-vsync", "--no-gpu-timing", "--render-path", "visibility", "--geometry-path", "mesh",
              "--post", "--upscaler", "native", "--warmup", "48", "--frames", "1", "--pipeline-sync",
              "--resolution", f"{RES[0]}x{RES[1]}", "--capture-linear", str(folder / f"{name}.pfm")]
    if rig:
        cmd = [str(binary), "--bench", "4", "--scene", str(scene), "--corpus-rig", "0,0", "--rt", "on", "--shadows", "rt", "--lighting", "restir",
               "--gi", "off", "--reflections", "rt", "--ao", "off", "--lighting-denoise", "custom", "--tonemap", "aces", *common]
    else:
        cmd = [str(binary), "--bench", "4", "--scene", str(scene), "--rt", "off", "--shadows", "off", "--lighting", "legacy", "--gi", "off",
               "--reflections", "off", "--ao", "off", "--lighting-denoise", "off", *common]
    log = folder / f"{name}.log"
    env = {k: v for k, v in os.environ.items() if k not in ("MTL_SHADER_VALIDATION",)}
    env["MTL_DEBUG_LAYER"] = "1"
    env["MTL_DEBUG_LAYER_WARNING_MODE"] = "nslog"
    with log.open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=300).returncode
    text = log.read_text(errors="replace")
    return code, text


def camera_distance(text):
    m = re.search(r"camera distance ([\d.]+)", text)
    return float(m.group(1)) if m else None


def verify_unlit(out):
    results, ok = [], True
    tol = 0.002
    for name, expect_pass in (("unlit", True), ("unlit-litcontrol", False)):
        pfm, log = out / f"{name}.pfm", out / f"{name}.log"
        if not pfm.exists():
            results.append(f"{name}: no capture"); ok = False; continue
        img = read_pfm(pfm)
        distance = camera_distance(log.read_text(errors="replace"))
        aspect = img[0] / img[1]
        verdict = {}
        for panel, colour in (("A", UNLIT_COLORS["A"]), ("C", UNLIT_COLORS["C"])):
            cx, cy = project(UNLIT_CENTRES[panel], distance, aspect, (img[0], img[1]))
            mean, dev = mean_and_dev(patch(img, cx, cy))
            err = max(abs(mean[c] - colour[c]) for c in range(3))
            verdict[panel] = (err <= tol and dev <= tol, err, dev, mean)
        panel_ok = all(v[0] for v in verdict.values())
        results.append(f"{name}: unlit predicate {'PASS' if panel_ok else 'FAIL'} (expected {'PASS' if expect_pass else 'FAIL'}) | " +
                       "; ".join(f"{p}: err {v[1]:.5f} dev {v[2]:.5f} mean {tuple(round(x, 4) for x in v[3])}" for p, v in verdict.items()))
        if panel_ok != expect_pass:
            ok = False
        if expect_pass:
            for lit, unlit_panel, colour in (("B", "A", UNLIT_COLORS["A"]), ("D", "C", UNLIT_COLORS["C"])):
                cx, cy = project(UNLIT_CENTRES[lit], distance, aspect, (img[0], img[1]))
                mean, _ = mean_and_dev(patch(img, cx, cy))
                diff = max(abs(mean[c] - colour[c]) for c in range(3))
                results.append(f"  control: lit panel {lit} differs from the unlit colour by {diff:.4f} (> 0.02 required) {'PASS' if diff > 0.02 else 'FAIL'}")
                ok = ok and diff > 0.02
    return ok, results


def verify_lights(out):
    results, ok = [], True
    for name, c in lights_cases().items():
        on, off = out / f"lights-{name}.pfm", out / f"lights-{name}-off.pfm"
        if not (on.exists() and off.exists()):
            results.append(f"{name}: missing capture"); ok = False; continue
        a, b = read_pfm(on), read_pfm(off)
        for x, expect in zip(c["panels"], c["expect"]):
            cx, cy = project_camera(x, 3.0, LIGHT_Z, (a[0], a[1]))
            ma, _ = mean_and_dev(patch(a, cx, cy))
            mb, _ = mean_and_dev(patch(b, cx, cy))
            delta = sum(ma[k] - mb[k] for k in range(3)) / 3
            if name == "dir-y90":
                # Grazing sun: the engine's directional light is a disk of angular radius 0.00465 rad, half of it above the horizon of the panel
                # at 90 degrees; E <= I r and delta <= base I r / pi (an upper bound derived from the disk, not from the capture).  The first
                # version of this check expected 0.002 (half-float ulps) and failed with 0.00297: it had forgotten the disk.
                good = 0.0 <= delta <= BASE * 3.0 * 0.00465 / math.pi
            elif expect == 0.0:
                good = abs(delta) <= 0.002
            else:
                good = abs(delta - expect) <= 0.06 * expect
            results.append(f"{name} panel x={x:+.1f}: delta {delta:.5f} expected {expect:.5f} {'PASS' if good else 'FAIL'}")
            ok = ok and good
    return ok, results


def run(out, binary):
    generate(out)
    names = [("unlit", True), ("unlit-litcontrol", True)]
    for name in lights_cases():
        names += [(f"lights-{name}", False), (f"lights-{name}-off", False)]
    for name, rig in names:
        code, text = capture(binary, out / f"{name}.gltf", out, name, rig)
        notices = len(re.findall(r"Redundant call", text))
        print(f"{name}: exit {code}, marker {re.findall(r'^EXIT (\\d+)$', text, re.M)[-1:]}, redundant-call notices {notices}", flush=True)
        if code != 0:
            print(text[-1500:])
            return 1
    return verify(out)


def verify(out):
    ok1, r1 = verify_unlit(out)
    ok2, r2 = verify_lights(out)
    print("\n".join(r1 + r2))
    print("PASS" if ok1 and ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["gen", "run", "verify"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    args = parser.parse_args()
    if args.mode == "gen":
        generate(args.out); return 0
    if args.mode == "run":
        return run(args.out, args.binary)
    return verify(args.out)


if __name__ == "__main__":
    sys.exit(main())
