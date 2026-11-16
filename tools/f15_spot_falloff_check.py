#!/usr/bin/env python3
"""F15 closure: the spot cone falloff of every direct-lighting path against the analytic engine convention smoothstep(cosOuter, cosInner, cos angle).

One spot light (inner 0.3, outer 0.6 rad, 40 W/sr, at (0,3,-3) looking -Z) lights white-ish panels (base 0.8, roughness 1) at 0, 0.38, 0.5 and 0.55 rad
from the axis and one outside the cone, in a glTF scene (KHR_lights_punctual, no rig).  Each capture is differenced with the light at intensity 0, and the
delta of a panel is divided by the delta of the on-axis panel, which cancels every common factor (BRDF, exposure, the 2-3 % Lambert deficit):
measured ratio r_i = delta_i / delta_0 against the analytic r_i = (cos theta_i / d_i^2) / (1 / 9) x smoothstep(t_i).  Paths: `legacy` (resolve + forward
lights), `restir` (ReSTIR DI) and `clustered` (clustered DI); RT reflection secondary lighting, the probes and the GI call the same evaluators
(restir_common.h `lightSample`, gi_common.h) and are covered by the two implementations measured here.  PASS per path: every panel within 3 % of the
smoothstep ratio (relative), the outside panel below 0.002; NEGATIVE CONTROL: the linear falloff t and the specification's informative t^2 must each miss
at least one panel by more than 3 % (at these angles they differ from smoothstep by 13 / 34 / 45 % and more), otherwise the fixture cannot tell the
falloffs apart.

  python3 "$H/gpulock.py" python3 -I tools/f15_spot_falloff_check.py run OUTDIR [BINARY]
  python3 -I tools/f15_spot_falloff_check.py verify OUTDIR

F15 closure round 3: SECONDARY lighting arms (`arms-run` / `arms-verify` / `selftest`).  The sentence above ("covered by the two implementations measured
here") was an argument from the code, not a measurement; the arms measure it.  The three secondary sites that light a surface away from the camera-visible one:

  rtrefl   RT reflection secondary lighting (reflections.metal reflectionSecondary -> diSampleLight): a small lit panel seen in a smooth metal floor
           (--reflections rt);
  probe    probe capture lighting (reflection_probes.metal, raster capture): the same panel in a roughness 0.05 metal floor (--reflections probes);
  gi       GI hit shading (gi_common.h giSecondaryRadiance): the panel's bounce onto a diffuse floor that the spot never lights directly (--gi ddgi).

Design (everything but the light's axis is identical between captures, so the geometry term cancels exactly): one spot (inner 0.3, outer 0.6 rad) at
(0, 1.5, 0) and a small white-ish panel facing +Z at (0, 1.5, -3); the light node is TILTED about X by `tilt` rad (0, 0.38, 0.45, 0.5, 0.55 and 0.85 outside the
cone) so the panel is `tilt` off the axis.  Each arm is captured with the light on at every tilt and with intensity 0; the DELTA (on - off, linear, mean RGB) of a
patch is divided by the delta at tilt 0.  Model (independent of the renderer): the mean over the panel (central part for the mirror arms, whole panel through the
form factor to the floor region for gi) of cos(panel) / d^2 x falloff(t); falloff = smoothstep (engine convention), linear and the informative t^2 as controls.
Thresholds fixed before any capture was compared: every in-cone tilt within `TOL_ARM` of the smoothstep model (relative), the outside delta below 1 % of the
tilt 0 delta, and BOTH controls (linear, t^2) must miss at least one tilt by more than `TOL_ARM` (`selftest` proves the controls discriminate on the model
itself for every arm: the minimum over arms of the maximum miss is above twice the tolerance).

  python3 "$H/gpulock.py" python3 -I tools/f15_spot_falloff_check.py arms-run OUTDIR [BINARY] [--arms rtrefl,probe,gi]
  python3 -I tools/f15_spot_falloff_check.py arms-verify OUTDIR [--arms ...]
  python3 -I tools/f15_spot_falloff_check.py selftest
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_unlit_lights_check as base  # noqa: E402

INNER, OUTER = 0.3, 0.6
ANGLES = [0.0, 0.38, 0.5, 0.55]
OUTSIDE_X = 5.2
TOL = 0.03
PATHS = {
    "legacy": ["--rt", "off", "--shadows", "off", "--lighting", "legacy", "--reflections", "off", "--lighting-denoise", "off"],
    "restir": ["--rt", "on", "--shadows", "off", "--lighting", "restir", "--reflections", "off", "--lighting-denoise", "off"],
    "clustered": ["--rt", "on", "--shadows", "off", "--lighting", "clustered", "--reflections", "off", "--lighting-denoise", "off"],
}


def panels():
    return [3.0 * math.tan(a) for a in ANGLES] + [OUTSIDE_X]


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def t_of(angle):
    return (math.cos(angle) - math.cos(OUTER)) / (math.cos(INNER) - math.cos(OUTER))


def geometry(x):
    d2 = x * x + 9.0
    return (3.0 / math.sqrt(d2)) / d2


def model_ratio(x, angle, falloff):
    return geometry(x) / geometry(0.0) * falloff(t_of(angle))


MODELS = {"smoothstep": smoothstep, "linear": lambda t: min(1.0, max(0.0, t)), "t2": lambda t: min(1.0, max(0.0, t)) ** 2}


def scene(intensity):
    c = {"panels": panels(), "node": {"translation": [0.0, 3.0, -3.0]},
         "light": {"type": "spot", "spot": {"innerConeAngle": INNER, "outerConeAngle": OUTER}}, "intensity": 40.0}
    return base.lights_scene(c["panels"], c["node"], c["light"], intensity)


def capture(binary, out, path, name, gltf):
    cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--render-path", "visibility", "--geometry-path", "mesh", "--gi", "off", "--ao", "off",
           *PATHS[path], "--post", "--upscaler", "native", "--warmup", "48", "--frames", "1", "--fixed-timestep", "--offscreen", "--no-ui", "--no-vsync",
           "--no-gpu-timing", "--pipeline-sync", "--resolution", "1280x720", "--capture-linear", str(out / f"{name}.pfm")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_")}
    with (out / f"{name}.log").open("w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=300).returncode


def verify(out):
    ok, lines = True, []
    discriminates = {"linear": False, "t2": False}
    for path in PATHS:
        on, off = out / f"{path}-on.pfm", out / f"{path}-off.pfm"
        if not (on.exists() and off.exists()):
            lines.append(f"{path}: missing capture"); ok = False; continue
        a, b = base.read_pfm(on), base.read_pfm(off)
        deltas = []
        for x in panels():
            cx, cy = base.project_camera(x, 3.0, base.LIGHT_Z, (a[0], a[1]))
            ma, _ = base.mean_and_dev(base.patch(a, cx, cy))
            mb, _ = base.mean_and_dev(base.patch(b, cx, cy))
            deltas.append(sum(ma[k] - mb[k] for k in range(3)) / 3)
        path_ok = abs(deltas[-1]) <= 0.002 and deltas[0] > 0.1
        worst = {name: 0.0 for name in MODELS}
        for x, angle, delta in zip(panels(), ANGLES, deltas):
            if angle == 0.0:
                continue
            r = delta / deltas[0]
            errs = {name: abs(r - model_ratio(x, angle, f)) / model_ratio(x, angle, f) for name, f in MODELS.items()}
            for name in MODELS:
                worst[name] = max(worst[name], errs[name])
            lines.append(f"{path} angle {angle}: ratio {r:.4f}  smoothstep {model_ratio(x, angle, smoothstep):.4f} (err {errs['smoothstep']*100:.2f} %)  "
                         f"linear {model_ratio(x, angle, MODELS['linear']):.4f} ({errs['linear']*100:.1f} %)  t^2 {model_ratio(x, angle, MODELS['t2']):.4f} ({errs['t2']*100:.1f} %)")
        path_ok = path_ok and worst["smoothstep"] <= TOL
        for name in ("linear", "t2"):
            if worst[name] > TOL:
                discriminates[name] = True
        lines.append(f"{path}: on-axis delta {deltas[0]:.4f}, outside {deltas[-1]:.5f}, worst smoothstep error {worst['smoothstep']*100:.2f} % -> {'PASS' if path_ok else 'FAIL'}")
        ok = ok and path_ok
    lines.append(f"negative control: linear fails {discriminates['linear']}, t^2 fails {discriminates['t2']} (both must be True)")
    ok = ok and all(discriminates.values())
    lines.append("PASS" if ok else "FAIL")
    return ok, lines


# --- round 3: the secondary lighting arms -------------------------------------------------------------------------------------
# Geometry (default glTF camera at (0, 3, 0) looking -Z; glTF lights are ignored under --corpus-rig, so no rig): a panel P at (0, 3, -2.5) whose FRONT
# faces -Z (its back faces the camera and is culled: the camera never sees it), a spot at (0, 3, -4) pointing +Z at P (distance D = 1.5), tilted up by `tilt`,
# and a wall W at z = -5 facing +Z that the spot never lights (it points away).  The camera sees W: a mirror (arms rtrefl, probe) that shows the lit front of P
# at the mirror image z = -7.5, or a diffuse wall (arm gi) that receives P's bounce.

import f15_vertex_color_secondary as vcs  # noqa: E402

LIGHT = (0.0, 3.0, -4.0)
PANEL_Z, WALL_Z = -2.5, -5.0
TILTS = [0.0, 0.38, 0.45, 0.5, 0.55]
OUTSIDE_TILT = 0.85
TOL_ARM = 0.07
ARMS = {
    "rtrefl": {"panel_half": 0.3, "wall": "mirror", "intensity": 10.0, "args": ["--rt", "on", "--gi", "off", "--reflections", "rt"], "patch": 4},
    "probe": {"panel_half": 0.3, "wall": "probe", "intensity": 10.0, "args": ["--rt", "on", "--gi", "off", "--reflections", "probes", "--reflection-capture-probe"], "patch": 4},
    "gi": {"panel_half": 0.15, "wall": "diffuse", "intensity": 200.0, "warmup": 160, "args": ["--rt", "on", "--gi", "ddgi", "--gi-grid", "16x8x16", "--gi-rays", "256",
                                                                               "--reflections", "off"], "patch": 0},
}
ARMS["girestir"] = {"panel_half": 0.15, "wall": "diffuse", "intensity": 200.0, "warmup": 160, "args": ["--rt", "on", "--gi", "restir", "--reflections", "off"], "patch": 0}
WALL_REGION = (-0.3, 0.3, -0.3, 0.3)  # gi: wall rectangle (x0, x1, dy0, dy1) around (0, 3) on W
CENTRE_HALF = 0.054                   # mirror arms: half the world footprint of the 9 x 9 pixel patch at the mirror image (7.5 m): 4.5 x 2 tan(30 deg) / 720 x 7.5 m (a first run used 0.05: 7.44 % at tilt 0.55 for rtrefl)


def tilt_quat(angle):
    """Rotation that points the light at +Z tilted up by `angle` about X: q = qY(180) * qX(angle)."""
    sx, cx = math.sin(angle / 2), math.cos(angle / 2)
    ax, ay, az, aw = 0.0, 1.0, 0.0, 0.0
    bx, by, bz, bw = sx, 0.0, 0.0, cx
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def tilt_name(t):
    return f"tilt{int(round(t * 100)):02d}"


def arm_scene(arm, tilt, intensity):
    cfg = ARMS[arm]
    s = cfg["panel_half"]
    panel = {"name": "panel", "pbrMetallicRoughness": {"baseColorFactor": [base.BASE, base.BASE, base.BASE, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    wall_pbr = {"mirror": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0, "roughnessFactor": 0.0},
                "probe": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 1.0, "roughnessFactor": 0.05},
                "diffuse": {"baseColorFactor": [0.9, 0.9, 0.9, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}}[cfg["wall"]]
    # normal = cross(U, -V): the wall faces +Z (U = +X, -V = +Y), the panel faces -Z (U = -X, -V = +Y)
    quads = [vcs.quad({"name": "wall", "pbrMetallicRoughness": wall_pbr}, (-2.0, 5.0, WALL_Z), (4.0, 0, 0), (0, -4.0, 0)),
             vcs.quad(panel, (s, LIGHT[1] + s, PANEL_Z), (-2 * s, 0, 0), (0, -2 * s, 0))]
    light = {"type": "spot", "spot": {"innerConeAngle": INNER, "outerConeAngle": OUTER}, "intensity": intensity}
    node = {"translation": list(LIGHT), "rotation": tilt_quat(tilt), "extensions": {"KHR_lights_punctual": {"light": 0}}}
    return quads, [node], {"extensionsUsed": ["KHR_lights_punctual"], "extensionsRequired": ["KHR_lights_punctual"],
                           "extensions": {"KHR_lights_punctual": {"lights": [light]}}}


def lit_value(tilt, x, y, falloff):
    """cos(panel) / d^2 x falloff of the panel point (x, y, PANEL_Z) for the light at LIGHT pointing +Z tilted up by `tilt` (axis = (0, sin, cos))."""
    w = (x - LIGHT[0], y - LIGHT[1], PANEL_Z - LIGHT[2])
    d = math.sqrt(sum(c * c for c in w))
    cos_a = (w[1] * math.sin(tilt) + w[2] * math.cos(tilt)) / d
    alpha = math.acos(max(-1.0, min(1.0, cos_a)))
    return (w[2] / d) / (d * d) * falloff(t_of(alpha))


def grid(half, n):
    return [(-half + 2 * half * (i + 0.5) / n, -half + 2 * half * (j + 0.5) / n) for i in range(n) for j in range(n)]


def model_value(arm, tilt, falloff):
    """The model's unnormalised response of the arm's measurement for one tilt."""
    cfg = ARMS[arm]
    if cfg["wall"] != "diffuse":
        pts = grid(CENTRE_HALF, 9)
        return sum(lit_value(tilt, x, LIGHT[1] + v, falloff) for x, v in pts) / len(pts)
    x0, x1, y0, y1 = WALL_REGION  # gi: the bounce onto the wall region through the form factor, mean over the region
    total = 0.0
    panel_pts = grid(cfg["panel_half"], 12)
    area = (2 * cfg["panel_half"]) ** 2 / len(panel_pts)
    for a in range(5):
        for b in range(5):
            P = (x0 + (x1 - x0) * (a + .5) / 5, LIGHT[1] + y0 + (y1 - y0) * (b + .5) / 5, WALL_Z)
            for x, v in panel_pts:
                w = (P[0] - x, P[1] - (LIGHT[1] + v), WALL_Z - PANEL_Z)
                r2 = sum(c * c for c in w)
                r = math.sqrt(r2)
                cos_factor = (PANEL_Z - WALL_Z) / r  # both normals are +-Z and the panel faces the wall: the same cosine at both ends
                total += lit_value(tilt, x, LIGHT[1] + v, falloff) * cos_factor * cos_factor / r2 * area
    return total / 25.0


def model_ratio(arm, tilt, falloff):
    return model_value(arm, tilt, falloff) / model_value(arm, 0.0, falloff)


def arm_capture(binary, folder, arm, name, gltf):
    cfg = ARMS[arm]
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--bench", "4", "--scene", str(gltf), "--render-path", "visibility", "--geometry-path", "mesh",
           "--shadows", "rt", "--lighting", "restir", *cfg["args"], "--ao", "off", "--lighting-denoise", "custom", "--post", "--tonemap", "aces",
           "--upscaler", "native", "--resolution", f"{base.RES[0]}x{base.RES[1]}", "--warmup", str(cfg.get("warmup", 48)), "--frames", "1", "--fixed-timestep", "--offscreen", "--no-ui",
           "--no-vsync", "--no-gpu-timing", "--pipeline-sync", "--capture-linear", str(folder / f"{name}.pfm")]
    if os.environ.get("F15_SPOT_HARVEST"):
        cmd += ["--harvest-pipelines", str(pathlib.Path(os.environ["F15_SPOT_HARVEST"]) / f"{arm}-{name}.raw.json")]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_")}
    env["MTL_DEBUG_LAYER"] = "1"
    env["MTL_DEBUG_LAYER_WARNING_MODE"] = "nslog"
    with (folder / f"{name}.log").open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=600).returncode
    text = (folder / f"{name}.log").read_text(errors="replace")
    return code, ("Failed with error:" in text and not os.environ.get("F15_VC_ALLOW_MISS")) or not re.search(r"^EXIT 0$", text, re.M)


def arm_names():
    return ["off"] + [tilt_name(t) for t in TILTS + [OUTSIDE_TILT]]


def arms_run(out, binary, arms):
    for arm in arms:
        folder = out / arm
        scale = float(os.environ.get("F15_SPOT_INTENSITY_SCALE", "1"))  # a linearity probe: the ratios must not depend on the absolute intensity
        jobs = [("off", 0.0, 0.0)] + [(tilt_name(t), t, ARMS[arm]["intensity"] * scale) for t in TILTS + [OUTSIDE_TILT]]
        for name, tilt, intensity in jobs:
            folder.mkdir(parents=True, exist_ok=True)
            quads, nodes, extra = arm_scene(arm, tilt, intensity)
            vcs.write_gltf(folder / f"{name}.gltf", quads, nodes, extra)
            code, bad = arm_capture(binary, folder, arm, name, folder / f"{name}.gltf")
            print(f"{arm}-{name}: exit {code}{' (log shows a failure)' if bad else ''}", flush=True)
            if code != 0 or bad:
                return 1
    return 0


def arm_delta(on, off, cx, cy, radius):
    a, b = base.patch(on, cx, cy, radius), base.patch(off, cx, cy, radius)
    return sum(sum(p[k] - q[k] for k in range(3)) / 3 for p, q in zip(a, b)) / len(a)


def arm_centre(size, on0, off):
    """Pixel centre of the measured patch of a mirror arm: the mirror image of the panel centre (camera model), refined by the delta blob at tilt 0."""
    w, h = on0[0], on0[1]
    px, py = base.project_camera(LIGHT[0], LIGHT[1], 2 * WALL_Z - PANEL_Z, size)
    window, peak = [], 0.0
    for y in range(py - 30, py + 31):
        for x in range(px - 30, px + 31):
            if 0 <= x < w and 0 <= y < h:
                v = arm_delta(on0, off, x, y, 0)
                window.append((x, y, v))
                peak = max(peak, v)
    wsum = sx = sy = 0.0
    for x, y, v in window:
        if v >= 0.5 * peak:
            wsum += v
            sx += v * x
            sy += v * y
    return (int(round(sx / wsum)), int(round(sy / wsum)), px, py, peak) if wsum and peak > 0 else None


def wall_pixels(size):
    x0, x1, y0, y1 = WALL_REGION
    ax, ay = base.project_camera(x0, LIGHT[1] + y1, WALL_Z, size)
    bx, by = base.project_camera(x1, LIGHT[1] + y0, WALL_Z, size)
    return [(x, y) for y in range(ay, by + 1, 2) for x in range(ax, bx + 1, 2)]


def arms_verify(out, arms):
    ok, lines = True, []
    for arm in arms:
        folder = out / arm
        files = {n: folder / f"{n}.pfm" for n in arm_names()}
        if not all(f.exists() for f in files.values()):
            lines.append(f"{arm}: missing capture"); ok = False; continue
        img = {n: base.read_pfm(f) for n, f in files.items()}
        w, h = img["off"][0], img["off"][1]
        deltas = {}
        if ARMS[arm]["wall"] == "diffuse":
            pixels = wall_pixels((w, h))
            for n in arm_names()[1:]:
                deltas[n] = sum(arm_delta(img[n], img["off"], px, py, 0) for px, py in pixels) / len(pixels)
            where = f"{len(pixels)} wall pixels"
        else:
            centre = arm_centre((w, h), img[tilt_name(0.0)], img["off"])
            if centre is None:
                lines.append(f"{arm}: no delta blob near the predicted image of the panel"); ok = False; continue
            cx, cy, px, py, peak = centre
            for n in arm_names()[1:]:
                deltas[n] = arm_delta(img[n], img["off"], cx, cy, ARMS[arm]["patch"])
            where = f"centre ({cx},{cy}), predicted ({px},{py}), blob peak {peak:.3f}"
        d0 = deltas[tilt_name(0.0)]
        if not d0 > 0.0:
            lines.append(f"{arm}: {where}; tilt 0 delta {d0:.5f} -> FAIL"); ok = False; continue
        worst = {name: 0.0 for name in MODELS}
        for t in TILTS[1:]:
            r = deltas[tilt_name(t)] / d0
            errs = {name: abs(r - model_ratio(arm, t, f)) / model_ratio(arm, t, f) for name, f in MODELS.items()}
            for name in MODELS:
                worst[name] = max(worst[name], errs[name])
            lines.append(f"{arm} tilt {t}: ratio {r:.4f}  smoothstep {model_ratio(arm, t, smoothstep):.4f} (err {errs['smoothstep']*100:.2f} %)  "
                         f"linear {model_ratio(arm, t, MODELS['linear']):.4f} ({errs['linear']*100:.1f} %)  t^2 {model_ratio(arm, t, MODELS['t2']):.4f} ({errs['t2']*100:.1f} %)")
        outside = deltas[tilt_name(OUTSIDE_TILT)] / d0
        arm_ok = worst["smoothstep"] <= TOL_ARM and abs(outside) <= 0.01
        discr = {n: worst[n] > TOL_ARM for n in ("linear", "t2")}
        arm_ok = arm_ok and all(discr.values())
        lines.append(f"{arm}: {where}; tilt 0 delta {d0:.5f}, outside ratio {outside:.5f} (<= 0.01), worst smoothstep error {worst['smoothstep']*100:.2f} % (<= {TOL_ARM*100:.0f} %), "
                     f"controls: linear fails {discr['linear']}, t^2 fails {discr['t2']} -> {'PASS' if arm_ok else 'FAIL'}")
        ok = ok and arm_ok
    lines.append("PASS" if ok else "FAIL")
    return ok, lines


def selftest():
    for t in (0.0, 0.5):  # the tilt quaternion points the light at +Z tilted up by the angle
        qx, qy, qz, qw = tilt_quat(t)
        v = (0.0, 0.0, -1.0)
        c1 = (qy * v[2] - qz * v[1], qz * v[0] - qx * v[2], qx * v[1] - qy * v[0])
        c2 = (qy * c1[2] - qz * c1[1], qz * c1[0] - qx * c1[2], qx * c1[1] - qy * c1[0])
        r = tuple(v[i] + 2 * qw * c1[i] + 2 * c2[i] for i in range(3))
        assert abs(r[0]) < 1e-9 and abs(r[1] - math.sin(t)) < 1e-9 and abs(r[2] - math.cos(t)) < 1e-9, (t, r)
    miss_all = []
    for arm in ARMS:
        worst = {n: 0.0 for n in ("linear", "t2")}
        for t in TILTS[1:]:
            ref = model_ratio(arm, t, smoothstep)
            for n in worst:
                worst[n] = max(worst[n], abs(model_ratio(arm, t, MODELS[n]) - ref) / ref)
        print(f"{arm}: model control miss linear {worst['linear']*100:.1f} %, t^2 {worst['t2']*100:.1f} %; "
              + " ".join(f"{t}: {model_ratio(arm, t, smoothstep):.4f}" for t in TILTS[1:]))
        miss_all.append(min(worst.values()))
        assert model_ratio(arm, OUTSIDE_TILT, smoothstep) < 0.01, arm
    assert min(miss_all) > 2 * TOL_ARM, (min(miss_all), TOL_ARM)
    print("selftest ok")



def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("arms-run", "arms-verify", "selftest"):
        parser = argparse.ArgumentParser()
        parser.add_argument("mode", choices=["arms-run", "arms-verify", "selftest"])
        parser.add_argument("out", type=pathlib.Path, nargs="?", default=pathlib.Path("build/f15-spot-arms"))
        parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
        parser.add_argument("--arms", default=",".join(ARMS))
        args = parser.parse_args()
        if args.mode == "selftest":
            selftest()
            return 0
        arms = args.arms.split(",")
        if args.mode == "arms-run":
            code = arms_run(args.out, args.binary, arms)
            if code:
                return code
        ok, lines = arms_verify(args.out, arms)
        print("\n".join(lines))
        return 0 if ok else 1
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "verify"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "run":
        (out / "on.gltf").write_text(json.dumps(scene(40.0)))
        (out / "off.gltf").write_text(json.dumps(scene(0.0)))
        for path in PATHS:
            for tag in ("on", "off"):
                code = capture(args.binary, out, path, f"{path}-{tag}", out / f"{tag}.gltf")
                print(f"{path}-{tag}: exit {code}", flush=True)
                if code != 0:
                    return 1
    ok, lines = verify(out)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
