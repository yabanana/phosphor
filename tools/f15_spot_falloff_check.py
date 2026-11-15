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


def main():
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
