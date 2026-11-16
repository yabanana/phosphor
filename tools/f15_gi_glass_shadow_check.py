#!/usr/bin/env python3
"""F15.1 (ticket 14, round 3): sun shadow rays cast from SECONDARY hits carry the transmittance of glass (ADR-0004 amendment).

The primary RT sun / DI shadows through glass are gated by `transmissive-shadow-*@rt|@restir` (tools/f15_check.py).  This tool gates the
sites that trace a sun / light shadow ray from a secondary vertex: the GI modes (ddgi probe rays, the radiance cache and ReSTIR GI candidates
all shade their hit with giSecondaryRadiance -> giShadowTransmittance) and the lighting of what a REFLECTION ray hits (reflectionSecondary ->
diEndpointTransmittance).  AO, froxel fog, ReSTIR reconnection and sky rays see glass as opaque (declared, plan "Rischi"): not tested here.

Method (every number fixed here BEFORE any capture, decision 11: transmission bound mean <= 5 %, p95 <= 15 %).  The environment is 0 and ONE
sun lights the scene; a huge thin-walled GREEN pane (tint (0.4, 0.9, 0.45), ior 1.5, transmission 1) hangs 60 m above the scene, 120 m wide,
so EVERY sun ray that can reach anything below it crosses the pane.  Light transport is linear in the source: every radiance below the pane
(direct, one bounce, any number of bounces) is the radiance of the SAME scene without the pane times the pane's transmittance T, per channel,
T_c = (1 - F_Schlick(cos 30 degrees, f0 = 0.04)) x tint_c  (the thin pane has ONE Fresnel surface, ticket 14).  The pane lies outside the
probe grid (the fixture pins the grid to y < 8 m), where the atlas lookup is 0, so it adds no bounce of its own (ray length 100 m: the sun
shadow ray of a secondary hit reaches it).  Scenes (bench class TransmissiveShadowValidation::indirectScenarios, `-open` = no pane):
  gi        a 1 m opaque cube on the floor; the camera sees its +Z face, which the sun (from -Z, 30 degrees from the vertical) never reaches:
            the face is lit by the floor's bounce ONLY.  Region = the inner 60 % of the face.  Modes: ddgi, cache, restir (GI = the site).
  refl      a polished metal wall (roughness 0.02, base 1, metallic 1) at z = -1.5 facing +Z, the sun behind it: the camera sees the sunlit
            floor in the mirror, and a mirror pixel = F(1) x the lighting of the floor point the reflection ray hit.  `--reflections rt`.
            Region = mirror pixels whose reflected floor point is farther than 0.3 m from the wall's shadow and inside +-5 m.
Gate per region and channel: R_c = mean(pane) / mean(open) must be within 5 % of T_c, and the p95 of the per-pixel |R_c(x) / T_c - 1| (ratio of
the two captures' pixels) <= 15 %.  Controls (all must FAIL the same gate): the engine run with `--debug-transmittance-inject
--debug-transmittance-dedup off` (every surface factor applied twice: the ratio is T^2), and two wrong expectations on the same data
(transmittance ignored: T = 1; tint ignored: T = (1 - F) for every channel).

  GPU captures (wrap in `python3 gpulock.py`):  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_gi_glass_shadow_check.py \
      --run --binary build/release/phosphor --out build/f15-gi-glass [--only gi-ddgi,refl] [--chunk]
  CPU only:                                      ... --evaluate --out build/f15-gi-glass
Exit codes: 0 gates met and controls fail, 1 a gate failed or a control passed, 2 usage, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

WIDTH, HEIGHT = 1280, 720
FRAMES = 128
ACCUMULATE = 64              # frames averaged (the last 64 of 128)
MEAN_BOUND, P95_BOUND = 0.05, 0.15
TINT = (0.4, 0.9, 0.45)
F0 = 0.04
SUN_DEGREES = 30.0
SUN_DIRECTION = (0.0, math.cos(math.radians(SUN_DEGREES)), -math.sin(math.radians(SUN_DEGREES)))  # towardSun (unit)
PANE_FRESNEL = F0 + (1.0 - F0) * (1.0 - math.cos(math.radians(SUN_DEGREES))) ** 5
TRANSMITTANCE = tuple((1.0 - PANE_FRESNEL) * c for c in TINT)

# camera of the bench (spec): target, distance, elevation (degrees above the floor, looking from +Z), 60 degree vertical field of view
CAMERAS = {"gi": ((0.0, 0.4, 0.3), 4.0, 30.0), "refl": ((0.0, 0.5, 1.0), 5.0, 25.0)}
CUBE_FACE = {"z": 0.5, "x": (-0.3, 0.3), "y": (0.2, 0.8)}        # inner 60 % of the +Z face of the 1 m cube
MIRROR = {"z": -1.5 + 0.025, "box": ((-3.0, 0.0, -1.525), (3.0, 2.0, -1.475)), "x": (-2.6, 2.6), "y": (0.1, 1.9)}
FLOOR_LIMIT = 5.0
SHADOW_MARGIN = 0.3

GI_ARGS = ["--gi-spacing", "0.5", "--gi-grid", "32x16x32", "--gi-probe-anchor", "0,4.5,0"]
# key -> (scenario family, extra engine arguments).  `gi-*`: the GI mode is the site; `refl`: --reflections rt, GI off.
KEYS = {
    "gi-ddgi": ("gi", ["--gi", "ddgi", *GI_ARGS]),
    "gi-cache": ("gi", ["--gi", "cache", *GI_ARGS]),
    "gi-restir": ("gi", ["--gi", "restir", *GI_ARGS]),
    "refl": ("refl", []),
}
CONTROL_ARGS = ["--debug-transmittance-inject", "--debug-transmittance-dedup", "off"]
EXPECTATIONS = {  # name -> per-channel expected ratio (wrong expectations on the same data)
    "transmittance-ignored": (1.0, 1.0, 1.0),
    "tint-ignored": (1.0 - PANE_FRESNEL,) * 3,
}


def scenario(family, pane):
    return f"transmissive-shadow-{family}-{'pane' if pane else 'open'}"


def capture_command(binary, key, pane, folder, control=False):
    family, extra = KEYS[key]
    lighting_gi = ["--lighting", "restir", "--lighting-denoise", "off"] if family == "gi" else ["--lighting", "legacy", "--lighting-denoise", "custom"]
    reflections = ["--reflections", "rt"] if family == "refl" else ["--reflections", "off"]
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario(family, pane),
            "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", "--shadows", "rt",
            *lighting_gi, *reflections, "--ao", "off", "--gi", "off", *extra, "--reflection-samples", "8", "--lighting-seed", "1",
            "--post", "--upscaler", "native", "--resolution", f"{WIDTH}x{HEIGHT}", "--frames", str(FRAMES), "--warmup", "0",
            "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--pipeline-sync", "--specular-ms", "on",
            "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", str(FRAMES - 1),
            "--capture-linear-accumulate", str(ACCUMULATE), "--capture-linear-signal", "hdr",
            *(CONTROL_ARGS if control else []), "--report", str(folder / "report.json")]


def run_capture(command, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "engine.log").open("wb") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
    text = (folder / "engine.log").read_text(errors="replace")
    problems = []
    if code != 0 or "EXIT 0" not in text:
        problems.append(f"exit {code}")
    if re.search(r"Failed with error:|failed assertion|Validation Error", text):
        problems.append("archive or validation error line")
    misses = re.findall(r"archive hits (\d+), misses (\d+)", text)
    if misses and int(misses[-1][1]) != 0:
        problems.append(f"{misses[-1][1]} archive misses")
    fallback = re.search(r"fallbacks (\d+) \(draws (\d+)\)", text)
    if fallback and int(fallback.group(2)) != 0:
        problems.append(f"{fallback.group(2)} fallback draws")
    if not (folder / "engine.pfm").is_file():
        problems.append("no capture")
    return problems


def read_pfm(np, path):
    data = pathlib.Path(path).read_bytes()
    parts = data.split(b"\n", 3)
    assert parts[0] == b"PF", "colour PFM expected"
    width, height = map(int, parts[1].split())
    scale = float(parts[2])
    pixels = np.frombuffer(parts[3], dtype="<f4" if scale < 0 else ">f4", count=width * height * 3)
    return pixels.reshape(height, width, 3)[::-1].astype(np.float64)  # PFM rows run bottom to top


# --- geometry ---------------------------------------------------------------------------------------------------------------------------

def camera_rays(np, family):
    target, distance, elevation = CAMERAS[family]
    e = math.radians(elevation)
    position = np.array(target) + distance * np.array([0.0, math.sin(e), math.cos(e)])
    forward = np.array(target) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_half = math.tan(math.radians(30.0))
    xs = ((np.arange(WIDTH) + 0.5) / WIDTH * 2.0 - 1.0) * tan_half * WIDTH / HEIGHT
    ys = (1.0 - (np.arange(HEIGHT) + 0.5) / HEIGHT * 2.0) * tan_half
    directions = forward + xs[None, :, None] * right + ys[:, None, None] * up
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    return position, directions


def ray_hits_box(np, origins, directions, lo, hi):
    """Slab test, arrays (..., 3): True where the ray (t > 0) meets the box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = 1.0 / directions
        t0, t1 = (np.array(lo) - origins) * inverse, (np.array(hi) - origins) * inverse
        near = np.nanmax(np.minimum(t0, t1), axis=-1)
        far = np.nanmin(np.maximum(t0, t1), axis=-1)
    return (far >= np.maximum(near, 1e-6))


def region_mask(np, family):
    position, directions = camera_rays(np, family)
    if family == "gi":
        t = (CUBE_FACE["z"] - position[2]) / directions[..., 2]
        point = position + t[..., None] * directions
        return (t > 0) & (point[..., 0] >= CUBE_FACE["x"][0]) & (point[..., 0] <= CUBE_FACE["x"][1]) & \
               (point[..., 1] >= CUBE_FACE["y"][0]) & (point[..., 1] <= CUBE_FACE["y"][1])
    t = (MIRROR["z"] - position[2]) / directions[..., 2]
    point = position + t[..., None] * directions
    inside = (t > 0) & (point[..., 0] >= MIRROR["x"][0]) & (point[..., 0] <= MIRROR["x"][1]) & \
             (point[..., 1] >= MIRROR["y"][0]) & (point[..., 1] <= MIRROR["y"][1])
    reflected = directions.copy()
    reflected[..., 2] = -reflected[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -point[..., 1] / reflected[..., 1]
    floor = point + s[..., None] * reflected
    inside &= (reflected[..., 1] < 0) & (np.abs(floor[..., 0]) <= FLOOR_LIMIT) & (np.abs(floor[..., 2]) <= FLOOR_LIMIT) & (floor[..., 2] > MIRROR["z"] + 0.05)
    sun = np.array(SUN_DIRECTION)
    lit = np.ones(inside.shape, dtype=bool)
    for dx, dz in ((0.0, 0.0), (SHADOW_MARGIN, 0.0), (-SHADOW_MARGIN, 0.0), (0.0, SHADOW_MARGIN), (0.0, -SHADOW_MARGIN)):
        probe = floor + np.array([dx, 0.0, dz])
        lit &= ~ray_hits_box(np, probe + 1e-4 * sun, np.broadcast_to(sun, probe.shape), *MIRROR["box"])
    return inside & lit


# --- statistics -------------------------------------------------------------------------------------------------------------------------

def luminance_floor_mask(np, open_image, mask):
    """Pixels whose pane-free value is positive in every channel (a ratio is defined)."""
    return mask & np.all(open_image > 1e-6, axis=2)


def ratio_statistics(np, pane, open_, mask, expected):
    mask = luminance_floor_mask(np, open_, mask)
    count = int(mask.sum())
    if count < 200:
        raise RuntimeError(f"region too small ({count} pixels)")
    mean_pane, mean_open = pane[mask].mean(axis=0), open_[mask].mean(axis=0)
    ratio = mean_pane / mean_open
    mean_error = np.abs(ratio / np.array(expected) - 1.0)
    pixel_error = np.abs(pane[mask] / open_[mask] / np.array(expected) - 1.0)
    p95 = np.percentile(pixel_error, 95, axis=0)
    return {"pixels": count, "mean_open": mean_open.tolist(), "ratio": ratio.tolist(), "expected": list(expected),
            "mean_error": mean_error.tolist(), "p95_error": p95.tolist(),
            "passes": bool(mean_error.max() <= MEAN_BOUND and p95.max() <= P95_BOUND)}


def evaluate_key(np, key, folder):
    family = KEYS[key][0]
    mask = region_mask(np, family)
    open_ = read_pfm(np, folder / "open" / "engine.pfm")
    pane = read_pfm(np, folder / "pane" / "engine.pfm")
    result = {"key": key, "region_pixels": int(mask.sum()), "transmittance": list(TRANSMITTANCE)}
    result["gate"] = ratio_statistics(np, pane, open_, mask, TRANSMITTANCE)
    controls = {}
    for name, expected in EXPECTATIONS.items():
        controls[name] = ratio_statistics(np, pane, open_, mask, expected)
    control_capture = folder / "control" / "engine.pfm"
    if control_capture.is_file():
        controls["transmittance-doubled"] = ratio_statistics(np, read_pfm(np, control_capture), open_, mask, TRANSMITTANCE)
    result["controls"] = controls
    result["controls_fail"] = all(not c["passes"] for c in controls.values())
    result["has_engine_control"] = control_capture.is_file()
    return result


def evaluate(out, keys):
    import numpy as np
    results = [evaluate_key(np, key, out / key) for key in keys]
    (out / "result.json").write_text(json.dumps(results, indent=2))
    ok = True
    for r in results:
        gate = r["gate"]
        print(f"{r['key']}: region {gate['pixels']} px, ratio {['%.4f' % v for v in gate['ratio']]} expected {['%.4f' % v for v in gate['expected']]}, "
              f"mean err {['%.4f' % v for v in gate['mean_error']]}, p95 err {['%.4f' % v for v in gate['p95_error']]} -> {'PASS' if gate['passes'] else 'FAIL'}")
        for name, c in r["controls"].items():
            print(f"   control {name}: mean err max {max(c['mean_error']):.4f}, p95 max {max(c['p95_error']):.4f} -> {'passes (BAD)' if c['passes'] else 'fails as required'}")
        if not r["has_engine_control"]:
            print("   (no engine control capture: run with --control)")
        ok &= gate["passes"] and r["controls_fail"]
    return 0 if ok else 1


def run(args, keys):
    binary = args.binary.resolve()
    for key in keys:
        for name, pane, control in (("open", False, False), ("pane", True, False)) + ((("control", True, True),) if args.control else ()):
            folder = args.out / key / name
            problems = run_capture(capture_command(binary, key, pane, folder, control), folder)
            if problems:
                print(f"{key}/{name}: capture failed: {problems} ({folder / 'engine.log'})")
                return 6
            print(f"{key}/{name}: captured")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="capture (GPU; wrap in gpulock.py) then evaluate")
    mode.add_argument("--evaluate", action="store_true", help="evaluate existing captures (CPU)")
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-gi-glass")
    parser.add_argument("--only", help="comma separated keys: " + ", ".join(KEYS))
    parser.add_argument("--control", action="store_true", help="also capture the engine control (--debug-transmittance-inject, dedup off)")
    parser.add_argument("--no-evaluate", action="store_true", help="--run: captures only")
    args = parser.parse_args(argv)
    keys = args.only.split(",") if args.only else list(KEYS)
    if any(k not in KEYS for k in keys):
        print("unknown key")
        return 2
    if args.run:
        code = run(args, keys)
        if code or args.no_evaluate:
            return code
    return evaluate(args.out, keys)


if __name__ == "__main__":
    sys.exit(main())
