#!/usr/bin/env python3
"""F15.1/F15.3 (ticket 20) chess fixture runner: back-light, side-light and shadow rigs on the pinned ABeautifulGame.

Captures the declared rigs of `--material-scenario chess-*` (src/testbench/chess_fixture.*) SERIALLY (the lock file of
tools/f15_check.py plus a check that no other `phosphor`, `soc_bench` or `f*_spike` runs), then evaluates with numpy only (no
Mitsuba: the checks are monotonicity and ratio statistics inside masks of the rig's own geometry, not a comparison with a path
tracer; the volumetric subsurface ground truth cannot represent these textured meshes, see fixtures/chess/README.md).

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_chess_check.py --run --expect green --gltf <ABeautifulGame.gltf>

Plan mode (default) prints the exact capture commands and writes plan.json.  `--evaluate` judges existing captures.  `--o7`
runs one process per rig with --warmup 64 and checks 0 GPU allocations, 0 archive misses, 0 render-thread compiles, 0 fallbacks.

Gates (each also has declared CONTROLS that must FAIL, built from the same captures: an engine that ignores the feature):
- back-light (frames averaged with --capture-linear-accumulate 112, RT glass transport): over the mask of the black pawn tops
  (a pure-emissive capture of the same camera, eroded 2 px, its centroid checked against the projection of the tops)
  * monotone: the mean luminance at transmission 0, 0.25, 0.5, 0.75, 1 rises at every step by at least 1 % of the final value
    (controls: transmission ignored, order reversed)
  * visible: the mean luminance at transmission 1 with the emissive backdrop is at least 1.5 x the one without it (the scene behind the
    glass is what rises) (control: backdrop removed)
  * tint: the chroma of the transmitted light follows the asset's tint: with e = tint / tint_g (R, B against G), the measured
    ratio asset/white-tint per channel normalised by its green value is within 15 % of e (control: tint ignored)
- side-light (view black: King_Black, Queen_Black; sun 3 W/m^2, environment 0, no shadows): the terminator bleed of each piece, the
  mean luminance of the geometrically unlit pixels (the weight 0 capture <= 1 % of its p90) over the mean of the lit ones (>= 25 % of its
  p90), rises at every step of weight 0 -> 0.5 -> 1 and of scale x0.5 -> x1 -> x2 (weight 1) by at least 0.002 (controls: subsurface
  ignored, scale ignored, order reversed).  View white (King_White, Queen_White 0.001 m): weight 0 -> 1 never falls, King rises
  by at least 0.002, the Queen's rise is reported.
- shadow (--shadows rt with --rt on, and csm with --rt off; sun 60 degrees from the vertical behind the black pawn row): the receiver
  region = floor pixels the opaque pawn top shadows (B < 0.5 H) and that are lit when the tops are hidden (H > 10 % of its board
  p99): per channel T_c = sum(A - B) / sum(H - B) with A the glass tops, B the opaque control, H the tops hidden.  Gate: 0.05 <= T_luma <= 0.95
  (attenuated, neither opaque nor clear) and the chroma follows the tint within 20 % (curved glass: qualitative) (controls: transmission
  ignored = A := B; tint ignored = A := the white-tint capture).
- support: the materials section of every report lists each chess feature as implemented or approximated (an approximation names itself).
- pipelines: every capture log has 0 archive misses, 0 render-thread compiles and 0 fallback draws; --o7 adds 0 GPU allocations.

Exit codes: 0 ok, 1 unexpected gate result, 2 usage, 3 non-discriminating control, 4 GPU busy, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import time
import traceback

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_check as fc  # noqa: E402  (capture, lock and log helpers shared with the material runner)
from f15_common import sha256_file  # noqa: E402

EXIT_OK, EXIT_UNEXPECTED, EXIT_USAGE, EXIT_NON_DISCRIMINATING, EXIT_GPU_BUSY, EXIT_INTERNAL, EXIT_CAPTURE_FAILED = range(7)
FRAMES, RESOLUTION = 128, "1920x1080"
WIDTH, HEIGHT = 1920, 1080
OVERRIDES = REPO / "fixtures/chess/chess-subsurface.overrides.json"

# Camera constants mirrored from src/testbench/chess_fixture.cpp (tests/test_chess_fixture.cpp compares them).
CAMERA_BACKLIGHT = ((0.0, 0.16, -0.4), (0.0, 0.07, 0.156))
TOP_CENTRES_X = (-0.2192, -0.15675, -0.0938, -0.0313, 0.03125, 0.0938, 0.1561, 0.21875)  # black pawn tops (asset, x of the bbox centre)
TOP_Y, TOP_Z = 0.0704, 0.1564
FOV_Y_DEGREES = 60.0

# Gate constants, declared before any measurement (a bound is never loosened to fit a result).
ERODE_PIXELS = 2
MASK_MIN_PIXELS = 3000
MASK_CENTROID_TOLERANCE_PIXELS = 10.0
MONOTONE_STEP = 0.01          # relative to the transmission-1 mean luminance
VISIBLE_RATIO = 1.5
TINT = (0.2995, 0.5, 0.45)    # the asset's baseColorFactor of Pawn_Top_Black
TINT_TOLERANCE = 0.15
BLEED_STEP = 0.002
UNLIT_FRACTION, LIT_FRACTION = 0.01, 0.25
REGION_MIN_PIXELS = 300
SHADOW_T_RANGE = (0.05, 0.95)
SHADOW_TINT_TOLERANCE = 0.20
SHADOW_SHADOWED, SHADOW_LIT = 0.5, 0.10
SHADOW_REGION_MIN_PIXELS = 500

BACKLIGHT_FLAGS = {"rt": True, "shadows": "off", "reflections": "rt", "denoise": "off", "accumulate": 112}
SIDELIGHT_FLAGS = {"rt": False, "shadows": "off", "reflections": "off", "denoise": "custom", "accumulate": 8}
MASK_FLAGS = {"rt": False, "shadows": "off", "reflections": "off", "denoise": "custom", "accumulate": 0}
SHADOW_RT_FLAGS = {"rt": True, "shadows": "rt", "reflections": "off", "denoise": "custom", "accumulate": 8}
SHADOW_CSM_FLAGS = {"rt": False, "shadows": "csm", "reflections": "off", "denoise": "custom", "accumulate": 8}

CAPTURES = {}


def _add(group, name, scenario, flags):
    CAPTURES[name] = {"group": group, "scenario": scenario, "flags": flags}


for _name, _t in (("bl-t000", 0.0), ("bl-t025", 0.25), ("bl-t050", 0.5), ("bl-t075", 0.75), ("bl-t100", 1.0)):
    _add("backlight", _name, f"chess-backlight/transmission={_t:g}", BACKLIGHT_FLAGS)
_add("backlight", "bl-white", "chess-backlight/tint=white", BACKLIGHT_FLAGS)
_add("backlight", "bl-nopanel", "chess-backlight/panel=off", BACKLIGHT_FLAGS)
_add("backlight", "bl-mask", "chess-backlight/mask=1", MASK_FLAGS)
_add("sidelight", "sl-w000", "chess-sidelight/weight=0", SIDELIGHT_FLAGS)
_add("sidelight", "sl-w050", "chess-sidelight/weight=0.5", SIDELIGHT_FLAGS)
_add("sidelight", "sl-w100", "chess-sidelight/weight=1", SIDELIGHT_FLAGS)
_add("sidelight", "sl-s050", "chess-sidelight/weight=1/scale=0.5", SIDELIGHT_FLAGS)
_add("sidelight", "sl-s200", "chess-sidelight/weight=1/scale=2", SIDELIGHT_FLAGS)
_add("sidelight", "sl-mask", "chess-sidelight/mask=1", MASK_FLAGS)
_add("sidelight", "sw-w000", "chess-sidelight/view=white/weight=0", SIDELIGHT_FLAGS)
_add("sidelight", "sw-w100", "chess-sidelight/view=white/weight=1", SIDELIGHT_FLAGS)
_add("sidelight", "sw-mask", "chess-sidelight/view=white/mask=1", MASK_FLAGS)
for _mode, _flags in (("rt", SHADOW_RT_FLAGS), ("csm", SHADOW_CSM_FLAGS)):
    _add("shadow", f"sh-{_mode}-on", "chess-shadow", _flags)
    _add("shadow", f"sh-{_mode}-opaque", "chess-shadow/tops=opaque", _flags)
    _add("shadow", f"sh-{_mode}-hidden", "chess-shadow/tops=hidden", _flags)
    _add("shadow", f"sh-{_mode}-white", "chess-shadow/tint=white", _flags)

# One process per rig for the O7 / archive check (--warmup 64, no capture).
O7_SCENARIOS = {"o7-backlight": ("chess-backlight", BACKLIGHT_FLAGS), "o7-sidelight": ("chess-sidelight", SIDELIGHT_FLAGS),
                "o7-shadow-rt": ("chess-shadow", SHADOW_RT_FLAGS), "o7-shadow-csm": ("chess-shadow", SHADOW_CSM_FLAGS)}

LUMA = (0.2126, 0.7152, 0.0722)


# --- commands -------------------------------------------------------------------------------------------------

def command_flags(binary, gltf, overrides, scenario, flags, seed=1):
    rt = flags.get("rt", False)
    return [str(binary), "--scene", str(gltf), "--material-overrides", str(overrides), "--bench", "2", "--material-scenario", scenario,
            "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on" if rt else "off", "--shadows", flags.get("shadows", "off"),
            "--lighting", "legacy", "--gi", "off", "--reflections", flags.get("reflections", "off"), "--ao", "off",
            "--lighting-denoise", flags.get("denoise", "custom"), "--reflection-samples", "8", "--lighting-seed", str(seed),
            "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1",
            "--pipeline-sync"]


def capture_command(binary, gltf, overrides, name, folder, seed=1):
    spec = CAPTURES[name]
    last = str(FRAMES - 1)
    command = command_flags(binary, gltf, overrides, spec["scenario"], spec["flags"], seed)
    command += ["--frames", str(FRAMES), "--warmup", "0", "--fixed-timestep",
                "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", last]
    if spec["flags"].get("accumulate"):
        command += ["--capture-linear-accumulate", str(spec["flags"]["accumulate"])]
    return command + ["--capture-linear-signal", "hdr", "--report", str(folder / "report.json")]


def o7_command(binary, gltf, overrides, name, folder, seed=1):
    scenario, flags = O7_SCENARIOS[name]
    return command_flags(binary, gltf, overrides, scenario, flags, seed) + ["--frames", "64", "--warmup", "64", "--fixed-timestep",
                                                                          "--report", str(folder / "report.json")]


# --- image statistics (numpy passed in: tests and the runner share it) -----------------------------------------

def luminance(np, image):
    return image @ np.asarray(LUMA, dtype=np.float32)


def erode(np, mask, pixels):
    mask = mask.copy()
    for _ in range(pixels):
        step = mask.copy()
        step[1:, :] &= mask[:-1, :]
        step[:-1, :] &= mask[1:, :]
        step[:, 1:] &= mask[:, :-1]
        step[:, :-1] &= mask[:, 1:]
        mask = step
    return mask


def mask_from_capture(np, image, threshold=0.5, erode_pixels=ERODE_PIXELS):
    """The pure-emissive mask capture: white (>= threshold luminance) where the rig's target material is, black elsewhere."""
    return erode(np, luminance(np, image) > threshold, erode_pixels)


def project(position, target, point, width=WIDTH, height=HEIGHT, fov=FOV_Y_DEGREES):
    """Pixel (column, row, depth) of a world point for the engine's camera (vertical field of view, up = +Y, row 0 at the top)."""
    f = [target[i] - position[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in f))
    f = [c / n for c in f]
    right = [f[1] * 0 - f[2] * 1, f[2] * 0 - f[0] * 0, f[0] * 1 - f[1] * 0]  # cross(f, up) with up = (0, 1, 0)
    n = math.sqrt(sum(c * c for c in right))
    right = [c / n for c in right]
    up = [right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2], right[0] * f[1] - right[1] * f[0]]
    d = [point[i] - position[i] for i in range(3)]
    depth = sum(d[i] * f[i] for i in range(3))
    tan_half = math.tan(math.radians(fov) / 2)
    ndc_x = sum(d[i] * right[i] for i in range(3)) / (depth * tan_half * width / height)
    ndc_y = sum(d[i] * up[i] for i in range(3)) / (depth * tan_half)
    return ((ndc_x * 0.5 + 0.5) * width, (0.5 - ndc_y * 0.5) * height, depth)


def check_backlight_mask(np, mask):
    """The mask has the area and sits where the black pawn tops project (a geometric check independent of the engine's materials)."""
    count = int(mask.sum())
    columns, rows = [], []
    for x in TOP_CENTRES_X:
        c, r, _ = project(CAMERA_BACKLIGHT[0], CAMERA_BACKLIGHT[1], (x, TOP_Y, TOP_Z), mask.shape[1], mask.shape[0])
        columns.append(c)
        rows.append(r)
    ys, xs = np.nonzero(mask)
    result = {"pixels": count, "min_pixels": MASK_MIN_PIXELS}
    if count == 0:
        return {**result, "pass": False}
    expected_column, expected_row = sum(columns) / len(columns), sum(rows) / len(rows)
    result.update({"centroid_column": float(xs.mean()), "centroid_row": float(ys.mean()), "expected_column": expected_column,
                   "expected_row": expected_row})
    result["pass"] = bool(count >= MASK_MIN_PIXELS and abs(xs.mean() - expected_column) <= MASK_CENTROID_TOLERANCE_PIXELS and
                          abs(ys.mean() - expected_row) <= MASK_CENTROID_TOLERANCE_PIXELS)
    return result


def mean_rgb(np, image, mask):
    return image[mask].mean(axis=0).astype(np.float64)


def evaluate_backlight(np, images, mask):
    """Gates of the back-light rig.  images: bl-t000 .. bl-t100, bl-white, bl-nopanel (linear RGB); mask: boolean."""
    keys = ("bl-t000", "bl-t025", "bl-t050", "bl-t075", "bl-t100")
    luma = [float(luminance(np, images[k])[mask].mean()) for k in keys]
    step = MONOTONE_STEP * luma[-1]
    steps = [luma[i + 1] - luma[i] for i in range(4)]
    monotone = {"luminance": luma, "steps": steps, "min_step": step, "pass": bool(all(s >= step for s in steps))}
    with_panel = luma[-1]
    without = float(luminance(np, images["bl-nopanel"])[mask].mean())
    ratio = with_panel / without if without > 0 else math.inf
    visible = {"with_panel": with_panel, "without_panel": without, "ratio": ratio, "min_ratio": VISIBLE_RATIO, "pass": bool(ratio >= VISIBLE_RATIO)}
    asset, white = mean_rgb(np, images["bl-t100"], mask), mean_rgb(np, images["bl-white"], mask)
    measured = asset / np.maximum(white, 1e-9)
    chroma = [float(measured[0] / measured[1]), 1.0, float(measured[2] / measured[1])]
    expected = [TINT[0] / TINT[1], 1.0, TINT[2] / TINT[1]]
    relative = [abs(chroma[i] / expected[i] - 1.0) for i in range(3)]
    tint = {"measured_ratio_rgb": [float(v) for v in measured], "chroma_rg_gg_bg": chroma, "expected": expected, "relative_error": relative,
            "tolerance": TINT_TOLERANCE, "pass": bool(max(relative) <= TINT_TOLERANCE)}
    gates = {"monotone": monotone, "visible": visible, "tint": tint}
    return {"gates": gates, "pass": all(g["pass"] for g in gates.values())}


def backlight_controls(np, images, mask):
    """Variants an engine without the feature would produce; each must make the evaluation fail."""
    keys = ("bl-t000", "bl-t025", "bl-t050", "bl-t075", "bl-t100")
    ignored = dict(images)
    for k in keys:
        ignored[k] = images["bl-t000"]
    reversed_ = dict(images)
    for a, b in zip(keys, keys[::-1]):
        reversed_[a] = images[b]
    no_backdrop = dict(images)
    no_backdrop["bl-t100"] = images["bl-nopanel"]
    untinted = dict(images)
    untinted["bl-t100"] = images["bl-white"]
    variants = {"transmission-ignored": ignored, "order-reversed": reversed_, "backdrop-removed": no_backdrop, "tint-ignored": untinted}
    return {name: evaluate_backlight(np, variant, mask) for name, variant in variants.items()}


def split_pieces(np, mask, king_left=True):
    """King and Queen: the pieces stand 7 mm apart on either side of the image centre column (the camera is on x = 0).  The black view
    looks along -Z (image right = +X: the King, x < 0, is on the left); the white view looks along +Z (the King is on the right)."""
    columns = np.arange(mask.shape[1])[None, :]
    centre = mask.shape[1] // 2
    left, right = mask & (columns < centre), mask & (columns >= centre)
    return {"king": left, "queen": right} if king_left else {"king": right, "queen": left}


def terminator_bleed(np, image, base, piece):
    """Mean luminance of the geometrically unlit pixels over the mean of the lit ones (regions from the weight 0 capture `base`)."""
    y0 = luminance(np, base)
    values = y0[piece]
    if values.size == 0:
        return None
    p90 = float(np.percentile(values, 90))
    lit, unlit = piece & (y0 >= LIT_FRACTION * p90), piece & (y0 <= UNLIT_FRACTION * p90)
    if int(lit.sum()) < REGION_MIN_PIXELS or int(unlit.sum()) < REGION_MIN_PIXELS:
        return None
    y = luminance(np, image)
    return {"bleed": float(y[unlit].mean() / y[lit].mean()), "lit_pixels": int(lit.sum()), "unlit_pixels": int(unlit.sum()), "p90_base": p90}


def evaluate_sidelight(np, images, masks_black, masks_white, bases=None):
    """Gates of the side-light rig.  images: sl-w000, sl-w050, sl-w100, sl-s050, sl-s200, sw-w000, sw-w100.  `bases` = the (black, white)
    weight 0 captures that define the lit and unlit regions (the controls keep the true ones: only the statistics change)."""
    bases = bases or (images["sl-w000"], images["sw-w000"])
    pieces = {}
    for piece, mask in masks_black.items():
        base = bases[0]
        stat = {k: terminator_bleed(np, images[k], base, mask) for k in ("sl-w000", "sl-w050", "sl-w100", "sl-s050", "sl-s200")}
        entry = {"bleed": {k: (v["bleed"] if v else None) for k, v in stat.items()}, "regions": {k: v and v["lit_pixels"] for k, v in stat.items()}}
        ok = all(v is not None for v in stat.values())
        entry["weight"] = {"series": [stat[k]["bleed"] for k in ("sl-w000", "sl-w050", "sl-w100")] if ok else None}
        entry["scale"] = {"series": [stat[k]["bleed"] for k in ("sl-s050", "sl-w100", "sl-s200")] if ok else None}
        for key in ("weight", "scale"):
            series = entry[key]["series"]
            entry[key]["steps"] = [series[i + 1] - series[i] for i in range(2)] if ok else None
            entry[key]["min_step"] = BLEED_STEP
            entry[key]["pass"] = bool(ok and all(s >= BLEED_STEP for s in entry[key]["steps"]))
        entry["pass"] = entry["weight"]["pass"] and entry["scale"]["pass"]
        pieces[f"{piece}-black"] = entry
    for piece, mask in masks_white.items():
        base = bases[1]
        a, b = terminator_bleed(np, images["sw-w000"], base, mask), terminator_bleed(np, images["sw-w100"], base, mask)
        entry = {"bleed": {"sw-w000": a and a["bleed"], "sw-w100": b and b["bleed"]}}
        ok = a is not None and b is not None
        rise = (b["bleed"] - a["bleed"]) if ok else None
        entry["rise"] = rise
        if piece == "king":
            entry["pass"] = bool(ok and rise >= BLEED_STEP)
            entry["gated"] = "no fall and a rise of at least 0.002 (0.003 m)"
        else:
            entry["pass"] = bool(ok and rise >= -1e-4)
            entry["gated"] = "no fall (0.001 m: reported)"
        pieces[f"{piece}-white"] = entry
    return {"pieces": pieces, "pass": all(p["pass"] for p in pieces.values())}


def sidelight_controls(np, images, masks_black, masks_white):
    ignored = dict(images)
    for k in ("sl-w050", "sl-w100", "sl-s050", "sl-s200"):
        ignored[k] = images["sl-w000"]
    scale_ignored = dict(images)
    for k in ("sl-s050", "sl-s200"):
        scale_ignored[k] = images["sl-w100"]
    reversed_ = dict(images)
    reversed_["sl-w000"], reversed_["sl-w100"] = images["sl-w100"], images["sl-w000"]
    reversed_["sl-s050"], reversed_["sl-s200"] = images["sl-s200"], images["sl-s050"]
    variants = {"subsurface-ignored": ignored, "scale-ignored": scale_ignored, "order-reversed": reversed_}
    # The base (region) capture stays the true weight 0 one in every variant so only the ordering of the statistics changes.
    bases = (images["sl-w000"], images["sw-w000"])
    return {name: evaluate_sidelight(np, variant, masks_black, masks_white, bases) for name, variant in variants.items()}


def shadow_region(np, hidden, opaque):
    yh, yb = luminance(np, hidden), luminance(np, opaque)
    board = float(np.percentile(yh[yh > 0], 99)) if bool((yh > 0).any()) else 0.0
    return erode(np, (yb < SHADOW_SHADOWED * yh) & (yh > SHADOW_LIT * board), 1)


def evaluate_shadow(np, images, region=None):
    """images: on (glass tops), opaque (B), hidden (H), white (white tint).  T_c = sum_R(A - B) / sum_R(H - B)."""
    a, b, h = images["on"], images["opaque"], images["hidden"]
    region = shadow_region(np, h, b) if region is None else region
    count = int(region.sum())
    result = {"region_pixels": count, "min_pixels": SHADOW_REGION_MIN_PIXELS}
    if count < SHADOW_REGION_MIN_PIXELS:
        return {**result, "pass": False, "reason": "region too small"}
    num = (a[region] - b[region]).sum(axis=0).astype(np.float64)
    den = (h[region] - b[region]).sum(axis=0).astype(np.float64)
    t = num / np.maximum(den, 1e-12)
    t_luma = float(sum(LUMA[i] * den[i] * t[i] for i in range(3)) / sum(LUMA[i] * den[i] for i in range(3)))
    chroma = [float(t[0] / t[1]) if t[1] > 0 else math.inf, 1.0, float(t[2] / t[1]) if t[1] > 0 else math.inf]
    expected = [TINT[0] / TINT[1], 1.0, TINT[2] / TINT[1]]
    relative = [abs(chroma[i] / expected[i] - 1.0) if math.isfinite(chroma[i]) else math.inf for i in range(3)]
    attenuated = bool(SHADOW_T_RANGE[0] <= t_luma <= SHADOW_T_RANGE[1])
    tinted = bool(max(relative) <= SHADOW_TINT_TOLERANCE)
    result.update({"transmittance_rgb": [float(v) for v in t], "transmittance_luma": t_luma, "range": SHADOW_T_RANGE, "attenuated": attenuated,
                   "chroma": chroma, "expected_chroma": expected, "chroma_relative_error": relative, "tolerance": SHADOW_TINT_TOLERANCE,
                   "tinted": tinted, "pass": attenuated and tinted})
    return result


def shadow_controls(np, images):
    """Transmission ignored (the glass tops cast the opaque shadow) and tint ignored (the white-tint capture): each must fail."""
    region = shadow_region(np, images["hidden"], images["opaque"])
    return {"transmission-ignored": evaluate_shadow(np, {**images, "on": images["opaque"]}, region),
            "tint-ignored": evaluate_shadow(np, {**images, "on": images["white"]}, region)}


# --- support report and pipeline health ---------------------------------------------------------------------------

CHESS_MATERIALS = ("Pawn_Top_Black", "Pawn_Top_White", "King_Black", "King_White", "Queen_Black", "Queen_White")


def evaluate_support(report, required_features):
    """Every chess feature of the six materials is implemented or an approximation that names itself.  `required_features`:
    {material: [feature, ...]}.  Returns (pass, rows)."""
    materials = {m["name"]: m for m in report["materials"]["materials"]}
    rows, ok = [], True
    for name, features in required_features.items():
        entry = materials.get(name)
        if entry is None:
            rows.append({"material": name, "feature": None, "support": "missing"})
            ok = False
            continue
        by_feature = {f["feature"]: f for f in entry["features"]}
        for feature in features:
            f = by_feature.get(feature)
            level = f["support"] if f else "absent"
            named = bool(f and (f["support"] == "implemented" or (f["support"] == "approximated" and f["approximation"])))
            rows.append({"material": name, "feature": feature, "support": level, "approximation": f["approximation"] if f else None,
                         "label": f["label"] if f else None, "pass": named})
            ok = ok and named
    return ok, rows


PIPELINES = re.compile(r"PIPELINES requests (\d+) \| archive hits (\d+), misses (\d+).*?render-thread compiles (\d+).*?fallbacks (\d+) \(draws (\d+)\)")


def pipeline_health(log_text):
    """Archive misses, render-thread compiles and fallback draws of the LAST PIPELINES summary of a log (None when absent)."""
    found = PIPELINES.findall(log_text)
    if not found:
        return None
    requests, hits, misses, compiles, fallbacks, draws = (int(v) for v in found[-1])
    return {"requests": requests, "archive_misses": misses, "render_thread_compiles": compiles, "fallback_draws": draws,
            "pass": misses == 0 and compiles == 0 and draws == 0}


# --- scenario evaluation -----------------------------------------------------------------------------------------

def read_images(folder, names):
    from f12_reference import read_pfm
    return {n: read_pfm(folder / n / "engine.pfm") for n in names}


def evaluate_group_backlight(np, out):
    names = [n for n, s in CAPTURES.items() if s["group"] == "backlight"]
    images = read_images(out, names)
    mask = mask_from_capture(np, images["bl-mask"])
    geometry = check_backlight_mask(np, mask)
    result = evaluate_backlight(np, images, mask)
    controls = backlight_controls(np, images, mask) if geometry["pass"] else {}
    return {"group": "backlight", "mask": geometry, "result": result, "controls": {k: v["pass"] for k, v in controls.items()},
            "controls_fail": all(not v["pass"] for v in controls.values()) and bool(controls),
            "pass": bool(geometry["pass"] and result["pass"])}


def evaluate_group_sidelight(np, out):
    names = [n for n, s in CAPTURES.items() if s["group"] == "sidelight"]
    images = read_images(out, names)
    masks_black = split_pieces(np, mask_from_capture(np, images["sl-mask"], erode_pixels=1))
    masks_white = split_pieces(np, mask_from_capture(np, images["sw-mask"], erode_pixels=1), king_left=False)
    result = evaluate_sidelight(np, images, masks_black, masks_white)
    controls = sidelight_controls(np, images, masks_black, masks_white)
    # the controls only concern the black view (the white view has two images)
    black_pass = {k: all(p["pass"] for name, p in v["pieces"].items() if name.endswith("-black")) for k, v in controls.items()}
    return {"group": "sidelight", "result": result, "controls": black_pass, "controls_fail": all(not v for v in black_pass.values()),
            "mask_pixels": {"black": {k: int(m.sum()) for k, m in masks_black.items()}, "white": {k: int(m.sum()) for k, m in masks_white.items()}},
            "pass": bool(result["pass"])}


def evaluate_group_shadow(np, out):
    entries = {}
    for mode in ("rt", "csm"):
        images = read_images(out, [f"sh-{mode}-{v}" for v in ("on", "opaque", "hidden", "white")])
        images = {v: images[f"sh-{mode}-{v}"] for v in ("on", "opaque", "hidden", "white")}
        result = evaluate_shadow(np, images)
        controls = shadow_controls(np, images)
        entries[mode] = {"result": result, "controls": {k: v["pass"] for k, v in controls.items()},
                         "controls_fail": all(not v["pass"] for v in controls.values())}
    return {"group": "shadow", "modes": entries, "pass": all(e["result"]["pass"] for e in entries.values()),
            "controls_fail": all(e["controls_fail"] for e in entries.values())}


REQUIRED_FEATURES = {"Pawn_Top_Black": ["transmission", "volume"], "Pawn_Top_White": ["transmission", "volume"],
                     "King_Black": ["subsurface", "volume"], "King_White": ["subsurface", "volume"],
                     "Queen_Black": ["subsurface", "volume"], "Queen_White": ["subsurface", "volume"]}


def evaluate_support_group(out):
    rows = {}
    ok = True
    for name, features in (("bl-t100", ["Pawn_Top_Black", "Pawn_Top_White"]), ("sl-w100", ["King_Black", "King_White", "Queen_Black", "Queen_White"])):
        report_path = out / name / "report.json"
        report = json.loads(report_path.read_text())
        passed, found = evaluate_support(report, {m: REQUIRED_FEATURES[m] for m in features})
        rows[name] = found
        ok = ok and passed
    return {"group": "support", "rows": rows, "pass": ok}


def evaluate_pipelines(out, names):
    health = {}
    for n in names:
        log = out / n / "engine.log"
        health[n] = pipeline_health(log.read_text(errors="replace")) if log.is_file() else None
    return {"health": health, "pass": all(h is not None and h["pass"] for h in health.values())}


def verdict(entries, expect):
    """(status, exit code): every group green with every control failing = GREEN."""
    discriminating = all(e.get("controls_fail", True) for e in entries)
    gates = all(e["pass"] for e in entries)
    if not discriminating:
        return "NON_DISCRIMINATING", EXIT_NON_DISCRIMINATING
    if expect == "green":
        return ("GREEN", EXIT_OK) if gates else ("RED", EXIT_UNEXPECTED)
    return ("RED", EXIT_OK) if not gates else ("GREEN", EXIT_UNEXPECTED)


# --- main -------------------------------------------------------------------------------------------------------

def find_gltf(arg):
    candidates = [arg] if arg else [os.environ.get("PHOSPHOR_CHESS_GLTF"), str(REPO / "build/khronos-corpus/ABeautifulGame/glTF/ABeautifulGame.gltf")]
    for c in candidates:
        if c and pathlib.Path(c).is_file():
            return pathlib.Path(c).resolve()
    return None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-chess")
    parser.add_argument("--gltf", default=None, help="ABeautifulGame.gltf (default: $PHOSPHOR_CHESS_GLTF or build/khronos-corpus)")
    parser.add_argument("--overrides", type=pathlib.Path, default=OVERRIDES)
    parser.add_argument("--group", action="append", default=[], choices=["backlight", "sidelight", "shadow"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--o7", action="store_true", help="one process per rig with --warmup 64: 0 GPU allocations, 0 archive misses, 0 compiles")
    parser.add_argument("--expect", choices=["red", "green"], default="green")
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def run_serial(parser, args, items, make_command, out, binary):
    if not binary.is_file():
        parser.error("--run requires an existing compiled renderer; this runner never builds")
    for n in items:
        folder = out / n
        if folder.exists() and any(folder.iterdir()):
            parser.error(f"{folder} is not empty: choose a new --out")
    with fc.gpu_lock(args.gpu_lock):
        for position, n in enumerate(items):
            busy = fc.running_gpu_processes()
            if busy:
                raise fc.GpuBusy(f"refusing to start {n}: other GPU jobs are running: {', '.join(busy)}")
            (out / n).mkdir(parents=True, exist_ok=True)
            status = fc.capture(make_command(n, out / n), out / n / "engine.log", args.timeout)
            if not status["passed"]:
                raise fc.CaptureFailed(f"{n}: capture failed: {status['failures']}")
            if position + 1 < len(items):
                time.sleep(2)


def evaluate_o7(out):
    rows = {}
    for n in O7_SCENARIOS:
        report = json.loads((out / n / "report.json").read_text())
        health = pipeline_health((out / n / "engine.log").read_text(errors="replace"))
        rows[n] = {"gpu_allocations": report.get("gpu_allocations"), "pipelines": health,
                   "pass": report.get("gpu_allocations") == 0 and health is not None and health["pass"]}
    return rows


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    groups = args.group or ["backlight", "sidelight", "shadow"]
    out = args.out.resolve()
    binary = args.binary.resolve()
    gltf = find_gltf(args.gltf)
    if gltf is None:
        parser.error("ABeautifulGame.gltf not found: pass --gltf or set PHOSPHOR_CHESS_GLTF (tools/fetch_khronos_corpus.py downloads it)")
    names = [n for n, s in CAPTURES.items() if s["group"] in groups]
    overrides = args.overrides.resolve()
    if args.o7:
        if args.run:
            parser.error("--o7 and --run are exclusive")
        run_serial(parser, args, list(O7_SCENARIOS), lambda n, f: o7_command(binary, gltf, overrides, n, f, args.seed), out, binary)
        rows = evaluate_o7(out)
        (out / "o7.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows, indent=2))
        ok = all(r["pass"] for r in rows.values())
        print(f"{'GREEN' if ok else 'RED'} (O7)")
        return EXIT_OK if ok else EXIT_UNEXPECTED
    if not args.run and not args.evaluate:
        out.mkdir(parents=True, exist_ok=True)
        plan = {"state": "NOT_EXECUTED", "gltf": str(gltf), "commands": {n: capture_command(binary, gltf, overrides, n, out / n, args.seed) for n in names}}
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        for n in names:
            print(f"# {n}: {CAPTURES[n]['scenario']}")
            print(" ".join(plan["commands"][n]))
        print(f"NOT_EXECUTED: planned {len(names)} chess captures, plan at {out / 'plan.json'}")
        return EXIT_OK
    if args.run:
        run_serial(parser, args, names, lambda n, f: capture_command(binary, gltf, overrides, n, f, args.seed), out, binary)
    import numpy as np
    entries = []
    if "backlight" in groups:
        entries.append(evaluate_group_backlight(np, out))
    if "sidelight" in groups:
        entries.append(evaluate_group_sidelight(np, out))
    if "shadow" in groups:
        entries.append(evaluate_group_shadow(np, out))
    if set(groups) == {"backlight", "sidelight", "shadow"}:
        entries.append({**evaluate_support_group(out), "controls_fail": True})
    pipelines = evaluate_pipelines(out, names)
    entries.append({"group": "pipelines", **pipelines, "controls_fail": True})
    status, code = verdict(entries, args.expect)
    document = {"status": status, "mode": args.expect, "gltf": str(gltf), "overrides_sha256": sha256_file(overrides), "entries": entries}
    (out / "results.json").write_text(json.dumps(document, indent=2, default=str) + "\n")
    for e in entries:
        print(f"{e['group']}: pass={e['pass']}" + (f" controls_fail={e['controls_fail']}" if "controls_fail" in e else ""))
    print(f"{status} (exit {code}); details in {out / 'results.json'}")
    return code


def main(argv=None):
    try:
        return run(argv)
    except SystemExit:
        raise
    except fc.GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return EXIT_GPU_BUSY
    except fc.CaptureFailed as error:
        print(str(error), file=sys.stderr)
        return EXIT_CAPTURE_FAILED
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
