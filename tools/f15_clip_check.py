#!/usr/bin/env python3
"""F15.1 ticket 08: specular-extension clip, recovery after an extension edit, lobe-contract controls and the 2x2
adaptive-shading check.  A separate runner (the F13/F14 corpus runner freezes a STOP_AFTER_F14 plan of its own cases);
it reuses its metric machinery unchanged: tools/temporal_light_metrics.py and the thresholds of f13_f14_check.THRESHOLDS
(linear RMSE p95 <= 0.06, max <= 0.15, residual flicker mean <= 0.02, error-change ghost <= 0.25, recovery <= 4 frames
within RMSE 0.06), judged against the higher-sample control exactly like the F13 cases (HIGHER_SAMPLE_CONTROL_NOT_
INDEPENDENT_OR_CONVERGED).

Cases (one renderer process each, serial, API + shader validation like the F13 runner):
  clip-raw, clip-custom, clip-rt8   the ior x specular grid (bench 2 --material-scenario specular-grid-edit, constant
                                    environment 0.5, RT reflections): raw 1 spp, custom denoiser 1 spp, 8 spp control
  edit-custom, edit-rt8             the same with --material-toggle-every 60: the two probe spheres change their
                                    extension at frames 59, 119, 179 (ior 1.5/2.4 and specularColor 1/25); the recovery
                                    gate and the clip gates run on the feature and parameter spheres
  negative-lobe-flip                custom denoiser whose sampled-lobe key changes every frame (PHOSPHOR_DIAGNOSTIC_
                                    DENOISE_LOBE=flip: history rejected across lobe keys): must FAIL the clip gates
  lobe-stripes                      custom denoiser with the lobe key forced to 1 on 16 px stripes (…=stripes: neighbours
                                    rejected across lobe keys): the last frame must differ from clip-custom
  adapt-off, adapt-off-repeat, adapt-on     specular-grid-rough at 1920x1080 under the plain resolve, without and with
                                    --adaptive-shading: pixels of spheres with feature bits must be identical
  adapt-control-off, adapt-control-on       base-grid-sun (no features) the same way: reuse must change pixels (the
                                    comparison can fail)
Stale control: the edit clip judged with the control delayed by 5 frames must FAIL the recovery gate (a history that
kept the old value for 5 frames).

Usage (numpy is needed only by the adaptive comparison):
  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_clip_check.py --run --out build/f15-clip
  ... --evaluate --out build/f15-clip      (existing captures, no GPU process)
  ... --plan --out build/f15-clip          (commands only)
Exit codes: 0 every expectation met, 1 an expectation failed, 2 usage, 3 GPU busy, 4 capture failed.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from f13_f14_check import THRESHOLDS, enforce_gates  # noqa: E402
from run_checked import run_checked  # noqa: E402
from temporal_light_metrics import analyse_clips, analyse_paths, clip_files, read_pfm  # noqa: E402

REPO = HERE.parent
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_BUSY, EXIT_CAPTURE = 0, 1, 2, 3, 4

# --- declared geometry of the grid fixtures (src/testbench/material_validation.cpp) -------------------------------------
SPACING, RADIUS, CAMERA_DISTANCE, CENTRE_Y, FOV_DEGREES = 1.0, 0.4, 5.5, 2.0, 60.0
COLUMNS, ROWS = 5, 5
# toggle spheres: row-major index row * COLUMNS + column (MaterialValidation::toggleFeatureSphere / ...ParameterSphere)
FEATURE_SPHERE, PARAMETER_SPHERE = (3, 0), (2, 1)  # (column, row)
ROI_SCALE = 0.5  # predeclared radiance scale: the declared environment radiance, the bound of a white reflector F <= 1
CLIP_FRAMES, EDIT_FRAMES, TOGGLE_EVERY = 160, 240, 60
EDIT_EVENTS = [TOGGLE_EVERY * k - 1 for k in (1, 2, 3)]  # the edit of event k is visible at frame k * N - 1 (frame 239 is the last one)
STALE_FRAMES = 5  # one more than recovery_frames_max
RESOLUTION, ADAPT_RESOLUTION = "640x360", "1920x1080"
ADAPT_FRAMES = 24
# KHR grid of specular-grid-rough (identical to specular-grid-env): neutral = ior 1.5, weight 1, colour 1 (no feature bit)
GRID_IOR = [0.0, 1.0, 1.33, 1.5, 2.4]
GRID_FACTOR = [1.0, 0.5, 1.0, 0.5, 0.5]
GRID_COLOR = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 1, 0.5), (2, 2, 2)]
GPU_PROCESS = re.compile(r"^(phosphor|soc_bench|f\d+_spike)$")
GPU_ERROR = re.compile(r"GPU timeout|command buffers failed|Shader Validation Error|Metal Validation Error|failed assertion", re.I)
CHECK_LINE = re.compile(r"LIGHTING check frame (\d+)\s*\|\s*(PASS|FAIL)")
LOBE_LINE = "F15 specular denoiser: lobe-contract kernels"


class GpuBusy(RuntimeError):
    pass


class CaptureFailed(RuntimeError):
    pass


# --- geometry ----------------------------------------------------------------------------------------------------------
def sphere_screen(column, row, width, height):
    """Normalized (cx, cy, rx, ry) of one sphere's centre and radius for the frozen camera (centre plane projection)."""
    aspect = width / height
    tan_half = math.tan(math.radians(FOV_DEGREES) / 2)
    x = (column - 0.5 * (COLUMNS - 1)) * SPACING
    y = row * SPACING - CENTRE_Y
    ndc_x = x / (CAMERA_DISTANCE * tan_half * aspect)
    ndc_y = y / (CAMERA_DISTANCE * tan_half)
    return (0.5 + 0.5 * ndc_x, 0.5 - 0.5 * ndc_y,
            0.5 * RADIUS / (CAMERA_DISTANCE * tan_half * aspect), 0.5 * RADIUS / (CAMERA_DISTANCE * tan_half))


def inner_roi(column, row, width, height, fraction=0.55):
    cx, cy, rx, ry = sphere_screen(column, row, width, height)
    return [round(cx - fraction * rx, 4), round(cy - fraction * ry, 4), round(cx + fraction * rx, 4), round(cy + fraction * ry, 4)]


def grid_roi(width, height):
    first = sphere_screen(0, ROWS - 1, width, height)
    last = sphere_screen(COLUMNS - 1, 0, width, height)
    return [round(first[0] - first[2], 4), round(first[1] - first[3], 4), round(last[0] + last[2], 4), round(last[1] + last[3], 4)]


def disc_mask(np, column, row, width, height, fraction):
    cx, cy, rx, ry = sphere_screen(column, row, width, height)
    yy, xx = np.mgrid[0:height, 0:width]
    return (((xx + 0.5 - cx * width) / (fraction * rx * width)) ** 2 + ((yy + 0.5 - cy * height) / (fraction * ry * height)) ** 2) <= 1.0


def is_neutral(column, row):
    return GRID_IOR[column] == 1.5 and GRID_FACTOR[row] == 1.0 and tuple(GRID_COLOR[row]) == (1, 1, 1)


# --- ROI configuration (frozen before any run) --------------------------------------------------------------------------
def roi_config(frames, events, width=640, height=360):
    """Gating ROIs: the whole grid and the two probe spheres.  The two Fresnel = 1 (ior 0) spheres are diagnostic only (see
    DIAGNOSTIC_ROIS): their error against the 8 spp control exceeds the F13 gates, like the base metals of the same
    roughness do at a smaller scale (tools/f15_clip_check.py docstring and the ticket 08 handoff)."""
    rois = [("grid", grid_roi(width, height)),
            ("feature-sphere", inner_roi(*FEATURE_SPHERE, width, height)),
            ("parameter-sphere", inner_roi(*PARAMETER_SPHERE, width, height))]
    config = {"schema": 1, "linear": True, "frozen_before_run": True, "reconstruction_support_radius": 1,
              "events": [{"name": f"edit-{frame}", "frame": frame, "window": 16, "stable_frames": 2} for frame in events],
              "minimum_change_energy": 1e-8,
              "rois": [{"name": name, "normalized": region, "scale": ROI_SCALE, "require_change": False,
                        "thresholds": dict(THRESHOLDS), "ghost_metric": "error-change"} for name, region in rois]}
    enforce_gates(config)
    return config


def diagnostic_roi_config(frames, events, width=640, height=360):
    """The two ior 0 (Fresnel = 1) spheres of the grid, column 0, rows 0 (weight 1) and 1 (weight 0.5), same thresholds."""
    config = roi_config(frames, events, width, height)
    template = config["rois"][0]
    config["rois"] = [dict(template, name=name, normalized=inner_roi(0, row, width, height))
                      for name, row in (("ior0-sphere-weight1", 0), ("ior0-sphere-weight0.5", 1))]
    enforce_gates(config)
    return config


# --- cases ---------------------------------------------------------------------------------------------------------------
def case_table(out, binary):
    def clip(name, frames, reflections_denoise, spp, extra=(), env=None, scenario="specular-grid-edit"):
        folder = out / name
        command = [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario,
                   "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", "--shadows", "rt",
                   "--lighting", "brute", "--gi", "off", "--reflections", "rt", "--ao", "off",
                   "--lighting-denoise", reflections_denoise, "--reflection-samples", str(spp), "--lighting-seed", "1",
                   "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(frames), "--warmup", "0",
                   "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1",
                   "--capture-linear-sequence", str(folder / "linear"), "--capture-linear-signal", "hdr",
                   "--report", str(folder / "report.json"), *extra]
        return {"name": name, "kind": "clip", "frames": frames, "command": command, "env": env or {},
                "denoise": reflections_denoise, "folder": str(folder)}

    def adapt(name, scenario, adaptive):
        folder = out / name
        command = [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario,
                   "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "off", "--shadows", "off",
                   "--lighting", "legacy", "--gi", "off", "--reflections", "off", "--ao", "off", "--lighting-denoise", "off",
                   "--post", "--upscaler", "temporal", "--resolution", ADAPT_RESOLUTION, "--frames", str(ADAPT_FRAMES),
                   "--warmup", "0", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing",
                   "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", str(ADAPT_FRAMES - 1),
                   "--capture-linear-signal", "hdr", "--report", str(folder / "report.json"),
                   *(["--adaptive-shading"] if adaptive else [])]
        return {"name": name, "kind": "adapt", "scenario": scenario, "adaptive": adaptive, "command": command,
                "env": {}, "folder": str(folder)}

    toggle = ["--material-toggle-every", str(TOGGLE_EVERY)]
    return [clip("clip-raw", CLIP_FRAMES, "off", 1), clip("clip-custom", CLIP_FRAMES, "custom", 1),
            clip("clip-rt8", CLIP_FRAMES, "off", 8),
            clip("edit-custom", EDIT_FRAMES, "custom", 1, toggle), clip("edit-rt8", EDIT_FRAMES, "off", 8, toggle),
            clip("negative-lobe-flip", CLIP_FRAMES, "custom", 1, env={"PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE": "flip"}),
            clip("lobe-stripes", CLIP_FRAMES, "custom", 1, env={"PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE": "stripes"}),
            adapt("adapt-off", "specular-grid-rough", False), adapt("adapt-off-repeat", "specular-grid-rough", False),
            adapt("adapt-on", "specular-grid-rough", True),
            adapt("adapt-control-off", "base-grid-sun", False), adapt("adapt-control-on", "base-grid-sun", True)]


# --- guard and execution ---------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def gpu_lock(path):
    with pathlib.Path(path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GpuBusy("another cooperating Phosphor GPU verification owns the lock")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def running_gpu_processes():
    text = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, check=True).stdout
    return sorted({pathlib.Path(line.strip()).name for line in text.splitlines() if GPU_PROCESS.match(pathlib.Path(line.strip()).name)})


def run_cases(cases, timeout):
    import os
    env = dict(os.environ, MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1", MTL_DEBUG_LAYER_WARNING_MODE="nslog")
    for position, case in enumerate(cases):
        folder = pathlib.Path(case["folder"])
        if folder.exists() and any(folder.iterdir()):
            raise CaptureFailed(f"{folder} is not empty: choose a new --out")
    with gpu_lock(pathlib.Path("/tmp/phosphor-gpu-verification.lock")):
        for position, case in enumerate(cases):
            busy = running_gpu_processes()
            if busy:
                raise GpuBusy(f"refusing to start {case['name']}: other GPU jobs are running: {', '.join(busy)}")
            folder = pathlib.Path(case["folder"])
            folder.mkdir(parents=True, exist_ok=True)
            status = run_checked(case["command"], folder / "log.txt", timeout=timeout, env=dict(env, **case["env"]))
            if not status["passed"]:
                raise CaptureFailed(f"{case['name']}: capture failed: {status['failures']}")
            if position + 1 < len(cases):
                time.sleep(2)  # serial and quiet between GPU jobs


# --- evaluation -------------------------------------------------------------------------------------------------------------
def functional(case):
    """Process-level evidence of one case: clean exit, no validation message, LIGHTING readbacks PASS (clips)."""
    folder = pathlib.Path(case["folder"])
    log = (folder / "log.txt").read_text(errors="replace")
    errors = []
    if not re.search(r"^EXIT 0$", log, re.M):
        errors.append("EXIT marker is not 0")
    if GPU_ERROR.search(log):
        errors.append("GPU/API/shader validation failure in the log")
    if case["kind"] == "clip":
        lines = list(CHECK_LINE.finditer(log))
        if not lines or any(m[2] != "PASS" for m in lines):
            errors.append("LIGHTING readback lines missing or failing")
        want_lobe = case["denoise"] == "custom"
        if want_lobe and LOBE_LINE not in log:
            errors.append("the lobe-contract denoiser kernels were never served")
        if not want_lobe and LOBE_LINE in log:
            errors.append("lobe denoiser kernels in a run without the denoiser")
    return errors


def metrics_row(result):
    return {name: {"passed": all(r["checks"].values()), "checks": r["checks"],
                   **{k: v for k, v in r["metrics"].items() if k in ("linear_rmse_p95", "linear_rmse_max", "residual_flicker_mean",
                                                                      "error_change_ghost_fraction", "recovery_frames")}}
            for name, r in result["regions"].items()}


def stale_clip(control, candidate, config, delay):
    """The control delayed by `delay` frames as a candidate: the best possible history that kept the old value."""
    refs, ids = clip_files(control)
    cands, candidate_ids = clip_files(candidate)
    if ids != candidate_ids:
        raise ValueError("frame sets differ")
    return analyse_clips((read_pfm(p) for p in refs[delay:]), (read_pfm(p) for p in cands[:-delay]), config, ids[delay:])


def event_visibility(control, config):
    """Largest per-pixel change of the control across each edit frame in the feature and parameter sphere ROIs."""
    refs, ids = clip_files(control)
    seen = {}
    for event in config["events"]:
        frame = event["frame"]
        before, after = read_pfm(refs[ids.index(frame - 1)]), read_pfm(refs[ids.index(frame)])
        worst = {}
        for roi in config["rois"]:
            if roi["name"] == "grid":
                continue
            x0, y0, x1, y1 = roi["normalized"]
            xs, ys = range(int(x0 * before.width), int(x1 * before.width) + 1), range(int(y0 * before.height), int(y1 * before.height) + 1)
            worst[roi["name"]] = sum(max(abs(a - b) for a, b in zip(before.pixel(x, y), after.pixel(x, y))) for x in xs for y in ys) / (len(xs) * len(ys))
        seen[event["name"]] = worst
    return seen


def evaluate_clips(out):
    result = {}
    static = roi_config(CLIP_FRAMES, [])
    edit = roi_config(EDIT_FRAMES, EDIT_EVENTS)
    for name in ("clip-raw", "clip-custom", "negative-lobe-flip", "lobe-stripes"):
        metrics = analyse_paths(out / "clip-rt8" / "linear", out / name / "linear", static)
        result[name] = {"role": "control" if name == "clip-raw" else "gate", "metrics": metrics_row(metrics)}
    result["clip-custom"]["passed"] = all(v["passed"] for v in result["clip-custom"]["metrics"].values())
    result["clip-custom"]["expectation"] = "every ROI check passes against the 8 spp control"
    result["negative-lobe-flip"]["passed"] = any(not v["passed"] for v in result["negative-lobe-flip"]["metrics"].values())
    result["negative-lobe-flip"]["expectation"] = "the clip gates FAIL (history rejected every frame)"
    result["clip-raw"]["passed"] = True
    result["clip-raw"]["expectation"] = "control: reported, does not gate"
    diagnostic = analyse_paths(out / "clip-rt8" / "linear", out / "clip-custom" / "linear", diagnostic_roi_config(CLIP_FRAMES, []))
    result["clip-custom"]["diagnostic_rois"] = metrics_row(diagnostic)
    # stripes: neighbours rejected across lobe keys must change the image
    last = f"frame-{CLIP_FRAMES - 1:06d}.pfm"
    a, b = read_pfm(out / "clip-custom" / "linear" / last), read_pfm(out / "lobe-stripes" / "linear" / last)
    differing = sum(1 for i in range(0, len(a.rgb), 3) if any(a.rgb[i + k] != b.rgb[i + k] for k in range(3)))
    result["lobe-stripes"]["differing_pixels_vs_custom"] = differing
    result["lobe-stripes"]["passed"] = differing > 0
    result["lobe-stripes"]["expectation"] = "last frame differs from clip-custom (neighbour rejection is live)"
    # edit clip
    edit_metrics = analyse_paths(out / "edit-rt8" / "linear", out / "edit-custom" / "linear", edit)
    result["edit-custom"] = {"role": "gate", "metrics": metrics_row(edit_metrics), "events": EDIT_EVENTS,
                             "expectation": "every ROI check incl. recovery <= 4 frames at RMSE 0.06 passes against the 8 spp control with the same edits"}
    # Gate of the edit clip: every check on the two probe spheres, and the recovery criterion on every ROI.  The grid ROI's steady
    # metrics (p95, flicker) are those of clip-custom, where they gate; here they include the edit frames and are reported only.
    result["edit-custom"]["expectation"] = ("recovery <= 4 frames at RMSE 0.06 on every ROI, every check on the probe spheres, against the 8 spp "
                                            "control with the same edits (the grid ROI's p95 and flicker include the edit frames: reported, gated on clip-custom)")
    result["edit-custom"]["passed"] = all(v["checks"]["recovery"] for v in result["edit-custom"]["metrics"].values()) and \
        all(v["passed"] for k, v in result["edit-custom"]["metrics"].items() if k != "grid")
    result["edit-custom"]["event_visibility_control_mean_abs"] = event_visibility(out / "edit-rt8" / "linear", edit)
    result["edit-custom"]["diagnostic_rois"] = metrics_row(analyse_paths(out / "edit-rt8" / "linear", out / "edit-custom" / "linear",
                                                                         diagnostic_roi_config(EDIT_FRAMES, EDIT_EVENTS)))
    stale = stale_clip(out / "edit-rt8" / "linear", out / "edit-rt8" / "linear", edit, STALE_FRAMES)
    stale_row = metrics_row(stale)
    result["stale-control"] = {"role": "negative", "metrics": stale_row, "delay_frames": STALE_FRAMES,
                               "expectation": "the control delayed by 5 frames FAILS the recovery gate on a probe sphere",
                               "passed": any(not v["checks"]["recovery"] for k, v in stale_row.items() if k != "grid")}
    return result


def evaluate_adaptive(out):
    import numpy as np
    width, height = (int(v) for v in ADAPT_RESOLUTION.split("x"))
    read = lambda name: np.array(read_pfm(out / name / "engine.pfm").rgb, dtype=np.float32).reshape(height, width, 3)
    reused = lambda name: int(re.search(r'"reused_pixels_last":\s*(\d+)', (out / name / "report.json").read_text())[1])
    off, repeat, on = read("adapt-off"), read("adapt-off-repeat"), read("adapt-on")
    # feature spheres: eroded discs (certainly on the sphere); the neutral sphere (no feature bit, reuse allowed): a dilated disc
    # that covers its whole silhouette (spheres are 1 m apart, radius 0.4 m: the discs never overlap).
    feature, neutral = np.zeros((height, width), bool), np.zeros((height, width), bool)
    for row in range(ROWS):
        for column in range(COLUMNS):
            if is_neutral(column, row):
                neutral |= disc_mask(np, column, row, width, height, 1.25)
            else:
                feature |= disc_mask(np, column, row, width, height, 0.9)
    differ = np.any(off != on, axis=2)
    control = np.any(read("adapt-control-off") != read("adapt-control-on"), axis=2)
    row = {"resolution": ADAPT_RESOLUTION, "feature_sphere_pixels": int(feature.sum()), "neutral_sphere_pixels": int(neutral.sum()),
           "repeat_identical": bool(np.array_equal(off, repeat)),
           "feature_pixels_differing": int((differ & feature).sum()), "differing_pixels_total": int(differ.sum()),
           "differing_outside_neutral_sphere": int((differ & ~neutral).sum()),
           "reused_pixels_adapt_on": reused("adapt-on"), "reused_pixels_adapt_off": reused("adapt-off"),
           "control_differing_pixels": int(control.sum()), "control_reused_pixels": reused("adapt-control-on")}
    row["passed"] = (row["repeat_identical"] and row["feature_pixels_differing"] == 0 and row["differing_outside_neutral_sphere"] == 0 and
                     row["reused_pixels_adapt_on"] > 0 and row["control_differing_pixels"] > 0 and row["feature_sphere_pixels"] > 0)
    row["expectation"] = ("two identical runs agree, feature-sphere pixels are identical with --adaptive-shading, reuse is live "
                          "(reused > 0, the control scene changes pixels)")
    return row


def verdict(clips, adaptive):
    entries = {k: v["passed"] for k, v in clips.items()}
    entries["adaptive-shading"] = adaptive["passed"]
    return all(entries.values()), entries


def markdown(clips, adaptive, status):
    lines = [f"# F15.1 ticket 08 clip checks: {status}", ""]
    for name, entry in clips.items():
        lines.append(f"## {name}: {'PASS' if entry['passed'] else 'FAIL'}  ({entry.get('expectation', '')})")
        for roi, m in entry.get("metrics", {}).items():
            lines.append(f"- {roi}: p95 {m['linear_rmse_p95']:.4f}  max {m['linear_rmse_max']:.4f}  flicker {m['residual_flicker_mean']:.4f}  "
                         f"ghost {m['error_change_ghost_fraction']:.4f}  recovery {m['recovery_frames']}  {'ok' if m['passed'] else 'checks ' + str({k: v for k, v in m['checks'].items() if not v})}")
        for roi, m in entry.get("diagnostic_rois", {}).items():
            lines.append(f"- (diagnostic, not gating) {roi}: p95 {m['linear_rmse_p95']:.4f}  max {m['linear_rmse_max']:.4f}  flicker {m['residual_flicker_mean']:.4f}  recovery {m['recovery_frames']}")
        if "differing_pixels_vs_custom" in entry:
            lines.append(f"- differing pixels vs clip-custom: {entry['differing_pixels_vs_custom']}")
        if "event_visibility_control_mean_abs" in entry:
            lines.append(f"- control change across each edit frame (mean abs per ROI): {entry['event_visibility_control_mean_abs']}")
        lines.append("")
    lines.append(f"## adaptive-shading: {'PASS' if adaptive['passed'] else 'FAIL'}  ({adaptive['expectation']})")
    lines += [f"- {k}: {v}" for k, v in adaptive.items() if k not in ("passed", "expectation")]
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-clip")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="capture every case serially, then evaluate")
    mode.add_argument("--evaluate", action="store_true", help="evaluate existing captures (no GPU process)")
    mode.add_argument("--plan", action="store_true", help="write the commands only")
    parser.add_argument("--timeout", type=float, default=600)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out, binary = args.out.resolve(), args.binary.resolve()
    cases = case_table(out, binary)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 1, "thresholds": THRESHOLDS, "roi_scale": ROI_SCALE, "clip_frames": CLIP_FRAMES, "edit_frames": EDIT_FRAMES,
                "edit_events": EDIT_EVENTS, "resolution": RESOLUTION, "adaptive_resolution": ADAPT_RESOLUTION,
                "cases": [{k: v for k, v in c.items()} for c in cases],
                "reference": "HIGHER_SAMPLE_CONTROL_NOT_INDEPENDENT_OR_CONVERGED (clip-rt8, edit-rt8)"}
    if binary.is_file():
        manifest["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.plan:
        for case in cases:
            print(" ".join([f"{k}={v}" for k, v in case["env"].items()] + case["command"]))
        print(f"NOT_EXECUTED: planned {len(cases)} cases, manifest at {out / 'manifest.json'}")
        return EXIT_OK
    try:
        if args.run:
            if not binary.is_file():
                print("--run requires an existing compiled renderer; this runner never builds", file=sys.stderr)
                return EXIT_USAGE
            run_cases(cases, args.timeout)
        errors = {c["name"]: functional(c) for c in cases}
        clips = evaluate_clips(out)
        for name, entry in clips.items():
            if name in errors and errors[name]:
                entry["passed"] = False
                entry["functional_errors"] = errors[name]
        adaptive = evaluate_adaptive(out)
        adaptive["functional_errors"] = {n: e for n, e in errors.items() if e and n.startswith("adapt")}
        adaptive["passed"] = adaptive["passed"] and not adaptive["functional_errors"]
        ok, entries = verdict(clips, adaptive)
    except GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return EXIT_BUSY
    except CaptureFailed as error:
        print(str(error), file=sys.stderr)
        return EXIT_CAPTURE
    status = "ALL EXPECTATIONS MET" if ok else "EXPECTATION FAILED"
    (out / "results.json").write_text(json.dumps({"status": status, "entries": entries, "clips": clips, "adaptive": adaptive}, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(clips, adaptive, status))
    print(markdown(clips, adaptive, status))
    print(f"{status} (exit {EXIT_OK if ok else EXIT_FAILED})")
    return EXIT_OK if ok else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
