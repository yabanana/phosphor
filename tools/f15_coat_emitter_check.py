#!/usr/bin/env python3
"""F15.1 ticket 09: emitter ownership under a clearcoat (no double count between DI and the coat lobe's transport).

Fixture: ReflectionValidation `coat-emitter` (the F13.1 rough-emitter geometry with two dark dielectric floor strips under a
full clearcoat of roughness 0.3 / 0.5; the glossy reflection of the black-body panel is the coat lobe).  The same frame is
rendered with three ownership regimes of the panel's specular lobe, converged (8 spp, no denoiser):
  legacy-ssr   legacy lighting: no sampled emitters, the SSR transport sees the panel whole (the single-count ground truth)
  brute-ssr    brute DI owns the whole lobe outside the RT band (SSR sees no emission)
  legacy-rt    legacy lighting + RT reflections (RT band by the coat roughness, SSR fallback sees the panel)
  brute-rt     brute DI + RT reflections: the panel lobe is split by the coat roughness (smoothstep share)
A double count would show as brute-rt/brute-ssr above legacy-ssr.  Gate: the mean radiance of each floor ROI of every
regime is within TOLERANCE of the legacy-ssr mean.  Negative control: the sum legacy-ssr + brute-ssr (what counting the
panel twice would give) must FAIL the same gate.

Usage: uv run --offline --no-project --python 3.13 python -I tools/f15_coat_emitter_check.py --run --out build/f15-coat-emitter
       ... --evaluate --out DIR   (existing captures)
Exit codes: 0 every expectation met, 1 failed, 2 usage, 3 GPU busy, 4 capture failed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_checked import run_checked  # noqa: E402
from temporal_light_metrics import read_pfm  # noqa: E402

REPO = HERE.parent
RESOLUTION, FRAMES = "1280x720", 120  # 120: the lobe kernels compile at run time under shader validation; capture after they are served
ROIS = {"floor-coat0.3": (0.38, 0.60, 0.49, 0.97), "floor-coat0.5": (0.51, 0.60, 0.64, 0.97)}  # the F13.1 rough-emitter ROIs
TOLERANCE = 0.20  # relative, per ROI: DI area-light sampling and SSR/RT transport integrate the same lobe differently (+8..16 % on the UNCOATED F13.1 fixture too, measured with --fixture rough-emitter); a double count is +100 %
MINIMUM = 0.01    # the reflection must exist (mean luminance above the dark floor)
GPU_PROCESS = re.compile(r"^(phosphor|soc_bench|f\d+_spike)$")


def cases(out, binary, fixture="coat-emitter"):
    common = ["--scene", "procedural", "--bench", "6", "--reflection-scene", fixture, "--render-path", "visibility",
              "--geometry-path", "mesh", "--gi", "off", "--ao", "off", "--lighting-denoise", "off", "--reflection-samples", "8",
              "--lighting-seed", "1", "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(FRAMES),
              "--warmup", "0", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1",
              "--capture-linear-signal", "hdr", "--capture-linear-frame", str(FRAMES - 1)]
    table = {"legacy-ssr": ["--rt", "off", "--shadows", "off", "--lighting", "legacy", "--reflections", "ssr"],
             "brute-ssr": ["--rt", "on", "--shadows", "rt", "--lighting", "brute", "--reflections", "ssr"],
             "legacy-rt": ["--rt", "on", "--shadows", "off", "--lighting", "legacy", "--reflections", "rt"],
             "brute-rt": ["--rt", "on", "--shadows", "rt", "--lighting", "brute", "--reflections", "rt"]}
    return [{"name": n, "folder": str(out / n),
             "command": [str(binary), *common, *flags, "--capture-linear", str(out / n / "engine.pfm"),
                         "--report", str(out / n / "report.json")]}
            for n, flags in table.items()]


def run(cases_, timeout):
    env = dict(os.environ, MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1", MTL_DEBUG_LAYER_WARNING_MODE="nslog")
    for i, case in enumerate(cases_):
        busy = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True).stdout.split("\n")
        if any(GPU_PROCESS.match(pathlib.Path(x.strip()).name) for x in busy):
            print("GPU busy", file=sys.stderr)
            return 3
        folder = pathlib.Path(case["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        status = run_checked(case["command"], folder / "log.txt", timeout=timeout, env=env)
        if not status["passed"]:
            print(f"{case['name']}: capture failed: {status['failures']}", file=sys.stderr)
            return 4
        if i + 1 < len(cases_):
            time.sleep(2)
    return 0


def roi_means(path):
    image = read_pfm(path)
    w, h = image.width, image.height
    means = {}
    for name, (x0, y0, x1, y1) in ROIS.items():
        total, count = 0.0, 0
        for y in range(int(y0 * h), int(y1 * h)):
            for x in range(int(x0 * w), int(x1 * w)):
                r, g, b = image.pixel(x, y)
                total += 0.2126 * r + 0.7152 * g + 0.0722 * b
                count += 1
        means[name] = total / count
    return means


def evaluate(out):
    names = ["legacy-ssr", "brute-ssr", "legacy-rt", "brute-rt"]
    means = {n: roi_means(out / n / "engine.pfm") for n in names}
    truth = means["legacy-ssr"]
    rows, ok = {}, True
    for roi in ROIS:
        row = {"legacy-ssr": truth[roi]}
        for n in names[1:]:
            row[n] = means[n][roi]
            row[n + "-ratio"] = means[n][roi] / truth[roi]
        row["double-count-control-ratio"] = (truth[roi] + means["brute-ssr"][roi]) / truth[roi]
        gate = truth[roi] > MINIMUM and all(abs(row[n + "-ratio"] - 1) <= TOLERANCE for n in names[1:])
        control_fails = abs(row["double-count-control-ratio"] - 1) > TOLERANCE
        row["gate"] = gate
        row["negative_control_fails_gate"] = control_fails
        rows[roi] = row
        ok = ok and gate and control_fails
    logs = {}
    for n in names:
        text = (out / n / "log.txt").read_text(errors="replace")
        logs[n] = bool(re.search(r"^EXIT 0$", text, re.M)) and "Validation Error" not in text
        ok = ok and logs[n]
    return ok, {"tolerance": TOLERANCE, "rows": rows, "clean_exit": logs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    p.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-coat-emitter")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--fixture", default="coat-emitter", help="diagnostic: rough-emitter runs the uncoated metal twin")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--run", action="store_true")
    m.add_argument("--evaluate", action="store_true")
    a = p.parse_args(argv)
    out = a.out.resolve()
    if a.run:
        code = run(cases(out, a.binary.resolve(), a.fixture), a.timeout)
        if code:
            return code
    ok, result = evaluate(out)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("ALL EXPECTATIONS MET (exit 0)" if ok else "EXPECTATION FAILED (exit 1)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
