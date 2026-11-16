#!/usr/bin/env python3
"""F15.1 ticket 11 (round 3): the denoiser lobe-id criterion for anisotropic highlights, measured on the GPU.

Ticket 08: "neighbours and history are rejected across lobe ids".  An anisotropic highlight must therefore keep ONE lobe id from frame to
frame (the transport writes id 0, the base lobe, for a pixel without a coat) so that the specular history of the highlight accumulates.
This runner renders the anisotropy grid (bench 2 `--material-scenario aniso-grid-env`, 3 strengths x 4 rotations over a dielectric and a
metal, RT reflections, the custom denoiser, 1 spp, static camera) and reads the denoiser's own GPU counters of the SPECULAR signal
(PHOSPHOR_DIAGNOSTIC_DENOISE_HISTORY=1, `DENOISE-HISTORY` lines: pixels with a valid history, pixels whose history was reused, longest history):

  aniso-custom   the engine as it is: after the warm-up the reused fraction of the valid history pixels is high and the longest history
                 reaches the frame count (the history survives across frames: the lobe id is stable).
  aniso-flip     negative control, PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE=flip (the sampled lobe key changes every frame: history rejected across
                 lobe ids): the reused fraction collapses and no history grows.  The same counters must therefore be able to fail.
  base-custom    the isotropic base grid (`base-grid-env`) under the same configuration: the reference behaviour of a plain base lobe, the
                 anisotropic fractions must be within BASE_TOLERANCE of it.

Criteria fixed before running: aniso-custom reused fraction >= 0.9 x the base grid's and >= 0.5 absolute at the last frame; longest history of
aniso-custom >= FRAMES - 2; the flip control's reused fraction <= 0.1 x aniso-custom's and its longest history <= 2.

Usage (the GPU lock is taken by the caller: python3 gpulock.py python tools/f15_aniso_lobe_check.py --run --out DIR):
  python -I tools/f15_aniso_lobe_check.py --run --out build/f15-aniso-lobe [--binary build/release/phosphor]
  python -I tools/f15_aniso_lobe_check.py --evaluate --out build/f15-aniso-lobe
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
FRAMES = 40
RESOLUTION = "640x360"
BASE_FRACTION = 0.9
ABSOLUTE_REUSED = 0.5
FLIP_RATIO = 0.1
FLIP_LONGEST = 2
SPECULAR_SIGNAL = 1  # DENOISE_SIGNAL_SPECULAR: checked against the printed signal ids below (the signal with the longest history under reflections)


def cases(binary, out):
    def command(scenario, folder):
        return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario,
                "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", "--shadows", "rt",
                "--lighting", "brute", "--gi", "off", "--reflections", "rt", "--ao", "off",
                "--lighting-denoise", "custom", "--reflection-samples", "1", "--lighting-seed", "1",
                "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(FRAMES), "--warmup", "0",
                "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1", "--pipeline-sync",
                "--report", str(folder / "report.json")]
    plan = []
    for name, scenario, env in (("aniso-custom", "aniso-grid-env", {}),
                                ("aniso-flip", "aniso-grid-env", {"PHOSPHOR_DIAGNOSTIC_DENOISE_LOBE": "flip"}),
                                ("base-custom", "base-grid-env", {})):
        folder = out / name
        plan.append({"name": name, "folder": folder, "command": command(scenario, folder),
                     "env": {"PHOSPHOR_DIAGNOSTIC_DENOISE_HISTORY": "1", **env}})
    return plan


def run(plan):
    for case in plan:
        case["folder"].mkdir(parents=True, exist_ok=True)
        result = subprocess.run(case["command"], env={**os.environ, **case["env"]}, capture_output=True, text=True, timeout=600, cwd=REPO)
        (case["folder"] / "stdout.txt").write_text(result.stdout)
        (case["folder"] / "stderr.txt").write_text(result.stderr)
        archive_misses = result.stderr.count("Failed with error:")
        print(f"{case['name']}: exit {result.returncode}, archive misses {archive_misses}")
        if result.returncode != 0 or archive_misses:
            raise SystemExit(4)


def history_lines(folder):
    rows = []
    for line in (folder / "stdout.txt").read_text().splitlines():
        m = re.match(r"DENOISE-HISTORY slot (\d+) signal (\d+) (\{.*\})", line)
        if m:
            row = json.loads(m.group(3))
            row["signal"] = int(m.group(2))
            rows.append(row)
    return rows


def summarise(folder):
    rows = history_lines(folder)
    signals = sorted({r["signal"] for r in rows})
    out = {}
    for signal in signals:
        last = [r for r in rows if r["signal"] == signal][-1]
        out[signal] = {"frames": len([r for r in rows if r["signal"] == signal]), "last": last,
                       "reused_fraction": last["reused"] / max(last["valid"], 1)}
    return out


def evaluate(out):
    summary = {name: summarise(out / name) for name in ("aniso-custom", "aniso-flip", "base-custom")}
    report = {"frames": FRAMES, "cases": {}, "criteria": {"base_fraction": BASE_FRACTION, "absolute_reused": ABSOLUTE_REUSED,
                                                          "flip_ratio": FLIP_RATIO, "flip_longest": FLIP_LONGEST}}
    ok = True
    # the specular signal is the one whose history the flip control destroys
    flip = summary["aniso-flip"]
    custom = summary["aniso-custom"]
    base = summary["base-custom"]
    checks = []
    for signal in sorted(custom):
        if signal not in flip or signal not in base:
            continue
        c, f, b = custom[signal], flip[signal], base[signal]
        discriminates = f["reused_fraction"] <= FLIP_RATIO * max(c["reused_fraction"], 1e-9) and f["last"]["max_length"] <= FLIP_LONGEST
        passes = (c["reused_fraction"] >= max(BASE_FRACTION * b["reused_fraction"], 0.0) and c["reused_fraction"] >= ABSOLUTE_REUSED
                  and c["last"]["max_length"] >= FRAMES - 2)
        checks.append({"signal": signal, "aniso_reused_fraction": c["reused_fraction"], "aniso_max_length": c["last"]["max_length"],
                       "flip_reused_fraction": f["reused_fraction"], "flip_max_length": f["last"]["max_length"],
                       "base_reused_fraction": b["reused_fraction"], "base_max_length": b["last"]["max_length"],
                       "control_discriminates": discriminates, "aniso_passes": passes})
    report["checks"] = checks
    # the gate is the signal on which the control discriminates (the specular signal); a run where none does is not a proof
    gated = [c for c in checks if c["control_discriminates"]]
    ok = bool(gated) and all(c["aniso_passes"] for c in gated)
    report["gated_signals"] = [c["signal"] for c in gated]
    report["passed"] = ok
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-aniso-lobe")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(cases(args.binary, args.out))
    return evaluate(args.out)


if __name__ == "__main__":
    sys.exit(main())
