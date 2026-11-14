#!/usr/bin/env python3
"""F15.4 (ticket 23): resolve GPU time of the terrain patch per active-layer count, A/B/A, for the binning decision (ticket 24).

PREPARED, NOT RUN BY THIS TICKET: the numbers need a quiet machine (no other agent, solver or benchmark: frame times become bimodal) and belong to
the orchestrator.  The script refuses to start while another GPU process runs or the verification lock is held, and records the load average.

Arms (all on the same measured view: the terrain patch at 1920x1080, mesh path, visibility resolve, saturated clocks `--no-vsync --gpu-timing-serial`,
warmup 64 frames, 256 measured frames, no capture, `--pipeline-sync` off so the AOT archive serves):
  single       the patch with layer 0 alone as an ordinary material (`terrain/single=0/coat=off`): the single-material resolve
  layers-N     the layered stack with N active layers per pixel (`terrain/weights=1,..,0/coat=off`, N = 1..4: constant mask with N unit weights)
  full         the generated mask, height contrast and coat finish (`terrain`): the real mix of active-layer counts and the feature class
  and, with --binning, each of them again with `--material-binning on` (per-class bins vs the union pipeline: the >= 5 % rule on the terrain).
A/B/A: per replica the order is single, layers-N, single for every N (and likewise full); the A value is the mean of the two single runs of that
replica, the effect is B / A - 1 per replica, reported as the median of 3 replicas with the min..max range.

  python3 -I tools/f15_layers_timing.py plan                         print the commands
  python3 -I tools/f15_layers_timing.py run --out build/f15-layers-timing [--binning] [--replicas 3] [--resolution 1920x1080]
  python3 -I tools/f15_layers_timing.py summarize --out DIR          rebuild the table from the reports of a run

The resolve unit is the report's `gpu_timing` unit whose name contains "resolve" (summed when the resolve is split in dispatches or bins).  Nothing
here claims a number: results are written to DIR/results.json and DIR/summary.md for perf-log.md / opt-log.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

FRAMES, WARMUP = 320, 64
ARMS = {
    "single": "terrain/single=0/coat=off",
    "layers-1": "terrain/weights=1,0,0,0/coat=off",
    "layers-2": "terrain/weights=1,1,0,0/coat=off",
    "layers-3": "terrain/weights=1,1,1,0/coat=off",
    "layers-4": "terrain/weights=1,1,1,1/coat=off",
    "full": "terrain",
}
COMPARED = ["layers-1", "layers-2", "layers-3", "layers-4", "full"]


def command(binary, scene, out, resolution, binning):
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scene, "--render-path", "visibility", "--geometry-path", "mesh",
            "--rt", "off", "--shadows", "off", "--lighting", "legacy", "--gi", "off", "--reflections", "ssr", "--ao", "off", "--lighting-denoise", "custom",
            "--post", "--upscaler", "native", "--resolution", resolution, "--frames", str(FRAMES), "--warmup", str(WARMUP), "--fixed-timestep", "--no-ui",
            "--no-vsync", "--gpu-timing-serial", "--material-binning", "on" if binning else "off", "--report", str(out)]


def schedule(replicas, binning_modes):
    """Run order: per binning mode and replica: A B A for every compared arm."""
    runs = []
    for binning in binning_modes:
        for replica in range(replicas):
            for arm in COMPARED:
                for position, name in enumerate(("single", arm, "single")):
                    runs.append({"binning": binning, "replica": replica, "arm": arm, "slot": f"{'ABA'[position]}{position}", "scene_arm": name})
    return runs


def run_name(run):
    return f"{'bin' if run['binning'] else 'union'}-r{run['replica']}-{run['arm']}-{run['slot']}"


def resolve_ms(report):
    """Mean GPU ms per frame of the units whose name contains "resolve" (the report's gpu_timing)."""
    timing = report.get("gpu_timing") or {}
    units = timing.get("passes") or timing.get("units") or []
    total = 0.0
    found = False
    for unit in units:
        if "resolve" in str(unit.get("name", "")).lower():
            total += float(unit.get("gpu_ms", unit.get("mean_ms", 0.0)))
            found = True
    return total if found else None


def summarize(out):
    reports = {}
    for path in sorted(out.glob("*/report.json")):
        reports[path.parent.name] = resolve_ms(json.loads(path.read_text()))
    effects = {}
    for mode in ("union", "bin"):
        for arm in COMPARED:
            per_replica = []
            replica = 0
            while f"{mode}-r{replica}-{arm}-A0" in reports:
                a0, b, a1 = (reports.get(f"{mode}-r{replica}-{arm}-{s}") for s in ("A0", "B1", "A2"))
                if None in (a0, b, a1) or (a0 + a1) == 0:
                    break
                single = 0.5 * (a0 + a1)
                per_replica.append({"single_ms": single, "arm_ms": b, "effect": b / single - 1.0, "single_spread": abs(a0 - a1) / single})
                replica += 1
            if per_replica:
                effects[f"{mode}/{arm}"] = {"replicas": per_replica, "median_effect": statistics.median(r["effect"] for r in per_replica),
                                            "min_effect": min(r["effect"] for r in per_replica), "max_effect": max(r["effect"] for r in per_replica),
                                            "median_arm_ms": statistics.median(r["arm_ms"] for r in per_replica),
                                            "median_single_ms": statistics.median(r["single_ms"] for r in per_replica)}
    return {"reports": reports, "effects": effects}


def markdown(result, load):
    lines = ["# F15.4 resolve GPU time per active-layer count (A/B/A, median of replicas)", "", f"load average at the start: {load}", "",
             "| mode | arm | single ms | arm ms | effect (median) | min..max | replicas |", "|---|---|---|---|---|---|---|"]
    for key, e in result["effects"].items():
        mode, arm = key.split("/")
        lines.append(f"| {mode} | {arm} | {e['median_single_ms']:.3f} | {e['median_arm_ms']:.3f} | {e['median_effect'] * 100:+.1f} % | "
                     f"{e['min_effect'] * 100:+.1f} .. {e['max_effect'] * 100:+.1f} % | {len(e['replicas'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "run", "summarize"])
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-layers-timing")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--binning", action="store_true", help="also measure each arm with --material-binning on")
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    args = parser.parse_args(argv)
    out = args.out.resolve()
    modes = [False, True] if args.binning else [False]
    runs = schedule(args.replicas, modes)
    if args.mode == "plan":
        for run in runs:
            print(" ".join(command(args.binary, ARMS[run["scene_arm"] if run["scene_arm"] == "single" else run["arm"]], out / run_name(run) / "report.json",
                                    args.resolution, run["binning"])))
        print(f"# {len(runs)} runs", file=sys.stderr)
        return 0
    if args.mode == "summarize":
        result = summarize(out)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "summary.md").write_text(markdown(result, "n/a"))
        print(markdown(result, "n/a"))
        return 0
    import f15_check  # the shared lock and GPU guard

    if not args.binary.is_file():
        parser.error("the renderer is not built")
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty")
    load = os.getloadavg()
    try:
        lock = f15_check.gpu_lock(args.gpu_lock)
        lock.__enter__()
    except f15_check.GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return 4
    with contextlib.ExitStack() as stack:
        stack.push(lock)
        for run in runs:
            busy = f15_check.running_gpu_processes()
            if busy:
                print(f"GPU busy: {busy}", file=sys.stderr)
                return 4
            folder = out / run_name(run)
            folder.mkdir(parents=True, exist_ok=True)
            scene = ARMS["single" if run["scene_arm"] == "single" else run["arm"]]
            code = subprocess.run(command(args.binary, scene, folder / "report.json", args.resolution, run["binning"]),
                                  stdout=(folder / "log.txt").open("w"), stderr=subprocess.STDOUT).returncode
            if code:
                print(f"{run_name(run)}: exit {code}", file=sys.stderr)
                return 6
            time.sleep(3)  # quiet between GPU jobs
    result = summarize(out)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(result, ", ".join(f"{x:.2f}" for x in load)))
    print(markdown(result, ", ".join(f"{x:.2f}" for x in load)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
