#!/usr/bin/env python3
"""F15.4 (ticket 23): attribute the layered-resolve cost to the exact-maths module and to the in-stack code, interleaved A/B rotation.

PREPARED FOR A QUIET MACHINE (no other agent or benchmark): the script takes the shared GPU lock and refuses to start while another GPU process runs.

An ARM is `label=BINARY[:MODULE]` (MODULE exact|fast, default exact; passed as `--debug-material-module`).  The terrain arms of tools/f15_layers_timing.py
(`single`, `layers-1`, `layers-4`, `full`, ...) are each run once per ARM per replica; the order of (arm, scene) is rotated by one position per replica so no
arm is always first.  Every run uses `--pipeline-sync` (the measured frames never serve a fallback pipeline: the fast-module pipelines with a feature mask are
not in the archive) and the F15.4 view (1920x1080, mesh path, visibility resolve, saturated clocks).  The report's "Material resolve" mean is the figure.

  python3 -I tools/f15_module_ab.py plan --arm tip=build/tip/phosphor:exact --arm tipfast=build/tip/phosphor:fast --scene single --scene layers-1
  python3 -I tools/f15_module_ab.py run  --out build/ab/quiet --arm ... --scene ... [--replicas 3] --terrain-textures DIR
  python3 -I tools/f15_module_ab.py summarize --out build/ab/quiet
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_layers_timing as base  # noqa: E402


def parse_arm(text):
    label, _, rest = text.partition("=")
    binary, _, module = rest.partition(":")
    if not label or not binary or module not in ("", "exact", "fast"):
        raise SystemExit(f"bad --arm {text!r}: label=BINARY[:exact|fast]")
    return {"label": label, "binary": pathlib.Path(binary), "module": module or "exact"}


def schedule(arms, scenes, replicas):
    runs = []
    pairs = [(a, s) for s in scenes for a in arms]
    for replica in range(replicas):
        shift = replica % len(pairs)
        for arm, scene in pairs[shift:] + pairs[:shift]:
            runs.append({"replica": replica, "arm": arm, "scene": scene})
    return runs


def command(run, out, resolution, textures):
    cmd = base.command(run["arm"]["binary"], base.ARMS[run["scene"]], out / "report.json", resolution, False, textures)
    return cmd + ["--pipeline-sync", "--debug-material-module", run["arm"]["module"]]


def name(run):
    return f"r{run['replica']}-{run['arm']['label']}-{run['scene']}"


def summarize(out):
    table = {}
    for path in sorted(out.glob("*/report.json")):
        _replica, rest = path.parent.name.split("-", 1)
        ms = base.resolve_ms(json.loads(path.read_text()))
        table.setdefault(rest, []).append(ms)
    lines = ["| arm-scene | Material resolve ms (median) | min..max | replicas |", "|---|---|---|---|"]
    for key, values in sorted(table.items()):
        lines.append(f"| {key} | {statistics.median(values):.3f} | {min(values):.3f} .. {max(values):.3f} | {len(values)} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "run", "summarize"])
    parser.add_argument("--arm", action="append", default=[], type=parse_arm)
    parser.add_argument("--scene", action="append", default=[], choices=sorted(base.ARMS))
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-module-ab")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--terrain-textures", type=pathlib.Path, default=None)
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if args.mode == "summarize":
        print(summarize(out))
        return 0
    runs = schedule(args.arm, args.scene or ["single", "layers-1", "layers-4", "full"], args.replicas)
    if args.mode == "plan":
        for run in runs:
            print(" ".join(command(run, out / name(run), args.resolution, args.terrain_textures)))
        print(f"# {len(runs)} runs", file=sys.stderr)
        return 0
    import f15_check

    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty")
    try:
        lock = f15_check.gpu_lock(args.gpu_lock)
        lock.__enter__()
    except f15_check.GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return 4
    print("load average at start:", ", ".join(f"{x:.2f}" for x in os.getloadavg()))
    with lock:
        for run in runs:
            busy = f15_check.running_gpu_processes()
            if busy:
                print(f"GPU busy: {busy}", file=sys.stderr)
                return 4
            folder = out / name(run)
            folder.mkdir(parents=True, exist_ok=True)
            code = subprocess.run(command(run, folder, args.resolution, args.terrain_textures), stdout=(folder / "log.txt").open("w"),
                                  stderr=subprocess.STDOUT).returncode
            log = (folder / "log.txt").read_text(errors="replace")
            if code or "fallbacks 0 (draws 0)" not in log:
                print(f"{name(run)}: exit {code}, fallback line missing or non-zero: not a valid measurement", file=sys.stderr)
                return 6
            time.sleep(3)
    print(summarize(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
