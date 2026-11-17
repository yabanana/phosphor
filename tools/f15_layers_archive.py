#!/usr/bin/env python3
"""F15.4 (ticket 23): targeted pipeline-archive update for the layered-material keys.

A layered scene specialises its pipelines with MATERIAL_FEATURES = 256 (layers) or 258 (layers + clearcoat) next to the lighting constants of each
runner key, so the archive needs those pipelines exactly as the keys request them.  Two steps, both label-independent and additive (existing
entries are never touched or duplicated; tools/filter_pipeline_harvest.py, tools/material_class_archive.py):

  harvest  --out DIR     run every non-cc0 `layers-*` key of tools/f15_check.py once (its exact command, 3 frames, synchronous pipelines, binning off
                         and on) with --harvest-pipelines: DIR/<key>-<mode>.json.  Serial, under the shared GPU lock (exit 3 = busy: wait and retry).
  update   --harvest DIR [--plan PLAN] [--archive shaders/pipelines.mtl4-json]
                         keep from the harvests the pipelines specialised with constant 22 that the archive lacks (every pipeline kind) plus the
                         RESOLVE_BIN class pipelines of the plan, merge, check the plan and the budgets, write the archive.

Exit codes: 0 ok, 1 check failed, 2 usage, 3 GPU busy, 6 a run failed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_check as fc  # noqa: E402


def keys():
    return [k for k, spec in fc.SCENARIOS.items() if k.startswith("layers-") and not spec.get("cc0")]


def command(binary, key, folder, binning):
    cmd = fc.capture_command(binary, key, folder)
    for flag in ("--capture-linear", "--capture-linear-frame", "--capture-linear-signal", "--export-reference", "--export-reference-frame", "--report"):
        if flag in cmd:
            i = cmd.index(flag)
            del cmd[i:i + 2]
    cmd[cmd.index("--frames") + 1] = "3"
    return cmd + ["--material-binning", "on" if binning else "off", "--harvest-pipelines", str(folder.parent / f"{key}-{'bin' if binning else 'union'}.json")]


def harvest(args):
    import fcntl
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.gpu_lock).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU lock held", file=sys.stderr)
            return 3
        for key in keys():
            for binning in (False, True):
                busy = fc.running_gpu_processes()
                if busy:
                    print(f"GPU busy: {busy}", file=sys.stderr)
                    return 3
                folder = out / f"{key}-{'bin' if binning else 'union'}"
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / "engine.log").open("wb") as log:
                    status = subprocess.run(command(args.binary.resolve(), key, folder, binning), stdout=log, stderr=subprocess.STDOUT).returncode
                if status != 0:
                    print(f"{key} (binning {binning}): exit {status}", file=sys.stderr)
                    return 6
                print(f"harvested {key} binning={binning}", flush=True)
    return 0


def update(args):
    import filter_pipeline_harvest as fph
    import material_class_archive as mca
    import merge_pipeline_harvests as mph

    archive_path = args.archive.resolve()
    archive = json.loads(archive_path.read_text())
    harvests = sorted(h for h in args.harvest.resolve().glob("*.json") if h.name != "report.json")
    if not harvests:
        print("no harvests", file=sys.stderr)
        return 2
    combined, label = {}, None
    for h in harvests:
        raw = mca.normalise(json.loads(h.read_text()))
        label = label or raw["libraries"][0]["label"]
        combined = mph.merge(combined, mph.normalize_library(raw, label))
    kept = fph.filter_harvest(combined, mca.MATERIAL_FEATURES_INDEX, present=archive)
    merged = mph.merge(archive, mph.normalize_library(kept, archive["libraries"][0]["label"]))
    plan = json.loads(args.plan.read_text())
    bin_index = next(c["index"] for p in plan["pipelines"] for c in p["constants"] if c["value"] & 0x3000)
    merged = mca.update(plan, merged, [json.loads(h.read_text()) for h in harvests], bin_index)
    problems, report = mca.check(plan, merged)
    added = {k: len(v) for k, v in kept["pipeline_descriptors"].items()}
    print(json.dumps({"added": added, **report}, sort_keys=True))
    for p in problems:
        print("FAIL:", p, file=sys.stderr)
    if problems:
        return 1
    archive_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--out", type=pathlib.Path, required=True)
    h.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    h.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    u = sub.add_parser("update")
    u.add_argument("--harvest", type=pathlib.Path, required=True)
    u.add_argument("--plan", type=pathlib.Path, default=REPO / "build/release/generated/pipeline/material_class_plan.json")
    u.add_argument("--archive", type=pathlib.Path, default=REPO / "shaders/pipelines.mtl4-json")
    args = parser.parse_args(argv)
    return harvest(args) if args.mode == "harvest" else update(args)


if __name__ == "__main__":
    raise SystemExit(main())
