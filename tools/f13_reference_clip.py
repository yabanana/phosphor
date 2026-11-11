#!/usr/bin/env python3
"""Per-frame F13 reference clips for the scripted reflection fixtures.

The fixture scripts (src/testbench/reflection_validation.cpp) are re-derived
here from their published constants, applied to ONE exported base snapshot,
and checked against further engine snapshots before any frame is rendered.
Frames with an identical scene state share one converged reference image.
No renderer image is read to build or converge the references.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import shutil
import struct
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import snapshot, write_pfm  # noqa: E402
import f13_reference as reference  # noqa: E402
import battery_pause  # noqa: E402

DT = struct.unpack("<f", struct.pack("<f", 1.0 / 60.0))[0]  # engine float fixed step, summed in double
LIGHT_STEP, CAMERA_CUT, PERIOD = 2.0, 4.0, 8.0
TARGET = (0.0, 1.1, 0.0)
PANEL_SLOT, OCCLUDER_SLOT, PANEL_MATERIAL = 64, 66, 3


def seconds(frame):
    """Fixture time when frame `frame` is rendered: one positive update per frame, starting at frame 0."""
    total = 0.0
    for _ in range(frame + 1):
        total += DT
    return total


def normalise(v):
    length = math.sqrt(sum(c * c for c in v))
    return [c / length for c in v]


def look(position):
    return normalise([t - p for t, p in zip(TARGET, position)])


def translate(world, position):
    world = list(world)
    world[12:15] = position
    return world


def state(scene, base, frame):
    """Scene description for `frame`, derived from the frame-0 base snapshot and the fixture script."""
    data = copy.deepcopy(base)
    data["frame"] = frame
    t = seconds(frame)
    phase = math.fmod(t, PERIOD)
    if scene == "disocclusion":
        x = 0.0 if phase < 2 else 1.8 * (phase - 2) / 2 if phase < 4 else 1.8
        position = [-0.8 + 0.45 * phase, 2.2, 7.0] if phase < CAMERA_CUT else [-1.8, 2.5, 6.5]
        data["camera"]["position"] = position
        data["camera"]["direction"] = look(position)
        data["camera"]["up"] = [0.0, 1.0, 0.0]
        for instance in data["instances"]:
            if instance["slot"] == OCCLUDER_SLOT:
                instance["world"] = translate(instance["world"], [x, 1.25, 2.0])
    elif scene == "moving-light":
        for instance in data["instances"]:
            if instance["slot"] == PANEL_SLOT:
                instance["world"] = translate(instance["world"], [1.0 + 0.7 * math.sin(phase), 1.3, 8.4])
        toward = normalise([-0.35 + 0.3 * math.sin(phase * 0.6), 0.85, 0.4])
        data["lights"][0]["direction"] = [-c for c in toward]
        step = int(t / LIGHT_STEP) % 4
        emission = [0.0, 0.0, 0.0] if step == 1 else [0.2, 0.5, 8.0] if step == 2 else [8.0, 0.4, 0.2]
        data["materials"][PANEL_MATERIAL]["emissive"] = emission
        data["lights"][0]["intensity"] = 6.0 if step == 1 else 12.0
    elif scene != "mirror":
        raise ValueError(f"no script for fixture {scene}")
    return data


def window(regions, width, height, margin):
    """Smallest pixel rectangle covering every region, widened by `margin` pixels (the metrics' support radius)."""
    if not regions:
        raise ValueError("--window-regions needs --regions")
    boxes = [(math.floor(x0 * width), math.floor(y0 * height), math.ceil(x1 * width), math.ceil(y1 * height))
             for x0, y0, x1, y1 in regions.values()]
    return (max(0, min(b[0] for b in boxes) - margin), max(0, min(b[1] for b in boxes) - margin),
            min(width, max(b[2] for b in boxes) + margin), min(height, max(b[3] for b in boxes) + margin))


def key(data):
    """Identity of everything the reference reads; equal keys render identical images."""
    fields = {"camera": data["camera"], "instances": data["instances"], "materials": data["materials"],
              "lights": data["lights"], "sky": data["sky"], "sun": data.get("sun_angular_radius")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def compare_state(predicted, exported):
    """Largest absolute difference between a reconstructed and an exported snapshot."""
    worst = {"camera_position": 0.0, "camera_direction": 0.0, "instance_translation": 0.0,
             "instance_linear": 0.0, "emission": 0.0, "sun_direction": 0.0, "sun_intensity": 0.0}
    def diff(a, b): return max(abs(x - y) for x, y in zip(a, b))
    worst["camera_position"] = diff(predicted["camera"]["position"], exported["camera"]["position"])
    worst["camera_direction"] = diff(normalise(predicted["camera"]["direction"]), normalise(exported["camera"]["direction"]))
    slots = {i["slot"]: i for i in exported["instances"]}
    if sorted(slots) != sorted(i["slot"] for i in predicted["instances"]):
        raise ValueError("instance set differs from the exported frame")
    for instance in predicted["instances"]:
        other = slots[instance["slot"]]["world"]
        worst["instance_translation"] = max(worst["instance_translation"], diff(instance["world"][12:15], other[12:15]))
        worst["instance_linear"] = max(worst["instance_linear"], diff(instance["world"][:12], other[:12]))
    for a, b in zip(predicted["materials"], exported["materials"]):
        worst["emission"] = max(worst["emission"], diff(a["emissive"], b["emissive"]))
    worst["sun_direction"] = diff(predicted["lights"][0]["direction"], exported["lights"][0]["direction"])
    worst["sun_intensity"] = abs(predicted["lights"][0]["intensity"] - exported["lights"][0]["intensity"])
    return worst


def command_check(args):
    """Reconstruct every exported frame from the frame-0 export and compare state and primary hits."""
    exports = pathlib.Path(args.exports)
    root, base = snapshot(exports / "snapshot-000000")
    rows, passed = [], True
    for folder in sorted(exports.glob("snapshot-*")):
        frame = int(folder.name.split("-")[1])
        _, exported = snapshot(folder)
        if exported["frame"] != frame:
            raise ValueError(f"{folder} holds frame {exported['frame']}")
        predicted = state(args.scene, base, frame)
        worst = compare_state(predicted, exported)
        context = reference.Context(root, predicted, args.threads, "regularized")
        positions = reference.validate_positions(context.np, context, exports / f"positions-{frame:06d}.pfm")
        ok = (worst["camera_position"] <= 1e-5 and worst["camera_direction"] <= 1e-5 and worst["instance_translation"] <= 1e-5
              and worst["instance_linear"] <= 1e-6 and worst["emission"] <= 1e-6 and worst["sun_direction"] <= 1e-5
              and worst["sun_intensity"] == 0 and positions["passed"])
        passed &= ok
        rows.append({"frame": frame, "state": worst, "positions": {k: positions[k] for k in
                     ("coverage_mismatch", "silhouette_pixels", "unexplained_mismatch", "max_position_error_m", "passed")},
                     "passed": ok})
        print(json.dumps(rows[-1]), flush=True)
    report = {"schema": 1, "scene": args.scene, "frames": rows, "passed": passed,
              "tool_sha256": reference.sha(__file__), "reference_tool_sha256": reference.sha(reference.__file__)}
    pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


def command_render(args):
    """Converged reference per distinct state; writes frame-NNNNNN.pfm for every requested frame."""
    root, base = snapshot(args.base)
    if base["frame"] != 0:
        raise ValueError("clip reconstruction starts from the frame-0 export")
    out = pathlib.Path(args.out).resolve()
    (out / "linear").mkdir(parents=True, exist_ok=args.resume)
    (out / "states").mkdir(exist_ok=True)
    regions = json.loads(pathlib.Path(args.regions).read_text()) if args.regions else None
    frames = list(range(args.first, args.last + 1))
    groups = {}
    for frame in frames:
        groups.setdefault(key(state(args.scene, base, frame)), []).append(frame)
    index = []
    block = 0.0  # render seconds since the last recharge break
    for number, (identity, members) in enumerate(groups.items()):
        folder = out / "states" / identity[:16]
        report_path = folder / "reference_report.json"
        if not (args.resume and report_path.is_file() and json.loads(report_path.read_text()).get("converged")):
            if folder.exists():
                shutil.rmtree(folder)
            data = state(args.scene, base, members[0])
            context = reference.Context(root, data, args.threads, args.secondary_specular)
            if args.window_regions:
                context.window = window(regions, context.width, context.height, args.window_margin)
            namespace = argparse.Namespace(spp=args.spp, spp_per_pass=args.spp_per_pass)
            started = time.monotonic()
            report = reference.render_frame(context, namespace, folder, None, regions)
            report["state"] = identity
            report["frames"] = members
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            elapsed = time.monotonic() - started
            block += elapsed
            gain = None
            if args.work_block_minutes and block >= 60.0 * args.work_block_minutes:
                gain, block = args.block_gain, 0.0
            if args.rest_ratio > 0 or args.battery_low is not None or gain:
                battery_pause.rest(elapsed, args.rest_ratio, args.battery_low, args.battery_high, gain,
                                   args.battery_cap, sys.stdout)
        report = json.loads(report_path.read_text())
        index.append({"state": identity, "frames": members, "converged": report["converged"],
                      "spp_per_seed": report["records"][-1]["spp_per_seed"]})
        print(json.dumps({"group": number + 1, "of": len(groups), **index[-1]}), flush=True)
        for frame in members:
            target = out / "linear" / f"frame-{frame:06d}.pfm"
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copyfile(folder / "reference.pfm", target)
    converged = all(row["converged"] for row in index)
    manifest = {"schema": 1, "scene": args.scene, "base_snapshot": str(pathlib.Path(args.base).resolve()),
                "base_snapshot_sha256": reference.scene_digest(root), "frames": [frames[0], frames[-1]],
                "states": index, "converged": converged, "model": reference.MODEL,
                "secondary_specular": args.secondary_specular, "regions": regions or {},
                "window_regions": bool(args.window_regions), "window_margin": args.window_margin,
                "tool_sha256": reference.sha(__file__), "reference_tool_sha256": reference.sha(reference.__file__),
                "candidate_read": False, "phase_accepted": False}
    (out / "clip_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if converged else 1


def command_refs(args):
    """Write the runner's --references mapping for converged clips, bound to the frozen manifest bytes."""
    manifest_path = pathlib.Path(args.manifest).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    frames = manifest["frames"]
    mapping = {}
    for item in args.case:
        name, folder = item.split("=", 1)
        folder = pathlib.Path(folder).resolve()
        if name not in {case["name"] for case in manifest["cases"]}:
            raise ValueError(f"{name} is not a case of the frozen manifest")
        report = json.loads((folder / "clip_report.json").read_text())
        files = sorted((folder / "linear").glob("frame-*.pfm"))
        if not report["converged"] or len(files) != frames or report["frames"] != [0, frames - 1]:
            raise ValueError(f"{folder} is not a complete converged {frames}-frame clip")
        mapping[name] = {"linear": str(folder / "linear"),
                         "provenance": {"validated": True, "independent": True,
                                        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                                        "clip_report": str(folder / "clip_report.json"),
                                        "clip_report_sha256": reference.sha(folder / "clip_report.json"),
                                        "window_regions": report.get("window_regions", False)}}
    pathlib.Path(args.out).write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({name: row["provenance"]["manifest_sha256"][:16] for name, row in mapping.items()}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check")
    p.add_argument("scene")
    p.add_argument("exports", help="folder with snapshot-NNNNNN/ and positions-NNNNNN.pfm engine exports")
    p.add_argument("--report", required=True)
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.set_defaults(function=command_check)
    p = sub.add_parser("render")
    p.add_argument("scene")
    p.add_argument("base", help="frame-0 engine snapshot of this fixture")
    p.add_argument("--out", required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=299)
    p.add_argument("--regions")
    p.add_argument("--spp", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--spp-per-pass", type=int, default=32)
    p.add_argument("--secondary-specular", choices=["regularized", "raw", "off"], default="regularized")
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--resume", action="store_true", help="keep converged states already in --out")
    p.add_argument("--window-regions", action="store_true", help="render only the regions' bounding box (+margin)")
    p.add_argument("--window-margin", type=int, default=2)
    p.add_argument("--rest-ratio", type=float, default=0.0,
                   help="after each rendered state, idle for this fraction of its render time")
    p.add_argument("--battery-low", type=int, help="after each state, wait on battery or below this percent")
    p.add_argument("--battery-high", type=int, help="charge percent to reach before resuming")
    p.add_argument("--work-block-minutes", type=float, help="take a recharge break after this much render time")
    p.add_argument("--block-gain", type=int, default=15, help="charge points a recharge break waits for")
    p.add_argument("--battery-cap", type=int, default=90, help="upper bound of the recharge break target")
    p.set_defaults(function=command_render)
    p = sub.add_parser("refs", help="runner --references file for converged clips")
    p.add_argument("manifest", help="frozen-manifest.json of the plan to compare")
    p.add_argument("--case", action="append", required=True, metavar="NAME=CLIP_DIR")
    p.add_argument("--out", required=True)
    p.set_defaults(function=command_refs)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F13 reference clip failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
