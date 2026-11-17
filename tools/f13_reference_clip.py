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
from f12_reference import read_pfm, snapshot, write_pfm  # noqa: E402
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
    elif scene not in ("mirror", "roughness", "rough-emitter", "probe-box"):  # static fixtures: the base state for every frame
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


def render_config(args, regions, scene_sha256):
    """Every input that changes a converged state image besides the scene state key."""
    return {"model": reference.MODEL, "secondary_specular": args.secondary_specular,
            "regularization_scale": args.regularization_scale, "spp": args.spp, "spp_per_pass": args.spp_per_pass,
            "regions": regions or {}, "window_regions": bool(args.window_regions), "window_margin": args.window_margin,
            "scene_sha256": scene_sha256, **({"mirror_bounces": args.mirror_bounces} if getattr(args, "mirror_bounces", 0) else {})}


def reusable(report_path, config, resume):
    """A saved state is reused only when it converged under exactly the requested configuration."""
    if not (resume and report_path.is_file()):
        return False
    saved = json.loads(report_path.read_text())
    return bool(saved.get("converged")) and saved.get("render_config") == config


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
    config = render_config(args, regions, reference.scene_digest(root))
    index = []
    block = 0.0  # render seconds since the last recharge break
    for number, (identity, members) in enumerate(groups.items()):
        folder = out / "states" / identity[:16]
        report_path = folder / "reference_report.json"
        if not reusable(report_path, config, args.resume):
            if folder.exists():
                shutil.rmtree(folder)
            data = state(args.scene, base, members[0])
            context = reference.Context(root, data, args.threads, args.secondary_specular,
                                        regularization_scale=args.regularization_scale,
                                        mirror_bounces=getattr(args, "mirror_bounces", 0))
            if args.window_regions:
                context.window = window(regions, context.width, context.height, args.window_margin)
            namespace = argparse.Namespace(spp=args.spp, spp_per_pass=args.spp_per_pass)
            started = time.monotonic()
            report = reference.render_frame(context, namespace, folder, None, regions)
            report["state"] = identity
            report["frames"] = members
            report["render_config"] = config
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
                "base_snapshot_sha256": config["scene_sha256"], "frames": [frames[0], frames[-1]],
                "states": index, "converged": converged, "model": reference.MODEL,
                "secondary_specular": args.secondary_specular, "regularization_scale": args.regularization_scale,
                "regions": regions or {}, **({"mirror_bounces": args.mirror_bounces} if getattr(args, "mirror_bounces", 0) else {}),
                "window_regions": bool(args.window_regions), "window_margin": args.window_margin,
                "tool_sha256": reference.sha(__file__), "reference_tool_sha256": reference.sha(reference.__file__),
                "candidate_read": False, "phase_accepted": False}
    (out / "clip_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if converged else 1


# ---------------------------------------------------------------------------------------------------------------------------
# Correlated difference estimator: new_clip = old_converged + (render_new - render_old), same seeds and sample streams.
# ---------------------------------------------------------------------------------------------------------------------------

def seed_stream(seed, done, shift=0):
    """Mitsuba stream seed of the pass starting at sample `done` (the same formula as f13_reference.converge)."""
    return ((seed + shift) * 1000003 + done) & 0x7FFFFFFF


class Accumulator:
    """Running per-seed sums over the converge() streams; extend() continues exactly where the last call stopped."""

    def __init__(self, context, spp_per_pass, shift=0):
        self.context, self.spp_per_pass, self.shift, self.done = context, spp_per_pass, shift, 0
        self.sums = {seed: context.np.zeros((context.height, context.width, 3)) for seed in reference.SEEDS}

    def extend(self, spp):
        while self.done < spp:
            step = min(self.spp_per_pass, spp - self.done)
            for seed in reference.SEEDS:
                self.sums[seed] += self.context.render_pass(seed_stream(seed, self.done, self.shift), step)
            self.done += step

    def means(self):
        return [self.sums[seed] / self.done for seed in reference.SEEDS]


def accumulate(context, spp, spp_per_pass, shift=0):
    """Per-seed mean images over `spp` samples, with the converge() streams (shift != 0 decorrelates them on purpose)."""
    accumulator = Accumulator(context, spp_per_pass, shift)
    accumulator.extend(spp)
    return accumulator.means()


def combine(np, old, minus, plus):
    """new = old + mean over seeds of (plus_seed - minus_seed); also returns the per-seed differences."""
    differences = [p - m for p, m in zip(plus, minus)]
    return old + np.mean(differences, axis=0), differences


def difference_statistics(np, old, differences, area):
    """Effect size and the noise of the seed-averaged difference inside `area` (a pair of slices)."""
    mean_diff = np.mean(differences, axis=0)[area]
    effect = float(np.sqrt(np.mean(mean_diff ** 2)))
    # The two seeds are independent estimates of the same difference: half their difference is the noise of their mean.
    noise = float(np.sqrt(np.mean((differences[0][area] - differences[1][area]) ** 2))) / 2.0
    support = float(np.sqrt(np.mean(old[area] ** 2)))
    return {"support": support, "mean_old": float(np.mean(old[area])), "mean_difference": float(np.mean(mean_diff)),
            "relative_mean_effect": float(np.mean(mean_diff)) / max(float(np.mean(old[area])), 1e-12),
            "effect_rms": effect, "difference_noise_rms": noise,
            "effect_rms_over_support": effect / max(support, 1e-12), "noise_over_support": noise / max(support, 1e-12),
            "noise_over_effect": noise / max(effect, 1e-12)}


def read_pfm_array(path):
    """Linear PFM as float64 (height, width, 3), in the row order write_pfm takes."""
    return read_pfm(path).astype("float64")


def correction_pair(root, data, args, regions, window_margin, use_window):
    """[(ms_lobe, context)] for the old model (ms off, no chain) and the new one (ms on, mirror chain)."""
    contexts = []
    for ms, bounces in ((False, 0), (True, args.mirror_bounces)):
        reference.Context.ms_lobe = ms  # read when the tables are built and in lobes(): set again before every render
        context = reference.Context(root, data, args.threads, args.secondary_specular,
                                    regularization_scale=args.regularization_scale, mirror_bounces=bounces)
        if use_window:
            context.window = window(regions, context.width, context.height, window_margin)
        contexts.append((ms, context))
    return contexts


def render_difference(contexts, args, shift=0):
    """Render the old and the new model with the same seed streams (shift decorrelates the new one: negative control)."""
    results = []
    for ms, context in contexts:
        reference.Context.ms_lobe = ms
        results.append(accumulate(context, args.spp[-1] if isinstance(args.spp, list) else args.spp, args.spp_per_pass,
                                  shift if ms else 0))
    return results


def adaptive_difference(contexts, args, old_image, slices, sigma):
    """Extend both renders through the spp checkpoints until the difference noise of every region is small against the old
    reference's own noise (`sigma[region]` = rms noise of its mean of two seeds); returns (off, on, spp, noise per region)."""
    np = contexts[0][1].np
    accumulators = [(ms, Accumulator(context, args.spp_per_pass)) for ms, context in contexts]
    for target in args.spp:
        for ms, accumulator in accumulators:
            reference.Context.ms_lobe = ms
            accumulator.extend(target)
        off, on = accumulators[0][1].means(), accumulators[1][1].means()
        differences = [p - m for p, m in zip(on, off)]
        noise = {name: difference_statistics(np, old_image, differences, area)["difference_noise_rms"] for name, area in slices.items()}
        if all(noise[name] <= args.max_noise_ratio * sigma[name] for name in slices):
            break
    return off, on, target, noise


def old_noise(report, names):
    """rms noise of the old clip's mean-of-two-seeds per region, from its last convergence record (seed_nrmse is the seed-pair error)."""
    regions = report["records"][-1]["regions"]
    return {name: regions[name]["support"] * regions[name]["seed_nrmse"] / 2.0 for name in names}


def command_correct(args):
    """Corrected clip: old converged references plus a same-seed (model_new - model_old) difference per state."""
    old_dir = pathlib.Path(args.old).resolve()
    old_report_path = old_dir / "clip_report.json"
    old = json.loads(old_report_path.read_text())
    root, base = snapshot(old["base_snapshot"])
    if reference.scene_digest(root) != old["base_snapshot_sha256"]:
        raise ValueError("base snapshot digest differs from the old clip report")
    regions, use_window, margin = old["regions"], old.get("window_regions", False), old.get("window_margin", 2)
    args.secondary_specular, args.regularization_scale = old["secondary_specular"], old["regularization_scale"]
    out = pathlib.Path(args.out).resolve()
    (out / "linear").mkdir(parents=True, exist_ok=True)
    (out / "states").mkdir(exist_ok=True)
    config = {"method": "difference", "spp": list(args.spp), "max_noise_ratio": args.max_noise_ratio, "spp_per_pass": args.spp_per_pass, "mirror_bounces": args.mirror_bounces,
              "old_clip_report_sha256": reference.sha(old_report_path), "regularization_scale": old["regularization_scale"],
              "secondary_specular": old["secondary_specular"], "regions": regions, "window_regions": use_window}
    index = []
    states = old["states"][args.first_state:args.last_state]
    for number, row in enumerate(states):
        identity, members = row["state"], row["frames"]
        folder = out / "states" / identity[:16]
        report_path = folder / "reference_report.json"
        saved = json.loads(report_path.read_text()) if report_path.is_file() else None
        if not (args.resume and saved and saved.get("render_config") == config and (folder / "reference.pfm").is_file()):
            if folder.exists():
                shutil.rmtree(folder)
            data = state(old["scene"], base, members[0])
            if key(data) != identity:
                raise ValueError(f"state {identity[:16]} is not reproduced by the script reconstruction")
            started = time.monotonic()
            contexts = correction_pair(root, data, args, regions, margin, use_window)
            np = contexts[0][1].np
            old_image = read_pfm_array(old_dir / "states" / identity[:16] / "reference.pfm")
            window_xyxy = contexts[0][1].window
            slices = reference.region_slices(regions, contexts[0][1].width, contexts[0][1].height, window_xyxy)
            old_report = json.loads((old_dir / "states" / identity[:16] / "reference_report.json").read_text())
            sigma = old_noise(old_report, slices)
            off, on, used_spp, noise = adaptive_difference(contexts, args, old_image, slices, sigma)
            image, differences = combine(np, old_image, off, on)
            stats = {name: difference_statistics(np, old_image, differences, area) for name, area in slices.items()}
            folder.mkdir(parents=True)
            write_pfm(folder / "reference.pfm", image)
            write_pfm(folder / "difference.pfm", np.mean(differences, axis=0))
            elapsed = time.monotonic() - started
            report = {"schema": 1, "frame": data["frame"], "state": identity, "frames": members, "converged": True,
                      "method": "old converged + same-seed (new - old) difference", "spp_per_seed": used_spp,
                      "old_reference_noise_rms": sigma, "noise_over_old_noise": {n: noise[n] / sigma[n] for n in noise},
                      "window_xyxy": list(window_xyxy), "difference_statistics": stats, "render_config": config,
                      "seconds": round(elapsed, 2)}
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            if args.rest_ratio > 0 or args.battery_low is not None:
                battery_pause.rest(elapsed, args.rest_ratio, args.battery_low, args.battery_high, None, args.battery_cap, sys.stdout)
        report = json.loads(report_path.read_text())
        index.append({"state": identity, "frames": members, "converged": True, "spp_per_seed": report["spp_per_seed"],
                      "difference_statistics": report["difference_statistics"]})
        print(json.dumps({"state": number + 1, "of": len(states), "frames": members, "seconds": report["seconds"],
                          "spp_per_seed": report["spp_per_seed"],
                          "noise_over_old_noise": {n: round(v, 4) for n, v in report["noise_over_old_noise"].items()}}), flush=True)
        for frame in members:
            target = out / "linear" / f"frame-{frame:06d}.pfm"
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copyfile(folder / "reference.pfm", target)
    complete = len(index) == len(old["states"])
    manifest = {k: old[k] for k in ("schema", "scene", "base_snapshot", "base_snapshot_sha256", "frames", "model",
                                   "secondary_specular", "regularization_scale", "regions", "window_regions", "window_margin")}
    manifest.update(states=index, converged=complete, method="difference_estimator", correction_spp_checkpoints=list(args.spp), max_noise_ratio=args.max_noise_ratio,
                    correction_spp_per_seed_range=[min(r["spp_per_seed"] for r in index), max(r["spp_per_seed"] for r in index)],
                    old_clip=str(old_dir), old_clip_report_sha256=config["old_clip_report_sha256"], specular_ms=True,
                    mirror_bounces=args.mirror_bounces, tool_sha256=reference.sha(__file__),
                    reference_tool_sha256=reference.sha(reference.__file__), candidate_read=False, phase_accepted=False,
                    note="new = old converged reference + mean over seeds of (render(ms on, mirror chain) - render(ms off, no chain)), "
                         "both with the identical seed streams of f13_reference.converge")
    (out / "clip_report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if complete else 1


def region_error(np, image, truth, area):
    """rms(image - truth) / rms(truth) inside `area`."""
    return float(np.sqrt(np.mean((image[area] - truth[area]) ** 2)) / max(np.sqrt(np.mean(truth[area] ** 2)), 1e-12))


def command_check_correction(args):
    """Accuracy evidence on chosen states: corrected frame vs a full converged new-model render, and the seed-correlation control."""
    old_dir = pathlib.Path(args.old).resolve()
    old = json.loads((old_dir / "clip_report.json").read_text())
    root, base = snapshot(old["base_snapshot"])
    regions, use_window, margin = old["regions"], old.get("window_regions", False), old.get("window_margin", 2)
    args.secondary_specular, args.regularization_scale = old["secondary_specular"], old["regularization_scale"]
    results = []
    for index in args.state_index:
        row = old["states"][index]
        data = state(old["scene"], base, row["frames"][0])
        old_image = read_pfm_array(old_dir / "states" / row["state"][:16] / "reference.pfm")
        contexts = correction_pair(root, data, args, regions, margin, use_window)
        off_context, on_context = contexts[0][1], contexts[1][1]
        np = off_context.np
        slices = reference.region_slices(regions, off_context.width, off_context.height, off_context.window)
        area = slices.get("primary-signal", slices["frame"])
        reference.Context.ms_lobe = True
        full_dir = pathlib.Path(args.workdir) / f"full-{row['state'][:16]}"
        report_file = full_dir / "reference_report.json"
        if not report_file.is_file():
            reference.render_frame(on_context, argparse.Namespace(spp=args.full_spp, spp_per_pass=args.spp_per_pass), full_dir, None, regions)
        full = read_pfm_array(full_dir / "reference.pfm")
        full_report = json.loads(report_file.read_text())
        old_report = json.loads((old_dir / "states" / row["state"][:16] / "reference_report.json").read_text())
        # Independent truth: the same new model with streams that share nothing with the old clip (shift) -- the honest test.
        indep_files = [full_dir / f"independent-{seed}.pfm" for seed in reference.SEEDS]
        if not all(f.is_file() for f in indep_files):
            reference.Context.ms_lobe = True
            for f, picture in zip(indep_files, accumulate(on_context, args.independent_spp, args.spp_per_pass, shift=104729)):
                write_pfm(f, picture)
        indep_seeds = [read_pfm_array(f) for f in indep_files]
        independent = np.mean(indep_seeds, axis=0)
        indep_support, indep_nrmse = reference.seed_agreement(np, indep_seeds[0][area], indep_seeds[1][area])
        entry = {"state_index": index, "frames": row["frames"],
                 "independent": {"spp_per_seed": args.independent_spp, "seed_nrmse": indep_nrmse},
                 "full": {"spp_per_seed": full_report["records"][-1]["spp_per_seed"], "converged": full_report["converged"],
                          "seed_nrmse": full_report["records"][-1]["regions"].get("primary-signal", {}).get("seed_nrmse")},
                 "old_seed_nrmse": old_report["records"][-1]["regions"].get("primary-signal", {}).get("seed_nrmse"),
                 "old_vs_full_nrmse": region_error(np, old_image, full, area),
                 "old_vs_independent_nrmse": region_error(np, old_image, independent, area), "corrections": {}}
        for spp in args.spp_list:
            args.spp = spp  # scalar here: render_difference renders exactly this many
            record = {}
            for label, shift in (("same_seed", 0), ("different_seed", 7919)):
                started = time.monotonic()
                off, on = render_difference(contexts, args, shift)
                corrected, differences = combine(np, old_image, off, on)
                stats = difference_statistics(np, old_image, differences, area)
                measured = region_error(np, corrected, independent, area)
                floor = math.hypot(entry["old_seed_nrmse"], indep_nrmse) / 2.0  # noise of two means-of-two-seeds, relative
                record[label] = {"corrected_vs_full_nrmse": region_error(np, corrected, full, area),
                                 "corrected_vs_independent_nrmse": measured, "independent_noise_floor": floor,
                                 "excess_over_floor": math.sqrt(max(measured ** 2 - floor ** 2, 0.0)),
                                 "difference_noise_rms": stats["difference_noise_rms"], "effect_rms": stats["effect_rms"],
                                 "noise_over_effect": stats["noise_over_effect"], "relative_mean_effect": stats["relative_mean_effect"],
                                 "seconds": round(time.monotonic() - started, 1)}
            entry["corrections"][str(spp)] = record
            print(json.dumps({"state_index": index, "spp": spp, **record}), flush=True)
        results.append(entry)
        pathlib.Path(args.report).write_text(json.dumps({"schema": 1, "old_clip": str(old_dir), "results": results}, indent=2) + "\n")
    return 0


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
    p.add_argument("--regularization-scale", type=float, default=8.0,
                   help="k in alpha' = max(alpha_secondary, k * alpha_camera); default 8, owner decision D5 (engine default)")
    p.add_argument("--mirror-bounces", type=int, default=0,
                   help="F15 ticket 22: extra ideal-mirror hits after the camera's specular lobe (0 = declared F13 model)")
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--specular-ms", choices=["on", "off"], default="on",
                   help="ADR-0007: the Kulla-Conty multiple-scattering lobe of the engine default (on) or the Appendix B lobe alone (off)")
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
    p = sub.add_parser("correct", help="old clip + same-seed (ms on, mirror chain) - (ms off, no chain) difference estimator")
    p.add_argument("old", help="old clip folder (clip_report.json, states/)")
    p.add_argument("--out", required=True)
    p.add_argument("--spp", type=int, nargs="+", default=[256, 512, 1024],
                   help="samples-per-seed checkpoints of the two difference renders; stops at the first that meets --max-noise-ratio")
    p.add_argument("--max-noise-ratio", type=float, default=0.2,
                   help="difference noise (rms) must be at most this fraction of the old reference's own noise, in every region")
    p.add_argument("--spp-per-pass", type=int, default=32)
    p.add_argument("--mirror-bounces", type=int, default=3)
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--first-state", type=int, default=0)
    p.add_argument("--last-state", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rest-ratio", type=float, default=0.0)
    p.add_argument("--battery-low", type=int)
    p.add_argument("--battery-high", type=int)
    p.add_argument("--battery-cap", type=int, default=90)
    p.set_defaults(function=command_correct)
    p = sub.add_parser("check-correction", help="accuracy and seed-correlation evidence for the difference estimator")
    p.add_argument("old")
    p.add_argument("--state-index", type=int, nargs="+", required=True)
    p.add_argument("--spp-list", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--full-spp", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--independent-spp", type=int, default=4096, help="samples per seed of the shifted-stream new-model truth")
    p.add_argument("--spp-per-pass", type=int, default=32)
    p.add_argument("--mirror-bounces", type=int, default=3)
    p.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    p.add_argument("--workdir", required=True)
    p.add_argument("--report", required=True)
    p.set_defaults(function=command_check_correction)
    p = sub.add_parser("refs", help="runner --references file for converged clips")
    p.add_argument("manifest", help="frozen-manifest.json of the plan to compare")
    p.add_argument("--case", action="append", required=True, metavar="NAME=CLIP_DIR")
    p.add_argument("--out", required=True)
    p.set_defaults(function=command_refs)
    args = parser.parse_args(argv)
    if getattr(args, "specular_ms", None):
        reference.Context.ms_lobe = args.specular_ms == "on"
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F13 reference clip failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
