#!/usr/bin/env python3
"""Independent F13.2 ambient-occlusion reference clip for a static fixture.

Visibility V(x) = integral over the hemisphere of (cos/pi) * [no surface closer
than the world radius], the engine's RTAO contract (shaders/ao.metal): rays
start at the receiver point, offset by `bias` along its geometric normal, with
cosine-distributed directions about it; any intersection before the radius
occludes. Receivers are the engine's exported points and geometric normals
(the quantities RTAO consumes); Mitsuba 3 (llvm_ad_rgb) independently traces
the occlusion rays through a scene built from the same frame's snapshot. The
exported points are also checked against Mitsuba's own primary hits and the
mismatch fraction is capped. Two seeds
converge to the shared criterion of tools/f13_reference.py. The scene must be
static: the one converged image is written for every requested frame, in the
clip layout tools/f13_reference_clip.py `refs` binds to a frozen manifest.
No renderer image is read.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import snapshot, write_pfm  # noqa: E402
import f13_reference as reference  # noqa: E402


def visibility_pass(context, receivers, seed, spp, radius, bias):
    """Per-pixel SUM of `spp` visibility samples (float64, height x width)."""
    mi, dr, np = context.mi, context.dr, context.np
    points, normals, valid_np = receivers
    pixels = context.width * context.height
    sampler = mi.load_dict({"type": "independent"})
    sampler.seed(seed, pixels * spp)
    pixel = dr.repeat(dr.arange(mi.UInt32, pixels), spp)
    p = mi.Point3f(*(mi.Float(points[:, i].copy()) for i in range(3)))
    n = mi.Normal3f(*(mi.Float(normals[:, i].copy()) for i in range(3)))
    valid = dr.gather(mi.Bool, mi.Bool(valid_np), pixel)
    p, n = dr.gather(mi.Point3f, p, pixel), dr.gather(mi.Normal3f, n, pixel)
    frame = mi.Frame3f(n)
    direction = frame.to_world(mi.warp.square_to_cosine_hemisphere(sampler.next_2d()))
    shadow = mi.Ray3f(p + n * bias, direction)
    shadow.maxt = radius
    occluded = context.scene.ray_test(shadow, valid)
    value = dr.select(valid, dr.select(occluded, 0.0, 1.0), 0.0)  # no receiver: 0, the engine's AO signal convention
    out = dr.zeros(mi.Float64, pixels)
    dr.scatter_add(out, mi.Float64(value), pixel)
    return np.asarray(out, dtype=np.float64).reshape(context.height, context.width)


def converge(context, receivers, checkpoints, spp_per_pass, radius, bias, regions, log=print):
    np = context.np
    sums = {seed: np.zeros((context.height, context.width)) for seed in reference.SEEDS}
    done, records = 0, []
    slices = reference.region_slices(regions, context.width, context.height)
    live = receivers[2].reshape(context.height, context.width).astype(np.float64)
    for target in checkpoints:
        start = time.monotonic()
        while done < target:
            step = min(spp_per_pass, target - done)
            for seed in reference.SEEDS:
                sums[seed] += visibility_pass(context, receivers, (seed * 1000003 + done) & 0x7FFFFFFF, step, radius, bias)
            done += step
        images = [sums[seed] / done for seed in reference.SEEDS]
        # Occlusion (1 - V) on receivers is the signal that has to agree; V is near 1 in open areas.
        agreement = {name: dict(zip(("support", "seed_nrmse"),
                                    reference.seed_agreement(np, (1 - images[0][area]) * live[area],
                                                             (1 - images[1][area]) * live[area])))
                     for name, area in slices.items()}
        converged = all(v["support"] > 1e-6 and v["seed_nrmse"] <= reference.CONVERGENCE for v in agreement.values())
        records.append({"spp_per_seed": done, "regions": agreement, "converged": converged,
                        "seconds": round(time.monotonic() - start, 2)})
        log(json.dumps(records[-1]), flush=True)
        if converged:
            break
    return np.mean(images, axis=0), records


def command_render(args):
    root, data = snapshot(args.snapshot)
    context = reference.Context(root, data, args.threads, "regularized")
    np = context.np
    out = pathlib.Path(args.out).resolve()
    (out / "linear").mkdir(parents=True, exist_ok=False)
    positions = reference.validate_positions(np, context, args.positions)
    mismatch = positions["coverage_mismatch"] + positions["over_tolerance"]
    positions["mismatch_fraction"] = mismatch / positions["pixels"]
    positions["mismatch_fraction_cap"] = 1e-3
    if positions["mismatch_fraction"] > 1e-3:
        raise ValueError(f"engine receivers disagree with Mitsuba primary hits: {positions}")
    point = reference.read_pfm(args.positions).astype(np.float64).reshape(-1, 3)
    normal = reference.read_pfm(args.normals).astype(np.float64).reshape(-1, 3)
    length = np.linalg.norm(normal, axis=1)
    valid = (np.abs(point).sum(axis=1) > 0) & (length > 0.5)
    normal[valid] /= length[valid, None]
    normal[~valid] = (0, 0, 1)
    regions = json.loads(pathlib.Path(args.regions).read_text()) if args.regions else None
    image, records = converge(context, (point.astype(np.float32), normal.astype(np.float32), valid), args.spp,
                              args.spp_per_pass, args.radius, args.bias, regions)
    rgb = np.repeat(image[:, :, None], 3, axis=2).astype(np.float32)
    write_pfm(out / "reference.pfm", rgb)
    for frame in range(args.first, args.last + 1):
        write_pfm(out / "linear" / f"frame-{frame:06d}.pfm", rgb)
    report = {"schema": 1, "kind": "f13-ao-reference", "scene_snapshot": str(pathlib.Path(args.snapshot).resolve()),
              "snapshot_sha256": reference.scene_digest(root), "frames": [args.first, args.last],
              "states": [{"frames": [args.first, args.last], "converged": records[-1]["converged"],
                          "spp_per_seed": records[-1]["spp_per_seed"]}],
              "converged": records[-1]["converged"], "records": records, "positions": positions,
              "model": "cosine-weighted binary visibility within a world radius (RTAO contract)",
              "radius_m": args.radius, "origin_bias_m": args.bias, "regions": regions or {},
              "tool_sha256": reference.sha(__file__), "reference_tool_sha256": reference.sha(reference.__file__),
              "static_scene_required": True, "candidate_read": False, "phase_accepted": False}
    (out / "clip_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["converged"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", help="engine --export-reference snapshot of the static fixture")
    parser.add_argument("--positions", required=True, help="engine shadow-position PFM of the same frame")
    parser.add_argument("--normals", required=True, help="engine shadow-normal PFM (geometric normals) of the same frame")
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, default=1.0, help="world radius, metres (runner: --ao-radius 1.0)")
    parser.add_argument("--bias", type=float, default=0.001, help="origin offset along the normal (AOSettings)")
    parser.add_argument("--regions")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=299)
    parser.add_argument("--spp", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    parser.add_argument("--spp-per-pass", type=int, default=64)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if any(n < 1 for n in args.spp) or args.spp != sorted(set(args.spp)) or not args.radius > 0 or args.bias < 0:
        parser.error("strictly increasing positive SPP, positive radius, non-negative bias required")
    try:
        return command_render(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F13 AO reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
