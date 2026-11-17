#!/usr/bin/env python3
"""F15.1 (ticket 17/18, RT tier): moving-camera clip of the walk's temporal accumulation against a converged control.

The product output of `--subsurface-mode rt-walk` goes through a temporal accumulation (reprojection with the engine's motion vectors, rejection by depth and
material) and an edge-aware spatial prefilter (shaders/subsurface_accumulate.metal).  A static-camera gate cannot see a ghost: this clip slides the camera
+-1.5 m sideways (`--material-motion`, a continuous strafe looking at the same target, no cut; about 3 pixels per frame at 1280x720) over the side-lit spheres of `subsurface-volumetric-sun` and judges the product output frame by
frame against the CONVERGED CONTROL: the per-frame mean of K = 96 independent runs (different `--lighting-seed`, each 64 walks per pixel, accumulation OFF), pooled
over a 9 x 9 window (the product's own prefilter size) (the control's own noise).  Metrics, declared before any run: relative luminance error per ROI pixel |Y - Y_ctl| / max(Y_ctl, 5 % of the
ROI mean) on the pixels of the lit spheres (Y_ctl > 0.08, eroded by 6 px), per frame: the mean and the p95; GATES (the bound of the side-lit key, never loosened):
every frame after the warm-up (frame >= 24) has mean <= 0.10 and p95 <= 0.25.  The control must itself be converged (halves A / B judged against each other:
mean <= 1/3 of the gate, F13 CONTROL_FRACTION).  Ghost: the p95 over the frames of the error of the product against a history that follows the camera is
reported as `ghost_ratio` = worst-frame mean error / median-frame mean error.  NEGATIVE CONTROL: `--debug-subsurface-history reject-never` (the history is read at
the same pixel with no motion and no depth rejection) must FAIL the gates.

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_walk_clip_check.py --run --out build/f15-walk-clip
  ... --evaluate --out DIR       (existing captures, no GPU process)

Exit codes: 0 expectations met, 1 an expectation failed, 2 usage, 4 GPU busy, 5 capture failed, 6 control not converged.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_check as fc  # noqa: E402

FRAMES, WARMUP = 96, 24
RESOLUTION = "1280x720"
CONTROL_SEEDS = 128
GATE = {"mean": 0.10, "p95": 0.25, "floor_fraction": 0.05}   # the side-lit bound (decision 11, ticket 17)
CONTROL_FRACTION = 1.0 / 3.0
ROI_THRESHOLD, ROI_ERODE, CONTROL_POOL = 0.08, 6, 9
SCENE = "subsurface-volumetric-sun"


def command(binary, folder, extra, seed):
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", SCENE, "--render-path", "visibility", "--geometry-path", "mesh",
            "--rt", "on", "--shadows", "off", "--lighting", "legacy", "--gi", "off", "--reflections", "off", "--ao", "off", "--lighting-denoise", "custom",
            "--lighting-seed", str(seed), "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(FRAMES), "--warmup", "0",
            "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1", "--pipeline-sync", "--material-motion",
            "--capture-linear-sequence", str(folder / "linear"), "--capture-linear-signal", "hdr", "--report", str(folder / "report.json"), *extra]


CASES = {
    "product": ["--subsurface-mode", "rt-walk"],
    "negative-reject-never": ["--subsurface-mode", "rt-walk", "--debug-subsurface-history", "reject-never"],
}


def read_frames(np, folder):
    from f12_reference import read_pfm
    files = sorted((folder / "linear").glob("frame-*.pfm"))
    return np.stack([read_pfm(f) for f in files])


def box_mean(np, image, window):
    half = window // 2
    padded = np.pad(image, [(half, half), (half, half)] + [(0, 0)] * (image.ndim - 2), mode="edge")
    c = padded.cumsum(axis=0).cumsum(axis=1)
    c = np.pad(c, [(1, 0), (1, 0)] + [(0, 0)] * (image.ndim - 2))
    h, w = image.shape[:2]
    return (c[window:window + h, window:window + w] - c[:h, window:window + w] - c[window:window + h, :w] + c[:h, :w]) / (window * window)


def luminance(np, frames):
    return frames @ np.asarray(fc.LUMA_WEIGHTS, dtype=np.float32)


def erode(np, mask, pixels):
    for _ in range(pixels):
        step = mask.copy()
        step[1:, :] &= mask[:-1, :]; step[:-1, :] &= mask[1:, :]; step[:, 1:] &= mask[:, :-1]; step[:, :-1] &= mask[:, 1:]
        mask = step
    return mask


def frame_errors(np, candidate_y, control_y):
    """Per frame (>= WARMUP) the mean and p95 relative luminance error on the ROI of the control (declared metric)."""
    rows = []
    for f in range(WARMUP, len(candidate_y)):
        roi = erode(np, control_y[f] > ROI_THRESHOLD, ROI_ERODE)
        ref = control_y[f][roi]
        floor = GATE["floor_fraction"] * float(ref.mean())
        error = np.abs(candidate_y[f][roi] - ref) / np.maximum(np.maximum(ref, floor), 1e-12)
        rows.append({"frame": f, "pixels": int(roi.sum()), "mean": float(error.mean()), "p95": float(np.percentile(error, 95))})
    return rows


def judge(rows):
    worst_mean, worst_p95 = max(r["mean"] for r in rows), max(r["p95"] for r in rows)
    means = sorted(r["mean"] for r in rows)
    median = means[len(means) // 2]
    return {"worst_mean": worst_mean, "worst_p95": worst_p95, "median_mean": median, "ghost_ratio": worst_mean / max(median, 1e-9),
            "pass": bool(worst_mean <= GATE["mean"] and worst_p95 <= GATE["p95"])}


def evaluate(out):
    import numpy as np
    halves = [np.load(out / f"control-{h}.npy") for h in "ab"]
    control = 0.5 * (halves[0] + halves[1])
    pooled = lambda frames: np.stack([box_mean(np, f, CONTROL_POOL) for f in frames])  # noqa: E731
    control_y = luminance(np, pooled(control))
    a_y, b_y = luminance(np, pooled(halves[0])), luminance(np, pooled(halves[1]))
    self_test = judge(frame_errors(np, a_y, b_y))
    converged = self_test["worst_mean"] <= CONTROL_FRACTION * GATE["mean"] and self_test["worst_p95"] <= CONTROL_FRACTION * GATE["p95"]
    result = {"control_self_test": {**self_test, "converged": converged, "limits": {"mean": CONTROL_FRACTION * GATE["mean"], "p95": CONTROL_FRACTION * GATE["p95"]}}}
    if not converged:
        result["status"] = "CONTROL_NOT_CONVERGED"
        return result, 6
    ok = True
    for name in CASES:
        frames = read_frames(np, out / name)
        verdict = judge(frame_errors(np, luminance(np, frames), control_y))
        expected_pass = not name.startswith("negative")
        verdict["expected_pass"] = expected_pass
        verdict["met"] = verdict["pass"] == expected_pass
        ok &= verdict["met"]
        result[name] = verdict
    result["status"] = "MET" if ok else "NOT_MET"
    return result, 0 if ok else 1


def run(args):
    import numpy as np
    out = args.out.resolve()
    binary = args.binary.resolve()
    if args.run:
        reuse = (out / "control-a.npy").exists() and (out / "control-b.npy").exists()  # --run again with the converged control already captured: only the candidates
        if out.exists() and any(out.iterdir()) and not reuse:
            print(f"{out} is not empty", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        for name in CASES:
            shutil.rmtree(out / name, ignore_errors=True)
        try:
            with fc.gpu_lock(args.gpu_lock):
                sums = {"a": None, "b": None}
                counts = {"a": 0, "b": 0}
                for k in range(0 if reuse else CONTROL_SEEDS):
                    half = "ab"[k % 2]
                    folder = out / f"control-{k}"
                    status = fc.capture(command(binary, folder, ["--subsurface-mode", "rt-walk", "--subsurface-accumulate", "off", "--subsurface-samples", "64"], 1000 + k),
                                        folder / "engine.log", args.timeout)
                    if not status["passed"]:
                        print(f"control {k}: capture failed: {status['failures']}", file=sys.stderr)
                        return 5
                    fc.require_report_walk(folder) if hasattr(fc, "require_report_walk") else None
                    frames = read_frames(np, folder).astype(np.float64)
                    sums[half] = frames if sums[half] is None else sums[half] + frames
                    counts[half] += 1
                    shutil.rmtree(folder)
                for half in "ab" if not reuse else "":
                    np.save(out / f"control-{half}.npy", (sums[half] / counts[half]).astype(np.float32))
                for name, extra in CASES.items():
                    folder = out / name
                    status = fc.capture(command(binary, folder, extra, 1), folder / "engine.log", args.timeout)
                    if not status["passed"]:
                        print(f"{name}: capture failed: {status['failures']}", file=sys.stderr)
                        return 5
                    report = json.loads((folder / "report.json").read_text()).get("subsurface") or {}
                    if report.get("effective") != "rt-walk":
                        print(f"{name}: subsurface.effective is {report.get('effective')!r} (reason {report.get('fallback_reason')!r}): capture error", file=sys.stderr)
                        return 5
        except fc.GpuBusy as error:
            print(f"GPU busy: {error}", file=sys.stderr)
            return 4
    result, code = evaluate(out)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-walk-clip")
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    parser.add_argument("--timeout", type=float, default=900)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
