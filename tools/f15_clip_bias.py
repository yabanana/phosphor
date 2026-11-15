#!/usr/bin/env python3
"""F15.1 ticket 13: control-noise-corrected clip metrics for the rough-glass ROIs.

The converged control of the transmission clip is the per-frame mean of N independent 8 spp renders.  It converges as 1/sqrt(N): the rough ROIs
(roughness 0.1 and 0.3) need about 460 / 1,800 seeds before the control's own noise falls to a third of the gates, so a raw judgement against a
128-seed control reads the CONTROL's noise as candidate error.  This module removes that known term instead of buying seeds:

  candidate = mu + b + c      (b = bias of the denoised image, c = its residual noise)
  control   = mu + n          (n = the noise of the N-seed mean, variance sigma^2 / N per pixel)
  E[mean_roi (candidate - control)^2] = mean_roi (b + c)^2 + mean_roi n^2

The control is accumulated in two independent halves A and B of N / 2 seeds, and Var(A - B) = 4 sigma^2 / N = 4 Var(control), so
mean_roi n^2 is estimated, per frame and ROI, as mean_roi (A - B)^2 / 4 (the ROI holds hundreds of samples).  The corrected per-frame RMSE is
sqrt(max(0, RMSE^2 - mean (A - B)^2 / 4)) / scale.  The residual flicker (mean |e_t - e_(t-1)| per sample) is corrected in quadrature (Gaussian
noise: E|x| = sqrt(2/pi) sd, variances add) with the control's own flicker, estimated from the halves as mean |d(A - B)| / 2 / scale.
The thresholds are the unchanged F13.6 gates of the ROI (p95 of the per-frame RMSE, its max, the mean flicker): only the control noise term is
removed.  What the method cannot do is certify a candidate whose error sits below the control noise: the report states the control noise floor
(p95 of the estimated control RMSE) next to every corrected value, and the judge refuses (not judged) when that floor exceeds the gate itself.

Discrimination is established by the negative controls of the runner (raw 1 spp, the reflect-axis and endpoint-off denoisers), which must FAIL the
same corrected gates, and by tools/test_f15_clip_bias.py (synthetic clips of known noise and bias: the corrected value recovers a known bias and is
~0 for an unbiased candidate whose raw value is dominated by the control noise).
"""
from __future__ import annotations

import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from temporal_light_metrics import bounds, clip_files, percentile, read_pfm  # noqa: E402


def roi_series(folder, config):
    """{roi name: float64 array (frames, samples)} of the PFM frames of `folder` on the ROIs of `config` (top-down, RGB)."""
    import numpy as np
    files, _ = clip_files(folder)
    series = {roi["name"]: [] for roi in config["rois"]}
    for path in files:
        image = read_pfm(path)
        pixels = np.frombuffer(image.rgb, dtype=np.float32).reshape(image.height, image.width, 3)
        for roi in config["rois"]:
            x, y, w, h = bounds(roi, image.width, image.height)
            series[roi["name"]].append(pixels[y:y + h, x:x + w, :].astype(np.float64).reshape(-1))
    return {name: np.stack(frames) for name, frames in series.items()}


def corrected_metrics(candidate, control, half_a, half_b, scale):
    """Corrected and raw clip metrics of one ROI.  Arrays are (frames, samples) in the same layout; `scale` is the ROI radiance scale."""
    import numpy as np
    error = candidate - control
    raw_square = (error ** 2).mean(axis=1)
    noise_square = ((half_a - half_b) ** 2).mean(axis=1) / 4.0
    corrected = np.sqrt(np.maximum(raw_square - noise_square, 0.0)) / scale
    raw = np.sqrt(raw_square) / scale
    floor = np.sqrt(noise_square) / scale
    flicker_raw = float(np.abs(np.diff(error, axis=0)).mean()) / scale
    flicker_control = float(np.abs(np.diff(half_a - half_b, axis=0)).mean()) / 2.0 / scale
    flicker = math.sqrt(max(flicker_raw ** 2 - flicker_control ** 2, 0.0))
    return {"p95": float(percentile(list(corrected), 0.95)), "max": float(corrected.max()), "flicker": flicker,
            "raw_p95": percentile(list(raw), 0.95), "raw_max": float(raw.max()), "raw_flicker": flicker_raw,
            "control_floor_p95": percentile(list(floor), 0.95), "control_floor_max": float(floor.max()), "control_flicker": flicker_control}


def judge(metrics, thresholds):
    """Verdict on the unchanged gates.  `judgeable` is False when the control noise floor alone is above a gate (nothing can be said)."""
    checks = {"spatial_p95": bool(metrics["p95"] <= thresholds["linear_rmse_p95_max"]), "worst_frame": bool(metrics["max"] <= thresholds["linear_rmse_max"]),
              "flicker": bool(metrics["flicker"] <= thresholds["residual_flicker_mean_max"])}
    judgeable = bool(metrics["control_floor_p95"] <= thresholds["linear_rmse_p95_max"] and metrics["control_floor_max"] <= thresholds["linear_rmse_max"]
                 and metrics["control_flicker"] <= thresholds["residual_flicker_mean_max"])
    return {"passed": bool(all(checks.values())), "checks": checks, "judgeable": judgeable}


def analyse(control_dir, half_a_dir, half_b_dir, candidate_dir, config):
    """{roi: {metrics, verdict}} of a candidate clip against the converged control, corrected for the control noise."""
    control, half_a, half_b, candidate = (roi_series(d, config) for d in (control_dir, half_a_dir, half_b_dir, candidate_dir))
    result = {}
    for roi in config["rois"]:
        name = roi["name"]
        metrics = corrected_metrics(candidate[name], control[name], half_a[name], half_b[name], roi["scale"])
        result[name] = {"metrics": metrics, "verdict": judge(metrics, roi["thresholds"])}
    return result
