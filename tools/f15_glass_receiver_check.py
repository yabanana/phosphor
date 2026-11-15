#!/usr/bin/env python3
"""F15 closure (tickets 13 and 15): two DECLARED checks of glass that is itself lit by a delta light.  Every threshold below is fixed from an
independent criterion BEFORE any comparison (decision 11: transmission bound, mean relative error <= 5 %, p95 <= 15 %), never loosened.

1. BTDF of thin-walled glass under delta lights (ticket 13).  Scenes `glass-btdf-sun` and `glass-btdf-point` (bench class
   GlassReceiverValidation): seven vertical thin-walled panes facing the camera (columns roughness {0.1, 0.3, 0.6}, row 0 white, row 1 tinted
   (0.4, 0.9, 0.45), a seventh white pane with transmission factor 0.5), ONE delta light BEHIND them, environment 0, nothing else lit: a pane
   pixel is the transmitted lobe alone,
       L = (1 - F(V.H_T)) t (1 - metallic) tint  D(N.H_T) G1(|N.L|) G1(N.V) / (4 |N.L| N.V)  |N.L|  E_light,
   H_T = normalize(V + L') with L' the light mirrored through the surface (the glTF README thin-walled specular_btdf), separable Smith G, Schlick F
   of the dielectric (ior 1.5: f0 = 0.04) at V.H_T, alpha = roughness^2.  The expectation is evaluated here per pixel (numpy, from those formulas,
   the pixel-centre camera ray of the capture, the sun averaged over its disk with 64 points, the point light with its inverse-square window).
   Gate per pane: mean and p95 of the per-pixel relative error |engine - expected| / max(expected, 0.05 mean(expected)) over the pane region
   (pixels whose centre ray hits the inner 0.8 of the pane and whose expected value is at least 5 % of the pane maximum).  Negative controls
   (each must FAIL the same gate on the same capture): tint ignored, transmission factor ignored, light not mirrored, no transmitted lobe.
2. Glass receiver behind glass under CSM (ticket 15).  Scenes `glass-receiver-stack` (a tinted-clear pane A above a rough tinted pane B; the sun
   reflection on B is shadowed by A) and `glass-receiver-single` (B alone, the unshadowed baseline), captured with `--shadows csm` and with
   `--shadows rt`.  The ratio R_c = sum(stack_c) / sum(single_c) over the B region (pixels whose centre ray hits the inner 0.8 of B and
   whose baseline channel is at least 20 % of its region mean) must equal the transmittance of A, T_A = (1 - F_Schlick(cos)) tint_A
   (thin-walled, one surface, cos = sun elevation sine, tint_A = (0.6, 0.8, 0.9): the tint is what makes the no-attenuation control fail): gate |R_c / T_A - 1| <= 5 % per channel and p95 of the per-pixel |ratio / T_A - 1| <= 15 %.
   Negative controls (the same data, wrong expectations; each must FAIL): the receiver's own surface counted again (T_A (1 - F_B) tint_B)
   and no attenuation at all (1).  RT is the independent truth: the RT pair must pass with the same bound.

3. Edge quality of the half-resolution transmittance map (ticket 15).  Scene `transmissive-shadow-pane-sun` (the thin GREEN pane, 0.9 m, at
   x = -0.7, height 0.8 m, sun 25 degrees from the vertical towards -Z: its shadow on the floor is x in [-1.15, -0.25], z in [-0.077, 0.823])
   captured with `--shadows csm` (default divisor 2) and `--shadows rt` (the ground truth of ticket 14).  The RED channel steps from the lit
   floor (1) to 0.4 (1 - F) in the shadow; per image row (resp. column) the sub-pixel position of the mid-level crossing of each vertical
   (resp. horizontal) shadow edge is measured on both images and mapped to world metres by the floor-plane ray of the pixel.  Bound derived
   BEFORE any measurement from the quantisation of the map: a receiver reads the nearest texel, so the position of a straight edge is
   displaced by a constant |e| <= 0.5 texel along it (the same for every row) and the PCSS disk is symmetric (it does not move the mid-level).
   Gate: mean |position difference| <= 0.5 texel and p95 <= 1 texel + half a pixel footprint, one texel = the covering cascade's texel x the
   transmittance divisor.  FIRST RUN (2026-11-15): the texel was taken from the ticket-15 note "cascade radius <= 9 m" (17.6 mm at divisor 2)
   and the gate failed (mean 15.1 / 4.7 / 1.9 / 13.7 mm on the four edges); that premise was wrong for this camera: the shadow cascades
   split the view range [0.05, 120] m with lambda 0.6, the floor lies in cascade 0 (view depth 0.05 .. 12.2 m, bounding-sphere radius 15.7 m),
   so the opaque texel is 15.33 mm and the divisor-2 transmittance texel 30.67 mm (`cascade_texel`, the formulas of makeShadowCascades,
   cross-checked against the C++ function by tests/test_glass_receiver_validation.cpp).  The criterion (0.5 texel mean, 1 texel p95) is
   unchanged; only the texel size was corrected, from the engine's own geometry and not from the measurement.  Negative control: the same
   CSM capture shifted by 5 pixels (about 27 mm on the vertical edges) must FAIL; the divisor 4 capture (`edge4@csm`, texel 61.3 mm) is
   reported against its own bound.

4. Glass casters with `--shadow-cache on` (ticket 15): the same scene (the green pane's shadow, `cache@csm`) must transmit like the no-cache
   capture: the ratio of the mean floor colour inside the shadow interior to a lit region, per channel, within 5 % of T = (1 - F_Schlick(cos 25
   degrees)) tint = the thin pane's transmittance (closed form).  Negative controls (same data, wrong expectations: own surface counted twice,
   opaque shadow T = 0) must FAIL; the pre-fix binary (`--before-binary`: transmissive casters stay opaque with the cache) must fail the gate.

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_glass_receiver_check.py --run --out build/f15-check/glass-receiver
  ... --evaluate --out DIR      (existing captures, no GPU process)   [--only btdf|stack|edge|cache]
Exit codes: 0 gates met and controls fail, 1 a gate failed or a control passed, 2 usage, 4 GPU busy, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

RESOLUTION = (1280, 720)
FRAMES = 64
FOV_Y = 60.0
MEAN_BOUND, P95_BOUND = 0.05, 0.15  # decision 11, transmission
REGION_MARGIN = 0.8                # inner fraction of a pane
TINT = (0.4, 0.9, 0.45)
F0 = 0.04                          # ior 1.5
SUN_ANGULAR_RADIUS = 0.00465
SUN_SAMPLES = 64

# ---- scene constants (mirrors src/testbench/glass_receiver_validation.cpp; tests/test_glass_receiver_validation.cpp pins the geometry) ----
BTDF_ROUGHNESS = (0.1, 0.3, 0.6)
BTDF_PANES = [  # (centre, roughness, tint, transmission) of the seven vertical panes facing +Z, unit squares
    ((-1.2, 0.6, 0.0), 0.1, (1.0, 1.0, 1.0), 1.0), ((0.0, 0.6, 0.0), 0.3, (1.0, 1.0, 1.0), 1.0), ((1.2, 0.6, 0.0), 0.6, (1.0, 1.0, 1.0), 1.0),
    ((-1.2, 1.8, 0.0), 0.1, TINT, 1.0), ((0.0, 1.8, 0.0), 0.3, TINT, 1.0), ((1.2, 1.8, 0.0), 0.6, TINT, 1.0),
    ((2.4, 0.6, 0.0), 0.3, (1.0, 1.0, 1.0), 0.5)]
BTDF_CAMERA = {"target": (0.0, 1.2, 0.0), "distance": 5.5, "elevation": 0.0}
BTDF_SUN = {"toward": (0.05, 0.08, -1.0), "irradiance": 3.0}
BTDF_POINT = {"position": (0.4, 1.4, -2.0), "intensity": 60.0, "range": 100.0}
STACK_CAMERA = {"target": (0.0, 0.5, 0.1), "distance": 4.5, "elevation": 38.0}
STACK_A_TINT = (0.6, 0.8, 0.9)                                          # the pane above (clear glass of a visible tint: attenuates every channel by > 5 %)
STACK_B = {"centre": (0.0, 0.5, 0.1), "size": (1.6, 0.6), "tint": TINT}  # horizontal, facing +Y

BTDF_MODES = {  # key: (scenario, lighting, rt, accumulate)
    "btdf-sun@legacy": ("glass-btdf-sun", "legacy", "off", 0),
    "btdf-point@legacy": ("glass-btdf-point", "legacy", "off", 0),
    "btdf-sun@restir": ("glass-btdf-sun", "restir", "on", 8),
    "btdf-point@restir": ("glass-btdf-point", "restir", "on", 8),
    "btdf-point@clustered": ("glass-btdf-point", "clustered", "on", 0),
}
EDGE_MODES = {  # key: (shadows, rt, divisor)
    "edge@csm": ("csm", "off", 2), "edge@rt": ("rt", "on", 2), "edge4@csm": ("csm", "off", 4),
}
EDGE_CAMERA = {"target": (0.0, 0.0, 0.3), "distance": 6.2, "elevation": 38.0}
EDGE_SHADOW = {"x": (-1.15, -0.25), "z": (-0.077, 0.823)}  # the green pane's shadow on the floor (declared geometry, see the docstring)


def cascade_texel(view_depth, divisor=1, near=0.05, fov_deg=60.0, aspect=16 / 9, distance=120.0, lam=0.6, resolution=2048):
    """Texel size (m) of the shadow cascade covering `view_depth` times the transmittance divisor: the formulas of makeShadowCascades
    (src/renderer/shadow_settings.cpp) for the engine camera (near 0.05, fov 60 degrees, 16:9) and the default settings."""
    splits = [near] + [lam * near * (distance / near) ** (i / 4) + (1 - lam) * (near + (distance - near) * i / 4) for i in (1, 2, 3)] + [distance]
    ci = next(i for i in range(4) if splits[i] <= view_depth <= splits[i + 1])
    t = math.tan(math.radians(fov_deg) / 2)
    d0 = splits[ci] if ci == 0 else splits[ci] - (splits[ci] - splits[ci - 1]) * 0.1
    d1 = splits[ci + 1]
    corners = [(sx * t * aspect * d, sy * t * d, d) for d in (d0, d1) for sx in (-1, 1) for sy in (-1, 1)]
    centre = [sum(p[k] for p in corners) / 8 for k in range(3)]
    radius = max(math.dist(p, centre) for p in corners)
    radius = math.ceil(radius * 16) / 16 * resolution / (resolution - 2)
    return 2 * radius / resolution * divisor


EDGE_TEXEL = cascade_texel(6.2, 2)                           # metres: the divisor-2 transmittance texel of the cascade covering the floor (30.67 mm)
EDGE_MEAN_TEXELS, EDGE_P95_TEXELS = 0.5, 1.0
STACK_MODES = {  # key: (scenario, shadows, rt)
    "stack@csm": ("glass-receiver-stack", "csm", "off"), "single@csm": ("glass-receiver-single", "csm", "off"),
    "stack@rt": ("glass-receiver-stack", "rt", "on"), "single@rt": ("glass-receiver-single", "rt", "on"),
    # fail-before evidence: the engine with the receiver's own surface left in the map sum (`--debug-receiver-own off` = the behaviour before the fix)
    "stackold@csm": ("glass-receiver-stack", "csm", "off"),
}
STACK_EXTRA = {"stackold@csm": ["--debug-receiver-own", "off"]}


def camera_rays(np, camera):
    """Pixel-centre rays of the capture (unjittered): the formula of f15_transmission_reference.camera_directions."""
    import math
    width, height = RESOLUTION
    el = math.radians(camera["elevation"])
    target = np.asarray(camera["target"], dtype=np.float64)
    position = target + camera["distance"] * np.array([0.0, math.sin(el), math.cos(el)])
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    scale = height / (2 * math.tan(math.radians(FOV_Y) / 2))
    px = np.arange(width) + 0.5 - width / 2
    py = height / 2 - (np.arange(height) + 0.5)
    x, y = np.meshgrid(px / scale, py / scale)
    d = forward[None, None, :] + right[None, None, :] * x[..., None] + up[None, None, :] * y[..., None]
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return position, d


def plane_hits(np, position, directions, centre, axis_u, axis_v, half_u, half_v, normal, margin):
    """Boolean mask of pixels whose ray hits the rectangle (inner `margin` fraction), the hit points (h, w, 3) and the hit distance."""
    n = np.asarray(normal, dtype=np.float64)
    c = np.asarray(centre, dtype=np.float64)
    denom = directions @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((c - position) @ n) / denom
    p = position[None, None, :] + directions * t[..., None]
    rel = p - c
    inside = (t > 0) & (np.abs(rel @ np.asarray(axis_u)) <= margin * half_u) & (np.abs(rel @ np.asarray(axis_v)) <= margin * half_v)
    return inside, p


# ---------------------------------------------------------------------------- BTDF

def g1(np, c, alpha):
    c = np.abs(c)
    a2 = alpha * alpha
    return 2.0 * c / np.maximum(c + np.sqrt(a2 + (1.0 - a2) * c * c), 1e-12)


def thin_btdf(np, n, v, l, alpha, mirror=True):
    """README thin-walled specular_btdf of the light direction `l` (towards the light, behind the surface): returns f_t |N.L| (1 - F) without
    the transmission factor and the tint, 0 where the lobe does not exist.  `n`, `v`, `l` (..., 3), `alpha` scalar."""
    nl_abs = -np.sum(n * l, axis=-1)
    nv = np.sum(n * v, axis=-1)
    lm = l + 2.0 * nl_abs[..., None] * n if mirror else l
    h = v + lm
    h_len2 = np.sum(h * h, axis=-1)
    h = h / np.sqrt(np.maximum(h_len2, 1e-30))[..., None]
    nh = np.sum(n * h, axis=-1)
    vh = np.sum(v * h, axis=-1)
    ok = (nl_abs > 0) & (nv > 0) & (nh > 0) & (vh > 0) & (h_len2 > 1e-20)
    a2 = alpha * alpha
    d_den = nh * nh * (a2 - 1.0) + 1.0
    big_d = a2 / (np.pi * d_den * d_den)
    f = F0 + (1.0 - F0) * (1.0 - np.clip(vh, 0, 1)) ** 5
    nlc = np.where(ok, nl_abs, 1.0)
    nvc = np.where(ok, nv, 1.0)
    f_t = big_d * g1(np, nlc, alpha) * g1(np, nvc, alpha) / (4.0 * nlc * nvc)
    return np.where(ok, (1.0 - f) * f_t * nl_abs, 0.0)


def sun_disk_directions(np, toward):
    """SUN_SAMPLES directions of the sun disk (equal solid angle, golden-angle spiral)."""
    import math
    t = np.asarray(toward, dtype=np.float64)
    t /= np.linalg.norm(t)
    helper = np.array([0.0, 1.0, 0.0]) if abs(t[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, t)
    right /= np.linalg.norm(right)
    up = np.cross(t, right)
    k = np.arange(SUN_SAMPLES)
    z = 1.0 - (k + 0.5) / SUN_SAMPLES * (1.0 - math.cos(SUN_ANGULAR_RADIUS))
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = 2.399963229728653 * k
    return t[None, :] * z[:, None] + (right[None, :] * np.cos(phi)[:, None] + up[None, :] * np.sin(phi)[:, None]) * s[:, None]


def btdf_expected(np, key, pane_index, position, directions, mask, hits, variant="exact"):
    """Expected pixel radiance (n, 3) of the masked pixels of one pane under the light of `key`."""
    centre, roughness, tint, transmission = BTDF_PANES[pane_index]
    alpha = roughness * roughness
    v = -directions[mask]
    n = np.broadcast_to(np.array([0.0, 0.0, 1.0]), v.shape)
    scene = BTDF_MODES[key][0]
    if scene == "glass-btdf-sun":
        total = np.zeros(v.shape[0])
        for l in sun_disk_directions(np, BTDF_SUN["toward"]):
            total += thin_btdf(np, n, v, np.broadcast_to(l, v.shape), alpha, mirror=variant != "not-mirrored")
        scalar = total / SUN_SAMPLES * BTDF_SUN["irradiance"]
    else:
        to_light = np.asarray(BTDF_POINT["position"]) - hits[mask]
        dist = np.linalg.norm(to_light, axis=1)
        l = to_light / dist[:, None]
        r = dist / BTDF_POINT["range"]
        window = np.clip(1.0 - r ** 4, 0.0, 1.0)
        scalar = thin_btdf(np, n, v, l, alpha, mirror=variant != "not-mirrored") * BTDF_POINT["intensity"] * window * window / np.maximum(dist * dist, 1e-4)
    factor = 1.0 if variant == "factor-ignored" else transmission
    colour = np.ones(3) if variant == "tint-ignored" else np.asarray(tint)
    if variant == "no-lobe":
        scalar = np.zeros_like(scalar)
    return scalar[:, None] * factor * colour[None, :]


def relative_stats(np, got, want, floor_from=None):
    """Mean and p95 of the per-pixel relative error, the floor 0.05 x the mean of the expectation (max over RGB per pixel); `floor_from`
    = the expectation that fixes the floor when `want` is a control (a wrong expectation must not define its own floor)."""
    floor = 0.05 * float(np.mean(want if floor_from is None else floor_from))
    err = np.max(np.abs(got - want) / np.maximum(want, floor), axis=1)
    return float(np.mean(err)), float(np.percentile(err, 95))


def evaluate_btdf(np, out, keys):
    from f12_reference import read_pfm
    position, directions = camera_rays(np, BTDF_CAMERA)
    results = {}
    for key in keys:
        image = read_pfm(out / key / "engine.pfm")
        entry = {"panes": [], "passed": True}
        for i, (centre, roughness, tint, transmission) in enumerate(BTDF_PANES):
            inside, hits = plane_hits(np, position, directions, centre, (1, 0, 0), (0, 1, 0), 0.5, 0.5, (0, 0, 1), REGION_MARGIN)
            want_all = btdf_expected(np, key, i, position, directions, inside, hits)
            keep = np.max(want_all, axis=1) >= 0.05 * np.max(want_all)
            mask = inside.copy()
            mask[inside] = keep
            want = want_all[keep]
            got = image[mask]
            mean_err, p95_err = relative_stats(np, got, want)
            controls = {}
            for variant in ("tint-ignored", "factor-ignored", "not-mirrored", "no-lobe"):
                if variant == "tint-ignored" and tint == (1.0, 1.0, 1.0):
                    continue  # identical expectations: cannot fail
                if variant == "factor-ignored" and transmission == 1.0:
                    continue
                other = btdf_expected(np, key, i, position, directions, mask, hits, variant)
                m, p = relative_stats(np, got, other, floor_from=want)
                controls[variant] = {"mean": round(m, 4), "p95": round(p, 4), "fails": bool(m > MEAN_BOUND or p > P95_BOUND)}
            ok = mean_err <= MEAN_BOUND and p95_err <= P95_BOUND
            entry["panes"].append({"pane": i, "roughness": roughness, "pixels": int(mask.sum()), "expected_mean": [round(float(v), 5) for v in want.mean(axis=0)],
                                   "engine_mean": [round(float(v), 5) for v in got.mean(axis=0)], "mean_error": round(mean_err, 4), "p95_error": round(p95_err, 4),
                                   "pass": bool(ok), "controls": controls})
            entry["passed"] &= bool(ok) and all(c["fails"] for c in controls.values())
        results[key] = entry
    return results


# ---------------------------------------------------------------------------- stack

def schlick_pass(cosine):
    return 1.0 - (F0 + (1.0 - F0) * (1.0 - cosine) ** 5)


def evaluate_stack(np, out, shadows):
    import math
    from f12_reference import read_pfm
    position, directions = camera_rays(np, STACK_CAMERA)
    centre, size = STACK_B["centre"], STACK_B["size"]
    inside, _ = plane_hits(np, position, directions, centre, (1, 0, 0), (0, 0, 1), 0.5 * size[0], 0.5 * size[1], (0, 1, 0), REGION_MARGIN)
    cos_sun = math.sin(math.radians(STACK_CAMERA["elevation"]))  # the sun is the mirror of the camera: its angle to the +Y normal
    t_a = schlick_pass(cos_sun) * np.asarray(STACK_A_TINT)  # per channel: (1 - F) tint of the one surface of A
    tint = np.asarray(STACK_B["tint"])
    results = {}
    for mode in shadows:
        before = mode == "csm-before"  # the same scene with the receiver's own surface left in the sum: the gate must FAIL
        stack = read_pfm(out / ("stackold@csm" if before else f"stack@{mode}") / "engine.pfm")
        single = read_pfm(out / f"single@{'csm' if before else mode}" / "engine.pfm")
        base = single[inside]
        region = np.all(base >= 0.2 * base.mean(axis=0)[None, :], axis=1)
        mask = inside.copy()
        mask[inside] = region
        s_sum, b_sum = stack[mask].sum(axis=0), single[mask].sum(axis=0)
        ratio_channels = s_sum / np.maximum(b_sum, 1e-12)
        per_pixel = stack[mask] / np.maximum(single[mask], 1e-9)
        expectations = {"transmittance-of-A": t_a, "own-counted": t_a * schlick_pass(cos_sun) * tint, "no-attenuation": np.ones(3)}
        table = {}
        for name, want in expectations.items():
            mean_err = np.abs(ratio_channels / want - 1.0)
            p95 = np.percentile(np.abs(per_pixel / want[None, :] - 1.0), 95, axis=0)
            fails = bool(np.any(mean_err > MEAN_BOUND) or np.any(p95 > P95_BOUND))
            table[name] = {"expected": [round(float(v), 4) for v in want], "mean_error": [round(float(v), 4) for v in mean_err],
                           "p95_error": [round(float(v), 4) for v in p95], "fails": fails}
        gate = not table["transmittance-of-A"]["fails"]
        controls = all(table[k]["fails"] for k in ("own-counted", "no-attenuation"))
        if before:  # fail-before: the gate fails and the engine matches the own-counted expectation (the cause)
            gate, controls = (not gate), not table["own-counted"]["fails"]
        results[mode] = {"pixels": int(mask.sum()), "baseline_mean": [round(float(v), 5) for v in single[mask].mean(axis=0)],
                         "ratio": [round(float(v), 4) for v in ratio_channels], "t_a": [round(float(v), 4) for v in t_a], "expectations": table,
                         "gate_pass": gate, "controls_fail": controls, "passed": bool(gate and controls)}
    return results


# ---------------------------------------------------------------------------- edge quality

def floor_points(np, position, directions):
    """World (x, z) of the floor plane y = 0 under every pixel centre ray (NaN above the horizon)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -position[1] / directions[..., 1]
    p = position[None, None, :] + directions * t[..., None]
    p = np.where((t > 0)[..., None], p, np.nan)
    return p[..., 0], p[..., 2]


def crossing(np, profile, coord, level):
    """Sub-pixel (interpolated) world coordinate where `profile` crosses `level`, closest to the window centre; NaN when it does not."""
    s = np.sign(profile - level)
    idx = np.nonzero(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return float("nan")
    centre = len(profile) / 2.0
    k = idx[np.argmin(np.abs(idx + 0.5 - centre))]
    f = (level - profile[k]) / (profile[k + 1] - profile[k])
    return float(coord[k] + f * (coord[k + 1] - coord[k]))


def edge_contours(np, image, wx, wz, window=0.12, margin=0.12):
    """Mid-level crossings of the RED channel across the four edges of the green pane's floor shadow: {edge: {line index: (world position,
    footprint)}}.  x0 / x1 (the vertical edges): one crossing per image row, positions along x, rows whose whole z is inside the shadow extent
    minus `margin`; z0 / z1 (the horizontal edges): one per image column, positions along z.  Lit side first for x0 / z0 (the shadow is
    entered), last for x1 / z1."""
    red = image[..., 0]
    height, width = red.shape
    out = {"x0": {}, "x1": {}, "z0": {}, "z1": {}}
    zlo, zhi = EDGE_SHADOW["z"]
    xlo, xhi = EDGE_SHADOW["x"]
    for name, edge in (("x0", xlo), ("x1", xhi)):
        for r in range(height):
            if not (np.isfinite(wz[r, 0]) and zlo + margin < wz[r, 0] < zhi - margin):
                continue
            cols = np.nonzero(np.abs(wx[r] - edge) <= window)[0]
            if len(cols) < 8:
                continue
            a, b = cols.min(), cols.max() + 1
            profile, coord = red[r, a:b + 1], wx[r, a:b + 1]
            lit, shade = (profile[:3].mean(), profile[-3:].mean()) if name == "x0" else (profile[-3:].mean(), profile[:3].mean())
            if not lit - shade > 0.2 * max(lit, 1e-6):
                continue
            c = crossing(np, profile, coord, 0.5 * (lit + shade))
            if np.isfinite(c):
                out[name][r] = (c, abs(float(wx[r, a + 1] - wx[r, a])))
    for name, edge in (("z0", zlo), ("z1", zhi)):
        for cidx in range(width):
            if not (np.isfinite(wx[0, cidx]) and xlo + margin < wx[height // 2, cidx] < xhi - margin):
                continue
            rows = np.nonzero(np.abs(wz[:, cidx] - edge) <= window)[0]
            if len(rows) < 8:
                continue
            a, b = rows.min(), rows.max() + 1
            profile, coord = red[a:b + 1, cidx], wz[a:b + 1, cidx]
            # image rows grow towards the camera (z grows): z0 (the far edge) has the lit floor first, z1 the shadow first
            lit, shade = (profile[:3].mean(), profile[-3:].mean()) if name == "z0" else (profile[-3:].mean(), profile[:3].mean())
            if not lit - shade > 0.2 * max(lit, 1e-6):
                continue
            c = crossing(np, profile, coord, 0.5 * (lit + shade))
            if np.isfinite(c):
                out[name][cidx] = (c, abs(float(wz[a + 1, cidx] - wz[a, cidx])))
    return out


def evaluate_edges(np, out):
    from f12_reference import read_pfm
    position, directions = camera_rays(np, EDGE_CAMERA)
    wx, wz = floor_points(np, position, directions)
    truth = edge_contours(np, read_pfm(out / "edge@rt" / "engine.pfm"), wx, wz)
    images = {"csm": read_pfm(out / "edge@csm" / "engine.pfm")}
    images["control-shift-5px"] = np.roll(images["csm"], 5, axis=1)
    if (out / "edge4@csm" / "engine.pfm").exists():
        images["divisor-4"] = read_pfm(out / "edge4@csm" / "engine.pfm")
    result = {"texel_m": EDGE_TEXEL, "bounds_texels": {"mean": EDGE_MEAN_TEXELS, "p95": EDGE_P95_TEXELS}, "runs": {}}
    for name, image in images.items():
        got = edge_contours(np, image, wx, wz)
        table = {}
        texel = cascade_texel(6.2, 4 if name == "divisor-4" else 2)
        for edge in ("x0", "x1", "z0", "z1"):
            shared = sorted(set(truth[edge]) & set(got[edge]))
            if len(shared) < 20:
                table[edge] = {"samples": len(shared), "pass": False, "reason": "too few contour samples"}
                continue
            d = np.array([abs(got[edge][k][0] - truth[edge][k][0]) for k in shared])
            half_pixel = 0.5 * float(np.mean([truth[edge][k][1] for k in shared]))
            mean_limit, p95_limit = EDGE_MEAN_TEXELS * texel, EDGE_P95_TEXELS * texel + half_pixel
            table[edge] = {"samples": len(shared), "mean_mm": round(float(d.mean()) * 1000, 2), "p95_mm": round(float(np.percentile(d, 95)) * 1000, 2),
                           "mean_limit_mm": round(mean_limit * 1000, 2), "p95_limit_mm": round(p95_limit * 1000, 2),
                           "pass": bool(d.mean() <= mean_limit and np.percentile(d, 95) <= p95_limit)}
        result["runs"][name] = {"edges": table, "all_pass": all(t["pass"] for t in table.values())}
    # the shift control moves the vertical edges only (x); the horizontal ones are insensitive to it and listed, not required to fail
    shift = result["runs"]["control-shift-5px"]["edges"]
    result["gate_pass"] = result["runs"]["csm"]["all_pass"]
    result["control_fails"] = not shift["x0"]["pass"] and not shift["x1"]["pass"]
    result["passed"] = bool(result["gate_pass"] and result["control_fails"])
    return result


# ---------------------------------------------------------------------------- shadow cache + glass

CACHE_REGION = {"shadow": ((-1.05, -0.35), (0.05, 0.70)), "lit": ((-1.05, -0.35), (-0.65, -0.20))}  # world (x range, z range) on the floor
CACHE_MODES = {  # key: (shadows, rt, extra arguments, uses the pre-fix binary)
    "cache@csm": ("csm", "off", ["--shadow-cache", "on"], False),
    "cachebefore@csm": ("csm", "off", ["--shadow-cache", "on"], True),
}


def evaluate_cache(np, out):
    """The transmittance of the green pane's shadow with `--shadow-cache on`: the ratio of the mean floor colour inside its shadow to the mean
    in a lit region of the same floor, per channel, against T = (1 - F_Schlick(cos 25 degrees)) x tint (thin-walled, one surface); the no-cache
    capture (`edge@csm`) must give the same; the pre-fix binary (transmissive casters opaque with the cache) must fail the gate (T = 0)."""
    from f12_reference import read_pfm
    position, directions = camera_rays(np, EDGE_CAMERA)
    wx, wz = floor_points(np, position, directions)

    def region(box):
        (x0, x1), (z0, z1) = box
        return np.isfinite(wx) & (wx > x0) & (wx < x1) & (wz > z0) & (wz < z1)

    shadow, lit = region(CACHE_REGION["shadow"]), region(CACHE_REGION["lit"])
    cos_sun = math.cos(math.radians(25.0))
    expected = schlick_pass(cos_sun) * np.asarray(TINT)
    result = {"expected_T": [round(float(v), 4) for v in expected], "shadow_pixels": int(shadow.sum()), "lit_pixels": int(lit.sum()), "runs": {}}
    wrong = {"own-counted": expected * schlick_pass(cos_sun) * np.asarray(TINT), "opaque": np.zeros(3), "no-attenuation": np.ones(3)}
    for key in ("edge@csm", "cache@csm", "cachebefore@csm"):
        if not (out / key / "engine.pfm").exists():
            continue
        image = read_pfm(out / key / "engine.pfm")
        ratio = image[shadow].mean(axis=0) / image[lit].mean(axis=0)
        err = np.abs(ratio - expected) / expected
        controls = {name: bool(np.any(np.abs(ratio - w) / np.maximum(w, 0.05 * expected) > MEAN_BOUND)) for name, w in wrong.items()}
        gate = bool(np.all(err <= MEAN_BOUND))
        # `cachebefore@csm` is evidence that the gate can fail on the old behaviour: it passes this check by FAILING the gate
        passed = (not gate) if key == "cachebefore@csm" else (gate and all(controls[n] for n in ("own-counted", "opaque")))
        result["runs"][key] = {"ratio": [round(float(v), 4) for v in ratio], "relative_error": [round(float(v), 4) for v in err],
                               "gate_pass": gate, "controls_fail": controls, "passed": bool(passed)}
    result["passed"] = bool(result["runs"] and all(r["passed"] for r in result["runs"].values()) and "cache@csm" in result["runs"])
    return result


# ---------------------------------------------------------------------------- runner

def capture_command(binary, folder, scenario, lighting, rt, shadows, accumulate, reflections="off"):
    last = str(FRAMES - 1)
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario, "--render-path", "visibility",
            "--geometry-path", "mesh", "--rt", rt, "--shadows", shadows, "--lighting", lighting, "--gi", "off", "--reflections", reflections,
            "--ao", "off", "--lighting-denoise", "off" if lighting != "legacy" else "custom", "--reflection-samples", "8", "--lighting-seed", "1",
            "--post", "--upscaler", "native", "--resolution", f"{RESOLUTION[0]}x{RESOLUTION[1]}", "--frames", str(FRAMES), "--warmup", "0",
            "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", *(["--debug-lighting", "1"] if shadows != "off" or lighting != "legacy" else []), "--pipeline-sync",
            "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", last,
            *(["--capture-linear-accumulate", str(accumulate)] if accumulate else []),
            "--capture-linear-signal", "hdr", "--report", str(folder / "report.json")]


def run(args, which):
    import f15_check as fc
    out, binary = args.out.resolve(), args.binary.resolve()
    jobs = []
    if which in ("all", "btdf"):
        jobs += [(key, capture_command(binary, out / key, sc, li, rt, "off", acc)) for key, (sc, li, rt, acc) in BTDF_MODES.items()]
    if which in ("all", "stack"):
        jobs += [(key, capture_command(binary, out / key, sc, "legacy", rt, sh, 8) + STACK_EXTRA.get(key, [])) for key, (sc, sh, rt) in STACK_MODES.items()]
    if which in ("all", "edge"):
        for key, (sh, rt, divisor) in EDGE_MODES.items():
            command = capture_command(binary, out / key, "transmissive-shadow-pane-sun", "legacy", rt, sh, 8)
            command += ["--shadow-transmittance-divisor", str(divisor)] if sh == "csm" else []
            jobs.append((key, command))
    if which in ("all", "cache"):
        for key, (sh, rt, extra, old) in CACHE_MODES.items():
            if old and not args.before_binary:
                continue
            command = capture_command((args.before_binary if old else binary).resolve(), out / key, "transmissive-shadow-pane-sun", "legacy", rt, sh, 8) + extra
            jobs.append((key, command))
        if which == "cache" and (out / "edge@csm" / "engine.pfm").exists() is False:
            jobs.append(("edge@csm", capture_command(binary, out / "edge@csm", "transmissive-shadow-pane-sun", "legacy", "off", "csm", 8) + ["--shadow-transmittance-divisor", "2"]))
    if args.harvest:  # targeted archive update: the same command lines, harvesting instead of capturing (no evaluation)
        harvested = []
        for key, command in jobs:
            drop = {"--capture-linear", "--capture-linear-frame", "--capture-linear-accumulate", "--capture-linear-signal", "--report"}
            kept, skip = [], False
            for word in command:
                if skip:
                    skip = False
                    continue
                if word in drop:
                    skip = True
                    continue
                kept.append(word)
            kept[kept.index("--frames") + 1] = "12"
            (args.harvest / key).mkdir(parents=True, exist_ok=True)
            harvested.append((key, kept + ["--harvest-pipelines", str(args.harvest / key / "raw.json")]))
        jobs = harvested
    import contextlib
    with (contextlib.nullcontext() if args.no_lock else fc.gpu_lock(args.gpu_lock)):  # --no-lock: the caller holds the lock (gpulock.py)
        for key, command in jobs:
            busy = fc.running_gpu_processes()
            if busy:
                raise fc.GpuBusy(f"other GPU jobs are running: {', '.join(busy)}")
            folder = out / key
            folder.mkdir(parents=True, exist_ok=True)
            if args.harvest:
                result = subprocess.run(command, stdout=open(args.harvest / key / "app.log", "w"), stderr=subprocess.STDOUT, timeout=args.timeout)
                print(f"harvest {key}: exit {result.returncode}")
                continue
            status = fc.capture(command, folder / "engine.log", args.timeout)
            if not status["passed"]:
                print(f"capture {key} failed: {status['failures']}")
                return 6
            draws = fc.fallback_draws((folder / "engine.log").read_text(errors="replace"))
            if draws is None or draws > 0:
                print(f"capture {key} served fallback pipelines ({draws})")
                return 6
            time.sleep(2)
    if args.harvest:
        return 0
    return evaluate(out, which)


def evaluate(out, which):
    import numpy as np
    result = {"bounds": {"mean": MEAN_BOUND, "p95": P95_BOUND}}
    if which in ("all", "btdf"):
        keys = [k for k in BTDF_MODES if (out / k / "engine.pfm").exists()]
        result["btdf"] = evaluate_btdf(np, out, keys)
    if which in ("all", "stack"):
        modes = [m for m in ("csm", "rt") if (out / f"stack@{m}" / "engine.pfm").exists() and (out / f"single@{m}" / "engine.pfm").exists()]
        if (out / "stackold@csm" / "engine.pfm").exists() and (out / "single@csm" / "engine.pfm").exists():
            modes.append("csm-before")
        result["stack"] = evaluate_stack(np, out, modes)
    if which in ("all", "edge") and (out / "edge@csm" / "engine.pfm").exists() and (out / "edge@rt" / "engine.pfm").exists():
        result["edge"] = {"pane-sun": evaluate_edges(np, out)}
    if which in ("all", "cache") and (out / "cache@csm" / "engine.pfm").exists():
        result["cache"] = {"pane-sun": evaluate_cache(np, out)}
    passed = all(v["passed"] for part in ("btdf", "stack", "edge", "cache") for v in result.get(part, {}).values())
    result["passed"] = bool(passed)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build" / "release" / "phosphor")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-lock", default="/tmp/phosphor-gpu-verification.lock")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--harvest", type=pathlib.Path, default=None, help="write pipeline harvests (--harvest-pipelines) of every capture command here instead of capturing")
    parser.add_argument("--no-lock", action="store_true", help="do not take the GPU lock (run under tools of the caller that holds it)")
    parser.add_argument("--only", choices=("all", "btdf", "stack", "edge", "cache"), default="all")
    parser.add_argument("--before-binary", type=pathlib.Path, default=None, help="the pre-fix binary for the fail-before captures")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluate:
        return evaluate(args.out.resolve(), args.only)
    try:
        return run(args, args.only)
    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        return 4 if type(error).__name__ == "GpuBusy" else 5


if __name__ == "__main__":
    sys.exit(main())
