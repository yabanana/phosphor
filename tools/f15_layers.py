#!/usr/bin/env python3
"""F15.4 (ticket 23): references and statistics of the layered-material keys of tools/f15_check.py (pure numpy; no GPU, no Mitsuba).

Kinds (each judged by an evaluator in this module, called from f15_check.evaluate_layer_scenario):

  layer-equal   two engine captures of the same camera: a layer stack whose mask is the weight vector e_j (or has weight-0 layers /
                fewer layers) and the SINGLE-material twin (same textures, mesh uvs scaled by the power-of-two tiling).  The gate is the
                ticket's criterion: 0 differing pixels (bitwise on the linear HDR signal).  The statistics always carry the bitwise, ULP,
                relative and 8-bit counts, so a codegen-induced ULP difference is reported with numbers and never re-defined away.
                `envelope` (the coat-zero key) gates the declared relative envelope of ticket 21 instead (1e-5 per channel).
                Negative controls: cross pairs (stack e_k against the twin of layer j != k) must fail the same gate.
  layer-ramp    flat quad, two plain layers, mask ramp along u: per pixel the blend factor (L - L0) / (L1 - L0) against the two single
                twins must rise monotonically along u (the transition ramp), end at 0 / 1 and follow the mask; the reversed ramp
                (control) must fail.
  layer-furnace flat quad, four albedo-1 layers (metallic 0/1/0/1, roughness .1/.4/.7/1) blended by a bilinear mask, environment 1,
                grazing view: no engine pixel above 1 + 0.5 %; the independent reference of the BLENDED single BSDF in parameter space
                (tools/f15_reference.py environment quadrature) and its deviation from the BSDF-space mix are recorded; the additive
                blend (the sum of the layers' BSDFs, weights ignored) is the negative control and must exceed the bound.

Everything here is a function of arrays so that tools/test_f15_layers.py checks it on synthetic data.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

# Declared constants (mirrored in src/testbench/terrain_fixture.h; tests/test_terrain_fixture.cpp and tools/test_f15_layers.py compare them).
FURNACE_METALLIC = (0.0, 1.0, 0.0, 1.0)
FURNACE_ROUGHNESS = (0.1, 0.4, 0.7, 1.0)
RAMP_DARK, RAMP_LIGHT, RAMP_ROUGHNESS = 0.05, 0.9, 0.8
PATCH_SIZE = 8.0
MASK_SIZE = 256            # flat mask width/height (ramp: 256 x 4)
EDGE_MARGIN = 3.0 / MASK_SIZE  # uv margin excluded at the plane edges (the repeat sampler blends the opposite edge there)
ENVELOPE = 1e-5            # ticket 21: declared relative envelope between a specialised and the all-lobes pipeline
# The coat-weight-0 key: the clearcoat guide bit routes the pixel through other composite/denoise code, so its capture differs from the coat-free
# twin by float noise (measured: median 1e-7, worst 3e-6 absolute at a pixel of 0.033, 1.0e-4 floored relative, 0 pixels in the 8-bit image).  The bound
# 5e-4 sits 5x above that and far below what a wrong base F0 does (the wrong-F0 control, ior 1.4: F0 0.040 -> 0.028, must fail it).
COAT_ZERO_ENVELOPE = 5e-4
FURNACE_LIMIT = 1.0 + 0.005
RAMP_TOLERANCE = 1e-3      # monotonicity tolerance on the blend factor
RAMP_MASK_ERROR = 0.02     # |blend factor - u| bound (mask quantisation 1/255 + the sampler's half-texel)


# --- image statistics ----------------------------------------------------------------------------------------------

def ordered_ints(np, image):
    """Float32 bit patterns as integers whose difference is the distance in representable floats."""
    bits = np.ascontiguousarray(image, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def srgb8(np, image):
    x = np.clip(image, 0.0, 1.0)
    encoded = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.floor(encoded * 255.0 + 0.5).astype(np.int32)


def compare_images(np, a, b, roi=None):
    """Bitwise, ULP, relative and 8-bit differences of two linear float32 images (over `roi`, a boolean (h, w) mask, or everywhere)."""
    if a.shape != b.shape:
        raise ValueError(f"image sizes differ: {a.shape} vs {b.shape}")
    if roi is None:
        roi = np.ones(a.shape[:2], dtype=bool)
    a = np.asarray(a, dtype=np.float32)[roi]
    b = np.asarray(b, dtype=np.float32)[roi]
    if a.size == 0:
        return {"pixels": 0, "bitwise_differing": 0, "eight_bit_differing": 0, "max_ulp": 0, "max_relative": 0.0, "mean_relative": 0.0, "max_relative_floored": 0.0}
    differing = (a.view(np.uint32) != b.view(np.uint32)).any(axis=1)
    ulp = np.abs(ordered_ints(np, a) - ordered_ints(np, b)).max(axis=1)
    floor = 1e-4  # relative error floor: radiance below 1e-4 is not a visible difference
    relative = (np.abs(a.astype(np.float64) - b.astype(np.float64)) / np.maximum(np.abs(b.astype(np.float64)), floor)).max(axis=1)
    eight = (srgb8(np, a) != srgb8(np, b)).any(axis=1)
    # The runner's own error convention (tools/f15_check.py relative_error): the denominator has the floor 0.05 x the mean radiance of the
    # compared pixels, so that a dark pixel's rounding noise (absolute 1e-8) is not read as a 1e-4 relative difference.
    floor_fraction = 0.05 * float(np.abs(b.astype(np.float64)).mean())
    floored = (np.abs(a.astype(np.float64) - b.astype(np.float64)) / np.maximum(np.abs(b.astype(np.float64)), floor_fraction)).max(axis=1)
    return {"pixels": int(len(a)), "bitwise_differing": int(differing.sum()), "eight_bit_differing": int(eight.sum()),
            "max_ulp": int(ulp.max()), "max_relative": float(relative.max()), "mean_relative": float(relative.mean()),
            "max_relative_floored": float(floored.max())}


def evaluate_equal(np, candidate, reference, envelope=False, roi=None):
    """The `layer-equal` gate: exact (0 bitwise-differing pixels) or an envelope on the floored relative error (`envelope` True = ENVELOPE,
    a number = that bound; the plain relative error with an absolute 1e-4 floor, the bitwise, ULP and 8-bit counts are reported next to it)."""
    stats = compare_images(np, candidate, reference, roi)
    non_empty = float(np.asarray(candidate).max()) > 0.0 and float(np.asarray(reference).max()) > 0.0
    bound = (ENVELOPE if envelope is True else float(envelope)) if envelope else 0
    passed = (stats["max_relative_floored"] <= bound) if envelope else (stats["bitwise_differing"] == 0)
    return {"kind": "layer-equal", "gate": "envelope" if envelope else "exact", "limit": bound, **stats,
            "non_empty": bool(non_empty), "pass": bool(passed and non_empty)}


# --- the transition ramp --------------------------------------------------------------------------------------------

def witness_positions(np, snapshot_dir):
    """(roi (h, w) bool, x (h, w), z (h, w)) of the primary surface of a flat-quad capture, from the engine's primary-surface witness."""
    data = json.loads((pathlib.Path(snapshot_dir) / "primary-surfaces.json").read_text())
    h, w = data["height"], data["width"]
    roi = np.zeros((h, w), dtype=bool)
    x = np.zeros((h, w))
    z = np.zeros((h, w))
    for index, pixel in enumerate(data["pixels"]):
        if pixel is not None:
            roi[index // w, index % w] = True
            x[index // w, index % w] = pixel["position"][0]
            z[index // w, index % w] = pixel["position"][2]
    return roi, x, z


def plane_uv(np, x, z):
    """uv of the flat quad (generatePlane: u along +x, v along +z, centred, side PATCH_SIZE)."""
    return (x + 0.5 * PATCH_SIZE) / PATCH_SIZE, (z + 0.5 * PATCH_SIZE) / PATCH_SIZE


def luminance(np, image):
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def evaluate_ramp(np, candidate, endpoint0, endpoint1, roi, u):
    """Blend factor t = (L - L0) / (L1 - L0) per pixel against the single twins; it must rise with u (per image row), follow the mask u
    within RAMP_MASK_ERROR and be ~0 / ~1 at the ends of the ramp.  The pixel is affine in the layer albedo (dielectric, sun only), so t is
    the layers' weight of the blend."""
    interior = roi & (u > EDGE_MARGIN * 2) & (u < 1.0 - EDGE_MARGIN * 2)
    l0, l1, l = luminance(np, endpoint0), luminance(np, endpoint1), luminance(np, candidate)
    span = l1 - l0
    usable = interior & (span > 0.02 * float(span[interior].mean())) if interior.any() else interior
    t = np.zeros_like(l)
    t[usable] = (l[usable] - l0[usable]) / span[usable]
    reversals, worst_drop, rows = 0, 0.0, 0
    for y in range(t.shape[0]):
        row = np.flatnonzero(usable[y])
        if row.size < 8:
            continue
        rows += 1
        order = row[np.argsort(u[y, row])]
        drop = float(np.max(np.maximum.accumulate(t[y, order]) - t[y, order]))  # largest fall below the running maximum
        worst_drop = max(worst_drop, drop)
        reversals += int(drop > RAMP_TOLERANCE)
    error = np.abs(t[usable] - u[usable]) if usable.any() else np.zeros(1)
    low = usable & (u < 0.05)
    high = usable & (u > 0.95)
    end_low = float(t[low].mean()) if low.any() else 1.0  # reported: the blend factor near the ends of the ramp (~0 and ~1)
    end_high = float(1.0 - t[high].mean()) if high.any() else 1.0
    slope = float(np.polyfit(u[usable], t[usable], 1)[0]) if usable.sum() > 2 else 0.0
    ok = bool(rows > 0 and reversals == 0 and float(error.max()) <= RAMP_MASK_ERROR)
    return {"kind": "layer-ramp", "rows": rows, "usable_pixels": int(usable.sum()), "rows_with_reversal": reversals, "worst_drop": worst_drop,
            "max_mask_error": float(error.max()), "mean_mask_error": float(error.mean()), "mean_t_near_low_end": end_low, "mean_one_minus_t_near_high_end": end_high,
            "fitted_slope": slope, "tolerance": RAMP_TOLERANCE, "mask_error_limit": RAMP_MASK_ERROR, "pass": ok}


# --- the furnace ------------------------------------------------------------------------------------------------------

def bilinear_weights(np, u, v):
    return np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=-1)


def quantise_roughness(r, steps=64):
    return [round(float(x) * steps) / steps for x in r]


def furnace_reference(np, u, v, n_dot_v, radiance=1.0, declared=False):
    """Outgoing radiance of the pixels under the constant environment, as RGB images' worth of (n, 3) arrays:
    blended   the BLENDED single BSDF in parameter space (weights -> albedo 1, metallic and roughness blended, ONE evaluation);
    mixed     the BSDF-space mix  sum w_i E_i  (every layer evaluated alone);
    additive  sum_i [w_i > 0] E_i  (the negative control: the layers' BSDFs added without weights);
    the quadrature is f15_reference.environment_radiance (or the engine's declared split-sum model with `declared`)."""
    import f15_reference as ref
    w = bilinear_weights(np, u, v).reshape(-1, 4)
    metallic = np.array(FURNACE_METALLIC)
    rough = np.array(FURNACE_ROUGHNESS)
    n = len(w)
    base = np.ones((n, 3))
    mu = np.asarray(n_dot_v, dtype=np.float64).reshape(-1)
    evaluate = ref.declared_environment_radiance if declared else ref.environment_radiance

    def radiance_of(m, r):
        return evaluate(base, np.asarray(m, dtype=np.float64), np.asarray(quantise_roughness(r)), mu, radiance)

    blended = radiance_of(w @ metallic, w @ rough)
    layers = [radiance_of(np.full(n, metallic[i]), np.full(n, rough[i])) for i in range(4)]
    mixed = sum(w[:, i:i + 1] * layers[i] for i in range(4))
    additive = sum((w[:, i:i + 1] > 0.0) * layers[i] for i in range(4))
    return {"blended": blended, "mixed": mixed, "additive": additive, "layers": layers}


def evaluate_furnace(np, candidate, roi, reference, controls):
    """Gate: no engine pixel above FURNACE_LIMIT.  Recorded: the engine against the blended reference, the reference's own maximum, the
    deviation of the parameter blend from the BSDF-space mix.  Controls (additive blend) must exceed the limit."""
    values = candidate[roi]
    engine_max = float(values.max())
    blended = reference["blended"]
    lum = lambda a: a[:, 0] * 0.2126 + a[:, 1] * 0.7152 + a[:, 2] * 0.0722  # noqa: E731
    deviation = lum(blended) - lum(reference["mixed"])
    engine_error = np.abs(lum(values) - lum(blended)) / np.maximum(lum(blended), 0.05 * float(lum(blended).mean()))
    result = {"kind": "layer-furnace", "limit": FURNACE_LIMIT, "max_pixel": engine_max, "mean_pixel": float(values.mean()), "pixels": int(len(values)),
              "reference_max": float(blended.max()), "reference_mean": float(blended.mean()),
              "engine_vs_reference_mean_relative": float(engine_error.mean()), "engine_vs_reference_p99_relative": float(np.percentile(engine_error, 99)),
              "deviation_from_bsdf_mix": {"max_abs": float(np.abs(deviation).max()), "mean": float(deviation.mean()), "min": float(deviation.min()),
                                          "max": float(deviation.max()),
                                          "max_relative": float((np.abs(deviation) / np.maximum(lum(reference["mixed"]), 1e-6)).max())},
              "pass": bool(engine_max <= FURNACE_LIMIT)}
    result["controls"] = {name: {"max_pixel": float(image.max()), "fails_gate": bool(float(image.max()) > FURNACE_LIMIT)} for name, image in controls.items()}
    return result
