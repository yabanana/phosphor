#!/usr/bin/env python3
"""F15.3 serial material-validation runner (decision 11 thresholds, fixed before any comparison).

Plan mode (default) prints the exact capture and reference commands and writes
`plan.json`.  `--run` captures every scenario SERIALLY (lock file plus a check
that no other `phosphor`, `soc_bench` or `f*_spike` process runs), then
evaluates.  `--evaluate` evaluates existing captures without launching a GPU
process.  The reference is `tools/f15_reference.py` (glTF Appendix B, written
from the specification; Mitsuba only casts rays), so evaluating needs
Mitsuba and numpy:

  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/f15_check.py ...

Exit codes are distinct (0 ok, 1 unexpected gate result, 2 usage, 3 non-discriminating
control, 4 GPU busy, 5 internal error, 6 capture failed): see the --help epilog.

Gates (relative error = |a - b| / max(b, 0.05 * mean_ROI(b)), per pixel, the
maximum over RGB; ROI = sphere pixels without the silhouette, from the
reference ray casting):
- direct (base-grid-sun, roughness-extremes-sun): per metallic row, p99 <= 1 % and max <= 3 % over the
  lit ROI, the max outside the declared sun-quadrature mask (narrow lobes).
- environment (base-grid-env, roughness-extremes): mean <= 5 % per sphere; the
  per-sphere table is published.  roughness-extremes holds sun and
  environment: only the environment criterion gates, direct statistics are
  reported.
- furnace (furnace-base): no engine pixel above 1.0 + 0.5 %; energy loss per
  sphere is recorded, not gated.
- KHR_materials_ior/specular (F15.1, ticket 05; always asserted green, no red state): specular-grid-sun uses
  the direct gate per (metallic, specular weight, colour) group, specular-grid-env-resolve the declared-model
  gate (split-sum f0 * A + f90 * B per lobe), both with the bounds above unchanged.  Negative controls on the
  reference side that must fail: specular extension ignored, f90 forced to 1, specular weight not applied to f0.
- Point light (F15.1, ticket 07): specular-grid-point is the ior x specular grid lit by one white point light,
  gated like a direct grid.  Lighting variants share the bench scene ("<bench>@<lighting>" keys): legacy
  (resolve local lights, reference without visibility), @restir and @clustered (DI with RT visibility,
  reference shadow rays); DI reads the lobe guide (ticket 07), so all three gate like a direct grid.  A scenario
  with a "pending" key would be asserted to FAIL until its ticket lands (`--expect green` then requires that
  failure); none is pending now.
- Clearcoat (F15.3, ticket 09): clearcoat-grid-sun / -env-resolve / -env / -point (legacy, @restir, @clustered) and
  furnace-clearcoat (+ @resolve) against the independent KHR_materials_clearcoat model of tools/f15_reference.py.
  The engine side landed with F15.1 ticket 09 (they were "pending" with the coat-ignored control as the matching
  image until then): every gate passes.  The furnace bound is an upper bound that a coat redistributes energy under,
  so the furnace gates are asserted green in every mode.  Negative controls (each must fail its gate): coat ignored, coat F0
  from the material ior (the Sample Renderer deviation), coat Fresnel at V.H, base not scaled by 1 - c F_c.
- Sheen (F15.3, ticket 10): sheen-grid-sun / -env-resolve / -env (+ @probes) / -point (legacy, @restir, @clustered)
  and furnace-sheen (+ @resolve) against the independent KHR_materials_sheen model of tools/f15_reference.py (Charlie D
  and V, base scaled by the albedo scaling, E from the reference's own quadrature; the declared environment model reads
  the engine table src/renderer/sheen_table.h generated and checked by tools/sheen_table.py).  The engine side landed
  with F15.1 ticket 10 (the keys were "pending" with the sheen-ignored control as the matching image until then): every
  gate passes; the furnace gates are asserted green in every mode.  Negative controls (each must fail its gate): sheen ignored, base not scaled by the albedo scaling,
  Ashikhmin visibility, alpha = roughness, view-only scaling for direct light (direct keys only).
- Anisotropy (F15.3, ticket 11): aniso-grid-sun / -env-resolve / -env / -point (legacy, @restir, @clustered), furnace-aniso
  (+ @resolve) and aniso-mirrored-sun against the independent KHR_materials_anisotropy model of tools/f15_reference.py
  (Burley D with alpha_t = mix(alpha, 1, strength^2), the README's height-correlated V, the lobe frame from the mesh tangent
  attribute of the snapshot and the world matrix; exact environment by quadrature of the anisotropic lobe; declared
  environment = split sum at an effective roughness).  The engine side landed with F15.1 ticket 11 (the keys were "pending" with
  the anisotropy-ignored control as the matching image until then): every gate passes; the furnace gates are asserted green in
  every mode.
  Negative controls (each must fail its gate): anisotropy ignored, rotation flipped, handedness ignored, alpha_t/alpha_b
  swapped, D normalised by alpha_b^2 (energy), world-matrix determinant ignored (mirrored fixture only).
- Iridescence (F15.3, ticket 12): iridescence-grid-sun / -env-resolve / -env / -point (legacy, @restir, @clustered),
  furnace-iridescence (+ @resolve) against the independent KHR_materials_iridescence model of tools/f15_reference.py (the
  specification's thin-film Fresnel transcribed from its text; V.H per light sample; the exact environment integrates the film
  at each V.H, the declared environment holds it at N.V in the split sum).  The engine side landed with F15.1 ticket 12 (the
  keys were "pending" with the iridescence-ignored control as the matching image until then): every gate passes; the furnace
  gates are asserted green in every mode.  Negative controls (each must fail its
  gate): film ignored, film ior = 1.0, film ior = the default 1.3, N.V instead of V.H, minimum thickness, diffuse not scaled
  by 1 - max F_irid (energy).
- Thin subsurface / diffuse transmission (F15.3, ticket 16): subsurface-thin-sun / -env-resolve / -env / -point (legacy,
  @restir, @clustered), furnace-thin-subsurface (+ @resolve) against the independent KHR_materials_diffuse_transmission model of
  tools/f15_reference.py (reflected (1 - w) and transmitted w * colour Lambert lobes under the literal fresnel_mix, sun at the disk
  centre, exact environment T <= Cd, declared environment (1 - E_d)(1 - 0.16 N.V); shadow rays start on the side of the light).
  The engine side landed with F15.1 ticket 16 (the keys were "pending" with transmission-ignored or its declared twin as the
  matching image until then): every gate passes; the furnace transport bound and the declared furnace gate are asserted green
  (the double-counted control gains energy and must fail them).
  Negative controls (each must fail its gate): transmission ignored, colour ignored, lobe on the front hemisphere
  (back lights ignored), diffuse not reduced by w (energy), no Fresnel weight on the transmitted part (not in the declared keys).
- Transmission + volume (F15.3, ticket 13): the `transmission-*` keys (bench class TransmissionValidation: glass spheres, slabs and
  panes in front of an emissive checkerboard or emissive panels under a constant environment) against the Mitsuba 3.9.1
  path-traced ground truth of tools/f15_transmission_reference.py (pixel-centre rays, roughdielectric / dielectric + a homogeneous
  absorbing medium, the glTF thin-walled interface; every model decision is in that module's docstring).  The engine image is
  the LINEAR HDR signal averaged over the last 32 frames (`--capture-linear-accumulate 112`, `--lighting-denoise off`: the gate measures the unfiltered
  stochastic estimator, as the DI keys do; the custom denoiser trades noise for edge blur, measured 5-6 % on rough refractions).  Gate "transmission" (decision 11): per glass object on the glass mask (silhouette band of 2 px excluded), mean
  relative luminance error <= 5 % and p95 <= 15 %, with the absolute floor 0.05 * mean_ROI(reference).  The engine side landed with
  F15.1 ticket 13 (the keys were "pending", asserted to FAIL with the `transmission-ignored` opaque dielectric as the matching image, until
  then): every gate must pass.  Negative controls (each must fail the gate): transmission
  ignored, base colour tint ignored, volume attenuation ignored, ior ignored (no refraction bend), thin-walled treated as a
  volume, back faces of the closed glass mesh rejected (today's `reflections.metal` rule), emitter radiance counted twice, the
  glass behind the front glass black.  `transmission-rtoff` is the declared `--rt off` model (probes / environment along the
  refracted ray, closed form from the DFG table) with the same bound.
- Transmissive shadows (F15.3, tickets 14 and 15): the `transmissive-shadow-*` keys (bench class TransmissiveShadowValidation: a
  horizontal diffuse floor under horizontal planar casters, lit by the sun within 30 degrees of their normal or by one point light,
  environment 0) against tools/f15_shadow_reference.py (Mitsuba 3.9.1 rays from each receiver point through the casters with
  Mitsuba's BSDFs, Fresnel and the Beer-Lambert wrapper; the receiver irradiance through a parallel slab is the straight-line
  transmittance up to a lateral shift of millimetres).  The engine image is the linear HDR signal of the last 8 frames.  Gate
  "transmissive-shadow" (decision 11 transmission bound): per caster SHADOW REGION on the floor (the receiver pixels whose shadow ray,
  and the 8 ring neighbours at the declared world envelope, first hit that caster; the camera never sees the floor through a caster):
  mean relative error <= 5 %, p95 <= 15 %, absolute floor 0.05 * mean over the whole mask of the GROUND TRUTH (also for the controls and
  for the pending twin).  The envelope is declared BEFORE any measurement (SHADOW_ENVELOPE): `rt` 0.06 m = sun penumbra at the caster
  height (3.7 mm) + the shadow-mask bilateral filter (about 1.5 cm at this view) + the horizontal run of the straight ray through the
  thickest slab (0.1 m * tan 20 degrees = 3.6 cm: the straight ray may leave through a side wall where the refracted one leaves
  through the top); `csm` 0.08 m = 2 texels of the cascade covering the receiver (cascade radius <= 9 m, 1024 texels: <= 17.6 mm per
  texel; PCSS disk radius max(penumbra, 0.5 texel), nearest-texel quantisation, depth/normal bias slop) + the same 3.6 cm + the penumbra.
  The sun keys run with `--shadows rt` (`@rt`, ticket 14) and `--shadows csm` (`@csm`, ticket 15); `transmissive-shadow-point@restir`
  is the DI visibility through glass (ReSTIR DI, RT visibility rays).  The RT and DI keys (ticket 14) are gated since
  F15.1 ticket 14 (RGB transmittance of straight shadow rays); the CSM keys are gated since
  F15.1 ticket 15 (the single-layer transmittance map of the cascades); before it they were "pending" (asserted to FAIL: the CSM casters
  shadowed opaquely) and their image passed the gate against the `transmittance-ignored` control (red for the right reason).  Negative controls (the declared straight-line model with one ingredient
  removed; each must fail the gate): transmittance ignored (opaque casters), tint ignored, Fresnel ignored, Beer-Lambert ignored,
  transmittance applied twice.  A control equal to the reference within the bound on a fixture is not listed there (water has
  Fresnel 2 % per surface: Fresnel-ignored is 4.6 %).  ORCHESTRATOR DECISIONS (2026-11-14): the CSM keys (`@csm`, kind
  `transmissive-shadow-ratio`) gate the TRANSMITTANCE RATIO, not the absolute luminance: per caster region T = mean luminance of the region /
  mean luminance of the declared lit reference region (floor outside every shadow and envelope, within 0.5 m of the shadows), compared with the
  truth's T (relative error <= 5 % with the floor 0.05 * mean T of the truth; p95 of the per-pixel luminance / lit mean against the truth's <= 15 %);
  the lit-floor level of the CSM against the truth (a pre-existing F10 bias) is a separate NOT gated diagnostic (`lit_floor_diagnostic`).  RT and
  DI keys keep the absolute bound.  `transmissive-shadow-curved@rt` (kind `transmissive-shadow-curved`, gated since ticket 14) gates a QUALITATIVE
  property only: per caster region 0.001 < T < 0.995 (lighter than an opaque caster, darker than unshadowed) and the chroma statistic R/B (mean
  red over mean blue of the region) follows the tint (base x attenuation colour): within 2 % of 1 for a neutral caster, on the tint's side of 1 with
  at least half its log magnitude for a tinted one; the numbers against the declared straight-line model (caustics excluded) are reported, never
  gated; controls: shadow-ignored (T = 1) and transmittance-ignored (T = 0).  The declared model itself must
  pass the gate against the Mitsuba truth on every gated key (`model_feasible`): otherwise the bound is unreachable for an engine
  implementing the declaration and the fixture is blocked (exit 3).
- Volumetric subsurface (F15.3, ticket 17): subsurface-volumetric-sun / -radius / -subpixel (+ the capture-only -subpixel-off) / -backlit
  against the independent ground truth of tools/f15_subsurface_reference.py (Mitsuba 3.9.1 `volpath` random walk of a homogeneous medium
  inside a rough dielectric boundary; OpenPBR parameters mapped through the draft's Kulla-Conty inversion; the reference of a piece with
  weight w is (1 - w) base model + w random walk).  Gate kinds: `subsurface` = the decision-11 bound on the piece mask (per piece: mean
  relative LUMINANCE error <= 10 %, p95 <= 25 %, absolute floor 5 % of the piece's mean reference luminance), `radius-monotonic` = the
  declared terminator statistic (mean luminance of the geometrically unlit ROI pixels over the mean of the lit ones: the light the
  medium carries across the terminator) must increase at every step of the radius sweep (no reversal beyond 0.01, and an increase of at
  least 0.01 wherever the ground truth increases by that much), `equal` = the engine capture equals the engine capture of the pair scene
  without any block (sub-pixel radius = subsurface off), `subsurface-report` = measured and reported, never gated (back-lit thick pieces:
  declared limit of screen-space diffusion).  The engine side landed with F15.1 ticket 17 (the keys were "pending": "ticket 17" with
  `pending_matches` = subsurface-off, the base-model image, until then): the sweep, sub-pixel and back-lit keys pass; the side-lit bound
  key misses the bound on the gated pieces and is asserted red as "pending": "ticket 18" (hybrid RT single scattering, the decision rule
  of ticket 10).  Negative controls: subsurface-off,
  diffusion-off (the colour replaces the diffuse albedo, no diffusion), radius-ignored (the median radius everywhere), subsurface-leak
  (a sub-pixel radius that is not skipped).
- declared (base-grid-env-resolve; also furnace-resolve next to its furnace
  gate): the engine image against the independent declared-environment model of
  tools/f15_reference.py (DFG table, ADR-0006): per sphere mean relative error
  <= 0.5 % and per-pixel p99 <= 1 % (the absolute floor rule above).  Its
  negative controls (diffuse unweighted, Karis fit instead of the table,
  diffuse weighted by the metallic-blended F0 Fresnel) must all fail it.
- Layered materials (F15.4, ticket 23): the `layers-*` keys (bench `terrain`, testbench/terrain_fixture.h; statistics and the blended-BSDF
  quadrature in tools/f15_layers.py, no external reference tool).  `layer-equal` keys compare two ENGINE captures: the layer stack with the constant
  mask e_j (weight 1 on layer j; height contrast on) against the single material of layer j (same textures, mesh uvs scaled by the power-of-two
  tiling) must have 0 bitwise-differing pixels (the ticket criterion; bitwise / ULP / relative / 8-bit counts are always reported, a codegen ULP
  difference is never redefined away); `layers-weight-zero`: weight 0 on two layers equals the stack without them; `layers-coat-zero`: the coat
  finish present with coat weight 0 leaves the base untouched (declared envelope 1e-5 relative: a different class pipeline).  `layer-ramp`: on a
  flat quad with two plain layers the blend factor (L - L0) / (L1 - L0) against the two single twins must rise monotonically along the mask ramp
  and follow it within 0.02.  `layer-furnace`: four albedo-1 layers (metallic 0/1/0/1) under environment 1: no engine pixel above 1.005; the
  independent reference of the blended single BSDF (tools/f15_reference.py quadrature) and its deviation from the BSDF-space mix are recorded.
  Negative controls (each must fail its gate): the single material of another layer (equality), a stack of the wrong layer or weights (weight
  zero), the real cavity coat (coat zero), the reversed ramp, the additive blend of the layers' BSDFs (furnace).  The `-cc0` keys repeat the
  endpoint equality with the owner-approved CC0 sets (TERRAIN_MATERIALS_DIR or build/terrain-materials; skipped when absent).
Design note: the environment gate cannot see a missing diffuse Fresnel weighting
(it moves a sphere mean by at most about 4.5 %, below the 5 % bound, measured with
the reference's `diffuse-unweighted` control), so that defect is owned by the
furnace fixture; the environment grid requires the alpha and Fresnel controls.
The literal Appendix B dielectric also gains energy at grazing views (the
specification says so): the reference's own furnace maximum is recorded next to
the engine's, and an engine that implements the literal environment quadrature
would fail the furnace gate, which is evaluated on the engine pixels only.
Negative controls are rendered on the reference side with deliberately wrong
models; a control that does not exceed the gate marks its fixture
non-discriminating, which blocks (exit 3).  Expected red state (`--expect red`,
default, before ticket 03): the direct grid passes at metallic 0 and 1 and
fails at intermediate metallic, the furnace fails; the environment grids are
measured, not predicted.  `--expect green` requires every gate to pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f15_common import sha256_file  # noqa: E402

EXIT_OK, EXIT_UNEXPECTED, EXIT_USAGE, EXIT_NON_DISCRIMINATING, EXIT_GPU_BUSY, EXIT_INTERNAL, EXIT_CAPTURE_FAILED = range(7)
EXIT_CODES_HELP = """exit codes:
  0  expected state confirmed (red: the documented red state; green: every gate passes)
  1  unexpected gate result
  2  usage error (also argparse errors, a missing renderer, a non-empty output folder)
  3  a negative control does not fail its gate (non-discriminating fixture)
  4  GPU busy: another GPU job runs or the verification lock is held
  5  internal error (any exception inside this tool)
  6  scenario capture failed
"""


class GpuBusy(RuntimeError):
    """Another cooperating GPU verification owns the lock, or another GPU job is running."""


class CaptureFailed(RuntimeError):
    """A scenario capture process failed."""


THRESHOLDS = {
    "relative_error": "abs(a - b) / max(b, floor_fraction * mean_ROI(b)), per pixel, maximum over RGB",
    "floor_fraction": 0.05,
    "direct": {"p99": 0.01, "max_outside_quadrature_mask": 0.03},
    "environment": {"mean_per_sphere": 0.05},
    "furnace": {"max_pixel": 1.005, "energy_loss": "recorded, not gated"},
    # Stochastic transport (furnace-base): the gate is the per-sphere mean, i.e. the
    # energy in expectation; per-pixel maxima are estimator noise there (decision 11
    # amendment: a pure-metal sphere, which cannot gain energy, reaches 1.013-1.019
    # across three seeds). The per-pixel bound stays on the deterministic furnace-resolve.
    "furnace_transport": {"mean_per_sphere": 1.005},
    # Declared-environment model (orchestrator, review 2026-11-13): fixed before any comparison.
    "declared": {"mean_per_sphere": 0.005, "p99_per_sphere": 0.01,
                 "model": "tools/f15_reference.py declared environment model (DFG table of src/renderer/dfg_table.h)"},
    # F15.3 ticket 13 (decision 11): transmission vs Mitsuba ground truth on the glass mask, static accumulation, per glass object.
    "transmission": {"mean_per_object": 0.05, "p95_per_object": 0.15, "floor": "0.05 * mean over the whole glass ROI of the reference",
                     "mask": "primary hit on a glass instance, silhouette band of 2 px excluded"},
    # F15.3 tickets 14/15 (decision 11): transmissive shadows vs Mitsuba on the receiver's shadow regions, per caster.
    "transmissive_shadow": {"mean_per_object": 0.05, "p95_per_object": 0.15,
                            "floor": "0.05 * mean over the whole shadow mask of the GROUND TRUTH (controls and twin included)",
                            "mask": "floor pixels whose shadow ray and 8 ring neighbours at the world envelope first hit the same caster",
                            "envelope_m": {"rt": 0.06, "csm": 0.08},
                            "ratio_gate": "CSM keys (orchestrator): T = mean luminance of the caster's shadow region / mean luminance of the declared lit "
                                          "reference region (floor outside every shadow and envelope, within 0.5 m of the shadows); relative error of T <= 5 % "
                                          "(floor 0.05 * mean T of the truth), p95 of the per-pixel luminance / lit mean against the truth's <= 15 %; "
                                          "the lit-floor level is a separate not gated diagnostic",
                            "curved": {"transmittance": [0.001, 0.995], "chroma_clear": 0.02, "chroma_tinted_fraction": 0.5}},
    # F15.3 (ticket 17), decision 11: subsurface vs random-walk ground truth on the piece mask.
    "subsurface": {"mean": 0.10, "p95": 0.25, "floor_fraction": 0.05,
                   "metric": "relative LUMINANCE error |Y_a - Y_b| / max(Y_b, 0.05 * mean_piece(Y_b)), Rec.709 weights, per piece ROI"},
    "radius_sweep": {"tolerance": 0.01,
                     "statistic": "terminator bleed = mean Y over the geometrically unlit ROI pixels of a piece / mean Y over its lit ROI pixels"},
    "equal": {"max_relative_luminance_error": 0.002, "min_mean_luminance": 1e-3},
    "source": ".scratch/f15/decisions/11-reference-method-and-thresholds.md (incl. review amendments)",
}
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
FRAMES, RESOLUTION = 128, "1280x720"
ACCUMULATE = 112  # frames averaged by --capture-linear-accumulate for the stochastic transmission transport (frames 16..127).  Measured
#                   (F15.1 ticket 13): the refraction estimator is unbiased (per-object mean ratio to Mitsuba within 0.5 %) but its per-pixel
#                   noise at 8 samples x 32 frames is 4-8 % on rough spheres (two seeds, same tolerance as the 256-sample Mitsuba noise),
#                   which the 5 % per-object bound cannot absorb; 112 frames bring it to 2-4 %.
REFERENCE_SPP = 1024  # Mitsuba samples per glass-mask pixel
TRANSMISSION_KINDS = ("transmission", "transmission-declared")
SHADOW_KINDS = ("transmissive-shadow", "transmissive-shadow-ratio", "transmissive-shadow-curved")
# Declared BEFORE any measurement (tickets 14/15, see the docstring): the world radius of the ring that excludes the shadow boundary.
SHADOW_ENVELOPE = {"rt": 0.06, "csm": 0.08}
SHADOW_ACCUMULATE = 8   # frames averaged for the shadow keys (frames 120..127): the engine's shadow transport may be stochastic
SHADOW_PANE_CONTROLS = ["transmittance-ignored", "tint-ignored", "fresnel-ignored", "double-counted"]
SHADOW_WATER_CONTROLS = ["transmittance-ignored", "beer-lambert-ignored", "double-counted"]      # Fresnel-ignored: 4.6 % < the bound
SHADOW_VOLUME_CONTROLS = ["transmittance-ignored", "fresnel-ignored", "beer-lambert-ignored", "double-counted"]  # base white: no tint control
SHADOW_POINT_CONTROLS = ["transmittance-ignored", "tint-ignored", "fresnel-ignored", "double-counted"]

ANISO_DIRECT_CONTROLS = ["aniso-ignored", "aniso-rotation-flipped", "aniso-bitangent-flipped", "aniso-alpha-swapped",
                         "aniso-d-normalised-by-ab"]

IRID_DIRECT_CONTROLS = ["iridescence-ignored", "iridescence-film-ior-ignored", "iridescence-film-ior-default",
                        "iridescence-at-nv", "iridescence-thickness-min", "iridescence-diffuse-unweighted"]
IRID_DECLARED_CONTROLS = ["declared-iridescence-ignored", "declared-iridescence-film-ior-ignored",
                          "declared-iridescence-film-ior-default", "declared-iridescence-diffuse-unweighted"]

THIN_SUN_CONTROLS = ["transmission-ignored", "transmission-untinted", "transmission-front-only",
                     "transmission-double-counted", "transmission-fresnel-ignored"]
# The point-light carriers (single back light, single-sided quads) have no front-lit pixel: the reflected part is dark, so the
# double-counted control equals the reference there (the sun and environment keys own it).
THIN_POINT_CONTROLS = ["transmission-ignored", "transmission-untinted", "transmission-front-only", "transmission-fresnel-ignored"]
THIN_DECLARED_CONTROLS = ["declared-transmission-ignored", "declared-transmission-untinted", "declared-transmission-front-only",
                          "declared-transmission-double-counted"]
# The furnace carries a white transmission colour, so the untinted control equals the reference there; and in a uniform
# environment the declared front-only control (transmitted weight = the diffuse weight it replaces) is the image of the ignored
# control (measured: identical), so the furnace keeps the two controls that differ.
THIN_FURNACE_DECLARED_CONTROLS = ["declared-transmission-ignored", "declared-transmission-double-counted"]

# F15.3 (ticket 19): the combination keys.  Per combination the controls drop one lobe at a time (the per-lobe `ignored` controls of
# the single-lobe tickets act on the scene's main model, so they keep the other lobes), the layering controls remove the layer
# attenuation (`additive-layering`: lobes summed, the furnace control) or swap the order (`order-swapped`: coat below sheen), and
# `transmission-uncoated` lets the thin transmitted lobe escape the coat.  The declared environment has the same per-lobe controls
# (`declared-*-ignored`) and the declared twins of the layering controls.
COMBO_COAT_SHEEN_CONTROLS = ["coat-ignored", "sheen-ignored", "order-swapped", "additive-layering"]
COMBO_COAT_SHEEN_DECLARED = ["declared-coat-ignored", "declared-sheen-ignored", "declared-order-swapped", "declared-additive-layering"]
COMBO_ANISO_COAT_CONTROLS = ["aniso-ignored", "aniso-rotation-flipped", "coat-ignored", "additive-layering"]
COMBO_ANISO_COAT_DECLARED = ["declared-aniso-ignored", "declared-coat-ignored", "declared-additive-layering"]
COMBO_ANISO_SHEEN_CONTROLS = ["aniso-ignored", "aniso-rotation-flipped", "sheen-ignored", "additive-layering"]
COMBO_ANISO_SHEEN_DECLARED = ["declared-aniso-ignored", "declared-sheen-ignored", "declared-additive-layering"]
COMBO_SPECULAR_SHEEN_CONTROLS = ["specular-ignored", "sheen-ignored", "additive-layering"]
COMBO_SPECULAR_SHEEN_DECLARED = ["declared-specular-ignored", "declared-sheen-ignored", "declared-additive-layering"]
COMBO_IRID_COAT_CONTROLS = ["iridescence-ignored", "coat-ignored", "additive-layering"]
COMBO_IRID_COAT_DECLARED = ["declared-iridescence-ignored", "declared-coat-ignored", "declared-additive-layering"]
COMBO_IRID_SHEEN_CONTROLS = ["iridescence-ignored", "sheen-ignored", "additive-layering"]
COMBO_IRID_SHEEN_DECLARED = ["declared-iridescence-ignored", "declared-sheen-ignored", "declared-additive-layering"]
COMBO_IRID_ANISO_CONTROLS = ["iridescence-ignored", "aniso-ignored", "aniso-rotation-flipped", "additive-layering"]
COMBO_IRID_ANISO_DECLARED = ["declared-iridescence-ignored", "declared-aniso-ignored", "declared-additive-layering"]
COMBO_IRID_SPECULAR_CONTROLS = ["iridescence-ignored", "specular-ignored", "additive-layering"]
COMBO_IRID_SPECULAR_DECLARED = ["declared-iridescence-ignored", "declared-specular-ignored", "declared-additive-layering"]
COMBO_THIN_COAT_CONTROLS = ["transmission-ignored", "coat-ignored", "transmission-uncoated", "additive-layering"]
# `transmission-uncoated` did not move the sphere mean of the exact environment by 5 % on the original two rows (the attenuated transmitted share is
# small); the grid has a third row (weight 1, white transmission colour) where it must (ticket 19, F15.3 closure).
COMBO_THIN_COAT_ENV_CONTROLS = ["transmission-ignored", "coat-ignored", "transmission-uncoated", "additive-layering"]
COMBO_THIN_COAT_DECLARED = ["declared-transmission-ignored", "declared-coat-ignored", "declared-additive-layering"]
# F15.3 closure (ticket 16): thin subsurface under sheen, anisotropy, iridescence and ior/specular (glTF Sample Renderer layering).  The exact environment
# of the combination has no per-ior transmitted-albedo table yet (the key is gated by the declared model, like iridescence + anisotropy was).
COMBO_THIN_LOBES_CONTROLS = ["transmission-ignored", "sheen-ignored", "aniso-ignored", "iridescence-ignored", "specular-ignored",
                             "transmission-default-fresnel", "transmission-sheen-unscaled"]
COMBO_THIN_LOBES_DECLARED = ["declared-transmission-ignored", "declared-sheen-ignored", "declared-aniso-ignored", "declared-iridescence-ignored",
                             "declared-specular-ignored", "declared-transmission-default-weight"]
# The exact-environment keys integrate over the whole sphere: a rotation of the lobe by the sign of the angle moves the sphere mean by less
# than the 5 % bound there (measured: aniso-rotation-flipped passes the env gate of the anisotropy + coat / sheen grids), so those keys drop it.
COMBO_ANISO_COAT_ENV_CONTROLS = ["aniso-ignored", "coat-ignored", "additive-layering"]
COMBO_ANISO_SHEEN_ENV_CONTROLS = ["aniso-ignored", "sheen-ignored", "additive-layering"]
# Orchestrator decision (2026-11-14, measured on the integrated engine, build/f15-check/t19probe-1): on the iridescence + anisotropy grid
# `additive-layering` (the diffuse not weighted by the film) moves the exact-environment sphere mean by 4.0 % < 5 %: it does not discriminate
# there and is dropped from the env key only (it stays on -sun and -env@resolve, where the per-pixel and declared gates see it).
# Redesign (F15.3 closure): a third row of dielectrics under a film of index 1.8 (400 nm) carries the control again (reference model: 13.5 %).
COMBO_IRID_ANISO_ENV_CONTROLS = ["iridescence-ignored", "aniso-ignored", "additive-layering"]
# The anisotropy + coat / sheen sphere grids cannot show the rotation flip in a constant environment (mirror symmetry, proven in
# tools/test_f15_reference.py); the yawed, rolled panel grid does (ticket 19, F15.3 closure).
COMBO_ANISO_PANEL_CONTROLS = ["aniso-ignored", "aniso-rotation-flipped", "coat-ignored", "sheen-ignored"]
COMBO_ANISO_PANEL_ENV_CONTROLS = ["aniso-ignored", "aniso-rotation-flipped"]
COMBO_IRID_SPECULAR_ENV_CONTROLS = ["iridescence-ignored", "specular-ignored", "additive-layering"]
COMBO_ALL_LOBES_CONTROLS = ["coat-ignored", "sheen-ignored", "aniso-ignored", "specular-ignored", "additive-layering"]
COMBO_ALL_LOBES_DECLARED = ["declared-coat-ignored", "declared-sheen-ignored", "declared-aniso-ignored", "declared-specular-ignored",
                            "declared-additive-layering"]

VOLUMETRIC_BOUND_CONTROLS = ["subsurface-off", "diffusion-off", "radius-ignored"]
VOLUMETRIC_SWEEP_CONTROLS = ["subsurface-off", "diffusion-off", "radius-ignored"]

SCENARIOS = {
    "base-grid-sun": {"kind": "direct", "controls": ["alpha-roughness", "fresnel-off", "engine-f0-blend"],
                      "terms": "sun disk only (environment 0, no shadows, no GI, reflections source off)"},
    "base-grid-env": {"kind": "environment", "controls": ["alpha-roughness", "fresnel-off"],
                      "terms": "constant environment only (radiance 0.5, no sun)"},
    "furnace-base": {"kind": "furnace-transport", "controls": ["diffuse-unweighted"],
                     "terms": "constant environment radiance 1, albedo 1, no sun, stochastic specular transport"},
    "furnace-resolve": {"kind": "furnace", "controls": ["diffuse-unweighted"], "reflections": "off",
                        "declared": ["declared-diffuse-unweighted", "declared-karis-fit", "declared-diffuse-f0-fresnel"],
                        "terms": "constant environment radiance 1, albedo 1, no sun, deterministic resolve terms"},
    "base-grid-env-resolve": {"kind": "declared", "controls": ["declared-diffuse-unweighted", "declared-karis-fit", "declared-diffuse-f0-fresnel"], "reflections": "off",
                              "terms": "constant environment only (radiance 0.5, no sun), deterministic resolve terms "
                                       "(--reflections off --lighting-denoise custom)"},
    "roughness-extremes-sun": {"kind": "direct", "controls": ["alpha-roughness", "fresnel-off", "engine-f0-blend"],
                               "terms": "sun disk only (environment 0, no shadows, no GI) on the roughness extremes 0 and 1"},
    "roughness-extremes": {"kind": "environment", "controls": ["alpha-roughness"],
                           "terms": "sun disk plus constant environment 0.3; only the environment criterion gates"},
    # F15.1 (ticket 05): KHR_materials_ior x KHR_materials_specular; asserted green in every mode.
    "specular-grid-sun": {"kind": "direct", "always_green": True,
                          "controls": ["specular-ignored", "f90-one", "specular-weight-not-on-f0"],
                          "terms": "sun disk only (environment 0, no shadows, no GI) on the ior x specular grid"},
    "specular-grid-env-resolve": {"kind": "declared", "always_green": True, "reflections": "off",
                                  "controls": ["declared-specular-ignored", "declared-f90-one",
                                               "declared-specular-weight-not-on-f0", "declared-extension-diffuse-base"],
                                  "terms": "constant environment only (radiance 0.5, no sun) on the ior x specular grid, "
                                           "deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    # F15.1 (ticket 07): the ior x specular grid lit by one point light (bench scenario specular-grid-point; a
    # key "<bench>@<lighting>" is a lighting variant of the same bench scenario).  legacy: the resolve's local
    # lights, unshadowed (reference without visibility).  restir / clustered: DI lights with RT visibility
    # (reference shadow rays), captured unfiltered (--lighting-denoise off: the gate measures the DI estimator;
    # the custom denoiser blurs the sharp highlights by up to 18 % per pixel, measured 2026-11-13, and is
    # ticket 08's), terminator band 2 px (RT shadow-ray offsets on the tessellated facets).  Before ticket 07
    # these two variants carried "pending" / "pending_matches" (DI evaluated the base BRDF, so the gate failed and
    # the image matched the specular-ignored control); DI now reads the lobe guide and they must pass.
    "specular-grid-point": {"kind": "direct", "always_green": True, "lighting": "legacy", "visibility": "none",
                            "controls": ["specular-ignored", "f90-one", "specular-weight-not-on-f0"],
                            "terms": "one point light only (no sun, environment 0, no GI), resolve local lights "
                                     "(--lighting legacy), on the ior x specular grid"},
    "specular-grid-point@restir": {"kind": "direct", "lighting": "restir", "visibility": "rays", "denoise": "off",
                                   "terminator_band": 2,
                                   "controls": ["specular-ignored", "f90-one", "specular-weight-not-on-f0"],
                                   "terms": "one point light only, ReSTIR DI with RT visibility (--lighting restir "
                                            "--rt on), unfiltered (--lighting-denoise off), on the ior x specular grid"},
    # F15.1 (ticket 08): specular transport of ior/specular.  specular-grid-env: SSR (like base-grid-env, kind
    # "environment": the reference is the literal quadrature, mean <= 5 % per sphere); @rt: RT reflections;
    # @probes: the deterministic prefiltered-probe path, which with no captured probe is the environment split-sum
    # weight of the extended lobe, so it gates against the declared model (kind "declared"), captured unfiltered
    # (--lighting-denoise off: the custom denoiser lowers the near-unit specular signal of the ior 0 spheres by
    # 0.74 % at the sphere centre, measured 2026-11-13, ticket 08b's; the gate measures the transport weight).
    "specular-grid-env": {"kind": "environment", "always_green": True, "controls": ['specular-ignored', 'f90-one', 'specular-weight-not-on-f0'],
                          "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the "
                                   "ior x specular grid"},
    "specular-grid-env@rt": {"kind": "environment", "always_green": True, "rt": True, "reflections": "rt",
                             "controls": ['specular-ignored', 'f90-one', 'specular-weight-not-on-f0'],
                             "terms": "constant environment only (radiance 0.5, no sun), RT reflections (--rt on "
                                      "--reflections rt), on the ior x specular grid"},
    "specular-grid-env@probes": {"kind": "declared", "always_green": True, "reflections": "probes", "denoise": "off",
                                 "controls": ['declared-specular-ignored', 'declared-f90-one', 'declared-specular-weight-not-on-f0', 'declared-extension-diffuse-base'],
                                 "terms": "constant environment only (radiance 0.5, no sun), probe path "
                                          "(--reflections probes, no captured probe: environment split-sum weight), "
                                          "on the ior x specular grid"},
    # Extended furnace (albedo 1, environment 1): the transport gates the per-sphere mean (energy in expectation),
    # @resolve the deterministic resolve terms per pixel plus the declared model.
    "furnace-specular": {"kind": "furnace-transport", "always_green": True, "controls": ["diffuse-unweighted"],
                         "terms": "constant environment radiance 1, albedo 1, no sun, stochastic specular transport "
                                  "on the ior x specular grid"},
    "furnace-specular@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                                 "controls": ["diffuse-unweighted"],
                                 "declared": ["declared-karis-fit", "declared-extension-diffuse-base"],
                                 "terms": "constant environment radiance 1, albedo 1, no sun, deterministic resolve "
                                          "terms on the ior x specular grid"},
    # F15.3 (ticket 09): KHR_materials_clearcoat.  Written test-first (every gate the coat-blind engine could fail was
    # "pending" and asserted red with the coat-ignored control as the matching image); the engine side landed with
    # F15.1 ticket 09, the pending markers are gone and every gate must pass.  The coat grid
    # rows are coat factor {0, 0.25, 0.5, 1}, coat 1 over a base of ior 2.4 (the only row where "coat F0 from the
    # material ior" differs) and a metal under a full coat; the columns are coat roughness {0, 0.05, 0.15, 0.4, 1}
    # over a rough dielectric base (roughness 0.6).  Direct grids group their rows by (metallic, ior, coat factor).
    "clearcoat-grid-sun": {"kind": "direct",
                           "controls": ["coat-ignored", "coat-f0-from-ior", "coat-fresnel-vh", "coat-base-unattenuated"],
                           "terms": "sun disk only (environment 0, no shadows, no GI) on the coat factor x coat roughness grid"},
    "clearcoat-grid-env-resolve": {"kind": "declared",
                                   "reflections": "off",
                                   "controls": ["declared-coat-ignored", "declared-coat-f0-from-ior",
                                                "declared-coat-base-unattenuated"],
                                   "terms": "constant environment only (radiance 0.5, no sun) on the coat grid, deterministic "
                                            "resolve terms (--reflections off --lighting-denoise custom)"},
    "clearcoat-grid-env": {"kind": "environment",
                           "controls": ["coat-ignored", "coat-f0-from-ior", "coat-base-unattenuated"],
                           "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the coat grid"},
    "clearcoat-grid-point": {"kind": "direct",
                             "lighting": "legacy", "visibility": "none",
                             "controls": ["coat-ignored", "coat-f0-from-ior", "coat-fresnel-vh", "coat-base-unattenuated"],
                             "terms": "one point light only (no sun, environment 0, no GI), resolve local lights "
                                      "(--lighting legacy), on the coat grid"},
    "clearcoat-grid-point@restir": {"kind": "direct",
                                    "lighting": "restir", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                    "controls": ["coat-ignored", "coat-f0-from-ior", "coat-fresnel-vh",
                                                 "coat-base-unattenuated"],
                                    "terms": "one point light only, ReSTIR DI with RT visibility (--lighting restir --rt on), "
                                             "unfiltered (--lighting-denoise off), on the coat grid"},
    "clearcoat-grid-point@clustered": {"kind": "direct",
                                       "lighting": "clustered", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                       "controls": ["coat-ignored", "coat-f0-from-ior", "coat-fresnel-vh",
                                                    "coat-base-unattenuated"],
                                       "terms": "one point light only, clustered DI with RT visibility (--lighting clustered "
                                                "--rt on), unfiltered (--lighting-denoise off), on the coat grid"},
    # Furnace over coated materials (albedo 1, environment 1).  The furnace bound is an UPPER bound that today's engine
    # (no coat) satisfies trivially, so the engine-pixel gates are asserted green in every mode: they guard the
    # implementation against energy gain (the unattenuated-base control must fail them).  The declared-model gate of
    # the deterministic resolve is the one that is red today ("pending_gate": "declared"): a coat redistributes
    # energy, the declared coat model differs from the uncoated pixels by more than the 0.5 % mean bound.
    "furnace-clearcoat": {"kind": "furnace-transport", "always_green": True, "controls": ["coat-base-unattenuated"],
                          "terms": "constant environment radiance 1, albedo 1, no sun, stochastic specular transport on the "
                                   "coat grid"},
    "furnace-clearcoat@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                                  "controls": ["coat-base-unattenuated"],
                                  "declared": ["declared-coat-ignored", "declared-coat-base-unattenuated"],
                                  "terms": "constant environment radiance 1, albedo 1, no sun, deterministic resolve terms "
                                           "on the coat grid"},
    # F15.3 (ticket 10): KHR_materials_sheen.  Rows are the sheen colour {0, white, orange, blue}, columns the sheen
    # roughness {0.1, 0.3, 0.5, 0.75, 1} over a dark rough dielectric (grey 0.1, roughness 0.7).  Direct grids group their
    # rows by (metallic, sheen colour).  The engine side landed with ticket 10 (the keys were pending until then).  The sun and
    # point controls include the view-only scaling (`sheen-cheap-scaling`): for direct light the reference uses the
    # notes' min(E(N.V), E(N.L)) form, which differs from the cheap form near the terminator; the environment forms use the
    # cheap form for both, so that control has no environment key.  Sun: the Charlie lobe and the scaling are
    # point-evaluated at the disk centre (documented in the reference).
    "sheen-grid-sun": {"kind": "direct",
                       "controls": ["sheen-ignored", "sheen-unscaled-base", "sheen-ashikhmin-visibility",
                                    "sheen-roughness-linear", "sheen-cheap-scaling"],
                       "terms": "sun disk only (environment 0, no shadows, no GI) on the sheen colour x sheen roughness grid"},
    "sheen-grid-env-resolve": {"kind": "declared", "reflections": "off",
                               "controls": ["declared-sheen-ignored", "declared-sheen-unscaled-base",
                                            "declared-sheen-roughness-linear"],
                               "terms": "constant environment only (radiance 0.5, no sun) on the sheen grid, deterministic "
                                        "resolve terms (--reflections off --lighting-denoise custom)"},
    "sheen-grid-env": {"kind": "environment",
                       "controls": ["sheen-ignored", "sheen-unscaled-base", "sheen-ashikhmin-visibility",
                                    "sheen-roughness-linear"],
                       "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport (sheen is not "
                                "sampled: the environment term of the composition), on the sheen grid"},
    # The probe path with the built-in analytic probe (a constant environment, no captured probe): the composition's
    # sheen term on the probe-lit path.  A fixture with a captured probe is open (see the ticket 10 comments).
    "sheen-grid-env@probes": {"kind": "declared", "reflections": "probes", "denoise": "off",
                              "controls": ["declared-sheen-ignored", "declared-sheen-unscaled-base",
                                           "declared-sheen-roughness-linear"],
                              "terms": "constant environment only (radiance 0.5, no sun), probe path (--reflections probes, "
                                       "analytic constant probe, unfiltered), on the sheen grid"},
    # ticket 10 (F15.3 closure): a COOKED probe of constant radiance 0.2 (written by capture_command) lights the specular and sheen terms while the declared
    # environment is 0.5; black base (no diffuse).  The reference runs with --sky-override 0.2.  The control that lights the sheen term with the declared 0.5
    # (the engine before the fix, decision 06 violated) must fail.
    "sheen-probe-env": {"kind": "declared", "reflections": "probes", "lighting": "clustered", "denoise": "off", "probe_constant": 0.2,
                        "reference_args": ["--sky-override", "0.2"],
                        "controls": ["declared-sheen-ignored", "declared-sheen-declared-radiance"],
                        "terms": "cooked constant probe (radiance 0.2) with the declared environment 0.5, black base, probe path (--reflections probes), sheen grid"},
    "sheen-grid-point": {"kind": "direct", "lighting": "legacy", "visibility": "none",
                         "controls": ["sheen-ignored", "sheen-unscaled-base", "sheen-ashikhmin-visibility",
                                      "sheen-roughness-linear", "sheen-cheap-scaling"],
                         "terms": "one point light only (no sun, environment 0, no GI), resolve local lights "
                                  "(--lighting legacy), on the sheen grid"},
    "sheen-grid-point@restir": {"kind": "direct", "lighting": "restir", "visibility": "rays", "denoise": "off",
                                "terminator_band": 2,
                                "controls": ["sheen-ignored", "sheen-unscaled-base", "sheen-ashikhmin-visibility",
                                             "sheen-roughness-linear", "sheen-cheap-scaling"],
                                "terms": "one point light only, ReSTIR DI with RT visibility (--lighting restir --rt on), "
                                         "unfiltered (--lighting-denoise off), on the sheen grid"},
    "sheen-grid-point@clustered": {"kind": "direct", "lighting": "clustered", "visibility": "rays", "denoise": "off",
                                   "terminator_band": 2,
                                   "controls": ["sheen-ignored", "sheen-unscaled-base", "sheen-ashikhmin-visibility",
                                                "sheen-roughness-linear", "sheen-cheap-scaling"],
                                   "terms": "one point light only, clustered DI with RT visibility (--lighting clustered "
                                            "--rt on), unfiltered (--lighting-denoise off), on the sheen grid"},
    # Furnace over sheen materials (base albedo 1, environment 1): an upper bound asserted green in every mode (the
    # unscaled-base control must fail it); the declared gate of the deterministic resolve checks the redistribution
    # S E + base (1 - max3(S) E) against the declared model.
    "furnace-sheen": {"kind": "furnace-transport", "always_green": True, "controls": ["sheen-unscaled-base"],
                      "terms": "constant environment radiance 1, base albedo 1, no sun, stochastic specular transport on the "
                               "sheen grid"},
    "furnace-sheen@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                              "controls": ["sheen-unscaled-base"],
                              "declared": ["declared-sheen-ignored", "declared-sheen-unscaled-base"],
                              "terms": "constant environment radiance 1, base albedo 1, no sun, deterministic resolve terms "
                                       "on the sheen grid"},
    # F15.3 (ticket 11): KHR_materials_anisotropy.  Rows are the strength {0, 0.5, 1} over a copper dielectric and then a
    # copper metal, columns the rotation {0, pi/4, pi/2, -pi/4}, base roughness 0.3, on COMBED spheres (uniform tangent field
    # facing the camera, handedness -1, tangents from the mesh attribute of the snapshot).  Direct grids group their rows by
    # (metallic, strength).  The engine side landed with ticket 11 (the keys were pending until then).  Direct controls: ignored, rotation
    # flipped, handedness ignored (the same lobe as the rotation flip on a mesh of handedness -1: the mirrored fixture
    # separates them), alpha_t / alpha_b swapped, D normalised by alpha_b^2 (gains energy).  The exact environment key uses
    # the same five controls (the per-sphere mean over a combed sphere sees the lobe orientation: measured); the declared
    # environment key (split sum at an effective roughness, see the reference) requires `declared-aniso-ignored` and
    # `-alpha-t` (the declared fit is independent of rotation, handedness and axis swap).
    "aniso-grid-sun": {"kind": "direct",
                       "controls": ANISO_DIRECT_CONTROLS,
                       "terms": "sun disk only (environment 0, no shadows, no GI) on the strength x rotation grid"},
    "aniso-narrow-sun": {"kind": "direct", "controls": ["aniso-ignored", "aniso-rotation-flipped", "aniso-alpha-swapped"],
                         "reference_args": ["--exact-aniso-disk"],
                         "terms": "sun disk only on the strength x rotation grid with NARROW lobes (alpha_b 0.0064 .. 0.04); the reference integrates the disk "
                                  "exactly (polar grid, no quadrature mask) instead of the engine's spiral rule"},
    "aniso-grid-env-resolve": {"kind": "declared",
                               "reflections": "off", "controls": ["declared-aniso-ignored", "declared-aniso-alpha-t"],
                               "terms": "constant environment only (radiance 0.5, no sun) on the anisotropy grid, deterministic "
                                        "resolve terms (--reflections off --lighting-denoise custom)"},
    "aniso-grid-env": {"kind": "environment",
                       "controls": ANISO_DIRECT_CONTROLS,
                       "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport (anisotropic VNDF "
                                "sampling), on the anisotropy grid"},
    "aniso-grid-point": {"kind": "direct",
                         "lighting": "legacy", "visibility": "none", "controls": ANISO_DIRECT_CONTROLS,
                         "terms": "one point light only (no sun, environment 0, no GI), resolve local lights "
                                  "(--lighting legacy), on the anisotropy grid"},
    "aniso-grid-point@restir": {"kind": "direct",
                                "lighting": "restir", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                "controls": ANISO_DIRECT_CONTROLS,
                                "terms": "one point light only, ReSTIR DI with RT visibility (--lighting restir --rt on), "
                                         "unfiltered (--lighting-denoise off), on the anisotropy grid"},
    "aniso-grid-point@clustered": {"kind": "direct",
                                   "lighting": "clustered", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                   "controls": ANISO_DIRECT_CONTROLS,
                                   "terms": "one point light only, clustered DI with RT visibility (--lighting clustered "
                                            "--rt on), unfiltered (--lighting-denoise off), on the anisotropy grid"},
    # Furnace over anisotropic materials (base albedo 1, environment 1): an upper bound, asserted green in every mode (the
    # energy-gaining control must fail it).
    "furnace-aniso": {"kind": "furnace-transport", "always_green": True, "controls": ["aniso-d-normalised-by-ab"],
                      "terms": "constant environment radiance 1, base albedo 1, no sun, stochastic specular transport on the "
                               "anisotropy grid"},
    "furnace-aniso@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                              "controls": ["aniso-d-normalised-by-ab"],
                              "declared": ["declared-aniso-ignored", "declared-aniso-alpha-t"],
                              "terms": "constant environment radiance 1, base albedo 1, no sun, deterministic resolve terms "
                                       "on the anisotropy grid"},
    # The mirrored carrier fixture (sun only): columns = (plain, mirrored) x (handedness -1, +1) at rotation pi/4.  The
    # mirrored instances carry INSTANCE_FLAG_MIRRORED; the reference puts the lobe on the world image of the model-space
    # bitangent (w sign(det)), so rotation sign, handedness w and the determinant each flip the lobe on their own.
    "aniso-mirrored-sun": {"kind": "direct",
                           "controls": ["aniso-ignored", "aniso-rotation-flipped", "aniso-bitangent-flipped",
                                        "aniso-alpha-swapped", "aniso-mirror-unaware", "aniso-d-normalised-by-ab"],
                           "terms": "sun disk only on plain and mirrored combed spheres of both tangent handedness values"},
    # F15.3 (ticket 12): KHR_materials_iridescence.  Rows are (factor, film ior) {(0.5, 1.3), (1, 1.3), (1, 1.8)} over a copper
    # dielectric and then a copper metal, columns the thickness {0, 200, 400, 600, 800, 1200} nm, base roughness 0.3.  Direct
    # grids group their rows by (metallic, factor, ior).  The engine side landed with ticket 12 (the keys were "pending" with the iridescence-ignored
    # reference as the matching image until then).  Controls:
    # ignored, film ior = 1.0 (outside ior), film ior = the default 1.3 (differs on the 1.8 rows only), N.V instead of V.H (the
    # Sample Renderer deviation), the minimum instead of the maximum thickness, diffuse not scaled by 1 - max F_irid (gains
    # energy).  `iridescence-thickness-max` of the ticket text cannot differ on texture-free data (open for ticket 22).  The
    # declared environment key (split sum with the film at N.V) has no at-nv control: the declared model is N.V by construction.
    "iridescence-grid-sun": {"kind": "direct",
                             "controls": IRID_DIRECT_CONTROLS,
                             "terms": "sun disk only (environment 0, no shadows, no GI) on the factor x thickness grid"},
    "iridescence-grid-env-resolve": {"kind": "declared", "reflections": "off", "controls": IRID_DECLARED_CONTROLS,
                                     "terms": "constant environment only (radiance 0.5, no sun) on the iridescence grid, "
                                              "deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "iridescence-grid-env": {"kind": "environment",
                             "controls": IRID_DIRECT_CONTROLS,
                             "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport (the weight uses "
                                      "the film at the sampled V.H), on the iridescence grid"},
    "iridescence-grid-point": {"kind": "direct",
                               "lighting": "legacy", "visibility": "none", "controls": IRID_DIRECT_CONTROLS,
                               "terms": "one point light only (no sun, environment 0, no GI), resolve local lights "
                                        "(--lighting legacy), on the iridescence grid"},
    "iridescence-grid-point@restir": {"kind": "direct",
                                      "lighting": "restir", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                      "controls": IRID_DIRECT_CONTROLS,
                                      "terms": "one point light only, ReSTIR DI with RT visibility (--lighting restir --rt on), "
                                               "unfiltered (--lighting-denoise off), on the iridescence grid"},
    "iridescence-grid-point@clustered": {"kind": "direct",
                                         "lighting": "clustered", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                         "controls": IRID_DIRECT_CONTROLS,
                                         "terms": "one point light only, clustered DI with RT visibility (--lighting clustered "
                                                  "--rt on), unfiltered (--lighting-denoise off), on the iridescence grid"},
    # Furnace over iridescent materials (base albedo 1, environment 1): an upper bound (the engine-side gates 1.005, orchestrator
    # decision: the literal Appendix B dielectric gains energy at grazing angles by itself), asserted green in every mode (the
    # diffuse-unscaled control gains energy and must fail it).
    "furnace-iridescence": {"kind": "furnace-transport", "always_green": True, "controls": ["iridescence-diffuse-unweighted"],
                            "terms": "constant environment radiance 1, base albedo 1, no sun, stochastic specular transport on "
                                     "the iridescence grid"},
    "furnace-iridescence@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                                    "controls": ["iridescence-diffuse-unweighted"],
                                    "declared": IRID_DECLARED_CONTROLS,
                                    "terms": "constant environment radiance 1, base albedo 1, no sun, deterministic resolve "
                                             "terms on the iridescence grid"},
    # F15.3 (ticket 16): thin subsurface (diffuse transmission).  Rows are the weight {0, 0.25, 0.5, 1} (no weight-0 row on the
    # point-light quads: nothing reaches them), columns the transmission colour {white, warm, teal}, over a blue-grey base.  Direct
    # grids group their rows by (metallic, weight).  The engine side landed with ticket 16 (the keys were "pending" with the
    # transmission-ignored reference as the matching image until then): every gate must pass.  The reference's shadow rays start on
    # the side of the light (a thin carrier does not occlude its own transmitted light), as the engine's DI visibility rays do.  The
    # declared environment keys have no fresnel-ignored control (the declared model has no V.H).  The furnace gates are asserted green.
    "subsurface-thin-sun": {"kind": "direct",
                            "controls": THIN_SUN_CONTROLS,
                            "terms": "sun disk only BEHIND the spheres (environment 0, no shadows, no GI) on the weight x colour grid"},
    "subsurface-thin-env-resolve": {"kind": "declared",
                                    "reflections": "off", "controls": THIN_DECLARED_CONTROLS,
                                    "terms": "constant environment only (radiance 0.5, no sun) on the thin subsurface grid, "
                                             "deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "subsurface-thin-env": {"kind": "environment",
                            "controls": THIN_SUN_CONTROLS,
                            "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport (the back-side "
                                     "environment of the composition), on the thin subsurface grid"},
    "subsurface-thin-point": {"kind": "direct",
                              "lighting": "legacy", "visibility": "none", "controls": THIN_POINT_CONTROLS,
                              "terms": "one point light BEHIND single-sided quads only (no sun, environment 0, no GI), resolve "
                                       "local lights (--lighting legacy)"},
    "subsurface-thin-point@restir": {"kind": "direct",
                                     "lighting": "restir", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                     "controls": THIN_POINT_CONTROLS,
                                     "terms": "one point light behind the quads only, ReSTIR DI with RT visibility "
                                              "(--lighting restir --rt on), unfiltered (--lighting-denoise off)"},
    "subsurface-thin-point@clustered": {"kind": "direct",
                                        "lighting": "clustered", "visibility": "rays", "denoise": "off", "terminator_band": 2,
                                        "controls": THIN_POINT_CONTROLS,
                                        "terms": "one point light behind the quads only, clustered DI with RT visibility "
                                                 "(--lighting clustered --rt on), unfiltered (--lighting-denoise off)"},
    # Furnace over thin materials (white base, white transmission colour, environment 1): an upper bound that the blind engine
    # satisfies, asserted green in every mode (the double-counted control gains energy and must fail it); only the declared gate
    # (the declared model moves energy from the diffuse to the transmitted lobe).
    "furnace-thin-subsurface": {"kind": "furnace-transport", "always_green": True, "controls": ["transmission-double-counted"],
                                "terms": "constant environment radiance 1, white base and transmission colour, no sun, stochastic "
                                         "specular transport on the thin subsurface grid"},
    "furnace-thin-subsurface@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                                        "controls": ["transmission-double-counted"], "declared": THIN_FURNACE_DECLARED_CONTROLS,
                                        "terms": "constant environment radiance 1, white base and transmission colour, no sun, "
                                                 "deterministic resolve terms on the thin subsurface grid"},
    # --- F15.3 (ticket 19): critical lobe combinations (generated by the table below; see the constants COMBO_*) ---
    "combo-coat-sheen-sun": {"kind": "direct", "controls": COMBO_COAT_SHEEN_CONTROLS,
                                     "terms": "sun disk only (environment 0, no shadows, no GI) on the clearcoat over sheen grid"},
    "combo-coat-sheen-env": {"kind": "environment", "controls": COMBO_COAT_SHEEN_CONTROLS,
                                     "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the clearcoat over sheen grid"},
    "combo-coat-sheen-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_COAT_SHEEN_DECLARED,
                                             "terms": "constant environment only (radiance 0.5, no sun) on the clearcoat over sheen grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-aniso-coat-sun": {"kind": "direct", "controls": COMBO_ANISO_COAT_CONTROLS,
                                     "terms": "sun disk only (environment 0, no shadows, no GI) on the anisotropy + clearcoat grid"},
    "combo-aniso-coat-env": {"kind": "environment", "controls": COMBO_ANISO_COAT_ENV_CONTROLS,
                                     "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the anisotropy + clearcoat grid"},
    "combo-aniso-coat-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_ANISO_COAT_DECLARED,
                                             "terms": "constant environment only (radiance 0.5, no sun) on the anisotropy + clearcoat grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-aniso-panel-sun": {"kind": "direct", "controls": COMBO_ANISO_PANEL_CONTROLS,
                                      "terms": "sun disk only (environment 0, no shadows, no GI) on the anisotropy panels (yaw 50, roll 20 degrees) under a coat, a sheen and alone"},
    "combo-aniso-panel-env": {"kind": "environment", "controls": COMBO_ANISO_PANEL_ENV_CONTROLS,
                                      "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the anisotropy panels (yaw 50, roll 20 degrees)"},
    "combo-thin-lobes-sun": {"kind": "direct", "controls": COMBO_THIN_LOBES_CONTROLS,
                             "terms": "sun disk only (environment 0, no shadows, no GI) from the side on the thin subsurface under sheen / anisotropy / iridescence / ior grid"},
    "combo-thin-lobes-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_THIN_LOBES_DECLARED,
                                     "reference_args": ["--no-exact-environment"],
                                     "terms": "constant environment only (radiance 0.5, no sun) on the thin subsurface under sheen / anisotropy / iridescence / ior grid, "
                                              "deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-aniso-sheen-sun": {"kind": "direct", "controls": COMBO_ANISO_SHEEN_CONTROLS,
                                      "terms": "sun disk only (environment 0, no shadows, no GI) on the anisotropy + sheen grid"},
    "combo-aniso-sheen-env": {"kind": "environment", "controls": COMBO_ANISO_SHEEN_ENV_CONTROLS,
                                      "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the anisotropy + sheen grid"},
    "combo-aniso-sheen-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_ANISO_SHEEN_DECLARED,
                                              "terms": "constant environment only (radiance 0.5, no sun) on the anisotropy + sheen grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-specular-sheen-sun": {"kind": "direct", "controls": COMBO_SPECULAR_SHEEN_CONTROLS,
                                         "terms": "sun disk only (environment 0, no shadows, no GI) on the ior/specular + sheen grid"},
    "combo-specular-sheen-env": {"kind": "environment", "controls": COMBO_SPECULAR_SHEEN_CONTROLS,
                                         "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the ior/specular + sheen grid"},
    "combo-specular-sheen-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_SPECULAR_SHEEN_DECLARED,
                                                 "terms": "constant environment only (radiance 0.5, no sun) on the ior/specular + sheen grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-irid-coat-sun": {"kind": "direct", "controls": COMBO_IRID_COAT_CONTROLS,
                                    "terms": "sun disk only (environment 0, no shadows, no GI) on the iridescence + clearcoat grid"},
    "combo-irid-coat-env": {"kind": "environment", "controls": COMBO_IRID_COAT_CONTROLS,
                                    "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the iridescence + clearcoat grid"},
    "combo-irid-coat-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_IRID_COAT_DECLARED,
                                            "terms": "constant environment only (radiance 0.5, no sun) on the iridescence + clearcoat grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-irid-sheen-sun": {"kind": "direct", "controls": COMBO_IRID_SHEEN_CONTROLS,
                                     "terms": "sun disk only (environment 0, no shadows, no GI) on the iridescence + sheen grid"},
    "combo-irid-sheen-env": {"kind": "environment", "controls": COMBO_IRID_SHEEN_CONTROLS,
                                     "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the iridescence + sheen grid"},
    "combo-irid-sheen-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_IRID_SHEEN_DECLARED,
                                             "terms": "constant environment only (radiance 0.5, no sun) on the iridescence + sheen grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-irid-aniso-sun": {"kind": "direct", "controls": COMBO_IRID_ANISO_CONTROLS,
                                     "terms": "sun disk only (environment 0, no shadows, no GI) on the iridescence + anisotropy grid"},
    "combo-irid-aniso-env": {"kind": "environment", "controls": COMBO_IRID_ANISO_ENV_CONTROLS,
                                     "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the iridescence + anisotropy grid"},
    "combo-irid-aniso-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_IRID_ANISO_DECLARED,
                                             "terms": "constant environment only (radiance 0.5, no sun) on the iridescence + anisotropy grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-irid-specular-sun": {"kind": "direct", "controls": COMBO_IRID_SPECULAR_CONTROLS,
                                        "terms": "sun disk only (environment 0, no shadows, no GI) on the iridescence + ior/specular grid"},
    "combo-irid-specular-env": {"kind": "environment", "controls": COMBO_IRID_SPECULAR_ENV_CONTROLS,
                                        "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the iridescence + ior/specular grid"},
    "combo-irid-specular-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_IRID_SPECULAR_DECLARED,
                                                "terms": "constant environment only (radiance 0.5, no sun) on the iridescence + ior/specular grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-thin-coat-sun": {"kind": "direct", "controls": COMBO_THIN_COAT_CONTROLS,
                                    "terms": "sun disk only (environment 0, no shadows, no GI) on the thin subsurface + clearcoat grid"},
    "combo-thin-coat-env": {"kind": "environment", "controls": COMBO_THIN_COAT_ENV_CONTROLS,
                                    "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the thin subsurface + clearcoat grid"},
    "combo-thin-coat-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_THIN_COAT_DECLARED,
                                            "terms": "constant environment only (radiance 0.5, no sun) on the thin subsurface + clearcoat grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "combo-all-lobes-sun": {"kind": "direct", "controls": COMBO_ALL_LOBES_CONTROLS,
                                    "terms": "sun disk only (environment 0, no shadows, no GI) on the ior/specular + clearcoat + sheen + anisotropy grid"},
    "combo-all-lobes-env": {"kind": "environment", "controls": COMBO_ALL_LOBES_CONTROLS,
                                    "terms": "constant environment only (radiance 0.5, no sun), SSR specular transport, on the ior/specular + clearcoat + sheen + anisotropy grid"},
    "combo-all-lobes-env@resolve": {"kind": "declared", "reflections": "off", "controls": COMBO_ALL_LOBES_DECLARED,
                                            "terms": "constant environment only (radiance 0.5, no sun) on the ior/specular + clearcoat + sheen + anisotropy grid, deterministic resolve terms (--reflections off --lighting-denoise custom)"},
    "furnace-combinations": {"kind": "furnace-transport", "always_green": True, "controls": ["additive-layering"],
                "terms": "constant environment radiance 1, white base, no sun, stochastic specular transport on every implemented lobe combination"},
    "furnace-combinations@resolve": {"kind": "furnace", "always_green": True, "reflections": "off",
                "controls": ["additive-layering"], "declared": ["declared-additive-layering"],
                "terms": "constant environment radiance 1, white base, no sun, deterministic resolve terms on every implemented lobe combination"},
    # --- end of the ticket 19 combination keys ---
    # F15.3 (ticket 13): KHR_materials_transmission + KHR_materials_volume, ray-traced (decision 09, ADR-0004).  Written
    # test-first (the keys were "pending" with the `transmission-ignored` opaque dielectric as the matching image until the engine side
    # landed with F15.1 ticket 13): every gate must pass.  All RT keys
    # trace the glass with RT whatever the reflection band (--rt on --reflections rt); the engine image is the mean of
    # frames 96..127 (--capture-linear-accumulate).  Only the controls that change the fixture are listed (a control
    # identical to the reference would be non-discriminating: e.g. `transmission-not-tinted` needs a tinted base).
    "transmission-roughness-ior": {"kind": "transmission",
                                   "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                   "controls": ["transmission-ignored", "ior-ignored"],
                                   "terms": "constant environment 0.5 + emissive checkerboard behind 12 clear volume spheres "
                                            "(roughness {0, .1, .3, .6} x ior {1.33, 1.5, 2.0}), RT glass transport"},
    "transmission-attenuation": {"kind": "transmission",
                                 "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                 "controls": ["transmission-ignored", "volume-ignored", "ior-ignored"],
                                 "terms": "constant environment 0.5 + emissive checkerboard behind 6 absorbing volume spheres "
                                          "(attenuation colour x distance), RT glass transport"},
    "transmission-factor-tint": {"kind": "transmission",
                                 "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                 "controls": ["transmission-ignored", "transmission-not-tinted", "volume-ignored", "ior-ignored",
                                              "thin-as-volume"],
                                 "terms": "constant environment 0.5 + emissive checkerboard behind thin panes and volume slabs "
                                          "(factor {1, .5} x tint), RT glass transport"},
    "transmission-thin-volume": {"kind": "transmission",
                                 "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                 "controls": ["transmission-ignored", "volume-ignored", "ior-ignored", "thin-as-volume"],
                                 "terms": "constant environment 0.5 + emissive checkerboard behind a thin-walled sphere, a clear "
                                          "and an absorbing volume sphere, RT glass transport"},
    # F15 closure (ticket 13): the glTF tint rule (baseColor once per closed solid at its entering interface, at every thin-walled
    # surface) on TINTED closed bodies, and ROUGH thin-walled glass against `gltf_thin_rough` (the README's thin-walled specular_btdf).
    # Written fail-before: the engine of 3ef8f41 tinted the camera-visible entry only (a tinted thin sphere lost its second wall's
    # tint) and the reference tinted every transmitting event; thin-walled glass above roughness 0 was refused by the reference.
    "transmission-tinted-bodies": {"kind": "transmission",
                                   "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                   "controls": ["transmission-ignored", "tint-every-event", "transmission-not-tinted", "volume-ignored", "ior-ignored"],
                                   "terms": "constant environment 0.5 + emissive checkerboard behind TINTED closed bodies (smooth, rough "
                                            "and absorbing solid spheres, a cube, an absorbing slab) and a tinted thin-walled sphere, RT glass transport"},
    "transmission-thin-rough": {"kind": "transmission",
                                "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                "controls": ["transmission-ignored", "thin-rough-as-smooth", "thin-rough-refracted", "transmission-not-tinted"],
                                "terms": "constant environment 0.5 + emissive checkerboard behind rough thin-walled panes (roughness {0.1, 0.3, 0.6}, "
                                         "white and tinted), rough thin-walled spheres and a half-transmission pane, RT glass transport"},
    "transmission-single-sided": {"kind": "transmission",
                                  "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                  "controls": ["transmission-ignored", "backface-rejected", "ior-ignored", "volume-ignored"],
                                  "terms": "constant environment 0.5 + emissive checkerboard behind SINGLE-SIDED closed glass "
                                           "meshes (lens sphere, rotated cube, absorbing sphere), RT glass transport"},
    "transmission-glass-behind-glass": {"kind": "transmission",
                                        "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off",
                                        "controls": ["transmission-ignored", "back-glass-black", "ior-ignored", "volume-ignored"],
                                        "terms": "constant environment 0.5 + emissive checkerboard behind overlapping glass "
                                                 "(sphere behind sphere, pane in front of a sphere), RT glass transport"},
    "transmission-emissive-panel": {"kind": "transmission",
                                    "rt": True, "reflections": "rt", "accumulate": ACCUMULATE, "denoise": "off", "lighting": "restir",
                                    "visibility": "rays",
                                    "controls": ["transmission-ignored", "emitter-double-counted", "transmission-not-tinted"],
                                    "terms": "constant environment 0.5 + two emissive panels (radiance 3) behind a clear and a "
                                             "tinted thin pane, ReSTIR DI with RT visibility (--lighting restir --rt on): the "
                                             "panel radiance is counted once (transport), not again by DI"},
    # The declared --rt off model: no backdrop (probes/environment only), the probe path with the analytic constant probe.
    "transmission-rtoff": {"kind": "transmission-declared",
                           "reflections": "probes", "denoise": "off",
                           "controls": ["declared-transmission-ignored", "declared-not-tinted", "declared-volume-ignored"],
                           "terms": "constant environment 0.5 only (no backdrop), --rt off probe path: the declared model "
                                    "L ((1 - E)((1 - t) base + t tint T) + E) of tools/f15_transmission_reference.py"},
    # F15.3 (ticket 19): the critical transmission combinations of the declared --rt off model: iridescence + transmission + volume and
    # the transmission OVER the diffuse transmission (thin subsurface), rows = transmission factor {1, 0.5}.  The ray-traced Mitsuba
    # ground truth of a thin-film dielectric is NOT built (no thin-film BSDF plugin for the two interface sides): the closed form of
    # tools/f15_transmission_reference.py declared_pixel gates the --rt off path only.  Pending ("ticket 19": today's engine ignores
    # the transmission and keeps the film and the thin subsurface, the image of `declared-transmission-ignored`).
    "transmission-rtoff-combinations": {"kind": "transmission-declared", "reflections": "probes", "denoise": "off",
                                        "controls": ["declared-transmission-ignored", "declared-iridescence-ignored",
                                                     "declared-diffuse-transmission-ignored",
                                                     "declared-diffuse-transmission-not-overridden", "declared-volume-ignored"],
                                        "terms": "constant environment 0.5 only (no backdrop), --rt off probe path: the declared model with the "
                                                 "film (iridescence + transmission + volume) and the transmission replacing the diffuse "
                                                 "transmission, tools/f15_transmission_reference.py declared_pixel"},
    # F15.3 (tickets 14/15): attenuated coloured shadows through transmissive casters (decision 06 shadow casters, ADR-0004 amendment).
    # Written test-first: the engine shadowed opaquely, so every gated key was "pending" (asserted to FAIL, its image equal to
    # `transmittance-ignored`).  The RT and DI keys (ticket 14) landed with F15.1 ticket 14 and the CSM keys (ticket 15) with F15.1 ticket 15:
    # every key is gated now.  --reflections off: the floor is lit by the light only (no SSR of the lit floor into
    # the shadow).  The sun keys exist as @rt (ticket 14) and @csm (ticket 15); the CSM keys share the bench scene and the RT reference.
    "transmissive-shadow-pane-sun@rt": {"kind": "transmissive-shadow",
                                        "rt": True, "shadows": "rt", "reflections": "off", "accumulate": SHADOW_ACCUMULATE,
                                        "envelope": "rt", "controls": SHADOW_PANE_CONTROLS,
                                        "terms": "sun 25 degrees from the pane normal, no environment, RT sun shadows (--shadows rt): "
                                                 "thin clear pane, thin green pane, thick clear pane, opaque slab over a diffuse floor"},
    "transmissive-shadow-pane-sun@csm": {"kind": "transmissive-shadow-ratio",
                                         "shadows": "csm", "reflections": "off", "accumulate": SHADOW_ACCUMULATE,
                                         "envelope": "csm", "controls": SHADOW_PANE_CONTROLS,
                                         "terms": "the pane fixture with the CSM path (--shadows csm, --rt off): single-layer transmittance map"},
    "transmissive-shadow-water-sun@rt": {"kind": "transmissive-shadow",
                                         "rt": True, "shadows": "rt", "reflections": "off", "accumulate": SHADOW_ACCUMULATE,
                                         "envelope": "rt", "controls": SHADOW_WATER_CONTROLS,
                                         "terms": "sun 20 degrees from the normal, RT sun shadows: clear and murky flat shallow water "
                                                  "(ior 1.33 slabs 0.12 m), opaque slab"},
    "transmissive-shadow-water-sun@csm": {"kind": "transmissive-shadow-ratio",
                                          "shadows": "csm", "reflections": "off", "accumulate": SHADOW_ACCUMULATE,
                                          "envelope": "csm", "controls": SHADOW_WATER_CONTROLS,
                                          "terms": "the shallow water fixture with the CSM path"},
    "transmissive-shadow-volume-pane@rt": {"kind": "transmissive-shadow",
                                           "rt": True, "shadows": "rt", "reflections": "off", "accumulate": SHADOW_ACCUMULATE,
                                           "envelope": "rt", "controls": SHADOW_VOLUME_CONTROLS,
                                           "terms": "sun 20 degrees from the normal, RT sun shadows: three thick glass panes (0.1 m) with "
                                                    "volume attenuation (colour x distance)"},
    "transmissive-shadow-volume-pane@csm": {"kind": "transmissive-shadow-ratio",
                                            "shadows": "csm", "reflections": "off", "accumulate": SHADOW_ACCUMULATE,
                                            "envelope": "csm", "controls": SHADOW_VOLUME_CONTROLS,
                                            "terms": "the attenuating thick pane fixture with the CSM path"},
    "transmissive-shadow-point@restir": {"kind": "transmissive-shadow",
                                         "lighting": "restir", "rt": True, "reflections": "off", "denoise": "off",
                                         "accumulate": SHADOW_ACCUMULATE, "envelope": "rt", "controls": SHADOW_POINT_CONTROLS,
                                         "terms": "one point light 4.5 m above the floor, ReSTIR DI with RT visibility (--lighting restir "
                                                  "--rt on), unfiltered: thin clear, thin green, thick clear pane and an opaque slab"},
    "transmissive-shadow-curved@rt": {"kind": "transmissive-shadow-curved", "rt": True, "shadows": "rt",
                                      "reflections": "off", "accumulate": SHADOW_ACCUMULATE, "envelope": "rt",
                                      "controls": ["shadow-ignored", "transmittance-ignored"],
                                      "reference_args": ["--no-ground-truth"],
                                      "terms": "sun 25 degrees from the vertical, RT sun shadows, clear and absorbing glass spheres: "
                                               "QUALITATIVE gate (0 < mean transmittance < 1, chroma follows the tint); the numbers against the declared "
                                               "straight-line model are reported, never gated (caustics are excluded)"},
    "specular-grid-point@clustered": {"kind": "direct", "lighting": "clustered", "visibility": "rays", "denoise": "off",
                                      "terminator_band": 2,
                                      "controls": ["specular-ignored", "f90-one", "specular-weight-not-on-f0"],
                                      "terms": "one point light only, clustered DI with RT visibility "
                                               "(--lighting clustered --rt on), unfiltered (--lighting-denoise off), "
                                               "on the ior x specular grid"},
    # F15.3 (ticket 17): the volumetric subsurface mode.  The reference is the Mitsuba random walk (tools/f15_subsurface_reference.py), not
    # the Appendix B model: "reference": "subsurface".  Captures run with --reflections off (deterministic resolve terms: the neighbours'
    # glow must not appear in SSR reflections).  The engine side (screen-space Burley diffusion) landed with F15.1 ticket 17: the sweep,
    # sub-pixel and back-lit keys pass.  The side-lit bound key MISSES the decision-11 bound on the gated pieces (see the ticket 17
    # comment: the diffusion is energy-normalised to the colour, the Mitsuba rough boundary is not, and the best-case shape error is
    # above the bound too): the decision rule of ticket 10 (hybrid RT single scattering, ticket 18) applies, so the key stays asserted
    # red ("pending": "ticket 18") with the bound unchanged; ticket 18 (or the orchestrator's decision) removes the marker.
    "subsurface-volumetric-sun": {"kind": "subsurface", "reference": "subsurface", "reflections": "off",
                                  "pending": "ticket 18",
                                  "controls": VOLUMETRIC_BOUND_CONTROLS,
                                  "terms": "sun only from the side (environment 0, no shadows, no GI) on skin and jade spheres, weight {0, 0.5, 1} at the "
                                           "nominal radius 0.08 m and radius x0.5 / x1 / x2 at weight 1"},
    # Re-measured on the valid ground truth (roughness 0.35, boundary-aware albedo; ticket 17 record): the engine's terminator bleed
    # saturates at large radii (skin 0.137 -> 0.147 where the truth rises 0.30 -> 0.43), so the sweep key is asserted red like the
    # side-lit key until ticket 18 (profile shape).
    "subsurface-volumetric-radius": {"kind": "radius-monotonic", "reference": "subsurface", "reflections": "off", "pending": "ticket 18",
                                     "controls": VOLUMETRIC_SWEEP_CONTROLS,
                                     "terms": "sun only from the side on skin and jade spheres of weight 1, radius x0.25 / x0.5 / x1 / x2 / x3 of 0.08 m"},
    "subsurface-volumetric-subpixel": {"kind": "equal", "reference": "subsurface", "reflections": "off",
                                       "reference_args": ["--no-ground-truth"],
                                       "pair": "subsurface-volumetric-subpixel-off", "controls": ["subsurface-leak"],
                                       "terms": "sun only from the side on spheres with weight {1, 0.5} and a sub-pixel radius (<= 0.002 m), subsurface colour = base "
                                                "colour; must equal the engine capture of the same scene without any block"},
    "subsurface-volumetric-subpixel-off": {"kind": "equal", "capture_only": True, "reflections": "off", "controls": [],
                                           "terms": "the sub-pixel scene without any subsurface block (the pair of the key above; capture only)"},
    "subsurface-volumetric-backlit": {"kind": "subsurface-report", "reference": "subsurface", "reflections": "off", "controls": [],
                                      "terms": "sun BEHIND thick weight-1 spheres (radius 0.08 and 0.16 m): measured and reported, not gated "
                                               "(declared limit of screen-space diffusion)"},
}


# --- F15.3 texture closure: TEXTURED lobe fixtures (tickets 09, 10, 11, 12, 16, 19, 22) --------------------------------------------
# Small deterministic textures (32 x 16, generated by src/testbench/material_validation.cpp) bound to the lobe slots of 2 x 3 spheres;
# the reference samples the EXPORTED texels (tools/f15_reference.py "KHR texture fixtures": level-0 bilinear, repeat, binary16, ROI
# without ambiguous-footprint pixels).  Gates and thresholds are the existing ones (direct p99 <= 1 %, max <= 3 %; declared mean <= 0.5 %,
# p99 <= 1 %), each sphere its own row.  Every control must fail.
def _texture_keys():
    lobes = {
        "coat": (["texture-v-flipped", "texture-coat-normal-ignored", "texture-coat-normal-y-flipped", "texture-coat-factor-ignored",
                  "texture-coat-roughness-ignored"], "clearcoat normal (RGB), factor (R) and roughness (G) textures"),
        "aniso": (["texture-v-flipped", "texture-aniso-direction-ignored", "texture-aniso-direction-y-flipped",
                   "texture-aniso-strength-ignored"], "anisotropy direction (RG) and strength (B) texture"),
        "irid": (["texture-v-flipped", "texture-irid-thickness-ignored", "texture-irid-thickness-inverted",
                  "texture-irid-factor-ignored"], "iridescence thickness (G) and factor (R) textures"),
        "sheen": (["texture-v-flipped", "texture-sheen-color-ignored", "texture-sheen-color-srgb-encoded",
                   "texture-sheen-roughness-ignored"], "sheen colour (sRGB RGB) and roughness (A) textures"),
        "thin": (["texture-v-flipped", "texture-thin-weight-ignored", "texture-thin-color-ignored", "texture-thin-color-srgb-encoded"],
                 "thin subsurface weight (A) and colour (sRGB RGB) textures"),
    }
    keys = {}
    for name, (controls, what) in lobes.items():
        keys[f"texture-{name}-sun"] = {
            "kind": "direct", "always_green": True, "controls": controls, "reference_args": ["--textures"],
            "terms": f"sun disk only (environment 0, no shadows, no GI) on 2 x 3 spheres with {what}"}
        keys[f"texture-{name}-env-resolve"] = {
            "kind": "declared", "always_green": True, "reflections": "off",
            # a constant environment has no angular signature of the anisotropy direction: the direction controls live on the sun key
            "controls": ["declared-" + c for c in controls if not (name == "aniso" and "direction" in c)],
            "reference_args": ["--textures", "--no-exact-environment"],
            "terms": f"constant environment only (radiance 0.5, no sun) on 2 x 3 spheres with {what}, deterministic resolve terms "
                     "(--reflections off --lighting-denoise custom)"}
    return keys


TEXTURE_SCENARIOS = {
    "texture-coat-sun": {**_texture_keys()["texture-coat-sun"]},
    "texture-coat-env-resolve": {**_texture_keys()["texture-coat-env-resolve"]},
    "texture-aniso-sun": {**_texture_keys()["texture-aniso-sun"]},
    "texture-aniso-env-resolve": {**_texture_keys()["texture-aniso-env-resolve"]},
    "texture-irid-sun": {**_texture_keys()["texture-irid-sun"]},
    "texture-irid-env-resolve": {**_texture_keys()["texture-irid-env-resolve"]},
    "texture-sheen-sun": {**_texture_keys()["texture-sheen-sun"]},
    "texture-sheen-env-resolve": {**_texture_keys()["texture-sheen-env-resolve"]},
    "texture-thin-sun": {**_texture_keys()["texture-thin-sun"]},
    "texture-thin-env-resolve": {**_texture_keys()["texture-thin-env-resolve"]},
}
SCENARIOS.update(TEXTURE_SCENARIOS)


# --- F15.4 (ticket 23): layered materials on the terrain patch fixture (tools/f15_layers.py) ------------------------------
LAYER_KINDS = ("layer-equal", "layer-ramp", "layer-furnace")
TERRAIN_MATERIALS_ENV = "TERRAIN_MATERIALS_DIR"


def terrain_materials_dir():
    """The unpacked CC0 terrain sets (tools/fetch_terrain_materials.py) or None: the `cc0` keys are skipped without them."""
    import os
    candidate = os.environ.get(TERRAIN_MATERIALS_ENV)
    path = pathlib.Path(candidate) if candidate else REPO / "build" / "terrain-materials"
    return path if (path / "Grass004").is_dir() else None


def _layer_keys():
    keys = {}
    weights = ["1,0,0,0", "0,1,0,0", "0,0,1,0", "0,0,0,1"]
    for source, suffix in (("procedural", ""), ("cc0", "-cc0")):
        extra = {"cc0": True} if source == "cc0" else {}
        for j in range(4):
            keys[f"layers-single-{j}{suffix}"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, **extra,
                                                  "scene": f"terrain/single={j}/coat=off",
                                                  "terms": f"the terrain patch with layer {j} alone as an ordinary material (no stack, mesh uvs x the tiling)"}
        for j in range(4):
            other = (j + 1) % 4
            keys[f"layers-endpoint-{j}{suffix}"] = {
                "kind": "layer-equal", "always_green": True, "snapshot": False, **extra, "scene": f"terrain/weights={weights[j]}/coat=off",
                "pair": f"layers-single-{j}{suffix}", "controls": [f"layers-single-{other}{suffix}"], "needs": [f"layers-single-{other}{suffix}"],
                "terms": f"the terrain stack with the constant mask e_{j} (weight 1 on layer {j}, 0 on the others, height contrast on) must equal the single "
                         f"material of layer {j} bit for bit; the control compares it with the single material of layer {other}"}
    keys["layers-two-layers"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/weights=1,1,0,0/layers=2/coat=off",
                                 "terms": "the terrain stack of only two layers, constant mask (1, 1)"}
    keys["layers-weight-zero"] = {
        "kind": "layer-equal", "always_green": True, "snapshot": False, "scene": "terrain/weights=1,1,0,0/coat=off", "pair": "layers-two-layers",
        "controls": ["layers-endpoint-0"], "needs": ["layers-endpoint-0"],
        "terms": "four layers with weight 0 on layers 2 and 3 must equal the stack without them (weight 0 leaves the others unchanged); the control "
                 "is the weight vector (1, 0, 0, 0), which differs"}
    keys["layers-patch-nocoat"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/coat=off",
                                   "terms": "the full terrain patch (generated mask, height contrast) without the coat finish"}
    keys["layers-patch"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain",
                            "terms": "the full terrain patch with the cavity coat finish (the measured view)"}
    keys["layers-coat-spec"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/coat=spec",
                                "terms": "the full terrain patch with a specular block (ior 1.5, factor 1 - 2^-24) instead of a coat: the extended-path twin of coat weight 0"}
    keys["layers-coat-wrong-f0"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/coat=spec/ior=1.4",
                                    "terms": "the specular-block twin with ior 1.4 (base F0 0.028 instead of 0.040): the negative control of the coat-zero gate"}
    keys["layers-coat-union"] = {"kind": "layer-equal", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/coat=union",
                                 "terms": "the full terrain patch with NO coat on the stack in a scene whose feature union has the clearcoat bit (an unused library material "
                                          "carries a coat): the exact twin of coat weight 0 (the same pipeline, the same code path of a pixel without a coat)"}
    keys["layers-coat-zero"] = {
        "kind": "layer-equal", "always_green": True, "snapshot": False, "scene": "terrain/coat=zero", "pair": "layers-coat-union",
        "info_pair": "layers-coat-spec", "controls": ["layers-patch", "layers-coat-wrong-f0"], "needs": ["layers-patch", "layers-coat-spec", "layers-coat-wrong-f0"],
        "terms": "the clearcoat finish present with coat weight 0 everywhere must leave the layered base untouched (a feature layer changes the coat "
                 "weight only): EXACTLY (0 pixels) equal to the same stack without a coat in a scene of the same feature union (a coat that resolves to weight 0 "
                 "takes the path of a pixel without a coat, by construction); the controls are the real cavity coat and a base F0 changed by the ior (the specular-block "
                 "twin at 1.4 instead of 1.5).  The distance to the specular-block twin of the same F0 (factor 1 - 2^-24, F0 0.04 to 6e-8) is reported, not gated"}
    keys["layers-ramp-single-0"] = {"kind": "layer-ramp", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/ramp=on/single=0", "reflections": "off",
                                    "terms": "the flat quad with the dark plain layer alone"}
    keys["layers-ramp-single-1"] = {"kind": "layer-ramp", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/ramp=on/single=1", "reflections": "off",
                                    "terms": "the flat quad with the light plain layer alone"}
    keys["layers-ramp-reverse"] = {"kind": "layer-ramp", "capture_only": True, "controls": [], "snapshot": False, "scene": "terrain/ramp=reverse", "reflections": "off",
                                   "terms": "the reversed ramp (the negative control of the monotonic gate)"}
    keys["layers-ramp"] = {
        "kind": "layer-ramp", "always_green": True, "snapshot": True, "scene": "terrain/ramp=on", "endpoints": ["layers-ramp-single-0", "layers-ramp-single-1"],
        "controls": ["layers-ramp-reverse"], "needs": ["layers-ramp-single-0", "layers-ramp-single-1", "layers-ramp-reverse"],
        "reflections": "off",
        "terms": "a flat quad, two plain layers (albedo 0.05 and 0.9), the weight of the second rising linearly with u under one sun: the blend factor "
                 "against the two single-material twins must rise monotonically along u and follow the mask; the reversed ramp must fail"}
    keys["layers-furnace"] = {
        "kind": "layer-furnace", "always_green": True, "snapshot": True, "scene": "terrain/furnace=on", "reflections": "off",
        "controls": ["additive-blend"],
        "terms": "a flat quad at a grazing view under a constant environment of radiance 1, four albedo-1 layers (metallic 0/1/0/1, roughness .1/.4/.7/1) "
                 "blended by a bilinear mask: the blended single BSDF must not exceed 1 + 0.5 %; the additive sum of the layers' BSDFs is the control"}
    return keys


SCENARIOS.update(_layer_keys())

# F15.4 (ticket 23, the layers' feature blocks): per block B (layer 1 of a flat two-layer quad carries it, layer 0 is plain) the stack at each endpoint
# of a constant mask must equal the single material (0 pixels), the stack at weight 1 must NOT equal the same layer without its block (negative
# control: the old behaviour, blocks ignored), and the transition ramp must rise monotonically (and, for the blocks whose luminance is affine in the
# blended parameter, follow the mask like the base ramp).  tests/test_terrain_fixture.cpp kBlockTemplates mirrors the scene strings.
LAYER_BLOCKS = ("coat", "sheen", "aniso", "irid", "spec", "emissive", "thin", "all")
# Blocks whose luminance is affine in the blended parameter (coat factor, sheen colour, film strength, subsurface weight, emission): the blend
# factor of the ramp follows the mask u like the base ramp.  Anisotropy and specular/ior act non-linearly (alpha_t = mix(alpha, 1, s^2); F0 from ior):
# monotonic only, the deviation from the mask is reported.
LAYER_BLOCKS_AFFINE = ("coat", "sheen", "irid", "emissive", "thin")


def _layer_block_keys():
    keys = {}
    for b in LAYER_BLOCKS:
        base = {"snapshot": False, "reflections": "off"}
        keys[f"layers-blk-{b}-s0"] = {"kind": "layer-equal", "capture_only": True, "controls": [], **base, "scene": f"terrain/block={b}/single=0",
                                      "terms": f"layer 0 (plain) alone as an ordinary material on the flat quad ({b} scenes)"}
        keys[f"layers-blk-{b}-s1"] = {"kind": "layer-equal", "capture_only": True, "controls": [], **base, "scene": f"terrain/block={b}/single=1",
                                      "terms": f"layer 1 alone with its {b} block as an ordinary material (mesh uvs x its tiling)"}
        keys[f"layers-blk-{b}-s1x"] = {"kind": "layer-equal", "capture_only": True, "controls": [], **base, "scene": f"terrain/block={b}/single=1/blockless=on",
                                       "terms": f"layer 1 alone WITHOUT the {b} block (the old behaviour: the block ignored)"}
        keys[f"layers-blk-{b}-e1"] = {
            "kind": "layer-equal", "always_green": True, **base, "scene": f"terrain/block={b}/weights=0,1,0,0", "pair": f"layers-blk-{b}-s1",
            **({"pending": "ticket 23: the seven blocks together differ from the single material by 2 ULP (39,045 of 921,600 pixels, 0 in the 8-bit image): each block alone is exact"} if b == "all" else {}),
            "controls": [f"layers-blk-{b}-s1x", f"layers-blk-{b}-s0"], "needs": [f"layers-blk-{b}-s1x", f"layers-blk-{b}-s0"],
            "terms": f"the stack with the constant mask (0, 1) must render layer 1 with its {b} block exactly like the single material (0 pixels); the "
                     f"controls are the same layer without the block (blocks ignored) and the plain layer 0"}
        keys[f"layers-blk-{b}-e0"] = {
            "kind": "layer-equal", "always_green": True, **base, "scene": f"terrain/block={b}/weights=1,0,0,0", "pair": f"layers-blk-{b}-s0",
            "controls": [f"layers-blk-{b}-s1"], "needs": [f"layers-blk-{b}-s1"],
            "terms": f"the stack with the constant mask (1, 0), the {b} layer at weight 0, must equal the plain single layer 0 (weight 0 has no effect); the "
                     f"control is the layer carrying the block"}
        keys[f"layers-blk-{b}-ramp-s0"] = {"kind": "layer-ramp", "capture_only": True, "controls": [], **base, "scene": f"terrain/block={b}/ramp=on/single=0",
                                           "terms": f"the flat quad, plain layer 0 alone ({b} ramp endpoint)"}
        keys[f"layers-blk-{b}-ramp-s1"] = {"kind": "layer-ramp", "capture_only": True, "controls": [], **base, "scene": f"terrain/block={b}/ramp=on/single=1",
                                           "terms": f"the flat quad, layer 1 with its {b} block alone ({b} ramp endpoint)"}
        keys[f"layers-blk-{b}-ramp-rev"] = {"kind": "layer-ramp", "capture_only": True, "controls": [], **base, "scene": f"terrain/block={b}/ramp=reverse",
                                            "terms": f"the reversed {b} ramp (negative control of the monotonic gate)"}
        keys[f"layers-blk-{b}-ramp"] = {
            "kind": "layer-ramp", "always_green": True, "snapshot": True, "reflections": "off", "scene": f"terrain/block={b}/ramp=on",
            "endpoints": [f"layers-blk-{b}-ramp-s0", f"layers-blk-{b}-ramp-s1"], "controls": [f"layers-blk-{b}-ramp-rev"],
            "needs": [f"layers-blk-{b}-ramp-s0", f"layers-blk-{b}-ramp-s1", f"layers-blk-{b}-ramp-rev"], "affine": b in LAYER_BLOCKS_AFFINE, "pooled": True,
            "terms": f"two layers of one dielectric, the {b} block on layer 1, its weight rising linearly with u: the blend factor against the two single "
                     f"twins must rise monotonically along u"
                     + (" and follow the mask" if b in LAYER_BLOCKS_AFFINE else " (non-linear block: the deviation from the mask is reported)")
                     + "; the reversed ramp must fail"}
    return keys


SCENARIOS.update(_layer_block_keys())

# F15.4 (ticket 23, secondary hits): the flat quad of two coloured layers seen through a mirror sphere (RT reflection hits, the captured reflection
# probe) and as the bounce on a wall (GI hits).  The stack at weight 1 on one layer must render EXACTLY like the single material of that layer (the
# secondary-hit path samples the blended material); the negative controls are the other layer and the stack's own record alone (what secondary
# hits saw before the blend), which differ.
LAYER_HIT_MODES = dict(refl=dict(rt=True, reflections="rt"), probe=dict(reflections="probes", probe=True),
                       gi=dict(lighting="brute", shadows="rt", gi="ddgi", reflections="off"))


def _layer_hit_keys():
    keys = {}
    for mode, extra in LAYER_HIT_MODES.items():
        base = {"snapshot": False, **extra}
        keys[f"layers-hit-{mode}-s0"] = {"kind": "layer-equal", "capture_only": True, "controls": [], **base, "scene": "terrain/hit=on/single=0",
                                         "terms": f"layer 0 (green) alone as an ordinary material, {mode} hits"}
        keys[f"layers-hit-{mode}-s1"] = {"kind": "layer-equal", "capture_only": True, "controls": [], **base, "scene": "terrain/hit=on/single=1",
                                         "terms": f"layer 1 (red) alone as an ordinary material, {mode} hits"}
        keys[f"layers-hit-{mode}-own"] = {"kind": "layer-equal", "capture_only": True, "controls": [], **base, "scene": "terrain/hit=on/owner=on",
                                          "terms": f"the stack's own record alone (the pre-blend secondary-hit appearance), {mode} hits"}
        keys[f"layers-hit-{mode}-e1"] = {
            "kind": "layer-equal", "always_green": True, **base, "scene": "terrain/hit=on/weights=0,1,0,0", "pair": f"layers-hit-{mode}-s1",
            "controls": [f"layers-hit-{mode}-own", f"layers-hit-{mode}-s0"], "needs": [f"layers-hit-{mode}-own", f"layers-hit-{mode}-s0"],
            "terms": f"the stack with the constant mask (0, 1) seen at secondary hits ({mode}) must equal the single material of layer 1 exactly (0 pixels); "
                     f"the controls are the stack's own record alone (the old behaviour) and layer 0"}
        keys[f"layers-hit-{mode}-e0"] = {
            "kind": "layer-equal", "always_green": True, **base, "scene": "terrain/hit=on/weights=1,0,0,0", "pair": f"layers-hit-{mode}-s0",
            "controls": [f"layers-hit-{mode}-s1"], "needs": [f"layers-hit-{mode}-s1"],
            "terms": f"the stack with the constant mask (1, 0) seen at secondary hits ({mode}) must equal the single material of layer 0; the control is layer 1"}
    return keys


SCENARIOS.update(_layer_hit_keys())
# Red-state expectations (before ticket 03): direct metallic rows 0 and 1 pass, intermediate rows fail.
GPU_PROCESS = re.compile(r"^(phosphor|soc_bench|f\d+_spike)$")


# --- guard and manifest -------------------------------------------------------

@contextmanager
def gpu_lock(path):
    import fcntl
    with pathlib.Path(path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GpuBusy("another cooperating Phosphor GPU verification owns the lock")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def running_gpu_processes(ps_output=None):
    """Names of other GPU jobs (`phosphor`, `soc_bench`, `f<N>_spike`) from `ps -axo pid=,comm=`."""
    if ps_output is None:
        ps_output = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True).stdout
    found = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] != str(os.getpid()) and GPU_PROCESS.match(pathlib.Path(parts[1]).name):
            found.append(f"{parts[0]}:{pathlib.Path(parts[1]).name}")
    return found


def command_text(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def parse_sw_vers(text):
    """`sw_vers` output ("Key:\tvalue" lines) as a dict; tolerant of None, blank, short or odd output."""
    info = {}
    for line in (text or "").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            info[key.strip()] = value.strip()
    return info


def os_description(info):
    """One line such as `macOS 27.3 (26D1)`; empty when sw_vers told us nothing."""
    parts = [info.get("ProductName"), info.get("ProductVersion")]
    text = " ".join(p for p in parts if p)
    if info.get("BuildVersion"):
        text += f" ({info['BuildVersion']})"
    return text.strip()


def build_manifest(binary, metallib, scenarios, seed):
    git = lambda *a: command_text(["git", "-C", str(REPO), *a])  # noqa: E731
    power = command_text(["pmset", "-g", "batt"]) or ""
    tools = {name: sha256_file(REPO / "tools" / name) for name in
             ("f15_check.py", "f15_reference.py", "f15_transmission_reference.py", "f15_shadow_reference.py", "f15_subsurface_reference.py",
              "f15_common.py", "f15_layers.py",
              "f12_reference.py",
              "run_checked.py")}
    return {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "binary": str(binary), "binary_sha256": sha256_file(binary) if binary.is_file() else None,
            "metallib_sha256": sha256_file(metallib) if metallib.is_file() else None,
            "tool_sha256": tools, "seed": seed, "reference_seed": None,
            "reference_note": "the Appendix B reference is deterministic (no random numbers)",
            "os": parse_sw_vers(command_text(["sw_vers"])), "sdk_version": command_text(["xcrun", "--show-sdk-version"]),
            "power": {"pmset_batt": power, "on_ac": "AC Power" in power},
            "frames": FRAMES, "resolution": RESOLUTION, "thresholds": THRESHOLDS,
            "scenarios": {name: SCENARIOS[name] for name in scenarios}}


# --- commands -----------------------------------------------------------------

def bench_scenario(name):
    """The bench scenario of a runner key ("<bench>@<lighting>" variants share the bench scene)."""
    return SCENARIOS.get(name, {}).get("scene") or name.split("@", 1)[0]


def write_constant_probe(folder, radiance):
    """Six 64x64 linear RGB PFM faces (px nx py ny pz nz) of a constant grey radiance: the cooked probe whose radiance is known exactly."""
    import struct
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = struct.pack("<f", float(radiance)) * (64 * 64 * 3)
    for face in ("px", "nx", "py", "ny", "pz", "nz"):
        (folder / f"{face}.pfm").write_bytes(b"PF\n64 64\n-1.0\n" + payload)
    return folder


def capture_command(binary, name, folder, seed=1):
    """The exact single GPU process of one scenario; CPU-side everything else."""
    last = str(FRAMES - 1)
    lighting = SCENARIOS[name].get("lighting", "legacy")
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", bench_scenario(name),
            "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "off" if lighting == "legacy" and not SCENARIOS[name].get("rt") else "on",
            "--shadows", SCENARIOS[name].get("shadows", "off"),
            "--lighting", lighting, "--gi", SCENARIOS[name].get("gi", "off"), "--reflections", SCENARIOS[name].get("reflections", "ssr"), "--ao", "off",
            "--lighting-denoise", SCENARIOS[name].get("denoise", "custom"), "--reflection-samples", "8",
            "--lighting-seed", str(seed),
            "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(FRAMES),
            "--warmup", "0", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1",
            # Synchronous pipelines: a cold compiler cache must never let a capture use a fallback variant (a fallback-served capture once
            # equalled the base model, ticket 17); fallback_draws() rejects any capture whose log still reports fallback draws.
            "--pipeline-sync", *(["--reflection-capture-probe"] if SCENARIOS[name].get("probe") else []),
            "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", last,
            *(["--capture-linear-accumulate", str(SCENARIOS[name]["accumulate"])] if SCENARIOS[name].get("accumulate") else []),
            "--capture-linear-signal", "hdr",
            *(["--export-reference", str(folder / "snapshot"), "--export-reference-frame", last] if SCENARIOS[name].get("snapshot", True) else []),
            *(["--terrain-textures", str(terrain_materials_dir() or "build/terrain-materials")] if SCENARIOS[name].get("cc0") else []),
            *(["--reflection-probe", str(write_constant_probe(folder / "probe", SCENARIOS[name]["probe_constant"]))] if SCENARIOS[name].get("probe_constant") else []),
            "--report", str(folder / "report.json")]


def scenario_controls(name):
    """(literal-model controls, declared-model controls) the reference must render for one scenario."""
    spec = SCENARIOS[name]
    if spec["kind"] == "declared":
        return [], list(spec["controls"])
    return list(spec["controls"]), list(spec.get("declared", []))


def reference_script(name):
    """The independent reference tool of one scenario: the Appendix B model, (ticket 13) the transmission ground truth or
    (ticket 17) the volumetric random walk."""
    if SCENARIOS[name]["kind"] in LAYER_KINDS:
        return "f15_layers.py"
    if SCENARIOS[name]["kind"] in SHADOW_KINDS:
        return "f15_shadow_reference.py"
    if SCENARIOS[name]["kind"] in TRANSMISSION_KINDS:
        return "f15_transmission_reference.py"
    return "f15_subsurface_reference.py" if SCENARIOS[name].get("reference") == "subsurface" else "f15_reference.py"


def reference_arguments(name):
    """Extra arguments of the reference tool of one scenario (the sub-pixel key does not read the random-walk ground truth)."""
    return list(SCENARIOS[name].get("reference_args", []))


def reference_command(name, folder):
    if SCENARIOS[name]["kind"] in LAYER_KINDS:  # no external reference tool: the statistics and the BSDF quadrature of tools/f15_layers.py
        return ["python3", "-I", str(REPO / "tools" / "f15_layers.py"), "evaluate", str(folder)]
    if SCENARIOS[name]["kind"] in SHADOW_KINDS:
        return ["uv", "run", "--offline", "--no-project", "--python", "3.13", "--with", "mitsuba==3.9.1", "--with", "numpy", "python", "-I",
                str(REPO / "tools" / "f15_shadow_reference.py"), "render", str(folder / "snapshot"), "--out", str(folder / "reference"),
                "--margin", str(SHADOW_ENVELOPE[SCENARIOS[name]["envelope"]]), *reference_arguments(name),
                "--controls", *SCENARIOS[name]["controls"]]
    if SCENARIOS[name]["kind"] in TRANSMISSION_KINDS:
        declared = SCENARIOS[name]["kind"] == "transmission-declared"
        command = ["uv", "run", "--offline", "--no-project", "--python", "3.13", "--with", "mitsuba==3.9.1",
                   "--with", "numpy", "python", "-I", str(REPO / "tools" / "f15_transmission_reference.py"),
                   "declared" if declared else "render", str(folder / "snapshot"), "--out", str(folder / "reference")]
        if not declared:
            command += ["--spp", str(SCENARIOS[name].get("spp", REFERENCE_SPP))]
        return command + ["--controls", *SCENARIOS[name]["controls"]]
    controls, declared = scenario_controls(name)
    command = ["uv", "run", "--offline", "--no-project", "--python", "3.13", "--with", "mitsuba==3.9.1",
               "--with", "numpy", "python", "-I", str(REPO / "tools" / reference_script(name)), "render",
               str(folder / "snapshot"), "--out", str(folder / "reference"),
               "--visibility", SCENARIOS[name].get("visibility", "none"), "--controls", *controls]
    if declared:
        command += ["--declared-controls", *declared]
    return command + reference_arguments(name)


# --- statistics (pure numpy) --------------------------------------------------

def relative_error(np, candidate, reference, roi, floor_reference=None):
    """Per-pixel relative error over the ROI (max over RGB), with the absolute floor of decision 11 (taken from `floor_reference`
    when given: the shadow keys derive it from the ground truth for every candidate and control)."""
    floor = THRESHOLDS["floor_fraction"] * float((reference if floor_reference is None else floor_reference)[roi].mean())
    # A reference that is exactly 0 over the whole ROI (a control image of a surface nothing reaches, e.g. a back-lit carrier
    # without transmission) has floor 0: 0 / 0 is the error 0 of two equal black pixels, a non-zero candidate is infinitely wrong.
    denominator = np.maximum(np.maximum(reference[roi], floor), np.finfo(reference.dtype).tiny)
    with np.errstate(over="ignore"):
        error = np.abs(candidate[roi] - reference[roi]) / denominator
    return error.max(axis=-1), floor


def terminator_band(np, lit, sphere, width=1):
    """Lit pixels with an unlit pixel of the same sphere within `width` pixels (Chebyshev; width 1 = the
    8-neighbours): the shadow terminator, where N.L changes sign inside one pixel and interpolated normals
    of two independent implementations may disagree on the sign (a shading discontinuity, excluded like the
    silhouette; decision 11 ROI rule).  Width 2 for RT-visibility scenarios (decision 11 amendment
    2026-11-13: the engine's shadow-ray offset on the tessellated facets moves the shadow terminator by up
    to 2 pixels)."""
    padded_lit = np.pad(lit, width, constant_values=True)
    padded_sphere = np.pad(sphere, width, constant_values=-1)
    band = np.zeros_like(lit)
    h, w = lit.shape
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dy == 0 and dx == 0:
                continue
            n_lit = padded_lit[width + dy:width + dy + h, width + dx:width + dx + w]
            n_sphere = padded_sphere[width + dy:width + dy + h, width + dx:width + dx + w]
            band |= lit & ~n_lit & (n_sphere == sphere)
    return band


def evaluate_direct(np, candidate, reference, masks, spheres, band=1):
    """Per metallic row: p99 and max (outside the quadrature mask) over lit ROI
    pixels, the shadow terminator excluded (see terminator_band, `band` pixels wide)."""
    gate = THRESHOLDS["direct"]
    lit = masks["roi"] & masks["lit"] & ~terminator_band(np, masks["lit"], masks["sphere"], band)
    error_all, floor = relative_error(np, candidate, reference, lit)
    narrow = masks["narrow"][lit]
    sphere = masks["sphere"][lit]
    rows = {}
    group_of = lambda s: s.get("group", s["metallic"])  # noqa: E731 - specular grids group by (metallic, weight, colour)
    for value in sorted({group_of(s) for s in spheres}, key=lambda g: (isinstance(g, str), g)):
        members = [s["instance"] for s in spheres if group_of(s) == value]
        pick = np.isin(sphere, members)
        if not pick.any():
            rows[value] = {"pixels": 0, "pass": False, "note": "no lit ROI pixels"}
            continue
        e, outside = error_all[pick], error_all[pick & ~narrow]
        row = {"pixels": int(pick.sum()), "mean": float(e.mean()), "p95": float(np.percentile(e, 95)),
               "p99": float(np.percentile(e, 99)), "max_all": float(e.max()),
               "max_outside_mask": float(outside.max()) if outside.size else 0.0,
               "masked_pixels": int((pick & narrow).sum())}
        row["pass"] = row["p99"] <= gate["p99"] and row["max_outside_mask"] <= gate["max_outside_quadrature_mask"]
        rows[value] = row
    return {"kind": "direct", "floor": floor, "rows": {str(k): v for k, v in rows.items()},
            "pass": all(r["pass"] for r in rows.values())}


def evaluate_environment(np, candidate, reference, masks, spheres, direct_info=None):
    """Mean relative error per sphere (the published table) against the 5 % bound."""
    gate = THRESHOLDS["environment"]["mean_per_sphere"]
    roi = masks["roi"]
    error, floor = relative_error(np, candidate, reference, roi)
    sphere = masks["sphere"][roi]
    table = []
    for s in spheres:
        pick = sphere == s["instance"]
        if not pick.any():
            table.append({**{k: s[k] for k in ("instance", "metallic", "roughness")}, "pixels": 0, "pass": False})
            continue
        e = error[pick]
        table.append({"instance": s["instance"], "metallic": s["metallic"], "roughness": s["roughness"],
                      "pixels": int(pick.sum()), "mean": float(e.mean()), "p95": float(np.percentile(e, 95)),
                      "pass": bool(e.mean() <= gate)})
    out = {"kind": "environment", "floor": floor, "spheres": table, "pass": all(t["pass"] for t in table),
           "worst_mean": max(t.get("mean", 0.0) for t in table)}
    if direct_info is not None:
        out["direct_statistics_not_gated"] = direct_info
    return out


def evaluate_declared(np, candidate, reference, masks, spheres):
    """Per sphere: mean and p99 of the relative error against the declared-environment model."""
    gate = THRESHOLDS["declared"]
    roi = masks["roi"]
    error, floor = relative_error(np, candidate, reference, roi)
    sphere = masks["sphere"][roi]
    table = []
    for s in spheres:
        pick = sphere == s["instance"]
        if not pick.any():
            table.append({**{k: s[k] for k in ("instance", "metallic", "roughness")}, "pixels": 0, "pass": False})
            continue
        e = error[pick]
        mean, p99 = float(e.mean()), float(np.percentile(e, 99))
        table.append({"instance": s["instance"], "metallic": s["metallic"], "roughness": s["roughness"],
                      "pixels": int(pick.sum()), "mean": mean, "p99": p99,
                      "pass": bool(mean <= gate["mean_per_sphere"] and p99 <= gate["p99_per_sphere"])})
    return {"kind": "declared", "floor": floor, "spheres": table, "pass": all(t["pass"] for t in table),
            "worst_mean": max(t.get("mean", 0.0) for t in table), "worst_p99": max(t.get("p99", 0.0) for t in table)}


def evaluate_furnace(np, candidate, reference, masks, spheres):
    limit = THRESHOLDS["furnace"]["max_pixel"]
    roi = masks["roi"]
    sphere = masks["sphere"][roi]
    values = candidate[roi]
    table = []
    for s in spheres:
        pick = sphere == s["instance"]
        if not pick.any():
            table.append({"instance": s["instance"], "pixels": 0})
            continue
        mean = float(values[pick].mean())
        table.append({"instance": s["instance"], "metallic": s["metallic"], "roughness": s["roughness"],
                      "pixels": int(pick.sum()), "max": float(values[pick].max()), "mean": mean,
                      "energy_loss": 1.0 - mean, "reference_max": float(reference[roi][pick].max()),
                      "reference_energy_loss": 1.0 - float(reference[roi][pick].mean())})
    maximum = float(values.max())
    return {"kind": "furnace", "limit": limit, "max_pixel": maximum, "spheres": table, "pass": maximum <= limit,
            "reference_max_pixel": float(reference[roi].max())}


def evaluate_furnace_transport(np, candidate, reference, masks, spheres):
    """Stochastic transport: gate on the per-sphere mean (energy in expectation)."""
    result = evaluate_furnace(np, candidate, reference, masks, spheres)
    limit = THRESHOLDS["furnace_transport"]["mean_per_sphere"]
    means = [s["mean"] for s in result["spheres"] if s.get("pixels")]
    result.update({"kind": "furnace-transport", "limit": limit, "max_sphere_mean": max(means),
                   "pass": max(means) <= limit})
    return result


def evaluate_transmission(np, candidate, reference, masks, objects, gate_name="transmission", kind="transmission", floor_reference=None,
                          gated=True):
    """Decision 11 transmission bound, per glass object on the glass mask: mean and p95 of the relative luminance error
    (absolute floor 0.05 * mean over the whole ROI of the reference).  `objects` are the reference report's glass objects.
    The transmissive-shadow keys (tickets 14/15) use the same bound per caster shadow region (`gate_name`, `floor_reference` =
    the ground truth); `gated` False measures and passes (the curved-glass report)."""
    gate = THRESHOLDS[gate_name]
    roi = masks["roi"]
    error, floor = relative_error(np, candidate, reference, roi, floor_reference)
    instance = masks["sphere"][roi]
    table = []
    for o in objects:
        pick = instance == o["instance"]
        label = {"instance": o["instance"], "label": o.get("label", ""), "metallic": 0.0, "roughness": o.get("roughness", 0.0)}
        if not pick.any():
            table.append({**label, "pixels": 0, "pass": False})
            continue
        e = error[pick]
        mean, p95 = float(e.mean()), float(np.percentile(e, 95))
        table.append({**label, "pixels": int(pick.sum()), "mean": mean, "p95": p95,
                      "pass": bool(mean <= gate["mean_per_object"] and p95 <= gate["p95_per_object"])})
    # Orchestrator decision: the pooled glass-mask statistic of decision 11 is reported for information (not gated).
    pooled = {"pixels": int(error.size), "mean": float(error.mean()), "p95": float(np.percentile(error, 95))} if error.size else {}
    return {"kind": kind, "floor": floor, "pooled_not_gated": pooled, "objects": table,
            "pass": all(t["pass"] for t in table) if gated else True, "gated": gated,
            "worst_mean": max(t.get("mean", 0.0) for t in table), "worst_p95": max(t.get("p95", 0.0) for t in table)}
def ratio_statistics(np, candidate, reference, masks, objects, floor_from=None):
    """Orchestrator decision (ticket 15): per caster region the TRANSMITTANCE RATIO T = mean luminance in the region / mean luminance of the
    declared lit reference region of the same image, compared with the reference's: relative error of T (floor 0.05 * mean T of `floor_from`,
    the ground truth, by default the reference) and the p95 of the per-pixel error of luminance / lit mean (same floor rule).  Also the lit-floor
    level of the candidate against the reference (diagnostic, never gated)."""
    gate = THRESHOLDS["transmissive_shadow"]
    lit, roi = masks["lit"], masks["roi"]
    truth = reference if floor_from is None else floor_from
    ya, yb, yt = luminance(np, candidate), luminance(np, reference), luminance(np, truth)
    lit_a, lit_b, lit_t = float(ya[lit].mean()), float(yb[lit].mean()), float(yt[lit].mean())
    tiny = np.finfo(float).tiny
    regions = {o["instance"]: roi & (masks["sphere"] == o["instance"]) for o in objects}
    present = [float(yt[r].mean()) / lit_t for r in regions.values() if r.any()]
    floor_t = THRESHOLDS["floor_fraction"] * (sum(present) / len(present) if present else 0.0)
    floor_pixel = THRESHOLDS["floor_fraction"] * float((yt[roi] / lit_t).mean())
    table = []
    for o in objects:
        pick = regions[o["instance"]]
        label = {"instance": o["instance"], "label": o.get("label", ""), "metallic": 0.0, "roughness": 0.0}
        if not pick.any():
            table.append({**label, "pixels": 0, "pass": False})
            continue
        ta, tb = float(ya[pick].mean()) / max(lit_a, tiny), float(yb[pick].mean()) / lit_b
        mean = abs(ta - tb) / max(tb, floor_t, tiny)
        pixel = np.abs(ya[pick] / max(lit_a, tiny) - yb[pick] / lit_b) / np.maximum(yb[pick] / lit_b, max(floor_pixel, tiny))
        p95 = float(np.percentile(pixel, 95))
        table.append({**label, "pixels": int(pick.sum()), "mean": float(mean), "p95": p95, "transmittance": ta, "reference_transmittance": tb,
                      "pass": bool(mean <= gate["mean_per_object"] and p95 <= gate["p95_per_object"])})
    ratio = ya[lit] / np.maximum(yb[lit], tiny)
    diagnostic = {"lit_pixels": int(lit.sum()), "candidate_lit_mean_luminance": lit_a, "reference_lit_mean_luminance": lit_b,
                  "lit_level_ratio": lit_a / lit_b if lit_b else None, "lit_level_median_ratio": float(np.median(ratio)),
                  "lit_level_p5_ratio": float(np.percentile(ratio, 5))}
    return table, diagnostic


def evaluate_shadow_ratio(np, candidate, reference, masks, objects, floor_from=None):
    table, diagnostic = ratio_statistics(np, candidate, reference, masks, objects, floor_from)
    return {"kind": "transmissive-shadow-ratio", "floor": 0.0, "pooled_not_gated": {}, "objects": table, "pass": all(t["pass"] for t in table),
            "gated": True, "worst_mean": max(t.get("mean", 0.0) for t in table), "worst_p95": max(t.get("p95", 0.0) for t in table),
            "lit_floor_diagnostic": diagnostic}


def evaluate_shadow_curved(np, candidate, reference, masks, objects):
    """Orchestrator decision: qualitative gate of the curved-glass key.  Per caster region T (as above) must satisfy 0 < T < 1 (lighter than an
    opaque caster, darker than unshadowed) and the chroma statistic R/B (mean red over mean blue of the region) must follow the tint (tint =
    base x attenuation colour): within 2 % of 1 (log) for a neutral caster, on the tint's side of 1 with at least half its log magnitude for a
    tinted one.  The numbers against the declared model are reported, never gated."""
    gate = THRESHOLDS["transmissive_shadow"]["curved"]
    lit = masks["lit"]
    ya = luminance(np, candidate)
    lit_a = float(ya[lit].mean())
    numbers = evaluate_transmission(np, candidate, reference, masks, objects, "transmissive_shadow", "transmissive-shadow-curved", reference, gated=False)
    table = []
    for o, n in zip(objects, numbers["objects"]):
        pick = masks["roi"] & (masks["sphere"] == o["instance"])
        label = {"instance": o["instance"], "label": o.get("label", ""), "metallic": 0.0, "roughness": 0.0}
        if not pick.any():
            table.append({**label, "pixels": 0, "pass": False})
            continue
        t = float(ya[pick].mean()) / max(lit_a, np.finfo(float).tiny)
        mean_rgb = candidate[pick].mean(axis=0)
        chroma = float(mean_rgb[0] / mean_rgb[2]) if mean_rgb[2] > 0 else 0.0
        tint = [b * a for b, a in zip(o.get("base", [1, 1, 1]), o.get("attenuation_color", [1, 1, 1]))]
        expected = math.log(tint[0] / tint[2])
        if abs(expected) < 1e-6:
            chroma_ok = chroma > 0 and abs(math.log(chroma)) <= gate["chroma_clear"]
        else:
            chroma_ok = chroma > 0 and math.log(chroma) * expected > 0 and abs(math.log(chroma)) >= gate["chroma_tinted_fraction"] * abs(expected)
        transmittance_ok = gate["transmittance"][0] < t < gate["transmittance"][1]
        table.append({**label, "pixels": int(pick.sum()), "mean": n.get("mean", 0.0), "p95": n.get("p95", 0.0), "transmittance": t,
                      "chroma_r_over_b": chroma, "expected_chroma_r_over_b": math.exp(expected), "transmittance_ok": transmittance_ok,
                      "chroma_ok": bool(chroma_ok), "pass": bool(transmittance_ok and chroma_ok)})
    return {"kind": "transmissive-shadow-curved", "floor": numbers["floor"], "pooled_not_gated": {}, "objects": table,
            "pass": all(t["pass"] for t in table), "gated": True, "numbers_gated": False,
            "worst_mean": max(t.get("mean", 0.0) for t in table), "worst_p95": max(t.get("p95", 0.0) for t in table)}


def luminance(np, image):
    return image[..., 0] * LUMA_WEIGHTS[0] + image[..., 1] * LUMA_WEIGHTS[1] + image[..., 2] * LUMA_WEIGHTS[2]


def piece_roi(masks, instance):
    return masks["roi"] & (masks["sphere"] == instance)


def luminance_error(np, candidate, reference, roi, floor_fraction):
    """Per-pixel relative luminance error over a ROI with the absolute floor of decision 11 (floor = fraction * mean_ROI(reference Y))."""
    ya, yb = luminance(np, candidate)[roi], luminance(np, reference)[roi]
    floor = floor_fraction * float(yb.mean())
    denominator = np.maximum(np.maximum(yb, floor), np.finfo(yb.dtype).tiny)
    with np.errstate(over="ignore"):
        return np.abs(ya - yb) / denominator, floor


def evaluate_subsurface(np, candidate, reference, masks, spheres, gated=True):
    """Ticket 17, decision 11: per PIECE (every sphere of the scene, the neutral weight-0 ones included) the mean and p95 of the relative
    luminance error over the piece's ROI against the random-walk ground truth; the gate is every piece mean <= 10 % and p95 <= 25 %.
    `gated` False (the back-lit report) measures the same numbers and passes."""
    gate = THRESHOLDS["subsurface"]
    pieces = []
    for s in spheres:
        roi = piece_roi(masks, s["instance"])
        if not roi.any():
            pieces.append({"instance": s["instance"], "weight": s.get("weight", 0.0), "pixels": 0, "pass": False})
            continue
        error, floor = luminance_error(np, candidate, reference, roi, gate["floor_fraction"])
        ya, yb = luminance(np, candidate)[roi], luminance(np, reference)[roi]
        mean, p95 = float(error.mean()), float(np.percentile(error, 95))
        met = bool(mean <= gate["mean"] and p95 <= gate["p95"])
        in_regime = bool(s.get("diffusion_regime", True))  # orchestrator decision: pieces outside the diffusion regime are reported only
        pieces.append({"instance": s["instance"], "weight": s.get("weight", 0.0), "radius": s.get("radius"),
                       "group": s.get("series"), "pixels": int(roi.sum()), "mean": mean, "p95": p95, "floor": floor,
                       "candidate_mean_luminance": float(ya.mean()), "reference_mean_luminance": float(yb.mean()),
                       "gated": in_regime, "bound_met": met, "pass": met or not in_regime})
    passed = all(piece["pass"] for piece in pieces) and any(piece["gated"] for piece in pieces)
    return {"kind": "subsurface" if gated else "subsurface-report", "gated": gated, "pieces": pieces,
            "worst_mean": max(piece.get("mean", 0.0) for piece in pieces), "worst_p95": max(piece.get("p95", 0.0) for piece in pieces),
            "limits": {"mean": gate["mean"], "p95": gate["p95"]}, "pass": passed if gated else True,
            "bound_met": all(piece.get("bound_met", False) for piece in pieces)}


def terminator_bleed(np, image, masks, instance):
    """The declared terminator statistic of one piece: mean luminance of its geometrically unlit ROI pixels over the mean luminance of
    its lit ROI pixels (0 for a base model without light across the terminator, growing with the radius); None without either set."""
    roi = piece_roi(masks, instance)
    y = luminance(np, image)
    lit, unlit = roi & masks["lit"].astype(bool), roi & ~masks["lit"].astype(bool)
    if not lit.any() or not unlit.any() or float(y[lit].mean()) <= 0.0:
        return None
    return float(y[unlit].mean() / y[lit].mean())


def evaluate_radius_monotonic(np, candidate, reference, masks, spheres):
    """Ticket 17: along every sweep series (pieces that differ only in radius, ordered by radius) the terminator bleed of the CANDIDATE must
    never fall by more than the tolerance and must rise by at least the tolerance at every step where the ground truth does.  A flat
    candidate against a flat reference passes (the engine image of a model without diffusion matches the subsurface-off control);
    a flat candidate against a rising ground truth fails (radius ignored)."""
    tolerance = THRESHOLDS["radius_sweep"]["tolerance"]
    groups = {}
    for s in spheres:
        if s.get("series") is not None:
            groups.setdefault(s["series"], []).append(s)
    series = []
    for label, members in groups.items():
        members = sorted(members, key=lambda m: m["radius"])
        cand = [terminator_bleed(np, candidate, masks, m["instance"]) for m in members]
        ref = [terminator_bleed(np, reference, masks, m["instance"]) for m in members]
        steps = []
        for i in range(len(members) - 1):
            if None in (cand[i], cand[i + 1], ref[i], ref[i + 1]):
                steps.append({"pass": False, "note": "a piece has no lit or no unlit ROI pixels"})
                continue
            dc, dr = cand[i + 1] - cand[i], ref[i + 1] - ref[i]
            ok = dc >= -tolerance and (dr < tolerance or dc >= tolerance)
            steps.append({"radius_from": members[i]["radius"], "radius_to": members[i + 1]["radius"], "candidate_step": dc,
                          "reference_step": dr, "reference_rises": bool(dr >= tolerance), "pass": bool(ok)})
        series.append({"series": label, "radii": [m["radius"] for m in members], "candidate": cand, "reference": ref, "steps": steps,
                       "pass": all(step["pass"] for step in steps)})
    reference_monotonic = bool(series) and all(step.get("reference_rises") for item in series for step in item["steps"])
    return {"kind": "radius-monotonic", "tolerance": tolerance, "series": series, "pass": bool(series) and all(x["pass"] for x in series),
            "reference_monotonic": reference_monotonic, "statistic": THRESHOLDS["radius_sweep"]["statistic"]}


def evaluate_radius_statistic_match(np, candidate, reference, masks, spheres):
    """The twin of the sweep gate (`pending_matches`): the candidate's terminator bleed equals the control's within the tolerance on every
    piece, i.e. the engine image is the image of the model the control describes (subsurface off: no light across the terminator)."""
    tolerance = THRESHOLDS["radius_sweep"]["tolerance"]
    pieces = []
    for s in spheres:
        if s.get("series") is None:
            continue
        c, r = terminator_bleed(np, candidate, masks, s["instance"]), terminator_bleed(np, reference, masks, s["instance"])
        pieces.append({"instance": s["instance"], "candidate": c, "reference": r,
                       "pass": c is not None and r is not None and abs(c - r) <= tolerance})
    return {"kind": "radius-statistic-match", "tolerance": tolerance, "pieces": pieces,
            "pass": bool(pieces) and all(piece["pass"] for piece in pieces)}


def evaluate_equal(np, candidate, reference, masks, spheres):
    """Ticket 17 sub-pixel: per piece the maximum relative luminance error of the candidate against the engine capture of the pair scene
    (no block) over the ROI stays at floating-point noise, and the candidate is not an empty image."""
    gate = THRESHOLDS["equal"]
    pieces = []
    for s in spheres:
        roi = piece_roi(masks, s["instance"])
        if not roi.any():
            pieces.append({"instance": s["instance"], "pixels": 0, "pass": False})
            continue
        error, floor = luminance_error(np, candidate, reference, roi, THRESHOLDS["subsurface"]["floor_fraction"])
        mean_luminance = float(luminance(np, candidate)[roi].mean())
        pieces.append({"instance": s["instance"], "weight": s.get("weight", 0.0), "radius": s.get("radius"), "pixels": int(roi.sum()),
                       "max": float(error.max()), "mean": float(error.mean()), "mean_luminance": mean_luminance,
                       "pass": bool(float(error.max()) <= gate["max_relative_luminance_error"] and
                                    mean_luminance >= gate["min_mean_luminance"])})
    return {"kind": "equal", "pieces": pieces, "worst_max": max(piece.get("max", 0.0) for piece in pieces),
            "limit": gate["max_relative_luminance_error"], "pass": all(piece["pass"] for piece in pieces)}


def evaluate(np, kind, candidate, reference, masks, spheres, band=1):
    if kind in TRANSMISSION_KINDS:
        return evaluate_transmission(np, candidate, reference, masks, spheres)
    if kind == "transmissive-shadow-ratio":
        return evaluate_shadow_ratio(np, candidate, reference, masks, spheres)
    if kind == "transmissive-shadow-curved":
        return evaluate_shadow_curved(np, candidate, reference, masks, spheres)
    if kind in SHADOW_KINDS:
        return evaluate_transmission(np, candidate, reference, masks, spheres, "transmissive_shadow", kind, reference)
    if kind == "subsurface":
        return evaluate_subsurface(np, candidate, reference, masks, spheres)
    if kind == "subsurface-report":
        return evaluate_subsurface(np, candidate, reference, masks, spheres, gated=False)
    if kind == "radius-monotonic":
        return evaluate_radius_monotonic(np, candidate, reference, masks, spheres)
    if kind == "radius-statistic-match":
        return evaluate_radius_statistic_match(np, candidate, reference, masks, spheres)
    if kind == "equal":
        return evaluate_equal(np, candidate, reference, masks, spheres)
    if kind == "direct":
        return evaluate_direct(np, candidate, reference, masks, spheres, band)
    if kind == "furnace":
        return evaluate_furnace(np, candidate, reference, masks, spheres)
    if kind == "furnace-transport":
        return evaluate_furnace_transport(np, candidate, reference, masks, spheres)
    if kind == "declared":
        return evaluate_declared(np, candidate, reference, masks, spheres)
    return evaluate_environment(np, candidate, reference, masks, spheres)


def expectation(kind, mode, result, always_green=False, pending=None):
    """Red-state assertions of ticket 02; None when a fixture is measured but not predicted.  A `pending`
    scenario (its engine side lands with that ticket) is asserted to FAIL in every mode."""
    if pending:
        return {"asserted": True, "expected_pass": False, "met": not result["pass"], "pending": pending}
    if mode == "green" or always_green:
        return {"asserted": True, "expected_pass": True, "met": result["pass"]}
    if kind == "declared":
        return {"asserted": False, "met": True, "note": "declared-model gate: measured, asserted in green mode only"}
    if kind in ("furnace", "furnace-transport"):
        return {"asserted": True, "expected_pass": False, "met": not result["pass"]}
    if kind == "direct":
        rows = {}
        for key, row in result["rows"].items():
            value = float(key)
            want = value in (0.0, 1.0)
            rows[key] = {"expected_pass": want, "observed_pass": row["pass"], "met": row["pass"] == want}
        return {"asserted": True, "rows": rows, "met": all(r["met"] for r in rows.values())}
    return {"asserted": False, "met": True, "note": "measured, not predicted before ticket 03"}


def control_verdicts(np, kind, reference, controls, masks, spheres, required, band=1):
    """Every required control rendered on the reference side must FAIL the gate of its fixture."""
    verdicts = {}
    for name in required:
        result = evaluate(np, kind, controls[name], reference, masks, spheres, band)
        verdicts[name] = {"fails_gate": not result["pass"], "gate_result": summarize(result)}
    return verdicts


def summarize(result):
    keep = {k: v for k, v in result.items() if k not in ("spheres", "rows", "objects", "pieces", "series")}
    if "objects" in result:
        keep["objects_failing"] = [t.get("label") or t["instance"] for t in result["objects"] if not t["pass"]]
    if "pieces" in result:
        keep["pieces_failing"] = [p["instance"] for p in result["pieces"] if not p["pass"]]
    if "series" in result:
        keep["series_failing"] = [x["series"] for x in result["series"] if not x["pass"]]
    if "rows" in result:
        keep["rows_failing"] = [k for k, v in result["rows"].items() if not v["pass"]]
    if result["kind"] in ("environment", "declared"):
        keep["spheres_failing"] = sum(1 for t in result["spheres"] if not t["pass"])
    return keep


# --- evaluation of one scenario ------------------------------------------------

def cached_reference_usable(reference_dir, snapshot_dir, literal_controls, declared_controls, visibility="none"):
    """A cached reference is reused only when it was rendered from THIS snapshot (recorded SHA-256 equals the
    snapshot's digest) with the same point-light visibility and holds every requested control; anything else
    is re-rendered."""
    from f12_reference import scene_digest
    meta_path = reference_dir / "reference.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("snapshot_sha256") != scene_digest(pathlib.Path(snapshot_dir).resolve()):
            return False
        if meta.get("visibility", "none") != visibility:
            return False
        files = [reference_dir / "reference.pfm", reference_dir / "masks.npz"]
        if not set(literal_controls) <= set(meta.get("controls", {})):
            return False
        files += [reference_dir / meta["controls"][c]["file"] for c in literal_controls]
        if declared_controls:
            declared = meta.get("declared")
            if not declared or not set(declared_controls) <= set(declared.get("controls", {})):
                return False
            files += [reference_dir / declared["file"]] + [reference_dir / declared["controls"][c]["file"]
                                                           for c in declared_controls]
        return all(f.is_file() for f in files)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def judge(np, kind, candidate, reference, masks, spheres, controls, mode):
    """One gate with its negative controls: result, verdicts, expectation."""
    result = evaluate(np, kind, candidate, reference, masks, spheres)
    verdicts = control_verdicts(np, kind, reference, controls, masks, spheres, list(controls))
    return result, verdicts, expectation(kind, mode, result)


def cached_transmission_reference_usable(reference_dir, snapshot_dir, controls, spp, tool="f15_transmission_reference.py",
                                         sample_key="spp", margin=None, base_tool=None):
    """A cached transmission (or transmissive-shadow) reference is reused only when it was rendered from THIS snapshot by THIS tool
    with the same sample count (and world margin) and holds every requested control file.  `base_tool`: a module the tool imports
    whose code shapes the images (the shadow reference builds its glass through f15_transmission_reference.py); its sha is
    recorded by the report as `tool_sha256_base` and a change re-renders."""
    from f12_reference import scene_digest
    meta_path = reference_dir / "reference.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("snapshot_sha256") != scene_digest(pathlib.Path(snapshot_dir).resolve()):
            return False
        if meta.get("tool_sha256") != sha256_file(REPO / "tools" / tool):
            return False
        if base_tool is not None and meta.get("tool_sha256_base") != sha256_file(REPO / "tools" / base_tool):
            return False
        if meta.get("kind") == "mitsuba" and meta.get(sample_key) != spp:
            return False
        if margin is not None and meta.get("margin_m") != margin:
            return False
        files = [reference_dir / "reference.pfm", reference_dir / "masks.npz"]
        if not set(controls) <= set(meta.get("controls", {})):
            return False
        files += [reference_dir / meta["controls"][c]["file"] for c in controls]
        return all(f.is_file() for f in files)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def evaluate_transmission_scenario(np, name, folder, mode):
    """One `transmission-*` key: the engine image against the Mitsuba (or declared) reference on the glass mask, the
    pending_matches twin and every negative control (all rendered by tools/f15_transmission_reference.py)."""
    from f12_reference import read_pfm
    import f15_transmission_reference as tref

    spec = SCENARIOS[name]
    declared = spec["kind"] == "transmission-declared"
    controls = list(spec["controls"])
    spp = spec.get("spp", REFERENCE_SPP)
    reference_dir = folder / "reference"
    if not cached_transmission_reference_usable(reference_dir, folder / "snapshot", controls, spp):
        if (reference_dir / "reference.json").is_file():
            print(f"{name}: the cached reference does not match this snapshot, tool or controls; re-rendering", file=sys.stderr)
        arguments = ["declared" if declared else "render", str(folder / "snapshot"), "--out", str(reference_dir)]
        if not declared:
            arguments += ["--spp", str(spp)]
        code = tref.main(arguments + ["--controls", *controls])
        if code:
            raise RuntimeError(f"{name}: reference render failed ({code})")
    report = json.loads((reference_dir / "reference.json").read_text())
    reference = read_pfm(reference_dir / "reference.pfm")
    candidate = read_pfm(folder / "engine.pfm")
    if candidate.shape != reference.shape:
        raise RuntimeError(f"{name}: engine capture {candidate.shape} and reference {reference.shape} differ in size")
    masks = dict(np.load(reference_dir / "masks.npz"))
    objects = report["objects"]
    images = {c: read_pfm(reference_dir / report["controls"][c]["file"]) for c in controls}
    result = evaluate_transmission(np, candidate, reference, masks, objects)
    verdicts = control_verdicts(np, spec["kind"], reference, images, masks, objects, controls)
    expected = expectation(spec["kind"], mode, result, spec.get("always_green", False), spec.get("pending"))
    twins = []
    if spec.get("pending_matches") in images:
        twin = evaluate_transmission(np, candidate, images[spec["pending_matches"]], masks, objects)
        expected = {**expected, "matches_control": spec["pending_matches"], "control_gate": summarize(twin),
                    "met": expected["met"] and twin["pass"]}
        twins.append(twin["pass"])
    entry = {"scenario": name, "kind": spec["kind"], "terms": spec["terms"], "reference_terms": report["terms"],
             "reference_maximum_radiance": report["maximum_radiance"], "quadrature_mask_pixels": 0,
             "roi_pixels": report["roi_pixels"], "snapshot_sha256": report["snapshot_sha256"], "result": result,
             "expectation": expected, "controls": verdicts, "pending": spec.get("pending"),
             "reference_settings": {k: report.get(k) for k in ("spp", "max_depth", "erosion_pixels", "engine_crossings", "kind")},
             "reference_noise": None if declared else report["reference"].get("noise"),
             "control_noise": {} if declared else {c: report["controls"][c].get("noise") for c in controls}}
    entry["discriminating"] = all(v["fails_gate"] for v in verdicts.values())
    entry["gates_pass"] = result["pass"]
    entry["pending_matches_met"] = all(twins)
    entry["expectation_met"] = expected["met"]
    return entry


def evaluate_shadow_scenario(np, name, folder, mode):
    """One `transmissive-shadow-*` key: the engine image against the Mitsuba ground truth on the caster shadow regions, the
    pending_matches twin, every negative control, and the declared straight-line model (which must pass the gate: feasibility)."""
    from f12_reference import read_pfm
    import f15_shadow_reference as sref

    spec = SCENARIOS[name]
    controls = list(spec["controls"])
    samples = spec.get("samples", sref.SAMPLES)
    margin = SHADOW_ENVELOPE[spec["envelope"]]
    reference_dir = folder / "reference"
    if not cached_transmission_reference_usable(reference_dir, folder / "snapshot", controls, samples, "f15_shadow_reference.py", "samples",
                                                margin, "f15_transmission_reference.py"):
        if (reference_dir / "reference.json").is_file():
            print(f"{name}: the cached reference does not match this snapshot, tool, margin or controls; re-rendering", file=sys.stderr)
        code = sref.main(["render", str(folder / "snapshot"), "--out", str(reference_dir), "--margin", str(margin), "--samples", str(samples),
                          *reference_arguments(name), "--controls", *controls])
        if code:
            raise RuntimeError(f"{name}: reference render failed ({code})")
    report = json.loads((reference_dir / "reference.json").read_text())
    reference = read_pfm(reference_dir / "reference.pfm")
    candidate = read_pfm(folder / "engine.pfm")
    if candidate.shape != reference.shape:
        raise RuntimeError(f"{name}: engine capture {candidate.shape} and reference {reference.shape} differ in size")
    masks = dict(np.load(reference_dir / "masks.npz"))
    objects = report["objects"]
    kind = spec["kind"]
    gated = True
    images = {c: read_pfm(reference_dir / report["controls"][c]["file"]) for c in controls}
    result = evaluate(np, kind, candidate, reference, masks, objects)
    verdicts = control_verdicts(np, kind, reference, images, masks, objects, controls)
    model = evaluate(np, kind, read_pfm(reference_dir / report["model"]["file"]), reference, masks, objects)
    expected = expectation(kind, mode, result, spec.get("always_green", False), spec.get("pending"))
    twins = []
    if spec.get("pending_matches") in images:
        # the twin's absolute floor is the ground truth's, not the control image's (an all-black control has floor 0)
        if kind == "transmissive-shadow-ratio":  # the floors are the truth's (an all-black twin has T = 0 and no floor of its own)
            twin = evaluate_shadow_ratio(np, candidate, images[spec["pending_matches"]], masks, objects, floor_from=reference)
        else:
            twin = evaluate_transmission(np, candidate, images[spec["pending_matches"]], masks, objects, "transmissive_shadow", kind, reference,
                                         gated=True)
        expected = {**expected, "matches_control": spec["pending_matches"], "control_gate": summarize(twin),
                    "met": expected["met"] and twin["pass"]}
        twins.append(twin["pass"])
    entry = {"scenario": name, "kind": kind, "terms": spec["terms"], "reference_terms": report["terms"],
             "reference_maximum_radiance": report["maximum_radiance"], "quadrature_mask_pixels": 0,
             "roi_pixels": report["roi_pixels"], "snapshot_sha256": report["snapshot_sha256"], "result": result,
             "expectation": expected, "controls": verdicts, "pending": spec.get("pending"), "gated": gated,
             "model_gate": summarize(model), "model_feasible": bool(model["pass"]), "envelope_m": margin,
             "reference_settings": {k: report.get(k) for k in ("samples", "max_bounces", "margin_m", "cone_factor", "point_light_radius",
                                                               "kind", "light")},
             "reference_noise": report["reference"].get("noise"), "model_vs_truth": report["model"].get("vs_truth"),
             "lit_floor_diagnostic": result.get("lit_floor_diagnostic")}
    entry["discriminating"] = all(v["fails_gate"] for v in verdicts.values()) and bool(model["pass"])
    entry["gates_pass"] = result["pass"]
    entry["pending_matches_met"] = all(twins)
    entry["expectation_met"] = expected["met"]
    return entry


def evaluate_layer_scenario(np, name, folder, mode):
    """One `layers-*` key (F15.4, ticket 23): exact equality of two engine captures (the stack at a weight vector against the single material),
    the transition ramp, or the furnace of the blended BSDF; every negative control must fail its gate."""
    from f12_reference import read_pfm
    import f15_layers as layers

    spec = SCENARIOS[name]
    kind = spec["kind"]
    candidate = read_pfm(folder / "engine.pfm")
    sibling = lambda key: read_pfm(folder.parent / key / "engine.pfm")  # noqa: E731
    entry = {"scenario": name, "kind": kind, "terms": spec["terms"], "reference_terms": ["engine capture pair" if kind != "layer-furnace" else
                                                                                          "tools/f15_reference.py environment quadrature of the blended BSDF"],
             "reference_maximum_radiance": None, "quadrature_mask_pixels": 0, "snapshot_sha256": None}
    verdicts = {}
    if kind == "layer-equal":
        pair = sibling(spec["pair"])
        envelope = spec.get("envelope", False)
        result = layers.evaluate_equal(np, candidate, pair, envelope=envelope)
        for control in spec["controls"]:
            c = layers.evaluate_equal(np, sibling(control), pair, envelope=envelope)
            verdicts[control] = {"fails_gate": not c["pass"], "gate_result": c}
        if spec.get("info_pair"):  # reported, never gated: the distance to the same scene on the plain base code path
            result["information_vs_" + spec["info_pair"]] = layers.compare_images(np, candidate, sibling(spec["info_pair"]))
        entry["roi_pixels"] = result["pixels"]
    elif kind == "layer-ramp":
        roi, x, z = layers.witness_positions(np, folder / "snapshot")
        u, _ = layers.plane_uv(np, x, z)
        e0, e1 = (sibling(k) for k in spec["endpoints"])
        affine = spec.get("affine", True)
        ramp = layers.evaluate_ramp_pooled if spec.get("pooled") else layers.evaluate_ramp
        result = ramp(np, candidate, e0, e1, roi, u, affine=affine)
        for control in spec["controls"]:
            c = ramp(np, sibling(control), e0, e1, roi, u, affine=affine)
            verdicts[control] = {"fails_gate": not c["pass"], "gate_result": c}
        entry["roi_pixels"] = int(roi.sum())
    else:
        roi, x, z = layers.witness_positions(np, folder / "snapshot")
        u, v = layers.plane_uv(np, x, z)
        scene = json.loads((folder / "snapshot" / "scene.json").read_text())
        position = np.array(scene["camera"]["position"], dtype=np.float64)
        inside = roi & (u > layers.EDGE_MARGIN * 2) & (u < 1 - layers.EDGE_MARGIN * 2) & (v > layers.EDGE_MARGIN * 2) & (v < 1 - layers.EDGE_MARGIN * 2)
        px, py, pz = x[inside], np.zeros(int(inside.sum())), z[inside]
        view = position[None, :] - np.stack([px, py, pz], axis=1)
        n_dot_v = view[:, 1] / np.linalg.norm(view, axis=1)
        exact = layers.furnace_reference(np, u[inside], v[inside], n_dot_v, declared=False)
        declared = layers.furnace_reference(np, u[inside], v[inside], n_dot_v, declared=True)
        result = layers.evaluate_furnace(np, candidate, inside, exact, {"additive-blend": exact["additive"]}, declared)
        verdicts = {c: {"fails_gate": v_["fails_gate"], "gate_result": v_} for c, v_ in result.pop("controls").items()}
        entry["roi_pixels"] = int(inside.sum())
        entry["reference_maximum_radiance"] = result["reference_max"]
    expected = expectation(kind, mode, result, spec.get("always_green", True), spec.get("pending"))
    entry.update(result=result, expectation=expected, controls=verdicts, pending=spec.get("pending"))
    entry["discriminating"] = all(v["fails_gate"] for v in verdicts.values())
    entry["gates_pass"] = result["pass"]
    entry["pending_matches_met"] = True
    entry["expectation_met"] = expected["met"]
    return entry


def evaluate_scenario(np, name, folder, mode):
    from f12_reference import read_pfm
    import f15_reference

    if SCENARIOS[name]["kind"] in SHADOW_KINDS:
        return evaluate_shadow_scenario(np, name, folder, mode)
    if SCENARIOS[name]["kind"] in TRANSMISSION_KINDS:
        return evaluate_transmission_scenario(np, name, folder, mode)
    if SCENARIOS[name]["kind"] in LAYER_KINDS:
        return evaluate_layer_scenario(np, name, folder, mode)

    spec = SCENARIOS[name]
    literal_controls, declared_controls = scenario_controls(name)
    reference_dir = folder / "reference"
    visibility = spec.get("visibility", "none")
    if spec.get("reference") == "subsurface":
        import f15_subsurface_reference as reference_tool
    else:
        reference_tool = f15_reference
    if not cached_reference_usable(reference_dir, folder / "snapshot", literal_controls, declared_controls,
                                   visibility):
        if (reference_dir / "reference.json").is_file():
            print(f"{name}: the cached reference does not match this snapshot or its controls; re-rendering",
                  file=sys.stderr)
        arguments = ["render", str(folder / "snapshot"), "--out", str(reference_dir), "--visibility", visibility,
                     "--controls", *literal_controls]
        if declared_controls:
            arguments += ["--declared-controls", *declared_controls]
        code = reference_tool.main(arguments + reference_arguments(name))
        if code:
            raise RuntimeError(f"{name}: reference render failed ({code})")
    report = json.loads((reference_dir / "reference.json").read_text())
    reference = read_pfm(reference_dir / "reference.pfm")
    candidate = read_pfm(folder / "engine.pfm")
    if spec.get("pair"):
        # An `equal` key compares the engine capture with the engine capture of its pair scene (same geometry, no block).
        reference = read_pfm(folder.parent / spec["pair"] / "engine.pfm")
    if candidate.shape != reference.shape:
        raise RuntimeError(f"{name}: engine capture {candidate.shape} and reference {reference.shape} differ in size")
    masks = dict(np.load(reference_dir / "masks.npz"))
    spheres = report["spheres"]
    entry = {"scenario": name, "kind": spec["kind"], "terms": spec["terms"], "reference_terms": report["terms"],
             "reference_maximum_radiance": report["maximum_radiance"],
             "quadrature_mask_pixels": report["quadrature_mask_pixels"], "roi_pixels": report["roi_pixels"],
             "snapshot_sha256": report["snapshot_sha256"]}
    checks = []  # (discriminating, gate result)
    twins = []  # pass/fail of the engine image against the pending_matches control, per gate that has one

    if spec["kind"] != "declared":
        band = spec.get("terminator_band", 1)
        controls = {c: read_pfm(reference_dir / report["controls"][c]["file"]) for c in literal_controls}
        result = evaluate(np, spec["kind"], candidate, reference, masks, spheres, band)
        if name == "roughness-extremes":  # sun + environment: direct statistics reported, environment gates
            direct = evaluate_direct(np, candidate, reference, masks, spheres)
            result = evaluate_environment(np, candidate, reference, masks, spheres, summarize(direct))
        verdicts = control_verdicts(np, spec["kind"], reference, controls, masks, spheres, literal_controls, band)
        # "pending_gate": "declared" keeps the literal gate asserted green (only the declared-model gate is pending).
        declared_only = spec.get("pending_gate") == "declared"
        expected = expectation(spec["kind"], mode, result, spec.get("always_green", False),
                               None if declared_only else spec.get("pending"))
        if spec.get("pending_matches") in controls and not declared_only:
            # Red for the right reason: before its ticket the engine must pass the gate against the
            # control that models the missing piece (e.g. DI evaluating the base BRDF).
            twin = evaluate(np, spec.get("twin_kind", spec["kind"]), candidate, controls[spec["pending_matches"]], masks, spheres,
                            band)
            expected = {**expected, "matches_control": spec["pending_matches"], "control_gate": summarize(twin),
                        "met": expected["met"] and twin["pass"]}
            twins.append(twin["pass"])
        entry.update(result=result, expectation=expected, controls=verdicts)
        checks.append((all(v["fails_gate"] for v in verdicts.values()), result["pass"]))

    if declared_controls:
        declared_report = report["declared"]
        declared_reference = read_pfm(reference_dir / declared_report["file"])
        controls = {c: read_pfm(reference_dir / declared_report["controls"][c]["file"]) for c in declared_controls}
        result = evaluate_declared(np, candidate, declared_reference, masks, spheres)
        verdicts = control_verdicts(np, "declared", declared_reference, controls, masks, spheres, declared_controls)
        expected = expectation("declared", mode, result, spec.get("always_green", False), spec.get("pending"))
        if spec.get("pending_matches") in controls:
            twin = evaluate_declared(np, candidate, controls[spec["pending_matches"]], masks, spheres)
            expected = {**expected, "matches_control": spec["pending_matches"], "control_gate": summarize(twin),
                        "met": expected["met"] and twin["pass"]}
            twins.append(twin["pass"])
        if spec["kind"] == "declared":
            entry.update(result=result, expectation=expected, controls=verdicts)
        else:
            entry["declared"] = {"result": result, "expectation": expected, "controls": verdicts,
                                 "dfg_table_sha256": declared_report.get("dfg_table_sha256")}
        checks.append((all(v["fails_gate"] for v in verdicts.values()), result["pass"]))

    # A radius sweep whose ground truth does not rise at every step cannot tell a sweep-blind engine apart: the fixture is redesigned.
    reference_ok = entry.get("result", {}).get("reference_monotonic", True)
    entry["discriminating"] = all(d for d, _ in checks) and reference_ok
    entry["gates_pass"] = all(p for _, p in checks)
    entry["pending"] = spec.get("pending")
    entry["pending_matches_met"] = all(twins)  # the engine image passes the gate against its pending_matches control
    entry["expectation_met"] = entry["expectation"]["met"] and entry.get("declared", {}).get("expectation", {"met": True})["met"]
    return entry


def verdict(entries, mode):
    """(status, exit code): blocking non-discrimination first, then the expectation."""
    if any(not e["discriminating"] for e in entries):
        return "BLOCKED_NON_DISCRIMINATING_FIXTURE", EXIT_NON_DISCRIMINATING
    if mode == "green":
        # A pending scenario must FAIL (its engine side has not landed); every other one must pass.
        # ... and, with a pending_matches control, must pass the gate against it (red for the right reason).
        passes = all(e.get("gates_pass", e["result"]["pass"]) != bool(e.get("pending")) and
                     e.get("pending_matches_met", True) for e in entries)
        return ("GREEN", EXIT_OK) if passes else ("RED", EXIT_UNEXPECTED)
    if all(e.get("expectation_met", e["expectation"]["met"]) for e in entries):
        return "EXPECTED_RED_CONFIRMED", EXIT_OK
    return "UNEXPECTED_RESULT", EXIT_UNEXPECTED


def result_lines(r):
    """Markdown table of one gate result."""
    lines = []
    if r["kind"] == "direct":
        lines += ["", "| metallic | pixels | mean | p95 | p99 | max outside mask | pass |", "|---|---|---|---|---|---|---|"]
        for k, row in r["rows"].items():
            lines.append(f"| {k} | {row['pixels']} | {row.get('mean', 0):.4f} | {row.get('p95', 0):.4f} | "
                         f"{row.get('p99', 0):.4f} | {row.get('max_outside_mask', 0):.4f} | {row['pass']} |")
    elif r["kind"] in ("subsurface", "subsurface-report"):
        lines += ["", f"worst piece mean {r['worst_mean']:.4f} (limit {r['limits']['mean']}), worst p95 {r['worst_p95']:.4f} "
                      f"(limit {r['limits']['p95']}){'' if r['gated'] else '; not gated (reported)'}", "",
                  "| piece | weight | radius | pixels | mean | p95 | candidate Y | reference Y | in regime (gated) | bound met |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in r["pieces"]:
            lines.append(f"| {t['instance']} | {t.get('weight', 0):g} | {t.get('radius') or 0:g} | {t['pixels']} | {t.get('mean', 0):.4f} | "
                         f"{t.get('p95', 0):.4f} | {t.get('candidate_mean_luminance', 0):.4f} | "
                         f"{t.get('reference_mean_luminance', 0):.4f} | {t.get('gated')} | {t.get('bound_met')} |")
    elif r["kind"] == "layer-equal":
        lines += ["", f"gate {r['gate']} (limit {r['limit']}): {r['bitwise_differing']} of {r['pixels']} pixels differ bitwise, max {r['max_ulp']} ULP, "
                      f"max relative {r['max_relative']:.3e} (floored by 0.05 x mean: {r['max_relative_floored']:.3e}), {r['eight_bit_differing']} pixels differ in the 8-bit image"]
    elif r["kind"] == "layer-ramp":
        lines += ["", f"{r['usable_pixels']} pixels in {r['rows']} rows; rows with a reversal {r['rows_with_reversal']} (worst drop {r['worst_drop']:.2e}, "
                      f"tolerance {r['tolerance']}); max |t - u| {r['max_mask_error']:.4f} (limit {r['mask_error_limit']}); ends (mean t / 1-t)"
                      f" {r['mean_t_near_low_end']:.4f} / {r['mean_one_minus_t_near_high_end']:.4f}; fitted slope {r['fitted_slope']:.4f}"]
    elif r["kind"] == "layer-furnace":
        d = r["deviation_from_bsdf_mix"]
        lines += ["", f"max pixel {r['max_pixel']:.5f} (limit {r['limit']}), mean {r['mean_pixel']:.5f}; reference (blended BSDF) max {r['reference_max']:.5f}, "
                      f"mean {r['reference_mean']:.5f}; engine vs reference mean relative {r['engine_vs_reference_mean_relative']:.4f}, p99 "
                      f"{r['engine_vs_reference_p99_relative']:.4f}; parameter blend minus BSDF-space mix: max |d| {d['max_abs']:.5f}, mean {d['mean']:.5f}, "
                      f"range [{d['min']:.5f}, {d['max']:.5f}]"]
        m = r.get("declared_model")
        if m:
            md = m["deviation_from_bsdf_mix"]
            lines += ["", f"engine model (ADR-0006 declared split sum): blended max {m['blended_max']:.5f}, BSDF-space mix max {m['mixed_max']:.5f}, additive max "
                          f"{m['additive_max']:.5f}; engine vs declared mean relative {m['engine_vs_declared_mean_relative']:.4f}, p99 "
                          f"{m['engine_vs_declared_p99_relative']:.4f}; blend minus mix: max |d| {md['max_abs']:.5f}, mean {md['mean']:.5f}, range "
                          f"[{md['min']:.5f}, {md['max']:.5f}]"]
    elif r["kind"] == "radius-monotonic":
        lines += ["", f"statistic: {r['statistic']}; tolerance {r['tolerance']}; ground truth rises at every step: "
                      f"{r['reference_monotonic']}", "", "| series | radii | candidate | reference | pass |", "|---|---|---|---|---|"]
        for x in r["series"]:
            fmt = lambda values: ", ".join("-" if v is None else f"{v:.4f}" for v in values)  # noqa: E731
            lines.append(f"| {x['series']} | {', '.join(f'{v:g}' for v in x['radii'])} | {fmt(x['candidate'])} | {fmt(x['reference'])} | "
                         f"{x['pass']} |")
    elif r["kind"] == "equal":
        lines += ["", f"worst per-piece max relative luminance error {r['worst_max']:.6f} (limit {r['limit']})", "",
                  "| piece | weight | radius | pixels | max | mean | mean Y | pass |", "|---|---|---|---|---|---|---|---|"]
        for t in r["pieces"]:
            lines.append(f"| {t['instance']} | {t.get('weight', 0):g} | {t.get('radius') or 0:g} | {t['pixels']} | {t.get('max', 0):.6f} | "
                         f"{t.get('mean', 0):.6f} | {t.get('mean_luminance', 0):.4f} | {t['pass']} |")
    elif r["kind"] == "environment":
        lines += ["", "| metallic | roughness | mean | p95 | pass |", "|---|---|---|---|---|"]
        for t in r["spheres"]:
            lines.append(f"| {t['metallic']} | {t['roughness']} | {t.get('mean', 0):.4f} | {t.get('p95', 0):.4f} | {t['pass']} |")
    elif r["kind"] in ("transmission", *SHADOW_KINDS):
        limits = THRESHOLDS["transmissive_shadow" if r["kind"] in SHADOW_KINDS else "transmission"]
        pooled = r.get("pooled_not_gated") or {}
        if pooled:
            lines += ["", f"pooled glass mask (information, not gated): mean {pooled['mean']:.4f}, p95 {pooled['p95']:.4f}, {pooled['pixels']} pixels"]
        diagnostic = r.get("lit_floor_diagnostic")
        if diagnostic:
            lines += ["", f"lit-floor diagnostic (not gated): candidate / reference level {diagnostic['lit_level_ratio']:.4f} (median "
                          f"{diagnostic['lit_level_median_ratio']:.4f}, p5 {diagnostic['lit_level_p5_ratio']:.4f}, {diagnostic['lit_pixels']} pixels)"]
        lines += ["", f"worst object mean {r['worst_mean']:.4f} (limit {limits['mean_per_object']}), "
                      f"worst p95 {r['worst_p95']:.4f} (limit {limits['p95_per_object']}), floor {r['floor']:.4f}"
                      f"{'' if r.get('gated', True) else '; not gated (reported)'}", "",
                  "| object | pixels | mean | p95 | pass |", "|---|---|---|---|---|"]
        for t in r["objects"]:
            lines.append(f"| {t['label']} | {t['pixels']} | {t.get('mean', 0):.4f} | {t.get('p95', 0):.4f} | {t['pass']} |")
    elif r["kind"] == "declared":
        lines += ["", f"worst sphere mean {r['worst_mean']:.5f} (limit {THRESHOLDS['declared']['mean_per_sphere']}), "
                      f"worst p99 {r['worst_p99']:.5f} (limit {THRESHOLDS['declared']['p99_per_sphere']})", "",
                  "| metallic | roughness | mean | p99 | pass |", "|---|---|---|---|---|"]
        for t in r["spheres"]:
            lines.append(f"| {t['metallic']} | {t['roughness']} | {t.get('mean', 0):.5f} | {t.get('p99', 0):.5f} | {t['pass']} |")
    else:
        gate = (f"max sphere mean {r['max_sphere_mean']:.5f} (limit {r['limit']}, energy in expectation), max pixel {r['max_pixel']:.5f}"
                if r["kind"] == "furnace-transport" else f"max pixel {r['max_pixel']:.5f} (limit {r['limit']})")
        lines += ["", f"{gate}, reference max {r['reference_max_pixel']:.5f}", "",
                  "| metallic | roughness | max | mean | energy loss | reference loss |", "|---|---|---|---|---|---|"]
        for t in r["spheres"]:
            lines.append(f"| {t.get('metallic')} | {t.get('roughness')} | {t.get('max', 0):.4f} | {t.get('mean', 0):.4f} | "
                         f"{t.get('energy_loss', 0):.4f} | {t.get('reference_energy_loss', 0):.4f} |")
    return lines


def control_text(controls):
    return ", ".join(f"{k}: {'fails' if v['fails_gate'] else 'PASSES (non-discriminating)'}" for k, v in controls.items())


def markdown(manifest, entries, status):
    os_text = os_description(manifest.get("os") or {})
    lines = [f"# F15.3 material validation: {status}", "",
             f"commit `{manifest.get('commit')}`, binary `{manifest.get('binary_sha256')}`, {os_text}", "",
             "| scenario | gate | declared-model gate | expectation | negative controls |", "|---|---|---|---|---|"]
    for e in entries:
        ex = e["expectation"]
        declared = e.get("declared")
        declared_cell = "-" if not declared else ("pass" if declared["result"]["pass"] else "FAIL")
        controls = control_text(e["controls"])
        if declared:
            controls += "; declared: " + control_text(declared["controls"])
        lines.append(f"| {e['scenario']} | {'pass' if e['result']['pass'] else 'FAIL'} | {declared_cell} | "
                     f"{'met' if e.get('expectation_met', ex['met']) else 'NOT met'}"
                     f"{'' if ex.get('asserted') else ' (not asserted)'} | {controls} |")
    for e in entries:
        lines += ["", f"## {e['scenario']} ({e['kind']})", f"Image contains: {e['terms']}.",
                  f"Reference terms: {', '.join(e['reference_terms'])}; ROI pixels {e.get('roi_pixels', 0)}; "
                  f"quadrature mask pixels {e['quadrature_mask_pixels']}."]
        lines += result_lines(e["result"])
        if e.get("declared"):
            lines += ["", f"### {e['scenario']}: declared environment model (DFG table)"]
            lines += result_lines(e["declared"]["result"])
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------

PIPELINES_FALLBACK_DRAWS = re.compile(r"PIPELINES .*?fallbacks \d+ \(draws (\d+)\)")


def fallback_draws(log_text):
    """Fallback draws of the LAST pipeline summary line of a capture log (None when the log has none)."""
    found = PIPELINES_FALLBACK_DRAWS.findall(log_text)
    return int(found[-1]) if found else None


REFERENCE_BASE_TOOL = {"f15_subsurface_reference.py": "f15_reference.py", "f15_shadow_reference.py": "f15_transmission_reference.py"}


def seed_reference_from_cache(cache_folder, folder):
    """Copy `cache_folder`/reference into `folder`/reference when the cached reference was rendered by the CURRENT tool (and base module):
    the scenario's own cache check (snapshot digest, controls, samples) then decides whether it is usable or re-rendered.  Returns the reason
    when nothing is copied (reported, never fatal)."""
    import shutil
    source, target = pathlib.Path(cache_folder) / "reference", pathlib.Path(folder) / "reference"
    meta_path = source / "reference.json"
    if target.exists():
        return "target already has a reference"
    if not meta_path.is_file():
        return "no cached reference"
    try:
        meta = json.loads(meta_path.read_text())
        tool = pathlib.Path(meta["tool"]).name
        if meta.get("tool_sha256") != sha256_file(REPO / "tools" / tool):
            return f"cached reference made by another revision of {tool}"
        base = REFERENCE_BASE_TOOL.get(tool)
        if base and meta.get("tool_sha256_base") != sha256_file(REPO / "tools" / base):
            return f"cached reference made by another revision of {base}"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"unreadable cached reference ({error})"
    shutil.copytree(source, target)
    return None


def capture(command, log, timeout):
    """One renderer process through run_checked (tests replace this)."""
    from run_checked import run_checked
    return run_checked(command, log, timeout=timeout)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, epilog=EXIT_CODES_HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-material")
    parser.add_argument("--only", action="append", default=[], choices=sorted(SCENARIOS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="capture every scenario serially, then evaluate")
    mode.add_argument("--evaluate", action="store_true", help="evaluate existing captures (no GPU process)")
    parser.add_argument("--expect", choices=["red", "green"], default="red")
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reference-cache", type=pathlib.Path, default=None,
                        help="an earlier --out whose references are reused when the current tools rendered them (the per-scenario cache checks "
                             "still apply: snapshot digest, controls, samples)")
    return parser


def run_captures(parser, args, names, out, binary):
    """Serial captures under the lock; the GPU guard runs before EVERY capture."""
    if not binary.is_file():
        parser.error("--run requires an existing compiled renderer; this runner never builds")
    folders = {n: out / n for n in names}
    for n, folder in folders.items():  # before anything starts: a stale folder is a usage error
        if folder.exists() and any(folder.iterdir()):
            parser.error(f"{folder} is not empty: choose a new --out")
    with gpu_lock(args.gpu_lock):
        for position, n in enumerate(names):
            busy = running_gpu_processes()
            if busy:
                raise GpuBusy(f"refusing to start {n}: other GPU jobs are running: {', '.join(busy)}")
            folders[n].mkdir(parents=True, exist_ok=True)
            status = capture(capture_command(binary, n, folders[n], args.seed), folders[n] / "engine.log", args.timeout)
            if not status["passed"]:
                raise CaptureFailed(f"{n}: capture failed: {status['failures']}")
            log = folders[n] / "engine.log"
            draws = fallback_draws(log.read_text(errors="replace")) if log.is_file() else None
            if draws is None or draws > 0:
                raise CaptureFailed(f"{n}: capture served fallback pipelines (fallback draws {draws}); its image is not the engine's")
            if position + 1 < len(names):
                time.sleep(2)  # serial and quiet between GPU jobs


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)  # SystemExit(2) on a usage error
    names = args.only or list(SCENARIOS)
    if not args.only and terrain_materials_dir() is None and (args.run or args.evaluate):
        skipped = [n for n in names if SCENARIOS[n].get("cc0")]
        names = [n for n in names if not SCENARIOS[n].get("cc0")]
        print(f"skipping {len(skipped)} cc0 keys: no unpacked CC0 terrain sets (set {TERRAIN_MATERIALS_ENV} or run tools/fetch_terrain_materials.py)", file=sys.stderr)
    for n in list(names):  # dependencies (pair, endpoints, control captures) are captured too
        for dep in [SCENARIOS[n].get("pair"), *SCENARIOS[n].get("endpoints", []), *SCENARIOS[n].get("needs", [])]:
            if dep and dep not in names:
                names.append(dep)
    out = args.out.resolve()
    binary = args.binary.resolve()
    metallib = binary.parent / "shaders" / "phosphor.metallib"
    manifest = build_manifest(binary, metallib, names, args.seed)

    if not args.run and not args.evaluate:
        out.mkdir(parents=True, exist_ok=True)
        plan = {"state": "NOT_EXECUTED", **manifest, "commands": {
            n: {"capture": capture_command(binary, n, out / n, args.seed), "reference": reference_command(n, out / n)}
            for n in names}}
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        for n in names:
            print(f"# {n}: {SCENARIOS[n]['terms']}")
            print(" ".join(plan["commands"][n]["capture"]))
            print(" ".join(plan["commands"][n]["reference"]))
        print(f"NOT_EXECUTED: planned {len(names)} F15.3 scenarios, plan at {out / 'plan.json'}")
        return EXIT_OK

    if args.run:
        run_captures(parser, args, names, out, binary)

    if args.reference_cache:
        for n in names:
            if not SCENARIOS[n].get("capture_only"):
                reason = seed_reference_from_cache(args.reference_cache.resolve() / n, out / n)
                print(f"{n}: reference cache {'reused' if reason is None else 'not used: ' + reason}", file=sys.stderr)
    import numpy as np
    entries = [evaluate_scenario(np, n, out / n, args.expect) for n in names if not SCENARIOS[n].get("capture_only")]
    status, code = verdict(entries, args.expect)
    manifest["state"] = status
    document = {**manifest, "mode": args.expect, "status": status, "scenarios": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "results.json").write_text(json.dumps(document, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(manifest, entries, status))
    print(markdown(manifest, entries, status))
    print(f"{status} (exit {code})")
    return code


def main(argv=None):
    """Run and map every failure to its documented exit code (see EXIT_CODES_HELP)."""
    try:
        return run(argv)
    except SystemExit:
        raise  # argparse: usage error (2) or --help (0)
    except GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return EXIT_GPU_BUSY
    except CaptureFailed as error:
        print(str(error), file=sys.stderr)
        return EXIT_CAPTURE_FAILED
    except Exception:  # noqa: BLE001 - every other failure is an internal error, never a gate result
        traceback.print_exc()
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
