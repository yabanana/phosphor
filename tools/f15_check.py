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
- declared (base-grid-env-resolve; also furnace-resolve next to its furnace
  gate): the engine image against the independent declared-environment model of
  tools/f15_reference.py (DFG table, ADR-0006): per sphere mean relative error
  <= 0.5 % and per-pixel p99 <= 1 % (the absolute floor rule above).  Its
  negative controls (diffuse unweighted, Karis fit instead of the table,
  diffuse weighted by the metallic-blended F0 Fresnel) must all fail it.
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
    "source": ".scratch/f15/decisions/11-reference-method-and-thresholds.md (incl. review amendments)",
}
FRAMES, RESOLUTION = 128, "1280x720"

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
    "specular-grid-point@clustered": {"kind": "direct", "lighting": "clustered", "visibility": "rays", "denoise": "off",
                                      "terminator_band": 2,
                                      "controls": ["specular-ignored", "f90-one", "specular-weight-not-on-f0"],
                                      "terms": "one point light only, clustered DI with RT visibility "
                                               "(--lighting clustered --rt on), unfiltered (--lighting-denoise off), "
                                               "on the ior x specular grid"},
}
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
             ("f15_check.py", "f15_reference.py", "f15_common.py", "f12_reference.py", "run_checked.py")}
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
    return name.split("@", 1)[0]


def capture_command(binary, name, folder, seed=1):
    """The exact single GPU process of one scenario; CPU-side everything else."""
    last = str(FRAMES - 1)
    lighting = SCENARIOS[name].get("lighting", "legacy")
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", bench_scenario(name),
            "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "off" if lighting == "legacy" and not SCENARIOS[name].get("rt") else "on",
            "--shadows", "off",
            "--lighting", lighting, "--gi", "off", "--reflections", SCENARIOS[name].get("reflections", "ssr"), "--ao", "off",
            "--lighting-denoise", SCENARIOS[name].get("denoise", "custom"), "--reflection-samples", "8",
            "--lighting-seed", str(seed),
            "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(FRAMES),
            "--warmup", "0", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1",
            "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", last,
            "--capture-linear-signal", "hdr",
            "--export-reference", str(folder / "snapshot"), "--export-reference-frame", last,
            "--report", str(folder / "report.json")]


def scenario_controls(name):
    """(literal-model controls, declared-model controls) the reference must render for one scenario."""
    spec = SCENARIOS[name]
    if spec["kind"] == "declared":
        return [], list(spec["controls"])
    return list(spec["controls"]), list(spec.get("declared", []))


def reference_command(name, folder):
    controls, declared = scenario_controls(name)
    command = ["uv", "run", "--offline", "--no-project", "--python", "3.13", "--with", "mitsuba==3.9.1",
               "--with", "numpy", "python", "-I", str(REPO / "tools" / "f15_reference.py"), "render",
               str(folder / "snapshot"), "--out", str(folder / "reference"),
               "--visibility", SCENARIOS[name].get("visibility", "none"), "--controls", *controls]
    if declared:
        command += ["--declared-controls", *declared]
    return command


# --- statistics (pure numpy) --------------------------------------------------

def relative_error(np, candidate, reference, roi):
    """Per-pixel relative error over the ROI (max over RGB), with the absolute floor of decision 11."""
    floor = THRESHOLDS["floor_fraction"] * float(reference[roi].mean())
    error = np.abs(candidate[roi] - reference[roi]) / np.maximum(reference[roi], floor)
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


def evaluate(np, kind, candidate, reference, masks, spheres, band=1):
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
    keep = {k: v for k, v in result.items() if k not in ("spheres", "rows")}
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


def evaluate_scenario(np, name, folder, mode):
    from f12_reference import read_pfm
    import f15_reference

    spec = SCENARIOS[name]
    literal_controls, declared_controls = scenario_controls(name)
    reference_dir = folder / "reference"
    visibility = spec.get("visibility", "none")
    if not cached_reference_usable(reference_dir, folder / "snapshot", literal_controls, declared_controls,
                                   visibility):
        if (reference_dir / "reference.json").is_file():
            print(f"{name}: the cached reference does not match this snapshot or its controls; re-rendering",
                  file=sys.stderr)
        arguments = ["render", str(folder / "snapshot"), "--out", str(reference_dir), "--visibility", visibility,
                     "--controls", *literal_controls]
        if declared_controls:
            arguments += ["--declared-controls", *declared_controls]
        code = f15_reference.main(arguments)
        if code:
            raise RuntimeError(f"{name}: reference render failed ({code})")
    report = json.loads((reference_dir / "reference.json").read_text())
    reference = read_pfm(reference_dir / "reference.pfm")
    candidate = read_pfm(folder / "engine.pfm")
    if candidate.shape != reference.shape:
        raise RuntimeError(f"{name}: engine capture {candidate.shape} and reference {reference.shape} differ in size")
    masks = dict(np.load(reference_dir / "masks.npz"))
    spheres = report["spheres"]
    entry = {"scenario": name, "kind": spec["kind"], "terms": spec["terms"], "reference_terms": report["terms"],
             "reference_maximum_radiance": report["maximum_radiance"],
             "quadrature_mask_pixels": report["quadrature_mask_pixels"], "roi_pixels": report["roi_pixels"],
             "snapshot_sha256": report["snapshot_sha256"]}
    checks = []  # (discriminating, gate result)

    if spec["kind"] != "declared":
        band = spec.get("terminator_band", 1)
        controls = {c: read_pfm(reference_dir / report["controls"][c]["file"]) for c in literal_controls}
        result = evaluate(np, spec["kind"], candidate, reference, masks, spheres, band)
        if name == "roughness-extremes":  # sun + environment: direct statistics reported, environment gates
            direct = evaluate_direct(np, candidate, reference, masks, spheres)
            result = evaluate_environment(np, candidate, reference, masks, spheres, summarize(direct))
        verdicts = control_verdicts(np, spec["kind"], reference, controls, masks, spheres, literal_controls, band)
        expected = expectation(spec["kind"], mode, result, spec.get("always_green", False), spec.get("pending"))
        if spec.get("pending_matches"):
            # Red for the right reason: before its ticket the engine must pass the gate against the
            # control that models the missing piece (e.g. DI evaluating the base BRDF).
            twin = evaluate(np, spec["kind"], candidate, controls[spec["pending_matches"]], masks, spheres, band)
            expected = {**expected, "matches_control": spec["pending_matches"], "control_gate": summarize(twin),
                        "met": expected["met"] and twin["pass"]}
        entry.update(result=result, expectation=expected, controls=verdicts)
        checks.append((all(v["fails_gate"] for v in verdicts.values()), result["pass"]))

    if declared_controls:
        declared_report = report["declared"]
        declared_reference = read_pfm(reference_dir / declared_report["file"])
        controls = {c: read_pfm(reference_dir / declared_report["controls"][c]["file"]) for c in declared_controls}
        result = evaluate_declared(np, candidate, declared_reference, masks, spheres)
        verdicts = control_verdicts(np, "declared", declared_reference, controls, masks, spheres, declared_controls)
        expected = expectation("declared", mode, result, spec.get("always_green", False), spec.get("pending"))
        if spec["kind"] == "declared":
            entry.update(result=result, expectation=expected, controls=verdicts)
        else:
            entry["declared"] = {"result": result, "expectation": expected, "controls": verdicts,
                                 "dfg_table_sha256": declared_report.get("dfg_table_sha256")}
        checks.append((all(v["fails_gate"] for v in verdicts.values()), result["pass"]))

    entry["discriminating"] = all(d for d, _ in checks)
    entry["gates_pass"] = all(p for _, p in checks)
    entry["pending"] = spec.get("pending")
    entry["expectation_met"] = entry["expectation"]["met"] and entry.get("declared", {}).get("expectation", {"met": True})["met"]
    return entry


def verdict(entries, mode):
    """(status, exit code): blocking non-discrimination first, then the expectation."""
    if any(not e["discriminating"] for e in entries):
        return "BLOCKED_NON_DISCRIMINATING_FIXTURE", EXIT_NON_DISCRIMINATING
    if mode == "green":
        # A pending scenario must FAIL (its engine side has not landed); every other one must pass.
        passes = all(e.get("gates_pass", e["result"]["pass"]) != bool(e.get("pending")) for e in entries)
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
    elif r["kind"] == "environment":
        lines += ["", "| metallic | roughness | mean | p95 | pass |", "|---|---|---|---|---|"]
        for t in r["spheres"]:
            lines.append(f"| {t['metallic']} | {t['roughness']} | {t.get('mean', 0):.4f} | {t.get('p95', 0):.4f} | {t['pass']} |")
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
                  f"Reference terms: {', '.join(e['reference_terms'])}; ROI pixels {e['roi_pixels']}; "
                  f"quadrature mask pixels {e['quadrature_mask_pixels']}."]
        lines += result_lines(e["result"])
        if e.get("declared"):
            lines += ["", f"### {e['scenario']}: declared environment model (DFG table)"]
            lines += result_lines(e["declared"]["result"])
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------

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
            if position + 1 < len(names):
                time.sleep(2)  # serial and quiet between GPU jobs


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)  # SystemExit(2) on a usage error
    names = args.only or list(SCENARIOS)
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

    import numpy as np
    entries = [evaluate_scenario(np, n, out / n, args.expect) for n in names]
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
