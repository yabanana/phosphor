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

Gates (relative error = |a - b| / max(b, 0.05 * mean_ROI(b)), per pixel, the
maximum over RGB; ROI = sphere pixels without the silhouette, from the
reference ray casting):
- direct (base-grid-sun): per metallic row, p99 <= 1 % and max <= 3 % over the
  lit ROI, the max outside the declared sun-quadrature mask (narrow lobes).
- environment (base-grid-env, roughness-extremes): mean <= 5 % per sphere; the
  per-sphere table is published.  roughness-extremes holds sun and
  environment: only the environment criterion gates, direct statistics are
  reported.
- furnace (furnace-base): no engine pixel above 1.0 + 0.5 %; energy loss per
  sphere is recorded, not gated.
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
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

THRESHOLDS = {
    "relative_error": "abs(a - b) / max(b, floor_fraction * mean_ROI(b)), per pixel, maximum over RGB",
    "floor_fraction": 0.05,
    "direct": {"p99": 0.01, "max_outside_quadrature_mask": 0.03},
    "environment": {"mean_per_sphere": 0.05},
    "furnace": {"max_pixel": 1.005, "energy_loss": "recorded, not gated"},
    "source": ".scratch/f15/decisions/11-reference-method-and-thresholds.md (incl. review amendments)",
}
FRAMES, RESOLUTION = 128, "1280x720"

SCENARIOS = {
    "base-grid-sun": {"kind": "direct", "controls": ["alpha-roughness", "fresnel-off", "engine-f0-blend"],
                      "terms": "sun disk only (environment 0, no shadows, no GI, reflections source off)"},
    "base-grid-env": {"kind": "environment", "controls": ["alpha-roughness", "fresnel-off"],
                      "terms": "constant environment only (radiance 0.5, no sun)"},
    "furnace-base": {"kind": "furnace", "controls": ["diffuse-unweighted"],
                     "terms": "constant environment radiance 1, albedo 1, no sun"},
    "roughness-extremes": {"kind": "environment", "controls": ["alpha-roughness"],
                           "terms": "sun disk plus constant environment 0.3; only the environment criterion gates"},
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
            raise RuntimeError("another cooperating Phosphor GPU verification owns the lock")
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


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_text(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def build_manifest(binary, metallib, scenarios, seed):
    git = lambda *a: command_text(["git", "-C", str(REPO), *a])  # noqa: E731
    power = command_text(["pmset", "-g", "batt"]) or ""
    tools = {name: sha(REPO / "tools" / name) for name in
             ("f15_check.py", "f15_reference.py", "f12_reference.py", "run_checked.py")}
    return {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "binary": str(binary), "binary_sha256": sha(binary) if binary.is_file() else None,
            "metallib_sha256": sha(metallib) if metallib.is_file() else None,
            "tool_sha256": tools, "seed": seed, "reference_seed": None,
            "reference_note": "the Appendix B reference is deterministic (no random numbers)",
            "os": command_text(["sw_vers"]), "sdk_version": command_text(["xcrun", "--show-sdk-version"]),
            "power": {"pmset_batt": power, "on_ac": "AC Power" in power},
            "frames": FRAMES, "resolution": RESOLUTION, "thresholds": THRESHOLDS,
            "scenarios": {name: SCENARIOS[name] for name in scenarios}}


# --- commands -----------------------------------------------------------------

def capture_command(binary, name, folder, seed=1):
    """The exact single GPU process of one scenario; CPU-side everything else."""
    last = str(FRAMES - 1)
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", name,
            "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "off", "--shadows", "off",
            "--lighting", "legacy", "--gi", "off", "--reflections", "ssr", "--ao", "off",
            "--lighting-denoise", "custom", "--reflection-samples", "8", "--lighting-seed", str(seed),
            "--post", "--upscaler", "native", "--resolution", RESOLUTION, "--frames", str(FRAMES),
            "--warmup", "0", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1",
            "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", last,
            "--capture-linear-signal", "hdr",
            "--export-reference", str(folder / "snapshot"), "--export-reference-frame", last,
            "--report", str(folder / "report.json")]


def reference_command(name, folder):
    controls = SCENARIOS[name]["controls"]
    return ["uv", "run", "--offline", "--no-project", "--python", "3.13", "--with", "mitsuba==3.9.1",
            "--with", "numpy", "python", "-I", str(REPO / "tools" / "f15_reference.py"), "render",
            str(folder / "snapshot"), "--out", str(folder / "reference"), "--controls", *controls]


# --- statistics (pure numpy) --------------------------------------------------

def relative_error(np, candidate, reference, roi):
    """Per-pixel relative error over the ROI (max over RGB), with the absolute floor of decision 11."""
    floor = THRESHOLDS["floor_fraction"] * float(reference[roi].mean())
    error = np.abs(candidate[roi] - reference[roi]) / np.maximum(reference[roi], floor)
    return error.max(axis=-1), floor


def evaluate_direct(np, candidate, reference, masks, spheres):
    """Per metallic row: p99 and max (outside the quadrature mask) over lit ROI pixels."""
    gate = THRESHOLDS["direct"]
    lit = masks["roi"] & masks["lit"]
    error_all, floor = relative_error(np, candidate, reference, lit)
    narrow = masks["narrow"][lit]
    sphere = masks["sphere"][lit]
    rows = {}
    for value in sorted({s["metallic"] for s in spheres}):
        members = [s["instance"] for s in spheres if s["metallic"] == value]
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


def evaluate(np, kind, candidate, reference, masks, spheres):
    if kind == "direct":
        return evaluate_direct(np, candidate, reference, masks, spheres)
    if kind == "furnace":
        return evaluate_furnace(np, candidate, reference, masks, spheres)
    return evaluate_environment(np, candidate, reference, masks, spheres)


def expectation(kind, mode, result):
    """Red-state assertions of ticket 02; None when a fixture is measured but not predicted."""
    if mode == "green":
        return {"asserted": True, "expected_pass": True, "met": result["pass"]}
    if kind == "furnace":
        return {"asserted": True, "expected_pass": False, "met": not result["pass"]}
    if kind == "direct":
        rows = {}
        for key, row in result["rows"].items():
            value = float(key)
            want = value in (0.0, 1.0)
            rows[key] = {"expected_pass": want, "observed_pass": row["pass"], "met": row["pass"] == want}
        return {"asserted": True, "rows": rows, "met": all(r["met"] for r in rows.values())}
    return {"asserted": False, "met": True, "note": "measured, not predicted before ticket 03"}


def control_verdicts(np, kind, reference, controls, masks, spheres, required):
    """Every required control rendered on the reference side must FAIL the gate of its fixture."""
    verdicts = {}
    for name in required:
        result = evaluate(np, kind, controls[name], reference, masks, spheres)
        verdicts[name] = {"fails_gate": not result["pass"], "gate_result": summarize(result)}
    return verdicts


def summarize(result):
    keep = {k: v for k, v in result.items() if k not in ("spheres", "rows")}
    if "rows" in result:
        keep["rows_failing"] = [k for k, v in result["rows"].items() if not v["pass"]]
    if result["kind"] == "environment":
        keep["spheres_failing"] = sum(1 for t in result["spheres"] if not t["pass"])
    return keep


# --- evaluation of one scenario ------------------------------------------------

def evaluate_scenario(np, name, folder, mode):
    from f12_reference import read_pfm
    import f15_reference

    spec = SCENARIOS[name]
    reference_dir = folder / "reference"
    if not (reference_dir / "reference.json").is_file():
        code = f15_reference.main(["render", str(folder / "snapshot"), "--out", str(reference_dir),
                                   "--controls", *spec["controls"]])
        if code:
            raise RuntimeError(f"{name}: reference render failed ({code})")
    report = json.loads((reference_dir / "reference.json").read_text())
    reference = read_pfm(reference_dir / "reference.pfm")
    candidate = read_pfm(folder / "engine.pfm")
    if candidate.shape != reference.shape:
        raise RuntimeError(f"{name}: engine capture {candidate.shape} and reference {reference.shape} differ in size")
    masks = dict(np.load(reference_dir / "masks.npz"))
    spheres = report["spheres"]
    controls = {c: read_pfm(reference_dir / report["controls"][c]["file"]) for c in spec["controls"]}
    result = evaluate(np, spec["kind"], candidate, reference, masks, spheres)
    if name == "roughness-extremes":  # sun + environment: direct statistics reported, environment gates
        direct = evaluate_direct(np, candidate, reference, masks, spheres)
        result = evaluate_environment(np, candidate, reference, masks, spheres, summarize(direct))
    verdicts = control_verdicts(np, spec["kind"], reference, controls, masks, spheres, spec["controls"])
    expected = expectation(spec["kind"], mode, result)
    return {"scenario": name, "kind": spec["kind"], "terms": spec["terms"], "reference_terms": report["terms"],
            "reference_maximum_radiance": report["maximum_radiance"], "quadrature_mask_pixels": report["quadrature_mask_pixels"],
            "roi_pixels": report["roi_pixels"], "snapshot_sha256": report["snapshot_sha256"], "result": result,
            "expectation": expected, "controls": verdicts,
            "discriminating": all(v["fails_gate"] for v in verdicts.values())}


def verdict(entries, mode):
    """(status, exit code): blocking non-discrimination first, then the expectation."""
    if any(not e["discriminating"] for e in entries):
        return "BLOCKED_NON_DISCRIMINATING_FIXTURE", 3
    if mode == "green":
        return ("GREEN", 0) if all(e["result"]["pass"] for e in entries) else ("RED", 1)
    if all(e["expectation"]["met"] for e in entries):
        return "EXPECTED_RED_CONFIRMED", 0
    return "UNEXPECTED_RESULT", 1


def markdown(manifest, entries, status):
    lines = [f"# F15.3 material validation: {status}", "",
             f"commit `{manifest.get('commit')}`, binary `{manifest.get('binary_sha256')}`, "
             f"{manifest.get('os', '').splitlines()[1] if manifest.get('os') else ''}", "",
             "| scenario | gate | expectation | negative controls |", "|---|---|---|---|"]
    for e in entries:
        controls = ", ".join(f"{k}: {'fails' if v['fails_gate'] else 'PASSES (non-discriminating)'}"
                             for k, v in e["controls"].items())
        ex = e["expectation"]
        lines.append(f"| {e['scenario']} | {'pass' if e['result']['pass'] else 'FAIL'} | "
                     f"{'met' if ex['met'] else 'NOT met'}{'' if ex.get('asserted') else ' (not asserted)'} | {controls} |")
    for e in entries:
        r = e["result"]
        lines += ["", f"## {e['scenario']} ({e['kind']})", f"Image contains: {e['terms']}.",
                  f"Reference terms: {', '.join(e['reference_terms'])}; ROI pixels {e['roi_pixels']}; "
                  f"quadrature mask pixels {e['quadrature_mask_pixels']}."]
        if r["kind"] == "direct":
            lines += ["", "| metallic | pixels | mean | p95 | p99 | max outside mask | pass |", "|---|---|---|---|---|---|---|"]
            for k, row in r["rows"].items():
                lines.append(f"| {k} | {row['pixels']} | {row.get('mean', 0):.4f} | {row.get('p95', 0):.4f} | "
                             f"{row.get('p99', 0):.4f} | {row.get('max_outside_mask', 0):.4f} | {row['pass']} |")
        elif r["kind"] == "environment":
            lines += ["", "| metallic | roughness | mean | p95 | pass |", "|---|---|---|---|---|"]
            for t in r["spheres"]:
                lines.append(f"| {t['metallic']} | {t['roughness']} | {t.get('mean', 0):.4f} | {t.get('p95', 0):.4f} | {t['pass']} |")
        else:
            lines += ["", f"max pixel {r['max_pixel']:.5f} (limit {r['limit']}), reference max {r['reference_max_pixel']:.5f}", "",
                      "| metallic | roughness | max | mean | energy loss | reference loss |", "|---|---|---|---|---|---|"]
            for t in r["spheres"]:
                lines.append(f"| {t.get('metallic')} | {t.get('roughness')} | {t.get('max', 0):.4f} | {t.get('mean', 0):.4f} | "
                             f"{t.get('energy_loss', 0):.4f} | {t.get('reference_energy_loss', 0):.4f} |")
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = parser.parse_args(argv)
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
        return 0

    if args.run:
        if not binary.is_file():
            parser.error("--run requires an existing compiled renderer; this runner never builds")
        from run_checked import run_checked
        with gpu_lock(args.gpu_lock):
            busy = running_gpu_processes()
            if busy:
                print(f"refusing to start: other GPU jobs are running: {', '.join(busy)}", file=sys.stderr)
                return 4
            for n in names:
                folder = out / n
                if folder.exists() and any(folder.iterdir()):
                    parser.error(f"{folder} is not empty: choose a new --out")
                folder.mkdir(parents=True, exist_ok=True)
                status = run_checked(capture_command(binary, n, folder, args.seed), folder / "engine.log",
                                     timeout=args.timeout)
                if not status["passed"]:
                    print(f"{n}: capture failed: {status['failures']}", file=sys.stderr)
                    return 2
                time.sleep(2)  # serial and quiet between GPU jobs

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


if __name__ == "__main__":
    raise SystemExit(main())
