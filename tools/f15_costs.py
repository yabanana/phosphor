#!/usr/bin/env python3
"""F15 (ticket 24) quiet-machine cost measurements: transmission RT, transmissive shadows (RT and CSM map), subsurface diffusion, iridescence,
GI-back lookup and binning vs union on the chess fixture.  Each item is a COST TO RECORD, not a gate.

Method (CLAUDE.md measurement rules): A/B/A per replica (order A, B, A; effect = B minus the mean of the two A runs), 3 replicas, 256 measured
frames after 64 warm-up frames, `--fixed-timestep --offscreen --no-ui --no-vsync --gpu-timing-serial`, per-pass `passes[].gpu_ms` p50 and mean,
frame `gpu_ms` p50/p95, `gpu_allocations` (must be 0).  Items without a clean A/B switch are "abs" items: the absolute pass times of the
fixture (median of the replicas), nothing else is claimed for them.  For every run the 1-minute load average at its start and the contention
sentinel (p95 of two constant-work passes of the post chain, `Display tone map` and `Luminance histogram`) are logged: a run set is only
usable when the sentinel does not move between the A runs.

  python3 -I tools/f15_costs.py plan [--only ITEM,...]                 print every command
  python3 -I tools/f15_costs.py run --out build/f15-costs/run [--only ITEM,...] [--replicas 3]
  python3 -I tools/f15_costs.py summarize --out DIR                    rebuild results.json + summary.md from the reports

`run` takes the shared GPU lock (/tmp/phosphor-gpu-verification.lock, the same file as tools/f15_check.py and the orchestrator's gpulock.py: do not
wrap it in gpulock.py) and refuses to start, or stops, while another phosphor / soc_bench / f<N>_spike process runs.  The binary is only read.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

FRAMES, WARMUP, REPLICAS = 256, 64, 3
DEFAULT_GLTF = pathlib.Path("/Users/danielsan/Documents/phosphor/.claude/worktrees/f15-plan-c38b51/build/khronos-corpus/ABeautifulGame/glTF/ABeautifulGame.gltf")
OVERRIDES = REPO / "fixtures/chess/chess-subsurface.overrides.json"
SENTINELS = ("Display tone map", "Luminance histogram")
NOISE_BAND_MS = 0.02  # run-to-run band of a pass p50 on this machine (two A runs of one replica differ by less in a quiet run)

# Chess rig flags (mirrors tools/f15_chess_check.py, which tests/test_f15_chess_check.py pins).
CHESS_FLAGS = {
    "backlight": {"rt": True, "shadows": "off", "reflections": "rt", "denoise": "off"},
    "sidelight": {"rt": False, "shadows": "off", "reflections": "off", "denoise": "custom"},
    "shadow-rt": {"rt": True, "shadows": "rt", "reflections": "off", "denoise": "custom"},
    "shadow-csm": {"rt": False, "shadows": "csm", "reflections": "off", "denoise": "custom", "shadow_map_size": 8192},
}


def chess(rig, scenario, resolution, extra=()):
    return {"family": "chess", "flags": rig, "scenario": scenario, "resolution": resolution, "extra": list(extra)}


def proc(key, resolution, extra=(), scenario=None):
    """A procedural fixture served by a tools/f15_check.py SCENARIOS key (its lighting / rt / shadows / reflections / denoise)."""
    return {"family": "proc", "key": key, "scenario": scenario, "resolution": resolution, "extra": list(extra)}


def ab(title, a, b, note, metric=None):
    return {"title": title, "kind": "ab", "arms": {"A": a, "B": b}, "note": note, "metric": metric}


def absolute(title, arms, note):
    return {"title": title, "kind": "abs", "arms": arms, "note": note}


R720, R1080 = "1280x720", "1920x1080"
GI_BACK = {"family": "gi-back"}

ITEMS = {
    "t1-transmission-chess-backlight": ab(
        "1. Transmission RT: back-light rig, pawn tops transmission 0 (A) vs 1 (B)",
        chess("backlight", "chess-backlight/transmission=0", R720), chess("backlight", "chess-backlight/transmission=1", R720),
        "same rig, RT reflections and RT glass transport enabled in both arms; A has the transmission factor 0 (no glass transport), B factor 1"),
    "t1-transmission-fixtures": absolute(
        "1b. Transmission RT, procedural glass fixtures (absolute)",
        {"glass-behind-glass": proc("transmission-glass-behind-glass", R720), "roughness-ior": proc("transmission-roughness-ior", R720),
         "rtoff-declared": proc("transmission-rtoff", R720)},
        "no opaque-glass switch exists on the procedural fixtures: absolute pass times only (transmission-rtoff = the declared --rt off model)"),
    "t2-shadow-chess-rt": ab(
        "2a. RT sun shadows: chess shadow rig, pawn tops opaque (A) vs glass (B)",
        chess("shadow-rt", "chess-shadow/tops=opaque", R1080), chess("shadow-rt", "chess-shadow/tops=on", R1080),
        "tops=opaque sets the transmission factor of the pawn tops to 0 (glass treated as opaque)"),
    "t2-shadow-chess-csm": ab(
        "2b. CSM sun shadows (map 8192): chess shadow rig, pawn tops opaque (A) vs glass (B)",
        chess("shadow-csm", "chess-shadow/tops=opaque", R1080), chess("shadow-csm", "chess-shadow/tops=on", R1080),
        "the transmittance map is built only when the scene has a transmissive caster; same flags in both arms"),
    "t2-shadow-pane-rt": absolute(
        "2c. RT sun shadows through glass, procedural pane fixtures (absolute)",
        {"pane-sun": proc("transmissive-shadow-pane-sun@rt", R720), "volume-pane": proc("transmissive-shadow-volume-pane@rt", R720),
         "water-sun": proc("transmissive-shadow-water-sun@rt", R720)},
        "no opaque switch on the procedural fixtures (chess item 2a is the glass vs opaque A/B)"),
    "t2-csm-map-pane": ab(
        "2d. CSM transmittance map, pane-sun@csm: map off (A, --debug-transmittance-map off) vs on (B)",
        proc("transmissive-shadow-pane-sun@csm", R720, ["--debug-transmittance-map", "off"]), proc("transmissive-shadow-pane-sun@csm", R720),
        "off = the receiver ignores the map (glass casts nothing); the map passes still run with a 1x1 stand-in, so the pass delta is a lower bound of the "
        "map passes; heap_bytes shows the memory"),
    "t2-csm-map-volume": ab(
        "2e. CSM transmittance map, volume-pane@csm: map off (A) vs on (B)",
        proc("transmissive-shadow-volume-pane@csm", R720, ["--debug-transmittance-map", "off"]), proc("transmissive-shadow-volume-pane@csm", R720),
        "see 2d"),
    "t3-subsurface-chess-sidelight": ab(
        "3. Subsurface diffusion: chess side-light rig, sss=off (A) vs on (B)",
        chess("sidelight", "chess-sidelight/sss=off", R1080), chess("sidelight", "chess-sidelight/sss=on", R1080),
        "sss=off drops the overrides (no subsurface block on any material); B applies the overrides (kings and queens, volumetric screen-space diffusion)"),
    "t3-subsurface-fixture": absolute(
        "3b. Subsurface diffusion, procedural side-lit spheres (absolute)",
        {"volumetric-sun": proc("subsurface-volumetric-sun", R1080), "subpixel-off": proc("subsurface-volumetric-subpixel-off", R1080),
         "thin-sun": proc("subsurface-thin-sun", R1080)},
        "volumetric-sun has the diffusion pass; subpixel-off is another scene without any block (not a matched control)"),
    "t4-iridescence": absolute(
        "4. Iridescence per light, iridescence grid (absolute, no strength switch)",
        {"grid-sun": proc("iridescence-grid-sun", R1080), "grid-point-legacy": proc("iridescence-grid-point", R1080),
         "grid-point-restir": proc("iridescence-grid-point@restir", R1080), "grid-point-clustered": proc("iridescence-grid-point@clustered", R1080)},
        "absolute pass times of the grid; the film-off and light-count arms are t4-iridescence-film-*"),
    # F15 closure (ticket 12): film-free twins (src/testbench/material_validation.cpp, `iridescence-cost-*`): every sphere keeps the iridescence
    # block (same feature union, same specialised pipelines) but thickness 0 disables the film, so A (no film) vs B (film) isolates the film
    # evaluation; the 8-light twins give its cost per light as the difference of the two deltas (8 lights minus 1 light) / 7.
    "t4-iridescence-film-1light-legacy": ab(
        "4a. Iridescence film cost, 1 point light, resolve lights (legacy): film off (A) vs film (B)",
        proc("iridescence-grid-point", R1080, scenario="iridescence-cost-nofilm"), proc("iridescence-grid-point", R1080),
        "A = iridescence-cost-nofilm (thickness 0 everywhere), B = iridescence-grid-point; same rig, same pipelines, 1 light"),
    "t4-iridescence-film-8lights-legacy": ab(
        "4b. Iridescence film cost, 8 point lights, resolve lights (legacy): film off (A) vs film (B)",
        proc("iridescence-grid-point", R1080, scenario="iridescence-cost-nofilm-lights8"), proc("iridescence-grid-point", R1080, scenario="iridescence-cost-lights8"),
        "per-light film cost = (delta of 4b - delta of 4a) / 7 on the Material resolve pass"),
    "t4-iridescence-film-1light-clustered": ab(
        "4c. Iridescence film cost, 1 point light, clustered DI: film off (A) vs film (B)",
        proc("iridescence-grid-point@clustered", R1080, scenario="iridescence-cost-nofilm"), proc("iridescence-grid-point@clustered", R1080),
        "A = iridescence-cost-nofilm, B = iridescence-grid-point, clustered DI with RT visibility"),
    "t4-iridescence-film-8lights-clustered": ab(
        "4d. Iridescence film cost, 8 point lights, clustered DI: film off (A) vs film (B)",
        proc("iridescence-grid-point@clustered", R1080, scenario="iridescence-cost-nofilm-lights8"),
        proc("iridescence-grid-point@clustered", R1080, scenario="iridescence-cost-lights8"),
        "per-light film cost = (delta of 4d - delta of 4c) / 7 on the DI shading pass"),
    "t5-gi-back": ab(
        "5. GI-back lookup: --thin-gi-back off (A) vs on (B), subsurface-thin-gi-back + DDGI",
        dict(GI_BACK, back="off", resolution=R720), dict(GI_BACK, back="on", resolution=R720), "the second DDGI probe lookup at -N of thin subsurface pixels"),
    "t6-binning-chess-sidelight": ab(
        "6a. Binning vs union, chess side-light (A = union, --material-binning off; B = on)",
        chess("sidelight", "chess-sidelight", R1080, ["--material-binning", "off"]), chess("sidelight", "chess-sidelight", R1080, ["--material-binning", "on"]),
        "resolve = Material resolve + Material classification (decision 05: binning adopted only if it lowers the resolve by >= 5 % beyond noise)", "resolve"),
    "t6-binning-chess-shadow": ab(
        "6b. Binning vs union, chess shadow rig (RT sun shadows)",
        chess("shadow-rt", "chess-shadow", R1080, ["--material-binning", "off"]), chess("shadow-rt", "chess-shadow", R1080, ["--material-binning", "on"]),
        "see 6a", "resolve"),
    "t6-binning-chess-backlight": ab(
        "6c. Binning vs union, chess back-light rig (RT glass transport, 1280x720)",
        chess("backlight", "chess-backlight", R720, ["--material-binning", "off"]), chess("backlight", "chess-backlight", R720, ["--material-binning", "on"]),
        "see 6a", "resolve"),
}


def command(args, arm, report):
    """The exact phosphor command line of one arm."""
    common = ["--resolution", arm["resolution"], "--frames", str(FRAMES), "--warmup", str(WARMUP), "--fixed-timestep", "--offscreen", "--no-ui",
              "--no-vsync", "--gpu-timing-serial", "--report", str(report)]
    family = arm["family"]
    if family == "chess":
        f = CHESS_FLAGS[arm["flags"]]
        return [str(args.binary), "--scene", str(args.gltf), "--material-overrides", str(OVERRIDES), "--bench", "2", "--material-scenario", arm["scenario"],
                "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on" if f["rt"] else "off", "--shadows", f["shadows"],
                "--lighting", "legacy", "--gi", "off", "--reflections", f["reflections"], "--ao", "off", "--lighting-denoise", f["denoise"],
                "--reflection-samples", "8", "--lighting-seed", "1", "--post", "--upscaler", "native",
                *(["--shadow-map-size", str(f["shadow_map_size"])] if f.get("shadow_map_size") else []), *arm["extra"], *common]
    if family == "gi-back":
        return [str(args.binary), "--scene", "procedural", "--bench", "2", "--material-scenario", "subsurface-thin-gi-back", "--render-path", "visibility",
                "--geometry-path", "mesh", "--rt", "on", "--shadows", "off", "--lighting", "restir", "--gi", "ddgi", "--reflections", "ssr", "--ao", "off",
                "--lighting-denoise", "custom", "--reflection-samples", "8", "--lighting-seed", "1", "--thin-gi-back", arm["back"], "--post",
                "--upscaler", "native", *common]
    import f15_check as fc
    key = arm["key"]
    spec = fc.SCENARIOS[key]
    lighting = spec.get("lighting", "legacy")
    scenario = arm["scenario"] or fc.bench_scenario(key)
    return [str(args.binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario, "--render-path", "visibility", "--geometry-path", "mesh",
            "--rt", "off" if lighting == "legacy" and not spec.get("rt") else "on", "--shadows", spec.get("shadows", "off"), "--lighting", lighting,
            "--gi", "off", "--reflections", spec.get("reflections", "ssr"), "--ao", "off", "--lighting-denoise", spec.get("denoise", "custom"),
            "--reflection-samples", "8", "--lighting-seed", "1", "--post", "--upscaler", "native", *arm["extra"], *common]


def schedule(item, replicas):
    runs = []
    for replica in range(replicas):
        if item["kind"] == "ab":
            for slot, arm in (("A0", "A"), ("B1", "B"), ("A2", "A")):
                runs.append({"replica": replica, "slot": slot, "arm": arm})
        else:
            for name in item["arms"]:
                runs.append({"replica": replica, "slot": name, "arm": name})
    return runs


def run_dir(out, item, run):
    return out / item / f"r{run['replica']}-{run['slot']}"


# --- report reading ----------------------------------------------------------------------------------------------

def unit_stats(unit):
    gpu = unit.get("gpu_ms")
    if not isinstance(gpu, dict):
        return {"p50": float(gpu), "mean": float(gpu), "p95": float(gpu)}
    return {k: float(gpu.get(k, 0.0)) for k in ("p50", "mean", "p95")}


def read_run(folder):
    report = json.loads((folder / "report.json").read_text())
    passes = {}
    for unit in report.get("passes") or []:
        s = unit_stats(unit)
        if unit["name"] in passes:  # same name in several units: sum the costs
            passes[unit["name"]] = {k: passes[unit["name"]][k] + s[k] for k in s}
        else:
            passes[unit["name"]] = s
    frame = report["gpu_ms"]
    log = (folder / "log.txt").read_text() if (folder / "log.txt").exists() else ""
    pipelines = re.search(r"fallbacks (\d+) \(draws (\d+)\)", log)
    compiles = re.search(r"render-thread compiles (\d+)", log)
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
    return {"frame": {"p50": frame["p50"], "p95": frame["p95"], "mean": frame["mean"]}, "passes": passes, "gpu_allocations": report.get("gpu_allocations"),
            "heap_bytes": report.get("graph", {}).get("heap_bytes"), "heap_unaliased_bytes": report.get("graph", {}).get("heap_unaliased_bytes"),
            "fallback_draws": int(pipelines.group(2)) if pipelines else None, "render_thread_compiles": int(compiles.group(1)) if compiles else None,
            "load1": meta.get("load1"), "sentinel": {n: passes.get(n, {}).get("p95") for n in SENTINELS}}


def resolve_sum(passes, key):
    return sum(v[key] for n, v in passes.items() if n.lower().startswith(("material resolve", "material classification")))


def med(values):
    return statistics.median(values) if values else None


def ab_effects(runs):
    """runs: {(replica, slot): run}.  Per pass and for the frame: per-replica B - mean(A0, A2) (p50 and mean), reduced to median and min..max."""
    replicas = sorted({r for r, _ in runs})
    names = sorted({n for run in runs.values() for n in run["passes"]})
    out = {"passes": {}, "frame": {}, "resolve": {}, "replicas": []}

    def effect(getter):
        values, a_values = [], []
        for r in replicas:
            if not all((r, s) in runs for s in ("A0", "B1", "A2")):
                continue
            a = 0.5 * (getter(runs[(r, "A0")]) + getter(runs[(r, "A2")]))
            values.append((getter(runs[(r, "B1")]) - a, a, abs(getter(runs[(r, "A0")]) - getter(runs[(r, "A2")]))))
        if not values:
            return None
        deltas = [v[0] for v in values]
        return {"median": med(deltas), "min": min(deltas), "max": max(deltas), "a_median": med([v[1] for v in values]),
                "b_median": med([v[1] + v[0] for v in values]), "a_spread_max": max(v[2] for v in values), "n": len(deltas)}

    for n in names:
        for key in ("p50", "mean"):
            e = effect(lambda run, n=n, key=key: run["passes"].get(n, {}).get(key, 0.0))
            if e:
                out["passes"].setdefault(n, {})[key] = e
    for key in ("p50", "p95", "mean"):
        e = effect(lambda run, key=key: run["frame"][key])
        if e:
            out["frame"][key] = e
    for key in ("p50", "mean"):
        e = effect(lambda run, key=key: resolve_sum(run["passes"], key))
        if e:
            out["resolve"][key] = e
    # relative resolve effect per replica (decision 05 rule)
    rel = []
    for r in replicas:
        if all((r, s) in runs for s in ("A0", "B1", "A2")):
            a0, b, a2 = (resolve_sum(runs[(r, s)]["passes"], "mean") for s in ("A0", "B1", "A2"))
            a = 0.5 * (a0 + a2)
            if a > 0:
                rel.append({"a_ms": a, "b_ms": b, "relative": b / a - 1.0, "a_spread": abs(a0 - a2) / a})
    out["resolve_relative"] = {"replicas": rel, "median": med([x["relative"] for x in rel]), "min": min((x["relative"] for x in rel), default=None),
                               "max": max((x["relative"] for x in rel), default=None), "noise": max((x["a_spread"] for x in rel), default=None)}
    return out


def abs_summary(runs):
    arms = sorted({slot for _, slot in runs})
    out = {}
    for arm in arms:
        rs = [run for (r, slot), run in sorted(runs.items()) if slot == arm]
        names = sorted({n for run in rs for n in run["passes"]})
        out[arm] = {"frame": {k: {"median": med([run["frame"][k] for run in rs]), "min": min(run["frame"][k] for run in rs), "max": max(run["frame"][k] for run in rs)}
                              for k in ("p50", "p95")},
                    "passes": {n: {"p50": med([run["passes"].get(n, {}).get("p50", 0.0) for run in rs]),
                                   "min": min(run["passes"].get(n, {}).get("p50", 0.0) for run in rs),
                                   "max": max(run["passes"].get(n, {}).get("p50", 0.0) for run in rs)} for n in names},
                    "replicas": len(rs)}
    return out


def summarize(out, replicas=None):
    result = {"items": {}}
    for item_dir in sorted(p for p in out.iterdir() if p.is_dir() and p.name in ITEMS):
        item = ITEMS[item_dir.name]
        runs = {}
        for folder in sorted(item_dir.glob("r*-*")):
            if not (folder / "report.json").exists():
                continue
            replica, slot = re.match(r"r(\d+)-(.*)", folder.name).groups()
            runs[(int(replica), slot)] = read_run(folder)
        entry = {"title": item["title"], "kind": item["kind"], "note": item["note"],
                 "runs": {f"r{r}-{s}": {k: v for k, v in run.items() if k != "passes"} | {"frame": run["frame"]} for (r, s), run in runs.items()}}
        entry.update(ab_effects(runs) if item["kind"] == "ab" else {"absolute": abs_summary(runs)})
        extra = item_dir / "extra.json"
        if extra.exists():
            entry["extra"] = json.loads(extra.read_text())
        result["items"][item_dir.name] = entry
    return result


def fmt(e, digits=3):
    return f"{e['median']:+.{digits}f} ({e['min']:+.{digits}f}..{e['max']:+.{digits}f})"


def markdown(result, meta):
    lines = ["# F15 ticket 24: quiet-machine cost measurements", "", f"binary: {meta.get('binary')}; started: {meta.get('started')}; load average (1/5/15 min) at the start: {meta.get('load')}",
             f"frames {FRAMES}, warm-up {WARMUP}, replicas {meta.get('replicas', REPLICAS)}; effect = B - mean(A0, A2) per replica, shown as median (min..max) in ms.", ""]
    for name, e in result["items"].items():
        lines += [f"## {e['title']}", "", e["note"], ""]
        runs = e["runs"]
        loads = [r["load1"] for r in runs.values() if r.get("load1") is not None]
        sentinels = {n: [r["sentinel"][n] for r in runs.values() if r["sentinel"].get(n) is not None] for n in SENTINELS}
        alloc = sorted({r["gpu_allocations"] for r in runs.values()})
        fallbacks = sorted({r["fallback_draws"] for r in runs.values() if r["fallback_draws"] is not None})
        lines.append(f"load1 at run start {min(loads):.2f}..{max(loads):.2f}; sentinel p95 " +
                     "; ".join(f"{n} {min(v):.4f}..{max(v):.4f} ms" for n, v in sentinels.items() if v) +
                     f"; GPU allocations {alloc}; fallback draws {fallbacks}" if loads else "no load data")
        lines.append("")
        if e["kind"] == "ab":
            lines += ["| metric | A ms | B ms | B - A (median, min..max) | A-A spread max |", "|---|---|---|---|---|"]
            for key in ("p50", "p95", "mean"):
                f = e["frame"].get(key)
                if f:
                    lines.append(f"| frame GPU {key} | {f['a_median']:.3f} | {f['b_median']:.3f} | {fmt(f)} | {f['a_spread_max']:.3f} |")
            if e["resolve"]:
                f = e["resolve"]["mean"]
                lines.append(f"| resolve (mean) | {f['a_median']:.3f} | {f['b_median']:.3f} | {fmt(f)} | {f['a_spread_max']:.3f} |")
            lines.append("")
            rows = [(n, v["p50"], v.get("mean")) for n, v in e["passes"].items() if "p50" in v and (abs(v["p50"]["median"]) >= 0.003 or abs(v["p50"]["b_median"]) + abs(v["p50"]["a_median"]) >= 0.2)]
            rows.sort(key=lambda r: -abs(r[1]["median"]))
            lines += ["| pass | A p50 | B p50 | B - A p50 (median, min..max) | B - A mean (median) |", "|---|---|---|---|---|"]
            for n, p50, mean in rows[:14]:
                lines.append(f"| {n} | {p50['a_median']:.3f} | {p50['b_median']:.3f} | {fmt(p50)} | {mean['median']:+.3f} |" if mean else f"| {n} | {p50['a_median']:.3f} | {p50['b_median']:.3f} | {fmt(p50)} | |")
            rel = e.get("resolve_relative", {})
            if rel.get("median") is not None and "binning" in name:
                verdict = "adopted (>= 5 % lower beyond noise)" if rel["median"] <= -0.05 and rel["max"] <= -0.05 + 0 else "NOT adopted (does not lower the resolve by 5 % beyond noise)"
                lines += ["", f"Resolve (Material resolve + Material classification, mean): relative effect of binning {rel['median'] * 100:+.1f} % "
                          f"({rel['min'] * 100:+.1f} .. {rel['max'] * 100:+.1f} %), A-A noise up to {rel['noise'] * 100:.1f} %: {verdict}."]
            heaps = {s: r.get("heap_bytes") for s, r in runs.items() if r.get("heap_bytes")}
            if heaps and name.startswith("t2-csm"):
                a = med([v for k, v in heaps.items() if "-A" in k]); b = med([v for k, v in heaps.items() if "-B" in k])
                lines += ["", f"graph heap_bytes (aliased) A {a / 2**20:.2f} MiB, B {b / 2**20:.2f} MiB (B - A {((b - a) / 2**20):+.2f} MiB); heap_unaliased_bytes "
                          f"A {med([r['heap_unaliased_bytes'] for k, r in runs.items() if '-A' in k]) / 2**20:.2f} MiB, B {med([r['heap_unaliased_bytes'] for k, r in runs.items() if '-B' in k]) / 2**20:.2f} MiB"]
        else:
            for arm, a in e["absolute"].items():
                lines += [f"**{arm}** ({a['replicas']} replicas): frame GPU p50 {a['frame']['p50']['median']:.3f} ({a['frame']['p50']['min']:.3f}..{a['frame']['p50']['max']:.3f}), "
                          f"p95 {a['frame']['p95']['median']:.3f} ms", "", "| pass | p50 median ms | min..max |", "|---|---|---|"]
                top = sorted(a["passes"].items(), key=lambda kv: -kv[1]["p50"])[:10]
                lines += [f"| {n} | {v['p50']:.3f} | {v['min']:.3f}..{v['max']:.3f} |" for n, v in top] + [""]
        if "extra" in e:
            lines += ["", "extra: " + json.dumps(e["extra"])]
        lines.append("")
    return "\n".join(lines) + "\n"


# --- subsurface pixel count (side-lit mask capture, standard library only) -----------------------------------------

def pfm_luminance_count(path, threshold=0.2):
    import array
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        data = array.array("f")
        data.frombytes(f.read(width * height * 12))
        if (scale > 0) == (sys.byteorder == "little"):
            data.byteswap()
    count = 0
    for i in range(0, len(data), 3):
        if 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2] > threshold:
            count += 1
    return {"pixels_above_threshold": count, "width": width, "height": height, "threshold": threshold}


def main(argv=None):
    global FRAMES, WARMUP
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["plan", "run", "summarize"])
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("/Users/danielsan/Documents/phosphor/.claude/worktrees/f15-plan-c38b51/build/release/phosphor"))
    parser.add_argument("--gltf", type=pathlib.Path, default=DEFAULT_GLTF)
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-costs/run")
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--only", default="", help="comma-separated item names (default: all)")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--rest", type=float, default=3.0, help="seconds of rest between runs")
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    parser.add_argument("--frames", type=int, default=FRAMES, help="measured frames (smoke tests only; the record uses 256)")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args(argv)
    FRAMES, WARMUP = args.frames, args.warmup
    names = [n for n in args.only.split(",") if n] or list(ITEMS)
    unknown = [n for n in names if n not in ITEMS]
    if unknown:
        parser.error(f"unknown items {unknown}; known: {list(ITEMS)}")
    out = args.out.resolve()
    if args.mode == "plan":
        for name in names:
            for run in schedule(ITEMS[name], 1):
                arm = ITEMS[name]["arms"][run["arm"]]
                print(f"# {name} {run['slot']}")
                print(" ".join(command(args, arm, out / name / "report.json")))
        return 0
    if args.mode == "summarize":
        meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
        result = summarize(out)
        (out / "results.json").write_text(json.dumps({"meta": meta, **result}, indent=2) + "\n")
        (out / "summary.md").write_text(markdown(result, meta))
        print(markdown(result, meta))
        return 0
    import f15_check
    if not args.binary.is_file() or not args.gltf.is_file():
        parser.error("binary or glTF missing")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"binary": str(args.binary), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "load": [round(x, 2) for x in os.getloadavg()], "replicas": args.replicas,
            "frames": FRAMES, "warmup": WARMUP}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    busy = f15_check.running_gpu_processes()
    if busy:
        print(f"GPU busy: {busy}", file=sys.stderr)
        return 4
    try:
        lock = f15_check.gpu_lock(args.gpu_lock)
        lock.__enter__()
    except f15_check.GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return 4
    with contextlib.ExitStack() as stack:
        stack.push(lock)
        for name in names:
            item = ITEMS[name]
            for run in schedule(item, args.replicas):
                busy = f15_check.running_gpu_processes()
                if busy:
                    print(f"GPU busy: {busy}", file=sys.stderr)
                    return 4
                folder = run_dir(out, name, run)
                if (folder / "report.json").exists():
                    continue  # resume
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "meta.json").write_text(json.dumps({"load1": os.getloadavg()[0], "time": time.strftime("%H:%M:%S")}) + "\n")
                cmd = command(args, item["arms"][run["arm"]], folder / "report.json")
                with (folder / "log.txt").open("w") as log:
                    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                print(f"{name} r{run['replica']} {run['slot']}: exit {code}", flush=True)
                if code:
                    print(f"failed: {' '.join(cmd)}", file=sys.stderr)
                    return 6
                time.sleep(args.rest)
            if name == "t3-subsurface-chess-sidelight" and not (out / name / "extra.json").exists():
                mask = out / name / "mask"
                mask.mkdir(exist_ok=True)
                cmd = command(args, chess("sidelight", "chess-sidelight/mask=1", R1080), mask / "report.json")
                cmd = [c for c in cmd if c not in ("--gpu-timing-serial",)]
                cmd[cmd.index("--frames") + 1] = "2"
                cmd[cmd.index("--warmup") + 1] = "0"
                cmd += ["--capture-linear", str(mask / "mask.pfm"), "--capture-linear-frame", "1", "--capture-linear-signal", "hdr", "--pipeline-sync"]
                code = subprocess.run(cmd, stdout=(mask / "log.txt").open("w"), stderr=subprocess.STDOUT, timeout=args.timeout).returncode
                if code == 0:
                    (out / name / "extra.json").write_text(json.dumps({"subsurface_mask_pixels": pfm_luminance_count(mask / "mask.pfm")}) + "\n")
                    (mask / "mask.pfm").unlink()
                else:
                    print(f"mask capture failed (exit {code})", file=sys.stderr)
    result = summarize(out)
    (out / "results.json").write_text(json.dumps({"meta": meta, **result}, indent=2) + "\n")
    (out / "summary.md").write_text(markdown(result, meta))
    print(markdown(result, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
