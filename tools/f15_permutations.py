#!/usr/bin/env python3
"""F15.2 (ticket 21) serial runner of the material permutation generator: harvest, equivalence and budgets on the fixtures.

Fixtures are the scenes of the committed manifests (shaders/material_classes/*.json, exported by tools/class_manifest;
an appended manifest -- chess, Khronos corpus -- is picked up the same way) and, for each of them, every key of
tools/f15_check.py SCENARIOS that uses the scene (lighting variants exercise `visibility_lit_resolve`; a scene without a
key runs the base-grid-sun command with the scene substituted).  Every GPU process runs serially under the shared
verification lock (exit 3 when it is held or another phosphor/soc_bench/f*_spike runs: wait and retry, never kill).

  harvest      --out DIR   one `--harvest-pipelines` run per fixture scene (binning on, archive ignored): DIR/<scene>.json,
                           then the targeted archive update is one command:
                             tools/material_class_archive.py update --plan <build>/generated/pipeline/material_class_plan.json \\
                                 --archive shaders/pipelines.mtl4-json --harvest DIR/*.json --output shaders/pipelines.mtl4-json
  equivalence  --out DIR   per case: ref = all-lobes resolve, no bins (--resolve-features all --material-binning off),
                           cls = per-class bins (--material-binning on), ctl = cls with every lobe dropped from the class
                           pipelines (--debug-class-drop-lobes 511, the negative control).  Each arm writes the linear HDR
                           capture and the 8-bit final image.  Measured (2026-11-14, M5 Max): a pipeline specialised by a
                           different MATERIAL_FEATURES value is NOT bit-identical to the all-lobes one in the linear float
                           capture -- the compiler schedules the shortened code differently (docs/adr/0003).
                           Declared envelope (set from the first full corpus, where the worst case was 3.1e-6 relative = 37 ULP):
                           per pixel and channel at most RELATIVE_TOLERANCE relative difference; reported next to it: the max ULP,
                           the bitwise differing pixels and the 8-bit differing pixels (image_diff, tolerance 0; max delta must
                           be <= 1 LSB).  A repeated ref run must be bit-identical (the run is deterministic).
                           ctl must leave the envelope for at least one scene of every feature bit the fixtures use; scenes whose
                           resolve colour cannot show a lobe (it acts in the lighting kernels, or the parameters are neutral) are
                           listed as not discriminated.  An overflow run (--debug-class-budget 2) of every scene with >= 4 classes
                           must stay inside the envelope.  The app-exported manifest of one scene must equal its committed entry.
  summarize    --results F re-evaluate the gates of an earlier equivalence.json (no GPU).
  budget       --out DIR   per case a measured-style run (warmup 64, archive, no --pipeline-sync, binning on): 0 render-thread
                           compiles, 0 fallback draws, 0 archive misses, class bins within the budget.

Results: DIR/<mode>.json.  Exit codes: 0 ok, 1 a gate failed, 2 usage, 3 GPU busy, 5 internal error, 6 a run failed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import pathlib
import re
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_check as fc  # noqa: E402
import f15_export_manifests as ext  # noqa: E402

MANIFEST_DIR = REPO / "shaders/material_classes"
LOCK = pathlib.Path("/tmp/phosphor-gpu-verification.lock")
EQUIV_FRAMES, EQUIV_RESOLUTION = 8, "640x360"
RELATIVE_TOLERANCE = 1e-5  # declared codegen envelope of a specialised pipeline (first corpus: at most 3.1e-6 = 37 ULP, iridescence code)
IMAGE_DIFF = REPO / "build/release/image_diff"
BUDGET_WARMUP, BUDGET_FRAMES = 64, 32
DROP_ALL = "511"


class Busy(RuntimeError):
    pass


class RunFailed(RuntimeError):
    pass


def manifest_scenes():
    """[{name, family, scenario, union, classes}] of every committed manifest, sorted by name."""
    out = []
    for path in sorted(MANIFEST_DIR.glob("*.json")):
        for scene in json.loads(path.read_text())["scenes"]:
            family, _, scenario = scene["name"].partition(":")
            out.append({"name": scene["name"], "family": family, "scenario": scenario, "union": scene["union"],
                        "classes": [c["mask"] for c in scene["classes"]], "volumetric": scene["volumetric"]})
    return sorted(out, key=lambda s: s["name"])


CORPUS = pathlib.Path(__import__("os").environ.get("KHRONOS_CORPUS_DIR", "")) if __import__("os").environ.get("KHRONOS_CORPUS_DIR") else None


def cases(scenes):
    """One case per f15_check key that uses a scene, plus a base-grid-sun substitution for a scene without key."""
    by_bench = {}
    for key in fc.SCENARIOS:
        if fc.SCENARIOS[key].get("capture_only") or fc.SCENARIOS[key].get("cc0"):  # cc0 keys share the procedural keys' scenes
            continue
        by_bench.setdefault(fc.bench_scenario(key), []).append(key)
    out = []
    for s in scenes:
        if s["family"] in ("chess", "corpus"):  # the chess rigs and the Khronos models: their own command (tools/f15_export_manifests.py)
            out.append({"id": s["name"], "template": None, "scene": s, "external": True})
            continue
        keys = by_bench.get(s["scenario"], [])
        if keys:
            out += [{"id": k, "template": k, "scene": s} for k in keys]
        else:
            out.append({"id": s["scenario"] + "#base", "template": "base-grid-sun", "scene": s})
    return out


def set_flag(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = value
    else:
        cmd += [flag, value]
    return cmd


def drop_flag(cmd, flag):
    cmd = list(cmd)
    if flag in cmd:
        i = cmd.index(flag)
        del cmd[i:i + 2]
    return cmd


def base_command(binary, case, folder, frames, resolution, capture=True):
    if case.get("external"):
        if CORPUS is None:
            raise SystemExit("chess and corpus scenes need KHRONOS_CORPUS_DIR (the owner-approved read-only corpus)")
        cmd = ext.scene_command(binary, case["scene"]["family"], case["scene"]["scenario"], CORPUS, frames=frames, warmup=0)
        if capture:
            cmd += ["--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", str(frames - 1), "--capture-linear-signal", "hdr",
                    "--capture", str(folder / "final.png")]
        return cmd + ["--report", str(folder / "report.json")]
    cmd = fc.capture_command(binary, case["template"], folder)
    cmd = set_flag(cmd, "--material-scenario", case["scene"]["scenario"])
    cmd = set_flag(cmd, "--resolution", resolution)
    cmd = set_flag(cmd, "--frames", str(frames))
    for flag in ("--capture-linear-accumulate", "--export-reference", "--export-reference-frame"):
        cmd = drop_flag(cmd, flag)
    if capture:
        cmd = set_flag(cmd, "--capture-linear-frame", str(frames - 1))
        cmd = set_flag(cmd, "--capture", str(folder / "final.png"))
    else:
        for flag in ("--capture-linear", "--capture-linear-frame", "--capture-linear-signal"):
            cmd = drop_flag(cmd, flag)
    return cmd


def run(cmd, log, timeout):
    busy = fc.running_gpu_processes()
    if busy:
        raise Busy("other GPU jobs are running: " + ", ".join(busy))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        try:
            status = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            status = -9
    text = log.read_text(errors="replace")
    if status != 0 or "EXIT 0" not in text:
        raise RunFailed(f"{' '.join(cmd[:1])} failed (status {status}); log {log}")
    return text


def read_pfm(path):
    data = pathlib.Path(path).read_bytes()
    header = data.split(b"\n", 3)
    if header[0] not in (b"PF", b"Pf"):
        raise ValueError(f"{path}: not a PFM")
    width, height = map(int, header[1].split())
    channels = 3 if header[0] == b"PF" else 1
    return width, height, channels, data[sum(len(h) + 1 for h in header[:3]):]


def ordered(value):
    """Float bit pattern as a monotonic integer (units in the last place are differences of these)."""
    return value if value >= 0 else -(value & 0x7FFFFFFF)


def compare_pfm(a, b):
    """Bitwise differing pixels, total pixels, max ULP and max relative difference of two PFM captures (a size mismatch: everything)."""
    import array
    import struct
    wa, ha, ca, da = read_pfm(a)
    wb, hb, cb, db = read_pfm(b)
    if (wa, ha, ca) != (wb, hb, cb):
        return {"pixels": wa * ha, "bitwise": wa * ha, "max_ulp": 1 << 30, "max_rel": float("inf")}
    if da == db:
        return {"pixels": wa * ha, "bitwise": 0, "max_ulp": 0, "max_rel": 0.0}
    fa, fb = array.array("I"), array.array("I")
    fa.frombytes(da)
    fb.frombytes(db)
    bitwise, max_ulp, max_rel = 0, 0, 0.0
    for i in range(0, len(fa), ca):
        if fa[i:i + ca] == fb[i:i + ca]:
            continue
        bitwise += 1
        for k in range(ca):
            if fa[i + k] == fb[i + k]:
                continue
            ua, ub = struct.unpack("<i", struct.pack("<I", fa[i + k]))[0], struct.unpack("<i", struct.pack("<I", fb[i + k]))[0]
            max_ulp = max(max_ulp, abs(ordered(ua) - ordered(ub)))
            x, y = struct.unpack("<f", struct.pack("<I", fa[i + k]))[0], struct.unpack("<f", struct.pack("<I", fb[i + k]))[0]
            max_rel = max(max_rel, abs(x - y) / max(abs(x), abs(y), 1e-30))
    return {"pixels": wa * ha, "bitwise": bitwise, "max_ulp": max_ulp, "max_rel": max_rel}


def compare_png(a, b):
    """8-bit differing pixels and max delta of two final images (image_diff, tolerance 0)."""
    result = subprocess.run([str(IMAGE_DIFF), str(a), str(b)], capture_output=True, text=True)
    m = re.search(r"(\d+) of (\d+) pixels differ \(tolerance 0\), max delta (\d+)", result.stdout)
    if not m:
        raise RuntimeError(f"image_diff: {result.stdout.strip()} {result.stderr.strip()}")
    return {"differ": int(m.group(1)), "max_delta": int(m.group(3))}


def within_envelope(c):
    return c["max_rel"] <= RELATIVE_TOLERANCE


def report(folder):
    return json.loads((folder / "report.json").read_text())


def pipelines_of(text):
    """The last PIPELINES line of a log as integers: requests, archive misses, render-thread compiles, fallback draws."""
    line = [l for l in text.splitlines() if l.startswith("PIPELINES ")][-1]
    pick = lambda pattern: int(re.search(pattern, line).group(1))  # noqa: E731
    return {"requests": pick(r"requests (\d+)"), "archive_misses": pick(r"misses (\d+)"),
            "render_thread_compiles": pick(r"render-thread compiles (\d+)"), "fallback_draws": pick(r"draws (\d+)")}


def selected(args, all_cases):
    cs = [c for c in all_cases if not args.only or c["scene"]["name"] in args.only or c["id"] in args.only]
    if not cs:
        raise SystemExit("no case selected")
    return cs


def do_harvest(args, binary, out):
    scenes = manifest_scenes()
    out.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        if args.only and s["name"] not in args.only:
            continue
        case = {"id": s["scenario"], "template": next((k for k in fc.SCENARIOS if fc.bench_scenario(k) == s["scenario"]), "base-grid-sun"), "scene": s}
        if s["family"] in ("chess", "corpus"):
            case = {"id": s["name"], "template": None, "scene": s, "external": True}
        folder = out / s["name"].replace(":", "_")
        cmd = base_command(binary, case, folder, 3, EQUIV_RESOLUTION, capture=False)
        cmd += ["--material-binning", "on", "--harvest-pipelines", str(out / (s["name"].replace(":", "_") + ".json"))]  # synchronous compiles: every request is recorded
        run(cmd, folder / "engine.log", args.timeout)
        print(f"harvested {s['name']}", flush=True)
    return 0


def do_equivalence(args, binary, out):
    all_cases = selected(args, cases(manifest_scenes()))
    results, failures = [], []
    repeated = {c["id"] for c in all_cases[:: max(1, len(all_cases) // 6)]}  # a spread of cases also runs ref twice
    for case in all_cases:
        arms = {"ref": ["--material-binning", "off", "--resolve-features", "all"], "cls": ["--material-binning", "on"]}
        has_features = case["scene"]["union"] != 0
        if has_features:
            arms["ctl"] = ["--material-binning", "on", "--debug-class-drop-lobes", DROP_ALL]
        if len(case["scene"]["classes"]) >= 4 and has_features:
            # Overflow bin: only 2 classes keep a bin, every other class runs the all-lobes overflow pipeline (the same
            # envelope applies: those pixels are the all-lobes pipeline, the others their class pipeline).
            arms["ovf"] = ["--material-binning", "on", "--debug-class-budget", "2"]
        if case["id"] in repeated:
            arms["ref2"] = arms["ref"]
        folders, logs = {}, {}
        for arm, extra in arms.items():
            folders[arm] = out / case["id"].replace("@", "_at_") / arm
            cmd = base_command(binary, case, folders[arm], EQUIV_FRAMES, EQUIV_RESOLUTION) + extra
            logs[arm] = run(cmd, folders[arm] / "engine.log", args.timeout)
            time.sleep(0.5)
        entry = {"case": case["id"], "scene": case["scene"]["name"], "union": case["scene"]["union"], "classes": case["scene"]["classes"]}
        for arm in arms:
            if arm != "ref":
                entry[arm] = compare_pfm(folders["ref"] / "engine.pfm", folders[arm] / "engine.pfm")
                entry[arm]["png"] = compare_png(folders["ref"] / "final.png", folders[arm] / "final.png")
        entry["pixels"] = entry["cls"]["pixels"]
        rep = report(folders["cls"])["rendering"]
        entry["feature_binned_frames"] = rep.get("feature_binned_frames_total")
        entry["class_bins"] = rep.get("class_bins")
        entry["class_overflow"] = rep.get("class_overflow")
        entry["cls_pipelines"] = pipelines_of(logs["cls"])
        entry["fallback_draws"] = {a: pipelines_of(t)["fallback_draws"] for a, t in logs.items()}
        entry["discriminates"] = has_features and not within_envelope(entry["ctl"])
        problems = []
        if not within_envelope(entry["cls"]):
            problems.append(f"class pipelines leave the declared envelope: relative {entry['cls']['max_rel']:.2e} ({entry['cls']['max_ulp']} ULP)")
        if entry["cls"]["png"]["max_delta"] > 1:
            problems.append(f"8-bit difference above 1 LSB ({entry['cls']['png']})")
        if "ref2" in entry and entry["ref2"]["bitwise"] != 0:
            problems.append(f"the all-lobes reference is not repeatable ({entry['ref2']['bitwise']} pixels)")
        if "ovf" in entry:
            ovf_rep = report(folders["ovf"])["rendering"]
            entry["ovf_class_overflow"] = ovf_rep.get("class_overflow")
            if not within_envelope(entry["ovf"]) or entry["ovf"]["png"]["max_delta"] > 1:
                problems.append(f"overflow-bin run leaves the envelope: {entry['ovf']}")
            if not entry["ovf_class_overflow"] or ovf_rep.get("feature_binned_frames_total") != EQUIV_FRAMES:
                problems.append(f"the overflow run did not use the overflow bin on every frame ({ovf_rep.get('class_overflow')} overflow classes)")
        if has_features and entry["feature_binned_frames"] != EQUIV_FRAMES:
            problems.append(f"only {entry['feature_binned_frames']} of {EQUIV_FRAMES} frames used the class bins")
        if any(entry["fallback_draws"].values()):
            problems.append(f"fallback draws {entry['fallback_draws']}")
        entry["problems"] = problems
        results.append(entry)
        failures += [f"{case['id']}: {m}" for m in problems]
        ctl = entry.get("ctl")
        print(f"{case['id']}: cls {entry['cls']['bitwise']} px bitwise ({entry['cls']['max_ulp']} ULP, 8-bit {entry['cls']['png']['differ']} px)"
              f", ctl {'-' if not ctl else str(ctl['bitwise']) + ' px / ' + str(ctl['max_ulp']) + ' ULP'}"
              f"{', ref2 ' + str(entry['ref2']['bitwise']) + ' px' if 'ref2' in entry else ''}"
              f"{'  FAIL ' + '; '.join(problems) if problems else ''}", flush=True)
    summary, summary_failures = summarize(results)
    failures += summary_failures
    export_failures = check_app_export(args, binary, out)
    summary["app_export_failures"] = export_failures
    (out / "equivalence.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    print(json.dumps(summary))
    return 1 if failures or export_failures else 0


def summarize(results):
    """(summary, extra failures) of per-case results.  Negative-control coverage is judged per LOBE: every feature bit that a
    fixture uses must be discriminated (ctl outside the envelope) by at least one scene.  Scenes whose lobes act in the lighting
    kernels, or whose parameters are neutral, cannot discriminate through the resolve's colour; they are listed, not failed."""
    failures = []
    by_scene = {}
    for r in results:
        by_scene.setdefault(r["scene"], []).append(r)
    undiscriminated = sorted(n for n, rs in by_scene.items() if rs[0]["union"] != 0 and not any(r["discriminates"] for r in rs))
    used_bits = {b for r in results for b in range(9) if r["union"] >> b & 1}
    discriminated_bits = {b for r in results if r["discriminates"] for b in range(9) if r["union"] >> b & 1}
    for b in sorted(used_bits - discriminated_bits):
        failures.append(f"feature bit {b}: used by a fixture but no scene's negative control leaves the envelope")
    scenes_per_bit = {b: sum(1 for rs in by_scene.values() if rs[0]["union"] >> b & 1 and any(r["discriminates"] for r in rs)) for b in sorted(used_bits)}
    summary = {"cases": len(results), "scenes": len(by_scene), "failures": failures,
               "cls_bitwise_zero_cases": sum(1 for r in results if r["cls"]["bitwise"] == 0),
               "cls_max_ulp": max(r["cls"]["max_ulp"] for r in results),
               "cls_max_rel": max(r["cls"]["max_rel"] for r in results),
               "cls_8bit_differing_pixels_total": sum(r["cls"]["png"]["differ"] for r in results),
               "cls_8bit_max_delta": max(r["cls"]["png"]["max_delta"] for r in results),
               "controls": sum(1 for r in results if "ctl" in r), "controls_leave_envelope": sum(1 for r in results if r["discriminates"]),
               "scenes_with_features": sum(1 for rs in by_scene.values() if rs[0]["union"] != 0),
               "scenes_discriminated": sum(1 for rs in by_scene.values() if any(r["discriminates"] for r in rs)),
               "scenes_not_discriminated": undiscriminated, "discriminating_scenes_per_feature_bit": scenes_per_bit,
               "overflow_runs": sum(1 for r in results if "ovf" in r),
               "overflow_max_ulp": max([r["ovf"]["max_ulp"] for r in results if "ovf" in r] or [0]),
               "repeats": sum(1 for r in results if "ref2" in r), "repeats_bit_identical": sum(1 for r in results if "ref2" in r and r["ref2"]["bitwise"] == 0),
               "relative_tolerance": RELATIVE_TOLERANCE,
               "frames": EQUIV_FRAMES, "resolution": EQUIV_RESOLUTION}
    return summary, failures


def check_app_export(args, binary, out):
    """`--export-class-manifest` of the app equals the committed manifest entry of the scene (same classes and union)."""
    scene = next(s for s in manifest_scenes() if s["name"] == "material:combo-all-lobes-sun")
    case = {"id": "combo-all-lobes-sun", "template": "combo-all-lobes-sun", "scene": scene}
    folder = out / "app-export"
    path = folder / "manifest.json"
    cmd = base_command(binary, case, folder, 2, EQUIV_RESOLUTION, capture=False) + ["--export-class-manifest", str(path)]
    run(cmd, folder / "engine.log", args.timeout)
    exported = json.loads(path.read_text())["scenes"][0]
    problems = []
    if [c["mask"] for c in exported["classes"]] != scene["classes"] or exported["union"] != scene["union"]:
        problems.append(f"app export {exported['classes']} != committed {scene['classes']}")
    return problems


def do_budget(args, binary, out):
    all_cases = selected(args, cases(manifest_scenes()))
    results, failures = [], []
    for case in all_cases:
        folder = out / case["id"].replace("@", "_at_")
        cmd = base_command(binary, case, folder, BUDGET_FRAMES, EQUIV_RESOLUTION, capture=False)
        cmd = [c for c in cmd if c != "--pipeline-sync"]
        cmd = set_flag(cmd, "--warmup", str(BUDGET_WARMUP)) + ["--material-binning", "on"]
        text = run(cmd, folder / "engine.log", args.timeout)
        p = pipelines_of(text)
        rep = report(folder)
        measured = rep["pipelines"]["measured"]  # deltas over the measured frames (after the warmup)
        entry = {"case": case["id"], "scene": case["scene"]["name"], "whole_run": p, "measured": measured,
                 "class_bins": rep["rendering"].get("class_bins"), "class_overflow": rep["rendering"].get("class_overflow"),
                 "feature_binned_frames": rep["rendering"].get("feature_binned_frames_total"),
                 "frames": BUDGET_FRAMES, "warmup": BUDGET_WARMUP}
        problems = []
        if measured["renderThreadCompiles"]:
            problems.append(f"{measured['renderThreadCompiles']} render-thread compiles in the measured frames")
        if measured["fallbackDraws"]:
            problems.append(f"{measured['fallbackDraws']} fallback draws in the measured frames")
        if measured["compilerCalls"]:
            problems.append(f"{measured['compilerCalls']} compiler calls in the measured frames")
        if p["archive_misses"]:
            problems.append(f"{p['archive_misses']} archive misses (the class table / archive are out of step)")
        if (entry["class_bins"] or 0) > 16 or entry["class_overflow"]:
            problems.append("more classes than the bin budget")
        entry["problems"] = problems
        results.append(entry)
        failures += [f"{case['id']}: {m}" for m in problems]
        print(f"{case['id']}: measured fallback draws {measured['fallbackDraws']}, rt-compiles {measured['renderThreadCompiles']}, compiler calls "
              f"{measured['compilerCalls']}; whole run: misses {p['archive_misses']}, fallback draws {p['fallback_draws']}; bins {entry['class_bins']}"
              f"{'  FAIL ' + '; '.join(problems) if problems else ''}", flush=True)
    summary = {"cases": len(results), "failures": failures, "max_class_bins": max((r["class_bins"] or 0) for r in results),
               "measured_fallback_draws": sum(r["measured"]["fallbackDraws"] for r in results),
               "measured_render_thread_compiles": sum(r["measured"]["renderThreadCompiles"] for r in results),
               "measured_compiler_calls": sum(r["measured"]["compilerCalls"] for r in results),
               "whole_run_archive_misses": sum(r["whole_run"]["archive_misses"] for r in results),
               "whole_run_fallback_draws_total": sum(r["whole_run"]["fallback_draws"] for r in results),
               "frames": BUDGET_FRAMES, "warmup": BUDGET_WARMUP}
    (out / "budget.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    print(json.dumps(summary))
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["harvest", "equivalence", "budget", "summarize"])
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--only", action="append", default=[], help="a scene (family:scenario) or a case key; repeatable")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--results", type=pathlib.Path, help="summarize: an earlier equivalence.json (no GPU process)")
    args = parser.parse_args(argv)
    if args.mode == "summarize":
        results = json.loads(args.results.read_text())["results"]
        summary, failures = summarize(results)
        failures += [f"{r['case']}: {m}" for r in results for m in r["problems"]]
        print(json.dumps(summary, indent=2))
        return 1 if failures else 0
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"{binary} does not exist (this runner never builds)")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"{out} is not empty: choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    try:
        with LOCK.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("GPU busy: the verification lock is held", file=sys.stderr)
                return 3
            return {"harvest": do_harvest, "equivalence": do_equivalence, "budget": do_budget}[args.mode](args, binary, out)
    except Busy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return 3
    except RunFailed as error:
        print(f"run failed: {error}", file=sys.stderr)
        return 6
    except Exception as error:  # noqa: BLE001
        print(f"internal error: {error!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
