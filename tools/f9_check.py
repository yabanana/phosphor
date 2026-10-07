#!/usr/bin/env python3
"""Sequential F9 engine correctness checks; never run another GPU job concurrently.

Each case preserves its command, raw process status, EXIT marker, renderer
report and validation log. No retries and no automatic acceptance of a failed
case. This runner covers F9 functional checks only: throughput, proxy error
budgets, image review, previous-phase regressions and physical-device
certification remain separate exit criteria in docs/plans/F9-EXECUTION.md.

Use --plan-only to write commands without running the renderer, --list to
print case names, or --only GLOB repeatedly to select a subset. --self-test
checks the evidence evaluator on CPU fixtures and never launches the engine.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fnmatch
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys

from run_checked import run_checked


CHECK_LINE = re.compile(r"^RT checks (\d+) failures (\d+) \| (PASS|FAIL)(?:\b.*)$", re.M)
GPU_ERROR = re.compile(r"failed assertion|GPU timeout|command buffers failed|Shader Validation Error|Metal Validation Error")
NOT_COVERED = [
    "F9 TLAS <=0.5 ms gate and refit degradation study at 100K instances",
    "three-run 1080p ray throughput and A/B/A performance against OPT-4.16",
    "proxy full-versus-proxy error metrics and visual review of debug captures",
    "interpretation of V-buffer edge/tie/alpha mismatches",
    "F5-F8 regression batteries, archive/hot-reload/lifetime/leaks and CI",
    "physical Apple9/M3 or other unavailable hardware certification",
]


@dataclass
class Case:
    name: str
    flags: list[str]
    family: str = "native"
    negative: bool = False
    probe: str | None = None
    visibility: bool = False
    proxy: bool = False
    require_zero_allocations: bool = True
    needs_sponza: bool = False
    capture: bool = False
    transition: str | None = None
    deform: bool = False


def make_cases(families: list[str], proxy_manifest: Path, quick: bool) -> list[Case]:
    cases = []
    for family in families:
        def add(name, flags, **kw):
            cases.append(Case(f"{family}-{name}", flags, family=family, **kw))
        for bench in (1, 5, 7, 8):
            extra = ["--instances", "10000", "--scene-meshes", "4"] if bench == 8 else []
            add(f"bench{bench}", ["--scene", "procedural", "--bench", str(bench), *extra])
        sponza = ["--scene", "assets/sponza/Sponza.gltf", "--bench", "4"]
        add("sponza-full", sponza, needs_sponza=True)
        for bench, scene_flags in ((1, ["--scene", "procedural", "--bench", "1"]), (4, sponza)):
            add(f"visibility-bench{bench}", [*scene_flags, "--render-path", "visibility", "--geometry-path", "mesh"],
                visibility=True, needs_sponza=bench == 4)
        for probe in ("primary", "shadow", "ao", "diffuse"):
            add(f"probe-{probe}", [*sponza, "--rt-probe", probe], probe=probe, needs_sponza=True)
        for path in ("indexed", "mesh"):
            add(f"debug-view-{path}", ["--scene", "procedural", "--bench", "1", "--geometry-path", path,
                                       "--debug-view", "rt"], capture=True)
        add("proxy-sponza", [*sponza, "--rt-proxy", "manifest", "--rt-proxy-manifest", str(proxy_manifest)],
            proxy=True, needs_sponza=True)
        if not quick:
            for name, cadence in (("deform", "1"), ("deform-inflight", "17")):
                add(name, ["--scene", "procedural", "--bench", "1", "--debug-rt", cadence,
                           "--debug-rt-deform", "--debug-view", "rt", "--frames", "120", "--warmup", "0"],
                    deform=True, probe="primary", require_zero_allocations=False)
            for transition in ("mask", "emissive", "reassign", "full-upload"):
                add(f"proxy-transition-{transition}", [*sponza, "--rt-proxy", "manifest", "--rt-proxy-manifest", str(proxy_manifest),
                    "--debug-rt-proxy-transition", transition, "--warmup", "0", "--frames", "24"],
                    transition=transition, needs_sponza=True, require_zero_allocations=False)
            for flight in (1, 2, 3):
                add(f"flight-{flight}", ["--scene", "procedural", "--bench", "1", "--frames-in-flight", str(flight)])
            add("dynamic-churn", ["--scene", "procedural", "--bench", "8", "--instances", "10000",
                                  "--scene-meshes", "4", "--dynamic-cpu", "10", "--churn", "8"])
            add("switch-resize", ["--scene", "procedural", "--bench", "1", "--instances", "10000",
                                  "--frames", "180", "--warmup", "0", "--switch-every", "20", "--resize-every", "45"],
                require_zero_allocations=False)
            add("periodic-rebuild", ["--scene", "procedural", "--bench", "8", "--instances", "10000",
                                     "--rt-tlas-rebuild-every", "5"])
        for corruption in ("transform", "mask", "blas"):
            add(f"negative-{corruption}", ["--scene", "procedural", "--bench", "1", "--frames", "12",
                                           "--debug-rt-corrupt", corruption], negative=True)
        add("negative-blas-multimesh", ["--scene", "procedural", "--bench", "8", "--instances", "10000",
                                        "--scene-meshes", "4", "--frames", "12", "--debug-rt-corrupt", "blas"], negative=True)
    return cases


def read_report(path: Path):
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as error:
        return None, f"unreadable renderer report: {error}"


def numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate(case: Case, status: dict, text: str, report, report_error: str | None = None) -> dict:
    failures = list(status.get("failures", []))
    expected = 1 if case.negative else 0
    if status.get("returncode") != expected:
        failures.append(f"raw process exit must be {expected}")
    if status.get("exit_marker") != expected:
        failures.append(f"final EXIT marker must be {expected}")
    if status.get("timed_out"):
        failures.append("timed-out processes cannot pass")
    if GPU_ERROR.search(text):
        failures.append("GPU/validation failure unrelated to the expected checker result")
    matches = list(CHECK_LINE.finditer(text))
    check = None
    if not matches:
        failures.append("missing final RT checks N failures M | PASS/FAIL line")
    else:
        match = matches[-1]
        check = {"checks": int(match[1]), "failures": int(match[2]), "status": match[3]}
        if check["checks"] <= 0:
            failures.append("checker executed zero checks")
        if case.negative:
            if check["status"] != "FAIL" or check["failures"] <= 0:
                failures.append("negative control did not produce an RT checker failure")
        elif check["status"] != "PASS" or check["failures"] != 0:
            failures.append("positive RT checker failed")
    if report_error:
        failures.append(report_error)
    rt = report.get("rt") if isinstance(report, dict) else None
    if report is None and not case.negative:
        failures.append("positive run is missing its renderer report")
    if report is not None:
        if not isinstance(report, dict):
            failures.append("renderer report must be a JSON object")
            report = {}
        if not numeric(report.get("schema_version")) or report["schema_version"] < 9:
            failures.append("renderer report must have schema_version >= 9")
        if not isinstance(rt, dict) or rt.get("enabled") is not True:
            failures.append("renderer report must contain rt.enabled=true")
            rt = {}
        for field in ("checks", "check_failures", "opaque_alpha_tests"):
            if not numeric(rt.get(field)) or rt[field] < 0:
                failures.append(f"missing/invalid rt.{field}")
        if numeric(rt.get("checks")) and rt["checks"] <= 0:
            failures.append("report records zero RT checks")
        if numeric(rt.get("check_failures")) and (rt["check_failures"] > 0) != case.negative:
            failures.append("report check_failures disagrees with the expected outcome")
        if not case.negative:
            if rt.get("opaque_alpha_tests") != 0:
                failures.append("opaque geometry invoked the alpha intersection function")
            if case.require_zero_allocations and report.get("gpu_allocations") != 0:
                failures.append("nonzero or missing measured-frame GPU allocation count")
            if case.family == "apple9" and rt.get("effective_family") != "apple9":
                failures.append("forced Apple9 run did not report effective_family=apple9")
            if case.probe:
                if rt.get("probe") != case.probe:
                    failures.append("reported RT probe differs from requested probe")
                if not numeric(rt.get("probe_rays")) or rt["probe_rays"] <= 0:
                    failures.append("probe traced no rays")
            if case.visibility:
                if not numeric(rt.get("visibility_compared")) or rt["visibility_compared"] <= 0:
                    failures.append("V-buffer agreement compared no pixels")
                if not numeric(rt.get("visibility_mismatches")) or rt["visibility_mismatches"] < 0:
                    failures.append("V-buffer mismatch count missing/invalid")
            if case.proxy:
                if rt.get("proxy_mode") != "manifest":
                    failures.append("requested proxy manifest was not reported as active")
                if not numeric(rt.get("proxy_meshes")) or rt["proxy_meshes"] <= 0:
                    failures.append("manifest selected no proxy meshes")
    if case.deform:
        for field in ("blas_refits", "blas_builds", "blas_count", "compactions"):
            if not isinstance(rt, dict) or not numeric(rt.get(field)) or rt[field] < 0:
                failures.append(f"missing/invalid deformation counter rt.{field}")
        if isinstance(rt, dict):
            if not numeric(rt.get("blas_refits")) or rt["blas_refits"] <= 0:
                failures.append("deformation executed no BLAS refit")
            if not (numeric(rt.get("blas_builds")) and numeric(rt.get("blas_count")) and rt["blas_builds"] > rt["blas_count"]):
                failures.append("deformation executed no BLAS rebuild after initial loading")
            if not numeric(rt.get("compactions")) or rt["compactions"] <= 0:
                failures.append("deformation executed no BLAS compaction")
    transition_evidence = None
    if case.transition:
        pattern = (r"^RT-PROXY-TRANSITION (mask|emissive|reassign|full-upload) mesh (\d+) source (\d+) "
                   r"before (\d+) after (\d+) material_full ([01]) instances_full ([01]) "
                   r"material_records (\d+) instance_records (\d+) applied ([01]) promoted ([01]) verified ([01]) "
                   r"\| (PASS|FAIL)(?:[: ].*)?$")
        records = list(re.finditer(pattern, text, re.M))
        if not records:
            failures.append("missing proxy-transition evidence; checker PASS alone is insufficient")
        else:
            match = records[-1]
            keys = ("mesh", "source", "before", "after", "material_full", "instances_full", "material_records",
                    "instance_records", "applied", "promoted", "verified")
            transition_evidence = {"mode": match[1], **dict(zip(keys, map(int, match.groups()[1:12]))), "status": match[13]}
            e = transition_evidence
            if e["mode"] != case.transition or e["status"] != "PASS" or not all(e[k] == 1 for k in ("applied", "promoted", "verified")):
                failures.append("proxy transition did not apply, promote and verify the requested change")
            if not (e["mesh"] < 0xffffffff and 0 < e["before"] < e["source"] == e["after"]):
                failures.append("proxy transition did not go from reduced indices to the full source")
            if case.transition == "full-upload":
                if not (e["material_full"] == e["instances_full"] == 1 and e["material_records"] == e["instance_records"] == 0):
                    failures.append("full-upload transition did not exercise full buffers and empty delta arrays")
            elif case.transition == "reassign":
                if e["instances_full"] or e["instance_records"] <= 0:
                    failures.append("reassignment transition did not exercise instance deltas")
            elif e["material_full"] or e["material_records"] <= 0:
                failures.append("material transition did not exercise material deltas")
        if not isinstance(rt, dict) or rt.get("proxy_mode") != "manifest":
            failures.append("proxy transition requires an active measured manifest")
        # These cases use primary diagnostics, warmup=0 and a fixed extent:
        # reload must not erase the completed but uncollected frame slots.
        dimensions = [report.get(k) if isinstance(report, dict) else None for k in ("frames", "width", "height")]
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in dimensions):
            failures.append("proxy transition needs valid frame count and extent for exact ray accounting")
        elif not isinstance(rt, dict) or rt.get("probe_rays") != dimensions[0] * min(512, dimensions[1] * dimensions[2]):
            failures.append("proxy transition lost measured rays across the reload")
        if not isinstance(rt, dict) or not numeric(rt.get("blas_count")) or rt["blas_count"] <= 0:
            failures.append("proxy transition has no valid live BLAS count")
        else:
            for key in ("blas_builds", "compactions"):
                if not numeric(rt.get(key)) or rt[key] < 2 * rt["blas_count"]:
                    failures.append(f"proxy transition rt.{key} does not include both scene loads")
    return {"name": case.name, "passed": not failures, "failures": failures,
            "transition_evidence": transition_evidence,
            "checker": check, "rt_report": rt,
            "report_present": report is not None,
            "visibility_requires_review": case.visibility,
            "proxy_error_requires_separate_validation": case.proxy}


def self_test():
    # Pure CPU fixtures deliberately separate log, raw status and report so a
    # pretty PASS line cannot mask a crash, missing report or zero-work check.
    import copy
    case = Case("fixture", [])
    status = {"returncode": 0, "exit_marker": 0, "timed_out": False, "failures": []}
    report = {"schema_version": 9, "gpu_allocations": 0,
              "rt": {"enabled": True, "checks": 2, "check_failures": 0, "opaque_alpha_tests": 0}}
    text = "RT checks 2 failures 0 | PASS\nEXIT 0\n"
    tests = 0
    def expect(expected, *args):
        nonlocal tests
        result = evaluate(*args)
        if result["passed"] != expected:
            raise RuntimeError(f"evaluator fixture {tests} failed: {result}")
        tests += 1
    expect(True, case, status, text, report)
    for bad in ({"returncode": -6}, {"returncode": 1}, {"exit_marker": None}, {"timed_out": True}):
        expect(False, case, {**status, **bad}, text, report)
    expect(False, case, status, "RT checks 0 failures 0 | PASS\nEXIT 0\n", report)
    expect(False, case, status, text, None)
    expect(False, case, status, text, [])
    expect(False, case, status, text + "Shader Validation Error\n", report)
    bad_report = copy.deepcopy(report); bad_report["rt"]["opaque_alpha_tests"] = 1
    expect(False, case, status, text, bad_report)
    bad_report = copy.deepcopy(report); bad_report["rt"]["checks"] = 0
    expect(False, case, status, text, bad_report)
    bad_report = copy.deepcopy(report); bad_report["rt"]["check_failures"] = "0"
    expect(False, case, status, text, bad_report)
    apple9 = Case("apple9", [], family="apple9")
    expect(False, apple9, status, text, report)
    family_report = copy.deepcopy(report); family_report["rt"]["effective_family"] = "apple9"
    expect(True, apple9, status, text, family_report)
    expect(False, Case("probe", [], probe="shadow"), status, text, report)
    expect(False, Case("visibility", [], visibility=True), status, text, report)
    expect(False, Case("proxy", [], proxy=True), status, text, report)
    negative = Case("negative", [], negative=True)
    neg_status = {**status, "returncode": 1, "exit_marker": 1}
    neg_text = "RT checks 1 failures 1 | FAIL\nEXIT 1\n"
    expect(True, negative, neg_status, neg_text, None)
    expect(False, negative, {**neg_status, "returncode": -11}, neg_text, None)
    expect(False, negative, neg_status, text, None)
    expect(False, negative, neg_status, neg_text + "GPU timeout\n", None)
    bad_report = copy.deepcopy(report); bad_report["rt"]["check_failures"] = 1
    expect(True, negative, neg_status, neg_text, bad_report)
    expect(False, negative, neg_status, neg_text, report)
    transition = Case("transition", [], transition="mask", require_zero_allocations=False)
    transition_report = copy.deepcopy(report)
    transition_report.update(frames=24, width=640, height=360)
    transition_report["rt"].update(proxy_mode="manifest", probe_rays=24*512, blas_count=103, blas_builds=206, compactions=206)
    marker = ("RT-PROXY-TRANSITION mask mesh 2 source 300 before 120 after 300 material_full 0 instances_full 0 "
              "material_records 1 instance_records 0 applied 1 promoted 1 verified 1 | PASS\n")
    expect(True, transition, status, text + marker, transition_report)
    expect(False, transition, status, text, transition_report)
    expect(False, transition, status, text + marker.replace("before 120", "before 300"), transition_report)
    expect(False, transition, status, text + marker.replace("verified 1", "verified 0"), transition_report)
    expect(False, transition, status, text + marker.replace("material_records 1", "material_records 0"), transition_report)
    for key, value in (("probe_rays", 21*512), ("blas_builds", 103), ("compactions", 103)):
        bad = copy.deepcopy(transition_report); bad["rt"][key] = value
        expect(False, transition, status, text + marker, bad)
    full = Case("transition-full", [], transition="full-upload", require_zero_allocations=False)
    full_marker = marker.replace("mask mesh", "full-upload mesh").replace("material_full 0", "material_full 1").replace("instances_full 0", "instances_full 1").replace("material_records 1", "material_records 0")
    expect(True, full, status, text + full_marker, transition_report)
    expect(False, full, status, text + full_marker.replace("instances_full 1", "instances_full 0"), transition_report)
    deform = Case("deform", [], deform=True, probe="primary", require_zero_allocations=False)
    deform_report = copy.deepcopy(report)
    deform_report["rt"].update(probe="primary", probe_rays=512, blas_count=2, blas_builds=6, blas_refits=68, compactions=2)
    expect(True, deform, status, text, deform_report)
    for field, value in (("blas_refits", 0), ("blas_builds", 2), ("compactions", 0), ("probe", "none")):
        bad = copy.deepcopy(deform_report); bad["rt"][field] = value
        expect(False, deform, status, text, bad)
    print(f"f9_check: {tests} CPU evidence-evaluator fixtures passed; no GPU invocation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build/release"))
    parser.add_argument("--out", type=Path, default=Path("build/f9-check"))
    parser.add_argument("--proxy-manifest", type=Path, default=Path("assets/manifests/sponza.rtproxy.json"))
    parser.add_argument("--families", default="native,apple9", help="comma-separated native,apple9")
    parser.add_argument("--only", action="append", default=[], metavar="GLOB")
    parser.add_argument("--quick", action="store_true", help="shorter runs; omit lifecycle/flight/rebuild cases")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    families = args.families.split(",")
    if not families or len(set(families)) != len(families) or any(f not in ("native", "apple9") for f in families):
        parser.error("--families must contain unique native and/or apple9 entries")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    repo = Path(__file__).resolve().parents[1]
    build, out, proxy_manifest = args.build.resolve(), args.out.resolve(), args.proxy_manifest.resolve()
    cases = make_cases(families, proxy_manifest, args.quick)
    if args.only:
        cases = [case for case in cases if any(fnmatch.fnmatchcase(case.name, pattern) for pattern in args.only)]
    if not cases:
        parser.error("no cases selected")
    if args.list:
        print("\n".join(case.name for case in cases))
        return 0
    app = build / "phosphor"
    common = [str(app), "--offscreen", "--no-ui", "--no-vsync", "--fixed-timestep", "--rt", "on",
              "--debug-rt", "1", "--warmup", "16" if args.quick else "60", "--frames", "16" if args.quick else "48",
              "--resolution", "640x360"]
    entries = []
    for case in cases:
        report_path = out / f"{case.name}.json"
        command = [*common, *case.flags, "--report", str(report_path)]
        if case.family == "apple9":
            command += ["--force-family", "apple9"]
        if case.capture:
            command += ["--capture", str(out / f"{case.name}.png")]
        entries.append({**asdict(case), "command": command, "expected_exit": 1 if case.negative else 0,
                        "log": str(out / f"{case.name}.log"), "report": str(report_path)})
    if out.exists() and any(out.iterdir()):
        parser.error(f"output directory must be empty to prevent stale evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    plan = {"runner_schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "state": "PLANNED", "scope": "F9 functional correctness only",
            "full_matrix": not args.only and not args.quick and set(families) == {"native", "apple9"},
            "not_covered": NOT_COVERED, "validation_env": {"MTL_DEBUG_LAYER": "1", "MTL_SHADER_VALIDATION": "1",
                                                          "MTL_DEBUG_LAYER_WARNING_MODE": "nslog"}, "cases": entries}
    (out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        for entry in entries:
            print(shlex.join(entry["command"]))
        print(f"PLANNED: {len(entries)} commands; no renderer execution; {out / 'manifest.json'}")
        return 0
    os.chdir(repo) # relative scene paths and run_checked's source manifest
    summary = {"scope": plan["scope"], "state": "RUNNING", "planned_cases": len(entries), "results": [],
               "not_covered": NOT_COVERED, "phase_accepted": False}
    summary_path = out / "summary.json"
    def save():
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    save()
    try:
        if not app.is_file() or not os.access(app, os.X_OK):
            raise RuntimeError(f"renderer executable missing: {app}")
        if any(case.needs_sponza for case in cases) and not (repo / "assets/sponza/Sponza.gltf").is_file():
            raise RuntimeError("Sponza is required by selected cases but assets/sponza/Sponza.gltf is missing")
        if any(case.proxy or case.transition for case in cases) and not proxy_manifest.is_file():
            raise RuntimeError(f"proxy manifest required by selected cases: {proxy_manifest}")
        env = {**os.environ, **plan["validation_env"]}
        for case, entry in zip(cases, entries):
            expected_word = "FAIL" if case.negative else "PASS"
            required = (rf"^RT checks [1-9]\d* failures \d+ \| {expected_word}\b",)
            status = run_checked(entry["command"], Path(entry["log"]), expected=entry["expected_exit"],
                                 required=required, timeout=args.timeout, env=env)
            text = Path(entry["log"]).read_text(errors="replace")
            report, error = read_report(Path(entry["report"]))
            result = evaluate(case, status, text, report, error)
            if case.capture:
                capture = out / f"{case.name}.png"
                if not capture.is_file() or not capture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                    result["passed"] = False
                    result["failures"].append("requested RT debug capture is missing or is not PNG")
                result["capture"] = str(capture)
            result.update(command=entry["command"], raw_status=entry["log"] + ".status.json",
                          log=entry["log"], report=entry["report"] if report is not None else None)
            summary["results"].append(result)
            save()
            print(("PASS" if result["passed"] else "FAIL") + ": " + case.name, flush=True)
            if not result["passed"]:
                raise RuntimeError("; ".join(result["failures"]))
        summary["state"] = "FUNCTIONAL_CASES_PASSED"
        summary["passed"] = True
    except (RuntimeError, OSError, ValueError, KeyboardInterrupt) as error:
        summary["state"] = "FAILED_OR_INTERRUPTED"
        summary["passed"] = False
        summary["error"] = str(error) or type(error).__name__
        print(summary["error"], file=sys.stderr, flush=True)
    finally:
        summary["completed_cases"] = len(summary["results"])
        summary["not_run"] = [entry["name"] for entry in entries[len(summary["results"]):]]
        save()
    return 0 if summary.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
