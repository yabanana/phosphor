#!/usr/bin/env python3
"""perf_table.py -- Markdown tables from the perf history CSVs (F4.5/F4.6).

  tools/perf_table.py latest                       newest recorded commit
  tools/perf_table.py compare <commitA> <commitB>  delta per bench/config/metric
  tools/perf_table.py passes <commit> [--bench N]  per-unit GPU table
  tools/perf_table.py report FILE.json             one bench_all-style report
  tools/perf_table.py --self-test                  exercise parsing/formatting

Common options: --csv PATH, --passes-csv PATH (default docs/perf-history.csv
and docs/perf-history-passes.csv next to this script's repo).  Commits match by
prefix.  When a commit was recorded several times, the last rows win.

compare flags a metric with "**" when |delta| > 2 * max(sdA, sdB), i.e. the
change is outside the run-to-run noise of BOTH sets; otherwise "~" (within
noise).  A metric without sd (single run, or absent) is never flagged.  Records
come from tools/perf_record.sh; GPU columns are only meaningful with vsync=1.
Python 3 standard library only.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
TESTDATA = os.path.join(HERE, "testdata")

METRICS = [("frame", "Frame ms"), ("cpu", "CPU ms"), ("gpu", "GPU ms"),
           ("gpupass", "GPU pass-sum ms")]


def num(v):
    """CSV/JSON cell to float, None when empty or not a number."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt(v, digits=3):
    v = num(v)
    if v is None:
        return "-"
    return f"{round(v, digits):g}"


def fmt_count(v):
    """Counter mean as a plain integer (no exponent notation)."""
    v = num(v)
    return "-" if v is None else f"{round(v):d}"


def fmt_ms(mean, p99=None):
    """'mean (p99)' as in docs/perf-log.md."""
    if num(mean) is None:
        return "-"
    return f"{fmt(mean)} ({fmt(p99)})" if num(p99) is not None else fmt(mean)


def fmt_ms95(mean, p95, p99):
    """'mean (p95, p99)' (schema 6); a missing percentile shows '-'."""
    if num(mean) is None:
        return "-"
    return f"{fmt(mean)} ({fmt(p95)}, {fmt(p99)})"


def fmt_sd(mean, sd):
    if num(mean) is None:
        return "-"
    m, s = num(mean), num(sd) or 0.0
    cv = (s / m * 100.0) if m else 0.0
    return f"{fmt(m)} ± {fmt(s)} ({cv:.1f}%)"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_commit(rows, prefix):
    """Full recorded commit string matching `prefix` (latest wins)."""
    found = None
    for r in rows:
        c = r["commit"]
        if c.startswith(prefix) or prefix.startswith(c):
            found = c
    if found is None:
        raise SystemExit(f"error: commit {prefix} not in the history")
    return found


def rows_of(rows, commit):
    """Last row per (bench, vsync) of a commit, ordered."""
    last = {}
    for r in rows:
        if r["commit"] == commit:
            last[(int(r["bench"]), r["vsync"])] = r
    return [last[k] for k in sorted(last, key=lambda k: (k[1] != "1", k[0]))]


def describe(r):
    dirty = " (dirty tree)" if r.get("dirty") == "1" else ""
    return (f"**{r['date']}** · commit `{r['commit']}`{dirty} · {r['machine']} · "
            f"{r['chip']} · macOS {r['macos']} · {r['power']} · "
            f"{r['frames']} frames x {r['runs']} runs, --no-ui")


def cmd_latest(rows):
    if not rows:
        raise SystemExit("error: the history is empty")
    commit = rows[-1]["commit"]
    sel = rows_of(rows, commit)
    out = [describe(sel[-1]), ""]
    for vs, label in (("1", "vsync on"), ("0", "vsync off")):
        part = [r for r in sel if r["vsync"] == vs]
        if not part:
            continue
        out += [f"{label} (mean ± sd across runs (CV%); p99 of the median run in parentheses)", ""]
        body = []
        for r in part:
            body.append([r["bench"], r["name"], r["resolution"],
                         fmt_sd(r["frame_mean"], r["frame_sd"]) + f" [{fmt(r['frame_p99'])}]",
                         fmt_sd(r["cpu_mean"], r["cpu_sd"]) + f" [{fmt(r['cpu_p99'])}]",
                         fmt_sd(r["gpu_mean"], r["gpu_sd"]) + f" [{fmt(r['gpu_p99'])}]",
                         fmt_sd(r["gpupass_mean"], r["gpupass_sd"]),
                         r["gpu_allocations_max"]])
        out += [table(["#", "Bench", "Resolution", "Frame ms [p99]", "CPU ms [p99]",
                       "GPU ms [p99]", "GPU pass-sum ms", "GPU allocs"], body), ""]
    return "\n".join(out)


def compare_metric(a, b, key):
    ma, mb = num(a[key + "_mean"]), num(b[key + "_mean"])
    if ma is None or mb is None:
        return None
    sda, sdb = num(a[key + "_sd"]) or 0.0, num(b[key + "_sd"]) or 0.0
    delta = mb - ma
    pct = (delta / ma * 100.0) if ma else 0.0
    thr = 2.0 * max(sda, sdb)
    flag = "**" if thr > 0 and abs(delta) > thr else "~"
    return ma, mb, delta, pct, flag


def cmd_compare(rows, pa, pb):
    ca, cb = resolve_commit(rows, pa), resolve_commit(rows, pb)
    ra = {(r["bench"], r["vsync"]): r for r in rows_of(rows, ca)}
    rb = {(r["bench"], r["vsync"]): r for r in rows_of(rows, cb)}
    out = [f"`{ca}` (A) vs `{cb}` (B); delta = B - A; `**` = |delta| > 2 sd of both "
           "sets, `~` = within noise", ""]
    body = []
    for key in sorted(set(ra) & set(rb), key=lambda k: (k[1] != "1", int(k[0]))):
        a, b = ra[key], rb[key]
        for m, label in METRICS:
            if m in ("gpu", "gpupass") and key[1] == "0":
                continue  # GPU ms are only meaningful with vsync
            res = compare_metric(a, b, m)
            if res is None:
                continue
            ma, mb, d, pct, flag = res
            body.append([a["bench"], a["name"], "on" if key[1] == "1" else "off", label,
                         fmt(ma), fmt(mb), f"{d:+.3f}", f"{pct:+.1f}%", flag])
    if not body:
        out.append("_no bench/config in common_")
    else:
        out.append(table(["#", "Bench", "vsync", "Metric", "A", "B", "Delta", "Delta %", "Sig."], body))
    return "\n".join(out)


def cmd_passes(prows, prefix, bench):
    commit = resolve_commit(prows, prefix)
    last = {}
    for r in prows:
        if r["commit"] == commit and (bench is None or int(r["bench"]) == bench):
            last[(r["bench"], r["vsync"], r["unit"])] = r
    if not last:
        return f"_no per-pass rows for `{commit}`_"
    out = [f"Per-unit GPU ms for `{commit}` (mean over runs ± sd across runs)", ""]
    groups = {}
    for (b, vs, _), r in last.items():
        groups.setdefault((b, vs), []).append(r)
    for (b, vs) in sorted(groups, key=lambda k: (int(k[0]), k[1] != "1")):
        body = []
        for r in groups[(b, vs)]:
            dram = num(r["dram_bytes"])
            body.append([r["unit"], r["queue"] or "-", "yes" if r["fused"] == "1" else "no",
                         fmt(r["gpu_mean"]), fmt(r["gpu_sd"]), fmt(r["gpu_p99"]),
                         fmt(r["gpu_max"]), fmt(dram / 1048576.0) if dram is not None else "-"])
        out += [f"#### Bench {b} (vsync {'on' if vs == '1' else 'off'})", "",
                table(["Unit", "Queue", "Fused", "GPU ms mean", "sd", "p99", "max", "DRAM MiB"], body), ""]
    return "\n".join(out)


def cmd_report(path):
    """bench_all-style rows of one JSON report, schema v1..v6 (v3 per-pass "work" is not shown;
    v6 adds p95 to every summary, a "hardware" line and a "meshlets" table)."""
    with open(path) as f:
        j = json.load(f)
    g = lambda *ks: _dig(j, ks)
    v = j.get("schema_version", 1)
    out = [f"{j.get('bench', '?')} · {j.get('width')}x{j.get('height')} · schema v{v}", "",
           table(["FPS", "Frame ms (p95, p99)", "CPU ms (p95, p99)", "GPU ms (p95, p99)", "Wait ms (p99)",
                  "GPU pass-sum ms"],
                 [[fmt(j.get("fps"), 0),
                   fmt_ms95(g("frame_ms", "mean"), g("frame_ms", "p95"), g("frame_ms", "p99")),
                   fmt_ms95(g("cpu_ms", "mean"), g("cpu_ms", "p95"), g("cpu_ms", "p99")),
                   fmt_ms95(g("gpu_ms", "mean"), g("gpu_ms", "p95"), g("gpu_ms", "p99")),
                   fmt_ms(g("wait_ms", "mean"), g("wait_ms", "p99")),
                   fmt_ms(g("gpu_pass_sum_ms", "mean"), g("gpu_pass_sum_ms", "p99"))]])]
    body = []
    for p in j.get("passes") or []:
        gm = p.get("gpu_ms") or {}
        dram = num(p.get("dram_bytes"))
        body.append([p.get("name", "?"), p.get("queue", "-"), "yes" if p.get("fused") else "no",
                     fmt(gm.get("mean")), fmt(gm.get("p99")), fmt(gm.get("max")),
                     fmt(dram / 1048576.0) if dram is not None else "-"])
    out += ["", table(["Unit", "Queue", "Fused", "GPU ms mean", "p99", "max", "DRAM MiB"], body)
            if body else "_no per-pass data_"]
    # Schema 5: the persistent GPU scene and the CPU phases (absent in older reports).
    sc = j.get("scene")
    if isinstance(sc, dict):
        mean = lambda k: fmt(_dig(sc, (k, "mean")), 0)
        out += ["", f"scene: gpu-driven {sc.get('mode', '?')} · {sc.get('instances', '-')} instances · "
                    f"{sc.get('slots', '-')} slots · {sc.get('buckets', '-')} buckets · "
                    f"{sc.get('materials', '-')} materials · queue overflow {sc.get('queue_overflow', '-')}",
                "",
                table(["Visible", "Culled frustum", "Culled distance", "Culled size", "Draw cmds", "CPU cmds",
                       "Upload bytes", "Delta records"],
                      [[mean("visible"), mean("culled_frustum"), mean("culled_distance"), mean("culled_size"),
                        mean("draw_commands"), mean("cpu_commands"), mean("upload_bytes"), mean("delta_records")]])]
    cp = j.get("cpu_phases")
    if isinstance(cp, dict):
        names = [("sim", "Sim"), ("scene_sync", "Scene sync"), ("prepare", "Prepare"), ("ui", "UI"),
                 ("graph", "Graph"), ("submit", "Submit")]
        out += ["", "CPU ms per frame phase (mean, p99)", "",
                table([n for _, n in names],
                      [[fmt_ms(_dig(cp, (k, "mean")), _dig(cp, (k, "p99"))) for k, _ in names]])]
    # Schema 6: hardware manifest and meshlet path.
    hw = j.get("hardware")
    if isinstance(hw, dict):
        mem = num(hw.get("memory_bytes"))
        unv = hw.get("unverified_devices") or []
        out += ["", f"hardware: {hw.get('physical_device', '?')} ({hw.get('physical_family', '?')}, "
                    f"{fmt(mem / 1073741824.0, 1) if mem is not None else '-'} GiB) · effective "
                    f"{hw.get('effective_capabilities', '?')} · preset {hw.get('preset') or 'none'} · "
                    f"validation scope {hw.get('validation_scope', '?')} · unverified devices: "
                    f"{', '.join(unv) if unv else 'none'}"]
    ml = j.get("meshlets")
    if isinstance(ml, dict):
        out += ["", table(["Path", "Cull", "Hi-Z req/eff", "Cook", "Meshlets", "Capacity", "Overflow frames",
                           "History resets", "Checks/failures"],
                          [[ml.get("path", "?"), ml.get("cull", "?"),
                            f"{ml.get('hiz_requested', '?')}/{ml.get('hiz_effective', '?')}", ml.get("cook", "?"),
                            ml.get("meshlets", "-"), ml.get("candidate_capacity", "-"),
                            ml.get("overflow_frames", "-"), ml.get("history_resets", "-"),
                            f"{ml.get('checks', '-')}/{ml.get('check_failures', '-')}"]]), ""]
        keys = ["candidates", "drawn_a", "frustum", "cone", "history_rejected", "drawn_b", "occluded_b",
                "primitives"]
        out += ["Meshlet counters per frame (mean)", "",
                table(keys, [[fmt_count(_dig(ml, (k, "mean"))) for k in keys]])]
    return "\n".join(out)


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def self_test():
    def check(cond, msg):
        if not cond:
            raise SystemExit(f"self-test FAILED: {msg}")

    check(fmt(None) == "-" and fmt("") == "-" and fmt("1.23456") == "1.235", "fmt")
    check(fmt_sd("2", "0.1") == "2 ± 0.1 (5.0%)", "fmt_sd")
    check(fmt_ms(1.5, None) == "1.5" and fmt_ms(None) == "-", "fmt_ms")
    v1 = cmd_report(os.path.join(TESTDATA, "report_v1.json"))
    check("schema v1" in v1 and "_no per-pass data_" in v1, "v1 report")
    check("| - |" in v1 or "- |" in v1, "v1 pass-sum shown as -")
    v2 = cmd_report(os.path.join(TESTDATA, "report_v2.json"))
    check("schema v2" in v2 and "Lighting" in v2 and "| 50 |" in v2, "v2 report (DRAM MiB)")
    check("scene:" not in v1 and "scene:" not in v2 and "CPU ms per frame phase" not in v2,
          "older reports show no scene section")
    v5 = cmd_report(os.path.join(TESTDATA, "report_v5.json"))
    check("schema v5" in v5 and "scene: gpu-driven on · 1000000 instances" in v5, "v5 scene header")
    check("| 350000 | 650000 | 0 | 0 | 24 | 3 | 262144 | 2730 |" in v5, "v5 scene row (visible, cpu cmds, upload)")
    check("CPU ms per frame phase" in v5 and "| 0.9 (1) |" in v5, "v5 cpu phases")
    check("(-, 9)" in v5 and "hardware:" not in v5 and "Meshlet counters" not in v5,
          "v5: missing p95 shows -, no hardware/meshlets sections")
    check("(-, " in v1, "v1: missing p95 shows -")
    v6 = cmd_report(os.path.join(TESTDATA, "report_v6.json"))
    check("schema v6" in v6, "v6 header")
    check("8.5 (9.2, 9.8)" in v6, "v6 frame p95/p99")
    check("hardware: Apple M5 Max (apple10, 128 GiB) · effective apple9 · preset t0-apple9 · "
          "validation scope development · unverified devices: M3 Base, M4 Pro" in v6, "v6 hardware line")
    check("| mesh | two-phase | auto/compute | standard-64v124t | 41210 | 1048576 | 0 | 0 | 12/0 |" in v6,
          "v6 meshlets row")
    check("| 350000 | 120000 | 150000 | 80000 | 7000 | 45000 | 6500 | 12500000 |" in v6, "v6 meshlet counters")
    rows = read_csv(os.path.join(TESTDATA, "history.csv"))
    prows = read_csv(os.path.join(TESTDATA, "history_passes.csv"))
    check(rows[0]["machine"] == "Mac17,6", "quoted comma in CSV")
    check(resolve_commit(rows, "bbb") == "bbbbbbb", "commit prefix")
    lat = cmd_latest(rows)
    check("`bbbbbbb`" in lat and "PBR Material Grid" in lat and "vsync off" in lat, "latest")
    cmp_ = cmd_compare(rows, "aaa", "bbb")
    # bench 1 vsync GPU: 1.337 -> 1.200, sd 0.02 -> delta -0.137 > 0.04: flagged
    check("-0.137" in cmp_ and "**" in cmp_, "compare flags")
    # bench 1 vsync CPU: 0.115 -> 0.110, delta -0.005 <= 2*0.004: noise
    check("| ~ |" in cmp_, "compare noise")
    check("PBR" not in cmp_, "compare only common benches")
    ps = cmd_passes(prows, "bbb", None)
    check("Cluster build" in ps and "| 50 |" in ps, "passes")
    check("Cluster build" not in cmd_passes(prows, "aaa", 1), "passes per commit")
    print("perf_table.py self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--csv", default=os.path.join(DOCS, "perf-history.csv"))
    ap.add_argument("--passes-csv", default=os.path.join(DOCS, "perf-history-passes.csv"))
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("latest")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    p = sub.add_parser("passes")
    p.add_argument("commit")
    p.add_argument("--bench", type=int)
    r = sub.add_parser("report")
    r.add_argument("file")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.cmd == "latest":
        print(cmd_latest(read_csv(args.csv)))
    elif args.cmd == "compare":
        print(cmd_compare(read_csv(args.csv), args.a, args.b))
    elif args.cmd == "passes":
        print(cmd_passes(read_csv(args.passes_csv), args.commit, args.bench))
    elif args.cmd == "report":
        print(cmd_report(args.file))
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
