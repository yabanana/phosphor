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


def fmt_ms(mean, p99=None):
    """'mean (p99)' as in docs/perf-log.md."""
    if num(mean) is None:
        return "-"
    return f"{fmt(mean)} ({fmt(p99)})" if num(p99) is not None else fmt(mean)


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
    """bench_all-style rows of one JSON report, schema v1 or v2."""
    with open(path) as f:
        j = json.load(f)
    g = lambda *ks: _dig(j, ks)
    v = j.get("schema_version", 1)
    out = [f"{j.get('bench', '?')} · {j.get('width')}x{j.get('height')} · schema v{v}", "",
           table(["FPS", "Frame ms (p99)", "CPU ms (p99)", "GPU ms (p99)", "Wait ms (p99)", "GPU pass-sum ms"],
                 [[fmt(j.get("fps"), 0),
                   fmt_ms(g("frame_ms", "mean"), g("frame_ms", "p99")),
                   fmt_ms(g("cpu_ms", "mean"), g("cpu_ms", "p99")),
                   fmt_ms(g("gpu_ms", "mean"), g("gpu_ms", "p99")),
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
