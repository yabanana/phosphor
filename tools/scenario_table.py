#!/usr/bin/env python3
"""scenario_table.py -- Markdown summary of tools/graph_scenarios.sh (OPT-1).

  tools/scenario_table.py DIR [--diffs FILE] [--validation FILE] [--base off]
  tools/scenario_table.py --self-test

DIR holds the schema-4 reports written by graph_scenarios.sh, named
s<scenario>-v<views>-<mode>-r<round>.json.  One row per (scenario, views,
mode): GPU frame span p50 (median over rounds) and the CV between rounds
(sample sd / mean of the per-round p50), the graph metrics of the report
(DRAM / heap / max live MiB, render passes, memoryless, barriers, plan
status) and the deltas against the base mode (off) for span, DRAM and heap.

--diffs: TSV "scenario<TAB>views<TAB>mode<TAB>pixels" (image_diff of the
capture against the base mode, -1 = capture missing/unreadable).  A mode with
pixels != 0 is flagged "**DIFF n**" and the exit status is 1.
--validation: TSV "scenario<TAB>views<TAB>mode<TAB>lines" (non-INFO lines of
the run under the Metal debug layer + shader validation); lines != 0 is
flagged "**n**" and the exit status is 1.  Exit 2: no reports.
Python 3 standard library only.
"""
import glob
import json
import os
import re
import statistics
import sys

MIB = 1024.0 * 1024.0
NAME = re.compile(r"^s(\d+)-v(\d+)-([a-z]+)-r(\d+)\.json$")


def load_reports(directory):
    """{(scenario, views, mode): [report, ...]} ordered by round."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "s*-v*-*-r*.json"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            rep = json.load(f)
        out.setdefault((int(m[1]), int(m[2]), m[3]), []).append((int(m[4]), rep))
    return {k: [r for _, r in sorted(v, key=lambda t: t[0])] for k, v in out.items()}


def load_tsv(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                out[(int(p[0]), int(p[1]), p[2])] = int(p[3])
    return out


def span_p50(rep):
    v = (rep.get("gpu_frame_span_ms") or {}).get("p50")
    return v if v is not None else (rep.get("gpu_ms") or {}).get("p50")


def cv_percent(values):
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean * 100.0 if mean else 0.0


def pct(base, now):
    if base is None or now is None or base == 0:
        return "-"
    return f"{(now - base) / base * 100.0:+.1f}%"


def f(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}"


def build_table(reports, diffs, validation, base="off"):
    """Returns (markdown, failures)."""
    failures = 0
    head = ("| scenario | views | mode | GPU span p50 ms | CV % | d span | DRAM MiB | d DRAM | heap MiB | d heap |"
            " max live MiB | render passes | memoryless | barriers | plan | pixel diff | validation |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for key in sorted(reports, key=lambda k: (k[0], k[1], k[2] != base, k[2])):
        s, v, mode = key
        reps = reports[key]
        spans = [x for x in (span_p50(r) for r in reps) if x is not None]
        span = statistics.median(spans) if spans else None
        last = reps[-1]
        g = last.get("graph") or {}
        b = reports.get((s, v, base))
        bspan = statistics.median([x for x in (span_p50(r) for r in b) if x is not None] or [0]) if b else None
        bg = (b[-1].get("graph") or {}) if b else {}

        def mib(d, k):
            return d.get(k) / MIB if d.get(k) is not None else None

        pixels = diffs.get(key)
        if mode == base and pixels is None:
            dtxt = "ref"
        elif pixels is None:
            dtxt = "-"
        elif pixels == 0:
            dtxt = "0"
        else:
            dtxt = f"**DIFF {pixels}**"
            failures += 1
        val = validation.get(key)
        if val is None:
            vtxt = "-"
        elif val == 0:
            vtxt = "0"
        else:
            vtxt = f"**{val}**"
            failures += 1
        bad_mode = mode != base
        rows.append(
            f"| {s} | {v} | {mode} | {f(span)} | {f(cv_percent(spans), 1)} | "
            f"{pct(bspan, span) if bad_mode else '-'} | {f(mib(g, 'dram_bytes'), 2)} | "
            f"{pct(bg.get('dram_bytes'), g.get('dram_bytes')) if bad_mode else '-'} | "
            f"{f(mib(g, 'heap_bytes'), 2)} | {pct(bg.get('heap_bytes'), g.get('heap_bytes')) if bad_mode else '-'} | "
            f"{f(mib(g, 'max_live_bytes'), 2)} | {g.get('render_passes', '-')} | {g.get('memoryless', '-')} | "
            f"{g.get('barriers', '-')} | {g.get('plan') or '-'} | {dtxt} | {vtxt} |")
    return head + "\n".join(rows) + "\n", failures


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        def w(mode, rnd, span, dram, heap):
            rep = {"gpu_frame_span_ms": {"p50": span},
                   "graph": {"dram_bytes": dram * MIB, "heap_bytes": heap * MIB, "max_live_bytes": heap * MIB,
                             "render_passes": 5, "memoryless": 1, "barriers": 7, "plan": "none"}}
            with open(os.path.join(d, f"s0-v1-{mode}-r{rnd}.json"), "w") as fh:
                json.dump(rep, fh)
        for r, sp in enumerate([2.0, 2.2, 2.1], 1):
            w("off", r, sp, 500, 160)
            w("plan", r, sp * 0.9, 400, 120)
        rep = load_reports(d)
        table, fails = build_table(rep, {(0, 1, "plan"): 0, (0, 1, "greedy"): 0}, {(0, 1, "plan"): 0})
        assert fails == 0, table
        assert "-10.0%" in table and "-20.0%" in table and "-25.0%" in table, table
        table, fails = build_table(rep, {(0, 1, "plan"): 12}, {(0, 1, "plan"): 3})
        assert fails == 2 and "**DIFF 12**" in table and "**3**" in table, table
    print("scenario_table self-test ok")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    args = [a for a in argv if not a.startswith("--")]
    opts = {}
    it = iter(argv)
    for a in it:
        if a in ("--diffs", "--validation", "--base"):
            opts[a[2:]] = next(it)
            args = [x for x in args if x != opts[a[2:]]]
    if len(args) != 1:
        print(__doc__)
        return 2
    reports = load_reports(args[0])
    if not reports:
        print(f"scenario_table: no reports in {args[0]}", file=sys.stderr)
        return 2
    table, failures = build_table(reports, load_tsv(opts.get("diffs")), load_tsv(opts.get("validation")),
                                  opts.get("base", "off"))
    print(table)
    if failures:
        print(f"**{failures} FAILURE(S): pixel differences or validation lines**")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
