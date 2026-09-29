#!/usr/bin/env python3
"""xctrace_passes.py -- per-pass GPU breakdown from `xctrace export` tables (F4.4).

Reads three tables exported from a Metal System Trace (see tools/gpu_trace.sh):

  metal-gpu-intervals                  one interval per encoder and channel
  metal-shader-profiler-intervals      "Shader Timeline": per-shader samples
  gpu-performance-state-intervals      GPU clock state over time

and, for the phosphor process only, prints a Markdown report and writes a JSON
file with (a) duration statistics per encoder label and channel, (b) time and
mean percent-of-kick per shader function, (c) the share of each render-graph
pass inside each fused encoder (shaders are mapped to passes), (d) the time
spent in each GPU performance state.

NO HARDWARE COUNTERS (occupancy / bandwidth / limiters) ARE AVAILABLE HEADLESS;
see docs/opt-log.md, section "F4", point 3.

The tables are 100+ MB: they are parsed with a streaming iterparse.  xctrace
de-duplicates values with id="N" / ref="N"; the raw element text (integer ns,
float percent) is used for arithmetic, never the formatted `fmt` attribute.

Usage:
  xctrace_passes.py --dir DIR [--process phosphor] [--report report.json]
                    [--map PASS=shader1,shader2 ...] [--out-md F] [--out-json F]
  xctrace_passes.py --self-test
"""
import argparse
import bisect
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

NO_COUNTERS = ("no hardware counters (occupancy/bandwidth/limiters) are "
               "available headless; see opt-log F4")

# Shader function -> pass, used when the app report carries no shaders (JSON
# report v1, or v2 without "shaders").  Keys are render-graph pass names.
DEFAULT_MAP = {
    "Forward": ["forward_vs", "forward_fs"],
    "ImGui overlay": ["imgui_vs", "imgui_fs"],
}

FILES = {
    "gpu": "metal-gpu-intervals.xml",
    "shaders": "metal-shader-profiler-intervals.xml",
    "perf": "gpu-performance-state-intervals.xml",
}

# Tags whose id can be the target of a later ref (all direct row children we
# read, plus `process`, which xctrace defines nested inside formatted labels).
_KEEP = {"start-time", "duration", "gpu-channel-name", "formatted-label",
         "process", "percent", "metal-object-label", "gpu-performance-state",
         "metal-nesting-level"}


def iter_rows(path):
    """Yield {mnemonic: (raw_text, fmt)} per row; id/ref resolved, memory flat."""
    ids = {}
    cols = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "col":
            m = el.find("mnemonic")
            if m is not None:
                cols.append(m.text)
        elif el.tag == "row":
            for x in el.iter("process"):
                if "id" in x.attrib:
                    ids[x.attrib["id"]] = (x.text, x.attrib.get("fmt"))
            vals = []
            for ch in el:
                if "ref" in ch.attrib:
                    v = ids.get(ch.attrib["ref"])
                elif ch.tag == "sentinel":
                    v = None
                else:
                    v = (ch.text, ch.attrib.get("fmt"))
                    if "id" in ch.attrib and ch.tag in _KEEP:
                        ids[ch.attrib["id"]] = v
                vals.append(v)
            yield dict(zip(cols, vals))
            el.clear()


def is_process(value, name, pid=None):
    """value = (raw, fmt) of a `process` column, fmt like 'phosphor (29945)'."""
    if not value or not value[1]:
        return False
    fmt = value[1]
    if pid is not None:
        return fmt.endswith("(%s)" % pid)
    return fmt == name or fmt.startswith(name + " (")


def clean_label(fmt):
    """'Command Buffer 0:Forward + ImGui overlay  (phosphor (1)) 0xab' -> pass label."""
    s = re.sub(r"\s+\(.*\)\s+0x[0-9a-fA-F]+$", "", fmt or "").strip()
    s = re.sub(r"^Command Buffer \d+:", "", s)
    return s or "(unlabelled)"


def strip_pso(name):
    """'forward_fs (19)' -> 'forward_fs'."""
    return re.sub(r"\s+\(\d+\)$", "", name or "").strip()


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def stats_ms(ns_values):
    v = sorted(x / 1e6 for x in ns_values)
    return {"count": len(v), "mean_ms": sum(v) / len(v) if v else 0.0,
            "p50_ms": percentile(v, 0.5), "p99_ms": percentile(v, 0.99)}


def load_encoders(path, process, pid=None):
    """-> list of (start_ns, dur_ns, channel, label), phosphor rows only."""
    out = []
    for r in iter_rows(path):
        if not is_process(r.get("process"), process, pid):
            continue
        lab, ch = r.get("event-label"), r.get("channel-name")
        if not lab or not ch or not r.get("start") or not r.get("duration"):
            continue
        out.append((int(r["start"][0]), int(r["duration"][0]), ch[0], clean_label(lab[1])))
    return out


def load_shaders(path, process, pid=None):
    """-> list of (start_ns, dur_ns, channel, shader, percent_of_kick)."""
    out = []
    for r in iter_rows(path):
        if not is_process(r.get("process"), process, pid):
            continue
        name, ch = r.get("pso-label"), r.get("channel-name")
        if not name or not ch or not r.get("start") or not r.get("duration"):
            continue
        pct = r.get("percent-of-kick")
        out.append((int(r["start"][0]), int(r["duration"][0]), ch[0],
                    strip_pso(name[0] or name[1]), float(pct[0]) if pct and pct[0] else 0.0))
    return out


def load_perf_states(path):
    """-> {state: total_ns} (device-wide table: not per process)."""
    tot = defaultdict(int)
    for r in iter_rows(path):
        st, d = r.get("gpu-performance-state"), r.get("duration")
        if st and d:
            tot[st[0]] += int(d[0])
    return dict(tot)


def encoder_stats(encoders):
    """-> [{label, channel, count, mean_ms, p50_ms, p99_ms}] sorted by total time."""
    g = defaultdict(list)
    for _, d, ch, lab in encoders:
        g[(lab, ch)].append(d)
    rows = [dict(label=k[0], channel=k[1], **stats_ms(v)) for k, v in g.items()]
    rows.sort(key=lambda r: -(r["mean_ms"] * r["count"]))
    return rows


def attribute_shaders(encoders, shaders):
    """Assign each shader sample to the encoder (same channel) containing its start.

    -> {(label, channel, shader): {"total_ns", "pct_sum", "n"}}; samples that fall
    in no encoder go to the label '(unattributed)'.
    """
    by_ch = defaultdict(list)
    for s, d, ch, lab in encoders:
        by_ch[ch].append((s, s + d, lab))
    for ch in by_ch:
        by_ch[ch].sort()
    starts = {ch: [x[0] for x in v] for ch, v in by_ch.items()}
    agg = defaultdict(lambda: {"total_ns": 0, "pct_sum": 0.0, "n": 0})
    for s, d, ch, name, pct in shaders:
        lab = "(unattributed)"
        ivs = by_ch.get(ch)
        if ivs:
            i = bisect.bisect_right(starts[ch], s) - 1
            # depth>0 intervals overlap: look back a few candidates for the one covering s
            for j in range(i, max(i - 8, -1), -1):
                if ivs[j][0] <= s < ivs[j][1]:
                    lab = ivs[j][2]
                    break
        a = agg[(lab, ch, name)]
        a["total_ns"] += d
        a["pct_sum"] += pct
        a["n"] += 1
    return agg


def _set_ci(m, key, value):
    for k in [k for k in m if k.lower() == key.lower()]:
        del m[k]
    m[key] = value


def build_pass_map(report, overrides):
    """pass name -> list of shader function names (later sources win, case-insensitively)."""
    m = {k: list(v) for k, v in DEFAULT_MAP.items()}
    source = "built-in engine map"
    if report and report.get("schema_version", 1) >= 2:
        used = False
        for unit in report.get("passes", []):
            sh = [s for s in unit.get("shaders", []) if s]
            ps = unit.get("passes", [])
            if sh and len(ps) == 1:            # unfused unit: shaders belong to the pass
                _set_ci(m, ps[0], sh)
                used = True
        if used:
            source = "report v2 passes[].shaders"
    for spec in overrides or []:
        k, _, v = spec.partition("=")
        _set_ci(m, k.strip(), [x.strip() for x in v.split(",") if x.strip()])
        source += " + --map"
    return m, source


def pass_shares(agg, pass_map):
    """-> {label: {"total_ms", "passes": {pass: {"ms", "share"}}}} inside each encoder."""
    shader_to_pass = {}
    for p, shs in pass_map.items():
        for s in shs:
            shader_to_pass[s] = p
    per_label = defaultdict(lambda: defaultdict(int))
    for (lab, _ch, sh), a in agg.items():
        per_label[lab][shader_to_pass.get(sh, "(other)")] += a["total_ns"]
    out = {}
    for lab, d in per_label.items():
        total = sum(d.values())
        # keep the display case of pass names as written in the label
        names = {p.lower(): p for p in lab.split(" + ")}
        out[lab] = {
            "total_ms": total / 1e6,
            "passes": {names.get(p.lower(), p): {"ms": v / 1e6, "share": v / total if total else 0.0}
                       for p, v in sorted(d.items(), key=lambda kv: -kv[1])},
        }
    return out


def analyse(gpu_path, shader_path, perf_path, process="phosphor", pid=None,
            report=None, overrides=None):
    result = {"note": NO_COUNTERS, "process": process}
    enc = load_encoders(gpu_path, process, pid) if gpu_path else []
    result["encoders"] = encoder_stats(enc)
    sh = load_shaders(shader_path, process, pid) if shader_path else []
    agg = attribute_shaders(enc, sh)
    pass_map, src = build_pass_map(report, overrides)
    result["shaders"] = sorted(
        ({"encoder": lab, "channel": ch, "shader": name,
          "total_ms": a["total_ns"] / 1e6, "samples": a["n"],
          "mean_percent_of_kick": a["pct_sum"] / a["n"]}
         for (lab, ch, name), a in agg.items()),
        key=lambda r: -r["total_ms"])
    result["pass_map"] = {"source": src, "map": pass_map}
    result["pass_shares"] = pass_shares(agg, pass_map)
    perf = load_perf_states(perf_path) if perf_path else {}
    tot = sum(perf.values())
    result["perf_states"] = [{"state": k, "ms": v / 1e6, "share": v / tot if tot else 0.0}
                             for k, v in sorted(perf.items(), key=lambda kv: -kv[1])]
    if not sh:
        result["warning"] = ("the Shader Timeline table has no rows for this process: "
                             "record more seconds and/or with the 'Metal GPU Counters' "
                             "instrument (XCTRACE_INSTRUMENT); pass shares are unavailable")
    return result


def to_markdown(res):
    L = ["# GPU trace per pass (Metal System Trace)", "",
         "> **%s**" % NO_COUNTERS, "",
         "Process: `%s`. Durations are GPU wall-clock intervals per encoder and channel; "
         "in a fused render encoder Vertex/Fragment intervals overlap in time and "
         "must not be summed." % res["process"], ""]
    if res.get("warning"):
        L += ["> WARNING: %s" % res["warning"], ""]
    L += ["## Encoders by label and channel", "",
          "| Encoder | Channel | Count | Mean ms | p50 ms | p99 ms |", "|---|---|---:|---:|---:|---:|"]
    for r in res["encoders"]:
        L.append("| %s | %s | %d | %.3f | %.3f | %.3f |" % (
            r["label"], r["channel"], r["count"], r["mean_ms"], r["p50_ms"], r["p99_ms"]))
    L += ["", "## Shader Timeline (per encoder, channel and shader function)", "",
          "| Encoder | Channel | Shader | Total ms | Samples | Mean % of kick |",
          "|---|---|---|---:|---:|---:|"]
    for r in res["shaders"]:
        L.append("| %s | %s | %s | %.3f | %d | %.1f |" % (
            r["encoder"], r["channel"], r["shader"], r["total_ms"], r["samples"],
            r["mean_percent_of_kick"]))
    L += ["", "## Pass share inside each encoder (shader sample time)", "",
          "Shader-to-pass map: %s.  Shares are sampled shader wall time (sum over "
          "channels); on a TBDR GPU the fragments of fused passes interleave per tile, "
          "so treat them as indicative, not as exclusive per-pass cost." % res["pass_map"]["source"], "",
          "| Encoder | Pass | Shader ms | Share |", "|---|---|---:|---:|"]
    for lab, d in res["pass_shares"].items():
        for p, v in d["passes"].items():
            L.append("| %s | %s | %.3f | %.1f%% |" % (lab, p, v["ms"], 100 * v["share"]))
    L += ["", "## GPU performance state (device-wide, time in state)", "",
          "| State | Time ms | Share |", "|---|---:|---:|"]
    for r in res["perf_states"]:
        L.append("| %s | %.1f | %.1f%% |" % (r["state"], r["ms"], 100 * r["share"]))
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def self_test():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "xctrace")
    res = analyse(os.path.join(here, FILES["gpu"]), os.path.join(here, FILES["shaders"]),
                  os.path.join(here, FILES["perf"]), "phosphor")
    exp = json.load(open(os.path.join(here, "expected.json")))
    failures = []

    def check(name, got, want):
        if isinstance(want, float):
            ok = abs(got - want) < 1e-6
        else:
            ok = got == want
        if not ok:
            failures.append("%s: got %r want %r" % (name, got, want))

    check("clean_label", clean_label("Command Buffer 0:A + B  (phosphor (1)) 0x1f"), "A + B")
    check("strip_pso", strip_pso("forward_fs (19)"), "forward_fs")
    check("percentile", percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
    check("no_counters_note", "no hardware counters" in NO_COUNTERS, True)
    for section in ("encoders", "shaders", "pass_shares", "perf_states"):
        check(section, json.loads(json.dumps(res[section])), exp[section])
    rep = {"schema_version": 2, "passes": [
        {"name": "Shadow", "passes": ["Shadow"], "shaders": ["shadow_vs"]},
        {"name": "Forward + ImGui overlay", "passes": ["Forward", "ImGui overlay"], "shaders": ["a", "b"]}]}
    m, src = build_pass_map(rep, ["forward=my_vs,my_fs"])
    check("map_report", m.get("Shadow"), ["shadow_vs"])
    check("map_override_ci", m.get("forward"), ["my_vs", "my_fs"])
    check("map_default_kept", m.get("ImGui overlay"), ["imgui_vs", "imgui_fs"])
    check("map_fused_ignored", "Forward + ImGui overlay" in m, False)
    check("map_source", src, "report v2 passes[].shaders + --map")
    # other processes must be filtered out
    check("only_phosphor", all(r["label"] != "WindowServer thing" for r in res["encoders"]), True)
    if failures:
        print("self-test FAILED:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print("self-test OK (%d encoder rows, %d shader rows)" % (len(res["encoders"]), len(res["shaders"])))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="directory holding the three exported XML tables")
    ap.add_argument("--gpu-intervals")
    ap.add_argument("--shader-intervals")
    ap.add_argument("--perf-state")
    ap.add_argument("--process", default="phosphor", help="process name (default phosphor)")
    ap.add_argument("--pid", help="restrict to this pid")
    ap.add_argument("--report", help="app JSON report (v2 passes[].shaders used for the pass map)")
    ap.add_argument("--map", action="append", default=[], metavar="PASS=SHADER1,SHADER2",
                    help="override/add a pass-to-shaders mapping (repeatable)")
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--update-expected", action="store_true",
                    help="regenerate tools/testdata/xctrace/expected.json (review the diff!)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.update_expected:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "xctrace")
        r = analyse(*(os.path.join(here, FILES[k]) for k in ("gpu", "shaders", "perf")))
        with open(os.path.join(here, "expected.json"), "w") as f:
            json.dump({k: r[k] for k in ("encoders", "shaders", "pass_shares", "perf_states")}, f, indent=1)
        return 0

    def pick(explicit, key):
        if explicit:
            return explicit
        if a.dir:
            p = os.path.join(a.dir, FILES[key])
            return p if os.path.exists(p) else None
        return None

    gpu, shd, perf = pick(a.gpu_intervals, "gpu"), pick(a.shader_intervals, "shaders"), pick(a.perf_state, "perf")
    if not (gpu or shd or perf):
        ap.error("give --dir or at least one table")
    report = json.load(open(a.report)) if a.report else None
    res = analyse(gpu, shd, perf, a.process, a.pid, report, a.map)
    md = to_markdown(res)
    print(md)
    if a.out_md:
        with open(a.out_md, "w") as f:
            f.write(md)
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
