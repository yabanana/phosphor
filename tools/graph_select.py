#!/usr/bin/env python3
"""graph_select.py -- adopt OPT-1 graph plans by measurement (macOS).

The cost model ranks candidate plans (tools/graph_opt --top K), the device
decides: for every scenario family found in the candidate files, run the app
with --graph-opt off and with --graph-opt plan for each candidate, in ROUNDS
rounds of rotating order (OPT-0: bimodal render-pass costs and drifting clocks
need interleaving), compare the captures with the "off" capture of the same
round (every plan must give the same image: 0 pixels), and keep the candidate
with the lowest median GPU frame span.  A candidate is adopted only if it beats
"off" by more than the noise floor (--min-gain, default 1%); otherwise the
family's baseline plan (greedy order, end-of-F4 policies) is written, so the
plans file never makes a scenario slower than its measurement.

  tools/graph_select.py --app build/release/phosphor --candidates DIR \\
      [--scenarios 0,1,2,3] [--views 1] [--rounds 3] [--frames 600] \\
      [--out shaders/graph-plans.json] [--report table.md] [--work DIR]

Refuses to start while another phosphor / soc_bench / graph_solver process runs
(measured trap: concurrent load makes frame times bimodal).  Exit 1 on any
image difference or failed run.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

SCENARIO_NAMES = ["deferred", "forward-plus", "post-chain", "async-compute"]


def busy():
    for name in ("phosphor", "soc_bench", "graph_solver"):
        if subprocess.run(["pgrep", "-x", name], capture_output=True).returncode == 0:
            return name
    return None


def run(app, args, report, capture, log):
    cmd = [app] + args + ["--frames", str(ARGS.frames), "--warmup", "120", "--no-ui", "--no-vsync",
                          "--report", report, "--capture", capture]
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"run failed ({rc}): {' '.join(cmd)} (log {log})")
    return json.load(open(report))


def image_diff(a, b):
    out = subprocess.run([ARGS.image_diff, a, b], capture_output=True, text=True)
    line = (out.stdout + out.stderr).strip().splitlines()[-1] if (out.stdout + out.stderr).strip() else "?"
    try:
        return int(line.split(" of ")[0]), line
    except ValueError:
        return -1, line


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--app", default="build/release/phosphor")
    p.add_argument("--image-diff", default="build/release/image_diff")
    p.add_argument("--candidates", required=True, action="append",
                   help="directory with plans_<i>.json (graph_opt --top); repeatable")
    p.add_argument("--scenarios", default="0,1,2,3")
    p.add_argument("--views", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--out", default=None)
    p.add_argument("--report", default=None)
    p.add_argument("--work", default="build/graph-select")
    ARGS = p.parse_args()
    other = busy()
    if other:
        print(f"graph_select: '{other}' is running: measurements would be bimodal; stop it first", file=sys.stderr)
        return 2
    os.makedirs(ARGS.work, exist_ok=True)
    files = []
    for d in ARGS.candidates:
        files += sorted(glob.glob(os.path.join(d, "plans_*.json")), key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
    candidates = {}  # family -> [(file, plan)], distinct plans only
    def same(a, b):
        return all(a[k] == b[k] for k in ("order", "remat", "async", "alias", "barriers"))
    for f in files:
        for plan in json.load(open(f))["plans"]:
            lst = candidates.setdefault(plan["family"], [])
            if not any(same(plan, q) for _, q in lst):
                lst.append((f, plan))

    failures = 0
    winners, rows = [], []
    for s in [int(x) for x in ARGS.scenarios.split(",")]:
        family = next((fam for fam in candidates if fam.startswith(f"scenario:{SCENARIO_NAMES[s]}:")
                       and f":v{ARGS.views}:" in fam), None)
        if family is None:
            print(f"graph_select: no candidates for scenario {s} views {ARGS.views}", file=sys.stderr)
            failures += 1
            continue
        variants = [("off", None, None)] + [(f"cand{i}", f, plan) for i, (f, plan) in enumerate(candidates[family])]
        spans = {name: [] for name, _, _ in variants}
        reports = {}
        diffs = {name: 0 for name, _, _ in variants}
        for r in range(ARGS.rounds):
            order = variants[r % len(variants):] + variants[:r % len(variants)]
            captures = {}
            for name, f, plan in order:
                tag = f"s{s}v{ARGS.views}-{name}-r{r}"
                args = ["--graph-scenario", str(s), "--graph-scenario-views", str(ARGS.views)]
                args += ["--graph-opt", "off"] if f is None else ["--graph-opt", "plan", "--graph-plan", f]
                rep = run(ARGS.app, args, f"{ARGS.work}/{tag}.json", f"{ARGS.work}/{tag}.png", f"{ARGS.work}/{tag}.log")
                if f is not None and rep.get("graph", {}).get("plan") != "applied":
                    print(f"graph_select: {tag}: plan not applied ({rep.get('graph', {}).get('plan')})", file=sys.stderr)
                    failures += 1
                spans[name].append(rep["gpu_frame_span_ms"]["p50"])
                reports[name] = rep
                captures[name] = f"{ARGS.work}/{tag}.png"
            for name in captures:
                if name == "off":
                    continue
                n, line = image_diff(captures["off"], captures[name])
                diffs[name] = max(diffs[name], n if n >= 0 else 1)
                if n != 0:
                    print(f"graph_select: s{s} {name} round {r}: {line}", file=sys.stderr)
                    failures += 1
        base = statistics.median(spans["off"])
        med = {name: statistics.median(spans[name]) for name, _, _ in variants}
        size = {name: reports[name].get("graph", {}).get("dram_bytes", 0) + reports[name].get("graph", {}).get("heap_bytes", 0)
                for name, _, _ in variants}
        clean = [name for name, _, _ in variants[1:] if diffs[name] == 0]
        # 1. a measured time gain beyond the noise floor wins (best time);
        # 2. else, among candidates not slower than off beyond the noise,
        #    the one that cuts DRAM bytes + heap the most (the OPT-1 goal);
        # 3. else the baseline plan.
        faster = [n for n in clean if med[n] < base * (1 - ARGS.min_gain)]
        reason = "no gain"
        if faster:
            best_name = min(faster, key=lambda n: med[n])
            reason = "faster"
        else:
            neutral = [n for n in clean if med[n] <= base * (1 + ARGS.min_gain) and size[n] < size["off"] * 0.97]
            best_name = min(neutral, key=lambda n: (size[n], med[n])) if neutral else "off"
            reason = "fewer bytes/heap, time neutral" if neutral else "no gain"
        adopted = best_name != "off"
        chosen = None
        if adopted:
            chosen = next(plan for name, _, plan in variants if name == best_name)
        else:
            # The baseline candidate: greedy order, end-of-F4 policies, default choices.
            for name, _, plan in variants[1:]:
                if plan["method"] == "greedy" and plan["alias"] == "greedy" and plan["barriers"] == "conservative" \
                        and plan["predicted"]["time_ms"] == plan["baseline"]["time_ms"]:
                    chosen = plan
                    break
        if chosen is not None:
            chosen = dict(chosen)
            chosen["method"] = chosen["method"] + f" (measured: {reason})"
            winners.append(chosen)
        for name, f, plan in variants:
            g = reports[name].get("graph", {})
            v = spans[name]
            cv = statistics.pstdev(v) / statistics.mean(v) * 100 if len(v) > 1 else 0.0
            rows.append({
                "scenario": s, "variant": name,
                "choices": "-" if plan is None else f"remat={','.join(plan['remat']) or '-'} async={','.join(plan['async']) or '-'}",
                "policies": "off" if plan is None else f"{plan['alias']}/{plan['barriers']}",
                "predicted": None if plan is None else plan["predicted"]["time_ms"],
                "span": statistics.median(v), "cv": cv, "delta": (statistics.median(v) / base - 1) * 100,
                "dram": g.get("dram_bytes", 0) / 2**20, "heap": g.get("heap_bytes", 0) / 2**20,
                "maxlive": g.get("max_live_bytes", 0) / 2**20, "rp": g.get("render_passes", 0),
                "ml": g.get("memoryless", 0), "barriers": g.get("barriers", 0), "diff": diffs[name],
                "chosen": (name == best_name and adopted) or (name == "off" and not adopted)})

    lines = ["| scen | variant | choices | policies | pred ms | frame p50 ms | CV % | vs off | DRAM MiB | heap MiB | max live MiB | RP | ML | barriers | diff px | adopted |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        pred = "-" if r["predicted"] is None else f"{r['predicted']:.3f}"
        lines.append(f"| {r['scenario']} | {r['variant']} | {r['choices']} | {r['policies']} | {pred} | {r['span']:.4f} | "
                     f"{r['cv']:.2f} | {r['delta']:+.2f}% | {r['dram']:.1f} | {r['heap']:.1f} | {r['maxlive']:.1f} | "
                     f"{r['rp']} | {r['ml']} | {r['barriers']} | {'**' + str(r['diff']) + '**' if r['diff'] else 0} | "
                     f"{'**yes**' if r['chosen'] else ''} |")
    table = "\n".join(lines) + "\n"
    print(table)
    if ARGS.report:
        open(ARGS.report, "w").write(table)
    if ARGS.out:
        existing = {"schema": 1, "plans": []}
        if os.path.exists(ARGS.out):
            existing = json.load(open(ARGS.out))
        keep = [p for p in existing["plans"] if p["family"] not in {w["family"] for w in winners}]
        json.dump({"schema": 1, "plans": keep + winners}, open(ARGS.out, "w"), indent=2, sort_keys=True)
        open(ARGS.out, "a").write("\n")
        print(f"graph_select: {len(winners)} plan(s) written to {ARGS.out}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
