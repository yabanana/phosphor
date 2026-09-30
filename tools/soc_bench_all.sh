#!/usr/bin/env bash
# soc_bench_all.sh -- OPT-0 exit battery of the SoC suite (bench/soc):
#   1. full suite x 3 runs -> bench/results/<chip>-<os>.json (--window: B-26 too)
#   2. --validate (quick suite under API + shader validation): 0 messages
#   3. --force-family apple9 (quick): the Apple9 fallback paths run
#   4. leaks --atExit on the quick suite: 0 leaks
# and prints, from the results JSON, every benchmark whose negative control
# failed and every metric with a CV between runs above 2%.
#
#   tools/soc_bench_all.sh [build-dir] [extra soc_bench args...]
#
# Exit status: 0 only if every step passed.  Run on a quiet machine (other
# GPU clients change the numbers, and B-27's idle phase fails its control).
set -uo pipefail

build_dir=${1:-build/release}
shift || true
bench="$build_dir/soc_bench"
[[ -x "$bench" ]] || { echo "error: build $bench first (cmake --build $build_dir --target soc_bench)" >&2; exit 2; }
log_dir="$build_dir/soc-bench-all"
mkdir -p "$log_dir"
status=0

step() { echo "== $*"; }

# Keep the display on: with the display asleep the window benchmarks
# present nothing (B-26), the idle GPU draws ~0 W (B-27) and the 2 GiB
# pointer chase read 1006 ns instead of ~475 (B-08), all observed when the
# display went off in the middle of a battery.  caffeinate needs no root.
keep_awake=(caffeinate -d -i -m -s)

step "full suite x3 (window benchmarks included)"
"${keep_awake[@]}" "$bench" --runs 3 --window "$@" 2>&1 | tee "$log_dir/full.log"
full_rc=${PIPESTATUS[0]}
results=$(sed -n 's/^\[soc\] results: //p' "$log_dir/full.log" | tail -1)
[[ $full_rc -eq 0 ]] || { echo "FAIL: full suite exit $full_rc"; status=1; }

step "--validate"
"${keep_awake[@]}" "$bench" --validate --window 2>&1 | tee "$log_dir/validate.log" | tail -3
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "FAIL: validation"; status=1; }

step "--force-family apple9 (quick)"
"${keep_awake[@]}" "$bench" --quick --runs 1 --force-family apple9 --out "$log_dir/apple9.json" 2>&1 | tee "$log_dir/apple9.log" | tail -3
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "FAIL: apple9 fallback run"; status=1; }

step "leaks (quick)"
leaks --atExit -- "$bench" --quick --runs 1 --out "$log_dir/leaks.json" > "$log_dir/leaks.log" 2>&1
grep -E "leaks for [0-9]+ total leaked bytes|Process [0-9]+: [0-9]+ leaks" "$log_dir/leaks.log" | tail -2
grep -qE ": 0 leaks for 0 total leaked bytes" "$log_dir/leaks.log" || { echo "FAIL: leaks"; status=1; }

if [[ -n "$results" && -f "$results" ]]; then
    step "controls and CV ($results)"
    python3 - "$results" <<'EOF' || status=1
import json, sys
d = json.load(open(sys.argv[1]))
bad = 0
for b in d["benchmarks"]:
    neg = b["negative_control"]["status"]
    if neg == "fail" or b["status"] == "failed":
        bad += 1
        print(f"  {b['id']} {b['status']} negative={neg}: {b['negative_control']['detail']} {b['notes']}")
    elif neg == "n/a":
        print(f"  {b['id']}: negative control n/a ({b['status']}) {b['notes'][:120]}")
    high = [f"{m['name']} {m['run_cv']*100:.1f}%" for m in b["metrics"] if m["run_cv"] > 0.02]
    if high:
        print(f"  {b['id']} CV>2%: " + ", ".join(high[:12]) + (" ..." if len(high) > 12 else ""))
print(f"{len(d['benchmarks'])} benchmarks, {bad} failed")
sys.exit(1 if bad else 0)
EOF
fi
echo "soc_bench_all: $([[ $status -eq 0 ]] && echo PASS || echo FAIL)"
exit $status
