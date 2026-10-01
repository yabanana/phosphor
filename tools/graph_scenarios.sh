#!/usr/bin/env bash
# graph_scenarios.sh -- OPT-1 measurement harness (macOS): graph scenarios x
# graph-compilation modes, rotating rounds, pixel check, validation check.
#
#   tools/graph_scenarios.sh [options] [release-dir] [debug-dir]
#
# Defaults: release-dir build/release, debug-dir build.  For every scenario,
# view count and mode it runs, N rounds, in ROTATING order of the modes (round
# r starts with mode r mod M, so a slow start or thermal drift does not always
# hit the same mode):
#   phosphor --graph-scenario S [--graph-scenario-views V] --graph-opt MODE
#            [--graph-plan FILE] --frames F --warmup W --no-ui --no-vsync
#            --report R --capture C
# then
#   * image_diff of every mode's capture against the base mode (off) of the
#     same round: must be 0 pixels (printed, and counted as a failure);
#   * one run per scenario, views and mode under the Metal debug layer and
#     shader validation (debug-dir binary, MTL_DEBUG_LAYER=1
#     MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog): the
#     non-INFO lines of its log are counted (must be 0);
#   * tools/scenario_table.py summary (GPU frame span p50 = median of rounds,
#     CV between rounds, DRAM / heap / max live MiB, render passes,
#     memoryless, barriers, plan status, deltas vs off, diff and validation).
#
# Options:
#   --scenarios "0 1 2 3"   scenarios (default 0 1 2 3)
#   --views "1 2"           view counts (default 1; one family per value)
#   --modes "off greedy plan"  (default; `off` is the reference and is added
#                           when missing)
#   --rounds N              measured rounds (default 3)
#   --frames N              measured frames per run (default 600)
#   --warmup N              warm-up frames (default 120)
#   --plan FILE             plans file for mode plan (default: generated with
#                           <release-dir>/graph_opt into <out>/plans.json)
#   --out DIR               results (default <release-dir>/graph-results)
#   --no-validation         skip the validation runs
#
# Exit status: 0 all checks passed, 1 a pixel diff / validation line / run
# failure, 2 refused to start (another phosphor, soc_bench or graph_solver
# process is running: concurrent GPU/CPU load makes frame times bimodal).
# The whole run is wrapped in `caffeinate -d` (display stays on).  Keep the
# window visible.  Bash 3.2 compatible.
set -uo pipefail

scenarios="0 1 2 3"
views="1"
modes="off greedy plan"
rounds=3
frames=600
warmup=120
plan_file=""
out_dir=""
validate=1
orig_args=("$@")
pos=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenarios) scenarios=$2; shift 2 ;;
        --views) views=$2; shift 2 ;;
        --modes) modes=$2; shift 2 ;;
        --rounds) rounds=$2; shift 2 ;;
        --frames) frames=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --plan) plan_file=$2; shift 2 ;;
        --out) out_dir=$2; shift 2 ;;
        --no-validation) validate=0; shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        --*) echo "error: unknown option $1" >&2; exit 2 ;;
        *) pos+=("$1"); shift ;;
    esac
done
release_dir=${pos[0]:-build/release}
debug_dir=${pos[1]:-build}
out_dir=${out_dir:-$release_dir/graph-results}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Refuse to start beside other GPU/CPU heavy processes (exact process names).
busy=""
for name in phosphor soc_bench graph_solver; do
    if pgrep -x "$name" >/dev/null 2>&1; then busy="$busy $name"; fi
done
if [[ -n "$busy" ]]; then
    echo "error: refusing to start, running:$busy (concurrent load makes frame times bimodal)" >&2
    exit 2
fi

# Re-exec under caffeinate -d (keeps the display on).
if [[ -z "${GRAPH_SCENARIOS_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
    export GRAPH_SCENARIOS_CAFFEINATED=1
    exec caffeinate -d "$0" "${orig_args[@]}"
fi

app="$release_dir/phosphor"
diff_tool="$release_dir/image_diff"
opt_tool="$release_dir/graph_opt"
[[ -x "$app" ]] || { echo "error: $app not found" >&2; exit 2; }
[[ -x "$diff_tool" ]] || { echo "error: $diff_tool not found" >&2; exit 2; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 2; }
case " $modes " in *" off "*) ;; *) modes="off $modes" ;; esac
mode_arr=($modes)
nmodes=${#mode_arr[@]}
if [[ $validate == 1 && ! -x "$debug_dir/phosphor" ]]; then
    echo "warning: $debug_dir/phosphor not found: validation runs skipped" >&2
    validate=0
fi

mkdir -p "$out_dir"
rm -f "$out_dir"/s*-v*-*-r*.json "$out_dir"/s*-v*-*-r*.png "$out_dir/diffs.tsv" "$out_dir/validation.tsv"
: > "$out_dir/diffs.tsv"
: > "$out_dir/validation.tsv"
failures=0

case " $modes " in
    *" plan "*)
        if [[ -z "$plan_file" ]]; then
            plan_file="$out_dir/plans.json"
            [[ -x "$opt_tool" ]] || { echo "error: $opt_tool not found (needed to generate plans)" >&2; exit 2; }
            rm -f "$plan_file"
            for s in $scenarios; do
                for v in $views; do
                    "$opt_tool" --scenario "$s" --views "$v" --out "$plan_file" --merge >/dev/null ||
                        { echo "error: graph_opt failed for scenario $s views $v" >&2; exit 1; }
                done
            done
        fi
        ;;
esac

# run_one MODE SCENARIO VIEWS TAG FRAMES WARMUP -> log in $out_dir/$TAG.log
run_one() {
    local mode=$1 s=$2 v=$3 tag=$4 nframes=$5 nwarm=$6
    local args=(--graph-scenario "$s" --graph-scenario-views "$v" --graph-opt "$mode")
    [[ $mode == plan ]] && args+=(--graph-plan "$plan_file")
    env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
        "$app" "${args[@]}" --frames "$nframes" --warmup "$nwarm" --no-ui --no-vsync \
        --report "$out_dir/$tag.json" --capture "$out_dir/$tag.png" >"$out_dir/$tag.log" 2>&1
}

for round in $(seq 1 "$rounds"); do
    for s in $scenarios; do
        for v in $views; do
            for k in $(seq 0 $((nmodes - 1))); do
                mode=${mode_arr[$(( (k + round - 1) % nmodes ))]}
                tag="s$s-v$v-$mode-r$round"
                echo "round $round/$rounds scenario $s views $v mode $mode"
                if ! run_one "$mode" "$s" "$v" "$tag" "$frames" "$warmup"; then
                    echo "FAIL: run $tag exited non-zero (see $out_dir/$tag.log)" >&2
                    failures=$((failures + 1))
                fi
            done
            # Pixel check of this round against its own `off` capture.
            for mode in "${mode_arr[@]}"; do
                [[ $mode == off ]] && continue
                ref="$out_dir/s$s-v$v-off-r$round.png"
                cand="$out_dir/s$s-v$v-$mode-r$round.png"
                pixels=-1
                if [[ -f $ref && -f $cand ]]; then
                    res=$("$diff_tool" "$ref" "$cand" 2>&1) || true
                    n=$(sed -nE 's/^([0-9]+) of [0-9]+ pixels differ.*/\1/p' <<<"$res")
                    [[ -n $n ]] && pixels=$n
                else
                    res="capture missing"
                fi
                echo "  pixel diff s$s v$v $mode vs off (round $round): $res"
                if [[ $pixels != 0 ]]; then failures=$((failures + 1)); fi
                # the table takes the worst round
                prev=$(awk -F'\t' -v s="$s" -v v="$v" -v m="$mode" '$1==s&&$2==v&&$3==m{print $4}' "$out_dir/diffs.tsv")
                if [[ -z $prev ]]; then
                    printf '%s\t%s\t%s\t%s\n' "$s" "$v" "$mode" "$pixels" >>"$out_dir/diffs.tsv"
                elif [[ $pixels != 0 && $prev == 0 ]]; then
                    printf '%s\t%s\t%s\t%s\n' "$s" "$v" "$mode" "$pixels" >>"$out_dir/diffs.tsv"
                fi
            done
        done
    done
done

# Validation: one run per scenario, views and mode with the debug build.
ignore='^\[INFO\]|^BENCH|^PIPELINES|^STARTUP|^SWITCH|^GRAPH-TRANSIENTS|^ASYNC-COMPUTE|Validation Enabled'
if [[ $validate == 1 ]]; then
    for s in $scenarios; do
        for v in $views; do
            for mode in "${mode_arr[@]}"; do
                tag="s$s-v$v-$mode-validation"
                args=(--graph-scenario "$s" --graph-scenario-views "$v" --graph-opt "$mode")
                [[ $mode == plan ]] && args+=(--graph-plan "$plan_file")
                status=0
                MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog \
                    "$debug_dir/phosphor" "${args[@]}" --frames 30 --warmup 5 --no-ui --no-vsync \
                    --report "$out_dir/$tag.report" >"$out_dir/$tag.log" 2>&1 || status=$?
                lines=$(grep -Ev "$ignore" "$out_dir/$tag.log" | grep -c . || true)
                [[ $status -eq 0 ]] || lines=$((lines + 1))
                echo "validation s$s v$v $mode: $lines non-INFO line(s) (exit $status)"
                [[ $lines -eq 0 ]] || failures=$((failures + 1))
                printf '%s\t%s\t%s\t%s\n' "$s" "$v" "$mode" "$lines" >>"$out_dir/validation.tsv"
            done
        done
    done
fi

echo
python3 "$script_dir/scenario_table.py" "$out_dir" --diffs "$out_dir/diffs.tsv" \
    --validation "$out_dir/validation.tsv" | tee "$out_dir/summary.md"
table_status=${PIPESTATUS[0]}
[[ $table_status -eq 0 ]] || failures=$((failures + 1))

if [[ $failures -ne 0 ]]; then
    echo "graph_scenarios: $failures failure(s)" >&2
    exit 1
fi
echo "graph_scenarios: all checks passed (results in $out_dir)"
