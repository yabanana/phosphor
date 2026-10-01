#!/usr/bin/env bash
# bench_all.sh -- run every test bench in benchmark mode and print one
# Markdown table row per bench (the format of docs/perf-log.md).
#
#   tools/bench_all.sh [options] [build-dir] [frames] [runs]
#
# Defaults: build/release, 600 measured frames, 3 runs per bench (the row
# reports the run with the median mean frame time).  Uses --no-vsync --no-ui
# and never enables Metal validation.  JSON reports are kept in
# <build-dir>/bench-results/.  Keep the window visible while it runs: macOS
# throttles occluded windows.  The default output is the single table below.
#
# Options (each also settable through an environment variable = 1):
#   --stats   (STATS=1)   after the main table print a second table with the
#                         across-run sample standard deviation of the mean and
#                         the coefficient of variation (CV% = sd / mean) for
#                         frame / CPU / GPU / GPU pass-sum.  Needs runs >= 2.
#   --passes  (PASSES=1)  after each bench print a per-pass table from the
#                         median run: unit, queue, fused, GPU mean / p99 / max,
#                         the across-run sd of the pass mean, DRAM MiB.  Needs a
#                         schema v2 report ("passes"); v1 reports print a note.
#   --vsync   (VSYNC=1)   run with vsync on (the default is --no-vsync).  GPU
#                         ms are only meaningful with vsync: without it frames
#                         in flight overlap and the command-buffer interval
#                         includes the overlap.
#
# Missing v2 fields (older binaries write schema v1) are shown as "-".
# Works with the macOS default bash 3.2 (no associative arrays, no mapfile).
set -euo pipefail

stats=${STATS:-0}
passes=${PASSES:-0}
vsync=${VSYNC:-0}
pos=()
for arg in "$@"; do
    case "$arg" in
        --stats)  stats=1 ;;
        --passes) passes=1 ;;
        --vsync)  vsync=1 ;;
        -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
        --*) echo "error: unknown option $arg" >&2; exit 1 ;;
        *) pos+=("$arg") ;;
    esac
done

build_dir=${pos[0]:-build/release}
frames=${pos[1]:-600}
runs=${pos[2]:-3}
app="$build_dir/phosphor"
out_dir="$build_dir/bench-results"

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DCMAKE_BUILD_TYPE=Release)" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 1; }
mkdir -p "$out_dir"

vsync_args=(--no-vsync)
[[ "$vsync" == 1 ]] && vsync_args=()

# jq helpers shared by the extra tables.  Input: the slurped array of the
# reports of one bench, sorted by run.  sd is the sample (n-1) deviation.
# shellcheck disable=SC2016  # jq programs, not shell expansions
jq_defs='
def r3: (. * 1000 | round) / 1000;
def sd: if length < 2 then 0 else (add / length) as $m
        | (map((. - $m) * (. - $m)) | add / (length - 1) | sqrt) end;
def cell: if . == null then "-" else r3 | tostring end;
def cv: (add / length) as $m | if $m == 0 then 0 else (sd / $m * 100) end;
def median: sort_by(.frame_ms.mean) | .[((length + 1) / 2 | floor) - 1];
def stat($f): [.[] | getpath($f) | select(. != null)]
    | if length == 0 then "-"
      else "\((add / length) | r3) ± \(sd | r3) (\(cv | . * 10 | round / 10)%)" end;
'

echo "| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |"
echo "|---|---|---|---|---|---|---|---|"
stats_rows=()
pass_blocks=()
for bench in 1 2 3 4 5 6 7 8; do
    reports=()
    for run in $(seq 1 "$runs"); do
        report="$out_dir/bench$bench-run$run.json"
        env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
            "$app" --bench "$bench" --frames "$frames" ${vsync_args[@]+"${vsync_args[@]}"} --no-ui --report "$report" >/dev/null 2>&1
        reports+=("$report")
    done
    # Median run by mean frame time.
    median=$(for r in "${reports[@]}"; do echo "$(jq '.frame_ms.mean' "$r") $r"; done | sort -n | \
             awk -v n="$runs" 'NR == int((n + 1) / 2) { print $2 }')
    jq -r --arg n "$bench" '"| \($n) | \(.bench) | \(.width)x\(.height) | \(.fps | round) | " +
        "\(.frame_ms.mean * 1000 | round / 1000) (\(.frame_ms.p99 * 1000 | round / 1000)) | " +
        "\(.cpu_ms.mean * 1000 | round / 1000) (\(.cpu_ms.p99 * 1000 | round / 1000)) | " +
        "\(.gpu_ms.mean * 1000 | round / 1000) (\(.gpu_ms.p99 * 1000 | round / 1000)) | " +
        "\(.wait_ms.mean * 1000 | round / 1000) (\(.wait_ms.p99 * 1000 | round / 1000)) |"' "$median"

    if [[ "$stats" == 1 ]]; then
        stats_rows+=("$(jq -s -r --arg n "$bench" "$jq_defs"'
            "| \($n) | \(.[0].bench) | " +
            "\(stat(["frame_ms","mean"])) | \(stat(["cpu_ms","mean"])) | " +
            "\(stat(["gpu_ms","mean"])) | \(stat(["gpu_pass_sum_ms","mean"])) |"' "${reports[@]}")")
    fi
    if [[ "$passes" == 1 ]]; then
        pass_blocks+=("$(jq -s -r --arg n "$bench" "$jq_defs"'
            median as $m
            | "\n#### Bench \($n) — \($m.bench) (passes, median run of \(length))\n",
              (if (($m.passes // []) | length) == 0
               then "_no per-pass data (report schema v1 or GPU timing off)_"
               else
                 "| Unit | Queue | Fused | GPU ms mean | sd (runs) | p99 | max | DRAM MiB |",
                 "|---|---|---|---|---|---|---|---|",
                 ($m.passes[] as $p
                  | [ .[] | (.passes // [])[] | select(.name == $p.name) | .gpu_ms.mean ] as $means
                  | "| \($p.name) | \($p.queue // "-") | \(if $p.fused then "yes" else "no" end) | " +
                    "\($p.gpu_ms.mean | cell) | \($means | sd | cell) | \($p.gpu_ms.p99 | cell) | " +
                    "\($p.gpu_ms.max | cell) | " +
                    "\(if $p.dram_bytes == null then "-" else ($p.dram_bytes / 1048576 | r3 | tostring) end) |")
               end)' "${reports[@]}")")
    fi
done

if [[ "$stats" == 1 ]]; then
    echo
    echo "Across-run mean ± sample sd (CV%) of the per-run means, $runs runs:"
    echo
    echo "| # | Bench | Frame ms | CPU ms | GPU ms | GPU pass-sum ms |"
    echo "|---|---|---|---|---|---|"
    printf '%s\n' "${stats_rows[@]}"
fi
if [[ "$passes" == 1 ]]; then
    printf '%s\n' "${pass_blocks[@]}"
fi
