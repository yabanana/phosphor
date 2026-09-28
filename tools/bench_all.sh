#!/usr/bin/env bash
# bench_all.sh -- run every test bench in benchmark mode and print one
# Markdown table row per bench (the format of docs/perf-log.md).
#
#   tools/bench_all.sh [build-dir] [frames] [runs]
#
# Defaults: build/release, 600 measured frames, 3 runs per bench (the row
# reports the run with the median mean frame time).  Uses --no-vsync --no-ui
# and never enables Metal validation.  JSON reports are kept in
# <build-dir>/bench-results/.  Keep the window visible while it runs: macOS
# throttles occluded windows.
set -euo pipefail

build_dir=${1:-build/release}
frames=${2:-600}
runs=${3:-3}
app="$build_dir/phosphor"
out_dir="$build_dir/bench-results"

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DCMAKE_BUILD_TYPE=Release)" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 1; }
mkdir -p "$out_dir"

echo "| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |"
echo "|---|---|---|---|---|---|---|---|"
for bench in 1 2 3 4 5 6 7; do
    reports=()
    for run in $(seq 1 "$runs"); do
        report="$out_dir/bench$bench-run$run.json"
        env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
            "$app" --bench "$bench" --frames "$frames" --no-vsync --no-ui --report "$report" >/dev/null 2>&1
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
done
