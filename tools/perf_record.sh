#!/usr/bin/env bash
# perf_record.sh -- run every bench N times in BOTH vsync configurations and
# append the results to the machine-readable perf history (F4.5/F4.6).
#
#   tools/perf_record.sh [build-dir] [frames] [runs]
#
# Defaults: build/release, 600 measured frames, 3 runs per bench and config.
# Always --no-ui and no Metal validation.  Two configurations per bench:
# vsync=0 (--no-vsync: throughput, frames in flight overlap) and vsync=1
# (default: the GPU ms columns are only meaningful here).
#
# Output (rows are APPENDED; the header is written when a file is missing):
#   docs/perf-history.csv         one row per bench x vsync: date (UTC ISO),
#       commit, dirty, machine (hw.model), chip, macOS, power (AC/battery),
#       bench id/name, resolution, vsync, ui, frames, runs, and for
#       frame / cpu / gpu / gpupass: mean of the per-run means, sample sd across
#       runs, p99 of the median run (median = run with the median mean frame
#       time); plus max GPU allocations and max CPU heap bytes/blocks delta.
#       Empty cells = not reported (schema v1 binaries have no gpupass).
#   docs/perf-history-passes.csv  one row per bench x vsync x unit: gpu mean
#       (mean over runs), sd across runs, p99 (median run), max (over runs),
#       dram_bytes.  Empty for schema v1 binaries.
# Raw JSON reports: <build-dir>/perf-record/<commit>/bench<N>-<vsync|novsync>-run<R>.json
#
# Read the history with tools/perf_table.py (latest / compare / passes).
# Warns when the binary is older than HEAD's commit (set STRICT=1 to refuse).
# Do not commit rows recorded from a dirty tree or a stale binary.  Keep the
# window visible while it runs.  Works with macOS default bash 3.2.
set -euo pipefail

build_dir=${1:-build/release}
frames=${2:-600}
runs=${3:-3}
app="$build_dir/phosphor"
root=$(git rev-parse --show-toplevel)
hist="$root/docs/perf-history.csv"
hist_passes="$root/docs/perf-history-passes.csv"

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DCMAKE_BUILD_TYPE=Release)" >&2; exit 1; }
command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 1; }

commit=$(git rev-parse --short HEAD)
dirty=0
[[ -n "$(git status --porcelain --untracked-files=no)" ]] && dirty=1
commit_ts=$(git log -1 --format=%ct)
bin_ts=$(stat -f %m "$app")
if (( bin_ts < commit_ts )); then
    echo "warning: $app is older than HEAD ($commit):" >&2
    echo "  binary mtime $(date -u -r "$bin_ts" +%FT%TZ), commit time $(date -u -r "$commit_ts" +%FT%TZ)" >&2
    [[ "${STRICT:-0}" == 1 ]] && { echo "error: STRICT=1, refusing to record a stale binary" >&2; exit 1; }
fi
[[ "$dirty" == 1 ]] && echo "warning: working tree is dirty (recorded with dirty=1)" >&2

date_iso=$(date -u +%FT%TZ)
machine=$(sysctl -n hw.model)
chip=$(sysctl -n machdep.cpu.brand_string)
macos=$(sw_vers -productVersion)
if pmset -g batt | head -1 | grep -q 'AC Power'; then power=AC; else power=battery; fi

hist_header='date,commit,dirty,machine,chip,macos,power,bench,name,resolution,vsync,ui,frames,runs,frame_mean,frame_sd,frame_p99,cpu_mean,cpu_sd,cpu_p99,gpu_mean,gpu_sd,gpu_p99,gpupass_mean,gpupass_sd,gpupass_p99,gpu_allocations_max,cpu_heap_bytes_delta_max,cpu_heap_blocks_delta_max'
passes_header='date,commit,bench,vsync,unit,queue,fused,gpu_mean,gpu_sd,gpu_p99,gpu_max,dram_bytes'
[[ -f "$hist" ]] || echo "$hist_header" > "$hist"
[[ -f "$hist_passes" ]] || echo "$passes_header" > "$hist_passes"

out_dir="$build_dir/perf-record/$commit"
mkdir -p "$out_dir"

# shellcheck disable=SC2016  # jq programs, not shell expansions
jq_defs='
def r4: (. * 10000 | round) / 10000;
def sd: if length < 2 then 0 else (add / length) as $m
        | (map((. - $m) * (. - $m)) | add / (length - 1) | sqrt) end;
def mean: if length == 0 then null else add / length end;
def med: sort_by(.frame_ms.mean) | .[((length + 1) / 2 | floor) - 1];
# mean of means, sd across runs, p99 of the median run; null when absent.
def trio($f; $m): [.[] | getpath($f + ["mean"]) | select(. != null)] as $v
    | if ($v | length) == 0 then [null, null, null]
      else [($v | mean | r4), ($v | sd | r4), ($m | getpath($f + ["p99"]) | r4)] end;
'

for bench in 1 2 3 4 5 6 7 8; do
    for cfg in vsync novsync; do
        vs_args=()
        vs=1
        if [[ "$cfg" == novsync ]]; then vs_args=(--no-vsync); vs=0; fi
        reports=()
        for run in $(seq 1 "$runs"); do
            report="$out_dir/bench$bench-$cfg-run$run.json"
            env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
                "$app" --bench "$bench" --frames "$frames" ${vs_args[@]+"${vs_args[@]}"} --no-ui --report "$report" >/dev/null 2>&1
            reports+=("$report")
        done
        jq -s -r --arg date "$date_iso" --arg commit "$commit" --argjson dirty "$dirty" \
            --arg machine "$machine" --arg chip "$chip" --arg macos "$macos" --arg power "$power" \
            --argjson bench "$bench" --argjson vs "$vs" --argjson runs "$runs" "$jq_defs"'
            med as $m
            | trio(["frame_ms"]; $m) as $f | trio(["cpu_ms"]; $m) as $c
            | trio(["gpu_ms"]; $m) as $g | trio(["gpu_pass_sum_ms"]; $m) as $p
            | [$date, $commit, $dirty, $machine, $chip, $macos, $power, $bench, $m.bench,
               "\($m.width)x\($m.height)", $vs, (if $m.ui then 1 else 0 end), $m.frames, $runs]
              + $f + $c + $g + $p
              + [([.[].gpu_allocations // 0] | max),
                 ([.[].cpu_heap_bytes_delta // 0] | max),
                 ([.[].cpu_heap_blocks_delta // 0] | max)]
            | @csv' "${reports[@]}" >> "$hist"
        jq -s -r --arg date "$date_iso" --arg commit "$commit" \
            --argjson bench "$bench" --argjson vs "$vs" "$jq_defs"'
            med as $m
            | ($m.passes // [])[] as $p
            | [ .[] | (.passes // [])[] | select(.name == $p.name) | .gpu_ms ] as $g
            | [$date, $commit, $bench, $vs, $p.name, ($p.queue // ""), (if $p.fused then 1 else 0 end),
               ([$g[].mean] | mean | r4), ([$g[].mean] | sd | r4), ($p.gpu_ms.p99 | r4),
               ([$g[].max] | max | r4), ($p.dram_bytes // null)]
            | @csv' "${reports[@]}" >> "$hist_passes"
        echo "recorded bench $bench ($cfg, $runs runs)"
    done
done
echo "appended to $hist and $hist_passes (commit $commit, dirty=$dirty); reports in $out_dir"
