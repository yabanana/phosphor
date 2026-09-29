#!/usr/bin/env bash
# gpu_trace.sh -- per-pass GPU breakdown from a headless Metal System Trace
# (F4.4, partial by design: no hardware counters are available headless; see
# docs/opt-log.md, section "F4", point 3).
#
#   tools/gpu_trace.sh <build-dir> <bench 1..N> [seconds] [extra app args...]
#
# Records `xcrun xctrace record --template 'Metal System Trace'` around
# <build-dir>/phosphor (default 4 s; xctrace itself needs ~40 s of wall time),
# exports only the tables the parser needs and runs tools/xctrace_passes.py.
# Output: <build-dir>/gpu-trace/<timestamp>-benchN/{run.trace,report.json,
# *.xml,passes.md,passes.json}.
#
# Environment:
#   FRAMES             frames requested from the app (default seconds*500, sized
#                      to outlast the time limit; when xctrace stops the app no
#                      report is written and the built-in engine map is used)
#   TEMPLATE           xctrace template (default 'Metal System Trace')
#   XCTRACE_INSTRUMENT one extra instrument name, e.g. 'Metal GPU Counters'
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <build-dir> <bench> [seconds] [extra app args...]" >&2
    exit 2
fi
build_dir=$1
bench=$2
seconds=${3:-4}
shift $(( $# < 3 ? $# : 3 ))
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
app="$build_dir/phosphor"
template=${TEMPLATE:-Metal System Trace}
frames=${FRAMES:-$(( seconds * 500 ))}

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DCMAKE_BUILD_TYPE=Release)" >&2; exit 1; }
command -v xcrun >/dev/null || { echo "error: xcrun not found (macOS with Xcode required)" >&2; exit 1; }

out="$build_dir/gpu-trace/$(date +%Y%m%d-%H%M%S)-bench$bench"
mkdir -p "$out"
trace="$out/run.trace"

instrument=()
if [[ -n ${XCTRACE_INSTRUMENT:-} ]]; then
    instrument=(--instrument "$XCTRACE_INSTRUMENT")
fi

echo "recording $seconds s of bench $bench into $trace (xctrace takes ~40 s wall)..." >&2
# xctrace exits non-zero when the launched app is stopped at the time limit.
env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
    xcrun xctrace record --template "$template" "${instrument[@]}" \
    --time-limit "${seconds}s" --output "$trace" --no-prompt \
    --launch -- "$app" --bench "$bench" --frames "$frames" --warmup 30 \
    --no-vsync --report "$out/report.json" "$@" \
    >"$out/record.log" 2>&1 || echo "note: xctrace record exited non-zero (see $out/record.log)" >&2
[[ -d "$trace" ]] || { cat "$out/record.log" >&2; echo "error: no trace produced" >&2; exit 1; }

for schema in metal-gpu-intervals metal-shader-profiler-intervals gpu-performance-state-intervals; do
    echo "exporting $schema..." >&2
    xcrun xctrace export --input "$trace" \
        --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" \
        --output "$out/$schema.xml" >>"$out/record.log" 2>&1 \
        || echo "warning: export of $schema failed (see $out/record.log)" >&2
done

report_args=()
[[ -f "$out/report.json" ]] && report_args=(--report "$out/report.json")
python3 "$here/xctrace_passes.py" --dir "$out" --process phosphor \
    "${report_args[@]}" --out-md "$out/passes.md" --out-json "$out/passes.json"
echo "tables: $out/passes.md, $out/passes.json" >&2
