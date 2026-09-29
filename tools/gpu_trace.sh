#!/usr/bin/env bash
# gpu_trace.sh -- per-pass GPU breakdown from a headless Metal System Trace
# (F4.4, partial by design: no hardware counters are available headless; see
# docs/opt-log.md, section "F4", point 3).
#
#   tools/gpu_trace.sh <build-dir> <bench 1..N> [seconds] [extra app args...]
#
# Records `xcrun xctrace record --template 'Metal System Trace'` around
# <build-dir>/phosphor (default 8 s: the Shader Timeline can be empty in 4 s traces of
# slow-starting benches; xctrace adds ~20-40 s of wall time),
# exports only the tables the parser needs and runs tools/xctrace_passes.py.
# Output: <build-dir>/gpu-trace/<timestamp>-benchN/{run.trace,report.json,
# *.xml,passes.md,passes.json}.
#
# The app is staged into a temp dir first: a process launched by xctrace from
# under ~/Documents hangs forever in dyld's open() (verified with `sample`).
# Do not add --target-stdout to the app run: with it xctrace records no Shader
# Timeline rows (verified).  Extra app args must not be relative paths (the
# app runs from a temporary cwd).
#
# Environment:
#   FRAMES             frames requested from the app (default seconds*500, sized
#                      to outlast the time limit; when xctrace stops the app no
#                      report is written and the built-in engine map is used)
#   TEMPLATE           xctrace template (default 'Metal System Trace')
#   XCTRACE_INSTRUMENT extra instrument (default 'Metal GPU Counters', which is what
#                      enables the Shader Timeline; set to an empty string to drop it)
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <build-dir> <bench> [seconds] [extra app args...]" >&2
    exit 2
fi
build_dir=$(cd "$1" && pwd)
bench=$2
seconds=${3:-8}
shift $(( $# < 3 ? $# : 3 ))
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
app="$build_dir/phosphor"
template=${TEMPLATE:-Metal System Trace}
instrument_name=${XCTRACE_INSTRUMENT-Metal GPU Counters}
frames=${FRAMES:-$(( seconds * 500 ))}

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DCMAKE_BUILD_TYPE=Release)" >&2; exit 1; }
command -v xcrun >/dev/null || { echo "error: xcrun not found (macOS with Xcode required)" >&2; exit 1; }

out="$build_dir/gpu-trace/$(date +%Y%m%d-%H%M%S)-bench$bench"
mkdir -p "$out"
trace="$out/run.trace"

# A process launched by xctrace blocks forever in dyld's open() when its binary
# or cwd lives under ~/Documents (privacy protection, no prompt is shown), so
# the app is staged into a temporary directory and run from there.
stage=$(mktemp -d "${TMPDIR:-/tmp}/phosphor-gpu-trace.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp "$app" "$stage/phosphor"
[[ -d "$build_dir/shaders" ]] && cp -R "$build_dir/shaders" "$stage/shaders"
[[ -d "$build_dir/assets" ]] && cp -RL "$build_dir/assets" "$stage/assets"

instrument=()
[[ -n "$instrument_name" ]] && instrument=(--instrument "$instrument_name")

echo "recording $seconds s of bench $bench into $trace (xctrace takes ~40 s wall)..." >&2
# xctrace exits non-zero when the launched app is stopped at the time limit.
# cwd = stage: the launched process inherits it (see above).
cd "$stage"
env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
    xcrun xctrace record --template "$template" "${instrument[@]}" \
    --time-limit "${seconds}s" --output "$trace" --no-prompt \
    --launch -- "$stage/phosphor" --bench "$bench" --frames "$frames" --warmup 30 \
    --no-vsync --report "$stage/report.json" "$@" \
    >"$out/record.log" 2>&1 || echo "note: xctrace record exited non-zero (see $out/record.log)" >&2
[[ -f "$stage/report.json" ]] && cp "$stage/report.json" "$out/report.json"
[[ -d "$trace" ]] || { cat "$out/record.log" >&2; echo "error: no trace produced" >&2; exit 1; }

for schema in metal-gpu-intervals metal-shader-profiler-intervals gpu-performance-state-intervals; do
    echo "exporting $schema..." >&2
    xcrun xctrace export --input "$trace" \
        --xpath "/trace-toc/run[@number=\"1\"]/data/table[@schema=\"$schema\"]" \
        --output "$out/$schema.xml" >>"$out/record.log" 2>&1 \
        || echo "warning: export of $schema failed (see $out/record.log)" >&2
done

# Other phosphor processes may run on the machine: select ours by pid (the
# launched process is the first <process> element of the table of contents).
pid=$(xcrun xctrace export --input "$trace" --toc 2>/dev/null \
    | sed -n 's/.*<process [^>]*type="launched"[^>]*pid="\([0-9]*\)".*/\1/p' | head -1)
pid_args=()
if [[ -n $pid ]]; then
    pid_args=(--pid "$pid")
    echo "launched pid: $pid" >&2
fi

report_args=()
[[ -f "$out/report.json" ]] && report_args=(--report "$out/report.json")
python3 "$here/xctrace_passes.py" --dir "$out" --process phosphor \
    "${pid_args[@]}" "${report_args[@]}" --out-md "$out/passes.md" --out-json "$out/passes.json"
echo "tables: $out/passes.md, $out/passes.json" >&2
