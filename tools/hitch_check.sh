#!/usr/bin/env bash
# hitch_check.sh -- check that switching benches causes no frame hitch (F3 exit
# criterion): runs the app with a bench switch every 20 frames, records the
# frame trace and prints the app's SWITCH and PIPELINES report lines.
#
#   tools/hitch_check.sh [build-dir] [frames] [extra app args...]
#
# Defaults: build/release, 300 frames.  Trace CSV: <build-dir>/hitch-trace.csv.
# Exits 1 if the SWITCH line reports "hitches N" with N > 0.  Extra arguments
# (also via $EXTRA_ARGS) are passed to the app, e.g. --debug-split-encoding.
# Keep the window visible while it runs: macOS throttles occluded windows.
# The first 60 frames are a warm-up (not measured): while the window appears
# SDL/Cocoa event pumping costs up to ~15 ms per frame, and a switch at frame
# 20 was flagged as a hitch on main and on every later build alike (F4,
# measured with Tracy: the time is in "Events", not in the switch).
set -euo pipefail

build_dir=${1:-build/release}
frames=${2:-300}
shift $(( $# < 2 ? $# : 2 ))
app="$build_dir/phosphor"
csv="$build_dir/hitch-trace.csv"
log=$(mktemp)
trap 'rm -f "$log"' EXIT

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DCMAKE_BUILD_TYPE=Release)" >&2; exit 1; }

# shellcheck disable=SC2086 # EXTRA_ARGS is a whitespace-separated list
env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
    "$app" --frames "$frames" --warmup 60 --switch-every 20 --no-vsync --no-ui \
    --frame-trace "$csv" ${EXTRA_ARGS:-} "$@" >"$log" 2>&1 || {
    cat "$log" >&2
    echo "error: $app failed" >&2
    exit 1
}

grep -E '^(SWITCH|PIPELINES) ' "$log" || true

switch_line=$(grep -E '^SWITCH ' "$log" | tail -n 1 || true)
[[ -n "$switch_line" ]] || { echo "error: no SWITCH line in the app output" >&2; exit 1; }
hitches=$(sed -nE 's/.*hitches ([0-9]+).*/\1/p' <<<"$switch_line")
[[ -n "$hitches" ]] || { echo "error: cannot parse hitches from: $switch_line" >&2; exit 1; }

if (( hitches > 0 )); then
    echo "FAIL: $hitches switch(es) with hitches (trace: $csv)" >&2
    exit 1
fi
echo "OK: no hitches (trace: $csv)"
