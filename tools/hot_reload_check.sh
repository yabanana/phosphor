#!/usr/bin/env bash
# hot_reload_check.sh -- F3.6 shader hot reload checks (Debug build).
#
#   tools/hot_reload_check.sh [build-dir]
#
# 1. Self-test: --debug-hot-reload swaps in the probe library built by CMake
#    (forward pass = opaque magenta) through the reload path, with every F2
#    debug flag on, under API + shader validation; the app checks the
#    captured frame exactly (HOT-RELOAD line) and exits 1 on failure.
# 2. Negative control: "reloading" the normal library must FAIL the check.
# 3. Real watcher: copies the shaders to a temporary --shader-dir, then while
#    the app runs appends a syntax error (the build must fail and the
#    pipelines stay), then makes forward_fs return magenta (the watcher must
#    rebuild the library and the pipelines must be swapped once).
# Default build dir: build (Debug: hot reload is compiled in Debug only).
set -euo pipefail

build_dir=${1:-build}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
app="$build_dir/phosphor"
probe="$build_dir/shaders/hot-reload-probe.metallib"
out_dir="$build_dir/hot-reload-check"
[[ -x "$app" && -f "$probe" ]] || { echo "error: build $app and $probe first" >&2; exit 2; }
mkdir -p "$out_dir"
failures=0

unexpected() { # log -> count of lines that are not ours / not expected
    grep -Ev '^\[INFO\]|^BENCH|^SCENE|^EXIT 0$|^MESHLET |^PIPELINES|^STARTUP|^HOT-RELOAD|^GRAPH-TRANSIENTS|^ASYNC-COMPUTE|Validation Enabled' "$1" |
        grep -Ev "${2:-^$}" | grep -c . || true
}

# --- 1. self-test ---------------------------------------------------------------
log="$out_dir/self-test.log"
status=0
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog \
    "$app" --bench 3 --frames 90 --warmup 0 --no-ui --capture "$out_dir/self-test.png" \
    --debug-hot-reload "$probe" --debug-split-encoding --debug-async-compute --debug-graph-transients \
    >"$log" 2>&1 || status=$?
n=$(unexpected "$log")
if [[ $status -eq 0 && $n -eq 0 ]] && grep -q '^HOT-RELOAD .*PASS' "$log" && grep -q '^GRAPH-TRANSIENTS .*PASS' "$log" &&
   grep -q '^ASYNC-COMPUTE .*PASS' "$log"; then
    echo "self-test: PASS ($(grep '^HOT-RELOAD' "$log"))"
else
    echo "self-test: FAIL (exit $status, $n unexpected lines, see $log)"; failures=$((failures + 1))
fi

# --- 2. negative control --------------------------------------------------------
log="$out_dir/negative.log"
status=0
"$app" --bench 1 --frames 60 --warmup 0 --no-ui --capture "$out_dir/negative.png" \
    --debug-hot-reload "$build_dir/shaders/phosphor.metallib" >"$log" 2>&1 || status=$?
if [[ $status -ne 0 ]] && grep -q '^HOT-RELOAD .*FAIL' "$log"; then
    echo "negative control: PASS (the check fails without the probe)"
else
    echo "negative control: FAIL (the check did not catch a missing swap, see $log)"; failures=$((failures + 1))
fi

# --- 3. real watcher --------------------------------------------------------------
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
cp "$repo_dir"/shaders/*.metal "$dir/"
cp "$repo_dir"/shaders/*.h "$dir/"
log="$out_dir/watcher.log"
# 1500 frames with vsync: >= 12 s even on a 120 Hz display.
MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog \
    "$app" --bench 1 --frames 1500 --warmup 0 --no-ui --shader-dir "$dir" --capture "$out_dir/watcher.png" \
    >"$log" 2>&1 &
pid=$!
sleep 3
printf '\nthis is not metal;\n' >>"$dir/forward.metal"          # broken edit
sleep 3
# Valid edit (magenta), written in one rename: two separate writes let the
# 250 ms watcher poll fall in between and reload twice (measured, F4).
{ printf '#define PHOSPHOR_HOT_RELOAD_PROBE 1\n'; cat "$repo_dir/shaders/forward.metal"; } > "$dir/forward.metal.tmp"
mv "$dir/forward.metal.tmp" "$dir/forward.metal"
status=0
wait "$pid" || status=$?
# The failed build prints the compiler's diagnostics (expected here).
n=$(unexpected "$log" '^\[ERROR\].*build failed|In file included from|forward\.metal:[0-9]+:[0-9]+: error|^this is not metal|^[[:space:]]*\^|errors? generated|^[[:space:]]*$')
if [[ $status -eq 0 && $n -eq 0 ]] && grep -q 'build failed, pipelines unchanged' "$log" &&
   grep -q 'Shaders reloaded: generation' "$log" && grep -q '^PIPELINES .*reloads 1 (failed 0)' "$log"; then
    echo "watcher: PASS (broken edit rejected, valid edit reloaded once)"
else
    echo "watcher: FAIL (exit $status, $n unexpected lines, see $log)"; failures=$((failures + 1))
fi

if [[ $failures -ne 0 ]]; then
    echo "hot reload check: $failures failure(s)"
    exit 1
fi
echo "hot reload check: all pass"
