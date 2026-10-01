#!/usr/bin/env bash
# visual_check.sh -- regression check of every test bench:
#   * Metal API + shader validation must print nothing,
#   * the capture must match the reference image pixel for pixel.
#
#   tools/visual_check.sh [build-dir] [reference-dir] [--update]
#
# EXTRA_ARGS (environment) is appended to every run, e.g.
# EXTRA_ARGS=--debug-graph-transients (its GRAPH-TRANSIENTS line is expected; --debug-async-compute: ASYNC-COMPUTE).
# BENCH8_ARGS (environment, default empty) is appended to bench 8 only
# ("1M Instances"): under API + shader validation the 1M-instance default
# can be slow; if a run exceeds ~2 minutes use BENCH8_ARGS="--instances 100000"
# (the scene is a different size then, so references must be taken with the
# same value).
# Defaults: build (Debug), build/reference.  --update (re)creates the
# references instead of comparing.  Captures use --fixed-timestep and the
# same frame count, so animated benches are deterministic; --inject-input
# proves that keyboard/mouse input cannot change a capture.
set -euo pipefail

build_dir=${1:-build}
ref_dir=${2:-build/reference}
update=${3:-}
app="$build_dir/phosphor"
diff_tool="$build_dir/image_diff"
out_dir="$build_dir/visual-check"
warmup=30

[[ -x "$app" && -x "$diff_tool" ]] || { echo "error: build $app and $diff_tool first" >&2; exit 2; }
mkdir -p "$out_dir" "$ref_dir"

failures=0
for bench in 1 2 3 4 5 6 7 8; do
    capture="$out_dir/bench$bench.png"
    log="$out_dir/bench$bench.log"
    status=0
    extra_bench_args=""
    [[ "$bench" == 8 ]] && extra_bench_args=${BENCH8_ARGS:-}
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog \
        "$app" --bench "$bench" --warmup "$warmup" --frames 1 --no-ui --fixed-timestep --inject-input \
        ${EXTRA_ARGS:-} ${extra_bench_args:-} --capture "$capture" >"$log" 2>&1 || status=$?
    # Anything besides our INFO lines, the BENCH/PIPELINES/STARTUP/SWITCH
    # summaries and the two "Validation Enabled" banners is a problem.
    messages=$(grep -Ev '^\[INFO\]|^BENCH|^GRAPH-TRANSIENTS|^ASYNC-COMPUTE|^PIPELINES|^STARTUP|^SWITCH|Validation Enabled' "$log" | grep -c . || true)
    # F3: every pipeline must have been built (the PIPELINES summary line).
    pipeline_failures=$(sed -nE 's/^PIPELINES .*\| failures ([0-9]+) .*/\1/p' "$log")
    if [[ "${pipeline_failures:-missing}" != "0" ]]; then
        messages=$((messages + 1))
        echo "bench $bench: PIPELINES line missing or failures=${pipeline_failures:-?}"
    fi

    if [[ "$update" == "--update" ]]; then
        cp "$capture" "$ref_dir/bench$bench.png"
        result="reference updated"
    else
        result=$("$diff_tool" "$ref_dir/bench$bench.png" "$capture" --diff "$out_dir/bench$bench-diff.png") || failures=$((failures + 1))
    fi
    if [[ $status -ne 0 || $messages -ne 0 ]]; then
        failures=$((failures + 1))
        echo "bench $bench: FAIL exit=$status, $messages validation/log messages (see $log)"
    else
        echo "bench $bench: $result"
    fi
done

if [[ $failures -ne 0 ]]; then
    echo "visual check: $failures failure(s)"
    exit 1
fi
echo "visual check: all benches pass"
