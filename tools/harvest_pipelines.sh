#!/usr/bin/env bash
# harvest_pipelines.sh -- regenerate shaders/pipelines.mtl4-json (F3.4).
#
#   tools/harvest_pipelines.sh [build-dir]
#
# Runs the app in harvest mode (it records every pipeline descriptor it
# creates, including the whole variant table, through
# MTL4PipelineDataSetSerializer), then normalises the JSON so it is stable
# and machine independent:
#   * the absolute metallib path becomes the placeholder @PHOSPHOR_METALLIB@
#     (cmake/PipelineArchive.cmake substitutes it at build time),
#   * `jq -S` sorts the keys and pretty-prints (small, reviewable diffs).
# The labels inside are only cross-references; metal-tt recomputes the keys
# from the real metallib, so the file stays valid when shader bodies change.
# Only new/removed functions or changed pipeline state require a re-harvest.
#
# The graph-debug, async-probe and known-cost pipelines are requested too (their flags
# are on), so the archive covers every self-check.
# EXTRA_ARGS (environment) is appended to the app command line.
# Default build dir: build (Debug).  Needs jq.
set -euo pipefail

build_dir=${1:-build}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
app="$build_dir/phosphor"
out="$repo_dir/shaders/pipelines.mtl4-json"

command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 2; }
[[ -x "$app" ]] || { echo "error: build $app first" >&2; exit 2; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
raw="$tmp/harvest.mtl4-json"

# shellcheck disable=SC2086 # EXTRA_ARGS is intentionally word-split
# F6: the mesh path (two-phase) requests every indexed pipeline too, plus the
# mesh variants, meshlet kernels, Hi-Z kernels (both backends in harvest
# mode) and the Hi-Z view.
"$app" --harvest-pipelines "$raw" --frames 1 --no-ui --debug-graph-transients --debug-async-compute --debug-gpu-cost 1 \
    --geometry-path mesh --meshlet-cull two-phase --post --upscaler native ${EXTRA_ARGS:-} >"$tmp/app.log" 2>&1 \
    || { echo "error: harvest run failed:" >&2; tail -20 "$tmp/app.log" >&2; exit 1; }
[[ -s "$raw" ]] || { echo "error: the app did not write $raw" >&2; tail -20 "$tmp/app.log" >&2; exit 1; }

# MetalFX owns its private shader libraries and archives. Creating a scaler
# during capture records those too; they must never be relabeled as ours.
[[ $(jq '.libraries | length' "$raw") == 1 ]] || {
    echo "error: harvest captured external libraries (use --upscaler native)" >&2; exit 1;
}

jq -S '.libraries |= map(.path = "@PHOSPHOR_METALLIB@")' "$raw" >"$tmp/normalised.json"
[[ -s "$tmp/normalised.json" ]] || { echo "error: jq produced no output" >&2; exit 1; }
mv "$tmp/normalised.json" "$out"

echo "wrote $out"
jq -r '
    "render pipeline descriptors:  \((.pipeline_descriptors.render_pipeline_descriptors // []) | length)",
    "compute pipeline descriptors: \((.pipeline_descriptors.compute_pipeline_descriptors // []) | length)",
    "functions: \([.function_descriptors.library_function_descriptors[]?.name] | unique | join(", "))"
' "$out"
