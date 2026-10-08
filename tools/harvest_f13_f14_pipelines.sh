#!/usr/bin/env bash
# SOURCE ONLY / NOT EXECUTED. Tester runs after owned F9 integration and build.
# Serial compiler-recorded descriptors; never fabricate an offline archive.
set -euo pipefail
build_dir=${1:-build/lighting}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
app="$build_dir/phosphor"
destination=${2:-"$repo_dir/shaders/pipelines.mtl4-json"}
command -v jq >/dev/null || { echo 'jq is required' >&2; exit 2; }
[[ -x "$app" ]] || { echo "Build $app in the tester checkout first" >&2; exit 2; }
[[ -f "$destination" ]] || { echo 'Existing baseline pipeline corpus is required' >&2; exit 2; }
if [[ -n ${HARVEST_ARTIFACT_DIR:-} ]]; then
    work=$HARVEST_ARTIFACT_DIR
    mkdir "$work" # refuse overwriting earlier raw evidence
else
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
fi
corpora=("$destination")
for mode in surface-rt surface-apple9 denoised-request volumes-full volumes-csm; do
    case "$mode" in
      surface-rt) flags=(--rt on --shadows rt --lighting restir --gi restir --reflections rt --ao rtao --lighting-denoise custom --reflection-capture-probe) ;;
      surface-apple9) flags=(--force-family apple9 --rt off --shadows off --lighting legacy --gi off --reflections probes --ao gtao --lighting-denoise custom --reflection-capture-probe) ;;
      denoised-request) flags=(--rt on --shadows rt --lighting brute --gi ddgi --reflections rt --ao rtao --lighting-denoise metalfx) ;;
      volumes-full) flags=(--rt on --shadows rt --lighting brute --gi ddgi --reflections rt --ao rtao --lighting-denoise custom --atmosphere on --fog on --clouds on --cloud-full-rate) ;;
      volumes-csm) flags=(--force-family apple9 --rt off --shadows csm --lighting legacy --gi off --reflections probes --ao gtao --lighting-denoise custom --atmosphere on --fog on --clouds on) ;;
    esac
    # A source signal capture explicitly waits for final PSOs before graph
    # construction. This is harvesting, never a steady-state performance run.
    "$app" --harvest-pipelines "$work/$mode.json" --bench 6 --reflection-scene roughness --frames 2 --warmup 0 \
        --no-ui --offscreen --render-path visibility --post --upscaler native --debug-lighting 1 \
        --capture-linear-signal specular --capture-linear "$work/$mode.pfm" --capture-linear-frame 1 \
        "${flags[@]}" >"$work/$mode.log" 2>&1 || { tail -30 "$work/$mode.log" >&2; exit 1; }
    [[ $(jq '.libraries | length' "$work/$mode.json") == 1 ]] || { echo 'Unexpected framework/private library in harvest' >&2; exit 1; }
    corpora+=("$work/$mode.json")
done
# Requested SDK mode can report custom MissingFactory. This script cannot
# reconcile the owned gateway or certify SDK output units/lifetime. Engine MSL
# descriptors are harvested; external SDK implementation stays framework-owned.
python3 "$repo_dir/tools/merge_pipeline_harvests.py" --output "$work/merged.json" "${corpora[@]}"
[[ -s "$work/merged.json" ]] || exit 1
jq -S . "$work/merged.json" >"$work/final.json"
mv "$work/final.json" "$destination"
echo "Wrote compiler-harvested F13/F14 engine pipelines to $destination; SDK acceptance remains separate"
