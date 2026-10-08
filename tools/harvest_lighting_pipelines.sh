#!/usr/bin/env bash
# WRITTEN ONLY. Run by the tester after F9 alignment and a successful MSL/app
# build. Serial harvest covers both cache raster and solar RT consumer PSOs.
# Does not overwrite the baseline script until both captures succeed.
set -euo pipefail
build_dir=${1:-build/lighting}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
app="$build_dir/phosphor"
destination=${2:-"$repo_dir/shaders/pipelines.mtl4-json"}
command -v jq >/dev/null || { echo 'jq is required' >&2; exit 2; }
[[ -x "$app" ]] || { echo "Build $app in the tester checkout first" >&2; exit 2; }
if [[ -n ${HARVEST_ARTIFACT_DIR:-} ]]; then
    work=$HARVEST_ARTIFACT_DIR
    mkdir "$work" # refuse overwriting earlier raw evidence
else
    work=$(mktemp -d)
    trap 'rm -rf "$work"' EXIT
fi
for mode in cache rt; do
    if [[ "$mode" == cache ]]; then shadow=(--shadows csm --shadow-cache on); else shadow=(--shadows rt); fi
    "$app" --harvest-pipelines "$work/$mode.json" --bench 6 --lighting-scene cornell --frames 1 --warmup 0 \
        --no-ui --offscreen --render-path visibility --rt on --lighting restir --gi restir \
        --debug-lighting 1 --capture-linear-signal indirect-diffuse --capture-linear "$work/$mode.pfm" \
        "${shadow[@]}" >"$work/$mode.log" 2>&1 || { tail -30 "$work/$mode.log" >&2; exit 1; }
    [[ $(jq '.libraries | length' "$work/$mode.json") == 1 ]] || { echo 'Unexpected framework/private library in harvest' >&2; exit 1; }
done
# Function records have labels, but pipeline descriptors do not. Deduplicate
# complete records or all unlabeled pipelines collapse into a single entry.
# Preserve baseline variants and actual compiler-recorded linkage metadata.
python3 "$repo_dir/tools/merge_pipeline_harvests.py" --output "$work/merged.json" "$destination" "$work/cache.json" "$work/rt.json"
[[ -s "$work/merged.json" ]] || exit 1
jq -S . "$work/merged.json" >"$work/final.json"
mv "$work/final.json" "$destination"
echo "Wrote compiler-harvested lighting pipelines to $destination"
