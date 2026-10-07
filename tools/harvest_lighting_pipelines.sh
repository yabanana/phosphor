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
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for mode in cache rt; do
    if [[ "$mode" == cache ]]; then shadow=(--shadows csm --shadow-cache on); else shadow=(--shadows rt); fi
    "$app" --harvest-pipelines "$work/$mode.json" --bench 6 --lighting-scene cornell --frames 1 --warmup 0 \
        --no-ui --offscreen --render-path visibility --rt on --lighting restir --gi restir \
        --debug-lighting 1 --capture-linear-signal indirect-diffuse --capture-linear "$work/$mode.pfm" \
        "${shadow[@]}" >"$work/$mode.log" 2>&1 || { tail -30 "$work/$mode.log" >&2; exit 1; }
    [[ $(jq '.libraries | length' "$work/$mode.json") == 1 ]] || { echo 'Unexpected framework/private library in harvest' >&2; exit 1; }
done
# Merge label-addressed descriptor arrays. Preserve every existing family of
# descriptors and the actual compiler-recorded linked-function metadata.
jq -s 'reduce .[] as $doc ({};
    reduce ($doc|keys_unsorted[]) as $key (. ;
      if $key=="libraries" then .libraries=(($doc.libraries|map(.path="@PHOSPHOR_METALLIB@"))+(.libraries//[])|unique_by(.label))
      elif ($doc[$key]|type)=="object" then
        .[$key]=(reduce ($doc[$key]|keys_unsorted[]) as $sub (.[$key]//{};
          .[$sub]=((.[$sub]//[])+$doc[$key][$sub]|unique_by(.label))))
      else .[$key]=$doc[$key] end))' "$destination" "$work/cache.json" "$work/rt.json" >"$work/merged.json"
[[ -s "$work/merged.json" ]] || exit 1
jq -S . "$work/merged.json" >"$work/final.json"
mv "$work/final.json" "$destination"
echo "Wrote compiler-harvested lighting pipelines to $destination"
