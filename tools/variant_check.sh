#!/usr/bin/env bash
# variant_check.sh -- pixel check of every forward variant (F3.3) on every bench.
#
#   tools/variant_check.sh [build-dir] [max-pixels]
#
# For each bench and debug mode the generic pipeline is captured
# (--debug-pipeline-fallback --debug-mode D), then every variant of the
# generated table is forced (--force-variant V) and captured with the debug
# mode it was specialised for.  The app logs the variant's values ("Forward
# variant V requested: light types 0xL, emissive E, debug mode D"); the
# bench's own needs come from a normal run ("light types 0xS, emissive SE").
#   * compatible variant: debug mode 1/2 (normals / base colour do not
#     depend on lights or emission), or L contains every light type of the
#     scene.  The only allowed difference is codegen -- every pixel within 1
#     of the generic capture and at most max-pixels (default 100) pixels
#     different at all.  EMISSIVE does not decide compatibility: false only
#     skips the emissive texture fetch (the add of factor x 1 stays, F3.3),
#     so it matches the generic on every bench (their emissive textures are
#     white); the engine picks false only when no material emits;
#   * incompatible variant (lit mode without a light type the scene uses):
#     it MUST differ (negative control: forcing works and the check can
#     fail).
# No Metal validation (pixels only; visual_check covers validation).
# Default build dir: build/release.  ~11 min for 8 benches x 42 variants.
set -euo pipefail

build_dir=${1:-build/release}
max_pixels=${2:-100}
app="$build_dir/phosphor"
diff_tool="$build_dir/image_diff"
out_dir="$build_dir/variant-check"
[[ -x "$app" && -x "$diff_tool" ]] || { echo "error: build $app and $diff_tool first" >&2; exit 2; }
mkdir -p "$out_dir"

capture() { # png log args...
    local png=$1 log=$2
    shift 2
    env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION "$app" --warmup 30 --frames 1 --no-ui --fixed-timestep \
        "$@" --capture "$png" >"$log" 2>&1
}
variant_values() { # log -> "V L E D" of the (single) forward variant requested
    sed -nE 's/.*Forward variant ([0-9]+) requested: light types 0x([0-9a-f]+), emissive ([01]), debug mode ([0-9]).*/\1 \2 \3 \4/p' "$1" | head -1
}

failures=0 compatible=0 incompatible=0 worst_pixels=0
for bench in 1 2 3 4 5 6 7 8; do
    capture "$out_dir/scene-$bench.png" "$out_dir/scene-$bench.log" --bench "$bench"
    read -r _ scene_lights scene_emissive _ <<<"$(variant_values "$out_dir/scene-$bench.log")"
    for mode in 0 1 2; do
        capture "$out_dir/generic-$bench-$mode.png" "$out_dir/generic-$bench-$mode.log" \
            --bench "$bench" --debug-mode "$mode" --debug-pipeline-fallback
    done
    bench_fail=0
    for v in $(seq 0 41); do
        png="$out_dir/v$v-b$bench.png" log="$out_dir/v$v-b$bench.log"
        # The debug mode of the frame matches the variant's (found in the log
        # of a first run, then captured again if it differs from mode 0).
        capture "$png" "$log" --bench "$bench" --force-variant "$v"
        read -r vv lights emissive mode <<<"$(variant_values "$log")"
        [[ "$vv" == "$v" ]] || { echo "bench $bench v$v: FAIL (no variant log line)"; failures=$((failures + 1)); continue; }
        if [[ "$mode" != 0 ]]; then
            capture "$png" "$log" --bench "$bench" --force-variant "$v" --debug-mode "$mode"
        fi
        result=$("$diff_tool" "$out_dir/generic-$bench-$mode.png" "$png" || true)
        pixels=$(sed -nE 's/^([0-9]+) of .*/\1/p' <<<"$result")
        delta=$(sed -nE 's/.*max delta ([0-9]+).*/\1/p' <<<"$result")
        if [[ $mode != 0 ]] || (( (16#$lights & 16#$scene_lights) == 16#$scene_lights )); then
            compatible=$((compatible + 1))
            (( pixels > worst_pixels )) && worst_pixels=$pixels
            if (( delta > 1 || pixels > max_pixels )); then
                echo "bench $bench v$v (lights 0x$lights emissive $emissive mode $mode): FAIL compatible but $result"
                failures=$((failures + 1)); bench_fail=1
            fi
        else
            incompatible=$((incompatible + 1))
            if (( pixels == 0 )); then
                echo "bench $bench v$v (lights 0x$lights emissive $emissive mode $mode): FAIL incompatible but identical"
                failures=$((failures + 1)); bench_fail=1
            fi
        fi
    done
    (( bench_fail )) || echo "bench $bench (scene lights 0x$scene_lights, emissive $scene_emissive): PASS"
done

echo "variants: $compatible compatible (worst $worst_pixels pixels, delta <= 1 required), $incompatible incompatible (must differ)"
if [[ $failures -ne 0 ]]; then
    echo "variant check: $failures failure(s)"
    exit 1
fi
echo "variant check: all pass"
