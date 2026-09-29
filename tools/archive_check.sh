#!/usr/bin/env bash
# archive_check.sh -- F3.5 miss scenarios with REAL pipeline archives.
#
#   tools/archive_check.sh [build-dir] [reference-dir]
#
# Every scenario runs bench 1 under Metal API + shader validation with an
# explicit --pipeline-archive, parses the app's PIPELINES line (requests,
# archive hits / misses / unavailable, compiler calls), compares the capture
# with the visual_check reference (0 pixels may differ: a miss or an
# unavailable archive only changes HOW a pipeline is obtained, never the
# image) and counts unexpected log lines (must be 0).
#
# Scenarios (all archives are built here with metal-tt from the build's
# substituted pipelines script, $build/shaders/pipelines.mtl4-json):
#   full     the build's archive: nothing may miss or be unavailable
#            (compiler calls are reported; ideally 0).
#   a        foreign-OS archive from $CI_ARCHIVE (e.g. the pipeline-archive-
#            macos26 CI artifact opened on a newer macOS): the OS rejects it
#            at open, so every result is "unavailable".  SKIP if unset.
#   b        archive built with -arch applegpu_g16s only (no slice for the
#            local GPU): rejected at open -> unavailable.
#   c        partial archive: the compute descriptors of the graph-debug
#            kernels (debug_*) are removed from the script.  Without
#            --debug-graph-transients the kernels are never requested, so
#            misses must stay 0 (negative control); with it, misses must equal
#            the number of dropped descriptors.
#   d        stale archive: built from a metallib in which one constant of
#            forward.metal was changed.  metal-tt keys are derived from the
#            library contents per function, so exactly the pipelines using
#            forward_vs / forward_fs miss (count taken from the script) and
#            the others still hit.  The app runs with the real metallib.
#   e        nonexistent archive path -> unavailable.
#
# Expected counts that depend on the app are derived from the script (JSON)
# or from the full run, never hard-coded.  Exit 1 if any scenario fails.
# Needs: jq, xcrun metal / metal-tt.  Default dirs: build, build/reference.
# Environment: CI_ARCHIVE=<file> enables scenario a.
set -euo pipefail

build_dir=${1:-build}
ref_dir=${2:-build/reference}
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
app="$build_dir/phosphor"
diff_tool="$build_dir/image_diff"
script_json="$build_dir/shaders/pipelines.mtl4-json"
metallib="$build_dir/shaders/phosphor.metallib"
full_archive="$build_dir/shaders/phosphor-archive.metallib"
out_dir="$build_dir/archive-check"
warmup=30

command -v jq >/dev/null || { echo "error: jq is required" >&2; exit 2; }
for f in "$app" "$diff_tool"; do [[ -x "$f" ]] || { echo "error: build $f first" >&2; exit 2; }; done
for f in "$script_json" "$metallib" "$full_archive" "$ref_dir/bench1.png"; do
    [[ -f "$f" ]] || { echo "error: missing $f (build all targets; tools/visual_check.sh --update for the reference)" >&2; exit 2; }
done
rm -rf "$out_dir"
mkdir -p "$out_dir"
# Absolute paths: metal-tt resolves the library path in the script.
metallib=$(cd "$(dirname "$metallib")" && pwd)/$(basename "$metallib")

native_arch=$(xcrun metal-arch 2>/dev/null || true)

failures=0

# make_script IN OUT [LIBRARY]: copy of a script whose library path is LIBRARY.
make_script() {
    jq --arg p "${3:-$metallib}" '.libraries |= map(.path = $p)' "$1" >"$2"
}
# make_archive SCRIPT OUT [ARCH]
make_archive() {
    local arch=()
    [[ -n "${3:-}" ]] && arch=(-arch "$3")
    # ${arr[@]+...}: empty arrays are "unbound" for bash 3.2 (macOS /bin/bash) under set -u.
    xcrun -sdk macosx metal-tt "$1" ${arch[@]+"${arch[@]}"} -o "$2" >"$2.log" 2>&1 \
        || { echo "error: metal-tt failed for $1:" >&2; cat "$2.log" >&2; exit 2; }
}

# run_case NAME ARCHIVE ALLOW_ARCHIVE_WARNINGS [app args...]
# Sets: requests hits misses unavailable compiler_calls messages status pixels.
run_case() {
    local name=$1 archive=$2 allow_warn=$3
    shift 3
    local capture="$out_dir/$name.png" log="$out_dir/$name.log"
    status=0
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog \
        "$app" --bench 1 --warmup "$warmup" --frames 1 --no-ui --fixed-timestep \
        --pipeline-archive "$archive" "$@" --capture "$capture" >"$log" 2>&1 || status=$?
    local line
    line=$(grep -m1 'PIPELINES' "$log" || true)
    requests=$(sed -nE 's/.*requests ([0-9]+).*/\1/p' <<<"$line")
    hits=$(sed -nE 's/.*archive hits ([0-9]+),.*/\1/p' <<<"$line")
    misses=$(sed -nE 's/.*misses ([0-9]+),.*/\1/p' <<<"$line")
    unavailable=$(sed -nE 's/.*unavailable ([0-9]+).*/\1/p' <<<"$line")
    compiler_calls=$(sed -nE 's/.*compiler calls ([0-9]+).*/\1/p' <<<"$line")
    : "${requests:=?}" "${hits:=?}" "${misses:=?}" "${unavailable:=?}" "${compiler_calls:=?}"
    # Anything besides INFO / BENCH / PIPELINES / SWITCH lines and the
    # "Validation Enabled" banners is a problem.  A rejected archive may be
    # reported by the app as a warning mentioning the archive: tolerated only
    # in the scenarios that expect a rejection.
    local ignore='^\[INFO\]|^BENCH|^PIPELINES|^SWITCH|^GRAPH-TRANSIENTS|^ASYNC-COMPUTE|Validation Enabled'
    [[ $allow_warn == 1 ]] && ignore="$ignore|^\[WARN\].*[Aa]rchive"
    messages=$(grep -Ev "$ignore" "$log" | grep -c . || true)
    diff_ok=1
    pixels=$("$diff_tool" "$ref_dir/bench1.png" "$capture" --diff "$out_dir/$name-diff.png" 2>&1) || diff_ok=0
}

# verdict NAME DETAILS [failure reasons...]: prints PASS/FAIL for the last run.
verdict() {
    local name=$1 details=$2 problems=("${@:3}")
    [[ $status -eq 0 ]] || problems+=("exit status $status")
    [[ $messages -eq 0 ]] || problems+=("$messages unexpected log lines (see $out_dir/$name.log)")
    [[ $requests != '?' ]] || problems+=("no PIPELINES line in the log")
    [[ $diff_ok -eq 1 ]] || problems+=("capture differs from the reference: $pixels")
    local summary="requests $requests hits $hits misses $misses unavailable $unavailable compiler calls $compiler_calls; $details"
    if [[ ${#problems[@]} -eq 0 ]]; then
        echo "$name: PASS ($summary)"
    else
        failures=$((failures + 1))
        echo "$name: FAIL ($summary)"
        printf '    - %s\n' ${problems[@]+"${problems[@]}"}
    fi
}

# --- full ---------------------------------------------------------------
run_case full "$full_archive" 0
full_requests=$requests
full_hits=$hits
problems=()
[[ $misses == 0 ]] || problems+=("misses $misses != 0")
[[ $unavailable == 0 ]] || problems+=("unavailable $unavailable != 0")
[[ $hits != 0 && $hits != '?' ]] || problems+=("no archive hit at all")
verdict full "compiler calls reported only" ${problems[@]+"${problems[@]}"}

# --- a: foreign-OS archive ------------------------------------------------
if [[ -n "${CI_ARCHIVE:-}" && -f "${CI_ARCHIVE:-}" ]]; then
    run_case a-foreign-os "$CI_ARCHIVE" 1
    problems=()
    [[ $unavailable != 0 && $unavailable != '?' ]] || problems+=("nothing reported unavailable")
    [[ $hits == 0 ]] || problems+=("hits $hits != 0")
    [[ $misses == 0 ]] || problems+=("misses $misses != 0 (a rejected archive is unavailable, not a miss)")
    verdict a-foreign-os "archive $CI_ARCHIVE; unavailable vs requests $unavailable/$requests" ${problems[@]+"${problems[@]}"}
else
    echo "a-foreign-os: SKIP (set CI_ARCHIVE to the pipeline-archive-macos26 CI artifact)"
fi

# --- b: archive without the local GPU slice --------------------------------
if [[ -n "$native_arch" ]]; then
    foreign_arch=applegpu_g16s
    [[ $native_arch != "$foreign_arch" ]] || foreign_arch=applegpu_g17s
    make_archive "$script_json" "$out_dir/b.metallib" "$foreign_arch"
    run_case b-foreign-arch "$out_dir/b.metallib" 1
    problems=()
    [[ $unavailable != 0 && $unavailable != '?' ]] || problems+=("nothing reported unavailable")
    [[ $hits == 0 ]] || problems+=("hits $hits != 0")
    [[ $misses == 0 ]] || problems+=("misses $misses != 0")
    verdict b-foreign-arch "archive only for $foreign_arch, local $native_arch; unavailable vs requests $unavailable/$requests" ${problems[@]+"${problems[@]}"}
else
    echo "b-foreign-arch: SKIP (xcrun metal-arch failed)"
fi

# --- c: partial archive ------------------------------------------------------
# Drop the compute descriptors whose function name is debug_*; follow the
# "fnd:<label>" references (labels are the function descriptors' labels).
# shellcheck disable=SC2016 # jq program, $drop is a jq variable
debug_filter='
    [.function_descriptors.library_function_descriptors[]
       | select(.name | test("^debug_(fill|reduce|expand|checksum)$")) | "fnd:" + .label] as $drop
  | .pipeline_descriptors.compute_pipeline_descriptors |=
      map(select(.compute_function_descriptor as $f | $drop | index($f) | not))'
dropped=$(jq '
    [.function_descriptors.library_function_descriptors[]
       | select(.name | test("^debug_(fill|reduce|expand|checksum)$")) | "fnd:" + .label] as $drop
  | [.pipeline_descriptors.compute_pipeline_descriptors[]
       | select(.compute_function_descriptor as $f | $drop | index($f))] | length' "$script_json")
jq "$debug_filter" "$script_json" >"$out_dir/c.mtl4-json"
make_archive "$out_dir/c.mtl4-json" "$out_dir/c.metallib" "$native_arch"
if [[ $dropped -eq 0 ]]; then
    echo "c-partial: SKIP (the script has no debug_* compute descriptors; re-harvest with the whole table)"
else
    run_case c-partial-nodbg "$out_dir/c.metallib" 0
    problems=()
    [[ $misses == 0 ]] || problems+=("misses $misses != 0 (negative control: dropped kernels are not requested)")
    [[ $unavailable == 0 ]] || problems+=("unavailable $unavailable != 0")
    verdict c-partial-nodbg "$dropped debug descriptors dropped, not requested" ${problems[@]+"${problems[@]}"}

    run_case c-partial-dbg "$out_dir/c.metallib" 0 --debug-graph-transients
    problems=()
    [[ $misses == "$dropped" ]] || problems+=("misses $misses != dropped descriptors $dropped")
    [[ $unavailable == 0 ]] || problems+=("unavailable $unavailable != 0")
    verdict c-partial-dbg "$dropped debug descriptors dropped, requested with --debug-graph-transients" ${problems[@]+"${problems[@]}"}
fi

# --- d: stale archive (modified metallib) ------------------------------------
# Change one constant in forward.metal, compile exactly like App.cmake (same
# flags, deployment target from the build's cache) and translate the script
# against that metallib.  Expected misses = descriptors that use forward_vs or
# forward_fs; every other requested pipeline must still hit.
stale_dir="$out_dir/stale"
mkdir -p "$stale_dir"
cp "$repo_dir"/shaders/*.metal "$stale_dir/"
if ! sed -i.bak 's/float3(0\.30, 0\.36, 0\.45)/float3(0.31, 0.36, 0.45)/' "$stale_dir/forward.metal" \
    || cmp -s "$stale_dir/forward.metal" "$repo_dir/shaders/forward.metal"; then
    echo "d-stale: FAIL (could not modify the constant in forward.metal; update this script)"
    failures=$((failures + 1))
else
    rm -f "$stale_dir/forward.metal.bak"
    target=$(sed -nE 's/^CMAKE_OSX_DEPLOYMENT_TARGET:[A-Z]+=(.*)$/\1/p' "$build_dir/CMakeCache.txt")
    airs=()
    for src in "$stale_dir"/*.metal; do
        xcrun -sdk macosx metal -std=metal4.0 "-mmacosx-version-min=${target:-26.0}" \
            -I "$repo_dir/src" -Wall -c "$src" -o "${src%.metal}.air" >"$src.log" 2>&1 \
            || { echo "error: compiling $src failed:" >&2; cat "$src.log" >&2; exit 2; }
        airs+=("${src%.metal}.air")
    done
    xcrun -sdk macosx metallib "${airs[@]}" -o "$stale_dir/stale.metallib"
    make_script "$script_json" "$out_dir/d.mtl4-json" "$(cd "$stale_dir" && pwd)/stale.metallib"
    make_archive "$out_dir/d.mtl4-json" "$out_dir/d.metallib" "$native_arch"
    expected_stale=$(jq '
        [.function_descriptors.library_function_descriptors[]
           | select(.name == "forward_vs" or .name == "forward_fs") | "fnd:" + .label] as $fwd
      | ([.pipeline_descriptors.render_pipeline_descriptors[]
            | select((.vertex_function_descriptor as $v | $fwd | index($v))
                  or (.fragment_function_descriptor as $f | $fwd | index($f)))] | length)
      + ([.pipeline_descriptors.compute_pipeline_descriptors[]
            | select(.compute_function_descriptor as $c | $fwd | index($c))] | length)' "$script_json")
    run_case d-stale "$out_dir/d.metallib" 0
    problems=()
    [[ $expected_stale -gt 0 ]] || problems+=("script has no forward pipeline: nothing to miss")
    [[ $misses == "$expected_stale" ]] || problems+=("misses $misses != forward descriptors $expected_stale")
    [[ $hits == $((full_hits - expected_stale)) ]] || problems+=("hits $hits != full-run hits $full_hits minus $expected_stale")
    [[ $unavailable == 0 ]] || problems+=("unavailable $unavailable != 0")
    verdict d-stale "expected misses $expected_stale (forward_vs/forward_fs users), full-run requests $full_requests" ${problems[@]+"${problems[@]}"}
fi

# --- e: nonexistent archive ---------------------------------------------------
run_case e-missing "$out_dir/does-not-exist.metallib" 1
problems=()
[[ $unavailable != 0 && $unavailable != '?' ]] || problems+=("nothing reported unavailable")
[[ $hits == 0 ]] || problems+=("hits $hits != 0")
[[ $misses == 0 ]] || problems+=("misses $misses != 0")
verdict e-missing "unavailable vs requests $unavailable/$requests" ${problems[@]+"${problems[@]}"}

echo
if [[ $failures -ne 0 ]]; then
    echo "archive check: $failures failure(s)"
    exit 1
fi
echo "archive check: all scenarios pass"
