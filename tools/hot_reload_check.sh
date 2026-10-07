#!/usr/bin/env bash
# hot_reload_check.sh -- F3.6 shader hot reload checks (Debug build).
#
#   tools/hot_reload_check.sh [build-dir] [--rt-only]
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
# --rt-only runs the F9 IFT checks instead: a valid reload on Sponza must
# preserve RT correctness, then a temporary alpha-counter poison must become
# observable after watcher reload on every frame slot. Original shaders are
# never edited. The expected checker failure must exit 1 AND print EXIT 1;
# a validation error, signal or timeout never satisfies the negative control.
# Default build dir: build (Debug: hot reload is compiled in Debug only).
set -euo pipefail

build_dir=build
rt_only=0
build_argument=0
for argument in "$@"; do
    case "$argument" in
        --rt-only) rt_only=1 ;;
        --help|-h) echo "usage: tools/hot_reload_check.sh [build-dir] [--rt-only]"; exit 0 ;;
        --*) echo "error: unknown option $argument" >&2; exit 2 ;;
        *)
            [[ $build_argument -eq 0 ]] || { echo "error: only one build directory is allowed" >&2; exit 2; }
            build_dir=$argument
            build_argument=1
            ;;
    esac
done
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

# --- F9: targeted RT/IFT reload -------------------------------------------------
if [[ $rt_only -eq 1 ]]; then
    scene="$repo_dir/assets/sponza/Sponza.gltf"
    [[ -f "$scene" ]] || { echo "error: Sponza is required for RT alpha hot reload" >&2; exit 2; }
    python_cmd=(python3)
    if command -v mise >/dev/null 2>&1; then python_cmd=(mise exec -- python3); fi
    rt_out="$out_dir/rt"
    mkdir -p "$rt_out"
    rt_temp=$(mktemp -d)
    rt_pid=
    cleanup_rt() {
        if [[ -n "$rt_pid" ]] && kill -0 "$rt_pid" 2>/dev/null; then
            kill "$rt_pid" 2>/dev/null || true
            wait "$rt_pid" 2>/dev/null || true
        fi
        rm -rf "$rt_temp"
    }
    trap cleanup_rt EXIT
    cp "$repo_dir"/shaders/*.metal "$rt_temp/"
    cp "$repo_dir"/shaders/*.h "$rt_temp/"
    cp "$repo_dir/shaders/rt_common.h" "$rt_out/original-rt_common.h"

    # Positive: ordinary reload, three frame slots, real MASK materials. A
    # magenta raster capture proves the new library became active; exact RT
    # checks must pass before AND after the pipeline-generation change.
    positive_log="$rt_out/positive.log"
    positive_report="$rt_out/positive.json"
    "${python_cmd[@]}" "$repo_dir/tools/run_checked.py" --log "$positive_log" --timeout 180 \
        --require '^HOT-RELOAD .*PASS$' --require '^RT checks [1-9][0-9]* failures 0 \| PASS$' \
        --require '^PIPELINES .*reloads 1 \(failed 0\)' -- env MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
        MTL_DEBUG_LAYER_WARNING_MODE=nslog "$app" --bench 4 --scene "$scene" --rt on --debug-rt 1 \
        --rt-probe primary --frames-in-flight 3 --frames 120 --warmup 0 --no-ui --offscreen --resolution 640x360 \
        --no-pipeline-archive --debug-hot-reload "$probe" --capture "$rt_out/positive.png" --report "$positive_report"
    "${python_cmd[@]}" - "$positive_log" "$positive_report" <<'PY_RT_POSITIVE'
import json, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
report = json.loads(Path(sys.argv[2]).read_text())
reload = re.search(r'Shaders reloaded: generation', text)
if not reload:
    raise SystemExit('RT reload positive: missing applied pipeline generation')
checks = list(re.finditer(r'^RT check frame (\d+).*\| PASS$', text, re.M))
if not any(check.start() < reload.start() for check in checks):
    raise SystemExit('RT reload positive: no exact RT check before reload')
slots = {int(check[1]) % 3 for check in checks if check.start() > reload.start()}
rt = report.get('rt', {})
if slots != {0, 1, 2} or rt.get('alpha_tests', 0) <= 0 or rt.get('opaque_alpha_tests') != 0 or rt.get('check_failures') != 0:
    raise SystemExit('RT reload positive: missing alpha work or passing checks for all three slots after reload')
print('RT reload positive: PASS (MASK traversal and all three IFT slots remain correct after reload)')
PY_RT_POSITIVE

    # Negative: poison ONLY the temporary intersection-function body. The
    # raster stays unchanged; if stale IFT handles keep executing the old
    # function, the intended counter violation never appears and this fails.
    negative_log="$rt_out/ift-poison.log"
    negative_report="$rt_out/ift-poison.json"
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog \
        "$app" --bench 4 --scene "$scene" --rt on --debug-rt 1 --rt-probe primary --frames-in-flight 3 \
        --frames 1800 --warmup 0 --no-ui --resolution 640x360 --no-pipeline-archive --shader-dir "$rt_temp" \
        --report "$negative_report" >"$negative_log" 2>&1 &
    rt_pid=$!
    # Edit only after the original alpha function has produced checked frames.
    ready=0
    for attempt in {1..150}; do
        if rg -q '^RT check frame [0-9]+ .*\| PASS$' "$negative_log"; then ready=1; break; fi
        if ! kill -0 "$rt_pid" 2>/dev/null; then break; fi
        sleep 0.1
    done
    [[ $ready -eq 1 ]] || { echo "RT IFT poison: no checked baseline before watcher edit" >&2; exit 1; }
    "${python_cmd[@]}" - "$rt_temp/rt_common.h" <<'PY_RT_POISON'
from pathlib import Path
import sys
path = Path(sys.argv[1])
source = path.read_text()
needle = '    ++payload.alphaTests;'
if source.count(needle) != 1:
    raise SystemExit('RT IFT poison: alpha function anchor changed; refusing an ambiguous edit')
poison = needle + '\n    ++payload.opaqueAlphaTests; // F9 hot-reload negative control'
replacement = path.with_suffix('.h.tmp')
replacement.write_text(source.replace(needle, poison, 1))
replacement.replace(path)
PY_RT_POISON
    # Bound the background run; keep exact raw status and an EXIT marker. A
    # watchdog termination is an error, never the expected negative result.
    timed_out=0
    for attempt in {1..1800}; do
        if ! kill -0 "$rt_pid" 2>/dev/null; then break; fi
        sleep 0.1
    done
    if kill -0 "$rt_pid" 2>/dev/null; then
        timed_out=1
        kill "$rt_pid" 2>/dev/null || true
    fi
    status=0
    wait "$rt_pid" || status=$?
    rt_pid=
    "${python_cmd[@]}" - "$negative_log" "$negative_report" "$status" "$timed_out" \
        "$repo_dir/shaders/rt_common.h" "$rt_out/original-rt_common.h" <<'PY_RT_NEGATIVE'
import json, re, sys
from pathlib import Path
log, report_path = Path(sys.argv[1]), Path(sys.argv[2])
text = log.read_text()
status, timeout = int(sys.argv[3]), bool(int(sys.argv[4]))
markers = re.findall(r'^EXIT (\d+)$', text, re.M)
exit_marker = int(markers[-1]) if markers else None
reload = re.search(r'Shaders reloaded: generation', text)
before = bool(reload and re.search(r'^RT check frame \d+ .*\| PASS$', text[:reload.start()], re.M))
slots = set()
if reload:
    for check in re.finditer(r'^RT check frame (\d+) .*opaque-alpha ([1-9][0-9]*) .*\| FAIL', text[reload.end():], re.M):
        slots.add(int(check[1]) % 3)
errors = []
if status != 1 or exit_marker != 1 or timeout: errors.append('expected raw exit 1 and EXIT 1 without timeout')
if re.search(r'failed assertion|\[ERROR\]|GPU timeout|command buffers failed|Shader Validation Error|Metal Validation Error|\berror:', text):
    errors.append('unexpected GPU, shader, build or validation failure')
if not before or slots != {0, 1, 2}: errors.append('poison was not observed on all three IFT slots after a checked baseline')
if not re.search(r'^PIPELINES .*reloads 1 \(failed 0\)', text, re.M): errors.append('expected exactly one successful reload')
if Path(sys.argv[5]).read_bytes() != Path(sys.argv[6]).read_bytes(): errors.append('original RT shader source changed')
report = json.loads(report_path.read_text()) if report_path.is_file() else {}
rt = report.get('rt', {})
if rt.get('check_failures', 0) <= 0 or rt.get('opaque_alpha_tests', 0) <= 0:
    errors.append('report does not confirm the intentional alpha counter violation')
result = {'returncode': status, 'exit_marker': exit_marker, 'expected_exit': 1, 'timed_out': timeout,
          'poison_slots': sorted(slots), 'passed': not errors, 'failures': errors}
log.with_suffix('.log.status.json').write_text(json.dumps(result, indent=2)+'\n')
if errors: raise SystemExit('RT IFT poison: FAIL: ' + '; '.join(errors))
print('RT IFT poison: PASS (new intersection function executed on all slots; intended checker exit 1)')
PY_RT_NEGATIVE
    echo "hot reload RT check: all pass"
    exit 0
fi

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
