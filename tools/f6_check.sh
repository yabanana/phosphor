#!/usr/bin/env zsh
# f6_check.sh -- F6 verification battery (docs/plans/F6.md, package F).
#
#   tools/f6_check.sh [debug-build] [release-build] [--quick] [--perf]
#
# Defaults: build, build/release.  Runs ONE GPU process at a time:
#   1. unit tests (ctest of the debug build)
#   2. visual_check of the mesh path against the indexed references
#      (build/reference-f6base for benches 1-7; bench 8 against the mesh-path
#      references build/reference-f6mesh: draw-order depth ties, see
#      docs/opt-log.md "F6"): --meshlet-cull off, frustum, two-phase, and
#      two-phase with --debug-graph-transients, --debug-async-compute,
#      --hiz-path sampler (Apple10), --force-family apple9
#   3. self-checks: --debug-meshlets PASS on benches 1, 3, 7 (classic and
#      --culling-script), 8; every negative control (id, depth, count) must
#      FAIL for its own reason; --debug-gpu-scene PASS in the mesh path and its
#      four negative controls FAIL; forced overflow draws the full frame
#   4. scripted Culling Viz event frames (occluder removed, cuts, pan, rising
#      view): two-phase == indexed pixel for pixel
#   5. bench switch + resize with UI under API + shader validation, mesh
#      two-phase + both self-checks: 0 messages
#   6. (--perf) the frozen gate preset culling-viz-f6-1920x1080, 3 replicas,
#      native / Apple9 / sampler: frame p95 <= 16.67 ms, 0 GPU allocations
# Exits 1 on any failure; every failure is printed.
setopt pipefail
dbg=${1:-build}
rel=${2:-build/release}
quick=0 perf=0
for a in "$@"; do
    [[ $a == --quick ]] && quick=1
    [[ $a == --perf ]] && perf=1
done
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
app="$dbg/phosphor"
diff_tool="$dbg/image_diff"
out="$dbg/f6-check"
mkdir -p "$out"
[[ -x $app && -x $diff_tool ]] || { echo "error: build $app and $diff_tool first" >&2; exit 2; }
reference_base=${F6_REFERENCE_BASE:-build/reference-f6base}
reference_mesh=${F6_REFERENCE_MESH:-build/reference-f6mesh}
run_app() {
    local logfile=$1 expected=$2
    shift 2
    mise exec -- python3 tools/run_checked.py --log "$logfile" --expect-exit "$expected" -- "$@"
}
failures=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }
ok() { echo "ok: $*"; }
val() { MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog "$@"; }
noise='^\[INFO\]|^BENCH|^SCENE|^MESHLET |^MESHLETS .*PASS|^GPU-SCENE .*PASS|^PIPELINES|^STARTUP|^SWITCH|^HITCH|^EXIT 0$|^GRAPH-TRANSIENTS|^ASYNC-COMPUTE|Validation Enabled|^\[WARN\]'

# 1. unit tests
if ctest --test-dir "$dbg" --output-on-failure >"$out/ctest.log" 2>&1; then ok "ctest"; else fail "ctest (see $out/ctest.log)"; fi

# 2. visual checks of the mesh path
refs_ok=1
for d in "$reference_base" "$reference_mesh"; do [[ -d $d ]] || refs_ok=0; done
if (( refs_ok )); then
    mkdir -p "$out/refs"
    for b in 1 2 3 4 5 6 7; do cp "$reference_base/bench$b.png" "$out/refs/"; done
    cp "$reference_mesh/bench8.png" "$out/refs/"
    variants=("--meshlet-cull off" "--meshlet-cull frustum" "--meshlet-cull two-phase"
              "--meshlet-cull two-phase --debug-graph-transients" "--meshlet-cull two-phase --debug-async-compute"
              "--meshlet-cull two-phase --hiz-path sampler" "--meshlet-cull two-phase --force-family apple9")
    (( quick )) && variants=("--meshlet-cull two-phase")
    for v in $variants; do
        if EXTRA_ARGS="--geometry-path mesh $v" tools/visual_check.sh "$dbg" "$out/refs" >"$out/visual.log" 2>&1; then
            ok "visual_check mesh $v"
        else
            fail "visual_check mesh $v: $(grep -E 'FAIL|differ' "$out/visual.log" | grep -v ' 0 of ' | head -3 | tr '\n' ' ')"
        fi
    done
else
    fail "references missing: take build/reference-f6base (indexed) and build/reference-f6mesh (mesh off) with visual_check --update"
fi

# 3. self-checks and negative controls
for sc in "1:--bench 1" "3:--bench 3" "7:--bench 7" "7s:--bench 7 --culling-script --resolution 1920x1080" "8:--bench 8 --instances 200000"; do
    name=${sc%%:*}; args=${sc#*:}
    run_app "$out/mesh-check-$name.log" 0 "$app" --scene procedural ${=args} --warmup 5 --frames 40 --no-ui --fixed-timestep --geometry-path mesh --meshlet-cull two-phase --debug-meshlets 9
    st=$?
    line=$(grep '^MESHLETS checks' "$out/mesh-check-$name.log")
    [[ $st == 0 && $line == *"failures 0 | PASS"* ]] && ok "meshlet self-check bench $name: $line" || fail "meshlet self-check bench $name: exit $st, $line"

done
typeset -A reason
reason=(id candidates depth pyramids count overflow)
for k in id depth count; do
    run_app "$out/mesh-negative-$k.log" 1 "$app" --scene procedural --bench 7 --warmup 5 --frames 12 --no-ui --fixed-timestep --geometry-path mesh --meshlet-cull two-phase --debug-meshlets 7 --debug-meshlets-corrupt $k
    st=$?
    log=$(cat "$out/mesh-negative-$k.log")
    if [[ $st == 0 && $log == *"| FAIL |"*"${reason[$k]}:"* ]]; then ok "negative control $k fails for the expected reason"
    else fail "negative control $k: unexpected process status or failure reason"; fi

done
for c in none delta plane command touch; do
    ca=(); [[ $c != none ]] && ca=(--debug-gpu-scene-corrupt $c)
    expected=1; [[ $c == none ]] && expected=0
    run_app "$out/scene-$c.log" $expected "$app" --scene procedural --bench 8 --instances 100000 --warmup 5 --frames 20 --no-ui --fixed-timestep --geometry-path mesh --debug-gpu-scene 5 $ca
    st=$?
    line=$(grep '^GPU-SCENE checks' "$out/scene-$c.log")
    if [[ $c == none ]]; then [[ $st == 0 && $line == *PASS* ]] && ok "gpu-scene check (mesh path)" || fail "gpu-scene check: $line"
    else [[ $st == 0 && $line == *FAIL* ]] && ok "gpu-scene negative control $c" || fail "gpu-scene negative control $c: $line"; fi

done
for b in 7 8; do
    val run_app "$out/overflow-b$b.log" 1 "$app" --scene procedural --bench $b --warmup 30 --frames 1 --no-ui --fixed-timestep --inject-input --geometry-path mesh --meshlet-cull two-phase --debug-meshlets 1 --debug-meshlets-corrupt count --capture "$out/overflow-b$b.png" || fail "overflow process bench $b"
    msgs=$(grep -Ev "$noise|^MESHLETS|^EXIT 1$" "$out/overflow-b$b.log" | grep -c .)  # the check fails on purpose
    r=$("$diff_tool" "$reference_base/bench$b.png" "$out/overflow-b$b.png")
    [[ $r == "0 of "* && $msgs == 0 ]] && ok "forced overflow bench $b: full frame via the indexed fallback" \
        || fail "forced overflow bench $b: $r, $msgs messages"
done

# 4. scripted Culling Viz event frames
frames=(179 180 181 360 361 600 720 721 960 1150)
(( quick )) && frames=(181 361)
S=(--bench 7 --culling-script --resolution 1920x1080 --fixed-timestep --frames 1 --no-ui)
for w in $frames; do
    if run_app "$out/ev$w-idx.log" 0 "$app" --scene procedural $S --warmup $w --capture "$out/ev$w-idx.png" &&        run_app "$out/ev$w-two.log" 0 "$app" --scene procedural $S --warmup $w --geometry-path mesh --meshlet-cull two-phase --capture "$out/ev$w-two.png"; then
        r=$("$diff_tool" "$out/ev$w-idx.png" "$out/ev$w-two.png")
        [[ $r == "0 of "* ]] && ok "event frame $w: two-phase == indexed" || fail "event frame $w: $r"
    else fail "event frame $w: process failed"; fi

done

# 5. switch + resize under validation
val run_app "$out/switch.log" 0 "$app" --scene procedural --frames 400 --warmup 0 --switch-every 20 --resize-every 45 --instances 100000 --culling-script --geometry-path mesh --meshlet-cull two-phase --debug-meshlets 13 --debug-gpu-scene 17
st=$?
msgs=$(grep -Ev "$noise" "$out/switch.log" | grep -c .)
[[ $st == 0 && $msgs == 0 ]] && ok "switch + resize under validation ($(grep -c 'Render graph compiled' "$out/switch.log") compiles)" \
    || fail "switch + resize: exit $st, $msgs messages (see $out/switch.log)"

# 6. gate
if (( perf )); then
    P=(--bench 7 --culling-script --resolution 1920x1080 --fixed-timestep --no-ui --warmup 120 --frames 1200)
    for cfg in "native:" "apple9:--force-family apple9" "sampler:--hiz-path sampler"; do
        n=${cfg%%:*}; a=(${=cfg#*:})
        for r in 1 2 3; do
            if ! run_app "$out/gate-$n-r$r.log" 0 "$rel/phosphor" $P --geometry-path mesh --meshlet-cull two-phase $a --report "$out/gate-$n-r$r.json"; then
                fail "gate $n r$r process failure"; continue
            fi
            p95=$(jq '.frame_ms.p95' "$out/gate-$n-r$r.json"); alloc=$(jq '.gpu_allocations' "$out/gate-$n-r$r.json")
            if (( p95 <= 16.67 && alloc == 0 )); then ok "gate $n r$r: p95 $p95 ms, allocations $alloc"
            else fail "gate $n r$r: p95 $p95 ms, allocations $alloc"; fi
        done
    done
fi

echo
(( failures == 0 )) && { echo "f6 check: all pass"; exit 0; }
echo "f6 check: $failures failure(s)"
exit 1
