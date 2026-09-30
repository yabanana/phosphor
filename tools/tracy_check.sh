#!/usr/bin/env bash
# tracy_check.sh -- end-to-end check of the Tracy integration (F4.2): runs the
# app with a live headless capture and verifies what arrived.
#
#   tools/tracy_check.sh [build-dir]        (configured with -DPHOSPHOR_TRACY=ON)
#
# Defaults: build/tracy.  Environment: BENCH (default 3), FRAMES (default 300),
# EXTRA_ARGS (extra app arguments).  Outputs in <build-dir>/tracy-check/.
#
# Steps
#   1. builds tracy-capture and tracy-csvexport from the Tracy source CMake
#      fetched (<build-dir>/_deps/tracy-src) into <build-dir>/tracy-tools
#      (needs network the first time: Tracy's CPM dependencies);
#   2. starts the app with TRACY_NO_EXIT=1 and waits until it listens on port
#      8086 -- starting tracy-capture earlier fails with "disconnected during
#      the initial connection handshake" (measured) -- then captures;
#   3. checks with tracy-csvexport:
#        CPU zones   every required zone present with a count > 0;
#        GPU zones   when the report JSON has "passes" (F4.1 GPU timing), each
#                    unit's mean from Tracy must match passes[].gpu_ms.mean
#                    (2% or 0.01 ms) -- FAIL on mismatch or missing zone; with
#                    no GPU zones and no "passes" -> SKIP (executor not wired);
#        memory      csvexport cannot export memory events: the capture's
#                    string table must contain the pool names ("cpu" for
#                    operator new, "GPU Geometry"/"GPU Textures"/... for
#                    GpuMemory); Tracy only records a pool name when it receives
#                    an allocation event of that pool.  Presence, not counts.
#   Prints "TRACY-CHECK PASS" or "TRACY-CHECK FAIL" and exits 0 / 1.
set -euo pipefail

build_dir=${1:-build/tracy}
bench=${BENCH:-3}
frames=${FRAMES:-300}
app="$build_dir/phosphor"
src="$build_dir/_deps/tracy-src"
tools="$build_dir/tracy-tools"
out="$build_dir/tracy-check"
capture="$tools/capture/tracy-capture"
csvexport="$tools/csvexport/tracy-csvexport"
port=8086

# Zones every run must contain.  "Pipeline compile" needs a real compile, so
# the app runs with --no-pipeline-archive.
required_cpu=("Events" "Simulation" "Scene extract" "Scene prepare" "UI" "Forward encode" "ImGui encode"
              "Graph execute" "Submit" "Bench switch" "Pipeline resolve" "Pipeline compile" "Compile job")

fail=0
note() { echo "tracy_check: $*"; }
bad()  { echo "FAIL: $*" >&2; fail=1; }

[[ -x "$app" ]] || { echo "error: $app not found (configure with -DPHOSPHOR_TRACY=ON, Release)" >&2; exit 1; }
[[ -d "$src" ]] || { echo "error: $src missing: is $build_dir configured with -DPHOSPHOR_TRACY=ON?" >&2; exit 1; }
command -v python3 >/dev/null || { echo "error: python3 not found" >&2; exit 1; }
command -v zstd >/dev/null || { echo "error: zstd not found (brew install zstd)" >&2; exit 1; }
mkdir -p "$out"

# --- 1. capture / export tools ---------------------------------------------
build_tool() { # <subdir> <binary>
    local sub=$1 bin=$2
    [[ -x "$tools/$sub/$bin" ]] && return 0
    note "building $bin from $src/$sub (first time only)"
    if ! cmake -S "$src/$sub" -B "$tools/$sub" -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DCPM_SOURCE_CACHE="$tools/cpm-cache" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 >"$out/tool-$sub.log" 2>&1 ||
       ! cmake --build "$tools/$sub" --target "$bin" >>"$out/tool-$sub.log" 2>&1; then
        tail -20 "$out/tool-$sub.log" >&2
        echo "error: building $bin failed" >&2
        exit 1
    fi
}
build_tool capture tracy-capture
build_tool csvexport tracy-csvexport

# --- 2. run + capture --------------------------------------------------------
if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "error: port $port is already in use (another Tracy client?)" >&2
    exit 1
fi
trace="$out/run.tracy"
report="$out/report.json"
applog="$out/app.log"
rm -f "$trace" "$report"

# shellcheck disable=SC2086 # EXTRA_ARGS is a whitespace-separated list
TRACY_NO_EXIT=1 env -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION \
    "$app" --bench "$bench" --frames "$frames" --warmup 0 --no-pipeline-archive --switch-every 100 \
    --report "$report" ${EXTRA_ARGS:-} >"$applog" 2>&1 &
app_pid=$!
trap 'kill "$app_pid" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
    lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1 && break
    kill -0 "$app_pid" 2>/dev/null || { cat "$applog" >&2; echo "error: the app exited before listening" >&2; exit 1; }
    sleep 0.1
done
lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1 || { echo "error: the app never listened on $port" >&2; exit 1; }

"$capture" -o "$trace" -a 127.0.0.1 -p $port -f >"$out/capture.log" 2>&1 \
    || { cat "$out/capture.log" >&2; echo "error: tracy-capture failed" >&2; exit 1; }
wait "$app_pid" || { cat "$applog" >&2; echo "error: the app failed" >&2; exit 1; }
trap - EXIT
[[ -s "$trace" ]] || { echo "error: empty capture" >&2; exit 1; }
grep -E '^(Frames|Zones|Time span|Trace size)' "$out/capture.log" || true

# --- 3. verification -----------------------------------------------------------
"$csvexport" "$trace" >"$out/cpu.csv"
"$csvexport" -g "$trace" >"$out/gpu.csv" || true

# CPU zones
for zone in "${required_cpu[@]}"; do
    count=$(python3 - "$out/cpu.csv" "$zone" <<'PY'
import csv, sys
n = 0
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        if row["name"] == sys.argv[2]:
            n += int(row["counts"])
print(n)
PY
)
    if (( count > 0 )); then
        echo "CPU zone  ok    $zone ($count)"
    else
        bad "CPU zone '$zone' missing"
    fi
done

# GPU zones vs the report JSON
gpu_result=$(python3 - "$out/gpu.csv" "$report" <<'PY'
import csv, json, sys
from collections import defaultdict

zones = defaultdict(list)
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        zones[row["name"]].append(float(row["GPU execution time"]) / 1e6)  # ns -> ms
try:
    passes = json.load(open(sys.argv[2])).get("passes") or []
except (OSError, ValueError):
    passes = []
if not zones and not passes:
    print("SKIP no GPU zones (no timed passes in the report: executor not wired to TracyGpuZones)")
    sys.exit(0)
bad = 0
if passes and not zones:
    print("FAIL the report has %d timed units but Tracy received no GPU zones" % len(passes))
    sys.exit(0)
if zones and not passes:
    print("FAIL Tracy has GPU zones (%s) but the report has no passes" % ", ".join(sorted(zones)))
    sys.exit(0)
for p in passes:
    name, expect = p["name"], p["gpu_ms"]["mean"]
    if name not in zones:
        print("FAIL GPU zone '%s' missing in Tracy" % name)
        bad += 1
        continue
    got = sum(zones[name]) / len(zones[name])
    tol = max(0.02 * abs(expect), 0.01)
    ok = abs(got - expect) <= tol
    bad += not ok
    print("%s GPU zone %-40s tracy %.4f ms (n=%d)  report %.4f ms  tol %.4f" %
          ("ok  " if ok else "FAIL", name, got, len(zones[name]), expect, tol))
extra = sorted(set(zones) - {p["name"] for p in passes})
if extra:
    print("note: Tracy GPU zones not in the report: %s" % ", ".join(extra))
print("PASS" if not bad else "FAIL %d unit(s) mismatch" % bad)
PY
)
while IFS= read -r line; do echo "gpu: $line"; done <<<"$gpu_result"
case "$gpu_result" in
    *FAIL*) bad "GPU zones do not match the report" ;;
esac

# Memory pools: string table of the trace.  A .tracy file is a 6-byte header
# followed by blocks [u32 compressed size][zstd frame]; each block is
# decompressed on its own (one stream from offset 11 stops at the first block
# boundary and misses the pools once the trace is larger than one block).
pools=$(python3 - "$trace" <<'PY' | LC_ALL=C strings -n 3 \
    | LC_ALL=C grep -aoE 'cpu|GPU (heap pages|Geometry|Textures|Upload|Transient|Render targets|Other)' | sort -u || true
import struct, subprocess, sys
data = open(sys.argv[1], "rb").read()
pos, out = 6, sys.stdout.buffer
while pos + 4 <= len(data):
    size = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    block = data[pos:pos + size]
    pos += size
    out.write(subprocess.run(["zstd", "-dc"], input=block, capture_output=True).stdout)
PY
)
if grep -qx 'cpu' <<<"$pools"; then echo "memory ok    pool 'cpu' (operator new/delete)"; else bad "memory pool 'cpu' missing"; fi
if grep -qE '^GPU ' <<<"$pools"; then
    echo "memory ok    GPU pools: $(grep -E '^GPU ' <<<"$pools" | tr '\n' ',' | sed 's/,$//')"
else
    bad "no GPU memory pool in the capture"
fi

if (( fail )); then
    echo "TRACY-CHECK FAIL (outputs in $out)"
    exit 1
fi
echo "TRACY-CHECK PASS (outputs in $out)"
