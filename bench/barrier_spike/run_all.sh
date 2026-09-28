#!/usr/bin/env bash
# Runs every barrier_spike case/variant as its own process, once under the
# Metal validation layers and once without, and prints a Markdown table.
#
#   bench/barrier_spike/run_all.sh [path/to/barrier_spike] [out_dir] [filter-regex]
#
# Raw stdout/stderr/exit codes are kept in out_dir (default: a temp dir, path
# printed on stderr).
set -u

BIN="${1:-build/spike/barrier_spike}"
OUT="${2:-$(mktemp -d "${TMPDIR:-/tmp}/barrier_spike.XXXXXX")}"
FILTER="${3:-.}"
TIMEOUT=180

mkdir -p "$OUT"
echo "raw output in $OUT" >&2

run_one() {   # <tag> <case> <variant> [env assignments...]
    local tag="$1" c="$2" v="$3"
    shift 3
    local base="$OUT/$c.$v.$tag"
    env "$@" "$BIN" "$c" "$v" >"$base.out" 2>"$base.err" &
    local pid=$!
    ( sleep "$TIMEOUT"; kill -9 "$pid" 2>/dev/null ) &
    local killer=$!
    wait "$pid"
    echo $? >"$base.rc"
    kill "$killer" 2>/dev/null
    wait "$killer" 2>/dev/null
}

"$BIN" --list | grep -E "$FILTER" | while read -r c v; do
    echo "running $c $v" >&2
    run_one val "$c" "$v" MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog
    run_one raw "$c" "$v" -u MTL_DEBUG_LAYER -u MTL_SHADER_VALIDATION -u MTL_DEBUG_LAYER_WARNING_MODE
done

python3 - "$OUT" "$BIN" "$FILTER" <<'PY'
import os, re, subprocess, sys

out, binary, flt = sys.argv[1], sys.argv[2], sys.argv[3]
pairs = [l.split() for l in subprocess.run([binary, "--list"], capture_output=True, text=True).stdout.splitlines()]
pairs = [p for p in pairs if re.search(flt, " ".join(p))]

def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""

def parse(base):
    o, e, rc = read(base + ".out"), read(base + ".err"), read(base + ".rc").strip()
    m = re.search(r"SPIKE \S+ \S+ result=(\S+) gpu_ms=([\d.]+) wrong_reps=(\d+)/(\d+)", o)
    return {"out": o, "err": e, "rc": rc,
            "result": m.group(1) if m else None, "ms": m.group(2) if m else "-",
            "wrong": f"{m.group(3)}/{m.group(4)}" if m else "-"}

def messages(err):
    """Distinct validation lines (banner and timestamps removed)."""
    lines = []
    for l in err.splitlines():
        l = re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+[+-]?\d* \S+\[\d+:\d+\] ", "", l.strip())
        if not l or l == "'" or re.match(r"Metal (API|GPU) Validation Enabled", l):
            continue
        if l not in lines:
            lines.append(l)
    return lines

rows = []
for c, v in pairs:
    val = parse(f"{out}/{c}.{v}.val")
    raw = parse(f"{out}/{c}.{v}.raw")
    msgs = messages(val["err"])
    if val["result"] is None:
        kind = "crash (rc=%s)" % val["rc"]
    elif msgs:
        kind = "warning/error"
    else:
        kind = "none"
    text = " / ".join(m.replace("|", "\\|") for m in msgs[:4])
    if len(text) > 500:
        text = text[:500] + "..."
    validation = kind + (f": `{text}`" if text and kind != "none" else "")
    rawstat = raw["wrong"] if raw["result"] else f"crash (rc={raw['rc']})"
    valstat = val["wrong"] if val["result"] else "-"
    ms = raw["ms"] if raw["result"] else val["ms"]
    rows.append((c, v, validation, valstat, rawstat, ms))

print("| case | variant | validation messages | wrong reps (validation) | wrong reps (no validation) | GPU ms |")
print("|---|---|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
PY
