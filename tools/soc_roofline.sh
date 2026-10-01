#!/usr/bin/env bash
# soc_roofline.sh -- OPT-0.4: predicted vs measured time of the testbenches'
# timed units, and the roofline per chip.
#
#   tools/soc_roofline.sh [build-dir] [results.json]
#
# For each bench 1..8: a report v3 at saturated clocks (--no-vsync
# --gpu-timing-serial --no-ui), a capture without UI whose pixels that
# differ from the clear colour give the covered pixels (fragment work),
# the forward variant the engine logs -> tools/air_ops.py --forward-variant
# (cheapest-path op counts of that variant), then soc_model predict.
# Outputs: bench/results/roofline/*, docs/img/roofline-<chip>.{svg,md}.
set -euo pipefail

build_dir=${1:-build/release}
results=${2:-bench/results/m5max-macos27.2.json}
app="$build_dir/phosphor"
tool="$build_dir/soc_model"
[[ -x "$app" && -x "$tool" ]] || { echo "error: build $app and $tool first" >&2; exit 2; }
work=$(mktemp -d)
out=bench/results/roofline
mkdir -p "$out" docs/img
slug=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["machine"]["slug"])' "$results")
model="bench/results/$(basename "$results" .json)-model.json"
"$tool" model --results "$results" --out "$model"
python3 tools/air_ops.py --out bench/results/shader_ops.json > /dev/null

preds=()
for b in 1 2 3 4 5 6 7 8; do
    caffeinate -d -i "$app" --bench "$b" --frames 300 --warmup 60 --no-vsync --gpu-timing-serial --no-ui \
        --report "$out/report-bench$b.json" > "$work/run$b.log" 2>&1
    caffeinate -d -i "$app" --bench "$b" --frames 30 --warmup 30 --fixed-timestep --no-ui \
        --capture "$work/cap$b.png" > "$work/cap$b.log" 2>&1
    sips -s format bmp "$work/cap$b.png" --out "$work/cap$b.bmp" > /dev/null
    # Variant the forward pass ran with: "light types 0x2, emissive 0, debug mode 0".
    read -r lt em dbg < <(sed -n 's/.*Forward variant [0-9]* requested: light types \(0x[0-9a-f]*\), emissive \([01]\), debug mode \([0-9]*\).*/\1 \2 \3/p' \
        "$work/run$b.log" | tail -1)
    python3 tools/air_ops.py --forward-variant "$lt" "$em" "$dbg" --out "$work/variant$b.json" > /dev/null
    python3 - "$work/cap$b.bmp" "$out/work-bench$b.json" bench/results/shader_ops.json "$work/variant$b.json" \
        "$work/ops$b.json" <<'EOF'
import json, struct, sys
bmp, work_out, generic, variant, ops_out = sys.argv[1:]
d = open(bmp, "rb").read()
off, = struct.unpack_from("<I", d, 10)
w, h = struct.unpack_from("<ii", d, 18)
bpp, = struct.unpack_from("<H", d, 28)
B = bpp // 8
stride = (w * B + 3) & ~3
cov = 0
for y in range(abs(h)):
    row = d[off + y * stride: off + y * stride + w * B]
    for x in range(0, w * B, B):
        # Clear colour of the forward pass (0.02, 0.025, 0.035) in sRGB: 39, 44, 52 (BGR in the BMP).
        if abs(row[x + 2] - 39) > 2 or abs(row[x + 1] - 44) > 2 or abs(row[x] - 52) > 2:
            cov += 1
json.dump({"Forward": {"fragments": cov}}, open(work_out, "w"))
ops = json.load(open(generic))
ops["functions"].update(json.load(open(variant))["functions"])
json.dump(ops, open(ops_out, "w"))
print("covered pixels %d of %d" % (cov, w * abs(h)))
EOF
    "$tool" predict --model "$model" --report "$out/report-bench$b.json" --ops "$work/ops$b.json" \
        --work "$out/work-bench$b.json" --out "$out/pred-bench$b.json"
    preds+=(--pred "$out/pred-bench$b.json")
done
python3 tools/roofline.py "${preds[@]}" --out-dir docs/img
python3 - "$out" <<'EOF'
import json, sys, glob
bad = 0
for f in sorted(glob.glob(sys.argv[1] + "/pred-bench*.json")):
    for u in json.load(open(f))["units"]:
        ok = u["predicted_ms"] <= u["measured_ms"]
        bad += not ok
        print("%-40s measured %8.3f ms  lower bound %8.4f ms  %s" % (f.split("/")[-1] + " " + u["name"], u["measured_ms"],
                                                                  u["predicted_ms"], "ok" if ok else "VIOLATED"))
sys.exit(1 if bad else 0)
EOF
rm -rf "$work"
echo "soc_roofline: docs/img/roofline-$slug.svg"
