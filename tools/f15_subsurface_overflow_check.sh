#!/bin/zsh
# F15.1 (ticket 17) forced-overflow check (macOS, GPU): the volumetric subsurface fixture (bench scenario subsurface-volumetric-sun by
# default) drawn by the meshlet/visibility path must match the same frame forced through the indexed overflow fallback
# (--debug-meshlets 1 --debug-meshlets-corrupt count draws every frame there; the meshlet self-check then reports FAIL and exit 1 by
# design).  The fallback fragment shaders write the diffuse-irradiance attachment like the resolve kernels (SSS_OUTPUT, colour attachment 6),
# so the diffusion pass sees the same input.  Tolerance: mean |difference| <= 2e-3 over the frame and over the pixels the scene covers
# (any channel of either capture above 0.01), the measure of tools/f13_overflow_emission_check.sh.
# Usage: tools/f15_subsurface_overflow_check.sh BINARY OUTDIR [SCENARIO]   (exit 0 = PASS)
B=$1; P=$2; S=${3:-subsurface-volumetric-sun}
[[ -x $B && -n $P ]] || { echo "usage: $0 BINARY OUTDIR [SCENARIO]"; exit 2; }
mkdir -p $P
BASE=(--scene procedural --bench 2 --material-scenario $S --render-path visibility --geometry-path mesh --rt off --shadows off --lighting legacy --gi off --reflections off --ao off --lighting-denoise custom --reflection-samples 8 --lighting-seed 1 --post --upscaler native --resolution 640x360 --frames 64 --warmup 0 --fixed-timestep --no-ui --no-vsync --no-gpu-timing --debug-lighting 1 --capture-linear-signal hdr --capture-linear-frame 63)
for V in normal overflow; do
  pgrep -x phosphor >/dev/null && { echo "another phosphor running"; exit 3; }
  EXTRA=(); [[ $V == overflow ]] && EXTRA=(--debug-meshlets 1 --debug-meshlets-corrupt count)
  $B $BASE $EXTRA --capture-linear $P/$V.pfm > $P/$V.log 2>&1; code=$?
  echo "$V exit $code $(grep -m1 -o 'overflow frames [0-9]*' $P/$V.log) | $(grep -c 'Pipeline archive miss' $P/$V.log) archive misses | $(tail -1 $P/$V.log)"
  # Normal: exit 0.  Overflow: the meshlet self-check fails by design (exit 1); anything else is a crash.
  if [[ $V == normal ]]; then [[ $code -eq 0 ]] || { echo FAIL; exit 1; }; else [[ $code -le 1 ]] || { echo FAIL; exit 1; }; fi
done
uv run --offline --no-project --python 3.13 --with numpy python -I - $P <<'PY'
import sys
import numpy as np
def rd(p):
    b = open(p, 'rb').read(); parts = b.split(b'\n', 3); w, h = map(int, parts[1].split())
    return np.frombuffer(parts[3][:w * h * 12], dtype='<f4').reshape(h, w, 3)
d = sys.argv[1]; a = rd(d + '/normal.pfm'); b = rd(d + '/overflow.pfm')
diff = np.abs(a - b)
covered = (np.maximum(a, b).max(axis=2) > 0.01)
frame, fmax = float(diff.mean()), float(diff.max())
obj = float(diff[covered].mean()) if covered.any() else 0.0
print('frame   mean|overflow-normal| %.6f  max %.4f' % (frame, fmax))
print('covered mean|overflow-normal| %.6f  (%d pixels)' % (obj, int(covered.sum())))
worst = max(frame, obj)
print('worst %.6f (limit 2e-3)' % worst)
print('PASS' if worst <= 2e-3 else 'FAIL'); sys.exit(0 if worst <= 2e-3 else 1)
PY
