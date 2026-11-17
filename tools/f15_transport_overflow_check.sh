#!/bin/zsh
# F15.1 (ticket 08) forced-overflow check (macOS, GPU): the ior x specular grid with the SSR specular transport
# (bench scenario specular-grid-env, the ticket's fixture) drawn by the meshlet/visibility path must match the same
# frame forced through the indexed overflow fallback (--debug-meshlets 1 --debug-meshlets-corrupt count draws every
# frame there; the meshlet self-check then reports FAIL and exit 1 by design).  Overflow receivers must produce the
# lobe guide like the meshlet path (ticket 06) or the transport and the composite would fall back to the base
# lobe on the receivers.  Same tolerance as tools/f13_overflow_emission_check.sh: mean |difference| <= 2e-3 over the
# frame and over the ROI of every sphere (a disc of 70 % of the sphere radius around its projected centre).
# Usage: tools/f15_transport_overflow_check.sh BINARY OUTDIR   (exit 0 = PASS)
B=$1; P=$2
[[ -x $B && -n $P ]] || { echo "usage: $0 BINARY OUTDIR"; exit 2; }
mkdir -p $P
BASE=(--scene procedural --bench 2 --material-scenario specular-grid-env --render-path visibility --geometry-path mesh --rt off --shadows off --lighting legacy --gi off --reflections ssr --ao off --lighting-denoise custom --reflection-samples 8 --lighting-seed 1 --post --upscaler native --resolution 640x360 --frames 64 --warmup 0 --fixed-timestep --no-ui --no-vsync --no-gpu-timing --debug-lighting 1 --capture-linear-signal hdr --capture-linear-frame 63)
for V in normal overflow; do
  pgrep -x phosphor >/dev/null && { echo "another phosphor running"; exit 3; }
  EXTRA=(); [[ $V == overflow ]] && EXTRA=(--debug-meshlets 1 --debug-meshlets-corrupt count)
  $B $BASE $EXTRA --capture-linear $P/$V.pfm > $P/$V.log 2>&1; code=$?
  echo "$V exit $code $(grep -m1 -o 'overflow frames [0-9]*' $P/$V.log) | $(grep -c 'Pipeline archive miss' $P/$V.log) archive misses | $(tail -1 $P/$V.log)"
  # Normal: exit 0.  Overflow: the meshlet self-check fails by design (exit 1); anything else is a crash.
  if [[ $V == normal ]]; then [[ $code -eq 0 ]] || { echo FAIL; exit 1; }; else [[ $code -le 1 ]] || { echo FAIL; exit 1; }; fi
done
python3 - $P <<'PY'
import sys, struct, math
def rd(p):
    b = open(p, 'rb').read(); parts = b.split(b'\n', 3); w, h = map(int, parts[1].split())
    px = struct.unpack('<%df' % (w * h * 3), parts[3][:w * h * 12]); return w, h, px
d = sys.argv[1]; w, h, a = rd(d + '/normal.pfm'); _, _, b = rd(d + '/overflow.pfm')
def mean_abs(x0, y0, x1, y1, disc=None):
    s = 0.0; n = 0; mx = 0.0
    for y in range(max(0, int(y0)), min(h, int(y1) + 1)):
        yy = h - 1 - y  # PFM rows are bottom-up
        for x in range(max(0, int(x0)), min(w, int(x1) + 1)):
            if disc and (x - disc[0]) ** 2 + (y - disc[1]) ** 2 > disc[2] ** 2: continue
            i = (yy * w + x) * 3
            for c in range(3):
                dv = abs(b[i + c] - a[i + c]); s += dv; mx = max(mx, dv); n += 1
    return s / n, mx
# Camera of MaterialValidation::getDefaultCamera: (0, 2, 5.5) looking at (0, 2, 0), 60 degree vertical field of view;
# spheres at x = -2 + column, y = row (5 x 5 grid), radius 0.4.
aspect = w / h; t = math.tan(math.radians(30)); dist = 5.5
worst = 0.0
frame, fmax = mean_abs(0, 0, w - 1, h - 1)
print('frame        mean|overflow-normal| %.6f  max %.4f' % (frame, fmax)); worst = frame
for row in range(5):
    for col in range(5):
        X, Y = -2.0 + col, float(row) - 2.0
        cx = (0.5 + 0.5 * X / (dist * t * aspect)) * w; cy = (0.5 - 0.5 * Y / (dist * t)) * h
        r = 0.4 * 0.7 / (dist * t) * 0.5 * h
        m, mx = mean_abs(cx - r, cy - r, cx + r, cy + r, (cx, cy, r))
        print('sphere r%dc%d   mean|d| %.6f  max %.4f' % (row, col, m, mx)); worst = max(worst, m)
print('worst mean |d| %.6f (limit 2e-3)' % worst)
print('PASS' if worst <= 2e-3 else 'FAIL'); sys.exit(0 if worst <= 2e-3 else 1)
PY
