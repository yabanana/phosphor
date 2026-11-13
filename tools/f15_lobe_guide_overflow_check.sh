#!/bin/zsh
# F15.1 (ticket 06) lobe guide overflow check (macOS, GPU): the ior/specular fixture
# is run once through the meshlet/visibility path and once forced through the
# indexed overflow fallback (--debug-meshlets 1 --debug-meshlets-corrupt count
# draws every frame there; the meshlet self-check then reports FAIL and exits 1
# by design). Both runs read back the lobe guide (--debug-lobe-guide 1) and compare
# it with its CPU reconstruction; the raw guide bytes of the last checked frame
# must be identical (cmp), i.e. the overflow receivers produce the same guide as
# the visibility resolve.
# Usage: tools/f15_lobe_guide_overflow_check.sh BINARY OUTDIR   (exit 0 = PASS)
B=$1; P=$2
[[ -x $B && -n $P ]] || { echo "usage: $0 BINARY OUTDIR"; exit 2; }
mkdir -p $P
pgrep -x phosphor >/dev/null && { echo "another phosphor running"; exit 3; }
BASE=(--scene procedural --bench 2 --material-scenario specular-grid-sun --render-path visibility --geometry-path mesh --rt off --shadows off --lighting legacy --gi off --reflections ssr --ao off --lighting-denoise custom --lighting-seed 1 --post --upscaler native --resolution 640x360 --frames 64 --warmup 0 --fixed-timestep --no-ui --no-vsync --no-gpu-timing)
ok=1
for V in normal overflow; do
  pgrep -x phosphor >/dev/null && { echo "another phosphor running"; exit 3; }
  EXTRA=(); [[ $V == overflow ]] && EXTRA=(--debug-meshlets 1 --debug-meshlets-corrupt count)
  rm -f $P/$V.bin
  $B $BASE $EXTRA --debug-lobe-guide 1 --debug-lobe-guide-dump $P/$V.bin > $P/$V.log 2>&1; code=$?
  passes=$(grep -c 'LOBE-GUIDE check .* | PASS$' $P/$V.log)
  fails=$(grep -c 'LOBE-GUIDE check .* | FAIL' $P/$V.log)
  echo "$V exit $code | lobe-guide PASS lines $passes FAIL lines $fails | $(grep 'LOBE-GUIDE check' $P/$V.log | tail -1)"
  [[ $passes -ge 1 && $fails -eq 0 ]] || ok=0
  # Normal: exit 0. Overflow: the meshlet self-check fails by design (exit 1); anything else is a crash.
  if [[ $V == normal ]]; then [[ $code -eq 0 ]] || ok=0; else [[ $code -le 1 ]] || ok=0; fi
  [[ -s $P/$V.bin ]] || { echo "$V: no guide dump"; ok=0; }
done
if [[ -s $P/normal.bin && -s $P/overflow.bin ]]; then
  if cmp -s $P/normal.bin $P/overflow.bin; then echo "guide dumps identical ($(stat -f %z $P/normal.bin) bytes)"
  else echo "guide dumps DIFFER: $(cmp -l $P/normal.bin $P/overflow.bin | wc -l | tr -d ' ') bytes"; ok=0; fi
fi
[[ $ok -eq 1 ]] && { echo PASS; exit 0; } || { echo FAIL; exit 1; }
