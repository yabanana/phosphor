#!/bin/zsh
# F13.1 overflow emission check (macOS, GPU): the rough-emitter SSR frame drawn
# by the visibility path must match the same frame forced through the indexed
# overflow fallback (--debug-meshlets-corrupt count draws every frame there; the
# meshlet self-check then reports FAIL and exit 1 by design). Overflow receivers
# must carry their resolve emission (GPUDISurface::pad[1]) or the SSR source
# keeps the emitter that DI already integrates: before the fix the floor ROIs
# read +0.149 / +0.161 above the visibility path.
# Usage: tools/f13_overflow_emission_check.sh BINARY OUTDIR   (exit 0 = PASS)
B=$1; P=$2; mkdir -p $P
BASE=(--scene procedural --bench 6 --reflection-scene rough-emitter --render-path visibility --geometry-path mesh --no-ui --no-vsync --fixed-timestep --frames 64 --warmup 0 --resolution 640x360 --post --upscaler native --no-gpu-timing --capture-linear-signal hdr --capture-linear-frame 63 --rt on --shadows rt --lighting brute --gi ddgi --reflections ssr --ao off --lighting-denoise custom)
for V in normal overflow; do
  pgrep -x phosphor >/dev/null && { echo "another phosphor running"; exit 3; }
  EXTRA=(); [[ $V == overflow ]] && EXTRA=(--debug-meshlets 1 --debug-meshlets-corrupt count)
  $B $BASE $EXTRA --capture-linear $P/$V.pfm > $P/$V.log 2>&1; echo "$V exit $? $(grep -m1 -o 'overflow[^,]*' $P/$V.log) | $(tail -1 $P/$V.log)"
done
python3 - $P <<'PY'
import sys,struct
def rd(p):
    b=open(p,'rb').read(); parts=b.split(b'\n',3); w,h=map(int,parts[1].split()); sc=float(parts[2]); px=struct.unpack('<%df'%(w*h*3),parts[3][:w*h*12]); return w,h,px
d=sys.argv[1]; w,h,a=rd(d+'/normal.pfm'); _,_,b=rd(d+'/overflow.pfm')
def roi(n,box):
    x0,y0,x1,y1=box; s=0; sa=0; n_=0; mx=0
    for y in range(int(y0*h),int(y1*h)):
        yy=h-1-y  # PFM rows bottom-up
        for x in range(int(x0*w),int(x1*w)):
            i=(yy*w+x)*3
            for c in range(3): dv=b[i+c]-a[i+c]; s+=dv; sa+=a[i+c]; mx=max(mx,abs(dv)); n_+=1
    print('%-11s mean(overflow-normal) %+.5f  mean normal %.5f  max|d| %.4f'%(n,s/n_,sa/n_,mx)); return abs(s/n_)
worst=max(roi('floor-r0.3',(0.38,0.60,0.49,0.97)),roi('floor-r0.5',(0.51,0.60,0.64,0.97)),roi('frame',(0,0,1,1)))
# Tolerance: indexed and visibility shading differ only in derivatives (measured |mean| <= 4e-5).
print('PASS' if worst<=2e-3 else 'FAIL'); sys.exit(0 if worst<=2e-3 else 1)
PY
