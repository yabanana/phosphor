#!/usr/bin/env python3
"""CPU-only oracle for actual wide-emission raster captures; never launches the renderer.

The fixture fills the camera with opaque emission [368640,128,64]. No tone map,
exposure fit, percentile clipping, resize or foreground mask is allowed here.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from temporal_light_metrics import read_pfm
TARGET=(368640.0,128.0,64.0)
# Budget of 16 FP32 operations for the constant residual/composition path.
# These absolute radiance bounds are fixed before GPU execution, not fitted.
GAMMA16=(16*2**-24)/(1-16*2**-24)

def evaluate(rgb):
    if not rgb or len(rgb)%3:raise ValueError("nonempty complete RGB image required")
    worst=[0.0]*3;lo=[math.inf]*3;hi=[-math.inf]*3;sums=[0.0]*3;failures=0
    for index,value in enumerate(rgb):
        c=index%3;target=TARGET[c]
        if not math.isfinite(value):failures+=1;worst[c]=math.inf;continue
        delta=abs(value-target);worst[c]=max(worst[c],delta);lo[c]=min(lo[c],value);hi[c]=max(hi[c],value);sums[c]+=value
        failures+=delta>GAMMA16*target
    finite=lambda values:[v if math.isfinite(v) else None for v in values]
    return {"passed":failures==0,"pixels":len(rgb)//3,"failed_components":failures,
        "target":TARGET,"gamma16":GAMMA16,"absolute_bounds":[GAMMA16*v for v in TARGET],
        "worst_absolute_error":finite(worst),"minimum":finite(lo),"maximum":finite(hi),
        "mean":[v/(len(rgb)//3) for v in sums],"half_clipped_or_nonfinite_is_failure":True}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("capture",type=Path);p.add_argument("--report",type=Path)
    a=p.parse_args(argv);image=read_pfm(a.capture);result=evaluate(image.rgb)
    result.update({"width":image.width,"height":image.height,"capture":str(a.capture.resolve()),"reference":"analytic opaque emission; no incoming radiance"})
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if a.report:
        if a.report.exists():p.error("refuse reference result overwrite")
        a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(text)
    print(text,end="");return int(not result["passed"])
if __name__=="__main__":raise SystemExit(main())
