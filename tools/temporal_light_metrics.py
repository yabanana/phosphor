#!/usr/bin/env python3
"""Independent linear-PFM temporal/ROI metrics. SOURCE ONLY / NOT EXECUTED.

No renderer, GPU, simulation or dependency installation is launched. RGB remains
linear/unexposed; no automatic resize, gamma, lag correction or threshold tuning.
Standard library works alone; NumPy, when installed, only accelerates finite scan.
"""
from __future__ import annotations
import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


@dataclass
class Image:
    width: int
    height: int
    rgb: object
    def pixel(self,x,y):
        i=(y*self.width+x)*3
        return (float(self.rgb[i]),float(self.rgb[i+1]),float(self.rgb[i+2]))


def solid(width,height,value):
    values=(value,value,value) if isinstance(value,(int,float)) else value
    return Image(width,height,array("f",values*(width*height)))


def read_pfm(path):
    with Path(path).open("rb") as stream:
        if stream.readline().strip()!=b"PF":raise ValueError("expected RGB Float32 PFM")
        def line():
            value=stream.readline()
            while value.startswith(b"#"):value=stream.readline()
            return value
        width,height=map(int,line().split());scale=float(line());raw=stream.read()
    if width<1 or height<1 or width>32768 or height>32768 or not math.isfinite(scale) or not scale:
        raise ValueError("invalid PFM header")
    if len(raw)!=width*height*12:raise ValueError("PFM payload length mismatch")
    values=array("f");values.frombytes(raw)
    if (scale<0)!=(sys.byteorder=="little"):values.byteswap()
    factor=abs(scale)
    if factor!=1:values=array("f",(x*factor for x in values))
    row=width*3;upright=array("f")
    for y in range(height-1,-1,-1):upright.extend(values[y*row:(y+1)*row])
    try:
        import numpy as np
        finite=bool(np.isfinite(np.frombuffer(upright,dtype=np.float32)).all())
    except ImportError:
        finite=all(math.isfinite(x) for x in upright)
    if not finite:raise ValueError(f"{path}: nonfinite linear pixels")
    return Image(width,height,upright)


def percentile(values,q):
    if not values:raise ValueError("empty percentile")
    values=sorted(values);position=(len(values)-1)*q
    lo=math.floor(position);hi=math.ceil(position)
    return values[lo]*(hi-position)+values[hi]*(position-lo) if hi!=lo else values[lo]


def bounds(roi,width,height):
    if "xywh" in roi:
        x,y,w,h=roi["xywh"]
    else:
        x0,y0,x1,y1=roi["normalized"]
        x,y=math.floor(x0*width),math.floor(y0*height)
        w,h=math.ceil(x1*width)-x,math.ceil(y1*height)-y
    if any(not isinstance(n,int) for n in (x,y,w,h)) or min(x,y)<0 or min(w,h)<1 or x+w>width or y+h>height:
        raise ValueError("ROI lies outside exact image extent")
    return x,y,w,h


def samples(image,region):
    x,y,w,h=region
    return array("d",(component for yy in range(y,y+h) for xx in range(x,x+w) for component in image.pixel(xx,yy)))


def spatial_envelope_residual(candidate,reference,region,radius):
    x,y,w,h=region;output=array("d")
    for yy in range(y,y+h):
        for xx in range(x,x+w):
            nearby=[reference.pixel(nx,ny)
                    for ny in range(max(0,yy-radius),min(reference.height,yy+radius+1))
                    for nx in range(max(0,xx-radius),min(reference.width,xx+radius+1))]
            current=candidate.pixel(xx,yy)
            for channel in range(3):
                lo=min(p[channel] for p in nearby);hi=max(p[channel] for p in nearby)
                output.append(current[channel]-hi if current[channel]>hi else current[channel]-lo if current[channel]<lo else 0)
    return output


def validate_config(config):
    if config.get("linear") is not True or config.get("frozen_before_run") is not True:
        raise ValueError("linear, pre-frozen ROI configuration required")
    if not config.get("rois"):raise ValueError("at least one frozen named ROI required")
    names=set()
    for roi in config["rois"]:
        if not roi.get("name") or roi["name"] in names:raise ValueError("unique ROI names required")
        names.add(roi["name"])
        if not math.isfinite(roi.get("scale",0)) or roi["scale"]<=0:raise ValueError("positive predeclared radiance scale required")
        required=("linear_rmse_p95_max","linear_rmse_max","residual_flicker_mean_max",
                  "support_ghost_fraction_max","recovery_frames_max","recovery_linear_rmse")
        for field in required:
            v=roi.get("thresholds",{}).get(field)
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0:
                raise ValueError(f"missing/nonfinite frozen threshold {field}")
    return config


def analyse_clips(reference,candidate,config,frame_ids=None):
    """Stream matched exact-sized frames; no temporal alignment fitted to data."""
    validate_config(config)
    iterator=iter(zip(reference,candidate,strict=True))
    states={r["name"]:{"definition":r,"frames":[],"prev_ref":None,"prev_error":None} for r in config["rois"]}
    ids=iter(frame_ids) if frame_ids is not None else None
    count=0;shape=None
    for ref,cur in iterator:
        frame=next(ids) if ids is not None else count
        if shape is None:shape=(ref.width,ref.height)
        if (ref.width,ref.height)!=shape or (cur.width,cur.height)!=shape:raise ValueError("exact dimensions changed or differ")
        if len(ref.rgb)!=ref.width*ref.height*3 or len(cur.rgb)!=len(ref.rgb):raise ValueError("RGB array length mismatch")
        if not all(math.isfinite(float(x)) for x in cur.rgb) or not all(math.isfinite(float(x)) for x in ref.rgb):
            raise ValueError("nonfinite clip pixel")
        for name,state in states.items():
            roi=state["definition"];region=bounds(roi,*shape);R=samples(ref,region);C=samples(cur,region)
            E=array("d",(c-r for c,r in zip(C,R)));scale=roi["scale"]
            rms=math.sqrt(math.fsum(e*e for e in E)/len(E))/scale
            signed=math.fsum(E)/len(E)/scale
            flicker=ghost=support=0.0;change_energy=0.0
            if state["prev_ref"] is not None:
                D=array("d",(p-r for p,r in zip(state["prev_ref"],R)))
                change_energy=math.fsum(d*d for d in D)
                flicker=math.fsum(abs(e-p) for e,p in zip(E,state["prev_error"]))/len(E)/scale
                if change_energy>1e-20:
                    ghost=max(0.0,math.fsum(e*d for e,d in zip(E,D))/change_energy)
                    outside=spatial_envelope_residual(cur,ref,region,int(config.get("reconstruction_support_radius",1)))
                    support=max(0.0,math.fsum(e*d for e,d in zip(outside,D))/change_energy)
            state["frames"].append({"frame":frame,"linear_rmse":rms,"signed_bias":signed,
                "residual_flicker":flicker,"old_frame_attraction":ghost,"support_ghost_fraction":support,
                "reference_change_energy":change_energy/(scale*scale)})
            state["prev_ref"],state["prev_error"]=R,E
        count+=1
    if count<2:raise ValueError("temporal metrics require at least two matching frames")
    regions={};passed=True
    for name,state in states.items():
        roi=state["definition"];rows=state["frames"];thresholds=roi["thresholds"]
        by_frame={r["frame"]:i for i,r in enumerate(rows)}
        recovery={}
        for event in config.get("events",[]):
            if event.get("rois") and name not in event["rois"]:continue
            event_frame=event["frame"]
            if event_frame not in by_frame:raise ValueError(f"missing event frame {event_frame}")
            start=by_frame[event_frame];end=min(len(rows),start+event.get("window",16))
            stable=int(event.get("stable_frames",2))
            recovered=None
            for index in range(start,end):
                if index+stable>end:break
                if all(rows[j]["linear_rmse"]<=thresholds["recovery_linear_rmse"] for j in range(index,index+stable)):
                    recovered=rows[index]["frame"]-event_frame;break
            recovery[event["name"]]=recovered
        metrics={"linear_rmse_p95":percentile([r["linear_rmse"] for r in rows],0.95),
                 "linear_rmse_max":max(r["linear_rmse"] for r in rows),
                 "residual_flicker_mean":math.fsum(r["residual_flicker"] for r in rows)/len(rows),
                 "support_ghost_fraction_max":max(r["support_ghost_fraction"] for r in rows),
                 "old_frame_attraction_max":max(r["old_frame_attraction"] for r in rows),
                 "signed_bias_mean":math.fsum(r["signed_bias"] for r in rows)/len(rows),
                 "reference_change_energy":math.fsum(r["reference_change_energy"] for r in rows),
                 "recovery_frames":recovery}
        checks={"spatial_p95":metrics["linear_rmse_p95"]<=thresholds["linear_rmse_p95_max"],
                "worst_frame":metrics["linear_rmse_max"]<=thresholds["linear_rmse_max"],
                "flicker":metrics["residual_flicker_mean"]<=thresholds["residual_flicker_mean_max"],
                "ghost_support":metrics["support_ghost_fraction_max"]<=thresholds["support_ghost_fraction_max"],
                "recovery":all(n is not None and n<=thresholds["recovery_frames_max"] for n in recovery.values()),
                "dynamic_sensitivity":not roi.get("require_change",False) or metrics["reference_change_energy"]>config.get("minimum_change_energy",1e-8)}
        passed=passed and all(checks.values())
        regions[name]={"metrics":metrics,"checks":checks,"frames":rows}
    return {"schema":1,"state":"NUMERICAL_CLIP_CHECK_ONLY","passed":passed,"frame_count":count,"extent":shape,
            "linear":True,"regions":regions,"phase_accepted":False}


def clip_files(folder):
    files=sorted(Path(folder).glob("frame-*.pfm"))
    if not files:raise ValueError("no linear frame-*.pfm captures")
    ids=[]
    for p in files:
        m=re.fullmatch(r"frame-(\d+)",p.stem)
        if not m:raise ValueError("ambiguous frame filename")
        ids.append(int(m[1]))
    if len(set(ids))!=len(ids):raise ValueError("duplicate frame IDs")
    return files,ids


def analyse_paths(reference,candidate,config):
    refs,ids=clip_files(reference);candidates,candidate_ids=clip_files(candidate)
    if ids!=candidate_ids:raise ValueError("missing/misaligned raw candidate and reference frame IDs")
    return analyse_clips((read_pfm(p) for p in refs),(read_pfm(p) for p in candidates),config,ids)


def homogeneous_fog(extinction,source,distance):
    """Independent exact SI Beer-Lambert/constant-source oracle."""
    if not math.isfinite(extinction) or not math.isfinite(distance) or min(extinction,distance)<0:
        raise ValueError("invalid homogeneous medium units")
    if len(source)!=3 or not all(math.isfinite(v) and v>=0 for v in source):raise ValueError("invalid source")
    transmittance=math.exp(-extinction*distance)
    integral=-math.expm1(-extinction*distance)/extinction if extinction else distance
    return {"transmittance":transmittance,"radiance":[v*integral for v in source]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",required=True);parser.add_argument("--candidate",required=True)
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    try:
        raw=Path(args.config).read_bytes();config=json.loads(raw)
        result=analyse_paths(args.reference,args.candidate,config)
        result["frozen_config_sha256"]=hashlib.sha256(raw).hexdigest()
        Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
        return 0 if result["passed"] else 1
    except (ValueError,OSError) as error:
        print(f"temporal light metrics failure: {error}",file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
