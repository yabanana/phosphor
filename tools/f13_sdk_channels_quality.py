#!/usr/bin/env python3
"""CPU-only analysis of existing authored SDK channels captures.

Requires a protocol frozen before examining SDK pixels. Never launches a
renderer, builds, installs packages, aligns clips, crops from image intensity,
changes gates or overwrites evidence. NumPy is an analysis dependency only.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from temporal_light_metrics import Image,analyse_clips,read_pfm,validate_config

CAPS={"linear_rmse_p95_max":.06,"linear_rmse_max":.15,"residual_flicker_mean_max":.02,
      "support_ghost_fraction_max":.25,"recovery_frames_max":4,"recovery_linear_rmse":.06}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reference(frame,width=128,height=96):
    out=np.full((height,width,3),.125,dtype=np.float64)
    center=width//4+frame%(width//2)
    out[height//2-3:height//2+4,center-3:center+4]=(.75,.5,.25)
    return out

def envelope(ref,radius):
    h,w=ref.shape[:2];padded=np.pad(ref,((radius,radius),(radius,radius),(0,0)),mode="edge")
    low=np.full_like(ref,np.inf);high=np.full_like(ref,-np.inf)
    for y in range(2*radius+1):
        for x in range(2*radius+1):
            value=padded[y:y+h,x:x+w];low=np.minimum(low,value);high=np.maximum(high,value)
    return low,high

def analyse(references,candidates,config):
    """Vectorized equivalent of the existing fixed F8/F13 metric definitions."""
    validate_config(config)
    regions={};passed=True;radius=config["reconstruction_support_radius"]
    envelopes=[envelope(r,radius) for r in references]
    for roi in config["rois"]:
        x,y,w,h=roi["xywh"];region=np.s_[y:y+h,x:x+w];rows=[];previous_ref=previous_error=None
        for frame,(ref,candidate) in enumerate(zip(references,candidates,strict=True)):
            if ref.shape!=candidate.shape or not np.isfinite(candidate).all():raise ValueError("invalid exact-sized finite frame")
            R=ref[region];C=candidate[region];E=C-R;scale=roi["scale"]
            flicker=ghost=support=change=0.
            if previous_ref is not None:
                D=previous_ref-R;change=float(np.sum(D*D))
                flicker=float(np.mean(np.abs(E-previous_error)))/scale
                if change>1e-20:
                    ghost=max(0.,float(np.sum(E*D))/change)
                    low,high=envelopes[frame];outside=C-np.clip(C,low[region],high[region])
                    support=max(0.,float(np.sum(outside*D))/change)
            rows.append({"frame":frame,"linear_rmse":float(np.sqrt(np.mean(E*E)))/scale,
                         "signed_bias":float(np.mean(E))/scale,"residual_flicker":flicker,
                         "old_frame_attraction":ghost,"support_ghost_fraction":support,
                         "reference_change_energy":change/(scale*scale)})
            previous_ref,previous_error=R,E
        limits=roi["thresholds"];recovery={}
        for event in config["events"]:
            start=event["frame"];end=min(len(rows),start+event["window"]);stable=event["stable_frames"]
            recovery[event["name"]]=next((i-start for i in range(start,end-stable+1)
                if all(rows[j]["linear_rmse"]<=limits["recovery_linear_rmse"] for j in range(i,i+stable))),None)
        metrics={"linear_rmse_p95":float(np.quantile([r["linear_rmse"] for r in rows],.95)),
                 "linear_rmse_max":max(r["linear_rmse"] for r in rows),
                 "residual_flicker_mean":float(np.mean([r["residual_flicker"] for r in rows])),
                 "support_ghost_fraction_max":max(r["support_ghost_fraction"] for r in rows),
                 "old_frame_attraction_max":max(r["old_frame_attraction"] for r in rows),
                 "signed_bias_mean":float(np.mean([r["signed_bias"] for r in rows])),
                 "reference_change_energy":sum(r["reference_change_energy"] for r in rows),"recovery_frames":recovery}
        checks={"spatial_p95":metrics["linear_rmse_p95"]<=limits["linear_rmse_p95_max"],
                "worst_frame":metrics["linear_rmse_max"]<=limits["linear_rmse_max"],
                "flicker":metrics["residual_flicker_mean"]<=limits["residual_flicker_mean_max"],
                "ghost_support":metrics["support_ghost_fraction_max"]<=limits["support_ghost_fraction_max"],
                "recovery":all(v is not None and v<=limits["recovery_frames_max"] for v in recovery.values()),
                "dynamic_sensitivity":not roi.get("require_change") or metrics["reference_change_energy"]>config["minimum_change_energy"]}
        regions[roi["name"]]={"metrics":metrics,"checks":checks,"frames":rows};passed&=all(checks.values())
    return {"passed":passed,"frame_count":len(candidates),"regions":regions,"phase_accepted":False}

def self_test():
    cfg={"linear":True,"frozen_before_run":True,"reconstruction_support_radius":1,"minimum_change_energy":1e-8,
         "events":[{"name":"change","frame":2,"window":4,"stable_frames":2}],
         "rois":[{"name":"all","xywh":[0,0,12,8],"scale":1,"require_change":True,"thresholds":CAPS}]}
    refs=[]
    for f in range(6):
        a=np.full((8,12,3),.125);a[2:5,2+f:4+f]=(.75,.5,.25);refs.append(a)
    candidates=[refs[0]]+refs[:-1]
    fast=analyse(refs,candidates,cfg)["regions"]["all"]
    images=lambda values:[Image(12,8,array("f",x.ravel())) for x in values]
    slow=analyse_clips(images(refs),images(candidates),cfg)["regions"]["all"]
    assert fast["checks"]==slow["checks"]
    for a,b in zip(fast["frames"],slow["frames"]):
        for key in a:assert abs(a[key]-b[key])<1e-12,(key,a[key],b[key])
    for key,a in fast["metrics"].items():
        b=slow["metrics"][key]
        assert a==b if isinstance(a,dict) else abs(a-b)<1e-12,(key,a,b)
    print("Vectorized metric agrees with existing scalar definitions; CPU only")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path);parser.add_argument("--output",type=Path)
    parser.add_argument("--self-test",action="store_true");args=parser.parse_args()
    if args.self_test:self_test();return 0
    if not args.protocol or not args.output:parser.error("--protocol and new --output required")
    protocol=json.loads(args.protocol.read_text());config=protocol["config"]
    if protocol["state"]!="FROZEN_BEFORE_SDK_PIXEL_READ" or protocol["extent"]!=[128,96] or protocol["preExposure"]!=1 or protocol["expected_frames"]!=list(range(96)):
        raise ValueError("unsupported frozen fixture contract")
    if config["reconstruction_support_radius"]!=1 or any(r["thresholds"]!=CAPS for r in config["rois"]):raise ValueError("original caps/support required")
    if [r["xywh"] for r in config["rois"]]!=[[0,0,128,96],[28,44,72,9]]:raise ValueError("only full image and geometry-defined swept ROI permitted")
    if args.output.exists():raise ValueError("refuse existing output directory")
    args.output.mkdir(parents=True)
    result={"schema":"phosphor.sdk-channels-quality.v1","protocol_sha256":sha(args.protocol),
            "state":"CPU_ANALYSIS_RUNNING","phase_accepted":False,"production_policy_promoted":False}
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    root=Path(protocol["captures"]);refs=[reference(f) for f in range(96)];sdk=[];physical=[];artifacts=[];provenance=None
    # Metadata validation happens before pixels. The analytic source is authored
    # from its geometry/time contract, never inferred from candidate images.
    records=[]
    for frame in range(96):
        path=root/f"frame-{frame:06d}-view-0.json";r=json.loads(path.read_text())
        if not (r.get("actual_sdk_encoded") and r.get("finite") and r.get("channels_passed") and r.get("passed")):raise ValueError("missing successful native frame")
        if [r[k] for k in ("input_width","input_height","output_width","output_height")]!=[128,96,128,96] or r["preExposure"]!=1 or r["phase"]!=frame%64 or r["frame"]!=frame or r["view"]!=0 or r["scenario"]!="channels":raise ValueError("frame contract mismatch")
        if provenance is None:provenance=r["provenance"]
        if r["provenance"]!=provenance:raise ValueError("source/binary/manifest changed across clip")
        records.append(r);artifacts.append({"path":str(path),"sha256":sha(path)})
    for frame in range(96):
        for suffix,frames in (("sdk",sdk),("physical",physical)):
            path=root/f"frame-{frame:06d}-view-0-{suffix}.pfm";image=read_pfm(path)
            if (image.width,image.height)!=(128,96):raise ValueError("PFM extent disagrees with actual record")
            frames.append(np.asarray(image.rgb,dtype=np.float64).reshape(96,128,3));artifacts.append({"path":str(path),"sha256":sha(path)})
    controls={"exact-positive":refs,"constant-background-negative":[np.full_like(r,.125) for r in refs],
              "one-frame-lag-negative":[refs[0]]+refs[:-1],"opposite-direction-negative":[reference((-f)%64) for f in range(96)]}
    result["controls"]={name:analyse(refs,clip,config) for name,clip in controls.items()}
    result["control_sensitivity_passed"]=result["controls"]["exact-positive"]["passed"] and all(not r["passed"] for name,r in result["controls"].items() if name!="exact-positive")
    result["sdk"]=analyse(refs,sdk,config);result["physical"]=analyse(refs,physical,config)
    diagnostic=[];region=np.s_[44:53,28:100];xx,yy=np.meshgrid(np.arange(28,100),np.arange(44,53))
    for f,cur in enumerate(sdk):
        cx=32+f%64;patch=cur[45:52,cx-3:cx+4];contrast=np.mean((patch-.125)/np.array([.625,.375,.125]),axis=(0,1))
        weight=np.maximum(cur[region][:,:,0]-.125,0);mass=float(weight.sum())
        centroid=[float((weight*xx).sum()/mass),float((weight*yy).sum()/mass)] if mass else None
        diagnostic.append({"frame":f,"authored_center":[cx,48],"patch_contrast_ratio_rgb":contrast.tolist(),"positive_red_centroid":centroid})
    result.update(state="CPU_ANALYSIS_COMPLETE",provenance=provenance,input_artifacts=artifacts,
                  sdk_physical_max_absolute_difference=max(float(np.max(np.abs(a-b))) for a,b in zip(sdk,physical)),
                  diagnostics=diagnostic,scope=protocol["limitations"])
    result["passed"]=result["control_sensitivity_passed"] and result["sdk"]["passed"] and result["physical"]["passed"]
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"passed":result["passed"],"controls":{n:r["passed"] for n,r in result["controls"].items()},
                      "sdk":{n:r["metrics"] for n,r in result["sdk"]["regions"].items()}},indent=2))
    return 0 if result["passed"] else 1

if __name__=="__main__":sys.exit(main())
