#!/usr/bin/env python3
"""Minimum physical custom-AO temporal control. Default PLAN ONLY / NOT EXECUTED.

--run is the only renderer execution switch. No builds, downloads, retries,
SDK/lifetime/GI adoption, same-renderer reference or threshold fitting. The
reference is an independent exact cosine-disk probability on known receivers.
Only ROI/halo pixels have this oracle; no fabricated full-frame reference image.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from f13_f14_check import gpu_lock,json_bytes,sha,source_hash,GPU_ERROR
from run_checked import run_checked
from temporal_light_metrics import read_pfm,percentile

PROTOCOL_PATH=Path(__file__).resolve().parent/"testdata"/"f13_ao_temporal_protocol.json"
CAPS={"linear_rmse_p95_max":.06,"linear_rmse_max":.15,"residual_flicker_mean_max":.02,
      "support_ghost_fraction_max":.25,"recovery_frames_max":4,"recovery_linear_rmse":.06}
CHECK=re.compile(r"LIGHTING check frame (\d+)\s*\|\s*(PASS|FAIL)")

def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def integer(x):return isinstance(x,int) and not isinstance(x,bool) and x>=0
def vector(value):
    if not isinstance(value,list) or len(value)!=3 or not all(number(x) for x in value):raise ValueError("finite three-component vector required")
    return tuple(value)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def unit(v):
    length=math.sqrt(dot(v,v))
    if not math.isfinite(length) or length<1e-12:raise ValueError("degenerate camera basis")
    return tuple(x/length for x in v)
def close(a,b,tolerance):return len(a)==len(b) and all(abs(x-y)<=tolerance for x,y in zip(a,b))

def visibility(receiver_x,wall_x,radius):
    """Infinite perpendicular wall, either side; coincident receiver is invalid."""
    if not all(number(x) for x in (receiver_x,wall_x,radius)) or radius<=0:raise ValueError("invalid AO metre geometry")
    distance=abs(wall_x-receiver_x)
    if distance<1e-12:raise ValueError("receiver coincides with wall")
    if distance>=radius:return 1.0
    a=distance/radius
    cap=math.acos(a)-a*math.sqrt(max(0,1-a*a))
    return 1-cap/math.pi

def receiver(camera,x,y,width,height,plane_z):
    p=vector(camera["position"]);front=unit(vector(camera["front"]));up=unit(vector(camera["up"]))
    right=unit(cross(front,up));up=unit(cross(right,front));fov=camera["fov_y"]
    if not number(fov) or not 0<fov<math.pi:raise ValueError("camera fov_y must be radians in (0,pi)")
    horizontal=(2*(x+.5)/width-1)*(width/height)*math.tan(fov/2)
    vertical=(1-2*(y+.5)/height)*math.tan(fov/2)
    ray=tuple(front[i]+right[i]*horizontal+up[i]*vertical for i in range(3))
    if abs(ray[2])<1e-12:raise ValueError("pixel ray parallel to receiver")
    t=(plane_z-p[2])/ray[2]
    if t<=0:raise ValueError("plane is behind pixel camera ray")
    return tuple(p[i]+t*ray[i] for i in range(3))

def load_protocol(path=PROTOCOL_PATH):
    raw=Path(path).read_bytes();p=json.loads(raw)
    if p.get("schema")!="phos.f13-ao-temporal-protocol.v1" or p.get("state")!="NOT_EXECUTED" or p.get("thresholds")!=CAPS:
        raise ValueError("incompatible frozen AO protocol/caps")
    if (p.get("frames"),p.get("width"),p.get("height"),p.get("step_ordinal"))!=(64,128,96,32):raise ValueError("fixed physical fixture protocol required")
    if p.get("roi",{}).get("xywh")!=[48,38,16,20] or p["roi"].get("scale")!=1:raise ValueError("fixed receiver ROI/unit visibility scale required")
    if p.get("history",{}).get("atrous_iterations")!=3 or p["history"].get("spatial_support_radius_pixels")!=14:raise ValueError("actual three-step atrous support required")
    return p,hashlib.sha256(raw).hexdigest()

def physical_reference(metadata,protocol):
    """Exact ROI plus halo; reject unsupported geometry instead of fitting it."""
    p=protocol["physics"];cam=metadata["camera"];plane=metadata["plane"];ordinal=metadata["fixture_ordinal"]
    tol=p["camera_tolerance"]
    for key,expected in (("position",p["camera_position"]),("front",p["camera_front"]),("up",p["camera_up"])):
        if not close(vector(cam[key]),expected,tol):raise ValueError("actual camera differs from fixed physical receiver contract")
    if not number(plane.get("z")) or abs(plane["z"]-p["plane_z_m"])>tol or not number(plane.get("size")) or abs(plane["size"]-p["plane_full_size_m"])>tol:
        raise ValueError("actual receiver plane differs from physical contract")
    if not close(vector(plane["normal"]),p["plane_normal"],tol):raise ValueError("receiver is not the +Z geometric plane")
    expected_wall=p["wall_x_m"][int(ordinal>=protocol["step_ordinal"])]
    wall=metadata["wall_x"];radius=metadata["radius"]
    if not number(wall) or abs(wall-expected_wall)>tol or not number(radius) or abs(radius-p["radius_m"])>tol:raise ValueError("actual wall step/radius differs from frozen metre geometry")
    x0,y0,w,h=protocol["roi"]["xywh"];halo=protocol["history"]["spatial_support_radius_pixels"]
    width,height=metadata["width"],metadata["height"]
    if x0-halo<0 or y0-halo<0 or x0+w+halo>width or y0+h+halo>height:raise ValueError("ROI+actual filter footprint outside frame")
    values={};half_plane=plane["size"]*.5;half_y,half_z=p["wall_half_extent_yz_m"]
    for y in range(y0-halo,y0+h+halo):
        for x in range(x0-halo,x0+w+halo):
            point=receiver(cam,x,y,width,height,plane["z"])
            if abs(point[0])>=half_plane or abs(point[1])>=half_plane:raise ValueError("ROI/halo does not identify an interior known plane receiver")
            if point[0]>=wall or cam["position"][0]>=wall:raise ValueError("selected receiver/primary camera is outside the declared left-wall domain")
            if abs(point[1])+radius>=half_y or abs(point[2])+radius>=half_z:raise ValueError("finite wall cannot support the infinite-wall analytic oracle")
            values[(x,y)]=visibility(point[0],wall,radius)
    return values

def validate_metadata(records,protocol,provenance):
    if len(records)!=protocol["frames"]:raise ValueError("exactly64 immutable metadata records required")
    last_frame=None;baseline_identity=None;post_identity=None;references=[]
    for ordinal,m in enumerate(records):
        if m.get("schema")!=protocol["metadata_schema"] or m.get("fixture_ordinal")!=ordinal:raise ValueError("missing/misaligned fixture ordinal")
        if not integer(m.get("frame")) or last_frame is not None and m["frame"]!=last_frame+1:raise ValueError("actual GPU submission IDs must be contiguous")
        last_frame=m["frame"]
        for key,value in (("view",0),("signal",9),("width",128),("height",96)):
            if m.get(key)!=value:raise ValueError("actual capture "+key+" mismatch")
        if any(m.get("provenance",{}).get(key)!=value for key,value in provenance.items()):raise ValueError("sidecar differs from frozen source/binary/metallib/manifest")
        h=m.get("history",{})
        for key in ("frame","view","pixels","valid","reused","max_length","invalid","invalid_output","epoch","revision","atrous_iterations"):
            if not integer(h.get(key)):raise ValueError("missing actual completed GPU history field: "+key)
        if h["frame"]!=m["frame"] or h["view"]!=m["view"] or h["pixels"]!=128*96:raise ValueError("GPU history readback belongs to another frame/view/extent")
        if h["valid"]!=h["pixels"] or not 0<=h["reused"]<=h["valid"] or not 1<=h["max_length"]<=32:
            raise ValueError("actual history count/length bounds invalid")
        if h["invalid"] or h["invalid_output"] or h["atrous_iterations"]!=3 or not isinstance(h.get("reset"),bool):raise ValueError("invalid completed history/output/filter/reset contract")
        identity=(h["epoch"],h["revision"])
        if 16<=ordinal<=31:
            if h["reset"] or not h["reused"] or h["max_length"]<=1:raise ValueError("baseline never exercised genuine temporal history reuse")
            if baseline_identity is not None and identity!=baseline_identity:raise ValueError("stable baseline history identity changed")
            baseline_identity=identity
        if ordinal==32:
            if not h["reset"] or h["reused"] or h["max_length"]!=1 or identity==baseline_identity:raise ValueError("wall step did not produce a real completed history reset/new identity")
            post_identity=identity
        if ordinal>32:
            if h["reset"] or identity!=post_identity or not h["reused"] or h["max_length"]<=1:raise ValueError("stable post-step history never resumed valid reuse")
        references.append(physical_reference(m,protocol))
    camera=records[0]["camera"]
    if any(m["camera"]!=camera for m in records[1:]):raise ValueError("camera changed during a supposedly fixed-receiver control")
    return references

def roi_values(values,protocol):
    x,y,w,h=protocol["roi"]["xywh"]
    return [values[(xx,yy)] for yy in range(y,y+h) for xx in range(x,x+w)]

def persistent_recovery(rows,start,threshold):
    """First recovery which remains valid through the complete post-step clip."""
    for i in range(start,len(rows)-1):
        if all(row["linear_rmse"]<=threshold for row in rows[i:]):return i-start
    return None

def quality(images,references,protocol):
    if len(images)!=64 or len(references)!=64:raise ValueError("64 actual scalar images and analytic states required")
    rows=[];previous_reference=previous_error=None;x0,y0,w,h=protocol["roi"]["xywh"]
    radius=protocol["history"]["spatial_support_radius_pixels"];step=protocol["step_ordinal"]
    anchor=roi_values(references[protocol["identifiability"]["event_anchor_ordinal"]],protocol)
    for ordinal,(image,reference) in enumerate(zip(images,references)):
        if (image.width,image.height)!=(128,96):raise ValueError("exact native scalar capture extent required")
        if len(image.rgb)!=128*96*3:raise ValueError("actual scalar PFM payload incomplete")
        for i in range(0,len(image.rgb),3):
            a,b,c=(float(image.rgb[i+j]) for j in range(3))
            if not all(math.isfinite(v) and 0<=v<=1 for v in (a,b,c)) or a!=b or a!=c:
                raise ValueError("actual AO must be finite visibility[0,1], identically replicated RGB")
        expected=roi_values(reference,protocol)
        actual=[image.pixel(x,y)[0] for y in range(y0,y0+h) for x in range(x0,x0+w)]
        error=[a-b for a,b in zip(actual,expected)];rms=math.sqrt(math.fsum(e*e for e in error)/len(error))
        flicker=ghost=event_ghost=0.0
        if previous_reference is not None:
            change=[a-b for a,b in zip(previous_reference,expected)];energy=math.fsum(d*d for d in change)
            flicker=math.fsum(abs(a-b) for a,b in zip(error,previous_error))/len(error)
            if energy>1e-20:
                outside=[]
                for y in range(y0,y0+h):
                    for x in range(x0,x0+w):
                        nearby=[reference[(xx,yy)] for yy in range(y-radius,y+radius+1) for xx in range(x-radius,x+radius+1)]
                        value=image.pixel(x,y)[0];lo,hi=min(nearby),max(nearby)
                        outside.append(value-hi if value>hi else value-lo if value<lo else 0)
                ghost=max(0,math.fsum(e*d for e,d in zip(outside,change))/energy)
        if ordinal>=step:
            for current,old,e in zip(expected,anchor,error):
                d=old-current
                if abs(d)<1e-12:raise ValueError("event-anchored receiver pixel has no independent reference change")
                event_ghost=max(event_ghost,max(0,e*d/(d*d)))
        rows.append({"fixture_ordinal":ordinal,"linear_rmse":rms,"residual_flicker":flicker,"support_ghost_fraction":ghost,
            "event_old_state_fraction_max":event_ghost})
        previous_reference,previous_error=expected,error
    baseline=roi_values(references[31],protocol);post=roi_values(references[32],protocol)
    change=math.fsum(abs(a-b) for a,b in zip(baseline,post))/len(baseline)
    if min(baseline)<protocol["identifiability"]["minimum_baseline_visibility"] or change<protocol["identifiability"]["minimum_step_reference_mean_change"]:
        raise ValueError("physical AO control lacks the frozen reference energy/change; no fitting or relaxed gate")
    measured=rows[16:64];recovered=persistent_recovery(rows,step,CAPS["recovery_linear_rmse"])
    metrics={"linear_rmse_p95":percentile([r["linear_rmse"] for r in measured],.95),
        "linear_rmse_max":max(r["linear_rmse"] for r in measured),
        "residual_flicker_mean":math.fsum(r["residual_flicker"] for r in measured)/len(measured),
        "support_ghost_fraction_max":max(r["support_ghost_fraction"] for r in measured),
        "event_old_state_fraction_max":max(r["event_old_state_fraction_max"] for r in rows[step:]),
        "persistent_recovery_frames":recovered,"reference_mean_change":change,"baseline_minimum_visibility":min(baseline)}
    checks={"spatial_p95":metrics["linear_rmse_p95"]<=CAPS["linear_rmse_p95_max"],"worst_frame":metrics["linear_rmse_max"]<=CAPS["linear_rmse_max"],
        "flicker":metrics["residual_flicker_mean"]<=CAPS["residual_flicker_mean_max"],"ghost_support":metrics["support_ghost_fraction_max"]<=CAPS["support_ghost_fraction_max"],
        "event_ghost":metrics["event_old_state_fraction_max"]<=CAPS["support_ghost_fraction_max"],
        "persistent_recovery":recovered is not None and recovered<=CAPS["recovery_frames_max"]}
    return {"passed":all(checks.values()),"metrics":metrics,"checks":checks,"frames":rows,"scope":protocol["scope"],"phase_accepted":False}

def capture_files(folder):
    pfms=sorted(Path(folder).glob("frame-*.pfm"));jsons=sorted(Path(folder).glob("frame-*.json"))
    if len(pfms)!=64 or len(jsons)!=64:raise ValueError("missing captured frames/immutable sidecars, including final drain")
    if [p.stem for p in pfms]!=[p.stem for p in jsons]:raise ValueError("PFM/sidecar frame IDs differ")
    ids=[]
    for p in pfms:
        match=re.fullmatch(r"frame-(\d+)",p.stem)
        if not match:raise ValueError("ambiguous scalar frame filename")
        ids.append(int(match[1]))
    if ids!=list(range(ids[0],ids[0]+64)):raise ValueError("noncontiguous actual frame filenames")
    return pfms,jsons,ids

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,default=Path("build/release/phosphor"));parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--run",action="store_true");parser.add_argument("--manifest",type=Path)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument("--metallib",type=Path)
    parser.add_argument("--timeout",type=float,default=300);parser.add_argument("--gpu-lock",type=Path,default=Path("/tmp/phosphor-gpu-verification.lock"))
    args=parser.parse_args(argv);protocol,protocol_sha=load_protocol()
    if not math.isfinite(args.timeout) or args.timeout<=0:parser.error("positive finite outer timeout required")
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);capture=out/"linear";report=out/"renderer.json";log=out/"renderer.log"
    command=[str(args.binary.resolve()),"--scene","procedural","--bench","6","--reflection-scene","ao-temporal-wall",
        "--render-path","visibility","--geometry-path","mesh","--rt","on","--ao","rtao","--ao-radius","2",
        "--shadows","off","--lighting","legacy","--gi","off","--reflections","off","--lighting-denoise","custom",
        "--post","--upscaler","native","--temporal-views","1","--resolution","128x96","--frames","64","--warmup","0",
        "--fixed-timestep","--no-ui","--no-vsync","--no-gpu-timing","--debug-lighting","1",
        "--capture-linear-sequence",str(capture),"--capture-linear-signal","ao-filtered","--report",str(report)]
    manifest={"schema":"phos.f13-ao-temporal-run.v1","state":"NOT_EXECUTED","frozen_before_run":True,
        "created_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":protocol_sha,"protocol":protocol,
        "command":command,"capture":str(capture),"report":str(report),"log":str(log),"phase_accepted":False}
    path=args.manifest.resolve() if args.manifest else out/"frozen-manifest.json"
    if args.manifest:
        if not args.run:parser.error("existing frozen manifest is consumed only with --run")
        raw=path.read_bytes();manifest=json.loads(raw)
        if manifest.get("protocol_sha256")!=protocol_sha or manifest.get("protocol")!=protocol or manifest.get("command")!=command:
            parser.error("frozen AO protocol/command/destination differs; never mutate existing evidence")
    else:
        if path.exists():parser.error("refuse frozen manifest overwrite")
        raw=json_bytes(manifest);path.write_bytes(raw)
    if not args.run:print(f"NOT_EXECUTED: physical custom-AO plan frozen at {path}");return 0
    if not args.binary.is_file():parser.error("existing compiled renderer required; never build")
    if any(p.exists() for p in (capture,report,log,out/"results.json",out/"checkpoint.json")):parser.error("refuse existing execution artifacts")
    binary_sha=sha(args.binary);source_sha=source_hash(args.source_root.resolve());manifest_sha=hashlib.sha256(raw).hexdigest()
    metallib=args.metallib.resolve() if args.metallib else args.binary.resolve().parent/"shaders"/"phosphor.metallib"
    metallib_sha=sha(metallib) if metallib.is_file() else None
    provenance={"source_sha":source_sha,"binary_sha":binary_sha,"metallib_sha":metallib_sha,"manifest_sha":manifest_sha}
    checkpoint={"schema":"phos.f13-ao-temporal-checkpoint.v1","state":"FROZEN_BEFORE_RUN","provenance":provenance,
        "protocol_sha256":protocol_sha,"metallib_path":str(metallib),"timeout_seconds":args.timeout,"phase_accepted":False}
    (out/"checkpoint.json").write_bytes(json_bytes(checkpoint))
    env=dict(os.environ,MTL_DEBUG_LAYER="1",MTL_SHADER_VALIDATION="1",MTL_DEBUG_LAYER_WARNING_MODE="nslog",
        PHOSPHOR_SOURCE_SHA=source_sha,PHOSPHOR_BINARY_SHA=binary_sha,PHOSPHOR_MANIFEST_SHA=manifest_sha)
    if metallib_sha is None:env.pop("PHOSPHOR_METALLIB_SHA",None)
    else:env["PHOSPHOR_METALLIB_SHA"]=metallib_sha
    errors=[];evaluation=None
    with gpu_lock(args.gpu_lock):
        if sha(args.binary)!=binary_sha or source_hash(args.source_root.resolve())!=source_sha:raise RuntimeError("source/binary changed before AO control")
        if metallib.is_file()!=(metallib_sha is not None) or metallib_sha is not None and sha(metallib)!=metallib_sha:raise RuntimeError("metallib changed before AO control")
        status=run_checked(command,log,expected=0,timeout=args.timeout,env=env);errors+=status["failures"]
        if sha(args.binary)!=binary_sha or source_hash(args.source_root.resolve())!=source_sha:errors.append("source/binary changed during actual AO control")
        if metallib.is_file()!=(metallib_sha is not None) or metallib_sha is not None and sha(metallib)!=metallib_sha:errors.append("metallib changed during actual AO control")
        text=log.read_text(errors="replace");lines=list(CHECK.finditer(text))
        if GPU_ERROR.search(text) or len(lines)!=64 or any(m[2]!="PASS" for m in lines):errors.append("missing/failed actual lighting readbacks or GPU validation")
        try:
            renderer=json.loads(report.read_text());lighting=renderer.get("lighting",{})
            if renderer.get("schema_version",0)<10 or lighting.get("checks")!=64 or lighting.get("failures")!=0:raise ValueError("actual renderer checker identity/count failed")
            if renderer.get("rt",{}).get("enabled") is not True:raise ValueError("actual RT was unavailable; GTAO fallback cannot certify this RTAO control")
            for key,value in (("ao","rtao"),("denoise_requested","custom"),("denoise_effective","custom"),("reflections","off"),("gi","off")):
                if lighting.get(key)!=value:raise ValueError("actual lighting control differs: "+key)
            pfms,sidecars,ids=capture_files(capture);records=[json.loads(p.read_text()) for p in sidecars]
            if [m.get("frame") for m in records]!=ids:raise ValueError("sidecar not bound to actual image filename")
            references=validate_metadata(records,protocol,provenance)
            evaluation=quality([read_pfm(p) for p in pfms],references,protocol)
            if not evaluation["passed"]:errors.append("physical custom-AO temporal gates failed")
        except (ValueError,OSError,KeyError,TypeError,IndexError) as error:errors.append(str(error))
    result={"schema":"phos.f13-ao-temporal-result.v1","state":"EXECUTED_CHECKS_ONLY","passed":not errors,
        "errors":errors,"provenance":provenance,"quality":evaluation,"scope":protocol["scope"],
        "phase_accepted":False,"sdk_accepted":False,"gi_accepted":False}
    (out/"results.json").write_bytes(json_bytes(result));checkpoint["state"]="EXECUTED_CHECKS_ONLY";checkpoint["passed"]=not errors
    (out/"checkpoint.json").write_bytes(json_bytes(checkpoint))
    return int(bool(errors))

if __name__=="__main__":raise SystemExit(main())
