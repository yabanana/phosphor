#!/usr/bin/env python3
"""Genuine SDK fixture planner/readback checker. Default NOT EXECUTED.

Only --run launches existing binaries, serially. No builds, SDK enabling,
gateway patch application, retries or production output-policy promotion.
All equations below independently inspect actual SDK/physical PFM pixels and
actual packed GPU guide samples, rather than accepting a summary PASS flag.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from f13_f14_check import gpu_lock,json_bytes,sha,source_hash,GPU_ERROR
from run_checked import run_checked
from temporal_light_metrics import read_pfm

SCHEMA="phosphor.metalfx-fixture.v1"
MANUAL_EXPOSURE_REQUESTED_FP32=struct.unpack("<f",struct.pack("<f",.5/368640))[0]
MANUAL_EXPOSURE_R16=struct.unpack("<e",struct.pack("<e",MANUAL_EXPOSURE_REQUESTED_FP32))[0]
GATES={"constant_relative_max":0.01,"normalized_gain_error_max":0.02,
       "shape_relative_l1_max":0.02,"support_disagreement_max":0.02,
       "support_relative_peak":0.001,"restore_relative_max":1e-5,
       "normal_abs_max":0.002,"roughness_abs_max":0.001,"albedo_abs_max":0.001,
       "depth_abs_max":1e-6,"packed_color_relative_max":0.001}


def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def pixels(values):
    if not values or any(not number(v) or v<0 for v in values):raise ValueError("empty/nonfinite/negative actual radiance")
def relative_max(actual,expected):
    if len(actual)!=len(expected) or not actual:raise ValueError("oracle extent mismatch")
    return max(abs(a-b)/max(abs(b),1e-6) for a,b in zip(actual,expected))


def constant_oracle(sdk,restored,physical,factor):
    """Two output-unit hypotheses; only the explicit experiment may restore."""
    pixels(sdk);pixels(restored)
    if len(sdk)!=len(restored) or len(sdk)%3 or len(physical)!=3 or not number(factor) or factor<=0:
        raise ValueError("constant oracle invalid RGB/factor")
    pixels(physical);target=[physical[i%3] for i in range(len(sdk))]
    pre=relative_max([x/factor for x in sdk],target)
    unscaled=relative_max(sdk,target);error=relative_max(restored,target)
    restore=relative_max(restored,[x/factor for x in sdk])
    return {"sdk_preexposed_hypothesis":pre<=GATES["constant_relative_max"],
            "sdk_physical_hypothesis":unscaled<=GATES["constant_relative_max"],
            "preexposed_relative_error":pre,"physical_relative_error":unscaled,
            "restored_relative_error":error,"restore_equation_error":restore,
            "passed":error<=GATES["constant_relative_max"] and restore<=GATES["restore_relative_max"]}


def impulse_oracle(unit,scaled,factor=1/64):
    """Scaling of the observed native filter; no identity/unit-energy premise."""
    pixels(unit);pixels(scaled)
    if len(unit)!=len(scaled) or not number(factor) or factor<=0:raise ValueError("impulse pair extent/factor mismatch")
    energy=sum(unit)
    if energy<=1e-12:raise ValueError("black native output cannot prove metamorphic scaling")
    normalized=[v/factor for v in scaled];gain=sum(normalized)/energy
    shape=sum(abs(a-b) for a,b in zip(unit,normalized))/energy
    floor=max(unit)*GATES["support_relative_peak"]
    support=[(a>floor,b>floor) for a,b in zip(unit,normalized)]
    populated=sum(a or b for a,b in support);different=sum(a!=b for a,b in support)
    disagreement=different/populated if populated else 1
    return {"normalized_gain":gain,"shape_relative_error":shape,"support_disagreement":disagreement,
            "passed":abs(gain-1)<=GATES["normalized_gain_error_max"] and
                shape<=GATES["shape_relative_l1_max"] and disagreement<=GATES["support_disagreement_max"]}


def leaks_at_exit(log):
    rows=re.findall(r"\b(\d+) leaks? for (\d+) total leaked bytes\b",log)
    if not rows:return {"state":"NOT_VERIFIED","passed":False,"reason":"supported leaks at-exit summary was not produced"}
    leaks,byte_count=map(int,rows[-1])
    return {"state":"EXECUTED_PROCESS_AT_EXIT_CHECK","passed":leaks==0 and byte_count==0,
            "leaks":leaks,"leaked_bytes":byte_count,"native_internal_lifetime_accepted":False}


def guide_oracle(record,manual_control=False):
    """Expected authored inputs and actual SDK packed channels in declared units."""
    w,h=record["input_width"],record["input_height"];frame=record["frame"];scene=record["scenario"]
    if not isinstance(w,int) or not isinstance(h,int) or w<16 or h<16:raise ValueError("invalid authored extent")
    phase=frame%max(1,w//2) if scene=="channels" else (frame//48)%2
    if record.get("phase")!=phase:raise ValueError("phase differs from analytic frame script")
    color=[368640,128,64] if scene=="wide-hdr" else [.5,.25,.125]
    factor=record["preExposure"]
    for packed,key in ((False,"input_samples"),(True,"packed_samples")):
        samples=record.get(key)
        if not isinstance(samples,list) or len(samples)!=4:raise ValueError("missing actual GPU guide probes")
        for i,s in enumerate(samples):
            x=(w//4,3*w//4,w//2,min(w//4+phase,w-1))[i];right=x>=w//2
            n=[.6,0,.8] if scene=="channels" and right else [0,0,1]
            rough=(.9 if right else .04) if scene=="channels" else .5
            center=w//4+phase;patch=abs(x-center)<=3
            rgb=([.75,.5,.25] if patch else [.125]*3) if scene=="channels" else color
            if scene=="impulse":rgb=[.5]*3 if abs(x-w//2)<=3 else [0]*3
            if packed:rgb=[v*factor for v in rgb]
            for key,expected,tolerance in (("normal",n,GATES["normal_abs_max"]),
                    ("motion",[-1 if scene=="channels" and phase>0 and patch else 0,0],0)):
                actual=s.get(key)
                if not isinstance(actual,list) or len(actual)!=len(expected) or any(not number(a) or abs(a-b)>tolerance for a,b in zip(actual,expected)):
                    raise ValueError("actual "+key+" differs from world/current-to-previous input-pixel contract")
            for key,expected,tolerance in (("roughness",rough,GATES["roughness_abs_max"]),("depth",.1/4,GATES["depth_abs_max"]),
                    ("diffuseR",.6,GATES["albedo_abs_max"]),("specularR",.04,GATES["albedo_abs_max"])):
                if not number(s.get(key)) or abs(s[key]-expected)>tolerance:raise ValueError("actual "+key+" guide differs from analytic plane")
            actual=s.get("color")
            if not isinstance(actual,list) or len(actual)!=3 or any(not number(a) or abs(a-b)>(GATES["packed_color_relative_max"] if packed else 1e-6)*max(1,abs(b)) for a,b in zip(actual,rgb)):
                raise ValueError("actual packed/authored color differs from declared pre-exposure")
            if packed and any(s.get(key)!=expected for key,expected in (("exposure",MANUAL_EXPOSURE_R16 if manual_control else 1),("hitDistance",0),("reactive",0),("strength",0))):
                raise ValueError("provided manual exposure/optional neutral SDK guide mismatch")
    return True


def exposure_mode_oracle(record,expected_auto,manual_control=False,require_readback=False):
    """The descriptor mode is observed; the internal SDK exposure is not public."""
    required={"auto_exposure_requested":expected_auto,"exposure_descriptor_configured":True,
        "auto_exposure_enabled":expected_auto,"exposure_mode":"sdk-auto" if expected_auto else "manual",
        "provided_manual_exposure_texture_value":MANUAL_EXPOSURE_R16 if manual_control else 1,"manual_exposure_texture_ignored":expected_auto,
        "packed_exposure_is_provided_manual_value":True}
    if manual_control:
        required.update({"manual_exposure_control":True,"requested_manual_exposure_fp32":MANUAL_EXPOSURE_REQUESTED_FP32,
            "expected_manual_exposure_r16":MANUAL_EXPOSURE_R16})
        if require_readback:
            required.update({"actual_manual_exposure_readback":True,"actual_provided_manual_exposure_texture_value":MANUAL_EXPOSURE_R16})
    if any(record.get(key)!=value or isinstance(value,bool) and record.get(key) is not value for key,value in required.items()):
        raise ValueError("actual SDK descriptor/manual texture semantics differ from frozen exposure mode")
    return True


def make_cases(out,frames,gateway,auto_exposure=False,manual_control=False):
    if manual_control:
        if gateway!="native" or frames!=96 or auto_exposure:
            raise ValueError("manual exposure control requires native gateway, exactly 96 frames and automatic exposure off")
        return [{"name":"wide-hdr-manual-exposure","scenario":"wide-hdr","preexposed":True,
            "auto_exposure":False,"manual_exposure_control":True,"expected_exit":0,"expected_state":"FIXTURE_CHECKS_PASSED","frames":96,
            "physical_target":[368640,128,64],"pre_exposure":1/64,"packed_target":[5760,2,1],
            "requested_manual_exposure_fp32":MANUAL_EXPOSURE_REQUESTED_FP32,"expected_manual_exposure_r16":MANUAL_EXPOSURE_R16}]
    if auto_exposure:
        if gateway!="native" or frames!=96:
            raise ValueError("automatic exposure experiment requires native gateway and exactly 96 frames")
        return [{"name":"wide-hdr-auto-exposure","scenario":"wide-hdr","preexposed":True,
            "auto_exposure":True,"expected_exit":0,"expected_state":"FIXTURE_CHECKS_PASSED","frames":96,
            "physical_target":[368640,128,64],"pre_exposure":1/64,"packed_target":[5760,2,1]}]
    if gateway=="expected-missing":
        return [{"name":"missing-factory","scenario":"constant","preexposed":False,"expected_exit":1,
                 "expected_state":"NOT_EXECUTED_NATIVE","missing_factory":True,"frames":frames}]
    return [{"name":name,"scenario":scenario,"preexposed":exposed,"expected_exit":expected,
             "expected_state":"NOT_EXECUTED_NATIVE" if expected else "FIXTURE_CHECKS_PASSED","frames":max(frames,300) if scenario=="lifecycle" else frames,
             "lifecycle":scenario=="lifecycle"}
        for name,scenario,exposed,expected in (("constant-unit","constant",False,0),
            ("constant-scaled-pair","constant",True,0),("impulse-scaled-pair","impulse",True,0),
            ("channels-motion-normal-roughness","channels",False,0),("lifecycle-views-resize-cuts","lifecycle",False,0),
            ("wide-hdr-scaled","wide-hdr",True,0),("wide-hdr-unverified-policy","wide-hdr",False,1))]


def evaluate_capture(case,folder,provenance):
    """Never read a renderer boolean as sufficient numerical proof."""
    summary=json.loads((folder/"summary.json").read_text());errors=[];experiments=[]
    if summary.get("schema")!=SCHEMA or summary.get("state")!=case["expected_state"]:errors.append("fixture state/schema differs from frozen case")
    files=sorted(folder.glob("frame-*-view-*.json"))
    if case["expected_exit"]:
        if summary.get("passed") is not False or summary.get("native_frames")!=0 or files:errors.append("unsupported policy/factory was represented as native evidence")
        if not summary.get("fallback_reason"):errors.append("missing explicit unavailable/fallback diagnostic")
        if case.get("missing_factory") and summary.get("factory_installed") is not False:errors.append("case expects a build without the injected gateway")
        if case["name"]=="wide-hdr-unverified-policy" and "pre-exposure" not in summary.get("fallback_reason",""):errors.append("unverified nonunit output mapping did not select fallback")
        return {"passed":not errors,"errors":errors,"native_numerical_proof":False,"summary":summary}
    prewarm=json.loads((folder/"prewarm.json").read_text())
    expected_auto=case.get("auto_exposure",False);manual_control=case.get("manual_exposure_control",False)
    bounded_exposure=expected_auto or manual_control
    for record in (summary,prewarm):
        # Old manual artifacts remain inspectable; automatic mode needs explicit
        # evidence from the actual descriptor, including ignored manual texture.
        if bounded_exposure or "auto_exposure_enabled" in record:
            try:exposure_mode_oracle(record,expected_auto,manual_control)
            except ValueError as error:errors.append(str(error))
    if bounded_exposure and (summary.get("native_frames")!=96 or len(files)!=96):
        errors.append("exposure experiment requires exactly 96 actual native frames")
    if prewarm.get("schema")!="phosphor.metalfx-prewarm.v1" or prewarm.get("ready") is not True or prewarm.get("state")!="READY":errors.append("initial native SDK prewarm did not complete")
    if prewarm.get("actual_encoded_before_counting")!=0 or prewarm.get("phase")!=0 or prewarm.get("frame")!=0:errors.append("fixture advanced frame/phase/native work during initial prewarm")
    expected_views=4 if case.get("lifecycle") else 1
    if prewarm.get("active_views")!=expected_views or summary.get("initial_active_views")!=expected_views:errors.append("not every actual active view was prewarmed")
    if prewarm.get("budget_ms")!=case.get("prewarm_ms") or prewarm.get("budget_ms")!=summary.get("initial_prewarm_budget_ms") or summary.get("initial_prewarm_ready") is not True:errors.append("prewarm summary differs from actual initial wait")
    first_record=next(iter(sorted(folder.glob("frame-*-view-*.json"))),None)
    if first_record is not None:
        row=json.loads(first_record.read_text())
        for key in ("input_width","input_height","output_width","output_height"):
            if prewarm.get(key)!=row.get(key):errors.append("initial requested extent differs from first native output")
    expected_count=summary.get("native_frames")
    if not isinstance(expected_count,int) or expected_count<=0 or len(files)!=expected_count:errors.append("zero/incomplete actual native records")
    if summary.get("checked_frames")!=expected_count or summary.get("pack_checks")!=expected_count or summary.get("actual_encoded_frames")!=expected_count:
        errors.append("SDK encoded/readback/pack-check frame identity mismatch")
    if summary.get("failures")!=0 or summary.get("passed") is not True:errors.append("real host checker failed")
    steady=0;exposures=set();pairs={};views=set();extents=set();frame_tags=set()
    for path in files:
        record=json.loads(path.read_text());p=record.get("provenance",{})
        if record.get("schema")!=SCHEMA or record.get("actual_sdk_encoded") is not True or record.get("scenario")!=case["scenario"]:
            errors.append("record is not a tagged actual SDK output");continue
        for field in ("source_sha","binary_sha","manifest_sha"):
            if p.get(field)!=provenance[field]:errors.append("immutable execution provenance mismatch: "+field)
        if not isinstance(p.get("shader_generation"),int):errors.append("missing pipeline generation")
        frame=record.get("frame")
        if not isinstance(frame,int) or frame<0:errors.append("invalid encoded frame tag");continue
        frame_tags.add(frame)
        factor=1/64 if case["scenario"]=="wide-hdr" or case["preexposed"] and (frame//48)%2 else 1
        if record.get("preExposure")!=factor:errors.append("SDK scalar differs from frozen input script");continue
        try:
            if bounded_exposure or "auto_exposure_enabled" in record:exposure_mode_oracle(record,expected_auto,manual_control,True)
            guide_oracle(record,manual_control)
            stem=path.with_suffix("");sdk=read_pfm(str(stem)+"-sdk.pfm");physical=read_pfm(str(stem)+"-physical.pfm")
            pixels(sdk.rgb);pixels(physical.rgb)
            if (sdk.width,sdk.height)!=(record.get("output_width"),record.get("output_height")) or (physical.width,physical.height)!=(sdk.width,sdk.height):
                raise ValueError("actual SDK/physical PFM extent mismatch")
            restore_error=relative_max(physical.rgb,[v/factor for v in sdk.rgb])
            if restore_error>GATES["restore_relative_max"]:raise ValueError("actual radiance restore differs from independent SDK/factor equation")
            if record.get("passed") is not True:raise ValueError("tagged actual frame checker failed")
            views.add(record["view"]);extents.add((record["input_width"],record["input_height"],sdk.width,sdk.height))
            if record.get("steady") is True:
                if frame%48<8:raise ValueError("warm-up frame tagged steady")
                steady+=1;exposures.add(factor)
                if case["scenario"] in ("constant","lifecycle","wide-hdr"):
                    target=[368640,128,64] if case["scenario"]=="wide-hdr" else [.5,.25,.125]
                    check=constant_oracle(sdk.rgb,physical.rgb,target,factor)
                    experiments.append({"frame":frame,"view":record["view"],"preExposure":factor,**check})
                    if not check["passed"]:raise ValueError("actual constant unit equation failed")
                if case["scenario"]=="impulse":
                    key=(record["view"],sdk.width,sdk.height);pairs.setdefault(key,{})[factor]=sdk.rgb
        except (KeyError,TypeError,ValueError,OSError) as error:errors.append(path.name+": "+str(error))
    if bounded_exposure and frame_tags!=set(range(96)):errors.append("exposure native frame sequence is not 0..95")
    if not steady:errors.append("zero steady native readback records")
    if case["preexposed"] and case["scenario"]!="wide-hdr" and exposures!={1,1/64}:errors.append("unit/scaled native experiment pair incomplete")
    if case["scenario"]=="impulse":
        if not pairs:errors.append("no actual impulse SDK captures")
        for key,pair in pairs.items():
            if set(pair)!={1,1/64}:errors.append("impulse view/extent pair incomplete");continue
            try:
                check=impulse_oracle(pair[1],pair[1/64]);experiments.append({"view_extent":key,**check})
                if not check["passed"]:errors.append("observed native impulse scaling/gain/support failed")
            except ValueError as error:errors.append(str(error))
    if case.get("lifecycle"):
        if views!={0,1,2,3} or len(extents)<2:errors.append("actual native resize/four-view captures missing")
        if summary.get("reload_requested_after_native_work") is not True or summary.get("captured_generation_count",0)<2:
            errors.append("actual native encodes before/after cache-owned hot reload missing")
        for key,minimum in (("factory_requests",2),("retirements_submitted",1),("history_resets",2),("supplied_cuts",1)):
            if not number(summary.get(key)) or summary[key]<minimum:errors.append("lifecycle did not exercise "+key)
    if summary.get("production_policy_promoted") is not False or summary.get("phase_accepted") is not False:errors.append("fixture improperly promotes production policy/phase")
    return {"passed":not errors,"errors":errors,"native_numerical_proof":not errors,"experiments":experiments,"summary":summary}


def frozen_auto_exposure_experiment(manifest):
    cases=manifest.get("cases",[]);automatic=manifest.get("auto_exposure_experiment",False)
    if type(automatic) is not bool:raise ValueError("invalid frozen automatic exposure selection")
    if automatic and (manifest.get("gateway")!="native" or len(cases)!=1):
        raise ValueError("automatic exposure is a single native wide-HDR experiment")
    for case in cases:
        enabled=case.get("auto_exposure",False);command=case.get("command",[])
        if type(enabled) is not bool or enabled!=automatic or command.count("--denoised-fixture-auto-exposure")!=int(enabled):
            raise ValueError("automatic exposure command/case differs from frozen experiment")
        if not enabled:continue
        required={"name":"wide-hdr-auto-exposure","scenario":"wide-hdr","preexposed":True,"frames":96,
            "physical_target":[368640,128,64],"pre_exposure":1/64,"packed_target":[5760,2,1],
            "expected_exit":0,"expected_state":"FIXTURE_CHECKS_PASSED"}
        if any(case.get(key)!=value for key,value in required.items()):
            raise ValueError("wide-HDR physical target, pre-exposure, frame count or acceptance changed")
        for flag,value in (("--frames","96"),("--denoised-fixture","wide-hdr")):
            if command.count(flag)!=1 or command.index(flag)+1>=len(command) or command[command.index(flag)+1]!=value:
                raise ValueError("frozen automatic exposure command changes the fixture or frame count")
        if command.count("--denoised-fixture-pre-exposed")!=1 or any(flag in command for flag in ("--temporal-views","--resize-every","--history-reset-every")):
            raise ValueError("automatic exposure experiment changed its exposure policy or lifecycle")
    return automatic


def frozen_manual_exposure_control(manifest):
    cases=manifest.get("cases",[]);control=manifest.get("manual_exposure_control",False)
    if type(control) is not bool:raise ValueError("invalid frozen manual exposure control")
    if control and (manifest.get("gateway")!="native" or manifest.get("auto_exposure_experiment",False) or len(cases)!=1):
        raise ValueError("manual exposure is a single native wide-HDR control with automatic exposure off")
    for case in cases:
        enabled=case.get("manual_exposure_control",False);command=case.get("command",[])
        if type(enabled) is not bool or enabled!=control or command.count("--denoised-fixture-manual-exposure-control")!=int(enabled):
            raise ValueError("manual exposure command/case differs from frozen control")
        if not enabled:continue
        required={"name":"wide-hdr-manual-exposure","scenario":"wide-hdr","preexposed":True,"frames":96,
            "physical_target":[368640,128,64],"pre_exposure":1/64,"packed_target":[5760,2,1],
            "requested_manual_exposure_fp32":MANUAL_EXPOSURE_REQUESTED_FP32,"expected_manual_exposure_r16":MANUAL_EXPOSURE_R16,
            "expected_exit":0,"expected_state":"FIXTURE_CHECKS_PASSED","auto_exposure":False}
        if any(case.get(key)!=value for key,value in required.items()):
            raise ValueError("manual exposure control changed its target, units, exposure, frames or acceptance")
        for flag,value in (("--frames","96"),("--denoised-fixture","wide-hdr")):
            if command.count(flag)!=1 or command.index(flag)+1>=len(command) or command[command.index(flag)+1]!=value:
                raise ValueError("frozen manual exposure command changes fixture or frame count")
        if command.count("--denoised-fixture-pre-exposed")!=1 or any(flag in command for flag in ("--denoised-fixture-auto-exposure","--temporal-views","--resize-every","--history-reset-every")):
            raise ValueError("manual exposure control changed exposure policy or lifecycle")
    return control


def frozen_prewarm_budget(manifest,outer_timeout):
    budget=manifest.get("prewarm_ms")
    if not isinstance(budget,int) or isinstance(budget,bool) or not 1<=budget<=120000:
        raise ValueError("SDK plan must freeze bounded initial prewarm")
    for case in manifest.get("cases",[]):
        command=case.get("command",[]);flag="--denoised-fixture-prewarm-ms"
        if case.get("prewarm_ms")!=budget or command.count(flag)!=1:
            raise ValueError("frozen case prewarm differs from manifest")
        pos=command.index(flag)
        if pos+1>=len(command) or command[pos+1]!=str(budget):
            raise ValueError("actual frozen command uses a different initial wait budget")
    if not number(outer_timeout) or outer_timeout<=budget/1000:
        raise ValueError("outer process timeout must exceed frozen initial prewarm")
    return budget


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,default=Path("build/release/phosphor"));p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--metallib",type=Path)
    p.add_argument("--run",action="store_true");p.add_argument("--manifest",type=Path);p.add_argument("--only",action="append",default=[])
    p.add_argument("--gateway",choices=["expected-missing","native"],default="expected-missing")
    p.add_argument("--frames",type=int,default=192);p.add_argument("--resolution",default="128x96")
    p.add_argument("--denoised-fixture-auto-exposure",action="store_true",help="freeze only the 96-frame native wide-HDR automatic-exposure experiment")
    p.add_argument("--denoised-fixture-manual-exposure-control",action="store_true",help="freeze one 96-frame wide-HDR control at analytic exposure 0.5/368640; no output compensation")
    p.add_argument("--leaks-at-exit",action="store_true",help="wrap only native lifecycle with existing macOS leaks; never install")
    p.add_argument("--leaks-tool",type=Path,default=Path("/usr/bin/leaks"))
    p.add_argument("--prewarm-ms",type=int,default=120000);p.add_argument("--timeout",type=float,default=600);p.add_argument("--gpu-lock",type=Path,default=Path("/tmp/phosphor-gpu-verification.lock"))
    a=p.parse_args(argv)
    if a.frames<96 or not re.fullmatch(r"[1-9]\d*x[1-9]\d*",a.resolution) or min(map(int,a.resolution.split("x")))<16:
        p.error("at least96 frames and valid >=16px dimensions required for paired steady experiments")
    if not math.isfinite(a.timeout) or a.timeout<=0:p.error("positive finite timeout required")
    if not 1<=a.prewarm_ms<=120000:p.error("prewarm 1..120000ms required")
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    try:cases=make_cases(out,a.frames,a.gateway,a.denoised_fixture_auto_exposure,a.denoised_fixture_manual_exposure_control)
    except ValueError as error:p.error(str(error))
    if a.only:cases=[c for c in cases if any(fnmatch.fnmatch(c["name"],x) for x in a.only)]
    if not cases:p.error("no cases selected")
    for case in cases:
        case["prewarm_ms"]=a.prewarm_ms
        if case.get("lifecycle") and a.leaks_at_exit:case["leaks_tool"]=str(a.leaks_tool.resolve())
        folder=out/case["name"];case["capture"]=str(folder/"actual-sdk");case["log"]=str(folder/"renderer.log")
        case["command"]=[str(a.binary.resolve()),"--scene","procedural","--bench","1","--render-path","visibility",
            "--geometry-path","mesh","--no-ui","--no-vsync","--fixed-timestep","--frames",str(case["frames"]),"--warmup","0",
            "--resolution",a.resolution,"--post","--upscaler","native","--no-gpu-timing",
            "--denoised-fixture",case["scenario"],"--denoised-fixture-output",case["capture"],
            "--denoised-fixture-prewarm-ms",str(a.prewarm_ms),
            *(["--denoised-fixture-pre-exposed"] if case["preexposed"] else []),
            *(["--denoised-fixture-auto-exposure"] if case.get("auto_exposure") else []),
            *(["--denoised-fixture-manual-exposure-control"] if case.get("manual_exposure_control") else []),
            *(["--temporal-views","4","--resize-every","90","--history-reset-every","60"] if case.get("lifecycle") else [])]
    manifest={"schema":1,"state":"NOT_EXECUTED","frozen_before_run":True,"hard_stop":"STOP_AFTER_F14",
        "created_utc":datetime.now(timezone.utc).isoformat(),"thresholds":GATES,"gateway":a.gateway,"cases":cases,
        "auto_exposure_experiment":a.denoised_fixture_auto_exposure,"manual_exposure_control":a.denoised_fixture_manual_exposure_control,
        "prewarm_ms":a.prewarm_ms,"production_policy_promoted":False,"source_only_delivery":True,"binary_sha256":sha(a.binary) if a.binary.is_file() else None}
    path=a.manifest.resolve() if a.manifest else out/"sdk-frozen-manifest.json"
    if a.manifest:
        if not a.run or a.only or a.denoised_fixture_auto_exposure or a.denoised_fixture_manual_exposure_control:p.error("consume existing manifest only with --run and unchanged case selection")
        raw=path.read_bytes();manifest=json.loads(raw);cases=manifest["cases"]
        if manifest.get("thresholds")!=GATES or manifest.get("hard_stop")!="STOP_AFTER_F14" or manifest.get("frozen_before_run") is not True:p.error("incompatible frozen SDK plan")

        if any(Path(c["command"][0]).resolve()!=a.binary.resolve() or Path(c["capture"]).parent.parent!=out for c in cases):p.error("binary/output differs from frozen manifest")
    else:
        try:
            frozen_auto_exposure_experiment(manifest)
            frozen_manual_exposure_control(manifest)
            frozen_prewarm_budget(manifest,a.timeout)
        except ValueError as error:p.error(str(error))
        if path.exists():p.error("refuse frozen plan overwrite; choose new output or --run --manifest")
        raw=json_bytes(manifest);path.write_bytes(raw)
    try:
        frozen_auto_exposure_experiment(manifest)
        frozen_manual_exposure_control(manifest)
        a.prewarm_ms=frozen_prewarm_budget(manifest,a.timeout)
    except ValueError as error:p.error(str(error))
    if not a.run:print(f"NOT_EXECUTED: SDK fixture plan frozen at {path}");return 0
    if not a.binary.is_file():p.error("existing compiled binary required; runner never builds")
    binary_sha=sha(a.binary);source_sha=source_hash(a.source_root.resolve());manifest_sha=hashlib.sha256(raw).hexdigest()
    metallib=a.metallib.resolve() if a.metallib else a.binary.resolve().parent/"shaders"/"phosphor.metallib"
    metal_sha=sha(metallib) if metallib.is_file() else None
    if manifest.get("binary_sha256") not in (None,binary_sha):p.error("binary differs from frozen SDK plan")
    provenance={"source_sha":source_sha,"binary_sha":binary_sha,"manifest_sha":manifest_sha,
        "source_hash_method":"sha256-sorted-src-shaders-tools-path-and-bytes","metallib_path":str(metallib),"metallib_sha":metal_sha,
        "timeout_seconds":a.timeout,"frozen_before_first_case":True}
    prov_path=out/"sdk-execution-provenance.json"
    if prov_path.exists():p.error("refuse execution evidence overwrite")
    prov_path.write_bytes(json_bytes(provenance))
    env=dict(os.environ,MTL_DEBUG_LAYER="1",MTL_SHADER_VALIDATION="1",MTL_DEBUG_LAYER_WARNING_MODE="nslog",
        PHOSPHOR_SOURCE_SHA=source_sha,PHOSPHOR_BINARY_SHA=binary_sha,PHOSPHOR_MANIFEST_SHA=manifest_sha)
    results=[]
    with gpu_lock(a.gpu_lock):
        for case in cases:
            if sha(a.binary)!=binary_sha or source_hash(a.source_root.resolve())!=source_sha:raise RuntimeError("source/binary changed between SDK experiments")
            if metallib.is_file()!=(metal_sha is not None) or metal_sha is not None and sha(metallib)!=metal_sha:raise RuntimeError("metallib changed between SDK experiments")
            folder=Path(case["capture"])
            if folder.exists() and any(folder.iterdir()):raise RuntimeError("refuse existing actual SDK evidence destination")
            command=case["command"]
            if case.get("leaks_tool"):
                if not Path(case["leaks_tool"]).is_file():
                    results.append({"name":case["name"],"state":"NOT_EXECUTED","passed":False,"errors":["existing macOS leaks tool unavailable; no installation"],"phase_accepted":False});continue
                command=[case["leaks_tool"],"--atExit","--",*command]
            status=run_checked(command,Path(case["log"]),expected=case["expected_exit"],timeout=a.timeout,env=env)
            if sha(a.binary)!=binary_sha or source_hash(a.source_root.resolve())!=source_sha:raise RuntimeError("source/binary changed during SDK experiment")
            if metallib.is_file()!=(metal_sha is not None) or metal_sha is not None and sha(metallib)!=metal_sha:raise RuntimeError("metallib changed during SDK experiment")
            errors=list(status["failures"]);log=Path(case["log"]).read_text(errors="replace")
            if GPU_ERROR.search(log):errors.append("GPU/API/shader failure cannot satisfy SDK fixture")
            try:
                evaluation=evaluate_capture(case,folder,provenance);errors+=evaluation["errors"]
            except (OSError,ValueError,KeyError,TypeError) as error:evaluation={"native_numerical_proof":False};errors.append(str(error))
            if case.get("lifecycle"):
                evaluation["process_at_exit"]=leaks_at_exit(log) if case.get("leaks_tool") else {"state":"NOT_EXECUTED","passed":False}
                evaluation["final_destruction_verified"]=False # Submission/readback counters never prove deallocation.
                if case.get("leaks_tool") and not evaluation["process_at_exit"]["passed"]:errors.append("requested process-at-exit leak check missing or failed")
            results.append({"name":case["name"],"state":"EXECUTED_CHECKS_ONLY","passed":not errors,"errors":errors,
                "evaluation":evaluation,"phase_accepted":False,"production_policy_promoted":False})
    (out/"sdk-results.json").write_bytes(json_bytes({"schema":1,"provenance":provenance,"results":results,
        "phase_accepted":False,"production_policy_promoted":False}))
    return int(any(not r["passed"] for r in results))

if __name__=="__main__":raise SystemExit(main())
