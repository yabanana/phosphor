#!/usr/bin/env python3
"""F13/F14 corpus planner and SERIAL tester. Default PLAN ONLY / NOT EXECUTED.

--run is the sole switch that launches the renderer. No build, download,
reference synthesis, automatic retries, gate relaxation or native SDK enablement.
Missing independent reference/control evidence remains pending and cannot pass.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from run_checked import run_checked
from temporal_light_metrics import analyse_paths,homogeneous_fog,read_pfm

PHASE_LIMIT="STOP_AFTER_F14"
GPU_ERROR=re.compile(r"GPU timeout|command buffers failed|Shader Validation Error|Metal Validation Error|failed assertion",re.I)
CHECK=re.compile(r"LIGHTING check frame (\d+)\s*\|\s*(PASS|FAIL)")
THRESHOLDS={"linear_rmse_p95_max":0.06,"linear_rmse_max":0.15,"residual_flicker_mean_max":0.02,
            "support_ghost_fraction_max":0.25,"recovery_frames_max":4,"recovery_linear_rmse":0.06}
MISSING_CONTROLS=[
    "F13 foreign-view/normal/motion/history corruption CLI and independent raw-state oracle",
    "F14 omitted/stale LUT update and invalid cloud-history corruption CLI",
    "F14 GPU LUT readback versus independent adaptive-Simpson/Gauss reference",
    "F14 homogeneous-fog parameter/readback export and exact Beer-Lambert comparison",
    "native denoised SDK scalar/impulse output-unit and lifetime proof after owner/tester gateway reconciliation",
]


def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def numeric(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def default_roi(scene,frames):
    region=[0.33,0.20,0.67,0.73] if scene in ("mirror","moving-light","disocclusion") else [0.15,0.20,0.85,0.85]
    dynamic=scene in ("moving-light","disocclusion","probe-parallax")
    events=[]
    if scene=="moving-light" and frames>135:events.append({"name":"emissive-step","frame":119,"window":16,"stable_frames":2})
    if scene=="disocclusion" and frames>255:events.append({"name":"camera-cut","frame":239,"window":16,"stable_frames":2})
    return {"schema":1,"linear":True,"frozen_before_run":True,"reconstruction_support_radius":1,"events":events,
            "minimum_change_energy":1e-8,"rois":[{"name":"primary-signal","normalized":region,"scale":1.0,
                "require_change":dynamic,"thresholds":dict(THRESHOLDS)}],
            "threshold_note":"F8 numeric caps retained; this is linear radiance per frozen ROI scale, not F8 SDR. Existing F8 gates still required unchanged."}


def enforce_gates(config):
    from temporal_light_metrics import validate_config
    validate_config(config)
    for roi in config["rois"]:
        for field,limit in THRESHOLDS.items():
            if roi["thresholds"][field]>limit:raise ValueError(f"cannot relax retained {field} gate")


def make_cases(out,frames,sdk_policy):
    cases=[]
    def add(name,phase,scene,flags,**extra):
        cases.append({"name":name,"phase":phase,"scene":scene,"flags":flags,"state":"NOT_EXECUTED",**extra})
    base_rt=["--rt","on","--shadows","rt","--lighting","brute","--gi","ddgi"]
    no_rt=["--rt","off","--shadows","off","--lighting","legacy","--gi","off"]
    add("probe-capture-unshadowed","F13","probe-parallax",
        [*no_rt,"--reflections","off","--ao","off","--lighting-denoise","off","--reflection-capture-probe",
         "--reflection-probe",str(out/"cooked-probe")],probe_capture=True,
        expected={"reflections":"off","ao":"off","denoise_requested":"off","denoise_effective":"off"})
    for scene in ("mirror","roughness","moving-light","disocclusion"):
        for variant,denoise,samples in (("raw","off",1),("custom","custom",1),("rt8-control","off",8)):
            name=f"{scene}-{variant}"
            add(name,"F13",scene,[*base_rt,"--reflections","rt","--reflection-samples",str(samples),
                "--ao","off","--lighting-denoise",denoise],
                expected={"reflections":"rt","ao":"off","denoise_requested":denoise,"denoise_effective":denoise},
                quality_pair=scene+"-rt8-control" if variant=="custom" else None,
                reference_kind="HIGHER_SAMPLE_CONTROL_NOT_INDEPENDENT_OR_CONVERGED" if variant=="rt8-control" else None)
    add("mirror-ssr","F13","mirror",[ *no_rt,"--reflections","ssr","--ao","off","--lighting-denoise","custom"],
        expected={"reflections":"ssr","ao":"off","denoise_requested":"custom","denoise_effective":"custom"})
    add("probe-parallax-cooked","F13","probe-parallax",[ *no_rt,"--reflections","probes","--ao","off","--lighting-denoise","custom",
         "--reflection-probe",str(out/"cooked-probe")],needs_probe=True,
        expected={"reflections":"probes","ao":"off","denoise_requested":"custom","denoise_effective":"custom"})
    for ao in ("gtao","rtao"):
        add("ao-cavity-"+ao,"F13","ao-cavity",[*(base_rt if ao=="rtao" else no_rt),
            "--reflections","off","--ao",ao,"--ao-radius","1.0","--lighting-denoise","custom"],
            expected={"reflections":"off","ao":ao,"denoise_requested":"custom","denoise_effective":"custom"},signal="ao")
    for signal in ("specular","ao"):
        add("raw-signal-"+signal,"F13","roughness",[ *base_rt,"--reflections","rt","--ao","rtao","--lighting-denoise","off"],
            signal=signal,raw_diagnostic=True,
            expected={"reflections":"rt","ao":"rtao","denoise_requested":"off","denoise_effective":"off"})
    add("metalfx-"+sdk_policy,"F13","roughness",[ *base_rt,"--reflections","rt","--ao","rtao","--lighting-denoise","metalfx"],
        expected={"reflections":"rt","ao":"rtao","denoise_requested":"metalfx","denoise_effective":"custom" if sdk_policy=="fallback" else "metalfx"},
        sdk_policy=sdk_policy)
    for views in (2,4):
        add("lifecycle-views"+str(views),"F13","disocclusion",[ *base_rt,"--reflections","rt","--ao","rtao",
            "--lighting-denoise","custom","--temporal-views",str(views),"--resize-every","90","--history-reset-every","60"],
            capture=False,expected={"reflections":"rt","ao":"rtao","denoise_requested":"custom","denoise_effective":"custom"})
    volume_base=[ *base_rt,"--reflections","rt","--ao","rtao","--lighting-denoise","custom"]
    for name,hour,height in (("atmo-zenith",12,2),("atmo-horizon",6.05,2),("atmo-space",12,150000)):
        add(name,"F14","mirror",[ *volume_base,"--atmosphere","on","--fog","off","--clouds","off",
            "--day-length","1200","--start-hour",str(hour),"--planet-camera-height",str(height)],
            expected={"atmosphere":True,"fog":False,"clouds":False,"cloud_full_rate":False},
            numerical_oracle="atmosphere-lut-independent",roi_scale=25)
    add("atmo-time-jump","F14","moving-light",[ *volume_base,"--atmosphere","on","--fog","off","--clouds","off",
         "--day-length","16","--start-hour","5","--time-jump-every","60"],
         expected={"atmosphere":True,"fog":False,"clouds":False,"cloud_full_rate":False},numerical_oracle="sun-clock-history")
    add("fog-homogeneous-oracle","F14","ao-cavity",[ *volume_base,"--atmosphere","on","--fog","on","--clouds","off"],
        expected={"atmosphere":True,"fog":True,"clouds":False,"cloud_full_rate":False},
        numerical_oracle="fog-homogeneous",
        pending_hook="Force heightFalloff0 and export same-frame sigma/source/distance/T/Lo; ordinary --fog on alone is not a homogeneous fixture")
    for variant in ("full","reconstructed"):
        add("clouds-"+variant,"F14","disocclusion",[ *volume_base,"--atmosphere","on","--fog","off","--clouds","on",
            "--day-length","1200","--start-hour","11","--planet-camera-height","2000",
            *(["--cloud-full-rate"] if variant=="full" else [])],
            expected={"atmosphere":True,"fog":False,"clouds":True,"cloud_full_rate":variant=="full"},
            quality_pair="clouds-full" if variant=="reconstructed" else None,
            reference_kind="FULL_RATE_CONTROL_NOT_INDEPENDENT_OFFLINE_REFERENCE" if variant=="full" else None,roi_scale=25)
    add("all-volumes-moving","F14","moving-light",[ *volume_base,"--atmosphere","on","--fog","on","--clouds","on",
         "--day-length","16","--start-hour","5","--time-jump-every","60"],
         expected={"atmosphere":True,"fog":True,"clouds":True,"cloud_full_rate":False},roi_scale=25)
    for case in cases:
        for flag,field in (("--reflections","reflections"),("--ao","ao"),("--lighting-denoise","denoise_requested"),
                           ("--gi","gi"),("--lighting","direct"),("--shadows","shadows")):
            if flag in case["flags"]:
                value=case["flags"][case["flags"].index(flag)+1];case.setdefault("expected",{})[field]=value
                if field=="denoise_requested" and value!="metalfx":case["expected"]["denoise_effective"]=value
        case["rt_expected"]=case["flags"][case["flags"].index("--rt")+1]=="on"
        folder=out/case["name"];case["report"]=str(folder/"renderer.json");case["log"]=str(folder/"renderer.log")
        case["linear"]=str(folder/"linear");case["signal"]=case.get("signal","hdr")
        case["roi"]=default_roi(case["scene"],frames)
        if case.get("roi_scale"):
            for roi in case["roi"]["rois"]:roi["scale"]=case["roi_scale"]
    return cases


def evaluate(case,status,log,report):
    errors=list(status.get("failures",[]))
    if status.get("returncode")!=0 or status.get("exit_marker")!=0:errors.append("raw process/EXIT marker did not succeed")
    if status.get("timed_out") or GPU_ERROR.search(log):errors.append("timeout or GPU/API/shader validation failure")
    schema=report.get("schema_version") if isinstance(report,dict) else None
    if not numeric(schema) or schema<10:errors.append("schema10 renderer report missing")
    lighting=report.get("lighting",{}) if isinstance(report,dict) else {}
    rt=report.get("rt",{}) if isinstance(report,dict) else {}
    # Schema10 omits rt when the subsystem is not instantiated (rt.present=false).
    rt_enabled=rt.get("enabled") if isinstance(report,dict) and "rt" in report else False
    if rt_enabled is not case.get("rt_expected"):errors.append("actual RT enabled state differs from frozen baseline")
    if not numeric(lighting.get("checks")) or lighting.get("checks",0)<=0:errors.append("lighting checker executed zero/missing checks")
    if lighting.get("failures")!=0:errors.append("lighting checker reported failures")
    lines=list(CHECK.finditer(log))
    if not lines or any(m[2]!="PASS" for m in lines):errors.append("missing/failed actual LIGHTING readback lines")
    for field,value in case.get("expected",{}).items():
        if lighting.get(field)!=value:errors.append(f"effective/requested lighting.{field} mismatch")
    if case.get("sdk_policy")=="fallback":
        reason=lighting.get("denoise_fallback")
        if not isinstance(reason,str) or not re.search(r"factory|gateway",reason,re.I):
            errors.append("missing-factory case did not explicitly identify factory/gateway custom fallback")
    if case.get("sdk_policy")=="native":
        # The requested/effective string is necessary but insufficient for native
        # lifetime/output-unit acceptance; independent evidence stays pending.
        if lighting.get("denoise_effective")!="metalfx":errors.append("native SDK was not actually selected")
    if case.get("needs_probe") and not lighting.get("probe_source"):errors.append("missing cooked probe source provenance")
    return {"functional_passed":not errors,"errors":errors,"lighting":lighting}


def validate_probe(folder):
    # Directory is a cooked six-face PFM corpus. No synthetic substitution.
    faces=list(Path(folder).glob("*.pfm"))
    if len(faces)!=6:raise ValueError("cooked probe must contain exactly six PFM faces")
    dimensions=None
    for face in sorted(faces):
        image=read_pfm(face)
        if image.width!=image.height:raise ValueError("probe face is not square")
        shape=(image.width,image.height)
        if dimensions and shape!=dimensions:raise ValueError("probe faces have different exact extents")
        dimensions=shape
    return {"faces":{str(p):sha(p) for p in sorted(faces)},"extent":dimensions}


def numeric_evidence(case,evidence,binary_hash,manifest_hash):
    row=evidence.get(case["name"])
    if not row:return {"passed":False,"pending":"missing independent numerical/readback evidence"}
    provenance=row.get("provenance",{})
    if provenance.get("binary_sha256")!=binary_hash or provenance.get("manifest_sha256")!=manifest_hash:
        return {"passed":False,"pending":"numerical evidence does not match frozen binary/manifest"}
    errors=[]
    if case.get("numerical_oracle")=="fog-homogeneous":
        samples=row.get("samples",[])
        if not samples:errors.append("zero homogeneous-medium checks")
        for sample in samples:
            if sample.get("height_falloff")!=0:errors.append("medium is not actually homogeneous");continue
            expected=homogeneous_fog(sample["extinction"],sample["source"],sample["distance"])
            gpu_radiance=sample.get("gpu_radiance",[])
            if not numeric(sample.get("gpu_transmittance")) or len(gpu_radiance)!=3 or not all(numeric(x) and x>=0 for x in gpu_radiance):
                errors.append("nonfinite or incomplete GPU medium result");continue
            if not 0<=sample["gpu_transmittance"]<=1:errors.append("nonphysical GPU transmittance")
            if abs(sample["gpu_transmittance"]-expected["transmittance"])>1e-4:errors.append("Beer-Lambert T error")
            scale=max(1.0,max(expected["radiance"]))
            if max(abs(a-b) for a,b in zip(sample["gpu_radiance"],expected["radiance"]))>1e-4*scale:errors.append("constant-source integral error")
    else:
        samples=row.get("samples",[])
        if not samples:errors.append("zero independent numerical checks")
        for sample in samples:
            if sample.get("reference_method") not in ("adaptive-simpson","gauss-legendre","exact-sun-clock"):
                errors.append("reference is not the independent declared method");continue
            reference=sample["reference"];observed=sample["gpu"]
            if len(reference)!=len(observed) or not all(numeric(x) for x in reference+observed):errors.append("invalid numerical samples");continue
            if any(abs(a-b)>1e-4*max(1,abs(a)) for a,b in zip(reference,observed)):errors.append("independent numerical oracle mismatch")
            expected_epoch=sample.get("expected_epoch");observed_epoch=sample.get("gpu_epoch")
            if not numeric(expected_epoch) or not numeric(observed_epoch) or expected_epoch!=observed_epoch:
                errors.append("missing/stale LUT/history/sun clock epoch")
    return {"passed":not errors,"errors":errors}


@contextmanager
def gpu_lock(path):
    import fcntl
    with Path(path).open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError("another cooperating Phosphor GPU verification owns the lock")
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,default=Path("build/release/phosphor"));parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--run",action="store_true");parser.add_argument("--only",action="append",default=[])
    parser.add_argument("--manifest",type=Path,help="--run consumes this existing frozen plan without rewriting it")
    parser.add_argument("--frames",type=int,default=300);parser.add_argument("--resolution",default="640x360")
    parser.add_argument("--sdk-policy",choices=["fallback","native"],default="fallback")
    parser.add_argument("--roi-config",type=Path);parser.add_argument("--references",type=Path);parser.add_argument("--numeric-evidence",type=Path)
    parser.add_argument("--timeout",type=float,default=300);parser.add_argument("--gpu-lock",type=Path,default=Path("/tmp/phosphor-gpu-verification.lock"))
    args=parser.parse_args(argv)
    if args.frames<2 or not re.fullmatch(r"[1-9]\d*x[1-9]\d*",args.resolution):parser.error("positive resolution and >=2 frames required")
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cases=make_cases(out,args.frames,args.sdk_policy)
    if args.only:cases=[c for c in cases if any(fnmatch.fnmatch(c["name"],pattern) for pattern in args.only)]
    if not cases:parser.error("--only selected no cases")
    roi_overrides=json.loads(args.roi_config.read_text()) if args.roi_config else {}
    for case in cases:
        if case["name"] in roi_overrides:case["roi"]=roi_overrides[case["name"]]
        enforce_gates(case["roi"])
        common=[str(args.binary.resolve()),"--scene","procedural","--bench","6","--reflection-scene",case["scene"],
                "--render-path","visibility","--geometry-path","mesh","--no-ui","--no-vsync","--fixed-timestep",
                "--frames",str(args.frames),"--warmup","0","--resolution",args.resolution,"--post","--upscaler","native",
                "--debug-lighting","1","--no-gpu-timing","--report",case["report"]]
        if case.get("capture",True):common+=["--capture-linear-sequence",case["linear"],"--capture-linear-signal",case["signal"]]
        case["command"]=common+case["flags"]
    manifest={"schema":1,"state":"NOT_EXECUTED","hard_stop":PHASE_LIMIT,"linear":True,"frozen_before_run":True,
              "created_utc":datetime.now(timezone.utc).isoformat(),"frames":args.frames,"resolution":args.resolution,
              "thresholds":THRESHOLDS,"cases":cases,"missing_controls":MISSING_CONTROLS,
              "existing_F8_gates_required_unchanged":"tools/testdata/temporal_thresholds.json",
              "independent_reference_required":True,"native_gateway_owner_tester_only":True,
              "binary_sha256":sha(args.binary) if args.binary.is_file() else None}
    manifest_path=args.manifest.resolve() if args.manifest else out/"frozen-manifest.json"
    if args.manifest:
        if not args.run:parser.error("--manifest is only for --run consumption of a frozen plan")
        if args.only or args.roi_config:parser.error("do not mutate case/ROI selection of an existing frozen manifest")
        raw=manifest_path.read_bytes();manifest=json.loads(raw)
        if manifest.get("hard_stop")!=PHASE_LIMIT or manifest.get("frozen_before_run") is not True:parser.error("invalid frozen F14-limited plan")
        cases=manifest["cases"];args.frames=manifest["frames"]
        for case in cases:
            enforce_gates(case["roi"])
            if Path(case["command"][0]).resolve()!=args.binary.resolve():parser.error("binary path differs from frozen plan")
            if Path(case["report"]).resolve().parent.parent!=out:parser.error("--out differs from frozen artifact destinations")
    else:
        raw=json_bytes(manifest)
        if manifest_path.exists():parser.error("refuse overwrite of frozen manifest; choose new output or --run --manifest")
        manifest_path.write_bytes(raw)
    manifest_hash=hashlib.sha256(raw).hexdigest()
    for case in cases:
        folder=out/case["name"];folder.mkdir(parents=True,exist_ok=True)
        (folder/"roi.json").write_bytes(json_bytes(case["roi"]))
    if not args.run:
        print(f"NOT_EXECUTED: frozen {len(cases)} F13/F14 cases at {manifest_path}")
        return 0
    if not args.binary.is_file():parser.error("--run requires an existing compiled renderer; this runner never builds")
    binary_hash=sha(args.binary)
    if manifest.get("binary_sha256") is not None and manifest["binary_sha256"]!=binary_hash:parser.error("compiled binary differs from frozen plan")
    # A source-only plan may precede compilation. Freeze the actual binary before
    # the first case without altering the ROI/scene/threshold manifest identity.
    (out/"execution-provenance.json").write_bytes(json_bytes({"manifest_sha256":manifest_hash,
        "binary_sha256":binary_hash,"frozen_before_first_case":True}))
    refs=json.loads(args.references.read_text()) if args.references else {}
    evidence=json.loads(args.numeric_evidence.read_text()) if args.numeric_evidence else {}
    results=[];env=dict(os.environ,MTL_DEBUG_LAYER="1",MTL_SHADER_VALIDATION="1",MTL_DEBUG_LAYER_WARNING_MODE="0")
    try:
        with gpu_lock(args.gpu_lock):
            for case in cases:
                if sha(args.binary)!=binary_hash:raise RuntimeError("binary changed between corpus cases")
                if case.get("needs_probe"):
                    try:validate_probe(out/"cooked-probe")
                    except (ValueError,OSError) as error:
                        results.append({"name":case["name"],"state":"NOT_EXECUTED","functional_passed":False,
                            "errors":["cooked probe prerequisite failed: "+str(error)],"phase_accepted":False})
                        continue
                status=run_checked(case["command"],Path(case["log"]),timeout=args.timeout,env=env)
                log=Path(case["log"]).read_text(errors="replace")
                try:report=json.loads(Path(case["report"]).read_text())
                except (OSError,ValueError):report=None
                result=evaluate(case,status,log,report);result.update(name=case["name"],state="EXECUTED",phase_accepted=False)
                if case.get("probe_capture"):
                    try:result["probe"]=validate_probe(out/"cooked-probe")
                    except (ValueError,OSError) as error:result["errors"].append(str(error));result["functional_passed"]=False
                if case.get("capture",True) and not case.get("probe_capture"):
                    try:
                        captures=sorted(Path(case["linear"]).glob("frame-*.pfm"))
                        if len(captures)!=args.frames:raise ValueError("linear sequence count incomplete, including final readback drain")
                        for capture in captures:read_pfm(capture)
                    except (ValueError,OSError) as error:result["errors"].append(str(error));result["functional_passed"]=False
                reference=refs.get(case["name"])
                if reference:
                    provenance=reference.get("provenance",{})
                    if provenance.get("validated") is True and provenance.get("independent") is True and provenance.get("manifest_sha256")==manifest_hash:
                        try:result["independent_metrics"]=analyse_paths(reference["linear"],case["linear"],case["roi"])
                        except (ValueError,OSError) as error:result["independent_metrics"]={"passed":False,"error":str(error)}
                    else:result["independent_metrics"]={"passed":False,"error":"reference provenance is not frozen/validated"}
                if case.get("numerical_oracle"):result["numerical"]=numeric_evidence(case,evidence,binary_hash,manifest_hash)
                result["independent_reference_pending"]=case.get("capture",True) and not case.get("raw_diagnostic") and not case.get("probe_capture") and not reference
                results.append(result)
                (out/case["name"]/"result.json").write_bytes(json_bytes(result))
    except (RuntimeError,OSError,ValueError) as error:
        results.append({"state":"FAILURE","error":str(error),"functional_passed":False})
    by_name={r.get("name"):r for r in results}
    # Pair analysis happens AFTER every serial render, so a control later in the
    # frozen matrix is never accidentally missed.
    for case in cases:
        result=by_name.get(case["name"])
        if result is None or not case.get("quality_pair"):continue
        control=out/case["quality_pair"]/"linear"
        try:result["control_metrics"]=analyse_paths(control,case["linear"],case["roi"])
        except (ValueError,OSError) as error:result["control_metrics"]={"passed":False,"error":str(error)}
        result["control_reference_accepted"]=False
        (out/case["name"]/"result.json").write_bytes(json_bytes(result))
    summary={"schema":1,"state":"EXECUTED_CHECKS_ONLY","manifest_sha256":manifest_hash,"results":results,
             "functional_passed":all(r.get("functional_passed",False) for r in results) and len(results)==len(cases),
             "phase_accepted":False,"hard_stop":PHASE_LIMIT,"missing_controls":MISSING_CONTROLS}
    (out/"summary.json").write_bytes(json_bytes(summary))
    selected_evidence_complete=summary["functional_passed"] and all(not r.get("independent_reference_pending",False) and
        r.get("independent_metrics",{}).get("passed",True) and r.get("control_metrics",{}).get("passed",True) and
        r.get("numerical",{}).get("passed",True) for r in results)
    summary["selected_evidence_complete"]=selected_evidence_complete
    summary["corpus_complete"]=False # Missing controls listed above are real unimplemented hooks.
    (out/"summary.json").write_bytes(json_bytes(summary))
    print(f"F13/F14 functional checks {'PASS' if summary['functional_passed'] else 'FAIL'}; selected evidence {'complete' if selected_evidence_complete else 'PENDING/FAIL'}; remaining controls PENDING; phase acceptance NOT CLAIMED")
    return 0 if selected_evidence_complete else 1


if __name__=="__main__":raise SystemExit(main())
