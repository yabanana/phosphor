"""Read actual immutable F14 GPU snapshots; never invoke a renderer.

Sparse tolerances mirror predeclared VolumeOracleSettings and are exploratory,
not quality/phase certification. Independent solar and homogeneous equations
are recomputed from complete same-frame scalar ABI parameter words.
"""
import json
import math
from pathlib import Path
import struct
from temporal_light_metrics import homogeneous_fog

VOLUME_GATES={"transmittance":{"absolute":0.02,"relative":0.05},
    "multiscattering":{"absolute":0.02,"relative":0.25},"sky_view":{"absolute":0.02,"relative":0.25},
    "homogeneous_fog":{"absolute":2e-5,"relative":2e-4},"solar_disk":{"absolute":0.01,"relative":2e-4}}
def numeric(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def volume_words(block,count):
    if not isinstance(block,list) or len(block)!=count or any(not isinstance(v,int) or isinstance(v,bool) or v<0 or v>0xffffffff for v in block):
        raise ValueError("complete scalar u32 ABI parameter block missing")
    return struct.unpack("<"+"f"*count,struct.pack("<"+"I"*count,*block))

def analytic_component(row,blocks,settings):
    if row["kind"]=="solar_disk":
        p=volume_words(blocks["expected_atmosphere"],96);radius=p[11]
        if not math.isfinite(radius) or not 0<radius<math.pi/2:raise ValueError("invalid declared solar radius")
        ray=row["pixel"][0]
        if ray not in (0,1,2):raise ValueError("missing toward/away/tangent ray identity")
        return [p[i]/(math.pi*math.sin(radius)**2) if ray==0 else 0 for i in (28,29,30)]+[1]
    if row["kind"]!="homogeneous_fog":return None
    p=volume_words(blocks["fog"],112);words=blocks["fog"];gx,gy,gz=words[64:67]
    if min(gx,gy,gz)<=0 or p[75]!=0:raise ValueError("actual fog parameters are not homogeneous")
    index=row["index"];x=index%gx;y=(index//gx)%gy;z=index//(gx*gy)
    if index<0 or z>=gz:raise ValueError("fog prefix index outside actual grid")
    clip=[(x+.5)*2/gx-1,1-(y+.5)*2/gy,1,1]
    h=[sum(p[col*4+r]*clip[col] for col in range(4)) for r in range(4)]
    if not math.isfinite(h[3]) or abs(h[3])<1e-12:raise ValueError("invalid actual inverse projection")
    ray=[h[i]/h[3]-p[92+i] for i in range(3)];length=math.sqrt(sum(v*v for v in ray))
    if length<=0:raise ValueError("invalid fog world ray")
    ray=[v/length for v in ray];cosine=-sum(p[16+i*4+2]*ray[i] for i in range(3))
    near,far=p[72:74]
    if cosine<=0 or near<=0 or far<=near:raise ValueError("invalid actual metre fog depth")
    distance=(near*(far/near)**((z+1)/gz)-near)/cosine
    sigma=settings["fixture_extinction_m_inv"];source=settings["fixture_source"]
    if not numeric(sigma) or abs(sigma-.01)>2e-8 or len(source)!=3 or any(not numeric(a) or abs(a-b)>2e-8 for a,b in zip(source,(.01,.02,.03))):
        raise ValueError("homogeneous fixture differs from predeclared source/extinction")
    answer=homogeneous_fog(sigma,source,distance)
    return list(answer["radiance"])+[answer["transmittance"]]

def check_record(record,provenance,negative=False,require_fog=False):
    blocks=record.get("parameter_blocks",{});supplied=record.get("provenance",{})
    if record.get("schema")!="phosphor.volume-oracle.v1" or record.get("kind")!="f14-volume" or record.get("certification") is not False:
        raise ValueError("wrong actual F14 snapshot schema/certification")
    if any(supplied.get(field)!=expected for field,expected in provenance.items()):
        raise ValueError("actual F14 snapshot does not match frozen source/binary/manifest")
    if not isinstance(supplied.get("shader_generation"),int) or not supplied.get("source_hash_at_prepare") or supplied.get("source_hash_at_prepare")=="unavailable":
        raise ValueError("actual producer generation/source hash unavailable")
    if blocks.get("encoding")!="little-endian-u32-shared-scalar-ABI":raise ValueError("missing actual parameter ABI encoding")
    volume_words(blocks.get("expected_atmosphere"),96);volume_words(blocks.get("submitted_atmosphere"),96);volume_words(blocks.get("fog"),112)
    if not negative and blocks["expected_atmosphere"]!=blocks["submitted_atmosphere"]:raise ValueError("uncorrupted consumer received different atmosphere parameters")
    rows=record.get("cases",[])
    if not isinstance(rows,list) or not rows:raise ValueError("zero actual F14 numerical sample rows")
    failed=False;kinds=set();row_failures=0;epoch_failures=0;solar_rays=set();solar_wide=False
    for row in rows:
        kind=row.get("kind");kinds.add(kind);gates=VOLUME_GATES.get(kind)
        if gates is None or row.get("tolerance")!=gates:raise ValueError("numeric tolerance differs from frozen source contract")
        expected,actual=row.get("expected",[]),row.get("actual",[])
        if len(expected)!=4 or len(actual)!=4 or not all(numeric(v) for v in expected+actual):raise ValueError("nonfinite/incomplete actual RGB+T sample")
        if any(v<0 for v in actual) or not 0<=actual[3]<=1.000001:raise ValueError("nonphysical actual volume sample")
        required_epoch=blocks["fog"][99] if kind=="homogeneous_fog" else blocks["expected_atmosphere"][87 if kind in ("sky_view","solar_disk") else 86]
        if row.get("expected_epoch")!=required_epoch or not isinstance(row.get("actual_epoch"),int):raise ValueError("epoch lacks actual source ABI identity")
        epoch_bad=row["actual_epoch"]!=required_epoch;epoch_failures+=epoch_bad
        reference=analytic_component(row,blocks,record["settings"])
        if reference is not None and any(abs(a-b)>gates["absolute"]+gates["relative"]*abs(a) for a,b in zip(reference,expected)):
            raise ValueError("host reference differs from independent analytic equation")
        if kind=="solar_disk":
            solar_rays.add(row["pixel"][0])
            if row["pixel"][0]==0 and max(expected[:3])>65504:solar_wide=max(actual[:3])>65504
        mismatch=epoch_bad or any(abs(a-b)>gates["absolute"]+gates["relative"]*abs(a) for a,b in zip(expected,actual))
        if row.get("passed") is not (not mismatch):raise ValueError("boolean disagrees with recomputed numerical/epoch gate")
        failed|=mismatch;row_failures+=mismatch
    if not {"transmittance","multiscattering","sky_view","solar_disk"}<=kinds or solar_rays!={0,1,2}:
        raise ValueError("actual LUT or toward/away/tangent solar producer samples missing")
    if require_fog and "homogeneous_fog" not in kinds:raise ValueError("actual homogeneous integration was not collected")
    counters=record.get("gpu_counters",{})
    for field in ("nonfinite","invalid_units","invalid_history","history_reused"):
        if not isinstance(counters.get(field),int) or counters[field]<0:raise ValueError("missing actual device counters")
    failed|=any(counters[k]>0 for k in ("nonfinite","invalid_units","invalid_history"))
    if record.get("passed") is not (not failed):raise ValueError("snapshot result disagrees with actual rows/device counters")
    if record.get("numeric_failure_count")!=row_failures or record.get("epoch_mismatch_count")!=epoch_failures:
        raise ValueError("numeric/epoch aggregate does not match actual rows")
    return {"failed":failed,"armed":record.get("corruption_armed") is True,"checks":len(rows),"kinds":kinds,"solar_wide":solar_wide}

def temporal_record(record):
    """Verify host clock evidence and naturally eligible reuse, never invent it."""
    clock=record.get("clock",{});eligible=record.get("history_eligible",{})
    if not all(numeric(clock.get(k)) for k in ("seconds","delta_seconds","day_fraction")):
        raise ValueError("complete exact host clock missing")
    if not isinstance(clock.get("epoch"),int) or isinstance(clock["epoch"],bool) or clock["epoch"]<1:
        raise ValueError("clock epoch missing")
    if any(not isinstance(clock.get(k),bool) for k in ("reset","frozen_base")) or any(not isinstance(eligible.get(k),bool) for k in ("fog","clouds")):
        raise ValueError("clock reset/freeze or per-signal eligibility missing")
    params=volume_words(record.get("parameter_blocks",{}).get("expected_atmosphere"),96)
    gpu_seconds=struct.unpack("<f",struct.pack("<f",clock["seconds"]))[0]
    gpu_delta=struct.unpack("<f",struct.pack("<f",clock["delta_seconds"]))[0]
    if params[43]!=gpu_seconds or params[94]!=gpu_delta:raise ValueError("host clock disagrees with submitted GPU time ABI")
    natural=(eligible["fog"] or eligible["clouds"]) and not clock["reset"]
    if record.get("corruption_requested")==2 and record.get("corruption_armed") and not (natural and clock["frozen_base"]):
        raise ValueError("history corruption armed by overriding a reset or moving clock")
    reused=record.get("gpu_counters",{}).get("history_reused",0)
    return clock["frozen_base"] and natural and reused>0

def native_volume_evidence(case,binary_hash,manifest_hash,source_sha):
    paths=sorted(Path(case["volume_oracle"]).glob("frame-*-view-*.json"));errors=[];failed_snapshots=0;armed=0;checks=0;kinds=set();solar_wide=0;history_exercised=0
    if not paths:return {"passed":False,"pending":"zero actual native F14 readback snapshots"}
    negative=case.get("expected_exit",0)!=0
    for path in paths:
        try:
            record=json.loads(path.read_text())
            check=check_record(record,{"source_sha":source_sha,"binary_sha":binary_hash,"manifest":manifest_hash},
                negative,case.get("numerical_oracle")=="fog-homogeneous")
            if case.get("require_volume_history_reuse"):
                reused=temporal_record(record);history_exercised+=reused
                if negative and check["armed"] and (not reused or record["gpu_counters"]["invalid_history"]<=0):
                    errors.append(path.name+": foreign history was not actually consumed/detected")
            kinds|=check["kinds"];checks+=check["checks"];armed+=check["armed"];failed_snapshots+=check["failed"];solar_wide+=check["solar_wide"]
            if not negative and check["failed"]:errors.append(path.name+": actual sparse F14 oracle failed")
        except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError) as error:errors.append(path.name+": "+str(error))
    if negative and (not armed or not failed_snapshots):errors.append("negative never armed/exercised a real failed numerical/epoch/device-counter check")
    if case.get("require_volume_history_reuse") and not history_exercised:errors.append("no naturally eligible frozen-clock volume history was reused")
    if not negative and case.get("name") in ("atmo-zenith","atmo-space") and not solar_wide:errors.append("no actual solar output exceeded half range in declared wide-HDR case")
    return {"passed":not errors,"errors":errors,"actual_snapshots":len(paths),"actual_component_checks":checks,
        "failed_snapshots":failed_snapshots,"armed_snapshots":armed,"solar_wide_snapshots":solar_wide,"history_exercised_snapshots":history_exercised,"kinds":sorted(kinds),"certification":False}
