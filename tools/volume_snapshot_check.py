"""Read actual immutable F14 GPU snapshots; never invoke a renderer.

Sparse tolerances mirror predeclared VolumeOracleSettings and are exploratory,
not quality/phase certification. Independent disk and homogeneous equations
are recomputed from complete same-frame scalar ABI parameter words.
"""
import json
import math
from pathlib import Path
import struct
from temporal_light_metrics import homogeneous_fog

VOLUME_GATES={"transmittance":{"absolute":0.02,"relative":0.05},
    "multiscattering":{"absolute":0.02,"relative":0.25},"sky_view":{"absolute":0.02,"relative":0.25},
    "homogeneous_fog":{"absolute":2e-5,"relative":2e-4},"solar_disk":{"absolute":0.01,"relative":2e-4},
    # FP32 disk math uses the solar relative allowance, with a lower dark-ray floor.
    "lunar_disk":{"absolute":0.001,"relative":2e-4},
    # Angular rounding is amplified by the Gaussian for narrow star footprints.
    "star_field":{"absolute":2e-9,"relative":0.005}}
CELESTIAL_RAYS={"solar_disk":3,"lunar_disk":3,"star_field":64}

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
    if row["kind"]=="lunar_disk":
        words=blocks["expected_atmosphere"];p=volume_words(words,96)
        radius,phase=p[35],p[39]
        if not 0<radius<math.pi/2 or not 0<=phase<=1:raise ValueError("invalid declared lunar radius/phase")
        ray=row["pixel"][0]
        if ray not in (0,1,2):raise ValueError("missing lunar toward/away/tangent ray identity")
        if ray!=0 or not words[90]&1 or phase<=1e-6:return [0,0,0,1]
        moon=p[32:35];sun=p[24:27]
        lambert=max(0,-sum(a*b for a,b in zip(moon,sun))/math.hypot(*moon))
        flux_to_radiance=lambert/(phase*(2/3)*math.pi*math.sin(radius)**2)
        return [irradiance*flux_to_radiance for irradiance in p[36:39]]+[1]
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
    failed=False;kinds=set();row_failures=0;epoch_failures=0;ray_ids={kind:set() for kind in CELESTIAL_RAYS};solar_wide=False
    for row in rows:
        kind=row.get("kind");kinds.add(kind);gates=VOLUME_GATES.get(kind)
        if gates is None or row.get("tolerance")!=gates:raise ValueError("numeric tolerance differs from frozen source contract")
        expected,actual=row.get("expected",[]),row.get("actual",[])
        if len(expected)!=4 or len(actual)!=4 or not all(numeric(v) for v in expected+actual):raise ValueError("nonfinite/incomplete actual RGB+T sample")
        if any(v<0 for v in actual) or not 0<=actual[3]<=1.000001:raise ValueError("nonphysical actual volume sample")
        required_epoch=blocks["fog"][99] if kind=="homogeneous_fog" else blocks["expected_atmosphere"][87 if kind=="sky_view" or kind in CELESTIAL_RAYS else 86]
        if row.get("expected_epoch")!=required_epoch or not isinstance(row.get("actual_epoch"),int):raise ValueError("epoch lacks actual source ABI identity")
        epoch_bad=row["actual_epoch"]!=required_epoch;epoch_failures+=epoch_bad
        reference=analytic_component(row,blocks,record["settings"])
        if reference is not None and any(abs(a-b)>gates["absolute"]+gates["relative"]*abs(a) for a,b in zip(reference,expected)):
            raise ValueError("host reference differs from independent analytic equation")
        if kind in CELESTIAL_RAYS:
            ray=row["index"]
            if type(ray) is not int or row["pixel"]!=[ray,0] or ray in ray_ids[kind]:
                raise ValueError("invalid or duplicate "+kind+" ray identity")
            ray_ids[kind].add(ray)
            if kind=="solar_disk" and ray==0 and max(expected[:3])>65504:solar_wide=max(actual[:3])>65504
        mismatch=epoch_bad or any(abs(a-b)>gates["absolute"]+gates["relative"]*abs(a) for a,b in zip(expected,actual))
        if row.get("passed") is not (not mismatch):raise ValueError("boolean disagrees with recomputed numerical/epoch gate")
        failed|=mismatch;row_failures+=mismatch
    if not {"transmittance","multiscattering","sky_view"}<=kinds:
        raise ValueError("actual LUT producer samples missing")
    for kind,count in CELESTIAL_RAYS.items():
        if ray_ids[kind]!=set(range(count)):raise ValueError("actual "+kind+" producer samples missing")
    if require_fog and "homogeneous_fog" not in kinds:raise ValueError("actual homogeneous integration was not collected")
    counters=record.get("gpu_counters",{})
    for field in ("nonfinite","invalid_units","invalid_history","history_reused"):
        if not isinstance(counters.get(field),int) or counters[field]<0:raise ValueError("missing actual device counters")
    failed|=any(counters[k]>0 for k in ("nonfinite","invalid_units","invalid_history"))
    if record.get("passed") is not (not failed):raise ValueError("snapshot result disagrees with actual rows/device counters")
    if record.get("numeric_failure_count")!=row_failures or record.get("epoch_mismatch_count")!=epoch_failures:
        raise ValueError("numeric/epoch aggregate does not match actual rows")
    return {"failed":failed,"armed":record.get("corruption_armed") is True,"checks":len(rows),"kinds":kinds,"solar_wide":solar_wide}

def check_night(record,moon_control):
    words=record["parameter_blocks"]["expected_atmosphere"];p=volume_words(words,96)
    up=[camera-center for camera,center in zip(p[40:43],p[:3])]
    radius=math.hypot(*up)
    up=[v/radius for v in up]
    sun=p[24:27];moon=p[32:35]
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    if dot(up,sun)>=-math.sin(math.radians(18)) or dot(up,moon)<=math.sin(p[35]) or p[39]<=1e-6:
        raise ValueError("night fixture requires astronomical night and a visible lit lunar disk")
    for i in range(CELESTIAL_RAYS["star_field"]):
        y=.2+.1*(i//8);azimuth=2*math.pi*(i%8+.5)/8;horizontal=math.sqrt(1-y*y)
        ray=(horizontal*math.cos(azimuth),y,horizontal*math.sin(azimuth))
        if dot(up,ray)<=0 or dot(ray,sun)>=math.cos(p[11]) or dot(ray,moon)>=math.cos(p[35]):
            raise ValueError("star ray intersects a celestial disk or the local horizon")
    rows=record["cases"]
    lunar=next(row for row in rows if row["kind"]=="lunar_disk" and row["index"]==0)
    stars=[row["expected"][0] for row in rows if row["kind"]=="star_field"]
    if max(lunar["expected"][:3])<=VOLUME_GATES["lunar_disk"]["absolute"] or max(stars)<=10*VOLUME_GATES["star_field"]["absolute"]:
        raise ValueError("night fixture did not exercise nonzero lunar and stellar radiance")
    if moon_control:
        submitted=record["parameter_blocks"]["submitted_atmosphere"]
        if record.get("corruption_requested")!=5 or record.get("corruption_armed") is not True:
            raise ValueError("moon control was not armed")
        if submitted[:39]!=words[:39] or submitted[40:]!=words[40:] or volume_words(submitted,96)[39]!=p[39]*.5:
            raise ValueError("moon control must halve only the submitted phase")
        if [row for row in rows if not row["passed"]]!=[lunar] or lunar["actual_epoch"]!=lunar["expected_epoch"]:
            raise ValueError("moon control must fail the lunar centre radiance check")

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
    night_frames=case.get("night_frames")
    seen_night_frames=set()
    for path in paths:
        try:
            record=json.loads(path.read_text())
            check=check_record(record,{"source_sha":source_sha,"binary_sha":binary_hash,"manifest":manifest_hash},
                negative,case.get("numerical_oracle")=="fog-homogeneous")
            if night_frames is not None:
                check_night(record,negative)
                frame=record["frame"]
                if type(frame) is not int or frame in seen_night_frames:raise ValueError("invalid or duplicate night frame")
                seen_night_frames.add(frame)
            if case.get("require_volume_history_reuse"):
                reused=temporal_record(record);history_exercised+=reused
                if negative and check["armed"] and (not reused or record["gpu_counters"]["invalid_history"]<=0):
                    errors.append(path.name+": foreign history was not actually consumed/detected")
            kinds|=check["kinds"];checks+=check["checks"];armed+=check["armed"];failed_snapshots+=check["failed"];solar_wide+=check["solar_wide"]
            if not negative and check["failed"]:errors.append(path.name+": actual sparse F14 oracle failed")
        except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError) as error:errors.append(path.name+": "+str(error))
    if night_frames is not None and seen_night_frames!=set(range(night_frames)):
        errors.append("night fixture lacks valid evidence for every frozen frame")
    if negative and (not armed or not failed_snapshots):errors.append("negative never armed/exercised a real failed numerical/epoch/device-counter check")
    if case.get("require_volume_history_reuse") and not history_exercised:errors.append("no naturally eligible frozen-clock volume history was reused")
    if not negative and case.get("name") in ("atmo-zenith","atmo-space") and not solar_wide:errors.append("no actual solar output exceeded half range in declared wide-HDR case")
    return {"passed":not errors,"errors":errors,"actual_snapshots":len(paths),"actual_component_checks":checks,
        "failed_snapshots":failed_snapshots,"armed_snapshots":armed,"solar_wide_snapshots":solar_wide,"history_exercised_snapshots":history_exercised,"kinds":sorted(kinds),"certification":False}
