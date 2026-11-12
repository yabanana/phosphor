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

# F14.2 slicing: froxel view-depth bands (metres) and the declared caps. The
# implementation gate reuses the homogeneous-fog tolerance; the discretisation
# cap bounds the declared froxel algorithm against the continuous medium.
FOG_SLICING_BANDS=((0.0,1.0),(1.0,5.0),(5.0,25.0),(25.0,125.0),(125.0,math.inf))
FOG_SLICING_GATES={"implementation":VOLUME_GATES["homogeneous_fog"],"discretisation_relative":0.01,"minimum_band_samples":100}

def _mat(words,offset):return [words[offset+i] for i in range(16)]
def _apply(m,v):return [sum(m[4*c+r]*v[c] for c in range(4)) for r in range(4)]
def _pixel_ray(x,y,width,height,inverse,camera):
    ndc=[(x+.5)*2/width-1,1-(y+.5)*2/height,1,1];h=_apply(inverse,ndc)
    if not math.isfinite(h[3]) or abs(h[3])<1e-20:raise ValueError("invalid inverse projection")
    d=[h[i]/h[3]-camera[i] for i in range(3)];n=math.sqrt(sum(v*v for v in d))
    if not n>0:raise ValueError("degenerate pixel ray")
    return [v/n for v in d]

def fog_slicing_sample(params,row,sigma0,falloff,ratio):
    """Return (band index, continuous E, froxel model M) for one probe row.

    E integrates sigma(y)=sigma0*exp(-falloff*y) exactly along the pixel ray over
    the declared medium [near, min(depth, far)] in view depth, with a constant
    source/extinction ratio J: L = J (1 - T) + T L0. M is the declared froxel
    algorithm: bilinear over the four column-centre rays, each integrated
    slice by slice with the extinction of the slice centre (exponential
    slices); for a constant ratio only the column optical depth matters.
    """
    x,y,depth=row[0],row[1],row[2];before=row[3:6]
    width,height=params["width"],params["height"];gx,gy,gz=params["grid"]
    inverse,view,camera,near,far=params["inverse"],params["view"],params["camera"],params["near"],params["far"]
    ray=_pixel_ray(x,y,width,height,inverse,camera)
    cosine=-sum(view[4*c+2]*ray[c] for c in range(3))
    if not cosine>1e-5:raise ValueError("pixel ray behind the view plane")
    if 0<depth<=1:
        ndc=[(x+.5)*2/width-1,1-(y+.5)*2/height,depth,1];h=_apply(inverse,ndc)
        distance=min(far/cosine,math.sqrt(sum((h[i]/h[3]-camera[i])**2 for i in range(3))))
    else:distance=far/cosine
    view_depth=min(far,distance*cosine)
    band=next(i for i,(lo,hi) in enumerate(FOG_SLICING_BANDS) if lo<=view_depth<hi) if view_depth<far else len(FOG_SLICING_BANDS)
    def continuous():
        if view_depth<=near:return 0.0
        t0,t1=near/cosine,view_depth/cosine;rate=falloff*ray[1]
        scale=sigma0*math.exp(-falloff*camera[1])
        if abs(rate)<1e-12:return scale*(t1-t0)
        return scale*(math.exp(-rate*t0)-math.exp(-rate*t1))/rate
    def column_depth(cx,cy):
        if view_depth<=near:return 0.0
        cray=_pixel_ray(cx,cy,gx,gy,inverse,camera);ccos=max(1e-5,-sum(view[4*c+2]*cray[c] for c in range(3)))
        z=min(int(math.log(view_depth/near)/math.log(far/near)*gz),gz-1)
        def edge(i):return near*(far/near)**(i/gz)
        tau=0.0
        for i in range(z+1):
            end=edge(i+1) if i<z else view_depth
            sigma=sigma0
            if falloff>0:
                centre=near*(far/near)**((i+.5)/gz);sigma=sigma0*math.exp(max(-80.0,min(80.0,-falloff*(camera[1]+cray[1]*centre/ccos))))
            tau+=sigma*max(0.0,end-edge(i))/ccos
        return tau
    u=(x+.5)/width*gx-.5;v=(y+.5)/height*gy-.5;bx,by=math.floor(u),math.floor(v);fx,fy=u-bx,v-by
    model=[0.0,0.0,0.0];weight_t=0.0
    for corner in range(4):
        ix,iy=corner&1,corner>>1
        cx=min(max(bx+ix,0),gx-1);cy=min(max(by+iy,0),gy-1)
        w=(fx if ix else 1-fx)*(fy if iy else 1-fy);t=math.exp(-column_depth(cx,cy))
        for k in range(3):model[k]+=w*ratio[k]*(1-t)
        weight_t+=w*t
    model=[model[k]+weight_t*before[k] for k in range(3)]
    t=math.exp(-continuous());exact=[ratio[k]*(1-t)+t*before[k] for k in range(3)]
    return band,exact,model

def fog_slicing_evidence(paths,falloff):
    """Per-band slicing evidence from the actual composition probe rows."""
    gates=FOG_SLICING_GATES;impl=gates["implementation"];cap=gates["discretisation_relative"]
    bands=[{"samples":0,"max_measured_relative":0.0,"max_discretisation_relative":0.0,"max_implementation_error":0.0,"failures":0}
           for _ in range(len(FOG_SLICING_BANDS)+1)]
    errors=[];seen=set()
    for path in paths:
        record=json.loads(Path(path).read_text());probe=record.get("fog_apply")
        if not probe or probe.get("schema")!="phosphor.fog-apply-probe.v1":raise ValueError(path.name+": missing fog composition probe")
        words=record["parameter_blocks"]["fog"];f=volume_words(words,112)
        if abs(probe.get("fixture_height_falloff_m_inv",-1)-falloff)>1e-9 or abs(f[75]-falloff)>1e-9 or f[80]!=0:
            raise ValueError(path.name+": fixture falloff differs from the declared case")
        settings=record["settings"];sigma0=settings["fixture_extinction_m_inv"];source=settings["fixture_source"]
        if abs(sigma0-.01)>2e-8 or any(abs(a-b)>2e-8 for a,b in zip(source,(.01,.02,.03))):raise ValueError("fixture differs from predeclared source/extinction")
        ratio=[v/sigma0 for v in source];falloff_actual=f[75]
        params={"width":probe["width"],"height":probe["height"],"grid":words[64:67],"inverse":_mat(f,0),"view":_mat(f,16),
                "camera":list(f[92:95]),"near":f[72],"far":f[73]}
        if params["width"]!=words[68] or params["height"]!=words[69]:raise ValueError("probe extent differs from the fog output extent")
        for row in probe["samples"]:
            if len(row)!=9 or not all(numeric(v) for v in row):raise ValueError("incomplete probe row")
            if row[0]<0 or row[1]<0 or row[0]>=params["width"] or row[1]>=params["height"]:raise ValueError("probe pixel outside the output")
            seen.add((row[0],row[1]))
            band,exact,model=fog_slicing_sample(params,row,sigma0,falloff_actual,ratio);entry=bands[band];entry["samples"]+=1
            after=row[6:9];bad=False
            for a,e,m in zip(after,exact,model):
                implementation=abs(a-m);entry["max_implementation_error"]=max(entry["max_implementation_error"],implementation/(impl["absolute"]+impl["relative"]*abs(m)))
                floor=max(abs(e),1e-3);measured=abs(a-e)/floor;discretisation=abs(m-e)/floor
                entry["max_measured_relative"]=max(entry["max_measured_relative"],measured)
                entry["max_discretisation_relative"]=max(entry["max_discretisation_relative"],discretisation)
                bad|=implementation>impl["absolute"]+impl["relative"]*abs(m) or measured>cap+impl["absolute"]/floor or discretisation>cap
            entry["failures"]+=bad
    for index,entry in enumerate(bands):
        lo,hi=FOG_SLICING_BANDS[index] if index<len(FOG_SLICING_BANDS) else (None,None)
        name="beyond-far" if lo is None else f"{lo:g}m-far" if hi==math.inf else f"{lo:g}-{hi:g}m"
        entry["band"]=name
        if entry["samples"]<gates["minimum_band_samples"]:errors.append(f"slicing band {name} has {entry['samples']} samples (<{gates['minimum_band_samples']})")
        if entry["failures"]:errors.append(f"slicing band {name}: {entry['failures']} samples outside the declared gates")
    return {"passed":not errors,"errors":errors,"bands":bands,"distinct_pixels":len(seen),"gates":gates,"falloff_m_inv":falloff}

def camera_geometry_evidence(paths,spec):
    """Per-frame camera checks from the submitted atmosphere block (words
    0-3 planet centre/radius, 24-26 sun, 40-42 camera, 44-59 inverse VP)."""
    errors=[];report={}
    rows=[]
    for path in paths:
        record=json.loads(Path(path).read_text());p=volume_words(record["parameter_blocks"]["submitted_atmosphere"],96)
        camera=p[40:43];centre=p[0:3];radius=p[3]
        altitude=math.sqrt(sum((a-b)**2 for a,b in zip(camera,centre)))-radius
        inverse=list(p[44:60]);h=_apply(inverse,[0,0,1,1]);forward=[h[i]/h[3]-camera[i] for i in range(3)]
        n=math.sqrt(sum(v*v for v in forward));forward=[v/n for v in forward];sun=p[24:27];sn=math.sqrt(sum(v*v for v in sun))
        angle=math.degrees(math.acos(max(-1.0,min(1.0,sum(a*b/sn for a,b in zip(forward,sun))))))
        rows.append((record["frame"],altitude,angle,(sun[1]/sn)))
    if "flight_altitude" in spec:
        f=spec["flight_altitude"];below=sum(r[1]<f["below"] for r in rows);above=sum(r[1]>f["above"] for r in rows);inside=len(rows)-below-above
        report["flight_altitude"]={"frames_below":below,"frames_inside":inside,"frames_above":above,
                                   "first_m":rows[0][1] if rows else None,"last_m":rows[-1][1] if rows else None}
        counts={"below":below,"inside":inside,"above":above}
        if any(counts[k]<f["minimum_frames"] for k in f.get("require",("below","inside","above"))):
            errors.append("flight does not cross the declared cloud-layer boundary")
        if any(b[1]<a[1] for a,b in zip(rows,rows[1:])):errors.append("flight altitude is not monotonic")
    if "backlit_sun" in spec:
        b=spec["backlit_sun"];worst=max(abs(r[2]-b["offset_degrees"]) for r in rows) if rows else math.inf
        report["backlit_sun"]={"max_offset_error_degrees":worst,"sun_elevation_degrees":[math.degrees(math.asin(rows[0][3])),math.degrees(math.asin(rows[-1][3]))] if rows else None}
        if worst>b["tolerance_degrees"]:errors.append("sun is not at the declared offset from the view centre")
    return {"passed":not errors,"errors":errors,**report}

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
    slicing=None
    if case.get("numerical_oracle") in ("fog-slicing","fog-slicing-height"):
        try:
            slicing=fog_slicing_evidence(paths,case.get("fog_fixture_falloff",0.0))
            errors.extend(slicing["errors"])
        except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError) as error:errors.append("fog slicing: "+str(error))
    geometry=None
    if case.get("camera_geometry"):
        try:
            geometry=camera_geometry_evidence(paths,case["camera_geometry"]);errors.extend(geometry["errors"])
        except (OSError,ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as error:errors.append("camera geometry: "+str(error))
    history_reused=0
    for path in paths:
        try:history_reused+=json.loads(path.read_text()).get("gpu_counters",{}).get("history_reused",0)
        except (OSError,ValueError):pass
    if case.get("require_history_reuse_frames") and history_reused<=0:errors.append("temporal history was never reused in a case that declares it")
    return {"passed":not errors,"errors":errors,"actual_snapshots":len(paths),"actual_component_checks":checks,"history_reused_total":history_reused,
        **({"fog_slicing":slicing} if slicing is not None else {}),**({"camera_geometry":geometry} if geometry is not None else {}),
        "failed_snapshots":failed_snapshots,"armed_snapshots":armed,"solar_wide_snapshots":solar_wide,"history_exercised_snapshots":history_exercised,"kinds":sorted(kinds),"certification":False}
