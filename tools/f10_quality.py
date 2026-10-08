#!/usr/bin/env python3
"""Frozen F10 quality/temporal protocol. Default writes a plan, --run is tester work.

No package installation, retries, image-derived ROI or threshold fitting. Rendering
is sequential through run_checked. CPU analysis needs existing NumPy; --self-test
uses only the standard library and never launches the renderer. This script does
not test F11/F12 or certify unavailable hardware. See F10-QUALITY-PROTOCOL.md.
"""
from __future__ import annotations
from array import array
from datetime import datetime, timezone
import argparse
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from run_checked import run_checked
from f10_f12_check import CHECK, UNRELATED_ERROR, VALIDATION_ENV, expected_checks, file_record, integer, option, save_manifest

# FROZEN BEFORE ANY RENDER. Changing this policy creates a new experiment; never
# replace existing evidence or tune these values to candidate images.
POLICY = {
    'version':'f10-quality-v1', 'sun_angular_radius':0.00465, 'fov_y_degrees':60.0, 'aspect':16/9,
    'penumbra':{'centers_x':[-1.2,1.2], 'heights':[8.0,32.0], 'half_width':0.4, 'half_depth':1.0,
                'receiver_z_band':0.2, 'profile_margin':0.25, 'tail_begin_frame':64,
                'profile_fit':'equal_weight_isotonic_PAVA_v1', 'projected_disk_cone_error_bound':3.3e-5,
                'visibility_mae_max':0.05, 'visibility_abs_bias_max':0.03,
                'lit_loss_mean_max':0.02, 'umbra_mean_max':0.02,
                'width_relative_error_max':0.30, 'edge_center_error_pixels_max':1.5,
                'minimum_expected_width_pixels':3.0, 'minimum_profile_rows':8,
                'bias_negative_lit_loss_min':0.5, 'temporal_flicker_ratio_max':0.9},
    'guides':{'normal_length_error_max':0.005, 'normal_dot_min':0.999,
              'position_relative_error_max':2e-4, 'validity_mismatches_max':0},
    'overflow_mask':{'mae_max':0.001, 'max_error_max':1/16+1/1024},
    'mip_sensitivity':{'changed_position_metres':0.02, 'changed_fraction_min':0.005},
    'lifecycle':{'frames':160, 'capture_every':17, 'resolution_script':11,
                 'resize_every':40, 'history_reset_every':23,
                 'lit_loss_mean_max':0.02, 'umbra_mean_max':0.02},
}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest()
MESHLET_FAIL = re.compile(r'^MESHLETS frame \d+ [^\n]*\| FAIL \| overflow:[^\n]*$',re.M)


def disk_cdf(u):
    if u <= -1: return 0.0
    if u >= 1: return 1.0
    return 0.5+(math.asin(u)+u*math.sqrt(max(0.0,1-u*u)))/math.pi


def solar_plate_visibility(x, center, height):
    radius=height*math.tan(POLICY['sun_angular_radius'])
    half=POLICY['penumbra']['half_width']
    return 1-(disk_cdf((center+half-x)/radius)-disk_cdf((center-half-x)/radius))


def guide_pixel_errors(pa,na,pb,nb):
    """Scalar contract also used by CPU negative fixtures. No sign-insensitive N."""
    if not all(math.isfinite(v) for point in (pa,na,pb,nb) for v in point):
        return ['nonfinite']
    la=math.sqrt(sum(v*v for v in na));lb=math.sqrt(sum(v*v for v in nb))
    va,vb=la>0,lb>0
    errors=[]
    if any(valid and abs(length-1)>POLICY['guides']['normal_length_error_max'] for valid,length in ((va,la),(vb,lb))):
        errors.append('nonunit')
    if va != vb: errors.append('validity')
    if va and vb:
        scale=max(1,math.sqrt(sum(v*v for v in pa)),math.sqrt(sum(v*v for v in pb)))
        if math.sqrt(sum((a-b)**2 for a,b in zip(pa,pb)))/scale>POLICY['guides']['position_relative_error_max']:
            errors.append('position')
        if sum(a*b for a,b in zip(na,nb))/(la*lb)<POLICY['guides']['normal_dot_min']:
            errors.append('normal')
    return errors


def plan_jobs(app, out):
    result=[]
    base=[str(app),'--bench','6','--render-path','visibility','--rt','on','--warmup','0',
          '--fixed-timestep','--no-ui','--offscreen','--no-vsync','--no-gpu-timing','--debug-lighting','1']
    def add(name, group, scene, flags=(), frames=8, signal='shadow', sequence=0, overflow=False, **metadata):
        folder=out/name
        argv=base+['--lighting-scene',scene,'--frames',str(frames),*flags,
                   '--capture-linear-signal',signal,'--report',str(folder/'report.json')]
        if sequence: argv+=['--capture-linear-sequence',str(folder/'frames'),'--capture-every',str(sequence)]
        else: argv+=['--capture-linear',str(folder/'capture.pfm'),'--capture-linear-frame',str(frames-1)]
        if overflow: argv+=['--debug-meshlets','1','--debug-meshlets-corrupt','count']
        result.append({'name':name,'group':group,'argv':argv,'overflow':overflow,'expected_exit':1 if overflow else 0,
                       'signal':signal,'sequence_every':sequence,'frames':frames,'state':'NOT_EXECUTED',**metadata})
    common=['--resolution','1280x720','--contact-shadows','off']
    add('oracle_csm','oracle','shadow-penumbra',[*common,'--shadows','csm'],frames=128,kind='csm')
    for name,extra in [('oracle_rt',[]),('oracle_rt_reset',['--history-reset-every','1'])]:
        add(name,'oracle','shadow-penumbra',[*common,'--shadows','rt',*extra],frames=128,sequence=8,kind=name)
    add('oracle_bias_negative','oracle','shadow-penumbra',[*common,'--shadows','rt','--debug-lighting-corrupt','bias'],
        frames=64,quality_negative=True)
    for family,scales in [('native',[1.0,0.75,0.5]),('apple9',[0.5])]:
        for scale in scales:
            key=f'alpha_{family}_{int(scale*100):03d}'
            flags=['--resolution','640x360','--shadows','csm','--post','--upscaler','native',
                   '--render-scale',str(scale),'--history-reset-every','1']
            if family=='apple9':flags+=['--force-family','apple9']
            for overflow in (False,True):
                for signal,label in [('shadow-position','position'),('shadow-normal','normal'),('shadow','mask')]:
                    add(key+('_overflow_' if overflow else '_visibility_')+label,'alpha','alpha-mip-shadow',flags,
                        signal=signal,overflow=overflow,pair=key,variant='overflow' if overflow else 'visibility',component=label)
    for signal,label in [('shadow-position','position'),('shadow-normal','normal')]:
        add('alpha_neutral_'+label,'alpha','alpha-mip-shadow',
            ['--resolution','640x360','--shadows','csm','--post','--upscaler','native','--render-scale','.5',
             '--history-reset-every','1','--debug-neutral-mip-bias'],signal=signal,
            pair='alpha_native_050',variant='neutral',component=label)
    for views in (1,2,3,4):
        p=POLICY['lifecycle']
        add(f'lifecycle_v{views}','lifecycle','shadow-penumbra',
            ['--resolution','640x360','--shadows','rt','--post','--upscaler','native','--temporal-views',str(views),
             '--resolution-script',str(p['resolution_script']),'--resize-every',str(p['resize_every']),
             '--history-reset-every',str(p['history_reset_every']),'--temporal-script'],
            frames=p['frames'],sequence=p['capture_every'],views=views,scripted_camera=True)
    return result


def functional(job,status,text,report):
    failures=list(status.get('failures',[]))
    if status.get('returncode')!=job['expected_exit'] or status.get('exit_marker')!=job['expected_exit']:
        failures.append('unexpected raw exit/EXIT marker')
    if status.get('timed_out') or status.get('signal'):failures.append('timeout/signal')
    if UNRELATED_ERROR.search(text):failures.append('GPU/validation/runtime error')
    mesh_fail=list(MESHLET_FAIL.finditer(text))
    scrubbed=MESHLET_FAIL.sub('',text) if job['overflow'] else text
    if job['overflow']:
        # Accept only the exact summary of the already-required per-frame negative.
        n=job['frames']
        scrubbed=re.sub(rf'^MESHLETS checks {n} \| failures {n} \| FAIL$', '', scrubbed, flags=re.M)
        frame_ids=[int(re.search(r'^MESHLETS frame (\d+)',match.group()).group(1)) for match in mesh_fail]
        if frame_ids!=list(range(n)):failures.append('overflow frame IDs missing or duplicated')
    if re.search(r'\| FAIL\b',scrubbed):failures.append('unexpected checker FAIL')
    checks=CHECK.findall(text)
    if sorted(int(frame) for frame,_ in checks)!=expected_checks(job['argv']) or any(state!='PASS' for _,state in checks):
        failures.append('LIGHTING checks missing, duplicated, failed or missing ring tail')
    if not isinstance(report,dict):report={};failures.append('missing report')
    lighting=report.get('lighting',{})
    if not isinstance(lighting,dict):lighting={}
    if report.get('schema_version',0)<10 or report.get('frames')!=job['frames']:
        failures.append('schema/frame count mismatch')
    if lighting.get('checks')!=len(checks) or lighting.get('failures')!=0 or not checks:
        failures.append('lighting report/checker mismatch')
    if lighting.get('shadows')!=option(job['argv'],'--shadows') or not integer(lighting.get('sun_index')) or lighting['sun_index']==0xffffffff:
        failures.append('solar test needs the requested mode and an actual sun')
    if lighting.get('gi')!='off' or lighting.get('direct')!='legacy':failures.append('F10 signal contaminated by local/GI replacement')
    if job['overflow']:
        mesh=report.get('meshlets',{})
        if len(mesh_fail)!=job['frames'] or mesh.get('checks')!=job['frames'] or mesh.get('check_failures')!=job['frames'] or mesh.get('overflow_frames',0)<=0:
            failures.append('forced-overflow evidence is incomplete or failed for another reason')
    if option(job['argv'],'--force-family')=='apple9' and report.get('hardware',{}).get('effective_capabilities')!='apple9':
        failures.append('Apple9 capability restriction is not present in the report')
    return failures


def read_pfm(path):
    import numpy as np
    with Path(path).open('rb') as stream:
        if stream.readline().strip()!=b'PF':raise ValueError(f'{path}: not RGB Float32 PFM')
        width,height=map(int,stream.readline().split());scale=float(stream.readline());raw=stream.read()
    if width<=0 or height<=0 or not math.isfinite(scale) or not scale or len(raw)!=width*height*12:
        raise ValueError(f'{path}: invalid PFM layout')
    image=np.frombuffer(raw,dtype='<f4' if scale<0 else '>f4').reshape(height,width,3)[::-1].astype(np.float64)*abs(scale)
    if not np.isfinite(image).all():raise ValueError(f'{path}: nonfinite capture')
    return image


def camera(frame=0,views=1,scripted=False):
    import numpy as np
    eye=np.array([0.,3.,6.]);target=np.zeros(3)
    if scripted:
        time=((frame+1)/60)%8;phase=int(time/2)
        if phase==0:eye[0]+=time*1.2
        elif phase==1:eye+=np.array([3.,.8,2.]);eye[2]+=(time-2)*.7
        elif phase==2:eye+=np.array([-2.,0,1.]);target[0]+=math.sin(time*5)*2
        else:eye[0]+=(8-time)*.8
    offset=np.array([2.*(frame%views),0,0]);eye+=offset;target+=offset
    forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward)
    return eye,forward,right,up


def receiver_grid(width,height,frame=0,views=1,scripted=False):
    import numpy as np
    eye,forward,right,up=camera(frame,views,scripted)
    px=(np.arange(width)+.5)/width*2-1;py=1-(np.arange(height)+.5)/height*2
    tangent=math.tan(math.radians(POLICY['fov_y_degrees'])/2)
    direction=forward[None,None,:]+right[None,None,:]*(px[None,:,None]*tangent*POLICY['aspect'])+up[None,None,:]*(py[:,None,None]*tangent)
    denominator=direction[:,:,1]
    distance=np.divide(-eye[1],denominator,out=np.full_like(denominator,np.nan),where=np.abs(denominator)>1e-12)
    point=eye[None,None,:]+direction*distance[:,:,None]
    valid=(distance>0)&np.isfinite(point).all(axis=2)&(np.abs(point[:,:,0])<5.5)&(np.abs(point[:,:,2])<.2)
    return point,valid


def oracle_image(points):
    import numpy as np
    expected=np.ones(points.shape[:2]);p=POLICY['penumbra'];x=points[:,:,0]
    def cdf(u):
        q=np.clip(u,-1,1)
        return .5+(np.arcsin(q)+q*np.sqrt(np.maximum(0,1-q*q)))/math.pi
    for center,height in zip(p['centers_x'],p['heights']):
        radius=height*math.tan(POLICY['sun_angular_radius'])
        expected-=cdf((center+p['half_width']-x)/radius)-cdf((center-p['half_width']-x)/radius)
    return expected


def mask_metrics(image,frame=0,views=1,scripted=False):
    import numpy as np
    value=image[:,:,0]
    if value.min() < -1e-4 or value.max()>1.0001:raise ValueError('shadow R must be dimensionless visibility in [0,1]')
    points,roi=receiver_grid(value.shape[1],value.shape[0],frame,views,scripted)
    expected=oracle_image(points)
    # ROIs are analytic geometry predicates fixed before images, not masks
    # selected from good-looking candidate pixels. Background is never an oracle.
    lit=roi&(expected>1-1e-8);umbra=roi&(expected<1e-8);penumbra=roi&(expected>.1)&(expected<.9)
    if min(np.count_nonzero(lit),np.count_nonzero(umbra))<32:raise ValueError('oracle ROI has insufficient lit/umbra coverage')
    delta=value-expected
    result={'pixels':int(roi.sum()),'lit_pixels':int(lit.sum()),'umbra_pixels':int(umbra.sum()),
            'mae':float(np.abs(delta[roi]).mean()),'signed_bias':float(delta[roi].mean()),
            'lit_loss_mean':float((1-value[lit]).mean()),'umbra_mean':float(value[umbra].mean())}
    return result,(points,roi,expected,penumbra)


def crossing(xs,ys,level):
    candidates=[]
    for i in range(len(xs)-1):
        if ys[i]<=level<=ys[i+1] and ys[i+1]>ys[i]:
            candidates.append(xs[i]+(level-ys[i])*(xs[i+1]-xs[i])/(ys[i+1]-ys[i]))
    return candidates[0] if len(candidates)==1 else None


def isotonic(values):
    # Fixed equal-weight PAVA, used ONLY to locate profile crossings. Raw
    # images still determine MAE, bias, lit/umbra error and temporal flicker.
    blocks=[]
    for value in values:
        blocks.append([float(value),1])
        while len(blocks)>1 and blocks[-2][0]/blocks[-2][1]>blocks[-1][0]/blocks[-1][1]:
            total,count=blocks.pop();blocks[-1][0]+=total;blocks[-1][1]+=count
    return [total/count for total,count in blocks for _ in range(count)]


def profiles(image,aux):
    import numpy as np
    points,roi,expected,_=aux;p=POLICY['penumbra'];out=[]
    for center,height in zip(p['centers_x'],p['heights']):
        edge=center+p['half_width'];radius=height*math.tan(POLICY['sun_angular_radius'])
        widths=[];pixel_errors=[];expected_pixels=[];undefined=0
        for row in range(image.shape[0]):
            choose=roi[row]&(np.abs(points[row,:,0]-edge)<radius+p['profile_margin'])
            xs=points[row,choose,0]
            if len(xs)<6:continue
            values=isotonic(image[row,choose,0]);truth=expected[row,choose]
            crossings=[crossing(xs,values,t) for t in (.1,.5,.9)]
            target=[crossing(xs,truth,t) for t in (.1,.5,.9)]
            if any(t is None for t in target):continue
            step=float(np.median(np.diff(xs)));width=target[2]-target[0]
            expected_pixels.append(width/step)
            if any(t is None for t in crossings):
                undefined+=1;continue
            widths.append(abs((crossings[2]-crossings[0])/width-1));pixel_errors.append(abs(crossings[1]-target[1])/step)
        if len(expected_pixels)<p['minimum_profile_rows'] or min(expected_pixels,default=0)<p['minimum_expected_width_pixels']:
            raise ValueError('penumbra is unresolved at this camera/resolution; do not infer a width gate')
        out.append({'height':height,'rows':len(expected_pixels),'undefined_rows':undefined,
                    'median_width_relative_error':float(np.median(widths)) if widths else None,
                    'median_edge_error_pixels':float(np.median(pixel_errors)) if pixel_errors else None,
                    'minimum_expected_width_pixels':float(min(expected_pixels))})
    return out


def guide_comparison(position_a,normal_a,position_b,normal_b):
    import numpy as np
    if len({x.shape for x in (position_a,normal_a,position_b,normal_b)})!=1:raise ValueError('guide dimensions differ')
    la=np.linalg.norm(normal_a,axis=2);lb=np.linalg.norm(normal_b,axis=2);va=la>0;vb=lb>0;both=va&vb
    p=POLICY['guides']
    bad_unit=((np.abs(la-1)>p['normal_length_error_max'])&va)|((np.abs(lb-1)>p['normal_length_error_max'])&vb)
    denom=np.maximum(1,np.maximum(np.linalg.norm(position_a,axis=2),np.linalg.norm(position_b,axis=2)))
    position_error=np.linalg.norm(position_a-position_b,axis=2)/denom
    dot=np.divide((normal_a*normal_b).sum(axis=2),la*lb,out=np.ones_like(la),where=both)
    result={'validity_mismatches':int(np.count_nonzero(va!=vb)), 'nonunit_normals':int(bad_unit.sum()),
            'position_mismatches':int(((position_error>p['position_relative_error_max'])&both).sum()),
            'normal_mismatches':int(((dot<p['normal_dot_min'])&both).sum()),'valid_pixels':int(both.sum()),
            'max_position_relative_error':float(position_error[both].max(initial=0))}
    result['passed']=result['valid_pixels']>0 and not any(result[k] for k in ('validity_mismatches','nonunit_normals','position_mismatches','normal_mismatches'))
    return result


def frames_for(job,out):
    folder=out/job['name']
    if not job['sequence_every']:return [(job['frames']-1,folder/'capture.pfm')]
    return [(frame,folder/'frames'/f'frame-{frame:06d}.pfm') for frame in range(0,job['frames'],job['sequence_every'])]


def analyze(manifest,out):
    import numpy as np
    results=[];pending=[];selected={j['name'] for j in manifest['jobs']}
    complete={j['name']:j for j in manifest['jobs'] if j.get('functional_passed')}
    def need(names,label):
        if all(name in complete for name in names):return True
        pending.append({'gate':label,'missing':[name for name in names if name not in complete]});return False
    p=POLICY['penumbra'];oracle_arrays={}
    for name in ('oracle_csm','oracle_rt','oracle_rt_reset'):
        if name not in complete:continue
        job=complete[name];pictures=[read_pfm(path) for frame,path in frames_for(job,out) if frame>=p['tail_begin_frame']]
        if not pictures:raise ValueError('oracle lacks predeclared settled tail')
        average=np.mean(pictures,axis=0);metrics,aux=mask_metrics(average);widths=profiles(average,aux)
        okay=metrics['mae']<=p['visibility_mae_max'] and abs(metrics['signed_bias'])<=p['visibility_abs_bias_max'] and metrics['lit_loss_mean']<=p['lit_loss_mean_max'] and metrics['umbra_mean']<=p['umbra_mean_max']
        okay &= all(not w['undefined_rows'] and w['median_width_relative_error']<=p['width_relative_error_max'] and w['median_edge_error_pixels']<=p['edge_center_error_pixels_max'] for w in widths)
        oracle_arrays[name]=(pictures,aux[3])
        # Reset is a stochastic control, not an independently required denoised-quality pass.
        if name!='oracle_rt_reset':results.append({'gate':name,'passed':bool(okay),'metrics':metrics,'profiles':widths})
    if 'oracle_rt_reset' in selected and need(['oracle_rt','oracle_rt_reset'],'temporal_flicker'):
        def flicker(item):
            images,roi=item
            if len(images)<2 or not roi.any():raise ValueError('no temporal penumbra samples')
            return float(np.sqrt(np.mean([np.square((b[:,:,0]-a[:,:,0])[roi]).mean() for a,b in zip(images,images[1:])])) )
        temporal=flicker(oracle_arrays['oracle_rt']);reset=flicker(oracle_arrays['oracle_rt_reset'])
        results.append({'gate':'temporal_flicker','passed':reset>1e-6 and temporal<=p['temporal_flicker_ratio_max']*reset,
                        'temporal_rms':temporal,'reset_rms':reset,'ratio_max':p['temporal_flicker_ratio_max']})
    if 'oracle_bias_negative' in selected and need(['oracle_rt','oracle_bias_negative'],'bias_negative_detected'):
        picture=read_pfm(out/'oracle_bias_negative/capture.pfm');metric,_=mask_metrics(picture)
        results.append({'gate':'bias_negative_detected','passed':metric['lit_loss_mean']>=p['bias_negative_lit_loss_min'],'metrics':metric})
    pairs=sorted({j['pair'] for j in manifest['jobs'] if j.get('group')=='alpha' and j.get('variant')!='neutral'})
    for pair in pairs:
        names=[f'{pair}_{variant}_{component}' for variant in ('visibility','overflow') for component in ('position','normal','mask')]
        if not need(names,pair):continue
        images={name:read_pfm(out/name/'capture.pfm') for name in names}
        geom=guide_comparison(images[names[0]],images[names[1]],images[names[3]],images[names[4]])
        difference=np.abs(images[names[2]][:,:,0]-images[names[5]][:,:,0]);mask={'mae':float(difference.mean()),'maximum':float(difference.max())}
        okay=geom['passed'] and mask['mae']<=POLICY['overflow_mask']['mae_max'] and mask['maximum']<=POLICY['overflow_mask']['max_error_max']
        results.append({'gate':pair,'passed':bool(okay),'guides':geom,'shadow':mask})
    names=['alpha_native_050_visibility_position','alpha_native_050_visibility_normal','alpha_neutral_position','alpha_neutral_normal']
    if 'alpha_neutral_position' in selected and need(names,'mip_sensitivity'):
        pa,na,pb,nb=[read_pfm(out/name/'capture.pfm') for name in names]
        valid_a=np.linalg.norm(na,axis=2)>0;valid_b=np.linalg.norm(nb,axis=2)>0
        changed=(valid_a!=valid_b)|((valid_a&valid_b)&(np.linalg.norm(pa-pb,axis=2)>POLICY['mip_sensitivity']['changed_position_metres']))
        fraction=float(changed.mean());results.append({'gate':'mip_sensitivity','passed':fraction>=POLICY['mip_sensitivity']['changed_fraction_min'],'changed_fraction':fraction})
    for name,job in complete.items():
        if job['group']!='lifecycle':continue
        records=[];views=set();extents=set();policy=POLICY['lifecycle']
        for frame,path in frames_for(job,out):
            picture=read_pfm(path);metric,_=mask_metrics(picture,frame,job['views'],True)
            views.add(frame%job['views']);extents.add(picture.shape[:2])
            records.append({'frame':frame,'view':frame%job['views'],'extent':list(picture.shape[:2]),**metric})
        okay=len(views)==job['views'] and len(extents)>=4 and all(r['lit_loss_mean']<=policy['lit_loss_mean_max'] and r['umbra_mean']<=policy['umbra_mean_max'] for r in records)
        results.append({'gate':name,'passed':bool(okay),'records':records})
    return {'policy_sha256':POLICY_SHA,'results':results,'pending':pending,
            'passed':bool(results) and not pending and all(r['passed'] for r in results),
            'phase_accepted':False,'scope':'Selected F10 fixed corpus gates only; hardware/performance/contact remain separate'}


def self_test():
    count=0
    def check(condition):
        nonlocal count
        if not condition:raise AssertionError(f'F10 protocol fixture {count} failed')
        count+=1
    for u in (-2,-1,-.5,0,.5,1,2):check(abs(disk_cdf(u)+disk_cdf(-u)-1)<1e-12)
    check(disk_cdf(0)==.5);check(disk_cdf(-1)==0);check(disk_cdf(1)==1)
    for center,height in zip(POLICY['penumbra']['centers_x'],POLICY['penumbra']['heights']):
        check(solar_plate_visibility(center,center,height)==0)
        check(solar_plate_visibility(center+1,center,height)==1)
        check(abs(solar_plate_visibility(center+.4,center,height)-.5)<1e-12)
    check(not guide_pixel_errors((0,0,0),(0,1,0),(0,0,0),(0,1,0)))
    check('normal' in guide_pixel_errors((0,0,0),(0,1,0),(0,0,0),(0,-1,0)))
    check('validity' in guide_pixel_errors((0,0,0),(0,1,0),(0,0,0),(0,0,0)))
    check('position' in guide_pixel_errors((0,1,0),(0,1,0),(0,1,.2),(0,1,0)))
    check('nonunit' in guide_pixel_errors((0,0,0),(0,.5,0),(0,0,0),(0,1,0)))
    check('nonfinite' in guide_pixel_errors((math.nan,0,0),(0,1,0),(0,0,0),(0,1,0)))
    check(isotonic([0,.6,.4,1])==[0,.5,.5,1])
    # Uniform cone projected disk differs from uniform disk only by the bounded
    # Jacobian (1+r^2)^(-3/2), below 3.3e-5 here, far below the frozen image gate.
    check(1-(1+math.tan(POLICY['sun_angular_radius'])**2)**-1.5<3.3e-5)
    for views in (1,2,3,4):check({f%views for f in range(0,160,17)}==set(range(views)))
    check({(f//11)%4 for f in range(0,160,17)}=={0,1,2,3})
    jobs=plan_jobs(Path('/NOT_EXECUTED/phosphor'),Path('/NOT_EXECUTED'))
    check(all('--debug-meshlets-corrupt' in j['argv'] and j['expected_exit']==1 for j in jobs if j['overflow']))
    check(all(option(j['argv'],'--lighting-scene')!='disocclusion' for j in jobs if j['group']=='lifecycle'))
    print(f'f10_quality: {count} scalar CPU fixtures passed; no renderer/GPU invocation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=Path,default=Path('build/lighting/phosphor'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--only',action='append',default=[])
    parser.add_argument('--list',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--analyze',type=Path,help='analyze an existing captured manifest, no rendering')
    parser.add_argument('--timeout',type=float,default=300);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return 0
    if args.analyze:
        out=args.analyze.resolve();manifest=json.loads((out/'manifest.json').read_text())
        if manifest['policy_sha256']!=POLICY_SHA:raise ValueError('policy changed since capture; preserve old experiment')
        result=analyze(manifest,out);save_manifest(out/'quality.json',result)
        print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
    if not args.output and not args.list:parser.error('--output is required')
    out=(args.output or Path('NOT_EXECUTED')).resolve();planned=plan_jobs(args.app.resolve(),out)
    if args.only:
        for pattern in args.only:
            if not any(fnmatch.fnmatchcase(j['name'],pattern) for j in planned):parser.error(f'no job matches {pattern}')
        planned=[j for j in planned if any(fnmatch.fnmatchcase(j['name'],p) for p in args.only)]
    if args.list:print('\n'.join(j['name'] for j in planned));return 0
    if out.exists():parser.error('output must be a new directory; no evidence/threshold replacement')
    out.mkdir(parents=True)
    manifest={'schema':1,'state':'NOT_EXECUTED','created_utc':datetime.now(timezone.utc).isoformat(),
              'policy':POLICY,'policy_sha256':POLICY_SHA,'jobs':planned,'phase_accepted':False,
              'validation_env':VALIDATION_ENV,'required_capture_signals':['shadow','shadow-position','shadow-normal'],
              'historical_note':'Old disocclusion fixture has no sun; its functional result is not solar-history acceptance.'}
    save_manifest(out/'manifest.json',manifest)
    if not args.run:print(json.dumps(manifest,indent=2));return 0
    if not math.isfinite(args.timeout) or args.timeout<=0:parser.error('timeout must be finite and positive')
    try:
        import numpy  # Existing environment only; fail before GPU work if absent.
        for job in planned:
            folder=out/job['name'];folder.mkdir();job['state']='RUNNING';save_manifest(out/'manifest.json',manifest)
            status=run_checked(job['argv'],folder/'run.log',expected=job['expected_exit'],timeout=args.timeout,
                               required=(r'^LIGHTING check frame \d+ \| PASS$',),env={**os.environ,**VALIDATION_ENV})
            job['execution']=status
            report=json.loads((folder/'report.json').read_text())
            errors=functional(job,status,(folder/'run.log').read_text(errors='replace'),report)
            artifacts=[]
            for frame,path in frames_for(job,out):
                picture=read_pfm(path);artifacts.append({**file_record(path),'frame':frame,'shape':list(picture.shape)})
            job.update(functional_passed=not errors,failures=errors,artifacts=artifacts,state='CAPTURED' if not errors else 'FAIL')
            save_manifest(out/'manifest.json',manifest)
            print(f"{job['state']}: {job['name']}",flush=True)
            if errors:raise RuntimeError('; '.join(errors))
        result=analyze(manifest,out);save_manifest(out/'quality.json',result)
        manifest['state']='GATES_PASSED' if result['passed'] else 'FAILED_OR_INCOMPLETE_GATES'
        manifest['quality_passed']=result['passed'];save_manifest(out/'manifest.json',manifest)
        return 0 if result['passed'] else 1
    except (Exception,KeyboardInterrupt) as error:
        manifest['state']='FAILED_OR_INTERRUPTED';manifest['error']=str(error) or type(error).__name__
        if 'job' in locals() and job['state']=='RUNNING':
            job.update(state='FAILED_OR_INTERRUPTED',functional_passed=False,failures=[manifest['error']])
        save_manifest(out/'manifest.json',manifest);print(manifest['error'],file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
