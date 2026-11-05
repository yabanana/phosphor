#!/usr/bin/env python3
"""Frozen dynamic-Sponza equivalence and A/B/A protocol; default plan only.

Equivalence isolates the selected cost changes. It is not an independent
physical lighting reference and cannot qualify F13.3 or the whole F13.6.
"""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from f13_f14_check import gpu_lock,sha,source_hash
from run_checked import run_checked
from temporal_light_metrics import read_pfm

REFERENCE={"PHOSPHOR_DIAGNOSTIC_DENOISE_GENERIC":"1",
           "PHOSPHOR_DIAGNOSTIC_DENOISE_GUIDE4_REFERENCE":"1",
           "PHOSPHOR_DIAGNOSTIC_DENOISE_UNBATCHED":"1"}
CAP=5e-5
DRIFT=.05
FRAME_BUDGET=1000/60


def compare(reference,candidate,frames):
    left=sorted(reference.glob('frame-*.pfm'));right=sorted(candidate.glob('frame-*.pfm'))
    if len(left)!=frames or [f.name for f in left]!=[f.name for f in right]:raise ValueError('incomplete frame sequence')
    worst=0.;squares=0.;count=0
    for a,b in zip(left,right):
        x=read_pfm(a);y=read_pfm(b)
        if (x.width,x.height)!=(y.width,y.height):raise ValueError('image extent mismatch')
        r=np.asarray(x.rgb,dtype=np.float64);c=np.asarray(y.rgb,dtype=np.float64)
        if not np.isfinite(r).all() or not np.isfinite(c).all():raise ValueError('nonfinite capture')
        error=np.abs(r-c)/max(float(np.max(np.abs(r))),1e-6)
        worst=max(worst,float(error.max()));squares+=float(np.sum(error*error));count+=error.size
    return {'normalized_max':worst,'normalized_rmse':(squares/count)**.5,'values':count,'passed':worst<=CAP}


def history(folder,frames,frozen_clock):
    records=[json.loads(p.read_text()) for p in sorted(folder.glob('frame-*.json'))]
    if len(records)!=frames:raise ValueError('missing immutable camera/history sidecars')
    reused=set();cuts=[]
    for ordinal,m in enumerate(records):
        if m.get('schema')!='phosphor.dynamic-lighting-capture.v1' or m.get('frame')!=ordinal:raise ValueError('capture identity mismatch')
        if len(m['camera']['view'])!=16 or len(m['camera']['unjittered_view_projection'])!=16:raise ValueError('camera witness missing')
        h=m['history']
        if set(h)!={'0','1','2','3'}:raise ValueError('selected signal checkpoint missing')
        for name,state in h.items():
            if state['frame']!=ordinal or state['view']!=m['view'] or state['invalid'] or state['invalid_output']:raise ValueError('invalid completed GPU checkpoint')
            if state['reused']:reused.add(name)
            if m['camera_cut'] and (not state['reset'] or state['reused']):raise ValueError('camera cut reused stale history')
        if m['camera_cut']:cuts.append(ordinal)
    if frozen_clock and reused!={'0','1','2','3'}:raise ValueError('camera sequence never exercised all signal histories')
    if frames>360 and len(cuts)<3:raise ValueError('script did not exercise three cuts')
    return {'frames':len(records),'reused_signals':sorted(reused),'cuts':cuts}


def timing(a,b,c):
    drift=abs(c['p50']-a['p50'])/a['p50']
    return {'baseline_drift':drift,'valid':drift<=DRIFT,
            'p50_reduction':1-b['p50']/((a['p50']+c['p50'])/2),
            'candidate_budget_passed':b['p95']<=FRAME_BUDGET}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=Path('build/release/phosphor'))
    p.add_argument('--scene',type=Path,default=Path('assets/sponza/Sponza.gltf'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--mode',choices=['quality','perf'],default='quality')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        a={'p50':30,'p95':32};b={'p50':27,'p95':29}
        assert timing(a,b,a)['valid'] and abs(timing(a,b,a)['p50_reduction']-.1)<1e-12
        assert not timing(a,b,{'p50':32,'p95':33})['valid']
        assert not timing(a,b,a)['candidate_budget_passed']
        assert not timing(a,{'p50':16.,'p95':16.6668},a)['candidate_budget_passed']
        print('dynamic Sponza evaluator self-test: PASS');return
    args.out.mkdir(parents=True) # immutable experiment, refuse overwrite
    base=[str(args.binary.resolve()),'--scene',str(args.scene.resolve()),'--bench','4',
          '--render-path','visibility','--geometry-path','mesh','--rt','on','--shadows','rt',
          '--lighting','restir','--gi','ddgi','--gi-grid','16x8x16','--gi-rays','64',
          '--reflections','rt','--ao','rtao','--lighting-denoise','custom','--atmosphere','on',
          '--fog','on','--clouds','on','--post','--upscaler','native','--temporal-script',
          '--fixed-timestep','--offscreen','--no-ui','--no-vsync']
    cases=[]
    if args.mode=='quality':
        for scene,flags in [('camera',['--atmo-freeze-clock']),('sun',['--day-length','64']),
                            ('apple9',['--force-family','apple9','--day-length','64'])]:
            for variant in ['reference','candidate']:
                cases.append({'name':scene+'-'+variant,'reference':variant=='reference',
                    'flags':['--resolution','322x182','--frames','384','--warmup','0','--no-gpu-timing','--debug-lighting','1',*flags]})
    else:
        for replica in range(3):
            for variant in ['A','B','A2']:
                cases.append({'name':f'{replica+1}-{variant}','reference':variant!='B',
                    'flags':['--resolution','1920x1080','--frames','256','--warmup','128','--gpu-timing-serial','--day-length','64']})
    scene=json.loads(args.scene.read_text())
    assets={uri:sha(args.scene.parent/uri) for section in ['buffers','images'] for item in scene.get(section,[])
            if (uri:=item.get('uri')) and not uri.startswith('data:')}
    archive=args.binary.parent/'shaders/phosphor-archive.metallib'
    identity={'binary':sha(args.binary),'metallib':sha(args.binary.parent/'shaders/phosphor.metallib'),
              'archive':sha(archive) if archive.is_file() else None,
              'scene':sha(args.scene),'external_assets':assets,'runner':sha(Path(__file__)),
              'source':source_hash(Path(__file__).resolve().parents[2])}
    protocol={'base':base,'cases':cases,'identity':identity,'reference_env':REFERENCE,'image_cap':CAP,
              'baseline_drift_cap':DRIFT,'gpu_p95_budget_ms':FRAME_BUDGET,'independent_reference':False,'frozen_before_gpu':True}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if not args.run:print('PLAN ONLY: '+str(args.out/'protocol.json'));return
    results={}
    with gpu_lock(Path('/tmp/phosphor-gpu-verification.lock')):
        for case in cases:
            if sha(args.binary)!=identity['binary'] or sha(args.binary.parent/'shaders/phosphor.metallib')!=identity['metallib']:raise RuntimeError('build changed')
            if (sha(archive) if archive.is_file() else None)!=identity['archive']:raise RuntimeError('archive changed')
            if source_hash(Path(__file__).resolve().parents[2])!=identity['source']:raise RuntimeError('source changed')
            out=args.out/case['name'];out.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('PHOSPHOR_DIAGNOSTIC_DENOISE')}
            if case['reference']:env.update(REFERENCE)
            command=base+case['flags']+['--report',str(out/'renderer.json')]
            if args.mode=='quality':
                env.update(MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1',MTL_DEBUG_LAYER_WARNING_MODE='nslog')
                command+=['--capture-linear-sequence',str(out/'frames')]
            else:
                for k in ['MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION','MTL_DEBUG_LAYER_WARNING_MODE']:env.pop(k,None)
            status=run_checked(command,out/'renderer.log',env=env,timeout=600)
            if not status['passed']:raise RuntimeError(str(status['failures']))
            report=json.loads((out/'renderer.json').read_text());results[case['name']]={'gpu_ms':report['gpu_ms'],'gpu_allocations':report['gpu_allocations']}
            counters=re.findall(r'archive hits (\d+), misses (\d+), unavailable (\d+)',(out/'renderer.log').read_text())
            if not counters:raise RuntimeError('missing pipeline archive counters')
            results[case['name']]['archive']=dict(zip(['hits','misses','unavailable'],map(int,counters[-1])))
            if args.mode=='quality':results[case['name']]['history']=history(out/'frames',384,case['name'].startswith('camera-'))
            elif report['gpu_allocations']:raise RuntimeError('steady GPU allocations')
            (args.out/'runs.json').write_text(json.dumps(results,indent=2)+'\n');print(case['name']+': PASS',flush=True)
    if args.mode=='quality':
        result={scene:compare(args.out/(scene+'-reference/frames'),args.out/(scene+'-candidate/frames'),384) for scene in ['camera','sun','apple9']}
        result['passed']=all(x['passed'] for x in result.values())
    else:
        result={'replicas':[timing(*(results[f'{i}-{v}']['gpu_ms'] for v in ['A','B','A2'])) for i in range(1,4)]}
        result['three_valid_replicas']=all(r['valid'] for r in result['replicas'])
        result['budget_passed']=all(r['valid'] and r['candidate_budget_passed'] for r in result['replicas'])
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if args.mode=='quality' and not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
