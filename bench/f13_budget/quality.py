#!/usr/bin/env python3
"""Frozen same-binary equivalence protocol for the selected F13/F14 cost fixes."""
import os,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from run_checked import run_checked
from f13_f14_check import gpu_lock
from temporal_light_metrics import read_pfm
import argparse
parser=argparse.ArgumentParser(description="F13/F14 same-binary image-equivalence experiment; default plan only")
parser.add_argument('--binary',type=Path,default=Path('build/lighting/phosphor'))
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--run',action='store_true')
parser.add_argument('--resolution',default='322x182')
args=parser.parse_args()
root=args.out.resolve()
if root.exists():raise SystemExit("Refuse to overwrite an existing experiment directory")
root.mkdir(parents=True)
base=[str(args.binary.resolve()),'--render-path','visibility','--geometry-path','mesh','--rt','on',
      '--lighting','restir','--lighting-denoise','custom','--post','--upscaler','native',
      '--resolution',args.resolution,'--frames','32','--warmup','0','--fixed-timestep','--no-vsync','--no-ui',
      '--scene',str(Path('assets/sponza/Sponza.gltf').resolve()),'--bench','4','--shadows','rt',
      '--gi','ddgi','--gi-grid','16x8x16','--gi-rays','64','--reflections','rt','--ao','rtao',
      '--atmosphere','on','--fog','on','--clouds','on','--no-gpu-timing','--debug-lighting','1']
limits={'tile_normalized_max':5e-5,'aerial_relative_max':.01,'relative_floor_peak_fraction':.001}
(root/'protocol.json').write_text(json.dumps({'base':base,'limits':limits,'views':['noon','sunrise','night'],'frames':32,'frozen_before_gpu':True},indent=2))
if not args.run:
 print('PLAN ONLY: pass --run to execute serial GPU work');raise SystemExit(0)
with gpu_lock(Path('/tmp/phosphor-gpu-verification.lock')):
 results=[]
 for label,hour in [('noon','12'),('sunrise','6'),('night','0')]:
  for variant,reference,aerial in [('reference','1','1'),('tiled','0','1'),('adaptive','0','0')]:
   out=root/(label+'-'+variant);env={k:v for k,v in os.environ.items() if not k.startswith('PHOSPHOR_DIAGNOSTIC')};env.update(MTL_DEBUG_LAYER='1',MTL_SHADER_VALIDATION='1',PHOSPHOR_DIAGNOSTIC_DENOISE_REFERENCE=reference,PHOSPHOR_DIAGNOSTIC_AERIAL_REFERENCE=aerial)
   cmd=base+['--start-hour',hour,'--capture-linear-sequence',str(out/'frames')]
   (out).mkdir(parents=True,exist_ok=True)
   (out/'command.json').write_text(json.dumps({'command':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('PHOSPHOR_DIAGNOSTIC','MTL_DEBUG','MTL_SHADER'))}},indent=2))
   status=run_checked(cmd,out/'renderer.log',env=env,timeout=180,
      required=[r'F13 atrous backend: '+('reference' if reference=='1' else r'tiled strides 1/2, reference 4\+'),
                r'F14 aerial quadrature: '+('reference' if aerial=='1' else 'distance-adaptive')])
   result={'name':label+'-'+variant,'status':status['passed'],'failures':status['failures']}
   if not status['passed']:
    result['passed']=False;results.append(result);print(result,flush=True);break
   if variant!='reference':
    worst=0.;relmax=0.;square=0.;n=0
    actual=sorted((out/'frames').glob('*.pfm'));expected=sorted((root/(label+'-reference')/'frames').glob('*.pfm'))
    if len(actual)!=32 or [f.name for f in actual]!=[f.name for f in expected]:raise RuntimeError('Missing/extra GPU captures')
    for f in sorted((out/'frames').glob('*.pfm')):
     a=np.asarray(read_pfm(f).rgb,dtype=np.float64);b=np.asarray(read_pfm(root/(label+'-reference')/'frames'/f.name).rgb,dtype=np.float64)
     if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise RuntimeError('Malformed/nonfinite capture')
     scale=max(float(np.max(np.abs(b))),1e-6);error=np.abs(a-b);worst=max(worst,float(error.max())/scale);relmax=max(relmax,float((error/np.maximum(np.abs(b),scale*.001)).max()));square+=float(np.sum((error/scale)**2));n+=a.size
    result.update(normalized_max=worst,relative_max=relmax,normalized_rmse=(square/n)**.5 if n else None,values=n)
    result['passed']=bool(n) and (worst<=5e-5 if variant=='tiled' else relmax<=.01)
   else:result['passed']=True
   results.append(result);print(result,flush=True)
 (root/'results.json').write_text(json.dumps(results,indent=2))

 raise SystemExit(0 if len(results)==9 and all(r['passed'] for r in results) else 1)
