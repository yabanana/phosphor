#!/usr/bin/env python3
"""Check scalar/LLVM CPU texture semantics on a captured scene."""
import sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'tools'))
from f12_reference import build_scene,snapshot
import argparse,subprocess,json,hashlib
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot',type=Path)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--worker',choices=('scalar_rgb','llvm_ad_rgb'))
args=parser.parse_args()
if not args.worker:
 args.out.mkdir(parents=True)
 for variant in ('scalar_rgb','llvm_ad_rgb'):
  subprocess.run([sys.executable,str(Path(__file__)),str(args.snapshot),'--out',str(args.out),'--worker',variant],check=True)
 a,b=[np.load(args.out/(v+'.npz')) for v in ('scalar_rgb','llvm_ad_rgb')]
 assert set(a.files)==set(b.files)
 comparisons={k:{'max_error':float(np.max(abs(a[k]-b[k]))),'equal':bool(np.array_equal(a[k],b[k]))} for k in a.files}
 result={'passed':all(v['equal'] for v in comparisons.values()),'values':sum(a[k].size for k in a.files),
         'comparisons':comparisons,'seed':20261110,'runner_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (args.out/'results.json').write_text(json.dumps(result,indent=2))
 print('CPU texture parity:',result['passed'],'values:',result['values'])
 raise SystemExit(0 if result['passed'] else 1)
variant=args.worker;root,data=snapshot(args.snapshot)
mi,_,_=build_scene(root,data,'diffuse',False,variant)
rng=np.random.default_rng(20261110)
uv=np.concatenate([rng.uniform(-3,3,(1024,2)),np.array([[0,0],[1,1],[-1,-1],[.5,.5],[.5/1024,.5/1024]])]).astype('float32')
outputs={}
for material in range(len(data['materials'])):
 for semantic in ['base','diffuse','metallic','roughness','mask','emissive','normal']:
  texture=mi.load_dict({'type':'phosphor_snapshot','material':material,'semantic':semantic})
  if variant=='scalar_rgb':
   values=[]
   for pair in uv:
    si=mi.SurfaceInteraction3f();si.uv=mi.Point2f(*pair);values.append(np.asarray(texture.eval(si)))
   value=np.array(values)
  else:
   import drjit as dr
   si=dr.zeros(mi.SurfaceInteraction3f,len(uv));si.uv=mi.Point2f(uv[:,0],uv[:,1])
   value=np.asarray(texture.eval(si)).T
  outputs[f'{material}-{semantic}']=value
np.savez(args.out/(variant+'.npz'),**outputs)
