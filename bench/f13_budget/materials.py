#!/usr/bin/env python3
"""Audit primary material factors; GPU witnesses are not independent references."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'tools'))
from f12_reference import snapshot,build_scene,primary_uv_footprint
from f12_oracle_validate import ply
spec=importlib.util.spec_from_file_location('budget_reference',REPO/'bench/f13_budget/reference.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot',type=Path)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists():parser.error('refuse existing audit overwrite')
root,data=snapshot(args.snapshot)
w=json.loads((root/data['primary_material_witness']).read_text())
if w.get('kind')!='gpu-primary-material-witness' or w.get('independent') is not False:
 raise ValueError('unsupported witness role')
if (w['frame'],w['width'],w['height'])!=(data['frame'],data['camera']['width'],data['camera']['height']) or len(w['pixels'])!=w['width']*w['height']:
 raise ValueError('witness identity or extent mismatch')
if w['width']<80 or w['height']<48:raise ValueError('fixed audit crop outside film')
mi,desc,differences=build_scene(root,data,'diffuse',False)
scene=mi.load_dict(desc,optimize=False);records=[]
instances={i['slot']:i for i in data['instances']}
meshes=[ply(root/name) for name in data['meshes']]
for row in range(24,48):
 for col in range(48,80):
  g=w['pixels'][row*w['width']+col]
  ray,_=scene.sensors()[0].sample_ray(0,0,[(col+.5)/w['width'],(row+.5)/w['height']],[0,0])
  _,hit=ref.primary_receiver(mi,scene,ray)
  if g is None or not hit.is_valid():
   records.append({'xy':[col,row],'matched':False});continue
  error=float(np.max(np.abs(np.asarray(hit.p)-g['position'])))
  slot=int(hit.shape.id().removeprefix('instance_'))
  instance=instances[slot]
  vertices,faces=meshes[instance['mesh']]
  triangle=vertices[faces[int(hit.prim_index)]]
  world=np.c_[triangle[:,:3],np.ones(3)]@np.asarray(instance['world']).reshape(4,4,order='F').T
  uv,dx,dy=primary_uv_footprint(data['camera'],world[:,:3],triangle[:,6:8],[col+.5,row+.5])
  cpu=np.asarray(hit.bsdf().eval_diffuse_reflectance(hit),dtype=float)
  gpu=np.asarray(g['albedo'])*(1-g['metallic'])
  records.append({'xy':[col,row],'matched':error<=1e-4 and slot==g['slot'],
    'position_error':error,'gpu_slot':g['slot'],'cpu_slot':slot,'material':g['material'],
    'uv':uv,'duv_dx':dx,'duv_dy':dy,'uv_error':float(np.max(np.abs(np.asarray(uv)-np.asarray(hit.uv)))),'gpu_diffuse':gpu.tolist(),'cpu_lod0_diffuse':cpu.tolist(),
    'absolute_error':float(np.max(np.abs(cpu-gpu)))})
matched=[r for r in records if r['matched']]
if not matched:raise ValueError('no matched primary positions in audit crop')
out={'frame':data['frame'],'diagnostic_only':True,'reference_accepted':False,'pixels':len(records),
     'matched_geometry':len(matched),'max_uv_error':max(r['uv_error'] for r in matched),'max_diffuse_absolute_error':max(r['absolute_error'] for r in matched),
     'p95_diffuse_absolute_error':float(np.percentile([r['absolute_error'] for r in matched],95)),
     'pixels_with_diffuse_error_over_1e_3':sum(r['absolute_error']>1e-3 for r in matched),'records':records}
args.out.write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='records'}))
