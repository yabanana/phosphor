#!/usr/bin/env python3
"""Known-value texture/sampling checks for the Mitsuba F12 adapter, CPU only."""
import argparse, copy, json, pathlib
import numpy as np
from f12_oracle_validate import load_reference

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-script',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ref=load_reference(a.reference_script);root,data=ref.snapshot(a.snapshot);out=pathlib.Path(a.output);out.mkdir(exist_ok=True,parents=True)
    # Four uniquely oriented texels. PFM round-trip preserves the top-left convention.
    rgb=np.asarray([[[1,0,0],[0,1,0]],[[0,0,1],[1,1,1]]],dtype=np.float32)
    alpha=np.repeat(np.asarray([[1,0],[.25,.75]],dtype=np.float32)[...,None],3,axis=2)
    ref.write_pfm(out/'known_rgb.pfm',rgb);ref.write_pfm(out/'known_alpha.pfm',alpha)
    assert np.array_equal(ref.read_pfm(out/'known_rgb.pfm'),rgb)
    data=copy.deepcopy(data);data['textures']=[{'id':0,'rgb':'known_rgb.pfm','alpha':'known_alpha.pfm'}]
    material=data['materials'][1];material['textures']=[0,0xffffffff,0xffffffff,0xffffffff,0];material['base']=[1,1,1,1];material['emissive']=[2,3,4];material['alpha_cutoff']=.5;material['metallic']=0
    mi,_,_=ref.build_scene(out,data,'diffuse',False)
    diffuse=mi.load_dict({'type':'phosphor_snapshot','material':1,'semantic':'diffuse'})
    emission=mi.load_dict({'type':'phosphor_snapshot','material':1,'semantic':'emissive'})
    mask=mi.load_dict({'type':'phosphor_snapshot','material':1,'semantic':'mask'})
    cases=[([.25,.25],[1,0,0],1),([.75,.25],[0,1,0],0),([.25,.75],[0,0,1],0),([.75,.75],[1,1,1],1),([-.25,.25],[0,1,0],0),([1.25,-.25],[0,0,1],0),([.5,.5],[.5,.5,.5],1),([.37,.63],[.3648,.24,.76],0)]
    results=[]
    for uv,expected,accepted in cases:
        si=mi.SurfaceInteraction3f();si.uv=uv
        expected=np.asarray(expected,dtype=np.float16).astype(np.float32)
        got=np.asarray(diffuse.eval_3(si));got_mask=float(mask.eval_1(si));got_emission=np.asarray(emission.eval_3(si));wanted_emission=expected*np.asarray([2,3,4])*accepted
        error=max(float(np.max(np.abs(got-expected))),float(np.max(np.abs(got_emission-wanted_emission))),abs(got_mask-accepted))
        results.append({'uv':uv,'max_error':error,'passed':error<1e-6})
    result={'cases':results,'uniform_area_proposal':not emission.is_spatially_varying(),'bsdf_texture_spatial':diffuse.is_spatially_varying(),'passed':all(r['passed'] for r in results) and not emission.is_spatially_varying()}
    (out/'texture_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
