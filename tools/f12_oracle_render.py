#!/usr/bin/env python3
"""Run a preregistered CPU-only Cornell oracle without loading candidate images."""
import argparse, copy, json, pathlib, time
import numpy as np
from f12_oracle_validate import load_reference, native_constants


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-script',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    ref=load_reference(a.reference_script);root,data=ref.snapshot(a.snapshot);protocol=json.loads(pathlib.Path(a.protocol).read_text());out=pathlib.Path(a.output);out.mkdir(exist_ok=True,parents=True)
    assert ref.scene_digest(root)==protocol['snapshot_sha256']
    mi,sd,diffs=ref.build_scene(root,data,'diffuse',False);assert not diffs
    import drjit as dr
    dr.set_thread_count(2)
    native=native_constants(sd,data);masks=dict(np.load(out/'masks.npz'));roi_names=list(protocol['regions']['slots'])+['interior']
    def render(scene_dict,spp,seed,depth):
        scene_dict['integrator']['max_depth']=depth
        scene=mi.load_dict(scene_dict);start=time.monotonic();image=np.asarray(mi.render(scene,spp=spp,seed=seed),dtype=np.float32)
        print(json.dumps({'event':'render','spp':spp,'seed':seed,'depth':depth,'seconds':time.monotonic()-start}),flush=True)
        return image
    # Cornells all active materials untextured; BSDF equality was checked per hit.
    # Compare path samples too, rather than assuming native constant nodes equivalent.
    generic=render(sd,64,1234,-1);fast=render(native,64,1234,-1)
    specialization_error=float(np.max(np.abs(generic-fast)))
    direct=render(native,64,1234,2)
    doubled=native_constants(sd,data)
    for ins in data['instances']:
        shape=doubled[f"instance_{ins['slot']}"]
        if 'emitter' in shape:shape['emitter']['radiance']['value']=[2*v for v in data['materials'][ins['material']]['emissive']]
    double=render(doubled,64,1234,-1)
    linearity_error=float(np.max(np.abs(double-2*fast)))
    black=native_constants(sd,data)
    for ins in data['instances']:black[f"material_{ins['material']}"]['reflectance']['value']=[0,0,0]
    zero=render(black,64,1234,-1)-render(black,64,1234,2)
    ref.write_pfm(out/'validation_full_generic_64.pfm',generic);ref.write_pfm(out/'validation_full_native_64.pfm',fast);ref.write_pfm(out/'validation_zero_indirect_64.pfm',zero)
    validation={'native_constant_max_abs_error':specialization_error,'emission_linearity_max_abs_error':linearity_error,'zero_indirect_max_abs_error':float(np.max(np.abs(zero))),'passed':bool(specialization_error<1e-5 and linearity_error<1e-5 and np.max(np.abs(zero))<1e-7),'scalar_threads':2}
    (out/'transport_validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation),flush=True)
    if not validation['passed']:return 2
    if a.validate_only:return 0
    records=[];last=[];accepted=False
    for spp in protocol['reference']['spp_checkpoints']:
        last=[]
        for seed in protocol['reference']['seeds']:
            filename=out/f'indirect_spp{spp}_seed{seed}.pfm'
            if filename.exists():image=ref.read_pfm(filename)
            else:
                full=render(native,spp,seed,-1);direct=render(native,spp,seed,2);image=full-direct
                ref.write_pfm(out/f'full_spp{spp}_seed{seed}.pfm',full);ref.write_pfm(out/f'direct_spp{spp}_seed{seed}.pfm',direct);ref.write_pfm(filename,image)
            last.append(image)
        metrics={name:ref.metric(last[0][masks[name]],last[1][masks[name]]) for name in roi_names}
        accepted=all(v['relative_rmse']<=protocol['reference']['convergence_relative_rmse'] for v in metrics.values())
        record={'spp':spp,'metrics':metrics,'converged':accepted};records.append(record)
        (out/'convergence_progress.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
        if accepted:break
    reference=np.mean(last,axis=0,dtype=np.float64).astype(np.float32);ref.write_pfm(out/'reference_indirect.pfm',reference)
    depth8=render(native,spp,1234,8)-ref.read_pfm(out/f'direct_spp{spp}_seed1234.pfm')
    ref.write_pfm(out/f'indirect_depth8_spp{spp}.pfm',depth8)
    report={'schema':1,'snapshot_sha256':ref.scene_digest(root),'signal':'indirect diffuse reflected radiance','renderer':'Mitsuba3','renderer_version':mi.__version__,'variant':'scalar_rgb','full_depth':-1,'direct_depth':2,'rr_depth':5,'model_differences':[],'native_constants_validated':True,'roles_equivalent':True,'converged':accepted,'state':'REFERENCE_CANDIDATE_CONVERGED' if accepted else 'NON_ACCEPTED_REFERENCE','records':records,'depth8_vs_unlimited':{name:ref.metric(last[0][masks[name]],depth8[masks[name]]) for name in roi_names},'reference':'reference_indirect.pfm','validation':'geometry, UV, units, one-sided emitter, decomposition, constant specialization and emission linearity checked CPU; static Cornell only'}
    (out/'reference_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':report['state'],'spp':spp}),flush=True)
    return 0 if accepted else 1
if __name__=='__main__':raise SystemExit(main())
