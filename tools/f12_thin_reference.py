#!/usr/bin/env python3
"""CPU-only bounded Mitsuba reference for the preregistered thin-wall masks."""
import argparse, hashlib, json, pathlib, time
import numpy as np
from f12_oracle_validate import load_reference, native_constants, geometry, ray, intersect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-script','snapshot','regions','protocol','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();ref=load_reference(args.reference_script);root,data=ref.snapshot(args.snapshot);protocol=json.loads(pathlib.Path(args.protocol).read_text())['thin_wall'];region_dir=pathlib.Path(args.regions);region_info=json.loads((region_dir/'regions.json').read_text())
    assert region_info['coverage_passed'] and region_info['snapshot_sha256']==ref.scene_digest(root)
    masks=dict(np.load(region_dir/'masks.npz'));bright=masks['bright_control'];dark=masks['dark_behind_wall'];selected=bright|dark
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
    mi,scene_dict,differences=ref.build_scene(root,data,'diffuse',False);assert not differences
    import drjit as dr
    dr.set_thread_count(2)
    scene=mi.load_dict(scene_dict);sensor=scene.sensors()[0];triangles,slots,uvs=geometry(root,data);camera=data['camera'];native=native_constants(scene_dict,data)
    maxima=np.zeros(4);wrong_hit=0
    for y,x in np.argwhere(selected):
        ro,rd=ray(camera,x+.5,y+.5);hit=intersect(triangles,ro,rd);assert hit is not None;i,t,bary=hit
        mr,_=sensor.sample_ray(0,0,[(x+.5)/camera['width'],(y+.5)/camera['height']],[0,0]);si=scene.ray_intersect(mr)
        if not si.is_valid() or si.shape.id()!=f'instance_{slots[i]}':wrong_hit+=1;continue
        normal=np.cross(triangles[i,1]-triangles[i,0],triangles[i,2]-triangles[i,0]);normal/=np.linalg.norm(normal)
        values=[np.max(np.abs(rd-np.asarray(mr.d))),np.max(np.abs(ro+t*rd-np.asarray(si.p))),np.max(np.abs(bary@uvs[i]-np.asarray(si.uv))),np.max(np.abs(normal-np.asarray(si.n)))];maxima=np.maximum(maxima,values)
    geometry_validation={'pixels':int(selected.sum()),'wrong_hits':wrong_hit,'max_ray_direction_error':maxima[0],'max_world_error':maxima[1],'max_uv_error':maxima[2],'max_normal_error':maxima[3],'passed':bool(wrong_hit==0 and np.all(maxima<[1e-6,1e-4,1e-5,1e-6]))}
    (out/'geometry_validation.json').write_text(json.dumps(geometry_validation,indent=2)+'\n');assert geometry_validation['passed']
    def render(dictionary,spp,seed,depth):
        dictionary['integrator']['max_depth']=depth;loaded=mi.load_dict(dictionary);start=time.monotonic();image=np.asarray(mi.render(loaded,spp=spp,seed=seed),dtype=np.float32);print(json.dumps({'spp':spp,'seed':seed,'depth':depth,'seconds':time.monotonic()-start}),flush=True);return image
    generic=render(scene_dict,64,1234,-1);optimized=render(native,64,1234,-1);maximum_error=float(np.max(np.abs(generic-optimized)))
    ref.write_pfm(out/'generic_full64.pfm',generic);ref.write_pfm(out/'native_full64.pfm',optimized)
    specialization={'max_abs_error':maximum_error,'passed':maximum_error<1e-5,'scope':'constant untextured active materials only; common texture/linearity adapter previously validated on Cornell'}
    (out/'specialization_validation.json').write_text(json.dumps(specialization,indent=2)+'\n');assert specialization['passed']
    records=[];converged=False;images=[];lum=np.asarray([.2126,.7152,.0722])
    for spp in protocol['reference']['spp_checkpoints']:
        images=[]
        for seed in protocol['reference']['seeds']:
            full=render(native,spp,seed,-1);direct=render(native,spp,seed,2);indirect=full-direct
            for name,image in [('full',full),('direct',direct),('indirect',indirect)]:ref.write_pfm(out/f'{name}_spp{spp}_seed{seed}.pfm',image)
            images.append(indirect)
        brightness=float(np.mean(np.mean(images,axis=0)[bright]@lum));assert brightness>1e-6
        bright_metric=ref.metric(images[0][bright],images[1][bright]);delta=(images[1]-images[0])[dark]
        dark_rmse=float(np.sqrt(np.mean(delta*delta)))/brightness;dark_bias=abs(float(np.mean(delta@lum)))/brightness
        gates=protocol['reference'];converged=bright_metric['relative_rmse']<=gates['bright_convergence_nrmse_max'] and dark_rmse<=gates['dark_seed_difference_rmse_over_bright_mean_max'] and dark_bias<=gates['dark_seed_mean_difference_over_bright_mean_max']
        record={'spp':spp,'bright':bright_metric,'bright_reference_luminance_mean':brightness,'dark_seed_difference_rmse_over_bright_mean':dark_rmse,'dark_seed_mean_difference_over_bright_mean':dark_bias,'converged':converged};records.append(record);(out/'convergence_progress.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(record),flush=True)
        if converged:break
    reference=np.mean(images,axis=0,dtype=np.float64).astype(np.float32);ref.write_pfm(out/'reference_indirect.pfm',reference)
    report={'schema':1,'state':'REFERENCE_CANDIDATE_CONVERGED' if converged else 'NON_ACCEPTED_REFERENCE','converged':converged,'renderer':'Mitsuba3','renderer_version':mi.__version__,'variant':'scalar_rgb','cpu_threads':2,'snapshot_sha256':ref.scene_digest(root),'protocol_sha256':hashlib.sha256(pathlib.Path(args.protocol).read_bytes()).hexdigest(),'reference_script_sha256':hashlib.sha256(pathlib.Path(args.reference_script).read_bytes()).hexdigest(),'records':records,'reference':'reference_indirect.pfm','full_depth':-1,'direct_depth':2,'rr_depth':5,'model_differences':[],'roles_equivalent':True,'regions':region_info,'geometry_validation':geometry_validation,'constant_specialization':specialization,'candidate_images_read':False}
    (out/'reference_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':report['state'],'spp':spp}),flush=True);return 0 if converged else 1
if __name__=='__main__':raise SystemExit(main())
