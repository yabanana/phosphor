#!/usr/bin/env python3
"""Compare only the preregistered F12 grid presets against the frozen oracle."""
import argparse, hashlib, json, math, pathlib
import numpy as np
from f12_oracle_validate import load_reference, ply
from f12_oracle_compare import evaluate_images


def option(command,name,default=None):
    if name not in command:return default
    index=command.index(name)
    if index+1>=len(command):raise ValueError(f'missing value for {name}')
    return command[index+1]


def validate_run(case,report,status,protocol):
    command=status['command']
    if not status.get('passed'):raise ValueError(f"{case['id']}: runtime check did not pass")
    if option(command,'--gi')!=case['mode']:raise ValueError('GI mode mismatch')
    actual_grid=tuple(map(int,option(command,'--gi-grid','8x4x8').split('x')))
    if actual_grid!=tuple(case['grid']):raise ValueError('grid mismatch')
    if '--gi-spacing' in command or '--gi-probe-anchor' in command:raise ValueError('fixed-volume sweep forbids spacing/anchor overrides')
    for flag,value in [('--resolution','128x72'),('--lighting-seed','1001'),('--capture-every','8'),('--warmup','256'),('--frames','256'),('--export-reference-frame','511')]:
        if option(command,flag)!=value:raise ValueError(f'{flag} differs from frozen sweep')
    if report['lighting']['gi_rays']!=protocol['ray_count_per_probe']:raise ValueError('ray budget mismatch')
    if report['frames']!=256 or report['rendering']['input_width_last']!=128 or report['rendering']['input_height_last']!=72:raise ValueError('capture dimensions/frame count mismatch')
    if report['hardware']['effective_capabilities']!='apple10':raise ValueError('this fixed grid sweep is native Apple10, not a forced family run')


def fitted_volume(root,data,counts):
    # Independent reconstruction from exported vertices and WORLD transforms.
    # Matches the declared conservative sphere-fit contract, not readback of
    # actual GPU probe placement. Relocation/classification remains unobserved.
    meshes=[ply(root/name)[0][:,:3] for name in data['meshes']]
    low=np.full(3,np.inf);high=-low
    world_points=[]
    for instance in data['instances']:
        points=meshes[instance['mesh']];centre=points.mean(axis=0);radius=np.max(np.linalg.norm(points-centre,axis=1))
        transform=np.asarray(instance['world']).reshape(4,4,order='F');linear=transform[:3,:3];gram=linear.T@linear
        scale=np.sqrt(np.max(np.sum(np.abs(gram),axis=1)));c=(transform@np.r_[centre,1])[:3]
        low=np.minimum(low,c-radius*scale);high=np.maximum(high,c+radius*scale)
        world_points.append(np.c_[points,np.ones(len(points))]@transform.T)
    origin=low-.25;spacing=(high-low+.5)/(np.asarray(counts)-1)
    coordinates=np.indices(counts).reshape(3,-1).T;positions=origin+coordinates*spacing
    interior=np.all((positions>[-2,0,-2])&(positions<[2,4,2]),axis=1);in_boxes=0
    for point in positions[interior]:
        for instance in data['instances']:
            if instance['slot'] not in [64,65]:continue
            local=np.linalg.inv(np.asarray(instance['world']).reshape(4,4,order='F'))@np.r_[point,1]
            if np.all(np.abs(local[:3])<.5):in_boxes+=1;break
    points=np.concatenate(world_points)[:,:3]
    return {'method':'CPU geometry reconstruction before relocation, not GPU probe readback','sphere_fit_min':low.tolist(),'sphere_fit_max':high.tolist(),'true_world_aabb_min':points.min(axis=0).tolist(),'true_world_aabb_max':points.max(axis=0).tolist(),'origin':origin.tolist(),'spacing':spacing.tolist(),'total_probes':len(positions),'initially_inside_room':int(interior.sum()),'of_inside_room_inside_boxes':in_boxes,'y_layers':(origin[1]+np.arange(counts[1])*spacing[1]).tolist()}


def cost_metadata(report,status):
    gi_passes=[p for p in report['passes'] if any(s.startswith(('ddgi_','gi_','radiance_cache')) for s in p.get('shaders',[]))]
    validation=status.get('validation_env',{})
    validated=any(str(validation.get(k,'0')).lower() not in ['0','','false'] for k in ['MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION'])
    return {'scope':'SINGLE_VALIDATION_RUN_NOT_PERFORMANCE_ADOPTION' if validated else 'SINGLE_RUN_NOT_PERFORMANCE_ADOPTION','measured_frames':report['frames'],'gpu_frame_ms':report['gpu_ms'],'cpu_ms':report['cpu_ms'],'gpu_allocations':report['gpu_allocations'],'engine_resource_bytes_last':report['rendering']['engine_resource_bytes_last'],'device_allocated_bytes_last':report['rendering']['device_allocated_bytes_last'],'gi_passes':[{k:p[k] for k in ['name','frames','gpu_ms']} for p in gi_passes],'gi_pass_min_samples':min((p['frames'] for p in gi_passes),default=0),'gi_pass_summary_sufficient':bool(gi_passes and all(p['frames']>=report['frames']*.9 for p in gi_passes)),'validation_env':validation,'gpu_timing':report['gpu_timing'],'commit':status['commit'],'binary_sha256':status['binary_sha256'],'artifacts':status.get('artifacts',{}),'tracked_patch_sha256':status.get('tracked_patch_sha256'),'command':status['command'],'hardware':report['hardware']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-script','reference-dir','protocol','grid-protocol','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--case',action='append',required=True,help='Frozen case ID=directory; all six required')
    args=parser.parse_args();paths={}
    for value in args.case:
        name,separator,path=value.partition('=')
        if not separator or name in paths:raise ValueError('malformed or duplicate case mapping')
        paths[name]=pathlib.Path(path)
    protocol=json.loads(pathlib.Path(args.protocol).read_text());grid_protocol=json.loads(pathlib.Path(args.grid_protocol).read_text())
    cases=grid_protocol['cases'];assert set(paths)=={c['id'] for c in cases},'exactly the preregistered case set required'
    ref=load_reference(args.reference_script);reference_dir=pathlib.Path(args.reference_dir)
    reference_report=json.loads((reference_dir/'reference_report.json').read_text())
    assert reference_report['state']=='REFERENCE_CANDIDATE_CONVERGED' and reference_report['snapshot_sha256']==protocol['snapshot_sha256']==grid_protocol['snapshot_sha256']
    for file in ['geometry_validation.json','transport_validation.json','texture-fixture/texture_validation.json']:assert json.loads((reference_dir/file).read_text())['passed']
    oracle=ref.read_pfm(reference_dir/reference_report['reference']);masks=dict(np.load(reference_dir/'masks.npz'))
    frames=list(range(protocol['candidate']['frames']['first'],protocol['candidate']['frames']['last_exclusive'],protocol['candidate']['frames']['step']))
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);result={}
    for case in cases:
        name=case['id'];root=paths[name];report=json.loads((root/'report.json').read_text());status=json.loads((root/'run.log.status.json').read_text())
        validate_run(case,report,status,grid_protocol)
        snapshot_root,data=ref.snapshot(root/'snapshot');assert ref.scene_digest(snapshot_root)==protocol['snapshot_sha256'],f'{name}: scene mismatch'
        images=[ref.read_pfm(root/'captures'/f'frame-{frame:06d}.pfm') for frame in frames]
        mean,quality=evaluate_images(ref,oracle,masks,images,protocol);ref.write_pfm(out/f'{name}_mean.pfm',mean)
        result[name]={'config':case,'quality':quality,'cost':cost_metadata(report,status),'volume':fitted_volume(snapshot_root,data,case['grid']),'ray_work_per_frame':math.prod(case['grid'])*grid_protocol['ray_count_per_probe']}
    record={'schema':1,'state':'BOUNDED_STATIC_CORNELL_COMPARISON_NOT_PHASE_ACCEPTANCE','quality_protocol_sha256':hashlib.sha256(pathlib.Path(args.protocol).read_bytes()).hexdigest(),'grid_protocol_sha256':hashlib.sha256(pathlib.Path(args.grid_protocol).read_bytes()).hexdigest(),'snapshot_sha256':protocol['snapshot_sha256'],'reference_spp_per_seed':reference_report['records'][-1]['spp'],'cases':result,'leak_certification':'UNAVAILABLE_INSUFFICIENT_GEOMETRIC_ROI','dynamic_recovery':'NOT_EXECUTED','performance_adoption':False}
    (out/'grid_quality_cost.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({name:{'roi_pass':r['quality']['roi_pass'],'worst_nrmse':max(m['relative_rmse'] for m in r['quality']['regions'].values()),'worst_absolute_bias':max(abs(m['relative_signed_bias']) for m in r['quality']['regions'].values()),'frame_gpu_p50_ms':r['cost']['gpu_frame_ms']['p50'],'frame_gpu_p95_ms':r['cost']['gpu_frame_ms']['p95'],'pass_min_samples':r['cost']['gi_pass_min_samples'],'initial_inside_room':r['volume']['initially_inside_room'],'total_probes':r['volume']['total_probes']} for name,r in result.items()},indent=2))
    return 0 if all(r['quality']['roi_pass'] for r in result.values()) else 1
if __name__=='__main__':raise SystemExit(main())
