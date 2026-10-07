#!/usr/bin/env python3
"""Validate physical emissive-step snapshots before deriving a linear oracle."""
import argparse, copy, hashlib, json, pathlib
from f12_oracle_validate import load_reference


def physical(data):
    result=copy.deepcopy(data)
    for key in ['frame','revisions']:result.pop(key,None)
    # Generation counters identify temporal resources, not physical transport.
    for item in result['instances']+result['sampled_lights']:item.pop('generation',None)
    return result


def equal_assets(left,right):
    names=lambda p:sorted(f.name for f in p.iterdir() if f.suffix in ['.ply','.pfm'])
    if names(left)!=names(right):raise ValueError('snapshot asset set changed')
    for name in names(left):
        if (left/name).read_bytes()!=(right/name).read_bytes():raise ValueError(f'physical asset changed: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-script','baseline-snapshot','before-snapshot','after-snapshot','reference-dir','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();ref=load_reference(args.reference_script)
    baseline_root,baseline=ref.snapshot(args.baseline_snapshot);before_root,before=ref.snapshot(args.before_snapshot);after_root,after=ref.snapshot(args.after_snapshot)
    if before['frame']!=255 or after['frame']!=256:raise ValueError('export exactly before255 and after256')
    equal_assets(baseline_root,before_root);equal_assets(before_root,after_root)
    if physical(baseline)!=physical(before):raise ValueError('pretransition scene differs physically from the validated Cornell reference')
    if before['lights'] or any(before['sky']):raise ValueError('linearity proof requires the unique area emitter and zero other light/sky')
    expected=physical(before);emitters=[]
    for index,material in enumerate(expected['materials']):
        if any(material['emissive']):
            if material['emissive']!=[12,12,12]:raise ValueError('expected Le12 in the sole emitting material')
            emitters.append(index);material['emissive']=[6,6,6]
    if len(emitters)!=1:raise ValueError('unique emitting material required')
    for light in expected['sampled_lights']:
        if light['type']!=6 or light.get('material')!=emitters[0]:raise ValueError('unexpected sampled-light source')
        light['emission']=[value*.5 for value in light['emission']]
    if expected!=physical(after):raise ValueError('posttransition scene is not exactly the same scene with half emission')
    source=pathlib.Path(args.reference_dir);provenance=json.loads((source/'reference_report.json').read_text())
    if provenance['state']!='REFERENCE_CANDIDATE_CONVERGED' or provenance['snapshot_sha256']!=ref.scene_digest(baseline_root):raise ValueError('independent baseline reference is not accepted for this scene')
    reference=ref.read_pfm(source/provenance['reference']);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);ref.write_pfm(out/'reference_after.pfm',reference*.5)
    report={'schema':1,'passed':True,'state':'DERIVED_LINEAR_TRANSPORT_REFERENCE','before_snapshot_sha256':ref.scene_digest(before_root),'after_snapshot_sha256':ref.scene_digest(after_root),'source_reference_snapshot_sha256':provenance['snapshot_sha256'],'source_reference_spp_per_seed':provenance['records'][-1]['spp'],'derivation':'exact linear transport: sole emitter Le12 to Le6, geometry/material response/camera/textures unchanged','scale':.5,'ignored_diagnostic_fields':['frame','revisions','instance/sample generation'],'reference_after':'reference_after.pfm'}
    (out/'step_snapshot_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
