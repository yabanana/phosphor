#!/usr/bin/env python3
"""Compare a physical thin-wall positive/negative pair against the frozen oracle."""
import argparse, hashlib, json, pathlib
import numpy as np
from f12_oracle_validate import load_reference
from f12_grid_compare import option, cost_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-script','reference-dir','regions','protocol','positive','negative','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();ref=load_reference(args.reference_script);reference_dir=pathlib.Path(args.reference_dir);reference_report=json.loads((reference_dir/'reference_report.json').read_text());assert reference_report['state']=='REFERENCE_CANDIDATE_CONVERGED'
    protocol=json.loads(pathlib.Path(args.protocol).read_text())['thin_wall'];region_dir=pathlib.Path(args.regions);region_info=json.loads((region_dir/'regions.json').read_text());assert region_info['coverage_passed'] and region_info['snapshot_sha256']==reference_report['snapshot_sha256']
    masks=dict(np.load(region_dir/'masks.npz'));bright=masks['bright_control'];dark=masks['dark_behind_wall'];oracle=ref.read_pfm(reference_dir/reference_report['reference']);lum=np.asarray([.2126,.7152,.0722]);bright_scale=float(np.mean(oracle[bright]@lum));assert bright_scale>1e-6
    frames=range(protocol['candidate_frames']['first'],protocol['candidate_frames']['last_exclusive'],protocol['candidate_frames']['step']);records={};means={};statuses={}
    frozen_options=['--gi','--gi-grid','--gi-spacing','--gi-probe-anchor','--lighting','--lighting-denoise','--gi-rays','--capture-linear-signal','--resolution','--lighting-seed','--capture-every','--frames','--warmup','--export-reference-frame']
    for name in ['positive','negative']:
        root=pathlib.Path(getattr(args,name));status=json.loads((root/'run.log.status.json').read_text());report=json.loads((root/'report.json').read_text());command=status['command'];statuses[name]=status
        if not status.get('passed') or report['lighting']['failures']!=0:raise ValueError('physical negative must keep ordinary invariants PASS')
        disabled=report['lighting'].get('gi_visibility_disabled')
        if disabled is not (name=='negative') or ('--debug-gi-no-visibility' in command)!=(name=='negative'):raise ValueError('explicit runtime visibility flag disagrees with pair member')
        for flag,value in [('--resolution','256x144'),('--lighting-scene','thin-walls'),('--lighting-seed','1001'),('--capture-every','8'),('--frames','256'),('--warmup','256'),('--export-reference-frame','511')]:
            if option(command,flag)!=value:raise ValueError(f'wrong frozen configuration: {flag}')
        if option(command,'--capture-linear-signal') not in ['indirect-diffuse','indirect-diffuse-filtered']:raise ValueError('wrong signal')
        snapshot_root,_=ref.snapshot(root/'snapshot');assert ref.scene_digest(snapshot_root)==reference_report['snapshot_sha256'],'physical scene/camera mismatch'
        images=[ref.read_pfm(root/'captures'/f'frame-{frame:06d}.pfm') for frame in frames];assert all(image.shape==oracle.shape for image in images)
        stacked=np.asarray(images);mean=np.mean(stacked,axis=0,dtype=np.float64).astype(np.float32);means[name]=mean;bright_metric=ref.metric(oracle[bright],mean[bright]);positive=np.maximum((mean-oracle)[dark]@lum,0)/bright_scale
        leak_mean=float(np.mean(positive));leak_p95=float(np.percentile(positive,95));leak_pass=leak_mean<=protocol['dark_mean_positive_error_over_bright_reference_mean_max'] and leak_p95<=protocol['dark_p95_positive_error_over_bright_reference_mean_max'];bright_pass=bright_metric['relative_rmse']<=protocol['bright_control_nrmse_max'] and abs(bright_metric['relative_signed_bias'])<=protocol['bright_control_absolute_relative_bias_max'];finite_nonnegative=bool(np.isfinite(stacked).all() and stacked.min()>=0)
        records[name]={'bright':bright_metric,'bright_passed':bright_pass,'dark_mean_positive_error_over_bright_reference_mean':leak_mean,'dark_p95_positive_error_over_bright_reference_mean':leak_p95,'leak_passed':leak_pass,'finite_nonnegative':finite_nonnegative,'quality_passed':bool(bright_pass and leak_pass and finite_nonnegative),'runtime_invariants_passed':True,'cost':cost_metadata(report,status)}
    a,b=statuses['positive'],statuses['negative']
    if a['binary_sha256']!=b['binary_sha256']:raise ValueError('positive/negative must use the same frozen executable')
    for flag in frozen_options:
        if option(a['command'],flag)!=option(b['command'],flag):raise ValueError(f'pair differs beyond physical ablation: {flag}')
    negative_detected=not records['negative']['leak_passed'];passed=records['positive']['quality_passed'] and negative_detected and records['negative']['finite_nonnegative']
    result={'schema':1,'state':'STATIC_THIN_WALL_QUALITY_AND_PHYSICAL_NEGATIVE_ONLY','passed':bool(passed),'negative_detected_by_leak_predicate':negative_detected,'protocol_sha256':hashlib.sha256(pathlib.Path(args.protocol).read_bytes()).hexdigest(),'snapshot_sha256':reference_report['snapshot_sha256'],'regions':region_info,'reference_spp_per_seed':reference_report['records'][-1]['spp'],'bright_reference_luminance_mean':bright_scale,'cases':records,'recovery':'SEPARATE_NOT_CERTIFIED'}
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for name,image in means.items():ref.write_pfm(out/f'{name}_mean.pfm',image)
    (out/'thin_quality_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':passed,'negative_detected':negative_detected,'cases':{name:{key:value for key,value in record.items() if key!='cost'} for name,record in records.items()}},indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
