#!/usr/bin/env python3
"""Apply the frozen F12 protocol; no ROI, exposure, seed or threshold fitting."""
import argparse, hashlib, json, pathlib
import numpy as np
from f12_oracle_validate import load_reference


def evaluate_images(ref, oracle, masks, images, protocol):
    """Pure frozen quality calculation shared by baseline and bounded grid runs."""
    names=list(protocol['regions']['slots']);candidate=protocol['candidate']
    frames=list(range(candidate['frames']['first'],candidate['frames']['last_exclusive'],candidate['frames']['step']))
    assert len(images)==len(frames) and all(image.shape==oracle.shape for image in images)
    stacked=np.asarray(images);mean=np.mean(stacked,axis=0,dtype=np.float64).astype(np.float32)
    lum=np.asarray([.2126,.7152,.0722]);scale=float(np.mean(oracle[masks['interior']]@lum))
    metrics={name:ref.metric(oracle[masks[name]],mean[masks[name]]) for name in names}
    roi_pass=all(v['relative_rmse']<=candidate['roi_relative_rmse_max'] and abs(v['relative_signed_bias'])<=candidate['roi_absolute_relative_luminance_bias_max'] for v in metrics.values())
    leak_mask=masks['occluded_floor'];valid_leak=int(leak_mask.sum())>=protocol['regions']['minimum_pixels']
    positive=np.maximum((mean-oracle)[leak_mask]@lum,0)/max(scale,1e-6)
    leak={'pixel_count':int(leak_mask.sum()),'sufficient':valid_leak,'mean_positive_over_interior':float(np.mean(positive)),'p95_positive_over_interior':float(np.percentile(positive,95)),'passed':bool(valid_leak and np.mean(positive)<=candidate['leak_positive_mean_over_reference_interior_mean_max'] and np.percentile(positive,95)<=candidate['leak_positive_p95_over_reference_interior_mean_max'])}
    per_frame=[{'frame':frame,**ref.metric(oracle[masks['interior']],image[masks['interior']])} for frame,image in zip(frames,images)]
    finite_nonnegative=bool(np.isfinite(stacked).all() and np.min(stacked)>=0)
    temporal_spread=float(np.sqrt(np.mean(np.var(stacked[:,masks['interior']],axis=0))))/max(float(np.sqrt(np.mean(oracle[masks['interior']]**2))),1e-6)
    return mean,{'roi_pass':roi_pass,'passed':roi_pass and leak['passed'] and finite_nonnegative,'regions':metrics,'leak':leak,'finite_nonnegative':finite_nonnegative,'minimum_pixel':float(stacked.min()),'maximum_pixel':float(stacked.max()),'temporal_standard_deviation_over_reference_rms':temporal_spread,'per_frame_interior':per_frame}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-script',required=True);p.add_argument('--reference-dir',required=True);p.add_argument('--protocol',required=True);p.add_argument('--candidate-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    ref=load_reference(a.reference_script);r=pathlib.Path(a.reference_dir);protocol=json.loads(pathlib.Path(a.protocol).read_text());report=json.loads((r/'reference_report.json').read_text())
    assert report['state']=='REFERENCE_CANDIDATE_CONVERGED' and report['snapshot_sha256']==protocol['snapshot_sha256']
    assert all(json.loads((r/f).read_text())['passed'] for f in ['geometry_validation.json','transport_validation.json','texture-fixture/texture_validation.json'])
    oracle=ref.read_pfm(r/report['reference']);masks=dict(np.load(r/'masks.npz'));names=list(protocol['regions']['slots']);candidate=protocol['candidate'];frames=list(range(candidate['frames']['first'],candidate['frames']['last_exclusive'],candidate['frames']['step']));lum=np.asarray([.2126,.7152,.0722]);scale=float(np.mean(oracle[masks['interior']]@lum))
    results={};out=pathlib.Path(a.output);out.mkdir(exist_ok=False,parents=True)
    for variant in candidate['variants']:
        root=pathlib.Path(a.candidate_root)/variant
        sr,_=ref.snapshot(root/'snapshot');assert ref.scene_digest(sr)==protocol['snapshot_sha256'],f'{variant}: scene/camera mismatch'
        # This is the first permitted access to candidate image pixels.
        images=[ref.read_pfm(root/'captures'/f'frame-{frame:06d}.pfm') for frame in frames]
        assert all(image.shape==oracle.shape for image in images)
        stacked=np.asarray(images);mean=np.mean(stacked,axis=0,dtype=np.float64).astype(np.float32);ref.write_pfm(out/f'{variant}_mean.pfm',mean)
        mean,results[variant]=evaluate_images(ref,oracle,masks,images,protocol)
        results[variant]['mean_image']=f'{variant}_mean.pfm'
    result={'schema':1,'state':'STATIC_CORNELL_QUALITY_ONLY_NOT_PHASE_ACCEPTANCE','protocol_sha256':hashlib.sha256(pathlib.Path(a.protocol).read_bytes()).hexdigest(),'snapshot_sha256':protocol['snapshot_sha256'],'reference_spp_per_seed':report['records'][-1]['spp'],'frames':frames,'region_pixels':{k:int(v.sum()) for k,v in masks.items()},'variants':results,'dynamic_recovery':'NOT_EXECUTED','leak_certification':'UNAVAILABLE_INSUFFICIENT_GEOMETRIC_ROI'}
    (out/'quality_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({v:{'roi_pass':q['roi_pass'],'passed':q['passed'],'bias':{k:m['relative_signed_bias'] for k,m in q['regions'].items()},'rmse':{k:m['relative_rmse'] for k,m in q['regions'].items()}} for v,q in results.items()},indent=2));return 0 if all(q['passed'] for q in results.values()) else 1
if __name__=='__main__':raise SystemExit(main())
