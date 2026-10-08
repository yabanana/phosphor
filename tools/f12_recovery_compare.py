#!/usr/bin/env python3
"""Measure emissive-step convergence without conflating F12 and F13 gates."""
import argparse, hashlib, json, pathlib
import numpy as np
from f12_oracle_validate import load_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-script','reference-dir','derived-reference-dir','protocol','quality-protocol','captures','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();ref=load_reference(args.reference_script);reference_dir=pathlib.Path(args.reference_dir);derived=pathlib.Path(args.derived_reference_dir)
    provenance=json.loads((derived/'step_snapshot_validation.json').read_text())
    assert provenance['passed'] and provenance['state']=='DERIVED_LINEAR_TRANSPORT_REFERENCE' and provenance['scale']==.5
    original=json.loads((reference_dir/'reference_report.json').read_text());assert original['state']=='REFERENCE_CANDIDATE_CONVERGED' and original['snapshot_sha256']==provenance['source_reference_snapshot_sha256']
    before_ref=ref.read_pfm(reference_dir/original['reference']);after_ref=ref.read_pfm(derived/provenance['reference_after']);assert np.array_equal(after_ref,before_ref*.5)
    protocol=json.loads(pathlib.Path(args.protocol).read_text())['recovery'];quality=json.loads(pathlib.Path(args.quality_protocol).read_text());masks=dict(np.load(reference_dir/'masks.npz'));names=list(quality['regions']['slots']);lum=np.asarray([.2126,.7152,.0722])
    captures=pathlib.Path(args.captures);before=protocol['baseline_candidate_frames'];transition=protocol['transition_frame'];bound=protocol['required_max_recovery_frames']
    def read(frame):
        image=ref.read_pfm(captures/f'frame-{frame:06d}.pfm')
        if image.shape!=before_ref.shape or np.min(image)<0:raise ValueError('wrong dimensions or negative candidate')
        return image
    pre_images=[read(frame) for frame in range(before['first'],before['last_exclusive'],before['step'])]
    pre_mean=np.mean(pre_images,axis=0,dtype=np.float64).astype(np.float32);pre_metrics={name:ref.metric(before_ref[masks[name]],pre_mean[masks[name]]) for name in names}
    def within(metric):return metric['relative_rmse']<=quality['candidate']['roi_relative_rmse_max'] and abs(metric['relative_signed_bias'])<=quality['candidate']['roi_absolute_relative_luminance_bias_max']
    scale={name:.5*float(np.mean(pre_mean[masks[name]]@lum)) for name in names}
    baseline_ok=all(within(v) for v in pre_metrics.values()) and min(scale.values())>1e-6
    def evaluate(image):
        metrics={name:ref.metric(after_ref[masks[name]],image[masks[name]]) for name in names}
        residual={name:abs(float(np.mean(image[masks[name]]@lum))-scale[name])/max(scale[name],1e-6) for name in names}
        return {'regions':metrics,'old_state_residual':residual,'physical_quality_pass':all(within(v) for v in metrics.values()),'history_residual_pass':all(v<=protocol['history_residual_max'] for v in residual.values())}
    post=[read(transition+offset) for offset in range(bound+1)];records=[]
    for offset,image in enumerate(post):
        instant=evaluate(image);trailing=evaluate(np.mean(post[max(0,offset-7):offset+1],axis=0,dtype=np.float64).astype(np.float32))
        valid=baseline_ok and instant['physical_quality_pass'] and instant['history_residual_pass'] and trailing['physical_quality_pass'] and trailing['history_residual_pass']
        records.append({'offset':offset,'frame':transition+offset,'instantaneous':instant,'trailing_up_to8_frames':trailing,'passed':valid})
    passing=[r['offset'] for r in records if r['passed']]
    sustained=[offset for offset in range(bound+1) if all(r['passed'] for r in records[offset:])]
    checkpoints=[records[offset] for offset in protocol['checkpoint_offsets']]
    negatives={name:evaluate(image) for name,image in [('retained_old_state',pre_mean),('black_post_state',np.zeros_like(pre_mean))]}
    assert all(not result['physical_quality_pass'] and not result['history_residual_pass'] for result in negatives.values()),'negative control must fail'
    report={'schema':1,'protocol_sha256':hashlib.sha256(pathlib.Path(args.protocol).read_bytes()).hexdigest(),'quality_protocol_sha256':hashlib.sha256(pathlib.Path(args.quality_protocol).read_bytes()).hexdigest(),'state':'F12_RECOVERY_ONLY_F13_HISTORY_GATE_SEPARATE','passed':bool(baseline_ok and records[-1]['passed']),'baseline_passed':baseline_ok,'baseline_regions':pre_metrics,'transition_frame':transition,'checkpoints':checkpoints,'first_instantaneous_and_trailing_pass_offset':passing[0] if passing else None,'first_pass_offset_that_stays_passing_through128':sustained[0] if sustained else None,'all_frames':records,'negative_controls':negatives,'f13_history_rejection_within4':'NOT_CERTIFIED_BY_IMAGE_CONVERGENCE; inspect separate history diagnostics at1/4','reference_derivation':provenance}
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'recovery_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['state','passed','baseline_passed','first_instantaneous_and_trailing_pass_offset','first_pass_offset_that_stays_passing_through128']},indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
