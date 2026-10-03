#!/usr/bin/env python3
"""Compare deterministic renderer clips against a supersampled reference.

Install tools/quality-requirements.txt in a mise-created virtual environment.
Reference downsampling happens in linear light. Residual temporal change and
old-frame attraction complement spatial PSNR; no optical-flow metric is claimed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def linear(rgb):
    return np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)


def srgb(rgb):
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= 0.0031308, rgb*12.92, 1.055*rgb**(1/2.4)-0.055)


def read(path, size=None):
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)/255
    result = linear(rgb)
    if size is not None and (result.shape[1], result.shape[0]) != size:
        result = np.stack([np.asarray(Image.fromarray(result[:, :, c]).resize(size, Image.Resampling.BOX))
                           for c in range(3)], axis=-1)
    return result


def measure(reference, candidate, thresholds, cuts=(), lag=0, spatial_control=None):
    refs = sorted(Path(reference).glob('frame-*.png'))
    candidates = {p.name:p for p in Path(candidate).glob('frame-*.png')}
    if not refs or any(p.name not in candidates for p in refs):
        raise ValueError('Missing matching clip frames')
    size = Image.open(candidates[refs[0].name]).size
    controls={p.name:p for p in Path(spatial_control).glob("frame-*.png")} if spatial_control else {}
    if controls and any(p.name not in controls for p in refs):raise ValueError("Missing spatial-control frames")
    rmse, psnr, flicker, ghosts, excess_ghosts = [], [], [], [], []
    previous_ref = previous_error = None
    frames = []
    for i, path in enumerate(refs):
        ref = read(path, size)
        candidate_path = candidates[refs[max(0, i-lag)].name]
        cur = read(candidate_path)
        error = cur-ref
        rms = float(np.sqrt(np.mean(error*error)))
        encoded_error = srgb(cur)-srgb(ref)
        mse = float(np.mean(encoded_error*encoded_error))
        rmse.append(rms)
        psnr.append(100.0 if mse == 0 else -10*np.log10(mse))
        ghost_fraction = residual_flicker = excess_fraction = 0.0
        if previous_ref is not None:
            residual_flicker = float(np.mean(np.abs(error-previous_error)))
            change = np.mean(np.abs(ref-previous_ref), axis=2)
            new_error = np.mean(np.abs(cur-ref), axis=2)
            old_error = np.mean(np.abs(cur-previous_ref), axis=2)
            changed = change > 0.06
            ghost = changed & (new_error > 0.04) & (old_error+0.02 < new_error)
            ghost_fraction = float(np.count_nonzero(ghost)/max(1, np.count_nonzero(changed)))
            if controls:
                spatial=read(controls[path.name])
                spatial_error=np.mean(np.abs(spatial-ref),axis=2)
                excess=ghost & (new_error > spatial_error+0.02)
                excess_fraction=float(np.count_nonzero(excess)/max(1,np.count_nonzero(changed)))
        flicker.append(residual_flicker);ghosts.append(ghost_fraction);excess_ghosts.append(excess_fraction)
        frames.append({'frame':int(path.stem.split('-')[1]),'linear_rmse':rms,'psnr_srgb_db':float(psnr[-1]),
                       'residual_flicker':residual_flicker,'ghost_fraction':ghost_fraction,'excess_ghost_fraction':excess_fraction})
        previous_ref, previous_error = ref, error
    by_frame = {item['frame']:i for i,item in enumerate(frames)}
    recovery = {}
    for cut in cuts:
        if cut not in by_frame:
            raise ValueError(f'Missing cut frame {cut}')
        start = by_frame[cut]
        recovery[str(cut)] = next((frames[j]['frame']-cut for j in range(start,len(frames))
                                  if rmse[j] <= thresholds['recovery_linear_rmse']), len(frames))
    metrics = {'frames':len(frames),'mean_psnr_srgb_db':float(np.mean(psnr)),
               'linear_rmse_p95':float(np.percentile(rmse,95)),'linear_rmse_max':max(rmse),
               'residual_flicker_mean':float(np.mean(flicker)),
               'ghost_fraction_max':max(ghosts),'excess_ghost_fraction_max':max(excess_ghosts),'recovery_frames':recovery}
    checks = {
        'spatial_psnr':metrics['mean_psnr_srgb_db'] >= thresholds['mean_psnr_srgb_min_db'],
        'spatial_p95':metrics['linear_rmse_p95'] <= thresholds['linear_rmse_p95_max'],
        'worst_frame':metrics['linear_rmse_max'] <= thresholds['linear_rmse_max'],
        'flicker':metrics['residual_flicker_mean'] <= thresholds['residual_flicker_mean_max'],
        'ghosting':bool(controls) and metrics['excess_ghost_fraction_max'] <= thresholds['excess_ghost_fraction_max'],
        'recovery':all(n <= thresholds['recovery_frames_max'] for n in recovery.values())}
    return {'passed':all(checks.values()),'metrics':metrics,'checks':checks,'frames':frames}


def contact_sheet(reference, candidate, output, selected):
    first = Image.open(Path(candidate)/f'frame-{selected[0]:06d}.png')
    width,height = first.size
    sheet = Image.new('RGB',(width*3,(height+24)*len(selected)),(24,24,24))
    draw = ImageDraw.Draw(sheet)
    for row,frame in enumerate(selected):
        name=f'frame-{frame:06d}.png'
        ref=read(Path(reference)/name,(width,height));cur=read(Path(candidate)/name)
        difference=np.clip(np.abs(cur-ref)*4,0,1)
        for column,(pixels,label) in enumerate(((srgb(ref),'reference'),(srgb(cur),'candidate'),(difference,'linear error x4'))):
            image=Image.fromarray(np.uint8(np.clip(pixels,0,1)*255))
            y=row*(height+24)
            draw.text((column*width+5,y+5),f'{frame}: {label}',fill='white')
            sheet.paste(image,(column*width,y+24))
    sheet.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--thresholds',type=Path,default=Path(__file__).parent/'testdata/temporal_thresholds.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cuts',type=int,nargs='*',default=[])
    p.add_argument('--negative-lag',type=int,default=0)
    p.add_argument('--spatial-control',type=Path,required=True)
    p.add_argument('--sheet-frames',type=int,nargs='*',default=[])
    a=p.parse_args();thresholds=json.loads(a.thresholds.read_text())
    result=measure(a.reference,a.candidate,thresholds,a.cuts,a.negative_lag,a.spatial_control)
    result.update({'thresholds':thresholds,'reference':str(a.reference),'candidate':str(a.candidate),
                   'negative_lag':a.negative_lag,'spatial_control':str(a.spatial_control),'numpy':np.__version__})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    if a.sheet_frames:contact_sheet(a.reference,a.candidate,a.out.with_suffix('.png'),a.sheet_frames)
    print(json.dumps({'passed':result['passed'],'metrics':result['metrics'],'checks':result['checks']},indent=2))
    raise SystemExit(not result['passed'])

if __name__=='__main__':main()
