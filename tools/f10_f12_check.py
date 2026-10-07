#!/usr/bin/env python3
"""F10-F12 tester runner. Written only: default emits a plan, never executes it.

Use --run deliberately in the TESTER checkout after F9 alignment/build. Jobs
are serialized. Failures are retained; no baseline/reference is overwritten.
A passing functional job is not phase acceptance, convergence or hardware proof.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import sys


def jobs(app, out):
    base = [str(app), '--render-path', 'visibility', '--rt', 'on', '--frames', '64',
            '--warmup', '0', '--fixed-timestep', '--no-ui', '--offscreen',
            '--resolution', '640x360', '--no-vsync', '--lighting-seed', '1', '--debug-lighting', '1']
    result = []
    def add(name, flags, failure=False, frame=None):
        cmd = base + flags + ['--report', str(out/(name+'.json'))]
        if frame is not None:
            cmd += ['--capture-linear', str(out/(name+'.pfm')), '--capture-linear-frame', str(frame)]
        result.append({'name': name, 'argv': cmd, 'expected_failure': failure, 'state': 'NOT_EXECUTED'})
    add('f10_offscreen_caster', ['--bench','6','--lighting-scene','offscreen-caster','--shadows','csm'], frame=63)
    add('f10_indexed_floor', ['--bench','6','--lighting-scene','offscreen-caster','--shadows','csm','--force-family','apple9'], frame=63)
    add('f10_cache', ['--bench','6','--lighting-scene','cache-stress','--shadows','csm','--shadow-cache','on'], frame=63)
    add('f10_caster_negative', ['--bench','6','--lighting-scene','offscreen-caster','--shadows','csm','--debug-lighting-corrupt','caster'], True)
    add('f10_cache_negative', ['--bench','6','--lighting-scene','cache-stress','--shadows','csm','--shadow-cache','on','--debug-lighting-corrupt','cache'], True)
    add('f10_solar_bias', ['--bench','6','--lighting-scene','shadow-bias','--shadows','rt','--contact-shadows','on'], frame=63)
    add('f10_solar_bias_negative', ['--bench','6','--lighting-scene','shadow-bias','--shadows','rt','--debug-lighting-corrupt','bias'], frame=63)
    add('f10_history_rejection', ['--bench','6','--lighting-scene','disocclusion','--shadows','rt','--history-reset-every','8'], frame=63)
    for mode in ('brute','clustered','restir'):
        add('f11_'+mode, ['--bench','5','--local-light-count','8','--stationary-lights','--area-lights','--lighting',mode], frame=63)
    add('f11_many', ['--bench','5','--lighting','restir','--local-light-count','1024'], frame=63)
    add('f11_reduced', ['--bench','5','--lighting','restir','--lighting-preset','reduced','--force-family','apple9'], frame=63)
    for kind in ('pdf','light'):
        add('f11_negative_'+kind,['--bench','5','--lighting','restir','--local-light-count','8','--stationary-lights','--debug-lighting-corrupt',kind],True)
    for mode in ('ddgi','cache','restir'):
        add('f12_'+mode,['--bench','6','--lighting-scene','cornell','--lighting','restir','--gi',mode,
                        '--capture-linear-signal','indirect-diffuse'],frame=63)
    add('f12_embedded',['--bench','6','--lighting-scene','thin-walls','--lighting','restir','--gi','ddgi',
                       '--gi-probe-anchor','0.9,1,-0.8','--gi-spacing','0.5','--capture-linear-signal','indirect-diffuse'],frame=63)
    for scenario in ('moving-sun','moving-emissive','disocclusion'):
        add('f12_'+scenario,['--bench','6','--lighting-scene',scenario,'--shadows','rt','--lighting','restir','--gi','ddgi',
                            '--capture-linear-sequence',str(out/scenario),'--capture-every','8','--capture-linear-signal','indirect-diffuse'])
    # Exact snapshot and candidate at the SAME frame. External renderer remains
    # a separate explicit command, with its material-model gate retained.
    add('f12_reference_snapshot',['--bench','6','--lighting-scene','cornell','--lighting','restir','--gi','ddgi',
                                 '--export-reference',str(out/'cornell-snapshot'),'--export-reference-frame','63',
                                 '--capture-linear-signal','indirect-diffuse'],frame=63)
    for kind in ('cache','probe','pdf'):
        add('f12_negative_'+kind,['--bench','6','--lighting-scene','cornell','--lighting','restir','--gi','restir',
                                  '--debug-gi-corrupt',kind],True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=pathlib.Path,default=pathlib.Path('build/lighting/phosphor'))
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    planned=jobs(args.app.resolve(),out)
    manifest={'schema':1,'state':'NON_VERIFIED_WRITER_PLAN','validation_scope':'M5 development; physical M3 pending',
              'units':'metres; pre-exposure Float32 PFM; fixed seed 1','jobs':planned,
              'required_followup':['source review and build','GPU/API validation','negative controls','per-view lifecycle',
                                   'frozen regional thresholds before independent reference','linear convergence sweep 64/256/1024/4096',
                                   'leaks and zero steady-state GPU allocations','F9/F7/F8 regressions','F13 denoise boundary'],
              'external_reference_argv':[sys.executable,str(pathlib.Path(__file__).with_name('f12_reference.py')),'render',str(out/'cornell-snapshot'),
                                         '--output',str(out/'offline'),'--spp','64','256','1024','4096','--signal','indirect','--material-model','diffuse']}
    if not args.run:
        print(json.dumps(manifest,indent=2))
        return 0
    if out.exists():
        parser.error('output must be a fresh directory; existing references are never replaced')
    out.mkdir(parents=True)
    failures=[]
    for job in planned:
        log=out/(job['name']+'.log')
        with log.open('w') as stream:
            completed=subprocess.run(job['argv'],stdout=stream,stderr=subprocess.STDOUT)
        job['returncode']=completed.returncode
        if completed.returncode<0:
            job['state']='SIGNAL_FAILURE';failures.append(job['name']);continue
        expected=job['expected_failure']
        # Check explicit engine diagnostics, not an unrelated loader/parse error.
        text=log.read_text(errors='replace')
        failed='LIGHTING check frame ' in text and '| FAIL' in text
        okay=(completed.returncode!=0 and failed) if expected else completed.returncode==0
        if not expected and okay:
            report=json.loads((out/(job['name']+'.json')).read_text())
            okay=report.get('lighting',{}).get('checks',0)>0 and report['lighting'].get('failures',1)==0
        job['state']='FUNCTIONAL_RESULT_ONLY' if okay else 'FAIL'
        if not okay: failures.append(job['name'])
    manifest['state']='TESTER_FUNCTIONAL_RUN_ONLY';manifest['failures']=failures
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
