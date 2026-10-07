#!/usr/bin/env python3
"""F10-F12 tester runner. Written only: default emits a plan, never executes it.

Use --run deliberately in the TESTER checkout after F9 alignment/build. Jobs
are serialized with exact exit/checker evidence, timeouts and API/shader
validation. --only GLOB/--frames select functional microcases; no retries.
Every invocation needs a new output directory, including a plan-only invocation.
The bias negative is an IMAGE/ORACLE control: its captures remain pending until
an independent, predeclared quality predicate is applied. A functional result
is never phase acceptance, image-quality acceptance, convergence or hardware proof.
"""
from __future__ import annotations
import argparse
from array import array
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile

from run_checked import run_checked


def jobs(app, out, frames=64):
    base = [str(app), '--render-path', 'visibility', '--rt', 'on', '--frames', str(frames),
            '--warmup', '0', '--fixed-timestep', '--no-ui', '--offscreen',
            '--resolution', '640x360', '--no-vsync', '--lighting-seed', '1', '--debug-lighting', '1']
    result = []
    def add(name, flags, failure=False, frame=None, quality_control=None):
        cmd = base + flags + ['--report', str(out/(name+'.json'))]
        if frame is not None:
            cmd += ['--capture-linear', str(out/(name+'.pfm')), '--capture-linear-frame', str(frame)]
        result.append({'name': name, 'argv': cmd, 'expected_failure': failure, 'expected_exit': 1 if failure else 0,
                       'quality_control': quality_control, 'quality_control_pending': quality_control is not None,
                       'state': 'NOT_EXECUTED'})
    add('f10_offscreen_caster', ['--bench','6','--lighting-scene','offscreen-caster','--shadows','csm'], frame=frames-1)
    add('f10_indexed_floor', ['--bench','6','--lighting-scene','offscreen-caster','--shadows','csm','--force-family','apple9'], frame=frames-1)
    add('f10_cache', ['--bench','6','--lighting-scene','cache-stress','--shadows','csm','--shadow-cache','on'], frame=frames-1)
    add('f10_caster_negative', ['--bench','6','--lighting-scene','offscreen-caster','--shadows','csm','--debug-lighting-corrupt','caster'], True)
    add('f10_cache_negative', ['--bench','6','--lighting-scene','cache-stress','--shadows','csm','--shadow-cache','on','--debug-lighting-corrupt','cache'], True)
    add('f10_solar_bias', ['--bench','6','--lighting-scene','shadow-bias','--shadows','rt','--contact-shadows','on'], frame=frames-1)
    add('f10_solar_bias_negative', ['--bench','6','--lighting-scene','shadow-bias','--shadows','rt','--contact-shadows','on','--debug-lighting-corrupt','bias'], frame=frames-1,
        quality_control={'kind':'image_oracle_negative','positive_job':'f10_solar_bias',
                         'requirement':'Separate frozen image/physical-oracle predicate; checker/process success is not detection.'})
    add('f10_history_rejection', ['--bench','6','--lighting-scene','disocclusion','--shadows','rt','--history-reset-every','8'], frame=frames-1)
    for mode in ('brute','clustered','restir'):
        add('f11_'+mode, ['--bench','5','--local-light-count','8','--stationary-lights','--area-lights','--lighting',mode], frame=frames-1)
    add('f11_many', ['--bench','5','--lighting','restir','--local-light-count','1024'], frame=frames-1)
    add('f11_reduced', ['--bench','5','--lighting','restir','--lighting-preset','reduced','--force-family','apple9'], frame=frames-1)
    for kind in ('pdf','light'):
        add('f11_negative_'+kind,['--bench','5','--lighting','restir','--local-light-count','8','--stationary-lights','--debug-lighting-corrupt',kind],True)
    for mode in ('ddgi','cache','restir'):
        add('f12_'+mode,['--bench','6','--lighting-scene','cornell','--lighting','restir','--gi',mode,
                        '--capture-linear-signal','indirect-diffuse'],frame=frames-1)
    add('f12_embedded',['--bench','6','--lighting-scene','thin-walls','--lighting','restir','--gi','ddgi',
                       '--gi-probe-anchor','0.9,1,-0.8','--gi-spacing','0.5','--capture-linear-signal','indirect-diffuse'],frame=frames-1)
    for scenario in ('moving-sun','moving-emissive','disocclusion'):
        add('f12_'+scenario,['--bench','6','--lighting-scene',scenario,'--shadows','rt','--lighting','restir','--gi','ddgi',
                            '--capture-linear-sequence',str(out/scenario),'--capture-every','8','--capture-linear-signal','indirect-diffuse'])
    # Exact snapshot and candidate at the SAME frame. External renderer remains
    # a separate explicit command, with its material-model gate retained.
    add('f12_reference_snapshot',['--bench','6','--lighting-scene','cornell','--lighting','restir','--gi','ddgi',
                                 '--export-reference',str(out/'cornell-snapshot'),'--export-reference-frame',str(frames-1),
                                 '--capture-linear-signal','indirect-diffuse'],frame=frames-1)
    for kind in ('cache','probe','pdf'):
        add('f12_negative_'+kind,['--bench','6','--lighting-scene','cornell','--lighting','restir','--gi','restir',
                                  '--debug-gi-corrupt',kind],True)
    return result


CHECK = re.compile(r'^LIGHTING check frame (\d+) \| (PASS|FAIL)$', re.M)
UNRELATED_ERROR = re.compile(
    r'failed assertion|\[ERROR\]|\berror:|GPU timeout|command buffers failed|'
    r'Shader Validation Error|Metal Validation Error|MTLCommandBufferErrorDomain|'
    r'GPU (?:page fault|hang|execution error)|IOAF code|kIOGPUCommandBufferCallbackError|'
    r'validation[^\n]*(?:failed|error)', re.I)
VALIDATION_ENV = {'MTL_DEBUG_LAYER':'1', 'MTL_SHADER_VALIDATION':'1', 'MTL_DEBUG_LAYER_WARNING_MODE':'nslog'}


def option(argv, flag, default=None):
    indices = [i for i, value in enumerate(argv[:-1]) if value == flag]
    return argv[indices[-1]+1] if indices else default


def expected_checks(argv):
    frames, warmup = int(option(argv, '--frames')), int(option(argv, '--warmup', '0'))
    cadence = int(option(argv, '--debug-lighting', '0'))
    if frames <= 0 or warmup < 0 or cadence <= 0:
        raise ValueError('functional jobs need positive frames/check cadence and nonnegative warmup')
    return list(range(cadence-1, warmup+frames, cadence))


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def evaluate(job, status, text, report):
    """Only exact process/checker evidence; never a lighting-quality verdict."""
    failures = list(status.get('failures', []))
    expected = job['expected_exit']
    if status.get('returncode') != expected:
        failures.append(f'raw exit must equal {expected}')
    if status.get('exit_marker') != expected:
        failures.append(f'final EXIT marker must equal {expected}')
    if status.get('signal') or status.get('returncode', 0) is not None and status.get('returncode', 0) < 0:
        failures.append('a signal is never a successful negative control')
    if status.get('timed_out'):
        failures.append('renderer timed out')
    if UNRELATED_ERROR.search(text):
        failures.append('unrelated runtime/GPU/validation error in log')
    # The intended LIGHTING failure is the only FAIL line an expected negative
    # may accept. Other engine self-check failures remain failures of this run.
    without_checks = CHECK.sub('', text)
    if re.search(r'\| FAIL\b', without_checks):
        failures.append('another engine checker failed')
    checked = [(int(frame), outcome) for frame, outcome in CHECK.findall(text)]
    observed = [frame for frame, _ in checked]
    expected_frames = expected_checks(job['argv'])
    if sorted(observed) != expected_frames:
        failures.append('checker frame coverage differs from requested cadence (missing/duplicate/tail readbacks)')
    failed_checks = sum(outcome == 'FAIL' for _, outcome in checked)
    if expected == 1 and failed_checks == 0:
        failures.append('negative control did not trigger the lighting checker')
    if expected == 0 and failed_checks:
        failures.append('positive lighting checker failed')
    if not isinstance(report, dict):
        failures.append('renderer JSON report is required')
        report = {}
    if not integer(report.get('schema_version')) or report.get('schema_version', 0) < 10:
        failures.append('report requires schema_version >= 10')
    lighting = report.get('lighting')
    if not isinstance(lighting, dict):
        failures.append('lighting report is missing')
        lighting = {}
    checks, errors = lighting.get('checks'), lighting.get('failures')
    if not integer(checks) or checks <= 0 or checks != len(checked):
        failures.append('lighting.checks disagrees with actual checker lines')
    if not integer(errors) or errors != failed_checks:
        failures.append('lighting.failures disagrees with actual checker failures')
    if not integer(report.get('frames')) or report.get('frames') != int(option(job['argv'], '--frames')):
        failures.append('renderer did not complete the requested measured frames')
    requested = {
        'shadows':option(job['argv'], '--shadows', 'off'),
        'direct':option(job['argv'], '--lighting', 'legacy'),
        'gi':option(job['argv'], '--gi', 'off'),
        'seed':int(option(job['argv'], '--lighting-seed', '1')),
        'contact':option(job['argv'], '--contact-shadows', 'off') == 'on',
        'shadow_cache':option(job['argv'], '--shadow-cache', 'off') == 'on',
        'preset':'reduced' if option(job['argv'], '--force-family') == 'apple9'
                 else option(job['argv'], '--lighting-preset', 'full'),
    }
    for field, value in requested.items():
        if type(lighting.get(field)) is not type(value) or lighting.get(field) != value:
            failures.append(f'lighting.{field} differs from requested {value!r}')
    if option(job['argv'], '--force-family') == 'apple9':
        hardware = report.get('hardware')
        if not isinstance(hardware, dict) or hardware.get('effective_capabilities') != 'apple9':
            failures.append('forced family was not reported as effective apple9')
    pending = bool(job.get('quality_control'))
    return {'functional_passed':not failures, 'failures':failures,
            'expected_check_frames':expected_frames, 'checker_lines':len(checked), 'checker_failures':failed_checks,
            'lighting_report':lighting, 'quality_control_pending':pending, 'quality_passed':None,
            'state':'FAIL' if failures else 'QUALITY_CONTROL_PENDING' if pending else 'FUNCTIONAL_RESULT_ONLY'}


def file_record(path):
    path = pathlib.Path(path)
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def pfm_record(path):
    """Check float capture integrity only. No image-error/energy threshold."""
    with pathlib.Path(path).open('rb') as stream:
        if stream.readline(32).strip() != b'PF':
            raise ValueError(f'{path}: expected RGB Float32 PFM')
        width, height = map(int, stream.readline(128).split())
        scale = float(stream.readline(64))
        if width < 1 or height < 1 or width > 32768 or height > 32768 or not math.isfinite(scale) or scale == 0:
            raise ValueError(f'{path}: invalid PFM dimensions/scale')
        raw = stream.read()
    if len(raw) != width*height*12:
        raise ValueError(f'{path}: truncated or padded PFM')
    values = array('f'); values.frombytes(raw)
    if (scale < 0) != (sys.byteorder == 'little'):
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{path}: nonfinite capture pixels')
    return {**file_record(path), 'width':width, 'height':height, 'quality_evaluated':False}


def collect_outputs(job):
    records = []
    capture = option(job['argv'], '--capture-linear')
    if capture:
        records.append({**pfm_record(capture), 'requested_frame':int(option(job['argv'], '--capture-linear-frame')),
                        'signal':option(job['argv'], '--capture-linear-signal', 'hdr')})
    sequence = option(job['argv'], '--capture-linear-sequence')
    if sequence:
        folder = pathlib.Path(sequence)
        total = int(option(job['argv'], '--frames')) + int(option(job['argv'], '--warmup', '0'))
        cadence = int(option(job['argv'], '--capture-every', '1'))
        expected = {f'frame-{frame:06d}.pfm' for frame in range(0, total, cadence)}
        observed = {path.name for path in folder.glob('*.pfm')}
        if observed != expected:
            raise ValueError(f'{folder}: capture sequence has missing/extra frames')
        records.extend(pfm_record(folder/name) for name in sorted(expected))
    snapshot = option(job['argv'], '--export-reference')
    if snapshot:
        root = pathlib.Path(snapshot)
        path = root/'scene.json'
        scene = json.loads(path.read_text())
        if scene.get('frame') != int(option(job['argv'], '--export-reference-frame')) or scene.get('linear') is not True:
            raise ValueError('reference snapshot does not describe the requested same frame/linear signal')
        records.append(file_record(path))
        names = scene['meshes'] + [texture[key] for texture in scene['textures'] for key in ('rgb','alpha')]
        for name in names:
            source = root/name
            if not source.resolve().is_relative_to(root.resolve()):
                raise ValueError('snapshot resource escapes export directory')
            records.append(pfm_record(source) if source.suffix == '.pfm' else file_record(source))
    return records


def save_manifest(path, manifest):
    temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(manifest, indent=2)+'\n')
    temporary.replace(path)


def self_test():
    """CPU fixtures, including the real status helper; never starts phosphor."""
    count = 0
    with tempfile.TemporaryDirectory(prefix='phosphor-f10-runner-') as folder:
        root = pathlib.Path(folder)
        positive = jobs(pathlib.Path('/unexecuted/phosphor'), root, 4)[0]
        negative = next(job for job in jobs(pathlib.Path('/unexecuted/phosphor'), root, 4) if job['expected_exit'] == 1)
        bias = next(job for job in jobs(pathlib.Path('/unexecuted/phosphor'), root, 4) if job.get('quality_control'))
        def report_for(job, failed=0, checks=4):
            return {'schema_version':10,'frames':4,'lighting':{
                'shadows':option(job['argv'],'--shadows','off'), 'direct':option(job['argv'],'--lighting','legacy'),
                'gi':option(job['argv'],'--gi','off'), 'seed':1,'contact':option(job['argv'],'--contact-shadows','off')=='on',
                'shadow_cache':False,'preset':'full','checks':checks,'failures':failed}}
        status = {'returncode':0,'exit_marker':0,'timed_out':False,'signal':None,'failures':[]}
        clean = ''.join(f'LIGHTING check frame {i} | PASS\n' for i in range(4))+'EXIT 0\n'
        bad = ''.join(f'LIGHTING check frame {i} | FAIL\n' for i in range(4))+'EXIT 1\n'
        def expect(want, job=positive, st=None, text=None, report=None):
            nonlocal count
            result = evaluate(job, status if st is None else st, clean if text is None else text,
                              report_for(job) if report is None else report)
            if result['functional_passed'] != want:
                raise AssertionError(result)
            count += 1
            return result
        expect(True)
        for field, value in [('returncode',2),('returncode',-11),('exit_marker',None),('timed_out',True),('signal',6)]:
            expect(False, st={**status,field:value})
        expect(False,text='LIGHTING check frame 0 | PASS\nEXIT 0\n',report=report_for(positive,checks=1))
        expect(False,text=clean.replace('frame 3','frame 2'))
        expect(False,text='EXIT 0\n',report=report_for(positive,checks=0))
        expect(False,report={'schema_version':10,'frames':4})
        expect(False,report=report_for(positive,checks=3))
        expect(False,report={**report_for(positive),'frames':3})
        broken_mode=report_for(positive);broken_mode['lighting']['shadows']='off'
        expect(False,report=broken_mode)
        ns={**status,'returncode':1,'exit_marker':1}
        expect(True,negative,ns,bad,report_for(negative,failed=4))
        for code in (0,2,-6):
            expect(False,negative,{**ns,'returncode':code},bad,report_for(negative,failed=4))
        for problem in ('Shader Validation Error','Metal Validation Error','GPU timeout','[ERROR] Fatal: unrelated','RT check | FAIL'):
            expect(False,negative,ns,bad+problem+'\n',report_for(negative,failed=4))
        expect(False,negative,ns,clean,report_for(negative))
        result=expect(True,bias,report=report_for(bias))
        if result['state']!='QUALITY_CONTROL_PENDING' or result['quality_passed'] is not None:
            raise AssertionError('bias negative must remain pending even with good functional evidence')
        for rc, marker, success in ((0,0,True),(2,1,False),(1,1,True)):
            job=positive if rc==0 else negative
            body=clean if rc==0 else bad
            source=f'print({body.rsplit("EXIT",1)[0]!r},end=""); print("EXIT {marker}"); raise SystemExit({rc})'
            checked=run_checked([sys.executable,'-c',source],root/f'helper-{rc}.log',expected=job['expected_exit'],timeout=5)
            expect(success,job,checked,body,report_for(job,failed=0 if rc==0 else 4))
        valid_pfm=root/'valid.pfm';valid_pfm.write_bytes(b'PF\n1 1\n-1.0\n'+bytes(12))
        if pfm_record(valid_pfm)['quality_evaluated'] is not False:
            raise AssertionError('capture integrity is not quality acceptance')
        count += 1
    print(f'f10_f12_check: {count} CPU fixtures passed; no renderer/GPU invocation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=pathlib.Path,default=pathlib.Path('build/lighting/phosphor'))
    parser.add_argument('--output',type=pathlib.Path)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--only',action='append',default=[],metavar='GLOB')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--frames',type=int,default=64,help='functional microcase length; not a performance preset')
    parser.add_argument('--timeout',type=float,default=240)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return 0
    if args.frames < 1 or args.frames > 1000000 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('frames must be 1..1000000; timeout must be finite and positive')
    if not args.output and not args.list:
        parser.error('--output is required unless --list or --self-test is used')
    out=(args.output or pathlib.Path('NOT_EXECUTED')).resolve()
    planned=jobs(args.app.resolve(),out,args.frames)
    if args.only:
        for pattern in args.only:
            if not any(fnmatch.fnmatchcase(job['name'],pattern) for job in planned):
                parser.error(f'--only pattern matches no cases: {pattern}')
        planned=[job for job in planned if any(fnmatch.fnmatchcase(job['name'],pattern) for pattern in args.only)]
    if args.list:
        print('\n'.join(job['name'] for job in planned));return 0
    if out.exists():
        parser.error('output must be a new directory; previous evidence is never overwritten')
    out.mkdir(parents=True)
    manifest={'schema':2,'state':'NOT_EXECUTED','created_utc':datetime.now(timezone.utc).isoformat(),
              'validation_scope':'Functional checks only; not phase, image quality, convergence or hardware acceptance',
              'validation_env':VALIDATION_ENV,'timeout_seconds':args.timeout,'jobs':planned,
              'functional_passed':None,'phase_accepted':False,'quality_passed':None,
              'required_followup':['Independent frozen image/oracle predicate for the bias negative',
                                   'Frozen regional thresholds before independent reference',
                                   'Linear convergence sweep, per-view lifecycle and F13 denoise boundary',
                                   'Leaks, zero steady-state GPU allocations and F9/F7/F8 regressions'],
              'external_reference_argv':[sys.executable,str(pathlib.Path(__file__).with_name('f12_reference.py')),'render',
                                         str(out/'cornell-snapshot'),'--output',str(out/'offline'),
                                         '--spp','64','256','1024','4096','--signal','indirect','--material-model','diffuse']}
    path=out/'manifest.json'
    save_manifest(path,manifest)
    if not args.run:
        print(json.dumps(manifest,indent=2));return 0
    env={**os.environ,**VALIDATION_ENV}
    manifest['state']='RUNNING';save_manifest(path,manifest)
    failures=[]
    try:
        for job in planned:
            job['state']='RUNNING';job['started_utc']=datetime.now(timezone.utc).isoformat()
            archive = pathlib.Path(job['argv'][0]).parent/'shaders/phosphor-archive.metallib'
            job['additional_input_artifacts'] = [file_record(archive)] if archive.is_file() else []
            save_manifest(path,manifest) # A killed/interrupted process remains identifiable.
            log=out/(job['name']+'.log')
            expected_word='FAIL' if job['expected_exit'] else 'PASS'
            status=run_checked(job['argv'],log,expected=job['expected_exit'],
                               required=(rf'^LIGHTING check frame \d+ \| {expected_word}$',),
                               timeout=args.timeout,env=env)
            job['status_file']=str(log.with_suffix('.log.status.json'))
            job['execution']=status # source/binary/shader hashes and raw exit from shared helper
            report_path=pathlib.Path(option(job['argv'],'--report'))
            try:
                report=json.loads(report_path.read_text())
            except (OSError,ValueError) as error:
                report=None;job['report_error']=str(error)
            result=evaluate(job,status,log.read_text(errors='replace'),report)
            for item in job['additional_input_artifacts']:
                original = pathlib.Path(item['path'])
                if not original.is_file() or file_record(original)['sha256'] != item['sha256']:
                    result['functional_passed']=False;result['state']='FAIL'
                    result['failures'].append('pipeline archive changed during the run')
            job.update(result)
            job['artifacts']=[file_record(log)]
            if report_path.is_file():job['artifacts'].append(file_record(report_path))
            if result['functional_passed']:
                try:
                    job['artifacts'].extend(collect_outputs(job))
                except (OSError,ValueError,KeyError,TypeError) as error:
                    job['functional_passed']=False;job['state']='FAIL';job['failures'].append(str(error))
            job['finished_utc']=datetime.now(timezone.utc).isoformat()
            save_manifest(path,manifest)
            print(f"{job['state']}: {job['name']}",flush=True)
            if not job['functional_passed']:
                failures.append(job['name'])
                break # Never retry or continue after an unexplained failure.
        manifest['functional_passed']=not failures
        manifest['state']='FUNCTIONAL_RUN_ONLY' if not failures else 'FAILED'
    except (KeyboardInterrupt,OSError,ValueError,RuntimeError,TypeError,KeyError) as error:
        manifest['functional_passed']=False;manifest['state']='INTERRUPTED_OR_FAILED'
        manifest['runner_error']=str(error) or type(error).__name__
        if 'job' in locals() and job['state']=='RUNNING':
            job['state']='INTERRUPTED_OR_FAILED';job['functional_passed']=False
            job['failures']=[manifest['runner_error']]
        print(manifest['runner_error'],file=sys.stderr,flush=True)
    finally:
        manifest['failures']=failures
        manifest['quality_control_pending']=[job['name'] for job in planned if job.get('quality_control_pending')]
        manifest['not_executed']=[job['name'] for job in planned if job['state']=='NOT_EXECUTED']
        save_manifest(path,manifest)
    return 0 if manifest['functional_passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
