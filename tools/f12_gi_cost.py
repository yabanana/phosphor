#!/usr/bin/env python3
"""Stage F12 cost commands and analyze reports. Never launches GPU processes."""
import argparse, datetime, hashlib, json, math, pathlib, statistics, subprocess, sys


def sha(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def git(root,*args):
    return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True).stdout


def cache_values(path):
    values={}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#','//')) and '=' in line and ':' in line.split('=',1)[0]:
            key,value=line.split('=',1);values[key.split(':',1)[0]]=value
    return values


def stage(args):
    protocol_path=pathlib.Path(args.protocol).resolve();protocol=json.loads(protocol_path.read_text());root=pathlib.Path(args.project).resolve();app=pathlib.Path(args.app).resolve();sponza=pathlib.Path(args.sponza).resolve();out=pathlib.Path(args.output).resolve()
    for path in [app,sponza,root/'tools/run_checked.py',app.parent/'CMakeCache.txt',app.parent/'shaders/phosphor.metallib']:
        if not path.is_file():raise ValueError(f'missing required artifact {path}')
    cache=cache_values(app.parent/'CMakeCache.txt')
    if cache.get('CMAKE_BUILD_TYPE')!='Release' or cache.get('PHOSPHOR_BUILD_APP')!='ON' or cache.get('PHOSPHOR_TRACY') not in ['OFF','FALSE','0']:raise ValueError('requires Release application without Tracy')
    # Pin the glTF and all external buffers/images, without silently permitting a fallback scene.
    scene=json.loads(sponza.read_text());assets={str(sponza):sha(sponza)}
    for item in scene.get('buffers',[])+scene.get('images',[]):
        uri=item.get('uri','')
        if not uri or uri.startswith('data:'):continue
        path=(sponza.parent/uri).resolve()
        if not path.is_file():raise ValueError(f'missing Sponza resource {path}')
        assets[str(path)]=sha(path)
    artifacts={str(app):sha(app),str(app.parent/'shaders/phosphor.metallib'):sha(app.parent/'shaders/phosphor.metallib')}
    # Archive/default-shader policy remains fixed as well as the primary binary.
    for path in sorted((app.parent/'shaders').iterdir()):
        if path.is_file() and (path.suffix in ['.metallib','.mtl4archive','.metalar'] or 'archive' in path.name.lower()):artifacts[str(path)]=sha(path)
    out.mkdir(parents=True,exist_ok=False);jobs=[]
    for replica in range(1,protocol['replicates']+1):
        for scene in protocol['scenes']:
            for member in protocol['order']:
                name=f"{scene['id']}-r{replica}-{member}";folder=out/name;folder.mkdir()
                scene_args=[str(sponza) if value=='{sponza}' else value for value in scene['args']]
                command=[str(app),*protocol['common_args'],*scene_args,*(protocol['B_args'] if member=='B' else protocol['A_args']),'--report',str(folder/'report.json')]
                checked=['env']
                for variable in protocol['environment_unset']:checked+=['-u',variable]
                checked += [str(pathlib.Path(args.python).resolve()),str(root/'tools/run_checked.py'),'--log',str(folder/'run.log'),'--timeout','300','--',*command]
                jobs.append({'id':name,'scene':scene['id'],'replica':replica,'member':member,'folder':str(folder),'command':command,'checked_command':checked})
    manifest={'schema':1,'state':'STAGED_NOT_EXECUTED','staged_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':protocol,'protocol_sha256':sha(protocol_path),'working_directory':str(root),'commit':git(root,'rev-parse','HEAD').decode().strip(),'tracked_patch_sha256':hashlib.sha256(git(root,'diff','--binary','HEAD')).hexdigest(),'cmake':{k:cache.get(k) for k in ['CMAKE_BUILD_TYPE','PHOSPHOR_BUILD_APP','PHOSPHOR_TRACY','CMAKE_CXX_COMPILER']},'cmake_cache_sha256':sha(app.parent/'CMakeCache.txt'),'artifacts':artifacts,'assets':assets,'jobs':jobs,'execution_policy':'Root executes checked_command sequentially from working_directory after ensuring quiet conditions. This tool never executes the application.'}
    (out/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n');print(str(out/'plan.json'));return 0


def analyze(args):
    plan_path=pathlib.Path(args.plan).resolve();plan=json.loads(plan_path.read_text());protocol=plan['protocol'];runs={};global_errors=[]
    for name,expected in {**plan['artifacts'],**plan['assets']}.items():
        if not pathlib.Path(name).is_file() or sha(name)!=expected:global_errors.append('pinned artifact/asset changed: '+name)
    previous_end=None
    for job in plan['jobs']:
        folder=pathlib.Path(job['folder']);errors=[]
        try:status=json.loads((folder/'run.log.status.json').read_text());report=json.loads((folder/'report.json').read_text())
        except (OSError,json.JSONDecodeError) as exc:runs[job['id']]={'valid':False,'errors':[str(exc)]};continue
        if not status.get('passed') or status.get('returncode')!=0 or status.get('exit_marker')!=0:errors.append('process/error gate failed')
        if status['command']!=job['command']:errors.append('command differs from frozen plan')
        started=datetime.datetime.fromisoformat(status['started_utc']).timestamp()
        duration=float(status['elapsed_seconds'])
        if not math.isfinite(duration) or duration<0:errors.append('invalid runtime duration')
        if previous_end is not None and started<previous_end-0.001:errors.append('runs overlap or ABA order changed')
        previous_end=started+duration
        if status['commit']!=plan['commit'] or status['tracked_patch_sha256']!=plan['tracked_patch_sha256']:errors.append('source provenance changed')
        for name,expected in status.get('artifacts',{}).items():
            if plan['artifacts'].get(name)!=expected:errors.append('runtime artifact mismatch: '+name)
        if status.get('binary_sha256')!=plan['artifacts'][job['command'][0]]:errors.append('binary hash mismatch')
        for artifact in [job['command'][0],str(pathlib.Path(job['command'][0]).parent/'shaders/phosphor.metallib')]:
            if status.get('artifacts',{}).get(artifact)!=plan['artifacts'][artifact]:errors.append('missing or mismatched runtime artifact: '+artifact)
        for variable in ['MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION','MTL_DEBUG_LAYER_WARNING_MODE']:
            if status.get('validation_env',{}).get(variable) not in [None,'','0']:errors.append('validation environment enabled: '+variable)
        view=report.get('rendering',{});light=report.get('lighting',{});hardware=report.get('hardware',{})
        expected_gi='ddgi' if job['member']=='B' else 'off'
        checks=[(report.get('frames')==512,'measured frame count'),(report.get('width')==1920 and report.get('height')==1080,'drawable resolution'),(view.get('input_width_last')==1920 and view.get('input_height_last')==1080,'native input resolution'),(report.get('gpu_allocations')==0,'O7 GPU allocations'),(view.get('gpu_failures')==0,'GPU failures'),(report.get('vsync') is False and report.get('ui') is False,'vsync/UI'),(report.get('gpu_timing') is True,'GPU timing'),(view.get('post') is True and view.get('upscaler_effective')=='native-spatial' and view.get('temporal_frames_total')==0 and view.get('native_frames_total')==640,'native post'),(view.get('offscreen') is False,'default windowed mode'),(not view.get('auto_exposure') and not view.get('edr'),'exposure/EDR'),(hardware.get('effective_capabilities')=='apple10' and hardware.get('physical_device')=='Apple M5 Max','hardware scope'),(light.get('gi')==expected_gi and light.get('direct')=='restir' and light.get('shadows')=='off' and light.get('reflections')=='off' and light.get('ao')=='off','lighting controls'),(light.get('denoise_effective')=='custom' and light.get('denoise_requested')=='custom','custom denoiser'),(light.get('checks')==0 and light.get('failures')==0 and not light.get('gi_visibility_disabled'),'lighting diagnostics disabled')]
        for passed,label in checks:
            if not passed:errors.append(label)
        if report.get('pipelines',{}).get('failures',0) or report.get('pipelines',{}).get('reloadFailures',0):errors.append('pipeline failure')
        for group,key in [('rt','checks'),('meshlets','checks'),('rendering','guide_checks')]:
            if report.get(group,{}).get(key,0):errors.append('diagnostic checker active: '+group)
        for field in ['gpu_ms','cpu_ms','frame_ms','wait_ms']:
            if any(not math.isfinite(report[field][key]) or report[field][key]<0 for key in ['mean','p50','p95','p99']):errors.append('nonfinite timing: '+field)
        if report['gpu_ms']['mean']<=0 or report['gpu_ms']['p50']<=0:errors.append('no positive GPU timing evidence')
        if job['scene']=='sponza' and 'sponza' not in view.get('asset','').lower():errors.append('Sponza asset fallback/mismatch')
        passes=report.get('passes',[]);coverage={'units':len(passes),'min_samples':min((p['frames'] for p in passes),default=0),'sufficient_units':sum(p['frames']>=.9*512 for p in passes),'steady_attribution_available':bool(passes and all(p['frames']>=.9*512 for p in passes))}
        runs[job['id']]={'valid':not errors,'errors':errors,'gpu_ms':report['gpu_ms'],'cpu_ms':report['cpu_ms'],'frame_ms':report['frame_ms'],'wait_ms':report['wait_ms'],'gpu_allocations':report['gpu_allocations'],'memory':{k:view.get(k) for k in ['engine_resource_bytes_last','device_allocated_bytes_last','parent_physical_footprint_last']},'passes':passes,'pass_coverage':coverage,'pipelines':report.get('pipelines',{}),'runtime_status':status}
    triples={};scenes={}
    for scene in protocol['scenes']:
        scene_id=scene['id'];rows=[]
        for replica in range(1,protocol['replicates']+1):
            members=[runs[f'{scene_id}-r{replica}-{m}'] for m in protocol['order']]
            if not all(r['valid'] for r in members):rows.append({'replica':replica,'valid':False,'reason':'invalid run'});continue
            a,b,c=members;paired={};drift={}
            for key in ['mean','p50','p95','p99']:
                baseline=(a['gpu_ms'][key]+c['gpu_ms'][key])/2;paired[key]={'A_mean':baseline,'B':b['gpu_ms'][key],'delta_ms':b['gpu_ms'][key]-baseline,'delta_fraction':b['gpu_ms'][key]/baseline-1 if baseline else None}
            for key in ['p50','p95']:drift[key]=abs(c['gpu_ms'][key]-a['gpu_ms'][key])/max((a['gpu_ms'][key]+c['gpu_ms'][key])/2,1e-12)
            valid=drift['p50']<=protocol['pair_drift_limits']['gpu_p50_relative'] and drift['p95']<=protocol['pair_drift_limits']['gpu_p95_relative']
            memory={key:{'A_mean':(a['memory'][key]+c['memory'][key])/2,'B':b['memory'][key],'delta':b['memory'][key]-(a['memory'][key]+c['memory'][key])/2} for key in a['memory'] if all(r['memory'][key] is not None for r in members)}
            rows.append({'replica':replica,'valid':valid,'baseline_drift':drift,'gpu_paired':paired,'memory_paired':memory})
        triples[scene_id]=rows;complete=len(rows)==3 and all(row['valid'] for row in rows)
        summary={}
        if complete:
            for key in ['mean','p50','p95','p99']:
                values=[row['gpu_paired'][key]['delta_ms'] for row in rows];summary[key]={'deltas_ms':values,'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),'sample_sd_ms':statistics.stdev(values)}
        scenes[scene_id]={'valid_three_paired_replicates':complete,'paired_gpu_delta_summary':summary}
    valid=not global_errors and all(r['valid'] for r in runs.values()) and all(s['valid_three_paired_replicates'] for s in scenes.values())
    result={'schema':1,'state':'COST_EVIDENCE_ONLY_NOT_DEFAULT_ADOPTION','valid_measurement':valid,'global_errors':global_errors,'plan_sha256':sha(plan_path),'runs':runs,'triples':triples,'scene_summaries':scenes,'adoption':'PENDING_BUDGET_DECISION; no time or memory budget invented','scope':protocol['scope']}
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'valid_measurement':valid,'global_errors':global_errors,'scene_summaries':scenes},indent=2));return 0 if valid else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    p=commands.add_parser('stage')
    for name in ['protocol','project','app','sponza','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--python',default=sys.executable);p.set_defaults(function=stage)
    p=commands.add_parser('analyze');p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.set_defaults(function=analyze)
    args=parser.parse_args()
    try:return args.function(args)
    except (ValueError,OSError,KeyError,AssertionError) as error:print(f'F12 cost failure: {error}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
