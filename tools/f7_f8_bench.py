#!/usr/bin/env python3
"""F7/F8 fixed-seed complete-frame comparisons. One GPU process at a time.

Three replicas rotate case order. Timings are offscreen throughput, not display
FPS; opaque MetalFX internal traffic is not a graph bandwidth measurement.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from run_checked import run_checked


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=Path('build/release'))
    p.add_argument('--out',type=Path,default=Path('build/f7-f8-bench'))
    p.add_argument('--frames',type=int,default=600)
    p.add_argument('--warmup',type=int,default=120)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION','MTL_DEBUG_LAYER_WARNING_MODE'):env.pop(key,None)
    post=['--post','--upscaler','temporal','--render-scale','.75','--auto-exposure','--material-binning','off']
    sponza=['--bench','4','--scene','assets/sponza/Sponza.gltf','--temporal-script']
    cases=[
        ('f7-forward',sponza+['--render-path','forward']),
        ('f7-generic',sponza+['--render-path','visibility','--material-binning','off']),
        ('f7-binned',sponza+['--render-path','visibility','--material-binning','on']),
        ('f8-native',sponza+['--post','--upscaler','native','--auto-exposure','--material-binning','off']),
        ('f8-temporal',sponza+post),
        ('f8-binned',sponza+post+['--material-binning','on']),
        ('f8-frustum',sponza+post+['--meshlet-cull','frustum']),
        ('f8-tile',sponza+post+['--tile-resolve','--meshlet-cull','frustum']),
        ('f8-adaptive-sponza',sponza+post+['--adaptive-shading']),
        ('f8-cornell',['--bench','6','--scene','procedural']+post),
        ('f8-adaptive-cornell',['--bench','6','--scene','procedural']+post+['--adaptive-shading']),
        ('f8-lights',['--bench','5','--scene','procedural']+post),
        ('f8-binned-lights',['--bench','5','--scene','procedural']+post+['--material-binning','on'])]
    conditions={}
    for name,command in [('power',['pmset','-g','batt']),('thermal',['pmset','-g','therm']),
                         ('os_build',['sw_vers','-buildVersion']),('metal',['xcrun','metal','--version'])]:
        r=subprocess.run(command,capture_output=True,text=True);conditions[name]=r.stdout.strip()
    manifest={'conditions':conditions,'frames':a.frames,'warmup':a.warmup,'resolution':[1920,1080],
              'order':[],'scope':'M5 offscreen throughput; no hardware DRAM/energy counter claim'}
    reports={name:[] for name,_ in cases}
    for replica in range(3):
        rotated=cases[replica:]+cases[:replica]
        manifest['order'].append([name for name,_ in rotated])
        (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name,flags in rotated:
            report=a.out/f'{name}-{replica+1}.json'
            command=[str(a.build/'phosphor'),'--frames',str(a.frames),'--warmup',str(a.warmup),'--offscreen',
                     '--fixed-timestep','--no-ui','--no-vsync','--resolution','1920x1080',*flags,'--report',str(report)]
            result=run_checked(command,a.out/f'{name}-{replica+1}.log',timeout=120,env=env)
            if not result['passed']:raise RuntimeError(f'{name}: {result["failures"]}')
            r=json.loads(report.read_text())
            assert r['gpu_allocations']==0,(name,'GPU allocation',r['gpu_allocations'])
            assert r['rendering']['command_buffer_rebuilds_measured']==0,(name,'command stream rebuilt')
            assert r['pipelines']['renderThreadCompiles']==0 and r['pipelines']['failures']==0
            assert r['rendering']['gpu_failures']==0
            reports[name].append(r)
            print(f'{name} r{replica+1}: frame {r["frame_ms"]["mean"]:.4f} ms, p95 {r["frame_ms"]["p95"]:.4f}',flush=True)
    summary={}
    for name,runs in reports.items():
        ordered=sorted(runs,key=lambda r:r['frame_ms']['mean']);median=ordered[1]
        summary[name]={'frame_mean_median_ms':statistics.median(r['frame_ms']['mean'] for r in runs),
                       'frame_mean_range_ms':[ordered[0]['frame_ms']['mean'],ordered[-1]['frame_ms']['mean']],
                       'p95_median_run_ms':median['frame_ms']['p95'],'p99_median_run_ms':median['frame_ms']['p99'],
                       'cpu_mean_ms':median['cpu_ms']['mean'],'gpu_span_mean_ms':median['gpu_ms']['mean'],
                       'device_allocated_bytes_last':median['rendering']['device_allocated_bytes_last'],
                       'graph_heap_bytes':median['graph']['heap_bytes'],'declared_graph_dram_bytes':median['graph']['dram_bytes'],
                       'shaded_pixels_last':median['rendering']['shaded_pixels_last'],
                       'reused_pixels_last':median['rendering']['reused_pixels_last']}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
