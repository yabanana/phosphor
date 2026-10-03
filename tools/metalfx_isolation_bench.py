#!/usr/bin/env python3
"""Matched direct/isolated MetalFX comparisons; one coordinated GPU workload at a time."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from run_checked import run_checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=Path('build/release'))
    p.add_argument('--out', type=Path, default=Path('build/metalfx-isolation-bench'))
    p.add_argument('--frames', type=int, default=600)
    p.add_argument('--warmup', type=int, default=120)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION', 'MTL_DEBUG_LAYER_WARNING_MODE'):
        env.pop(key, None)
    scenes = [('sponza', ['--bench', '4', '--scene', 'assets/sponza/Sponza.gltf', '--temporal-script']),
              ('lights', ['--bench', '5', '--scene', 'procedural']),
              ('two-views', ['--bench', '4', '--scene', 'assets/sponza/Sponza.gltf', '--temporal-script', '--temporal-views', '2'])]
    cases = [(name+'-'+mode, flags+['--metalfx-mode', mode]) for name, flags in scenes for mode in ('direct', 'isolated')]
    manifest = {'frames': a.frames, 'warmup': a.warmup, 'resolution': [1920, 1080], 'input_scale': .75,
                'order': [], 'conditions': {}, 'scope': 'Offscreen complete-frame throughput; GPU timing is graphics submission span including IPC gaps. Memory domains remain separate.'}
    for name, cmd in [('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm']), ('os', ['sw_vers']), ('metal', ['xcrun', 'metal', '--version'])]:
        manifest['conditions'][name] = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    reports = {name: [] for name, _ in cases}
    for replica in range(3):
        ordered = cases[replica:]+cases[:replica]
        manifest['order'].append([name for name, _ in ordered])
        (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        for name, flags in ordered:
            report = a.out/f'{name}-{replica+1}.json'
            cmd = [str(a.build/'phosphor'), '--offscreen', '--no-ui', '--no-vsync', '--fixed-timestep', '--resolution', '1920x1080',
                   '--warmup', str(a.warmup), '--frames', str(a.frames), '--post', '--upscaler', 'temporal', '--render-scale', '.75',
                   '--auto-exposure', '--material-binning', 'off', *flags, '--report', str(report)]
            status = run_checked(cmd, a.out/f'{name}-{replica+1}.log', env=env, timeout=120)
            if not status['passed']:
                raise RuntimeError(status['failures'])
            r = json.loads(report.read_text())
            assert r['gpu_allocations'] == 0, (name, 'parent GPU allocation')
            assert r['rendering']['command_buffer_rebuilds_measured'] == 0
            assert r['pipelines']['renderThreadCompiles'] == 0 and r['pipelines']['failures'] == 0
            assert r['rendering']['gpu_failures'] == 0 and r['rendering']['worker_failures'] == 0
            expected = 'metalfx-temporal-isolated' if name.endswith('isolated') else 'metalfx-temporal'
            assert r['rendering']['upscaler_effective'] == expected
            assert r['rendering']['native_frames_total'] == 0, (name, 'native fallback invalidates comparison')
            reports[name].append(r)
            print(f'{name} r{replica+1}: {r["frame_ms"]["mean"]:.4f} ms, p95 {r["frame_ms"]["p95"]:.4f}', flush=True)
    summary = {}
    for name, runs in reports.items():
        ordered = sorted(runs, key=lambda x: x['frame_ms']['mean']); median = ordered[1]
        summary[name] = {'frame_mean_median_ms': statistics.median(r['frame_ms']['mean'] for r in runs),
                         'frame_mean_range_ms': [ordered[0]['frame_ms']['mean'], ordered[-1]['frame_ms']['mean']],
                         'p95_median_run_ms': median['frame_ms']['p95'], 'p99_median_run_ms': median['frame_ms']['p99'],
                         'cpu_mean_ms': median['cpu_ms']['mean'], 'gpu_span_mean_ms': median['gpu_ms']['mean'],
                         'parent_device_bytes': median['rendering']['device_allocated_bytes_last'],
                         'worker_device_bytes': median['rendering']['worker_device_allocated_bytes_last'],
                         'shared_bridge_bytes': median['rendering']['worker_shared_bridge_bytes_last'],
                         'parent_physical_footprint': median['rendering']['parent_physical_footprint_last'],
                         'worker_physical_footprint': median['rendering']['worker_physical_footprint_last']}
    for scene, _ in scenes:
        summary[scene+'-relative'] = {'frame_delta_percent': 100*(summary[scene+'-isolated']['frame_mean_median_ms']/summary[scene+'-direct']['frame_mean_median_ms']-1)}
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
