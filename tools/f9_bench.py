#!/usr/bin/env python3
"""F9 serial GPU measurements. Run only with a quiet machine and stopped builders.

Three replicas use rotated case order. The baseline suite uses A/B/A for
each preset and replica. Queue-chain timings are not hardware busy counters.
This does not replace correctness, visual review or device certification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from run_checked import run_checked


def summary(values):
    mean = statistics.mean(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "cv_percent": 100 * statistics.pstdev(values) / mean if mean else 0}


def validate(report, rt=False):
    assert report['gpu_allocations'] == 0, 'steady-state GPU allocations'
    assert report['pipelines']['renderThreadCompiles'] == 0, 'render-thread compiler call'
    assert report['pipelines']['failures'] == 0, 'pipeline failure'
    assert report['rendering']['gpu_failures'] == 0, 'GPU failure'
    assert report['rendering']['command_buffer_rebuilds_measured'] == 0, 'command stream rebuild'
    if report.get('gpu_timing'):
        assert abs(report['gpu_pass_sum_ms']['mean'] - report['gpu_frame_span_ms']['mean']) <= .001, 'timestamp accounting'
    if rt:
        data = report['rt']
        assert data['enabled'] and data['tlas_refits'] > 0, 'no TLAS refit'
        assert data['blas_build_ms'] > 0, 'no measured GPU BLAS build'
        assert data['opaque_alpha_tests'] == 0, 'opaque geometry invoked alpha IFT'
        if data['probe'] != 'none':
            assert data['probe_rays'] > 0 and data['probe_ms']['mean'] > 0, 'no timed probe rays'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=Path('build/release'))
    p.add_argument('--baseline-app', type=Path)
    p.add_argument('--baseline-ref', help='Source revision of the supplied baseline binary')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--suite', choices=('tlas', 'probes', 'baseline'), required=True)
    p.add_argument('--frames', type=int, default=600)
    p.add_argument('--warmup', type=int, default=120)
    a = p.parse_args()
    if a.frames < 300 or a.warmup < 60:
        p.error('use at least 300 measured and 60 warmup frames')
    if a.suite == 'baseline' and (not a.baseline_app or not a.baseline_ref):
        p.error('--baseline-app and --baseline-ref are required for A/B/A')
    a.out.mkdir(parents=True, exist_ok=False)
    app = (a.build/'phosphor').resolve()
    env = os.environ.copy()
    for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION', 'MTL_DEBUG_LAYER_WARNING_MODE'):
        env.pop(key, None)
    conditions = {}
    for name, command in [('power', ['pmset', '-g', 'batt']), ('thermal', ['pmset', '-g', 'therm']),
                          ('os', ['sw_vers']), ('metal', ['xcrun', 'metal', '--version'])]:
        r = subprocess.run(command, capture_output=True, text=True)
        conditions[name] = {'returncode': r.returncode, 'output': r.stdout.strip()}
    assets = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(Path('assets/sponza').rglob('*')) if f.is_file()}
    binaries = {'candidate': str(app)}
    if a.baseline_app:
        binaries['baseline'] = str(a.baseline_app.resolve())
    hashes = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest() for key, value in binaries.items()}
    manifest = {'suite': a.suite, 'conditions': conditions, 'assets': assets, 'binaries': binaries,
                'binary_sha256': hashes, 'baseline_revision': a.baseline_ref,
                'candidate_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'frames': a.frames, 'warmup': a.warmup, 'replicas': 3,
                'resolution': [1920, 1080], 'order': [], 'scope': 'M5 development measurement',
                'timing': 'CPU complete frame plus GPU queue-chain contribution; serial GPU timing for RT suites',
                'tlas_gate_ms': .5, 'baseline_material_regression_percent': 3,
                'unmeasured': ['physical Apple9 hardware', 'energy', 'hardware DRAM traffic']}
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    common = ['--offscreen', '--no-ui', '--no-vsync', '--fixed-timestep', '--resolution', '1920x1080',
              '--frames', str(a.frames), '--warmup', str(a.warmup)]
    sponza = ['--scene', 'assets/sponza/Sponza.gltf', '--bench', '4']
    reports = {}

    def run(name, binary, flags, rt):
        report_path = (a.out/(name+'.json')).resolve()
        command = [str(binary), *common, *flags, '--report', str(report_path)]
        manifest['order'].append({'name': name, 'command': command})
        (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        result = run_checked(command, a.out/(name+'.log'), timeout=300, env=env)
        if not result['passed']:
            raise RuntimeError(f'{name}: {result["failures"]}')
        report = json.loads(report_path.read_text())
        validate(report, rt)
        reports[name] = report
        print(f'{name}: frame={report["frame_ms"]["mean"]:.4f} ms GPU={report["gpu_ms"]["mean"]:.4f} ms', flush=True)

    if a.suite == 'tlas':
        # Native no-probe is the update gate; traversal cases compare refit
        # degradation on the same dynamic scene, ray population and timeline.
        cases = [('update-native', [], None), ('update-apple9', ['--force-family', 'apple9'], None)]
        cases += [(f'interval-{interval}', [], interval) for interval in (0, 64, 256)]
        for replica in range(3):
            for name, family, interval in cases[replica:] + cases[:replica]:
                flags = ['--scene', 'procedural', '--bench', '8', '--instances', '100000', '--dynamic-cpu', '0',
                         '--rt', 'on', '--gpu-timing-serial', *family]
                if interval is not None:
                    flags += ['--rt-probe', 'primary', '--rt-tlas-rebuild-every', str(interval)]
                run(f'{name}-r{replica+1}', app, flags, True)
    elif a.suite == 'probes':
        cases = [(family, probe) for family in ('native', 'apple9') for probe in ('primary', 'shadow', 'ao', 'diffuse')]
        for replica in range(3):
            for family, probe in cases[replica:] + cases[:replica]:
                flags = [*sponza, '--rt', 'on', '--rt-probe', probe, '--gpu-timing-serial']
                if family == 'apple9':
                    flags += ['--force-family', 'apple9']
                run(f'{family}-{probe}-r{replica+1}', app, flags, True)
    else:
        cases = [('forward', [*sponza, '--render-path', 'forward']),
                 ('temporal', [*sponza, '--temporal-script', '--post', '--upscaler', 'temporal', '--render-scale', '.75',
                               '--auto-exposure', '--material-binning', 'off']),
                 ('instances', ['--scene', 'procedural', '--bench', '8', '--instances', '100000'])]
        for replica in range(3):
            for name, flags in cases[replica:] + cases[:replica]:
                for leg, binary in [('a1', a.baseline_app.resolve()), ('b', app), ('a2', a.baseline_app.resolve())]:
                    run(f'{name}-r{replica+1}-{leg}', binary, flags, False)
    results = {}
    for name, report in reports.items():
        base = name.split('-r')[0]
        if a.suite == 'baseline':
            base += '-' + name.rsplit('-', 1)[1]
        results.setdefault(base, []).append(report)
    metrics = {}
    for name, runs in results.items():
        metrics[name] = {key: summary([r[key]['mean'] for r in runs])
                         for key in ('frame_ms', 'cpu_ms', 'gpu_ms')}
        metrics[name]['frame_p99_ms'] = summary([r['frame_ms']['p99'] for r in runs])
        if a.suite != 'baseline':
            metrics[name].update({key: summary([r['rt'][key]['mean'] for r in runs])
                                  for key in ('tlas_update_ms', 'probe_ms', 'probe_ns_per_ray')})
            metrics[name]['probe_rays'] = [r['rt']['probe_rays'] for r in runs]
    if a.suite == 'baseline':
        for name, _ in cases:
            ratios = []
            for replica in range(1, 4):
                legs = [reports[f'{name}-r{replica}-{leg}']['frame_ms']['mean'] for leg in ('a1', 'b', 'a2')]
                ratios.append(100 * (legs[1] / ((legs[0] + legs[2]) / 2) - 1))
            metrics[name+'-candidate_delta_pct'] = {'replicas': ratios, 'median': statistics.median(ratios)}
    (a.out/'summary.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print('Measurements complete; correctness and phase acceptance require separate review.', flush=True)


if __name__ == '__main__':
    main()
