#!/usr/bin/env python3
"""F11 linear-energy gate. Runs are SERIAL; run only when the GPU is reserved.

  python tools/f11_energy_check.py --self-test
  python tools/f11_energy_check.py --build build/lighting --out build/f11-energy-v1 \
      --only point1,point8-fresh,areas-fresh,emissive-fresh

NumPy is required. SciPy is optional: --student-t stdlib uses a checked incomplete
beta CDF/inverse. Nothing is installed. --manifest-only performs no subprocess.
--only all includes clustered/fresh/spatial/temporal/full for all three MC fixtures.
--seeds 1:32 --reference-seeds 1001:1032 use inclusive, disjoint seed ranges.

The immutable manifest precedes ALL subprocesses, including the negative-control
shader compilation. No build/source artifact is modified. The negative copies
the app into an isolated directory because there is no --shader-library CLI.
Only a temporary restir_di.air is compiled; other canonical AIRs are linked as-is.
proposal *= 2 changes candidate weights only, and must preserve checker PASS while
halving the single-point energy. No live original shader or metallib is replaced.

The statistical unit is a complete independently seeded run, not a pixel/frame.
Welch Student-t intervals on run means are an approximate finite-sample diagnostic,
not a distribution-free bound. Fixed familywise alpha=.01; 2% CI precision gate.
RGBA32Float direct output is captured as Float32 PFM: NO half-format allowance.
gamma4096 is a conservative FP32 test budget, not a hardware operation guarantee.
Every measured frame is hashed; per-run means and official endpoint PFMs remain.
Intermediate owned captures are consumed online and deleted, unless requested.
Exit 0=all PASS, 1=FAIL or INCONCLUSIVE. No retries, adaptive seeds or retuned limits.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading

U32 = 2.0 ** -24
GAMMA4096 = 4096 * U32 / (1 - 4096 * U32)
ALPHA = .01
PRECISION = .02
PROPOSAL = 'const float proposal = selectedEntry.selectionPdf * sample.pdfArea;'
POISON = 'const float proposal = 2.0f * selectedEntry.selectionPdf * sample.pdfArea;'
CHECK = re.compile(r'^LIGHTING check frame (\d+) \| (PASS|FAIL)$', re.M)


def np_module():
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError('NumPy is required; select an existing Python environment with NumPy') from error
    return np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def betacf(a, b, x):
    """Lentz evaluation of the regularized-beta continued fraction."""
    small = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > small else small)
    h = d
    for m in range(1, 1001):
        for term in (m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m)),
                     -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))):
            d = 1 + term * d
            c = 1 + term / c
            d = 1.0 / (d if abs(d) > small else small)
            c = c if abs(c) > small else small
            delta = d * c
            h *= delta
        if abs(delta - 1) < 2e-14:
            return h
    raise ArithmeticError('incomplete beta failed to converge')


def beta_regularized(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a+1)/(a+b+2):
        return front * betacf(a, b, x) / a
    return 1 - front * betacf(b, a, 1-x) / b


def t_survival(x, df):
    if not math.isfinite(df) or df <= 0 or x < 0:
        raise ValueError('finite positive degrees of freedom and nonnegative t required')
    return .5 * beta_regularized(df/(df+x*x), df/2, .5)


def t_isf(p, df):
    if not 0 < p < .5:
        raise ValueError('one-sided tail probability must be in (0,.5)')
    lo, hi = 0., 1.
    while t_survival(hi, df) > p:
        hi *= 2
        if hi > 1e16:
            raise ArithmeticError('Student-t quantile unbounded')
    for _ in range(90):
        mid = (lo+hi)/2
        if t_survival(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def student_backend(choice):
    stdlib_stats_test()
    if choice != 'stdlib':
        try:
            import scipy
            from scipy.stats import t
        except ImportError:
            if choice == 'scipy':
                raise RuntimeError('--student-t scipy requested, but SciPy is not installed')
        else:
            for df in (1., 2., 7.5, 31., 62., 1000.):
                for p in (.025, .005, 1e-5, 1e-7):
                    actual, expected = float(t.isf(p, df)), t_isf(p, df)
                    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
                        raise RuntimeError('SciPy/stdlib Student-t verification disagreement')
            return lambda p, df: float(t.isf(p, df)), 'scipy-'+scipy.__version__+' cross-checked'
    return t_isf, 'stdlib regularized-beta CDF + bisection, analytic self-tests'


def stdlib_stats_test():
    for x in (0., .1, 1., 5., 100.):
        assert math.isclose(t_survival(x, 1), .5-math.atan(x)/math.pi, abs_tol=2e-14)
        assert math.isclose(t_survival(x, 2), .5*(1-x/math.sqrt(x*x+2)), abs_tol=2e-14)
    for df, expected in ((1, 12.7062047361747), (2, 4.30265272974946),
                         (30, 2.04227245630124), (60, 2.00029782201426)):
        assert math.isclose(t_isf(.025, df), expected, rel_tol=2e-10)


def read_pfm(path):
    np = np_module()
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'PF':
            raise ValueError(f'{path}: RGB PFM required')
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        data = stream.read()
    if width < 1 or height < 1 or not math.isfinite(scale) or scale == 0 or len(data) != width*height*12:
        raise ValueError(f'{path}: invalid PFM extent/scale/payload')
    image = np.frombuffer(data, dtype='<f4' if scale < 0 else '>f4').reshape(height, width, 3)[::-1]
    image = image.astype(np.float64) * abs(scale)
    if not np.isfinite(image).all() or (image < 0).any():
        raise ValueError(f'{path}: nonfinite or negative linear radiance')
    return image


def write_pfm(path, image):
    np = np_module()
    with Path(path).open('wb') as stream:
        stream.write(f'PF\n{image.shape[1]} {image.shape[0]}\n-1.0\n'.encode())
        stream.write(np.asarray(image[::-1], dtype='<f4').tobytes())


def rois(width, height):
    return [{'name': 'whole', 'rect': [0, 0, width, height]}] + [
        {'name': f'grid-{y}-{x}', 'rect': [x*width//4, y*height//4, (x+1)*width//4, (y+1)*height//4]}
        for y in range(4) for x in range(4)]


def roi_means(image, regions):
    np = np_module()
    return np.asarray([image[y0:y1, x0:x1].mean(axis=(0, 1))
                       for r in regions for x0, y0, x1, y1 in [r['rect']]])


def exact_metrics(image, reference, factor=1.):
    np = np_module()
    expected = reference * factor
    difference = np.abs(image-expected)
    budget = GAMMA4096 * np.abs(expected)
    positive = np.abs(expected) > 0
    relative = np.divide(difference, np.abs(expected), out=np.zeros_like(difference), where=positive)
    return {'violations': int((difference > budget).sum()),
            'max_relative_nonzero': float(relative.max()),
            'nonzero_on_black': int(((~positive) & (difference > 0)).sum()),
            'max_absolute': float(difference.max())}


class CaptureConsumer:
    """Consume only atomically published files in a newly created owned directory."""
    def __init__(self, folder, manifest, official, reference=None, factor=1.):
        np = np_module()
        self.folder, self.manifest, self.official = Path(folder), manifest, official
        self.reference, self.factor = reference, factor
        self.start, self.end = manifest['warmup'], manifest['warmup'] + manifest['frames']
        self.mean = np.zeros((manifest['height'], manifest['width'], 3), dtype=np.float64)
        self.count = 0
        self.hashes, self.frames = {}, []
        self.first_half = np.zeros_like(self.mean)
        self.second_half = np.zeros_like(self.mean)
        self.exact = {'violations': 0, 'max_relative_nonzero': 0., 'nonzero_on_black': 0, 'max_absolute': 0.}
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, name='CPU-PFM-consumer', daemon=True)

    def drain(self):
        for path in sorted(self.folder.glob('frame-*.pfm')):
            match = re.fullmatch(r'frame-(\d+)\.pfm', path.name)
            if not match:
                raise RuntimeError('unexpected capture name: '+str(path))
            frame = int(match[1])
            if frame in self.hashes:
                continue  # An official frame intentionally remains on disk.
            image = read_pfm(path)
            if image.shape != self.mean.shape or not 0 <= frame < self.end:
                raise RuntimeError('unexpected capture dimensions/frame: '+str(path))
            self.hashes[frame] = sha(path)
            measured = self.start <= frame < self.end
            keep = self.manifest['keep_all_pfm'] or (self.official and frame in (self.start, self.end-1))
            record = {'frame': frame, 'sha256': self.hashes[frame], 'measured': measured, 'retained': keep}
            if measured:
                self.count += 1
                self.mean += (image-self.mean)/self.count
                if frame < self.start + self.manifest['frames']//2:
                    self.first_half += image
                else:
                    self.second_half += image
                record['roi_rgb'] = roi_means(image, self.manifest['regions']).tolist()
                if self.reference is not None:
                    values = exact_metrics(image, self.reference, self.factor)
                    for key in ('violations', 'nonzero_on_black'):
                        self.exact[key] += values[key]
                    for key in ('max_relative_nonzero', 'max_absolute'):
                        self.exact[key] = max(self.exact[key], values[key])
            self.frames.append(record)
            if not keep:
                path.unlink()

    def watch(self):
        try:
            while not self.stop.wait(.04):
                self.drain()
        except Exception as error:
            self.error = error

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('capture CPU consumer did not stop')
        try:
            if self.error:
                raise self.error
            self.drain()
        finally:
            # A failed process may have lost later frames, but earlier files were
            # already consumed. Their checksums/statistics must survive the FAIL.
            with (self.folder.parent/'frames.jsonl').open('w') as stream:
                for frame in sorted(self.frames, key=lambda item: item['frame']):
                    stream.write(json.dumps(frame, allow_nan=False)+'\n')
        if set(self.hashes) != set(range(self.end)) or self.count != self.manifest['frames']:
            missing = sorted(set(range(self.end)) - set(self.hashes))
            raise RuntimeError(f'incomplete capture sequence, missing frames {missing[:32]}')
        if list(self.folder.glob('*.partial')):
            raise RuntimeError('unpublished PFM remains after process completion')
        write_pfm(self.folder.parent/'mean.pfm', self.mean)
        np_module().save(self.folder.parent/'mean-f64.npy', self.mean, allow_pickle=False)
        pairs = [(f, f+16) for f in range(self.start, self.end-16)]
        n0 = self.manifest['frames']//2
        return {'frames': self.count, 'roi_rgb': roi_means(self.mean, self.manifest['regions']).tolist(),
                'mean_pfm_sha256': sha(self.folder.parent/'mean.pfm'), 'exact_each_frame': self.exact,
                'period16': {'pairs': len(pairs), 'identical': sum(self.hashes[a] == self.hashes[b] for a, b in pairs)},
                'first_half_rgb': roi_means(self.first_half/n0, self.manifest['regions']).tolist(),
                'second_half_rgb': roi_means(self.second_half/(self.manifest['frames']-n0), self.manifest['regions']).tolist()}


def seed_list(text):
    result = []
    for part in text.split(','):
        limits = [int(value) for value in part.split(':')]
        if len(limits) == 1:
            result += limits
        elif len(limits) == 2 and 0 <= limits[0] <= limits[1] <= 0xffffffff:
            result.extend(range(limits[0], limits[1]+1))
        else:
            raise ValueError('seeds require N or inclusive FIRST:LAST')
    if not result or len(result) != len(set(result)) or any(not 0 <= s <= 0xffffffff for s in result):
        raise ValueError('seeds must be unique u32 values')
    return result


def cases(selected):
    result = OrderedDict()
    catalog = OrderedDict()
    for fixture in ('point1', 'point8', 'areas', 'emissive'):
        modes = ('clustered', 'fresh', 'pdf2') if fixture == 'point1' else ('clustered', 'fresh', 'spatial', 'temporal', 'full')
        for mode in modes:
            name = fixture+'-'+mode
            catalog[name] = {'name': name, 'fixture': fixture, 'mode': mode,
                             'exact': fixture == 'point1' or (fixture == 'point8' and mode == 'clustered'),
                             'factor': .5 if mode == 'pdf2' else 1.}
    for term in selected.split(','):
        matches = [key for key in catalog if term == 'all' or key == term or key.startswith(term+'-')]
        if not matches:
            raise ValueError('unknown --only selector '+term+'; use all, fixture or fixture-mode')
        for name in matches:
            result[name] = catalog[name]
    return list(result.values())


def fixture_args(fixture):
    if fixture == 'emissive':
        return ['--bench', '6', '--lighting-scene', 'cornell']
    return ['--bench', '5', '--local-light-count', {'point1': '1', 'point8': '8', 'areas': '0'}[fixture],
            '--stationary-lights'] + (['--area-lights'] if fixture == 'areas' else [])


def command_for(binary, fixture, mode, seed, directory, m, reference_frame=None):
    lighting = mode if mode in ('brute', 'clustered') else 'restir'
    spatial = 4 if mode in ('spatial', 'full') else 0
    command = [str(binary), '--render-path', 'visibility', '--rt', 'on', '--shadows', 'off', '--gi', 'off',
               '--shadow-map-size', '128', '--meshlet-cull', 'off', '--fixed-timestep', '--no-ui', '--offscreen',
               '--no-vsync', '--resolution', f"{m['width']}x{m['height']}", '--warmup', str(m['warmup']),
               '--frames', str(m['frames']), '--no-pipeline-archive', '--debug-lighting', '1',
               '--lighting', lighting, '--lighting-seed', str(seed), '--lighting-preset', 'full',
               '--lighting-candidates', '1' if fixture == 'point1' else '8',
               '--lighting-spatial-samples', str(spatial), '--capture-linear-signal', 'direct',
               '--capture-linear-sequence', str(directory/'captures'), '--capture-every', '1',
               '--export-reference', str(directory/'snapshot'), '--export-reference-frame',
               str(m['reference_frame'] if reference_frame is None else reference_frame), '--report', str(directory/'report.json')]
    if mode not in ('temporal', 'full'):
        command += ['--history-reset-every', '1']
    return command + fixture_args(fixture)


def snapshot_digest(folder, frame, m):
    folder = Path(folder)
    data = json.loads((folder/'scene.json').read_text())
    if data.get('schema') != 1 or data.get('linear') is not True or data.get('frame') != frame:
        raise RuntimeError('missing exact same-frame linear snapshot')
    camera = data.get('camera', {})
    if camera.get('jitter_pixels') != [0, 0] or (camera.get('width'), camera.get('height')) != (m['width'], m['height']):
        raise RuntimeError('receiver projection differs: jitter or extent changed')
    # Frame is the only normalized field: the extra stability witness intentionally
    # captures an earlier frame; every other JSON value and resource byte must match.
    del data['frame']
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
    files = {}
    for item in sorted(folder.iterdir()):
        if item.name != 'scene.json':
            if not item.is_file() or item.is_symlink():
                raise RuntimeError('unexpected snapshot resource')
            files[item.name] = sha(item)
            digest.update(item.name.encode()+files[item.name].encode())
    return digest.hexdigest(), files


def assert_inputs(m):
    for path, expected in m['immutable_inputs'].items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('frozen input changed: '+path)


def welch(candidate, reference, family, quantile):
    np = np_module()
    a, b = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError('Welch requires independent seed samples, at least two per estimator')
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    va, vb = a.var(axis=0, ddof=1)/len(a), b.var(axis=0, ddof=1)/len(b)
    se2 = va+vb
    tail = ALPHA/(2*family)
    rows = []
    for index in np.ndindex(mu_a.shape):
        error = float(mu_a[index]-mu_b[index])
        se = math.sqrt(float(se2[index]))
        df = None
        half = 0.
        if se > 0:
            df = float(se2[index]**2/(va[index]**2/(len(a)-1)+vb[index]**2/(len(b)-1)))
            half = quantile(tail, df)*se
        numeric = GAMMA4096 * abs(float(mu_b[index]))
        precision = PRECISION * abs(float(mu_b[index]))
        state = 'FAIL' if abs(error) > half+numeric else ('INCONCLUSIVE' if half > precision else 'PASS')
        rows.append({'roi': index[0], 'channel': 'RGB'[index[1]], 'candidate': float(mu_a[index]),
                     'reference': float(mu_b[index]), 'difference': error, 'standard_error': se,
                     'welch_df': df, 'ci_half_width': half, 'fp32_budget': numeric,
                     'precision_limit': precision, 'status': state})
    return rows


def freeze(args, quantile_name, checked_path, helper_path):
    repo, build, out = args.repo.resolve(), args.build.resolve(), args.out.resolve()
    if out.exists():
        raise ValueError('output directory exists; choose a fresh path (no overwrites/resume)')
    width, height = map(int, args.resolution.split('x'))
    if width < 4 or height < 4 or args.frames < 2 or args.warmup < 0:
        raise ValueError('extent >=4x4, frames>=2 and warmup>=0 required')
    selected = cases(args.only)
    seeds, ref_seeds = seed_list(args.seeds), seed_list(args.reference_seeds)
    if set(seeds) & set(ref_seeds):
        raise ValueError('candidate and reference seed sets must be disjoint')
    has_mc = any(not c['exact'] for c in selected)
    if has_mc and (len(seeds) < 32 or len(ref_seeds) < 32):
        raise ValueError('official MC protocol requires >=32 candidate and >=32 disjoint reference seeds')
    binary = build/'phosphor'
    cache_path, app_cmake_path = build/'CMakeCache.txt', repo/'cmake/App.cmake'
    helper = load_module('f11_compile_helpers', helper_path)
    cache = helper.read_cache(cache_path)
    source_home = Path(cache.get('CMAKE_HOME_DIRECTORY', '')).resolve()
    if source_home != repo:
        raise ValueError('CMake build belongs to another source directory')
    cmake = app_cmake_path.read_text()
    modules = helper.shader_modules(cmake, repo)
    flags = helper.metal_flags(cmake, cache, repo, build)
    if cache.get('PHOSPHOR_SHADER_DEBUG_INFO', 'ON') == 'ON' and cache.get('CMAKE_BUILD_TYPE') in ('Debug', 'RelWithDebInfo'):
        flags += ['-gline-tables-only', '-frecord-sources']
    inputs = {binary, build/'shaders/phosphor.metallib', cache_path, app_cmake_path,
              Path(__file__).resolve(), checked_path, helper_path}
    inputs.update(build/'shaders'/name for name in modules)
    inputs.update(p for p in (repo/'shaders').rglob('*') if p.is_file())
    inputs.update(p for p in (repo/'src').rglob('*.h'))
    inputs.update(p for p in (build/'generated').rglob('*') if p.is_file())
    inputs.update((repo/'src/testbench/many_lights.cpp', repo/'src/testbench/lighting_validation.cpp',
                   repo/'src/app/engine.cpp', repo/'src/platform/metal/direct_lighting_passes.cpp'))
    if (build/'shaders/graph-plans.json').is_file():
        inputs.add(build/'shaders/graph-plans.json')
    m = {'schema': 1, 'state': 'FROZEN_BEFORE_PROCESSES', 'created_utc': datetime.now(timezone.utc).isoformat(),
         'repo': str(repo), 'build': str(build), 'out': str(out), 'width': width, 'height': height,
         'frames': args.frames, 'warmup': args.warmup, 'reference_frame': args.frames+args.warmup-1,
         'candidate_seeds': seeds, 'reference_seeds': ref_seeds, 'cases': selected,
         'regions': rois(width, height), 'signal': 'direct RGBA32Float -> RGB Float32 linear PFM',
         'statistical_unit': 'one entire seed-run mean; never pixels or correlated frames',
         'family_comparisons': 51*sum(not c['exact'] for c in selected), 'familywise_alpha': ALPHA,
         'student_t': quantile_name, 'precision_relative': PRECISION, 'fp32_unit_roundoff': U32,
         'gamma_operations': 4096, 'gamma4096': GAMMA4096, 'keep_all_pfm': args.keep_all_pfm,
         'retention': 'all run means PFM+Float64 NPY; all frame hashes/ROI; first+last PFM of first seed per case; canonical full fixture snapshot',
         'expected_visibility': 'no local visibility with --shadows off; energy/proposal gate only',
         'source_stability': 'stationary bench5 or static Cornell, no post/jitter/DRS/camera animation; early+late snapshot witness',
         'immutable_inputs': {str(path): sha(path) for path in sorted(inputs)}, 'runs': [], 'negative_compile': []}
    fixtures = list(dict.fromkeys(c['fixture'] for c in selected))
    for fixture in fixtures:
        fixture_refs = ref_seeds[:1] if fixture == 'point1' else ref_seeds
        # The early witness runs the same full sequence but exports at warmup.
        # It is not counted as another statistically independent reference sample.
        configurations = [('stability', ref_seeds[0], args.warmup)] + [('reference', s, None) for s in fixture_refs]
        for kind, seed, frame in configurations:
            name = f'{fixture}-{kind}-{seed}'
            directory = out/name
            m['runs'].append({'name': name, 'fixture': fixture, 'kind': kind, 'seed': seed,
                              'mode': 'brute', 'directory': str(directory), 'reference_frame': m['reference_frame'] if frame is None else frame,
                              'official': seed == ref_seeds[0],
                              'command': command_for(binary, fixture, 'brute', seed, directory, m, frame)})
    negative_binary = out/'pdf2-app/phosphor'
    for case in selected:
        for seed in (seeds[:1] if case['exact'] else seeds):
            name = case['name']+'-'+str(seed)
            directory = out/name
            m['runs'].append({'name': name, 'fixture': case['fixture'], 'kind': 'candidate', 'seed': seed,
                              'case': case['name'], 'mode': case['mode'], 'directory': str(directory),
                              'official': seed == seeds[0], 'reference_frame': m['reference_frame'],
                              'command': command_for(negative_binary if case['mode'] == 'pdf2' else binary,
                                                     case['fixture'], case['mode'], seed, directory, m)})
    if any(c['mode'] == 'pdf2' for c in selected):
        source = (repo/'shaders/restir_di.metal').read_text()
        if source.count(PROPOSAL) != 1 or source.find(PROPOSAL) > source.find('kernel void restir_di_temporal'):
            raise ValueError('candidate proposal anchor changed; refusing an ambiguous shader mutation')
        poisoned = source.replace(PROPOSAL, POISON, 1)
        m['negative_source_sha256'] = hashlib.sha256(poisoned.encode()).hexdigest()
        air = out/'pdf2-build/restir_di.air'
        m['negative_compile'] = [
            ['xcrun', '-sdk', 'macosx', 'metal', *flags, '-c', str(out/'pdf2-build/shaders/restir_di.metal'), '-o', str(air)],
            ['xcrun', '-sdk', 'macosx', 'metallib', *[str(air if name == 'restir_di.air' else build/'shaders'/name) for name in modules],
             '-o', str(out/'pdf2-app/shaders/phosphor.metallib')]]
    out.mkdir(parents=True)
    json_write(out/'manifest.json', m)
    (out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n')
    return m


def compile_negative(m, run_checked, timeout):
    if not m['negative_compile']:
        return
    out, repo, build = Path(m['out']), Path(m['repo']), Path(m['build'])
    (out/'pdf2-build').mkdir()
    (out/'pdf2-app/shaders').mkdir(parents=True)
    shutil.copytree(repo/'shaders', out/'pdf2-build/shaders')
    path = out/'pdf2-build/shaders/restir_di.metal'
    path.write_text(path.read_text().replace(PROPOSAL, POISON, 1))
    if sha(path) != m['negative_source_sha256']:
        raise RuntimeError('poison source differs from frozen manifest')
    shutil.copy2(build/'phosphor', out/'pdf2-app/phosphor')
    graph = build/'shaders/graph-plans.json'
    if graph.is_file():
        shutil.copy2(graph, out/'pdf2-app/shaders/graph-plans.json')
    for i, command in enumerate(m['negative_compile']):
        assert_inputs(m)
        status = run_checked(command, out/f'pdf2-compile-{i}.log', marker=False, timeout=timeout)
        if not status['passed']:
            raise RuntimeError('negative shader compile/link failed: '+str(status['failures']))
    assert_inputs(m)
    json_write(out/'pdf2-artifacts.json', {str(p): sha(p) for p in
        (path, out/'pdf2-build/restir_di.air', out/'pdf2-app/phosphor', out/'pdf2-app/shaders/phosphor.metallib')})


def run_all(m, checked, quantile, timeout):
    np = np_module()
    out = Path(m['out'])
    manifest_hash = sha(out/'manifest.json')
    assert_inputs(m)
    compile_negative(m, checked, timeout)
    results, fixture_hashes, reference_images = {}, {}, {}
    for run in m['runs']:
        if sha(out/'manifest.json') != manifest_hash:
            raise RuntimeError('manifest changed after freeze')
        assert_inputs(m)
        directory = Path(run['directory'])
        directory.mkdir()
        (directory/'captures').mkdir()
        case = next((c for c in m['cases'] if c['name'] == run.get('case')), None)
        reference = reference_images.get(run['fixture']) if case and case['exact'] else None
        consumer = CaptureConsumer(directory/'captures', m, run['official'], reference, case['factor'] if case else 1.)
        consumer.thread.start()
        try:
            status = checked(run['command'], directory/'run.log', timeout=timeout,
                             required=(r'^LIGHTING check frame \d+ \| PASS$',))
        finally:
            consumer.stop.set()
            consumer.thread.join(timeout=30)
        errors = list(status['failures'])
        try:
            metrics = consumer.finish()
        except Exception as error:
            metrics = {}
            errors.append(str(error))
        report_path = directory/'report.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        lighting = report.get('lighting', {})
        lines = CHECK.findall((directory/'run.log').read_text(errors='replace'))
        expected_frames = set(range(m['warmup']+m['frames']))
        if {int(frame) for frame, value in lines if value == 'PASS'} != expected_frames or any(value != 'PASS' for _, value in lines):
            errors.append('missing per-frame lighting checker PASS, including shutdown-drained slots')
        if lighting.get('failures') != 0 or lighting.get('checks') != len(expected_frames):
            errors.append('lighting report does not confirm all-frame checker PASS')
        expected_mode = run['mode'] if run['mode'] in ('brute', 'clustered') else 'restir'
        if lighting.get('direct') != expected_mode or lighting.get('seed') != run['seed'] or lighting.get('gi') != 'off':
            errors.append('report mode/seed does not match frozen run')
        try:
            digest, files = snapshot_digest(directory/'snapshot', run['reference_frame'], m)
            if run['fixture'] not in fixture_hashes:
                fixture_hashes[run['fixture']] = digest
            elif digest != fixture_hashes[run['fixture']]:
                errors.append('receiver/light/material/geometry snapshot differs from static fixture witness')
            json_write(directory/'snapshot-checksums.json', {'same_frame': run['reference_frame'], 'normalized_digest': digest, 'resources': files})
            # Keep the canonical witness's full resource set. Other same-byte
            # snapshots retain scene.json + hashes, avoiding duplicate texture/PLY GB.
            if run['kind'] != 'stability' and digest == fixture_hashes[run['fixture']]:
                for name in files:
                    (directory/'snapshot'/name).unlink()
        except Exception as error:
            errors.append('snapshot: '+str(error))
        if run['kind'] == 'reference' and metrics:
            existing = reference_images.get(run['fixture'])
            if run['fixture'] in ('point1', 'point8') and existing is not None and not np.array_equal(existing, consumer.mean):
                errors.append('punctual brute reference varied across independent seeds')
            reference_images.setdefault(run['fixture'], consumer.mean.copy())
        if case and case['exact'] and metrics and metrics['exact_each_frame']['violations']:
            errors.append('per-pixel energy violates frozen FP32 budget on one or more measured frames')
        if case and case['factor'] != 1. and metrics:
            metrics['unity_energy_gate_rejected'] = exact_metrics(consumer.mean, reference)['violations'] > 0
            if not metrics['unity_energy_gate_rejected']:
                errors.append('negative control was not rejected by the unmodified energy gate')
        if not errors and metrics and np.sum(consumer.mean) <= 0:
            errors.append('black fixture cannot certify energy')
        result = {'run': run['name'], 'fixture': run['fixture'], 'kind': run['kind'], 'case': run.get('case'),
                  'passed': not errors, 'failures': errors, 'metrics': metrics}
        json_write(directory/'aggregate.json', result)
        results[run['name']] = result
        print(('PASS' if not errors else 'FAIL')+': '+run['name'], flush=True)
        assert_inputs(m)
        if errors:
            json_write(out/'summary.json', {'status': 'FAIL', 'failed_run': result, 'completed_runs': list(results)})
            raise RuntimeError('run failed; evidence retained in '+str(directory))
    comparisons = []
    for case in m['cases']:
        a = [r['metrics']['roi_rgb'] for r in results.values() if r['case'] == case['name']]
        b = [r['metrics']['roi_rgb'] for r in results.values() if r['kind'] == 'reference' and r['fixture'] == case['fixture']]
        if case['exact']:
            status, rows = 'PASS', []  # Every measured pixel/frame already checked above.
        else:
            rows = welch(a, b, m['family_comparisons'], quantile)
            status = 'FAIL' if any(r['status'] == 'FAIL' for r in rows) else ('INCONCLUSIVE' if any(r['status'] == 'INCONCLUSIVE' for r in rows) else 'PASS')
        energy = float(np.asarray(a).mean(axis=0)[0].sum()/np.asarray(b).mean(axis=0)[0].sum())
        comparisons.append({'case': case['name'], 'status': status, 'candidate_runs': len(a), 'reference_runs': len(b),
                            'energy_ratio': energy, 'expected_ratio': case['factor'], 'roi_tests': rows})
        print(f"{status}: {case['name']} energy/reference={energy:.9g}", flush=True)
    status = 'FAIL' if any(c['status'] == 'FAIL' for c in comparisons) else ('INCONCLUSIVE' if any(c['status'] == 'INCONCLUSIVE' for c in comparisons) else 'PASS')
    summary = {'status': status, 'manifest_sha256': manifest_hash, 'family_comparisons': m['family_comparisons'],
               'comparisons': comparisons, 'completed_runs': len(results),
               'scope': 'static direct energy only; does not certify moving scenes or local visibility'}
    json_write(out/'summary.json', summary)
    return status == 'PASS'


def self_test():
    np = np_module()
    stdlib_stats_test()
    quantile, backend = student_backend('auto')
    assert cases('point1')[2]['factor'] == .5 and len(cases('all')) == 18
    assert seed_list('1:3,9') == [1, 2, 3, 9]
    assert 51*sum(not c['exact'] for c in cases('point1,point8-fresh,areas-fresh,emissive-fresh')) == 153
    assert GAMMA4096 == 1/4095
    constant = np.full((32, 17, 3), 2.)
    assert all(row['status'] == 'PASS' for row in welch(constant, constant, 51, quantile))
    assert all(row['status'] == 'FAIL' for row in welch(constant*.5, constant, 51, quantile))
    symmetric = np.tile(np.array([-1., 1.])[:, None, None], (16, 17, 3))
    assert all(row['status'] == 'INCONCLUSIVE' for row in welch(constant+symmetric, constant-symmetric, 51, quantile))
    assert all(row['status'] == 'PASS' for row in welch(constant+symmetric*.001, constant-symmetric*.001, 51, quantile))
    assert exact_metrics(np.ones((4, 4, 3))*.5, np.ones((4, 4, 3)), .5)['violations'] == 0
    assert exact_metrics(np.ones((4, 4, 3))*.5, np.ones((4, 4, 3)))['violations'] == 48
    assert exact_metrics(np.ones((4, 4, 3)), np.zeros((4, 4, 3)))['nonzero_on_black'] == 48
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)/'captures'
        folder.mkdir()
        m = {'width': 8, 'height': 4, 'warmup': 2, 'frames': 4, 'regions': rois(8, 4), 'keep_all_pfm': False}
        for frame in range(6):
            write_pfm(folder/f'frame-{frame:06d}.pfm', np.full((4, 8, 3), frame))
        consumer = CaptureConsumer(folder, m, True)
        consumer.thread.start()
        metrics = consumer.finish()
        assert np.array_equal(consumer.mean, np.full((4, 8, 3), 3.5))
        assert metrics['frames'] == 4 and metrics['roi_rgb'][0] == [3.5]*3
        assert sorted(p.name for p in folder.glob('*.pfm')) == ['frame-000002.pfm', 'frame-000005.pfm']
        assert np.array_equal(read_pfm(folder.parent/'mean.pfm'), consumer.mean)
        # Atomic publication is respected: no *.partial is ever consumed.
        (folder/'frame-000006.pfm.partial').write_text('unfinished')
        try:
            consumer.finish()
            raise AssertionError('unfinished capture accepted')
        except RuntimeError as error:
            assert 'unpublished' in str(error)
    # Exercise orchestration with in-process fake outputs. This callable NEVER
    # launches a renderer/compiler or any other process, including in CI.
    for corruption in (None, 'energy', 'snapshot', 'checker'):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            m = {'out': str(out), 'width': 8, 'height': 4, 'warmup': 2, 'frames': 4,
                 'reference_frame': 5, 'regions': rois(8, 4), 'keep_all_pfm': False,
                 'immutable_inputs': {}, 'negative_compile': [], 'family_comparisons': 0,
                 'cases': cases('point1-fresh,point1-pdf2'), 'runs': []}
            specs = [('witness', 'stability', 'brute', 1001, 2, None),
                     ('reference', 'reference', 'brute', 1001, 5, None),
                     ('fresh', 'candidate', 'fresh', 1, 5, 'point1-fresh'),
                     ('negative', 'candidate', 'pdf2', 1, 5, 'point1-pdf2')]
            for name, kind, mode, seed, frame, case in specs:
                m['runs'].append({'name': name, 'kind': kind, 'mode': mode, 'seed': seed,
                                  'fixture': 'point1', 'case': case, 'official': True,
                                  'reference_frame': frame, 'directory': str(out/name), 'command': [name]})
            json_write(out/'manifest.json', m)
            def fake_checked(command, log, **kwargs):
                run = next(r for r in m['runs'] if r['name'] == command[0])
                directory = Path(run['directory'])
                radiance = 1. if run['mode'] == 'pdf2' else 2.
                if corruption == 'energy' and run['mode'] == 'fresh':
                    radiance *= .5
                for frame in range(6):
                    write_pfm(directory/'captures'/f'frame-{frame:06d}.pfm', np.full((4, 8, 3), radiance))
                camera = {'width': 8, 'height': 4, 'jitter_pixels': [0, 0], 'position': [0, 0, 0]}
                if corruption == 'snapshot' and run['kind'] == 'candidate':
                    camera['position'][0] = 1
                (directory/'snapshot').mkdir()
                json_write(directory/'snapshot/scene.json', {'schema': 1, 'linear': True, 'frame': run['reference_frame'], 'camera': camera})
                direct = run['mode'] if run['mode'] == 'brute' else 'restir'
                json_write(directory/'report.json', {'lighting': {'checks': 6, 'failures': 0, 'direct': direct, 'seed': run['seed'], 'gi': 'off'}})
                count = 5 if corruption == 'checker' and run['kind'] == 'candidate' else 6
                Path(log).write_text(''.join(f'LIGHTING check frame {frame} | PASS\n' for frame in range(count))+'EXIT 0\n')
                return {'passed': True, 'failures': []}
            try:
                passed = run_all(m, fake_checked, quantile, 1)
                assert corruption is None and passed
                assert json.loads((out/'negative/aggregate.json').read_text())['metrics']['unity_energy_gate_rejected']
            except RuntimeError:
                assert corruption is not None
                assert json.loads((out/'summary.json').read_text())['status'] == 'FAIL'
    print('PASS: Student-t analytic cases, exact/PDF2/black corruption, Welch FAIL/PASS/precision, PFM streaming, fake orchestration/receiver/checker negatives; '+backend)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--build', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--only', default='point1,point8-fresh,areas-fresh,emissive-fresh')
    parser.add_argument('--seeds', default='1:32')
    parser.add_argument('--reference-seeds', default='1001:1032')
    parser.add_argument('--resolution', default='128x72')
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--student-t', choices=('auto', 'stdlib', 'scipy'), default='auto')
    parser.add_argument('--keep-all-pfm', action='store_true')
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.build is None or args.out is None:
        parser.error('--build and --out are required')
    args.repo = args.repo.resolve()
    if not args.build.is_absolute():
        args.build = args.repo/args.build
    if not args.out.is_absolute():
        args.out = args.repo/args.out
    sys.path.insert(0, str(args.repo/'tools'))
    checked_path = args.repo/'tools/run_checked.py'
    helper_path = args.repo/'tools/rt_archive_reload_check.py'
    checked = load_module('f11_checked', checked_path)
    quantile, name = student_backend(args.student_t)
    m = freeze(args, name, checked_path, helper_path)
    print(f"Frozen {len(m['runs'])} serial runs; {m['family_comparisons']} statistical comparisons: {args.out/'manifest.json'}", flush=True)
    if args.manifest_only:
        return 0
    os.chdir(args.repo)  # run_checked records the integration repo, not the runner's staging folder.
    return 0 if run_all(m, checked.run_checked, quantile, args.timeout) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('FAIL: '+str(error), file=sys.stderr)
        raise SystemExit(1)
