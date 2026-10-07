#!/usr/bin/env python3
"""F9 Release check: Compiler lookupArchives must not hide a changed alpha function.

Run explicitly on the Mac, with no other GPU workload:
  mise exec -- python3 tools/rt_archive_reload_check.py \
      --build build/release --out build/f9-validation/archive-reload

The build must already contain the app, its AOT archive, probe AIR files and
hot-reload-probe.metallib. This script never builds/changes those artifacts.
It runs the normal probe, then compiles only a temporary rt_scene.metal against
a poisoned copy of rt_common.h and links a new probe library in the new output
directory. --debug-hot-reload works in Release without the Debug-only watcher.
No --pipeline-sync: that diagnostic disables PipelineCache::reload itself.

Archive hit counts are direct Archive API results. compilerCalls counts public
Compiler API calls; lookupArchives does not expose whether its hint hit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from run_checked import run_checked


VALIDATION_ERROR = re.compile(
    r'failed assertion|\[ERROR\]|GPU timeout|command buffers failed|Shader Validation Error|'
    r'Metal Validation Error|\berror:')
CHECK = re.compile(r'^RT check frame (\d+).*?opaque-alpha (\d+).*?\| (PASS|FAIL)(?:$|:)', re.M)
RELOAD = re.compile(r'Shaders reloaded: generation')


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cache(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(('#', '//')):
            continue
        match = re.match(r'([^:=]+):[^=]+=(.*)$', line)
        if match:
            result[match[1]] = match[2]
    return result


def metal_flags(app_cmake: str, cache: dict[str, str], repo: Path, build: Path) -> list[str]:
    """Read the Release flag list actually declared by cmake/App.cmake."""
    match = re.search(r'set\(PHOSPHOR_METAL_FLAGS\s+(.*?)\)', app_cmake, re.S)
    if not match:
        raise ValueError('cannot find PHOSPHOR_METAL_FLAGS in cmake/App.cmake')
    deployment = cache.get('CMAKE_OSX_DEPLOYMENT_TARGET', '')
    if not deployment:
        raise ValueError('CMAKE_OSX_DEPLOYMENT_TARGET is missing from the build cache')
    variables = {'CMAKE_OSX_DEPLOYMENT_TARGET': deployment,
                 'CMAKE_SOURCE_DIR': str(repo), 'CMAKE_BINARY_DIR': str(build)}
    def expand(token):
        def value(match):
            name = match[1]
            if name not in variables:
                raise ValueError(f'unsupported Metal flag variable {name}; update the runner explicitly')
            return variables[name]
        return re.sub(r'\$\{([^}]+)\}', value, token)
    # Expand after tokenizing: a source/build path containing spaces remains
    # one argument, just as an unquoted CMake path variable does.
    flags = [expand(token) for token in shlex.split(match[1], comments=True)]
    if '-fpreserve-invariance' not in flags or not any(flag.startswith('-std=metal') for flag in flags):
        raise ValueError('Metal standard/invariance contract changed; inspect App.cmake before running')
    return flags


def shader_modules(app_cmake: str, repo: Path) -> list[str]:
    """Follow the shader GLOB and explicit REMOVE/PREPEND operations in App.cmake."""
    sources = sorted((repo/'shaders').glob('*.metal'))
    for operation, arguments in re.findall(r'list\((REMOVE_ITEM|PREPEND)\s+PHOSPHOR_METAL_SHADERS\s+(.*?)\)', app_cmake, re.S):
        paths = []
        for token in shlex.split(arguments, comments=True):
            expanded = token.replace('${CMAKE_SOURCE_DIR}', str(repo))
            if '${' in expanded:
                raise ValueError('unsupported shader-list variable in App.cmake: '+token)
            paths.append(Path(expanded))
        if operation == 'REMOVE_ITEM':
            sources = [path for path in sources if path not in paths]
        else:
            sources = paths + sources
    modules = [path.stem+'.air' for path in sources]
    if not modules or modules[0] != 'material_passes.air' or len(modules) != len(set(modules)):
        raise ValueError('unexpected CMake shader link order or duplicate modules')
    return modules


def ordered_probe_air(probe_dir: Path, replacement: Path, modules: list[str]) -> list[Path]:
    # Follow the CMake link list, not globbed build leftovers. In particular,
    # older forward.air/visibility_resolve.air can exist on disk even though
    # these source files are now included by the first-linked material module.
    if 'material_passes.air' not in modules or 'rt_scene.air' not in modules:
        raise ValueError('current CMake shader list lacks the material or RT module')
    missing = [name for name in modules if not (probe_dir/name).is_file()]
    if missing:
        raise ValueError('probe AIR files are incomplete; build phosphor_probe_shaders first: '+str(missing))
    return [replacement if name == 'rt_scene.air' else probe_dir/name for name in modules]


def poison_alpha(source: str) -> str:
    needle = '    ++payload.alphaTests;'
    if source.count(needle) != 1:
        raise ValueError('alpha counter anchor changed; refusing an ambiguous shader edit')
    return source.replace(needle, needle + '\n    ++payload.opaqueAlphaTests; // F9 archive-hint reload negative control', 1)


def archive_environment() -> dict[str, str]:
    env = dict(os.environ)
    # Release has no automatic API-validation define. Explicit zeros also
    # prevent an inherited validation setting from disabling the AOT archive.
    for key in list(env):
        if key.startswith('MTL_SHADER_VALIDATION') or key in {
            'MTL_DEBUG_LAYER', 'MTL_DEBUG_LAYER_WARNING_MODE', 'MTL_CAPTURE_ENABLED', 'METAL_DEVICE_WRAPPER_TYPE'}:
            env.pop(key)
    env['MTL_DEBUG_LAYER'] = '0'
    env['MTL_SHADER_VALIDATION'] = '0'
    return env


def assess(text: str, report: dict, negative: bool) -> list[str]:
    errors = []
    if VALIDATION_ERROR.search(text):
        errors.append('unexpected GPU, compiler, or validation error')
    if 'Validation Enabled' in text:
        errors.append('Metal validation was enabled; this is not the archive-active path')
    reloads = list(RELOAD.finditer(text))
    if len(reloads) != 1:
        errors.append('expected exactly one applied shader generation')
    checks = list(CHECK.finditer(text))
    if reloads:
        before = [check for check in checks if check.start() < reloads[0].start()]
        after = [check for check in checks if check.start() > reloads[0].end()]
        if not before or any(check[3] != 'PASS' for check in before):
            errors.append('missing clean RT checks before reload')
        if negative:
            slots = {int(check[1]) % 3 for check in after if check[3] == 'FAIL' and int(check[2]) > 0}
            if slots != {0, 1, 2}:
                errors.append('changed alpha function was not observed on every IFT slot')
        else:
            slots = {int(check[1]) % 3 for check in after if check[3] == 'PASS'}
            if slots != {0, 1, 2} or any(check[3] != 'PASS' for check in checks):
                errors.append('RT checks did not remain correct on all slots after reload')
    if re.search(r'^HOT-RELOAD .*\| PASS$', text, re.M) is None:
        errors.append('probe capture does not prove the new raster library became active')
    pipelines = report.get('pipelines', {})
    for field in ('archiveHits', 'compilerCalls'):
        value = pipelines.get(field)
        if type(value) is not int or value <= 0:
            errors.append(f'pipelines.{field} must be positive')
    for field, expected in (('reloads', 1), ('reloadFailures', 0), ('failures', 0), ('renderThreadCompiles', 0)):
        if pipelines.get(field) != expected:
            errors.append(f'pipelines.{field} must equal {expected}')
    rt = report.get('rt', {})
    if rt.get('enabled') is not True or rt.get('checks', 0) <= 0 or rt.get('alpha_tests', 0) <= 0:
        errors.append('missing enabled, checked RT work with real MASK intersections')
    if negative:
        if rt.get('check_failures', 0) <= 0 or rt.get('opaque_alpha_tests', 0) <= 0:
            errors.append('report did not confirm the intentional alpha counter violation')
    elif rt.get('check_failures') != 0 or rt.get('opaque_alpha_tests') != 0:
        errors.append('normal probe produced RT checker failures')
    return errors


def assert_unchanged(files: dict[str, str]) -> None:
    for name, expected in files.items():
        path = Path(name)
        if not path.is_file() or checksum(path) != expected:
            raise RuntimeError('input/source/build artifact changed during check: ' + name)


def compile_step(command: list[str], log: Path, timeout: float, env: dict[str, str]) -> dict:
    started = time.monotonic()
    timed_out = False
    with log.open('wb') as output:
        try:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
            timed_out = True
    status = {'command': command, 'returncode': code, 'timed_out': timed_out,
              'elapsed_seconds': time.monotonic()-started, 'passed': code == 0 and not timed_out}
    log.with_suffix(log.suffix+'.status.json').write_text(json.dumps(status, indent=2)+'\n')
    if not status['passed']:
        raise RuntimeError(f'poison compilation/link failed; see {log}')
    return status


def run_case(name: str, command: list[str], out: Path, env: dict[str, str], immutable: dict[str, str],
             timeout: float, negative: bool) -> dict:
    log = out / f'{name}.log'
    report_path = out / f'{name}.json'
    command = [*command, '--report', str(report_path), '--capture', str(out / f'{name}.png')]
    status = run_checked(command, log, expected=int(negative), timeout=timeout, env=env,
                         required=(r'^HOT-RELOAD .*\| PASS$', r'^RT checks [1-9][0-9]* failures \d+ \| (PASS|FAIL)$'))
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    errors = list(status['failures']) + assess(log.read_text(errors='replace'), report, negative)
    assert_unchanged(immutable)
    result = {'name': name, 'passed': not errors, 'failures': errors,
              'expected_exit': int(negative), 'status': str(log)+'.status.json',
              'report': str(report_path), 'pipelines': report.get('pipelines'), 'rt': report.get('rt')}
    print(('PASS' if result['passed'] else 'FAIL') + ': ' + name, flush=True)
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        probe = root / 'probe'; probe.mkdir()
        for name in ('a.air', 'material_passes.air', 'rt_scene.air', 'z.air'):
            (probe/name).write_bytes(b'test')
        replacement = root / 'poison.air'
        shaders = root/'shaders'; shaders.mkdir()
        for name in ('a.metal', 'forward.metal', 'material_passes.metal', 'rt_scene.metal', 'visibility_resolve.metal', 'z.metal'):
            (shaders/name).write_text('// test')
        cmake = ('list(REMOVE_ITEM PHOSPHOR_METAL_SHADERS ${CMAKE_SOURCE_DIR}/shaders/forward.metal '
                 '${CMAKE_SOURCE_DIR}/shaders/visibility_resolve.metal ${CMAKE_SOURCE_DIR}/shaders/material_passes.metal)\n'
                 'list(PREPEND PHOSPHOR_METAL_SHADERS ${CMAKE_SOURCE_DIR}/shaders/material_passes.metal)')
        modules = shader_modules(cmake, root)
        assert modules == ['material_passes.air', 'a.air', 'rt_scene.air', 'z.air']
        linked = ordered_probe_air(probe, replacement, modules)
        assert linked[0].name == 'material_passes.air' and linked.count(replacement) == 1
        assert probe/'rt_scene.air' not in linked
        (probe/'forward.air').write_bytes(b'stale')
        assert probe/'forward.air' not in ordered_probe_air(probe, replacement, modules)
        flags = metal_flags('set(PHOSPHOR_METAL_FLAGS -std=metal4.0 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}\n'
                            ' -I ${CMAKE_SOURCE_DIR}/src -I ${CMAKE_BINARY_DIR}/generated -Wall -fpreserve-invariance)',
                            {'CMAKE_OSX_DEPLOYMENT_TARGET': '26.0'}, root/'source with spaces', root/'build with spaces')
        assert str(root/'source with spaces/src') in flags and '-mmacosx-version-min=26.0' in flags
        assert poison_alpha('    ++payload.alphaTests;\n').count('++payload.opaqueAlphaTests;') == 1
        report = {'pipelines': {'archiveHits': 25, 'compilerCalls': 2, 'reloads': 1, 'reloadFailures': 0,
                                'failures': 0, 'renderThreadCompiles': 0},
                  'rt': {'enabled': True, 'checks': 4, 'alpha_tests': 12, 'check_failures': 0, 'opaque_alpha_tests': 0}}
        before = 'RT check frame 0 | checked 512 | opaque-alpha 0 invalid-mesh 0 | PASS\n'
        reload = 'Shaders reloaded: generation 2\n'
        after = ''.join(f'RT check frame {i} | checked 512 | opaque-alpha 0 invalid-mesh 0 | PASS\n' for i in (6, 7, 8))
        tail = 'HOT-RELOAD reloads 1 | probe pixels 100 | PASS\n'
        assert not assess(before+reload+after+tail, report, False)
        poison = after.replace('opaque-alpha 0', 'opaque-alpha 3').replace('| PASS', '| FAIL')
        report['rt']['check_failures'] = 3; report['rt']['opaque_alpha_tests'] = 9
        valid = before+reload+poison+tail
        assert not assess(valid, report, True)
        assert assess(before+reload+after+tail, report, True)  # Stale IFT never sees poison.
        assert assess(valid+'Shader Validation Error\n', report, True)
        assert assess(valid+'Metal GPU Validation Enabled\n', report, True)
        report['pipelines']['archiveHits'] = 0
        assert assess(valid, report, True)
    print('rt_archive_reload_check: CPU controls passed; no compiler or GPU run')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--build', type=Path, default=Path('build/release'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--frame-delay-ms', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.out is None:
        parser.error('--out is required and must not exist')
    if args.frames < 120 or not 1 <= args.frame_delay_ms <= 1000:
        parser.error('at least 120 frames and a delay in 1..1000 ms are required for asynchronous reload')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    repo = Path(__file__).resolve().parents[1]
    build, out = args.build.resolve(), args.out.resolve()
    cache_path = build/'CMakeCache.txt'
    cache = read_cache(cache_path)
    if cache.get('CMAKE_BUILD_TYPE') != 'Release':
        parser.error('this archive-active test requires a configured Release build')
    if Path(cache.get('CMAKE_HOME_DIRECTORY', '')).resolve() != repo:
        parser.error('build CMAKE_HOME_DIRECTORY does not match this source checkout')
    if out.exists():
        parser.error('--out already exists; old evidence must not be overwritten')
    cmake_path = repo/'cmake/App.cmake'
    cmake_source = cmake_path.read_text()
    flags = metal_flags(cmake_source, cache, repo, build)
    modules = shader_modules(cmake_source, repo)
    app = build/'phosphor'
    library = build/'shaders/phosphor.metallib'
    archive = build/'shaders/phosphor-archive.metallib'
    normal_probe = build/'shaders/hot-reload-probe.metallib'
    scene = repo/'assets/sponza/Sponza.gltf'
    generated = build/'generated'
    original_scene = repo/'shaders/rt_scene.metal'
    original_common = repo/'shaders/rt_common.h'
    poison_air, poison_library = out/'rt_scene-poison.air', out/'poison.metallib'
    linked_air = ordered_probe_air(build/'shaders/probe', poison_air, modules)
    inputs = [app, library, archive, normal_probe, scene, cache_path, cmake_path,
              original_scene, original_common, repo/'src/renderer/gpu_types.h',
              *sorted((build/'shaders/probe').glob('*.air')), *sorted(generated.rglob('*.h'))]
    for path in inputs:
        if not path.is_file():
            parser.error('missing required input: '+str(path))
    if not os.access(app, os.X_OK):
        parser.error('renderer is not executable: '+str(app))
    if not generated.is_dir():
        parser.error('generated include directory missing: '+str(generated))
    # Hash relevant shader inputs too: compilation must not race an edit in
    # another task, and this runner never rewrites originals or build outputs.
    inputs.extend(sorted((repo/'shaders').glob('*.h')))
    inputs.extend(sorted((repo/'shaders').glob('*.metal')))
    immutable = {str(path): checksum(path) for path in inputs}
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    environment = archive_environment()
    common = [str(app), '--offscreen', '--no-ui', '--no-vsync', '--fixed-timestep', '--bench', '4', '--scene', str(scene),
              '--rt', 'on', '--rt-probe', 'primary', '--debug-rt', '1', '--frames-in-flight', '3', '--warmup', '0',
              '--frames', str(args.frames), '--resolution', '640x360', '--debug-frame-delay-ms', str(args.frame_delay_ms),
              '--pipeline-archive', str(archive)]
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'F9 archive-hint IFT hot reload',
                'build': str(build), 'build_type': cache['CMAKE_BUILD_TYPE'], 'metal_flags': flags, 'linked_probe_modules': modules,
                'immutable_inputs': immutable, 'state': 'RUNNING', 'results': [], 'compile_steps': []}
    manifest_path = out/'summary.json'
    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    try:
        positive = run_case('positive', [*common, '--debug-hot-reload', str(normal_probe)], out, environment,
                            immutable, args.timeout, False)
        manifest['results'].append(positive)
        save()
        if not positive['passed']:
            raise RuntimeError('normal archive-active reload failed; poison run not started')
        with tempfile.TemporaryDirectory(prefix='poison-source-', dir=out) as temporary:
            temporary = Path(temporary)
            shader = temporary/'rt_scene.metal'
            header = temporary/'rt_common.h'
            shutil.copyfile(original_scene, shader)
            header.write_text(poison_alpha(original_common.read_text()))
            # Keep the exact changed source as evidence after the temporary
            # compile directory is removed. The normal shader remains intact.
            shutil.copyfile(header, out/'poison-rt_common.h')
            compile_command = ['xcrun', '-sdk', 'macosx', 'metal', *flags, '-DPHOSPHOR_HOT_RELOAD_PROBE=1',
                               '-c', str(shader), '-o', str(poison_air)]
            link_command = ['xcrun', '-sdk', 'macosx', 'metallib', *map(str, linked_air), '-o', str(poison_library)]
            for name, command in (('compile', compile_command), ('link', link_command)):
                manifest['compile_steps'].append(compile_step(command, out/f'{name}.log', args.timeout, environment))
                save()
        assert_unchanged(immutable)
        generated_hashes = {str(path): checksum(path) for path in (poison_air, poison_library, out/'poison-rt_common.h')}
        manifest['generated_artifacts'] = generated_hashes
        negative = run_case('alpha-poison', [*common, '--debug-hot-reload', str(poison_library)], out, environment,
                            {**immutable, **generated_hashes}, args.timeout, True)
        manifest['results'].append(negative)
        manifest['state'] = 'PASS' if negative['passed'] else 'FAIL'
        save()
        return 0 if negative['passed'] else 1
    except Exception as error:
        manifest['state'] = 'FAIL'
        manifest['error'] = str(error)
        save()
        print('FAIL: '+str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
