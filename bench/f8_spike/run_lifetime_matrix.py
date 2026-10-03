#!/usr/bin/env python3
"""Build against explicit SDKs, run sequentially, and cross-check with leaks.

Use mise exec -- python3. A live weak target OR leaks is a failed lifetime gate.
The same host OS framework runs in every case; SDK != framework replacement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name('metalfx_lifetime_matrix.mm')


def invoke(command, log, env=None, timeout=120):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
        raw, code = result.stdout, result.returncode
    except subprocess.TimeoutExpired as error:
        raw, code = error.stdout or b'', None
    log.write_bytes(raw)
    return {'command': [str(x) for x in command], 'returncode': code,
            'elapsed_seconds': time.monotonic()-start, 'log': str(log)}, raw.decode(errors='replace')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdk', action='append', type=Path,
                   help='Repeat to compare installed SDKs; defaults to xcrun current SDK.')
    p.add_argument('--out', type=Path, default=ROOT/'build/metalfx-lifetime-matrix')
    p.add_argument('--count', type=int, default=8, choices=range(1, 17))
    args = p.parse_args()
    sdks = args.sdk or [Path(subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip())]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION', 'MTL_HUD_ENABLED', 'DYLD_INSERT_LIBRARIES'):
        env.pop(name, None)
    report = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'host': subprocess.check_output(['sw_vers'], text=True),
              'compiler': subprocess.check_output(['xcrun', 'clang++', '--version'], text=True),
              'scope': 'Different build SDKs; same physical GPU and OS MetalFX runtime. Creation/release only.',
              'leaks_scan_uses_weak_monitor': False,
              'cases': []}
    failed = False
    for index, sdk in enumerate(sdks):
        sdk = sdk.resolve(strict=True)
        binary = args.out/f'matrix-sdk-{index}'
        build, _ = invoke(['xcrun', 'clang++', '-std=c++20', '-fno-objc-arc', '-mmacosx-version-min=26.0',
                           '-isysroot', str(sdk), '-framework', 'Foundation', '-framework', 'Metal',
                           '-framework', 'MetalFX', str(SOURCE), '-o', str(binary)], args.out/f'build-{index}.log')
        if build['returncode'] != 0:
            raise RuntimeError(f'Build failed: {build}')
        subprocess.run(['codesign', '--force', '--sign', '-', '--entitlements',
                        str(ROOT/'cmake/debuggable.entitlements'), str(binary)], check=True, capture_output=True)
        versions = subprocess.check_output(['xcrun', 'vtool', '-show-build', str(binary)], text=True)
        for mode in ('temporal4', 'temporal3', 'spatial4', 'denoised4', 'denoised3'):
            name = f'sdk-{index}-{mode}'
            command = [str(binary), '--mode', mode, '--count', str(args.count)]
            run, text = invoke(command, args.out/(name+'.log'), env)
            final = re.search(r'^FINAL created=(\d+) failed=(\d+) live=(-?\d+)$', text, re.M)
            samples = [dict(zip(('released', 'live', 'allocated_delta'), map(int, row))) for row in
                       re.findall(r'^SAMPLE released=(\d+) live=(-?\d+) allocated_delta=(-?\d+)$', text, re.M)]
            leak_run, leak_text = invoke(['leaks', '--atExit', '--', *command, '--no-weak'], args.out/(name+'-leaks.log'), env)
            leak = re.search(r'Process \d+: (\d+) leaks? for (\d+) total leaked bytes', leak_text)
            passed = bool(final and int(final[2]) == 0 and int(final[3]) == 0 and run['returncode'] == 0
                          and leak and int(leak[1]) == 0 and leak_run['returncode'] == 0)
            row = {'sdk': str(sdk), 'mach_o_versions': versions, 'mode': mode, 'run': run, 'samples': samples,
                   'final_live_scalers': int(final[3]) if final else None, 'leaks_run': leak_run,
                   'leaked_allocations': int(leak[1]) if leak else None,
                   'leaked_cpu_bytes': int(leak[2]) if leak else None, 'lifetime_passed': passed}
            report['cases'].append(row)
            failed |= not passed
            (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'{name}: {"PASS" if passed else "FAIL"}; live={row["final_live_scalers"]}, '
                  f'leaked CPU bytes={row["leaked_cpu_bytes"]}', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
