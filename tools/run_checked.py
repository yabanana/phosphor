#!/usr/bin/env python3
"""Run one renderer process, preserving raw status, signal, EXIT marker and log.

A PASS line never overrides a failing process. No retries or concurrent GPU runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def run_checked(command, log, expected=0, required=(), timeout=120, marker=True, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    artifacts = {}
    for argument in command:
        path = Path(argument)
        if path.is_file() and path.name == 'phosphor':
            for item in (path, path.parent/'shaders/phosphor.metallib'):
                if item.is_file(): artifacts[str(item)] = hashlib.sha256(item.read_bytes()).hexdigest()
            break
    commit = subprocess.run(['git','rev-parse','HEAD'], capture_output=True, text=True).stdout.strip()
    patch = subprocess.run(['git','diff','--binary','HEAD'], capture_output=True).stdout
    begin = time.monotonic()
    timed_out = False
    with log.open('wb') as output:
        try:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = None
            timed_out = True
    text = log.read_text(errors='replace')
    markers = re.findall(r'^EXIT (\d+)$', text, re.M)
    end_marker = int(markers[-1]) if markers else None
    reasons = []
    if timed_out:
        reasons.append('process timeout')
    if code != expected:
        reasons.append(f'process exit {code}, expected {expected}')
    if marker and end_marker != expected:
        reasons.append(f'EXIT marker {end_marker}, expected {expected}')
    for expression in required:
        if re.search(expression, text, re.M) is None:
            reasons.append(f'missing required output: {expression}')
    if expected == 0 and re.search(r'failed assertion|\[ERROR\]|\| FAIL\b|GPU timeout|command buffers failed', text):
        reasons.append('error/validation failure in log')
    for name, checksum in artifacts.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != checksum:
            reasons.append('artifact changed during run: ' + name)
    binary_hash = next((v for k,v in artifacts.items() if Path(k).name == 'phosphor'),None)
    status = {'command': command, 'started_utc': started, 'elapsed_seconds': time.monotonic()-begin,
              'returncode': code, 'signal': -code if code is not None and code < 0 else None,
              'exit_marker': end_marker, 'expected_exit': expected, 'timed_out': timed_out,
              'binary_sha256': binary_hash, 'artifacts': artifacts, 'commit': commit,
              'tracked_patch_sha256': hashlib.sha256(patch).hexdigest(), 'macos': platform.mac_ver()[0],
              'validation_env': {k:(env or os.environ).get(k) for k in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION','MTL_DEBUG_LAYER_WARNING_MODE')},
              'passed': not reasons, 'failures': reasons}
    log.with_suffix(log.suffix+'.status.json').write_text(json.dumps(status, indent=2)+'\n')
    return status


def self_test():
    cases = [("print('EXIT 0')", 0, True),
             ("print('PASS\\nEXIT 0'); raise SystemExit(1)", 0, False),
             ("print('PASS')", 0, False),
             ("print('FAIL\\nEXIT 1'); raise SystemExit(1)", 1, True)]
    with tempfile.TemporaryDirectory() as folder:
        for i, (source, expected, passed) in enumerate(cases):
            result = run_checked([sys.executable, '-c', source], Path(folder)/f'{i}.log', expected)
            assert result['passed'] == passed, result
    print('run_checked: positive and false-PASS controls passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--expect-exit', type=int, default=0)
    parser.add_argument('--require', action='append', default=[])
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--no-exit-marker', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.log is None:
        parser.error('--log and a command after -- are required')
    result = run_checked(command, args.log, args.expect_exit, args.require, args.timeout, not args.no_exit_marker)
    print(('PASS' if result['passed'] else 'FAIL') + ': ' + str(args.log))
    if result['failures']:
        print('\n'.join(result['failures']), file=sys.stderr)
    raise SystemExit(not result['passed'])


if __name__ == '__main__':
    main()
