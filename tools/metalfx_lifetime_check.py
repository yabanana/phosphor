#!/usr/bin/env python3
"""Sequential MetalFX lifetime checks (direct default + isolated opt-in); sampling runs are not performance measurements."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from run_checked import run_checked


def children(pid):
    r = subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


# Public RUSAGE_INFO_V4 ABI prefix from sys/resource.h. Extra storage lets the
# kernel return the rest of v4 without conflating GPU/IOKit footprint with RSS.
class Usage(ctypes.Structure):
    _fields_ = [('uuid',ctypes.c_ubyte*16)]+[(n,ctypes.c_uint64) for n in
        ('user','system','idle_wakeups','interrupt_wakeups','pageins','wired','resident','physical')]+[('tail',ctypes.c_ubyte*1024)]

LIBPROC=ctypes.CDLL('/usr/lib/libproc.dylib')
LIBPROC.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
LIBPROC.proc_pid_rusage.restype=ctypes.c_int

def footprints(pids):
    result={}
    for pid in pids:
        usage=Usage()
        if LIBPROC.proc_pid_rusage(pid,4,ctypes.byref(usage))==0:
            result[pid]=usage.physical
    return result


def rss(pids):
    if not pids:
        return {}
    r = subprocess.run(['ps', '-o', 'pid=,rss=', '-p', ','.join(map(str, pids))], capture_output=True, text=True)
    return {int(p): int(k)*1024 for p, k in (line.split() for line in r.stdout.splitlines() if len(line.split()) == 2)}


def wait_gone(pids, seconds=10):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        alive = [pid for pid in pids if subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()]
        if not alive:
            return True
        time.sleep(.05)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=Path('build/release'))
    p.add_argument('--out', type=Path, default=Path('build/metalfx-lifetime-check'))
    p.add_argument('--settle-ms', type=int, default=64, help='Diagnostic pacing so workers render between output resizes; not a performance run.')
    p.add_argument('--frames', type=int, default=240, help='Resize-run length (resize every 30 frames, two views).')
    p.add_argument('--direct-only', action='store_true', help='In-process soak only: skip isolated runs and worker controls.')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    app = str(a.build/'phosphor')
    env = dict(os.environ)
    for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION', 'MTL_DEBUG_LAYER_WARNING_MODE'):
        env.pop(key, None)
    base = [app, '--offscreen', '--no-ui', '--no-vsync', '--fixed-timestep', '--warmup', '0', '--bench', '1',
            '--scene', 'procedural', '--resolution', '640x360', '--post', '--upscaler', 'temporal', '--render-scale', '.75']
    results = []
    for mode in ('direct',) if a.direct_only else ('direct', 'isolated'):
        command = [*base, '--metalfx-mode', mode, '--frames', str(a.frames), '--resize-every', '30', '--temporal-views', '2',
                   '--debug-frame-delay-ms', str(a.settle_ms), '--report', str(a.out/(mode+'-resize.json'))]
        samples, seen, limited = [], set(), False
        with (a.out/(mode+'-resize.log')).open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            begin = time.monotonic()
            while process.poll() is None:
                kids = children(process.pid); seen.update(kids)
                sizes = rss([process.pid, *kids])
                physical=footprints([process.pid,*kids])
                samples.append({'seconds': time.monotonic()-begin, 'parent_rss_bytes': sizes.get(process.pid, 0),
                                'child_rss_bytes': sum(sizes.get(pid, 0) for pid in kids),
                                'parent_physical_footprint':physical.get(process.pid,0),
                                'worker_physical_footprints':{pid:physical.get(pid,0) for pid in kids},'children': kids})
                if physical.get(process.pid,0) > 8*1024**3:
                    limited = True; process.terminate(); break
                if time.monotonic()-begin > 120+a.frames*a.settle_ms/1000:
                    process.kill(); raise RuntimeError(mode+' resize timeout')
                time.sleep(.1)
            code = process.wait(timeout=10)
        gone = wait_gone(seen)
        text = (a.out/(mode+'-resize.log')).read_text(errors='replace')
        if mode == 'isolated':
            assert code == 0 and not limited and gone and re.search(r'METALFX-WORKERS.*failures 0 shared-bytes 0 \| PASS', text), text[-2000:]
        else:
            # In-process scalers: every one destroyed (cycle released), no leak.
            assert code == 0 and not limited and 'EXIT 0' in text, text[-2000:]
            assert re.search(r'METALFX-LIFETIME adopted (\d+) .*retained 0 unknown 0 max-release-us \d+ \| PASS', text), text[-2000:]
        report=json.loads((a.out/(mode+'-resize.json')).read_text()) if not limited else {}
        if mode=='direct':
            assert report['rendering']['temporal_frames_total']>a.frames//2, 'Insufficient active temporal frames during resize'
        if mode=='isolated':
            assert report['rendering']['temporal_frames_total']>120, 'Insufficient active temporal frames during resize'
            assert report['rendering']['workers_spawned_total']>=14, 'Insufficient completed worker startups'
        footprint=[x['parent_physical_footprint'] for x in samples if x['parent_physical_footprint']]
        result = {'case': mode+'-resize', 'command': command, 'exit': code, 'stopped_at_8_gib_parent_physical_footprint': limited,
                  'parent_physical_footprint_peak': max(footprint, default=0),
                  'parent_physical_footprint_last': footprint[-1] if footprint else 0,
                  'all_observed_children_gone': gone, 'samples': samples, 'rendering': report.get('rendering',{})}
        results.append(result)
        (a.out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        print(mode+' resize: exit '+str(code)+', memory limit '+str(limited)+', all children gone '+str(gone)+
              ', parent footprint peak '+str(result['parent_physical_footprint_peak'])+' last '+
              str(result['parent_physical_footprint_last']), flush=True)
    if a.direct_only:
        return
    # Stop only our renderer process. Child exit must follow EOF even while
    # another worker exists; unrelated parent FDs must not leak through spawn.
    with (a.out/'parent-kill.log').open('wb') as log:
        proc = subprocess.Popen([*base, '--metalfx-mode', 'isolated', '--frames', '100000', '--temporal-views', '2'],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        begin = time.monotonic(); kids = []
        while time.monotonic()-begin < 30:
            kids = children(proc.pid)
            text = (a.out/'parent-kill.log').read_text(errors='replace')
            if len(kids) == 2 and 'STARTUP first frame submitted' in text:
                break
            if proc.poll() is not None:
                raise RuntimeError('renderer failed before parent-death control')
            time.sleep(.05)
        assert len(kids) == 2
        proc.kill(); code = proc.wait(timeout=5)
        assert code == -signal.SIGKILL
        gone = wait_gone(kids)
        assert gone, ('orphan workers', kids)
        results.append({'case': 'parent-death', 'parent_exit': code, 'children': kids, 'all_children_gone': gone})
        print('parent death: no orphan workers, PASS', flush=True)
    for name, flags in [('crash', ['--debug-metalfx-worker-crash', '4']), ('timeout', ['--debug-metalfx-worker-delay-ms', '1500'])]:
        r = run_checked([*base, '--metalfx-mode', 'isolated', '--frames', '20', *flags], a.out/(name+'.log'), expected=1, marker=False,
                        required=('MetalFX worker: frame IPC',), env=env, timeout=30)
        assert r['passed'], r
        results.append({'case': name, 'expected_failure_observed': True, 'elapsed_seconds': r['elapsed_seconds']})
    (a.out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
