#!/usr/bin/env python3
"""F15 (GPU safety) regression: a scene whose kings and queens are pure emissive (about 75k emissive triangles) must complete every frame.

The clustered direct-lighting fallback used to shade every light of an overflowed cell: pixels x lights visibility rays, frame 0 never
completed (ticket 20, "GPU timeout waiting for frame event").  The per-pixel light budget (src/renderer/light_budget.h) bounds it.

  python3 -I tools/f15_emissive_bound_check.py --gltf <ABeautifulGame.gltf> [--resolution 1280x720]

Runs `chess-sidelight/mask=1/emit=1` (the black kings and queens emissive white) SERIALLY under the cooperating GPU lock, RT on and
off, with the renderer's own timeout; passes when both runs exit 0, the report's worst GPU frame time is below the bound (default
100 ms, indicative: measured about 15 ms at 1280x720 with the budget, seconds without it) and no GPU allocation happens in the
measured frames (O7).  Pipelines compile without the archive (the emissive forward variant is not archived; not a lighting check).  Exit codes: 0 pass, 1 fail, 3 GPU busy.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_check as fc  # noqa: E402

SCENARIO = "chess-sidelight/mask=1/emit=1"
BOUND_MS = 100.0


def judge(report, bound_ms=BOUND_MS):
    """Failures of one report (empty = pass)."""
    failures = []
    worst = report["gpu_ms"]["max"]
    if not worst < bound_ms:
        failures.append(f"worst GPU frame {worst:.2f} ms is not below {bound_ms:g} ms")
    if report.get("gpu_allocations", 0) != 0:
        failures.append(f"{report['gpu_allocations']} GPU allocations in the measured frames (O7)")
    return failures


def command(binary, gltf, rt, resolution, report):
    return [str(binary), "--scene", str(gltf), "--bench", "2", "--material-scenario", SCENARIO, "--render-path", "visibility",
            "--geometry-path", "mesh", "--rt", "on" if rt else "off", "--shadows", "rt" if rt else "off", "--lighting", "legacy",
            "--gi", "off", "--reflections", "off", "--ao", "off", "--lighting-denoise", "custom", "--post", "--upscaler", "native",
            "--resolution", resolution, "--no-ui", "--no-vsync", "--frames", "16", "--warmup", "4", "--fixed-timestep",
            "--debug-lighting", "1", "--no-pipeline-archive", "--report", str(report)]  # the emissive forward variant is not in the archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gltf", required=True, type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--bound-ms", type=float, default=BOUND_MS)
    parser.add_argument("--gpu-lock", default="/tmp/phosphor-gpu-verification.lock")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if not args.binary.is_file() or not args.gltf.is_file():
        parser.error("needs an existing renderer binary and glTF")
    failed = False
    try:
        with fc.gpu_lock(args.gpu_lock):
            for rt in (True, False):
                busy = fc.running_gpu_processes()
                if busy:
                    raise fc.GpuBusy(f"other GPU jobs are running: {', '.join(busy)}")
                folder = args.out / ("rt" if rt else "no-rt")
                folder.mkdir(parents=True, exist_ok=True)
                status = fc.capture(command(args.binary, args.gltf, rt, args.resolution, folder / "report.json"), folder / "engine.log",
                                    args.timeout)
                if not status["passed"]:
                    print(f"FAIL rt={'on' if rt else 'off'}: run failed: {status['failures']}")
                    failed = True
                    continue
                report = json.loads((folder / "report.json").read_text())
                failures = judge(report, args.bound_ms)
                print(f"{'FAIL' if failures else 'PASS'} rt={'on' if rt else 'off'}: worst GPU frame {report['gpu_ms']['max']:.2f} ms "
                      f"(mean {report['gpu_ms']['mean']:.2f}), allocations {report.get('gpu_allocations')}", *failures)
                failed |= bool(failures)
    except fc.GpuBusy as error:
        print(f"BUSY: {error}")
        return 3
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
