#!/usr/bin/env python3
"""F15 closure (ticket 20): the chess back-light rig under API + shader validation.

The back-light rig (574k-triangle pawn tops, RT reflections and RT glass transport, 8 reflection samples) costs about 620 ms per frame at
1920x1080 without validation (measured, build/f15-chess-valid2).  Shader validation instruments every shader access and makes a frame several
times slower (640x360, the exact capture flags with --debug-lighting 1: 220 ms plain, 1126 ms validated, 5.1x; 383 ms without --debug-lighting), so the
1080p frame needs more than the engine's 5 s frame wait (src/platform/metal/metal_context.cpp, kWaitTimeoutMs): "GPU timeout waiting for frame event"
is the wait giving up on a slow frame, not a validation message, and a longer wait would put a multi-second command buffer in front of the
WindowServer watchdog (CLAUDE.md GPU safety: command buffers far below 1 s).  The rig is therefore validated at 640x360, 12 frames, without the
112-frame accumulation (the exact flags of tools/f15_chess_check.py otherwise, `--debug-lighting 1` included, `--no-pipeline-archive`: Metal rejects archive
lookups under shader validation).  PASS: every capture exits 0 with no GPU timeout, no `Shader Validation` / `failed assertion` / `Validation Error`
line and no `Redundant call to setDepthStencilState / setViewport` notice.

  python3 "$H/gpulock.py" python3 -I tools/f15_chess_validate.py run OUTDIR [BINARY] [--gltf FILE] [--rig bl-t100 --rig bl-t000 ...]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_chess_check as cc  # noqa: E402

DEFAULT_GLTF = pathlib.Path("/Users/danielsan/Documents/phosphor/.claude/worktrees/f15-plan-c38b51/build/khronos-corpus/ABeautifulGame/glTF/ABeautifulGame.gltf")
VALIDATION_ENV = {"MTL_DEBUG_LAYER": "1", "MTL_SHADER_VALIDATION": "1", "MTL_DEBUG_LAYER_WARNING_MODE": "nslog"}
BAD = re.compile(r"Shader Validation|failed assertion|Validation Error|GPU timeout")
REDUNDANT_STATE = re.compile(r"Redundant call to set(DepthStencilState|Viewport)")
RESOLUTION, FRAMES = "640x360", 12


def validation_command(binary, gltf, overrides, name, folder):
    """The exact capture command of tools/f15_chess_check.py at the validation resolution, without the accumulation."""
    command = cc.capture_command(binary, gltf, overrides, name, folder)
    out = list(command)
    out[out.index("--resolution") + 1] = RESOLUTION
    out[out.index("--frames") + 1] = str(FRAMES)
    out[out.index("--capture-linear-frame") + 1] = str(FRAMES - 1)
    if "--capture-linear-accumulate" in out:
        i = out.index("--capture-linear-accumulate")
        del out[i:i + 2]
    return out + ["--no-pipeline-archive"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    parser.add_argument("--gltf", type=pathlib.Path, default=DEFAULT_GLTF)
    parser.add_argument("--rig", action="append", default=[])
    args = parser.parse_args()
    rigs = args.rig or ["bl-t000", "bl-t100"]
    overrides = REPO / "fixtures/chess/chess-subsurface.overrides.json"
    ok = True
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTL_")}
    env.update(VALIDATION_ENV)
    for rig in rigs:
        folder = args.out / rig
        folder.mkdir(parents=True, exist_ok=True)
        command = validation_command(args.binary, args.gltf, overrides, rig, folder)
        log = folder / "run.log"
        with log.open("w") as f:
            code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=900).returncode
        text = log.read_text(errors="replace")
        bad = BAD.findall(text)
        redundant = len(REDUNDANT_STATE.findall(text))
        frame = re.search(r"frame ([\d.]+) ms", text)
        good = code == 0 and not bad and not redundant
        ok = ok and good
        print(f"{rig}: exit {code}, frame {frame.group(1) if frame else '-'} ms, validation/timeout lines {len(bad)}, redundant depth/viewport notices {redundant} -> "
              f"{'PASS' if good else 'FAIL'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
