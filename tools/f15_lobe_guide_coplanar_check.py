#!/usr/bin/env python3
"""F15 closure (ticket 06): the coplanar-overlap fixture that proves the raster order group of the lobe-guide producer.

`forward_di_receiver_guide_fs` (the overflow receivers of scenes with material features) writes the lobe guide of its pixel from a fragment
shader with [[early_fragment_tests]] and an EQUAL depth test: fragments of exactly coplanar, fully overlapping surfaces all pass, the colour
attachments keep the LAST one (primitive order), and the guide buffer, a device write, must keep the record of the same fragment.  Without
`raster_order_group(0)` on that buffer the writes of overlapping fragments are unordered.

Scene `lobe-guide-coplanar` (src/testbench/material_validation.cpp): ONE draw of 16 coplanar, fully overlapping quads (identical positions,
one depth per pixel) whose vertex normals are tilted by a different angle per layer, so the layer of a fragment shows in its shading normal.
The overflow receivers are forced with `--debug-meshlets 1 --debug-meshlets-corrupt count` (the meshlet self-check then FAILs by design:
exit 1, see tools/f15_lobe_guide_overflow_check.sh).  `--debug-lobe-guide 1` compares the guide with its CPU reconstruction on every frame;
for a coat-only material the coat normal of the guide IS the shading normal of the fragment that wrote it and the check compares it with the
shading normal of the colour attachments (the layer shown).

Four captures: `rog` and `norog` (`--debug-lobe-guide-no-rog`: the same body without the attribute) with equal shading cost per fragment, and
`rog-delay` / `norog-delay` with `--debug-lobe-guide-delay`: a bounded, data-dependent delay (up to 16384 dependent fma) before the guide write
that is longest for the first layer and zero for the last, so that overlapping fragments complete OUT of submission order (the situation of
fragments with different materials, textures or cache misses).  First result, recorded in the ticket: with equal cost the hardware completes
the fragments in order and the control PASSES (`norog` is not a proof); the delay variants are the fixture.  PASS criteria (fixed before the delay
captures): `rog` and `rog-delay`: every checked frame passes and at least 32 000 pixels are extended; `norog-delay` must FAIL on at least one
frame (otherwise the fixture does not prove the need).

  python3 "$H/gpulock.py" python3 -I tools/f15_lobe_guide_coplanar_check.py run OUTDIR [BINARY] [--frames N]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
W, H = 1280, 720
MIN_EXTENDED = 32000


def capture(binary, folder, variant, frames):
    delay = variant.endswith("-delay")
    variant = variant.replace("-delay", "")
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", "lobe-guide-coplanar", "--render-path", "visibility",
           "--geometry-path", "mesh", "--rt", "off", "--shadows", "off", "--lighting", "legacy", "--gi", "off", "--reflections", "off", "--ao", "off",
           "--lighting-denoise", "custom", "--lighting-seed", "1", "--post", "--upscaler", "native", "--resolution", f"{W}x{H}",
           "--frames", str(frames), "--warmup", "4", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--offscreen",
           "--pipeline-sync", "--debug-meshlets", "1", "--debug-meshlets-corrupt", "count", "--debug-lobe-guide", "1",
           "--capture", str(folder / "engine.png")]
    if variant == "norog":
        cmd += ["--debug-lobe-guide-no-rog"]
    if delay:
        cmd += ["--debug-lobe-guide-delay"]
    with (folder / "run.log").open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=600).returncode
    return code, (folder / "run.log").read_text(errors="replace")


CHECK = re.compile(r"LOBE-GUIDE check frame (\d+) \| pixels (\d+) extended (\d+) base (\d+) unverified (\d+) \| (PASS|FAIL|SKIP)(?: \((\d+) mismatches\))?")


def summarize(text):
    rows = CHECK.findall(text)
    passes = [r for r in rows if r[5] == "PASS"]
    fails = [r for r in rows if r[5] == "FAIL"]
    skips = [r for r in rows if r[5] == "SKIP"]
    extended = max((int(r[2]) for r in rows), default=0)
    return {"frames": len(rows), "pass": len(passes), "fail": len(fails), "skip": len(skips), "extended": extended,
            "mismatches": [int(r[6]) for r in fails if r[6]]}


def verdicts(results):
    lines = [f"{name}: {r}" for name, r in results.items()]
    good = all(results[v]["fail"] == 0 and results[v]["pass"] >= 1 and results[v]["extended"] >= MIN_EXTENDED for v in ("rog", "rog-delay"))
    control = results["norog-delay"]["fail"] >= 1 and results["norog-delay"]["extended"] >= MIN_EXTENDED
    lines.append(f"with the raster order group (equal cost / skewed cost): {results['rog']['fail']} / {results['rog-delay']['fail']} failing frames (must be 0) -> "
                 f"{'PASS' if good else 'FAIL'}")
    lines.append(f"informational, equal cost without the group: {results['norog']['fail']} of {results['norog']['frames']} frames fail")
    lines.append(f"control (no raster order group, skewed cost): {results['norog-delay']['fail']} of {results['norog-delay']['frames']} checked frames FAIL "
                 f"(must be >= 1) -> {'PASS' if control else 'FAIL: the control does not fail, the fixture does not prove the need'}")
    return good and control, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "verify"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    parser.add_argument("--frames", type=int, default=16)
    args = parser.parse_args()
    results = {}
    for variant in ("rog", "norog", "rog-delay", "norog-delay"):
        if args.mode == "run":
            code, text = capture(args.binary, args.out / variant, variant, args.frames)
            print(f"{variant}: exit {code}", flush=True)
        else:
            text = (args.out / variant / "run.log").read_text(errors="replace")
        results[variant] = summarize(text)
    ok, lines = verdicts(results)
    print("\n".join(lines))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
