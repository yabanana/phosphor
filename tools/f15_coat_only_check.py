#!/usr/bin/env python3
"""F15 closure (ticket 09): the coat-only feature union (mask 2) fixture.

`clearcoat-grid-coat-only-sun` is `clearcoat-grid-sun` without the ior block of row 4, so every sphere carries the clearcoat block only (feature
union 2; clearcoat-grid-sun is 3).  Criteria, fixed before any capture: with the archive complete for mask 2 the capture has 0 archive misses and
0 render-thread compiles after the first frame (the PIPELINES line), and every pixel OUTSIDE the band of row 4 equals the pixel of
clearcoat-grid-sun up to FLOAT ROUNDING (the rows 0-3 and 5 carry the same coat blocks; the two scenes differ only by the specialisation mask and
row 4's ior block).  The first version of this check required bit equality and failed: the two specialisations are different compiled programs
(constant folding reorders float operations), the first capture showed 34 050 pixels outside the band with an absolute difference of at most
1.8e-7 (1.5 ulp at 1.0) and none above 1e-6.  The criterion is therefore the rounding bound fixed from that independent fact (8 float32 ulps at
1.0 = 1e-6), not a loosened physical threshold: outside the band every channel differs by at most 1e-6.  Negative control: inside the band the
images must differ by more than 1e-3 in at least one pixel (row 4 lost its ior 2.4 block), otherwise the comparison cannot see a difference.

  python3 "$H/gpulock.py" python3 -I tools/f15_coat_only_check.py run OUTDIR [BINARY]
"""
from __future__ import annotations

import argparse
import array
import math
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
W, H = 1280, 720


def read_pfm(path):
    data = pathlib.Path(path).read_bytes()
    m = re.match(rb"(PF|Pf)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s", data)
    w, h, scale = int(m.group(2)), int(m.group(3)), float(m.group(4))
    values = array.array("f")
    values.frombytes(data[m.end():m.end() + w * h * 12])
    if (scale < 0) != (sys.byteorder == "little"):
        values.byteswap()
    return w, h, values


def capture(binary, scenario, folder, extra=()):
    folder.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario, "--render-path", "visibility", "--geometry-path", "mesh",
           "--rt", "off", "--shadows", "off", "--lighting", "legacy", "--gi", "off", "--reflections", "ssr", "--ao", "off", "--lighting-denoise", "custom",
           "--reflection-samples", "8", "--lighting-seed", "1", "--post", "--upscaler", "native", "--resolution", f"{W}x{H}", "--frames", "8",
           "--warmup", "8", "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--offscreen", "--capture-linear", str(folder / "engine.pfm"),
           "--capture-linear-frame", "7", "--capture-linear-signal", "hdr", *extra]
    log = folder / "run.log"
    with log.open("w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=600).returncode
    return code, log.read_text(errors="replace")


def band():
    """Pixel rows of sphere row 4 (centre y 4.0, radius 0.4, camera (0, 2.5, 5.5) looking -Z, 60 degree vertical fov), widened by 8 %."""
    depth = 5.5
    half_h = depth * math.tan(math.radians(30))
    top = (1 - (4.0 + 0.4 * 1.08 - 2.5) / half_h) / 2 * H
    bottom = (1 - (4.0 - 0.4 * 1.08 - 2.5) / half_h) / 2 * H
    return int(math.floor(top)) - 2, int(math.ceil(bottom)) + 2


def compare(folder):
    a, b = read_pfm(folder / "ref" / "engine.pfm"), read_pfm(folder / "coat-only" / "engine.pfm")
    top, bottom = band()
    outside = inside = 0
    max_outside = max_inside = 0.0
    for y in range(H):
        row = H - 1 - y
        offset = row * W * 3
        ra, rb = a[2][offset:offset + W * 3], b[2][offset:offset + W * 3]
        if ra == rb:
            continue
        for i in range(0, W * 3, 3):
            d = max(abs(ra[i + k] - rb[i + k]) for k in range(3))
            if top <= y <= bottom:
                max_inside = max(max_inside, d)
                if d > 1e-3:
                    inside += 1
            else:
                max_outside = max(max_outside, d)
                if d > 1e-6:
                    outside += 1
    return outside, inside, (top, bottom), max_outside, max_inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "verify"])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("binary", type=pathlib.Path, nargs="?", default=REPO / "build/release/phosphor")
    args = parser.parse_args()
    ok = True
    if args.mode == "run":
        for name, scenario in (("ref", "clearcoat-grid-sun"), ("coat-only", "clearcoat-grid-coat-only-sun")):
            code, text = capture(args.binary, scenario, args.out / name)
            pipelines = [l for l in text.splitlines() if l.startswith("PIPELINES")][-1:]
            misses = re.search(r"misses (\d+)", pipelines[0]).group(1) if pipelines else "?"
            compiles = re.search(r"render-thread compiles (\d+)", pipelines[0]).group(1) if pipelines else "?"
            print(f"{name}: exit {code}; archive misses {misses}; render-thread compiles {compiles}; {pipelines}", flush=True)
            if code != 0 or misses != "0":
                ok = False
    outside, inside, rows, max_outside, max_inside = compare(args.out)
    print(f"pixels differing by more than 1e-6 outside the row-4 band (rows {rows[0]}..{rows[1]}): {outside} (must be 0; largest difference {max_outside:.3g}); "
          f"inside the band: {inside} pixels differ by more than 1e-3 (must be > 0, the control; largest {max_inside:.3g})")
    ok = ok and outside == 0 and inside > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
