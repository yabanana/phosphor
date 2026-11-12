#!/usr/bin/env python3
"""Author an external cooked reflection probe (six linear RGB PFM faces).

The F13.5 cooked-probe path (`--reflection-probe DIR`) loads px/nx/py/ny/pz/nz
.pfm, 64x64, linear RGB. The engine never exports these files, so the runner's
`probe-parallax-cooked` case needs an input authored outside the engine. This
writer derives every texel direction from the Metal/D3D cube-map face table
(face major axis, s/t axes, t top-down), not from the engine's helpers, and
writes standard PFM (rows bottom to top, negative scale = little endian).

Pattern `analytic-gradient` reproduces the engine's built-in analytic probe,
L = mix((.02,.025,.03), (.1,.12,.16), d.y*.5+.5): a cooked run must then match
the analytic-environment run of the same scene. The gradient is vertical, so a
row-order or up-axis error on the four side faces inverts it and is visible.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

SIDE = 64
FACES = ("px", "nx", "py", "ny", "pz", "nz")


def face_direction(face, s, t):
    """Metal cube-map convention: s right, t down, both in [-1, 1]."""
    if face == "px": d = (1.0, -t, -s)
    elif face == "nx": d = (-1.0, -t, s)
    elif face == "py": d = (s, 1.0, t)
    elif face == "ny": d = (s, -1.0, -t)
    elif face == "pz": d = (s, -t, 1.0)
    elif face == "nz": d = (-s, -t, -1.0)
    else: raise ValueError(face)
    n = math.sqrt(sum(c * c for c in d))
    return tuple(c / n for c in d)


def analytic_gradient(d):
    low, high = (0.02, 0.025, 0.03), (0.1, 0.12, 0.16)
    w = d[1] * 0.5 + 0.5
    return tuple(a + (b - a) * w for a, b in zip(low, high))


def write_pfm(path, rows):
    """rows: top-down list of rows of (r, g, b)."""
    with open(path, "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (SIDE, SIDE))
        for row in reversed(rows):  # PFM stores the bottom row first
            f.write(b"".join(struct.pack("<3f", *texel) for texel in row))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", choices=("analytic-gradient",), default="analytic-gradient")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    digests = {}
    for face in FACES:
        rows = []
        for y in range(SIDE):
            t = (y + 0.5) / SIDE * 2.0 - 1.0
            rows.append([analytic_gradient(face_direction(face, (x + 0.5) / SIDE * 2.0 - 1.0, t)) for x in range(SIDE)])
        path = args.out / (face + ".pfm")
        write_pfm(path, rows)
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / "probe-author.json").write_text(json.dumps(
        {"schema": 1, "writer": "tools/probe_pfm_author.py", "pattern": args.pattern, "side": SIDE,
         "convention": "Metal cube faces px nx py ny pz nz, t top-down, PFM bottom-up", "sha256": digests}, indent=1) + "\n")
    print("authored", args.pattern, "probe at", args.out)


if __name__ == "__main__":
    main()
