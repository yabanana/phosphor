#!/usr/bin/env python3
"""Compare two raw lobe guide dumps (--debug-lobe-guide-dump): every word bit for bit except the oct-encoded direction words,
which are compared by angle (F15.1 ticket 11).

The oct fields of GPULobeGuide (clearcoatNormal, anisotropyDirection) are functions of the interpolated geometry; the visibility
resolve and the indexed overflow receivers interpolate in a different order, so the vectors agree to a tiny angle, not bit for bit
(renderer/lobe_guide_pack.h: "compare oct fields by angle").  Usage: f15_guide_dump_compare.py A B [--max-angle DEG]
Exit 0 = identical outside the oct words and every oct pair within the angle; 1 otherwise.
"""
import argparse
import sys

import numpy as np

WORDS = 20                      # sizeof(GPULobeGuide) / 4
OCT_WORDS = {"clearcoatNormal": 4, "anisotropyDirection": 8}


def decode(words):
    u = ((words & 0xFFFF).astype(np.int64) ^ 0x8000) - 0x8000
    v = ((words >> 16).astype(np.int64) ^ 0x8000) - 0x8000
    u = np.maximum(u / 32767.0, -1.0)
    v = np.maximum(v / 32767.0, -1.0)
    z = 1.0 - np.abs(u) - np.abs(v)
    fold = z < 0
    ou = np.where(fold, (1 - np.abs(v)) * np.where(u >= 0, 1.0, -1.0), u)
    ov = np.where(fold, (1 - np.abs(u)) * np.where(v >= 0, 1.0, -1.0), v)
    vec = np.stack([ou, ov, z], axis=1)
    return vec / np.linalg.norm(vec, axis=1)[:, None]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--max-angle", type=float, default=1.0, help="degrees allowed between oct vectors")
    args = parser.parse_args(argv)
    a = np.fromfile(args.a, dtype=np.uint32)
    b = np.fromfile(args.b, dtype=np.uint32)
    if a.size != b.size or a.size % WORDS:
        print(f"dump sizes differ or are not whole records: {a.size} vs {b.size} words")
        return 1
    a, b = a.reshape(-1, WORDS), b.reshape(-1, WORDS)
    ok = True
    for word in range(WORDS):
        differs = a[:, word] != b[:, word]
        if not differs.any():
            continue
        name = next((n for n, w in OCT_WORDS.items() if w == word), None)
        if name is None:
            print(f"word {word}: {int(differs.sum())} records differ (bit-exact word)")
            ok = False
            continue
        x, y = decode(a[differs, word]), decode(b[differs, word])
        angle = float(np.degrees(np.arccos(np.clip((x * y).sum(axis=1), -1.0, 1.0))).max())
        within = angle <= args.max_angle
        print(f"word {word} ({name}): {int(differs.sum())} records differ, max angle {angle:.4f} deg "
              f"({'within' if within else 'ABOVE'} {args.max_angle} deg)")
        ok = ok and within
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
