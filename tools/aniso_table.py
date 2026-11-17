#!/usr/bin/env python3
"""Generate src/renderer/aniso_table.h: the directional albedo (A, B) of the anisotropic GGX lobe, the split-sum scale and bias.

F15.1 (ticket 11, round 3).  The declared environment of an anisotropic material (the resolve's constant / hemisphere environment) is a
split sum, E = f0 A + f90 B, exactly like the isotropic DFG table (tools/dfg_table.py, renderer/dfg_table.h).  The isotropic table at an
effective roughness cannot follow the lobe (measured: rms fit 6.5 % at strength 1, the BEST single isotropic roughness still 4.0-4.8 %
against the exact albedo, decision 11 allows 5 %), because the albedo of a stretched lobe depends on the azimuth of the view vector
against the lobe axis T'.  This table integrates the anisotropic lobe itself:

    A(mu, phi, r, s) = integral over the incident hemisphere of D V cos(theta_i) (1 - w(V.H)),   B = same with w(V.H)
    w = (1 - V.H)^5 (the Schlick weight, so that E = f0 (1 - w) + f90 w integrates to f0 A + f90 B),
    alpha_b = max(r^2, 0.002), alpha_t = alpha_b + (1 - alpha_b) s^2     (KHR_materials_anisotropy, README rule),
    D the Burley anisotropic GGX, V the README's height-correlated Smith visibility (not clamped), as in shaders/materials/bsdf.h.

Dimensions (4-D, all four matter):
  * mu = N.V          : like the DFG table (the directional albedo is a function of the polar angle of the view).
  * phi = azimuth of V in the lobe frame (T', B'), folded into [0, pi/2] (the lobe is even in both axes).  An azimuth-averaged 3-D table
    would still leave 3.6-4.5 % at strength 1 against the exact albedo (measured, tools/test_aniso_table.py reports it): too close to the
    5 % bound and blind to the direction of the highlight; the azimuth costs two more texel reads.
  * r  = base roughness (alpha_b = max(r^2, 0.002)) and s = anisotropy strength (alpha_t follows): the two parameters of the lobe.  Strength
    0 is the isotropic lobe, the table's s = 0 slice is bit-compatible with the isotropic quadrature of the same code (tested).
Layout: AnisoTable[(((si * R + ri) * P + pi) * M + mi) * 2 + c], c 0 = A, 1 = B; node mi = mu index with node 0 evaluated at MU_FLOOR,
node pi = phi = pi/2 * pi / (P - 1), node ri = r = ri / (R - 1), node si = s = si / (S - 1).  Lookup: four-linear (renderer/aniso_table.h).

The integration is a deterministic half-vector quadrature, independent of the GPU and of any random sequence: the half vector is sampled by
the CDF of D cos(theta_h) (azimuth atan2(ab sin 2 pi u2, at cos 2 pi u2), radial u1), midpoint rule in (u1, u2), integrand
V cos_i 4 (V.H) / cos_h.  tools/test_aniso_table.py compares it against a wi-space quadrature (cosine-weighted midpoint rule over the
incident hemisphere, no half-vector substitution), against the reference's own integrator (tools/f15_reference.py) and against the
shipped header.

Usage: python3 tools/aniso_table.py [--out PATH] [--jobs N]
"""
import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "src/renderer/aniso_table.h"
M, P, R, S = 24, 9, 17, 9   # nodes over N.V, azimuth, roughness, strength
MU_FLOOR = 1e-3             # node 0 of the N.V axis
QUAD = (192, 192)           # (u1, u2) nodes of the half-vector quadrature


def alphas(r, s):
    """(alpha_t, alpha_b) of the README rule."""
    ab = max(r * r, 0.002)
    return ab + (1.0 - ab) * s * s, ab


def half_vector_integrals(mu, phi, at, ab, quad=QUAD):
    """(A, B) of one view (cos theta = mu, azimuth phi from T') and one lobe (alpha_t, alpha_b)."""
    n1, n2 = quad
    u1 = (np.arange(n1) + 0.5) / n1
    u2 = (np.arange(n2) + 0.5) / n2
    phi_h = np.arctan2(ab * np.sin(2 * math.pi * u2), at * np.cos(2 * math.pi * u2))
    tan2 = (u1[:, None] / (1.0 - u1[:, None])) / ((np.cos(phi_h) / at) ** 2 + (np.sin(phi_h) / ab) ** 2)[None, :]
    cos_h = 1.0 / np.sqrt(1.0 + tan2)
    sin_h = np.sqrt(np.maximum(0.0, 1.0 - cos_h * cos_h))
    hx, hy, hz = sin_h * np.cos(phi_h)[None, :], sin_h * np.sin(phi_h)[None, :], cos_h
    sin_v = math.sqrt(max(0.0, 1.0 - mu * mu))
    wo = (sin_v * math.cos(phi), sin_v * math.sin(phi), mu)
    vh = hx * wo[0] + hy * wo[1] + hz * wo[2]
    lx, ly, lz = 2.0 * vh * hx - wo[0], 2.0 * vh * hy - wo[1], 2.0 * vh * hz - wo[2]
    valid = (lz > 0) & (vh > 0)
    lz = np.maximum(lz, 0.0)
    gv = lz * math.sqrt((at * wo[0]) ** 2 + (ab * wo[1]) ** 2 + mu * mu)
    gl = mu * np.sqrt((at * lx) ** 2 + (ab * ly) ** 2 + lz ** 2)
    vis = 0.5 / np.maximum(gv + gl, 1e-12)
    base = np.where(valid, vis * lz * 4.0 * vh / hz, 0.0)
    w = (1.0 - np.clip(vh, 0.0, 1.0)) ** 5
    return float(np.mean(base * (1.0 - w))), float(np.mean(base * w))


def _mu_nodes():
    return [MU_FLOOR if i == 0 else i / (M - 1) for i in range(M)]


def _row(args):
    si, ri = args
    at, ab = alphas(ri / (R - 1), si / (S - 1))
    out = []
    for pi_ in range(P):
        phi = 0.5 * math.pi * pi_ / (P - 1)
        for mu in _mu_nodes():
            out.extend(half_vector_integrals(mu, phi, at, ab))
    return out


def build(jobs=3):
    """The flat table, [si][ri][pi][mi][c]."""
    with Pool(jobs) as pool:
        rows = pool.map(_row, [(si, ri) for si in range(S) for ri in range(R)])
    return np.asarray(rows, dtype=np.float64).reshape(S, R, P, M, 2)


def lookup(table, mu, phi, r, s):
    """(A, B) by the four-linear rule the shaders use (`anisoTableLookup`): cell = floor(value * (n - 1)) clamped to n - 2, fractional weights."""
    def cell(x, n):
        t = min(max(x, 0.0), 1.0) * (n - 1)
        i = min(int(t), n - 2)
        return i, t - i
    mi, fm = cell(mu, M)
    pi_, fp = cell(phi / (0.5 * math.pi), P)
    ri, fr = cell(r, R)
    si, fs = cell(s, S)
    out = np.zeros(2)
    for ds in (0, 1):
        for dr in (0, 1):
            for dp in (0, 1):
                for dm in (0, 1):
                    w = (fs if ds else 1 - fs) * (fr if dr else 1 - fr) * (fp if dp else 1 - fp) * (fm if dm else 1 - fm)
                    out += w * table[si + ds, ri + dr, pi_ + dp, mi + dm]
    return float(out[0]), float(out[1])


def parse(path=DEFAULT_OUT):
    """The shipped table of a generated header as an array [S, R, P, M, 2]."""
    import re
    text = Path(path).read_text()
    start = text.index("float AnisoTable[")
    body = text[text.index("{", start) + 1:text.index("};", start)]
    values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)]
    dims = [int(re.search(name + r"\s*=\s*(\d+)", text).group(1)) for name in ("AnisoTableMu", "AnisoTableAzimuth", "AnisoTableRoughness", "AnisoTableStrength")]
    if len(values) != int(np.prod(dims)) * 2:
        raise ValueError(f"{path}: {len(values)} values for dimensions {dims}")
    return np.asarray(values, dtype=np.float64).reshape(dims[3], dims[2], dims[1], dims[0], 2)


def write_header(table, out):
    values = table.reshape(-1)
    lines = [
        "#pragma once",
        "// GENERATED by tools/aniso_table.py; do not edit.",
        "// F15.1 (ticket 11, round 3): directional albedo (A, B) of the anisotropic GGX lobe (D Burley, V height-correlated Smith, the README rule",
        "// alpha_b = max(r^2, 0.002), alpha_t = mix(alpha_b, 1, s^2)); E = f0 A + f90 B under a constant environment.  Layout:",
        "// AnisoTable[(((strength * R + roughness) * P + azimuth) * M + mu) * 2 + c], c 0 = A, 1 = B; node coordinates mu = i / (M - 1) (node 0 at",
        "// 1e-3), azimuth = pi / 2 i / (P - 1) of V against the lobe axis T' (folded: the lobe is even in both axes), r = i / (R - 1), s = i / (S - 1).",
        "// Four-linear lookup (anisoTableLookup); one GPU buffer after the DFG and multiple-scattering tables (scene_renderer.cpp).",
        '#include "renderer/gpu_types.h"',
        "namespace phosphor {",
        f"PHOSPHOR_GPU_CONSTANT u32 AnisoTableMu = {M}, AnisoTableAzimuth = {P}, AnisoTableRoughness = {R}, AnisoTableStrength = {S};",
        "// The data array is C++ (the GPU buffer is filled from it) or an explicit opt-in of a compute kernel: fragment functions must read the",
        "// buffer (MTL_SHADER_VALIDATION corrupts their reads of program-scope constant arrays) and no other shader pays for 0.7 MB of source.",
        "#if !defined(__METAL_VERSION__) || defined(PHOSPHOR_ANISO_TABLE_DATA)",
        f"PHOSPHOR_GPU_CONSTANT float AnisoTable[{len(values)}] = {{",
    ]
    per_line = 8
    for i in range(0, len(values), per_line):
        lines.append("    " + ", ".join(f"{v:.8e}f" for v in values[i:i + per_line]) + ",")
    lines += [
        "};",
        "#endif",
        "#if defined(__METAL_VERSION__)",
        "#define PHOSPHOR_ANISO_TABLE const constant float*",
        "#else",
        "#define PHOSPHOR_ANISO_TABLE const float*",
        "#endif",
        "// Four-linear lookup on any copy of the table (the program-scope array above or the GPU buffer holding the same values).  `azimuth01` is",
        "// the azimuth of V against T' in [0, 1] (0 .. pi / 2, folded), `roughness` the material roughness r, `strength` s; component 0 = A, 1 = B.",
        "struct AnisoCell { u32 i; float f; };",
        "inline AnisoCell anisoTableCell(float x, u32 n) {",
        "    const float t = (x > 0 ? (x < 1 ? x : 1.0f) : 0.0f) * float(n - 1);",
        "    AnisoCell c;",
        "    c.i = u32(t) < n - 1 ? u32(t) : n - 2;",
        "    c.f = t - float(c.i);",
        "    return c;",
        "}",
        "inline float anisoTableLookup(PHOSPHOR_ANISO_TABLE table, float nv, float azimuth01, float roughness, float strength, u32 component) {",
        "    const AnisoCell cm = anisoTableCell(nv, AnisoTableMu), cp = anisoTableCell(azimuth01, AnisoTableAzimuth);",
        "    const AnisoCell cr = anisoTableCell(roughness, AnisoTableRoughness), cs = anisoTableCell(strength, AnisoTableStrength);",
        "    const float fm = cm.f, fp = cp.f, fr = cr.f, fs = cs.f;",
        "    float sum = 0.0f;",
        "    for (u32 ds = 0; ds < 2; ++ds) {",
        "        const float ws = ds ? fs : 1.0f - fs;",
        "        for (u32 dr = 0; dr < 2; ++dr) {",
        "            const float wr = ws * (dr ? fr : 1.0f - fr);",
        "            for (u32 dp = 0; dp < 2; ++dp) {",
        "                const float wp = wr * (dp ? fp : 1.0f - fp);",
        "                const u32 base = (((cs.i + ds) * AnisoTableRoughness + cr.i + dr) * AnisoTableAzimuth + cp.i + dp) * AnisoTableMu + cm.i;",
        "                sum += wp * ((1.0f - fm) * table[base * 2 + component] + fm * table[(base + 1) * 2 + component]);",
        "            }",
        "        }",
        "    }",
        "    return sum;",
        "}",
        "} // namespace phosphor",
        "",
    ]
    Path(out).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    table = build(args.jobs)
    write_header(table, args.out)
    print(f"wrote {args.out}: {table.size} values ({M} x {P} x {R} x {S} x 2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
