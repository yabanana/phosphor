#!/usr/bin/env python3
"""Generate src/renderer/ms_table.h: the lost-energy table L = 1 - E and the average albedo E_avg of the engine GGX lobe.

F15 closure, multiple-scattering energy compensation (the model is documented in shaders/materials/bsdf_energy.h).
Kulla and Conty (2017, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH course notes) need
  E(mu, alpha)  = directional albedo of the single-scattering lobe D Vis cos (Fresnel 1)
  E_avg(alpha)  = 2 * integral_0^1 E(mu, alpha) mu dmu
and the lobe is proportional to (1 - E(mu_o)) (1 - E(mu_i)) / (1 - E_avg): products of the LOST energy L = 1 - E, which is
1e-3 .. 1e-1 for a glossy lobe.  The split-sum DFG table of the engine (tools/dfg_table.py, 16384 Monte Carlo samples, E = A + B) is
accurate to ~1e-3 in E, i.e. 1-10 % in L: too coarse for this lobe (measured against the exact quadrature, direct-light fixtures
miss their 1 % bound at roughness <= 0.35).  This table is therefore integrated deterministically: E(mu, alpha) by the half-vector
quadrature of `albedo` below (u = the CDF of D cos(theta_h), the azimuth integrated over its exact valid interval; converged to
~1e-5, see tools/test_ms_table.py), on the same 64 x 64 grid as the DFG table (rows: roughness j/(N-1), alpha = max(r^2, 0.002);
columns: mu = i/(N-1), node 0 evaluated at 1e-3).  E_avg is integrated over the table's OWN piecewise-linear L, analytically per mu
segment (the engine's bilinear lookup), so that
    2 * integral (1 - E(mu)) mu dmu = 1 - E_avg
holds exactly at every roughness node: the compensation then redistributes exactly the energy the single-scattering lobe lacks
(white furnace 1 for F = 1, no gain for F < 1).  `--check` compares E_avg with an INDEPENDENT estimate (a wi-space midpoint quadrature
of D Vis cos cos, no half-vector substitution); tools/test_ms_table.py gates the difference and proves the check can fail.

Usage: python3 tools/ms_table.py [--out PATH] [--check] [--jobs N]
"""
import argparse
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "src/renderer/ms_table.h"
SIZE = 64  # = DfgTableSize: the same grid, so that the shader's lookup rule is the DFG one


def albedo(mu, roughness, n_u=16384, n_phi=256):
    """E(mu, roughness) = integral over the incident hemisphere of D Vis cos(theta_i), Fresnel 1 (alpha = max(r^2, 0.002), GGX D,
    height-correlated Smith Vis).  Half-vector substitution: dOmega_i = 4 (v.h) dOmega_h and D cos(theta_h) dOmega_h = dF (the CDF F of
    the half vector, u), so E = 1/pi int du int_0^phimax dphi Vis cos_i 4 (v.h) / cos(theta_h); the azimuth runs over the exact interval
    cos(phi) > -(mu / sin v) cot(2 theta_h) where the incident direction is above the horizon."""
    alpha = max(roughness * roughness, 0.002)
    a2 = alpha * alpha
    u = (np.arange(n_u) + 0.5) / n_u
    cos_h = np.sqrt((1 - u) / (1 + (a2 - 1) * u))
    sin_h = np.sqrt(np.maximum(0.0, 1 - cos_h ** 2))
    sin_v = max(math.sqrt(max(0.0, 1 - mu * mu)), 1e-12)
    c = -mu * (2 * cos_h ** 2 - 1) / (2 * cos_h * sin_h * sin_v)  # cos(phi) must exceed this for l_z > 0
    phi_max = np.where(c < 1, np.arccos(np.clip(c, -1.0, 1.0)), 0.0)
    k = (np.arange(n_phi) + 0.5) / n_phi
    phi = phi_max[:, None] * k[None, :]
    vh = sin_h[:, None] * np.cos(phi) * sin_v + cos_h[:, None] * mu
    lz = np.maximum(2 * vh * cos_h[:, None] - mu, 0.0)
    den = 2.0 * (mu * np.sqrt(a2 + (1 - a2) * lz * lz) + lz * math.sqrt(a2 + (1 - a2) * mu * mu))
    integrand = lz * 4 * vh / np.maximum(den, 1e-12) / cos_h[:, None]
    return float(np.sum(integrand.mean(axis=1) * phi_max) / n_u / math.pi)


def e_avg_from_nodes(e):
    """2 * integral of the piecewise-linear E (nodes i/(n-1)) times mu: exact per segment.
    Segment [a, b], end values ea, eb: integral of E mu dmu = (b-a)(ea (2a+b) + eb (a+2b)) / 6."""
    n = len(e)
    total = 0.0
    for i in range(n - 1):
        a, b = i / (n - 1), (i + 1) / (n - 1)
        total += (b - a) * (e[i] * (2 * a + b) + e[i + 1] * (a + 2 * b)) / 6.0
    return 2.0 * total


def _row(j, size=SIZE, quality=1):
    r = j / (size - 1)
    mus = [1e-3 if i == 0 else i / (size - 1) for i in range(size)]
    return [max(0.0, 1.0 - albedo(mu, r, 16384 * quality, 256 * quality)) for mu in mus]


def lost_table(size=SIZE, jobs=3, quality=1):
    """L[j][i] = 1 - E at roughness j/(size-1), mu node i."""
    with Pool(jobs) as pool:
        return pool.starmap(_row, [(j, size, quality) for j in range(size)])


LOG_FLOOR = 1e-9  # L is floored before the logarithm (a near mirror at normal incidence loses < 1e-9)


def log_lost_table(lost):
    """ln L of the table: the shader interpolates the logarithm (L spans four decades) and exponentiates."""
    return [[math.log(max(v, LOG_FLOOR)) for v in row] for row in lost]


def e_avg_from_log_nodes(log_row):
    """2 * integral of L mu dmu where ln L is piecewise linear in mu (nodes i/(n-1), the shader's interpolation): 8-point Gauss-Legendre per
    segment of the exponential."""
    n = len(log_row)
    t, w = np.polynomial.legendre.leggauss(8)
    t, w = 0.5 * (t + 1.0), 0.5 * w
    total = 0.0
    for i in range(n - 1):
        a, b = i / (n - 1), (i + 1) / (n - 1)
        value = np.exp(log_row[i] + (log_row[i + 1] - log_row[i]) * t)
        total += (b - a) * float(np.sum(w * value * (a + (b - a) * t)))
    return 2.0 * total


def loss_avg_table(lost):
    """1 - E_avg per roughness row = 2 integral of L mu dmu over the shader's interpolation of the table (stored as its logarithm)."""
    return [e_avg_from_log_nodes(row) for row in log_lost_table(lost)]


def parse_lost(path=DEFAULT_OUT):
    """The lost-energy table L (not its logarithm) of a generated header (rows x columns)."""
    import re
    text = Path(path).read_text()
    start = text.index("float MsLogLostTable[")
    body = text[text.index("{", start) + 1:text.index("};", start)]
    values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)]
    size = int(round(math.sqrt(len(values))))
    if size * size != len(values):
        raise ValueError(f"{path}: {len(values)} lost-energy values is not a square table")
    return [[math.exp(v) for v in values[i * size:(i + 1) * size]] for i in range(size)]


def independent_e_avg(roughness, n_theta=96, n_phi=96):
    """2 * integral over mu_o of mu_o * integral over the incident hemisphere of D V cos(theta_i): a fixed 3D midpoint
    rule (the azimuth of the outgoing direction is 0 by symmetry).  Written from the GGX / height-correlated Smith
    definitions in wi-space; shares no code or table with `albedo`."""
    alpha = max(roughness * roughness, 0.002)
    a2 = alpha * alpha
    mu_o = (np.arange(n_theta) + 0.5) / n_theta
    ti = (np.arange(n_theta) + 0.5) / n_theta * (math.pi / 2)
    dphi = 2 * math.pi / n_phi
    phi = (np.arange(n_phi) + 0.5) * dphi
    dtheta = (math.pi / 2) / n_theta
    ct, st = np.cos(ti)[:, None], np.sin(ti)[:, None]
    li = np.stack([st * np.cos(phi)[None, :], st * np.sin(phi)[None, :], ct * np.ones_like(phi)[None, :]], axis=-1)
    total = 0.0
    for mo in mu_o:
        v = np.array([math.sqrt(max(0.0, 1 - mo * mo)), 0.0, mo])
        h = li + v
        h = h / np.linalg.norm(h, axis=-1, keepdims=True)
        nh = h[..., 2]
        d = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        nl = li[..., 2]
        gv = nl * np.sqrt(mo * mo * (1 - a2) + a2)
        gl = mo * np.sqrt(nl * nl * (1 - a2) + a2)
        vis = 0.5 / np.maximum(gv + gl, 1e-9)
        e_mu = float(np.sum(d * vis * nl * st * dtheta * dphi))
        total += e_mu * mo / n_theta
    return 2.0 * total


def independent_albedo(mu, roughness, n_theta=512, n_phi=512):
    """E(mu, roughness) by a wi-space midpoint rule over the incident hemisphere of D Vis cos (no half-vector substitution): converges to
    ~1e-4 for roughness >= 0.5 at mid angles; the test of `albedo` (a lobe this narrow needs adaptive nodes in wi-space)."""
    a2 = max(roughness * roughness, 0.002) ** 2
    ti = (np.arange(n_theta) + 0.5) / n_theta * (math.pi / 2)
    phi = (np.arange(n_phi) + 0.5) * (2 * math.pi / n_phi)
    ct, st = np.cos(ti)[:, None], np.sin(ti)[:, None]
    v = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
    hx, hy, hz = st * np.cos(phi)[None, :] + v[0], st * np.sin(phi)[None, :], ct + v[2]
    nh = hz / np.sqrt(hx * hx + hy * hy + hz * hz)
    d = a2 / (math.pi * (nh * nh * (a2 - 1) + 1) ** 2)
    vis = 0.5 / np.maximum(ct * math.sqrt(mu * mu * (1 - a2) + a2) + mu * np.sqrt(ct * ct * (1 - a2) + a2), 1e-9)
    return float(np.sum(d * vis * ct * st) * (math.pi / 2 / n_theta) * (2 * math.pi / n_phi))


def check(loss_avg, rows=(0.4, 0.6, 0.8, 1.0), size=SIZE):
    """(roughness, tabulated E_avg, independent E_avg, relative difference) at the check roughnesses."""
    report = []
    for r in rows:
        j = r * (size - 1)
        j0 = min(int(j), size - 2)
        f = j - j0
        tabulated = 1.0 - math.exp(math.log(loss_avg[j0]) * (1 - f) + math.log(loss_avg[j0 + 1]) * f)
        reference = independent_e_avg(r)
        report.append((r, tabulated, reference, abs(tabulated - reference) / reference))
    return report


def render(lost, loss_avg, size=SIZE):
    def block(values, per_line):
        return ",\n".join("    " + ", ".join(f"{v:.8e}f" for v in values[i:i + per_line]) for i in range(0, len(values), per_line))

    flat = [v for row in log_lost_table(lost) for v in row]
    logs = [math.log(max(v, LOG_FLOOR)) for v in loss_avg]
    return f'''#pragma once
// GENERATED by tools/ms_table.py; do not edit.
// F15 closure (--specular-ms): the single-scattering GGX lobe (Fresnel 1) loses L(mu, alpha) = 1 - E(mu, alpha); E_avg = 2 int E mu dmu.
// Both tables store NATURAL LOGARITHMS (L spans four decades; the shader interpolates ln L linearly and exponentiates).
// MsLogLostTable[row * N + column] = ln L: row = roughness j/(N-1), column = mu i/(N-1) (node 0 evaluated at 1e-3), the layout and lookup rule
// of renderer/dfg_table.h; integrated deterministically (tools/ms_table.py `albedo`).  MsLogLossAvgTable[j] = ln(1 - E_avg(roughness j/(N-1)))
// with 1 - E_avg = 2 int L mu dmu over the same interpolation.  Shared by C++ and MSL (shaders/materials/bsdf_energy.h).
#include "renderer/gpu_types.h"
namespace phosphor {{
PHOSPHOR_GPU_CONSTANT u32 MsTableSize = {size};
PHOSPHOR_GPU_CONSTANT float MsLogLossAvgTable[{size}] = {{
{block(logs, 8)}
}};
PHOSPHOR_GPU_CONSTANT float MsLogLostTable[{size * size}] = {{
{block(flat, 8)}
}};
}} // namespace phosphor
'''


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--check", action="store_true", help="compare E_avg with the independent quadrature")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--from-header", action="store_true", help="re-use the lost-energy table of the existing header (no quadrature)")
    args = p.parse_args()
    lost = parse_lost(args.out) if args.from_header or args.check else lost_table(jobs=args.jobs)
    loss_avg = loss_avg_table(lost)
    if args.check:
        worst = 0.0
        for r, tab, ref, rel in check(loss_avg):
            print(f"roughness {r:.2f}: table {tab:.5f} independent {ref:.5f} rel {rel:.4f}")
            worst = max(worst, rel)
        return 0 if worst < 0.005 else 1
    args.out.write_text(render(lost, loss_avg))
    print(f"wrote {args.out} ({SIZE} x {SIZE} lost-energy entries, {SIZE} 1 - E_avg entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
