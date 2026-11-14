#!/usr/bin/env python3
"""F15.1 (ticket 18): the radial diffusion profile LUT of the volumetric subsurface mode.

A searchlight (normal incidence, delta beam) on a semi-infinite homogeneous medium (sigma_t 1, single-scatter albedo alpha, HG g) behind
the SAME boundary as the ground truth of tools/f15_subsurface_reference.py (GGX rough dielectric, ior 1.5, VNDF microfacet sampling, the
alpha_b of the albedo table): a vectorised numpy random walk (analog absorption, G2/G1 microfacet weights, Fresnel-chosen reflection or
refraction, exit on the first refraction out).  The exit flux is binned radially in x = r / mfp (log-spaced edges X_MIN..X_MAX, BINS bins) and
normalised to unit integral; the LUT stores the CUMULATIVE flux at the bin edges (the gather samples radii by this CDF and weights by the
density P(x) = dCDF / (pi (x1^2 - x0^2))).  The medium of a colour C is the boundary-aware inversion of tools/f15_subsurface_albedo.json (the
single-scatter albedo whose OBSERVED albedo is C), per g.  Beyond X_MAX the profile is truncated (the pieces are a few tens of mfp).

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_subsurface_profile.py generate [--photons N] [--out FILE]
  uv run ... python -I tools/f15_subsurface_profile.py check      (albedo cross-check against the Mitsuba table)

The data is tools/f15_subsurface_profile.json; src/renderer/subsurface_profile_lut.h is generated from it (command `header`).
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
LUT_PATH = TOOLS / "f15_subsurface_profile.json"
HEADER_PATH = REPO / "src" / "renderer" / "subsurface_profile_lut.h"
ALBEDO_PATH = TOOLS / "f15_subsurface_albedo.json"
IOR = 1.5
ALPHA_B = 0.1225                      # roughness 0.35 squared: the only boundary of the albedo table
X_MIN, X_MAX, BINS = 0.02, 64.0, 64   # edges: BINS + 1 log-spaced values; the first bin starts at 0 (CDF 0 at X_MIN is the first interval)
C_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.78, 0.85, 0.87]
G_GRID = [0.0, 0.3]
MAX_STEPS = 60000


def edges():
    return np.exp(np.linspace(math.log(X_MIN), math.log(X_MAX), BINS + 1))


# --- rough dielectric interface ---------------------------------------------------------------------------------------------

def _lambda(w, alpha):
    c2 = np.maximum(w[:, 2] ** 2, 1e-12)
    return 0.5 * (-1.0 + np.sqrt(1.0 + alpha * alpha * (1.0 - c2) / c2))


def interface(rng, wi, eta_i, eta_t, alpha):
    """GGX rough dielectric, VNDF sampling.  `wi` (n, 3) unit vectors with the sign of z telling the side (z > 0 = the side of index eta_i
    seen from +z).  Returns (wo, weight, reflected, valid): weight = G2 / G1(wi); valid False = the sampled direction left on the wrong side."""
    n = len(wi)
    flip = np.where(wi[:, 2] < 0.0, -1.0, 1.0)
    w = wi * flip[:, None]
    eta = eta_t / eta_i
    vh = np.stack([alpha * w[:, 0], alpha * w[:, 1], w[:, 2]], axis=1)
    vh /= np.linalg.norm(vh, axis=1, keepdims=True)
    lensq = vh[:, 0] ** 2 + vh[:, 1] ** 2
    ok = lensq > 1e-12
    inv = 1.0 / np.sqrt(np.where(ok, lensq, 1.0))
    t1v = np.stack([np.where(ok, -vh[:, 1] * inv, 1.0), np.where(ok, vh[:, 0] * inv, 0.0), np.zeros(n)], axis=1)
    t2v = np.cross(vh, t1v)
    u1, u2 = rng.random(n), rng.random(n)
    r = np.sqrt(u1)
    phi = 2.0 * math.pi * u2
    a, b = r * np.cos(phi), r * np.sin(phi)
    s = 0.5 * (1.0 + vh[:, 2])
    b = (1.0 - s) * np.sqrt(np.maximum(1.0 - a * a, 0.0)) + s * b
    nh = a[:, None] * t1v + b[:, None] * t2v + np.sqrt(np.maximum(1.0 - a * a - b * b, 0.0))[:, None] * vh
    m = np.stack([alpha * nh[:, 0], alpha * nh[:, 1], np.maximum(nh[:, 2], 0.0)], axis=1)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    cosi = np.clip(np.sum(w * m, axis=1), 1e-9, 1.0)
    sin2t = (1.0 - cosi ** 2) / (eta * eta)
    tir = sin2t >= 1.0
    cost = np.sqrt(np.maximum(1.0 - sin2t, 0.0))
    rs = (cosi - eta * cost) / (cosi + eta * cost)
    rp = (eta * cosi - cost) / (eta * cosi + cost)
    fres = np.where(tir, 1.0, 0.5 * (rs * rs + rp * rp))
    reflected = rng.random(n) < fres
    wo_r = 2.0 * np.sum(w * m, axis=1)[:, None] * m - w
    wo_t = -w / eta + ((cosi / eta - cost))[:, None] * m
    wo = np.where(reflected[:, None], wo_r, wo_t)
    valid = np.where(reflected, wo[:, 2] > 0.0, wo[:, 2] < 0.0)
    g2 = 1.0 / (1.0 + _lambda(w, alpha) + _lambda(wo, alpha))
    g1 = 1.0 / (1.0 + _lambda(w, alpha))
    weight = g2 / g1
    wo = wo * flip[:, None]
    wo /= np.linalg.norm(wo, axis=1, keepdims=True)
    return wo, weight, reflected, valid


def hg_scatter(rng, d, g):
    n = len(d)
    u = rng.random(n)
    cost = np.where(abs(g) < 1e-6, 1.0 - 2.0 * u, (1.0 + g * g - ((1.0 - g * g) / (1.0 - g + 2.0 * g * u)) ** 2) / (2.0 * g if g != 0 else 1.0))
    cost = np.clip(cost, -1.0, 1.0)
    sint = np.sqrt(np.maximum(1.0 - cost * cost, 0.0))
    phi = 2.0 * math.pi * rng.random(n)
    helper = np.where(np.abs(d[:, [0]]) > 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(d, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(d, t1)
    out = cost[:, None] * d + sint[:, None] * (np.cos(phi)[:, None] * t1 + np.sin(phi)[:, None] * t2)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def walk(alpha_ss, g, photons, seed, alpha_b=ALPHA_B, ior=IOR):
    """Radial exit-flux histogram (BINS bins) and the total exited flux per incident photon of one searchlight."""
    rng = np.random.default_rng(seed)
    e = edges()
    hist = np.zeros(BINS)
    # entry
    wi = np.tile(np.array([[0.0, 0.0, 1.0]]), (photons, 1))
    wo, wgt, refl, valid = interface(rng, wi, 1.0, ior, alpha_b)
    keep = valid & ~refl
    d = wo[keep]
    w = wgt[keep]
    pos = np.zeros((len(d), 3))
    total_exit = 0.0
    for _ in range(MAX_STEPS):
        if len(d) == 0:
            break
        s = -np.log(1.0 - rng.random(len(d)))
        new = pos + s[:, None] * d
        hit = new[:, 2] > 0.0
        # boundary hits: move to z = 0 and interact from inside
        if hit.any():
            # pos z is <= 0 and d z > 0: the distance to z = 0 is (0 - z) / dz
            t = (0.0 - pos[hit, 2]) / np.where(d[hit, 2] < 1e-12, 1e-12, d[hit, 2])
            bpos = pos[hit] + t[:, None] * d[hit]
            bpos[:, 2] = 0.0
            wo2, wg2, rf2, va2 = interface(rng, -d[hit], ior, 1.0, alpha_b)
            exits = va2 & ~rf2
            r = np.hypot(bpos[:, 0], bpos[:, 1])
            fw = w[hit] * wg2
            sel = exits & (r < X_MAX)
            idx = np.clip(np.searchsorted(e, r[sel]) - 1, 0, BINS - 1)
            np.add.at(hist, idx, fw[sel])
            total_exit += float(fw[exits].sum())
            stay = va2 & rf2
            newd = d.copy()
            newd[hit] = wo2
            neww = w.copy()
            neww[hit] = fw
            newp = new.copy()
            newp[hit] = bpos
            # photons that hit survive only when reflected inwards
            survive = np.ones(len(d), dtype=bool)
            survive[np.flatnonzero(hit)] = stay
            d, w, pos = newd[survive], neww[survive], newp[survive]
            hit = hit[survive]
            # reflected photons restart from the boundary without scattering: handled next iteration as a flight from z = 0
            scatter = ~hit
        else:
            pos = new
            scatter = np.ones(len(d), dtype=bool)
        # scatter / absorb the photons that flew inside
        absorbed = scatter & (rng.random(len(d)) > alpha_ss)
        r2 = np.hypot(pos[:, 0], pos[:, 1])
        dead = absorbed | (r2 > 2.0 * X_MAX)
        sc = scatter & ~dead
        if sc.any():
            d[sc] = hg_scatter(rng, d[sc], g)
        d, w, pos = d[~dead], w[~dead], pos[~dead]
    return hist, total_exit / photons


def profile_cdf(alpha_ss, g, photons, seed):
    """(cumulative unit-integral flux at the BINS + 1 edges, total exit flux per photon)."""
    hist, total = walk(alpha_ss, g, photons, seed)
    cum = np.concatenate([[0.0], np.cumsum(hist)])
    return cum / max(cum[-1], 1e-30), total


# --- the observed-albedo inversion (shared with the ground truth) -------------------------------------------------------------

def inverse_albedo(c, g):
    sys.path.insert(0, str(TOOLS))
    import f15_subsurface_reference as ref
    curve = ref.curve_for(float(g), ALPHA_B)
    return float(ref.invert_observed_albedo(float(c), curve)[0])


def generate(photons, out, log=print):
    rows = []
    for gi, g in enumerate(G_GRID):
        for ci, c in enumerate(C_GRID):
            a = inverse_albedo(c, g)
            cdf, total = profile_cdf(a, g, photons, 1000 * gi + ci + 1)
            log(f"g {g} C {c} alpha_ss {a:.6f} exit flux {total:.4f}")
            rows.append({"g": g, "C": c, "alpha_ss": a, "exit_flux": total, "cdf": [float(v) for v in cdf]})
    data = {"schema": 1, "tool": "tools/f15_subsurface_profile.py", "boundary": {"ior": IOR, "alpha_b": ALPHA_B, "model": "ggx rough dielectric, VNDF"},
            "x_min": X_MIN, "x_max": X_MAX, "bins": BINS, "photons": photons, "g_grid": G_GRID, "c_grid": C_GRID, "profiles": rows}
    pathlib.Path(out).write_text(json.dumps(data, indent=1) + "\n")
    return data


def header_text(data):
    """C++/MSL header: the CDF table [g][C][BINS + 1] as a flat float array (index (gi * NC + ci) * (BINS + 1) + k)."""
    values = [v for row in data["profiles"] for v in row["cdf"]]
    lines = ["// GENERATED by tools/f15_subsurface_profile.py (header); do not edit.  Radial exit-flux CDF of a searchlight on a semi-infinite medium",
             "// behind the ground truth's boundary (ior 1.5, GGX alpha_b %.4f), per HG g and observed colour C." % data["boundary"]["alpha_b"],
             "#pragma once", "#include \"renderer/gpu_types.h\"", "namespace phosphor {",
             f"PHOSPHOR_GPU_CONSTANT u32 SSS_PROFILE_BINS = {data['bins']}u;",
             f"PHOSPHOR_GPU_CONSTANT u32 SSS_PROFILE_COLORS = {len(data['c_grid'])}u;",
             f"PHOSPHOR_GPU_CONSTANT u32 SSS_PROFILE_GS = {len(data['g_grid'])}u;",
             f"PHOSPHOR_GPU_CONSTANT float SSS_PROFILE_X_MIN = {data['x_min']}f;",
             f"PHOSPHOR_GPU_CONSTANT float SSS_PROFILE_X_MAX = {data['x_max']}f;",
             "// colour grid and g grid", "#define SSS_PROFILE_C_GRID " + ", ".join(f"{c}f" for c in data["c_grid"]),
             "#define SSS_PROFILE_G_GRID " + ", ".join(f"{g}f" for g in data["g_grid"]),
             "#define SSS_PROFILE_CDF_VALUES " + ", ".join(f"{v:.7g}f" for v in values), "} // namespace phosphor", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["generate", "header", "check"])
    p.add_argument("--photons", type=int, default=200000)
    p.add_argument("--out", default=str(LUT_PATH))
    args = p.parse_args(argv)
    if args.command == "generate":
        generate(args.photons, args.out)
    elif args.command == "header":
        HEADER_PATH.write_text(header_text(json.loads(pathlib.Path(args.out).read_text())))
    else:
        table = json.loads(ALBEDO_PATH.read_text())
        curve = [c for c in table["curves"] if c["g"] == 0.0 and abs(c["alpha_b"] - ALPHA_B) < 1e-4 and c["ior"] == IOR][0]
        for point in curve["points"]:
            a = point["albedo"]
            if a in (0.5, 0.8, 0.95, 0.99, 0.999):
                _, total = profile_cdf(a, 0.0, args.photons, 7)
                print(f"alpha_ss {a}: walk exit flux {total:.4f}, table A(0 deg) {point['A_angle']['0']:.4f}, A(diffuse) {point['A']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
