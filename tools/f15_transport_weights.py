#!/usr/bin/env python3
"""F15.1 (ticket 13/14 closure, transport): CPU model of the per-sample weights of the Fresnel-branch estimator of a transmissive pixel.

Finding (glass-temporal agent): on smooth glass with ior 2.0 a raw transport sample of weight about 32 occurs about once per 92k pixel-samples
(0.5 radiance becomes 16.18) although the branch weights cannot reach 32.  Cause reproduced here: the branch probability p was computed from the
Fresnel factor at n.v, the refraction weight (1 - F(v.h)) from the SAMPLED half vector, so a tail sample of the visible-normal density at a grazing
view (p clamped to 31/32, weight / (1 - p) = x32) keeps (1 - F(v.h)) ~ 1; the reflection branch drew its half vector from the unbounded D(h) n.h
proposal.  "after" evaluates p at the half vector the refraction proposal draws from the same random numbers (a function of u only: unbiased for any p(u))
and draws the reflection from the visible-normal density.

Model (mirrors shaders/reflections.metal, reflection_common.h): one white dielectric interface of index `ior` (f0 = ((ior - 1) / (ior + 1))^2, Schlick), transmission
factor 1, albedo 1, metallic 0; pixels of a sphere (pixel-uniform, n.v = sqrt(1 - u)); GGX alpha; branch probability transmissionSelect(F, (1 - F)) clamped to
[1/32, 31/32].  Radiance weight of one sample: reflect = D V F (n.l) / pdf / p, refract = (1 - F(v.h)) G1(o) / (1 - p) (G1(o) <= 1: counted as 1, an upper bound).
Gate (fixed before measuring): after the fix NO sample may weigh more than 1.1 = the analytic maximum of the unclamped estimator (r + t <= 1, the clamp at the floor
keeps 32 r <= 1.03) over 4 alphas x N pixel-samples; the before model must exceed it (the fixture discriminates).  The mean weight per alpha must agree between
the two estimators (both are unbiased for the same integral) within 3 sigma of the "after" estimate.

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_transport_weights.py [--n 2000000]
Exit codes: 0 gate holds and the control (before) fails it, 1 the after model exceeds the bound, 3 the before model does not exceed it.
"""
import argparse
import sys

import numpy as np

BOUND = 1.1
FLOOR = 1.0 / 32.0


def fresnel(c, f0):
    return f0 + (1.0 - f0) * (1.0 - np.clip(c, 0.0, 1.0)) ** 5


def vndf(v, alpha, u1, u2):
    """Heitz 2018 visible normal sampling, normal = z; v (N, 3)."""
    vh = np.stack([alpha * v[:, 0], alpha * v[:, 1], v[:, 2]], 1)
    vh /= np.linalg.norm(vh, axis=1, keepdims=True)
    lensq = vh[:, 0] ** 2 + vh[:, 1] ** 2
    t1 = np.where(lensq[:, None] > 1e-12, np.stack([-vh[:, 1], vh[:, 0], np.zeros(len(vh))], 1) / np.sqrt(np.maximum(lensq, 1e-12))[:, None], np.array([1.0, 0, 0]))
    t2 = np.cross(vh, t1)
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    p1, p2 = r * np.cos(phi), r * np.sin(phi)
    s = 0.5 * (1 + vh[:, 2])
    p2 = (1 - s) * np.sqrt(np.maximum(0, 1 - p1 ** 2)) + s * p2
    nh = p1[:, None] * t1 + p2[:, None] * t2 + np.sqrt(np.maximum(0, 1 - p1 ** 2 - p2 ** 2))[:, None] * vh
    ne = np.stack([alpha * nh[:, 0], alpha * nh[:, 1], np.maximum(0, nh[:, 2])], 1)
    return ne / np.linalg.norm(ne, axis=1, keepdims=True)


def ggx(c, a):
    a2 = a * a
    return a2 / (np.pi * (c * c * (a2 - 1) + 1) ** 2)


def smith_v(nv, nl, a):
    a2 = a * a
    return 0.5 / np.maximum(nl * np.sqrt(nv * nv * (1 - a2) + a2) + nv * np.sqrt(nl * nl * (1 - a2) + a2), 1e-5)


def select(r, t):
    return np.clip(r / np.maximum(r + t, 1e-30), FLOOR, 1 - FLOOR)


def estimate(n, alpha, ior, fixed, rng):
    f0 = ((ior - 1) / (ior + 1)) ** 2
    nv = np.maximum(np.sqrt(1 - rng.random(n)), 1e-3)
    v = np.stack([np.sqrt(1 - nv ** 2), np.zeros(n), nv], 1)
    u1, u2, ub = rng.random(n), rng.random(n), rng.random(n)
    h = vndf(v, alpha, u1, u2)  # the refraction proposal (and, after the fix, the reflection proposal)
    vh_r = (v * h).sum(1)
    if fixed:
        f = fresnel(vh_r, f0)
        p = select(f, 1 - f)
        hx, hz, vh = h[:, 0], h[:, 2], vh_r
        o_z = 2 * vh * hz - nv
        nl = o_z
        pdf = ggx(hz, alpha) * (2 * nv / (nv + np.sqrt(alpha ** 2 + (1 - alpha ** 2) * nv ** 2))) / (4 * nv)
        ok = (vh > 0) & (nl > 0)
    else:
        f = fresnel(nv, f0)
        p = select(f, 1 - f)
        c = np.sqrt((1 - u1) / (1 + (alpha ** 2 - 1) * u1))
        r = np.sqrt(np.maximum(0, 1 - c * c))
        vh = np.sqrt(1 - nv ** 2) * (r * np.cos(2 * np.pi * u2)) + nv * c
        nl = 2 * vh * c - nv
        pdf = ggx(c, alpha) * c / (4 * np.maximum(vh, 1e-9))
        hz = c
        ok = (vh > 0) & (nl > 0)
    reflect = ub < p
    w_reflect = np.where(ok, ggx(hz, alpha) * smith_v(nv, np.maximum(nl, 1e-6), alpha) * fresnel(vh, f0) * nl / np.maximum(pdf, 1e-30), 0.0) / p
    w_refract = np.where(vh_r > 0, 1.0 - fresnel(vh_r, f0), 0.0) / (1 - p)
    return np.where(reflect, w_reflect, w_refract)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2_000_000)
    parser.add_argument("--ior", type=float, default=2.0)
    args = parser.parse_args(argv)
    worst = {"before": 0.0, "after": 0.0}
    over = {"before": 0, "after": 0}
    agree = True
    for alpha in (0.002, 0.0025, 0.04, 0.25):
        rows = {}
        for name, fixed in (("before", False), ("after", True)):
            w = estimate(args.n, alpha, args.ior, fixed, np.random.default_rng(7))
            rows[name] = w
            worst[name] = max(worst[name], float(w.max()))
            over[name] += int((w > BOUND).sum())
        sigma = rows["after"].std() / np.sqrt(args.n)
        diff = abs(rows["before"].mean() - rows["after"].mean())
        agree &= diff <= 3 * sigma + 3 * rows["before"].std() / np.sqrt(args.n)
        print(f"alpha {alpha}: before max {rows['before'].max():.3f} mean {rows['before'].mean():.5f} | after max {rows['after'].max():.3f} mean {rows['after'].mean():.5f}"
              f" (sigma {sigma:.5f})")
    print(f"samples above {BOUND}: before {over['before']}, after {over['after']}; means agree: {agree}")
    if over["after"] or not agree:
        return 1
    if not over["before"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
