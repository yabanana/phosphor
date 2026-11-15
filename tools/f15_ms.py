#!/usr/bin/env python3
"""F15 closure: independent reference of the multiple-scattering energy compensation of the GGX specular lobes (--specular-ms).

Written from the papers, not from shaders/materials/bsdf_energy.h:
  * Kulla and Conty, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH 2017 course notes: the energy a single-scattering
    microfacet lobe loses, L(mu) = 1 - E(mu), is returned by an extra reciprocal lobe
        f_ms(o, i) = L(mu_o) L(mu_i) / (pi (1 - E_avg))          (E_avg = 2 int E mu dmu, the cosine-weighted mean)
    coloured by the Fresnel of the extra bounces: F_ms = F_avg^2 E_avg / (1 - F_avg (1 - E_avg)), F_avg the cosine-weighted mean of the
    Schlick Fresnel, F_avg = f0 + (f90 - f0) / 21 (2 int mu (1 - mu)^5 dmu = 1/21).  E is the directional albedo of D Vis cos WITHOUT
    Fresnel.
  * Fdez-Aguera, "A Multiple-Scattering Microfacet Model for Real-Time Image-Based Lighting", JCGT 8(1), 2019: the same split for an
    environment; in a CONSTANT environment of radiance L the integral of f_ms cos over the hemisphere is exactly F_ms L(mu_o) L, because
    2 int L mu dmu = 1 - E_avg.

Two models, like every other lobe of tools/f15_reference.py:
  * literal ("kulla-conty"): E(mu) comes from a deterministic quadrature of D Vis cos (tools/ms_table.py `albedo`: half-vector
    substitution with the exact azimuth interval, converged to ~1e-5 at mid angles; tools/test_ms_table.py checks it against a wi-space
    midpoint quadrature that shares nothing with it) evaluated here on a fine mu grid -- NOT the engine's split-sum DFG table (a 16384-sample
    Monte Carlo of ~1e-3 accuracy, too coarse for the lost energy L) and NOT the engine's lost-energy table (which is the same quadrature
    sampled on a 64 x 64 grid: the literal-vs-engine comparison therefore measures the table interpolation and the shader);
  * declared ("table"): L(mu, r) and 1 - E_avg(r) of the engine tables src/renderer/ms_table.h (bilinear / linear in the logarithm, the engine's rule),
    parsed here from the header.
Negative controls: "off" (no compensation) and "double" (F_ms doubled: gains energy).
"""
import pathlib
import re

import f15_reference as ref
import ms_table

np = ref.np
MS_HEADER = ref.REPO / "src" / "renderer" / "ms_table.h"
_MS = {}
_EXACT = {}
MU_NODES = np.concatenate([np.linspace(1e-3, 0.1, 37)[:-1], np.linspace(0.1, 1.0, 181)])  # dense near grazing incidence


def _block(text, name):
    start = text.index(f"float {name}[")
    body = text[text.index("{", start) + 1:text.index("};", start)]
    return np.asarray([float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)], dtype=np.float64)


def load_ms_tables(path=MS_HEADER):
    """(ln(1 - E_avg) per roughness row, ln L table [roughness row, mu column]) of the engine header (both stored as logarithms)."""
    key = str(path)
    if key not in _MS:
        text = pathlib.Path(path).read_text()
        size = int(re.search(r"MsTableSize\s*=\s*(\d+)\s*;", text).group(1))
        loss, lost = _block(text, "MsLogLossAvgTable"), _block(text, "MsLogLostTable")
        if len(loss) != size or len(lost) != size * size:
            raise ValueError(f"{path}: expected {size} and {size * size} table values, parsed {len(loss)} and {len(lost)}")
        _MS[key] = (loss, lost.reshape(size, size))
    return _MS[key]


def table_loss_avg(roughness, path=MS_HEADER):
    """1 - E_avg: the engine table is linear in ln(1 - E_avg) between roughness nodes (clamped to [0, 1])."""
    loss, _ = load_ms_tables(path)
    y = np.clip(np.asarray(roughness, dtype=np.float64), 0.0, 1.0) * (len(loss) - 1)
    y0 = np.minimum(np.floor(y).astype(np.int64), len(loss) - 2)
    return np.exp(loss[y0] * (1 - (y - y0)) + loss[y0 + 1] * (y - y0))


def table_lost(mu, roughness, path=MS_HEADER):
    """L(mu, roughness) of the engine table: the bilinear rule of `dfg_lookup` (node = value * (size - 1), cell clamped to size - 2) on ln L,
    exponentiated."""
    _, table = load_ms_tables(path)
    size = table.shape[0]
    n = size - 1
    x = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0) * n
    y = np.clip(np.asarray(roughness, dtype=np.float64), 0.0, 1.0) * n
    x0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), size - 2)
    fx, fy = x - x0, y - y0
    return np.exp((table[y0, x0] * (1 - fx) + table[y0, x0 + 1] * fx) * (1 - fy) +
                  (table[y0 + 1, x0] * (1 - fx) + table[y0 + 1, x0 + 1] * fx) * fy)


def exact_lost(roughness, model=ref.APPENDIX_B):
    """(mu grid, L(mu) = 1 - E on it, 1 - E_avg) of one roughness from the quadrature (trapezoid over the grid for the average)."""
    alpha_root = float(np.sqrt(ref.effective_alpha(roughness, model)))  # the roughness whose alpha rule gives this alpha
    key = (round(alpha_root, 12), model["alpha"])
    if key not in _EXACT:
        r = float(ref.clamp_roughness(roughness))
        lost = np.array([max(0.0, 1.0 - ms_table.albedo(float(mu), r, 8192, 128)) for mu in MU_NODES])
        loss = 2.0 * float(np.sum(0.5 * (lost[1:] * MU_NODES[1:] + lost[:-1] * MU_NODES[:-1]) * np.diff(MU_NODES)))
        _EXACT[key] = (MU_NODES, lost, loss)
    return _EXACT[key]


def lobe_endpoints(base, metallic, dielectric=None):
    """(f0, f90) per pixel and channel of the specular lobe: mix(dielectric, metal) with metal = (base colour, 1)."""
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    n = len(m)
    if dielectric is None:
        f0d, f90d = np.full((n, 3), ref.DIELECTRIC_F0), np.ones((n, 3))
    else:
        f0d, _ = ref.dielectric_fresnel(dielectric["f0"], dielectric["weight"], np.zeros((n, 1)), dielectric["variant"])
        f90d, _ = ref.dielectric_fresnel(dielectric["f0"], dielectric["weight"], np.ones((n, 1)), dielectric["variant"])
    return (1 - m) * f0d + m * base, (1 - m) * f90d + m


def fms(f0, f90, loss_avg, kind="kulla-conty"):
    """F_ms per channel (n, 3) for the loss 1 - E_avg (n,)."""
    favg = f0 + (f90 - f0) / 21.0
    loss = np.asarray(loss_avg, dtype=np.float64)[:, None]
    value = favg * favg * (1.0 - loss) / (1.0 - favg * loss)
    return 2.0 * value if kind == "double" else value


def _per_roughness(roughness, function):
    """Apply function(selection, roughness value) over the unique roughnesses of a per-pixel array."""
    rough = np.asarray(roughness, dtype=np.float64)
    for r in np.unique(rough):
        function(np.flatnonzero(rough == r), float(r))


def lobe_value(base, metallic, roughness, n_dot_v, n_dot_l, model, dielectric=None, kind="kulla-conty"):
    """f_ms (n, 3), the literal model's extra lobe (without the cosine); 0 where n_dot_l <= 0."""
    out = np.zeros((len(n_dot_v), 3))
    f0, f90 = lobe_endpoints(base, metallic, dielectric)
    mu_v = np.clip(n_dot_v, ref.MIN_NDOTV, 1.0)
    mu_l = np.clip(n_dot_l, ref.MIN_NDOTV, 1.0)

    def fill(sel, r):
        mu, lost, loss = exact_lost(r, model)
        scale = np.interp(mu_v[sel], mu, lost) * np.interp(mu_l[sel], mu, lost) / (np.pi * loss)
        out[sel] = fms(f0[sel], f90[sel], np.full(len(sel), loss), kind) * scale[:, None]

    _per_roughness(roughness, fill)
    return np.where((np.asarray(n_dot_l) > 0)[:, None], out, 0.0)


def environment_value(base, metallic, roughness, n_dot_v, radiance, model, dielectric=None, kind="kulla-conty"):
    """Radiance (n, 3) of the extra lobe in a constant environment, by integrating f_ms cos with the reference's own L(mu)
    (a midpoint quadrature over cos(theta_i) of the very lobe, NOT the closed form, so that the identity
    2 int L mu dmu = 1 - E_avg is exercised rather than assumed)."""
    out = np.zeros((len(n_dot_v), 3))
    f0, f90 = lobe_endpoints(base, metallic, dielectric)
    mu_v = np.clip(n_dot_v, ref.MIN_NDOTV, 1.0)
    nodes = (np.arange(8192) + 0.5) / 8192.0

    def fill(sel, r):
        mu, lost, loss = exact_lost(r, model)
        # integral over the hemisphere of f_ms cos = 2 int L(mu_o) L(mu_i) mu_i dmu_i / (1 - E_avg) * F_ms
        integral = 2.0 * float(np.mean(np.interp(nodes, mu, lost) * nodes)) / loss
        out[sel] = fms(f0[sel], f90[sel], np.full(len(sel), loss), kind) * (np.interp(mu_v[sel], mu, lost) * integral)[:, None]

    _per_roughness(roughness, fill)
    return out * radiance


def declared_environment_value(base, metallic, roughness, n_dot_v, radiance, kind="table", dielectric=None):
    """The declared model: F_ms L(mu_o) Lrad with L and 1 - E_avg of the engine tables (src/renderer/ms_table.h)."""
    rough = ref.clamp_roughness(roughness)
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), ref.MIN_NDOTV)
    f0, f90 = lobe_endpoints(base, metallic, dielectric)
    return fms(f0, f90, table_loss_avg(rough), "double" if kind == "double" else "table") * table_lost(mu, rough)[:, None] * radiance


def isotropic(aniso, n):
    """(n,) mask of the pixels whose base lobe carries the multiple-scattering lobe: an anisotropic lobe carries none (ADR-0007: an isotropic table cannot
    follow its azimuth dependence; the effective isotropic roughness gained +3.3 % in the white furnace of a pure metal at strength 1)."""
    if aniso is None:
        return np.ones(n, dtype=bool)
    return ~(np.asarray(aniso["strength"], dtype=np.float64) > 0.0)


def effective_roughness(roughness, aniso, model):
    """Kept for the callers of the first version: the roughness of the lost energy (the pixel's own; anisotropic pixels are masked by `isotropic`)."""
    return roughness


def coat_lobe_value(coat, c, n_dot_v, n_dot_l, model, kind="kulla-conty"):
    """c f_ms of the coat's own dielectric lobe (f0 0.04, f90 1, the coat roughness; coat normal = shading normal), per pixel, no cosine."""
    n = len(c)
    value = lobe_value(np.ones((n, 3)), np.zeros(n), coat["roughness"], n_dot_v, n_dot_l, model, None, kind)
    return np.max(value, axis=1) * c


def coat_environment_value(coat, c, n_dot_v, radiance, model, kind="kulla-conty"):
    """Radiance (n, 3) of the coat's own lobe in a constant environment, c F_ms L(mu) L (exact model)."""
    n = len(c)
    return environment_value(np.ones((n, 3)), np.zeros(n), coat["roughness"], n_dot_v, radiance, model, None, kind) * c[:, None]


def declared_coat_environment_value(coat, c, n_dot_v, kind="table"):
    """Unit-radiance (n, 3) coat lobe of the declared model (engine tables)."""
    n = len(c)
    return declared_environment_value(np.ones((n, 3)), np.zeros(n), coat["roughness"], n_dot_v, 1.0, kind, None) * c[:, None]


def declared_complement(f0, f90, roughness, n_dot_v, kind="table"):
    """F_ms L(mu) (n, 3) of a dielectric layer with Fresnel endpoints (f0, f90) in the declared model (engine tables): the energy its multiple-scattering
    lobe returns, which the diffuse complement gives up (1 - E_d - F_ms L, ADR-0007: 1 - E_d already refilled the single-scattering loss)."""
    rough = ref.clamp_roughness(roughness)
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), ref.MIN_NDOTV)
    n = len(mu)
    f0 = np.broadcast_to(np.asarray(f0, dtype=np.float64), (n, 3))
    f90 = np.broadcast_to(np.asarray(f90, dtype=np.float64).reshape(n, -1) if np.ndim(f90) else np.full((n, 1), float(f90)), (n, 3))
    return fms(f0, f90, table_loss_avg(rough), "double" if kind == "double" else "table") * table_lost(mu, rough)[:, None]
