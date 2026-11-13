#!/usr/bin/env python3
"""F15.3 independent reference of the glTF 2.0 Appendix B base model.

Written from the specification (Khronos glTF 2.0, Appendix B "BRDF
Implementation"), not translated from the MSL.  It consumes the snapshot of
`exportOfflineReference` (geometry, world matrices, camera, sun, environment,
material factors) of the material-validation bench (`--material-scenario`) and
never reads a renderer image to choose rays, samples or masks.  Mitsuba 3 is
used only to cast pixel-centre rays on the snapshot meshes (hit position,
interpolated shading normal, shape); the BSDF and every integral are this
file's own.

Model (Appendix B):
  dielectric = mix(diffuse(baseColor), specular, F(f0 = 0.04))        [ior 1.5]
  metal      = specular * F(f0 = baseColor)
  material   = mix(dielectric, metal, metallic)
  specular   = D_GGX * Vis_SmithCorrelated, alpha = roughness^2
  F          = Schlick, f0 + (1 - f0) (1 - |V.H|)^5
  diffuse    = baseColor / pi

KHR_materials_ior + KHR_materials_specular (F15.1, ticket 05) are implemented from the specification
text of KHR_materials_specular (research/khr-materials.md section 9), in its UNFLATTENED form -- the engine
shader implements the flattened one, so the two are independent:
  f0   = min(((1 - ior) / (1 + ior))^2 * specularColor, 1)       (ior = 0: the factor is 1, effective IOR +inf)
  fr   = f0 + (1 - f0) (1 - |V.H|)^5
  dielectric = (1 - specular * max(fr)) * diffuse + specular * fr * specular_brdf
  metal      = unchanged; material = mix(dielectric, metal, metallic)
The extension values come from the snapshot's `material_extensions` records (the raw GPU words of record 0,
word offsets of src/renderer/gpu_types.h GPUMaterialExt0).  Environment: the declared split-sum model with
E_d = f0_scaled * A + f90 * B (f0_scaled = f0 * specular, f90 = specular).

KHR_materials_clearcoat (F15.3, ticket 09) is implemented from the ratified text (research/khr-materials.md
section 2), again in the layering form the shader does not use.  With c = clearcoatFactor, the coat normal Nc
(the shading normal: the fixtures carry no coat normal texture) and a_c = max(clearcoatRoughness^2, 0.002):
  F_c    = 0.04 + 0.96 (1 - |V.Nc|)^5          (coat IOR fixed at 1.5; V.Nc, not V.H, the specification's
                                                 deliberate simplification)
  f      = (1 - c F_c) f_material + c F_c D(Nc.H, a_c) Vis(Nc.V, Nc.L, a_c)       (coat term: no Fresnel inside)
  L_out  = (1 - c F_c) L_material + c F_c L_coat,  emission would be darkened by the same 1 - c F_c (none here)
  - the base keeps its own cosine max(N.L, 0), the coat term its own max(Nc.L, 0);
  - roughness clamps are the base ones: clearcoatRoughness in [0.04, 1], a_c = max(r_c^2, 0.002), Nc.V >= 1e-4;
  - "f_material" is the full extended material (ior/specular dielectric or metal), unchanged underneath;
  - sun disk: the coat lobe follows the declared disk rule with ITS OWN alpha and reflection axis (own sample
    count); the quadrature mask is the union of both lobes' masks;
  - exact environment: (1 - c F_c(mu)) * L_material + c F_c(mu) * E_coat(mu, a_c) * L, E_coat the directional
    albedo of D * Vis (Fresnel 1, the quadrature of `_spec_integrals`, A + B);
  - declared environment (split sum, DFG table): (1 - c F_c(mu)) * declared_material + c F_c(mu) *
    clamp(A + B, 0, 1) * L at the coat roughness (A, B of `dfg_lookup`);
  - point lights: the BSDF above at the light direction.

KHR_materials_sheen (F15.3, ticket 10) is implemented from the ratified text and its non-normative Charlie notes
(research/khr-materials.md section 3); the formulas, the albedo-scaling forms, the clamp of E and the sun-disk rule are
documented at the "KHR_materials_sheen" section below.  The Phosphor E table (`tools/sheen_table.py`) is generated and
checked there; this file integrates E itself (incident hemisphere) for the literal model and reads the engine header
only for the declared environment model.  Clearcoat sits above the sheen; textured sheen is refused.

Declared engine conventions (every item is part of the JSON report):
- roughness clamped to [0.04, 1], alpha = max(roughness^2, 0.002), N.V >= 1e-4.
- Sun: a disk of uniform radiance and angular radius r (the snapshot's
  sun_angular_radius).  Diffuse is evaluated at the disk centre; the specular
  term is the irradiance-normalised mean of f_s cos over the disk with the
  engine's declared rule (`sunDiskSpecular` in shaders/material_shading.h): a
  point evaluation when the lobe is wide, else a deterministic uniform
  solid-angle golden-angle spiral of 32 samples (128 when alpha < 2r) at the
  pixel centre.  `narrow` pixels are reported as the quadrature mask.
- Environment: one constant isotropic radiance L (the snapshot's sky).  The
  outgoing radiance is the exact numerical quadrature over the hemisphere of
  BSDF * cos * L, not a split-sum fit (half-vector parametrisation with the
  GGX CDF, midpoint rule; the table is a function of roughness and N.V only).
- Camera: one ray through each pixel centre, no reconstruction filter.
- Shadows: none for the sun (the scenarios declare `--shadows off`), no inter-reflection.
- Point lights (F15.1, ticket 07): white delta lights of the snapshot (type 1) with the engine's declared
  falloff, incident radiance = colour * intensity * window^2 / max(d^2, 1e-4) with
  window = clamp(1 - (d / max(range, 1e-3))^4, 0, 1); the BSDF is evaluated at the light direction.
  Visibility (`--visibility rays`, the DI runs: DI traces local-light visibility with RT) comes from Mitsuba
  shadow rays from the hit offset 1e-4 along the shading normal; `--visibility none` (the legacy resolve,
  whose local lights are unshadowed) skips it.  The `lit` mask holds N.L > 0 and, with rays, the visible
  pixels, so the runner's terminator band also excludes shadow boundaries.

Second, independent model (`--declared`): the DECLARED environment model of the
engine (ADR-0006), re-implemented here from its documentation, not translated
from the shader.  With E_d = 0.04 A + B the directional albedo of the
dielectric specular layer, the environment radiance of one pixel is
  L_env * ((1 - metallic) * baseColor * (1 - E_d) + clamp(f0 A + B, 0, 1))
with f0 = mix(0.04, baseColor, metallic), (A, B) the DFG table values of
`src/renderer/dfg_table.h` (parsed from that header; the bilinear rule of
`dfgTableLookup` is re-implemented in `dfg_lookup`), roughness clamped to
[0.04, 1] and N.V clamped at 1e-4.  It is a split-sum model, hence it differs
from the exact quadrature above by a few percent (the diffuse weight); it is the
gate of the deterministic resolve scenarios.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import np_module, scene_digest, snapshot, write_pfm  # noqa: E402
from f15_common import sha256_file  # noqa: E402

np = np_module()

MIN_ROUGHNESS, MIN_ALPHA, MIN_NDOTV = 0.04, 0.002, 1e-4
DIELECTRIC_IOR = 1.5
DIELECTRIC_F0 = ((1 - DIELECTRIC_IOR) / (1 + DIELECTRIC_IOR)) ** 2  # 0.04
GOLDEN_ANGLE = 2.39996323  # the engine's constant (float32 literal)
INVALID = 0xFFFFFFFF

MODEL = {
    "base": "glTF 2.0 Appendix B: mix(mix(diffuse, specular, F(0.04)), specular * F(baseColor), metallic)",
    "specular": "GGX D, height-correlated Smith visibility, alpha = max(roughness^2, 0.002), roughness in [0.04, 1]",
    "fresnel": "Schlick on |V.H|",
    "sun": "disk centre for diffuse; specular = mean over a point or a deterministic 32/128-sample golden-angle "
           "uniform-solid-angle spiral (engine sunDiskSpecular rule); narrow pixels form the quadrature mask",
    "environment": "exact hemisphere quadrature of BSDF * cos * L_env (constant isotropic), not a split-sum fit",
    "camera": "pixel-centre rays, no reconstruction filter",
    "clearcoat": "KHR_materials_clearcoat: F_c = 0.04 + 0.96 (1 - |V.Nc|)^5, f = (1 - c F_c) f_material + c F_c D Vis "
                 "(a_c = max(clearcoatRoughness^2, 0.002), roughness in [0.04, 1], Nc = shading normal); sun coat lobe "
                 "with its own alpha, axis and sample count (mask = union); exact environment (1 - c F_c) L + c F_c E_coat "
                 "L_env",
    "sheen": "KHR_materials_sheen: f = S D_charlie V_charlie + f_material min(1 - max3(S) E(N.V), 1 - max3(S) E(N.L)) "
             "(direct light; environment: 1 - max3(S) E(N.V), the notes' cheap form), alpha_g = roughness^2, roughness "
             "in [0.07, 1], E = Fresnel-free Charlie albedo clamped to 1 (own incident-hemisphere quadrature); sun: "
             "sheen lobe and scaling point-evaluated at the disk centre; coat above the sheen; exact environment "
             "(base (1 - max3(S) E) + S E) L",
    "shadows": "none",
    "declared_environment": "(1 - metallic) * baseColor * (1 - (0.04 A + B)) + clamp(f0 A + B, 0, 1), (A, B) from the "
                            "engine DFG table (bilinear), f0 = mix(0.04, baseColor, metallic), roughness in [0.04, 1]; "
                            "clearcoat: (1 - c F_c(mu)) * that + c F_c(mu) * clamp(A_c + B_c, 0, 1), (A_c, B_c) at the "
                            "coat roughness; sheen: material (1 - max3(S) E_t) + S E_t with E_t from the engine sheen table "
                            "(src/renderer/sheen_table.h, bilinear) under the coat",
}

# The declared model plus the deliberately wrong variants used as negative controls.
APPENDIX_B = {"alpha": "squared", "fresnel": "schlick", "weighting": "appendix-b"}
# F15.1: the dielectric of KHR_materials_ior/specular scenes.  "base" = Appendix B (F0 0.04, no extension).
SPECULAR_EXT = {**APPENDIX_B, "dielectric": "extension"}
CONTROLS = {
    "alpha-roughness": {**APPENDIX_B, "alpha": "linear"},          # alpha = roughness instead of roughness^2
    "fresnel-off": {**APPENDIX_B, "fresnel": "off"},               # F = f0 for every angle
    "diffuse-unweighted": {**APPENDIX_B, "weighting": "unweighted-diffuse"},  # diffuse not scaled by (1 - F)
    "engine-f0-blend": {**APPENDIX_B, "weighting": "engine-blend"},  # F of the metallic-blended f0 weights diffuse
    # F15.1 negative controls of the specular/ior grids (direct light).
    "specular-ignored": {**APPENDIX_B, "dielectric": "ignored"},          # the extension is not applied (f0 0.04)
    "f90-one": {**APPENDIX_B, "dielectric": "f90-one"},                    # f90 forced to 1 instead of the weight
    "specular-weight-not-on-f0": {**APPENDIX_B, "dielectric": "f0-unweighted"},  # f0 keeps no specular weight
    # F15.3 (ticket 09) negative controls of the clearcoat grids; they sit on the ior/specular model (the
    # fixtures combine both) and only change the coat.
    "coat-ignored": {**SPECULAR_EXT, "coat": "ignored"},                  # c = 0: no coat at all
    "coat-f0-from-ior": {**SPECULAR_EXT, "coat": "f0-from-ior"},          # Sample Renderer: coat F0 from the material ior
    "coat-fresnel-vh": {**SPECULAR_EXT, "coat": "fresnel-vh"},            # F_c at V.H per light sample, not |V.Nc|
    "coat-base-unattenuated": {**SPECULAR_EXT, "coat": "base-unattenuated"},  # base not scaled by 1 - c F_c
}
COAT_VARIANTS = ("spec", "ignored", "f0-from-ior", "fresnel-vh", "base-unattenuated")
# F15.3 (ticket 10) negative controls of the sheen grids: deltas applied to the scene's main model (the sheen fixtures
# are sheen-only, so that is Appendix B; a sheen + coat/specular scene keeps its own extension model).
SHEEN_CONTROLS = {
    "sheen-ignored": {"sheen": "ignored"},                          # no sheen layer at all
    "sheen-unscaled-base": {"sheen": "unscaled-base"},              # sheen added, base not scaled by the albedo scaling
    "sheen-ashikhmin-visibility": {"sheen": "ashikhmin-visibility"},  # the cheaper visibility (D and E follow it)
    "sheen-roughness-linear": {"sheen": "roughness-linear"},        # alpha_g = roughness instead of roughness^2
    "sheen-cheap-scaling": {"sheen": "cheap-scaling"},              # direct light with 1 - max3(S) E(N.V) only
}
ALL_CONTROLS = sorted({**CONTROLS, **SHEEN_CONTROLS})


def control_model(name, main_model):
    """The model of a literal-model control: the sheen controls modify the scene's main model, the older ones are
    complete model dicts of their own."""
    if name in SHEEN_CONTROLS:
        return {**main_model, **SHEEN_CONTROLS[name]}
    return CONTROLS[name]


# --- BSDF -----------------------------------------------------------------

def clamp_roughness(roughness):
    return np.clip(np.asarray(roughness, dtype=np.float64), MIN_ROUGHNESS, 1.0)


def effective_alpha(roughness, model=APPENDIX_B):
    r = clamp_roughness(roughness)
    return np.maximum(r * r if model["alpha"] == "squared" else r, MIN_ALPHA)


def ggx_d(n_dot_h, alpha):
    a2 = alpha * alpha
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (math.pi * d * d)


def smith_vis(n_dot_v, n_dot_l, alpha):
    """Appendix B Vis = 1 / (2 (|NV| sqrt(a^2 + (1-a^2) NL^2) + |NL| sqrt(a^2 + (1-a^2) NV^2)))."""
    a2 = alpha * alpha
    denominator = 2.0 * (np.abs(n_dot_v) * np.sqrt(a2 + (1 - a2) * n_dot_l * n_dot_l) +
                         np.abs(n_dot_l) * np.sqrt(a2 + (1 - a2) * n_dot_v * n_dot_v))
    return 1.0 / np.maximum(denominator, 1e-12)


def schlick_weight(v_dot_h, model=APPENDIX_B):
    if model["fresnel"] == "off":
        return np.zeros_like(np.asarray(v_dot_h, dtype=np.float64))
    return (1.0 - np.clip(v_dot_h, 0.0, 1.0)) ** 5


def dot(a, b):
    return np.sum(a * b, axis=-1)


def normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-300)


# --- KHR_materials_ior + KHR_materials_specular -------------------------------

def ior_factor(ior):
    """((1 - ior) / (1 + ior))^2 of KHR_materials_ior; ior = 0 is the permanent special mode: the effective IOR is
    +infinity, so the factor is its limit, 1 (the Fresnel term evaluates to 1 for specular 1 and colour <= 1)."""
    ior = np.asarray(ior, dtype=np.float64)
    safe = np.where(ior == 0.0, 1.0, ior)
    return np.where(ior == 0.0, 1.0, ((1.0 - safe) / (1.0 + safe)) ** 2)


def specular_f0(ior, specular_color):
    """f0 of fresnel_mix: min(((1 - ior) / (1 + ior))^2 * specularColor, 1) per channel (before the weight)."""
    return np.minimum(ior_factor(ior)[..., None] * np.asarray(specular_color, dtype=np.float64), 1.0)


def dielectric_fresnel(f0, weight, w, variant="extension"):
    """(F, max F): the dielectric Fresnel term (n, 3) and its channel maximum (n, 1) at Schlick weight `w` (n, 1).

    extension (the specification): fr = f0 + (1 - f0) w; F = weight * fr, so the diffuse base is scaled by
      1 - weight * max(fr) = 1 - max(F).  Wrong variants (negative controls): "f90-one" scales f0 by the weight
      but keeps f90 = 1; "f0-unweighted" keeps f90 = weight without the weight on f0; "ignored" drops the
      extension (f0 = 0.04, weight 1)."""
    weight = np.asarray(weight, dtype=np.float64)[..., None]
    if variant == "ignored":
        f0, weight = np.full_like(f0, DIELECTRIC_F0), np.ones_like(weight)
    if variant == "f90-one":
        f = weight * f0 + (1.0 - weight * f0) * w
    elif variant == "f0-unweighted":
        f = f0 + (weight - f0) * w
    else:
        f = weight * (f0 + (1.0 - f0) * w)
    return f, np.max(f, axis=-1, keepdims=True)


def take(dielectric, sel):
    """The per-pixel arrays of `dielectric` restricted to the pixel indices `sel` (None stays None)."""
    if dielectric is None:
        return None
    return {"f0": dielectric["f0"][sel], "weight": dielectric["weight"][sel], "variant": dielectric["variant"]}


# --- KHR_materials_clearcoat --------------------------------------------------

COAT_IOR = 1.5
COAT_F0 = ((1.0 - COAT_IOR) / (1.0 + COAT_IOR)) ** 2  # 0.04, fixed by the ratified text


def take_coat(coat, sel):
    """The per-pixel arrays of `coat` restricted to the pixel indices `sel` (None stays None)."""
    if coat is None:
        return None
    return {"factor": coat["factor"][sel], "roughness": coat["roughness"][sel], "f0_ior": coat["f0_ior"][sel],
            "variant": coat["variant"]}


def coat_weight(coat):
    """The coat factor c per pixel (the `ignored` control removes the coat: c = 0)."""
    if coat["variant"] == "ignored":
        return np.zeros_like(coat["factor"])
    return coat["factor"]


def coat_fresnel(coat, nc, v, h=None):
    """F_c per pixel.  Specification: F0 = 0.04 (coat IOR 1.5), Schlick on |V.Nc|.  Controls: `f0-from-ior` takes F0
    from the material ior (the Sample Renderer deviation), `fresnel-vh` evaluates Schlick on V.H (needs `h`)."""
    f0 = coat["f0_ior"] if coat["variant"] == "f0-from-ior" else COAT_F0
    x = dot(v, h) if (coat["variant"] == "fresnel-vh" and h is not None) else np.abs(dot(nc, v))
    return f0 + (1.0 - f0) * (1.0 - np.clip(x, 0.0, 1.0)) ** 5


def coat_attenuation(coat, c, fc):
    """1 - c F_c: the factor of the material under the coat (the `base-unattenuated` control keeps 1)."""
    return np.ones_like(fc) if coat["variant"] == "base-unattenuated" else 1.0 - c * fc


def coat_lobe(coat, n, v, l):
    """(attenuation (n,), coat BSDF value (n,) without cosine) at light direction `l` per pixel; the coat normal is
    the shading normal n (no coat normal texture in the fixtures).  The coat value is c F_c D Vis with the coat's own
    cosine max(Nc.L, 0) as its validity mask; the caller multiplies it by that cosine."""
    c = coat_weight(coat)
    h = normalize(v + l)
    fc = coat_fresnel(coat, n, v, h)
    alpha = effective_alpha(coat["roughness"])
    nl = dot(n, l)
    nv = np.maximum(np.abs(dot(n, v)), MIN_NDOTV)
    nh = np.clip(dot(n, h), 0.0, 1.0)
    value = np.where(nl > 0, c * fc * ggx_d(nh, alpha) * smith_vis(nv, np.maximum(nl, 0.0), alpha), 0.0)
    return coat_attenuation(coat, c, fc), value


_COAT_ALBEDO = {}


def coat_albedo_table(roughness):
    """(mu grid, A, B) of the coat lobe D * Vis alone: the quadratures of `_spec_integrals` (A with weight 1 - w,
    B with the Schlick weight w of V.H), so E_coat = A + B is the albedo of D * Vis (Fresnel 1)."""
    alpha = float(effective_alpha(roughness))
    key = round(alpha, 12)
    if key not in _COAT_ALBEDO:
        mus = np.linspace(MIN_NDOTV, 1.0, MU_NODES)
        a = np.empty(MU_NODES)
        b = np.empty(MU_NODES)
        for i, mu in enumerate(mus):
            a[i], b[i] = _spec_integrals(float(mu), alpha, APPENDIX_B)
        _COAT_ALBEDO[key] = (mus, a, b)
    return _COAT_ALBEDO[key]


def coat_albedo(mu, roughness):
    """(E_coat, B) per pixel (interpolated over N.V) for per-pixel coat roughness values: E_coat = A + B is the
    directional albedo of D * Vis, B its Schlick-weighted part (needed by the `fresnel-vh` control)."""
    mu = np.clip(mu, MIN_NDOTV, 1.0)
    albedo = np.empty(len(mu))
    weighted = np.empty(len(mu))
    rough = np.asarray(roughness, dtype=np.float64)
    for r in np.unique(rough):
        sel = rough == r
        mus, a, b = coat_albedo_table(r)
        albedo[sel] = np.interp(mu[sel], mus, a + b)
        weighted[sel] = np.interp(mu[sel], mus, b)
    return albedo, weighted


def coat_fresnel_mu(coat, mu):
    """F_c(mu) with mu = |V.Nc| (the environment forms): F0 0.04, or the material-ior F0 of the control."""
    f0 = coat["f0_ior"] if coat["variant"] == "f0-from-ior" else COAT_F0
    return f0 + (1.0 - f0) * (1.0 - np.clip(mu, 0.0, 1.0)) ** 5


def environment_coat(material, n_dot_v, coat, radiance):
    """Exact-quadrature environment radiance under the coat: `material` is the (n, 3) radiance of the material
    alone (environment_radiance, already multiplied by `radiance`); the result is
    (1 - c F_c(mu)) * material + c F_c(mu) * E_coat(mu, a_c) * radiance.  Control `fresnel-vh`: the coat term
    integrates Schlick on V.H, c * (f0 (A + B) + (1 - f0) B) * radiance, the attenuation stays at |V.Nc|."""
    c = coat_weight(coat)
    mu = np.clip(n_dot_v, MIN_NDOTV, 1.0)
    fc = coat_fresnel_mu(coat, mu)
    albedo, weighted = coat_albedo(mu, coat["roughness"])
    if coat["variant"] == "fresnel-vh":
        coat_term = c * (COAT_F0 * albedo + (1.0 - COAT_F0) * weighted)
    else:
        coat_term = c * fc * albedo
    return coat_attenuation(coat, c, fc)[:, None] * material + (coat_term * radiance)[:, None]




# --- KHR_materials_sheen -------------------------------------------------------
#
# Written from the ratified text and its non-normative Charlie notes (research/khr-materials.md section 3), not from the
# MSL.  With sheenColor S, sheenRoughness r (clamped to [0.07, 1], `SHEEN_MIN_ROUGHNESS`, the declared lower bound of
# the Charlie lobe) and alpha_g = r^2:
#   D_charlie = (2 + 1/alpha_g) sin(theta_h)^(1/alpha_g) / (2 pi)
#   V_charlie = 1 / ((1 + Lambda(N.V) + Lambda(N.L)) 4 N.V N.L)      (Lambda: the Estevez-Kulla fit of the notes)
#   f_sheen   = S D_charlie V_charlie
#   f         = f_sheen + f_material * scaling,  f_material = the full material under the sheen (dielectric or metal,
#               ior/specular included); clearcoat (when present) sits ABOVE this sum
#   scaling   = min(1 - max3(S) E(N.V), 1 - max3(S) E(N.L))   direct light (point and sun), the notes' exact form
#             = 1 - max3(S) E(N.V)                            indirect light (environment), the notes' "cheap form"
# E(mu, r) is the Fresnel-free directional albedo of D V N.L, clamped to 1 (the Charlie visibility fit is not energy
# conserving at grazing views of narrow lobes; see tools/sheen_table.py).  Here E comes from this file's own quadrature
# over the INCIDENT hemisphere (midpoint rule in theta_i x phi_i), independent of the engine table
# `src/renderer/sheen_table.h` (half-vector quadrature, tools/sheen_table.py), which only the declared environment
# model reads.  Sun disk: the sheen lobe (alpha_g >= 0.0049, a lobe about 4 degrees wide, against a 0.27 degree
# disk) and the albedo scaling are point-evaluated at the disk centre; the base terms keep their declared rule.
# Texture-driven sheen is refused (the fixtures are texture-free).

SHEEN_MIN_ROUGHNESS = 0.07
SHEEN_FEATURE_BIT = 4  # MaterialFeature::Sheen
SHEEN_E_CLAMP = 1.0
SHEEN_QUADRATURE = (512, 128)  # (theta_i midpoint nodes, phi_i midpoint nodes over [0, pi]; the integrand is symmetric)
SHEEN_VARIANTS = ("spec", "ignored", "unscaled-base", "ashikhmin-visibility", "roughness-linear", "cheap-scaling")


def sheen_alpha(roughness, variant="spec"):
    """alpha_g: roughness^2 (the specification); `roughness-linear` (control) uses the roughness itself."""
    r = np.clip(np.asarray(roughness, dtype=np.float64), SHEEN_MIN_ROUGHNESS, 1.0)
    return r if variant == "roughness-linear" else r * r


def charlie_d(n_dot_h, alpha):
    inv = 1.0 / np.asarray(alpha, dtype=np.float64)
    sin2h = np.maximum(1.0 - np.asarray(n_dot_h, dtype=np.float64) ** 2, 0.0)
    return (2.0 + inv) * sin2h ** (0.5 * inv) / (2.0 * math.pi)


def _charlie_l(x, alpha):
    """l(x, alpha_g) of the notes: a / (1 + b x^c) + d x + e with every coefficient mix(low, high, (1 - alpha_g)^2)."""
    t = (1.0 - alpha) ** 2
    a, b, c = 21.5473 * (1 - t) + 25.3245 * t, 3.82987 * (1 - t) + 3.32435 * t, 0.19823 * (1 - t) + 0.16801 * t
    d, e = -1.97760 * (1 - t) + -1.27393 * t, -4.32054 * (1 - t) + -4.85967 * t
    return a / (1.0 + b * np.power(x, c)) + d * x + e


def charlie_lambda(cos_theta, alpha):
    x = np.abs(np.asarray(cos_theta, dtype=np.float64))
    near = np.exp(_charlie_l(np.minimum(x, 0.5), alpha))
    far = np.exp(2.0 * _charlie_l(0.5, alpha) - _charlie_l(1.0 - np.minimum(x, 1.0), alpha))
    return np.where(x < 0.5, near, far)


def charlie_visibility(n_dot_v, n_dot_l, alpha, variant="spec"):
    """V_charlie of the notes; `ashikhmin-visibility` (control) is the cheaper 1 / (4 (N.L + N.V - N.L N.V))."""
    nv = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    nl = np.maximum(np.asarray(n_dot_l, dtype=np.float64), 1e-9)
    if variant == "ashikhmin-visibility":
        return 1.0 / (4.0 * (nl + nv - nl * nv))
    return 1.0 / ((1.0 + charlie_lambda(nv, alpha) + charlie_lambda(nl, alpha)) * 4.0 * nv * nl)


def take_sheen(sheen, sel):
    """The per-pixel arrays of `sheen` restricted to the pixel indices `sel` (None stays None)."""
    if sheen is None:
        return None
    return {"color": sheen["color"][sel], "roughness": sheen["roughness"][sel], "variant": sheen["variant"]}


def sheen_color(sheen):
    """The per-pixel sheenColor (the `ignored` control removes the layer: S = 0)."""
    return np.zeros_like(sheen["color"]) if sheen["variant"] == "ignored" else sheen["color"]


_SHEEN_ALBEDO = {}


def sheen_albedo_table(alpha, variant="spec"):
    """(mu grid, E) of D V N.L over the incident hemisphere for one alpha_g, clamped to SHEEN_E_CLAMP."""
    key = (round(float(alpha), 12), variant == "ashikhmin-visibility")
    if key not in _SHEEN_ALBEDO:
        n_theta, n_phi = SHEEN_QUADRATURE
        theta = (np.arange(n_theta) + 0.5) * (0.5 * math.pi / n_theta)
        phi = (np.arange(n_phi) + 0.5) * (math.pi / n_phi)
        cos_i, sin_i = np.cos(theta)[:, None], np.sin(theta)[:, None]
        cell = 2.0 * (0.5 * math.pi / n_theta) * (math.pi / n_phi) * sin_i  # x2: the phi range [pi, 2 pi] mirrors
        mus = np.linspace(MIN_NDOTV, 1.0, MU_NODES)
        albedo = np.empty(MU_NODES)
        lambda_i = charlie_lambda(cos_i, alpha)
        for k, mu in enumerate(mus):
            sin_o = math.sqrt(max(0.0, 1.0 - mu * mu))
            wi_dot_wo = sin_i * np.cos(phi)[None, :] * sin_o + cos_i * mu
            n_dot_h = np.clip((cos_i + mu) / np.sqrt(2.0 + 2.0 * wi_dot_wo), 0.0, 1.0)  # (wi + wo).n / |wi + wo|
            if variant == "ashikhmin-visibility":
                visibility = 1.0 / (4.0 * (cos_i + mu - cos_i * mu))
            else:
                visibility = 1.0 / ((1.0 + charlie_lambda(mu, alpha) + lambda_i) * 4.0 * mu * cos_i)
            albedo[k] = float(np.sum(charlie_d(n_dot_h, alpha) * visibility * cos_i * cell))
        _SHEEN_ALBEDO[key] = (mus, np.minimum(albedo, SHEEN_E_CLAMP))
    return _SHEEN_ALBEDO[key]


def sheen_albedo(mu, roughness, variant="spec"):
    """E per pixel (interpolated over N.V) for per-pixel sheen roughness values."""
    mu = np.clip(np.asarray(mu, dtype=np.float64), MIN_NDOTV, 1.0)
    rough = np.asarray(roughness, dtype=np.float64)
    out = np.empty(mu.shape)
    for r in np.unique(rough):
        sel = rough == r
        mus, e = sheen_albedo_table(float(sheen_alpha(r, variant)), variant)
        out[sel] = np.interp(mu[sel], mus, e)
    return out


def sheen_scaling(sheen, e_v, e_l=None):
    """The albedo scaling of the material under the sheen (n,): min form with `e_l` (direct light), the cheap form
    without (indirect light); the `cheap-scaling` control keeps the cheap form for direct light, `unscaled-base` is 1."""
    if sheen["variant"] == "unscaled-base":
        return np.ones_like(e_v)
    m = np.max(sheen_color(sheen), axis=-1)
    scale = 1.0 - m * e_v
    if e_l is None or sheen["variant"] == "cheap-scaling":
        return scale
    return np.minimum(scale, 1.0 - m * e_l)


def sheen_lobe(sheen, n_dot_v, n_dot_l, n_dot_h):
    """f_sheen = S D V per pixel (n, 3) without the cosine (the caller masks N.L > 0)."""
    alpha = sheen_alpha(sheen["roughness"], sheen["variant"])
    d = charlie_d(n_dot_h, alpha) * charlie_visibility(n_dot_v, n_dot_l, alpha, sheen["variant"])
    return sheen_color(sheen) * d[..., None]


def sheen_layer(value, sheen, n_dot_v, n_dot_l, n_dot_h):
    """The material value `value` (n, 3, no cosine, zero where N.L <= 0) under the sheen layer: f_sheen + f scaling."""
    mask = np.asarray(n_dot_l) > 0
    e_v = sheen_albedo(n_dot_v, sheen["roughness"], sheen["variant"])
    e_l = sheen_albedo(np.maximum(n_dot_l, MIN_NDOTV), sheen["roughness"], sheen["variant"])
    scale = sheen_scaling(sheen, e_v, e_l)
    return value * scale[..., None] + np.where(mask[..., None], sheen_lobe(sheen, n_dot_v, n_dot_l, n_dot_h), 0.0)


def sheen_environment(material, n_dot_v, sheen, radiance):
    """Exact-quadrature environment radiance under the sheen: `material` (n, 3) is the radiance of the material alone
    (already multiplied by `radiance`); the result is material (1 - max3(S) E(mu)) + S E(mu) radiance (cheap form)."""
    e = sheen_albedo(n_dot_v, sheen["roughness"], sheen["variant"])
    return material * sheen_scaling(sheen, e)[:, None] + sheen_color(sheen) * (e * radiance)[:, None]


def lobe_weights(base, metallic, v_dot_h, model=APPENDIX_B, dielectric=None):
    """(diffuse multiplier per channel, specular multiplier per channel) at one V.H.

    f = diffuse_multiplier * base/pi-style term + specular_multiplier * D * Vis; the
    diffuse multiplier already contains (1 - metallic) and the Fresnel weighting.
    `dielectric` (F15.1): per-pixel {"f0" (n,3), "weight" (n,), "variant"} of KHR_materials_ior/specular.
    """
    w = schlick_weight(v_dot_h, model)[..., None]
    m = np.asarray(metallic, dtype=np.float64)[..., None]
    if dielectric is not None:
        f_d, max_f = dielectric_fresnel(dielectric["f0"], dielectric["weight"], w, dielectric["variant"])
        f_metal = base + (1 - base) * w
        return (1 - m) * (1.0 - max_f), (1 - m) * f_d + m * f_metal
    f_dielectric = DIELECTRIC_F0 + (1 - DIELECTRIC_F0) * w
    f_metal = base + (1 - base) * w
    if model["weighting"] == "engine-blend":
        f0 = DIELECTRIC_F0 * (1 - m) + base * m
        f = f0 + (1 - f0) * w
        return (1 - f) * (1 - m), f
    diffuse = (1 - m) * (1.0 if model["weighting"] == "unweighted-diffuse" else (1 - f_dielectric))
    specular = (1 - m) * f_dielectric + m * f_metal
    return diffuse, specular


def bsdf(base, metallic, roughness, n, v, l, model=APPENDIX_B, dielectric=None, coat=None, sheen=None):
    """BSDF value f (without the cosine) for world unit vectors; (n, 3) arrays or broadcastable.

    `sheen` (F15.3, ticket 10, per-pixel {"color", "roughness", "variant"} of KHR_materials_sheen): the material under
    the sheen layer, f_sheen + f_material scaling (sheen_layer).
    `coat` (F15.3, per-pixel {"factor", "roughness", "f0_ior", "variant"} of KHR_materials_clearcoat): the coated
    value (1 - c F_c) f_material + c F_c D Vis (see the module docstring); the coat term has its own cosine mask.
    The coat sits above the sheen: the coat attenuates the whole sheen + material value."""
    alpha = effective_alpha(roughness, model)
    n_dot_l = dot(n, l)
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    h = normalize(v + l)
    n_dot_h = np.clip(dot(n, h), 0.0, 1.0)
    v_dot_h = dot(v, h)
    diffuse_w, specular_w = lobe_weights(base, metallic, v_dot_h, model, dielectric)
    spec = (ggx_d(n_dot_h, alpha) * smith_vis(n_dot_v, np.maximum(n_dot_l, 0.0), alpha))[..., None]
    value = np.where((n_dot_l > 0)[..., None], diffuse_w * base / math.pi + specular_w * spec, 0.0)
    if sheen is not None:
        value = sheen_layer(value, sheen, n_dot_v, n_dot_l, n_dot_h)
    if coat is None:
        return value
    attenuation, coat_value = coat_lobe(coat, n, v, l)
    return attenuation[..., None] * value + coat_value[..., None]


# --- sun disk ---------------------------------------------------------------

def sun_quadrature(n, v, l, roughness, radius, model=APPENDIX_B):
    """Per-pixel sample count and `narrow` flag of the engine's declared disk rule."""
    alpha = effective_alpha(roughness, model)
    n_dot_v = dot(n, v)
    reflected = 2.0 * n_dot_v[..., None] * n - v
    separation = np.arccos(np.clip(dot(reflected, l), -1.0, 1.0))
    narrow = (alpha < 0.05) & (separation < 4.0 * radius + 16.0 * alpha)
    count = np.where(~narrow, 1, np.where(alpha < 2.0 * radius, 128, 32))
    return count, narrow


def sun_disk_samples(l, radius, count):
    """Deterministic uniform solid-angle golden-angle spiral over the disk of centre l (unit)."""
    l = np.asarray(l, dtype=np.float64)
    if count == 1:
        return l[None, :]
    helper = np.array([0.0, 0.0, 1.0]) if abs(l[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    t = np.cross(helper, l)
    t /= np.linalg.norm(t)
    b = np.cross(l, t)
    k = np.arange(count, dtype=np.float64)
    z = 1.0 - (k + 0.5) / count * (1.0 - math.cos(radius))
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = GOLDEN_ANGLE * k
    return t[None, :] * (s * np.cos(phi))[:, None] + b[None, :] * (s * np.sin(phi))[:, None] + l[None, :] * z[:, None]


def sun_radiance(base, metallic, roughness, n, v, toward_sun, irradiance, radius, model=APPENDIX_B, dielectric=None,
                 coat=None, sheen=None):
    """Outgoing radiance per pixel from the sun disk; returns (radiance (n,3), narrow (n,)).

    `sheen` (F15.3, ticket 10): the material terms are scaled by the albedo scaling at the disk centre and the sheen
    lobe S D V cos is point-evaluated at the disk centre (the lobe is wide against the disk, see the sheen section);
    the coat, when present, attenuates the sheen lobe like the material (centre F_c).

    `coat` (F15.3): the material terms are scaled by 1 - c F_c (F_c at |V.Nc|, or per disk sample at V.H in the
    `fresnel-vh` control) and the coat lobe adds c F_c D Vis cos averaged over its own disk rule (alpha, reflection
    axis and sample count of the coat roughness); `narrow` is the union of both lobes' masks."""
    n_pixels = n.shape[0]
    l = np.asarray(toward_sun, dtype=np.float64)
    l = l / np.linalg.norm(l)
    count, narrow = sun_quadrature(n, v, l, roughness, radius, model)
    alpha = effective_alpha(roughness, model)
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    # Diffuse: centre of the disk.
    n_dot_l = np.clip(dot(n, np.broadcast_to(l, n.shape)), 0.0, 1.0)
    h_c = normalize(v + l)
    diffuse_w, _ = lobe_weights(base, metallic, dot(v, h_c), model, dielectric)
    total = np.where((n_dot_l > 0)[..., None], diffuse_w * base / math.pi * n_dot_l[..., None], 0.0)
    coat_c = None
    if coat is not None:
        coat_c = coat_weight(coat)
        total = total * coat_attenuation(coat, coat_c, coat_fresnel(coat, n, v, h_c))[:, None]
    # Specular: mean of f_spec cos over the samples (count = 1 reduces to the point evaluation).
    spec_sum = np.zeros((n_pixels, 3))
    for c in np.unique(count):
        sel = np.flatnonzero(count == c)
        samples = sun_disk_samples(l, radius, int(c))
        for sample in samples:
            lk = np.broadcast_to(sample, (sel.size, 3))
            nl = np.clip(dot(n[sel], lk), 0.0, 1.0)
            h = normalize(v[sel] + lk)
            nh = np.clip(dot(n[sel], h), 0.0, 1.0)
            _, specular_w = lobe_weights(base[sel], metallic[sel], dot(v[sel], h), model, take(dielectric, sel))
            term = (ggx_d(nh, alpha[sel]) * smith_vis(n_dot_v[sel], nl, alpha[sel]) * nl)[:, None] * specular_w
            if coat is not None:
                sub = take_coat(coat, sel)
                term = term * coat_attenuation(sub, coat_c[sel], coat_fresnel(sub, n[sel], v[sel], h))[:, None]
            spec_sum[sel] += np.where((nl > 0)[:, None], term, 0.0) / int(c)
    result = total + spec_sum
    if sheen is not None:
        e_v = sheen_albedo(n_dot_v, sheen["roughness"], sheen["variant"])
        e_l = sheen_albedo(np.maximum(n_dot_l, MIN_NDOTV), sheen["roughness"], sheen["variant"])
        scale = sheen_scaling(sheen, e_v, e_l)
        lobe = sheen_lobe(sheen, n_dot_v, n_dot_l, np.clip(dot(n, h_c), 0.0, 1.0)) * n_dot_l[:, None]
        lobe = np.where((n_dot_l > 0)[:, None], lobe, 0.0)
        if coat is not None:
            lobe = lobe * coat_attenuation(coat, coat_c, coat_fresnel(coat, n, v, h_c))[:, None]
        result = result * scale[:, None] + lobe
    if coat is not None:
        # The coat lobe: its own roughness drives the disk rule (axis = reflection about Nc = n, own alpha).
        count_c, narrow_c = sun_quadrature(n, v, l, coat["roughness"], radius, model)
        alpha_c = effective_alpha(coat["roughness"], model)
        coat_sum = np.zeros(n_pixels)
        for c in np.unique(count_c):
            sel = np.flatnonzero(count_c == c)
            sub = take_coat(coat, sel)
            for sample in sun_disk_samples(l, radius, int(c)):
                lk = np.broadcast_to(sample, (sel.size, 3))
                nl = np.clip(dot(n[sel], lk), 0.0, 1.0)
                h = normalize(v[sel] + lk)
                nh = np.clip(dot(n[sel], h), 0.0, 1.0)
                fc = coat_fresnel(sub, n[sel], v[sel], h)
                term = coat_c[sel] * fc * ggx_d(nh, alpha_c[sel]) * smith_vis(n_dot_v[sel], nl, alpha_c[sel]) * nl
                coat_sum[sel] += np.where(nl > 0, term, 0.0) / int(c)
        result = result + coat_sum[:, None]
        narrow = narrow | narrow_c
    return result * np.asarray(irradiance, dtype=np.float64), narrow


# --- point lights ------------------------------------------------------------

def point_window(distance, light_range):
    ratio = distance / max(float(light_range), 1e-3)
    return np.clip(1.0 - ratio ** 4, 0.0, 1.0) ** 2


def point_radiance(base, metallic, roughness, n, v, position, light, model=APPENDIX_B, dielectric=None, visible=None,
                   coat=None, sheen=None):
    """Outgoing radiance per pixel from one white delta point light; returns (radiance (n,3), lit (n,))."""
    delta = np.asarray(light["position"], dtype=np.float64)[None, :] - position
    distance = np.linalg.norm(delta, axis=1)
    l = delta / distance[:, None]
    geometry = point_window(distance, light["range"]) / np.maximum(distance * distance, 1e-4)
    incident = np.asarray(light["color"], dtype=np.float64)[None, :] * float(light["intensity"]) * geometry[:, None]
    n_dot_l = np.clip(dot(n, l), 0.0, 1.0)
    value = bsdf(base, metallic, roughness, n, v, l, model, dielectric, coat, sheen) * n_dot_l[:, None] * incident
    lit = dot(n, l) > 0
    if visible is not None:
        value = value * visible[:, None]
        lit = lit & visible
    return value, lit


# --- environment ------------------------------------------------------------

_TABLES = {}
MU_NODES = 257
U_NODES, PHI_NODES = 512, 256
DIFFUSE_MU_NODES, DIFFUSE_PHI_NODES = 128, 128


def _diffuse_integral(mu, weight_of_vh):
    """integral over wi of weight(V.H) * cos(theta_i) / pi for wo at cos(theta_o) = mu."""
    x, wgt = np.polynomial.legendre.leggauss(DIFFUSE_MU_NODES)
    cos_i = 0.5 * (x + 1.0)  # Gauss-Legendre in cos(theta_i) over [0, 1]
    wgt = 0.5 * wgt
    phi = (np.arange(DIFFUSE_PHI_NODES) + 0.5) * (2 * math.pi / DIFFUSE_PHI_NODES)
    sin_i = np.sqrt(1 - cos_i ** 2)
    wo = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
    wi_dot_wo = sin_i[:, None] * np.cos(phi)[None, :] * wo[0] + cos_i[:, None] * wo[2]
    vh = np.sqrt(np.clip((1 + wi_dot_wo) / 2, 0.0, 1.0))
    integrand = weight_of_vh(vh) * cos_i[:, None] / math.pi
    return float(np.sum(integrand.mean(axis=1) * 2 * math.pi * wgt))


def _spec_integrals(mu, alpha, model):
    """(A, B): integrals of D*Vis*cos*(1-w) and D*Vis*cos*w over wi, via the GGX CDF in half-vector space."""
    u = (np.arange(U_NODES) + 0.5) / U_NODES
    phi = (np.arange(PHI_NODES) + 0.5) * (2 * math.pi / PHI_NODES)
    cos_h = np.sqrt((1 - u) / (1 + (alpha * alpha - 1) * u))
    sin_h = np.sqrt(np.maximum(0.0, 1 - cos_h ** 2))
    wo = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
    vh = sin_h[:, None] * np.cos(phi)[None, :] * wo[0] + cos_h[:, None] * wo[2]
    wi_z = 2 * vh * cos_h[:, None] - wo[2]  # z of wi = 2 (wo.h) h - wo
    valid = (wi_z > 0) & (vh > 0)
    # With h sampled from D cos(theta_h): integral = mean( Vis * cos_i * 4 (wo.h) / cos_h * g ).
    vis = smith_vis(mu, np.maximum(wi_z, 0.0), alpha)
    base = np.where(valid, vis * wi_z * 4 * vh / cos_h[:, None], 0.0)
    w = schlick_weight(vh, model)
    return float(np.mean(base * (1 - w))), float(np.mean(base * w))


def environment_table(roughness, model=APPENDIX_B):
    """Directional-albedo components over N.V for one roughness: dict of arrays on a uniform mu grid."""
    alpha = float(effective_alpha(roughness, model))
    key = (round(alpha, 12), model["fresnel"], model["alpha"] == "squared")
    if key not in _TABLES:
        mus = np.linspace(MIN_NDOTV, 1.0, MU_NODES)
        a = np.empty(MU_NODES)
        b = np.empty(MU_NODES)
        cd = np.empty(MU_NODES)
        cu = np.empty(MU_NODES)
        for i, mu in enumerate(mus):
            a[i], b[i] = _spec_integrals(float(mu), alpha, model)
            cd[i] = _diffuse_integral(float(mu), lambda vh: 1.0 - DIELECTRIC_F0 - (1 - DIELECTRIC_F0) * schlick_weight(vh, model))
            cu[i] = _diffuse_integral(float(mu), lambda vh: np.ones_like(vh))
        _TABLES[key] = {"mu": mus, "A": a, "B": b, "Cd": cd, "Cu": cu}
    return _TABLES[key]


_EXTENSION_DIFFUSE = {}


def extension_diffuse_table(c0, c1, model=APPENDIX_B):
    """Diffuse integral over N.V (the mu grid of environment_table) of (1 - max_c F_c(V.H)) * cos / pi for a dielectric
    Fresnel F_c = c0_c + c1_c * w(V.H) (affine in the Schlick weight; see dielectric_fresnel)."""
    key = (tuple(np.round(c0, 12)), tuple(np.round(c1, 12)), model["fresnel"])
    if key not in _EXTENSION_DIFFUSE:
        mus = np.linspace(MIN_NDOTV, 1.0, MU_NODES)
        c0a, c1a = np.asarray(c0, dtype=np.float64), np.asarray(c1, dtype=np.float64)
        weight = lambda vh: 1.0 - np.max(c0a[:, None, None] + c1a[:, None, None] * schlick_weight(vh, model)[None], axis=0)  # noqa: E731
        _EXTENSION_DIFFUSE[key] = np.array([_diffuse_integral(float(mu), weight) for mu in mus])
    return _EXTENSION_DIFFUSE[key]


def environment_radiance(base, metallic, roughness, n_dot_v, radiance, model=APPENDIX_B, dielectric=None):
    """Outgoing radiance (n, 3) under a constant isotropic environment of radiance `radiance`.

    `dielectric` (F15.1, per-pixel {"f0", "weight", "variant"} of KHR_materials_ior/specular): the exact quadrature of
    the extended dielectric.  F is affine in the Schlick weight (F = c0 + c1 w), so the specular part is
    c0 * (A + B) + c1 * B per channel and the diffuse part integrates 1 - max(F) (extension_diffuse_table)."""
    n_pixels = len(n_dot_v)
    result = np.zeros((n_pixels, 3))
    mu = np.clip(n_dot_v, MIN_NDOTV, 1.0)
    for r in np.unique(np.asarray(roughness, dtype=np.float64)):
        sel = np.flatnonzero(np.asarray(roughness) == r)
        t = environment_table(r, model)
        a = np.interp(mu[sel], t["mu"], t["A"])[:, None]
        b = np.interp(mu[sel], t["mu"], t["B"])[:, None]
        cd = np.interp(mu[sel], t["mu"], t["Cd"])[:, None]
        cu = np.interp(mu[sel], t["mu"], t["Cu"])[:, None]
        m = np.asarray(metallic, dtype=np.float64)[sel, None]
        c = base[sel]
        if dielectric is not None:
            d = take(dielectric, sel)
            f_w0, _ = dielectric_fresnel(d["f0"], d["weight"], np.zeros((len(sel), 1)), d["variant"])
            f_w1, _ = dielectric_fresnel(d["f0"], d["weight"], np.ones((len(sel), 1)), d["variant"])
            c0, c1 = f_w0, f_w1 - f_w0
            cd_ext = np.empty((len(sel), 1))
            combos = np.unique(np.concatenate([c0, c1], axis=1), axis=0)
            for combo in combos:
                pick = np.flatnonzero(np.all(np.concatenate([c0, c1], axis=1) == combo, axis=1))
                table = extension_diffuse_table(combo[:3], combo[3:], model)
                cd_ext[pick, 0] = np.interp(mu[sel][pick], t["mu"], table)
            dielectric_value = c * cd_ext + c0 * (a + b) + c1 * b
            result[sel] = (1 - m) * dielectric_value + m * (c * a + b)
            continue
        if model["weighting"] == "engine-blend":
            f0 = DIELECTRIC_F0 * (1 - m) + c * m
            result[sel] = (1 - m) * c * cu + f0 * a + b
            continue
        diffuse = c * (cu if model["weighting"] == "unweighted-diffuse" else cd)
        dielectric_value = diffuse + DIELECTRIC_F0 * a + b
        metal = c * a + b
        result[sel] = (1 - m) * dielectric_value + m * metal
    return result * radiance


# --- declared environment model (the engine's, ADR-0006) -------------------------

DFG_HEADER = REPO / "src" / "renderer" / "dfg_table.h"
_DFG = {}

# "extension_diffuse": the environment diffuse weight of materials carrying a feature. Since ticket 08 the F13
# composite receives the lobe guide and weights the GI/environment diffuse of an extended pixel with
# 1 - max(E_dielectric) of its own specular lobe ("extension", the engine's state). "composite-base" (the weight
# 1 - E(0.04) of the base model, the engine's state before ticket 08) is kept as a negative control.
DECLARED = {"albedo": "dfg-table", "diffuse": "one-minus-dielectric-albedo", "dielectric": "extension",
            "extension_diffuse": "extension", "coat": "spec", "sheen": "spec"}
# Deliberately wrong variants that the declared-model gate must reject.
DECLARED_CONTROLS = {
    "declared-diffuse-unweighted": {**DECLARED, "diffuse": "unweighted"},            # no (1 - E) weight
    "declared-karis-fit": {**DECLARED, "albedo": "karis-fit"},                       # mobile fit instead of the table
    "declared-diffuse-f0-fresnel": {**DECLARED, "diffuse": "metallic-blended-f0"},   # F of the blended f0 weights diffuse
}
# F15.1: negative controls of the ior/specular grid of the deterministic resolve (kept apart: they only apply
# to scenes whose materials carry KHR_materials_ior/specular).
SPECULAR_DECLARED_CONTROLS = {
    "declared-specular-ignored": {**DECLARED, "dielectric": "ignored"},
    "declared-f90-one": {**DECLARED, "dielectric": "f90-one"},
    "declared-specular-weight-not-on-f0": {**DECLARED, "dielectric": "f0-unweighted"},
    # The composite weighting the environment diffuse of an extended pixel with the base model (ticket 08 closes it).
    "declared-extension-diffuse-base": {**DECLARED, "extension_diffuse": "composite-base"},
}
# F15.3 (ticket 09): negative controls of the clearcoat grids of the deterministic resolve.  The split-sum model has
# no V.H, so `coat-fresnel-vh` has no declared counterpart.
COAT_DECLARED_CONTROLS = {
    "declared-coat-ignored": {**DECLARED, "coat": "ignored"},
    "declared-coat-f0-from-ior": {**DECLARED, "coat": "f0-from-ior"},
    "declared-coat-base-unattenuated": {**DECLARED, "coat": "base-unattenuated"},
}
# F15.3 (ticket 10): negative controls of the sheen grids of the deterministic resolve.  The declared model reads the
# engine sheen table, so a different visibility has no declared counterpart (no table exists for it).
SHEEN_DECLARED_CONTROLS = {
    "declared-sheen-ignored": {**DECLARED, "sheen": "ignored"},
    "declared-sheen-unscaled-base": {**DECLARED, "sheen": "unscaled-base"},
    "declared-sheen-roughness-linear": {**DECLARED, "sheen": "roughness-linear"},
}
ALL_DECLARED_CONTROLS = {**DECLARED_CONTROLS, **SPECULAR_DECLARED_CONTROLS, **COAT_DECLARED_CONTROLS,
                         **SHEEN_DECLARED_CONTROLS}


def load_dfg_table(path=DFG_HEADER):
    """(size, table[size, size, 2]) parsed from the engine header: table[roughness row, N.V column, (A, B)]."""
    key = str(path)
    if key not in _DFG:
        text = pathlib.Path(path).read_text()
        size = int(re.search(r"DfgTableSize\s*=\s*(\d+)\s*;", text).group(1))
        start = text.index("DfgTable[")
        body = text[text.index("{", start) + 1:text.index("};", start)]
        values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)]
        if len(values) != size * size * 2:
            raise ValueError(f"{path}: expected {size * size * 2} DFG values, parsed {len(values)}")
        _DFG[key] = (size, np.asarray(values, dtype=np.float64).reshape(size, size, 2))
    return _DFG[key]


def dfg_lookup(n_dot_v, roughness, path=DFG_HEADER):
    """Bilinear (A, B) of the DFG table: the rule of `dfgTableLookup` (node = value * (size - 1), the cell
    index clamped to size - 2, rows = roughness, columns = N.V), written here from its documentation."""
    size, table = load_dfg_table(path)
    n = size - 1
    x = np.clip(np.asarray(n_dot_v, dtype=np.float64), 0.0, 1.0) * n
    y = np.clip(np.asarray(roughness, dtype=np.float64), 0.0, 1.0) * n
    x0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), size - 2)
    fx, fy = x - x0, y - y0

    def component(c):
        return ((table[y0, x0, c] * (1 - fx) + table[y0, x0 + 1, c] * fx) * (1 - fy) +
                (table[y0 + 1, x0, c] * (1 - fx) + table[y0 + 1, x0 + 1, c] * fx) * fy)

    return component(0), component(1)


SHEEN_HEADER = REPO / "src" / "renderer" / "sheen_table.h"
_SHEEN_TABLE = {}


def load_sheen_table(path=SHEEN_HEADER):
    """(size, table[size, size]) parsed from the engine header: table[sheen roughness row, N.V column] = E."""
    key = str(path)
    if key not in _SHEEN_TABLE:
        text = pathlib.Path(path).read_text()
        size = int(re.search(r"SheenTableSize\s*=\s*(\d+)\s*;", text).group(1))
        start = text.index("SheenTable[")
        body = text[text.index("{", start) + 1:text.index("};", start)]
        values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)]
        if len(values) != size * size:
            raise ValueError(f"{path}: expected {size * size} sheen values, parsed {len(values)}")
        _SHEEN_TABLE[key] = (size, np.asarray(values, dtype=np.float64).reshape(size, size))
    return _SHEEN_TABLE[key]


def sheen_lookup(n_dot_v, roughness, path=SHEEN_HEADER):
    """Bilinear E of the sheen table: the rule of `sheenTableLookup` (node = value * (size - 1), the cell index
    clamped to size - 2, rows = sheen roughness, columns = N.V), written here from its documentation."""
    size, table = load_sheen_table(path)
    n = size - 1
    x = np.clip(np.asarray(n_dot_v, dtype=np.float64), 0.0, 1.0) * n
    y = np.clip(np.asarray(roughness, dtype=np.float64), 0.0, 1.0) * n
    x0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), size - 2)
    fx, fy = x - x0, y - y0
    return ((table[y0, x0] * (1 - fx) + table[y0, x0 + 1] * fx) * (1 - fy) +
            (table[y0 + 1, x0] * (1 - fx) + table[y0 + 1, x0 + 1] * fx) * fy)


def declared_sheen_environment(material, n_dot_v, sheen, variant):
    """The declared sheen environment model for unit radiance: material (1 - max3(S) E_t) + S E_t, E_t = the table
    value at (N.V, sheen roughness) (clamped to [0.07, 1], the table's own clamp); variant["sheen"] names the control
    (`ignored`, `unscaled-base`, `roughness-linear`: the table is read at sqrt(r), i.e. alpha_g = r)."""
    kind = variant.get("sheen", "spec")
    if kind in ("ashikhmin-visibility", "cheap-scaling"):
        raise ValueError("the declared sheen model has no Ashikhmin table and no per-light scaling")
    sheen = {**sheen, "variant": kind}
    r = np.clip(np.asarray(sheen["roughness"], dtype=np.float64), SHEEN_MIN_ROUGHNESS, 1.0)
    e = sheen_lookup(np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV),
                     np.sqrt(r) if kind == "roughness-linear" else r)
    return material * sheen_scaling(sheen, e)[:, None] + sheen_color(sheen) * e[:, None]


def karis_ab(n_dot_v, roughness):
    """Karis 2014 mobile environment-BRDF fit (A, B); the former engine model."""
    c0 = (-1.0, -0.0275, -0.572, 0.022)
    c1 = (1.0, 0.0425, 1.04, -0.04)
    r = [roughness * c0[i] + c1[i] for i in range(4)]
    a004 = np.minimum(r[0] * r[0], np.exp2(-9.28 * n_dot_v)) * r[0] + r[1]
    return -1.04 * a004 + r[2], 1.04 * a004 + r[3]


def declared_environment_radiance(base, metallic, roughness, n_dot_v, radiance, variant=DECLARED, dielectric=None,
                                  coat=None, sheen=None):
    """Outgoing radiance (n, 3) of the engine's declared environment model (see the module docstring).

    `dielectric` (F15.1): per-pixel {"f0", "weight"} of KHR_materials_ior/specular (variant["dielectric"] picks the
    model or one of its wrong controls): E_d = f0_scaled * A + f90 * B per lobe, diffuse weight 1 - max(E_d).
    `coat` (F15.3, per-pixel {"factor", "roughness", "f0_ior", "variant"} of KHR_materials_clearcoat): the declared
    coat model (1 - c F_c(mu)) * declared_material + c F_c(mu) * clamp(A_c + B_c, 0, 1), (A_c, B_c) = dfg_lookup at
    the coat roughness; variant["coat"] names the control (`ignored`, `f0-from-ior`, `base-unattenuated`).
    `sheen` (F15.3, ticket 10): the material is first put under the sheen (declared_sheen_environment), the coat sits
    above that sum."""
    material = _declared_material_radiance(base, metallic, roughness, n_dot_v, variant, dielectric)
    if sheen is not None:
        material = declared_sheen_environment(material, n_dot_v, sheen, variant)
    if coat is not None:
        coat = {**coat, "variant": variant.get("coat", "spec")}
        if coat["variant"] == "fresnel-vh":
            raise ValueError("the declared split-sum model has no V.H: no fresnel-vh control")
        c = coat_weight(coat)
        mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
        a_c, b_c = dfg_lookup(mu, clamp_roughness(coat["roughness"]))
        fc = coat_fresnel_mu(coat, mu)
        material = (coat_attenuation(coat, c, fc)[:, None] * material +
                    (c * fc * np.clip(a_c + b_c, 0.0, 1.0))[:, None])
    return material * radiance


def _declared_material_radiance(base, metallic, roughness, n_dot_v, variant, dielectric):
    """The declared model of the material alone for unit environment radiance (see declared_environment_radiance)."""
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    rough = clamp_roughness(roughness)
    a, b = karis_ab(mu, rough) if variant["albedo"] == "karis-fit" else dfg_lookup(mu, rough)
    a, b = a[:, None], b[:, None]
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    if dielectric is not None:
        f0c, weight = dielectric["f0"], np.asarray(dielectric["weight"], dtype=np.float64)[:, None]
        kind = variant.get("dielectric", "extension")
        if kind == "ignored":
            f0c, weight = np.full_like(f0c, DIELECTRIC_F0), np.ones_like(weight)
        f0_scaled = f0c if kind == "f0-unweighted" else f0c * weight
        f90 = np.ones_like(weight) if kind == "f90-one" else weight
        e_d = np.clip(f0_scaled * a + f90 * b, 0.0, 1.0)
        e_m = np.clip(base * a + b, 0.0, 1.0)
        if variant.get("extension_diffuse", "extension") == "composite-base":
            diffuse_weight = 1.0 - np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)
        else:
            diffuse_weight = 1.0 - np.max(e_d, axis=-1, keepdims=True)
        return (1 - m) * base * diffuse_weight + (1 - m) * e_d + m * e_m
    f0 = DIELECTRIC_F0 * (1 - m) + base * m
    specular = np.clip(f0 * a + b, 0.0, 1.0)
    if variant["diffuse"] == "unweighted":
        weight = np.ones_like(a)
    elif variant["diffuse"] == "metallic-blended-f0":
        weight = 1.0 - np.clip(f0 * a + b, 0.0, 1.0)
    else:
        weight = 1.0 - np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)
    return (1 - m) * base * weight + specular


# --- ray casting ------------------------------------------------------------

def mitsuba_scene(root, data, threads):
    """The snapshot meshes as one Mitsuba scene (shape ids `instance_<index>`)."""
    import drjit as dr
    import mitsuba as mi

    mi.set_variant("llvm_ad_rgb")
    dr.set_flag(dr.JitFlag.FastMath, False)
    dr.set_thread_count(threads)
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    description = {"type": "scene"}
    for index, instance in enumerate(data["instances"]):
        description[f"instance_{index}"] = {
            "type": "ply", "filename": str(root / data["meshes"][instance["mesh"]]),
            "face_normals": False,
            "to_world": transform(np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F"))}
    return mi.load_dict(description)


SHADOW_OFFSET = 1e-4


def shadow_visibility(root, data, position, normal, target, threads):
    """1 where the segment from position (offset along normal) to target hits nothing, per row."""
    import drjit as dr  # noqa: F401  (variant set by mitsuba_scene)
    import mitsuba as mi

    scene = mitsuba_scene(root, data, threads)
    origin = position + SHADOW_OFFSET * normal
    delta = np.asarray(target, dtype=np.float64)[None, :] - origin
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / distance[:, None]
    ray = mi.Ray3f(mi.Point3f(mi.Float(origin[:, 0].copy()), mi.Float(origin[:, 1].copy()), mi.Float(origin[:, 2].copy())),
                   mi.Vector3f(mi.Float(direction[:, 0].copy()), mi.Float(direction[:, 1].copy()),
                               mi.Float(direction[:, 2].copy())))
    ray.maxt = mi.Float((distance * (1.0 - 1e-6)).copy())
    return ~np.asarray(scene.ray_test(ray))


def cast_primary_rays(root, data, threads):
    """Pixel-centre primary hits with Mitsuba: (valid, position, shading normal, instance index)."""
    import drjit as dr
    import mitsuba as mi

    scene = mitsuba_scene(root, data, threads)
    identifiers = np.asarray(dr.reinterpret_array(mi.UInt32, scene.shapes_dr()))
    index_of = {int(i): k for k, (shape, i) in enumerate(zip(scene.shapes(), identifiers))}
    names = [shape.id() for shape in scene.shapes()]
    camera = data["camera"]
    if any(camera.get("jitter_pixels", [0, 0])):
        raise ValueError("pixel-centre reference requires an unjittered camera")
    width, height = int(camera["width"]), int(camera["height"])
    forward = np.asarray(camera["direction"], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera["up"], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    scale = height / (2 * math.tan(camera["fov_y_radians"] / 2))
    pixel = np.arange(width * height)
    x = (pixel % width + 0.5 - width / 2) / scale
    y = (height / 2 - (pixel // width + 0.5)) / scale
    direction = forward[None, :] + right[None, :] * x[:, None] + up[None, :] * y[:, None]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ray = mi.Ray3f(mi.Point3f(*[float(c) for c in camera["position"]]),
                   mi.Vector3f(mi.Float(direction[:, 0].copy()), mi.Float(direction[:, 1].copy()),
                               mi.Float(direction[:, 2].copy())))
    si = scene.ray_intersect(ray)
    valid = np.asarray(si.is_valid())
    position = np.stack([np.asarray(si.p.x), np.asarray(si.p.y), np.asarray(si.p.z)], axis=1).astype(np.float64)
    normal = np.stack([np.asarray(si.sh_frame.n.x), np.asarray(si.sh_frame.n.y), np.asarray(si.sh_frame.n.z)],
                      axis=1).astype(np.float64)
    shape_id = np.asarray(dr.reinterpret_array(mi.UInt32, si.shape)).astype(np.int64)
    instance = np.full(width * height, -1, dtype=np.int64)
    for key, value in index_of.items():
        instance[valid & (shape_id == key)] = int(names[value].split("_")[1])
    valid &= instance >= 0
    return valid.reshape(height, width), position, normal, instance.reshape(height, width), (width, height)


# --- frame ------------------------------------------------------------------

SPECULAR_FEATURE_BIT = 1  # MaterialFeature::SpecularIor
CLEARCOAT_FEATURE_BIT = 2  # MaterialFeature::Clearcoat
EXTENSION_RECORD_WORDS = 20


def f32(word):
    """The float whose IEEE-754 binary32 bit pattern is the unsigned word."""
    import struct
    return struct.unpack("<f", struct.pack("<I", int(word) & 0xFFFFFFFF))[0]


def material_extension_words(data, material):
    """The 20 words of record 0 of a material with the ior/specular, clearcoat and/or sheen feature bits; raises when it
    has other features.  None when the material has no feature bit."""
    bits = int(material.get("feature_bits", 0))
    if bits == 0:
        return None
    if bits & ~(SPECULAR_FEATURE_BIT | CLEARCOAT_FEATURE_BIT | SHEEN_FEATURE_BIT):
        raise ValueError(f"material feature bits {bits:#x} are outside the declared ior/specular/clearcoat/sheen model")
    words = data["material_extensions"]["records"][int(material["extension_index"])]
    if len(words) != EXTENSION_RECORD_WORDS:
        raise ValueError("a material extension record must hold 20 words")
    return words


def material_specular(data, material):
    """(ior, specularFactor, specularColor[3]) of a material from the snapshot's extension records, or None when
    the material has no KHR_materials_ior/specular feature bit.  Raises when it has other features or textures.
    A clearcoat-only material has the neutral block (the engine ignores the record words without the bit)."""
    words = material_extension_words(data, material)
    if words is None or not int(material.get("feature_bits", 0)) & SPECULAR_FEATURE_BIT:
        if words is not None and (int(words[5]) != INVALID or int(words[6]) != INVALID):
            raise ValueError("a specular texture is set; the reference is texture-free")
        return None
    if int(words[5]) != INVALID or int(words[6]) != INVALID:
        raise ValueError("a specular texture is set; the reference is texture-free")
    # GPUMaterialExt0 words 0 (ior), 1 (specularFactor), 2-4 (specularColor).
    return f32(words[0]), f32(words[1]), [f32(words[2]), f32(words[3]), f32(words[4])]


def material_clearcoat(data, material):
    """(clearcoatFactor, clearcoatRoughness, clearcoatNormalScale) from word offsets 7, 8, 9 of GPUMaterialExt0, or
    None without the clearcoat feature bit.  Raises when a clear-coat texture (words 10, 11, 12: factor, roughness,
    normal) is set: the reference is texture-free, and the coat normal is the shading normal."""
    words = material_extension_words(data, material)
    if words is None or not int(material.get("feature_bits", 0)) & CLEARCOAT_FEATURE_BIT:
        return None
    if any(int(words[k]) != INVALID for k in (10, 11, 12)):
        raise ValueError("a clear-coat texture is set; the reference is texture-free")
    return f32(words[7]), f32(words[8]), f32(words[9])


def material_sheen(data, material):
    """(sheenColor[3], sheenRoughness) from word offsets 13-15 and 16 of GPUMaterialExt0, or None without the sheen
    feature bit.  Raises when a sheen texture (words 17, 18: colour, roughness) is set: the reference is texture-free.
    A zero sheenColor disables the layer (the engine clears the bit then); the formulas reduce to no sheen as well."""
    words = material_extension_words(data, material)
    if words is None or not int(material.get("feature_bits", 0)) & SHEEN_FEATURE_BIT:
        return None
    if any(int(words[k]) != INVALID for k in (17, 18)):
        raise ValueError("a sheen texture is set; the reference is texture-free")
    return [f32(words[13]), f32(words[14]), f32(words[15])], f32(words[16])


def uses_sheen(data):
    return any(int(m.get("feature_bits", 0)) & SHEEN_FEATURE_BIT for m in data["materials"])


def uses_specular_extension(data):
    return any(int(m.get("feature_bits", 0)) & SPECULAR_FEATURE_BIT for m in data["materials"])


def uses_clearcoat(data):
    return any(int(m.get("feature_bits", 0)) & CLEARCOAT_FEATURE_BIT for m in data["materials"])


def sphere_table(data):
    out = []
    uses = uses_specular_extension(data)
    uses_coat = uses_clearcoat(data)
    uses_sheen_scene = uses_sheen(data)
    for k, inst in enumerate(data["instances"]):
        material = data["materials"][inst["material"]]
        entry = {"instance": k, "slot": inst["slot"], "base": material["base"][:3],
                 "metallic": float(material["metallic"]), "roughness": float(material["roughness"])}
        spec = material_specular(data, material)
        if spec is None and uses:  # a neutral block (no feature bit) in a scene that has the extension
            spec = (1.5, 1.0, [1.0, 1.0, 1.0])
        if spec is not None:
            entry.update(ior=spec[0], specular=spec[1], specular_color=spec[2],
                         group=f"metallic {entry['metallic']:g} specular {spec[1]:g} colour "
                               f"{spec[2][0]:g},{spec[2][1]:g},{spec[2][2]:g}")
        if uses_coat:
            coat = material_clearcoat(data, material) or (0.0, 0.0, 1.0)
            ior = spec[0] if spec is not None else 1.5
            entry.update(coat=coat[0], coat_roughness=coat[1], coat_normal_scale=coat[2],
                         group=f"metallic {entry['metallic']:g} ior {ior:g} coat {coat[0]:g}")
        if uses_sheen_scene:
            sheen = material_sheen(data, material) or ([0.0, 0.0, 0.0], 0.0)
            c = sheen[0]
            entry.update(sheen_color=c, sheen_roughness=sheen[1],
                         group=f"{entry.get('group', 'metallic ' + format(entry['metallic'], 'g'))} sheen "
                               f"{c[0]:g},{c[1]:g},{c[2]:g}")
        out.append(entry)
    return out


def dielectric_arrays(spheres, instance_of_pixel, variant):
    """Per-pixel {"f0", "weight", "variant"} of the extension, neutral (ior 1.5, weight 1, colour 1) elsewhere."""
    f0 = np.empty((len(spheres), 3))
    weight = np.empty(len(spheres))
    for k, s in enumerate(spheres):
        ior, spec, color = (s["ior"], s["specular"], s["specular_color"]) if "ior" in s else (1.5, 1.0, [1.0] * 3)
        f0[k] = specular_f0(ior, color)
        weight[k] = spec
    return {"f0": f0[instance_of_pixel], "weight": weight[instance_of_pixel], "variant": variant}


def coat_arrays(spheres, instance_of_pixel, variant):
    """Per-pixel {"factor", "roughness", "f0_ior", "variant"} of KHR_materials_clearcoat (factor 0 where a sphere has
    no coat); `f0_ior` is the F0 of the material ior, used by the `f0-from-ior` control only."""
    factor = np.array([s.get("coat", 0.0) for s in spheres], dtype=np.float64)
    roughness = np.array([s.get("coat_roughness", 0.0) for s in spheres], dtype=np.float64)
    f0 = np.array([float(ior_factor(s.get("ior", DIELECTRIC_IOR))) for s in spheres], dtype=np.float64)
    return {"factor": factor[instance_of_pixel], "roughness": roughness[instance_of_pixel],
            "f0_ior": f0[instance_of_pixel], "variant": variant}


def sheen_arrays(spheres, instance_of_pixel, variant):
    """Per-pixel {"color", "roughness", "variant"} of KHR_materials_sheen (colour 0 where a sphere has no sheen)."""
    color = np.array([s.get("sheen_color", [0.0, 0.0, 0.0]) for s in spheres], dtype=np.float64)
    roughness = np.array([s.get("sheen_roughness", 0.0) for s in spheres], dtype=np.float64)
    return {"color": color[instance_of_pixel], "roughness": roughness[instance_of_pixel], "variant": variant}


def require_declared_scene(data):
    for instance in data["instances"]:
        material = data["materials"][instance["material"]]
        material_specular(data, material)  # only the ior/specular and clearcoat features, without textures
        material_clearcoat(data, material)
        material_sheen(data, material)
        if any(index != INVALID for index in material["textures"]):
            raise ValueError("instanced material samples a texture; the Appendix B reference is texture-free")
        if any(v > 0 for v in material["emissive"]) or material["alpha_cutoff"] > 0:
            raise ValueError("emissive or alpha-masked material is outside the declared model")
    if any(light["type"] not in (0, 1) for light in data["lights"]) or \
            sum(light["type"] == 0 for light in data["lights"]) > 1:
        raise ValueError("at most one directional light (the sun) and white point lights are modelled")
    for light in data["lights"]:
        if light["type"] == 1 and not (light["color"][0] == light["color"][1] == light["color"][2]):
            raise ValueError("point lights must be white")
    sky = data["sky"]
    if not (sky[0] == sky[1] == sky[2]):
        raise ValueError("the environment must be isotropic and grey")
    points = [light for light in data["lights"] if light["type"] == 1]
    for sampled in data["sampled_lights"]:
        # DI lists the delta point lights it samples; anything else (area/emissive) is outside the model.
        same = [light for light in points if sampled["type"] == 1 and
                max(abs(a - b) for a, b in zip(sampled["position"], light["position"])) <= 1e-5]
        if not same:
            raise ValueError("sampled (area) lights are outside the declared model")


def render_frame(root, data, model=APPENDIX_B, threads=None, cast=None, environment=None, visibility=None):
    """Reference radiance image (h, w, 3) float32 plus masks and the per-sphere table.

    `environment` None = exact quadrature of the literal model; a DECLARED variant dict = the engine's declared
    environment model (the sun term is the same in both).  `visibility`: None (no shadow rays) or a list with
    one boolean array per point light (shadow rays, see point_light_visibility)."""
    require_declared_scene(data)
    valid, position, normal, instance, (width, height) = cast or cast_primary_rays(
        root, data, threads or min(8, os.cpu_count() or 1))
    spheres = sphere_table(data)
    flat = valid.reshape(-1)
    inst = instance.reshape(-1)
    camera = np.asarray(data["camera"]["position"], dtype=np.float64)
    pick = np.flatnonzero(flat)
    n = normalize(normal[pick])
    v = normalize(camera[None, :] - position[pick])
    base = np.array([spheres[k]["base"] for k in inst[pick]], dtype=np.float64)
    metallic = np.clip(np.array([spheres[k]["metallic"] for k in inst[pick]]), 0.0, 1.0)
    roughness = np.array([spheres[k]["roughness"] for k in inst[pick]])
    # F15.1: KHR_materials_ior/specular of the scene (the model dicts name the dielectric variant).
    variant = model.get("dielectric", "base")
    sun_dielectric = None if variant == "base" else dielectric_arrays(spheres, inst[pick], variant)
    # F15.3: KHR_materials_clearcoat of the scene (None without any coat: the older scenes are unchanged).
    has_coat = uses_clearcoat(data)
    coat = coat_arrays(spheres, inst[pick], model.get("coat", "spec")) if has_coat else None
    env_coat = coat_arrays(spheres, inst[pick], (environment or {}).get("coat", "spec")) if has_coat else None
    # F15.3 (ticket 10): KHR_materials_sheen of the scene (None without any sheen: the older scenes are unchanged).
    has_sheen = uses_sheen(data)
    sheen = sheen_arrays(spheres, inst[pick], model.get("sheen", "spec")) if has_sheen else None
    env_sheen = sheen_arrays(spheres, inst[pick], (environment or {}).get("sheen", "spec")) if has_sheen else None
    env_variant = (environment or {}).get("dielectric", "extension")
    env_dielectric = (dielectric_arrays(spheres, inst[pick], env_variant)
                      if environment is not None and uses_specular_extension(data) else None)
    radiance = np.zeros((pick.size, 3))
    narrow = np.zeros(pick.size, dtype=bool)
    lit = np.zeros(pick.size, dtype=bool)
    terms = []
    sky = float(data["sky"][0])
    sun_radius = float(data.get("sun_angular_radius", 0.0))
    points = [light for light in data["lights"] if light["type"] == 1]
    for k, light in enumerate(points):
        visible = None if visibility is None else visibility[k]
        value, light_lit = point_radiance(base, metallic, roughness, n, v, position[pick], light, model, sun_dielectric,
                                          visible, coat, sheen)
        radiance += value
        lit = lit | light_lit
        terms.append("point-light-shadowed" if visible is not None else "point-light")
    for light in (light for light in data["lights"] if light["type"] == 0):
        toward = -normalize(np.asarray(light["direction"], dtype=np.float64))
        irradiance = np.asarray(light["color"], dtype=np.float64) * float(light["intensity"])
        if sun_radius <= 0:
            raise ValueError("a sun requires a positive angular radius")
        sun, narrow = sun_radiance(base, metallic, roughness, n, v, toward, irradiance, sun_radius, model,
                                   sun_dielectric, coat, sheen)
        radiance += sun
        lit = lit | (dot(n, np.broadcast_to(toward, n.shape)) > 0)
        terms.append("sun-disk")
    if sky > 0:
        if environment is None:
            material = environment_radiance(base, metallic, roughness, dot(n, v), sky, model, sun_dielectric)
            if sheen is not None:
                material = sheen_environment(material, dot(n, v), sheen, sky)
            radiance += material if coat is None else environment_coat(material, dot(n, v), coat, sky)
            terms.append("environment-quadrature")
        else:
            radiance += declared_environment_radiance(base, metallic, roughness, dot(n, v), sky, environment,
                                                      env_dielectric, env_coat, env_sheen)
            terms.append("environment-declared")
    image = np.full((height * width, 3), sky, dtype=np.float64)  # pixels without geometry show the environment
    image[pick] = radiance
    # ROI: sphere pixels whose 4-neighbours belong to the same sphere (silhouette excluded).
    sphere_id = instance.copy()
    same = np.ones((height, width), dtype=bool)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        same &= np.roll(sphere_id, shift, axis=axis) == sphere_id
    same[0, :] = same[-1, :] = same[:, 0] = same[:, -1] = False
    roi = valid & same
    narrow_full = np.zeros(height * width, dtype=bool)
    narrow_full[pick] = narrow
    lit_full = np.zeros(height * width, dtype=bool)
    lit_full[pick] = lit
    masks = {"roi": roi, "sphere": sphere_id.astype(np.int32), "narrow": narrow_full.reshape(height, width),
             "lit": lit_full.reshape(height, width)}
    return image.reshape(height, width, 3).astype(np.float32), masks, spheres, terms


def point_light_visibility(root, data, cast, threads):
    """Per point light, the shadow-ray visibility of the valid primary hits (pixel order of render_frame)."""
    valid, position, normal, _, _ = cast
    pick = np.flatnonzero(valid.reshape(-1))
    return [shadow_visibility(root, data, position[pick], normalize(normal[pick]), light["position"], threads)
            for light in data["lights"] if light["type"] == 1]


def command_render(args):
    root, data = snapshot(args.snapshot)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cast = cast_primary_rays(root, data, args.threads)
    visibility = point_light_visibility(root, data, cast, args.threads) if args.visibility == "rays" else None
    main_model = SPECULAR_EXT if uses_specular_extension(data) or uses_clearcoat(data) else APPENDIX_B  # sheen-only: base
    controls = {"": main_model, **{name: control_model(name, main_model) for name in args.controls}}
    report = {"schema": 1, "tool": "tools/f15_reference.py", "tool_sha256": sha256_file(__file__), "model": MODEL,
              "snapshot_sha256": scene_digest(root), "frame": data.get("frame"),
              "sun_angular_radius": data.get("sun_angular_radius"), "sky": data["sky"],
              "lights": data["lights"], "visibility": args.visibility, "controls": {}, "deterministic": True}
    if visibility is not None:
        report["shadowed_hits"] = [int((~v).sum()) for v in visibility]
    for name, model in controls.items():
        image, masks, spheres, terms = render_frame(root, data, model, cast=cast, visibility=visibility)
        stem = "reference" if not name else f"control-{name}"
        write_pfm(out / f"{stem}.pfm", image)
        if not name:
            np.savez_compressed(out / "masks.npz", **masks)
            report.update(terms=terms, spheres=spheres, model_variant=model, width=image.shape[1],
                          height=image.shape[0], maximum_radiance=float(image.max()),
                          roi_pixels=int(masks["roi"].sum()), quadrature_mask_pixels=int(masks["narrow"].sum()))
        else:
            report["controls"][name] = {"model_variant": model, "file": f"{stem}.pfm"}
    if args.declared or args.declared_controls:
        # The declared environment model on the same cast: the image and its negative controls.
        declared = {"model": DECLARED, "file": "declared.pfm", "dfg_table": str(DFG_HEADER.relative_to(REPO)),
                    "dfg_table_sha256": sha256_file(DFG_HEADER), "controls": {}}
        image, masks, _, terms = render_frame(root, data, model=main_model, cast=cast, environment=DECLARED,
                                              visibility=visibility)
        write_pfm(out / "declared.pfm", image)
        declared["terms"] = terms
        for name in args.declared_controls:
            image, *_ = render_frame(root, data, model=main_model, cast=cast, environment=ALL_DECLARED_CONTROLS[name],
                                     visibility=visibility)
            write_pfm(out / f"control-{name}.pfm", image)
            declared["controls"][name] = {"model_variant": ALL_DECLARED_CONTROLS[name], "file": f"control-{name}.pfm"}
        report["declared"] = declared
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference": str(out / "reference.pfm"), "terms": report["terms"],
                      "roi_pixels": report["roi_pixels"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="reference image, masks and negative controls for one exported snapshot")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--controls", nargs="*", default=[], choices=ALL_CONTROLS)
    p.add_argument("--declared", action="store_true",
                   help="also render the engine's declared environment model (declared.pfm)")
    p.add_argument("--declared-controls", nargs="*", default=[], choices=sorted(ALL_DECLARED_CONTROLS),
                   help="negative controls of the declared model (implies --declared)")
    p.add_argument("--visibility", choices=["none", "rays"], default="none",
                   help="point-light visibility: none (legacy resolve lights) or Mitsuba shadow rays (DI)")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.set_defaults(function=command_render)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
