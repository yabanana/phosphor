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

KHR_materials_anisotropy (F15.3, ticket 11) is implemented from the ratified text and the README's informative GLSL
(research/khr-materials.md section 4); the section "KHR_materials_anisotropy" below documents the frame, the interpretation
choices (tangent source, handedness on mirrored instances, the exact and the declared environment) and the controls.

KHR_materials_iridescence (F15.3, ticket 12) is implemented from the ratified text (research/khr-materials.md section 8): the
thin-film Fresnel function of the specification is transcribed here from that text, never from the MSL, and checked by an
independent spectral (Airy, CIE) evaluation in tools/test_f15_reference.py; the section "KHR_materials_iridescence" below
documents the composition with the base Fresnel, the per-light-sample V.H rule, the exact and the declared environment and the
controls.

KHR_materials_diffuse_transmission (RC) / subsurface thin mode (F15.3, ticket 16) is implemented from the RC text
(research/khr-materials.md section 10.1) and ADR-0005/decision 10 (thin mode = diffuse transmission lobe); the section
"Subsurface thin mode / diffuse transmission" below states the hemisphere convention, the Fresnel weighting of the transmitted
lobe, the exact and the declared environment and the controls.

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

DEFAULT_THREADS = int(os.environ.get("F15_REFERENCE_THREADS", min(8, os.cpu_count() or 1)))  # Mitsuba ray-cast threads (env override)
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
    "anisotropy": "KHR_materials_anisotropy: f_spec = F D_burley V_aniso with alpha_b = alpha, alpha_t = mix(alpha, 1, strength^2), "
                  "lobe direction T' = cos(rot) T + sin(rot) B (T the Gram-Schmidt of the interpolated mesh tangent against the "
                  "shading normal, B = cross(N, T) w sign(det of the world matrix)), B' = cross(N, T'); informative height-correlated "
                  "V (no clamp, equal to Appendix B when isotropic); sun: the disk rule with alpha_min = alpha_b; exact environment "
                  "by quadrature of the anisotropic lobe (table over N.V and the azimuth of V against T'); declared environment "
                  "= split sum at an effective roughness (the bent-normal radiance fetch is invisible in a constant environment)",
    "iridescence": "KHR_materials_iridescence: thin-film Fresnel of the specification (Belcour-Barla analytic form, "
                   "sensitivity / phase / XYZ-to-Rec709 evaluation, outside ior 1, film ior = iridescenceIor, base_f0 = 0.04 "
                   "for the dielectric and baseColor for the metal) replaces the Schlick Fresnel with weight "
                   "s = iridescenceFactor (thickness 0 disables the film): dielectric spec = (1 - s) F_schlick + s F_irid, "
                   "diffuse = (1 - s) (1 - F_schlick) + s (1 - max F_irid); metal spec = (1 - s) F_schlick + s F_irid; "
                   "cos(theta1) = V.H per light sample; in the exact environment quadrature F_irid is evaluated inside the "
                   "integral at each V.H; declared environment: F_irid at N.V as a constant in the split sum (A + B), diffuse "
                   "1 - max of the specular albedo",
    "thin-subsurface": "KHR_materials_diffuse_transmission / subsurface thin mode: base = mix(diffuse_brdf(baseColor), "
                       "diffuse_btdf(color), w) under fresnel_mix: reflected diffuse (1 - w) (1 - F(V.H)) baseColor / pi for N.L > 0, "
                       "transmitted (1 - metallic) w (1 - F(V.H)) color / pi |N.L| for N.L < 0 (thin-walled, same point, no "
                       "refraction, H = normalize(V + L) of the literal fresnel_mix); specular lobes unchanged; sun at the disk "
                       "centre; exact environment (1 - w) Cd + w T with T the back-hemisphere quadrature; declared environment "
                       "transmitted weight (1 - E_d(N.V)) (1 - 0.16 N.V); GI at -N is engine-only (no GI in the fixtures)",
    "shadows": "none",
    "declared_environment": "(1 - metallic) * baseColor * (1 - (0.04 A + B)) + clamp(f0 A + B, 0, 1), (A, B) from the "
                            "engine DFG table (bilinear), f0 = mix(0.04, baseColor, metallic), roughness in [0.04, 1]; "
                            "clearcoat: (1 - c F_c(mu)) * that + c F_c(mu) * clamp(A_c + B_c, 0, 1), (A_c, B_c) at the "
                            "coat roughness; sheen: material (1 - max3(S) E_t) + S E_t with E_t from the engine sheen table "
                            "(src/renderer/sheen_table.h, bilinear) under the coat",
}

# The declared model plus the deliberately wrong variants used as negative controls.
# F15 closure (ADR-0007): the engine applies the Kulla-Conty multiple-scattering lobe by default ("ms": "kulla-conty", tools/f15_ms.py); the
# glTF Appendix B model without it is {**APPENDIX_B, "ms": "off"} (`--specular-ms off`, the control `ms-ignored`).
APPENDIX_B = {"alpha": "squared", "fresnel": "schlick", "weighting": "appendix-b", "ms": "kulla-conty"}
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
# F15.3 (ticket 11) negative controls of the anisotropy grids (deltas on the scene's main model; the anisotropy fixtures are
# anisotropy-only, so that is Appendix B).  `bitangent-flipped` (w ignored) and `rotation-flipped` give the same lobe on a
# mesh whose handedness is -1 everywhere: the mirrored fixture carries both handedness values and mirrored instances.
ANISO_CONTROLS = {
    "aniso-ignored": {"aniso": "ignored"},                          # strength 0: the isotropic base model
    "aniso-rotation-flipped": {"aniso": "rotation-flipped"},        # rotation clockwise instead of counter-clockwise
    "aniso-bitangent-flipped": {"aniso": "bitangent-flipped"},      # tangent handedness w ignored (B = cross(N, T))
    "aniso-alpha-swapped": {"aniso": "alpha-swapped"},              # alpha_t and alpha_b exchanged
    "aniso-mirror-unaware": {"aniso": "mirror-unaware"},            # the determinant of the world matrix ignored
    "aniso-d-normalised-by-ab": {"aniso": "d-normalised-by-ab"},    # D prefactor 1/(pi alpha_b^2): gains energy
}
# F15.3 (ticket 12) negative controls of the iridescence grids (deltas on the scene's main model; the iridescence fixtures are
# iridescence-only, so that is Appendix B).  `thickness-max` of the ticket text (thickness = maximum whatever the texture) is the
# specification itself on texture-free fixtures (thickness = mix(min, max, 1) = max): it cannot fail there and is OPEN for the
# textured case (ticket 22); `thickness-min` is its texture-free counterpart (the minimum used as the thickness).
IRID_CONTROLS = {
    "iridescence-ignored": {"irid": "ignored"},                      # factor 0: the base model (the engine before ticket 12)
    "iridescence-film-ior-ignored": {"irid": "film-ior-ignored"},    # film ior = outside ior 1.0 (no index contrast on top)
    "iridescence-film-ior-default": {"irid": "film-ior-default"},    # film ior = the default 1.3 whatever the material says
    "iridescence-at-nv": {"irid": "at-nv"},                          # N.V instead of V.H per light sample (Sample Renderer)
    "iridescence-thickness-min": {"irid": "thickness-min"},          # the minimum thickness used instead of the maximum
    "iridescence-diffuse-unweighted": {"irid": "diffuse-unweighted"},  # diffuse not scaled by 1 - max F_irid: gains energy
}
# F15.3 (ticket 16) negative controls of the thin-subsurface grids (deltas on the scene's main model, Appendix B).
THIN_CONTROLS = {
    "transmission-ignored": {"thin": "ignored"},                     # weight 0: the base model (the engine before ticket 16)
    "transmission-untinted": {"thin": "untinted"},                   # the transmission colour ignored (white)
    "transmission-front-only": {"thin": "front-only"},               # the BTDF evaluated on the FRONT hemisphere: back lights ignored
    "transmission-double-counted": {"thin": "double-counted"},       # diffuse reflection not reduced by w: gains energy
    "transmission-fresnel-ignored": {"thin": "fresnel-ignored"},     # transmitted part not weighted by 1 - F(V.H)
}
# F15.3 (texture closure) negative controls of the textured fixtures (deltas on the scene's main model; see the KHR texture fixtures section).
TEXTURE_CONTROLS = {
    "texture-v-flipped": {"tex": "v-flipped"},                                  # every texture sampled at 1 - v (the origin convention)
    "texture-coat-normal-ignored": {"tex": "coat-normal-ignored"},              # Nc = N
    "texture-coat-normal-y-flipped": {"tex": "coat-normal-y-flipped"},          # the green channel read with the OpenGL (+Y down) sign
    "texture-coat-factor-ignored": {"tex": "coat-factor-ignored"},
    "texture-coat-roughness-ignored": {"tex": "coat-roughness-ignored"},
    "texture-aniso-direction-ignored": {"tex": "aniso-direction-ignored"},      # the lobe along +X rotated by `rotation` only
    "texture-aniso-direction-y-flipped": {"tex": "aniso-direction-y-flipped"},
    "texture-aniso-strength-ignored": {"tex": "aniso-strength-ignored"},        # strength = the factor
    "texture-irid-thickness-ignored": {"tex": "irid-thickness-ignored"},        # thickness = the maximum
    "texture-irid-thickness-inverted": {"tex": "irid-thickness-inverted"},      # t -> 1 - t
    "texture-irid-factor-ignored": {"tex": "irid-factor-ignored"},
    "texture-sheen-color-ignored": {"tex": "sheen-color-ignored"},
    "texture-sheen-color-srgb-encoded": {"tex": "sheen-color-srgb-encoded"},    # the texels used as stored (no sRGB decode)
    "texture-sheen-roughness-ignored": {"tex": "sheen-roughness-ignored"},
    "texture-thin-weight-ignored": {"tex": "thin-weight-ignored"},
    "texture-thin-color-ignored": {"tex": "thin-color-ignored"},
    "texture-thin-color-srgb-encoded": {"tex": "thin-color-srgb-encoded"},
}
# F15.3 (ticket 19) negative controls of the combination grids (deltas on the scene's main model).  `additive-layering` removes the
# layer attenuation of every layered lobe the scene has at once (coat: base not scaled by 1 - c F_c; sheen: base not scaled by the
# albedo scaling; iridescence: diffuse not scaled by 1 - max F_irid; thin: diffuse reflection not reduced by w): lobes summed, the
# furnace control that must exceed the bound.  `order-swapped` puts the coat BELOW the sheen; `transmission-uncoated` lets the thin
# transmitted lobe escape the coat.
COMBO_CONTROLS = {
    "additive-layering": {"coat": "base-unattenuated", "sheen": "unscaled-base", "irid": "diffuse-unweighted", "thin": "double-counted"},
    "order-swapped": {"order": "swapped"},
    "transmission-uncoated": {"thin": "uncoated"},
    # F15.3 closure (ticket 16): thin + ior / film / sheen controls.  `default-fresnel`: the transmitted lobe keeps the default 0.04 Fresnel under an
    # extended dielectric or a film (the engine before the closure); `sheen-unscaled`: the sheen layer does not scale the transmitted lobe.
    "transmission-default-fresnel": {"thin": "default-fresnel"},
    "transmission-sheen-unscaled": {"thin": "sheen-unscaled"},
}
# F15 closure (--specular-ms, tools/f15_ms.py): deltas on the scene's main model, which then carries "ms": "kulla-conty".
MS_CONTROLS = {
    "ms-ignored": {"ms": "off"},      # no multiple-scattering compensation (today's base model)
    "ms-double": {"ms": "double"},    # F_ms doubled: the lobe gains energy
}
ALL_CONTROLS = sorted({**CONTROLS, **SHEEN_CONTROLS, **ANISO_CONTROLS, **IRID_CONTROLS, **THIN_CONTROLS, **COMBO_CONTROLS,
                       **TEXTURE_CONTROLS, **MS_CONTROLS})


def control_model(name, main_model):
    """The model of a literal-model control: the sheen and anisotropy controls modify the scene's main model, the older
    ones are complete model dicts of their own."""
    if name in SHEEN_CONTROLS:
        return {**main_model, **SHEEN_CONTROLS[name]}
    if name in ANISO_CONTROLS:
        return {**main_model, **ANISO_CONTROLS[name]}
    if name in IRID_CONTROLS:
        return {**main_model, **IRID_CONTROLS[name]}
    if name in THIN_CONTROLS:
        return {**main_model, **THIN_CONTROLS[name]}
    if name in COMBO_CONTROLS:
        return {**main_model, **COMBO_CONTROLS[name]}
    if name in TEXTURE_CONTROLS:
        return {**main_model, **TEXTURE_CONTROLS[name]}
    if name in MS_CONTROLS:
        return {**main_model, **MS_CONTROLS[name]}
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
    out = {"factor": coat["factor"][sel], "roughness": coat["roughness"][sel], "f0_ior": coat["f0_ior"][sel],
           "variant": coat["variant"]}
    for key in ("normal", "mu"):  # texture closure: the per-pixel coat normal Nc and |V.Nc| of a coat normal texture
        if key in coat:
            out[key] = coat[key][sel]
    return out


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
    n = coat.get("normal", n)  # texture closure: the coat normal Nc (the shading normal without a coat normal texture)
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
        mus = np.linspace(math.sqrt(MIN_NDOTV), 1.0, MU_NODES) ** 2
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
    if len(np.unique(rough)) > 16:
        raise ValueError("a textured coat roughness has no exact environment table: render the declared model "
                         "(--no-exact-environment)")
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
    mu = np.clip(coat.get("mu", n_dot_v), MIN_NDOTV, 1.0)  # texture closure: |V.Nc| with a coat normal texture
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
    """(mu grid, E) of D V N.L over the incident hemisphere for one alpha_g.  E is UNCLAMPED here: `sheen_albedo`
    interpolates first and clamps to SHEEN_E_CLAMP after (a clamped table has a kink where E crosses 1, and interpolating
    across it was off by 0.011 near N.V 0.05 at sheen roughness 0.1: measured against a 2048 x 1024 integration).  The mu
    nodes are quadratic, mu = s^2 with s uniform in [sqrt(MIN_NDOTV), 1], where E changes fastest.
    Incident quadrature (ticket 10, N.V < 0.005): the integrand D V cos is a narrow peak around the horizon and the azimuth of
    the view for a grazing view (D is largest for a horizontal half vector), so the nodes are dense there: cos_i = u^2 and
    phi = pi w^3 with (u, w) midpoint nodes, the Jacobian 2 u * 3 pi w^2 (the previous uniform nodes gave E 0.81 against 0.876
    at N.V 0.001, roughness 0.3: proven against a dense independent quadrature in tools/test_f15_reference.py)."""
    key = (round(float(alpha), 12), variant == "ashikhmin-visibility")
    if key not in _SHEEN_ALBEDO:
        n_u, n_w = SHEEN_QUADRATURE
        u = (np.arange(n_u) + 0.5) / n_u
        w = (np.arange(n_w) + 0.5) / n_w
        cos_i = (u * u)[:, None]
        sin_i = np.sqrt(np.maximum(1.0 - cos_i * cos_i, 0.0))
        phi = math.pi * w ** 3
        cell = 2.0 * (2.0 * u / n_u)[:, None] * (3.0 * math.pi * w ** 2 / n_w)[None, :]  # x2: the phi range [pi, 2 pi] mirrors
        s_nodes = np.linspace(math.sqrt(MIN_NDOTV), 1.0, MU_NODES)
        mus = s_nodes * s_nodes
        albedo = np.empty(MU_NODES)
        lambda_i = charlie_lambda(cos_i, alpha)
        for k, mu in enumerate(mus):
            sin_o = math.sqrt(max(0.0, 1.0 - mu * mu))
            wi_dot_wo = sin_i * np.cos(phi)[None, :] * sin_o + cos_i * mu
            n_dot_h = np.clip((cos_i + mu) / np.sqrt(np.maximum(2.0 + 2.0 * wi_dot_wo, 1e-30)), 0.0, 1.0)  # (wi + wo).n / |wi + wo|
            if variant == "ashikhmin-visibility":
                visibility_cos = cos_i / (4.0 * (cos_i + mu - cos_i * mu))
            else:
                visibility_cos = 1.0 / ((1.0 + charlie_lambda(mu, alpha) + lambda_i) * 4.0 * mu)  # V N.L with N.L cancelled
            albedo[k] = float(np.sum(charlie_d(n_dot_h, alpha) * visibility_cos * cell))
        _SHEEN_ALBEDO[key] = (mus, albedo)
    return _SHEEN_ALBEDO[key]


SHEEN_ROUGHNESS_NODES = 96  # roughness nodes of the textured-roughness grid (sheen_albedo_grid): r = r_min + (1 - r_min) q^2, q uniform
SHEEN_EXACT_UNIQUE = 16     # up to this many distinct roughness values the exact table of each one is used
_SHEEN_GRID = {}


def sheen_albedo_grid(variant="spec"):
    """(s nodes, r nodes, E) of the exact tables on a roughness-node grid (E unclamped, s = sqrt(N.V)): the lookup for a
    continuous per-pixel roughness (a roughness texture) interpolates it bilinearly instead of integrating one table per value."""
    key = (variant == "ashikhmin-visibility", variant == "roughness-linear")
    if key not in _SHEEN_GRID:
        q = np.linspace(0.0, 1.0, SHEEN_ROUGHNESS_NODES)
        r_nodes = SHEEN_MIN_ROUGHNESS + (1.0 - SHEEN_MIN_ROUGHNESS) * q * q
        tables = [sheen_albedo_table(float(sheen_alpha(r, variant)), variant) for r in r_nodes]
        _SHEEN_GRID[key] = (np.sqrt(tables[0][0]), r_nodes, np.stack([t[1] for t in tables], axis=0))
    return _SHEEN_GRID[key]


def sheen_albedo(mu, roughness, variant="spec"):
    """E per pixel (interpolated over N.V) for per-pixel sheen roughness values: one exact table per distinct roughness up to
    SHEEN_EXACT_UNIQUE of them, the roughness-node grid (bilinear in sqrt(N.V) and roughness) for a continuous field."""
    mu = np.clip(np.asarray(mu, dtype=np.float64), MIN_NDOTV, 1.0)
    rough = np.asarray(roughness, dtype=np.float64)
    out = np.empty(mu.shape)
    unique = np.unique(rough)
    if len(unique) > SHEEN_EXACT_UNIQUE:
        s_nodes, r_nodes, table = sheen_albedo_grid(variant)
        r = np.clip(rough, SHEEN_MIN_ROUGHNESS, 1.0)
        fr = np.sqrt((r - SHEEN_MIN_ROUGHNESS) / (1.0 - SHEEN_MIN_ROUGHNESS)) * (len(r_nodes) - 1)
        i = np.clip(np.floor(fr).astype(np.int64), 0, len(r_nodes) - 2)
        t = fr - i
        fs = (np.sqrt(mu) - s_nodes[0]) / (s_nodes[1] - s_nodes[0])
        j = np.clip(np.floor(fs).astype(np.int64), 0, len(s_nodes) - 2)
        f = fs - j
        out = ((table[i, j] * (1 - f) + table[i, j + 1] * f) * (1 - t) + (table[i + 1, j] * (1 - f) + table[i + 1, j + 1] * f) * t)
        return np.minimum(out, SHEEN_E_CLAMP)
    for r in unique:
        sel = rough == r
        mus, e = sheen_albedo_table(float(sheen_alpha(r, variant)), variant)
        out[sel] = np.interp(mu[sel], mus, e)
    return np.minimum(out, SHEEN_E_CLAMP)


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


# --- KHR_materials_anisotropy ----------------------------------------------------
#
# Written from the ratified text and the README's informative GLSL (research/khr-materials.md section 4), not from the MSL.
# With strength s, rotation phi, alpha = max(roughness^2, 0.002) of the base material:
#   alpha_b = alpha                        (perpendicular to the lobe direction)
#   alpha_t = mix(alpha, 1, s^2)           (along the lobe direction: the highlight is stretched along it)
#   D(h)    = 1 / (pi alpha_t alpha_b ((h.t)^2/alpha_t^2 + (h.b)^2/alpha_b^2 + (h.n)^2)^2)          (Burley, normative)
#   V       = 0.5 / (N.L |(alpha_t T.V, alpha_b B.V, N.V)| + N.V |(alpha_t T.L, alpha_b B.L, N.L)|)   (README, informative)
#   f_spec  = F D V        (F, the diffuse weighting and the metal/dielectric mix are the base model's, unchanged)
# The README clamps V to [0, 1]; this reference does NOT (the isotropic limit then equals Appendix B exactly, which the
# unit tests assert; V > 1 needs N.L ~ 0 with alpha ~ 0.002 and carries no energy).
#
# FRAME.  T = the world tangent: the mesh's own TANGENT attribute (PLY properties tangent_x/y/z, handedness_x of the snapshot,
# mesh version 2) barycentrically interpolated at the hit, transformed by the instance's world matrix as a direction and
# Gram-Schmidt orthogonalised against the shading normal (so T is exactly in the tangent plane).  B = cross(N, T) * w * det
# where w = sign of the interpolated handedness and det = the sign of the determinant of the world matrix: the bitangent is
# the world image of the model-space bitangent cross(N, T) w, so a MIRRORED instance (negative scale, INSTANCE_FLAG_MIRRORED)
# renders as the mirror image of the same surface (cross(N_w, T_w) alone would flip it).  The lobe direction is the README's
# T' = cos(rot) T + sin(rot) B (counter-clockwise from T in the T/B plane), B' = cross(N, T'): only T' and B' enter D and V, so
# the rotation is invisible at 0 and pi/2 for the sign (T' = +-T, +-B), and the handedness only through the sign of sin(rot).
#
# SUN.  The disk rule (`sun_quadrature`) keys on the narrowest alpha, alpha_b = alpha, unchanged: alpha >= 0.05 evaluates the
# lobe at the disk centre (the anisotropic lobe is only wider along T').
# ENVIRONMENT, exact: the directional albedo of F-weighted D V N.L under a constant radiance depends on N.V, on the azimuth of
# the view vector in the (T', B') frame and on (alpha_t, alpha_b); `aniso_albedo_table` integrates it by quadrature of the
# anisotropic half-vector distribution (the D cos(theta_h) CDF of the anisotropic GGX, midpoint rule), the diffuse term is the
# base model's.  DECLARED environment: the Sample Renderer fetches the radiance along a "bent normal" (the README's IBL
# guidance) and keeps the isotropic split-sum weights at the material roughness; in a CONSTANT environment the bent normal has no
# effect, so that method alone is anisotropy-blind there.  The declared model here keeps the bent-normal fetch (invisible) and
# evaluates the split sum (DFG table) at an effective isotropic roughness r_eff = sqrt(alpha_eff), alpha_eff = `ANISO_FIT`
# of (alpha_t, alpha_b) (the root mean square): a declared FIT whose error against the exact quadrature is measured and reported
# (tools/test_f15_reference.py).  Textured anisotropy (anisotropyTexture, word 2) is refused.

ANISOTROPY_FEATURE_BIT = 8  # MaterialFeature::Anisotropy
ANISOTROPY_RECORD = 1       # GPUMaterialExt1: strength word 0, rotation word 1, texture word 2
ANISO_VARIANTS = ("spec", "ignored", "rotation-flipped", "bitangent-flipped", "alpha-swapped", "mirror-unaware",
                  "d-normalised-by-ab")
ANISO_QUADRATURE = (192, 192)      # (u1 nodes, u2 nodes) of the half-vector quadrature, midpoint rule
ANISO_MU_NODES, ANISO_PHI_NODES = 65, 33   # table nodes over N.V in [MIN_NDOTV, 1] and the view azimuth in [0, pi/2]
ANISO_FIT = "rms"                  # alpha_eff = sqrt((alpha_t^2 + alpha_b^2) / 2) of the declared environment


def aniso_alphas(alpha, strength, variant="spec"):
    """(alpha_t, alpha_b) of the README rule; `ignored` (control) forces strength 0, `alpha-swapped` swaps the two."""
    alpha = np.asarray(alpha, dtype=np.float64)
    s = np.zeros_like(np.asarray(strength, dtype=np.float64)) if variant == "ignored" else np.asarray(strength, dtype=np.float64)
    at, ab = alpha + (1.0 - alpha) * s * s, alpha.copy()
    if variant == "alpha-swapped":
        at, ab = ab, at
    return at, ab


def take_aniso(aniso, sel):
    """The per-pixel arrays of `aniso` restricted to the pixel indices `sel` (None stays None)."""
    if aniso is None:
        return None
    return {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in aniso.items()}


def aniso_axes(aniso, n):
    """(T', B') unit lobe axes per pixel in the tangent plane of the shading normals `n`: T' = cos(rot) T + sin(rot) B with
    B = cross(N, T) s_eff, s_eff = w det (the world image of the model-space bitangent); B' = cross(N, T').  Controls:
    `rotation-flipped` negates the angle, `bitangent-flipped` ignores the handedness w (s_eff = det), `mirror-unaware`
    ignores the determinant (s_eff = w)."""
    variant = aniso["variant"]
    w, det = aniso["w"], aniso["det"]
    sign = det if variant == "bitangent-flipped" else (w if variant == "mirror-unaware" else w * det)
    b0 = np.cross(n, aniso["tangent"]) * sign[:, None]
    rotation = -aniso["rotation"] if variant == "rotation-flipped" else aniso["rotation"]
    t = normalize(np.cos(rotation)[:, None] * aniso["tangent"] + np.sin(rotation)[:, None] * b0)
    return t, normalize(np.cross(n, t))


def aniso_dv(n, v, l, alpha, aniso):
    """D V per pixel (no Fresnel, no cosine) of the anisotropic lobe at world unit vectors; n.l <= 0 gives V of N.L = 0."""
    at, ab = aniso_alphas(alpha, aniso["strength"], aniso["variant"])
    t, b = aniso_axes(aniso, n)
    h = normalize(v + l)
    nh = np.clip(dot(n, h), 0.0, 1.0)
    th, bh = dot(t, h), dot(b, h)
    q = (th / at) ** 2 + (bh / ab) ** 2 + nh * nh
    d = 1.0 / (math.pi * at * ab * q * q)
    if aniso["variant"] == "d-normalised-by-ab":
        d = d * at / ab  # control: the prefactor 1/(pi ab^2): gains energy by alpha_t / alpha_b
    nv = np.maximum(dot(n, v), MIN_NDOTV)
    nl = np.maximum(dot(n, l), 0.0)
    ggx_v = nl * np.sqrt((at * dot(t, v)) ** 2 + (ab * dot(b, v)) ** 2 + nv * nv)
    ggx_l = nv * np.sqrt((at * dot(t, l)) ** 2 + (ab * dot(b, l)) ** 2 + nl * nl)
    return d * (0.5 / np.maximum(ggx_v + ggx_l, 1e-12))


def specular_dv(n, v, l, alpha, aniso=None):
    """D V per pixel: the base isotropic GGX / Smith product, or the anisotropic lobe."""
    if aniso is not None:
        return aniso_dv(n, v, l, alpha, aniso)
    h = normalize(v + l)
    return ggx_d(np.clip(dot(n, h), 0.0, 1.0), alpha) * smith_vis(np.maximum(dot(n, v), MIN_NDOTV),
                                                                  np.maximum(dot(n, l), 0.0), alpha)


_ANISO_ALBEDO = {}


def _aniso_spec_integrals(mu, phi_v, at, ab, variant="spec"):
    """(A, B): integrals of D V cos (1 - w) and D V cos w over the incident hemisphere (w the Schlick weight of V.H) for a
    view at cos(theta) = mu and azimuth phi_v from the lobe axis T' in the (T', B', N) frame.  The anisotropic half-vector
    distribution D cos(theta_h) is sampled by its CDF (azimuth atan2(ab sin 2 pi u2, at cos 2 pi u2), tan^2 theta_h =
    u1 / (1 - u1) / (cos^2 / at^2 + sin^2 / ab^2)), integral = mean(V cos_i 4 (wo.h) / cos_h g)."""
    n1, n2 = ANISO_QUADRATURE
    u1 = (np.arange(n1) + 0.5) / n1
    u2 = (np.arange(n2) + 0.5) / n2
    phi_h = np.arctan2(ab * np.sin(2 * math.pi * u2), at * np.cos(2 * math.pi * u2))
    tan2 = (u1[:, None] / (1.0 - u1[:, None])) / ((np.cos(phi_h) / at) ** 2 + (np.sin(phi_h) / ab) ** 2)[None, :]
    cos_h = 1.0 / np.sqrt(1.0 + tan2)
    sin_h = np.sqrt(np.maximum(0.0, 1.0 - cos_h * cos_h))
    h = np.stack([sin_h * np.cos(phi_h)[None, :], sin_h * np.sin(phi_h)[None, :], cos_h], axis=-1)
    sin_v = math.sqrt(max(0.0, 1.0 - mu * mu))
    wo = np.array([sin_v * math.cos(phi_v), sin_v * math.sin(phi_v), mu])
    vh = h @ wo
    wi = 2.0 * vh[..., None] * h - wo
    valid = (wi[..., 2] > 0) & (vh > 0)
    wi_z = np.maximum(wi[..., 2], 0.0)
    ggx_v = wi_z * np.sqrt((at * wo[0]) ** 2 + (ab * wo[1]) ** 2 + mu * mu)
    ggx_l = mu * np.sqrt((at * wi[..., 0]) ** 2 + (ab * wi[..., 1]) ** 2 + wi_z ** 2)
    vis = 0.5 / np.maximum(ggx_v + ggx_l, 1e-12)
    base = np.where(valid, vis * wi_z * 4.0 * vh / cos_h, 0.0)
    if variant == "d-normalised-by-ab":
        base = base * at / ab
    w = schlick_weight(vh)
    return float(np.mean(base * (1 - w))), float(np.mean(base * w))


def aniso_albedo_table(at, ab, variant="spec"):
    """(mu grid, phi grid, A[mu, phi], B[mu, phi]) of one (alpha_t, alpha_b)."""
    key = (round(float(at), 12), round(float(ab), 12), variant)
    if key not in _ANISO_ALBEDO:
        mus = np.linspace(MIN_NDOTV, 1.0, ANISO_MU_NODES)
        phis = np.linspace(0.0, 0.5 * math.pi, ANISO_PHI_NODES)
        a = np.empty((len(mus), len(phis)))
        b = np.empty_like(a)
        for i, mu in enumerate(mus):
            for j, phi in enumerate(phis):
                a[i, j], b[i, j] = _aniso_spec_integrals(float(mu), float(phi), float(at), float(ab), variant)
        _ANISO_ALBEDO[key] = (mus, phis, a, b)
    return _ANISO_ALBEDO[key]


_ANISO_HIST = {}


def _aniso_vh_histogram(mu, phi_v, at, ab, variant="spec"):
    """Histogram over V.H (IRID_BINS bins, linear splitting) of the weights of `_aniso_spec_integrals` (the same half-vector quadrature):
    sum_k hist[k] f(k / (bins - 1)) is the quadrature of f(V.H) D V cos, so hist . (1 - w) = A and hist . w = B (tested)."""
    n1, n2 = ANISO_QUADRATURE
    u1 = (np.arange(n1) + 0.5) / n1
    u2 = (np.arange(n2) + 0.5) / n2
    phi_h = np.arctan2(ab * np.sin(2 * math.pi * u2), at * np.cos(2 * math.pi * u2))
    tan2 = (u1[:, None] / (1.0 - u1[:, None])) / ((np.cos(phi_h) / at) ** 2 + (np.sin(phi_h) / ab) ** 2)[None, :]
    cos_h = 1.0 / np.sqrt(1.0 + tan2)
    sin_h = np.sqrt(np.maximum(0.0, 1.0 - cos_h * cos_h))
    h = np.stack([sin_h * np.cos(phi_h)[None, :], sin_h * np.sin(phi_h)[None, :], cos_h], axis=-1)
    sin_v = math.sqrt(max(0.0, 1.0 - mu * mu))
    wo = np.array([sin_v * math.cos(phi_v), sin_v * math.sin(phi_v), mu])
    vh = h @ wo
    wi = 2.0 * vh[..., None] * h - wo
    valid = (wi[..., 2] > 0) & (vh > 0)
    wi_z = np.maximum(wi[..., 2], 0.0)
    ggx_v = wi_z * np.sqrt((at * wo[0]) ** 2 + (ab * wo[1]) ** 2 + mu * mu)
    ggx_l = mu * np.sqrt((at * wi[..., 0]) ** 2 + (ab * wi[..., 1]) ** 2 + wi_z ** 2)
    vis = 0.5 / np.maximum(ggx_v + ggx_l, 1e-12)
    weight = np.where(valid, vis * wi_z * 4.0 * vh / cos_h, 0.0)
    if variant == "d-normalised-by-ab":
        weight = weight * at / ab
    return _vh_histogram(vh, weight / (n1 * n2))


def aniso_vh_histograms(at, ab, variant="spec"):
    """(mu grid, phi grid, hist[mu, phi, bins]) of one (alpha_t, alpha_b): the V.H histograms of the anisotropic quadrature
    (exact environment of iridescence + anisotropy, ticket 19)."""
    key = (round(float(at), 12), round(float(ab), 12), variant)
    if key not in _ANISO_HIST:
        mus = np.linspace(MIN_NDOTV, 1.0, ANISO_MU_NODES)
        phis = np.linspace(0.0, 0.5 * math.pi, ANISO_PHI_NODES)
        table = np.empty((len(mus), len(phis), IRID_BINS))
        for i, mu in enumerate(mus):
            for j, phi in enumerate(phis):
                table[i, j] = _aniso_vh_histogram(float(mu), float(phi), float(at), float(ab), variant)
        _ANISO_HIST[key] = (mus, phis, table)
    return _ANISO_HIST[key]


def aniso_film_spec(film_ior, thickness, f0, mu, aniso_env):
    """Exact specular integral (n, 3) of the thin-film Fresnel under the anisotropic lobe: the integral of D V cos F_irid(V.H) for the
    pixels of one film (ior, thickness nm, base f0 (3,)) at per-pixel N.V `mu` and the (alpha_t, alpha_b, azimuth) of `aniso_env`."""
    out = np.empty((len(mu), 3))
    bins = np.linspace(0.0, 1.0, IRID_BINS)
    film = thin_film_fresnel(bins, np.full(IRID_BINS, film_ior), np.full(IRID_BINS, thickness),
                             np.broadcast_to(np.asarray(f0, dtype=np.float64), (IRID_BINS, 3)))
    pairs = np.stack([np.round(aniso_env["at"], 12), np.round(aniso_env["ab"], 12)], axis=1)
    for at, ab in np.unique(pairs, axis=0):
        pick = np.flatnonzero((pairs[:, 0] == at) & (pairs[:, 1] == ab))
        mus, phis, hist = aniso_vh_histograms(at, ab, aniso_env["variant"])
        values = hist @ film
        for c in range(3):
            out[pick, c] = _bilinear(mus, phis, values[..., c], np.clip(mu[pick], MIN_NDOTV, 1.0), aniso_env["phi"][pick])
    return out


def _bilinear(mus, phis, table, mu, phi):
    """Bilinear interpolation of table[mu, phi] at per-pixel (mu, phi), both clamped to the grid."""
    x = np.clip((mu - mus[0]) / (mus[-1] - mus[0]), 0.0, 1.0) * (len(mus) - 1)
    y = np.clip(phi / phis[-1], 0.0, 1.0) * (len(phis) - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), len(mus) - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), len(phis) - 2)
    fx, fy = x - x0, y - y0
    return ((table[x0, y0] * (1 - fx) + table[x0 + 1, y0] * fx) * (1 - fy) +
            (table[x0, y0 + 1] * (1 - fx) + table[x0 + 1, y0 + 1] * fx) * fy)


def aniso_environment_inputs(aniso, roughness, n, v):
    """Per-pixel {"at", "ab", "phi", "variant"} of the exact environment: the alphas and the azimuth of V in the lobe frame
    folded into [0, pi/2] (the lobe is even in both axes)."""
    at, ab = aniso_alphas(effective_alpha(roughness), aniso["strength"], aniso["variant"])
    t, b = aniso_axes(aniso, n)
    phi = np.arctan2(np.abs(dot(v, b)), np.abs(dot(v, t)))
    return {"at": at, "ab": ab, "phi": phi, "variant": aniso["variant"]}


def aniso_spec_albedo(mu, env):
    """(A, B) per pixel from the tables: pixels with alpha_t = alpha_b use the isotropic table of the base model."""
    mu = np.clip(mu, MIN_NDOTV, 1.0)
    a = np.empty(len(mu))
    b = np.empty(len(mu))
    pairs = np.stack([np.round(env["at"], 12), np.round(env["ab"], 12)], axis=1)
    for at, ab in np.unique(pairs, axis=0):
        sel = np.flatnonzero((pairs[:, 0] == at) & (pairs[:, 1] == ab))
        if at == ab and env["variant"] != "d-normalised-by-ab":
            t = environment_table(math.sqrt(ab), APPENDIX_B)
            a[sel], b[sel] = np.interp(mu[sel], t["mu"], t["A"]), np.interp(mu[sel], t["mu"], t["B"])
            continue
        mus, phis, ta, tb = aniso_albedo_table(at, ab, env["variant"])
        a[sel] = _bilinear(mus, phis, ta, mu[sel], env["phi"][sel])
        b[sel] = _bilinear(mus, phis, tb, mu[sel], env["phi"][sel])
    return a, b


def declared_aniso_alpha(at, ab, kind="spec"):
    """alpha_eff of the declared environment fit: the root mean square sqrt((alpha_t^2 + alpha_b^2) / 2) (`spec`; of the
    simple isotropic fits it has the smallest sphere-weighted error against the exact quadrature: 1.1 % mean at strength 0.5,
    6.6 % at strength 1, measured over N.V >= 0.1 and all azimuths at alpha_b 0.09), the isotropic base alpha_b (`ignored`,
    the engine before ticket 11) or the stretched alpha_t (`alpha-t`, a control)."""
    if kind == "ignored":
        return np.asarray(ab, dtype=np.float64)
    if kind == "alpha-t":
        return np.asarray(at, dtype=np.float64)
    return np.sqrt(0.5 * (np.asarray(at, dtype=np.float64) ** 2 + np.asarray(ab, dtype=np.float64) ** 2))


# --- KHR_materials_iridescence -------------------------------------------------
#
# Source: the ratified text and the README's informative GLSL (research/khr-materials.md section 8), transcribed below into numpy.
# * strength s = iridescenceFactor (x iridescenceTexture.r: textures are refused), thickness = mix(min, max, thicknessTexture.g)
#   in nm = the MAXIMUM when there is no thickness texture; the Minimum may exceed the Maximum.
# * Thickness 0 disables the film (convention of the viewers and of the ticket: the closed form at d = 0 is close to, not equal to,
#   the Schlick Fresnel; the strength is forced to 0 there so the material is EXACTLY the base model, asserted in the tests).
# * Composition (README "iridescent_dielectric_layer" / "iridescent_conductor_layer", outside ior 1, base ior 1.5 so base_f0 = 0.04):
#     dielectric: mix(fresnel_mix(base, specular), rgb_mix(base, specular, F_irid), s), rgb_mix(b, sp, a) = (1 - max(a)) b + a sp
#                 -> spec weight (1 - s) F_schlick + s F_irid (per channel), diffuse weight (1 - s) (1 - F_schlick) + s (1 - max F_irid)
#     metal:      mix(conductor_fresnel(base_f0, specular), specular * F_irid(base_f0 = baseColor), s)
#   F_irid = thin-film Fresnel evaluated at cos(theta1) = V.H of the light sample (direct sun: the disk rule keys on alpha,
#   unchanged, so the V.H of every disk sample; point lights: V.H of the light direction).
# * Exact environment: the quadrature integrates F_irid INSIDE the integral at each V.H (decision 06: the declared
#   engine model evaluates it ONCE at N.V; the exact reference is the one that integrates at V.H).  Implemented through
#   histograms of the quadrature weights over V.H (shared by every material of one alpha): integral = histogram . F(bin).
#   DECLARED environment (decision 06 "thin-film Fresnel at N.V"): the split sum with F_irid(N.V) constant over the lobe,
#   specular albedo F_irid(N.V) clamp(A + B, 0, 1) at the material roughness, diffuse weight 1 - max of that albedo (the rule
#   the F13 composite applies to every extended lobe, "extension_diffuse").
# * Refused: textures (words 7, 8: the thickness texture case is OPEN for ticket 22), combinations with other features.

IRIDESCENCE_FEATURE_BIT = 16   # MaterialFeature::Iridescence
IRIDESCENCE_RECORD = 1         # GPUMaterialExt1: factor word 3, ior 4, thicknessMin 5, thicknessMax 6, factor tex 7, thickness tex 8
IRID_OUTSIDE_IOR = 1.0
IRID_DEFAULT_IOR = 1.3
IRID_VARIANTS = ("spec", "ignored", "film-ior-ignored", "film-ior-default", "at-nv", "thickness-min", "diffuse-unweighted")
IRID_BINS = 2048               # V.H bins of the environment histograms (linear splitting)
IRID_MU_NODES = 129            # N.V nodes of the iridescent environment tables

# The specification's constants, verbatim (evalSensitivity and XYZ_TO_REC709).
_SENS_VAL = np.array([5.4856e-13, 4.4201e-13, 5.2481e-13])
_SENS_POS = np.array([1.6810e+06, 1.7953e+06, 2.2084e+06])
_SENS_VAR = np.array([4.3278e+09, 9.3046e+09, 6.6121e+09])
# GLSL mat3(...) is column-major: rgb = x * (3.2404542, -0.9692660, 0.0556434) + y * (-1.5371385, 1.8760108, -0.2040259) + z * (...)
_XYZ_TO_REC709 = np.array([[3.2404542, -1.5371385, -0.4985314],
                           [-0.9692660, 1.8760108, 0.0415560],
                           [0.0556434, -0.2040259, 1.0572252]])


def ior_to_fresnel0(transmitted, incident):
    """IorToFresnel0 of the README: ((transmitted - incident) / (transmitted + incident))^2."""
    return ((transmitted - incident) / (transmitted + incident)) ** 2


def fresnel0_to_ior(f0):
    """Fresnel0ToIor of the README: (1 + sqrt(f0)) / (1 - sqrt(f0))."""
    root = np.sqrt(f0)
    return (1.0 + root) / (1.0 - root)


def schlick_f(f0, cos_theta):
    """F_Schlick of the README: f0 + (1 - f0) (1 - cos)^5."""
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5


def eval_sensitivity(opd, shift):
    """evalSensitivity(OPD, shift) of the README: the Rec709 RGB sensitivity of an optical path difference `opd` (nm, (n,))
    with the per-channel phase `shift` ((n, 3)); the matrix product is the GLSL `XYZ_TO_REC709 * xyz`."""
    phase = (2.0 * math.pi * np.asarray(opd, dtype=np.float64) * 1e-9)[..., None]
    xyz = _SENS_VAL * np.sqrt(2.0 * math.pi * _SENS_VAR) * np.cos(_SENS_POS * phase + shift) * np.exp(-(phase * phase) * _SENS_VAR)
    xyz[..., 0] += (9.7470e-14 * math.sqrt(2.0 * math.pi * 4.5282e+09) * np.cos(2.2399e+06 * phase[..., 0] + shift[..., 0]) *
                    np.exp(-4.5282e+09 * phase[..., 0] ** 2))
    xyz = xyz / 1.0685e-7
    return xyz @ _XYZ_TO_REC709.T


def thin_film_fresnel(cos_theta1, film_ior, thickness, base_f0, outside_ior=IRID_OUTSIDE_IOR, phase_shifts=True):
    """The specification's iridescent Fresnel (README "Iridescence Fresnel", Belcour-Barla 2017): (n, 3).

    cos_theta1 (n,) in [0, 1]; film_ior (n,); thickness (n,) in nm; base_f0 (n, 3).  The base ior comes from f0 + 0.0001 (the
    README), kept below 0.9999 so that a white metal (f0 = 1) gives the finite ior 4e4 instead of a division by zero.
    `phase_shifts` False is the test-only wrong model whose reflection phases are all zero."""
    cos1 = np.clip(np.asarray(cos_theta1, dtype=np.float64), 0.0, 1.0)
    film = np.asarray(film_ior, dtype=np.float64)
    base_f0 = np.asarray(base_f0, dtype=np.float64)
    r0 = ior_to_fresnel0(film, outside_ior)
    r12 = schlick_f(r0, cos1)
    r21 = r12
    t121 = 1.0 - r12
    base_ior = fresnel0_to_ior(np.minimum(base_f0 + 0.0001, 0.9999))
    r1 = ior_to_fresnel0(base_ior, film[..., None])
    sin_theta2_sq = (outside_ior / film) ** 2 * (1.0 - cos1 * cos1)
    cos_theta2_sq = 1.0 - sin_theta2_sq
    total_reflection = cos_theta2_sq < 0.0
    cos2 = np.sqrt(np.maximum(cos_theta2_sq, 0.0))
    r23 = schlick_f(r1, cos2[..., None])
    opd = 2.0 * film * np.asarray(thickness, dtype=np.float64) * cos2
    phi12 = np.where(film < outside_ior, math.pi, 0.0)
    phi21 = math.pi - phi12
    phi23 = np.where(base_ior < film[..., None], math.pi, 0.0)
    phi = phi21[..., None] + phi23
    if not phase_shifts:
        phi = np.zeros_like(phi)
    r123 = np.clip(r12[..., None] * r23, 1e-5, 0.9999)
    root123 = np.sqrt(r123)
    rs = (t121 * t121)[..., None] * r23 / (1.0 - r123)
    c0 = r21[..., None] + rs
    total = c0
    cm = rs - t121[..., None]
    for m in (1, 2):
        cm = cm * root123
        total = total + cm * (2.0 * eval_sensitivity(m * opd, m * phi))
    return np.where(total_reflection[..., None], 1.0, np.maximum(total, 0.0))


def take_irid(irid, sel):
    """The per-pixel arrays of `irid` restricted to the pixel indices `sel` (None stays None)."""
    if irid is None:
        return None
    return {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in irid.items()}


def irid_film_ior(irid):
    """Film ior per pixel under the variant (controls: the outside ior 1.0, or the default 1.3)."""
    variant = irid["variant"]
    if variant == "film-ior-ignored":
        return np.full_like(irid["ior"], IRID_OUTSIDE_IOR)
    if variant == "film-ior-default":
        return np.full_like(irid["ior"], IRID_DEFAULT_IOR)
    return irid["ior"]


def irid_thickness(irid):
    """Thickness (nm) per pixel: thickness = mix(min, max, 1) = max without a thickness texture (control: the minimum)."""
    return irid["thickness_min"] if irid["variant"] == "thickness-min" else irid["thickness"]


def irid_strength(irid):
    """s per pixel: iridescenceFactor, 0 under `ignored`, and 0 where the thickness is 0 (the film is disabled)."""
    s = np.zeros_like(irid["strength"]) if irid["variant"] == "ignored" else irid["strength"]
    return np.where(irid_thickness(irid) > 0.0, s, 0.0)


def irid_fresnel(irid, base_f0, cos_theta1):
    """F_irid (n, 3) of the pixels at `cos_theta1` (V.H per light sample; the `at-nv` control uses N.V)."""
    cos1 = irid["nv"] if irid["variant"] == "at-nv" else cos_theta1
    return thin_film_fresnel(cos1, irid_film_ior(irid), irid_thickness(irid), base_f0)


def irid_base_f0(base, dielectric):
    """The dielectric base_f0 handed to the thin film (n, 3): 0.04 without the ior/specular extension; with it (ticket 19, the
    README's `iridescent_fresnel(..., specular_weight * base_f0, ...)` with base_f0 = min(((1 - ior) / (1 + ior))^2 * specularColor, 1)) the
    scaled f0 `weight * f0` of the dielectric (the `ignored` control: 0.04)."""
    if dielectric is None:
        return np.full(base.shape, DIELECTRIC_F0)
    zero = np.zeros((base.shape[0], 1))
    return dielectric_fresnel(dielectric["f0"], dielectric["weight"], zero, dielectric["variant"])[0]


def irid_lobe_weights(base, metallic, v_dot_h, irid, model=APPENDIX_B, dielectric=None):
    """(diffuse multiplier per channel, specular multiplier per channel) of an iridescent material at one V.H (see above).

    `dielectric` (ticket 19, iridescence + ior/specular): the (1 - s) part keeps the extended Fresnel `weight * fr` and its
    diffuse weight 1 - max(F), the film takes the scaled f0 (irid_base_f0).  For a fractional strength the specification's two
    forms (mix of the two layers, or one rgb_mix with the mixed Fresnel) differ in the diffuse weight only; the fixtures of
    ticket 19 use s = 1 where they coincide, and this function implements the mix of the diffuse weights (the form of the
    iridescence-only text) for both."""
    if model["weighting"] != "appendix-b":
        raise ValueError("iridescence is modelled on the Appendix B weighting only")
    w = schlick_weight(v_dot_h, model)[..., None]
    m = np.asarray(metallic, dtype=np.float64)[..., None]
    s = irid_strength(irid)[..., None]
    if dielectric is None:
        f_dielectric = DIELECTRIC_F0 + (1 - DIELECTRIC_F0) * w
    else:
        f_dielectric, _ = dielectric_fresnel(dielectric["f0"], dielectric["weight"], w, dielectric["variant"])
    f_metal = base + (1 - base) * w
    cos1 = np.asarray(v_dot_h, dtype=np.float64)
    film_dielectric = irid_fresnel(irid, irid_base_f0(base, dielectric), cos1)
    film_metal = irid_fresnel(irid, base, cos1)
    film_weight = np.ones_like(f_dielectric) if irid["variant"] == "diffuse-unweighted" else \
        1.0 - np.max(film_dielectric, axis=-1, keepdims=True)
    diffuse = (1 - m) * ((1 - s) * (1.0 - np.max(f_dielectric, axis=-1, keepdims=True)) + s * film_weight)
    specular = (1 - m) * ((1 - s) * f_dielectric + s * film_dielectric) + m * ((1 - s) * f_metal + s * film_metal)
    return diffuse, specular


# --- Subsurface thin mode / diffuse transmission (F15.3, ticket 16) -------------------------------------------------------
#
# Source: KHR_materials_diffuse_transmission (Release Candidate), research/khr-materials.md section 10.1, normative text
#   dielectric_brdf = fresnel_mix(ior = 1.5,
#       base  = mix(diffuse_brdf(color = baseColor), diffuse_btdf(color = diffuseTransmissionColor), diffuseTransmission),
#       layer = specular_brdf(alpha = roughness^2))
#   diffuse_brdf(c) = c / pi if V and L are on the same hemisphere of N, else 0;   diffuse_btdf(c) = c / pi if on opposite ones.
# Phosphor's thin mode (ADR-0005, decision 10) maps weight -> diffuseTransmission and the subsurface colour ->
# diffuseTransmissionColor (the loader maps both glTF encodings and the description file to the same SubsurfaceBlock).
#
# Conventions and interpretations (every one is reported to the orchestrator):
# * Hemispheres: relative to the interpolated SHADING normal N; the camera side (N.V > 0 for a visible pixel) is the front, "back"
#   is the -N side.  Thin-walled: the transmitted ray leaves at the same point, no refraction, no offset, no thickness; a light with
#   N.L < 0 lights the front surface through the diffuse BTDF.
# * fresnel_mix is the Appendix B form f = (1 - F) base + F layer with F = Schlick(f0 = 0.04, |V.H|) and H = normalize(V + L) of the
#   literal text, ALSO for the transmitted lobe (the RC says "the specular reflection BSDF and Fresnel weighting are not affected";
#   the glTF Sample Renderer does the same: one F at V.H of V and L multiplies the whole base).  For L on the back hemisphere
#   V.H = sqrt((1 + V.L) / 2) < 0.71, so the transmitted lobe is attenuated by (1 - F) with F up to 1 at V = -L (a light straight
#   behind the camera ray transmits nothing: the albedo T(1) is 0.8075).  A degenerate H (|V + L|^2 <= 1e-20) gives 0 (engine rule).
#   ORCHESTRATOR: confirm this literal reading (alternative: (1 - F(N.V)), which the text does not say).
# * Reflected part:  f_r = (1 - w) (1 - metallic) (1 - F(V.H)) baseColor / pi   for N.L > 0 (the specular lobe is unchanged);
#   transmitted part: f_t = w (1 - metallic) (1 - F(V.H)) color / pi             for N.L < 0, radiance f_t |N.L| E.  Metals carry no
#   diffuse lobe, so no transmission.  The transmission colour is NOT multiplied by baseColor or by w twice (the README's
#   text/formula inconsistency of the colour factor is resolved as in research section 10.1: formula wins, the mix carries w once).
# * Sun: the diffuse and the transmitted term are point-evaluated at the disk centre (smooth; the engine's diffuse rule).
# * Environment (constant L): exact = (1 - w) c Cd(N.V) + w color T(N.V) + F0 A + B with Cd the front diffuse integral of
#   environment_table and T the back-hemisphere one (`thin_transmission_table`, the same (1 - F(V.H)) cos / pi integrand over L with
#   N.L < 0: T(1) = 0.8075, T(0) = 0.933).  T <= Cd everywhere (ratio 1 at grazing, 0.841 at N.V = 1), so reflected + transmitted never
#   exceeds the unextended model: "energy moved, not added".  Declared (engine) model: the transmitted weight is the declared
#   front diffuse weight times the ratio r(mu) = T / Cd ~ 1 - 0.16 mu (`THIN_RATIO_SLOPE`; fit error 1.4e-3, tested), i.e.
#   w color (1 - E_d(mu)) (1 - 0.16 mu) with E_d = clamp(0.04 A + B, 0, 1), and the declared front diffuse is scaled by (1 - w).
# * The GI term at -N (decision 06) and the secondary-hit base reduction are engine-only: the fixtures carry no GI, no RT hits.
# * Shadow rays of the point-light DI keys start on the side of the light (offset 1e-4 * sign(N.L) * N): a thin carrier does not
#   occlude its own transmitted light.
# * Refused (ValueError): volumetric mode (ticket 17), subsurface textures, non-zero anisotropy (the draft's forward/backward
#   scatter split is not modelled), any other feature bit.  Radius/radiusScale are volumetric-only and ignored in thin mode.

SUBSURFACE_FEATURE_BIT = 128  # MaterialFeature::Subsurface
SUBSURFACE_RECORD = 2         # GPUMaterialExt2: mode 0, weight 1, colour 2-4, radius 5, radiusScale 6-8, anisotropy 9, textures 10-12
THIN_VARIANTS = ("spec", "ignored", "untinted", "front-only", "double-counted", "fresnel-ignored")
THIN_RATIO_SLOPE = 0.16


def take_thin(thin, sel):
    """The per-pixel arrays of `thin` restricted to the pixel indices `sel` (None stays None)."""
    if thin is None:
        return None
    return {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in thin.items()}


def thin_weight(thin):
    """w per pixel (0 under `ignored`)."""
    return np.zeros_like(thin["weight"]) if thin["variant"] == "ignored" else thin["weight"]


def thin_color(thin):
    """Transmission colour (n, 3) (white under `untinted`)."""
    return np.ones_like(thin["color"]) if thin["variant"] == "untinted" else thin["color"]


def thin_reflect_scale(thin):
    """Factor of the reflected diffuse: 1 - w (1 under `double-counted`: the energy is not taken away)."""
    return np.ones_like(thin["weight"]) if thin["variant"] == "double-counted" else 1.0 - thin_weight(thin)


def thin_transmitted(thin, metallic, n, v, l, model=APPENDIX_B, dielectric=None, irid=None, base=None):
    """Transmitted radiance factor f_t |N.L| (n, 3) of the thin lobe for unit incident radiance (see the section comment).

    `front-only` (control): the same lobe evaluated on the FRONT hemisphere (cosine max(N.L, 0)): back lights are ignored and a
    front light gets an extra w color / pi term.  `fresnel-ignored` (control): no 1 - F(V.H) factor.

    `dielectric` / `irid` (F15.3 closure, ticket 16): thin + ior/specular and/or iridescence.  The transmitted lobe sits inside the base of the same
    fresnel_mix as the diffuse reflection (glTF Sample Renderer, research/khr-materials.md lines 453 and 460), so it takes the diffuse weight of the
    reflected lobe, (1 - metallic) (1 - max3 F(V.H)) with the extended dielectric Fresnel or the film Fresnel (`lobe_weights`)."""
    n_dot_l = dot(n, l)
    h = normalize(v + l)
    degenerate = dot(v + l, v + l) <= 1e-20
    if (dielectric is not None or irid is not None) and thin["variant"] != "default-fresnel":
        diffuse_w, _ = lobe_weights(np.ones(np.broadcast(n_dot_l, n_dot_l).shape + (3,)) if base is None else base, metallic, dot(v, h), model, dielectric, irid)
        fresnel_factor = np.ones_like(diffuse_w[..., 0]) * (1.0 - np.clip(np.asarray(metallic, dtype=np.float64), 0.0, 1.0)) if thin["variant"] == "fresnel-ignored" \
            else diffuse_w[..., 0]
        cosine = np.clip(n_dot_l, 0.0, 1.0) if thin["variant"] == "front-only" else np.clip(-n_dot_l, 0.0, 1.0)
        value = np.where(degenerate, 0.0, thin_weight(thin) * fresnel_factor * cosine / math.pi)
        return value[..., None] * thin_color(thin)
    f = DIELECTRIC_F0 + (1.0 - DIELECTRIC_F0) * schlick_weight(dot(v, h), model)
    fresnel = np.ones_like(f) if thin["variant"] == "fresnel-ignored" else 1.0 - f
    cosine = np.clip(n_dot_l, 0.0, 1.0) if thin["variant"] == "front-only" else np.clip(-n_dot_l, 0.0, 1.0)
    m = np.clip(np.asarray(metallic, dtype=np.float64), 0.0, 1.0)
    value = np.where(degenerate, 0.0, (1.0 - m) * thin_weight(thin) * fresnel * cosine / math.pi)
    return value[..., None] * thin_color(thin)


def thin_lit(thin, n, l):
    """Pixels a light of direction `l` reaches through any lobe of the thin material: the front ones (N.L > 0) and, where w > 0, the
    back ones (N.L < 0)."""
    n_dot_l = dot(n, l)
    return (n_dot_l > 0) | ((n_dot_l < 0) & (thin_weight(thin) > 0))


_THIN_TABLE = {}
_THIN_QUADRATURE = (192, 192)


def thin_transmission_table():
    """(mu grid, T) with T(mu) = integral over the BACK hemisphere of (1 - F(V.H)) |N.L| / pi dL, F = Schlick(0.04, V.H), for V at
    cos(theta) = mu from N: the exact transmitted albedo of unit weight and colour in a constant environment."""
    if "t" not in _THIN_TABLE:
        mus = np.linspace(MIN_NDOTV, 1.0, MU_NODES)
        x, wgt = np.polynomial.legendre.leggauss(_THIN_QUADRATURE[0])
        cos_l = 0.5 * (x + 1.0)
        wgt = 0.5 * wgt
        phi = (np.arange(_THIN_QUADRATURE[1]) + 0.5) * (2 * math.pi / _THIN_QUADRATURE[1])
        sin_l = np.sqrt(1 - cos_l ** 2)
        values = np.empty(MU_NODES)
        for i, mu in enumerate(mus):
            v_dot_l = sin_l[:, None] * np.cos(phi)[None, :] * math.sqrt(max(0.0, 1 - mu * mu)) - cos_l[:, None] * mu
            v_dot_h = np.sqrt(np.clip((1 + v_dot_l) / 2, 0.0, 1.0))
            f = DIELECTRIC_F0 + (1 - DIELECTRIC_F0) * (1.0 - v_dot_h) ** 5
            values[i] = float(np.sum(((1.0 - f) * cos_l[:, None] / math.pi).mean(axis=1) * 2 * math.pi * wgt))
        _THIN_TABLE["t"] = (mus, values)
    return _THIN_TABLE["t"]


def thin_ratio(mu):
    """The declared ratio of the transmitted to the front diffuse weight, 1 - 0.16 mu (exact: T / Cd, fit error 1.4e-3)."""
    return 1.0 - THIN_RATIO_SLOPE * np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)


def hemisphere_radiance(direction):
    """The legacy environment: mix(ground, sky, d.y 0.5 + 0.5) per unit direction (n, 3) -> (n, 3)."""
    t = (np.asarray(direction, dtype=np.float64)[:, 1:2] * 0.5 + 0.5)
    return np.asarray(HEMISPHERE_GROUND) * (1.0 - t) + np.asarray(HEMISPHERE_SKY) * t


def aniso_bent_normal(n, v, bitangent, roughness, strength):
    """The README's bent normal (research/khr-materials.md section 4), written from its GLSL:
    bent = normalize(cross(cross(B, V), B)); a = (1 - strength (1 - roughness))^4; N' = normalize(mix(bent, N, a))."""
    bent = np.cross(np.cross(bitangent, v), bitangent)
    length = np.linalg.norm(bent, axis=-1, keepdims=True)
    bent = np.where(length > 1e-6, bent / np.maximum(length, 1e-12), n)
    a = ((1.0 - strength * (1.0 - roughness)) ** 2) ** 2
    mixed = bent * (1.0 - a)[:, None] + n * a[:, None]
    norm = np.linalg.norm(mixed, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, mixed / np.maximum(norm, 1e-12), n)


def legacy_hemisphere_radiance(base, metallic, roughness, n, v, aniso, variant):
    """Declared radiance of the engine's LEGACY environment (no declared constant one): the declared split-sum weights of the material (the anisotropic
    effective roughness of `declared_aniso_alpha`) times hemisphere(N) for the diffuse and hemisphere(specularDir) for the specular, with
    specularDir = normalize(mix(reflect(-V, N'), N', roughness^2)) and N' the README bent normal of an anisotropic pixel (the control
    `bent-normal-ignored` fetches along N)."""
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    effective = np.asarray(roughness, dtype=np.float64)
    env_normal = n
    table_ab = None
    kind = variant.get("aniso", "spec")
    if aniso is not None:
        at, ab = aniso_alphas(effective_alpha(roughness), aniso["strength"])
        if kind in ("spec", "azimuth-ignored", "bent-normal-ignored"):  # F15.1 (ticket 11, round 3): the anisotropic albedo table
            table_ab = declared_aniso_ab(effective, n_dot_v, aniso, "azimuth-ignored" if kind == "azimuth-ignored" else "spec")
        else:  # the older fit (negative controls `rms-fit`, `ignored`, `alpha-t`)
            alpha_eff = declared_aniso_alpha(at, ab, "spec" if kind == "rms-fit" else kind)
            effective = np.where(alpha_eff == ab, effective, np.sqrt(alpha_eff))
        if kind != "bent-normal-ignored":
            _, b_axis = aniso_axes(aniso, n)
            env_normal = np.where((aniso["strength"] > 0)[:, None], aniso_bent_normal(n, v, b_axis, np.asarray(roughness, dtype=np.float64), aniso["strength"]), n)
    a, b = dfg_lookup(n_dot_v, clamp_roughness(effective)) if table_ab is None else table_ab[:2]
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    f0 = DIELECTRIC_F0 * (1 - m) + base * m
    spec_w = np.clip(f0 * a[:, None] + b[:, None], 0.0, 1.0)
    diffuse_w = 1.0 - np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)[:, None]
    ms_term = 0.0
    if variant.get("ms", "off") != "off":  # ADR-0007: the lobe times hemisphere(N) (anisotropic lobes carry none), its energy out of the diffuse complement
        import f15_ms
        mask = f15_ms.isotropic(aniso, len(n_dot_v))[:, None] if aniso is not None else 1.0
        lobe = f15_ms.declared_complement(DIELECTRIC_F0, 1.0, clamp_roughness(effective), n_dot_v, variant["ms"]) * mask
        diffuse_w = np.maximum(diffuse_w - np.max(lobe, axis=-1, keepdims=True), 0.0)
        ms_term = f15_ms.declared_environment_value(base, np.asarray(metallic, dtype=np.float64), clamp_roughness(effective), n_dot_v, 1.0,
                                                    variant["ms"], None) * mask * hemisphere_radiance(n)
    reflected = 2.0 * dot(env_normal, v)[:, None] * env_normal - v
    mix_k = (np.asarray(roughness, dtype=np.float64) ** 2)[:, None]
    direction = normalize(reflected * (1.0 - mix_k) + env_normal * mix_k)
    return (1 - m) * base * diffuse_w * hemisphere_radiance(n) + spec_w * hemisphere_radiance(direction) + ms_term


# F15.3 closure (ticket 16): the declared transmitted albedo of a thin material under an extended dielectric or a film.  The dielectric-only ratio fit
# `1 - 0.16 mu` missed the exact T by 5 % of the radiance at film 1.3 / 400 nm and at ior 1.8; a 4-node quadrature in V.H with N.V-polynomial weights
# (fitted on 96 films and 24 dielectrics, held-out error <= 0.13 PERCENT of T sphere-weighted) replaces it when the material is not the default dielectric:
#   T(mu) = sum_k a_k(mu) W(vh_k),  W = the diffuse weight 1 - max3 F(V.H) (the extension Fresnel, or the (1 - s) / s mix with the film).
# The same constants live in shaders/materials/bsdf.h (bsdfThinEnvAlbedo).
THIN_ENV_NODES = (0.15, 0.4, 0.65, 0.9)
THIN_ENV_POLY = ((-0.093795337, 0.193347433, 0.125582023, 0.027399036), (0.197867067, -0.097814745, 0.288317827, 0.183036859), (-0.179523756, -0.107787082, 0.045040643, 0.429318433), (0.074219065, 0.014610016, -0.461559537, 0.360719294))


def thin_env_is_default(dielectric, irid):
    """Per pixel (n,) bool, or a scalar bool without arrays: the declared transmitted term keeps the default-dielectric ratio fit (no film, f0 0.04 and f90 1)."""
    n = 1 if dielectric is None and irid is None else len(dielectric["weight"] if dielectric is not None else irid_strength(irid))
    default = np.ones(n, dtype=bool)
    if irid is not None:
        default &= ~(irid_strength(irid) > 0.0)
    if dielectric is not None:
        default &= np.all(np.abs(np.asarray(dielectric["f0"], dtype=np.float64) - DIELECTRIC_F0) < 1e-6, axis=-1) & (np.abs(np.asarray(dielectric["weight"]) - 1.0) < 1e-6)
    return default if (dielectric is not None or irid is not None) else bool(default[0])


def thin_env_albedo(base, mu, dielectric, irid):
    """T(mu) (n,) of the 4-node declared transmitted albedo (see THIN_ENV_NODES)."""
    n = len(mu)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)
    f0 = np.full((n, 3), DIELECTRIC_F0) if dielectric is None else np.asarray(dielectric["f0"], dtype=np.float64)
    weight = np.ones(n) if dielectric is None else np.asarray(dielectric["weight"], dtype=np.float64)
    film_f0 = None if irid is None else irid_base_f0(base, dielectric)
    total = np.zeros(n)
    for vh, poly in zip(THIN_ENV_NODES, THIN_ENV_POLY):
        _, max_f = dielectric_fresnel(f0, weight, np.full((n, 1), (1.0 - vh) ** 5), "extension")
        w_value = 1.0 - max_f[:, 0]
        if irid is not None:
            s = irid_strength(irid)
            film = thin_film_fresnel(np.full(n, vh), irid_film_ior(irid), irid_thickness(irid), film_f0)
            w_value = (1.0 - s) * w_value + s * (1.0 - film.max(axis=-1))
        total += np.polyval(poly, mu) * w_value
    return np.maximum(total, 0.0)


def declared_extension_diffuse_weight(metallic, roughness, n_dot_v, variant, dielectric, irid):
    """The declared diffuse weight 1 - max3 E of the underlying dielectric (n, 1): the extension (ior/specular) and/or the film, exactly the weights
    `_declared_material_radiance` and `declared_irid_environment` apply to the diffuse term (F15.3 closure, thin + ior / iridescence)."""
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    a, b = declared_dfg(mu, clamp_roughness(roughness))
    a, b = a[:, None], b[:, None]
    if dielectric is None:
        f0_film, e_ds = np.full((len(mu), 3), DIELECTRIC_F0), np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)
    else:
        kind = variant.get("dielectric", "extension")
        f0c, weight_s = dielectric["f0"], np.asarray(dielectric["weight"], dtype=np.float64)[:, None]
        if kind == "ignored":
            f0c, weight_s = np.full_like(f0c, DIELECTRIC_F0), np.ones_like(weight_s)
        f0_film = f0c * weight_s
        e_ds = np.clip((f0c if kind == "f0-unweighted" else f0_film) * a + (np.ones_like(weight_s) if kind == "f90-one" else weight_s) * b, 0.0, 1.0)
    lobe = 0.0
    if variant.get("ms", "off") != "off":  # ADR-0007: F_ms L of the film-free dielectric layer comes out of the complement
        import f15_ms
        f90_ms = np.ones((len(mu), 1)) if dielectric is None or variant.get("dielectric", "extension") == "f90-one" else \
            np.asarray(dielectric["weight"], dtype=np.float64)[:, None]
        lobe = f15_ms.declared_complement(f0_film, f90_ms, clamp_roughness(roughness), mu, variant["ms"])
    plain = np.maximum(1.0 - np.max(e_ds + lobe, axis=-1, keepdims=True), 0.0)
    if irid is None:
        return plain
    s = irid_strength(irid)[:, None]
    e_film = np.clip(thin_film_fresnel(mu, irid_film_ior(irid), irid_thickness(irid), f0_film) * np.clip(a + b, 0.0, 1.0), 0.0, 1.0)
    return (1.0 - s) * plain + s * np.maximum(1.0 - np.max(e_film + lobe, axis=-1, keepdims=True), 0.0)


def declared_thin_environment(material, metallic, base, roughness, n_dot_v, thin, diffuse_weight=None, albedo=None, ms_kind="off"):
    """The declared thin-subsurface model for unit radiance (decision 06): `material` is the declared base radiance
    (1 - m) c (1 - E_d) + E.  The reflected diffuse is scaled by (1 - w) and the transmitted lobe adds
        (1 - m) w color (1 - E_d(mu)) r(mu),   r(mu) = 1 - 0.16 mu,  E_d = clamp(0.04 A + B, 0, 1)  (A, B at the material roughness),
    so it never exceeds the diffuse weight it replaces (energy moved, not added).  Controls: `front-only` r = 1 (the transmitted
    weight equals the front diffuse's), `untinted`, `double-counted`, `ignored`."""
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    a, b = declared_dfg(mu, clamp_roughness(roughness))
    if thin["variant"] == "default-weight":
        diffuse_weight = None
    weight = np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)[:, None] if diffuse_weight is None else 1.0 - diffuse_weight  # E_d of the extension / film when given
    if ms_kind != "off" and diffuse_weight is None:  # ADR-0007: the diffuse complement 1 - E_d - F_ms L that the transmitted lobe and the front diffuse share
        import f15_ms
        weight = weight + np.max(f15_ms.declared_complement(DIELECTRIC_F0, 1.0, clamp_roughness(roughness), mu, ms_kind), axis=-1, keepdims=True)
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    w = thin_weight(thin)[:, None]
    ratio = np.ones_like(weight) if thin["variant"] == "front-only" else thin_ratio(mu)[:, None]
    transmitted = w * thin_color(thin) * ((1.0 - weight) * ratio if albedo is None else np.where(albedo[1][:, None], albedo[0][:, None], (1.0 - weight) * ratio))
    removed = (1.0 - thin_reflect_scale(thin))[:, None] * base * (1.0 - weight)
    return material + (1.0 - m) * (transmitted - removed)


def lobe_weights(base, metallic, v_dot_h, model=APPENDIX_B, dielectric=None, irid=None):
    """(diffuse multiplier per channel, specular multiplier per channel) at one V.H.

    f = diffuse_multiplier * base/pi-style term + specular_multiplier * D * Vis; the
    diffuse multiplier already contains (1 - metallic) and the Fresnel weighting.
    `dielectric` (F15.1): per-pixel {"f0" (n,3), "weight" (n,), "variant"} of KHR_materials_ior/specular.
    `irid` (F15.3, ticket 12): per-pixel dict of the iridescence section (the thin-film Fresnel replaces Schlick).
    """
    if irid is not None:
        return irid_lobe_weights(base, metallic, v_dot_h, irid, model, dielectric)
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


def bsdf(base, metallic, roughness, n, v, l, model=APPENDIX_B, dielectric=None, coat=None, sheen=None, aniso=None, irid=None,
         thin=None):
    """BSDF value f (without the cosine) for world unit vectors; (n, 3) arrays or broadcastable.

    `sheen` (F15.3, ticket 10, per-pixel {"color", "roughness", "variant"} of KHR_materials_sheen): the material under
    the sheen layer, f_sheen + f_material scaling (sheen_layer).
    `coat` (F15.3, per-pixel {"factor", "roughness", "f0_ior", "variant"} of KHR_materials_clearcoat): the coated
    value (1 - c F_c) f_material + c F_c D Vis (see the module docstring); the coat term has its own cosine mask.
    The coat sits above the sheen: the coat attenuates the whole sheen + material value.
    `aniso` (F15.3, ticket 11, per-pixel dict of the anisotropy section): the specular D V is the anisotropic lobe.
    `thin` (F15.3, ticket 16, per-pixel dict of the thin-subsurface section): the reflected diffuse is scaled by 1 - w; the
    transmitted lobe (N.L < 0) is NOT part of this value (see thin_transmitted: it has its own cosine)."""
    alpha = effective_alpha(roughness, model)
    n_dot_l = dot(n, l)
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    h = normalize(v + l)
    n_dot_h = np.clip(dot(n, h), 0.0, 1.0)
    v_dot_h = dot(v, h)
    diffuse_w, specular_w = lobe_weights(base, metallic, v_dot_h, model, dielectric, irid)
    if thin is not None:
        diffuse_w = diffuse_w * thin_reflect_scale(thin)[..., None]
    if aniso is None:
        spec = (ggx_d(n_dot_h, alpha) * smith_vis(n_dot_v, np.maximum(n_dot_l, 0.0), alpha))[..., None]
    else:  # F15.3 (ticket 11): the anisotropic lobe replaces D V (Fresnel and the diffuse weighting are unchanged)
        spec = aniso_dv(n, v, l, alpha, aniso)[..., None]
    value = np.where((n_dot_l > 0)[..., None], diffuse_w * base / math.pi + specular_w * spec, 0.0)
    if model.get("ms", "off") != "off":  # F15 closure (ADR-0007): the Kulla-Conty lobe (tools/f15_ms.py) of the material under the layers
        import f15_ms
        value = value + f15_ms.lobe_value(base, metallic, roughness, n_dot_v, n_dot_l, model, dielectric, model["ms"]) * \
            f15_ms.isotropic(aniso, len(n_dot_v))[:, None]
    if sheen is not None and coat is not None and model.get("order") == "swapped":
        # ticket 19 control: the coat BELOW the sheen (sheen over coat over material): the sheen is not coated, the coat lobe is
        # scaled by the sheen's albedo scaling
        attenuation, coat_value = coat_lobe(coat, n, v, l)
        return sheen_layer(attenuation[..., None] * value + coat_value[..., None], sheen, n_dot_v, n_dot_l, n_dot_h)
    if sheen is not None:
        value = sheen_layer(value, sheen, n_dot_v, n_dot_l, n_dot_h)
    if coat is None:
        return value
    attenuation, coat_value = coat_lobe(coat, n, v, l)
    if model.get("ms", "off") != "off":  # the coat's own dielectric lobe (not attenuated, like the coat lobe)
        import f15_ms
        coat_value = coat_value + f15_ms.coat_lobe_value(coat, coat_weight(coat), np.abs(dot(n, v)), n_dot_l, model, model["ms"])
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


# Exact disk integration of a NARROW anisotropic lobe (ticket 11, F15.3 closure): the engine's rule (golden-angle spiral, 32 / 128 samples) is
# not independent evidence for itself, so `--exact-aniso-disk` replaces it, for the narrow-lobe pixels of an anisotropic material only, by a
# polar midpoint grid (EXACT_DISK_RINGS rings equal in cos(theta) x EXACT_DISK_AZIMUTHS azimuths: uniform in solid angle, a different
# quadrature family) and drops those pixels from the quadrature mask.
EXACT_ANISO_DISK = False
LEGACY_ENVIRONMENT = False  # `--legacy-environment`: the declared image is the engine's legacy two-colour hemisphere environment (ticket 11, bent normal)
HEMISPHERE_SKY, HEMISPHERE_GROUND = (0.30, 0.36, 0.45), (0.10, 0.09, 0.08)  # shaders/material_shading.h hemisphere(): sky above, ground below
THIN_BACK_NDOTL = 1e-3  # N.L at which the sheen albedo scaling of a light BEHIND a thin surface is read (see thinSheenBackScale in shaders/material_shading.h)
# `--sky-override R`: the environment radiance of the SPECULAR / SHEEN terms is R instead of the snapshot sky (a captured probe of constant radiance R
# lights them; ticket 10).  Only valid for a black base (no diffuse term, which keeps the declared environment): the fixture's contract.
SKY_OVERRIDE = None
SKY_RATIO_CONTROL = 1.0  # declared sky / overridden sky: the `declared-radiance` control of the sheen term
EXACT_DISK_RINGS, EXACT_DISK_AZIMUTHS = 64, 128
EXACT_DISK_COUNT = EXACT_DISK_RINGS * EXACT_DISK_AZIMUTHS


def sun_disk_polar(l, radius, rings=EXACT_DISK_RINGS, azimuths=EXACT_DISK_AZIMUTHS):
    """(rings * azimuths, 3) unit directions of a polar midpoint grid over the disk of centre l, equal solid-angle weights."""
    l = np.asarray(l, dtype=np.float64)
    helper = np.array([0.0, 0.0, 1.0]) if abs(l[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    t = np.cross(helper, l)
    t /= np.linalg.norm(t)
    b = np.cross(l, t)
    z = 1.0 - (np.arange(rings) + 0.5) / rings * (1.0 - math.cos(radius))
    phi = (np.arange(azimuths) + 0.5) * (2.0 * math.pi / azimuths)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    s = np.sqrt(np.maximum(0.0, 1.0 - zz * zz))
    return (t[None, None, :] * (s * np.cos(pp))[..., None] + b[None, None, :] * (s * np.sin(pp))[..., None] +
            l[None, None, :] * zz[..., None]).reshape(-1, 3)


def sun_disk_samples(l, radius, count):
    """Deterministic uniform solid-angle golden-angle spiral over the disk of centre l (unit)."""
    l = np.asarray(l, dtype=np.float64)
    if count == 1:
        return l[None, :]
    if count == EXACT_DISK_COUNT:
        return sun_disk_polar(l, radius, EXACT_DISK_RINGS, EXACT_DISK_AZIMUTHS)
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
                 coat=None, sheen=None, aniso=None, irid=None, thin=None):
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
    if aniso is not None and EXACT_ANISO_DISK:  # the narrow pixels get the exact disk integration and stop being masked
        count = np.where(narrow, EXACT_DISK_COUNT, count)
        narrow = np.zeros_like(narrow)
    alpha = effective_alpha(roughness, model)
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    # Diffuse: centre of the disk.
    n_dot_l = np.clip(dot(n, np.broadcast_to(l, n.shape)), 0.0, 1.0)
    h_c = normalize(v + l)
    diffuse_w, _ = lobe_weights(base, metallic, dot(v, h_c), model, dielectric, irid)
    if thin is not None:
        diffuse_w = diffuse_w * thin_reflect_scale(thin)[..., None]
    total = np.where((n_dot_l > 0)[..., None], diffuse_w * base / math.pi * n_dot_l[..., None], 0.0)
    uncoated_thin = None
    if thin is not None:  # F15.3 (ticket 16): the transmitted lobe at the disk centre (smooth, like the diffuse term)
        transmitted = thin_transmitted(thin, metallic, n, v, np.broadcast_to(l, n.shape), model, dielectric, irid, base)
        if (coat is not None and thin["variant"] == "uncoated") or (sheen is not None and thin["variant"] == "sheen-unscaled"):
            uncoated_thin = transmitted  # ticket 19 / 16 controls: the coat (or the sheen scaling) does not cover the transmitted lobe, added at the end
        else:
            total = total + transmitted
    coat_c = None
    if coat is not None:
        coat_c = coat_weight(coat)
        total = total * coat_attenuation(coat, coat_c, coat_fresnel(coat, coat.get("normal", n), v, h_c))[:, None]
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
            _, specular_w = lobe_weights(base[sel], metallic[sel], dot(v[sel], h), model, take(dielectric, sel), take_irid(irid, sel))
            if aniso is None:
                lobe = ggx_d(nh, alpha[sel]) * smith_vis(n_dot_v[sel], nl, alpha[sel])
            else:  # F15.3 (ticket 11): the disk rule keys on alpha_b = alpha (unchanged), the lobe is anisotropic
                lobe = aniso_dv(n[sel], v[sel], lk, alpha[sel], take_aniso(aniso, sel))
            term = (lobe * nl)[:, None] * specular_w
            if coat is not None:
                sub = take_coat(coat, sel)
                term = term * coat_attenuation(sub, coat_c[sel], coat_fresnel(sub, sub.get("normal", n[sel]), v[sel], h))[:, None]
            spec_sum[sel] += np.where((nl > 0)[:, None], term, 0.0) / int(c)
    result = total + spec_sum
    if model.get("ms", "off") != "off":  # F15 closure (ADR-0007): the lobe at the disk centre (smooth against the disk), under the coat
        import f15_ms
        ms_centre = f15_ms.lobe_value(base, metallic, roughness, n_dot_v, n_dot_l, model, dielectric, model["ms"]) * n_dot_l[:, None] * \
            f15_ms.isotropic(aniso, n_pixels)[:, None]
        if coat is not None:
            ms_centre = ms_centre * coat_attenuation(coat, coat_c, coat_fresnel(coat, n, v, h_c))[:, None]
        result = result + ms_centre
    coat_sum = None
    if coat is not None:
        # The coat lobe: its own roughness drives the disk rule (axis = reflection about Nc = n, own alpha).
        nc_all = coat.get("normal", n)  # texture closure: the coat lobe lives on Nc
        count_c, narrow_c = sun_quadrature(nc_all, v, l, coat["roughness"], radius, model)
        alpha_c = effective_alpha(coat["roughness"], model)
        n_dot_v_c = np.maximum(dot(nc_all, v), MIN_NDOTV)
        coat_sum = np.zeros(n_pixels)
        for c in np.unique(count_c):
            sel = np.flatnonzero(count_c == c)
            sub = take_coat(coat, sel)
            for sample in sun_disk_samples(l, radius, int(c)):
                lk = np.broadcast_to(sample, (sel.size, 3))
                ncs = nc_all[sel]
                nl = np.clip(dot(ncs, lk), 0.0, 1.0)
                h = normalize(v[sel] + lk)
                nh = np.clip(dot(ncs, h), 0.0, 1.0)
                fc = coat_fresnel(sub, ncs, v[sel], h)
                term = coat_c[sel] * fc * ggx_d(nh, alpha_c[sel]) * smith_vis(n_dot_v_c[sel], nl, alpha_c[sel]) * nl
                coat_sum[sel] += np.where(nl > 0, term, 0.0) / int(c)
        narrow = narrow | narrow_c
        if model.get("ms", "off") != "off":  # the coat's own dielectric lobe at the disk centre
            import f15_ms
            coat_sum = coat_sum + f15_ms.coat_lobe_value(coat, coat_c, n_dot_v, n_dot_l, model, model["ms"]) * n_dot_l
    swapped = sheen is not None and coat is not None and model.get("order") == "swapped"  # ticket 19 control: coat BELOW the sheen
    if sheen is not None:
        e_v = sheen_albedo(n_dot_v, sheen["roughness"], sheen["variant"])
        e_l = sheen_albedo(np.maximum(n_dot_l, THIN_BACK_NDOTL if thin is not None else MIN_NDOTV), sheen["roughness"], sheen["variant"])
        scale = sheen_scaling(sheen, e_v, e_l)
        lobe = sheen_lobe(sheen, n_dot_v, n_dot_l, np.clip(dot(n, h_c), 0.0, 1.0)) * n_dot_l[:, None]
        lobe = np.where((n_dot_l > 0)[:, None], lobe, 0.0)
        if coat is not None and not swapped:
            lobe = lobe * coat_attenuation(coat, coat_c, coat_fresnel(coat, coat.get("normal", n), v, h_c))[:, None]
        if swapped:  # sheen over (coat over material): the coat lobe is scaled by the sheen's albedo scaling, the sheen is not coated
            result = (result + coat_sum[:, None]) * scale[:, None] + lobe
            coat_sum = None
        else:
            result = result * scale[:, None] + lobe
    if coat_sum is not None:
        result = result + coat_sum[:, None]
    if uncoated_thin is not None:
        result = result + uncoated_thin
    return result * np.asarray(irradiance, dtype=np.float64), narrow


# --- point lights ------------------------------------------------------------

def point_window(distance, light_range):
    ratio = distance / max(float(light_range), 1e-3)
    return np.clip(1.0 - ratio ** 4, 0.0, 1.0) ** 2


def point_radiance(base, metallic, roughness, n, v, position, light, model=APPENDIX_B, dielectric=None, visible=None,
                   coat=None, sheen=None, aniso=None, irid=None, thin=None):
    """Outgoing radiance per pixel from one white delta point light; returns (radiance (n,3), lit (n,))."""
    delta = np.asarray(light["position"], dtype=np.float64)[None, :] - position
    distance = np.linalg.norm(delta, axis=1)
    l = delta / distance[:, None]
    geometry = point_window(distance, light["range"]) / np.maximum(distance * distance, 1e-4)
    incident = np.asarray(light["color"], dtype=np.float64)[None, :] * float(light["intensity"]) * geometry[:, None]
    n_dot_l = np.clip(dot(n, l), 0.0, 1.0)
    value = bsdf(base, metallic, roughness, n, v, l, model, dielectric, coat, sheen, aniso, irid, thin) * n_dot_l[:, None] * incident
    lit = dot(n, l) > 0
    if thin is not None:  # F15.3 (ticket 16): lights behind the surface light it through the transmitted lobe
        transmitted = thin_transmitted(thin, metallic, n, v, l, model, dielectric, irid, base)
        if sheen is not None and thin["variant"] != "sheen-unscaled":  # F15.3 closure (ticket 16): the sheen layer scales the transmitted lobe like the reflected one, E at the smallest N.L
            e_v = sheen_albedo(np.maximum(dot(n, v), MIN_NDOTV), sheen["roughness"], sheen["variant"])
            e_l = sheen_albedo(np.full(len(e_v), THIN_BACK_NDOTL), sheen["roughness"], sheen["variant"])
            transmitted = transmitted * sheen_scaling(sheen, e_v, e_l)[:, None]
        if coat is not None and thin["variant"] != "uncoated":  # ticket 19: the coat sits above the transmitted lobe too
            transmitted = transmitted * coat_lobe(coat, n, v, l)[0][:, None]
        value = value + transmitted * incident
        lit = thin_lit(thin, n, l)
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


_IRID_HIST = {}


def _vh_histogram(vh, weight):
    """Linear-splitting histogram of `weight` over V.H in [0, 1] (IRID_BINS bins): sum_k hist[k] F(k / (bins - 1)) is the
    quadrature of F(V.H) weight."""
    x = np.clip(vh.reshape(-1), 0.0, 1.0) * (IRID_BINS - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), IRID_BINS - 2)
    f = x - i0
    w = weight.reshape(-1)
    return np.bincount(i0, w * (1 - f), minlength=IRID_BINS) + np.bincount(i0 + 1, w * f, minlength=IRID_BINS)


def irid_histograms(alpha):
    """(mus, spec_hist (mu, bins), diffuse_hist (mu, bins), cu (mu,)) of the environment quadrature of one alpha, the weights of
    `_spec_integrals` (D Vis cos over the GGX half-vector CDF) and of `_diffuse_integral` (cos / pi, Gauss-Legendre in cos) binned by
    V.H: the quadrature of any F(V.H) is a dot product with these (the Schlick weight (1 - w) A and w B split is the special case)."""
    key = round(float(alpha), 12)
    if key not in _IRID_HIST:
        mus = np.linspace(MIN_NDOTV, 1.0, IRID_MU_NODES)
        u = (np.arange(U_NODES) + 0.5) / U_NODES
        phi = (np.arange(PHI_NODES) + 0.5) * (2 * math.pi / PHI_NODES)
        cos_h = np.sqrt((1 - u) / (1 + (alpha * alpha - 1) * u))
        sin_h = np.sqrt(np.maximum(0.0, 1 - cos_h ** 2))
        x, gwt = np.polynomial.legendre.leggauss(DIFFUSE_MU_NODES)
        cos_i = 0.5 * (x + 1.0)
        gwt = 0.5 * gwt
        phi_d = (np.arange(DIFFUSE_PHI_NODES) + 0.5) * (2 * math.pi / DIFFUSE_PHI_NODES)
        sin_i = np.sqrt(1 - cos_i ** 2)
        spec = np.empty((len(mus), IRID_BINS))
        diffuse = np.empty((len(mus), IRID_BINS))
        cu = np.empty(len(mus))
        for k, mu in enumerate(mus):
            wo = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
            vh = sin_h[:, None] * np.cos(phi)[None, :] * wo[0] + cos_h[:, None] * wo[2]
            wi_z = 2 * vh * cos_h[:, None] - wo[2]
            valid = (wi_z > 0) & (vh > 0)
            vis = smith_vis(mu, np.maximum(wi_z, 0.0), alpha)
            weight = np.where(valid, vis * wi_z * 4 * vh / cos_h[:, None], 0.0) / (U_NODES * PHI_NODES)
            spec[k] = _vh_histogram(vh, weight)
            wi_dot_wo = sin_i[:, None] * np.cos(phi_d)[None, :] * wo[0] + cos_i[:, None] * wo[2]
            vh_d = np.sqrt(np.clip((1 + wi_dot_wo) / 2, 0.0, 1.0))
            weight_d = np.broadcast_to((cos_i * 2 * math.pi * gwt / math.pi / DIFFUSE_PHI_NODES)[:, None], vh_d.shape)
            diffuse[k] = _vh_histogram(vh_d, weight_d)
            cu[k] = diffuse[k].sum()
        _IRID_HIST[key] = (mus, spec, diffuse, cu)
    return _IRID_HIST[key]


_IRID_TABLE = {}


def irid_environment_table(alpha, film_ior, thickness, f0, variant):
    """(mus, spec (mu, 3), diffuse (mu,)): the exact directional integrals of the thin-film Fresnel over N.V for one film
    (alpha, film ior, thickness nm, base f0 (3,)): spec = integral of D Vis cos F_irid(V.H), diffuse = integral of
    (1 - max F_irid(V.H)) cos / pi (`diffuse-unweighted`: cos / pi alone)."""
    key = (round(float(alpha), 12), round(float(film_ior), 9), round(float(thickness), 9), tuple(np.round(f0, 12)), variant)
    if key not in _IRID_TABLE:
        mus, spec_hist, diffuse_hist, cu = irid_histograms(alpha)
        bins = np.linspace(0.0, 1.0, IRID_BINS)
        film = thin_film_fresnel(bins, np.full(IRID_BINS, film_ior), np.full(IRID_BINS, thickness),
                                 np.broadcast_to(np.asarray(f0, dtype=np.float64), (IRID_BINS, 3)))
        diffuse = cu if variant == "diffuse-unweighted" else diffuse_hist @ (1.0 - film.max(axis=-1))
        _IRID_TABLE[key] = (mus, spec_hist @ film, diffuse)
    return _IRID_TABLE[key]


def irid_environment_value(base, metallic, mu, alpha, a, b, cd, irid, ext=None, aniso_env=None):
    """Exact environment value (n, 3) for unit radiance of iridescent pixels sharing one alpha (a, b, cd: the base tables'
    A, B and diffuse (n, 1) at the pixels' N.V).  `at-nv` (control and declared model alike) evaluates F_irid once at N.V:
    spec F (A + B), diffuse (1 - max F) integral of cos / pi without the Fresnel (the F_irid constant over the lobe)."""
    n = len(mu)
    s = irid_strength(irid)[:, None]
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    film_ior, thickness = irid_film_ior(irid), irid_thickness(irid)
    spec_d, spec_m = np.empty((n, 3)), np.empty((n, 3))
    diff_d = np.empty((n, 1))
    # ticket 19: with the ior/specular extension (`ext` = {"f0_scaled" (n, 3), "plain" (n, 3)}) the film takes the scaled f0 and the
    # (1 - s) part is the extended dielectric radiance `plain` (unit radiance) computed by the caller.
    f0d = np.full((n, 3), DIELECTRIC_F0) if ext is None else ext["f0_scaled"]
    if irid["variant"] == "at-nv":
        fd = thin_film_fresnel(irid["nv"], film_ior, thickness, f0d)
        fm = thin_film_fresnel(irid["nv"], film_ior, thickness, base)
        _, _, _, cu_t = irid_histograms(alpha)
        cu = np.interp(mu, np.linspace(MIN_NDOTV, 1.0, IRID_MU_NODES), cu_t)[:, None]
        spec_d, spec_m = fd * (a + b), fm * (a + b)
        diff_d = (1.0 - fd.max(axis=-1, keepdims=True)) * cu
    else:
        pairs = np.concatenate([film_ior[:, None], thickness[:, None], base, f0d], axis=1)
        for key in np.unique(np.round(pairs, 9), axis=0):
            pick = np.flatnonzero(np.all(np.round(pairs, 9) == key, axis=1))
            for target, f0 in ((spec_d, key[5:8]), (spec_m, key[2:5])):
                if aniso_env is not None:  # ticket 19: the film under the anisotropic lobe (V.H histograms of the anisotropic quadrature)
                    target[pick] = aniso_film_spec(key[0], key[1], f0, mu[pick], take_aniso(aniso_env, pick))
                    continue
                mus, spec, diffuse = irid_environment_table(alpha, key[0], key[1], f0, irid["variant"])
                for c in range(3):
                    target[pick, c] = np.interp(mu[pick], mus, spec[:, c])
            mus, _, diffuse = irid_environment_table(alpha, key[0], key[1], key[5:8], irid["variant"])
            diff_d[pick, 0] = np.interp(mu[pick], mus, diffuse)
    plain = (base * cd + DIELECTRIC_F0 * a + b) if ext is None else ext["plain"]
    dielectric = (1 - s) * plain + s * (base * diff_d + spec_d)
    metal = (1 - s) * (base * a + b) + s * spec_m
    return (1 - m) * dielectric + m * metal


def layered_environment(material, n_dot_v, coat, sheen, radiance, swapped=False):
    """Exact-environment layering above the material radiance `material` (n, 3, already times `radiance`): the sheen layer over the
    material, the coat over the sheen (research/khr-materials.md section 12: clearcoat > sheen > base).  `swapped` is the ticket 19
    control that puts the coat below the sheen."""
    if sheen is not None and coat is not None and swapped:
        return sheen_environment(environment_coat(material, n_dot_v, coat, radiance), n_dot_v, sheen, radiance)
    if sheen is not None:
        material = sheen_environment(material, n_dot_v, sheen, radiance)
    if coat is not None:
        material = environment_coat(material, n_dot_v, coat, radiance)
    return material


_THIN_EXT_TABLE = {}
THIN_EXT_NODES, THIN_EXT_QUADRATURE = 65, (96, 96)


def thin_transmission_table_for(f0, weight, film=None, variant="extension"):
    """(mu grid, T) of the exact transmitted albedo under an extended dielectric and/or a thin film (F15.3 closure, ticket 16):
    T(mu) = integral over the BACK hemisphere of W(V.H) |N.L| / pi dL with W = 1 - max3 F_ext(V.H) (ior/specular, `dielectric_fresnel`), or, with a
    film (ior, thickness nm, strength s, film base f0 (3,)), W = (1 - s) (1 - max3 F_ext) + s (1 - max3 F_film(V.H)) (the weight of the reflected diffuse,
    `irid_lobe_weights`).  `thin_transmission_table()` is the special case f0 0.04, weight 1, no film."""
    f0 = np.asarray(f0, dtype=np.float64)
    if film is None and variant == "extension" and float(weight) == 1.0 and np.allclose(f0, DIELECTRIC_F0, rtol=0, atol=1e-12):
        return thin_transmission_table()  # the neutral dielectric: the default table of ticket 16 (bit for bit)
    key = (tuple(np.round(f0, 12)), round(float(weight), 12), None if film is None else tuple(np.round(np.append(film[:3], film[3]), 12)), variant)
    if key not in _THIN_EXT_TABLE:
        mus = np.linspace(MIN_NDOTV, 1.0, THIN_EXT_NODES)
        x, wgt = np.polynomial.legendre.leggauss(THIN_EXT_QUADRATURE[0])
        cos_l, wgt = 0.5 * (x + 1.0), 0.5 * wgt
        phi = (np.arange(THIN_EXT_QUADRATURE[1]) + 0.5) * (2 * math.pi / THIN_EXT_QUADRATURE[1])
        sin_l = np.sqrt(1 - cos_l ** 2)
        values = np.empty(THIN_EXT_NODES)
        for i, mu in enumerate(mus):
            v_dot_l = sin_l[:, None] * np.cos(phi)[None, :] * math.sqrt(max(0.0, 1 - mu * mu)) - cos_l[:, None] * mu
            vh = np.sqrt(np.clip((1 + v_dot_l) / 2, 0.0, 1.0)).reshape(-1)
            count = vh.size
            _, max_f = dielectric_fresnel(np.tile(f0, (count, 1)), np.full(count, float(weight)), (1.0 - vh)[:, None] ** 5, variant)
            w_dielectric = (1.0 - max_f[:, 0])
            if film is not None:
                ior, thickness, strength, f0_film = film[0], film[1], film[2], np.asarray(film[3], dtype=np.float64)
                f_film = thin_film_fresnel(vh, np.full(count, ior), np.full(count, thickness), np.tile(f0_film, (count, 1)))
                w_dielectric = (1.0 - strength) * w_dielectric + strength * (1.0 - f_film.max(axis=-1))
            values[i] = float(np.sum((w_dielectric.reshape(cos_l.size, -1) * cos_l[:, None] / math.pi).mean(axis=1) * 2 * math.pi * wgt))
        _THIN_EXT_TABLE[key] = (mus, values)
    return _THIN_EXT_TABLE[key]


def environment_radiance(base, metallic, roughness, n_dot_v, radiance, model=APPENDIX_B, dielectric=None, aniso_env=None,
                         irid_env=None, thin_env=None):
    """Exact environment radiance; a thin subsurface on top of an extended dielectric and/or a film takes the exact transmitted albedo of that
    material (`thin_transmission_table_for`) instead of the default 0.04 one (F15.3 closure, ticket 16):
        result = R0 + (1 - m) (w color T - (1 - reflect scale) base Dint)
    with R0 the thin-free radiance, Dint the diffuse integral of the material (R0(base 1) - R0(base 0) of the dielectric, m = 0) and T the table."""
    if thin_env is not None and (dielectric is not None or irid_env is not None) and thin_env["variant"] != "default-fresnel":
        n = len(n_dot_v)
        r0 = _environment_radiance_core(base, metallic, roughness, n_dot_v, radiance, model, dielectric, aniso_env, irid_env, None)
        zero_m = np.zeros(n)
        d1 = _environment_radiance_core(np.ones((n, 3)), zero_m, roughness, n_dot_v, 1.0, model, dielectric, aniso_env, irid_env, None)
        d0 = _environment_radiance_core(np.zeros((n, 3)), zero_m, roughness, n_dot_v, 1.0, model, dielectric, aniso_env, irid_env, None)
        dint = (d1 - d0)[:, 0:1]
        mu = np.clip(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV, 1.0)
        f0 = np.full((n, 3), DIELECTRIC_F0) if dielectric is None else np.asarray(dielectric["f0"], dtype=np.float64)
        weight = np.ones(n) if dielectric is None else np.asarray(dielectric["weight"], dtype=np.float64)
        dvariant = "extension" if dielectric is None else dielectric["variant"]
        film_f0 = irid_base_f0(base, dielectric) if irid_env is not None else None
        strength = np.zeros(n) if irid_env is None else irid_strength(irid_env)
        albedo = np.empty((n, 1))
        keys = np.concatenate([f0, weight[:, None], strength[:, None],
                               np.zeros((n, 2)) if irid_env is None else np.stack([irid_film_ior(irid_env), irid_thickness(irid_env)], axis=1),
                               np.zeros((n, 3)) if film_f0 is None else film_f0], axis=1)
        for key in np.unique(np.round(keys, 9), axis=0):
            pick = np.flatnonzero(np.all(np.round(keys, 9) == key, axis=1))
            film = None if key[4] <= 0.0 else (key[5], key[6], key[4], key[7:10])
            mus, table = thin_transmission_table_for(key[0:3], key[3], film, dvariant)
            albedo[pick, 0] = np.interp(mu[pick], mus, table)
        if thin_env["variant"] == "front-only":
            albedo = dint
        elif thin_env["variant"] == "fresnel-ignored":
            albedo = np.ones_like(dint)
        m = np.asarray(metallic, dtype=np.float64)[:, None]
        w = thin_weight(thin_env)[:, None]
        removed = (1.0 - thin_reflect_scale(thin_env))[:, None] * base * dint
        return r0 + (1.0 - m) * (w * thin_color(thin_env) * albedo - removed) * radiance
    return _environment_radiance_core(base, metallic, roughness, n_dot_v, radiance, model, dielectric, aniso_env, irid_env, thin_env)


def _environment_radiance_core(base, metallic, roughness, n_dot_v, radiance, model=APPENDIX_B, dielectric=None, aniso_env=None,
                               irid_env=None, thin_env=None):
    """Outgoing radiance (n, 3) under a constant isotropic environment of radiance `radiance`.

    `dielectric` (F15.1, per-pixel {"f0", "weight", "variant"} of KHR_materials_ior/specular): the exact quadrature of
    the extended dielectric.  F is affine in the Schlick weight (F = c0 + c1 w), so the specular part is
    c0 * (A + B) + c1 * B per channel and the diffuse part integrates 1 - max(F) (extension_diffuse_table).
    `aniso_env` (F15.3, ticket 11, aniso_environment_inputs): A and B come from the anisotropic quadrature tables; the diffuse
    integrals do not depend on the lobe."""
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
        if aniso_env is not None:
            a1, b1 = aniso_spec_albedo(mu[sel], take_aniso(aniso_env, sel))
            a, b = a1[:, None], b1[:, None]
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
            diffuse_ext = c * cd_ext
            if thin_env is not None:  # ticket 19 (thin + coat): (1 - w) c Cd + w color T, T the back-hemisphere quadrature
                tt = take_thin(thin_env, sel)
                mus, table = thin_transmission_table()
                albedo = np.interp(mu[sel], mus, table)[:, None]
                if tt["variant"] == "front-only":
                    albedo = cd_ext
                elif tt["variant"] == "fresnel-ignored":
                    albedo = np.ones_like(cd_ext)
                diffuse_ext = thin_reflect_scale(tt)[:, None] * diffuse_ext + thin_weight(tt)[:, None] * thin_color(tt) * albedo
            dielectric_value = diffuse_ext + c0 * (a + b) + c1 * b
            result[sel] = (1 - m) * dielectric_value + m * (c * a + b)
            if irid_env is not None:  # ticket 19: iridescence + ior/specular, the film takes the scaled f0 (c0)
                sub = take_irid(irid_env, sel)
                active = np.flatnonzero(irid_strength(sub) > 0.0)
                if active.size:
                    result[sel[active]] = irid_environment_value(
                        c[active], m[active, 0], mu[sel][active], float(effective_alpha(r, model)), a[active], b[active],
                        cd[active], take_irid(sub, active), {"f0_scaled": c0[active], "plain": dielectric_value[active]},
                        None if aniso_env is None else take_aniso(take_aniso(aniso_env, sel), active))
            continue
        if model["weighting"] == "engine-blend":
            f0 = DIELECTRIC_F0 * (1 - m) + c * m
            result[sel] = (1 - m) * c * cu + f0 * a + b
            continue
        diffuse = c * (cu if model["weighting"] == "unweighted-diffuse" else cd)
        if thin_env is not None:  # F15.3 (ticket 16): (1 - w) c Cd + w color T, T the back-hemisphere quadrature
            t = take_thin(thin_env, sel)
            mus, table = thin_transmission_table()
            albedo = np.interp(mu[sel], mus, table)[:, None]
            if t["variant"] == "front-only":
                albedo = cd
            elif t["variant"] == "fresnel-ignored":
                albedo = np.ones_like(cd)
            diffuse = thin_reflect_scale(t)[:, None] * diffuse + thin_weight(t)[:, None] * thin_color(t) * albedo
        dielectric_value = diffuse + DIELECTRIC_F0 * a + b
        metal = c * a + b
        result[sel] = (1 - m) * dielectric_value + m * metal
        if irid_env is not None:  # F15.3 (ticket 12): the pixels whose film is active integrate the thin-film Fresnel
            sub = take_irid(irid_env, sel)
            active = np.flatnonzero(irid_strength(sub) > 0.0)
            if active.size:
                part = take_irid(sub, active)
                result[sel[active]] = irid_environment_value(
                    c[active], m[active, 0], mu[sel][active], float(effective_alpha(r, model)), a[active], b[active], cd[active], part,
                    None, None if aniso_env is None else take_aniso(take_aniso(aniso_env, sel), active))
    return result * radiance


# --- declared environment model (the engine's, ADR-0006) -------------------------

DFG_HEADER = REPO / "src" / "renderer" / "dfg_table.h"
_DFG = {}

# "extension_diffuse": the environment diffuse weight of materials carrying a feature. Since ticket 08 the F13
# composite receives the lobe guide and weights the GI/environment diffuse of an extended pixel with
# 1 - max(E_dielectric) of its own specular lobe ("extension", the engine's state). "composite-base" (the weight
# 1 - E(0.04) of the base model, the engine's state before ticket 08) is kept as a negative control.
DECLARED = {"albedo": "dfg-table", "diffuse": "one-minus-dielectric-albedo", "dielectric": "extension",
            "extension_diffuse": "extension", "coat": "spec", "sheen": "spec", "aniso": "spec", "irid": "spec",
            "thin": "spec", "ms": "table"}  # ms: the declared form of the multiple-scattering lobe (engine tables), ADR-0007
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
    "declared-sheen-declared-radiance": {**DECLARED, "sheen": "declared-radiance"},   # ticket 10: sheen lit by the declared environment, not the probe
}
# F15.3 (ticket 11): negative controls of the anisotropy grids of the deterministic resolve.  The declared fit is symmetric
# in the lobe axes and independent of the rotation, so there is no declared rotation / handedness / swap control: those are
# gated by the exact quadrature and the direct keys.
ANISO_DECLARED_CONTROLS = {
    "declared-aniso-ignored": {**DECLARED, "aniso": "ignored"},      # alpha_eff = alpha_b: the engine before ticket 11
    "declared-aniso-alpha-t": {**DECLARED, "aniso": "alpha-t"},      # alpha_eff = alpha_t: the stretched axis only
    "declared-aniso-rms-fit": {**DECLARED, "aniso": "rms-fit"},      # the isotropic split sum at the rms effective roughness: the engine before round 3
    "declared-aniso-azimuth-ignored": {**DECLARED, "aniso": "azimuth-ignored"},  # the albedo table read at a fixed view azimuth (pi / 4)
    "declared-aniso-bent-normal-ignored": {**DECLARED, "aniso": "bent-normal-ignored"},  # legacy hemisphere: fetch along N, not the README's bent normal
}
# F15.3 (ticket 12): negative controls of the iridescence grids of the deterministic resolve.  The split-sum model evaluates
# the film at N.V by construction (decision 06), so `at-nv` has no declared counterpart.
IRID_DECLARED_CONTROLS = {
    "declared-iridescence-ignored": {**DECLARED, "irid": "ignored"},                  # the engine before ticket 12
    "declared-iridescence-film-ior-ignored": {**DECLARED, "irid": "film-ior-ignored"},
    "declared-iridescence-film-ior-default": {**DECLARED, "irid": "film-ior-default"},
    "declared-iridescence-diffuse-unweighted": {**DECLARED, "irid": "diffuse-unweighted"},
}
# F15.3 (ticket 16): negative controls of the thin-subsurface grids of the deterministic resolve.  The declared model has no
# Fresnel at V.H for the transmitted lobe (the ratio 1 - 0.16 N.V stands for it), so `fresnel-ignored` has no declared twin.
THIN_DECLARED_CONTROLS = {
    "declared-transmission-ignored": {**DECLARED, "thin": "ignored"},                 # the engine before ticket 16
    "declared-transmission-untinted": {**DECLARED, "thin": "untinted"},
    "declared-transmission-front-only": {**DECLARED, "thin": "front-only"},           # ratio 1: transmitted weight = front diffuse's
    "declared-transmission-double-counted": {**DECLARED, "thin": "double-counted"},
}
# F15.3 (ticket 19): the declared twins of the combination controls (`transmission-uncoated` has none: the declared transmitted weight
# lives inside the material the coat attenuates).
COMBO_DECLARED_CONTROLS = {
    "declared-additive-layering": {**DECLARED, "coat": "base-unattenuated", "sheen": "unscaled-base", "irid": "diffuse-unweighted",
                                   "thin": "double-counted"},
    "declared-order-swapped": {**DECLARED, "order": "swapped"},
    "declared-transmission-default-weight": {**DECLARED, "thin": "default-weight"},  # the transmitted weight of the default dielectric, not of the extension / film
}
# F15.3 (texture closure): the declared twins of the texture controls (the same per-pixel parameters feed the declared model).
TEXTURE_DECLARED_CONTROLS = {"declared-" + name: {**DECLARED, **delta} for name, delta in TEXTURE_CONTROLS.items()}
# F15 closure (--specular-ms): the declared environment model with the table form of the multiple-scattering lobe (engine DFG and E_avg
# tables), and its negative controls.
DECLARED_MS = DECLARED  # kept for the callers of the first version of the option
MS_DECLARED_CONTROLS = {
    "declared-ms-ignored": {**DECLARED, "ms": "off"},
    "declared-ms-double": {**DECLARED, "ms": "double"},
}
ALL_DECLARED_CONTROLS = {**DECLARED_CONTROLS, **SPECULAR_DECLARED_CONTROLS, **COAT_DECLARED_CONTROLS, **COMBO_DECLARED_CONTROLS,
                         **SHEEN_DECLARED_CONTROLS, **ANISO_DECLARED_CONTROLS, **IRID_DECLARED_CONTROLS,
                         **THIN_DECLARED_CONTROLS, **TEXTURE_DECLARED_CONTROLS, **MS_DECLARED_CONTROLS}


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
    """Bilinear E of the sheen table: the rule of `sheenTableLookup` (node = sqrt(N.V) * (size - 1) for the quadratic N.V
    axis, (roughness - 0.07) / 0.93 * (size - 1) for the rows, the cell index
    clamped to size - 2; the stored values are UNCLAMPED and the interpolated result is clamped to E <= 1), written here
    from its documentation."""
    size, table = load_sheen_table(path)
    n = size - 1
    x = np.sqrt(np.clip(np.asarray(n_dot_v, dtype=np.float64), 0.0, 1.0)) * n
    y = (np.clip(np.asarray(roughness, dtype=np.float64), SHEEN_MIN_ROUGHNESS, 1.0) - SHEEN_MIN_ROUGHNESS) / (1.0 - SHEEN_MIN_ROUGHNESS) * n
    x0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), size - 2)
    fx, fy = x - x0, y - y0
    e = ((table[y0, x0] * (1 - fx) + table[y0, x0 + 1] * fx) * (1 - fy) +
         (table[y0 + 1, x0] * (1 - fx) + table[y0 + 1, x0 + 1] * fx) * fy)
    return np.minimum(e, SHEEN_E_CLAMP)


ANISO_TABLE_HEADER = REPO / "src" / "renderer" / "aniso_table.h"
_ANISO_TABLE = {}


def load_aniso_table(path=ANISO_TABLE_HEADER):
    """table[strength, roughness, azimuth, N.V, (A, B)] parsed from the engine header (tools/aniso_table.py; layout in the header's comment)."""
    key = str(path)
    if key not in _ANISO_TABLE:
        text = pathlib.Path(path).read_text()
        dims = [int(re.search(name + r"\s*=\s*(\d+)", text).group(1))
                for name in ("AnisoTableMu", "AnisoTableAzimuth", "AnisoTableRoughness", "AnisoTableStrength")]
        start = text.index("AnisoTable[")
        body = text[text.index("{", start) + 1:text.index("};", start)]
        values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?=f\b)", body)]
        if len(values) != int(np.prod(dims)) * 2:
            raise ValueError(f"{path}: expected {int(np.prod(dims)) * 2} anisotropic albedo values, parsed {len(values)}")
        _ANISO_TABLE[key] = np.asarray(values, dtype=np.float64).reshape(dims[3], dims[2], dims[1], dims[0], 2)
    return _ANISO_TABLE[key]


def aniso_table_lookup(n_dot_v, azimuth, roughness, strength, path=ANISO_TABLE_HEADER):
    """Four-linear (A, B) of the anisotropic albedo table, the rule of `anisoTableLookup` written from its documentation: node = clip(x, 0, 1) (n - 1)
    for x = N.V, azimuth / (pi / 2), roughness and strength, the cell index clamped to n - 2."""
    table = load_aniso_table(path)
    ns, nr, npi, nm = table.shape[:4]
    coords = []
    for x, n in ((n_dot_v, nm), (np.asarray(azimuth, dtype=np.float64) / (0.5 * math.pi), npi), (roughness, nr), (strength, ns)):
        t = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0) * (n - 1)
        i = np.minimum(np.floor(t).astype(np.int64), n - 2)
        coords.append((i, t - i))
    (im, fm), (ip, fp), (ir, fr), (is_, fs) = coords
    out = 0.0
    for ds in (0, 1):
        for dr in (0, 1):
            for dp in (0, 1):
                for dm in (0, 1):
                    w = (fs if ds else 1 - fs) * (fr if dr else 1 - fr) * (fp if dp else 1 - fp) * (fm if dm else 1 - fm)
                    out = out + w[..., None] * table[is_ + ds, ir + dr, ip + dp, im + dm]
    return out[..., 0], out[..., 1]


# The declared (A, B) of the anisotropic pixels of one `declared_environment_radiance` call: the base-lobe sites of the declared model read it
# through `declared_dfg` instead of the isotropic DFG table (None = every pixel isotropic).
_DECLARED_AB = {"ab": None}


def declared_dfg(n_dot_v, roughness):
    """(A, B) of the declared base lobe: the anisotropic override of the running call, else the DFG table."""
    ab = _DECLARED_AB["ab"]
    if ab is None:
        return dfg_lookup(n_dot_v, roughness)
    if len(ab[0]) != len(np.atleast_1d(n_dot_v)):
        raise ValueError("the anisotropic declared (A, B) has a different pixel count than the call")
    return ab


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
    term = sheen_color(sheen) * e[:, None]
    if kind == "declared-radiance":  # ticket 10 control: the sheen term lit by the DECLARED environment instead of the probe (the engine before the fix)
        term = term * SKY_RATIO_CONTROL
    return material * sheen_scaling(sheen, e)[:, None] + term


def karis_ab(n_dot_v, roughness):
    """Karis 2014 mobile environment-BRDF fit (A, B); the former engine model."""
    c0 = (-1.0, -0.0275, -0.572, 0.022)
    c1 = (1.0, 0.0425, 1.04, -0.04)
    r = [roughness * c0[i] + c1[i] for i in range(4)]
    a004 = np.minimum(r[0] * r[0], np.exp2(-9.28 * n_dot_v)) * r[0] + r[1]
    return -1.04 * a004 + r[2], 1.04 * a004 + r[3]


def declared_aniso_ab(roughness, n_dot_v, aniso, kind="spec"):
    """(A, B, active) per pixel of the declared environment of an anisotropic material (F15.1, ticket 11, round 3): the anisotropic albedo table of the engine
    (`aniso_table_lookup`: N.V, the azimuth `aniso["phi"]` of V against the lobe axis T' folded into [0, pi / 2], the material roughness, the strength) for the
    pixels with strength > 0, the DFG table elsewhere.  `azimuth-ignored` (control) reads the table at the fixed azimuth pi / 4."""
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    rough = clamp_roughness(roughness)
    strength = np.asarray(aniso["strength"], dtype=np.float64)
    phi = np.full(len(mu), 0.25 * math.pi) if kind == "azimuth-ignored" else np.asarray(aniso["phi"], dtype=np.float64)
    a_t, b_t = aniso_table_lookup(mu, phi, rough, np.clip(strength, 0.0, 1.0))
    a_d, b_d = dfg_lookup(mu, rough)
    active = strength > 0.0
    return np.where(active, a_t, a_d), np.where(active, b_t, b_d), active


def declared_environment_radiance(base, metallic, roughness, n_dot_v, radiance, variant=DECLARED, dielectric=None,
                                  coat=None, sheen=None, aniso=None, irid=None, thin=None):
    """The declared environment model (see `_declared_environment_radiance`).  F15.1 (ticket 11, round 3): an anisotropic pixel's split sum reads the
    anisotropic albedo table (variant["aniso"] `spec`, `azimuth-ignored`); the older fits (`rms-fit` = the rms effective roughness of the decision 06
    amendment, `ignored`, `alpha-t`) stay as negative controls.  `aniso` then needs "phi" (`aniso_environment_inputs`)."""
    kind = variant.get("aniso", "spec")
    if aniso is None or kind not in ("spec", "azimuth-ignored"):
        return _declared_environment_radiance(base, metallic, roughness, n_dot_v, radiance, variant, dielectric, coat, sheen, aniso, irid, thin)
    if "phi" not in aniso:
        raise ValueError("the declared anisotropic environment needs the azimuth of the view in the lobe frame (aniso['phi'])")
    a, b, _ = declared_aniso_ab(roughness, n_dot_v, aniso, kind)
    _DECLARED_AB["ab"] = (a, b)
    try:
        return _declared_environment_radiance(base, metallic, roughness, n_dot_v, radiance, variant, dielectric, coat, sheen, aniso, irid, thin)
    finally:
        _DECLARED_AB["ab"] = None


def _declared_environment_radiance(base, metallic, roughness, n_dot_v, radiance, variant=DECLARED, dielectric=None,
                                   coat=None, sheen=None, aniso=None, irid=None, thin=None):
    """Outgoing radiance (n, 3) of the engine's declared environment model (see the module docstring).

    `irid` (F15.3, ticket 12, per-pixel dict of the iridescence section; variant["irid"] names the control): the iridescent pixels
    use declared_irid_environment (the split sum with the thin-film Fresnel at N.V).

    `dielectric` (F15.1): per-pixel {"f0", "weight"} of KHR_materials_ior/specular (variant["dielectric"] picks the
    model or one of its wrong controls): E_d = f0_scaled * A + f90 * B per lobe, diffuse weight 1 - max(E_d).
    `coat` (F15.3, per-pixel {"factor", "roughness", "f0_ior", "variant"} of KHR_materials_clearcoat): the declared
    coat model (1 - c F_c(mu)) * declared_material + c F_c(mu) * clamp(A_c + B_c, 0, 1), (A_c, B_c) = dfg_lookup at
    the coat roughness; variant["coat"] names the control (`ignored`, `f0-from-ior`, `base-unattenuated`).
    `sheen` (F15.3, ticket 10): the material is first put under the sheen (declared_sheen_environment), the coat sits
    above that sum.
    `aniso` (F15.3, ticket 11, per-pixel {"strength", ...}): the split sum runs at the effective isotropic roughness
    sqrt(alpha_eff) of `declared_aniso_alpha` (variant["aniso"] names the fit or the control); the bent-normal radiance fetch
    of the Sample Renderer has no effect in a constant environment.
    `thin` (F15.3, ticket 16, per-pixel {"weight", "color", "variant"}): declared_thin_environment."""
    ms_pixels = np.ones(len(np.asarray(n_dot_v)), dtype=bool)  # ADR-0007: anisotropic base lobes carry no multiple-scattering lobe
    if aniso is not None:
        alpha = effective_alpha(roughness)
        at, ab = aniso_alphas(alpha, aniso["strength"])
        kind = variant.get("aniso", "spec")
        if kind not in ("spec", "azimuth-ignored"):  # the older fit of the decision 06 amendment (negative controls)
            alpha_eff = declared_aniso_alpha(at, ab, "spec" if kind == "rms-fit" else kind)
            roughness = np.where(alpha_eff == ab, np.asarray(roughness, dtype=np.float64), np.sqrt(alpha_eff))
        ms_pixels = ~(np.asarray(aniso["strength"], dtype=np.float64) > 0.0)
    material = _declared_material_radiance(base, metallic, roughness, n_dot_v, variant, dielectric, ms_pixels)
    if irid is not None:
        material = declared_irid_environment(material, base, metallic, roughness, n_dot_v, irid, dielectric, variant, ms_pixels)
    if thin is not None:  # F15.3 (ticket 16)
        exact_fit = (dielectric is not None or irid is not None) and thin["variant"] != "default-weight"
        material = declared_thin_environment(material, metallic, base, roughness, n_dot_v, thin,
                                             None if dielectric is None and irid is None else
                                             declared_extension_diffuse_weight(metallic, roughness, n_dot_v, variant, dielectric, irid),
                                             (thin_env_albedo(base, n_dot_v, dielectric, irid), ~thin_env_is_default(dielectric, irid)) if exact_fit else None,
                                             variant.get("ms", "off"))
    if variant.get("ms", "off") != "off":  # F15 closure (ADR-0007): the declared lobe (engine tables) joins the material under sheen and coat
        import f15_ms
        material = material + f15_ms.declared_environment_value(base, metallic, roughness, n_dot_v, 1.0, variant["ms"], dielectric) * _ms_mask(ms_pixels)
    def under_coat(material):
        coat_v = {**coat, "variant": variant.get("coat", "spec")}
        if coat_v["variant"] == "fresnel-vh":
            raise ValueError("the declared split-sum model has no V.H: no fresnel-vh control")
        c = coat_weight(coat_v)
        mu = np.maximum(np.asarray(coat.get("mu", n_dot_v), dtype=np.float64), MIN_NDOTV)  # |V.Nc| with a coat normal texture
        a_c, b_c = dfg_lookup(mu, clamp_roughness(coat_v["roughness"]))
        fc = coat_fresnel_mu(coat_v, mu)
        return (coat_attenuation(coat_v, c, fc)[:, None] * material +
                (c * fc * np.clip(a_c + b_c, 0.0, 1.0))[:, None])
    if sheen is not None and coat is not None and variant.get("order") == "swapped":  # ticket 19 control: coat below sheen
        material = declared_sheen_environment(under_coat(material), n_dot_v, sheen, variant)
    else:
        if sheen is not None:
            material = declared_sheen_environment(material, n_dot_v, sheen, variant)
        if coat is not None:
            material = under_coat(material)
    if coat is not None and variant.get("ms", "off") != "off":  # the coat's own lobe, not attenuated
        import f15_ms
        material = material + f15_ms.declared_coat_environment_value(coat, coat_weight({**coat, "variant": variant.get("coat", "spec")}), n_dot_v,
                                                                     variant["ms"])
    return material * radiance


def _ms_mask(ms_pixels):
    """(n, 1) float mask of the pixels that carry the multiple-scattering lobe (None: all)."""
    return 1.0 if ms_pixels is None else np.asarray(ms_pixels, dtype=np.float64)[:, None]


def declared_irid_environment(material, base, metallic, roughness, n_dot_v, irid, dielectric=None, variant=None, ms_pixels=None):
    """The declared iridescence model for unit radiance (decision 06): pixels whose film is active use the split sum with the
    thin-film Fresnel evaluated ONCE at N.V and held constant over the lobe, E_irid = clamp(F_irid(N.V) clamp(A + B, 0, 1), 0, 1)
    (A, B at the material roughness, `dfg_lookup`); diffuse weight 1 - max(E_irid) (the composite's rule for an extended lobe).
        dielectric = (1 - s) (c (1 - E_s) + E_s) + s (c (1 - max E_irid) + E_irid),   E_s = clamp(0.04 A + B, 0, 1)
        metal      = (1 - s) clamp(c A + B, 0, 1) + s E_irid(base_f0 = c)
    `material` keeps the pixels whose strength is 0 (the declared model of the base)."""
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    a, b = declared_dfg(mu, clamp_roughness(roughness))
    a, b = a[:, None], b[:, None]
    s = irid_strength(irid)[:, None]
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    film, thickness = irid_film_ior(irid), irid_thickness(irid)
    carry = np.clip(a + b, 0.0, 1.0)
    # ticket 19 (iridescence + ior/specular): the (1 - s) layer is the declared extended dielectric (E = f0_scaled A + f90 B, diffuse
    # weight 1 - max E, the model of _declared_material_radiance) and the film takes the scaled f0 (irid_base_f0).
    ms_kind = (variant or DECLARED).get("ms", "off")  # ADR-0007: the film-free dielectric lobe's energy comes out of the diffuse complement
    if dielectric is None:
        f0_film = np.full(base.shape, DIELECTRIC_F0)
        e_ds = np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)
        lobe = 0.0
        if ms_kind != "off":
            import f15_ms
            lobe = f15_ms.declared_complement(DIELECTRIC_F0, 1.0, clamp_roughness(roughness), mu, ms_kind) * _ms_mask(ms_pixels)
        plain = base * (1 - np.max(e_ds + lobe, axis=-1, keepdims=True) if ms_kind != "off" else 1 - e_ds) + e_ds
    else:
        kind = (variant or DECLARED).get("dielectric", "extension")
        f0c, weight_s = dielectric["f0"], np.asarray(dielectric["weight"], dtype=np.float64)[:, None]
        if kind == "ignored":
            f0c, weight_s = np.full_like(f0c, DIELECTRIC_F0), np.ones_like(weight_s)
        f0_film = f0c * weight_s
        f90 = np.ones_like(weight_s) if kind == "f90-one" else weight_s
        e_ds = np.clip((f0c if kind == "f0-unweighted" else f0_film) * a + f90 * b, 0.0, 1.0)
        lobe = 0.0
        if ms_kind != "off":
            import f15_ms
            lobe = f15_ms.declared_complement(f0_film, f90, clamp_roughness(roughness), mu, ms_kind) * _ms_mask(ms_pixels)
        plain = base * (1.0 - np.max(e_ds + lobe, axis=-1, keepdims=True)) + e_ds
    e_d = np.clip(thin_film_fresnel(mu, film, thickness, f0_film) * carry, 0.0, 1.0)
    e_m = np.clip(thin_film_fresnel(mu, film, thickness, base) * carry, 0.0, 1.0)
    e_ms = np.clip(base * a + b, 0.0, 1.0)
    weight = np.ones_like(a) if irid["variant"] == "diffuse-unweighted" else np.maximum(1.0 - np.max(e_d + lobe, axis=-1, keepdims=True), 0.0)
    dielectric_value = (1 - s) * plain + s * (base * weight + e_d)
    metal = (1 - s) * e_ms + s * e_m
    return np.where(s > 0.0, (1 - m) * dielectric_value + m * metal, material)


def _declared_material_radiance(base, metallic, roughness, n_dot_v, variant, dielectric, ms_pixels=None):
    """The declared model of the material alone for unit environment radiance (see declared_environment_radiance)."""
    mu = np.maximum(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV)
    rough = clamp_roughness(roughness)
    a, b = karis_ab(mu, rough) if variant["albedo"] == "karis-fit" else declared_dfg(mu, rough)
    a, b = a[:, None], b[:, None]
    m = np.asarray(metallic, dtype=np.float64)[:, None]
    ms_on = variant.get("ms", "off") != "off"  # ADR-0007: the dielectric layer's lobe energy comes out of the diffuse complement
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
            import f15_ms
            lobe = f15_ms.declared_complement(f0_scaled, f90, rough, mu, variant["ms"]) * _ms_mask(ms_pixels) if ms_on else 0.0
            diffuse_weight = np.maximum(1.0 - np.max(e_d + lobe, axis=-1, keepdims=True), 0.0)
        return (1 - m) * base * diffuse_weight + (1 - m) * e_d + m * e_m
    f0 = DIELECTRIC_F0 * (1 - m) + base * m
    specular = np.clip(f0 * a + b, 0.0, 1.0)
    if variant["diffuse"] == "unweighted":
        weight = np.ones_like(a)
    elif variant["diffuse"] == "metallic-blended-f0":
        weight = 1.0 - np.clip(f0 * a + b, 0.0, 1.0)
    else:
        weight = 1.0 - np.clip(DIELECTRIC_F0 * a + b, 0.0, 1.0)
        if ms_on:
            import f15_ms
            weight = np.maximum(weight - np.max(f15_ms.declared_complement(DIELECTRIC_F0, 1.0, rough, mu, variant["ms"]) * _ms_mask(ms_pixels),
                                                axis=-1, keepdims=True), 0.0)
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


class Cast(tuple):
    """The 5-tuple of cast_primary_rays; `primitive` (h, w) holds the hit triangle index (face order of the PLY file) per
    pixel, -1 where nothing was hit (None for a cast built by hand)."""
    primitive = None


def cast_primary_rays(root, data, threads):
    """Pixel-centre primary hits with Mitsuba: (valid, position, shading normal, instance index) in a Cast (the triangle
    index is its `primitive` attribute)."""
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
    result = Cast((valid.reshape(height, width), position, normal, instance.reshape(height, width), (width, height)))
    result.primitive = np.where(valid, np.asarray(si.prim_index).astype(np.int64), -1).reshape(height, width)
    return result


# --- surface tangents (KHR_materials_anisotropy) -------------------------------

_PLY = {}


def read_ply(path):
    """(vertex properties {name: float32 array}, faces (n, 3) uint32) of a binary little-endian triangle PLY."""
    key = str(path)
    if key not in _PLY:
        raw = pathlib.Path(path).read_bytes()
        end = raw.index(b"end_header\n") + len(b"end_header\n")
        header = raw[:end].decode().splitlines()
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"{path}: only binary little-endian PLY files are read")
        count = int(next(line.split()[2] for line in header if line.startswith("element vertex")))
        faces = int(next(line.split()[2] for line in header if line.startswith("element face")))
        names = [line.split()[2] for line in header if line.startswith("property float")]
        vertices = np.frombuffer(raw, dtype=np.dtype([(n, "<f4") for n in names]), count=count, offset=end)
        face_dtype = np.dtype([("n", "u1"), ("i", "<u4", 3)])
        table = np.frombuffer(raw, dtype=face_dtype, count=faces, offset=end + vertices.nbytes)
        if (table["n"] != 3).any():
            raise ValueError(f"{path}: only triangle faces are read")
        _PLY[key] = ({n: np.array(vertices[n]) for n in names}, np.array(table["i"]))
    return _PLY[key]


def instance_matrix(instance):
    """The 4x4 world matrix of a snapshot instance (column-major words)."""
    return np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F")


def surface_tangents(root, data, cast, pick):
    """(tangent (n, 3), w (n,), det (n,)) per picked pixel: the barycentrically interpolated mesh tangent attribute at the
    hit transformed by the world matrix as a direction (not yet orthogonalised), the sign of the interpolated handedness and
    the sign of the world matrix determinant.  The barycentrics come from the hit position in the object space of the hit
    triangle (affine invariant), the triangle from Mitsuba's primitive index; the mesh must carry tangents (mesh version 2)."""
    valid, position, _, instance, _ = cast
    if getattr(cast, "primitive", None) is None:
        raise ValueError("the cast carries no primitive indices")
    inst = instance.reshape(-1)[pick]
    prim = cast.primitive.reshape(-1)[pick]
    hit = position[pick]
    tangent = np.zeros((len(pick), 3))
    w = np.zeros(len(pick))
    det = np.zeros(len(pick))
    for k in np.unique(inst):
        sel = np.flatnonzero(inst == k)
        item = data["instances"][int(k)]
        properties, faces = read_ply(root / data["meshes"][item["mesh"]])
        if "tangent_x" not in properties or "handedness_x" not in properties:
            raise ValueError("the mesh PLY has no tangent attribute (re-export with mesh version 2)")
        world = instance_matrix(item)
        triangle = faces[prim[sel]]
        position_o = np.stack([properties["x"], properties["y"], properties["z"]], axis=1).astype(np.float64)
        v = position_o[triangle]  # (m, 3 corners, 3)
        inverse = np.linalg.inv(world)
        p = (np.concatenate([hit[sel], np.ones((len(sel), 1))], axis=1) @ inverse.T)[:, :3]
        e1, e2, d = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], p - v[:, 0]
        d11, d12, d22 = dot(e1, e1), dot(e1, e2), dot(e2, e2)
        d1, d2 = dot(d, e1), dot(d, e2)
        denominator = d11 * d22 - d12 * d12
        b1 = (d22 * d1 - d12 * d2) / denominator
        b2 = (d11 * d2 - d12 * d1) / denominator
        bary = np.stack([1.0 - b1 - b2, b1, b2], axis=1)
        t_vertex = np.stack([properties["tangent_x"], properties["tangent_y"], properties["tangent_z"]], axis=1).astype(np.float64)
        t_object = np.einsum("mc,mcj->mj", bary, t_vertex[triangle])
        w_value = np.einsum("mc,mc->m", bary, properties["handedness_x"].astype(np.float64)[triangle])
        tangent[sel] = t_object @ world[:3, :3].T
        w[sel] = np.where(w_value < 0.0, -1.0, 1.0)
        det[sel] = np.sign(np.linalg.det(world[:3, :3]))
    return tangent, w, det


def orthogonal_tangent(tangent, n):
    """Gram-Schmidt of the world tangent against the shading normal; a (near) zero result takes any perpendicular."""
    t = tangent - dot(tangent, n)[:, None] * n
    length = np.linalg.norm(t, axis=1, keepdims=True)
    helper = np.where(np.abs(n[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    fallback = np.cross(n, helper)
    return normalize(np.where(length > 1e-6, t, fallback))


# --- frame ------------------------------------------------------------------

SPECULAR_FEATURE_BIT = 1  # MaterialFeature::SpecularIor
CLEARCOAT_FEATURE_BIT = 2  # MaterialFeature::Clearcoat
EXTENSION_RECORD_WORDS = 20
# F15.3 (ticket 19): every feature bit the reference models; the combinations it composes are listed at check_combination
# (anything else is refused with a ValueError, never silently approximated).
SUPPORTED_FEATURE_BITS = 1 | 2 | 4 | 8 | 16 | 128


def check_combination(bits):
    """The feature combinations the reference composes (ticket 19, ticket 16 closure): every subset of {ior/specular, clearcoat, sheen, anisotropy,
    iridescence, thin subsurface} (layering order of research/khr-materials.md section 12; the thin transmitted lobe sits inside the base, see
    thin_transmitted).  Nothing is refused any more; the function stays the single place a future refusal would live."""
    return  # F15.3 closure (ticket 16): thin subsurface is composed with every lobe (glTF Sample Renderer layering, see thin_transmitted)


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
    if bits & ~SUPPORTED_FEATURE_BITS:
        raise ValueError(f"material feature bits {bits:#x} are outside the declared "
                         "ior/specular/clearcoat/sheen/anisotropy/iridescence/thin-subsurface model")
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


def material_clearcoat(data, material, textures_ok=False):
    """(clearcoatFactor, clearcoatRoughness, clearcoatNormalScale) from word offsets 7, 8, 9 of GPUMaterialExt0, or
    None without the clearcoat feature bit.  Raises when a clear-coat texture (words 10, 11, 12: factor, roughness,
    normal) is set: the reference is texture-free, and the coat normal is the shading normal."""
    words = material_extension_words(data, material)
    if words is None or not int(material.get("feature_bits", 0)) & CLEARCOAT_FEATURE_BIT:
        return None
    if not textures_ok and any(int(words[k]) != INVALID for k in (10, 11, 12)):
        raise ValueError("a clear-coat texture is set; the reference is texture-free")
    return f32(words[7]), f32(words[8]), f32(words[9])


def material_sheen(data, material, textures_ok=False):
    """(sheenColor[3], sheenRoughness) from word offsets 13-15 and 16 of GPUMaterialExt0, or None without the sheen
    feature bit.  Raises when a sheen texture (words 17, 18: colour, roughness) is set: the reference is texture-free.
    A zero sheenColor disables the layer (the engine clears the bit then); the formulas reduce to no sheen as well."""
    words = material_extension_words(data, material)
    if words is None or not int(material.get("feature_bits", 0)) & SHEEN_FEATURE_BIT:
        return None
    if not textures_ok and any(int(words[k]) != INVALID for k in (17, 18)):
        raise ValueError("a sheen texture is set; the reference is texture-free")
    return [f32(words[13]), f32(words[14]), f32(words[15])], f32(words[16])


def material_anisotropy(data, material, textures_ok=False):
    """(strength, rotation) from words 0 and 1 of record 1 (GPUMaterialExt1) of the material's extension group, or None
    without the anisotropy feature bit.  Raises when the anisotropy texture (word 2) is set: the reference is texture-free
    (the direction texture is open for the Khronos corpus, ticket 22).  Ticket 19: composed with coat, sheen, ior/specular and
    iridescence (check_combination)."""
    bits = int(material.get("feature_bits", 0))
    if not bits & ANISOTROPY_FEATURE_BIT:
        return None
    check_combination(bits)
    words = data["material_extensions"]["records"][int(material["extension_index"]) + ANISOTROPY_RECORD]
    if len(words) != EXTENSION_RECORD_WORDS:
        raise ValueError("a material extension record must hold 20 words")
    if not textures_ok and int(words[2]) != INVALID:
        raise ValueError("an anisotropy texture is set; the reference is texture-free")
    return f32(words[0]), f32(words[1])


def material_iridescence(data, material, textures_ok=False):
    """(iridescenceFactor, iridescenceIor, thicknessMinimum, thicknessMaximum) from words 3-6 of record 1 (GPUMaterialExt1) of the
    material's extension group, or None without the iridescence feature bit.  Raises when the factor or thickness texture (words 7
    and 8) is set (the reference is texture-free: the thickness texture case is open for ticket 22).  Ticket 19: composed with coat,
    sheen, anisotropy and ior/specular (check_combination)."""
    bits = int(material.get("feature_bits", 0))
    if not bits & IRIDESCENCE_FEATURE_BIT:
        return None
    check_combination(bits)
    words = data["material_extensions"]["records"][int(material["extension_index"]) + IRIDESCENCE_RECORD]
    if len(words) != EXTENSION_RECORD_WORDS:
        raise ValueError("a material extension record must hold 20 words")
    if not textures_ok and (int(words[7]) != INVALID or int(words[8]) != INVALID):
        raise ValueError("an iridescence or thickness texture is set; the reference is texture-free")
    return f32(words[3]), f32(words[4]), f32(words[5]), f32(words[6])


def material_subsurface(data, material, textures_ok=False):
    """(weight, colour[3]) of the thin subsurface block from record 2 (GPUMaterialExt2) of the material's extension group
    (words 1 weight, 2-4 colour), or None without the subsurface feature bit.  Raises on volumetric mode (word 0 = 1: ticket 17),
    on a texture (words 10-12), on a non-zero anisotropy (word 9) and on any other feature bit."""
    bits = int(material.get("feature_bits", 0))
    if not bits & SUBSURFACE_FEATURE_BIT:
        return None
    check_combination(bits)
    words = data["material_extensions"]["records"][int(material["extension_index"]) + SUBSURFACE_RECORD]
    if len(words) != EXTENSION_RECORD_WORDS:
        raise ValueError("a material extension record must hold 20 words")
    if int(words[0]) != 0:
        raise ValueError("volumetric subsurface is outside the thin-mode model (ticket 17)")
    if any(int(words[k]) != INVALID for k in ((12,) if textures_ok else (10, 11, 12))):
        raise ValueError("a subsurface texture is set; the reference is texture-free")
    if f32(words[9]) != 0.0:
        raise ValueError("a non-zero subsurface anisotropy is outside the thin-mode model (the draft's scatter split is not modelled)")
    weight, color = f32(words[1]), [f32(words[2]), f32(words[3]), f32(words[4])]
    if not (0.0 <= weight <= 1.0) or any(c < 0.0 for c in color):
        raise ValueError("subsurface weight outside [0, 1] or a negative colour")
    return weight, color


def uses_subsurface(data):
    return any(int(m.get("feature_bits", 0)) & SUBSURFACE_FEATURE_BIT for m in data["materials"])


def uses_iridescence(data):
    return any(int(m.get("feature_bits", 0)) & IRIDESCENCE_FEATURE_BIT for m in data["materials"])


def uses_anisotropy(data):
    return any(int(m.get("feature_bits", 0)) & ANISOTROPY_FEATURE_BIT for m in data["materials"])


def uses_sheen(data):
    return any(int(m.get("feature_bits", 0)) & SHEEN_FEATURE_BIT for m in data["materials"])


def uses_specular_extension(data):
    return any(int(m.get("feature_bits", 0)) & SPECULAR_FEATURE_BIT for m in data["materials"])


def uses_clearcoat(data):
    return any(int(m.get("feature_bits", 0)) & CLEARCOAT_FEATURE_BIT for m in data["materials"])


# --- KHR texture fixtures (F15.3 texture closure, tickets 09-12, 16, 19, 22) ------------------------------------------------
#
# Textured lobe parameters, from the KHR_materials_* texts (research/khr-materials.md) and the engine's sampler, written here
# independently of the MSL.  Enabled by `--textures` (render_frame(textures=True)); without it a textured material is refused as before.
#
# Sampling definition (the engine's `kMaterialSampler`: linear, trilinear mips, repeat, anisotropy 8; the visibility resolve passes
# analytic UV gradients; the result is a half4):
#   * the texel centre of texel i is (i + 0.5) / size, v = 0 is the first (top) row, wrap = repeat; level 0 bilinear;
#   * the value is rounded to binary16 before the factor multiplication (shader: `half4(tex.sample(...))`);
#   * the exported RGB of an sRGB texture is ALREADY linear (the hardware decodes before filtering, ReferenceTexture contract);
#     alpha is linear.
#   * mip selection is NOT modelled: a pixel whose UV footprint (largest UV step to a 4-neighbour of the same instance, wrapped in u)
#     exceeds FOOTPRINT_LIMIT texels in either axis is dropped from the ROI of the render (`ambiguous`), the engine's LOD may be above
#     0 there.  Everywhere else the sampler magnifies (LOD <= 0) and level 0 is exact.  The criterion is a property of the texture
#     and the view, fixed before any comparison.
#   * UV = the mesh `s`, `t` of the PLY barycentrically interpolated at the hit (the hit position in the object space of the hit
#     triangle: affine invariant, equal to the engine's perspective-correct interpolation); no KHR_texture_transform.
#
# Channel semantics (texture word -> parameter):
#   clearcoat      factor c = saturate(factor * tex.r), roughness = clamp(roughnessFactor * tex.g, 0.04, 1),
#                  normal Nc = normalize(T x + B y + N z), (x, y, z) = (2 rgb - 1) with x, y scaled by clearcoatNormalScale;
#                  T the shading tangent orthogonalised against N, B = cross(N, T) w det (the anisotropy frame), N the shading normal
#   sheen          colour S = factor * tex.rgb (sRGB), roughness = factor * tex.a
#   anisotropy     direction d = normalize(2 rg - 1) (+X when zero), strength = saturate(strength * tex.b), the lobe direction is
#                  d rotated counter-clockwise by `rotation`, i.e. the lobe angle in the (T, B) frame is rotation + atan2(d.y, d.x)
#   iridescence    thickness = min (1 - tex.g) + max tex.g, factor = saturate(factor * tex.r)
#   thin           weight = saturate(weight * tex.a), colour = colour * tex.rgb (sRGB)
# Controls (render_frame model["tex"] / environment["tex"]): see TEXTURE_CONTROLS.
FOOTPRINT_LIMIT = 0.5
TEXTURE_WORDS = {  # slot: (record offset in the material's extension group, word, feature bit that owns the record words)
    "coat_factor": (0, 10, 2), "coat_roughness": (0, 11, 2), "coat_normal": (0, 12, 2),
    "sheen_color": (0, 17, 4), "sheen_roughness": (0, 18, 4),
    "aniso": (1, 2, 8),
    "irid_factor": (1, 7, 16), "irid_thickness": (1, 8, 16),
    "thin_weight": (2, 10, 128), "thin_color": (2, 11, 128),
}
TEXTURE_SLOT_ORDER = tuple(TEXTURE_WORDS)
# TEXTURE_CONTROLS: defined with the other control tables (top of the module).


def srgb_encode(linear):
    """The sRGB opto-electronic transfer function (IEC 61966-2-1): the inverse of the decode the hardware applies."""
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def texture_slots(data, material):
    """{slot: texture id} of the texture words a material sets, read only from the records of its feature bits."""
    bits = int(material.get("feature_bits", 0))
    out = {}
    if not bits:
        return out
    records = data["material_extensions"]["records"]
    base = int(material["extension_index"])
    for slot, (record, word, bit) in TEXTURE_WORDS.items():
        if bits & bit and int(records[base + record][word]) != INVALID:
            out[slot] = int(records[base + record][word])
    return out


def texture_slot_bits(data, material):
    """The slots a material binds as a bit set (bit k = TEXTURE_SLOT_ORDER[k]); 0 for a texture-free material."""
    slots = texture_slots(data, material)
    return sum(1 << k for k, name in enumerate(TEXTURE_SLOT_ORDER) if name in slots)


def uses_textures(data):
    return any(texture_slots(data, m) for m in data["materials"])


def surface_uv_image(root, data, cast):
    """((h, w, 2) interpolated mesh UV of the valid pixels, (h, w, 2) UV footprint): the footprint of a pixel is the largest absolute
    UV step (u wrapped to [0, 0.5], v plain) to its valid 4-neighbours of the same instance; 0 where no such neighbour exists."""
    valid, position, _, instance, (width, height) = cast
    if getattr(cast, "primitive", None) is None:
        raise ValueError("the cast carries no primitive indices")
    flat = valid.reshape(-1)
    inst = instance.reshape(-1)
    prim = cast.primitive.reshape(-1)
    uv = np.zeros((width * height, 2))
    pick = np.flatnonzero(flat)
    for k in np.unique(inst[pick]):
        sel = pick[inst[pick] == k]
        item = data["instances"][int(k)]
        properties, faces = read_ply(root / data["meshes"][item["mesh"]])
        if "s" not in properties or "t" not in properties:
            raise ValueError("the mesh PLY has no UV (s, t)")
        world = instance_matrix(item)
        triangle = faces[prim[sel]]
        position_o = np.stack([properties["x"], properties["y"], properties["z"]], axis=1).astype(np.float64)
        corners = position_o[triangle]
        p = (np.concatenate([position[sel], np.ones((len(sel), 1))], axis=1) @ np.linalg.inv(world).T)[:, :3]
        e1, e2, d = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], p - corners[:, 0]
        d11, d12, d22 = dot(e1, e1), dot(e1, e2), dot(e2, e2)
        d1, d2 = dot(d, e1), dot(d, e2)
        denominator = d11 * d22 - d12 * d12
        b1 = (d22 * d1 - d12 * d2) / denominator
        b2 = (d11 * d2 - d12 * d1) / denominator
        bary = np.stack([1.0 - b1 - b2, b1, b2], axis=1)
        st = np.stack([properties["s"], properties["t"]], axis=1).astype(np.float64)
        uv[sel] = np.einsum("mc,mcj->mj", bary, st[triangle])
    uv_image = uv.reshape(height, width, 2)
    inst_image = instance
    foot = np.zeros((height, width, 2))
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        other = np.roll(uv_image, shift, axis=axis)
        same = np.roll(inst_image, shift, axis=axis) == inst_image
        same &= np.roll(valid, shift, axis=axis) & valid
        delta = np.abs(other - uv_image)
        delta[..., 0] = np.minimum(delta[..., 0], 1.0 - delta[..., 0])
        foot = np.maximum(foot, np.where(same[..., None], delta, 0.0))
    return uv_image, foot


class TextureField:
    """Per-pixel sampling of the snapshot textures at the picked pixels of a cast (see the section comment)."""

    def __init__(self, root, data, cast, pick):
        from f12_reference import read_pfm
        uv_image, foot = surface_uv_image(root, data, cast)
        self.uv = uv_image.reshape(-1, 2)[pick]
        self.foot = foot.reshape(-1, 2)[pick]
        self.images = {}
        for t in data["textures"]:
            self.images[int(t["id"])] = (read_pfm(root / t["rgb"]).astype(np.float64),
                                         read_pfm(root / t["alpha"])[:, :, 0].astype(np.float64))
        self.ambiguous = np.zeros(len(pick), dtype=bool)
        self._cache = {}

    def sample_all(self, index, flip_v=False):
        """(n, 4) binary16-rounded bilinear RGBA of texture `index` at every picked pixel (cached)."""
        key = (index, flip_v)
        if key not in self._cache:
            rgb, alpha = self.images[index]
            h, w = alpha.shape
            uv = self.uv.copy()
            if flip_v:
                uv[:, 1] = 1.0 - uv[:, 1]
            x, y = uv[:, 0] * w - 0.5, uv[:, 1] * h - 0.5
            ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            fx, fy = (x - ix)[:, None], (y - iy)[:, None]
            image = np.concatenate([rgb, alpha[:, :, None]], axis=2)

            def texel(dx, dy):
                return image[(iy + dy) % h, (ix + dx) % w]
            value = ((texel(0, 0) * (1 - fx) + texel(1, 0) * fx) * (1 - fy) + (texel(0, 1) * (1 - fx) + texel(1, 1) * fx) * fy)
            self._cache[key] = value.astype(np.float16).astype(np.float64)
        return self._cache[key]

    def use(self, index, rows, flip_v=False):
        """The samples of texture `index` at pixel indices `rows`; flags the pixels whose footprint exceeds FOOTPRINT_LIMIT texels."""
        _, alpha = self.images[index]
        h, w = alpha.shape
        exceeds = (self.foot[rows, 0] * w > FOOTPRINT_LIMIT) | (self.foot[rows, 1] * h > FOOTPRINT_LIMIT)
        self.ambiguous[rows] |= exceeds
        return self.sample_all(index, flip_v)[rows]


def apply_textures(field, data, material_of_pixel, n, v, frame, coat, sheen, aniso, irid, thin, mode="spec"):
    """The per-pixel lobe dicts with the texture-driven parameters substituted (copies; None stays None).  `frame` = the
    (tangent, w, det) of `surface_tangents`, needed for a coat normal texture; `mode` names a control (TEXTURE_CONTROLS)."""
    flip = mode == "v-flipped"
    coat, sheen, aniso, irid, thin = [None if x is None else {k: (a.copy() if isinstance(a, np.ndarray) else a) for k, a in x.items()}
                                      for x in (coat, sheen, aniso, irid, thin)]
    for material_index in np.unique(material_of_pixel):
        rows = np.flatnonzero(material_of_pixel == material_index)
        material = data["materials"][int(material_index)]
        slots = texture_slots(data, material)
        if not slots:
            continue

        def sample(slot):
            return field.use(slots[slot], rows, flip) if slot in slots else None
        if coat is not None:
            t = sample("coat_factor")
            if t is not None and mode != "coat-factor-ignored":
                coat["factor"][rows] = np.clip(coat["factor"][rows] * t[:, 0], 0.0, 1.0)
            t = sample("coat_roughness")
            if t is not None and mode != "coat-roughness-ignored":
                coat["roughness"][rows] = coat["roughness"][rows] * t[:, 1]
            t = sample("coat_normal")
            if t is not None:
                words = data["material_extensions"]["records"][int(material["extension_index"])]
                scale = f32(words[9])
                nr = n[rows]
                tangent, w, det = frame
                tr = orthogonal_tangent(tangent[rows], nr)
                br = np.cross(nr, tr) * (w[rows] * det[rows])[:, None]
                cn = 2.0 * t[:, :3] - 1.0
                cn[:, :2] *= scale
                if mode == "coat-normal-y-flipped":
                    cn[:, 1] = -cn[:, 1]
                nc = normalize(tr * cn[:, 0:1] + br * cn[:, 1:2] + nr * cn[:, 2:3])
                if mode == "coat-normal-ignored":
                    nc = nr
                if "normal" not in coat:
                    coat["normal"] = n.copy()
                coat["normal"][rows] = nc
        if aniso is not None:
            t = sample("aniso")
            if t is not None:
                d = 2.0 * t[:, :2] - 1.0
                if mode == "aniso-direction-y-flipped":
                    d[:, 1] = -d[:, 1]
                length = np.linalg.norm(d, axis=1)
                angle = np.where(length > 1e-6, np.arctan2(d[:, 1], d[:, 0]), 0.0)
                if mode != "aniso-direction-ignored":
                    aniso["rotation"][rows] = aniso["rotation"][rows] + angle
                if mode != "aniso-strength-ignored":
                    aniso["strength"][rows] = np.clip(aniso["strength"][rows] * t[:, 2], 0.0, 1.0)
        if irid is not None:
            t = sample("irid_thickness")
            if t is not None and mode != "irid-thickness-ignored":
                weight = 1.0 - t[:, 1] if mode == "irid-thickness-inverted" else t[:, 1]
                lo, hi = irid["thickness_min"][rows], irid["thickness"][rows]
                irid["thickness"][rows] = lo * (1.0 - weight) + hi * weight
            t = sample("irid_factor")
            if t is not None and mode != "irid-factor-ignored":
                irid["strength"][rows] = np.clip(irid["strength"][rows] * t[:, 0], 0.0, 1.0)
        if sheen is not None:
            t = sample("sheen_color")
            if t is not None and mode != "sheen-color-ignored":
                rgb = srgb_encode(t[:, :3]) if mode == "sheen-color-srgb-encoded" else t[:, :3]
                sheen["color"][rows] = sheen["color"][rows] * rgb
            t = sample("sheen_roughness")
            if t is not None and mode != "sheen-roughness-ignored":
                sheen["roughness"][rows] = sheen["roughness"][rows] * t[:, 3]
        if thin is not None:
            t = sample("thin_weight")
            if t is not None and mode != "thin-weight-ignored":
                thin["weight"][rows] = np.clip(thin["weight"][rows] * t[:, 3], 0.0, 1.0)
            t = sample("thin_color")
            if t is not None and mode != "thin-color-ignored":
                rgb = srgb_encode(t[:, :3]) if mode == "thin-color-srgb-encoded" else t[:, :3]
                thin["color"][rows] = thin["color"][rows] * rgb
    if coat is not None and "normal" in coat:
        coat["mu"] = np.abs(dot(coat["normal"], v))
    return coat, sheen, aniso, irid, thin


def sphere_table(data, textures_ok=False):
    out = []
    uses = uses_specular_extension(data)
    uses_coat = uses_clearcoat(data)
    uses_sheen_scene = uses_sheen(data)
    uses_aniso = uses_anisotropy(data)
    uses_irid = uses_iridescence(data)
    uses_thin = uses_subsurface(data)
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
            coat = material_clearcoat(data, material, textures_ok) or (0.0, 0.0, 1.0)
            ior = spec[0] if spec is not None else 1.5
            entry.update(coat=coat[0], coat_roughness=coat[1], coat_normal_scale=coat[2],
                         group=f"metallic {entry['metallic']:g} ior {ior:g} coat {coat[0]:g}")
        if uses_sheen_scene:
            sheen = material_sheen(data, material, textures_ok) or ([0.0, 0.0, 0.0], 0.0)
            c = sheen[0]
            entry.update(sheen_color=c, sheen_roughness=sheen[1],
                         group=f"{entry.get('group', 'metallic ' + format(entry['metallic'], 'g'))} sheen "
                               f"{c[0]:g},{c[1]:g},{c[2]:g}")
        if uses_aniso:
            aniso = material_anisotropy(data, material, textures_ok) or (0.0, 0.0)
            det = float(np.sign(np.linalg.det(instance_matrix(inst)[:3, :3])))
            entry.update(anisotropy=aniso[0], anisotropy_rotation=aniso[1], mirrored=bool(det < 0),
                         group=f"metallic {entry['metallic']:g} anisotropy {aniso[0]:g}")
        if uses_irid:
            irid = material_iridescence(data, material, textures_ok) or (0.0, 1.3, 100.0, 400.0)
            entry.update(iridescence=irid[0], iridescence_ior=irid[1], thickness_min=irid[2], thickness=irid[3],
                         group=f"metallic {entry['metallic']:g} iridescence {irid[0]:g} ior {irid[1]:g}")
        if uses_thin:
            thin = material_subsurface(data, material, textures_ok) or (0.0, [1.0, 1.0, 1.0])
            entry.update(thin_weight=thin[0], thin_color=thin[1],
                         group=f"metallic {entry['metallic']:g} thin weight {thin[0]:g}")
        layered = sum(map(bool, (uses_coat, uses_sheen_scene, uses_aniso, uses_irid, uses_thin)))
        # ticket 19: a combination scene groups its spheres by the full set of lobe parameters.  The clearcoat grids of ticket 09 (coat
        # over a base with an ior block in some rows) keep their own grouping.
        if layered > 1 or (uses and layered == 1 and not uses_coat):
            tokens = [f"metallic {entry['metallic']:g}"]
            if "ior" in entry:
                tokens.append(f"ior {entry['ior']:g} spec {entry['specular']:g}")
            if entry.get("coat", 0.0) > 0.0:
                tokens.append(f"coat {entry['coat']:g}/{entry['coat_roughness']:g}")
            if any(c > 0.0 for c in entry.get("sheen_color", [0.0])):
                c = entry["sheen_color"]
                tokens.append(f"sheen {c[0]:g},{c[1]:g},{c[2]:g}/{entry['sheen_roughness']:g}")
            if entry.get("anisotropy", 0.0) > 0.0:
                tokens.append(f"aniso {entry['anisotropy']:g}@{entry['anisotropy_rotation']:g}")
            if entry.get("iridescence", 0.0) > 0.0:
                tokens.append(f"irid {entry['iridescence']:g}/{entry['iridescence_ior']:g}/{entry['thickness']:g}nm")
            if entry.get("thin_weight", 0.0) > 0.0:
                tokens.append(f"thin {entry['thin_weight']:g}")
            entry["group"] = " ".join(tokens)
        if textures_ok and texture_slot_bits(data, material):
            # texture closure: every textured sphere is its own gate row (a control that changes one sphere only must not hide in the 1 %)
            entry["group"] = f"metallic {entry['metallic']:g} sphere {k} textures {texture_slot_bits(data, material):#x}"
        out.append(entry)
    return out


def thin_arrays(spheres, instance_of_pixel, variant):
    """Per-pixel {"weight", "color", "variant"} of the thin subsurface block (weight 0 where a sphere has none)."""
    weight = np.array([s.get("thin_weight", 0.0) for s in spheres], dtype=np.float64)
    color = np.array([s.get("thin_color", [1.0, 1.0, 1.0]) for s in spheres], dtype=np.float64)
    return {"weight": weight[instance_of_pixel], "color": color[instance_of_pixel], "variant": variant}


def irid_arrays(spheres, instance_of_pixel, n_dot_v, variant):
    """Per-pixel iridescence dict (strength, film ior, thickness = the maximum, thickness minimum, N.V, variant); strength 0
    where a sphere has no film."""
    def column(key, default):
        return np.array([s.get(key, default) for s in spheres], dtype=np.float64)[instance_of_pixel]
    return {"strength": column("iridescence", 0.0), "ior": column("iridescence_ior", 1.3),
            "thickness": column("thickness", 400.0), "thickness_min": column("thickness_min", 100.0),
            "nv": np.clip(np.asarray(n_dot_v, dtype=np.float64), MIN_NDOTV, 1.0), "variant": variant}


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


def aniso_arrays(spheres, instance_of_pixel, surface, n, variant):
    """Per-pixel anisotropy dict (strength, rotation, orthogonalised world tangent, handedness w, det, variant)."""
    strength = np.array([s.get("anisotropy", 0.0) for s in spheres], dtype=np.float64)
    rotation = np.array([s.get("anisotropy_rotation", 0.0) for s in spheres], dtype=np.float64)
    tangent, w, det = surface
    return {"strength": strength[instance_of_pixel], "rotation": rotation[instance_of_pixel],
            "tangent": orthogonal_tangent(tangent, n), "w": w, "det": det, "variant": variant}


def require_declared_scene(data, textures_ok=False):
    for instance in data["instances"]:
        material = data["materials"][instance["material"]]
        material_specular(data, material)  # only the ior/specular and clearcoat features, without textures
        material_clearcoat(data, material, textures_ok)
        material_sheen(data, material, textures_ok)
        material_anisotropy(data, material, textures_ok)
        material_iridescence(data, material, textures_ok)
        material_subsurface(data, material, textures_ok)
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


def render_frame(root, data, model=APPENDIX_B, threads=None, cast=None, environment=None, visibility=None, exact_environment=True,
                 textures=False):
    """Reference radiance image (h, w, 3) float32 plus masks and the per-sphere table.

    `environment` None = exact quadrature of the literal model; a DECLARED variant dict = the engine's declared
    environment model (the sun term is the same in both).  `visibility`: None (no shadow rays) or a list with
    one boolean array per point light (shadow rays, see point_light_visibility).  `textures` (texture closure): sample the
    lobe textures of the materials (the KHR texture fixtures section); the ROI then drops the pixels whose texture footprint is
    ambiguous."""
    require_declared_scene(data, textures)
    cast = cast or cast_primary_rays(root, data, threads or DEFAULT_THREADS)
    valid, position, normal, instance, (width, height) = cast
    spheres = sphere_table(data, textures)
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
    # F15.3 (ticket 11): KHR_materials_anisotropy of the scene (None without any anisotropy).
    has_aniso = uses_anisotropy(data)
    aniso = env_aniso = None
    if has_aniso:
        surface = surface_tangents(root, data, cast, pick)
        aniso = aniso_arrays(spheres, inst[pick], surface, n, model.get("aniso", "spec"))
        env_aniso = aniso_arrays(spheres, inst[pick], surface, n, (environment or {}).get("aniso", "spec"))
    # F15.3 (ticket 12): KHR_materials_iridescence of the scene (None without any film).
    irid = env_irid = None
    if uses_iridescence(data):
        irid = irid_arrays(spheres, inst[pick], dot(n, v), model.get("irid", "spec"))
        env_irid = irid_arrays(spheres, inst[pick], dot(n, v), (environment or {}).get("irid", "spec"))
    # F15.3 (ticket 16): the thin subsurface block of the scene (None without any).
    thin = env_thin = None
    if uses_subsurface(data):
        thin = thin_arrays(spheres, inst[pick], model.get("thin", "spec"))
        env_thin = thin_arrays(spheres, inst[pick], (environment or {}).get("thin", "spec"))
    field = None
    if textures and uses_textures(data):
        field = TextureField(root, data, cast, pick)
        material_of_pixel = np.array([data["instances"][int(k)]["material"] for k in inst[pick]])
        frame = surface_tangents(root, data, cast, pick) if any("coat_normal" in texture_slots(data, m) for m in data["materials"]) else None
        coat, sheen, aniso, irid, thin = apply_textures(field, data, material_of_pixel, n, v, frame, coat, sheen, aniso, irid, thin,
                                                        model.get("tex", "spec"))
        if environment is not None:
            env_coat, env_sheen, env_aniso, env_irid, env_thin = apply_textures(
                field, data, material_of_pixel, n, v, frame, env_coat, env_sheen, env_aniso, env_irid, env_thin,
                environment.get("tex", "spec"))
    env_variant = (environment or {}).get("dielectric", "extension")
    env_dielectric = (dielectric_arrays(spheres, inst[pick], env_variant)
                      if environment is not None and uses_specular_extension(data) else None)
    radiance = np.zeros((pick.size, 3))
    narrow = np.zeros(pick.size, dtype=bool)
    lit = np.zeros(pick.size, dtype=bool)
    terms = []
    sky = float(data["sky"][0]) if SKY_OVERRIDE is None else float(SKY_OVERRIDE)  # ticket 10 probe fixture: the radiance of a cooked probe
    sun_radius = float(data.get("sun_angular_radius", 0.0))
    points = [light for light in data["lights"] if light["type"] == 1]
    for k, light in enumerate(points):
        visible = None if visibility is None else visibility[k]
        value, light_lit = point_radiance(base, metallic, roughness, n, v, position[pick], light, model, sun_dielectric,
                                          visible, coat, sheen, aniso, irid, thin)
        radiance += value
        lit = lit | light_lit
        terms.append("point-light-shadowed" if visible is not None else "point-light")
    for light in (light for light in data["lights"] if light["type"] == 0):
        toward = -normalize(np.asarray(light["direction"], dtype=np.float64))
        irradiance = np.asarray(light["color"], dtype=np.float64) * float(light["intensity"])
        if sun_radius <= 0:
            raise ValueError("a sun requires a positive angular radius")
        sun, narrow = sun_radiance(base, metallic, roughness, n, v, toward, irradiance, sun_radius, model,
                                   sun_dielectric, coat, sheen, aniso, irid, thin)
        radiance += sun
        toward_all = np.broadcast_to(toward, n.shape)
        lit = lit | (dot(n, toward_all) > 0) | (thin_lit(thin, n, toward_all) if thin is not None else False)
        terms.append("sun-disk")
    if sky > 0:
        if environment is None and not exact_environment:
            # ticket 19: a combination without an exact quadrature (iridescence + anisotropy) is gated by its declared model only;
            # the literal image then carries the direct terms alone (the environment key does not exist for it)
            terms.append("environment-not-modelled")
        elif environment is None:
            aniso_env = None if aniso is None else aniso_environment_inputs(aniso, roughness, n, v)
            material = environment_radiance(base, metallic, roughness, dot(n, v), sky, model, sun_dielectric, aniso_env, irid,
                                            thin)
            if model.get("ms", "off") != "off":  # F15 closure (ADR-0007): the lobe joins the material, so sheen and coat layer it like the rest
                import f15_ms
                material = material + f15_ms.environment_value(base, metallic, roughness, dot(n, v), sky, model, sun_dielectric, model["ms"]) * \
                    f15_ms.isotropic(aniso, pick.size)[:, None]
            if thin is not None and coat is not None and thin["variant"] == "uncoated":
                # ticket 19 control (F15.3 closure): the transmitted lobe escapes the coat.  Split the material radiance into the part
                # without the transmitted lobe (colour 0: the reflected diffuse keeps its 1 - w) and the transmitted term itself.
                reflected = environment_radiance(base, metallic, roughness, dot(n, v), sky, model, sun_dielectric, aniso_env, irid,
                                                 {**thin, "color": np.zeros_like(thin["color"])})
                radiance += layered_environment(reflected, dot(n, v), coat, sheen, sky, model.get("order") == "swapped") + (material - reflected)
            else:
                radiance += layered_environment(material, dot(n, v), coat, sheen, sky, model.get("order") == "swapped")
            if model.get("ms", "off") != "off" and coat is not None:  # the coat's own lobe, not attenuated
                radiance += f15_ms.coat_environment_value(coat, coat_weight(coat), dot(n, v), sky, model, model["ms"])
            terms.append("environment-quadrature")
        else:
            if env_aniso is not None:  # F15.1 (ticket 11, round 3): the declared anisotropic albedo table is indexed by the view azimuth in the lobe frame
                env_aniso = {**env_aniso, "phi": aniso_environment_inputs(env_aniso, roughness, n, v)["phi"]}
            if LEGACY_ENVIRONMENT:
                if any(x is not None for x in (env_dielectric, env_coat, env_sheen, env_irid, env_thin)):
                    raise ValueError("the legacy-environment reference models the base and anisotropy only")
                radiance += legacy_hemisphere_radiance(base, metallic, roughness, n, v, env_aniso, environment)
            else:
                radiance += declared_environment_radiance(base, metallic, roughness, dot(n, v), sky, environment,
                                                          env_dielectric, env_coat, env_sheen, env_aniso, env_irid, env_thin)
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
    if field is not None:  # texture closure: pixels whose UV footprint may select a mip level above 0
        ambiguous = np.zeros(height * width, dtype=bool)
        ambiguous[pick] = field.ambiguous
        roi &= ~ambiguous.reshape(height, width)
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
    thin = uses_subsurface(data)  # a thin carrier does not occlude its own transmitted light: rays start on the light's side
    out = []
    for light in (light for light in data["lights"] if light["type"] == 1):
        n = normalize(normal[pick])
        if thin:
            toward = np.asarray(light["position"], dtype=np.float64)[None, :] - position[pick]
            n = np.where((dot(n, toward) < 0)[:, None], -n, n)
        out.append(shadow_visibility(root, data, position[pick], n, light["position"], threads))
    return out


def command_render(args):
    global EXACT_ANISO_DISK, SKY_OVERRIDE, LEGACY_ENVIRONMENT
    EXACT_ANISO_DISK = bool(getattr(args, "exact_aniso_disk", False))
    LEGACY_ENVIRONMENT = bool(getattr(args, "legacy_environment", False))
    SKY_OVERRIDE = getattr(args, "sky_override", None)
    global SKY_RATIO_CONTROL
    SKY_RATIO_CONTROL = 1.0
    if SKY_OVERRIDE:
        SKY_RATIO_CONTROL = float(snapshot(args.snapshot)[1]["sky"][0]) / float(SKY_OVERRIDE)
    root, data = snapshot(args.snapshot)
    textures = bool(args.textures)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cast = cast_primary_rays(root, data, args.threads)
    visibility = point_light_visibility(root, data, cast, args.threads) if args.visibility == "rays" else None
    main_model = SPECULAR_EXT if uses_specular_extension(data) or uses_clearcoat(data) else APPENDIX_B  # sheen/anisotropy-only: base
    main_model = {**main_model, "ms": "kulla-conty" if args.specular_ms == "on" else "off"}  # ADR-0007: the engine default is on
    controls = {"": main_model}
    for name in args.controls:  # every control keeps the scene's multiple-scattering state, except the ms controls themselves
        model = control_model(name, main_model)
        controls[name] = model if name in MS_CONTROLS else {**model, "ms": main_model["ms"]}
    report = {"schema": 1, "tool": "tools/f15_reference.py", "tool_sha256": sha256_file(__file__), "model": MODEL,
              "snapshot_sha256": scene_digest(root), "frame": data.get("frame"),
              "sun_angular_radius": data.get("sun_angular_radius"), "sky": data["sky"],
              "lights": data["lights"], "visibility": args.visibility, "controls": {}, "deterministic": True}
    if visibility is not None:
        report["shadowed_hits"] = [int((~v).sum()) for v in visibility]
    for name, model in controls.items():
        image, masks, spheres, terms = render_frame(root, data, model, cast=cast, visibility=visibility,
                                                    exact_environment=not args.no_exact_environment, textures=textures)
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
        image, masks, _, terms = render_frame(root, data, model=main_model, cast=cast, environment={**DECLARED, "ms": "table" if args.specular_ms == "on" else "off"},
                                              visibility=visibility, textures=textures)
        write_pfm(out / "declared.pfm", image)
        declared["terms"] = terms
        for name in args.declared_controls:
            control_env = ALL_DECLARED_CONTROLS[name]
            if args.specular_ms == "off" and name not in MS_DECLARED_CONTROLS:
                control_env = {**control_env, "ms": "off"}
            image, *_ = render_frame(root, data, model=main_model, cast=cast, environment=control_env,
                                     visibility=visibility, textures=textures)
            write_pfm(out / f"control-{name}.pfm", image)
            declared["controls"][name] = {"model_variant": control_env, "file": f"control-{name}.pfm"}
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
    p.add_argument("--no-exact-environment", action="store_true",
                   help="skip the exact environment quadrature of the literal image (combinations without one: declared model only)")
    p.add_argument("--exact-aniso-disk", action="store_true",
                   help="integrate the sun disk of narrow anisotropic lobes on a polar grid instead of the engine's spiral rule (no quadrature mask there)")
    p.add_argument("--legacy-environment", action="store_true",
                   help="the declared image is the engine's legacy two-colour hemisphere environment (base and anisotropy only)")
    p.add_argument("--sky-override", type=float, default=None,
                   help="radiance of the environment terms instead of the snapshot sky (a cooked constant probe; black base only)")
    p.add_argument("--textures", action="store_true",
                   help="sample the lobe textures of the materials (texture closure; the ROI drops ambiguous-footprint pixels)")
    p.add_argument("--specular-ms", choices=["on", "off"], default="on",
                   help="F15 closure (ADR-0007): the engine runs --specular-ms on (default: the Kulla-Conty lobe in the literal and the declared model) or off")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.set_defaults(function=command_render)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
