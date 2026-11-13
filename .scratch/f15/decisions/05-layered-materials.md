# 05: Layered materials for terrain and complex surfaces

Type: research
Status: resolved
Blocked by: none

## Question

What layering models are established for real-time terrain and complex
surfaces (parameter blending vs shaded blending, height-based blending, mask
sources, normal blending such as RNM/whiteout/UDN, MaterialX/OpenPBR mix and
layer semantics, Unreal Substrate horizontal/vertical layering) and which
energy rules hold for each? Which bounded stack sizes and costs are reported?
Findings: `.scratch/f15/research/layered-materials.md`.

## Answer

Findings: [layered-materials.md](../research/layered-materials.md).

- Horizontal mix (MaterialX/OpenPBR `mix`, lerp of BSDFs) is energy conserving
  by construction; vertical layering uses albedo scaling
  (`f_top + T_top·f_base`). Parameter-space lerp (one BSDF evaluation) is the
  standard real-time approximation (Substrate fallback, HDRP LayeredLit) but is
  not proven energy conserving for GGX+Lambert: the suite must test it.
- Normal blending: surface-gradient sum (Mikkelsen 2020) is exact at weight 1,
  returns the geometric normal at zero coverage and supports per-layer UV
  frames; whiteout/RNM also pass the endpoints; linear and UDN fail.
- Bounds in production: 4 layers per pixel (HDRP LayeredLit, Unreal weightmaps
  of 4); no ratified glTF layering extension exists.
- No modern first-party terrain timings: Phosphor must measure its own.
- Decision (F15.4 baseline): up to 4 layers per pixel from one RGBA weight
  mask (plus optional height-based contrast that is exact at the endpoints),
  layers are ordinary material descriptions, base parameters blended in
  parameter space then one BSDF evaluation, normals by surface-gradient
  blending, weights normalised to sum to 1, the layer-presence bitmask is part
  of the feature class. Tests: exact equality at weights 0/1 (pixels, visibility
  path), monotonic transitions, white furnace ≤ 1 with an additive negative
  control that must fail, deviation from the BSDF-space mix recorded, cost per
  active-layer count on a terrain fixture.
