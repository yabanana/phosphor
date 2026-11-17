# 09: Transmission technique and the F15/F16 boundary

Type: grilling
Status: resolved
Blocked by: 04, 06

## Question

How does F15 render specular transmission and volume attenuation for surfaces
of the opaque visibility buffer (ray-traced refraction in the specular
transport, opaque colour pyramid, probes fallback), how do transmissive
surfaces cast shadows, and what is left to F16 (blended transparency, OIT,
liquids)?

## Answer

- F15 renders specular transmission with ray-traced refraction inside the
  existing specular transport of the visibility path: glass stays in the opaque
  visibility buffer (so its reflective lobes keep DI, reflections and the
  denoiser), each pixel's single transport sample chooses reflection or
  refraction by Fresnel, refracted rays cross at most 4 interfaces with
  Beer–Lambert attenuation over the real inside length (`thicknessFactor` is a
  raster hint and unused with RT, as the extension allows), thin-walled
  materials refract without offset. Ray budget per pixel is unchanged; extra
  traces only for interior crossings.
- Without `--rt on` the transmitted radiance comes from reflection probes or the
  environment along the refracted direction: declared approximation in the
  support report.
- Shadows of transmissive casters stay opaque in CSM and RT shadows (declared);
  attenuated/coloured shadows are deferred.
- F16 keeps blended transparency: OIT (F16.1), raster glass/liquids with a
  colour pyramid and absorption (F16.2), multi-layer ordering. Dispersion is a
  candidate for F16.2.
- Rationale: the opaque-colour-pyramid baseline would require removing glass
  from the visibility pass and a separate lit transparent pass — F16
  infrastructure — while the RT path reuses existing, validated machinery
  (METHOD: first path correct with existing tools). Owner confirmation
  requested because it fixes the F15/F16 boundary (ADR-0004).
- Reference: Mitsuba 3.9.1 rough dielectric + homogeneous absorbing medium on
  the same snapshot (independent ground truth), converged accumulation of the
  engine image; thresholds in ticket 11.

## Comments

- 2026-11-13, owner (AskUserQuestion): "RT in F15, raster in F16" — confirmed.
  Recorded as ADR-0004.
- 2026-11-13, review pass: `reflection_rt` traces RT only inside the
  roughness band `1 − smoothstep(rtRoughnessLow 0.2, rtRoughnessHigh 0.4)`
  (`reflections.metal:105`, `reflection_settings.h:8`) and otherwise falls back
  to SSR/radiance cache/probes. SSR cannot see behind the glass that occupies
  the pixel, so refracted samples are always traced when RT is on, whatever
  the roughness; without RT they use probes/environment, never SSR.
- Emitter ownership (no double count): refracted rays own the emitters they hit
  (area/emissive triangles); the DI BTDF term of thin-walled transmission
  covers only delta lights (directional, point, spot), which rays cannot hit —
  the same partition idea as F13's `DI_REFLECTED_EMITTER_SPECULAR`.
- 2026-11-13, owner (second AskUserQuestion): transmissive shadows are
  attenuated and coloured in F15 (supersedes "opaque"); ADR-0004 amended.
- Review: interface traversal must accept the inside back faces of the
  transmissive medium (`reflections.metal:21` rejects back faces of single-sided
  materials today).
