# 10: Subsurface model and its glTF encoding

Type: grilling
Status: resolved
Blocked by: 01, 03, 04

## Question

Which parameterisation does Phosphor's subsurface feature use (no ratified
Khronos extension; diffuse_transmission is RC, volume_scatter is an open PR),
which glTF encoding(s) the loader accepts, which real-time technique renders it
(screen-space diffusion profile, diffuse transmission, ray-traced), and which
independent reference proves it?

## Answer

- No ratified glTF subsurface exists: diffuse_transmission is RC,
  KHR_materials_scatter (PR #2579, head `a6cd5b4`) is a draft that replaces
  volume_scatter and diffuse_transmission and mirrors OpenPBR.
- Phosphor's subsurface feature uses the OpenPBR subsurface semantics (public
  ASWF specification): weight, multiple-scattering colour, per-channel mean free
  path (radius × scale, metres) and Henyey–Greenstein anisotropy, each with an
  optional texture; volumetric (solid) or thin-walled mode.
- Inputs: (a) Phosphor's own serialised material description (a versioned JSON
  file that can override materials of a loaded glTF by name) — used by the
  fixtures and by consumers such as Track Studio; (b) glTF draft
  KHR_materials_scatter with KHR_materials_volume (pinned head, mapped with the
  draft's Kulla–Conty albedo inversion, flagged `draft` in the support report);
  (c) KHR_materials_diffuse_transmission (RC) mapped to the thin-walled mode.
  Owner confirmation requested for accepting a draft extension in the loader.
- Rendering: thin mode = diffuse-transmission lobe; volumetric mode = Burley
  normalised diffusion in screen space on a diffuse-irradiance target written by
  the resolve/DI for subsurface pixels (Khronos/Golubev form: 21/55 samples,
  metres→pixels radius, sub-pixel skip, same-material and depth test), then
  recombined with the specular terms.
- Track Studio case: ABeautifulGame plus a material-description file for the
  four kings/queens carrying the MaterialX values (weight map, colour = base
  colour texture, radius = base colour texture, scale 0.003 / 0.001 m), with the
  conversion documented in the mapping note (ADR-0001).
- Reference: Mitsuba 3.9.1 volumetric path tracing with a homogeneous medium
  inside a dielectric boundary (random walk, ground truth). Decision rule fixed
  now: if the screen-space diffusion misses the decision-11 bound on the piece
  mask, the hybrid RT single-scattering variant is implemented within F15
  (timeboxed, ≤ 2 weeks); otherwise it stays a candidate.

## Comments

- 2026-11-13, owner (AskUserQuestion): "OpenPBR + bozza KHR" — confirmed.
  Recorded as ADR-0005.
