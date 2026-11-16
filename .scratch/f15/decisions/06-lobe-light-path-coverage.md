# 06: Lobe × light-path coverage matrix

Type: grilling
Status: resolved
Blocked by: 01, 04

## Question

For every lobe (base diffuse/specular with ior+specular, clearcoat, sheen,
anisotropy, iridescence, specular transmission + volume, diffuse transmission,
subsurface) which light paths evaluate it exactly, which approximate it (with
the declared approximation) and which ignore it: resolve analytic lights and
environment, DI (ReSTIR/clustered), GI composition (effective diffuse albedo),
reflections (sampling and weight), denoiser demodulation, secondary RT hits,
shadow casters (CSM and RT), and the CPU reference.

## Answer

Rule: the primary visible surface evaluates every implemented lobe for direct
light and for its environment/specular transport; secondary RT hits keep the
current base reduction (diffuse albedo + emission, GGX for reflected hits);
anything not exact is a declared approximation in the support report and in the
reference report. One shared MSL BSDF module (prefactor) serves resolve, DI,
reflections and composite; the Python reference is independent.

| Lobe | Analytic lights + sun disk (resolve, DI) | Environment / ambient | GI composition | Specular transport (RT/SSR/probes) | Denoiser | Secondary RT hits | Shadow casters |
|---|---|---|---|---|---|---|---|
| Base + ior + specular | exact (spec Fresnel, f0 colour, f90 = weight) | existing Karis split-sum fit with f0/f90 (approximation, already in use) | diffuse × `(1 − weight·max F)` | GGX sampling, weight uses spec Fresnel | edge-stopping roughness of the significant lobes | base reduction | unchanged |
| Clearcoat | exact (coat GGX, F0 0.04, coat normal; base and emission × `1 − c·F_c`) | coat term with the same fit at coat roughness | diffuse × coat attenuation | one sample chooses coat or base lobe by albedo weight (mixture pdf, MIS-correct one-sample) | edge-stopping roughness of the significant lobes | ignored (declared) | unchanged |
| Sheen | exact (Charlie D + Charlie V, base × E-table scaling) | Charlie directional albedo × environment (exact for a constant environment) | diffuse × sheen albedo scaling | not sampled; covered by the environment/probe term (declared) | edge-stopping roughness of the significant lobes | ignored | unchanged |
| Anisotropy | exact (anisotropic GGX D and V, rotated tangent) | fit with bent normal (Sample Renderer method, declared) | n/a | anisotropic VNDF sampling, exact weight | edge-stopping roughness of the significant lobes | ignored | unchanged |
| Iridescence | exact (thin-film Fresnel at V·H per light sample) | thin-film Fresnel at N·V | diffuse × `(1 − max F_irid)` | weight uses thin-film Fresnel at the sampled V·H | edge-stopping roughness of the significant lobes | ignored | unchanged |
| Specular transmission + volume | thin-walled: BTDF on the back hemisphere; solid volume: direct light through two interfaces excluded (caustic, declared) | via the transport fallback | diffuse × `(1 − transmission)` | one sample chooses reflection or refraction by Fresnel; rough GGX BTDF; ≤ 4 interface crossings; Beer–Lambert along the real inside path; probes/environment along the refracted ray without RT (declared) | IOR-scaled transmission roughness for edge-stopping | opaque base reduction (declared) | opaque in CSM and RT (declared; coloured shadows deferred) |
| Subsurface (volumetric) | diffuse part written to the diffuse-irradiance target; specular unchanged | included in the irradiance target | included in the irradiance target | unchanged | diffuse denoised before the diffusion pass | base reduction | unchanged |
| Subsurface (thin) / diffuse transmission | Lambert BTDF for lights on the back hemisphere | back-side environment | GI irradiance at −N for the transmitted part | unchanged | edge-stopping roughness of the significant lobes | base reduction | unchanged |

CPU reference: every lobe exact from the specification; transport differences
(split-sum fit, screen-space diffusion, excluded caustics) are measured, never
hidden.

Corrections after review (2026-11-13): the F13 denoiser filters BRDF-weighted
radiance with roughness/normal edge-stopping and does not demodulate by albedo
(`shaders/denoise.metal`), so the denoiser column gives the edge-stopping
roughness the lobe guide supplies, not an albedo. Emissive materials under a
clearcoat keep their uncoated emission when they act as sampled emitters in DI
(declared approximation; the camera-visible emission is coated).
Review pass additions (2026-11-13): (1) refraction samples ignore the
reflection RT roughness band and are always traced with RT (decision 09);
(2) emitter ownership extends the F13 partition: the coat lobe's DI/RT emitter
share uses the coat roughness, and thin-walled transmission gives emitters hit
by refracted rays to the transport and delta lights to DI.
Owner decisions and review findings (2026-11-13, second pass):
- Base model: glTF Appendix B everywhere (ADR-0006, ticket 03); the matrix rows
  above are relative to the corrected base.
- Shadow casters (supersedes "opaque"): transmissive casters attenuate and tint
  light along straight shadow rays (per-surface factor + Beer–Lambert from the
  signed entry/exit distance sum, order-independent) in RT sun shadows, DI
  visibility, GI shadow rays and reflection secondary lighting; CSM uses a
  single-layer transmittance map; caustics excluded (declared). Non-shadow GI
  and probe rays still hit glass as an opaque base reduction (declared).
- Denoiser: a specialised binding carries sampled-lobe id, axis
  (reflect/refract) and roughness; neighbours and history are rejected across
  lobe ids; `GPUDISurface` is unchanged.
- Sheen environment term lives in the composition (probe/GI radiance at N ×
  Charlie albedo), since the resolve omits ambient specular when the transport
  runs.
- The indexed overflow fallback produces the lobe guide and the subsurface
  irradiance target like the meshlet path (forced-overflow captures).
- Amendment (2026-11-13, ticket 11 harness): environment term of anisotropy.
  The Sample Renderer's bent normal changes only the fetch direction, so a
  constant environment cannot gate it. The declared model keeps the bent
  normal for the fetch direction and evaluates the split-sum DFG term at the
  effective roughness alpha_eff = sqrt((alpha_t^2 + alpha_b^2) / 2), not at
  the material roughness.
  - Against the exact anisotropic quadrature on the fixture:
    - dielectric: 1.5–3.5 %;
    - metal, strength 0.5: 2.3–3.4 %;
    - metal, strength 1: 6.4–7.0 %.
  - The physical gate stays the transport's (anisotropic VNDF sampling)
    against the exact quadrature.
  - A non-constant-environment fixture with an analytic reference, to gate
    the bent-normal fetch, is open (ticket 22 or a later fixture).
- Amendment (2026-11-16, ticket 11 round 3): the declared environment of an anisotropic material no longer runs the DFG table at an effective
  roughness. It reads the directional albedo (A, B) of the anisotropic lobe itself from `renderer/aniso_table.h` (generated by
  `tools/aniso_table.py`: 24 N.V x 9 azimuth of V against the lobe axis x 17 roughness x 9 strength nodes, four-linear lookup, the split sum
  `f0 A + f90 B` and the diffuse weight `1 - max E_dielectric` are unchanged). The resolve (all `anisoActive` pixels, including the
  combinations with coat, sheen, iridescence, ior/specular and thin subsurface that go through `bsdfSpecEnvIridAB`) uses it; the probe fit,
  the composite and the reflection-weight code keep the rms fit (declared, the transported specular signal is gated against the exact
  quadrature). Against the exact quadrature on the ticket grid the declared model is now within 0.1 % (was 6.5 %); the old fit stays as the
  negative control `declared-aniso-rms-fit`. Multiple-scattering compensation stays off for anisotropic lobes (ADR-0007).
