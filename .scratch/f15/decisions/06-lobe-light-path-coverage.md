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
| Base + ior + specular | exact (spec Fresnel, f0 colour, f90 = weight) | existing Karis split-sum fit with f0/f90 (approximation, already in use) | diffuse × `(1 − weight·max F)` | GGX sampling, weight uses spec Fresnel | combined directional albedo | base reduction | unchanged |
| Clearcoat | exact (coat GGX, F0 0.04, coat normal; base and emission × `1 − c·F_c`) | coat term with the same fit at coat roughness | diffuse × coat attenuation | one sample chooses coat or base lobe by albedo weight (mixture pdf, MIS-correct one-sample) | combined albedo | ignored (declared) | unchanged |
| Sheen | exact (Charlie D + Charlie V, base × E-table scaling) | Charlie directional albedo × environment (exact for a constant environment) | diffuse × sheen albedo scaling | not sampled; covered by the environment/probe term (declared) | combined albedo | ignored | unchanged |
| Anisotropy | exact (anisotropic GGX D and V, rotated tangent) | fit with bent normal (Sample Renderer method, declared) | n/a | anisotropic VNDF sampling, exact weight | combined albedo | ignored | unchanged |
| Iridescence | exact (thin-film Fresnel at V·H per light sample) | thin-film Fresnel at N·V | diffuse × `(1 − max F_irid)` | weight uses thin-film Fresnel at the sampled V·H | combined albedo | ignored | unchanged |
| Specular transmission + volume | thin-walled: BTDF on the back hemisphere; solid volume: direct light through two interfaces excluded (caustic, declared) | via the transport fallback | diffuse × `(1 − transmission)` | one sample chooses reflection or refraction by Fresnel; rough GGX BTDF; ≤ 4 interface crossings; Beer–Lambert along the real inside path; probes/environment along the refracted ray without RT (declared) | combined reflect+transmit albedo | opaque base reduction (declared) | opaque in CSM and RT (declared; coloured shadows deferred) |
| Subsurface (volumetric) | diffuse part written to the diffuse-irradiance target; specular unchanged | included in the irradiance target | included in the irradiance target | unchanged | diffuse denoised before the diffusion pass | base reduction | unchanged |
| Subsurface (thin) / diffuse transmission | Lambert BTDF for lights on the back hemisphere | back-side environment | GI irradiance at −N for the transmitted part | unchanged | combined albedo | base reduction | unchanged |

CPU reference: every lobe exact from the specification; transport differences
(split-sum fit, screen-space diffusion, excluded caustics) are measured, never
hidden.
