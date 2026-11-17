# The GGX specular lobes carry multiple-scattering energy compensation by default

Owner decision, 2026-11-15 ("accesa di default"). Supersedes the part of
[ADR-0006](0006-base-model-follows-gltf-appendix-b.md) and of F15 decision 01
(`.scratch/f15/decisions/01-khr-material-extensions.md`) that fixes the base
specular lobe to the single-scattering form of glTF 2.0 Appendix B. Everything
else of ADR-0006 stands (dielectric `mix(diffuse, specular, F(0.04))`, metal
`specular * F(baseColor)`, diffuse weighted by the dielectric Fresnel).

## Declared deviation from glTF Appendix B

Appendix B has no multiple-scattering lobe: its GGX + height-correlated Smith
specular is a single-scattering model and loses energy on rough surfaces. The
loss is measured by the white-furnace fixture (`ms-furnace@resolve`, constant
environment 1, pure white metal, f0 = f90 = 1): the single-scattering lobe
returns 1 - L(mu) of the light, i.e. a mean deficit of **59 % at roughness 1**
(E_avg 0.41), 20 % at roughness 0.85, 6 % at 0.5. A rough white metal is
visibly darker than a rough white dielectric of the same albedo. The engine
therefore adds the energy the lobe loses back as an extra reciprocal lobe.
This is a deviation from Appendix B on purpose; the independent references
model the same compensation, so no gate compares different models.

## Model

Kulla and Conty, "Revisiting Physically Based Shading at Imageworks", SIGGRAPH
2017 course notes, with the environment form of Fdez-Aguera, "A
Multiple-Scattering Microfacet Model for Real-Time Image-Based Lighting", JCGT
8(1), 2019. With `L(mu) = 1 - E(mu)` the energy that the single-scattering lobe
(Fresnel 1) loses at the cosine `mu`, `E_avg = 2 int E mu dmu`, and the
cosine-weighted Schlick Fresnel `F_avg = f0 + (f90 - f0) / 21`:

- `F_ms = F_avg^2 E_avg / (1 - F_avg (1 - E_avg))`
- light sample: `f_ms = F_ms L(mu_o) L(mu_i) / (pi (1 - E_avg))`, added to the
  specular lobe, times the cosine and the light like every other term;
- environment / GI: `F_ms L(mu_o)` times the radiance L(N) (GI irradiance / pi,
  the declared environment or the legacy hemisphere), exact for a constant
  environment because `2 int L mu dmu = 1 - E_avg` for the tables.

A white metal returns exactly E + L = 1 (furnace), a dielectric F0 0.04 gains at
most 0.2 % (F_ms <= F_avg^2): the furnace bound 1.005 is unchanged.

**Dielectric diffuse complement.** The engine's diffuse weight of a dielectric,
`1 - max3(E_d)`, already refills the energy its specular layer does not return,
the single-scattering loss included. The lobe's own energy `F_ms L` is therefore
taken out of that complement (`1 - max3(E_d + F_ms L)`): without it a dielectric
of Fresnel near 1 (ior 0, a saturated specular colour) reflects the loss twice
(measured +3.6 % in the furnace of the ior x specular grid). For F0 0.04 the
change is below 1e-3.

**Tables** (`tools/ms_table.py` -> `src/renderer/ms_table.h`): `L(mu, roughness)`
on the 64 x 64 grid of the DFG table and `1 - E_avg(roughness)`, as natural
logarithms (L spans four decades; the shader interpolates ln L and
exponentiates), integrated deterministically (half-vector quadrature with the
exact azimuth interval, converged to ~1e-5 at mid angles, checked against a
wi-space midpoint rule and an independent E_avg). They are NOT derived from the
split-sum DFG table, whose 16384-sample Monte Carlo error (~1e-3 in E) is 1-10 %
in L. The tables follow the DFG table in the same GPU buffer.

## Where it applies

Every GGX specular lobe the resolve and the lighting kernels shade:

- the base dielectric / metal lobe and the ior/specular lobe (f0, f90 of the
  material; iridescence uses the film-free f0, declared);
- punctual lights, the sun (disk centre), ReSTIR DI target and shading and the
  clustered DI kernels, the environment term, the GI-fed ambient, and the F13
  composite (so the signal travels with the diffuse ambient exactly like the
  sheen term);
- under sheen (the sheen albedo scaling of the material) and under the clear
  coat (attenuated by `1 - c F_c`), plus the coat's own dielectric lobe
  (f0 0.04, f90 1, coat roughness and normal, factor c, not attenuated).

Declared and not modelled: **anisotropic lobes carry no lobe** (the lost energy
of an anisotropic lobe depends on the azimuth of the view against the lobe axis,
which a table of L(mu, roughness) cannot follow: the effective isotropic roughness
of the declared environment fit gained +3.3 % in the white furnace of a pure metal
at strength 1, measured, so none is applied; an anisotropic pixel keeps the
Appendix B energy loss), the transmitted lobes, and the secondary-hit
shading of reflection rays, probe captures and GI rays (they keep their
documented base reductions). While a pipeline compiles, the fallback is the
generic variant of the same option (it carries the constant too).

## Off switch

`--specular-ms on|off` (default **on**; function constant `SPECULAR_MS`,
reserved index 25, set by `PipelineCache::request` on the functions that run
`shadeSurface`, runtime flag bits `DI_SPECULAR_MS` / `REFLECT_COMPOSE_SPECULAR_MS`
in the base DI and composite kernels); the report carries
`rendering.specular_ms`. With `off` the engine is bit-identical to the tree
before the option existed (ADR-0006 model): the evidence is in
`.scratch/f15/issues/03-base-model-appendix-b.md` (2026-11-15 section).

## Consequences

- visual_check references, the Release references of `archive_check`, the pipeline
  archive (every shading pipeline now specialises constant 25), the independent
  references and every F13/F14 reference that evaluates the base GGX lobe are
  re-taken or extended with the lobe; the F12 diagnostic signal stays pure
  Lambert (Mitsuba Cornell oracle).
- Cost is one table lookup per light sample and per pixel; Sponza D2 against the
  decision-11 band (median A/B/A GPU p95 delta <= 0.10 ms, absolute <= 16.67 ms)
  is measured by the orchestrator on a quiet machine.
