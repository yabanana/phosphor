# 08: Specular transport, composition and denoiser contract for multi-lobe materials

Roadmap: F15.1, F15.3

**What to build:** The specular transport samples from the shared BSDF with the guide's lobes (lobe choice with a mixture pdf), the composition uses the effective diffuse weight and gains the environment/probe terms that lobes need (e.g. sheen), and the denoiser receives through a new specialised binding the sampled-lobe id, axis (reflect/refract) and roughness: neighbours and history are rejected across lobe ids; `GPUDISurface` stays unchanged. Proven with ior/specular, ready for coat and transmission.

**Blocked by:** 06

**Status:** claimed (08a done and verified 2026-11-13; 08b in progress)

- [x] ior/specular grids pass with RT reflections, SSR and probes; the support report marks every light path of decision 06 for ior/specular
- [ ] Denoiser negative control: using the reflect axis for a refract-axis lobe fails the clip; history across lobe ids rejected (unit-tested on the CPU mirror of the rule)
- [ ] F13 temporal metrics on a specular-extension clip within the F13.6 contract
- [ ] Scenes without features unchanged (visual_check 0 px, D2 within band)
- [x] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
- [x] The composite's environment/GI diffuse uses the effective diffuse weight of extended materials: `tools/f15_reference.py` DECLARED `extension_diffuse` switches from `composite-base` to `extension` and `specular-grid-env-resolve` stays green against it (until then the support report names `transport_and_composite_diffuse_use_base_model`)
- [ ] Carried from ticket 05: a uniform-environment furnace over EXTENDED materials (specular grid at albedo 1) stays ≤ 1 per pixel once the composite uses the extended diffuse weight; indirect/denoised signals recover within the F13.6 recovery criterion after an extension edit; adaptive 2×2 shading reuse (`--adaptive-shading`) leaves feature-bit pixels identical to the run without reuse

## Comments

2026-11-13 — 08a (transport, composite, extended furnace, reference switch)
verified on M5. Code d278b6a/46c447f/5071e60/e88a93d (Sonnet agent,
reviewed). Batch log: `build/f15-exact/battery08.log`.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t08`): GREEN on
  all 18 keys, every control fails.
  - `specular-grid-env-resolve` against the declared model with
    `extension_diffuse = extension`: worst mean 1e-4. The new control
    `declared-extension-diffuse-base` fails.
  - `specular-grid-env` (SSR): worst sphere mean 2.32 %.
  - `@rt`: 4.70 %, against the 5 % environment gate.
  - `@probes`, declared and unfiltered: worst mean 1e-4.
  - `furnace-specular`: max sphere mean 1.00027 (transport).
  - `furnace-specular@resolve`: max pixel 1.0000001, declared worst mean
    4.5e-6.
- Seeds 2 and 3 give the same values: SSR 2.32 %, RT 4.70 / 4.71 %, furnace
  1.00027. The RT margin is systematic, not noise.
- Cause of the RT margin, a declared transport behaviour rather than the
  material. With GI off, RT secondary hits on neighbouring spheres get no
  environment term (base reduction: direct, emitters and GI only), so occluded
  directions are black while the reference ignores inter-reflection. On the
  base grid (roughness up to 1) the same configuration is 24.7 %
  (`build/f15-check/t08-basert`). The specular grid at roughness 0.35 passes
  at 4.7 % with discriminating controls.
- Forced overflow (`tools/f15_transport_overflow_check.sh`): worst
  per-sphere mean |d| 5.3e-5 (limit 2e-3).
- Scenes without features:
  - F13 exact captures 0 px;
  - visual_check 8/8 at 0 px.
- Archive +8 compute (`_lobe` SSR, RT, probe and composite kernels at 1 and
  511). archive_check: every local scenario PASS. hitch_check OK.
- The support report names the ior/specular approximation
  `secondary_rt_hits_use_base_reduction` (decision 06).
- Observation for 08b: on `@probes` the custom denoiser lowers the near-unit
  specular of the ior-0 spheres by about 0.74 %, so that key runs unfiltered.
