# 08: Specular transport, composition and denoiser contract for multi-lobe materials

Roadmap: F15.1, F15.3

**What to build:** The specular transport samples from the shared BSDF with the guide's lobes (lobe choice with a mixture pdf), the composition uses the effective diffuse weight and gains the environment/probe terms that lobes need (e.g. sheen), and the denoiser receives through a new specialised binding the sampled-lobe id, axis (reflect/refract) and roughness: neighbours and history are rejected across lobe ids; `GPUDISurface` stays unchanged. Proven with ior/specular, ready for coat and transmission.

**Blocked by:** 06

**Status:** resolved (2026-11-13; refract-axis clip control carried to ticket 13, D2 band to ticket 24)

- [x] ior/specular grids pass with RT reflections, SSR and probes; the support report marks every light path of decision 06 for ior/specular
- [ ] Denoiser negative control: using the reflect axis for a refract-axis lobe fails the clip; history across lobe ids rejected (unit-tested on the CPU mirror of the rule) — rule and history rejection unit-tested, GPU lobe-key controls fail the clip; the refract-axis CLIP control needs refraction → carried to ticket 13
- [x] F13 temporal metrics on a specular-extension clip within the F13.6 contract
- [ ] Scenes without features unchanged (visual_check 0 px, D2 within band) — visual_check and F13 exact captures 0 px; D2 band → ticket 24
- [x] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
- [x] The composite's environment/GI diffuse uses the effective diffuse weight of extended materials: `tools/f15_reference.py` DECLARED `extension_diffuse` switches from `composite-base` to `extension` and `specular-grid-env-resolve` stays green against it (until then the support report names `transport_and_composite_diffuse_use_base_model`)
- [x] Carried from ticket 05: a uniform-environment furnace over EXTENDED materials (specular grid at albedo 1) stays ≤ 1 per pixel once the composite uses the extended diffuse weight; indirect/denoised signals recover within the F13.6 recovery criterion after an extension edit; adaptive 2×2 shading reuse (`--adaptive-shading`) leaves feature-bit pixels identical to the run without reuse

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

2026-11-13 — 08b (denoiser lobe contract, specular-extension clip, recovery,
2x2) verified on M5. Code e205af2…e990908 (Sonnet agent, reviewed;
integrated with the parallel ticket-09 harness, conflicts resolved by hand),
archive 16d6a50 and 199fa13. Batch log: `build/f15-exact/battery08b.log`.
- Contract:
  - `GPUSpecularSample.flags` carries the sampled lobe id (bits 8-9) and
    axis (bit 10); the history state word stores the key (bits 20-22), and
    no record grows.
  - The `_lobe` denoiser kernels reject neighbours and history across keys.
    Lobe 1 (coat) takes the coat roughness from the guide.
  - CPU mirror unit tests cover packing, rejection across ids and axes, the
    roughness that follows the lobe, and the refract-axis rule (including
    total internal reflection).
- `tools/f15_clip_check.py` (`build/f15-clip-t08b`): ALL EXPECTATIONS MET.
  - Static clip, custom against the 8 spp control: p95 0.0551 (0.06), max
    0.1204 (0.15), flicker 0.0198 (0.02). The margins are thin.
  - The raw 1 spp control fails.
  - Recovery after a strong runtime edit: the probe spheres recover in
    0 frames. A control delayed by 5 frames fails, at 5 > 4.
  - Lobe-key flip control: grid p95 0.1003, flicker 0.0292, so it fails.
  - 2x2 adaptive shading: 0 of 282 626 feature-sphere pixels differ. Reuse
    is live (1 101 reused pixels on the neutral sphere); the control scene
    changes 2 276.
- Gate design decided after the first measurement: on the edit clip the grid
  ROI's steady metrics include the edit frames by construction (p95 0.0607,
  flicker 0.0203). So the steady metrics are gated on the static clip and
  the edit clip gates recovery. Recorded here as a post-measurement choice
  with that rationale.
- Diagnostic, not gating: the ior-0 near-mirror spheres alone exceed the F13
  thresholds (weight 1: p95 0.098, flicker 0.072). A base metal at roughness
  0.35 shows flicker 0.0285 too: this is the 1 spp strong-specular noise
  level, not an extension defect.
- Found and fixed (039e22b): since ticket 06 (6a64b0e), `di_receiver_pack`
  left buffer 3 unbound in scenes without features, and shader validation
  aborted every DI scene ("Buffer binding at index 3 ... never set"). The
  ticket-06 battery ran with API validation only and missed it. The F13
  mirror and specular-grid-env now run clean under shader validation.
  - The full F13/F14 corpus is rerun at closure (ticket 24).
- `tools/f15_check.py --run --expect green` (`build/f15-check/t08b2`): GREEN
  on all 25 keys; the clearcoat keys fail as pending (ticket 09).
- Other checks:
  - transport and lobe-guide overflow checks PASS;
  - F13 exact captures and visual_check 0 px;
  - archive_check every local scenario PASS;
  - hitch_check OK.
- Observations recorded for later tickets:
  - the custom denoiser lowers the near-unit ior-0 specular by 0.74 % on
    `@probes` and blurs DI metal highlights by up to 18 % per pixel; the
    lobe contract changes neither, and the F13 clip contract does not
    gate per-pixel blur;
  - with mixture sampling (ticket 09) the denoiser key must stay stable per
    pixel.
