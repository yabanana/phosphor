# 01: One shared BSDF module for resolve, DI and reflections

Roadmap: F15.1 (prefactor)

**What to build:** The primary-surface BSDF duplicated today in the resolve shading, the DI BRDF/target and the reflection sampling/weight becomes one MSL module that all three call, so a lobe added later is implemented once. Every image stays as it is (the model is corrected in ticket 03, not here).

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The three GPU copies call the shared module; no second GGX/Fresnel implementation remains in resolve, DI or reflections (the secondary-hit base reduction may keep its reduced form, documented)
- [ ] `tools/visual_check.sh` 0 pixels against references taken before the change on every bench, API + shader validation clean
- [ ] F13/F14 corpus subset touching DI/reflections green with unchanged numbers; any FP-contraction difference is explained and removed, not tolerated
- [ ] Sponza D2 preset GPU p95 within the decision-11 band of the parent commit (A/B/A, quiet machine)
- [ ] `ctest` and `metal_syntax_check` green

## Comments

- 2026-11-13 (orchestrator): implemented in `shaders/materials/bsdf.h`; the
  resolve (`material_shading.h`: `evaluateLight`, `sunDiskSpecular`, sun-disk
  diffuse, environment term), DI (`restir_common.h: diBRDFWeighted`) and the
  transport (`reflection_common.h: reflectionSpecularBRDF`, GGX sampling pdf)
  call it; expression shapes preserved. Secondary-hit reductions
  (`gi_common.h`, `reflections.metal`) keep their documented base reduction.
- Exactness: five F13-path captures (Sponza D2 preset 720p, Cornell
  restir/RT reflections/DDGI/RTAO, Cornell clustered/SSR/cache/GTAO, Many
  Lights restir/probes, PBR grid with atmosphere/fog/clouds), API validation,
  48 warm-up frames, fixed timestep: linear PFM byte-identical to the base
  (`build/f15-exact/base1` vs `t01`; base repeatability proven first with
  `base1` vs `base2`). Because DI, reflections and GI produce bit-identical
  frames after temporal warm-up, the F13/F14 corpus metrics of these paths
  cannot change; the corpus was not re-run for this ticket.
- `tools/visual_check.sh build/release build/reference-f15base`
  (BENCH8_ARGS="--instances 100000", API + shader validation): 8/8 benches
  0 pixels, no validation message. `ctest` 13/13.
- Pending: Sponza D2 A/B/A on a quiet machine (base binary
  `../f15-base` at 75924b2).
