# 01: One shared BSDF module for resolve, DI and reflections

Roadmap: F15.1 (prefactor)

**What to build:** The primary-surface BSDF duplicated today in the resolve shading, the DI BRDF/target and the reflection sampling/weight becomes one MSL module that all three call, so a lobe added later is implemented once. Every image stays as it is (the model is corrected in ticket 03, not here).

**Blocked by:** None (can start immediately)

**Status:** resolved (verified 2026-11-13, bac820d) with open criteria: literal F13/F14 corpus subset not run for this ticket (exactness argument only); metal_syntax_check not recorded

- [x] The three GPU copies call the shared module; no second GGX/Fresnel implementation remains in resolve, DI or reflections (the secondary-hit base reduction may keep its reduced form, documented)
- [x] `tools/visual_check.sh` 0 pixels against references taken before the change on every bench, API + shader validation clean
- [ ] F13/F14 corpus subset touching DI/reflections green with unchanged numbers; any FP-contraction difference is explained and removed, not tolerated — PARTIAL (audit 2026-11-15): argued by five byte-identical F13 captures (base vs t01), literal corpus subset not run
- [x] Sponza D2 preset GPU p95 within the decision-11 band of the parent commit (A/B/A, quiet machine) — D2 band MET under the owner's option A (ticket 24: 2026-11-15 median -0.038 ms against ticket 03 + 26fe229, 0 GPU allocations; round-2 quiet rotation t03 -> final -0.059 ms; the final binary's rotation is pending the final battery (ticket 24))
- [ ] `ctest` and `metal_syntax_check` green — PARTIAL (audit 2026-11-15): ctest 13/13 recorded, metal_syntax_check not recorded

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
- 2026-11-13 15:05, Sponza D2 A/B/A base 75924b2 vs HEAD 2e1057a (tickets
  01–04 + review fixes), `build/f15-d2/head-2e1057a`: GPU p95 pairs
  15.57/15.42, 15.45/15.65, 15.21/15.87 ms (median delta +0.20 ms) with the
  sky sentinel disturbed in 5 of 6 runs (p95 0.69–0.81 vs clean 0.674–0.677;
  WindowServer 40–60 %, Chrome 50 %): not a valid band measurement. Per-pass
  p50 attribution: deterministic +0.007 ms Material resolve, +0.006 ms
  material lighting guides, +0.005/+0.003 ms composites (DFG table reads of
  ticket 03), stochastic reflection pass −0.055…+0.096 ms (noise). Absolute
  GPU p95 15.42–15.87 ms ≤ 16.67 ms, 0 GPU allocations. Formal band check
  re-run at closure (ticket 24) on a quiet machine. The earlier 11:3x attempt
  (Track Studio cargo tests at 1600 % CPU) is discarded.

2026-11-15 — closure audit (ticket 24):

- 1 shared module, no second GGX/Fresnel: MET (bac820d; bsdf.h called from material_shading.h, restir_common.h, reflection_common.h; remaining duplicates removed in ticket 03). Ticked.
- 2 visual_check 0 px, API + shader validation: MET (8/8 vs reference-f15base, no validation message; run on the ticket-01 tree, before the 6a64b0e defect existed). Ticked.
- 3 F13/F14 corpus subset: PARTIAL (five F13 captures byte-identical base vs t01 with repeatability proven; literal corpus not run). Open.
- 4 D2 band: NOT MET (first run invalid, sentinel disturbed 5/6; later runs in ticket 24 fail the band). Open, ticket 24.
- 5 ctest + metal_syntax_check: PARTIAL (ctest 13/13; metal_syntax_check not recorded). Open.
