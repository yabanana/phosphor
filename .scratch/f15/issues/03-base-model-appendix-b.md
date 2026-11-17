# 03: Base model conforms to glTF Appendix B (owner decision) and baselines are re-taken

Roadmap: F15.1, F15.3 (prefactor)

**What to build:** Owner decision 2026-11-13 (ADR-0006): the shared BSDF module implements glTF 2.0 Appendix B — dielectric `mix(diffuse, specular, F(0.04))`, metal `specular · F(baseColor)`, `mix(dielectric, metal, metallic)` — and the environment/GI diffuse is weighted by the dielectric directional Fresnel, in resolve, forward, DI, reflection weights and composition. The F12 diagnostic signal `gi_reference_diffuse` stays pure Lambert so the Mitsuba Cornell oracle remains valid. Every baseline that depends on the model is re-taken deliberately with before/after evidence.

**Blocked by:** 01, 02

**Status:** resolved (verified 2026-11-13, aa58b66, 1ffc724) with open criteria (2026-11-16, round 2: multiple-scattering compensation is ON by default by owner decision, ADR-0007 (`--specular-ms on|off`, tables src/renderer/ms_table.h); with the option off the engine is bit-identical to 3ef8f41 (f15_exact x5 cmp, visual_check 8/8 0 px); keys 1-100 of the full f15_check re-run pass every gate against the compensated references with every control failing (build/f15-ms/full4), ms-furnace@resolve 0.00016 against the lobe-off control 0.60; keys 101-148 were captured but NOT evaluated and the F13/F14 corpus clips that need the MS-aware reference re-render are pending the orchestrator's final battery; Sponza D2 band recorded as met in ticket 24 under owner option A (720ee63: median -0.038 ms vs the ticket-03 baseline plus 26fe229, +0.020 ms vs 2e1057a), the checkbox line here still carries the older "not met" figure; the closure-log before/after diff is DELIVERED (ticket 24, 2026-11-16 'closure log: visual_check references per bench': base -> appB and appB -> final per bench, 0 px against `reference-f15-final` on all 8 benches on the final battery); archive_check / hitch_check are recorded in ticket 24 on 720ee63 (archive_check pass, hitch_check 0 hitches, 21 / 21 archive hits, 0 render-thread compiles) and in ticket 21 on 038eb10 (archive_check 18 hits / 0 misses, hitch_check 0 hitches 21 hits 0 misses); still pending the final battery (ticket 24): the final-tip D2 rotation and the evaluation of keys 101-148)

- [x] Ticket-02 base grids (all metallic values) and furnace pass decision-11 bounds; negative control (old weighting) fails
- [x] visual_check references re-taken; before/after diff per bench documented in the closure log (expected change: intermediate metallic and ambient/GI diffuse) — PARTIAL (audit 2026-11-15): references retaken (reference-f15-appB, 8/8 0 px) and diff recorded here; the closure-log entry is a ticket 24 deliverable — MET 2026-11-16: per-bench table in ticket 24 ("closure log: visual_check references per bench"), base -> appB and appB -> final
- [x] Declared material model of the independent references updated where they model the engine BSDF (`tools/f13_reference.py` and the clip references); affected Mitsuba clip references re-rendered with battery pacing; F13/F14 corpus green on the new baseline; F12 oracle unchanged
- [x] Sponza D2 GPU p95 re-measured A/B/A (within the decision-11 band; cost change expected ≈ 0); 0 GPU allocations — D2 band MET under the owner's option A (ticket 24: 2026-11-15 median -0.038 ms against ticket 03 + 26fe229, 0 GPU allocations; round-2 quiet rotation t03 -> final -0.059 ms; the final binary's rotation is pending the final battery (ticket 24))
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames — MET 2026-11-16 on the cumulative tip: the runs are recorded in ticket 24 (720ee63: archive_check pass, hitch_check 0 hitches 21 / 21 archive hits 0 render-thread compiles, 0 GPU allocations) and ticket 21 (038eb10: archive_check 18 hits / 0 misses, hitch_check 0 hitches 21 hits 0 misses); the audit's 'no run recorded in this ticket' (2026-11-15) is history. The caveat stays: archive_check compares with a same-mode no-archive baseline (archive_check there compares against a same-mode no-archive baseline: 12 px at 1 LSB on bench 1 since ticket 03 under shader validation)

## Comments

- 2026-11-13 (orchestrator): shared BSDF now Appendix B (diffuse weighted by
  the dielectric Fresnel F(0.04)); environment/GI diffuse weighted by
  1 − E(0.04) and environment specular from the engine's DFG table instead of
  the Karis fit (the fit under-estimated E at grazing views and by up to 14 %
  at roughness 0.3–0.6, which made the transport furnace gain energy:
  dielectric max 1.076 with Karis, 1.009 with the table). `tools/f13_reference.py`
  declares the Appendix B diffuse weighting (numerically inert for the F13
  fixtures, whose metallic is 0 or 1: no reference clip re-render needed). The
  F12 diagnostic signals stay pure Lambert.
- Shader-validation defect found and worked around: fragment functions read
  the DFG table through `FrameConstants::dfgTable` (evidence in
  `docs/research/2026-11-13-f15-shader-validation-constant-arrays.md`).
- Remaining shared-module duplicates removed (review finding on ticket 01):
  reflection-probe F0/DFG and the probe-prefilter and RT-regularisation alpha
  floors call `bsdf.h`; verified bit-identical on the probes and RT reflection
  captures before the model change.
- GPU: `tools/f15_check.py --run --expect green` → GREEN on M5 (direct grid all
  metallic rows p99 ≤ 0.09 %; environment grid within 5 %; furnace-resolve max
  1.00000; furnace-base max sphere mean 1.00001; every negative control fails).
  Captures `build/f15-check/t03e`.
- visual_check: before/after vs `build/reference-f15base` = intended change
  (PSNR 29.9–55.1 dB, max delta 2–64; metals brighter with the DFG table);
  new references `build/reference-f15-appB` (BENCH8_ARGS="--instances 100000"),
  verify run 8/8 at 0 px, no validation message.
- Pending: F13/F14 corpus on the new baseline (≈ 2.5 h, serial GPU) and the
  Sponza D2 A/B/A on a quiet machine (the first attempt was contaminated by an
  external 1600 % CPU job, sky sentinel 0.74–1.12 ms; discarded).
- F13/F14 corpus on aa58b66 (`build/f15-corpus-t03`, same cases, references
  and ROI as the closure's final4 run, 12:02–14:40): 89 cases, every role
  gates (control 30/30, cross-check 2/2, full-rate-pair 17/17, functional
  13/13, independent-clip 12/12, numerical 14/14, repeatability-pair 1/1).
  Independent-clip linear RMSE p95 unchanged to 4 decimals except
  metalfx-fallback 0.0309 → 0.0303.
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

- 1 base grids + furnace + control: MET (aa58b66; direct grid all metallic p99 <= 0.09 %, furnace-resolve 1.00000, furnace-base 1.00001; post-review worst mean 0.009 %, p99 0.037 %, three controls fail). Ticked.
- 2 references retaken, diff in closure log: PARTIAL (reference-f15-appB 8/8 0 px, PSNR 29.9-55.1 dB recorded here; closure log is ticket 24). Open.
- 3 reference model updated, corpus green, F12 oracle: MET with justification (f13_reference.py updated, inert for F13 fixtures so no clip re-render; corpus 89/89 at 1ffc724; F12 signal pure Lambert by design). Ticked.
- 4 D2 band: NOT MET (2nd run "not a valid band measurement", +0.20 ms; 0 GPU allocations OK). Open, ticket 24.
- 5 archive/hitch: NOT EVIDENCED in this ticket; later runs of 05-08 pass. Open.

2026-11-15 — multiple-scattering compensation, ON by default (owner decision; docs/adr/0007-specular-multiple-scattering-compensation.md, ticket 19 furnace):

- Model: Kulla-Conty 2017 f_ms + Fdez-Aguera 2019 environment form, tables `src/renderer/ms_table.h` generated by `tools/ms_table.py` (deterministic half-vector
  quadrature of D Vis cos, ln L = ln(1 - E) on the 64 x 64 DFG grid and ln(1 - E_avg); the DFG table's Monte Carlo E is 1-10 % off in L: measured, the first
  version with it failed the 1 % direct-light gates at roughness <= 0.35, p99 3.3 %).  `--specular-ms on|off`, default ON, `rendering.specular_ms` in the report.
- Hooks (additive): shaders/materials/bsdf_energy.h (one include line at the end of bsdf.h), shadeSurface (function constant SPECULAR_MS = reserved 25,
  lights/sun/environment/coat), restir_common.h (DI target+shading, `diBRDFWeightedMs` next to the untouched `diBRDFWeighted`), composite (`reflection_composite_ms`
  entry point, lobe kernel).  Option off is BIT-IDENTICAL to 3ef8f41: f15_exact.sh x5 `cmp` identical to the pre-change binary; visual_check 8/8 0 px (see the
  bench 6 note below).  Three attempts with a runtime flag inside the base DI / composite kernels flipped 1 LSB at 1-9 pixels of benches 1, 2 or 6 and of 3 of
  the 5 exactness cases (the compiler re-associates the products when the kernel changes): the base functions/kernels are now textually unchanged and the callers
  select a separate function / entry point.
- Corrections found by the full f15_check re-run: (1) a dielectric of Fresnel ~1 (ior 0, saturated specular colour) reflected the single-scattering loss twice
  (1 - E_d diffuse complement + the lobe): furnace-specular mean 1.036; the lobe's F_ms L now comes out of the diffuse complement (declared in the references);
  (2) anisotropic lobes carry no lobe (the effective isotropic roughness gained +3.3 % on a pure metal at strength 1: the lost energy follows the view azimuth).
- Fixtures (thresholds fixed before any comparison; negative controls must fail): `ms-furnace@resolve` (pure-metal mean within 0.5 % of 1: 0.00016; no pixel above
  1.005; control ms-ignored deviates by 0.60), `ms-grid-env-resolve` (declared: worst mean 8e-5, p99 3.7e-4), `ms-grid-sun` (p99 <= 0.42 %, max 1.14 %), `ms-grid-env`
  (composite, 2.56 % of 5 %), `ms-specular-point` (+ @restir, @clustered: p99 <= 0.21 %); unit tests tools/test_ms_table.py, tools/test_f15_ms.py (furnace of a rough metal
  within 0.5 %, off FAILS at roughness 1 with 59 % loss, no gain, doubled F_ms gains, a Fresnel-1 dielectric does not double count), tests/test_specular_ms.cpp.
- References: tools/f15_reference.py defaults to the lobe and models it under every layer the engine applies it to (base, ior/specular, sheen scaling, coat attenuation
  + the coat's own lobe, iridescence film-free f0, thin subsurface, declared diffuse complement); tools/f15_transmission_reference.py (declared glass pixel);
  the Mitsuba shadow/subsurface references take their floor/OFF radiance from f15_reference; tools/f13_reference.py (+ _clip) folds the lobe into the cosine-sampled
  strategy (closed-form furnace 1.0001 / 0.9994 / 1.0001 / 0.9999; a 64 spp furnace render: rough white metal 0.3546 -> 0.8867, converged only by the corpus re-render).
- Full f15_check re-run with the lobe on by default and the compensated references (build/f15-ms/full4, plan order, captures at --pipeline-sync, 0 fallback draws):
  keys 1-100 (base, specular, clearcoat, sheen, anisotropy, iridescence, thin subsurface, every combination, furnaces, transmission) all gates PASS, every
  negative control FAILS, existing thresholds unchanged.  Keys 101-148 (transmissive shadows, volumetric subsurface, chess/layers; 8 cc0 keys need the
  terrain sets) were captured but NOT evaluated: the battery rule (< 40 %) stopped the Mitsuba renders; resume with
  `tools/f15_check.py --evaluate --expect green --out build/f15-ms/full4 --only <key>...` (references of the subsurface pieces are cached in build/f15-reference-cache).
- Bench 6 / 1 / 2 note: visual_check with shader validation shows 1 LSB at 1-9 pixels in ONE bench on the FIRST run after a rebuild and 0 px on the next runs of the
  same binary (round 1: benches 1, 2; round 2: 0 px at 8/8; bench 6 three times 0 px against the orchestrator binary's own images): an environmental effect
  of the cold Metal shader cache, not of the option; f15_exact.sh (no shader validation) x5 is byte-identical to the pre-change binary with `--specular-ms off`.

2026-11-16 — status reconciliation (closure): the archive_check / hitch_check box is ticked on the runs that ticket 24 and ticket 21 record on the cumulative tip (720ee63, 038eb10; the final-tip rerun is pending the final battery, ticket 24); the Status no longer lists the closure-log diff as open (ticket 24 holds the per-bench table); the D2 'recorded in ticket 24 and the handoff' wording now reads pending the final battery. Keys 101-148 evaluation stays pending (not re-derived here).

## 2026-11-17 round 4: multiple-scattering mismatches of the final battery (owner option B)

Two keys failed with MS on (ADR-0007) and passed with MS off in engine AND reference; the wrong side of each was found and fixed.

- **Engine side (`sheen-probe-env`, mean 7.0 % on the worst sphere).**  Under `--reflections probes` the composite skips the declared-environment sheen
  (`REFLECT_COMPOSE_SHEEN_FROM_PROBE`) and the lobe probe kernel adds it with the probe radiance at N; the MS term `F_ms L` had no such guard: the composite lit it
  with the declared environment (0.5) instead of the probe radiance (0.2), and the probe pass added none.  Fix (`e6078212`): the `_lobe` probe kernel adds the MS
  term (same lobe, sheen scale, coat attenuation and coat lobe as the composite) lit by the probe radiance at N, roughest level, exactly like the sheen; the
  composite skips its environment-lit copy when `SHEEN_FROM_PROBE` and `SPECULAR_MS` are both set (lobe variant only).  Bit 15 of `GPUReflectionParams.flags`
  (`kProbeSpecularMs`, set by `reflection_passes.cpp` when the lobe guide is valid, `--specular-ms` is on and reflections are `probes`) tells the probe kernel;
  it is defined next to its two users, not in `gpu_types.h` (no ABI change; move it there at the next gpu_types edit).  Result: `sheen-probe-env` PASS,
  worst-sphere mean 7.02e-02 -> 1.78e-04 (p99 8.25e-02 -> 6.43e-04), threshold unchanged.
- **Non-lobe `_ms` composite under probes (open, not touched).**  The base `reflection_composite_ms` kernel still lights `msTerm` with the declared
  environment under probe reflections and the base probe kernel has no MS variant: no key exercises it (every probe key with a gradient from the
  environment has probe == environment radiance), kept byte-identical.
- **Reference side (`texture-coat-sun`, p99 1.55 % against 1 %).**  `tools/f15_reference.py` evaluated the coat's own MS lobe on the shading normal N while the
  base path (`coat_lobe`, `coat_attenuation`) and the engine (`msCoat`: |V.Nc|, Nc.L) use the coat normal Nc (glTF KHR_materials_clearcoat evaluates the coat
  BRDF with clearcoatNormal).  Fixed in `bsdf` (coat MS value), `sun_radiance` (centre attenuation F_c(|V.Nc|), coat lobe at |V.Nc|, Nc.L), the declared and the
  exact environment coat term (`coat["mu"]`).  Identical without a coat normal texture (Nc = N).  Unit test
  `tools/test_f15_ms.py Layers.test_the_coat_lobe_lives_on_the_coat_normal` (N != Nc analytic case) fails on the old reference and passes on the new.
- Selected for before/after: every key with `reflections: probes` or a probe capture, every key with a coat, `ms-*`, `furnace-*`, `layers-furnace`
  (enumerated from `SCENARIOS` by name/field).  Results in ticket 24's round-4 section if the orchestrator records them; per key BEFORE/AFTER are in the
  writer report (all unchanged to float noise except the two keys above and `texture-coat-env-resolve`: sphere-weighted mean 7.36e-04 -> 4.76e-04, the same Nc fix).
