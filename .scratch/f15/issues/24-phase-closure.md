# 24: F15 phase closure: residue audit, regressions, measurements and handoff

Roadmap: F15.1–F15.4

**What to build:** Severe residue audit (roadmap lines, plan, exit criteria, TODOs), full regression battery, Sponza budget re-measured, Apple9 forced path, documentation and per-ID handoff; roadmap boxes ticked only for verified scope.

**Blocked by:** 15, 18 (if triggered), 20, 21, 22, 23

**Status:** in progress, closure (2026-11-16): the old carry list below is history; its current classification is the section '2026-11-16 — carry classification at closure' at the end of this ticket. The final battery (final-tip chess numbers, the final D2 rotation, the 24-of-24 corpus run, the final `f15_check` run, `metal_syntax_check` on the real target) is pending and not claimed here.

- [ ] `ctest`, `metal_syntax_check`, `tools/visual_check.sh` (0 px vs the ticket-03 baseline), F13/F14 corpus, `variant_check`, `archive_check`, `hitch_check` green; API validation clean (shader validation where the F14.4 defect does not apply) — 2026-11-15: all green on 29814ac and, after the reflection fix, ctest / F13 exactness / visual_check / archive_check / f15_check on 720ee63 (corpus and hitch_check re-run on 720ee63 recorded below); STAYS UNTICKED until the real `metal_syntax_check` runs (it exists only on non-Apple hosts: the macOS emulation passes 48 / 49 files and the one failure is an emulation artefact in metalfx_lifetime.cpp, unchanged by F15 and identical to origin/main, whose Linux job is green; CI runs the real target on the owner's push)
- [x] Sponza D2 GPU p95 A/B/A against the ticket-03 baseline within the decision-11 band; 0 GPU allocations (2026-11-15: median -0.038 ms against ticket 03 + 26fe229 under the owner's option A, +0.020 ms against the plain ticket-03 baseline; 0 allocations)
- [x] `--force-family apple9` run of the material bench with features (2026-11-15: five feature scenarios, exit 0, 0 validation lines, report effective apple9)
- [x] ROADMAP F15.1–F15.4 ticked only where verified; approximations and deferred items listed (caustics, dispersion, multiple-scattering compensation, back-lit thick subsurface if the hybrid is not triggered or not met, non-uniform texture transforms, coated emission of sampled emitters, GI/probe rays hitting glass as opaque) (2026-11-15: no box ticked, every ID keeps documented partial items; status lines per ID in ROADMAP, the list in docs/F15_HANDOFF.md)
- [x] perf-log, opt-log, RENDERER.md and per-ID handoff written; no push or PR without the owner (2026-11-15: docs/F15_HANDOFF.md, RENDERER.md status, perf-log and opt-log entries; nothing pushed)

## Comments

2026-11-14 — orchestrator: consolidated carry list. Keep it current until closure. Each item names its source ticket and record.

Partial criteria (stay unticked):
- 13: the temporal clip with transmission fails F13.6 on the gated smooth glass (p95 0.14-0.17). The cause is refracted-image
  reprojection under the orbit. A per-branch history was tried as a spike and rejected. The refract-axis control is not
  demonstrable, and the rough ROIs are not judgeable (control cost).
- 17 and 18: volumetric subsurface misses the side-lit bound on valid OpenPBR ground truth (gated mean 0.25-0.36) because of
  confinement in near-lossless media; ticket 18 ended in its fallback.
  - The tabulated profile (0.14-0.27) and the V1 dmfp mapping are measured candidates, not adopted.
  - The back-lit view is near black.
  - Later EDGE candidate: an RT volumetric random walk.
- 20: the ticket-17 bound on the chess pieces cannot be built (the reference is texture-free and homogeneous). The scattering
  weight map is not in the pinned subset (weight 1).
- 21: class pipelines are not bit-identical to the all-lobes pipeline (up to 37 ULP, compiler scheduling), so binning stays
  opt-in; the A/B/A binning-vs-union measurement on the chess fixture still needs a quiet machine.
- 22: Khronos side by side is qualitative by design (no IBL, different tonemap). No numeric texture fixtures for coat normal,
  anisotropy direction, iridescence thickness, or subsurface weight/colour textures (the reference refuses textures).

Declared approximations and limits (support table entries):
- 13: one Fresnel-chosen sample per pixel (ADR-0004); at most 4 crossings / 8 events.
- 14: GI shadow rays and reflection secondary lighting through glass have no numeric fixture.
- 15: single-layer map; glass is opaque with `--shadow-cache on`; the froxel fog ignores glass; the receiver bias applies only
  while the map is active; CSM at 2048^2 cannot resolve 2 cm casters (the chess shadow rig uses 8192).
- 19: GI at -N is a DDGI probe lookup only (cache, ReSTIR GI and the legacy resolve path keep E(N)). The iridescence diffuse
  weight is evaluated at N.V (ADR-0006 energy form).
- 22:
  - D4 fix: refracted first opaque hits get the environment term; the F13 reflection estimators and the RT probe capture now
    get it too (bf0799f, F13 residue below);
  - sun shadow rays (RT) and the CSM depth pass both treat the back of a single-sided plane as a caster: declared behaviour,
    consistent across the two shadow paths ("Shadow depth remains conservatively two-sided", shadow_passes.cpp), not a defect;
  - D5 is the single-bounce reflection of neighbours; D7 is unbiased 1-spp noise;
  - refusals: KHR_materials_unlit and KHR_lights_punctual are unsupported by design; non-uniform KHR_texture_transform is
    deferred.

Added by the closure audit (2026-11-15; three read-only audits of tickets 01-23, bookkeeping 65c8a52; each item names its source ticket):
- Orphaned or unowned: 16 image equality of the glTF encodings (KHR_materials_diffuse_transmission, draft scatter) vs the description
  file (carried to 20, which never held it; not built); 06 no coplanar-overlap fixture for the raster_order_group need of the lobe-guide
  producer; 06 the visibility resolve does not write the lobe guide (the pre-resolve kernel does; deviation recorded in commits only);
  10 captured-probe sheen fixture and sheen texture numeric fixture, reference quadrature inaccurate for N.V < 0.005 at roughness >= 0.3;
  09 the coat-only feature union (mask 2) is not archived (no fixture uses it).
- Quiet-machine costs: measured 2026-11-15 (af56a4b, perf-log) except the iridescence per light (no film-free twin of the grid and no
  light-count option: only absolute times and the static 187 flops per `bsdfThinFilm`). Transmission on the chess back-light rig costs
  +10.67 ms frame p50 (`Local clustered RT shading` +9.18): recorded as a cost, an optimisation candidate, not a gate.
- Declared limits and fit gaps: 11 anisotropic declared-environment fit 6.4-7.0 % on metal at strength 1 and the composite complement
  fit not proven as a bound, narrow-lobe sun and non-constant environment for the bent normal not exercised, a primitive without tangents
  shades as strength 0; 13 thin-walled delta-light BTDF unvalidated, rough thin-walled glass refused, the reference tints at every
  transmitting event while the engine tints at the entry only; 14 AO, fog and reconnection/sky rays see glass as opaque, the interior of a
  closed volume at the light end clamped T <= 1, transmissive + alpha-masked caster = opaque-masked, dedup list of 8 surfaces; 15 a glass
  receiver's own underside is multiplied by its slab's transmittance, half-resolution edge quality not gated, mesh/indexed transmit
  pipelines requested without glass; 16 thin + sheen/anisotropy/iridescence/ior not composed (the reference refuses), back-lit thin
  receiver uses the standard sun visibility, 8 background compiler calls in restir/clustered captures not explained; 17 composite diffuse
  limits (DI/ReSTIR/clustered and GI composite diffuse, reflection ambient diffuse and DI local lights not diffused; combinations keep the
  base diffuse; weight/colour/radius-scale textures do not change the profile shape); 18 RT single scattering analysed, not implemented,
  LUT valid for roughness 0.35 only; 19 thin-lobe weight/colour textures without reference, dropped non-discriminating controls
  (aniso-rotation-flipped, additive-layering on combo-irid-aniso-env, transmission-uncoated on combo-thin-coat-env); 20 API notices
  "Redundant call to setDepthStencilState/setViewport" on chess and subsurface fixtures (the same on 1d64ba5), the chess back-light rig
  under shader validation times out; 22 A1 single-bounce reflections make mirror-in-mirror dark (named approximation), sheen textures
  without numeric fixture; 23 layers blend base parameters only (feature blocks, emissive and alpha are not blended: deviation from the
  plan wording), secondary hits/GI/probes see the stack's own record, the mask uses the primitive uv (no triplanar), the repeat sampler
  blends the opposite texel at the mask edge; 07/08 the denoised DI/near-unit specular energy bias (bounded and declared, closure doc) and
  thin clip margins (0.0551 vs 0.06, 0.0198 vs 0.02).
- Thresholds set after measurement / understated figures: 21 the 1e-5 envelope was fixed after the first corpus and is exceeded on 15 of
  24 textured real-asset scenes (worst 1.0e-2 relative, 99 533 ULP on ToyCar, 8-bit within 1 LSB), and 21 of 73 featured scenes cannot
  discriminate through the resolve colour: "up to 37 ULP" above is the fixture maximum only; 23 the coat-zero gate 5e-4 was floored
  after the first measurement and the 1e-5 envelope is exceeded on two textured terrain cases (6.8e-5, 655 ULP, 0 px in 8-bit);
  10 the references of benches 5/6 were retaken in 74ff194 (1 LSB shader-validation change), so "0 px vs the ticket-03 baseline" for
  those benches is against retaken references; 03 archive_check compares with a same-mode no-archive baseline (12 px at 1 LSB on bench 1
  under shader validation since ticket 03).
- Closure battery items with no record yet: metal_syntax_check (non-Apple target; emulate with the stubs on macOS), variant_check, the
  full F13/F14 corpus after ticket 08 and after 8fbda9c, shader validation of DI scenes (first exercised in 08b, 039e22b), the forced-
  overflow subsurface capture (`tools/f15_subsurface_overflow_check.sh`, promised by ticket 17), the final report schema number (04).
- Plan deliverables not yet written: the deferred-items list in the handoff, docs/F15_HANDOFF.md, RENDERER.md beyond the table row,
  ROADMAP status lines (F15.2/F15.4 stale, F15.1/F15.3 missing), opt-log terrain headings.

Found in F15, outside F15 code (owner 2026-11-14: fixed inside F15; record in docs/implementation/F13-F14-CLOSURE-2026-11-12.md,
section "Defects found during F15"; integrated by fast-forward to 6b6bf04; orchestrator re-verification on the integrated tree
1903a09: ctest 30/30 with KHRONOS_CORPUS_DIR, F13 exact captures identical to the fixing worktree's `final` (0 px, 5 cases; sponza-d2
328 px / max delta 70 against base03 as explained in the closure document, the other four 0 px), new baseline
`build/f15-exact/base-residues`; `visual_check` against reference-f15-appB 0 px on 8 benches):
- GPU safety, clustered DI hang with emissive meshes: fixed 1bf0e19 (F11.5 per-pixel bound), emissive variant archived 3015a46.
- C22-D3 silhouette ring (shading N.V <= 0, no specular estimate, composite removed 1 - E anyway): fixed e9569ac, fixture
  `f15_residue_check.py --case grazing` (11/11 ring pixels below 70 % on the control, 0/0 fixed).
- Shadowed secondary hit black in reflection estimators / probe capture with GI off: fixed bf0799f (`--case mirror-env`, 1.001 vs 0.004).
- DDGI wall behind glass black (probe in the gap went inactive on the pane's back faces): fixed 4422a13 (`f15_shadowed_hit_check.py
  --mode ddgi`, 1.248 vs 0.377, gate 0.90..2.0).
- Transport rays ending black on single-sided back faces: fixed 26fe229 (`--case backface`, 1.000 vs 0.00001; error words 16/17).
- FIXED, owner 2026-11-14 "Correggere in F15": with any GI mode the declared environment reaches no primary surface (the
  INDIRECT atlas excludes it by design, no other pass adds it): shadowed wall 16/255 with `--gi ddgi` vs 194 with GI off. Required:
  one ownership rule (no double count against the GI-off term and the reflection/probe paths), reference albedo x L_env with a
  negative control, F13/F14 corpus rerun with every image change explained (thresholds only tightened, re-cut references justified),
  Sponza D2 re-measured on a quiet machine in an interleaved A/B with the previous binary, closure-document row.
  RESULT: fixed 4fc1c8a (agent 8a6335d, F12.1), closure row 274d503 plus the orchestrator re-verification. Ownership rule in
  shaders/gi_common.h (giEscapeRadiance): GI off = the analytic unoccluded term; GI on = escaped probe/candidate rays return the declared
  radiance (occluded by geometry), secondary hits get it through the atlas, the explicit secondary sky ray is removed, the GI-off term is
  stripped. Fixture tools/f15_gi_environment_check.py (CPU cosine-weighted sky-visibility reference), orchestrator rerun against the
  c978151 control: open ddgi 1.018 / cache 1.019 / restir 1.003 against 0.039 / 0.016 / 0.016, slot 1.049 / 1.040 / 1.044 (an
  unoccluded term reads 3.94). ctest 33/33; exact sponza-d2 284 775 px (max 12, 50.53 dB: the atmosphere sky now reaches surfaces),
  the other four 0 px; visual_check 0 px; zero-environment scenes bit-identical in every GI mode (Cornell room, moving-sun restir);
  f10_f12 f12_* functional battery exit 0 (negatives exit 1). F13/F14 corpus (agent): 37 verdicts equal final4, no reference re-cut,
  no threshold moved, control vs fixed bit-identical on five representative cases (none of the 37 declares an environment). Pipeline
  descriptors unchanged. Sponza D2 interleaved A/B (2026-11-15, `build/d2-ab-envfix`, control c978151 vs 94da95c): GPU p95 deltas -0.10 /
  +0.14 / -0.04 ms (median -0.04 within the 0.10 ms band), p50 -0.05 / +0.01 / -0.05, 0 GPU allocations, sky sentinel 0.689-0.695 on both.
  Process note: the agent's first configure fetched the CMake dependencies over the network for a few seconds against the no-download
  rule; it deleted that directory and rebuilt offline from the local sources.
- IN PROGRESS, owner 2026-11-14 "Dimostrare o correggere": the F11.5 DI cap (1bf0e19) ratio budget/exact is 1.016 +- 0.008 with
  the denoiser off and 0.973 (sem 0.001) with `--lighting-denoise custom`. Measure the cause (clamp disabled/widened, same scene
  below the 128-light cap); fix a systematic loss with a failing-before fixture, or prove the expected behaviour against a
  reference. No "probably" may remain in the closure documents.
  RESULT (7e3f730): cause proven = the a-trous luminance edge-stopping weight (row-normalised) of the F13.4 denoiser loses energy
  on heavy-tailed noise; the cap's estimator is unbiased (raw 128 / raw 8192 strata 0.9996 +- 0.006); denoised/raw 0.977 +- 0.004
  (128), 0.988 +- 0.002 (8192), 1.0000 for exact lists; luminance weight off 1.004 +- 0.007; CPU reference in tests/test_denoise.cpp.
  Owner 2026-11-14: ONE timeboxed experiment with an energy-preserving a-trous behind an option (default unchanged), adopted only if
  the bias drops within noise on DI-cap + GI + AO + specular signals (8-seed protocol, CPU reference), no corpus/flicker/ghost gate
  gets worse (full F13/F14 corpus, option on vs off) and the Sponza D2 GPU p95 interleaved A/B (orchestrator, quiet machine) is no
  worse; otherwise fall back to B with the numbers. Spike in docs/opt-log.md, numbers in the closure document.
  RESULT (2026-11-15): option B. The experiment (ceec9db, d6e12c2, 2b7f823, 94da95c; flag default off) removes the bias within noise (DI cap
  0.9933 -> 0.9990, Queen 0.977 -> 1.006, specular 0.9941 -> 1.0000, GI spatial 0.9992 -> 1.0000) with identical corpus verdicts, but the Sponza
  D2 A/B (`build/d2-ab-energy`) costs +13.27 ms GPU p95 (median of 3 pairs; p50 15.3 -> 28.4 ms), all in the specular energy passes (14.73 ms vs
  1.64 ms). Shipped filter stays; residue declared with its bound in the closure row; A stays an open candidate (opt-log). Integration check:
  ctest 33/33, F13 exactness 0 px and PFM bit-identical against 4fc1c8a, visual_check 0 px. CLOSED.
  GI with temporal accumulation 0.926 +- 0.016 filtered/raw for BOTH filters: ISOLATED (opt-log): transient history lag behind the converging DDGI
  (raw rises 0.868 -> 1.000 of its final level; filtered/raw 0.926 over frames 16-64, 0.9979 +- 0.0019 over frames 192-256, spatial-only 0.9996).
  Still to explain: the negative-f14-lut exit -11 in its first corpus run (crash report 2026-11-14 17:07: SIGSEGV in
  the AGX compiler, `UserIntersectionProgram::VariantSetEntry::signalPending`, on a phosphor-compile worker inside
  `PipelineCache::compileFinal` of a compute pipeline with a linked intersection function, under the shader-validation compiler; a 2026-11-13
  report crashed in `RayPayloadSizeAlignImpl::merge` at `setComputePipelineState`). OPEN.
  AGX crash investigation (2026-11-15, base 1d64ba5, `bench/agx_link_stress` + `tools/agx_link_stress.py`): NOT REPRODUCED, cause unexplained.
  Both reports are the same family (driver state of pipelines with a statically linked intersection function; the 11-13 one is the render
  thread binding a pipeline at ~0.3 s after launch while 17 workers were still inside `newSpecializedFunction`; the 11-14 one is a worker
  messaging a bad `_MTLFunction*` inside `createAirLinkedIntersectionVariantSets`, under `MTL4GPUDebugCompiler`). The driver keeps the
  compiled variants by content: a fresh `MTL::Library` is served from memory (1.6 ms vs 3.4 s), so cold compiles need a salted metallib or
  salted MSL source. Attempts, all with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` unless noted, 0 crashes in every one: real case reruns
  4 warm (+12 by the corpus agent); real case on an app copy with a salted `rt_alpha_generic` (cold) 28; the 15 real RT kernels, cold, 18
  threads, plus a render-thread role (function handle + IFT) 45; synthetic salted linked kernels (+ unlinked ones) at T=1/18 (6 each, also
  without validation), T=18 with consumer 8+8+8, T=36 and T=64 with consumer and encoder bind 40 each. API review of PipelineCache found
  no misuse (library retained per job, descriptors owned per job, one thread-safe compiler, pool per job); no mitigation was added because
  nothing failed before it. Startup note: with the archive, D2 startup is 14-20 ms ready with 12 compiler calls (570-630 ms summed CPU,
  max 76 ms each), so the 573-606 ms figure is summed compile time, not wall time. OPEN.
- OPEN, found 2026-11-15 (blocks the D2 criterion): Sponza D2 A/B/A against the pre-F15 baseline 75924b2 (`build/d2-ab-f15-closure`, quiet
  machine, 3 pairs): GPU p50 +0.73 / +0.67 / +0.64 ms, p95 15.57 / 16.40, 15.46 / 16.53, 15.31 / 16.39 ms (median delta +1.07 ms against the
  0.10 ms band; absolute 16.39-16.53 <= 16.67). Per-pass p50 medians: `F13 world-radius RTAO` +0.459 (1.046 -> 1.504, the kernel is unchanged),
  `Solar RT visibility` +0.066, `Solar temporal moments` +0.065, `F13 reflection sample 0` +0.064, froxel RT source +0.012, Material resolve
  +0.009. Sponza uses no F15 feature, so the cost is paid by scenes that do not need it. Leading hypothesis to test with a measured spike: the
  shared RtPayload grew from 16 to ~124 bytes in c59c5af (ticket 14 transmittance list, counters, direction), so every RT query pays it; the RGB
  sun mask (same commit) for the temporal moments pass.
  2026-11-15 record (spike in `docs/opt-log.md`, entry "F15 ticket 24: Sponza D2 regression ... attribution and fix"; commit ec63efc; the formal
  quiet-machine A/B/A band is the orchestrator's, NOT claimed here; the sky sentinel read 0.81-0.82 in every run of this session, 75924b2 included):
  attributed and fixed by measurement with negative controls: RTAO +0.45 = the 124-byte RtPayload alone (lite kernel 1.047-1.058 vs 75924b2 1.046-1.058; the
  same lite kernel with the payload padded back to 124 bytes 1.494), Solar RT visibility +0.06 = the same payload (0.429-0.433 vs 0.432-0.435), froxel RT source
  +0.011 = the same payload (0.118), Solar temporal moments +0.065 = the 80-byte RGB history record (scalar 64-byte kernel 0.390 vs 0.389-0.390; shipped 0.454-0.457).
  Scenes without MATERIAL_FEATURE_TRANSMISSION use `ao_rtao` / `fog_inject_rt` (always) and `shadow_sun_rt_lite` / `shadow_temporal_scalar`; glass scenes keep the
  ticket-14/15 kernels (transmissive-shadow keys: 8/8 GREEN, every negative control fails; F13 exactness 5 cases 0 px and PFM bit-identical; visual_check 0 px;
  archive +4 compute descriptors, archive_check pass, 0 misses and 0 GPU allocations on D2; ctest 31 + 1 skipped, 926 doctest cases; API + shader validation
  clean on pane-sun@rt, pane-sun@csm and D2 with only the notices 1d64ba5 prints too).  Frame p50 after the fix +0.04..+0.11 ms against 75924b2 (before: +0.63..+0.75).
  STILL OPEN and NOT attributed: `F13 reflection sample 0` (p95 +0.14..+0.17 ms, p50 +0.02..+0.06 within a +-0.1 run-to-run spread on this machine; a lite-payload
  variant of `reflection_rt` and a `kReflectionBackfaceSkips`=0 variant did not separate from the noise) and three small passes whose p95 (not p50) is +0.04 (positive RAW
  pre-reflection lighting, DDGI indirect irradiance, denoise 3 atrous 1).  Whether the median p95 delta now meets 0.10 ms needs the orchestrator's quiet-machine A/B/A.
  Carried: the archive keeps the earlier `ao_rtao` / `fog_inject_rt` entries linking `rt_alpha_generic` (unrequested); lobe kernels (scenes with any feature, no glass) still trace
  with the full payload.
  Orchestrator integration (2026-11-15, 8fbda9c + d124f2b on the branch): ctest 33/33, F13 exactness 5 cases 0 px and PFM bit-identical against 94da95c,
  visual_check 0 px on 8 benches, archive_check all scenarios pass. A/B/A after the fix, sky sentinel 0.81 on every binary of the day:
  - against 75924b2 (`build/d2-ab-f15-closure-fix`): GPU p95 15.49 / 15.70, 15.51 / 15.60, 15.52 / 15.67 (median delta +0.15), p50 median +0.07;
  - against the ticket-03 baseline 2e1057a, which is the cost baseline of the plan ("after ticket 03 the scenes without features stay identical to the new
    baseline in images and cost", docs/plans/F15.md; built in worktree f15-t03-baseline, `build/d2-ab-t03-fix`): GPU p95 15.40 / 15.66, 15.35 / 15.70,
    15.60 / 15.68 (A/B/A deltas +0.29 / +0.23 / +0.08, median +0.23 against 0.10), p50 median +0.03; per pass the residual is only `F13 reflection
    sample 0` (p50 +0.04, p95 +0.42 / +0.51 / +0.18 on a 3.53 ms base), every other pass within +-0.07 p95 with mixed signs.
  - attribution (D2 agent, opt-log entries of 2026-11-15): the reflection tail is the F13 back-face continuation of 26fe229 (about 3 900 continued
    transport rays per frame on D2, about 200 exhaust the bound of 4, i.e. five traversals in one lane); with the bound at 0 the pass p95 returns to
    the 2e1057a level; restructuring the loop changes nothing; it cannot be specialised (single-sidedness is not a feature bit and Sponza has
    single-sided back faces: bound 0 changes sponza-d2 by 934 px, max delta 122). A one-traversal hardware back-face culling variant
    (`RT_PROBE_DIFFUSE_CULLED`) was discarded: per-ray comparison finds 24-35 rays/frame with a different first surface outside the bound-exhausted
    ones and ~2 800 with different t/u/v (the loop re-traces from an offset origin), and it recovered only 0.07 of the 0.22 ms.
  - orchestrator A/B/A, 2e1057a vs the current code with the continuation bound set to 0 (a never-committed probe build, `build/d2-ab-t03-skips0`):
    GPU p95 15.46 / 15.60, 15.44 / 15.49, 15.44 / 15.47 (A/B/A deltas +0.15 / +0.05 / +0.04, median +0.05 within the band), p50 median +0.07; so the
    F15 features alone meet the band and the excess is the continuation's cost. OWNER DECISION REQUESTED (2026-11-15): (A) declare it as the intrinsic
    cost of the 26fe229 correctness fix, or (B) reduce it by changing images under a new gate.
  - OWNER DECISION 2026-11-15: option A (relayed by the parent session): the continuation's cost is the F13 fix's, measured on its own with and
    without the continuation on the same executable; the F15 band stays 0.10 ms, measured against the ticket-03 baseline plus 26fe229; images unchanged.
  - Measured literally (orchestrator, quiet machine, `build/d2-skips/`, perf-log entry "F15 ticket 24: Sponza D2 band under the owner's option A"):
    backport of 26fe229 onto 2e1057a (`t03fe`, local branch cl/f15-d2-t03-plus-26fe229 / 91bba50, never merged; D2 exactness 49 px against 2e1057a
    as 26fe229 recorded; continuation counters equal to the closure tree's).  On the closure tree 29814ac the F15 band still failed: A/B/A
    t03fe -> 29814ac +0.20 / +0.22 / +0.05 (median +0.198), 6-replica rotation +0.104; the continuation cost +0.086 (A/B/A) / +0.139 (rotation) there
    but only +0.009 / +0.065 at the baseline: the reflection pass still traced with the 124-byte ticket-14 payload and the continuation multiplied it.
  - FIXED in F15 by 720ee63 (opt-log "follow-up 3", adopted): the base reflection_rt traces with RtPayloadLite and links rt_alpha_lite (templated
    reflectionTraceVisible / reflectionSecondary / rtTraceSurface / diEndpointTransmittance; lobe kernels and the probe capture keep RtPayload;
    archive +1 compute descriptor).  F13 exactness 5 cases 0 px and PFM bit-identical against 29814ac, visual_check 0 px, archive_check pass,
    ctest 33/33.  Decision-11 A/B/A: t03fe -> final -0.04 / +0.04 / -0.10 (median -0.038, WITHIN the band); 2e1057a -> final median +0.020;
    F13 continuation cost on the shipped code (s0lite -> s4lite, same executable) median +0.065 ms; absolute GPU p95 15.33-15.43 ms, 0 GPU
    allocations. CRITERION MET.
- Declared (owner noted, no objection): single-sided planes cast sun shadows from both sides in CSM (two-sided by contract) and RT.
- CLOSED, not a defect (eac5233, ticket 13 "C22-D3 follow-up"): the 67 IORTestGrid "dark ring" pixels are a metric artefact. With
  reflections off a white transmissive sphere reads albedo 1 x environment 0.5 = the clear colour, so the ring set of the off image
  holds only the 3x3 sun-glint halos (r/R 0.23 .. 0.47, none on a silhouette); 99.7 % of all glass-sphere pixels are below 70 % of
  that baseline because the glass shows the backdrop. Reference: white furnace `tools/f15_residue_check.py --case transmissive`
  (five smooth solid glass spheres, IOR 1.0 .. 2.42, environment 0.5 only). Orchestrator rerun on the c978151 Release binary
  (2026-11-14), identical to the agent's numbers: disc mean 1.0007, silhouette-ring mean 0.9984 (2 056 px), min 0.83, 0 of 83 540
  pixels below 70 %, fraction below 90 % 4e-5; absorbing-medium negative control (same binary) disc mean 0.2736, 79 209 dark pixels,
  ring mean 0.866 (grazing Fresnel reflection dominates there), gate fails as required. No engine change; the IORTestGrid count is
  withdrawn. Limits recorded in ticket 13: walks that end on lit geometry are covered by the existing transmission gates; glass
  pixels with shading N.V <= 0 keep the declared full diffuse (e9569ac).
- O7 caveat: a non-archived RT configuration with a cold compiler cache allocates one IFT per frame slot when its final variant
  arrives (ticket 13 record).

Quiet-machine measurements for this ticket:
- Sponza D2 A/B/A band;
- cost of transmission RT, transmissive shadows (RT and the CSM map), the subsurface diffusion pass, the iridescence per light
  (ticket 12) and the GI-back lookup;
- binning vs union on chess and terrain (ticket 23: `python3 -I tools/f15_layers_timing.py run --out build/f15-layers-timing --binning`;
  resolve GPU time per active-layer count; >= 5 % rule of decision 05). TERRAIN DONE 2026-11-15 (quiet machine, load 2.0, 90 runs, A/B/A x 3
  replicas, `build/f15-layers-timing/summary.md`): resolve mean ms union single 0.089, layers-1..4 0.136 / 0.162 / 0.183 / 0.208, full 0.204;
  binning (resolve + classification) single 0.106, layers-1..4 0.143 / 0.160 / 0.182 / 0.216, full 0.206. Binning never lowers the resolve by
  5 % (full +1 %, layers-4 +4 %, layers-2 -1 %): stays opt-in, union stays the default. Chess binning A/B/A: see the next entry;
- QUIET-MACHINE COSTS RECORDED 2026-11-15 (cfb9e11, `tools/f15_costs.py`, 129 runs, A/B/A x 3 replicas, 256 frames, 0 GPU allocations, 0 fallbacks; tables in `docs/perf-log.md`
  "F15 ticket 24: quiet-machine cost of the F15 features"; costs to record, not gates):
  - transmission RT (chess back-light 720p, tops transmission 0 vs 1): frame p50 +10.67 ms (+9.38..+10.85), `Local clustered RT shading` +9.18, each of the 8 reflection samples +0.17..0.18;
    no opaque-glass switch on the procedural fixtures (absolute 14.8 / 17.4 ms glass-behind-glass / roughness-ior, declared `--rt off` 4.2 ms);
  - transmissive shadows, chess shadow rig 1080p glass vs opaque tops: RT +0.148 ms frame (within A-A spread 0.167; `Solar temporal moments` +0.099 RGB history, `Solar RT visibility` +0.027);
    CSM 8192 +1.03 ms (`CSM transmittance` 4 x 0.195, PCSS +0.128) and +1 028 MiB of map memory; CSM 2048 pane fixtures: map receiver `CSM PCSS visibility` +0.075 ms, transmittance passes 4 x 0.021 ms;
  - subsurface diffusion (chess side-light 1080p, sss off vs on): frame p50 +0.411 ms, `Subsurface diffusion` 0.431 + clear 0.025 ms = 6.1 ns per subsurface pixel (71 152 px, whole-frame figure), +64 MiB unaliased heap;
  - iridescence per light: NOT measurable, no film-off twin and no light-count option (absolute pass times recorded; static 187 flops per `bsdfThinFilm` stays the only per-light figure);
  - GI-back lookup (`--thin-gi-back`): `DDGI back irradiance` pass +0.175 ms (720p), frame delta not significant (+0.34, A-A spread 0.38), +7.25 MiB;
  - binning vs union on chess (resolve + classification): side-light -1.8 %, shadow rig +6.7 %, back-light +9.8 % (A-A noise 2.6 / 0.9 / 4.3 %): the >= 5 % rule of decision 05 is not met (opt-log), with the
    terrain result union stays default, binning opt-in. CHESS DONE.
- Sponza D2 interleaved A/B for the GI-mode environment fix and for the energy-preserving a-trous experiment.

2026-11-15 13:10 — final-tree corpus: `tools/f13_f14_check.py --run` on 720ee63 (`build/f15-corpus-final`, frozen manifest and references
of the closure run) exit 0, 89 / 89 executed, gating control 30/30, cross-check 2/2, full-rate-pair 17/17, functional 13/13, independent-clip
12/12, numerical 14/14, repeatability-pair 1/1; `corpus_compare.py` against the 29814ac run: every case identical in state. hitch_check on
720ee63: 0 hitches, 21/21 archive hits, 0 render-thread compiles.
2026-11-15 ~12:20 — owner (via the parent session): before the F15 PR, finish every partial of this carry list and of the handoff "Open and
partial" section, plus earlier-phase fixes, in parallel; owner decisions of the same day: implement multiple-scattering compensation,
multi-bounce reflections, non-uniform texture transforms and back-lit thick subsurface; RT random walk as an optional subsurface tier; class
pipelines bit-identical within the D2 budget; optimise the transmission cost in F15. Work in progress on branches `f15c-*` (round 2).

### 2026-11-16 — closure log: visual_check references per bench (ticket 03 criterion 2)

`image_diff` of the committed-history references (8 benches, 2400x2400, tolerance 0; `BENCH8_ARGS="--instances 100000"`):
`build/reference-f15base` (2026-11-13 11:08, before ticket 03) -> `build/reference-f15-appB` (11:59, glTF Appendix B base model, ADR-0006) ->
`build/reference-f15-final` (2026-11-16 01:35, round 2: specular multiple-scattering compensation on by default, ADR-0007, and the other intended
round-2 image changes).  The final battery (`build/final2`, final tip) compares 0 px against `reference-f15-final` on all 8 benches.

| bench | base -> appB: pixels differ / max delta / PSNR | appB -> final: pixels differ / max delta / PSNR |
|---|---|---|
| 1 | 3 635 504 / 14 / 38.76 dB | 2 755 646 / 34 / 42.71 dB |
| 2 | 561 787 / 49 / 41.11 dB | 366 387 / 69 / 42.05 dB |
| 3 | 3 121 293 / 57 / 31.59 dB | 2 374 495 / 71 / 35.08 dB |
| 4 | 1 409 236 / 30 / 45.49 dB | 1 473 054 / 58 / 41.85 dB |
| 5 | 2 229 852 / 2 / 55.13 dB | 731 856 / 1 / 61.62 dB |
| 6 | 2 088 863 / 7 / 48.58 dB | 1 037 935 / 2 / 57.59 dB |
| 7 | 3 809 999 / 10 / 45.83 dB | 2 201 844 / 1 / 56.31 dB |
| 8 | 4 379 566 / 64 / 29.85 dB | 3 181 013 / 43 / 34.63 dB |

base -> appB is the expected change of ticket 03 (diffuse weighted by the dielectric Fresnel F(0.04), metals through the DFG table: intermediate
metallic and ambient / GI diffuse; ticket 03 recorded PSNR 29.9-55.1 dB, max delta 2-64, the same numbers).  appB -> final is not attributed
per bench to individual round-2 commits here (MS compensation is the change that touches every bench's specular energy).


### 2026-11-16 — carry classification at closure

Each item of the 2026-11-14 carry list and of the 2026-11-15 closure-audit additions is classified by the LATEST dated evidence in the tickets named. States: CLOSED (evidence),
OPEN by owner decision (date), DECLARED limit (kept in the support table or an ADR), PENDING the final battery (ticket 24). Numbers are quoted from the tickets and logs, not recomputed.
The old list above is left as written.

Partial criteria
- 13 temporal clip: CLOSED for the gated scope (b6c7ee6, 8d43c2e): smooth glass p95 0.0230 / 0.0236 / 0.0306 vs 0.06, refract-axis control 0.1746 / 0.1691 / 0.1445 fails as required (the old 'p95 0.14-0.17' is that control). Rough-glass ROIs with roughness <= 0.3: DECLARED out of the F13.6 gate by the owner (option b, 2026-11-15), measured as information (`transmission-rough-corrected`; r0.1 / r0.3 miss; named future fix: a sub-pixel gradient-aware rough-glass history). The final-tree clip run: PENDING the final battery.
- 17 and 18 volumetric subsurface: the optional RT tier `--subsurface-mode rt-walk` meets the bound (side-lit 0.0635 / 0.1708 vs 0.10 / 0.25, back-lit 0.0919 / 0.2288 vs 0.20 / 0.50, product output and moving-camera clip green): CLOSED for that tier (owner decision, 2026-11-15). The DEFAULT diffusion tier still misses the side-lit bound (gated mean 0.25-0.36) and its back-lit view is near black: DECLARED limit, fallback of ticket 18, default stays diffusion. The tabulated profile and V1 dmfp mapping stay measured candidates, not adopted. 'Later EDGE candidate: an RT volumetric random walk' is CLOSED as the optional tier. RT walk timings (4 / 16 / 64 walks 3.93 / 15.3 / 60.4 ms): recorded in perf-log 2026-11-16; final-tip rerun PENDING.
- 20 chess king/queen subsurface bound: OPEN by owner decision (2026-11-16, option A): `sssbound` evaluated RED on the RT walk (pooled per piece mean / p95 up to 1.431 / 3.072 against 0.10 / 0.25; the reference's own noise alone exceeds the bound), thresholds unchanged, follow-up D named and not started. Final-tip numbers: PENDING the final battery. The monotone gate is met. The 'scattering weight map not in the pinned subset' stays a DECLARED fixture limit.
- 21 class pipelines vs the all-lobes pipeline: CLOSED (038eb10, owner option (a) exact-maths module): 200 cases / 162 scenes 0 px, 0 ULP, repeated reference 7 of 7, terrain 19 cases 0 ULP (ticket 23, e0f6ea0). Binning vs union: CLOSED as a measurement (chess side-light -1.8 %, shadow rig +6.7 %, back-light +9.8 %; terrain full +1 %, layers-4 +4 %, layers-2 -1 %): the 5 % rule is not met, binning stays opt-in, union the default; the re-measure on the exact module: PENDING. The old ToyCar / ClearCoatCarPaint run-to-run variation: DECLARED localised, not root-caused (owner option a, 2026-11-16); the tip does not reproduce it (0 of 480 frame checks), the guard in `ReflectionPasses::check` stays. The final-tip equivalence run and the quiet cost of the exact module: PENDING.
- 22 Khronos side by side: DECLARED qualitative by design (no IBL, different tonemap). Numeric texture fixtures for coat, anisotropy, iridescence, sheen and thin weight / colour: CLOSED (`texture-{coat,aniso,irid,sheen,thin}-{sun,env-resolve}` GREEN, controls fail). The corpus page / report regeneration and the 24-of-24 corpus run on the final tip: PENDING the final battery.

Declared approximations and limits
- 13: one Fresnel-chosen sample per pixel (ADR-0004), at most 4 crossings / 8 events: DECLARED. Owner exclusions (a) AO, fog, ReSTIR reconnection and sky rays see glass as opaque and (b) secondary RT hits use the base reduction: DECLARED (dcfb3ea).
- 14 GI shadow rays and reflection secondary lighting through glass: CLOSED (round 3, e0f6ea0: gi-ddgi / gi-cache / gi-restir / refl at the expected ratio 0.3840 / 0.8640 / 0.4320, controls fail; the subsurface walk exit connection now carries transmittance).
- 15: single-layer map, receiver-plane bias only while the map is active, a solid receiver's light-facing face behind another layer still carries its own slab, the divisor 4 edge bound reported only, the froxel fog with CSM does not see glass (owner exclusion (a)): DECLARED. Glass with `--shadow-cache on`: CLOSED (round 2). CSM at 2048^2 cannot resolve 2 cm casters (the chess shadow rig uses 8192): DECLARED. The eight ticket-13 Mitsuba keys and the literal `--expect green` over every key: PENDING the final battery.
- 19: GI at -N is a probe-grid lookup in every GI mode and the legacy resolve: CLOSED (ticket 16, quad errors 4.7-6.7 % vs the 25 % indicative gate, `--thin-gi-back off` fails). Iridescence diffuse weight at N.V (ADR-0006 energy form) and the iridescence + specular 4.92 % bias: DECLARED. Radiometric accuracy of DDGI, Cache and ReSTIR GI themselves: DECLARED unverified. The cost of the extra GI-at-N dispatch: recorded (+0.175 ms DDGI back pass, 720p); final-tip rerun PENDING.
- 22: D4 fix CLOSED; sun shadow rays and the CSM depth pass treating the back of a single-sided plane as a caster: DECLARED (consistent across the two paths); D5 (single-bounce mirror reflection) CLOSED by multi-bounce reflections (`--reflection-bounces` default 3, owner option (b), 2026-11-15; the mirror / moving-light / disocclusion clips re-render: PENDING the final battery); D7 unbiased 1-spp noise: DECLARED. Refusals: KHR_materials_unlit and KHR_lights_punctual are now IMPLEMENTED (e8b6399, owner decision 2026-11-15; the 'unsupported by design' text above is superseded and docs/plans/F15.md line 22 is overridden as an approved scope addition); non-uniform KHR_texture_transform: CLOSED (per-slot implementation, tickets 04 / 22 / 23); a second UV set per material: DECLARED unsupported by decision.
- Multiple-scattering compensation (listed 'deferred' in box 4 of this ticket): CLOSED as implemented and ON by default (ADR-0007, ticket 03; `--specular-ms off` is bit-identical to 3ef8f41). Back-lit thick subsurface: CLOSED on the optional RT tier (ticket 17 / 18), DECLARED limit on the default tier. Non-uniform texture transforms: CLOSED (above). Coated emission of sampled emitters: not re-audited in this reconciliation (see the handoff list). GI / probe rays hitting glass as opaque: DECLARED owner exclusion (a) for the non-GI-shadow sites; GI shadow rays through glass: CLOSED (ticket 14 round 3). Caustics and dispersion: DECLARED out of scope.
- Spot falloff (ticket 04): OPEN by owner decision (2026-11-16, option a) with per-path bounds (RT reflection lighting PASS up to tilt 0.5, probe capture 16.3 % at tilt 0.55, DDGI not separable, ReSTIR GI 19.6 % in the steep tail); the post-hoc patch-size change is rejected (95e184e) and the final-tip run records tilt 0.55 as it measures: PENDING.

Added by the closure audit
- Orphaned or unowned: 16 image equality of the glTF encodings: CLOSED (`tools/f15_encoding_check.py`, 0 px, repeat 0, controls differ). 06 coplanar-overlap fixture: CLOSED (`lobe-guide-coplanar`: 20 / 20 with the raster order group, control fails 20 / 20 at skewed cost; the 'no coplanar-overlap fixture' wording above is superseded). 06 the visibility resolve does not write the guide: DECLARED deviation (the pre-resolve kernel covers every meshlet pixel). 06 criterion 3 (memory per category): stays PARTIAL, the report JSON has no memory per category, the allocation is logged per category since 8d54b5f. 10 captured-probe sheen fixture, sheen texture fixture and the N.V < 0.005 quadrature: CLOSED (`sheen-probe-env` GREEN, `texture-sheen-*` GREEN, quadrature bounded 3.8e-4 vs 1e-3); the captured-probe path is declared on the F13 composite path only. 09 coat-only union (mask 2): CLOSED (archived, 271d751, 0 archive misses).
- Quiet-machine costs: recorded in perf-log (2026-11-15 cfb9e11 and 2026-11-16 'F15 closure round 2'). Iridescence per light: CLOSED (+0.033 ms per extra light legacy, +0.036 ms clustered). Transmission on the chess back-light rig: recorded as a cost, not a gate (+10.67 ms frame p50 on 2026-11-15; perf-log 2026-11-16 records the glass transport B - A at +12.62 ms after the 128-byte payload fix vs +14.46 ms pre-fix, quiet A/B, adopted): final-tip cost PENDING the final battery.
- Declared limits and fit gaps: 11 anisotropic declared-environment fit: CLOSED (round 3 albedo table, 0.08 %); narrow-lobe sun `aniso-narrow-sun`: OPEN by owner decision (2026-11-16, option A: RED on the exact module, one pixel 3.77 % vs 3 %, follow-up E); bent-normal fixture `aniso-bent-env`: CLOSED (legacy hemisphere, GREEN); a primitive without tangents shades as strength 0 / gets (1, 0, 0, 1): DECLARED. 13 thin-walled delta-light BTDF, rough thin-walled glass and the tint rule: CLOSED (841fd49). 14 closed-volume interior at the light end and transmissive + alpha-masked caster: CLOSED (round 2); dedup list of 8 surfaces: DECLARED. 15 glass receiver underside and half-resolution edge: CLOSED (round 2); mesh / indexed transmit pipelines requested without glass: CLOSED (64 -> 62). 16 thin + sheen / anisotropy / iridescence / ior: CLOSED (composed and gated, `combo-thin-lobes-env` worst sphere mean 1.95 %); back-lit thin receiver: CLOSED (RT / DI `thin-backlit`, CSM 205d50b7); the 8 background compiler calls: CLOSED (explained). 17 composite diffuse limits: CLOSED for the RT tier (ambient, GI, DI local lights, texture-modulated medium), DECLARED for the default tier. 18 RT single scattering: CLOSED as the optional RT tier; LUT now generalised over roughness 0.15..0.7 and g 0..0.3, the roughness 0.15 knot has no fixture: DECLARED. 19 thin-lobe textures without reference and the three dropped controls: CLOSED (`texture-thin-*`, `combo-aniso-panel-*`). 20 API notices: CLOSED (65f2f10, 0 notices); chess back-light rig under shader validation: CLOSED as explained (the engine's 5 s frame wait) and validated at 640x360. 22 A1: CLOSED (above); sheen textures numeric fixture: CLOSED. 23 feature blocks, emissive and alpha not blended: CLOSED (blocks blended, coverage alpha blended; transmission, volume and volumetric subsurface not blendable by declaration: DECLARED); secondary hits see the stack: CLOSED (`layers-hit-*`); the mask uses the primitive uv (no triplanar): DECLARED, deferred to F19.2; the repeat sampler edge blend: CLOSED (clamp_to_edge). 07 / 08 denoised DI / near-unit specular energy bias: DECLARED residual (energy-preserving a-trous option B shipped, A stays an open candidate); thin clip margins: DECLARED.
- Thresholds set after measurement: 21 envelope: CLOSED (0 ULP, the envelope is moot). 23 coat-zero gate: CLOSED (exactly 0 px) and the textured-terrain overshoot is gone (round 3). 10 retaken references of benches 5 / 6: DECLARED (the per-bench table above gives the base -> appB -> final diffs). 03 archive_check same-mode no-archive baseline (12 px at 1 LSB on bench 1 under shader validation): DECLARED. 04 post-hoc fixture changes (rtrefl patch size, COLOR_0 fog sign threshold): rejected by the owner and not applied (95e184e).
- Closure battery items: `metal_syntax_check` on the real target: PENDING (CI on the owner's push; the macOS emulation passes 48 / 49 files). `variant_check`: PENDING the final battery. Full F13 / F14 corpus: CLOSED on 720ee63 (89 / 89), final-tip 24-of-24 run PENDING. Shader validation of DI scenes: CLOSED (first exercised in 08b, fix 039e22b). Forced-overflow subsurface capture (`tools/f15_subsurface_overflow_check.sh`, promised by ticket 17): PENDING the final battery. Final report schema number: CLOSED at 12 (ticket 04).
- Plan deliverables: docs/F15_HANDOFF.md, RENDERER.md and ROADMAP status lines are recorded as written on 2026-11-15 (box 5 above); the final per-ID update after the battery is PENDING and outside this reconciliation.

Found in F15 outside F15 code: all items marked CLOSED above in their own entries (C22-D3, wall behind glass, back-face transport rays, GI-mode environment, DI cap cause, a-trous option B) stay closed; the negative-f14-lut AGX compiler crash stays OPEN (NOT REPRODUCED in the 2026-11-15 stress study (agx_link_stress, 0 crashes in every attempt), cause unexplained, no mitigation added); GI with temporal accumulation 0.926 +- 0.016 filtered / raw is ISOLATED (transient history lag), not a defect.

### 2026-11-16 — final battery on the final tip (orchestrator, `build/final2`)

Tree: final tip (round 3 integrated, aniso b6c9b3d + records); binary rebuilt once more after 740b6b7 (+2 corpus tile pipelines, archive
only). Final sha256: phosphor `c2514d6fdbe7344ae50fa14528a7f89970f3d5a17aa6652705d556effc019787`, metallib
`5adf457e26bae033e65062dc429fbc8b48856099a125038e682d509c01f4614e`, archive `49e5d00e2965cc9b97c525f9f3585d5c698934b7febfc160e3f19278aa49cc9e`
(phosphor and metallib unchanged by the archive rebuild). Every step records whether the Mitsuba disocclusion correction was running
(owner rule: correctness may share the machine, timing waits for quiet); timing ran in a separate window (perf-log round 3).

| check | result |
|---|---|
| ctest | 46 / 46 (before and after the archive rebuild) |
| syntax (macOS emulation of metal_syntax_check) | 48 / 49 files, the known `__APPLE__` block of metalfx_lifetime.cpp (expected) |
| visual_check (8 benches, vs reference-f15-final, API + shader validation) | 0 px on every bench |
| variant_check, variant_check mesh | pass |
| archive_check | pass (before and after the archive rebuild) |
| hitch_check (quiet window) | 15 switches, 0 hitches, 21 / 21 archive hits |
| subsurface overflow | PASS, worst 1.6e-5 (limit 2e-3) |
| F13 exactness (5 PFMs vs exact-final-ms) | 5 / 5 SAME |
| JIT reflection_rt_lobe | 0 failures |
| XPC retry fixture | N=2 recovers, N=4 fails as required |
| forced Apple9 (6 scenarios, API + shader validation without RT) | all exit 0, 0 validation lines, effective apple9; native vs apple9 differ on layers-patch (197 318 px), subsurface-volumetric-sun (53 236 px) and combo-all-lobes-sun (17 343 px, max delta 203; CORRECTED 2026-11-17: this row first said 0 px, which was the a09f738 pre-check; the difference appeared between a09f738 and c2514d6f and is unchanged in round 4); Apple9 selects the `reduced` lighting preset (report `lighting.preset`), the attribution of the combo difference to the preset is not yet proven; transmission-glass-behind-glass, transmissive-shadow-pane-sun@rt, texture-coat-sun 0 px |
| transport fixtures volume / masked / thin-backlit / glass-weights / csm-thin | pass |
| glass receiver, thin GI back, texture transforms (primary, secondary, layers) | pass |
| COLOR_0 primary paths; secondary paths | pass; pass after a rerun with Pillow (the step's environment lacked it) |
| GI / reflection shadow rays through glass (gi-glass, with controls) | pass |
| spot falloff, DI paths | PASS after the tool fix 75166e0 (legacy / ReSTIR / clustered 2.17 / 2.16 / 2.16 % against 7 %) |
| spot falloff, secondary arms | RED as declared: RT reflection tilt 0.55 7.44 %, probe 17.59 %, GI 139.22 %, ReSTIR GI 19.60 % |
| unlit / punctual lights, unlit GI, residue, shadowed hit (off, ddgi), GI environment | pass |
| RT walk moving-camera clip | pass |
| anisotropic lobe-id runner | PASSED (history 40 / 40, flip control 0.0 / 1) |
| chess validation, chess rigs (green), chess O7 | pass, pass, GREEN (0 GPU allocations) |
| Khronos corpus (24 models) and validation | 24 / 24 render, validation pass; 0 archive misses after 740b6b7 (before: one visibility_tile miss each in ToyCar and GlamVelvetSofa, 0 fallback draws) |
| chess king / queen subsurface bound | RED as declared (ticket 20), equal to the pre-check |
| F15.2 permutation equivalence | 237 cases / 168 scenes 0 px / 0 ULP; controls leave the envelope in 151 of 216; every feature bit 0-8 discriminated; overflow 12 / 12 0 ULP; repeats 7 / 7 |
| transmission temporal clip | PASS on a converged control rebuilt for the final binary (128 seeds, half-vs-half 0.0055 / 0.0069 / 0.0096 <= gates / 3): smooth ior 1.33 / 1.5 / 2.0 p95 0.0230 / 0.0234 / 0.0307, max 0.0409 / 0.0480 / 0.0620 (gate 0.06); negative-refract-axis fails (0.1745 / 0.1690 / 0.1445); rough ROIs information (r0.1 0.0889, r0.3 0.0845, r0.6 0.0517); the battery run itself had no converged control (exit 5, not judged) |
| f15_check (286 scenarios, 209 evaluated) | captures on the final binary; missing references rendered by 3 CPU shards, then one `--evaluate`: RED, 21 scenarios miss their expectation, every cause diagnosed by A/B on the same binary (section "f15_check diagnosis" below); owner decisions pending |
| F13/F14 corpus with the MS references | first run (kept in `build/f15-corpus-r2-oracle112`): functional checks pass, but the F14 volume oracle rejected every frame (116-word fog block against 112) -> numerical 0 / 14, full-rate pairs 0 / 17, controls 15 / 30; oracle fixed (ed41156), full frozen-plan re-run in place |

### 2026-11-16 — f15_check diagnosis on the final binary (orchestrator; owner decisions requested)

- 16 layer-equal keys (`layers-endpoint-*` incl. cc0, `layers-blk-emissive-e0/e1`, `layers-hit-{refl,probe,gi}-e0/e1`): the stack has material
  features and resolves on the exact module, the single-material pair is featureless and stays on the fast module.  With the stack forced to
  the fast module (`--debug-material-module fast`) all 16 are 0 px bitwise against their pairs: the blending is exact.  Last green
  2026-11-15 20:54 (before fa6786d).
- `sheen-probe-env` (7.0 %) and `texture-coat-sun` (p99 1.55 %): pass with the round-2 numbers when engine AND reference run with MS off
  (sheen mean 1.2e-5, coat p99 0.34 %); with MS on the engine and the reference model the multiple-scattering energy differently on the
  probe-captured sheen path and the textured coat path (engine MS on / off +3.0 % and +2.3 % on the spheres).  Last green 2026-11-15
  21:44 (before 3812871); these keys were among the 101-148 never evaluated with MS in round 2.
- `combo-all-lobes-env` sphere 2 (ior / specular + coat + sheen + anisotropy): 5.12 % against 5 %, unchanged with MS off, the fast module or
  one reflection bounce; a build of 7704b21 (before the anisotropic albedo table) gives 4.07 % (PASS): the table (141655c) moves this
  4-lobe combination 0.84 % brighter while the reference is unchanged.
- `aniso-narrow-sun`: exact module, declared open (follow-up E).
- `layers-blk-all-e1`: its pending 2-ULP expectation is met by the exact module, asserted to pass since 858569f.
- The remaining failing rows are the declared subsurface default-tier keys (expected failures, expectation met).

### 2026-11-17 — identity of every final-battery result (owner request) and what replaces it

All results above came from phosphor c2514d6f / metallib 5adf457e.  Archive per result: `build/final2` steps 15:22-16:57 (build, ctest,
syntax, visual, variant, variant-mesh, archive_check, overflow, exactness, JIT, XPC, Apple9, every round-2/3 fixture, chess, Khronos corpus,
equivalence) and the f15_check captures (286 / 286, exit 0, 16:57-17:30) ran with archive 128a394a; the POST steps (ctest, archive_check,
Khronos corpus + validation, 18:00-18:15), the quiet window (hitch, D2, costs, metal-tt, 18:20-19:13), the transmission clip (19:13-19:42)
and the diagnosis runs ran with archive 49e5d00e.  The 2 added tile pipelines only change archive hits of two corpus models (ToyCar,
GlamVelvetSofa); images do not depend on the archive (captures use final pipelines, `--pipeline-sync`), which archive_check re-confirmed on
49e5d00e.  f15_check's GPU run was stopped by the orchestrator after the 286 captures because the reference rendering (every transmission
reference re-renders after the f15_reference.py change) held the GPU lock for hours; the references were rendered by CPU shards and one
`--evaluate` pass judged the captures.  The F13/F14 corpus re-run on c2514d6f was stopped deliberately after 37 of 89 cases (2026-11-17
~00:00) as a pre-check: the owner's round-4 fixes replace this binary, the GPU went to the fix verifications, ed41156 is validated on a
frozen F14-numerical subset (build/f15-corpus-oracle-check).  The owner's round-4 decisions (layer test flag, MS fixes, combo
investigation) change the binary: ONE complete final battery is re-run on the new hashes (f15_check in full, equivalence, archive_check,
visual, hitch, D2 on a quiet machine, the full F13/F14 corpus), and only those results are the final evidence.


### 2026-11-17 — round-4 final battery, part A (orchestrator, `build/battery-r4`, tip 817b080)

Identity for every row: phosphor e01bb209, metallib 0e5423c1, archive c2776814 (one binary, one archive, no rebuild between steps;
01:37-04:35, no Mitsuba running, battery 100 %).

| check | result |
|---|---|
| build (no-op) / ctest | 46 / 46 |
| metal_syntax_check | 49 files, 1 expected failure (metalfx_lifetime `__APPLE__` Objective-C block) |
| visual_check | every bench 0 px |
| variant_check (indexed / mesh) | 284 compatible (worst 26 px, delta <= 1), 52 incompatible differ; both pass |
| archive_check | all scenarios pass |
| subsurface overflow, exactness self-checks (5 scenes), JIT `reflection_rt_lobe`, XPC N=2 / N=4 | pass / pass / 0 failures / N=2 recovers, N=4 fails as required |
| forced Apple9 (6 scenes) | exit 0, 0 validation lines both families; native vs apple9 0 px on glass-behind-glass, pane@rt, texture-coat-sun; combo-all-lobes-sun 17 343 px, subsurface-volumetric-sun 53 236 px, layers-patch 197 318 px (reduced preset / diffusion fallback, unchanged since round 3) |
| round-2/3 fixtures (transport x5, glass receiver, thin GI back, texture transforms, vertex colour, GI glass shadow, spot DI, unlit, residue, shadowed hit, GI environment, walk clip, aniso lobe-id) | pass |
| spot falloff arms | RED as declared (probe / GI arms, owner decision) |
| chess, chess O7, chess validation | GREEN (every group passes, controls fail), O7 GREEN |
| chess king / queen subsurface bound | RED as declared (ticket 20) |
| Khronos corpus + validation | 24 / 24 render, 0 archive misses |
| transmission temporal clip | converged control rebuilt for this binary (half-vs-half 0.0055 / 0.0069 / 0.0096); smooth ior 1.33 / 1.5 / 2.0 p95 0.0230 / 0.0234 / 0.0307, max 0.0409 / 0.0480 / 0.0620 (gate 0.06); negative-refract-axis fails (0.149) |
| F15.2 permutation equivalence | 237 cases / 168 scenes 0 px / 0 ULP; controls leave the envelope in 151 of 216; feature bits 0-8 discriminated; overflow 12 / 12 0 ULP; repeats 7 / 7 bit identical |
| f15_check | 286 / 286 captures (03:54); missing references by 3 CPU shards (148 done, 0 errors); one `--evaluate`: RED only by `aniso-narrow-sun` (declared open, 3.77 % at one pixel, max 3 %).  FAIL rows 32 -> 13: the 16 layer-equal keys, `sheen-probe-env` (worst mean 1.8e-4, limit 5e-3), `texture-coat-sun` (worst p99 0.34 %) and `combo-all-lobes-env` (worst sphere mean 0.92 %, limit 5 %) now pass; the remaining rows are the two declared subsurface default-tier keys and the `@rt-walk-*` negative controls, which must fail |
| hitch (quiet window, first step) | 15 switches, 0 hitches, worst post-switch frame 1.22 ms |

### 2026-11-17 — round 4: before / after on the final binary, and two corrections to the round-3 record

Every f15_check key with a coat, sheen, probe reflections, MS or a furnace, round 3 (`build/final2`, c2514d6f) against round 4
(`build/battery-r4`, e01bb209), worst row over spheres; all pass on round 4.  Changed keys only (the other 60 are identical to the
printed precision; table generated from both results.json files):

| key | round 3 worst mean / p99 / max | round 4 worst mean / p99 / max |
|---|---|---|
| `sheen-probe-env` | 7.02e-2 / 8.25e-2 / - | 1.78e-4 / 6.43e-4 / - |
| `texture-coat-sun` | 1.27e-2 / 1.48e-1 / 2.83e-1 | 2.74e-4 / 3.41e-3 / 1.46e-2 |
| `texture-coat-env-resolve` | 7.36e-4 / 2.43e-3 / - | 4.76e-4 / 2.18e-3 / - |
| `combo-aniso-coat-env` | 1.69e-2 | 1.91e-2 |
| `combo-aniso-sheen-env` | 1.35e-2 | 1.01e-2 |
| `furnace-aniso` spheres 4-7 / 8-11 (metallic 0, roughness 0.3) | 0.9999-1.0000 | 0.9856-0.9915 / 0.9640-0.9680 |
| `furnace-combinations` spheres 1 / 2 / 6 / 9 | 0.9976 / 1.0000 / 0.9714 / 0.9790 | 0.9694 / 0.9808 / 0.9478 / 0.9764 |

Corrections:
- `texture-coat-sun` round 3: the worst sphere was sphere 3 (metallic 1, textures 0x4): p99 14.8 %, max 28.3 % (sphere 5: p99 5.1 %;
  sphere 0: p99 1.55 %).  The diagnosis section above and tickets 03 / 09 quoted sphere 0's 1.55 % as the key's p99; the key failed by far
  more.  On round 4 every sphere passes (worst p99 0.34 %, sphere 0; max 1.46 %).
- Energy cost of 817b080 (combo fix), measured on the final binary: besides `furnace-aniso` spheres 4-11 and `furnace-combinations` sphere 1
  (as reported by W3), `furnace-combinations` spheres 2 (1.0000 -> 0.9808) and 6 (0.9714 -> 0.9478) and 9 (0.9790 -> 0.9764) also lose
  energy.  The reference loses 0.011-0.015 on `furnace-aniso` spheres 8-11 and gains 0.006-0.012 on spheres 4-7; on
  `furnace-combinations` it loses 0.016 / 0.015 / 0.023 / 0.027 on spheres 1 / 2 / 6 / 9: the engine now loses up to 0.035 (aniso 8-11)
  and 0.052 (combinations 6) where the reference loses 0.014 and 0.023.  The furnace gate is an upper bound (1.005), so every furnace key
  passes; the `@resolve` furnaces are unchanged (no transported signal).
