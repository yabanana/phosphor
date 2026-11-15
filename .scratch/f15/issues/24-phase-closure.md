# 24: F15 phase closure: residue audit, regressions, measurements and handoff

Roadmap: F15.1–F15.4

**What to build:** Severe residue audit (roadmap lines, plan, exit criteria, TODOs), full regression battery, Sponza budget re-measured, Apple9 forced path, documentation and per-ID handoff; roadmap boxes ticked only for verified scope.

**Blocked by:** 15, 18 (if triggered), 20, 21, 22, 23

**Status:** ready-for-agent

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
