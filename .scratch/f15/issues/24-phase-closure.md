# 24: F15 phase closure: residue audit, regressions, measurements and handoff

Roadmap: F15.1–F15.4

**What to build:** Severe residue audit (roadmap lines, plan, exit criteria, TODOs), full regression battery, Sponza budget re-measured, Apple9 forced path, documentation and per-ID handoff; roadmap boxes ticked only for verified scope.

**Blocked by:** 15, 18 (if triggered), 20, 21, 22, 23

**Status:** ready-for-agent

- [ ] `ctest`, `metal_syntax_check`, `tools/visual_check.sh` (0 px vs the ticket-03 baseline), F13/F14 corpus, `variant_check`, `archive_check`, `hitch_check` green; API validation clean (shader validation where the F14.4 defect does not apply)
- [ ] Sponza D2 GPU p95 A/B/A against the ticket-03 baseline within the decision-11 band; 0 GPU allocations
- [ ] `--force-family apple9` run of the material bench with features
- [ ] ROADMAP F15.1–F15.4 ticked only where verified; approximations and deferred items listed (caustics, dispersion, multiple-scattering compensation, back-lit thick subsurface if the hybrid is not triggered or not met, non-uniform texture transforms, coated emission of sampled emitters, GI/probe rays hitting glass as opaque)
- [ ] perf-log, opt-log, RENDERER.md and per-ID handoff written; no push or PR without the owner

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
- IN PROGRESS, owner 2026-11-14 "Correggere in F15": with any GI mode the declared environment reaches no primary surface (the
  INDIRECT atlas excludes it by design, no other pass adds it): shadowed wall 16/255 with `--gi ddgi` vs 194 with GI off. Required:
  one ownership rule (no double count against the GI-off term and the reflection/probe paths), reference albedo x L_env with a
  negative control, F13/F14 corpus rerun with every image change explained (thresholds only tightened, re-cut references justified),
  Sponza D2 re-measured on a quiet machine in an interleaved A/B with the previous binary, closure-document row.
- IN PROGRESS, owner 2026-11-14 "Dimostrare o correggere": the F11.5 DI cap (1bf0e19) ratio budget/exact is 1.016 +- 0.008 with
  the denoiser off and 0.973 (sem 0.001) with `--lighting-denoise custom`. Measure the cause (clamp disabled/widened, same scene
  below the 128-light cap); fix a systematic loss with a failing-before fixture, or prove the expected behaviour against a
  reference. No "probably" may remain in the closure documents.
  RESULT (7e3f730): cause proven = the a-trous luminance edge-stopping weight (row-normalised) of the F13.4 denoiser loses energy
  on heavy-tailed noise; the cap's estimator is unbiased (raw 128 / raw 8192 strata 0.9996 +- 0.006); denoised/raw 0.977 +- 0.004
  (128), 0.988 +- 0.002 (8192), 1.0000 for exact lists; luminance weight off 1.004 +- 0.007; CPU reference in tests/test_denoise.cpp.
  Owner asked 2026-11-14: A) energy-preserving a-trous (reopens F13.4/F13.6 acceptance) or B) declared behaviour (recommended).
- Declared (owner noted, no objection): single-sided planes cast sun shadows from both sides in CSM (two-sided by contract) and RT.
- OPEN, F15 (transmission, ticket 13 territory): 67 dark pixels on the silhouette ring of the IORTestGrid transmissive spheres
  (N.V > 0 there, untouched by the D3 fix); hypothesis: a refraction sample that returns nothing at grazing incidence while the
  composite released 1 - t of the diffuse. Analyse before closure.
- O7 caveat: a non-archived RT configuration with a cold compiler cache allocates one IFT per frame slot when its final variant
  arrives (ticket 13 record).

Quiet-machine measurements for this ticket:
- Sponza D2 A/B/A band;
- cost of transmission RT, transmissive shadows (RT and the CSM map), the subsurface diffusion pass, the iridescence per light
  (ticket 12) and the GI-back lookup;
- binning vs union on chess.
