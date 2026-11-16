# 05: Feature plumbing proven by ior and specular (resolve and forward)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_ior` and `KHR_materials_specular` render for analytic lights, sun disk and environment in the resolve and forward paths through the new chain: extension records via SceneStore deltas with the layout of EVERY F15 feature frozen now (static_asserts), feature bits, the feature mask as a reserved function constant specialising forward and resolve per scene, and runtime edits and feature toggles.

**Blocked by:** 03, 04

**Status:** resolved (2026-11-13; furnace, recovery and 2x2 sub-clauses delivered and verified by ticket 08b; D2 band met under the owner's option A, ticket 24)

- [x] Extension records (layouts for all features frozen in `gpu_types.h`, `GPUMaterial::pad` retyped to `u32` with the same size) reach the GPU only through SceneStore deltas; `--debug-gpu-scene` and `verifyAgainstEcs` cover the extension buffer, with a corruption control that must FAIL
- [x] Reserved constant semantics fixed: undefined ⇒ no extension lobes (scenes without features keep byte-identical pipeline descriptors and archive keys, visual_check 0 px); the compile fallback uses an explicit all-lobes value; ADR-0003 wording matches (deviation for compute pipelines recorded there)
- [x] Extension at default values renders exactly like the base material (0 px)
- [x] ior/specular grids pass decision-11 direct-light thresholds (ior = 0 mode, f90 = specular weight); furnace bound holds; negative control (extension ignored) fails — grids and controls done; the furnace bound of EXTENDED materials needs the composite's extended diffuse weight → carried to ticket 08
- [x] Parameter edits change the next frame; turning a feature on/off at runtime is a counted structure change with fixed per-material record capacity, the all-lobes fallback serves until the specialisation compiles, and a hitch test toggling a feature shows 0 GPU allocations in measured frames (O7/O9)
- [x] Extension-record deltas bump every material-revision consumer (`gi_passes.cpp:91`, `reflection_passes.cpp:254`, `shadow_passes.cpp:298`, `direct_lighting_passes.cpp:108`, `engine.cpp:1718`, `acceleration_structures.cpp:620`); indirect/denoised signals recover within the F13.6 recovery criterion after an edit — revision bumps done; the F13.6 recovery measurement after an edit → carried to ticket 08
- [x] Adaptive 2×2 shading reuse excludes materials with feature bits; offline-reference snapshots versioned and extended with extension records — snapshot done; the 2×2 exclusion is in `visibility_resolve.metal` but not yet exercised on the GPU → carried to ticket 08
- [x] Sponza D2 within the decision-11 band (A/B/A) → carried to ticket 24 (quiet machine) — D2 band MET under the owner's option A (ticket 24: 2026-11-15 median -0.038 ms against ticket 03 + 26fe229, 0 GPU allocations; round-2 quiet rotation t03 -> final -0.059 ms; the final binary's rotation is recorded in ticket 24 and the handoff)
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — implementation and M5 verification (commits 18c582c…f1350a7 plus
the integration commit with this note).

- Layouts and records: `GPUMaterialExt0..3` frozen (80 B each, static_asserts
  and offset tests), `GPUMaterial::pad` → `extensionIndex` + `pad1`; records
  travel as SceneStore deltas (fifth scatter). `--debug-gpu-scene 1` on
  specular-grid-sun: 128 checks, 0 failures (PASS); `--debug-gpu-scene-corrupt
  extension`: FAIL on every check (negative control). `verifyAgainstEcs`
  checks groups, ownership and record bytes.
- Reserved constant: index 22, undefined ⇒ 0. Scenes without features:
  visual_check 8/8 at 0 px against `build/reference-f15-appB`; the exact F13
  captures (`build/f15-exact/t05`) equal base03.
- Neutral values: Sponza copy with named materials plus
  `--material-overrides` setting m6 to the neutral specular block: 0 px in
  forward and visibility.
  - m6 at ior 2.5 / colour (1, 0.6, 0.3): forward diff confined to the m6
    arches (1 204 318 px); the rest of the scene is bit-identical in the
    specialised pipeline.
  - Visibility resolve with the same override: 14 px elsewhere differ by
    1 LSB (codegen of the specialised kernel; recorded in ADR-0003).
- ior/specular: `tools/f15_check.py` GREEN (`build/f15-check/t05c`).
  - specular-grid-sun: worst sphere p99 0.27 %.
  - specular-grid-env-resolve: mean 0.010 %, p99 0.014 %.
  - Every negative control fails: specular-ignored, f90-one,
    specular-weight-not-on-f0 and their declared variants.
  - The environment reference declares `extension_diffuse = composite-base`:
    the F13 composite still weights the environment diffuse with the base
    model until ticket 08 (criterion added there). The support report names
    the approximation `di_transport_and_composite_diffuse_use_base_model`.
- Crash found and fixed: `FeaturePipelines::serve` returned a pending compute
  handle (no flexible fallback for compute), which led to SIGSEGV in
  `setComputePipelineState`. It now serves the specialisation or the
  all-lobes pipeline once either has an object, else the base pipeline
  (counted fallback). Deviation recorded in ADR-0003.
- Toggle/edit (`--material-toggle-every 30`, 300 frames):
  - neutral frames are bit-identical to each other, as are active frames;
    frame 0 differs (first-frame fallback before the specialisation is
    available);
  - 0 GPU allocations and 0 render-thread compiles;
  - frame p95 0.934 ms against 0.940 ms without toggles.
- Archive: targeted update of `shaders/pipelines.mtl4-json` (+16 descriptors
  with constant 22 = 1 and 511; compute 296→306, render 170→174, tile 3→5),
  0 misses in the material scenarios.
  - `tools/archive_check.sh` PASS on every local scenario (full, b, c×2,
    d-stale, e-missing). It now compares with a same-mode no-archive
    baseline: the shader-validation reference differs by 12 px at 1 LSB on
    bench 1 since ticket 03, archive on/off identical.
  - The stale scenario copies `shaders/materials/`.
  - `tools/hitch_check.sh` OK (15 switches, 0 hitches, 0 render-thread
    compiles).
- `--material-overrides` (ticket 04 residue assigned here): accepted with
  `--bench 4` and a glTF `--scene`, applied by the scene viewer after the
  load (before the store mirrors the library); unknown names abort the load
  (EXIT 1).
  - Sponza feature-specialised visibility pipelines are not in the archive
    (12 worker compiles, 0 render-thread compiles): harvesting product scenes
    belongs to the corpus tickets.
- ctest 15/15; `tools/test_f15_reference.py` 36 tests OK.
- Open, carried: furnace bound and F13.6 recovery of extended materials, and
  the GPU exercise of the 2×2 exclusion (ticket 08); Sponza D2 band (ticket
  24). An optional prewarm of the specialisation would remove the first-frame
  fallback (not required by the criteria).

2026-11-15 — closure audit (ticket 24):

- 1-3, 5, 9: MET as already ticked (18c582c, 8db5583, f8b995c, c44667a, 28c0988, c816623).
- 4 grids + furnace of extended materials + control: MET (grids and controls here; furnace of extended materials carried to and met in 08: furnace-specular 1.00027 transport, 1.0000001 resolve). Now ticked.
- 6 revision bumps + recovery: MET (bumps f1350a7/a7535f1; recovery in 08b: 0 frames, delayed control 5 > 4 fails; the six consumer files are not individually enumerated in the record). Now ticked.
- 7 2x2 exclusion + snapshots: MET (snapshots 28c0988; 2x2 GPU exercise in 08b: 0 of 282 626 pixels differ, control changes 2 276). Now ticked.
- 8 D2 band: NOT MET at HEAD (see annotation). Open, ticket 24.
Open notes: optional prewarm of the specialisation not built (declared); Sponza feature-specialised visibility pipelines not archived (12 worker compiles, declared).
