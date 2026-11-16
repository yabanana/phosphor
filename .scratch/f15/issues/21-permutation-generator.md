# 21: Material permutation generator and budgets

Roadmap: F15.2

**What to build:** The app exports feature-class manifests from fixtures and corpus; the generator (extension of `variant_gen`) turns manifests plus a declared list into the class table, MSL constants and pipeline descriptors and updates the archive in a targeted way; per-class tile binning measured against the union pipeline on the chess fixture (provisional decision; final with the terrain in ticket 23); budgets enforced and reported.

**Blocked by:** 19, 20

**Status:** partial (2026-11-14, integration step added the chess and corpus manifests: 74 pipelines, 38 permutations); (2026-11-16, round 2: criterion 2 is now met literally on the integrated tip by an exact-maths second copy of the visibility resolve kernels for scenes with F15 features (owner option (a), `-fmetal-math-mode=safe`, tools/exact_module_archive.py): the official `tools/f15_permutations.py equivalence` with RELATIVE_TOLERANCE 0.0, run twice on 038eb10, gives 200 cases (162 scenes) bit-identical (0 px, 0 ULP), repeated reference 7/7, overflow bin 12 runs 0 ULP, ToyCar, ClearCoatCarPaint and all nine `layers-blk-*` keys exact; full budget run 200 cases 0 fallback draws / 0 render-thread compiles, 5 featureless exactness cases cmp-identical, visual_check 0 px, archive_check and hitch_check pass; the earlier ToyCar / ClearCoatCarPaint run-to-run variation (8 of 15 runs differ on the old base 78d173d, 11 of 20 on a quiet machine) does not occur on the tip (0 of 24 runs) but was NOT root-caused or bisected; criterion 4 chess binning vs union A/B/A is recorded in ticket 24 (side-light -1.8 %, shadow rig +6.7 %, back-light +9.8 %, the 5 % rule is not met, binning stays opt-in); the quiet-machine D2 / feature-scene cost of the exact module is pending the orchestrator's final battery)

- [x] Generator determinism: two runs byte-identical (ctest); malformed manifests and schema errors rejected
- [ ] Every generated class pipeline equals the generic all-lobes resolve exactly (0 pixels) on every fixture; negative control (class missing a used lobe) differs -- PARTIAL: 77 of 116 fixture cases are bit-identical, the others stay within 3.1e-6 relative (37 ULP: the compiler schedules a specialised pipeline differently) and 2 cases differ by 1 pixel of 1 LSB in the 8-bit image; the control differs for every feature bit, but 21 of 73 scenes with features cannot discriminate through the resolve colour (listed below); chess/corpus fixtures not yet
- [x] Budget: ≤ 64 archived material permutations, ≤ 16 active classes per scene (beyond: logged generic fallback), archive build time +≤ 25 %, 0 render-thread compiles and 0 fallback draws in measured frames
- [ ] Binning vs union on the chess fixture (A/B/A, 3 replicas, quiet machine) recorded as provisional in opt-log/perf-log
- [x] Targeted archive update; `pipelines script` test green

## Verification record (2026-11-14, M5 Max, branch worktree-agent-a281fecfae84dabcf)

Commits: `a740944` (manifests, generator, plan), `71d67c6` (per-class bins), `9f41677` (archive update), `1b2a5cf` (runner), then the docs commit.

**Design.** Manifest schema `phosphor.material-class-manifest` v1 (`src/renderer/material_class_manifest.h`, closed objects: name, family, union,
volumetric, classes[{mask, materials, lobes}]); the same header is the exporter's writer and the generator's reader. `tools/class_manifest export` (GPU free)
and `phosphor --export-class-manifest FILE` (the scene the app runs) produce it; `shaders/material_classes/{material,transmission,shadow}.json` are the
committed exports of all 81 fixture scenes (a unit test fails when one is stale). `shaders/material_classes.def` is the declared list (budgets, kernels,
manifests, `declare` lines); appending an input is one line. `variant_gen classes` writes `material_classes.generated.h` (27 class masks, 81 scenes, 29
permutations, budgets), `.generated.metal.h` and `material_class_plan.json` (62 pipelines: kernel + function constants) into `build/generated/pipeline/`.
Per-class bins: `RESOLVE_BIN` (reserved constant 24) = `0x1000 | mask` (a class bin that resolves only the pixels of that mask, specialised with
`MATERIAL_FEATURES = mask`) or `0x2000` (the all-lobes overflow bin); up to 16 bins by descending material count; scenes without features keep the legacy
four classes and byte-identical descriptors.

**Verification.**
- `ctest` 23 of 23 (new: `test_material_classes.cpp` 10 cases: schema rejections, stale-manifest check, table and budget coverage of all
  fixtures, two generator runs byte-identical and independent of the manifest order, plan content, definition/budget/duplicate errors, class table; Python
  `f15_material_class_archive`, `f15_permutations_runner`); `pipelines script` 8 of 8.
- Archive: +62 compute descriptors (687 -> 749; render, mesh and tile unchanged; existing entries a prefix, the archive's 332 history duplicates not added to);
  distinct `MATERIAL_FEATURES` values 23 -> 29 (budget 64); `metal-tt` median of 3: 9.457 s -> 10.564 s (runs 9.28/9.50/9.46 and 10.34/10.56/10.76), +11.7 % (budget 25 %);
  archive 91.9 -> 103.0 MB (`tools/archive_build_time.py`, a build-time measure).
- Equivalence (`tools/f15_permutations.py equivalence`, 116 cases = every f15_check key of the 81 scenes plus 3 scenes without key, both resolve kernels, 640x360,
  8 frames, linear HDR bit pattern compared): class bins vs the all-lobes resolve: 77 cases bit-identical, max 3.09e-6 relative (37 ULP, `combo-irid-specular-sun`;
  iridescence code), 8-bit image 2 differing pixels in 2 cases (1 LSB each); all-lobes arm repeated 7 of 7 bit for bit; negative control (every lobe dropped from the
  class pipelines): 55 of 108 cases leave the envelope (up to 7e7 ULP), every feature bit 0..7 is discriminated by 4 to 17 scenes; overflow bin
  (`--debug-class-budget 2`, 4 scenes with >= 4 classes): within the envelope (max 2 ULP, overflow classes counted); app manifest export of `combo-all-lobes-sun`
  equals the committed entry. Scenes NOT discriminated by the resolve colour (the lobes act in the lighting/transport kernels, or the parameters are
  neutral; their equivalence is therefore vacuous on the HDR signal): aniso/clearcoat/iridescence/sheen/specular-grid-env, furnace-thin-subsurface,
  specular-grid-edit, subsurface-thin-env, -env-resolve, -gi-back, subsurface-volumetric-subpixel, transmissive-shadow-point and nine transmission scenes.
- Budget (`... budget`, 116 cases, warmup 64, 32 measured frames, archive, no `--pipeline-sync`): measured-frame fallback draws 0, render-thread compiles 0,
  compiler calls 0 in every case; archive misses 0; at most 11 classes (bins) in a scene. Whole-run transient (warmup only) 9..14 fallback draws per feature scene,
  the phase-branch binary shows 18 for `aniso-grid-sun` without bins (pre-existing).
- `tools/visual_check.sh` (validation on, BENCH8_ARGS=--instances 100000, references reference-f15-appB): 0 of 5,760,000 pixels, 8 of 8 benches;
  `tools/archive_check.sh` all scenarios pass; `tools/hitch_check.sh` 0 hitches (SWITCH n=15); `tools/f15_check.py --run --expect green --reference-cache build/f15-check/t19int`
  (112 keys, commit `37c59e4`, 62 min with the volumetric references re-rendered): GREEN, exit 0, every key met its expectation.

**Deviation (criterion 2).** A pipeline specialised with a different `MATERIAL_FEATURES` value is not bit-identical to the all-lobes pipeline: the Metal compiler
schedules the shortened code differently (ADR-0003 already recorded 1 LSB on Sponza). The gate is a declared envelope fixed from the first corpus (relative 1e-5
per channel; measured worst 3.1e-6) next to the bitwise, ULP and 8-bit counts; "exactly 0 pixels" is met in 77 of 116 cases and for the 8-bit image in all but two
pixels. If the owner wants strict bit identity the all-lobes pipeline would have to serve every class (no specialisation benefit).

**Open.** Chess fixture and Khronos corpus manifests (append a `manifest` line after `class_manifest`/`--export-class-manifest`; rerun `harvest` + `material_class_archive.py
update` for the new classes); the A/B/A binning vs union measurement (3 replicas, quiet machine) on the chess fixture, final decision with the terrain of ticket 23
(logged as provisional in `docs/opt-log.md`, no timing claim made); layers (bit 8) has no fixture yet.

## Integration step: chess and corpus manifests (2026-11-14, after the merge onto the phase branch)

Commits: manifests `tools/f15_export_manifests.py` + `chess.json`, `khronos-corpus.json`; archive update (+12 compute); table tests. After the merge `f15_material_class_archive` failed
on [38, 64, 106, 113, 224]; all five are explained by manifests: 38 = ToyCar (clearcoat+sheen+transmission on separate materials), 64 = IORTestGrid class (volume alone),
106 = AnisotropyBarnLamp, 113 = IridescenceSuzanne, 224 = chess-sidelight union (transmission+volume+subsurface). Nothing was forced into the plan.
- Manifests exported by the app (`phosphor --export-class-manifest`, serial, locked): 3 chess rigs (classes [0,96], [0,96,192], [0,96]) and 21 corpus models (ABeautifulGame and ToyCar
  included; DiffuseTransmissionTest, TextureTransformMultiTest and GlamVelvetSofa are refused by the loader for required extensions, as in the corpus runner). Max 5 classes in a scene.
- Plan 74 pipelines, 38 archived permutations (budget 64), <= 16 classes per scene; archive +12 compute descriptors (871 total; existing entries a prefix, no duplicates added), `check` passes.
- metal-tt (machine loaded by other agents, runs 13..27 s, interleaved, median of 3): branch archive without any RESOLVE_BIN pipeline 14.79 s / 113.3 MB, with all 74: 16.18 s / 126.4 MB
  (+9.4 % time, +11.6 % size; budget +25 %). The pre-ticket archive (commit 432f02a) measured 10.96 s / 91.9 MB in the same session, so the whole branch (tickets 20 and 22 included) is
  +34 % over it; the RESOLVE_BIN share is the +9.4 %.
- ctest 26/26 with KHRONOS_CORPUS_DIR; `visual_check` 0 px (8 benches); `archive_check` pass; `hitch_check` 0 hitches.
- Class equivalence on the 24 new scenes (8 frames, the runners' own rigs, both lighting paths): 4 bit-identical, 7 differing 8-bit pixels in total (max delta 1 LSB), all-lobes repeat bit-identical
  6 of 6, negative control leaves the envelope in 22 of 22 scenes with features, every feature bit 0..7 discriminated; overflow bin (2 scenes) runs. **The 1e-5 information envelope is exceeded
  on 15 of 24 textured real-asset scenes: worst 1.0e-2 relative (99,533 ULP, ToyCar) at isolated pixels, typically 4e-5..8e-4** (specular highlights and texture filtering amplify the codegen
  difference; the fixtures stayed at 3.1e-6). Reported, not gated further; 8-bit stays within 1 LSB.
- Budget run (warmup 64, 32 frames): 0 fallback draws, 0 render-thread compiles, 0 compiler calls in measured frames, 0 archive misses, at most 5 bins.
- Chess runner (`f15_chess_check --run --expect green`): GREEN, 26 engine runs with 0 archive misses. Corpus runner: 21 rendered, 3 refused, 0 misses.

2026-11-15 — closure audit (ticket 24):

- 1 determinism + malformed manifests: MET (ctest, 26/26 after integration). Ticked.
- 2 class pipeline equals generic exactly (0 px): PARTIAL (116 fixture cases: 77 bit-identical, max 3.09e-6 relative = 37 ULP, 8-bit 1 LSB in 2 pixels; chess/corpus scenes: 1e-5 envelope exceeded on 15 of 24 textured scenes, worst 1.0e-2 relative, 99 533 ULP ToyCar, 8-bit within 1 LSB; terrain 2 of 8 at 6.8e-5; envelope fixed after the first corpus; 21 of 73 featured scenes cannot discriminate through resolve colour). Open.
- 3 budget: MET (38 permutations, 41 with terrain; metal-tt +11.7 % / +9.4 %; 0/0). Ticked.
- 4 binning vs union on chess A/B/A: NOT MET (opt-log 2277 records correctness and provisional opt-in with no timing; chess A/B/A not run). Open.
- 5 targeted archive update + pipelines script test: MET (+62 then +12 descriptors, 8/8). Ticked.

## Exact-maths module for scenes with features (2026-11-15, owner option (a); branch f15c-bitexact-a from cl/f15-plan-c38b51 78d173d)

Candidate B (a global `-fmetal-math-mode=safe` on the whole material module) cost +0.47/+0.63/+0.68 ms on Sponza D2 and was rejected.  Replaced by a second, exact copy of the visibility
resolve kernels used only by scenes with F15 features.

**Design.** `shaders/material_passes_exact.metal` = `namespace phx_exact { #include "visibility_resolve.metal" }`, compiled as its own module with
`-fmetal-math-mode=safe -DPHOSPHOR_EXACT_MODULE=1 -Wno-modules-import-nested-redundant`, linked second (cmake/App.cmake, per-module flag hook; hot-reload build passes the same flags).  The namespace gives
the kernels the names `phx_exact::visibility_resolve` / `phx_exact::visibility_lit_resolve` and makes every inline function private (no link-time merge with the base module's differently compiled
copy; a namespaced kernel keeps its function-constant indices).  `shaders/materials/exact_math_begin/end.h` and `exact_pragma_on/off.h` are EMPTY in the base module; in the exact module they bracket bsdf.h,
the body of material_shading.h, resolveVisibilityPixel and the shared headers with contract(off), math_mode(safe) and a private namespace of `metal::precise` using-declarations (the precise names are not
exported: exporting them made every unqualified sqrt/length of the other kernels ambiguous).  layer_blend.h uses `metal::precise::exp` in the exact module.
Selection: `pipe::forward::exactModuleKernel/inExactModule`, applied in `withMaterialFeatures` (union-specialised and all-lobes pipelines) and in `VisibilityRenderer::binPipeline` (class bins incl. the
all-lobes overflow bin): a pipeline with MATERIAL_FEATURES or RESOLVE_BIN defined resolves from the copy; a featureless scene requests exactly today's descriptors.
Not moved (not needed by the criterion): forward, lobe-guide DI, reflection, denoise kernels and the adaptive/tile resolves of feature scenes stay in the base module.

**metal-tt relocation risk.** The separate module links second, the archive translates (build passes, archive 133 MB, `tools/archive_build_time.py` median 18.6 s of 3 on the loaded machine, branch before this change
16.18 s per the earlier record), `archive_check.sh` all scenarios pass (full: 18 hits / 0 misses), `hitch_check.sh` 0 hitches, 21 archive hits, 0 misses, 0 render-thread compiles.  No relocation failure
appeared with the namespaced, resolve-only module.

**Archive.** `tools/exact_module_archive.py update|prune|check` (new): +131 compute descriptors of the two exact kernels (permutation harvest of 112 scenes and the layers harvest of 40 runs, binary x2) and
-270 now-dead base-module resolve pipelines specialised with 22/24; `material_class_archive.py check`: 80 plan pipelines once each, 42 of 64 permutations; `material_classes.def` names the exact kernels.
Class pipelines of texture-transform scenes (constant 26) are archived by this harvest as well.  ctest 37/37.

**Verification so far.**
- Featureless scenes: the 5 `f15_exact.sh` cases (sponza-d2, cornell-restir, cornell-clustered-ssr, manylights-restir, grid-sky) are cmp-identical to `litefix/exact-new`; `visual_check`
  (EXTRA_ARGS=--pipeline-sync, BENCH8_ARGS="--instances 100000") 0 of 5,760,000 px on all 8 benches.  The generic-vs-variant forward difference (9/1/5/9/25 px, delta 1) is unchanged: that module is not touched.
- Equivalence (ref = all-lobes vs class bins, 640x360, 8 frames, --pipeline-sync): SheenTestGrid, TransmissionTest, ToyCar (second run), combo-irid-specular-sun, base-grid-sun: 0 px / 0 ULP.
  INTERMITTENT, not yet explained: the first ToyCar run showed 542 px on a single scanline (y = 265, x 600..680; relative up to 0.2); the same binaries rerun gave 0.  Needs the official runner's repeated-reference arm.
- Budget run (7 feature scenes): measured-frame fallback draws 0, render-thread compiles 0, compiler calls 0 everywhere; whole-run archive misses: chess 0, combo-all-lobes 0 and one
  `phx_exact::visibility_lit_resolve` in four scenes (harvest gap: a scene harvest only exercised its own lighting path), fixed in 3e52314 (requestBoth), re-harvest pending.  ToyCar also misses 8 pipelines of
  texture-transform scenes that are NOT exact-module pipelines (Forward v7 forward_fs, indexed overflow fallback, material_lighting_guides, DI receiver guides): the base archive holds only 12 constant-26 descriptors,
  so these predate this change (ticket 22 archive scope).
- `layers-blk-all-e1` (layers agent, 2 ULP): not on this base.  The exact module covers the layer stack (resolveLayerStack, layer_blend.h with precise exp, the blend arithmetic under contract(off) / safe maths); expected to be
  covered, to be re-tested by the orchestrator after the layers merge.

**Open (held by the owner's battery rule, 26 % at the time; nothing GPU or builds started after it):** re-harvest with 3e52314 + archive update; the official `tools/f15_permutations.py equivalence` on every arm
(repeated reference, overflow, app export, negative control; RELATIVE_TOLERANCE 0 is not set on this branch yet: set it only after the run is exact); the budget run on all scenes; the ToyCar scanline.
Cost commands for the orchestrator: `python3 -I tools/f15_costs.py plan` lists the A/B/A runs of the feature items and the chess rig (A = tip build, B = this build, `--only` per item); D2 sanity:
`zsh f15_exact.sh <build> OUT` case `sponza-d2` (cmp-identical image) plus the usual A/B/A of the D2 band.

### Results of the exact module (2026-11-15, branch f15c-bitexact-a)

**Official equivalence run** (`tools/f15_permutations.py equivalence`, every arm, build = ab0c133 + the archive of that commit, RELATIVE_TOLERANCE of the run 1e-5, the criterion now committed as 0):
154 cases (119 scenes): class bins vs all-lobes **152 bit-identical** (0 px, 0 ULP, 8-bit 0); repeated reference arm 7 of 7 bit-identical; overflow bin (6 runs) 0 ULP; app manifest export 0 failures;
negative control (every lobe dropped) leaves the envelope in 87 of 142 controls, every feature bit 0..8 discriminated by 6..26 scenes (the not-discriminated list is the known resolve-colour-neutral one).
**Two cases fail: `corpus:ClearCoatCarPaint` (323 px, 8-bit 237 px, max delta 15) and `corpus:ToyCar` (282 px, 8-bit 77 px, max delta 2)**, both texture-transform scenes. See the non-determinism study below.

**Non-determinism study (ToyCar class vs all-lobes, each run its own lock acquisition, ref and cls in separate processes; image hashes of the linear capture):**
- exact module, archive of 8d305b0 (x4): run 0 ref e09f8ccd / cls 8c6d6655 (186 px, a triangle at x 337..351, y 228..233); run 1 ref 8c6d6655 / cls 9fe2763e (526 px, a block at x 602..636, y 153..).
  The same few images appear in either arm: the variation is NOT class-vs-all-lobes.  Earlier, the first x1 run 542 px on the single scanline y = 265, x 600..680; the same binaries rerun gave 0.
- exact module, final archive (x5): run 0 cls cce33d66 differs from ref 8c6d6655 (520 px, rows 154.., x 474..513); run 1 both 8c6d6655.  The canonical image 8c6d6655 is the common output.
- exact module with `--no-pipeline-archive` (every pipeline JIT): 2 of 2 runs both arms 8c6d6655 (0 px).
- integrated tip build (BASE module, orchestrator `build/release`): 4 of 4 runs deterministic (ref 7f73553d, cls 40a288e0; they differ by the known 19343 px of the unscoped maths), no scanline.
- logs of a variant and a canonical run are the same in pipelines: all 8 frames binned, the same 4 archive misses (Forward v7, indexed overflow fallback, material_lighting_guides, DI receiver guides: the texture-transform
  archive gap of ticket 22), no fallback draws.
Reading: the variation needs archive-served pipelines in a texture-transform scene and is not caused by the class/all-lobes choice; with every pipeline JIT-compiled the output is stable and class = all-lobes.  Root cause NOT identified
(candidates not yet tested: the 4 JIT-compiled non-exact pipelines of the TT scenes feeding the resolve, an archive-vs-JIT codegen difference of an exact kernel, a GPU-driven culling order/ties).  It is not in the resolve dispatch
or in a missing barrier that I found: both arms use the same dispatch and the variation appears in the ref arm too.
Runs still queued when this was written: 20 runs of the exact build, 6 no-archive runs, 20 base runs (flake study files /private/tmp/claude-501/bx/flake_*.out).

**Budget run (full, 154 cases) of the exact build:** measured-frame fallback draws 0, render-thread compiles 0, compiler calls 0 in every case; whole-run archive misses 1 in every case = `phx_exact::visibility_lit_resolve` of the unbinned
all-lobes union pipeline that the first archive prune had dropped (scenes whose union is the full lobe set, and the lit generic dispatch).  Fixed in the follow-up archive (+35 descriptors from the harvests, 143 MB, metal-tt 19.9 s median of 3
on the loaded machine); re-run of the budget on it is pending.
**Featureless regression, archive_check, hitch_check** on the same build as above: 5 exactness cases cmp-identical, visual_check 0 px on 8 benches, archive_check all pass (full: 18 hits, 0 misses), hitch_check 0 hitches (21 hits, 0 misses).

**Flake study, counts (ToyCar class vs all-lobes, 640x360, 8 frames, `--pipeline-sync`, each run its own lock acquisition; killed after these):**
- exact build with the committed archive: **8 of 15 runs differ** (155..1139 px; the differing image is the class arm in some runs and the REFERENCE arm in others, e.g. run 11 ref 6eee5273, run 12 ref 8a546fa7); the other 7 runs are bit-identical to the canonical image 8c6d6655 in both arms.
- exact build with `--no-pipeline-archive`: **0 of 6** differ, both arms always 8c6d6655.
- integrated tip (base module, archive): **0 of 15** differ (always ref 7f73553d / cls 40a288e0: the known 19343 px of unscoped maths).
- Not found in the base build, so it is not a pre-existing nondeterminism of the resolve readback or dispatch (no scanline appears there in 15 runs).  It appears only with archive-served exact-module pipelines in a texture-transform scene
  (ToyCar, ClearCoatCarPaint; the other 152 cases, 119 scenes incl. non-TT corpus scenes, are exact with the same archive).  Open hypotheses to test next: (1) an archive-built exact kernel differs from its JIT build for the
  TEXTURE_TRANSFORM (26) specialisation (function constants live in namespace phx_exact; an archive entry that does not carry the constant would still run); (2) a data race that only the archived kernels' timing exposes (the
  extension records of the per-slot texture transforms are read through `frame.materialExtensions`).  Suggested next experiment: archive-built vs JIT-built single kernel with GPU capture, or a build whose archive omits only the TT exact entries.

**Archive-without-texture-transform experiment (12 ToyCar runs each, same method):** final archive (x6): **8 of 12** runs differ (160..1481 px, either arm); the same build with the 12 texture-transform exact entries removed from the
archive (so those kernels are JIT-compiled; x7): **4 of 12** differ (186..2210 px; runs 0, 1, 10, 11, the quiet middle runs 2..9 were all canonical).  So the TEXTURE_TRANSFORM archive entries are not the (only) cause, and
the rate follows the machine load (other agents' jobs ran in parallel) rather than the pipeline set.  Together with: base module 0 of 15, all-JIT exact module 0 of 6 and the shape of the variation (several rows of ~40 px wide blocks at
varying places, either arm, varying image hashes) this points to a TIMING-DEPENDENT hazard that the slower exact resolve (and the faster archived start) exposes in the feature-scene frame, not to arithmetic: the arithmetic is exact (152 of 154
cases, and the canonical image is reproduced by both arms in every quiet run).  Next step (not done): search the frame graph of the feature path for a resolve consumer or a bin/classify producer without the stage barrier the
Metal 4 no-hazard-tracking rule requires (shaders/visibility_resolve.metal bins buffer written by visibility_classify, `visibility_renderer.cpp` dispatch order of the class bins, the passes reading the resolve targets), by forcing
`--gpu-timing-serial` / a serialising barrier after every resolve dispatch: if the variation vanishes the hazard is confirmed.

`--gpu-timing-serial` (x6, 10 ToyCar runs): 7 of 10 differ (370..1569 px): that switch does not remove the variation, so the hazard hypothesis above is still unconfirmed (that flag serialises timed units, it is not a barrier after every resolve dispatch).

### Masked-defect hypothesis (orchestrator), 2026-11-15, branch f15c-bitexact-b

- (a) NaN / Inf / garbage in the HDR captures of the 12 final-archive ToyCar runs (differing and identical): none.  Every image of both arms has 0 NaN, 0 Inf, 0 negative values and the same maximum (4923.9); the differing
  pixels are ordinary values (relative differences 1e-8 .. 1e-1), not neutralised garbage.
- (c) Texture-transform data: the host writes every word the shader can read.  `material_extension_pack.cpp:161` builds the block from a value-initialised `std::array<GPUTextureTransformRecord, MATERIAL_TT_RECORDS>` and copies ALL
  MATERIAL_TT_RECORDS records (`:170`), `scene_store.cpp:389` stores all of them and `:431` zero-fills a freed block, so no `reserved` word, unused slot or tail record is uninitialised.  The shader reads a transform only when the material
  flag is set and the slot's `slotMask` bit is set (`texture_transform_core.h` texXfForOn / texXfHasOn / texXfGet; texXfFrame, sampleSlot); a material without the block has `rec == nullptr, mask == 0`.  I found no read of an
  unwritten or out-of-block word.
- (b) shader validation with `--no-pipeline-archive` on the exact module: NOT run (battery 35 %, owner rule).  Pending.

### Rebased tip (24563ce) + exact module: results (2026-11-16, branch f15c-bitexact-b, 038eb10)

**The ToyCar / ClearCoatCarPaint variation does not occur on the integrated tip.**  Official `tools/f15_permutations.py equivalence` (every arm, RELATIVE_TOLERANCE 0.0), run TWICE on two archive states:
200 cases (162 scenes) all bit-identical (0 px, 0 ULP, 8-bit 0), failures [], repeated reference 7 of 7, overflow bin 12 runs 0 ULP, app manifest export 0 failures, negative control leaves the envelope in 125 of 185 controls with every
feature bit 0..8 discriminated by 16..34 scenes.  ToyCar, ClearCoatCarPaint and all nine `layers-blk-*` keys, `layers-blk-all-e1` included, are exact.  Flake study on the rebased build, archive on, `--pipeline-sync`, each run its own lock:
ToyCar 0 of 12 differ, and 0 of 12 with a forced Dispatch->Dispatch device barrier after every bin dispatch (experiment, reverted); with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 --no-pipeline-archive` ToyCar 0 of 6 and ClearCoatCarPaint 0 of 6 differ and the logs
hold no validation message (check (b)).  Before the rebase (78d173d base) the same study gave 8 of 15 / 8 of 12 / 4 of 12 differing runs.  The cls arm now has 0 archive misses on ToyCar (was 4).
What changed between the two bases for these scenes: texxform-3 archives the per-slot-transform pipelines (the 4 Forward v7 / guide misses), texxform-layers and the flag-collision fix (24563ce: MATERIAL_FLAG_LAYER_TRANSFORM bit 2 collided with
MATERIAL_FLAG_UNLIT).  I did not bisect which of them removed the variation, so the old nondeterminism is NOT root-caused; it is not reproducible on the integrated tip in 24 + 12 runs and two full equivalence runs.
(a) no NaN/Inf in the old differing captures, (c) the host initialises every transform-block word: neither explains it.

**Archive / budget on the final build (038eb10):** full budget run, 200 cases: measured-frame fallback draws 0, render-thread compiles 0, compiler calls 0 in every case; 197 cases 0 archive misses; the three cases with misses miss ONLY pipelines of the
base module (ToyCar: indexed overflow fallback, Forward v7, DI guides, material_lighting_guides = the texture-transform gap owned by the texxform agent; aniso-bent-env: indexed overflow fallback forward_surface_fs; terrain/layerblocks=on/single=2/coat=off#base:
Forward v0, DI guides, the five denoise_*_lobe kernels, overflow fallback).  No exact-module pipeline misses anywhere.  The all-lobes fallback of the generic dispatch had to be archived too (serve() asks for it while the specialised pipeline is not final):
requestBoth for the generic dispatch only, +6 descriptors.
**Featureless / archive checks on the same build:** 5 exactness cases cmp-identical, visual_check 0 px on 8 benches, archive_check full 18 hits / 0 misses, hitch_check 0 hitches 21 hits 0 misses; generic-vs-variant forward 9/1/5/9/25 px unchanged (module untouched).
**Earlier harvests missed every scene whose name contains `/`** (nested output folders: the terrain / layerblocks fixtures); the final update uses `find` and includes them.

### Bisection of the old ToyCar variation (2026-11-16, timeboxed; ToyCar class vs all-lobes, 20 runs each, each run its own lock acquisition, battery 55 %)

- Old build still flaky on a quiet machine: the exact module on the old base 78d173d (snapshot x6, archive of that branch), rerun now: **11 of 20** runs differ (205..1303 px).  So the variation is not machine load; it is in the old build.
- (1) integrated tip + exact module with the texxform-3 archive entries REMOVED (9 archive descriptors; the cls arm then shows the old 4 misses and the ref arm 8, exactly the old flaky build's numbers: 138 hits / 4 misses and 135 / 8): **0 of 20** differ.
- (2) integrated tip + exact module with MATERIAL_FLAG_LAYER_TRANSFORM moved back to bit 2 (the collision with UNLIT that 24563ce fixes; static assert removed for the experiment, reverted): **0 of 20** differ.
  (The collision cannot be tested on the old base: MATERIAL_FLAG_LAYER_TRANSFORM does not exist there, texxform-layers added it.)
- Neither the archive entries nor the flag fix removes anything, and the old miss/hit pattern alone does not cause the variation (it is stable at the tip with it).  So the cause is another change in 78d173d..24563ce (11,654 lines in shaders and
  src/renderer).  Not bisected further: a source bisection needs one full rebuild plus archive harvest per step.
- Leading candidate, untested: texxform-layers (7fcbb3d) added `GPUTemporalParams::materialExtensions` and made every alpha site read the extension buffer from it ("the extension buffer for the alpha tests of layer stacks and per-slot transforms").
  On the old base the per-slot-transform alpha tests (`visibility_alpha_tt_fs`, `visibility_alpha_lit_tt_fs`, meshlet.metal:800/817) read `frame.materialExtensions`, and the old visibility_renderer.cpp never set an extension pointer of its own; if that pointer is stale or
  unset in a pass the alpha test reads another buffer's words, the visibility buffer differs at the triangles of the alpha-tested, transformed materials (the differing regions are triangle-shaped blocks, in either arm) and the result changes from run to run
  with the contents of that memory.  ToyCar and ClearCoatCarPaint are the corpus scenes with per-slot transforms next to alpha-tested or textured parts.  That also explains why the class-vs-all-lobes comparison, which shares the visibility pass,
  could not be the cause.  To confirm: build 7fcbb3d^ and 7fcbb3d, 20 ToyCar runs each.

### Frame-constants hypothesis (orchestrator), 2026-11-16, tip 91dcd80: refuted by reading, no GPU run needed

- At the tip the per-slot-transform alpha tests already read `frame.materialExtensions` (shaders/meshlet.metal:818 `visibility_alpha_tt_fs`, :837 `visibility_alpha_lit_tt_fs`: `texXfForOn(frame.materialExtensions, ...)`); the old base read it the same way (lines 807 / 825).
  `GPUTemporalParams::materialExtensions` feeds only `visibilityStackAlpha` (meshlet.metal:766, called from the alpha kernels), which texxform-layers added.  "Pointing the TT alpha tests back to frame.materialExtensions" is therefore a no-op: there is nothing to revert, and the old flake cannot have been a stale temporal pointer in the TT alpha test.
- Is `FrameConstants::materialExtensions` itself stale in a pass?  `SceneRenderer::prepareFrame` copies the constants AFTER `buffers_.stageFrame` (the only place the extension buffer is (re)allocated) and writes `frameConstantsData.materialExtensions = materialExtensionsAddress()` into the frame ring
  (scene_renderer.cpp:216-220); the other readers take `scene.materialExtensionsAddress()` at encode time of the same frame (reflection_passes.cpp, atmosphere_passes.cpp, visibility_renderer.cpp:185) or `constants.materialExtensions` set right after prepareFrame (engine.cpp:1659).
  A frame in flight keeps its own copy of the address and the old buffer is released through GpuMemory's deferred release, so I find no path where a pass sees an address from another frame.  ToyCar and ClearCoatCarPaint never grow the buffer (static scenes), so a mid-run reallocation cannot explain a run-to-run variation there either.
  No fixture added: there is no defect to fail before / pass after.  The old-base variation stays unexplained; the remaining candidates are the other changes of 78d173d..24563ce (a source bisection at 7fcbb3d^ / 7fcbb3d, 20 ToyCar runs each, is the next step if the owner wants it).

### 2026-11-16 round 3: the old ToyCar / ClearCoatCarPaint variation, reproduced and localised (PARTIAL: producer of the bad record not found, tip immune)

Old state = branch `f15c-bitexact-a` (78d173d + the first exact-maths module, archive of that branch).  Two builds: the frozen snapshot x6 of the bitexact-b worktree and an own rebuild of `0208fc0` (`build/old`, same canonical image hash).  Method: one `phosphor` per capture (ToyCar, corpus rig, `--pipeline-sync`, 1024x768, 8 frames, `cls` arm only), several captures per lock acquisition, images hashed.

**Reproduced.**  Rate per capture on a quiet GPU: HDR 5 of 20 differ (x6, 25 %, one arm; the two arms of the old study gave 55 %); `specular-filtered` signal 5 of 14 (x6) and 8 of 20 (own rebuild); `--frames-in-flight 1` 4 of 20; `--graph-no-alias` 6 of 20; untiled reference a-trous (`PHOSPHOR_DIAGNOSTIC_DENOISE_REFERENCE=1`) 6 of 20.

| point (ToyCar cls arm, one capture per run) | differing / runs |
|---|---|
| old state + exact module (x6), HDR | 5 / 20 |
| old state + exact module, `specular-filtered`, frame 7 | 5 / 14, 8 / 20 (rebuild), 6 / 20 (no alias), 4 / 20 (1 frame in flight) |
| same old tree WITHOUT the exact module (`snap/base`), `specular-filtered` | 0 / 20 |
| old + exact, `--no-pipeline-archive` | 0 / 14 |
| old + exact, `--reflections ssr` (no RT reflections) | 0 / 20 (`--shadows csm` still 11 / 20) |
| integrated tip e0f6ea0, HDR | 0 / 20; with `--debug-async-compute` 0 / 16, `--debug-split-encoding` 0 / 16, `--gpu-timing-serial` 0 / 16, `--debug-lighting 1` 0 / 60 |

A source bisection of 78d173d..24563ce was NOT done: the exact module exists only on the old branch and after fa6786d, the variation needs its archive (0 / 14 without it), so every bisect point needs a rebuild plus a harvest; the point table above is a runtime bisection of the old binary instead.

**Where it is (all numbers on the old state; each signal captured by its own runs).**  Stable in every run: raw specular (6 / 6 at frame 7, 8 / 8 at frames 1, 3, 5), shadow 8 / 8, direct 6 / 6, reactive 16 / 16, shadow position and normal 16 / 16 each, the DI surfaces and the lobe guide (`--debug-lobe-guide-dump`: identical bytes in 20 of 20 runs, 7 of them with a differing image, CPU check PASS in every frame).  Varying: the filtered specular, and the F13 specular temporal output (5 of 20 runs).  The variation is always ONE row segment of exactly 32 pixels aligned to 32 (temporal output; the 44 x 13 patches of the final image are that segment dilated by the two a-trous iterations: 32 + 2 x 6 wide, 1 + 2 x 6 tall).  Its values are the raw samples: the history of those 32 pixels was rejected in one frame.  With the failing-test mask captured per pixel (debug build of the old tree) the rejected test is `stochastic != history.STOCHASTIC` (frame 1: history length 1 instead of 2, 5 of 30 runs), i.e. the sample flags of those 32 pixels were not what the previous frame stored.  The sample flags of frame 0 itself differ in 2 of 40 runs (flags 0 instead of 0x9 for a 32-aligned group).

**The record itself is wrong.**  Reading the reduced hit-metadata buffer (`GPUSpecularSample`, 48 B per pixel) on the CPU after the frame completes (debug build, buffer made shared): in 23..35 of 480 frame checks per 60 runs a whole aligned 32-pixel group holds `flags 0, path 0, secondarySlot ~0, hitDistance 0, direction 0, pdf 0` while `radiance` is non-zero for the pixels that reflect something (the last good record before the group has flags 0x9).  That shape is `reflectionReduceEnd` of a sample without VALID, or an `empty` record with the radiance words written: the 32 records are not what the reflection kernel computes for a valid surface.  It is not the consumer reading early (the final buffer is wrong), not the clear pass (`reflection_signal_zero` writing the metadata removed: 28 / 480 events, 19 of 320 with a record dump; with the extra encoder barrier after it: 28 / 480), not the frame ring (1 frame in flight: still 4 / 20), not transient aliasing (no-alias 6 / 20), not the a-trous tiles, not the lobe guide, not archive entries (earlier rounds).  It needs the RT reflection kernel (SSR 0 / 20), the exact module in the old tree (0 / 20 without) and the archived pipelines (0 / 14 without), i.e. it depends on timing of the old frame, as the earlier hazard hypothesis said.  I did not find which store writes the zero flags (the graph, its barriers and the binding code were read; the clear, the encoder barrier and the frame ring are excluded by experiment).

**Tip.**  e0f6ea0 has the same graph and barrier set for these passes (`--dump-graph` of both) and never produces the record: 0 of 480 frame checks (60 runs, `--debug-lighting 1`), 60 / 60 identical images, plus the 3 scheduling probes above.  So the tip is immune at the data level, not masked by the denoiser, but I cannot show it is immune by construction; the cause is a property of the old frame (timing / code of the old kernels), not a commit I could name.  The earlier refuted candidates (archive entries, flag collision, frame constants) stay refuted.

**Guard added (the check that fails if it comes back).**  `ReflectionPasses::check` (only under `--debug-lighting N`, the hit-metadata buffer is allocated shared only then) fails the lighting check and the run (exit 1) when a reduced record has radiance but no flags.  The same code on the old tree fails 7 of 40 runs (exit 1, "REFLECTION metadata check slot N: K records with radiance but no flags"); on the tip 0 of 60 (EXIT 0, LIGHTING check PASS every frame).  Normal runs are untouched.
Verification of the commit: `f15_exact.sh`: the 5 PFMs `cmp`-identical to `exact-final-ms`; `visual_check` (`--pipeline-sync`, vs `reference-f15-final`): all 8 benches 0 px; `ctest`: 45 / 45 pass (terrain_materials_verify skipped as before); `f15_permutations.py equivalence` on `corpus:ToyCar` and `corpus:ClearCoatCarPaint` (tip): 0 px, 0 ULP, 8-bit 0, repeated reference 0, controls leave the envelope (101 618 and 173 067 px).
Open: the writer of the zeroed flags in the old state (next step: a debug copy of the reflection kernel that stores a per-group marker word, or the metadata buffer double-buffered per frame slot, to see whether the late store is the previous frame's or this frame's); the old nondeterminism remains UNEXPLAINED by construction.

### 2026-11-16 — owner decision on the old ToyCar variation (orchestrator)

Declared **localised, not root-caused** (owner option a): reproduced 5-8 of 20 runs on 78d173d + the first exact module; signature: whole aligned 32-pixel
groups (one SIMD group) of the reflection `GPUSpecularSample` record end with flags / path / slot / pdf 0 but non-zero radiance (23-35 of 480 frame checks in
the final buffer); needs RT reflections, the exact module and the archived (metal-tt) pipelines (0 of 14 with `--no-pipeline-archive`); excluded: the clear
pass, an extra barrier, the frame ring, transient aliasing, the a-trous tiles, the lobe guide, the archive entries; the writing store is not found.  The
tip never produces the record (0 of 480 frame checks, 60 / 60 identical images).  Guard kept: `ReflectionPasses::check` fails the run under
`--debug-lighting N` for a reduced record with radiance but no flags (old tree 7 / 40 runs fail, tip 0 / 60); the final battery's f15_check captures run
with `--debug-lighting 1`.  Follow-up (named, not started): a debug copy of the reflection kernel that stores a per-group marker word to find the store.
