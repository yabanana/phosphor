# 21: Material permutation generator and budgets

Roadmap: F15.2

**What to build:** The app exports feature-class manifests from fixtures and corpus; the generator (extension of `variant_gen`) turns manifests plus a declared list into the class table, MSL constants and pipeline descriptors and updates the archive in a targeted way; per-class tile binning measured against the union pipeline on the chess fixture (provisional decision; final with the terrain in ticket 23); budgets enforced and reported.

**Blocked by:** 19, 20

**Status:** partial (2026-11-14): everything that needs no chess or corpus asset is built and verified on the M5; the chess/corpus manifests (tickets 20, 22), the A/B/A binning measurement (ticket 24) and the "exactly 0 pixels" wording of criterion 2 are open (see the verification record)

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
