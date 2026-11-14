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
