# 23: Layered materials (layer stack) on a terrain fixture

Roadmap: F15.4

**What to build:** Up to four layers per pixel from an RGBA weight mask with optional endpoint-exact height contrast (metres), parameter-space blending with one BSDF evaluation, surface-gradient normals, weights normalised to sum 1, layer presence in the feature class; a procedural terrain patch with four pinned CC0 materials and a cavity coat finish; final binning decision on chess + terrain.

**Blocked by:** 21

**Status:** partial (2026-11-14): everything except the resolve GPU time per active-layer count and the final binning decision is built and verified on the M5; those two need a quiet machine (ticket 24); see the verification record

- [x] Doctest seam 3: weight normalisation, endpoint exactness, monotonic transitions, no NaN with all-zero weights
- [x] Each layer at weight 1 renders exactly like the single material (0 pixels, visibility path); weight 0 leaves the others unchanged
- [x] Coverage and feature layers distinct: a coat finish changes coat weight only, never the base F0 below
- [x] Transition ramp monotonic; furnace ≤ 1 + 0.5 % for blended BSDFs; additive-blend negative control exceeds it; deviation from a BSDF-space mix recorded
- [ ] Resolve GPU time at 1080p per active-layer count (1–4) vs single material, A/B/A, 3 replicas, quiet machine; final binning decision (≥ 5 % rule on chess + terrain) in opt-log
- [x] CC0 fetch pinned with hashes; sparse top-K masks recorded as an option only
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Verification record (2026-11-14, M5 Max, worktree agent-a5228d3ced304e941; commits `32692d2` .. the record commit)

**Status: partial.** Everything except the resolve GPU time per active-layer count and the final binning decision is built and verified; those two need a quiet
machine and belong to the orchestrator (ticket 24). `tools/f15_layers_timing.py` is the prepared measurement (plan / run / summarize, A/B/A, 3 replicas, refuses to
start while another GPU process runs); it was NOT run for a number (it exits 4 cleanly on a held lock; the script was only checked with `plan`).

**Design.**
- `src/renderer/layer_blend.h` (shared C++/MSL): weights clamped (NaN = 0), unused layers 0, normalised to sum 1, all zero -> layer 0 (declared rule, no NaN); optional height
  contrast `w'_i = n_i exp((h_i - hmax)/S) / sum` with S = 5 mm, `h_i = heightContrast_i * (height texel - 0.5)` in metres, hmax over the layers with n_i > 0 (exact at n_i = 0 and at a
  single weight 1); linear parameter blend in layer order (exact at e_j: 0 * p adds an exact zero); surface-gradient normal blend (Mikkelsen 2020: gradients `(tx/tz, ty/tz)` add; at e_j
  the layer's raw tangent-space normal is returned, so the caller's arithmetic is the single material's).
- `shaders/material_shading.h`: `resolveLayerStack` runs only when the scene feature union has bit 8 (function constant) and the material carries it; it samples the mask at the primitive uv,
  then per layer with non-zero weight the layer material's base colour, metallic-roughness, occlusion and normal (and the height texture) at `uv * uvScale` with the single-material
  expressions, blends, and hands base colour / roughness / metallic / occlusion / normal to the ONE existing BSDF evaluation. Layers contribute base parameters only; their feature
  blocks, emissive and alpha are not blended (features belong to the stack's own description).
- Description / record: `MaterialLayer` gains `uvScale` (> 0) and `heightTexture` (file keys `uvScale`, `heightTexture`; description version unchanged, both optional); record 3 of
  ADR-0002 keeps its layout (`uvScale[4]`, `heightTex[4]` take padding words); a material without a stack packs the pre-F15.4 bytes, so snapshots and the reference cache of every other
  scene stay valid (ADR-0002 addendum). Layers are ordinary library materials; the stack material's own base record is the appearance seen by secondary hits (declared limit: RT
  reflections, GI and probes do not see the blend).
- Class machinery (ticket 21): fixture family `terrain` in `class_manifest` (`shaders/material_classes/terrain.json`, one `manifest` line in `shaders/material_classes.def`): classes
  {0, layers 0x100, layers + coat 0x102, layers + specular 0x101}; budgets kept (41 archived permutations <= 64, at most 2 bins in a terrain scene). Targeted archive update with
  `tools/f15_layers_archive.py` (exact-key harvest, label-independent, additive): +41 compute, +9 render, +3 tile descriptors, nothing removed (every old descriptor is still present);
  `material_class_archive.py check` passes (80 plan pipelines). metal-tt median of 3 on the loaded machine: 18.8 s / 137.9 MB (ticket 21 end: 16.2 s / 126.4 MB), i.e. this ticket adds
  53 of about 1,380 descriptors (+4 %); the +25 % build-time budget is judged on that share, the machine was not quiet.
- Fixture (`TerrainFixture`, bench 2 `--material-scenario terrain[/opts]`, `src/testbench/terrain_fixture.*`, data in `src/scene/terrain_textures.*`): 8 m heightfield patch (128 x 128
  quads), deterministic generated RGBA mask (slope / height / noise rules, seed 23), four layers (grass, rock, ground, snow; tilings 16/8/16/8, powers of two), height contrast 3/6/4/2 cm,
  cavity coat finish (clearcoat on the stack's own description, factor texture = cavity map), sun 3 W/m2 + environment 0.25. Textures: deterministic procedural stand-ins by default;
  `--terrain-textures DIR` loads the owner-approved CC0 ambientCG sets (Grass004, Rock030, Ground037, Snow006, 1K JPG, `<DIR>/<Asset>/<Asset>_1K-JPG_{Color,NormalGL,Roughness,
  AmbientOcclusion,Displacement}.jpg`; JPG only, nothing else is opened, a missing map is an error). Options: `weights=`, `layers=`, `single=N` (the single-material twin: layer N alone,
  mesh uvs x its tiling), `height=`, `coat=on|off|zero|spec[/ior=]`, `ramp=on|reverse`, `furnace=on`. `python3 -I tools/fetch_terrain_materials.py verify DIR` (ctest
  `terrain_materials_verify`, env `TERRAIN_MATERIALS_DIR`, skips when absent) checks the sets against `tools/terrain_materials.json`: OK on `build/terrain-materials`.
- Runner: `tools/f15_check.py` kinds `layer-equal`, `layer-ramp`, `layer-furnace` (statistics and the blended-BSDF quadrature in `tools/f15_layers.py`, 14 unit tests); keys `layers-*`
  (26 procedural, 8 CC0 that are skipped when no CC0 directory exists).

**Verification (all on the M5 Max, serial under the shared lock).**
- Seam 3 (`tests/test_layer_blend.cpp`, 10 cases, about 105k assertions): normalisation (sum 1, clamps, unused layers); all-zero / negative / NaN / infinite inputs never give a NaN (layer 0);
  endpoint exactness bit for bit with and without the height contrast and with a tall absent layer; a worked height example (heights S ln 3 apart -> 0.75 / 0.25); 200 random monotone ramps
  (weights, height-contrast weights, parameters); parameter blend exact at e_j; surface-gradient worked example, raw normal at the endpoint, linearity, degenerate texel; record packing
  (the coat finish changes record 0 only, record 3 byte-identical; a stackless material packs the old record 3); description file round trip and validation.
  `tests/test_terrain_fixture.cpp` (6 cases): options, determinism, tilings, patch frames, the feature classes of every scenario. ctest 30 of 30 (with the Khronos corpus and the CC0 set present).
- Weight 1 = single material (visibility path, mesh geometry, 1280x720, linear HDR, frame 127): layers 0..3 procedural AND layers 0..3 CC0: 0 of 921,600 pixels differ bitwise (max 0 ULP, 0 in the
  8-bit image) in all 8 keys, with the height contrast on, normal maps on and the generated tilings; the controls (the stack against the single material of the NEXT layer) differ in 197,325 of
  921,600 pixels (max relative 26). Weight 0 leaves the others unchanged: weights (1,1,0,0) on a 4-layer stack equal the 2-layer stack, 0 of 921,600 pixels; the control ((1,0,0,0) against the 2-layer
  stack) differs. Criterion met; no ULP issue appeared.
- Coat finish (feature layer) changes the coat weight only: the record test above plus the capture `layers-coat-zero` (clearcoat block present, cavity map black) against the same stack under a
  specular block of the same F0 (same extended code path, no coat): 95,053 of 921,600 pixels differ bitwise, median 1e-7 relative, worst 3.0e-6 absolute at a pixel of 0.033 (floored relative error
  1.03e-4), 0 pixels differ in the 8-bit image. The gate is a DECLARED envelope of 5e-4 floored relative error (set after this first measurement, on the runner's own floor rule 0.05 x mean; the
  cause is float noise through other composite/denoise code for pixels carrying the clearcoat guide bit, not the base F0). The controls fail it by orders of magnitude: the real cavity coat (floored
  4.3, 68,060 pixels in the 8-bit image) and a base F0 changed by the ior (1.4: F0 0.040 -> 0.028; floored 0.278, 170,275 pixels in the 8-bit image). Against the plain uncoated stack (base code
  path) the same capture differs in 120,454 pixels (floored 1.0e-4, 1 pixel in the 8-bit image): reported, not gated.
- Transition ramp (flat quad, two plain layers, mask R = 1 - u, G = u, sun only, reflections off): the blend factor (L - L0)/(L1 - L0) against the two single twins over 242,160 pixels in 424 rows:
  0 rows with a reversal (worst drop 0), max |t - u| 0.0020 (limit 0.02), fitted slope 1.0039; the reversed ramp (control) reverses in all 424 rows (slope -1.004).
- Furnace (flat quad at a grazing view, environment 1, four albedo-1 layers metallic 0/1/0/1, roughness .1/.4/.7/1, bilinear mask, 402,278 pixels): engine max pixel 0.99997 (limit 1.005), mean 0.8236;
  the engine's declared model (ADR-0006) gives a blended maximum of 0.99996 and the engine matches it with mean relative 0.24 %, p99 1.17 %. The additive blend (the sum of the layers' BSDFs, weights
  ignored) reaches 3.65 (the control fails the gate). Deviation of the parameter-space blend from the BSDF-space mix `sum w_i E_i` (declared model): blend minus mix is positive everywhere, max 0.0859,
  mean 0.0508 (min 0.0022); with the literal Appendix B quadrature (tools/f15_reference.py) 0.0827 / 0.0490 / 0.0010 and a blended maximum of 1.175: the LITERAL glTF model with a metallic-blended F0
  is not energy bounded at intermediate metallic, which the ADR-0006 diffuse weighting of the engine model removes (that maximum is not what the resolve evaluates and is not gated).
- Pipelines: budget run (8 terrain cases, warmup 64, 32 measured frames, archive, no `--pipeline-sync`): 0 fallback draws, 0 render-thread compiles, 0 compiler calls, 0 archive misses and 0 GPU
  allocations in the measured frames, 2 class bins; `tools/archive_check.sh build/release <reference-f15-appB>`: all scenarios pass; `tools/hitch_check.sh`: 0 hitches (SWITCH n=15);
  `tools/visual_check.sh` (validation on, BENCH8_ARGS=--instances 100000, references `build/reference-f15-appB`): 0 of 5,760,000 pixels on 8 of 8 benches.
- Class equivalence (`f15_permutations.py equivalence`, 8 terrain cases, 640x360, 8 frames, class bins against the all-lobes resolve): 4 bit-identical, the others at most 4 ULP except
  `terrain/coat=off` and `terrain/coat=spec` (the textured full patch, layers and layers + specular): 655 ULP, 6.8e-5 relative, 0 pixels in the 8-bit image. The declared 1e-5 envelope of ticket 21
  (set on untextured fixtures) is exceeded on those two; reported as ticket 21 reported its textured scenes (chess/corpus), not re-declared. The all-lobes arm repeats bit for bit (8 of 8); the
  negative control (lobes dropped from the class pipelines) leaves the envelope in all 6 scenes with features; feature bit 8 is discriminated by 6 scenes.
- Full runner: `tools/f15_check.py --run --expect green --reference-cache build/f15-check/t19int` over ALL 141 keys (124 evaluated, 17 capture-only; 108 cached references reused, 0 re-rendered, the 12 `layers-*` gates included, 8 of them on the CC0 sets): GREEN, exit 0, every key met its expectation and every negative control failed its gate (commit `203c060`, clean tree, 124 keys, about 1 h 15 min with the GPU queue shared). The record-3 packing fix (unused words stay 0) is what kept the reference cache valid: the first full run, before it, found every extension scene's snapshot digest changed and was stopped.

**Deviations and honest limits.**
- The coat-zero gate is an envelope declared after a first measurement (above); the criterion text has no bound. The weight-1 and weight-0 gates are strictly 0 pixels.
- Two textured terrain cases exceed the ticket-21 equivalence envelope (6.8e-5), reported above.
- The literal Appendix B quadrature of the blended parameters exceeds 1 (1.175); only the engine model is bounded (see the furnace).
- Layer materials contribute base parameters only (no layer features, emissive, alpha); the stack material's own record serves secondary hits (RT reflections, GI and probes see one average
  appearance, not the blend). The mask uses the primitive uv (no triplanar or per-layer uv rotation); tilings are per-layer scalars. Top-K sparse masks (more than 4 layers per pixel, pruned to the
  4 strongest and renormalised, per tile) are recorded as an OPTION only: the decision-05 baseline is a hard 4 and nothing here implements pruning.
- The repeat sampler blends the opposite texel in the outermost half texel of the mask, so the ramp and furnace gates exclude a 3-texel margin at the plane edges.

**Open.** (1) Resolve GPU time at 1080p per active-layer count (1-4) and for the generated mix against the single material, A/B/A with 3 replicas on a quiet machine, and (2) the final binning
decision (>= 5 % rule on chess + terrain) in opt-log: both need the orchestrator (ticket 24); command `python3 -I tools/f15_layers_timing.py run --out build/f15-layers-timing --binning`; the opt-log
entry is written with the decision left open. No timing claim is made in this record.
