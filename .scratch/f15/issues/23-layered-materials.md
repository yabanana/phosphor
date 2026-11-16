# 23: Layered materials (layer stack) on a terrain fixture

Roadmap: F15.4

**What to build:** Up to four layers per pixel from an RGBA weight mask with optional endpoint-exact height contrast (metres), parameter-space blending with one BSDF evaluation, surface-gradient normals, weights normalised to sum 1, layer presence in the feature class; a procedural terrain patch with four pinned CC0 materials and a cavity coat finish; final binning decision on chess + terrain.

**Blocked by:** 21

**Status:** partial (2026-11-14); (2026-11-16, round 2: the writer "layers" closed items 1, 2, 3 and 5 of its brief: layer feature blocks (coat, sheen, anisotropy, iridescence, specular, emissive, thin subsurface) are blended and gated (stack at weight 0/1 equals the single material, 0 of 921 600 px; ramps monotone, controls fail), transmission, volume and volumetric subsurface are NOT blendable by declaration; secondary hits (reflection, probe, GI) see the stack (`layers-hit-*`, 0 px); the mask edge uses clamp_to_edge (edge max |t - u| 0.4999 -> 0.0021); the coat-zero gate is exactly 0 px (was 59 945 px); coverage alpha is blended at every alpha site (`stack-*` fixtures 1.0, stack as emitter fixed) and per-slot texture transforms work inside stacks; `layers-blk-all-e1` (2 ULP, previously the only pending key) is bit-exact on the integrated tip with the exact module (ticket 21, 2026-11-16); triplanar is deferred to F19.2 by declaration; criterion 5: terrain binning timing recorded 2026-11-15 (5cd4bf3, binning never lowers the resolve by 5 %) and chess binning A/B/A recorded in ticket 24 (side-light -1.8 %, shadow rig +6.7 %, back-light +9.8 %), so binning stays opt-in and union the default, but the resolve cost per layer count with the new blocks (`--binning` arms `blocks-1..4`) and the secondary-hit change timing on D2 are pending the orchestrator's final battery; the 1e-5 envelope overshoot on textured terrain is attributed to the height contrast gain and declared, not re-declared as an envelope; the checkbox line 5 of this ticket's audit still says chess A/B/A was not run, ticket 24's later record supersedes it)

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

## Integration (orchestrator, 2026-11-14)

Merged onto cl/f15-plan-c38b51 as cfa5e73 (after the F13 residue fixes). Conflicts: pipelines archive = union of both sides
(`tools/merge_pipeline_harvests.py`, no descriptor removed; 871/143/279/32 + ticket 23 -> 912/143/288/35), tests/CMakeLists.txt keeps
both new tests. Re-verified on the merged tree: ctest 32/32 (KHRONOS_CORPUS_DIR set, CC0 set present); `f15_check --only layers-*`
GREEN (12 evaluated keys, every control fails; `build/f15-check/cfa5e73-layers`); F13 exact captures 0 px on 5 cases against
`build/f15-exact/base-residues`; `visual_check` 0 px on 8 benches; `archive_check` all scenarios pass; `hitch_check` 0 hitches.
Carried to ticket 24: resolve GPU time per active-layer count and the binning decision (quiet machine, `tools/f15_layers_timing.py`).

2026-11-15 — closure audit (ticket 24):

- 1 doctest seam 3: MET (10 cases ~105k assertions). 2 weight 1 = single material: MET (8 keys, 0 of 921 600 px). Ticked.
- 3 coverage vs feature layers: MET with declared envelope (gate 5e-4 floored, set after the first measurement; 0 px in 8-bit). Ticked.
- 4 ramp, furnace, additive control: MET (furnace 0.99997, additive 3.65; blend minus BSDF mix max 0.0859). Ticked.
- 5 resolve time per layer count + final binning decision: PARTIAL (terrain A/B/A x3 on 2026-11-15: union single 0.089 ms, layers 1..4 0.136/0.162/0.183/0.208 ms; binning never lowers the resolve by 5 % (full +1 %, layers-4 +4 %); ran on procedural stand-in textures, not the CC0 sets; chess A/B/A not run; nothing written to opt-log/perf-log). Open.
- 6 CC0 fetch pinned: MET. 7 archive/hitch/O7: MET (+41 compute +9 render +3 tile; integration re-verified). Ticked.
Open: layers contribute base parameters only, feature blocks/emissive/alpha not blended (deviation from plan wording); secondary hits/GI/probes see the stack's own record; mask uses primitive uv; the 1e-5 envelope exceeded on two textured terrain cases.

## F15 closure (writer "layers", 2026-11-15): feature blocks, emissive, secondary hits, mask edge, exact coat zero

Commits `a1a2548` (item 3), `a5bf1d9` (items 1 and 5, fixtures, runner, archive), `64074eb` (item 2). Everything below ran on the M5 Max under the shared lock; no timing
claim is made (other agents ran in parallel).

**Item 3, mask edge: DONE.** The mask uses a `clamp_to_edge` sampler (`kMaskSampler`). The ramp gate gates the edge band (u < 1.5/256 and u > 1 - 1.5/256) on its own and
the 3-texel exclusion is gone (`EDGE_MARGIN` 0 for ramp and furnace). BEFORE (base build + the new gate): edge max |t - u| 0.4999 over 2,978 edge pixels, 412 of 424 rows
reverse. AFTER: 0.0021, 0 reversals, slope 1.0039; the reversed ramp (control) still fails; furnace without the margin 1.00000 (limit 1.005, additive 3.46).

**Item 1, features blended: DONE except the combination of all seven blocks (2 ULP).** Rules in `src/renderer/layer_blend.h` header and ADR-0002: per feature a layer's source is
its own block, else the stack's own block (the finish it inherits), else neutral; presence-like scalars (coat factor, sheen colour, anisotropy strength, film strength, specular
factor/colour, thin subsurface weight, emission) blend linearly with the final weights; the secondary parameters of a lobe (coat/sheen roughness, film ior/thickness, anisotropy
direction, subsurface colour, the coat normal by surface gradients) are weighted by w_i x presence_i; ior 0 is a mode that wins where at least half of the weight asks for it; a
sole layer (weight exactly 1) returns its own values bit for bit. Alpha: see below. Emissive: sum of w_i emissive_i x texel_i, the stack's own emissive is an additive finish.
The stack's flags carry the layers' blendable bits (record 3 `features` word), a lobe whose blended weight is 0 at the pixel is off.
- Not blendable, with reasons (declared; the SceneStore logs the dropped blocks once): transmission and volume (the refraction, the transmittance shadow/ray passes and the
  guide read ONE record per surface at traced hits, and a refracting layer has no parameter-space mean: interface and absorption are path-dependent), volumetric subsurface
  (a screen-space diffusion pass over a target keyed by the material). The layer still contributes the rest of its material.
- Gates per block (flat quad, layer 1 carries the block, checker factor textures, layer tilings 1 and 4): stack at constant mask (0,1) equals the single material of layer 1,
  0 of 921,600 px (coat, sheen, aniso, irid, spec, emissive, thin); stack at (1,0) equals the plain single layer 0, 0 px (the twin runs in the same feature union through an
  unused library material); the negative controls (the same layer without its block = the old behaviour, and the other layer) all fail; transition ramps: pooled blend factor
  monotone (0 reversals) and, for the blocks affine in the blended parameter (coat, sheen, irid, emissive, thin), max |t - u| 0.002 (limit 0.02); aniso and spec are gated
  monotone only (their luminance is not affine in the parameter: reported). The pooled estimator replaces the per-pixel ratio for blocks because the effect of a block on the
  luminance is tiny and changes sign across a textured layer (sheen conserves energy: L1 - L0 ~ 0.2 %); the thresholds (0.001 drop, 0.02 mask error) are unchanged.
- Furnace (base blend, four layers) 1.00000 with the additive control at 3.46; the deviation from the BSDF-space mix is the one recorded before (max 0.0859). A per-block furnace
  is not added: every block's energy rule is the one of its own ticket (coat attenuation, sheen albedo scaling, film replacing the Fresnel), applied to blended parameters.
- OPEN: all seven blocks on one layer (`layers-blk-all-e1`) differ from the single material by 2 ULP in 39,045 px (0 in the 8-bit image); each block alone is exact. Cause not
  isolated (FMA contraction of the many accumulations in the fused path; the emissive block alone was exactly this: the product feeds `color +=` and had to be one expression
  for both paths). The key is `pending` in the runner (asserted to fail until closed); the zero-weight and ramp keys of `all` pass.
- Alpha: NOT blended (PARTIAL). Coverage is decided from the stack's own record in about 15 independent passes (visibility alpha fragments, forward, CSM and cached shadow
  depth, RT any-hit intersection functions, ReSTIR emitter alpha, the probe capture, CPU light sampling and the RT checks), most of them with only `materials` and `textures` bound.
  Blending it in the resolve alone would disagree with those passes (holes without shadows). It is feasible, not impossible: one shared `layerStackAlpha` (mask + height weights
  x layer alpha) plus the extension pointer in GPUTemporalParams (visibility and forward), the shadow params and the RT entries (buffer 7 already holds it). Proposal for the owner:
  schedule it as its own ticket with an alpha-cutout ramp fixture and RT shadow / reflection coverage gates; until then a stack's coverage is its own alpha record.

**Item 2, secondary hits: DONE for base parameters, normal and emission.** `layerStackSurface` (one helper, `shaders/layer_resolve.h` + `gi_common.h`) feeds reflectionMaterial,
giMaterial (DDGI and ReSTIR GI hits) and the raster probe capture; features are not evaluated at secondary hits for any material (the documented reduction of ADR-0002).
Gates `layers-hit-{refl,probe,gi}`: stack at (0,1) equals the single material of layer 1 in 0 px in all three modes, (1,0) equals layer 0 (0 px), controls (the stack's own record,
the other layer) fail. Sampled emitters: a layer's emission is not an emitter for the DI/GI emitter sampling (declared; the stack's own emissive is).

**Item 4, triplanar: NOT REQUIRED in F15.4, deferred to F19.2.** Re-searched `docs/plans/F15.md` (F15.4 section), this ticket, `.scratch/f15/decisions/`, `docs/adr/`,
`.scratch/f15/research/layered-materials.md`: none requires triplanar; the mask is an RGBA weight mask at the primitive uv and the Mikkelsen surface gradients are used for the
per-layer uv frames. `research/track-studio-audit.md` (lines 325-327) says triplanar "va decisa come caso F15.4 oppure rinviata a F19 dichiarandolo": it is declared deferred to
F19.2 (ROADMAP "Texturing del terreno con virtual texture e materiali a strati"), where heightfield walls need it.

**Item 5, threshold: DONE, the gate is 0.** A coat that resolves to weight 0 switches the coat off at the pixel (`activeBits`), so the pixel takes the code path of a pixel without a
coat. `layers-coat-zero` now compares `terrain/coat=zero` with `terrain/coat=union` (no coat on the stack; an unused library material carries a coat so that the feature union and
the pipeline are the same): BEFORE 59,945 of 921,600 px differ bitwise (max 24 ULP); AFTER 0 px; controls (real cavity coat, the ior-1.4 twin) fail. The old twin (specular block
factor 1 - 2^-24) is reported only. The 1e-5 class-equivalence overshoot (655 ULP on the textured patch) is NOT the coat: `terrain/coat=off/height=off` measures 31 ULP (3.7e-6,
inside 1e-5) against 654 ULP with the height contrast on; the height blend is exp((h - hmax)/5 mm), a gain of ~200 on a 1-ULP codegen difference between the class and the
all-lobes pipeline. Declared as inherent to the height contrast; not re-declared as an envelope.

**Archive and regressions.** New permutations (unions 0x9F, 0x104, 0x108, 0x110, 0x180, 0x19F) entered by the targeted update (`.scratch/f15/harvest/layers-1.json`,
`tools/material_class_archive.py check` passes). Featureless scenes unchanged: Sponza D2 and the F13 exactness captures bit-identical to `build/litefix/exact-new`; visual_check
8 of 8 benches 0 px and the 5 exactness captures IDENTICAL on the final binary (also Sponza D2 and bench 6 on the secondary-hit variant). A first restructuring of the secondary-hit code moved Sponza D2 by 5,768 px: reverted to verbatim original
statements plus an override.

**Needs a quiet machine (orchestrator):** resolve cost at 1080p by active layer count with the new blocks on the CC0 terrain, A/B/A, 3 replicas:
`python3 -I tools/f15_layers_timing.py run --out build/f15-layers-timing --binning` (new arms `blocks-1..4` = `terrain/layerblocks=on/weights=...`, plus the old arms); and
Sponza D2 / RT reflection kernel timing after the secondary-hit change.

2026-11-15 (texxform agent, follow-up on the integrated tip 4e89658): layer-stack coverage alpha and per-slot transforms inside stacks.

- **Coverage alpha** = the stack's own record alpha x the blended layer alpha (`resolveLayerWeights`, `layerStackAlpha`, `layerCoverage` in `shaders/layer_resolve.h`: the weights are those of the base blend, a_i = layer i's baseColor[3] x
  its base colour texture alpha at uv * uvScale through the layer's own transform), tested against the STACK's cutoff, at every alpha site of `$H/layers-alpha-notes.md`: `material_shading.h` (before the alpha test:
  forward, resolve, guide kernels, indexed fallback), `meshlet.metal` (the four visibility alpha entries), `forward.metal` (the receiver prepasses), `shadows.metal` (CSM), `shadow_cache.metal`, `rt_intersections.metal`
  (any-hit, both entries: RT shadows, reflection rays, GI rays), `reflection_probes.metal` (probe capture).  Plumbing: `GPUTemporalParams` carries the extension-buffer pointer (offset 184, size 192, filled in
  `visibility_renderer.cpp`); the other sites already bound the extension buffer (ticket 22).  CPU mirrors: `rt_check.cpp` reports a stack's alpha decision Ambiguous (the CPU reference has no extension records) instead of
  deciding from the stack's own record; `rt_proxy.cpp` protection already tests the stack's own cutoff.  Emitter alpha of a stack (`restir_common.h`): the stack's own record (declared optional in the notes).
- **Fixtures** (`tools/f15_texture_transform_layers.py`, glTF + a material-overrides file: the override loader now resolves a layer's material NAME to the library index): a ramp mask R = u, G = 1 - u, layer 0 alpha 1, layer 1
  alpha 0, stack MASK cutoff 0.5: the coverage boundary sits at u = 0.5.  stack-visibility 1.0, stack-rt-shadow 1.0, stack-csm-shadow 1.0, stack-cache-shadow 1.0, stack-reflection 0.9975 (thresholds 0.97, set before the
  run); NEGATIVE controls: the stack's OWN alpha (coverage 1 everywhere) 0.5 / 0.65 / 0.65 / 0.65 / 0.50 and the inverted ramp 0.0 / 0.0 / 0.0 / 0.0 / 0.0025, all < 0.90: they fail as required.
- **Per-slot transforms inside stacks** (ticket 22 owner decision): `TT_SLOT_LAYER_MASK` (the mask through the stack's block), `TT_SLOT_LAYER_HEIGHT0 + i` (layer i's height on uv * uvScale through the stack's block), every texture of layer
  i (base colour, metal-rough, occlusion, normal with the texel axes rotated into the mesh frame, emissive, and the feature textures of `layerFeaturesOf`) through the block of the layer's SUB-material on uv * uvScale.
  A stack with any transform is PerSlot and never baked; a layer sub-material that is only UNIFORM still gets its block (`SceneStore::layerTargets`: nothing bakes a layer).  A transform on a layer texture is no
  longer Unsupported.  The transform block grew to 3 extension groups (25 slots).  Fixtures: layer-tex (layer 1 base colour palette through a per-slot transform, decoy transform on a second slot, weight 1) cell agreement
  1.0, controls (identity, uniform-bake = other slots' transforms) 0.09-0.15; layer-mask (halves mask mirrored by the mask's own transform) and layer-height (halves height swapped by the height's transform) measure R - G
  left - right -0.68 / -0.68 with the reference's sign (model -1), the identity predicts +1 and the controls disagree.
- Still declared: the stack's emission-as-emitter alpha uses the stack's own record; a layer sub-material used BOTH directly by a primitive and as a layer would apply a uniform transform twice (baked + block): not authored anywhere.

2026-11-15 (texxform agent, second part): the two declared limits are closed.

- **Stack as emitter** (`restir_common.h` diSampleTexturedLight, CPU `light_sampling` mirror with an optional coverage callback): an emissive stack's emitter alpha is its own record alpha x the blended layer coverage (LOD0,
  `layerCoverage`), tested against the stack's cutoff.  Fixture `stack-emitter` (`tools/f15_texture_transform_layers.py`): an emissive stack (strength 40, ramp mask, cutoff 0.5) lights the floor in front of it from the u >= 0.5
  half only; the gate is the floor luminance left - right (threshold 0.08) against a form-factor sum over the emitting half (model -0.556): AFTER -0.1142 (reference class), BEFORE (binary of adca5c8) -0.0013 (the own-alpha class: the whole
  quad emitted); controls own-alpha and inverted ramp disagree.  Doctest: the stack's own record never cuts, the ramp coverage cuts the u < 0.5 half.
- **Gates of the final build** (`5c775d8` + this section): the 116 `layers-*` keys in four chunks under the lock (`f15_check.py --run --expect green --reference-cache t19int`, references whose snapshot digest changed re-rendered by
  Mitsuba): all four chunks GREEN (exit 0) - see the ticket 22 record for the per-chunk numbers and the one chunk that had to be re-run (the CC0 sets must be reachable at build/terrain-materials; a symlink to the orchestrator's unpacked sets was
  missing in the first attempt).
