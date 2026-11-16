# 04: Material description, glTF extension parsing and support report

Roadmap: F15.1, F15.3

**What to build:** A versioned, portable material description with one optional block per material feature and a bounded layer stack; the glTF loader parses every F15 extension (and texture transforms, emissive strength, required extensions) into it and prints a support report; a versioned material-description file can define materials or override a glTF's materials by name. Rendering does not change: every feature is reported `unsupported` until its ticket lands.

**Blocked by:** None (can start immediately)

**Status:** resolved (verified 2026-11-13, ddca1ae..4727b0f; ctest 15/15, visual_check 8/8 0 px) with open criterion (2026-11-16, round 2: final report schema number closed at 12 (pinned by doctests; unlit adds a value inside the same section); per-slot KHR_texture_transform implemented by owner decision (analysis None/Uniform/PerSlot/Unsupported; a second UV set per material stays Unsupported and out of scope); KHR_materials_unlit and KHR_lights_punctual are rendered (e8b6399) and kept as an approved scope addition overriding docs/plans/F15.md line 22, unlit now Implemented together with COLOR_0, punctual spot falloff is the engine smoothstep, declared Approximated (`spot_falloff_smoothstep_not_informative_t2`) and measured equal on the legacy, ReSTIR and clustered paths (RT reflection lighting, probes and GI not captured separately); still open: Sponza identity (criterion 7) is evidenced by the F13 exact captures of tickets 05-08, not by visual_check, which covers procedural benches only)

- [x] Doctest seam 1: for each extension (clearcoat, sheen, anisotropy, transmission, volume, ior, specular, iridescence, emissive_strength, texture_transform, diffuse_transmission RC, draft KHR_materials_scatter at PR head a6cd5b4) parameters, defaults, ranges, texture channel and colour space match the specification text cited in research/khr-materials.md
- [x] Uniform UV0 transform baked into the primitive's UV0 (unit test with an independent worked example); a non-uniform transform reported unsupported for that material
- [x] An asset whose `extensionsRequired` lists an unsupported extension is refused with an error naming it (glTF 2.0 §3.2: SHOULD NOT be loaded); optional unknown extensions listed, the asset loads
- [x] `MATERIALS` log line and report-JSON section (schema bump) with implemented / approximated / unsupported per material and feature; draft/RC inputs labelled
- [x] Material-description file load/save round-trip and name-keyed override over a glTF, with validation errors for out-of-range values
- [x] Image URIs resolve only inside the asset's directory (no absolute paths, `..` or symlink escape)
- [ ] Sponza loads with an empty feature set and identical images (visual_check 0 px) — PARTIAL (audit 2026-11-15): visual_check covers procedural benches only; Sponza identity = F13 exact sponza-d2 0 px in tickets 05-08

## Comments

- 2026-11-13 (orchestrator): code review (standards + spec axes, fresh
  reviewers) of tickets 01/02/04; every finding fixed by a Sonnet implementer
  (12 commits, integrated 15:00) or closed in ticket 03 (shared-module
  duplicates). Verified on M5 after integration: build, ctest 15/15
  (phosphor_tests 768 cases), F13-path exact captures byte-identical to the
  post-ticket-03 baseline, visual_check 8/8 at 0 px vs
  `build/reference-f15-appB`, `tools/f15_check.py --run --expect green` →
  GREEN on 7 scenarios incl. `base-grid-env-resolve` and
  `roughness-extremes-sun`; declared-model gates: worst sphere mean 0.009 %
  (bound 0.5 %), worst p99 0.037 % (bound 1 %), the three declared controls
  (diffuse unweighted, Karis fit, F0-blended diffuse weight) fail. Captures
  `build/f15-check/review`.

2026-11-15 — closure audit (ticket 24):

- 1 doctest seam 1 per extension: MET (one TEST_CASE per extension in test_gltf_materials.cpp incl. scatter at a6cd5b4). Ticked.
- 2 uniform UV0 bake: MET (worked example; non-uniform unsupported, declared). Ticked.
- 3 extensionsRequired refused / optional listed: MET (1aa6b89). Ticked.
- 4 MATERIALS line + report JSON: MET (afd92fb schema 11, 4727b0f schema 12; later tickets bump again, final schema to be recorded in 24). Ticked.
- 5 description round-trip, name-keyed override: MET (e9b6423; engine application resolved in ticket 05, 8913c6f). Ticked.
- 6 image URI containment: MET (c6d0ec5; symlink subcase returns silently if the filesystem refuses symlinks, execution not stated). Ticked.
- 7 Sponza empty feature set: PARTIAL (visual_check 8/8 0 px procedural only; Sponza identity via F13 exact captures in 05-08). Open.

2026-11-15 — per-slot KHR_texture_transform (closure agent texxform; owner decision: implement the non-uniform transform, ticket 24 carry item "per-slot texture transform"):

- Analysis (`analyzeTextureTransforms`, material_description.cpp): status None / Uniform (every present texture shares one transform: baked into
  UV0 exactly as before) / PerSlot (slots differ, or transformed and plain slots mix: evaluated on the GPU) / Unsupported (named `reason`).
  `uvSet` = the glTF texCoord every present texture of the material samples.  Layer mask/height textures have no GPU transform slot
  (`textureTransformSlot` returns ~0): a transform or UV set on them is Unsupported.
- Loader: the UV set a material samples (usually 0) is copied into UV0 when it is another one (TEXCOORD_1 of TextureTransformMultiTest);
  Uniform transforms are baked into that set; PerSlot leaves UV0 untouched.  One UV stream per mesh is the stated limit: a material whose
  textures sample DIFFERENT sets is Unsupported ("different UV sets"), refused when KHR_texture_transform is required, reported in the
  support report otherwise.  What two sets per material would take: GPUVertex 48 -> 56 B (uv1; +17 % vertex bytes for such meshes), the
  meshlet cooker and vertex pulling of the visibility resolve / guide kernels / RT hit interpolation reading uv1, 2 bits per slot in the
  transform block (spare words of record 0) and `sampleSlot` choosing the set; no change to the material record.
- GPU layout (gpu_types.h): `MATERIAL_FLAG_TEXTURE_TRANSFORM` (flags bit 1), `GPUMaterial::pad1` -> `textureTransformIndex`, 20 slots
  `TT_SLOT_*`, `GPUTextureTransform` (2x2 + offset, 24 B), `GPUTextureTransformRecord` (3 transforms + slot mask).  A transform block is two
  adjacent extension groups (8 records) owned by the SceneStore (alloc on first need, kept zeroed when the transforms go away, reused through a
  free list, `verifyAgainstEcs` checks flags, index, records, ownership and the free list).  Scenes without transforms allocate nothing.
- `SceneStore::textureTransformInUse()` / `pipelineFeatureUnion()` carry the feature to the pipelines as the reserved function constant
  TEXTURE_TRANSFORM (26); `featureUnion()` and every lobe-bit consumer are unchanged.
- Support table: `textureTransformPerSlot` Implemented; the report line reads "KHR_texture_transform (per slot)".
- Tests: test_texture_transform.cpp (pack against the README formula, 20-slot map, store allocate / flag / free / reuse / verify, marker, pipeline
  constant), test_gltf_loader.cpp (UV-set copy, uniform bake on set 1, mixed sets), test_gltf_materials.cpp (required extension, per-slot rendered,
  mixed sets refused), test_material_description.cpp (analysis).

## F15 closure, loader-misc (2026-11-15)

- **Final report schema number: 12.**  `BENCH_REPORT_SCHEMA_VERSION == 12` (src/diagnostics/bench_report.h; 11 -> 12 in 4727b0f, the `materials` section is `phosphor.materials.v2`).  No later ticket bumped it
  (tickets 05-23 only add fields to the materials/report sections that already exist), and this closure does not either: unlit adds one more `encodings` entry ("unlit"), a value inside the same section.
  Doctests: tests/test_bench_report.cpp "F15 bench report: schema 12 carries the materials support section only when present" pins the number (`CHECK(BENCH_REPORT_SCHEMA_VERSION == 12)` and the
  document's `schema_version`), tests/test_launch_options.cpp compares the constant on four report paths, tests/test_gltf_materials.cpp "support report: summary line, detail and JSON" pins
  `"schema": "phosphor.materials.v2"`.  Ticket-04 box 4 ("schema bump") is therefore closed at 12.
- **KHR_materials_unlit and KHR_lights_punctual are rendered** (they were "reported unsupported, a model that requires them is refused", decision 02; docs/plans/F15.md line 22 scoped them out and the
  closure brief overrides it).  Unlit: `lowerUnlit` (emission = base colour, no reflection, features dropped, `MATERIAL_FLAG_UNLIT`), encoding "unlit" Approximated `no_vertex_color_stream`
  (the specification multiplies the base colour by COLOR_0 when present; the engine has no COLOR_0 stream for ANY material, so lit materials ignore it too; adding it touches GPUVertex, the meshlet
  and mesh paths and every shader that fetches a vertex: not small, not done).  Lights: `src/scene/gltf_lights.*` and the loader's light import (directional lux and point/spot candela used as
  W/m2 and W/sr, factor 1, `GltfLoader::setPhotometricToEngine(1/683)` for watts; infinite range = 1e5 m; negative intensity and invalid cones clamped with a warning).  The spot cone falloff is the engine's
  (smoothstep of cos in the forward, resolve and GI paths, LINEAR in ReSTIR's `restir_common.h`), not the informative formula of the specification (t^2); the extension leaves the blend between the cones
  implementation-defined.  The ReSTIR / resolve difference is an older inconsistency, reported to the orchestrator (owner decision: unify on one falloff changes every existing spot scene).
  Commit e8b6399.  Fixtures: tests/test_gltf_unlit_lights.cpp (CPU) and tools/f15_unlit_lights_check.py (GPU), see ticket 22.

### Follow-up (2026-11-15, orchestrator review): spot falloff measured on every DI path
My earlier statement that ReSTIR's falloff is linear came from reading `restir_common.h` and was WRONG: it computes `spot*spot*(3-2*spot)` = smoothstep (the evaluators of ReSTIR, clustered DI, reflection secondary lighting and probes are that function; legacy/forward/overlay and GI call the built-in `smoothstep`).
Measured (tools/f15_spot_falloff_check.py, one spot 0.3/0.6 rad, ratio to the on-axis panel, analytic smoothstep(cosOuter, cosInner) x geometry): legacy, ReSTIR and clustered all within 0.7-2.2 % at 0.38 / 0.5 / 0.55 rad, outside the cone 0.00000; the linear falloff misses by 10.6 / 12.4 / 47 % and
the informative t^2 by 39 / 118 / 153 %, so both negative controls fail as required.  No path differs: nothing to fix.  RT reflection secondary lighting, probes and GI were not captured separately (shared evaluators, argued from the code).  The deviation from the specification's informative t^2 is declared in the
support table (`punctualLights`, Approximated, `spot_falloff_smoothstep_not_informative_t2`, logged by the loader when a spot is imported; doctest).

## 2026-11-15 — COLOR_0 support level

`KHR_materials_unlit` is now `SupportLevel::Implemented` (the vertex colour factor is applied); COLOR_0 loading, stream and sites: see ticket 22, same date.

## 2026-11-16 round 3 — spot falloff measured at the secondary lighting sites (branch f15c-color0-spot)

The 2026-11-15 statement "RT reflection secondary lighting, probes and GI were not captured separately (shared evaluators, argued from the code)" is replaced by measurements
(`tools/f15_spot_falloff_check.py arms-run|arms-verify|selftest`, doc string of the tool).  Design: default glTF camera (no rig: lights are ignored under `--corpus-rig`), a small
panel P (front facing -Z, back culled so the camera never sees it) at (0,3,-2.5), the spot at (0,3,-4) pointing +Z (D = 1.5 m), the LIGHT NODE TILTED about X by 0 / 0.38 / 0.45 / 0.5 / 0.55 rad
(0.85 = outside) so only the angle to the axis changes (geometry term identical, it cancels), a wall at z = -5 the spot never lights.  DELTA (light on - intensity 0, linear PFM,
mean RGB) of a patch divided by the delta at tilt 0, against the independent model (mean over the panel of cos(panel)/d^2 x falloff; smoothstep = engine convention; linear and the
informative t^2 as controls).  Thresholds fixed before the first comparison: in-cone tilts within 7 % (relative) of the smoothstep model, outside ratio <= 1 %, both controls must miss
a tilt by > 7 % (`selftest`: model control miss linear 68.7 % / t^2 59.6 % mirror arms, 21.5 % / 53.3 % gi arms, all above 2 x 7 %).

| arm | site | measured ratios 0.38 / 0.45 / 0.5 / 0.55 | smoothstep model | worst error | verdict |
|---|---|---|---|---|---|
| rtrefl | RT reflection secondary lighting (mirror wall, `--reflections rt`) | 0.8820 / 0.6127 / 0.3601 / 0.1300 | 0.8819 / 0.6093 / 0.3541 / 0.1223 | 6.24 % (tilt 0.55), 0.01 / 0.56 / 1.67 % before | PASS; outside 0.00000; linear misses 11.4 / 6.6 / 9.8 / 37 %, t^2 40-163 % |
| probe | probe capture lighting (roughness 0.05 wall, `--reflections probes --reflection-capture-probe`) | 0.8658 / 0.6021 / 0.3600 / 0.1423 | same | 1.83 / 1.19 / 1.66 % up to 0.5, **16.27 % at 0.55** | PARTIAL: formula equal to 1.8 % where the signal is large; at the steep tail (ratio 0.12) the cube-map footprint / prefilter blurs the sliver (not separated from a real difference) |
| gi (DDGI) | GI hit shading, probe grid | 0.9610 / 0.8323 / 0.6541 / 0.4284 | 0.8251 / 0.5722 / 0.3576 / 0.1791 | 139 % | FAIL as a measurement: the response is wider than ANY falloff (also than linear 0.77 / 0.56 / 0.38 / 0.22); linear in intensity (x0.05 gives the same ratios), so not a saturation; the DDGI atlas + trilinear probe interpolation weights the panel with its own kernel, which the form-factor model does not know.  Not a proof that the GI falloff differs |
| girestir | GI hit shading, `--gi restir` (path sampling, no grid kernel) | 0.8155 / 0.5552 / 0.3327 / 0.1440 | same as gi | 1.17 / 2.96 / 6.97 / 19.6 % | PARTIAL: 1-3 % at the two large-signal tilts, drifts low at the tail (delta at tilt 0 only 0.005, ReSTIR GI noise not separated) |

Code facts that support equality (not a measurement): reflections.metal, reflection_probes.metal and the sampled-light branch of gi_common.h call `diSampleLight` (restir_common.h smoothstep) with the
same cone constants; the punctual branch of gi_common.h calls the built-in `smoothstep(cos(outer), cos(inner), cos)` (same function as legacy).  No code path with a different falloff was found, so no fix was made: nothing differs
measurably where the measurement is clean (rtrefl, probe up to tilt 0.5, girestir up to tilt 0.45).
Open: the probe tail (0.55) and the two GI arms; proposed follow-up: an emissive-pattern equivalence for GI (the same wall response for an emissive panel whose texture encodes the analytic lit pattern, which cancels the probe kernel).
Model note: the first rtrefl run used the central patch half-size 0.05 m (7.44 % at tilt 0.55, over the 7 % gate); the footprint of the 9 x 9 pixel patch at the mirror image is 0.054 m (4.5 px x 2 tan 30 deg / 720 x 7.5 m), derived from the
camera, not from the result; the tolerance was not changed.

**GI, kernel-agnostic follow-up (same day)**: `giemit-run` / `giemit-verify` (tool above).  The spot-lit panel (A, DDGI, GI only) is compared with an EMISSIVE panel of the same size and albedo whose 48 x 48 sRGB texture is the analytic lit radiance of the panel at that tilt (B; control B' = linear falloff), both through the same GI; the GI part of B is gi on - gi off (removes the direct emitter lighting), so the probe-grid kernel cancels.  Threshold fixed beforehand: B / A within 7 % of 1, control must miss.
Measured (wall ratio to tilt 0, tilts 0.38 / 0.45 / 0.5 / 0.55): A 0.961 / 0.832 / 0.654 / 0.428; B (smoothstep pattern) 0.830 / 0.584 / 0.371 / 0.185 = the form-factor model 0.825 / 0.572 / 0.358 / 0.179 within 1-3 %; B' (linear pattern) 0.775 / 0.560 / 0.386 / 0.231.  So the DDGI probe kernel reproduces the analytic pattern for an emissive panel, and the SPOT-LIT panel through the same DDGI is clearly WIDER than the analytic smoothstep (B / A 0.86 / 0.70 / 0.57 / 0.43, worst 56.7 %, FAIL; control 46 %).  The ReSTIR GI arm (no probe grid) matches the model (1.2 / 3.0 %), so the shading function `giSecondaryRadiance` is not the suspect by reading; the difference is specific to how DDGI stores a LIT hit (not found in the time box; candidates: the quantised radiance cache / hit binning of lit hits that an emission does not use, hysteresis on the changed light).  An albedo of 0.8 on B (multi-bounce at the panel) gives identical B ratios, a x0.05 intensity gives identical A ratios (not saturation, not multi-bounce).
Status: the DDGI GI spot response is an OPEN MEASURED DEFECT CANDIDATE (not fixed, not explained), reproduce with `tools/f15_spot_falloff_check.py arms-run OUT BIN --arms gi` and `giemit-run`.  Because the probe field smooths spatially, its effect on a real scene is a softer spot edge in DDGI indirect light only; direct lighting (legacy, ReSTIR, clustered) and the RT reflection arm are unaffected.
