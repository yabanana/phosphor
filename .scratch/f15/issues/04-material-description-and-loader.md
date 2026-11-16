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
