# 04: Material description, glTF extension parsing and support report

Roadmap: F15.1, F15.3

**What to build:** A versioned, portable material description with one optional block per material feature and a bounded layer stack; the glTF loader parses every F15 extension (and texture transforms, emissive strength, required extensions) into it and prints a support report; a versioned material-description file can define materials or override a glTF's materials by name. Rendering does not change: every feature is reported `unsupported` until its ticket lands.

**Blocked by:** None (can start immediately)

**Status:** resolved (verified 2026-11-13, ddca1ae..4727b0f; ctest 15/15, visual_check 8/8 0 px) with open criterion: Sponza identity is evidenced by the F13 exact captures of tickets 05-08, not by visual_check (procedural benches only)

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
