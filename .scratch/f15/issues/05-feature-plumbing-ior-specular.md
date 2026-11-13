# 05: Feature plumbing proven by ior and specular (resolve and forward)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_ior` and `KHR_materials_specular` render for analytic lights, sun disk and environment in the resolve and forward paths through the new chain: extension records via SceneStore deltas with the layout of EVERY F15 feature frozen now (static_asserts), feature bits, the feature mask as a reserved function constant specialising forward and resolve per scene, and runtime edits and feature toggles.

**Blocked by:** 03, 04

**Status:** ready-for-agent

- [ ] Extension records (layouts for all features frozen in `gpu_types.h`, `GPUMaterial::pad` retyped to `u32` with the same size) reach the GPU only through SceneStore deltas; `--debug-gpu-scene` and `verifyAgainstEcs` cover the extension buffer, with a corruption control that must FAIL
- [ ] Reserved constant semantics fixed: undefined ⇒ no extension lobes (scenes without features keep byte-identical pipeline descriptors and archive keys, visual_check 0 px); the compile fallback uses an explicit all-lobes value; ADR-0003 wording matches
- [ ] Extension at default values renders exactly like the base material (0 px)
- [ ] ior/specular grids pass decision-11 direct-light thresholds (ior = 0 mode, f90 = specular weight); furnace bound holds; negative control (extension ignored) fails
- [ ] Parameter edits change the next frame; turning a feature on/off at runtime is a counted structure change with fixed per-material record capacity, the all-lobes fallback serves until the specialisation compiles, and a hitch test toggling a feature shows 0 GPU allocations in measured frames (O7/O9)
- [ ] Extension-record deltas bump every material-revision consumer (`gi_passes.cpp:91`, `reflection_passes.cpp:254`, `shadow_passes.cpp:298`, `direct_lighting_passes.cpp:108`, `engine.cpp:1718`, `acceleration_structures.cpp:620`); indirect/denoised signals recover within the F13.6 recovery criterion after an edit
- [ ] Adaptive 2×2 shading reuse excludes materials with feature bits; offline-reference snapshots versioned and extended with extension records
- [ ] Sponza D2 within the decision-11 band (A/B/A)
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
