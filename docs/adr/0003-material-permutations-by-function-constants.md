# Material permutations are function-constant specialisations of one source

F15.2 asks for an offline generator of MSL permutations without an
uber-shader (O11). Phosphor generates the permutation set offline from material
descriptions (feature classes found in fixture and corpus manifests plus a
declared list) and specialises the hand-written lobe code with Metal function
constants keyed by feature class; it does not emit textual MSL per
permutation. Function constants remove dead lobes at compile time, and the
existing PipelineCache, AOT archive harvest, flexible fallback and exact
resolve-equivalence checks apply unchanged. The feature mask is a reserved
constant: undefined means "no extension lobes", so scenes without features keep
the measured generic resolve (FC 21 undefined) with byte-identical descriptors;
extended scenes use their union-specialised pipeline; an explicit all-lobes
value serves only as reference and compile fallback.

Implemented in F15.1 (ticket 05): the constant is `MATERIAL_FEATURES`, a
`reserved` uint at function-constant index 22 in `shaders/variants.def` (not an
axis: the dense forward table of 42 variants and `tools/variant_check.sh` are
unchanged). Its MSL value `kMaterialFeatures` is 0 when the constant is
undefined; its bits are `1 << MaterialFeature` (`MATERIAL_FEATURE_*` in
`gpu_types.h`). The all-lobes value is `MATERIAL_FEATURES_ALL` (every bit, 0x1FF).
Forward, meshlet-forward and visibility-resolve pipelines request constant 22 =
`SceneStore::featureUnion()` only when it is non-zero (`withMaterialFeatures`);
for union 0 the descriptor is returned unchanged, byte for byte.

## Consequences

- A permutation outside the generated set runs the all-lobes reference pipeline
  (not the generic one, which has no extension lobes; logged) until its
  specialisation compiles; budgets (≤ 64 archived permutations, ≤ 16
  active classes per scene, +≤ 25 % archive build time, 0 hitches) are tracked.
- Measured deviation (ticket 05, 2026-11-13): the all-lobes pipeline serves
  only once it has an object. Render pipelines get the cache's flexible
  fallback object while they compile, compute pipelines (the visibility
  resolve) get none; when neither the specialisation nor the all-lobes
  pipeline is available, the scene's base pipeline serves those frames
  (extension lobes absent, counted as a fallback use;
  `FeaturePipelines::serve`). Binding the pending compute handle crashed the
  driver (null pipeline state). Transient after a load or a feature toggle;
  the toggle test shows the image settles at the next archived or compiled
  specialisation.
- Codegen of a specialised pipeline is not guaranteed bit-identical for
  materials without feature bits: Sponza with one overridden material, forward
  0 px outside that material, visibility resolve 14 of 5.76 M px at 1 LSB
  (2026-11-13). Scenes without features keep byte-identical descriptors and
  images (the guarantee above); within extended scenes the decision-11 gates
  apply.

## F15.2 (ticket 21): generator, class table and per-class bins

- **Inputs.** A class is the feature mask of one material. Feature-class manifests (`phosphor.material-class-manifest` v1, strict schema,
  `src/renderer/material_class_manifest.h`) list per scene the classes with their material counts and the scene union. The F15 fixtures export
  GPU-free through `class_manifest export shaders/material_classes`; `phosphor --export-class-manifest FILE` exports the scene the app runs (chess
  fixture, Khronos corpus). The declared list is `shaders/material_classes.def`: budgets, the resolve kernels, the manifests, extra `declare`d classes.
  Appending an input is one `manifest` or `declare` line.
- **Generator.** `variant_gen classes` writes the class table (C++), the MSL constants and `material_class_plan.json` (every pipeline the table
  implies: kernel + function constants) into `build/generated/pipeline/`. Two runs are byte-identical; a malformed manifest, an unknown directive or more
  than 64 distinct `MATERIAL_FEATURES` values (class masks, scene unions, the all-lobes mask) fail the generation. No text MSL is emitted per
  permutation: the permutation is still `MATERIAL_FEATURES = mask` on the hand-written lobe code.
- **Archive.** `tools/material_class_archive.py update` adds exactly the plan to `shaders/pipelines.mtl4-json` from per-scene harvests
  (`tools/f15_permutations.py harvest`), with the label-independent identity of `filter_pipeline_harvest.py`; history duplicates are never added to.
- **Per-class bins.** With `--material-binning on` a scene with features is binned by material class: the classify kernel gives each tile a list per class
  (up to 16 bins by descending material count, plus one overflow bin), and each class is resolved by its own pipeline, specialised with
  `MATERIAL_FEATURES = class mask` and the new reserved constant `RESOLVE_BIN` (24; `0x1000 | mask` for a class bin that resolves only the pixels of that
  mask, `0x2000` for the overflow bin, which is the all-lobes pipeline and resolves the pixels outside the table). Beyond 16 classes the surplus runs the
  overflow bin (logged). A class pipeline not yet final makes that frame use the union-served generic dispatch (never a fallback pipeline).
  Scenes without features keep the legacy four classes and unchanged descriptors.
- **Measured deviation (2026-11-14).** A pipeline specialised by a different `MATERIAL_FEATURES` value is not bit-identical to the all-lobes one in the
  linear float capture: the compiler schedules the shortened code differently, a few ULP on a part of the sphere pixels and up to 37 ULP (3.1e-6 relative) on iridescence code (counts in the verification
  record). The equivalence gate is therefore a declared envelope (relative 1e-5 per channel; reported with the bitwise and the 8-bit counts), not "0 bitwise pixels".
  `docs/implementation/` records the numbers; the all-lobes arm repeats bit for bit.
