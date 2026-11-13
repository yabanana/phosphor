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
