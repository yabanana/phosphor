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

## Consequences

- A permutation outside the generated set runs the generic pipeline (logged)
  until its specialisation compiles; budgets (≤ 64 archived permutations, ≤ 16
  active classes per scene, +≤ 25 % archive build time, 0 hitches) are tracked.
