# 08: Permutation mechanism for F15.2 (O11, no uber-shader)

Type: grilling
Status: resolved
Blocked by: 07

## Question

Do generated permutations use Metal function constants keyed by feature class
(reusing PipelineCache, archive harvest and variant checks) or textual MSL
generation? How are feature classes chosen offline from material descriptions,
dispatched at runtime (scene-union pipeline vs tile binning per class), bounded
(variant and compile-time budget) and checked (determinism, equivalence with
the generic reference)?

## Answer

Evidence: `shaders/variants.def`, `tools/variant_gen`, `FC_RESOLVE_CLASS`
binning, `docs/RENDERING_F7_F8.md` ("binning and tile comparisons against
generic resolve remain exact"), archive counts (296/170/142/3), O11.

- **Mechanism**: Metal function constants keyed by feature class (bitmask of
  material features, plus the existing normal/emissive bits), not textual MSL
  generation: dead lobes are compiled out (O11), and PipelineCache, archive
  harvest, flexible fallback and variant checks are reused unchanged. The
  generator (an extension of `tools/variant_gen`) reads material-class
  manifests produced offline from the fixtures and corpus (an export option of
  the app) plus a declared list, and writes the class table, the MSL constant
  header and the pipeline descriptors for a targeted archive update (no full
  re-harvest). ADR-0003 records why this satisfies "generatore di
  permutazioni MSL" without an uber-shader.
- **Dispatch**: every scene is specialised by its feature union (forward and
  resolve, like `EMISSIVE`), so Sponza keeps its exact pipelines. Per-class tile
  binning (classes from the per-material class table) is the measured option;
  decision rule fixed now: binning becomes the default for scenes with extended
  materials only if it lowers the extended resolve time on the chess and
  terrain fixtures by ≥ 5 % beyond noise (A/B/A, 3 replicas, quiet machine);
  otherwise the union pipeline stays the default. The generic all-features
  pipeline remains the reference and the asynchronous-compile fallback.
- **Budget (fixed before measuring)**: ≤ 64 generated material permutations in
  the archive, ≤ 16 active feature classes per scene (beyond: generic fallback,
  logged), archive build time +≤ 25 % over the measured baseline, 0
  render-thread compiles and 0 fallback draws in measured frames with the
  archive (O9).
- **Checks**: generator determinism (two runs byte-identical, ctest); every
  class pipeline equals the generic resolve exactly (0 pixels, visibility path)
  on each fixture; negative control: forcing a class without a lobe the scene
  uses must differ.
- OPT-2.0 is not needed (ticket grilling Q4).
