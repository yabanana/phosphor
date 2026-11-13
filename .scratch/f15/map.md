# F15 — Materiali avanzati e varianti shader (wayfinder map)

## Destination

An approved, gap-free F15 plan: `docs/plans/F15.md` refined, a spec
(`.scratch/f15/spec.md`) and tracer-bullet implementation tickets
(`.scratch/f15/issues/`) with declared blocking edges, every design decision
below resolved with cited evidence, and the owner's approval recorded before
any F15 code is written.

## Notes

- 2026-11-13: every decision ticket resolved (owner confirmed 09 and 10);
  spec in `spec.md`, tracer-bullet tickets 01–18 in `issues/` (review pass:
  ticket 04 split into plumbing + lobe guide, denoiser edge-stopping
  corrected, Phosphor-only scope per ADR-0001 addendum); awaiting the owner's
  approval of the plan before any F15 code.

- Roadmap tasks F15.1–F15.4 (`docs/ROADMAP.md`), plan `docs/plans/F15.md`,
  method `docs/plans/METHOD.md`, sequencing `docs/plans/SEQUENCING.md`,
  hardware policy `docs/plans/HARDWARE_VALIDATION.md`, `CLAUDE.md`.
- Skills: `grilling` + `domain-modeling` (self-answered from code, docs and
  research; only owner-level choices go to the owner in one question),
  `research` (primary sources, findings in `.scratch/f15/research/`),
  `prototype` for measured spikes (serial GPU, quiet machine, negative control).
- Phosphor only (ADR-0001 addendum): Track Studio is paused; its six chess
  materials remain a fixture from public assets.
- Product lighting path = visibility buffer + F13 lighting; forward stays the
  default render path for the legacy benches.

## Decisions so far

<!-- one line per resolved ticket -->

- [The six Track Studio materials in MaterialX and glTF terms](decisions/03-track-studio-chess-materials.md): pawn tops = ABeautifulGame as is (transmission + volume); kings/queens need a Phosphor subsurface feature, glTF sample omits it; Track Studio is a consumer (ADR-0001)
- [Khronos reference corpus for F15.3](decisions/02-khronos-reference-corpus.md): Tier A of ~20 controlled models pinned at 8f49239; Khronos images qualitative only; numeric gates from Phosphor-side independent references; KHR_texture_transform joins the loader scope
- [Layered materials for terrain](decisions/05-layered-materials.md): ≤4 layers from an RGBA mask, parameter-space blend + one BSDF, surface-gradient normals, energy and 0/1 tests
- [KHR material extensions](decisions/01-khr-material-extensions.md): ratified set + texture_transform in scope, dispersion out, follow ratified text over the Sample Renderer, Phosphor-generated sheen LUT, tinygltf v2.9.7 Value parsing
- [Schema and GPU layout](decisions/07-schema-and-gpu-layout.md): portable versioned description; GPUMaterial frozen at 80 B + extension records via SceneStore deltas; lobe guide only for extended scenes; uniform UV transforms baked; support report + extensionsRequired refusal
- [Permutation mechanism](decisions/08-permutation-mechanism.md): function-constant feature classes from offline manifests; union specialisation default, binning by ≥5 % rule; budget 64/16/+25 %/0 hitches; exact equality vs generic
- [Real-time transmission and SSS](decisions/04-realtime-transmission-sss.md): pyramid vs RT refraction, Burley screen-space SSS vs hybrid; costs and fit recorded
- [Lobe × light-path coverage](decisions/06-lobe-light-path-coverage.md): primary surface exact for direct light and transport, secondary hits base reduction, shared MSL BSDF module, declared approximations table
- [Transmission and F16 boundary](decisions/09-transmission-scope.md): RT refraction in the specular transport, probes fallback, opaque shadows; raster transparency stays in F16 (owner confirmed, ADR-0004)
- [Subsurface model](decisions/10-subsurface-model.md): OpenPBR semantics; Phosphor description file + draft KHR_materials_scatter + RC diffuse_transmission inputs (owner confirmed, ADR-0005); Burley screen-space; hybrid RT if bound missed
- [Reference method and thresholds](decisions/11-reference-method-and-thresholds.md): fixtures, independent references, thresholds fixed before comparison, discrimination rule
- [F15.4 fixture and real-world cost](decisions/12-layered-fixture.md): procedural terrain patch + 4 CC0 materials + generated RGBA mask + cavity finish; resolve cost per active-layer count

## Not yet specified

- Cost budget of the extended resolve on the chess and terrain fixtures:
  measured in the first tickets that render them; reported, gated only through
  the Sponza no-regression rule and the binning decision rule.

## Out of scope

- Modifying Track Studio or its vendored Phosphor copy.
- F16 work (OIT, particles, decals) and any OPT task not selected by a measured need.
