# F15 — Advanced materials and shader permutations (spec)

Status: ready-for-agent (after owner approval of the plan)
Roadmap: F15.1–F15.4 (`docs/ROADMAP.md`); plan `docs/plans/F15.md`;
decisions `.scratch/f15/decisions/01–12`; ADR-0001…0005; glossary `GLOSSARY.md`.

## Problem Statement

Phosphor shades every surface with one metallic-roughness model (Lambert +
GGX). Content authored with the glTF material extensions — clearcoat paint,
sheen fabric, brushed (anisotropic) metal, iridescent films, glass, translucent
stone — loads without error but renders as plain PBR, and nothing tells the
author or the consuming application that features were dropped. A concrete
consumer hit this: Track Studio reported six Open Chess Set materials (two glass
pawn tops, two kings and two queens with subsurface scattering) rendered
incompletely. Phosphor also has no way to blend several materials over one
surface (terrain), and its shader specialisation enumerates a full product of
axes, which cannot scale to material features without an uber-shader.

## Solution

Phosphor gains a versioned material description with optional material
features defined by public specifications (ratified `KHR_materials_*`, OpenPBR
semantics for subsurface), loads them from glTF (and from its own material
description file), reports per material what it implements, approximates or
does not support, and renders each lobe in every light path of the visibility
renderer with declared approximations. Shader permutations are generated
offline from the feature classes that material descriptions actually use, so a
scene pays only for the lobes it contains and scenes without features (Sponza)
render exactly as before. A material-validation bench compares every lobe with
an independent reference under frozen lighting, and a terrain fixture proves a
bounded layer stack.

## User Stories

1. As a content author, I want clearcoat (factor, roughness, coat normal) from `KHR_materials_clearcoat` rendered, so that car paint and varnished wood look coated.
2. As a content author, I want sheen (colour, roughness) from `KHR_materials_sheen` rendered with energy-preserving base scaling, so that fabrics look like cloth.
3. As a content author, I want anisotropy (strength, rotation, direction texture) from `KHR_materials_anisotropy` rendered, so that brushed metal shows stretched highlights.
4. As a content author, I want iridescence (factor, film ior, thickness range and texture) from `KHR_materials_iridescence` rendered on metals and dielectrics, so that soap films and coated metals shift colour with angle.
5. As a content author, I want `KHR_materials_ior` and `KHR_materials_specular` honoured, so that dielectric reflectance and tint match the authored values (including the ior = 0 special mode).
6. As a content author, I want glass (`KHR_materials_transmission` + `KHR_materials_volume`) to show the scene behind it refracted, tinted and attenuated by its thickness, so that backlit glass looks like glass.
7. As a content author, I want thin-walled transmissive surfaces (no volume) to transmit without refraction offset, so that thin sheets of glass look right.
8. As a content author, I want subsurface scattering (weight, colour, mean free path, anisotropy; thin or volumetric) rendered, so that side-lit stone, wax and skin show soft light diffusion.
9. As a content author, I want `KHR_materials_diffuse_transmission` rendered as thin translucency, so that leaves and thin cloth glow when backlit.
10. As a content author, I want `KHR_materials_emissive_strength` honoured, so that HDR emitters authored above 1 are as bright as intended.
11. As a content author, I want `KHR_texture_transform` honoured when a material's textures share one transform, so that tiled and offset textures land where authored.
12. As a content author, I want every texture slot of every extension read with the channel and colour space the specification defines, so that maps authored for other viewers work unchanged.
13. As a content author, I want a support report listing, per material, which features are implemented, approximated (with the approximation named) or unsupported, so that I know what the image does not show.
14. As a content author, I want assets that require an extension Phosphor does not support (`extensionsRequired`) to be refused with a clear error, so that I am not shown a silently wrong image.
15. As a content author, I want unknown or unsupported optional extensions listed in the report while the asset still loads, so that partial support is visible.
16. As a consuming application, I want to set material parameters (including feature parameters) at runtime and see the change on the next frame, so that an editor can expose them live.
17. As a consuming application, I want to supply materials through Phosphor's versioned material description file, including overrides of materials of a loaded glTF by name, so that I can drive features glTF cannot encode yet (subsurface from MaterialX sources).
18. As a consuming application, I want the support report in the benchmark/report JSON, so that I can show "incomplete rendering" flags from Phosphor's own statement.
19. As a Phosphor developer, I want the conversion of the chess fixture's MaterialX values into Phosphor's description documented, so that the fixture is reproducible.
20. As a Phosphor developer, I want the six Open Chess Set materials of the roadmap's Track Studio case reproduced from public assets, so that the roadmap case is verified inside Phosphor.
21. As a renderer user, I want scenes without material features to keep their exact images and frame cost, so that F15 costs nothing where it is not used.
22. As a renderer user, I want transmission to work without ray tracing through probes/environment (declared approximation), so that the renderer degrades gracefully with `--rt off`.
23. As a renderer user, I want the forward path to show the new lobes for analytic and ambient light, so that the legacy benches render extended materials plausibly.
24. As a renderer user, I want transmissive objects to see other transmissive objects through bounded interface crossings, so that glass behind glass is not black.
25. As a renderer user, I want the lighting denoiser to stay stable on clearcoat, iridescent and transmissive surfaces, so that features do not introduce new temporal noise beyond the F13 contract.
26. As a terrain author, I want up to four materials blended per pixel by an RGBA mask with optional height contrast, so that terrain transitions look natural.
27. As a terrain author, I want each layer to be an ordinary material description (including features such as a coat finish), so that I reuse my material library.
28. As a terrain author, I want a layer at weight 1 to render exactly like the single material, and weight 0 to leave the others untouched, so that layering never alters authored materials.
29. As a terrain author, I want normals of blended layers combined so that each layer's detail survives and endpoints are exact, so that blends do not flatten relief.
30. As a terrain author, I want blended materials never to reflect more energy than they receive, so that blends do not glow.
31. As a Phosphor developer, I want one shared BSDF module used by resolve, DI and reflections, so that a lobe is implemented once and cannot diverge between light paths.
32. As a Phosphor developer, I want shader permutations generated offline from the feature classes material descriptions use, so that no uber-shader runs in the product and no runtime compile hitches occur.
33. As a Phosphor developer, I want the generator to be deterministic and its permutation count and compile time bounded and reported, so that variant growth stays visible (O11).
34. As a Phosphor developer, I want every generated permutation to equal the generic reference exactly on the visibility path, so that specialisation can never change an image.
35. As a Phosphor developer, I want per-lobe independent references (written from the specifications, Mitsuba for ray casting and ground-truth transport) with thresholds fixed before comparison, so that correctness is measured, not judged by eye.
36. As a Phosphor developer, I want a white furnace fixture for every lobe and combination, so that energy gain is caught.
37. As a Phosphor developer, I want each gated fixture to have a negative control that must fail, so that a fixture that cannot discriminate is detected.
38. As a Phosphor developer, I want the Khronos sample corpus fetched at a pinned revision and rendered side by side with the Khronos images, so that defects per feature are documented without claiming global parity.
39. As a Phosphor developer, I want Phosphor's deviations from the Khronos Sample Renderer (where it departs from the ratified text) listed, so that visual differences are explained rather than tuned away.
40. As a Phosphor developer, I want extension parameters to reach the GPU through the existing scene delta path, so that the GPU scene contract of F5 holds.
41. As a Phosphor developer, I want new per-pixel lobe data allocated only when a scene uses features, so that memory and bandwidth stay unchanged elsewhere.
42. As a Phosphor developer, I want the Apple9 fallback path exercised with features, so that nothing depends on Apple10.
43. As the owner, I want measured costs of the extended resolve (chess and terrain fixtures) and the Sponza budget re-measured A/B/A, so that F15's price is known.
44. As the owner, I want every approximation and every deferred item (coloured shadows, dispersion, multiple-scattering compensation, hybrid subsurface) listed in the handoff, so that nothing is silently assumed done.

## Implementation Decisions

- **Shared BSDF module (prefactor).** The GGX/Lambert evaluation duplicated in
  the resolve shading, DI target/BRDF and reflection sampling becomes one MSL
  module; images stay identical (exact where the arithmetic is unchanged; any
  contraction difference is investigated, not tolerated silently). The Python
  reference remains an independent implementation.
- **Material description** (portable, `phosphor_core`): versioned schema with
  base metallic-roughness parameters, one optional block per material feature
  (specular+ior, clearcoat, sheen, anisotropy, iridescence, transmission,
  volume, subsurface with thin/volumetric mode) and a bounded layer stack
  (≤ 4). Texture references carry index, channel, colour space and UV set.
  Validation is explicit (ranges, ior = 0 mode, companion requirements).
  The description models lobes, not extension JSON (future coat/fuzz
  extensions map onto the same features).
- **glTF loading.** tinygltf v2.9.7 stays; a portable parser converts the
  extension `Value` trees (including texture refs and their transforms) into
  descriptions. Emissive strength is folded into the emissive factor. A
  material whose textures share one UV0 transform has it baked into that
  primitive's UV0; otherwise texture_transform is reported unsupported for that
  material. `extensionsRequired` with an unsupported extension refuses the
  asset. Draft `KHR_materials_scatter` (pinned PR head `a6cd5b4`) and RC
  `KHR_materials_diffuse_transmission` map to the subsurface feature and are
  reported as draft/RC. Dispersion, unlit and lights_punctual are reported
  unsupported.
- **Material description file.** A versioned JSON serialisation of the
  description, loadable standalone or as name-keyed overrides over a glTF;
  used by fixtures (chess kings/queens) and consumers.
- **Support report.** Per material and feature: implemented / approximated
  (named approximation) / unsupported, plus unknown extensions; a `MATERIALS`
  log line and a report-JSON section (schema bump).
- **GPU data (ADR-0002).** `GPUMaterial` stays 80 bytes; feature bits in
  `flags`, first extension record index in its padding; extension parameters
  in a separate buffer of 80-byte records updated by `SceneStore` deltas
  (runtime edits reach the next frame). The store exposes the scene feature
  union. A per-pixel lobe guide (resolved feature parameters for DI,
  reflections and composite) is allocated and read only when the union is
  non-empty.
- **Light paths (decision 06).** The primary visible surface evaluates every
  implemented lobe for analytic lights, sun disk, DI and its specular
  transport; GI composition uses the effective diffuse weight of the layered
  model; secondary RT hits keep the base reduction; all approximations are
  named in the support report. Ratified text wins over the Sample Renderer
  (clearcoat F0 fixed at 1.5, `(1 − weight·max F)` base weighting,
  iridescence per light sample at V·H). No multiple-scattering compensation.
  The sheen E table is generated by a Phosphor tool with a negative control.
- **Transmission (ADR-0004).** Ray-traced refraction inside the specular
  transport: one Fresnel-chosen reflect/refract sample per pixel, rough GGX
  BTDF, ≤ 4 interface crossings, Beer–Lambert along the real inside path
  (a medium entered at a front face ends at any hit), thin-walled without
  offset; probes/environment along the refracted ray without RT; transmissive
  casters shadow opaquely. The F13 denoiser filters BRDF-weighted radiance with
  roughness edge-stopping: the lobe guide supplies the effective transport
  roughness (coat, IOR-scaled transmission) instead of the base roughness.
- **Subsurface (ADR-0005).** OpenPBR parameters. Thin mode = diffuse
  transmission lobe. Volumetric mode = Burley normalised diffusion in screen
  space on a diffuse-irradiance target that the resolve/DI write for subsurface
  pixels (21/55 samples, metres→pixels radius, sub-pixel skip, same-material and
  depth test), recombined with the specular terms. If the screen-space result
  misses its bound, the hybrid RT single-scattering variant is implemented in
  F15 (timeboxed ≤ 2 weeks).
- **Permutations (ADR-0003).** Function-constant feature classes. Every scene is
  specialised by its feature union (forward, resolve, DI/reflection kernels);
  the generator (extension of `variant_gen`) reads class manifests exported by
  the app from fixtures and corpus plus a declared list, writes the class table,
  MSL constants and pipeline descriptors, and updates the archive in a targeted
  way. Per-class tile binning is adopted as default for extended scenes only if
  it lowers the extended resolve time by ≥ 5 % beyond noise (A/B/A, 3 replicas,
  quiet machine). Budget: ≤ 64 archived material permutations, ≤ 16 active
  classes per scene (beyond: generic fallback, logged), archive build time
  +≤ 25 %, 0 render-thread compiles and 0 fallback draws in measured frames.
- **Layer stack (decision 05/12).** ≤ 4 layers from one RGBA weight mask with
  optional endpoint-exact height contrast; parameters blended in parameter space,
  then one BSDF evaluation; normals by surface-gradient blending; weights
  normalised to sum 1; layer presence is part of the feature class.
- **Material-validation bench.** Named scenarios (lobe grids, combinations,
  furnace, extreme roughness, chess fixture rigs, Khronos corpus viewer,
  terrain patch) with frozen light rig, camera, exposure and resolution, linear
  captures and offline-reference snapshots.
- **Chess fixture conversion note.** How the MaterialX Standard Surface values
  of the four subsurface pieces become Phosphor description values
  (transmission at depth 0 → tint in base colour; subsurface weight/colour/
  radius × scale → OpenPBR subsurface); documentation of the fixture only.

## Testing Decisions

- Good tests check external behaviour at agreed seams with expected values
  from independent sources (specification formulas, Mitsuba ground truth,
  known literals), never values recomputed the way the code computes them.
- **Seam 1 — material loading (portable, doctest):** glTF JSON (small inline or
  fixture files) → material descriptions + support report. Covers every
  extension's parameters, defaults, channels, colour spaces, texture transforms
  (bake vs unsupported), emissive strength folding, required-extension refusal,
  draft/RC labelling, description-file load/save/override. Prior art:
  `tests/test_procedural.cpp`, existing scene/loader tests.
- **Seam 2 — permutation generator (portable, ctest):** manifests → generated
  tables: determinism (byte-identical twice), budget enforcement, class keys.
  Prior art: the `pipelines script` staleness test, `variant_gen`.
- **Seam 3 — layer blend rules (portable, doctest on the shared C++/MSL header,
  `fp contract(off)` where CPU/GPU must agree):** weights normalisation, endpoint
  exactness, monotonicity, no NaN with all-zero weights. Prior art:
  `cull_math.h`, `transform_math.h` tests.
- **Seam 4 — rendered image (macOS, the F15 runner):** app run of a named
  scenario → linear capture + snapshot → independent reference → thresholds
  fixed in decision 11, plus a negative control per gated fixture that must fail.
  Prior art: `tools/f13_reference.py`, `tools/f13_f14_check.py`,
  `tools/f12_oracle_*`.
- **Regression seams (existing tools, unchanged):** `tools/visual_check.sh`
  (0 pixels on every bench), `ctest`, `metal_syntax_check` (Linux CI), the F13/F14
  corpus runner, `tools/variant_check.sh`, `tools/archive_check.sh`,
  `tools/hitch_check.sh`, Sponza D2 budget A/B/A on a quiet machine, API
  validation (shader validation only where the F14.4 instrumentation defect
  does not apply), `--force-family apple9`, 0 GPU allocations in measured frames.

## Out of Scope

- Track Studio: no work for or in it (ADR-0001; its development is paused by
  the owner); no host-API work (`rea/host-embedding` stays outside F15).
- F16: OIT, blended transparency, raster glass/liquids with a colour pyramid,
  dispersion, coloured/attenuated shadows of transmissive casters.
- Multiple-scattering energy compensation, `KHR_materials_unlit`,
  `KHR_lights_punctual`, per-texture (non-uniform) texture transforms, a second
  UV set.
- Image-based lighting from HDR environment maps (Khronos images stay
  qualitative).
- Any OPT task (OPT-2.0 included); the hybrid RT subsurface variant unless its
  decision rule triggers.
- Physical T0/M3 certification (`EXTERNAL_VALIDATION_PENDING`).

## Further Notes

- Package order: prefactor and reference harness → feature plumbing and lobe
  guide (proven by ior/specular) → lobes one at a time, each
  with its reference fixture (F15.1 + F15.3 incrementally) → critical
  combinations → chess fixture → generator (F15.2) → Khronos corpus run →
  layers (F15.4) → closure. This refines the plan's ".1 → .3 → .2 → .4": the
  reference harness comes first so every lobe lands with its test (tracer
  bullets); the generator comes after the lobes so its class set and binning
  decision are measured on real consumers.
- One GPU process at a time, quiet machine for measurements, 80 W adapter
  pacing (`tools/battery_pause.py`), commits per verified change with task IDs.
