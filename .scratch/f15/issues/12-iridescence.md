# 12: Iridescence lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_iridescence` per decision 06: Belcour–Barla thin-film Fresnel per light sample at V·H and at N·V for environment/weight terms, on metals and dielectrics.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Iridescence fixtures (thickness sweep on dielectric and metal, thickness texture, factor) pass decision-11 thresholds against an independent implementation of the specification's code
- [ ] Thickness 0 disables the effect; furnace bound holds; negative control (film ior ignored) fails
- [ ] Per-light cost measured; IridescenceDielectricSpheres/MetallicSpheres side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — test-first harness (independent reference, grid fixtures, runner keys, targeted archive update); no engine or
shader change. Commits: reference `b33c91b` (+ `94fc076`, `393ad91` test fixes for the old "bit 16 is outside the model" refusals),
fixtures and keys `b5ceb0a`, archive `2282a96`, and the commit with this note.

Reference (`tools/f15_reference.py`, section "KHR_materials_iridescence", from the ratified text and the README GLSL in
research/khr-materials.md section 8):
- Thin-film Fresnel: `ior_to_fresnel0`, `fresnel0_to_ior`, `schlick_f`, `eval_sensitivity` (constants, the matrix as GLSL
  column-major) and `thin_film_fresnel` transcribed line by line from the specification text, never from the MSL. Details of
  the text that were interpreted: `cosTheta2 = sqrt(1 - sinTheta2Sq)` (the text elides it), per-channel `phi23` and `baseIor`
  (base ior from `f0 + 0.0001`, kept below 0.9999 so a white metal gives ior 4e4 instead of a division by zero; the film is then
  invisible, F = 1), `Sm = 2 * evalSensitivity`, `phi` added componentwise to the XYZ-indexed sensitivity exactly as the README
  does (the "channel" of phi is mixed with the XYZ index in the original code; transcribed as written). Result clamped at 0, not
  at 1: the closed form overshoots (copper metal up to 1.076 in the sampled range).
- Independence check of the transcription (test `test_transcription_agrees_with_an_independent_spectral_evaluation`): a
  separate spectral evaluation written for the test (Airy summation with the exact unpolarised s/p Fresnel amplitudes of an
  air | film | base stack, real base ior of f0 as in the README, CIE 1931 matching functions by the Wyman-Sloan-Shirley fit,
  equal-energy illuminant normalised by the white of the equal-energy spectrum) over 72 films (dielectric and copper, ior 1.3 /
  1.8, six thicknesses, three angles): mean absolute difference 0.016, maximum 0.14 (the closed form is an approximation;
  the bound is mean < 0.025, max < 0.17). Negative control: the same transcription with all reflection phases set to zero is
  more than twice as far (asserted), so a sign or phase error in the transcription would be caught. Also: hand value (film
  ior = base ior: no interference, F = Schlick(R0, cos1) within 0.01), TIR returns 1, white metal F = 1, continuity in angle and
  thickness (steps < 0.03), and a wi-space Monte Carlo (uniform hemisphere, 400k samples) of the exact environment table.
- Composition (per the README layers, outside ior 1, base ior 1.5 so base_f0 = 0.04, metal base_f0 = baseColor):
  dielectric spec weight `(1 - s) F_schlick + s F_irid`, diffuse `(1 - s)(1 - F_schlick) + s (1 - max F_irid)`; metal
  `(1 - s) F_schlick + s F_irid`; s = iridescenceFactor (textures refused: factor/thickness textures are OPEN for ticket 22).
  Thickness = the MAXIMUM without a thickness texture (`mix(min, max, 1)`); the Minimum may exceed it.
- Thickness 0 disables the film: the README text transcribed in the research file does NOT state this rule (the closed form at
  d = 0 is near, not equal to, the plain Fresnel: copper normal incidence 0.953 / 0.635 / 0.532 against 0.95 / 0.64 / 0.54).
  It is the convention of the ticket and of the viewers; the reference forces s = 0 there, so the material is EXACTLY the base
  model (asserted by `assert_array_equal` on the BSDF, the environment and a rendered snapshot). ORCHESTRATOR: confirm this reading.
- Angle: cos(theta1) = V.H of each light sample (sun disk samples; point light: V.H of the light direction). The exact
  environment quadrature integrates the film INSIDE the integral at each V.H (V.H histograms of the quadrature weights, shared by
  all films of one alpha: integral = histogram . F(bin)); the DECLARED model (decision 06) evaluates the film once at N.V and holds
  it constant in the split sum: spec albedo `F_irid(N.V) clamp(A + B, 0, 1)`, diffuse weight `1 - max` of that albedo (the F13
  composite's rule for extended lobes, an extension of decision 06's "diffuse x (1 - max F_irid)": it keeps the furnace bound
  exact, `c (1 - max E) + E_c <= 1`). ORCHESTRATOR: confirm the `1 - max(E_irid)` (albedo) reading against the literal
  `1 - max F_irid`; the engine must implement the one the reference declares.
  Measured: declared against exact, sphere-weighted mean relative error over N.V >= 0.1: worst 3.8 % on the fixture films.
- Controls (`--controls`): `iridescence-ignored`, `iridescence-film-ior-ignored` (film ior = the outside ior 1.0),
  `iridescence-film-ior-default` (film ior always 1.3, differs on the 1.8 rows only), `iridescence-at-nv` (N.V instead of V.H:
  the Sample Renderer deviation; in the exact environment it holds F at N.V like the declared model), `iridescence-thickness-min`
  (the minimum instead of the maximum), `iridescence-diffuse-unweighted` (energy). The ticket's `iridescence-thickness-max`
  cannot fail on texture-free fixtures (the specification's thickness IS the maximum there): not registered, OPEN with the
  thickness texture (ticket 22). Declared: `declared-iridescence-ignored`, `-film-ior-ignored`, `-film-ior-default`,
  `-diffuse-unweighted` (no declared at-nv: the declared model is N.V by construction).
- Furnace finding: the exact quadrature of the literal model is NOT bounded by 1 even without the film (white dielectric,
  roughness 0.3: sphere mean 1.028, pixel maximum 1.33 at grazing N.V, a property of the specification's Appendix B). With the
  film the sphere mean is 1.021-1.038 (excess over the plain model <= 0.035 per N.V node, asserted < 0.05). The white metal
  furnace rows are blind to the film (base f0 1: F = 1). The gates are therefore the engine-side ones already used for the other
  furnaces (transport mean <= 1.005, resolve pixel <= 1.005): both pass with a film-blind engine (0.99982 / 1.0000).
- `tools/test_f15_reference.py`: 118 tests OK (16 new).

Fixtures (`src/testbench/material_validation.*`, `tests/test_material_validation.cpp`; 35 scenarios, doctest OK):
`iridescence-grid-sun`, `-env-resolve`, `-env`, `-point`, `furnace-iridescence`: 6 rows (factor, film ior) = (0.5, 1.3),
(1, 1.3), (1, 1.8) over a copper dielectric then a copper metal; 6 columns thickness {0, 200, 400, 600, 800, 1200} nm (the
maximum; the minimum keeps 100 nm), roughness 0.3 (alpha 0.09: no sun quadrature mask), base-grid rig (sun 3 W/m2, 250 W/sr point
light 9 m away, V.L about -0.5). Every sphere carries the block (thickness 0 included: bit set, film off by rule). The
thickness TEXTURE case is not built (reference refuses words 7/8): OPEN for ticket 22.

Runner keys (`tools/f15_check.py`, `tools/test_f15_check.py`: 62 tests OK), all `"pending": "ticket 12"`, `pending_matches` =
`iridescence-ignored` / `declared-iridescence-ignored`: `iridescence-grid-sun`, `iridescence-grid-point` (+ `@restir`,
`@clustered`), `iridescence-grid-env`, `iridescence-grid-env-resolve` (declared gate), `furnace-iridescence` (transport,
asserted green), `furnace-iridescence@resolve` (engine-pixel furnace asserted green, declared gate pending).

Measured red state (M5 Max, `build/f15-check/t12-red`, 1280x720, `--run --expect green`: GREEN, exit 0; every capture 0 archive
misses and 0 render-thread compiles after the archive update). Every pending key FAILS its gate and PASSES it against its
`pending_matches` control; every control fails its gate (all six literal, all four declared):
- iridescence-grid-sun: FAIL, p99 per group 0.31 / 0.94 / 0.53 (dielectric rows), 0.51 / 2.10 / 2.31 (metal rows).
- iridescence-grid-point (legacy): FAIL, p99 0.29 / 0.83 / 0.54 / 0.76 / 6.37 / 2.50; @restir and @clustered FAIL (identical
  numbers, same DI shading; p99 up to 6.35).
- iridescence-grid-env-resolve: FAIL, worst sphere mean 3.62 (limit 0.005); controls fail.
- iridescence-grid-env: FAIL, worst sphere mean 3.37; controls fail (at-nv: worst mean 0.125 on 6 spheres, the weakest; film-ior
  default 2.24; thickness-min 1.38).
- furnace-iridescence: gate PASS (sphere mean max 0.99982, pixel max 1.00643), asserted green; the energy control fails.
- furnace-iridescence@resolve: engine-pixel gate PASS (1.0000 / 0.9906 metal), declared gate FAIL (worst mean 0.124), vs
  declared-iridescence-ignored PASSES; controls fail.

Archive: targeted update, +17 descriptors with MATERIAL_FEATURES 16 (compute 383 -> 395, render 190 -> 194, tile 8 -> 9, mesh 142
unchanged), harvested from the eight captures. A scene combining iridescence with another feature needs its own union.

Open / not done:
- Thickness texture, factor texture: refused by the reference; OPEN for ticket 22.
- Iridescence + clearcoat/sheen/anisotropy/specular/ior combinations and the clearcoat-interface rule: refused, not modelled.
- Not built: IridescenceDielectricSpheres / IridescenceMetallicSpheres side by side, per-light cost, `tools/archive_check.sh`,
  `tools/hitch_check.sh` (not run here); the GPU allocation count of measured frames is unchanged by a harness.
- ctest 17/17 before the archive commit (`phosphor_tests` re-run green after it).
