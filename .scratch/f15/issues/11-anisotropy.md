# 11: Anisotropy lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_anisotropy` per decision 06: anisotropic GGX D and V with the rotated tangent frame, anisotropic VNDF sampling in the transport, bent-normal environment term (declared).

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Anisotropy fixtures (strength, rotation, direction texture) pass decision-11 thresholds; furnace bound holds
- [ ] Rotation direction and tangent handedness correct on mirrored instances (glTF conventions)
- [ ] Negative control (rotation sign flipped) fails
- [ ] AnisotropyStrengthTest, AnisotropyRotationTest, AnisotropyDiscTest side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — test-first harness (independent reference, combed-sphere fixtures, runner keys, snapshot tangents); no engine or
shader change. Commits `5c8f9d2` (reference), `21cd919` (snapshot mesh version 2), `3c9e3a8` (combed sphere), `b881a94`
(fixtures, keys, tests), `b61a47b` (archive) and the commit with this note.

Reference (`tools/f15_reference.py`, from the ratified text and the README's informative GLSL, research/khr-materials.md
section 4): f_spec = F D V, Burley D with alpha_b = alpha = max(roughness^2, 0.002), alpha_t = mix(alpha, 1, strength^2); the
README's height-correlated V WITHOUT its final clamp (the isotropic limit then equals Appendix B to 3e-14, asserted); F, the
diffuse weighting and the metal/dielectric mix are the base model's. Interpretations, all declared in the module and the report:
- Tangent source: the snapshot did NOT carry tangents. `exportOfflineReference` now writes `tangent_x/y/z` and `handedness_x`
  (GPUVertex tx, ty, tz, tw) after `s, t` in every PLY and `"mesh_version": 2` in scene.json (`REFERENCE_MESH_VERSION`; the
  postfixed names make Mitsuba load them silently as vertex attributes, v1 readers ignore them; doctest in `tests/test_gi.cpp`).
  The reference parses the PLY itself and interpolates the attribute at the Mitsuba hit by barycentrics computed in object space
  from the primitive index (Mitsuba's `si.uv` is the texture UV, `dp_du` follows the UV parameterisation, not the mesh tangent).
  A v1 snapshot is refused with a clear error.
- Frame: T = world image of the interpolated tangent, Gram-Schmidt against the shading normal; B = cross(N, T) * w * sign(det
  world matrix); lobe axes T' = cos(rot) T + sin(rot) B (counter-clockwise from T, the README's mat2), B' = cross(N, T'). The
  determinant factor is the world image of the model-space bitangent: a MIRRORED instance (negative scale) renders as the exact
  mirror image of the same surface. Asserted by an independent physical test: a mirrored carrier lit by the mirrored sun equals
  the horizontal flip of the plain one (camera on the mirror plane), and `aniso-mirror-unaware` breaks it.
  The sign of w and the sign of sin(rot) act together (T' only), so on a mesh of handedness -1 everywhere
  `aniso-bitangent-flipped` and `aniso-rotation-flipped` give identical images; the mirrored fixture carries both handedness values.
- Sun disk: the disk rule keys on alpha_b = alpha (unchanged); the fixtures use roughness 0.3 (alpha 0.09 >= 0.05), so the lobe
  is point-evaluated at the disk centre, no quadrature mask (a 4000-sample disk average agrees within 1 %, tested). A narrow
  anisotropic lobe (alpha < 0.05) would use the spiral rule on the reflection axis of the isotropic lobe: not exercised (OPEN,
  with strength rows over a narrower base roughness).
- Exact environment: quadrature of the anisotropic half-vector distribution (CDF sampling of D cos(theta_h), midpoint 192 x 192),
  tabulated over N.V (65 nodes) and the view azimuth against T' (33 nodes in [0, pi/2]); the albedo of D V cos is <= 1 at every
  node (white-furnace bound, tested) and agrees with an independent wi-space Monte Carlo (cosine sampling, 400k samples, within
  6 sigma, five nodes x two strengths); the diffuse integrals are the base model's.
- Declared environment: the Sample Renderer's bent-normal method keeps the isotropic split-sum weights at the material roughness
  and only bends the radiance fetch, so in a CONSTANT environment it is anisotropy-blind (the declared gate could not fail). The
  declared model therefore runs the split sum (DFG table) at an effective isotropic roughness sqrt(alpha_eff), alpha_eff =
  sqrt((alpha_t^2 + alpha_b^2) / 2) (root mean square; of the simple fits tried it has the smallest sphere-weighted error against
  the exact quadrature at N.V >= 0.1: geometric mean 4.3 / 48 %, mean 2.1 / 16 %, alpha_t 7.2 / 31 %, rms 1.1 / 6.6 % at strength
  0.5 / 1), independent of rotation and handedness. Measured error of the declared image against the exact quadrature on the
  fixture (sphere mean relative error, decision-11 floor): dielectric 1.5-3.5 % (the strength-0 rows already differ 1.9 %), metal
  strength 0.5: 2.3-3.4 %, metal strength 1: 6.4-7.0 %.
  OPEN QUESTION for the orchestrator: the fit is the weakest piece; a table of the anisotropic albedo over (N.V, azimuth, alpha_t,
  alpha_b) like the sheen table would remove the 7 % at strength 1, at the cost of a 4-D table.
- Refused: anisotropyTexture (word 2; the direction-texture case is OPEN for the Khronos corpus, ticket 22: the reference is
  texture-free), anisotropy combined with clearcoat or sheen (not modelled), meshes without tangents.
- Controls (`--controls`): `aniso-ignored`, `aniso-rotation-flipped`, `aniso-bitangent-flipped` (w ignored),
  `aniso-alpha-swapped`, `aniso-mirror-unaware` (determinant ignored), `aniso-d-normalised-by-ab` (D prefactor 1/(pi alpha_b^2):
  gains energy, the furnace control); declared: `declared-aniso-ignored`, `declared-aniso-alpha-t`. `tools/test_f15_reference.py`:
  102 tests OK (23 new: alpha rule, frame hand values per handedness / determinant, README D and V formulas vs the implementation,
  isotropic limit exact, D normalisation by an independent hemisphere quadrature, Monte Carlo albedo, furnace bound, control
  directions, sun point evaluation, record words and refusals, declared fit and its measured error, mirrored instance = mirror image).

Fixtures (`src/testbench/material_validation.*`, `tests/test_material_validation.cpp`; 30 scenarios, 828 doctest cases OK):
- Carrier: `ProceduralMeshes::generateCombedSphere` (new, `src/scene/procedural.*`, doctested): the UV sphere with the comb
  direction +X projected on every vertex tangent plane and a chosen handedness (-1 = the engine sphere's). A uniform lobe direction
  facing the camera instead of the UV sphere's circles around a pole (the field has three poles at the limb, normal = +-comb).
  A flat disc or quad was rejected: it sweeps almost no half-vector range under one sun or point light.
- `aniso-grid-sun`, `-env-resolve`, `-env`, `-point`, `furnace-aniso`: rows strength {0 (no extension), 0.5, 1} over a copper
  dielectric then a copper metal (6 rows), columns rotation {0, pi/4, pi/2, -pi/4} (the +-pi/4 pair tells the direction; 0 and
  pi/2 are sign-blind), base roughness 0.3, the base-grid rig (sun 3 W/m2, 250 W/sr point light 9 m from the centre, V.L about -0.5).
- `aniso-mirrored-sun`: two rows (dielectric, metal) at strength 1, four columns at rotation pi/4: plain and MIRRORED instances
  (scale x = -1, INSTANCE_FLAG_MIRRORED, asserted through SceneStore) over meshes of handedness -1 then +1 (two meshes).

Runner keys (`tools/f15_check.py`, `tools/test_f15_check.py`: 55 tests OK), all `"pending": "ticket 11"` with `pending_matches`
aniso-ignored / declared-aniso-ignored: `aniso-grid-sun`, `aniso-grid-point` (+ `@restir`, `@clustered`), `aniso-grid-env`,
`aniso-mirrored-sun` (five direct controls, the mirrored key also `aniso-mirror-unaware`), `aniso-grid-env-resolve` (declared
gate, two declared controls), `furnace-aniso` (transport, mean <= 1.005, asserted green), `furnace-aniso@resolve` (engine-pixel
furnace asserted green, declared gate pending). No `@probes` key (not requested).

Measured red state (M5 Max, `build/f15-check/t11-red`, 1280x720, `--run --expect green`: GREEN, exit 0; every capture 0 archive
misses and 0 render-thread compiles after the archive update). Gate (pending, FAILS) / same gate vs the ignored control (PASSES):
- aniso-grid-sun: FAIL, p99 per group 1.45 / 5.44 (dielectric strength 0.5 / 1), 3.20 / 22.9 (metal), strength-0 groups 0.001 and 0;
  vs aniso-ignored PASS; all five controls fail.
- aniso-grid-point (legacy): FAIL (p99 1.59 / 8.29 / 3.77 / 33.1); @restir and @clustered FAIL (p99 1.89 / 10.0 / 3.81 / 8.06,
  identical numbers: same DI shading); vs aniso-ignored PASS in all three; five controls fail each.
- aniso-grid-env-resolve: FAIL (worst sphere mean 0.951); vs declared-aniso-ignored PASS (worst mean 1.3e-5); both declared controls fail.
- aniso-grid-env: FAIL (worst sphere mean 0.841); vs aniso-ignored PASS (worst mean 0.019, bound 0.05); controls fail (rotation /
  handedness flipped 0.070, alpha swapped 0.083, energy 10.1).
- furnace-aniso: gate PASS (sphere mean max 0.99982, pixel max 1.0064), asserted green; the energy control fails (sphere mean 6.49).
- furnace-aniso@resolve: engine-pixel gate PASS (1.0000); declared gate FAIL (worst mean 0.794), vs declared-aniso-ignored PASS
  (1.3e-6); controls fail (energy 8.95, declared ignored / alpha-t).
- aniso-mirrored-sun: FAIL (p99 5.36 dielectric, 21.9 metal); vs aniso-ignored PASS; all six controls fail (including
  mirror-unaware).

Archive: targeted update, +17 descriptors with MATERIAL_FEATURES 8 (compute 371 -> 383, render 186 -> 190, tile 7 -> 8, mesh 142
unchanged), harvested from the nine captures. A scene combining anisotropy with another feature needs its own union.

Dropped, limited or open:
- The direction texture (anisotropyTexture) and texture-driven strength: refused by the reference, OPEN for ticket 22.
- Anisotropy + clearcoat/sheen combinations: refused (the coat sits above, the order is specified, not implemented in the reference).
- Narrow anisotropic lobes under the sun (alpha < 0.05) and the denoiser lobe-id criterion for anisotropic highlights: not covered.
- `aniso-bitangent-flipped` equals `aniso-rotation-flipped` on the grid (handedness -1 everywhere); only the mirrored fixture separates them.
- The declared environment fit error (above) and the Sample Renderer's bent-normal fetch (invisible in a constant environment, so not
  gated): a non-constant environment fixture with an analytic reference is OPEN.
- Not built: AnisotropyStrengthTest / AnisotropyRotationTest / AnisotropyDiscTest side by side, `tools/archive_check.sh` and
  `tools/hitch_check.sh` (not run here); the GPU allocation count of measured frames is unchanged by a harness and not re-measured.
- ctest: 17/17 (phosphor_tests 828 cases, f15_reference_python, f15_check_python, sheen_table_python, ...).

2026-11-13 — engine side (commits `a55207d` engine, `298136b` archive, `698b814` runner/tools). Verified on M5 Max, 1280x720:
- `f15_check.py --run --expect green` on the nine anisotropy keys (`build/f15-check/t11b`): GREEN. Direct grids (sun, point legacy / ReSTIR /
  clustered, mirrored): p99 <= 0.0003 per group (max outside mask <= 0.009), all five (mirrored: six) controls fail. aniso-grid-env-resolve
  (declared): worst sphere mean 1e-5, both declared controls fail. aniso-grid-env (transport vs exact quadrature): sphere mean error 0.3-1.9 %,
  five controls fail. furnace-aniso (transport): max sphere mean 1.0022 (bound 1.005); furnace-aniso@resolve: pass, declared worst 1e-5.
  Full battery (all keys) before the last two composite edits: only furnace-aniso red (1.0063, see deviation 2); the complete rerun of
  all keys is in the final report of this session's orchestration, not repeated here.
- `--debug-lobe-guide 8` on aniso-grid-sun: PASS (extended 27552, base 13864); with `--debug-lobe-guide-corrupt` FAIL (27552 mismatches).
- `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, aniso-grid-env, `--rt on --lighting restir --reflections rt`, 1500 frames: exit 0, no validation error.
- `tools/f15_lobe_guide_overflow_check.sh ... aniso-grid-sun`: PASS (the direction words differ by <= 0.54 deg between the visibility resolve
  and the indexed overflow receivers; every other word identical). Archive: 0 misses, 0 render-thread compiles in all nine captures.
- ctest 17/17 (new: lobe guide direction by angle, strength record, support table, denoiser sharper-axis rule).
Deviations: (1) the guide publishes the anisotropy bit only when a tangent frame exists (a primitive without tangents resolves it as
strength 0, so `--debug-lobe-guide` would flag such a mesh); (2) the composite uses the mean-alpha albedo for the diffuse complement
when it adds a transported specular signal (rms fit: furnace 1.0063 > 1.005 at strength-1 dielectric, because the diffuse weight at
sqrt(alpha_eff) is 0.008 too high against the exact albedo); (3) coat/sheen + anisotropy is implemented (resolve lights, DI, transport
mixture) but NOT validated (the harness refuses the combination). Carried: AnisotropyStrength/Rotation/DiscTest side by side and the
direction texture fixture (ticket 22), narrow-lobe sun disk, non-constant environment fixture for the bent normal.
