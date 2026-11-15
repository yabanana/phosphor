# 11: Anisotropy lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_anisotropy` per decision 06: anisotropic GGX D and V with the rotated tangent frame, anisotropic VNDF sampling in the transport, bent-normal environment term (declared).

**Blocked by:** 07, 08

**Status:** resolved with carried items (2026-11-13; Khronos side-by-side delivered by ticket 22; direction-texture numeric fixture NOT built: box 1 PARTIAL)

- [ ] Anisotropy fixtures (strength, rotation, direction texture) pass decision-11 thresholds; furnace bound holds — strength and rotation fixtures and the furnace pass; the direction-TEXTURE fixture needs textures in the reference → ticket 22 — PARTIAL (audit 2026-11-15): strength and rotation fixtures and furnace met; direction-TEXTURE fixture not built (ticket 22: qualitative only)
- [x] Rotation direction and tangent handedness correct on mirrored instances (glTF conventions)
- [x] Negative control (rotation sign flipped) fails
- [x] AnisotropyStrengthTest, AnisotropyRotationTest, AnisotropyDiscTest side by side → ticket 22 (done 2026-11-14, qualitative: only the sun highlight shows the anisotropy in a constant environment; open defect C22-D5, speckle on near-mirror strong anisotropy in one settled frame; `docs/research/2026-11-14-f15-khronos-corpus-report.md`)
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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
  Full runner (all keys, final binary, `build/f15-check/t11`): GREEN, exit 0. Before the composite complement edit only
  furnace-aniso was red (1.0063, see deviation 2).
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

2026-11-13 — orchestrator verification on M5 after integration (engine
commits a55207d-equivalents, archive regenerated with `--skip-present`).
Batch log: `build/f15-exact/battery11.log`.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t11`): GREEN on
  every key; the iridescence keys are pending until ticket 12.
  - Direct grids (sun, point, `@restir`, `@clustered`, mirrored): worst p99
    per group 0.061 % (mirrored 0.017 %). Every control fails; the mirrored
    fixture adds `aniso-mirror-unaware`.
  - `aniso-grid-env-resolve` (declared): worst mean 1.1e-5.
  - `aniso-grid-env` (transport against the exact quadrature): worst sphere
    mean 1.93 %.
  - `furnace-aniso`: 1.0022 (bound 1.005); `@resolve`: 1.0.
- `--debug-lobe-guide 16` on aniso-grid-sun: `pixels 41416 extended 27552 base
  13864 unverified 0 | PASS`. The direction is compared by angle; the corrupt
  control sets it to the normal.
- Guide overflow on aniso-grid-sun: PASS. The direction words differ by at
  most 0.54 degrees between the visibility resolve and the indexed receivers;
  every other word is identical.
- Shader validation: aniso-grid-env with RT reflections and ReSTIR exits
  clean.
- Scenes without features: F13 exact captures and visual_check 0 px.
  archive_check: every local scenario PASS. hitch_check OK. The clip runner
  meets every expectation.
- Accepted deviation, recorded as a declared approximation: when the
  composite adds a transported specular signal, the diffuse complement of an
  anisotropic pixel uses the roughness fit sqrt((alpha_t + alpha_b) / 2).
  - The resolve's declared environment and the probe fit keep the rms fit of
    the decision-06 amendment.
  - It was chosen after the furnace measured 1.0063 with the rms fit, which
    is energy gain: the rms complement exceeds 1 - E_true by about 0.008 at
    strength 1.
  - It is validated on the fixture only, not proven as a bound in general.
  - Candidate for a later fix: a tabulated anisotropic directional albedo
    E(mu, phi, alpha_t, alpha_b). It would also remove the declared model's
    6-7 % gap on metals at strength 1.
- Declared:
  - a primitive without a tangent frame is shaded as strength 0 (glTF
    requires tangents for anisotropy);
  - anisotropy combined with coat or sheen is implemented but has no
    reference (the harness refuses it) → ticket 19;
  - the narrow-lobe sun disk (alpha < 0.05) and a non-constant environment
    for the bent normal are not exercised.

### Ticket 22 closure (2026-11-14)

- Side by side done. The direction texture is exercised qualitatively by AnisotropyDiscTest, AnisotropyRotationTest and AnisotropyBarnLamp; the NUMERIC direction-texture fixture was not built (the reference refuses texture words): still open.

2026-11-15 — closure audit (ticket 24):

- 1 fixtures + furnace: PARTIAL (strength/rotation p99 <= 0.061 %, furnace 1.0022 vs 1.005 met; direction texture not built). Unticked.
- 2 mirrored instances: MET (aniso-mirrored-sun p99 <= 0.017 %, six controls). Ticked.
- 3 rotation-flip control: MET. Ticked.
- 4 side by side: MET qualitative (ticket 22; C22-D5 declared single-bounce reflection of neighbours). Ticked.
- 5 archive/hitch/O7: MET (+17 descriptors). Ticked.
Open: direction-texture fixture; declared-environment fit 6.4-7.0 % gap on metal strength 1 and composite complement fit not proven as a bound; narrow-lobe sun and non-constant environment for the bent normal not exercised; denoiser lobe-id for anisotropic highlights not covered; primitive without tangent shaded as strength 0 (declared).

## F15 closure, loader-misc (2026-11-15): tangent frame repair

- The residue "a primitive without a tangent frame is shaded as strength 0" had two real sources, not one: (a) the loader ALREADY generates MikkTSpace tangents for a primitive without TANGENT (completeGeometry;
  doctest "missing normals and tangents are generated"), so the common case was not a defect; (b) SUPPLIED tangents that are degenerate (zero, parallel to the normal, non-finite) and the MikkTSpace result on
  a degenerate UV mapping (it returns zero tangents) reach the shader as `dot(T, T) <= 1e-8`, which shades anisotropy as strength 0.
- Fix (e8b6399): `repairTangentFrames` (src/scene/mesh_processing.cpp) replaces every unusable tangent of a primitive whose material carries an anisotropy block by a perpendicular of the normal (authored
  handedness w kept; usable tangents untouched).  Only anisotropy primitives are touched, every other mesh keeps its data bit for bit.
- Fixtures (tests/test_gltf_unlit_lights.cpp "tangent frames: ..."): a +Z quad with supplied zero tangents is repaired on an anisotropy material (4 of 4 vertices usable, unit, orthogonal to N, w 1) and the
  SAME data on a plain material stays degenerate (4 of 4 unusable: the state before the fix, the negative control); a tangent parallel to the normal with w -1 is repaired with w -1 kept while the other three
  authored tangents are bit-identical; generated tangents on an all-equal UV mapping are usable after the repair; unit test of `repairTangentFrames` (3 repaired of 4, idempotent).
- Still declared: an anisotropy primitive built by a bench through `GpuScene::uploadMesh` without tangents gets (1, 0, 0, 1) (the synthetic fixtures use combed spheres on purpose); a repaired tangent frame
  has the arbitrary direction of `anyPerpendicular`, so the anisotropy ROTATION of such a vertex is relative to that arbitrary direction (glTF requires tangents for anisotropic materials; an asset that
  ships none, or degenerate ones, has no authored direction to honour).

### 2026-11-15 — direction/strength texture fixture and fit bounds (agent `tex`, branch f15c-tex-2)

- Anisotropy texture (RG direction, B strength) fixture: `texture-aniso-sun` and `texture-aniso-env-resolve` (`build/f15-check/tex3`, `tex4`): sun worst p99 0.65 %
  (copper), max outside mask 1.94 %; env-resolve worst mean 0.007 %, p99 0.06 %. Sun controls texture-v-flipped, aniso-direction-ignored,
  aniso-direction-y-flipped, aniso-strength-ignored all FAIL. The env-resolve key keeps v-flipped and strength-ignored only: the first run showed
  declared-aniso-direction-ignored / -y-flipped PASS (non-discriminating) and that is physics, not a fixture defect: a constant environment has no
  angular signature of the lobe direction (the declared model is rotation independent by design), so the direction controls live on the sun key.
- Declared-environment fit as a tested bound (grid measured with `aniso_albedo_table` against the split sum at the fit roughness, sphere-weighted
  N.V >= 0.1, base roughness 0.2 / 0.3 / 0.5 / 0.7, strength 0.25 / 0.5 / 0.75 / 1): rms fit error 0.4 / 1.1 / 2.0 / 6.5 % at strength 0.25 / 0.5 / 0.75 / 1
  (r 0.3; 6.5, 6.6, 6.5, 4.8 % at strength 1 for r 0.2, 0.3, 0.5, 0.7). The BEST single isotropic roughness (scan of 60 values) still leaves 4.0 / 4.2 / 4.8 /
  4.0 % at strength 1, so no fit of the declared split-sum form reaches below ~4 % and the decision-11 5 % is only met by an anisotropic albedo table
  (the 4-D table of this ticket's open question). Tested bounds (unit tests, thresholds fixed before): rms fit < 7 % at strength 1 (r 0.3, 0.7); best
  isotropic roughness > 3.5 %. The strength-1 metal gap therefore stays a DECLARED approximation of the declared model, now with a measured bound
  instead of an anecdote; a candidate engine tweak (roughness x ~0.97 at strength 1: best fit 0.815-0.827 vs rms 0.841-0.854) would give 4.0-4.8 % and is
  left as an owner decision (it re-opens every aniso key).
- Composite complement fit: the sphere-weighted mean of E_fit - E_exact (F0 0.04, N.V >= 0.1) is +0.0007 / +0.0023 / +0.0030 / +0.0005 for the mean-alpha
  roughness (r/strength 0.3/0.5, 0.3/1, 0.2/1, 0.7/1: never below zero, i.e. the complement does not gain energy on the sphere average) and -0.0042 for the rms
  fit at (0.3, 1) (the 1.0063 furnace gain). Unit test `test_composite_complement_never_gains_energy_on_the_sphere_average` (control: rms < -0.002). A per-pixel
  bound does NOT exist: E_fit - E_exact reaches -0.46 at grazing views along the stretched axis (N.V < 0.1), outside the sphere weight; the furnace fixture
  (pixel max 1.0022 < 1.005) is the per-pixel evidence and stays fixture-bound.
- Narrow-lobe sun and non-constant environment (bent normal): not built. The reference already integrates the sun disk of a narrow anisotropic lobe with the
  engine's disk rule (its `narrow` mask drops the highlight from the max gate), so a fixture would gate only the tails; a non-constant declared environment does not exist in the
  engine (the declared environment is isotropic), so the bent-normal fetch has nothing to be compared against. Stays declared.

### 2026-11-15 (2) — narrow-lobe sun and the bent-normal fixture (agent `tex`, round 2)

- Narrow-lobe sun: the reference now integrates the disk of a narrow anisotropic lobe EXACTLY (`--exact-aniso-disk`: a polar grid of 64 x 128 samples uniform in solid angle, a different
  family from the engine's golden-angle spiral; the narrow pixels leave the quadrature mask). Converged: 64 x 128 against 192 x 384 within 3e-10 relative; the engine's 32 / 128-sample
  spiral agrees with it to 3e-5 on highlight pixels (unit test `test_exact_disk_integration_of_a_narrow_anisotropic_lobe`). New fixture `aniso-narrow-sun` (6 rows x 4 rotations,
  alpha_b 0.0064 .. 0.04, all below the 0.05 rule): GREEN with the existing thresholds, p99 <= 0.35 %, max outside mask <= 1.12 % (3 % bound), no pixel masked (`tex17`); controls
  aniso-ignored, aniso-rotation-flipped, aniso-alpha-swapped FAIL. FINDING: the engine does not miss; the declared 32 / 128-sample rule is accurate for narrow anisotropic lobes.
- Bent-normal fixture: NOT built, with the reason measured: the material-scenario rig runs the declared F13 path in every scenario (`physicalGI` true: the constant declared
  environment, 0 when the scenario sets none), the legacy two-colour hemisphere is only reachable through the legacy bench scenes, not through `--material-scenario`; a fixture
  needs a host option that switches a material scenario to the hemisphere environment plus a reference for the hemisphere fetch (exact quadrature of the anisotropic lobe against a
  non-constant radiance, ~300 M lobe evaluations). Not attempted; the engine's bent-normal fetch stays unexercised.

### 2026-11-15 (3) — bent-normal fixture on the legacy hemisphere (agent `tex`, round 3)

- Host option `--legacy-environment` (`src/core/launch_options.{h,cpp}`, `src/app/engine.cpp`): a material scenario keeps the resolve on the legacy two-colour hemisphere (sky 0.30 0.36 0.45 above,
  ground 0.10 0.09 0.08 below) instead of its declared constant environment; requires `--material-scenario` and `--reflections off`. Without the option nothing changes.
- Reference (`tools/f15_reference.py --legacy-environment`): the declared split-sum weights (anisotropic effective roughness) times hemisphere(N) for the diffuse and hemisphere(specularDir) for the
  specular, specularDir = normalize(mix(reflect(-V, N'), N', roughness^2)), N' the README bent normal written from its GLSL (`aniso_bent_normal`; hand value in the unit test). Control
  `declared-aniso-bent-normal-ignored` fetches along N and must FAIL; `declared-aniso-ignored` too. Fixture key `aniso-bent-env` (scene `aniso-grid-env-resolve`, thresholds of the declared kind).
- Status: reference and unit test done (the bent-normal control moves a glancing metal pixel by > 2 %); the capture and the gate were NOT run (battery hold): UNVERIFIED on the GPU.
Result (round 3, `build/f15-check/tex22`): `aniso-bent-env` GREEN: worst sphere mean 0.059 %, p99 0.094 % against the legacy-hemisphere reference (declared kind, thresholds 0.5 % / 1 %);
controls `declared-aniso-bent-normal-ignored` (fetch along N) and `declared-aniso-ignored` FAIL. The capture runs the plain (non-lit) resolve with `--legacy-environment --reflections off --lighting-denoise off
--ao off` (no `--debug-lighting`, which needs F10-F12 lighting); the first attempt with the custom denoiser still used the declared constant environment (the reflection composite owns it there:
engine = the constant-sky image, 230 % off the hemisphere reference), hence the launch-option rule. Archive: +5 resolve class descriptors (plan update, `perm-harvest2`) and +6 constant-22
descriptors (`.scratch/f15/harvest/tex-5.json`).
