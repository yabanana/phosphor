# 10: Sheen lobe, sheen albedo table and its environment term

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_sheen` per decision 06: Charlie distribution and visibility, base scaled by the sheen directional albedo from a Phosphor-generated E(μ, roughness) table, and the sheen environment term in the composition (probe/GI radiance at N × Charlie albedo).

**Blocked by:** 07, 08

**Status:** resolved with carried items (2026-11-13; Khronos side-by-side delivered by ticket 22; captured-probe sheen fixture NOT built: box 2 PARTIAL)

- [x] Table generator reproducible and checked against an independent numerical integration; a corrupted table fails the check
- [ ] Sheen grid (colour × roughness) passes decision-11 thresholds under constant environment and under a probe-lit fixture; furnace bound holds — constant environment, the probe path (`@probes`, analytic) and the furnace pass; a CAPTURED-probe fixture is open (captured radiance does not reach the sheen term yet, declared) → ticket 22 — PARTIAL (audit 2026-11-15): constant environment and analytic probe met; no CAPTURED-probe fixture exists
- [x] Negative control (no albedo scaling) exceeds the furnace bound
- [x] SheenTestGrid and one Tier B sheen model side by side → ticket 22 (done 2026-11-14: SheenTestGrid and SpecularSilkPouf rendered; GlamVelvetSofa REFUSED, non-uniform KHR_texture_transform, defect C22-R2b; `docs/research/2026-11-14-f15-khronos-corpus-report.md`)
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — test-first harness (table tool, independent reference, fixtures, runner keys); no engine or shader change.
Commits `9ae1dee` (table tool, committed table, check), `a106397` (reference), `388bfb0` (fixtures, keys, tests),
`4351375` (archive) and the commit with this note.

Table (`tools/sheen_table.py`, `src/renderer/sheen_table.h`, 64x64, `SheenTable[4096]`, `SheenTableSize`,
`PHOSPHOR_SHEEN_TABLE`, `sheenTableLookup(table, nv, roughness)`: same layout, macros and bilinear rule as the DFG
table, rows = sheen roughness j/63, columns = N.V i/63, node 0 at 1e-3; the engine agent reads it through a buffer
like `dfgTable`, nothing is wired). E = Fresnel-free albedo of D_charlie V_charlie N.L, generated deterministically by
half-vector quadrature (256 Gauss-Legendre in cos theta_h x 256 phi), sheen roughness clamped to [0.07, 1], stored as
min(E, 1). Finding: with the Estevez-Kulla visibility fit the unclamped albedo EXCEEDS 1 at grazing views of narrow
lobes (2.92 at N.V 1e-3, roughness 0.07; 51 of 4096 entries clamp, all N.V < 0.06), so `1 - max3(S) E` would go negative:
the clamp is part of the model and mirrored by the reference (the Sample Renderer's unorm LUT saturates the same way).
`--check` (run: OK): regeneration (max |fresh - committed| 5e-10, byte-identical text), independent integration
(separately written BRDF, midpoint product rule over the INCIDENT hemisphere, 49 nodes, tolerance 2e-3 absolute: PASS;
measured worst difference 2.2e-3 only at the clamped narrow-lobe grazing node before the clamp, <= 7e-5 elsewhere
against a 2048x1024 grid), and the negative control (one entry perturbed by 0.05 FAILS as required). The unit tests
also fail a perturbation at every one of the 49 checked nodes in both directions, and a third method (fixed-seed cosine
Monte Carlo) agrees within 5 sigma. `tools/test_sheen_table.py`: 19 tests, registered in ctest (`sheen_table_python`).

Reference (`tools/f15_reference.py`, from the ratified text and the non-normative notes, research/khr-materials.md
section 3): f = S D_charlie V_charlie + f_material scaling; alpha_g = r^2; roughness clamp [0.07, 1] (declared lower
bound; fixture columns start at 0.1). Interpretations, all declared in the module and the report:
- Albedo scaling: DIRECT light (point, sun) uses the notes' exact min(1 - max3(S) E(N.V), 1 - max3(S) E(N.L)); the
  ENVIRONMENT uses the cheap form 1 - max3(S) E(N.V) (the notes' own indirect statement). Measured: the min form
  matters (control `sheen-cheap-scaling` fails the direct gate on every sheen row), because E(N.L) is large near the
  terminator, so the engine must implement the min form for analytic lights and the sun.
- E in the literal model is the reference's own incident-hemisphere quadrature (clamped); the declared model reads the
  engine header (`sheen_lookup`, bilinear rule re-implemented from its documentation).
- Sun disk: the Charlie lobe and the scaling are point-evaluated at the disk centre (lobe about 4 degrees wide against
  a 0.27 degree disk; a 4000-sample disk average agrees within 0.5 % at sheen roughness 0.1); base terms keep the
  declared rule; sheen adds no quadrature-mask pixels.
- Exact environment: base (1 - max3(S) E(mu)) + S E(mu) L (exact for a constant environment). Declared: the same with
  E from the table, split-sum base.
- Clearcoat sits ABOVE the sheen (coat attenuates the whole sheen + material value, direct, sun, exact and declared
  environment); implemented and unit-tested, but no coat + sheen fixture exists (open). Textured sheen (words 17/18)
  is refused; zero sheenColor is exactly the plain material (the engine clears the bit).
- Controls (`--controls`): `sheen-ignored`, `sheen-unscaled-base`, `sheen-ashikhmin-visibility` (D and E follow it),
  `sheen-roughness-linear`, `sheen-cheap-scaling`; declared: `declared-sheen-ignored`, `-unscaled-base`,
  `-roughness-linear` (no declared Ashikhmin: no table exists for it; no declared cheap-scaling: the environment form
  has no per-light factor). `tools/test_f15_reference.py`: 79 tests OK (23 new: hand values of D, Lambda, V, scaling
  forms, layering at one geometry, furnace bounds of the BSDF integral and of the environment, controls' directions,
  sun point evaluation vs disk average, coat above sheen, declared hand value, table agreement, record words 13-18).

Fixtures (`src/testbench/material_validation.*`, `tests/test_material_validation.cpp`; 24 scenarios, ctest 17/17):
rows = sheen colour {0, (1,1,1), (1,0.5,0.25), (0.25,0.5,1)} (row 0 carries no extension), columns = sheen roughness
{0.1, 0.3, 0.5, 0.75, 1}, dark rough base (grey 0.1, roughness 0.7, metallic 0), the base-grid rig (sun 3 W/m2 from
(-0.8, 0.35, -0.3), or the 250 W/sr point light 9 m from the grid centre, V.L about -0.5). Scenarios: `sheen-grid-sun`,
`-env-resolve`, `-env`, `-point`, `furnace-sheen`.

Runner keys (`tools/f15_check.py`, `tools/test_f15_check.py`: 47 tests OK), all `"pending": "ticket 10"` with
`pending_matches` sheen-ignored / declared-sheen-ignored: `sheen-grid-sun`, `sheen-grid-point` (+ `@restir`,
`@clustered`) direct gates per (metallic, sheen colour) row; `sheen-grid-env-resolve` declared gate; `sheen-grid-env`
(SSR transport, sheen not sampled) environment gate; `sheen-grid-env@probes` declared gate on the probe path with the
built-in analytic probe (constant environment); `furnace-sheen` (transport mean <= 1.005) asserted green;
`furnace-sheen@resolve` engine-pixel furnace asserted green, declared gate pending.

Measured red state (M5 Max, `build/f15-check/t10-red`, 1280x720, `--run --expect green`: GREEN, exit 0; every capture
0 archive misses, 0 render-thread compiles after the archive update). Gate (pending, FAILS) / same gate vs the
ignored control (PASSES) / controls:
- sheen-grid-sun: FAIL, p99 per sheen row 2.25 (white), 2.55 (orange), 2.86 (blue), colour-0 row 0.0004; vs sheen-ignored PASS; all five controls fail.
- sheen-grid-point (legacy): FAIL (p99 3.04 / 1.99 / 1.65); @restir and @clustered FAIL (p99 1.13-1.25); vs sheen-ignored PASS in all three; five controls fail.
- sheen-grid-env-resolve: FAIL (worst sphere mean 0.639, 15 spheres); vs declared-sheen-ignored PASS (worst mean 0.0001); controls fail (ignored 0.639, unscaled-base 0.226, roughness-linear 0.446).
- sheen-grid-env@probes: identical numbers to env-resolve (the probe path with no captured probe is the same deterministic term today).
- sheen-grid-env: FAIL (worst mean 0.638, 12 spheres); vs sheen-ignored PASS (worst mean 0.0092, bound 0.05); controls fail (0.641 / 0.225 / 0.114 Ashikhmin / 0.449).
- furnace-sheen: gate PASS (sphere mean max 1.00002, pixel max 1.0010), asserted green; unscaled-base control fails (sphere mean 1.277, pixel 1.638).
- furnace-sheen@resolve: engine-pixel gate PASS (1.0000); declared gate FAIL (worst mean 0.286), vs declared-sheen-ignored PASS; controls fail (unscaled 1.638; declared ignored 0.286, declared unscaled 0.381).

Archive: targeted update, +17 descriptors with MATERIAL_FEATURES 4 (compute 359 -> 371, render 182 -> 186, tile 6 -> 7,
mesh 142 unchanged), harvested from the nine captures. A scene combining sheen with another feature needs its own union.

Dropped, limited or open:
- `sheen-cheap-scaling` is not an environment/furnace control (identical there by construction); `sheen-ashikhmin-visibility`
  has no declared twin; `sheen-ignored` is not a furnace control (an upper bound the sheen-less engine meets).
- Probe-lit fixture: only the analytic probe is covered (`sheen-grid-env@probes`). A captured probe in a sphere grid would
  capture the neighbouring spheres, which the reference (no inter-reflection) cannot model; a captured-probe fixture needs a
  radiance field with an analytic reference (e.g. a probe-box with a constant-colour room): OPEN.
- Not built: coat + sheen fixture, SheenTestGrid / Tier B sheen model side by side, texture-driven sheen, the denoiser
  lobe-id criterion for the sheen environment term; `tools/archive_check.sh` and `tools/hitch_check.sh` not run here;
  the GPU allocation count of measured frames is unchanged by a harness and not re-measured.

2026-11-13 (engine) - KHR_materials_sheen implemented (F15.1); the runner is green on all keys.
Commits: table layout fix, engine, runner keys not pending, archive update (see `git log --grep "ticket 10"`).

What the engine does (decision 06; `bsdf.h`, `material_shading.h`, `restir_common.h`, `reflection_*`):
- Charlie D, the notes' lambda fit V, lobe S D V, roughness clamped to [0.07, 1] (stored clamped in the lobe guide).
- Analytic lights and the sun: base (ior/specular included) scaled by min(1 - m E(N.V), 1 - m E(N.L)); sheen lobe point-evaluated
  (the sun at the disk centre); coat above the sheen. ReSTIR / clustered DI: the same in target and shading (the emitter
  share of the sheen lobe is 1: the transport does not sample sheen).
- Environment / GI / transport: the base specular weight of the sampled lobe and the diffuse weight carry 1 - m E(N.V); the sheen
  term S E L(N) is added next to the diffuse ambient: resolve (declared environment or legacy hemisphere at N, GI irradiance / pi
  with GI) or, when the composite owns the diffuse, the composite (giSelected / pi with GI, the declared environment `p.environment`
  without GI, times AO like the diffuse). Probe path: split-sum weight scaled; captured probes do NOT light the sheen term
  (declared in the support table; a captured-probe fixture with an analytic reference is still open).
- Secondary RT hits ignore sheen (declared). Support table: approximated, "lobe_not_sampled_environment_term_in_composition;
  sun_disk_point_evaluated;secondary_rt_hits_ignore_sheen".

Finding fixed on the way: the harness table (uniform N.V axis, values clamped to 1) interpolated E across the clamp kink near
N.V 0.05 with errors of 0.03-0.05, which made the min-form light scaling of the dim terminator pixels differ from the reference
by up to 0.2 (gate 0.03) in the legacy sun and point grids. The table now stores the unclamped E at N.V = (i/63)^2 and roughness
0.07 + 0.93 j/63, and the lookup clamps after interpolating (max error 0.002 between nodes, unit test); the reference's own
albedo table had the same kink (0.011 off against a 2048 x 1024 integration) and was fixed the same way. The reference's incident
quadrature (512 x 128) is still inaccurate for N.V below 0.005 at sheen roughness >= 0.3 (E 0.81 against 0.876 at N.V 0.001): the
pixels it touches lie in the excluded terminator band today; a finer quadrature is open if a fixture ever needs it.
Archive: +8 compute descriptors (371 -> 379), the nine sheen captures show 0 misses and 0 render-thread compiles.
Open: SheenTestGrid / Tier B model (ticket 22), captured-probe sheen fixture, coat + sheen fixture, textured sheen on GPU (CPU guide
expectation marks textured sheen unverified), `archive_check.sh` / `hitch_check.sh` not run here.

2026-11-13 — engine side verified on M5. Code ded8a58 (Sonnet agent,
reviewed), table fix 883cc6d, runner 56c3c01, archive 53da879, guide control
1b2450c. Batch log: `build/f15-exact/battery10.log`.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t10`): GREEN on
  every key; the anisotropy keys are pending until ticket 11. Every control
  fails, and the unscaled-base control exceeds the furnace bound (1.277).
  - `sheen-grid-sun`: worst max outside the mask 0.45 %.
  - `-point`: 0.54 %; `@restir` and `@clustered`: 0.12 %.
  - `-env-resolve`: worst mean 7e-5; `@probes`: 7e-5.
  - `-env` (transport): worst sphere mean 0.94 %.
  - `furnace-sheen`: 1.00007; `@resolve`: max pixel 1.0000006.
- Table fix, made by the engine agent after the first engine run failed the
  sun/point grids:
  - The harness table interpolated values clamped to 1 across a kink near
    N.V 0.05, and the reference's declared albedo table had the same kink
    (0.011 off against a 2048x1024 integration).
  - The table is now unclamped on a quadratic N.V axis and clamped after
    interpolation, in both the engine table and the reference's declared
    lookup.
  - The exact quadrature of the reference is unchanged. This is a numerical
    fix of a shared table, checked against an independent integration, not
    a change of the gate to fit the engine.
- `--debug-lobe-guide 16` on sheen-grid-sun: `pixels 34100 extended 25438
  base 8662 unverified 0 | PASS`. The corrupt control now also perturbs the
  sheen roughness.
- Shader validation: sheen-grid-env with RT reflections and ReSTIR exits
  clean.
- Scenes without features:
  - F13 exact captures 0 px.
  - visual_check flagged benches 5 and 6 at 25 / 8 px, delta 1. Under API
    validation only they are identical to the preceding build (0 px), and the
    validated repeat agrees. This is shader-validation codegen after
    FrameConstants grew to 184 B; the references were retaken (74ff194,
    research note addendum).
- archive_check: every local scenario PASS. hitch_check OK. The transport and
  lobe-guide overflow checks PASS.
- Clip runner: the adaptive-shading bit comparison failed `repeat_identical`.
  One run captured with fallback pipelines, its worker compiles slowed by
  parallel agent builds. It now runs with `--pipeline-sync` (74ff194), to be
  re-verified in the next battery.
- Declared:
  - the sheen environment term in the composite is the declared environment
    (or the GI irradiance / pi);
  - captured probe radiance does not light the sheen term;
  - secondary RT hits ignore sheen.

### Ticket 22 closure (2026-11-14)

- SheenTestGrid and SpecularSilkPouf side by side done; ToyCar adds sheen on a cloth. GlamVelvetSofa refused (non-uniform texture transform). The captured-probe sheen fixture and the coat + sheen fixture remain open (no Khronos model combines them).

2026-11-15 — closure audit (ticket 24):

- 1 table generator: MET (9ae1dee, 883cc6d; 19 python tests). Ticked.
- 2 grid under constant env and probe-lit, furnace: PARTIAL (constant env, analytic @probes, furnace 1.00007 met; captured-probe fixture does not exist). Unticked.
- 3 negative control: MET (unscaled base mean 1.277). Ticked.
- 4 side by side: MET qualitative (ticket 22: SheenTestGrid, SpecularSilkPouf; GlamVelvetSofa refused). Ticked.
- 5 archive/hitch/O7: MET (+17, +8 descriptors). Ticked.
Open: captured-probe sheen fixture; sheen texture numeric fixture; reference quadrature inaccurate for N.V < 0.005 at roughness >= 0.3 (terminator band only). Note: references of benches 5/6 retaken in 74ff194 after a 1-LSB shader-validation change.

### 2026-11-15 — texture closure and grazing quadrature (agent `tex`, branch f15c-tex-2)

- Sheen colour (sRGB RGB) and roughness (A) texture fixture: `texture-sheen-sun` and `texture-sheen-env-resolve` GREEN (`build/f15-check/tex4`):
  sun worst p99 0.18 %, max outside mask 0.44 %; env-resolve worst mean 0.027 %, p99 0.11 %; controls texture-v-flipped,
  sheen-color-ignored, sheen-color-srgb-encoded (texels used as stored), sheen-roughness-ignored all FAIL (sun and declared twins).
- The per-roughness `sheen_albedo` loop is replaced for continuous fields: a 96 x 257 grid (roughness r = 0.07 + 0.93 q^2, N.V = s^2) interpolated
  bilinearly in (q, s) for more than 16 distinct roughness values; up to 16 values the exact table of each is used as before. Threshold fixed
  before comparing: |grid - exact table| <= 1e-3 in E at off-node roughness (unit test `test_roughness_grid_matches_the_exact_tables_at_off_node_roughness`,
  with a coarse-grid negative control that exceeds it).
- Grazing quadrature (N.V < 0.005, roughness >= 0.3): the incident quadrature is now dense near the horizon and the view azimuth (cos_i = u^2,
  phi = pi w^3, 512 x 128 midpoint nodes). Independent proof: a Gauss-Legendre grid (cos_i = u^3, phi = pi w^4, 700 x 500 and 1400 x 1000: equal
  to 1e-6) and a log-spaced midpoint grid (6000 x 6000: equal to 1e-7) give E = 0.871033 at N.V 0.001, roughness 0.3 (the old table gave 0.81; the
  ticket said 0.876, which was not an exact value). New table vs dense: max |error| 3.8e-4 over roughness 0.3 / 0.5 / 1 and N.V 1e-4 .. 5e-3
  (bound 1e-3, unit test `test_albedo_is_accurate_at_grazing_views_against_a_dense_quadrature`).
- Captured-probe sheen fixture: NOT built. The engine does not light the sheen term from captured probes (declared in this ticket); a fixture
  would measure a declared gap, not a correctness property, and needs an analytic radiance field plus an engine change (a probe lookup of S E L(N)).
  Stays open as a declared limitation.

### 2026-11-15 (2) — captured-probe sheen term (agent `tex`, round 2; decision 06: "probe/GI radiance at N x Charlie albedo")

- Defect confirmed and fixed. The probe kernel (`reflection_probe_only_lobe`) lit the specular lobe from the probe but the composite added the sheen term S E L(N) with the DECLARED
  environment; with `--reflections probes` and no GI the sheen term now takes the probe radiance at N (roughest filtered level, coat attenuation included): `shaders/reflection_probes.metal`
  (new block after `result+=coatResult`), `shaders/reflection_composite.metal` (the declared term is skipped under the new flag `REFLECT_COMPOSE_SHEEN_FROM_PROBE`, `src/renderer/gpu_types.h`,
  set in `src/platform/metal/reflection_passes.cpp`). With GI on the GI irradiance / pi stays L(N) (unchanged).
- Fixture `sheen-probe-env`: the sheen grid on a BLACK base (no diffuse term, which keeps the declared environment 0.5), lit by a cooked constant probe of radiance 0.2 written by
  `tools/f15_check.py` (`write_constant_probe`, `--reflection-probe`), `--lighting clustered --reflections probes --lighting-denoise off`; reference `--sky-override 0.2` (the declared
  sheen model at the probe radiance). Material scenes disabled every probe lookup (`!options.materialScene.empty()` zeroed the probe flags, "constant infinite incident radiance"): an
  explicit `--reflection-probe` is now honoured there (`reflection_passes.cpp`), nothing else changes.
- Evidence the fixture bites: before the probe lookup took effect (legacy lighting, then the probe flags still zeroed) the engine ignored the probe entirely (a 0.8 probe gave an image
  identical to the 0.2 probe: max difference 0.0) and the gate failed with sphere means of 150 % (engine = declared 0.5 everywhere, reference = 0.2). With the probe honoured but the composite
  still adding the declared sheen term the interior spheres matched and the sheen rows differed; after the change: GREEN, worst sphere mean 0.018 %, p99 0.065 % (`tex17`).
  Controls FAIL: declared-sheen-ignored and `declared-sheen-declared-radiance` (the sheen term lit by the declared 0.5, i.e. the engine before the fix).
- Declared remainder: with the custom denoiser on, the probe-lit specular+sheen signal passes through the denoiser and the boundary spheres of the grid differ by up to 1.9 % (measured
  with `--lighting-denoise custom`: worst sphere mean 0.83 %, p99 6.7 %): the fixture runs unfiltered like the other `@probes` keys. With `--lighting-denoise off` / legacy lighting the resolve
  owns the ambient and the sheen term there still uses the declared environment (decision 06 is met on the F13 composite path only).
- visual_check (`BENCH8_ARGS="--instances 100000"`, reference-f15-appB): benches 1-5, 7, 8 are 0 px; bench 6 differs by 9 px, max delta 1 (the known 1-LSB shader-validation codegen effect of tickets 10/74ff194, the
  references of benches 5/6 were retaken then). It was NOT re-checked under API validation only (battery hold); the five F13 exactness PFMs are byte-identical. Owner/orchestrator: rerun bench 6 without shader validation.
