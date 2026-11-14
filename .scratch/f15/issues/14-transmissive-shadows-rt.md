# 14: Attenuated coloured shadows through transmissive casters (ray tracing)

Roadmap: F15.1, F15.3

**What to build:** Owner decision 2026-11-13: shadow rays through transmissive surfaces accumulate an RGB transmittance instead of stopping — per-surface factor from transmission, tint and Fresnel, plus Beer–Lambert from the signed sum of entry/exit distances per medium (order-independent, so any-hit traversal needs no sorting); straight line, caustics excluded (declared). Used by the RT sun shadow mask (scalar → RGB), DI visibility, GI shadow rays and the reflection secondary lighting.

**Blocked by:** 13

**Status:** ready-for-agent

- [ ] Spike first (timeboxed, results in opt-log): Metal intersection-function invocation multiplicity for non-opaque triangles (duplicate invocations would double-count) and cost; mitigation chosen by measurement
- [ ] Planar-interface fixtures (glass pane, flat shallow water with sun within 30° of the normal) vs Mitsuba within the decision-11 transmission bound; curved-glass shadows (pawn tops) qualitative (caustics excluded)
- [ ] Opaque casters unchanged (visual_check 0 px on scenes without transmissive materials); negative control (transmittance ignored → opaque) fails the planar fixtures
- [ ] Sun shadow mask format change and DI/GI visibility changes keep F10/F11/F12 corpus cases green
- [ ] Cost on the glass fixture and Sponza D2 within band
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-14 — test-first harness of tickets 14 and 15 (no engine or shader change; the same record is in ticket 15). Commits: `b14f650` (bench
fixtures), `60054a8` (Mitsuba ground truth, declared model, controls, tests), `b587e30` (runner keys, gate kinds, envelopes), `dcd9181` (targeted
archive update: +5 compute, +2 render descriptors, MATERIAL_FEATURES; the ior 1.33 slab variant had missed in `forward_fs`).

Fixtures (`src/testbench/transmissive_shadow_validation.*`, bench class on `--material-scenario`; `tests/test_transmissive_shadow_validation.cpp`;
mirrored by `tools/f15_shadow_fixtures.py`): environment 0, no GI, a horizontal diffuse FLOOR (16 m, base 0.8, roughness 1) under horizontal planar
casters at 0.8 m (water 0.5 m); the sun (3 W/m2) sits BEHIND the casters as seen by the camera (towards -Z) so the shadows fall towards the camera: the
camera (38 degrees above the floor, 6.2 m) sees the floor shadows and never the floor through a caster (a test proves it). Scenarios: `pane-sun` (sun 25
degrees from the normal: thin clear pane, thin GREEN pane (0.4, 0.9, 0.45), thick clear slab 0.06 m (single-sided closed mesh: the shadow ray leaves through
a back face), opaque slab), `water-sun` (20 degrees: clear and murky ior 1.33 slabs 0.12 m, attenuation (0.45, 0.75, 0.9) at 0.3 m, opaque slab),
`volume-pane` (20 degrees: three 0.1 m slabs, attenuation (0.5, 0.8, 1)@0.2, (1, 0.6, 0.35)@0.2, grey 0.65@0.15), `point` (one point light 60 W/sr at
(0, 4.5, -0.3): panes of 0.7 m, <= 26 degrees from the vertical at the shadow centres), `curved` (clear and absorbing glass spheres, report only).

Runner keys (`tools/f15_check.py`, kind `transmissive-shadow`): `transmissive-shadow-{pane-sun,water-sun,volume-pane}@rt` (`--shadows rt --rt on`) and
`@csm` (`--shadows csm --rt off`), `transmissive-shadow-point@restir` (ReSTIR DI with RT visibility, unfiltered), report-only
`transmissive-shadow-curved@rt` (kind `transmissive-shadow-report`, `always_green`, reference = the declared model, `--no-ground-truth`). All gated keys
`"pending": "ticket 14"` (RT and DI) / `"ticket 15"` (CSM), `pending_matches` = `transmittance-ignored`. `--reflections off` everywhere (an SSR of the lit floor
would fill the shadow); frames 120..127 averaged (`--capture-linear-accumulate 8`).

Ground truth (`tools/f15_shadow_reference.py`, every decision in its docstring): Mitsuba 3.9.1. For each floor pixel (pixel-centre primary hit) rays leave the
receiver point inside a cone around the light, are followed through the casters with Mitsuba's own BSDFs (`dielectric`, the glTF thin-walled `gltf_thin`, the
`gltf_absorbing` Beer-Lambert wrapper of ticket 13), exact Fresnel, and a ray that escapes contributes its throughput if it hits the sun disk (angular radius
0.00465) or a stand-in sphere of radius 0.08 m around the point light; the floor and the opaque slab are black absorbers (reflected paths die: no secondary
bounce of the shadow ray). The shadowed radiance is the Appendix B floor radiance (`f15_reference.sun_radiance` / `point_radiance`, the specular lobe included:
both see the same visibility) times the RGB transmittance. Through a parallel slab the exit ray keeps its direction, so the receiver irradiance equals the
straight-line transmittance up to a lateral shift of 1-3 cm (the point light's sphere target is wider than the shift; the sun needs no margin): STATED. The
analytic DECLARED MODEL (straight shadow ray; entry surface transmission factor x tint x (1 - F), interior surfaces (1 - F), Schlick F at the straight-ray angle,
Beer-Lambert from the signed sum of entry/exit distances; opaque = 0) is implemented independently of Mitsuba's BSDFs (Mitsuba lists the crossed surfaces) and
COMPARED ONCE with the truth (`model.pfm`, `model-differences`): worst per-caster mean / p95 relative error pane-sun 0.39 / 0.82 % (csm region 0.41 / 1.04),
water 0.70 / 1.04 %, volume-pane 1.87 / 2.39 % (the straight path through the slab is 3.6 % longer than the refracted one at 20 degrees: bounded by the
attenuation chosen), point 1.11 / 2.03 %. The model passes the gate with room (`model_feasible`, a precondition of every key: otherwise exit 3).
Closed forms (tests, no Mitsuba): thin pane (1 - F_Schlick), thin tinted pane x tint, thick clear slab (1 - F)/(1 + F) with exact Fresnel (the product (1 - F)^2 of
the declaration differs by F^2/(1 - F^2) = 0.16 %), absorbing slab (1 - F)^2 a / (1 - F^2 a^2) with a = exp(-sigma_t t / cos theta_t).
Reference settings: 512 cone samples per receiver pixel in two independent halves (the noise estimate), per-depth radical-inverse (low-discrepancy) reflect/refract
choice shifted per pixel, <= 12 path vertices, sampling cone 1.0 x the sun disk / 1.3 x the point sphere (weighted by the solid-angle ratio), shadow ray origin 1 mm
off the floor, seed 1. Measured noise (relative standard error, max over RGB, from the half-sample difference): mean 0.13-0.31 %, p95 0.41-1.06 % over the eight keys.
Cost 2.4-3.6 s per key at 1280x720 (4 threads).

Gate (decision 11 transmission bound, `THRESHOLDS["transmissive_shadow"]`): per caster SHADOW REGION on the floor, mean relative error <= 5 %, p95 <= 15 % (max over
RGB per pixel), absolute floor 0.05 x the mean of the GROUND TRUTH over the whole mask (also for the controls and the pending twin: an all-black twin must not
define its own floor). The region of caster k: floor pixels whose shadow ray (towards the light centre) first hits k AND whose 8 ring neighbours at the declared WORLD
margin do too (pixels near a boundary, or behind no caster, are not gated). Envelopes declared BEFORE any measurement (constants committed in `b587e30`, first capture
afterwards, never changed): `rt` 0.06 m = sun penumbra at the caster height 3.7 mm + the shadow-mask bilateral filter (about 1.5 cm at this view) + the horizontal run of
the straight ray through the thickest slab (0.1 m x tan 20 degrees = 3.6 cm: the straight ray may leave through a side wall where the refracted one leaves through the
top; measured at 0.03 m the volume-pane p95 of the DECLARED MODEL was 12 % at the edges, at 0.06 m 2.4 %: the margin was widened from my first guess BEFORE the
envelope constants were committed and before any device run); `csm` 0.08 m = 2 texels of the cascade covering the receiver (cascade radius <= 9 m, 1024 texels:
<= 17.6 mm; PCSS disk radius max(penumbra, 0.5 texel), nearest-texel quantisation, depth/normal bias slop) + the same 3.6 cm + the penumbra. Region sizes at 1280x720: 3.1-4.0 k
pixels per caster (RT), 3.3-3.7 k (CSM).

Red state on the M5 Max (`build/f15-check/t14`, eight keys, `f15_check.py --run --expect green`: GREEN, exit 0; 0 archive misses, 0 render-thread compiles in every capture):
every gate FAILS (worst mean / p95 1.000 / 1.000: the engine's casters shadow opaquely, the truth transmits), every twin (engine vs `transmittance-ignored`) PASSES with
0 / 0 (the engine's shadowed pixels are exactly 0), the curved report passes (not gated, reported 1.000 / 1.000). Controls (all FAIL; worst caster mean / p95):
pane-sun@rt: transmittance-ignored 1.000 / 1.000, tint-ignored 1.501 / 1.508, fresnel-ignored 0.089 / 0.094 (the thick slab; the thin clear pane alone is 4.2 %, below the
bound), double-counted 0.616 / 0.616; water-sun@rt: transmittance 1.000, beer-lambert-ignored 0.396 / 0.402, double-counted 0.321 / 0.324; volume-pane@rt: transmittance 1.000,
fresnel-ignored 0.088 / 0.094, beer-lambert-ignored 0.716 / 0.726, double-counted 0.483 / 0.485; point@restir: transmittance 1.000, tint-ignored 1.492 / 1.507, fresnel-ignored
0.074 / 0.086, double-counted 0.617 / 0.619. The @csm keys give the same numbers within 0.001 (region sizes differ). Radiometry check (information): the engine's lit floor
(@rt, @restir, 700 k pixels) equals the reference's unshadowed floor radiance to 0.1 % mean (p5 0.9995, p95 1.0000), i.e. the reference floor model is the engine's.

Dropped controls (identical to the reference within the bound on the fixture): `fresnel-ignored` on water-sun (ior 1.33: 2 % per surface, 4.6 % mean < 5 %: it would PASS the
gate, so it is not listed); `tint-ignored` on water-sun and volume-pane (white base); `beer-lambert-ignored` on pane-sun and point (no attenuation there); every control on the
report-only key.

Interpretations (all declared): transmission factor and tint at the ENTRY surface only, interior crossings pure dielectric (ticket 13 orchestrator decision; the fixtures use
factor 1); a thin-walled single-quad pane has ONE Fresnel surface (glTF), a closed slab two; the specular lobe of the floor is shadowed by the same RGB visibility; reflected shadow
paths contribute nothing; the point light's reference target is a 0.08 m sphere (a delta light has no refracted path); a planar caster's lateral shift is ignored by the engine and
by the declaration; the curved key is measured against the declared straight-line model (caustics excluded) and never gated.

FINDING for ticket 15 (information, not gated): with `--shadows csm` the engine's LIT floor outside the shadows reads 0.948 of the reference on average (median 0.970, p5 0.80, 16 % of
the pixels below 0.9; the RT and DI keys read 0.999 / 1.000). The floor is itself a caster (the snapshot reader requires visible and caster roles to be equal), so this is the
CSM's own PCSS bias/noise on a large flat receiver, present before any transmittance. The ticket-15 bound (5 % mean) is applied unchanged to the shadow regions: the CSM visibility
term will have to come within the budget itself (a bias or filter fix is part of the CSM work) or the owner must interpret the bound; the harness cannot separate the two.

Not done / open: the engine side and every other criterion (spike of non-opaque intersection multiplicity, opaque casters 0 px by `visual_check`, F10/F11/F12 corpus cases with the RGB mask, Sponza D2
cost, `tools/archive_check.sh` / `hitch_check.sh` of the future pipelines, memory and cost of the CSM map); the engine's shadow noise under accumulation (today's shadows are exactly 0 in the
regions: nothing to average; the 8-frame window is a placeholder to be measured when the transport lands); GI shadow rays and reflection secondary lighting through glass have NO fixture in
this harness (the brief asked for sun, DI and CSM keys): candidates for the closure pass; `--lighting clustered` DI visibility is not keyed; what "qualitative" asserts for the curved key beyond
the numbers it prints (suggested: the shadow of a curved caster is neither black nor absent: 0 < mean transmittance < 1) is left to the owner; casters rotated off the horizontal (tilted panes) are not fixtures.

2026-11-14, ORCHESTRATOR DECISIONS on the open questions of the harness (taken before any engine result; commit `d3c4537`; they supersede the contrary
wording above: the CSM finding and the curved key):
1. CSM keys (ticket 15) gate the TRANSMITTANCE RATIO, not the absolute receiver luminance (kind `transmissive-shadow-ratio`). Per shadow region
   T = mean luminance inside the caster's shadow (envelope excluded) / mean luminance of a declared LIT reference region of the same floor next to it
   (floor pixels whose shadow ray and 8 ring neighbours at the envelope hit nothing, inside the x/z box of the shadows grown by 0.5 m), compared with the same
   ratio of the ground truth: relative error of T <= 5 % (absolute floor 0.05 x the mean T of the truth), and p95 of the per-pixel luminance / lit mean error
   against the truth's <= 15 % (same floor rule; the floors come from the truth for controls and twin). The lit-floor bias of today's CSM (a pre-existing F10
   property) is a separate NOT gated diagnostic line of the CSM keys (`lit_floor_diagnostic`). RT and DI keys keep the absolute bound.
   Measured on the existing captures (luminance basis, lit region 28-48 k pixels): CSM lit floor / truth 0.954 mean (median 0.982, p5 0.809) pane-sun, 0.977
   (0.9995, 0.879) water-sun, 0.976 (0.997, 0.879) volume-pane. Red state `--expect green`: GREEN, exit 0; the three @csm gates FAIL (T = 0 against the truth's
   0.92 / 0.96 / 0.73 for the clear, green and thick panes, 0.96 / 0.82 for water, 0.79 / 0.74 / 0.68 for the volume panes), the twins pass 0 / 0 with the truth's
   floors, and the controls fail the ratio gate: pane-sun tint-ignored 0.314 / 0.318, fresnel-ignored 0.089 / 0.096, double-counted 0.208 / 0.209; water-sun
   beer-lambert-ignored 0.171 / 0.176, double-counted 0.175 / 0.179; volume-pane fresnel-ignored 0.082 / 0.089, beer-lambert-ignored 0.347 / 0.355,
   double-counted 0.327 / 0.330 (transmittance-ignored 1.000 everywhere). The declared model passes the ratio gate on every @csm key (`model_feasible`).
   Tests: ratio gate ignores a uniform 5.2 % bias but fails an 8 % wrong shadow and the opaque engine; the absolute gate sees the bias; the lit region excludes
   shadows and envelope.
2. Curved glass (report-only key, now `transmissive-shadow-curved@rt`, kind `transmissive-shadow-curved`, pending "ticket 14"): the numbers against the declared
   model stay report-only (`numbers_gated: false`); a QUALITATIVE assertion is gated: per caster region 0.001 < T < 0.995 (lighter than an opaque caster, darker than
   unshadowed; T as above, lit region of the key) and the chroma statistic R/B = mean red over mean blue of the region follows the tint (tint = base x attenuation
   colour; expected R/B printed): within 2 % of 1 (log) for the neutral clear sphere, on the tint's side of 1 with at least half its log magnitude for the absorbing sphere
   (expected R/B 0.5). Controls `shadow-ignored` (T = 1: fails 0 < T < 1 and chroma) and `transmittance-ignored` (T = 0); today's engine (T = 0) fails, the declared model
   passes. The key therefore leaves `always_green` and becomes pending like the others (no pending_matches twin).
3. GI shadow rays, reflection secondary lighting through glass and clustered DI visibility have no fixture in this harness: recorded as ITEMS FOR THE ENGINE TICKET (14),
   no new fixture now.

- [ ] (orchestrator, 2026-11-14) engine ticket items without a harness fixture yet: GI shadow rays, reflection secondary lighting through glass, clustered DI visibility through glass
