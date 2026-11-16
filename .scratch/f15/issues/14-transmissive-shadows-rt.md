# 14: Attenuated coloured shadows through transmissive casters (ray tracing)

Roadmap: F15.1, F15.3

**What to build:** Owner decision 2026-11-13: shadow rays through transmissive surfaces accumulate an RGB transmittance instead of stopping — per-surface factor from transmission, tint and Fresnel, plus Beer–Lambert from the signed sum of entry/exit distances per medium (order-independent, so any-hit traversal needs no sorting); straight line, caustics excluded (declared). Used by the RT sun shadow mask (scalar → RGB), DI visibility, GI shadow rays and the reflection secondary lighting.

**Blocked by:** 13

**Status:** resolved with carried items (2026-11-16, round 2: the closed-volume interior at the light end (fixture `volume`, B/A 0.6446 / 0.8254 / 0.9570 vs 0.6429 / 0.8243 / 0.9565 expected, tip control fails) and the transmissive + alpha-masked caster (fixture `masked`, right ROI 0.8973 / 0.4985 / 0.2990 vs tint 0.9 / 0.5 / 0.3, tip control fails) are closed by the transport agent's engine change; dedup list kept at 8 surfaces; clustered DI through glass closed; owner exclusions named in the support table (dcfb3ea): (a) AO, fog, ReSTIR reconnection and sky rays see glass as opaque, (b) secondary RT hits use the base reduction; still without a numeric fixture: GI shadow rays and reflection secondary lighting through glass; Sponza D2 band recorded as met in ticket 24 under owner option A (payload cost fixed by ec63efc / 8fbda9c / 720ee63), RT transmissive-shadow frame cost +0.148 ms on the chess shadow rig recorded in ticket 24 as a cost, and the quiet-window cost of the +8 payload bytes on glass / feature scenes is pending the orchestrator's final battery)

- [x] Spike first (timeboxed, results in opt-log): Metal intersection-function invocation multiplicity for non-opaque triangles (duplicate invocations would double-count) and cost; mitigation chosen by measurement
- [x] (RT sun and ReSTIR point keys; the CSM keys are ticket 15) Planar-interface fixtures (glass pane, flat shallow water with sun within 30° of the normal) vs Mitsuba within the decision-11 transmission bound; curved-glass shadows (pawn tops) qualitative (caustics excluded)
- [x] Opaque casters unchanged (visual_check 0 px on scenes without transmissive materials); negative control (transmittance ignored → opaque) fails the planar fixtures
- [x] Sun shadow mask format change and DI/GI visibility changes keep F10/F11/F12 corpus cases green
- [ ] Cost on the glass fixture and Sponza D2 within band — ticket 24 (quiet machine); indicative loaded-machine pass time +0.12 ms on volume-pane — D2 band not met at HEAD (median p95 +0.23 ms vs 2e1057a; owner decision on the 26fe229 continuation cost pending, ticket 24)
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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

2026-11-14 — ENGINE SIDE (F15.1, commits `36a4392` math, `055c5dd` engine, `381947c` runner; spike in `docs/opt-log.md` "F15 ticket 14 spike").

Design. A new ray type `RT_PROBE_SHADOW_TRANSMITTED` (any-hit): the sun kernel, `diEndpointTransmittance` (ReSTIR and clustered DI, reflection secondary
lighting) and `giShadowTransmittance` use it; AO, fog and the reconnection/sky rays keep `RT_PROBE_SHADOW` (declared: opaque glass).  Transmissive
instances are non-opaque in the TLAS (`rt_write_instances`: feature bit TRANSMISSION), so `rt_alpha_generic` sees their triangles: every other ray type
accepts them at once (no counter change: `opaqueAlphaTests` still counts opaque instances only), a transmittance ray accumulates in log space into the
payload (`lnTransmittance`, zero = 1: order independent) and REJECTS the triangle, so traversal continues to the light; an opaque or alpha-masked caster
accepted ends the ray (hit -> T = 0).  Per crossed surface: ln(1 - Schlick F) at the straight-ray |cos| with the geometric normal (ior = ext0.ior with the
ior feature, else 1.5; ior 0 = no Fresnel); once at the ENTRY surface (front-facing hit of a solid, or the single surface of a thin-walled material:
volume feature absent or thickness 0): transmission factor (x texture) x (1 - metallic) x baseColor tint (x texture); solid media: +sigma_t t at an entry,
-sigma_t t at an exit with sigma_t = -ln(attenuationColor)/attenuationDistance per channel (sum = -sigma_t (t_exit - t_entry), clamped T <= 1).  The IFT
gets the extension buffer at index 7 (`TraceResources::extensions`, the materials buffer while the scene has no extended material); `GPURtParams`
grows to 32 bytes (`transmittanceFlags`).  Per-primitive duplicate filter: 8 `(slot, primitive)` pairs per ray (the spike measured multiplicity 1.000;
kept as insurance, controls `--debug-transmittance-dedup off|on`, `--debug-transmittance-inject`).  The sun mask is RGB end to end: raw RGBA16F,
temporal history 80 bytes (RGB mean next to the luminance scalar the variance estimator uses; the value itself while the channels are equal, so grey
scenes are bit-identical), filter and contact on three channels with the variance in `.w`, `shadeSurface` and the overflow forward path take a `float3`
sun visibility.  Channel 0 of the captured `shadow` signal is still the scalar visibility (F10 quality tools read it); `.y/.z` (moment, samples) became
`.y/.z` = G/B, variance `.w`.  CSM writes grey (ticket 15 adds the transmittance map).  Counters: `GPUShadowCounters` +4 words (rays through glass,
invocations, distinct, overflows), printed as `SHADOW-TRANSMITTANCE` with `--debug-lighting`.

Runner (`f15_check.py --run --expect green --only <5 keys>`, `build/f15-check/t14a`, GREEN, exit 0; worst caster mean / p95, limits 0.05 / 0.15):
pane-sun@rt 0.0037 / 0.0080 (controls transmittance-ignored, tint-ignored, fresnel-ignored, double-counted all FAIL); water-sun@rt 0.0077 / 0.0113
(transmittance-ignored, beer-lambert-ignored, double-counted FAIL); volume-pane@rt 0.0199 / 0.0253 (transmittance-ignored, fresnel-ignored,
beer-lambert-ignored, double-counted FAIL); point@restir 0.0111 / 0.0203 (transmittance-ignored, tint-ignored, fresnel-ignored, double-counted FAIL);
curved@rt qualitative gate PASS (0 < T < 0.995 on both spheres, chroma R/B follows the tint; shadow-ignored and transmittance-ignored FAIL; the numbers
against the declared model are 0.0018 / 0.0061 worst, report only).  `pending` / `pending_matches` are gone from those keys; the three @csm keys keep
"ticket 15".  0 archive misses and 0 render-thread compiles in every capture (no new pipeline descriptor: the new code lives in existing kernels, so no
targeted archive update was needed).

Other checks run: unit tests 869 / 869 (`phosphor_tests`) and ctest 19 / 19 except the shadow harness python test updated for the landed keys (green);
clustered DI through glass: `transmissive-shadow-point` with `--lighting clustered` equals the gated ReSTIR capture to 1.5e-7 absolute (245 829 pixels,
max abs 1.49e-7, none above 0.1 %); pixel identity without glass: six captures (shadow-penumbra RT, CSM, RT + contact; specular-grid-point ReSTIR + RT
reflections; clustered; base-grid-sun RT sun + DDGI + RT reflections) against the orchestrator tree's binary: 0 pixels differ, max abs 0.0 in all six;
shader + API validation (`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, pane-sun, RT sun + ReSTIR + DDGI + RT reflections, 24 frames): exit 0, 24
LIGHTING checks PASS; F10-F12 tester runner (`tools/f10_f12_check.py --run`, 26 jobs under validation): every job its expected exit (negative controls
exit 1), 0 failures, `f10_solar_bias_negative` stays QUALITY_CONTROL_PENDING (image control, as before).

Not done / open: GI shadow rays and reflection secondary lighting through glass compile, run clean under validation and pass the identity checks but
have NO numeric fixture (the harness has none; only the shared kernel of the DI keys is gated); cost on the glass fixture and Sponza D2 (ticket 24):
orchestrator on a quiet machine (my in-process pass measurement is in the opt log and is not quiet); `visual_check`, `archive_check`, `hitch_check`: not
run by me; the interior of a closed volume at the light end (a ray that ends inside a medium) is clamped T <= 1; a transmissive material that is also
alpha-masked is treated as an opaque-masked caster (declared); the dedup list is 8 surfaces (beyond it the filter is off and counted).

2026-11-14 — orchestrator mid-phase battery on the integrated tree (tickets 13, 14, 15, 17 and the ticket-19 harness; `build/f15-exact/battery_mid.log`):
- visual_check: all benches pass at 0 px (BENCH8_ARGS --instances 100000).
- F13 exact captures: 0 px against base03 (sponza-d2, cornell-restir, cornell-clustered-ssr, manylights-restir, grid-sky).
- archive_check: every local scenario PASS. hitch_check: no hitches.
- O7 (warmup 64): 0 GPU allocations and 0 render-thread compiles on all five transmissive-shadow scenes, on RT shadows + ReSTIR and on CSM.
- Shader + API validation of pane-sun, RT + DDGI + RT reflections and CSM + SSR: clean, exit 0.
- Negative controls: transmittance-ignored fails every RT/DI key (agent run, t14a).
- Ticked: opaque casters unchanged, and the archive/O7 criterion. The ticket adds no pipelines; the existing kernels run with 0 misses.

2026-11-15 — closure audit (ticket 24):

- 1 spike: MET (opt-log 2230, multiplicity 1.000, dedup list of 8 kept as insurance). 2 planar fixtures: MET (pane-sun 0.0037/0.0080, point@restir 0.0111/0.0203 vs 0.05/0.15; re-green after the payload fix). 3 opaque casters: MET (0 px, control fails). 4 F10/F11/F12 corpus: MET (26 jobs). Ticked.
- 5 cost + D2: NOT MET (only an in-process loaded figure +0.12 ms; D2 regression from the c59c5af payload fixed by ec63efc/8fbda9c; residual vs 2e1057a is the 26fe229 continuation, not this ticket). Open, ticket 24.
- 6 archive/hitch/O7: MET (no new descriptor; later +4 compute descriptors with the payload follow-up).
Stray checkbox line removed: clustered DI visibility through glass is CLOSED (clustered equals gated ReSTIR to 1.5e-7, 245 829 px); GI shadow rays and reflection secondary lighting through glass remain without a numeric fixture (declared).
Open (declared in the ticket, not in 24): AO, fog, reconnection/sky rays treat glass as opaque; interior of a closed volume at the light end clamped T <= 1; transmissive + alpha-masked = opaque-masked caster; dedup list 8 surfaces.

2026-11-15 — CLOSURE (agent "transport", branch f15c-transport-2): interior of a closed volume, transmissive + alpha-masked caster, dedup decision.

Fixtures (generated glTF, corpus rig, linear HDR PFM averaged over 16 frames, `tools/f15_transport_check.py {volume|masked|thin-backlit}`, unit tests `tools/test_f15_transport_check.py`
in ctest as `f15_transport_check_python`; each fixture has a control binary = the tip 3ef8f41 build that must fail the gate, and model controls that must fail on the fixed build):
- `volume` (the "ends inside a medium" clamp): wall, closed absorbing glass cube (ior 1.5, attenuation (0.6, 0.8, 0.95) at 0.3 m, solid), an emissive quad INSIDE the cube (ReSTIR DI, RT
  visibility).  Arms A = no attenuation, B = attenuation, C = emitter off; D = capture - C cancels sun and environment, B/A cancels the entry Fresnel factor.  Expected per ROI pixel (10 218
  pixels per ROI, left and right of the cube): T = mean over a 5x5 grid of the emitter of exp(-sigma_t |L_s - E_s|), weighted by the measured D_A.  Fixed build (x1, probes included): B/A =
  0.6446 / 0.8254 / 0.9570 against 0.6429 / 0.8243 / 0.9565 expected (relative errors 0.26 / 0.13 / 0.05 %, bound 5 %); left ROI the same within 0.2 %.  Tip build (control binary): B/A = 1.0418 in
  every channel, i.e. 1 / (1 - F): the clamp T <= 1 lost the open medium (fails the gate by 62 / 26 / 9 %).  Model control `clamp-model` (expect B/A = 1) fails on the fixed build by 35.5 / 17.5 / 4.3 %.
- `masked` (transmissive + alpha-masked caster): thin-walled glass pane (ior 1, tint (0.9, 0.5, 0.3)) with an RGBA base-colour texture, alpha 0 on the left half and 1 on the right, alphaMode MASK
  cutoff 0.5, sun with RT shadows; arm O = the same pane opaque, T_k = (M_k - O_k) / (lit - O_k) cancels the ambient.  Fixed build (final binary): left (texel below the cutoff) T = 0.9996 (expect 1),
  right (above the cutoff) T = 0.8973 / 0.4985 / 0.2990 (expect the tint 0.9 / 0.5 / 0.3: 0.3 % error each).  Tip build: left 0.9996, right 0.000 (the accepted masked texel was an opaque
  blocker): fails.  Model controls `opaque-model` and `ignored-model` fail on the fixed build (right ROI).
Engine: `rt_alpha_generic` sends a transmissive material with `alphaCutoff > 0` hit by a `RT_PROBE_SHADOW_TRANSMITTED` ray through the mask test: a texel below the cutoff is rejected, a texel at or
above it accumulates its transmittance and the ray continues (other ray types keep the accepted-hit rule).  The payload gains `openSigma[3]` (+sigma_t at a solid's entry, -sigma_t at its exit,
`transmissionShadowSurfaceOpen`); `rtShadowTransmittance(payload, endDistance)` adds -endDistance x max(net open, 0) per channel to the ln sum before the T <= 1 clamp: a segment that ends inside a
medium is attenuated over the rest of its length; a segment that starts inside one needs nothing.  `diEndpointTransmittance` (ReSTIR, clustered DI, reflection secondary lighting) and
`giShadowTransmittance` pass the segment length; the SUN does not (its ray never ends inside a medium: unchanged behaviour).  Declared: a volume mesh is closed (KHR_materials_volume); an OPEN mesh with
`thickness > 0` used as a local-light caster now attenuates over the whole segment instead of being clamped; nested media with different sigma use the net sum.  Unit test
`RT transmittance: a segment that ends inside a closed volume...` (tests/test_shadows.cpp).
Dedup list: KEPT at 8 (spike in docs/opt-log.md: no measurable benefit of a shorter list on the glass arm, Apple9 evidence pending); the dead `noDedup` word was removed (payload 136 -> 132 bytes with open sigma).
Identity: the five F13 exactness captures (sponza-d2, cornell-restir, cornell-clustered-ssr, manylights-restir, grid-sky) are bit-identical (`cmp` of the PFMs) to `build/litefix/exact-new` with the final binary.
Open / for the orchestrator: quiet-window cost of the +8 payload bytes on the glass arm and on feature scenes with the full payload (A/B commands in the opt-log entry); `visual_check` and archive checks not run by me
(no descriptor changed, no archive update needed).

2026-11-16 — GPU evidence on the integrated tip (branch f15c-transport-3 = 06dd3a7 + 3 commits; binary linked by hand because the pipeline archive step fails in this build tree, see below).

- Glass weights (ticket 13), `tools/f15_transport_check.py glass-weights`: rough ior-2 glass sphere (roughness 0.2, alpha 0.04), `--capture-linear-signal specular`, 1 sample per pixel, denoiser off, 4 seeds,
  1024x768, environment 0.5; counted pixels whose raw radiance exceeds 0.5 x 1.2.  Fix: 0 / 0 / 0 / 0 (peak 0.5002 = the environment, 180 729 lit pixels per frame).  Tip binary (control): 1302 / 1275 / 1289 / 1303
  pixels, peak 7.82 (a weight of 15.6).  Negative control, same counter at half the environment: 180 620 pixels on the fixed build (the statistic sees the sphere).
- Ticket 16, `thin-backlit`: the thin diffuse-transmission quad lit only by an emitter behind it (ReSTIR DI + RT visibility) is lit 3.83x its unlit value; with an opaque occluder between emitter and quad the
  light drops to exactly 0.000 of it: the RT visibility ray does not self-occlude the back-lit thin receiver (DI path; the sun cannot back-light a camera-visible receiver in the corpus rig, RT sun origin is on the light
  side by code).  There is no visibility-off arm (`--rt off` is refused with restir DI): the occluder is the control.  CSM self-shadowing stays with the thinwall work.
- Exactness: the 5 F13 captures (sponza-d2, cornell-restir, cornell-clustered-ssr, manylights-restir, grid-sky) with the final binary are `cmp`-identical to the same 5 captured with the orchestrator's tip binary.
  `visual_check` (EXTRA_ARGS=--pipeline-sync, BENCH8_ARGS="--instances 100000", references build/reference-f15-appB): all 8 benches 0 pixels, max delta 0.
- Archive: `metal-tt` ends with "applegpu-nt: Internal compiler error ... plugin wrapper-nt.dylib cannot emit pipeline" (no descriptor named) on every `phosphor-archive.metallib` step in this worktree: with my changes (3 runs),
  with all my shader changes reverted to the tip (2 runs, at fa6786d and at 06dd3a7), identical pipelines.mtl4-json to the orchestrator's (cmp).  I linked `phosphor` from `ninja -t commands` without the archive and ran with
  "archive hits 0, unavailable 136" (every pipeline compiled at start, `--pipeline-sync`): fallbacks 0.  The archive ICE is therefore environmental to this tree, not caused by this work; the orchestrator's archive build of the same content succeeds.
