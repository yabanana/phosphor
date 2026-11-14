# 15: Coloured transmittance map for transmissive casters (CSM)

Roadmap: F15.1, F15.3

**What to build:** The CSM path gets a single-layer transmittance map per cascade (nearest transmissive caster depth + RGB transmittance with multiplicative accumulation) so receivers behind glass are attenuated and coloured without RT (declared single-layer approximation).

**Blocked by:** 13

**Status:** resolved (2026-11-14; quiet cost → ticket 24; declared limits in the orchestrator record)

- [x] Planar fixtures with `--shadows csm` match the RT result of ticket 14 within the decision-11 transmission bound (or the CSM filtering envelope, declared before measuring)
- [x] Scenes without transmissive casters unchanged (visual_check 0 px); negative control (map ignored) fails
- [x] Memory and cost of the map per cascade measured
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-14 — test-first harness of tickets 14 and 15 (no engine or shader change; the same record is in ticket 14; this ticket owns the `@csm` keys: pending "ticket 15", envelope `csm`). Commits: `b14f650` (bench
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

2026-11-14 — ENGINE SIDE (F15.1 ticket 15; commits `7cc67b6` map, `677bbac` targeted archive update, `54324fb` runner keys gated, `839c28b` standalone
render groups, `63e3b47` divisor, `85355cb` stand-in maps with shadows off).  The four criteria above are ticked on this evidence.

Design.  Transmissive casters (KHR_materials_transmission with extension records, not alpha-masked: masked ones stay opaque-masked casters as in ticket 14)
leave the opaque cascade depth maps (`shadow_depth_vertex/mesh` reject them while `SHADOW_FLAG_TRANSMITTANCE_MAP` is set) and draw, per cascade, into ONE
RGBA32Float map in a standalone raster pass ("CSM transmittance N"; indexed `shadow_transmit_vertex` and mesh `shadow_transmit_mesh` + `shadow_transmit_fragment`,
new pipeline blend `TransmittanceMap`: RGB add, alpha max, key token `tmap`).  RGB = the SUM over every crossed surface of the ln RGB transmittance, alpha =
the NEAREST transmissive depth (reverse-Z maximum, 0 = none).  Per surface the math is the same function as the ray-traced any-hit (new
`transmissionShadowSurfaceLn` in `transmission_math.h`; `rt_intersections.metal` now calls it: pixel-identical RT captures, below): ln(1 - F) at the straight
ray's |cos| (geometric normal from the screen derivatives of the world position), the entry factor x (1 - metallic) x tint once per caster (entry of a solid or
the only surface of a thin-walled material), and for a solid the signed medium term +sigma_t d (entry) / -sigma_t d (exit).  d is measured from the cascade FAR
plane (`shadowTransmitDistance`: it grows towards the light like the distance of a ray from a receiver below the volume), so a closed volume cancels the
origin and a receiver behind it gets -sigma_t (t_exit - t_entry) exactly (test).  Entry/exit comes from `[[front_facing]]` x the instance's MIRRORED flag
(`shadowTransmitEntry`: the surface faces away from the light = a back face in the light's view = entry; checked on the slabs with attenuation, a wrong sign
gives T clamped to 1).  The receiver (`shadow_csm_pcss`) multiplies its opaque PCSS visibility by the mean over the same disk of exp(min(sum, 0)) of the taps
whose texel's nearest transmissive depth lies in front of the receiver (the declared single-layer approximation: the nearest depth decides "behind", every
layer in the texel contributes its factor like the ray-traced sum); cascades blend like the opaque term; the mask is RGB (the layout of ticket 14).  The passes
exist in the graph only while a transmissive caster exists (`graphVersion` bump when the answer changes; 54 passes / 15 render passes with the map against
50 / 11 without on the fixtures; no empty encoder); 1x1 zero stand-in maps keep the kernel's bindings valid otherwise (also with `--shadows off`, where the
PCSS pass exists).  `--debug-transmittance-map on|off` (off: the receiver ignores the map: glass casts nothing), `--shadow-transmittance-divisor 1|2|4` (default 2).

Findings, in the order they were found.  (1) CSM acne on a tilted floor that is also a caster: the first run of pane-sun@csm gave mean 0.6 % but p95 17.5 % (limit
15 %): the lit floor read 0.954 of the reference with ROWS of 0.82-0.95 along the shadow-map texel grid (a PCSS tap at up to half a texel plus the nearest-texel
quantisation moves the stored depth by about one texel x slope, the bias covers a quarter texel x slope).  Fix inside `shadowPcss`: with the map, the FILTER taps use a
receiver-plane bias (filter radius + a quarter texel) x slope / depthRange; the lit floor is then 0.9999 (median 1.0000, p5 0.9995) on all three keys.  DECLARED
DEVIATION: the extra bias is applied ONLY while `SHADOW_FLAG_TRANSMITTANCE_MAP` is set, because the criterion demands that scenes without transmissive casters are
bit-identical; promoting it to every CSM scene (the `filterBias` condition) would change every CSM reference of F10.  (2) The render graph fused the colour-only
transmit pass with the opaque cascade depth pass: validation (pixel format mismatch of the opaque pipelines against the encoder's colour attachment) and, on the
indexed path, the opaque depth state leaking into the transmit draws (the lower face of a closed slab lost to a depth test).  `PassBuilder::setStandalone()` (new,
tested) gives the pass its own render group.  (3) A ReSTIR scene with `--shadows off` failed to compile the graph (the PCSS pass exists there; stand-in maps).

Runner (`tools/f15_check.py`): `pending` / `pending_matches` removed from the three @csm keys, no gate, threshold or envelope changed; the python harness test expects
every key gated.  Final binary, captures of the full run `build/f15-check/t15` (112 keys captured serially under the GPU lock), shadow keys evaluated key by key
(same `evaluate_scenario` / `verdict` as the runner): GREEN.  Worst caster (limits mean 0.05 / p95 0.15): pane-sun@csm 0.0016 / 0.0081, water-sun@csm 0.0043 / 0.0093,
volume-pane@csm 0.0091 / 0.0148 (the opaque slab exactly 0 / 0); lit-floor diagnostic 0.9999 on all three (not gated).  Controls (all FAIL; worst mean / p95):
pane-sun transmittance-ignored 1.000 / 1.000, tint-ignored 0.314 / 0.318, fresnel-ignored 0.087 / 0.094, double-counted 0.208 / 0.209; water-sun transmittance 1.000,
beer-lambert-ignored 0.170 / 0.176, double-counted 0.176 / 0.179; volume-pane transmittance 1.000, fresnel-ignored 0.082 / 0.088, beer-lambert-ignored 0.346 / 0.353,
double-counted 0.327 / 0.331.  The @rt / @restir / @curved keys stay green on the same binary (pane-sun@rt 0.0032 / 0.0080, water@rt 0.0080 / 0.0117, volume@rt
0.0205 / 0.0259, point@restir 0.0111 / 0.0203, curved qualitative PASS).  Map-ignored negative control (`--debug-transmittance-map off`, evaluated against the same
references, EXPECTED_RED_CONFIRMED): pane-sun worst 0.369 / 0.373, water-sun 0.219 / 0.224, volume-pane 0.460 / 0.469.  The indexed path (`--force-family apple9`) was
checked on volume-pane (T per slab 0.787 / 0.736 / 0.679 against the truth's 0.790 / 0.742 / 0.685), and the three gates are identical at divisors 1, 2 and 4.

Whole-runner accounting (112 keys captured, 95 evaluated key by key): every evaluated key meets its expectation (pending keys fail as asserted) except that ONE key
blocks `--expect green`, `combo-irid-aniso-env` (ticket 19's harness: its `additive-layering` control does not fail the gate, BLOCKED_NON_DISCRIMINATING_FIXTURE, exit 3),
and `transmission-rtoff-combinations` cannot be captured (13 archive misses: specialised forward / resolve variants missing from the archive).  Both are identical on the
phase-head binary (13 misses with `build/base/phosphor`; the `combo-irid-aniso-env` capture is pixel-identical to the phase head's).  The eight ticket-13 Mitsuba keys
(`transmission-roughness-ior`, `-attenuation`, `-factor-tint`, `-thin-volume`, `-single-sided`, `-glass-behind-glass`, `-emissive-panel`, `-rtoff`) were NOT re-evaluated
by me: their references take hours of CPU on the loaded machine (one ran 45+ min before I stopped it); their captures are pixel-identical (0 px, max 0.0) to captures of
the phase-head binary, so their verdicts are the phase head's.  The closing battery of the integrated tree owns the literal single `--expect green` over every key.

Unchanged-scene checks.  `tools/visual_check.sh build/release build/reference-f15-appB` (copy of the orchestrator's references): benches 1-7 0 px; bench 8 differs from
the reference by default arguments, not by this change (the phase-head binary gives the same 5 592 801 px; with `BENCH8_ARGS="--instances 100000"`, the value the
reference was taken with, 0 px).  Captures against `build/base/phosphor` (the phase-branch head `b618511`), bit-identical (0 px, max abs 0.0): shadow-penumbra CSM
(contact on and off), CSM with `--shadow-cache on`, CSM `--force-family apple9` (indexed casters), base-grid-sun CSM, shadow-penumbra RT, volume-pane RT shadows (glass:
the refactored shared function), the ReSTIR point scene through glass with `--shadows off` (the run that found failure 3).

Pipelines and the archive.  Targeted update: harvest of the pane-sun@csm capture (`--harvest-pipelines`), `filter_pipeline_harvest.py --constant 22 --function
shadow_transmit_{vertex,mesh,fragment} --skip-present`, `merge_pipeline_harvests.py`: +1 render +1 mesh descriptor (253 / 142 -> 254 / 143).  Both pipelines are requested
at startup in the non-cache CSM path whether or not the scene has glass (archive hits).  `tools/archive_check.sh build/release build/reference-f15-appB`: all scenarios
PASS (full: 18 hits, 0 misses); `tools/hitch_check.sh build/release`, also with `--render-path visibility --geometry-path mesh --shadows csm --post --upscaler native`
(67 requests, 67 hits): 0 hitches, 0 render-thread compiles.  Warmup 64 / 128 frames on the three fixtures: 0 GPU allocations, 0 render-thread compiles, 0 archive
misses.  Shader + API validation (`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, volume-pane@csm mesh path, water-sun@csm indexed path, 128 frames each): exit 0, 128 LIGHTING
checks PASS each; the only messages are the "Redundant call to setDepthStencilState / setRenderPipelineState / setViewport" notices, 1280 per capture exactly as with the
phase-head binary, and the 31 flexible-pipeline "blending substate" notices (also identical).  `tools/f10_f12_check.py --run` (26 jobs under validation): every job its
expected exit, 25 FUNCTIONAL_RESULT_ONLY and `f10_solar_bias_negative` QUALITY_CONTROL_PENDING as before (one overlap: another agent's phosphor process was running when
it started; no failure).  Unit tests 883 / 883, ctest 19 / 19.

Memory (graph accounting, `report.json` `graph.heap_unaliased_bytes` and `heap_bytes`, volume-pane at 1280x720, default `--shadow-map-size 2048`; baseline = the phase head
with opaque glass: 295 960 576 / 121 241 600): divisor 1 (RGBA32Float 2048^2) +269 484 032 / +269 484 032 = 64.25 MiB per cascade; divisor 2 (default, 1024^2) +67 371 008 /
+67 371 008 = 16.06 MiB per cascade (the size of one opaque depth map); divisor 4 +16 842 752 / +16 842 752 = 4.02 MiB.  A scene without glass allocates a 1x1 RGBA32Float
(16 B) per process and nothing else.  Cost (LOADED machine, load average about 10: indicative only; per-pass p50 GPU ms on volume-pane at 1280x720, four cascades summed):
transmit passes 0.33 ms at divisor 1, 0.17 at 2, 0.08 at 4; the opaque cascades get cheaper (0.29 -> 0.12-0.19 ms: the glass left them); PCSS 0.35-0.56 ms (noise
dominates; baseline 0.43).  Frame totals are meaningless under this load; the quiet measurement is the orchestrator's.

Declared limitations / open.  (a) `--shadow-cache on`: transmissive casters stay opaque there (logged once; the static cache composes opaque depth only).  (b) The
froxel fog with CSM reads the opaque maps: glass no longer blocks fog light in CSM mode (in RT mode the fog rays treat glass as opaque: declared in ticket 14); a fog
that reads the map is not built.  (c) A receiver that is a glass surface itself (the underside of a slab) lies "behind" its own top surface and is multiplied by the slab's
transmittance: only its (negligible) direct term changes.  (d) Single layer: a receiver between two separated glass layers gets both (the nearest depth decides).  (e) The
receiver-plane bias is gated on the map (deviation of finding 1): the owner decides whether every CSM scene gets it.  (f) The edge quality of the half-resolution default is
not gated (the envelope excludes the edges); divisor 1 is one flag.  (g) The mesh and indexed transmit pipelines are requested even without glass (two archive hits).

2026-11-14 — orchestrator: integrated on `cl/f15-plan-c38b51` as f0d8bc0…140a96e, with no conflicts. Archive: +1 render, +1 mesh descriptor.
After the merge: build OK, ctest 20/20.
- Accepted as declared:
  - The receiver-plane bias applies only while a transmittance map is active, so scenes without glass stay bit-identical.
    Extending it to every CSM scene would change every F10 reference. Not done; it is an OPT/EDGE candidate if a measured need appears.
  - With `--shadow-cache on`, glass stays opaque.
  - The froxel fog does not see glass with CSM.
  - Two separate glass layers above a receiver both attenuate it (single-layer map).
  - Default divisor 2 (16 MiB per cascade at 2048^2), measured. The `@csm` gates were identical at divisors 1, 2 and 4.
- The cost on a quiet machine goes to ticket 24.
- The agent's two blocked all-keys items were fixed on the branch:
  - `transmission-rtoff-combinations` got a targeted archive update (83c0697) and is gated (322020d).
  - `combo-irid-aniso-env` dropped its non-discriminating control (7a59d8c).
- Orchestrator mid-phase battery (2026-11-14, `build/f15-exact/battery_mid.log`):
  - O7 on CSM: 0 allocations and 0 render-thread compiles on the five shadow scenes;
  - shader + API validation of pane-sun with CSM: clean;
  - visual_check 0 px; archive_check PASS; hitch_check OK.
