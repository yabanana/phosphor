# 13: Specular transmission and volume attenuation (ray-traced)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_transmission` + `KHR_materials_volume` per ADR-0004: Fresnel-chosen reflect/refract sample, rough GGX BTDF, always traced with RT whatever the reflection band, interface traversal that accepts the inside back faces of the transmissive medium (single-sided closed meshes included), ≤ 4 crossings, Beer–Lambert along the real inside path (a medium entered at a front face ends at any hit), thin-walled without offset; probes/environment along the refracted ray without RT; refract lobe id/axis/IOR-scaled roughness to the denoiser; emitters hit by refracted rays owned by the transport, delta lights by the thin-walled DI BTDF.

**Blocked by:** 07, 08

**Status:** resolved with carried items (2026-11-14; temporal clip PARTIAL: smooth glass fails F13.6 under orbit (refracted-image reprojection), carried to ticket 24 and the EDGE/F16 list; Khronos pages → ticket 22; Sponza D2 band and quiet cost → ticket 24)

- [x] Transmission fixtures (factor, roughness × ior, thin vs volume, attenuation) vs Mitsuba 3.9.1 rough dielectric (same roughness) + homogeneous absorbing medium on the glass mask, with nothing behind the glass lit through glass: decision-11 transmission bound
- [x] Single-sided closed glass mesh renders the exit correctly; negative control keeping today's back-face rejection (`reflections.metal:21`) fails
- [x] Glass behind glass is not black; crossings never exceed the bound; command buffers far below 1 s
- [x] `--rt off`: image compared with the declared probes/environment approximation (image check + negative control), and the support report says so
- [x] Emissive panel behind thin glass: no double count
- [ ] Ray budget per pixel unchanged on non-transmissive scenes (Sponza D2 within band); extra cost on the glass fixture measured — non-transmissive F13 captures 0 px; D2 band and quiet cost → ticket 24
- [ ] Temporal clip (orbit, moving light) within F13.6; transmissive pixels reactive for MetalFX temporal — reactive proven in both paths; clip FAILS on the gated smooth ROIs (p95 0.14-0.17): partial, carried
- [x] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [x] TransmissionTest, TransmissionRoughnessTest, TransmissionThinwallTestGrid, AttenuationTest side by side — ticket 22 (done 2026-11-14, qualitative; defect C22-D4 (dark squares behind glass in TransmissionRoughnessTest) FIXED in the ticket 22 follow-up: the wall behind the glass lacked the environment diffuse term in the sun shadow of the label panel, not a back-face hit; `docs/research/2026-11-14-f15-khronos-corpus-report.md`)
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-14 — test-first harness (no engine or shader change besides test infrastructure). Commits: `39a6329` (`--capture-linear-accumulate`),
`feafa7b` (bench), `86d0c86` (reference, gate, keys), `dd79fc3` (attenuation fixture), the pooled-statistic commit, the archive commit
(+32 compute, +11 render, +3 tile descriptors, MATERIAL_FEATURES; compute 410 -> 442, render 194 -> 205, tile 9 -> 12, mesh 142), the control fix.

Orchestrator decisions (2026-11-14, recorded as such):
1. The transmission factor applies at the camera-visible ENTRY surface only; interior crossings (exit and any further interface of the closed medium)
   are pure dielectric interfaces with factor 1 (declared approximation; the reference implements exactly this: image(t) = (1 - t) I0 + t I1).
2. Gate per glass object (mean <= 5 %, p95 <= 15 %, floor 0.05 * mean over the whole ROI) is accepted (stricter than the pooled mask); the pooled
   glass-mask statistic is reported for information (`pooled_not_gated`).
3. `--rt off` declared model: path length = the volume thickness hint (thicknessFactor times the instance scale along the thin axis, world units).
   EXACT formula (per pixel, metallic 0, constant environment L, (A, B) = DFG table at (N.V, roughness), f0 = ((ior - 1) / (ior + 1))^2):
   E = clamp(f0 A + B, 0, 1);  T = exp(-sigma_t * thickness_world), sigma_t = -ln(attenuationColor) / attenuationDistance (T = 1 for thin-walled, thickness 0,
   or no attenuation);  L_pixel = L * ((1 - E) * ((1 - t) * baseColor + t * baseColor * T) + E).  Implemented in `declared_pixel`,
   tools/f15_transmission_reference.py; gate = the transmission bound per object.

Fixtures (`src/testbench/transmission_validation.*`, separate bench class on `--material-scenario`; `tests/test_transmission_validation.cpp`, 842 doctest
cases; mirrored by `tools/f15_transmission_fixtures.py`): constant environment 0.5, a purely EMISSIVE backdrop (checkerboard of 0.4 m cells, radiance 2, 192 cells at
z = -3; or panels radiance 3) with nothing diffuse behind the glass (gating policy), camera 60 deg, 1280x720. Scenarios:
`transmission-roughness-ior` (12 clear volume spheres, roughness {0,.1,.3,.6} x ior {1.33,1.5,2}), `-attenuation` (6 spheres, colour x distance {.6,1.2,3},
roughness .1), `-factor-tint` (4 thin panes (factor {1,.5} x tint) + 4 yawed volume slabs, white base), `-thin-volume` (thin sphere, clear and absorbing volume
sphere), `-single-sided` (doubleSided = false closed meshes: lens sphere, yawed cube, absorbing sphere), `-glass-behind-glass` (sphere behind sphere, pane
in front of an ior 2 sphere), `-emissive-panel` (clear and tinted thin pane in front of emissive panels, ReSTIR DI), `-rtoff` (no backdrop, `--rt off --reflections probes`).

Ground truth (`tools/f15_transmission_reference.py`; every decision in its docstring): Mitsuba 3.9.1, pixel-CENTRE rays through `integrator.sample` (path
integrator; Mitsuba's film would jitter inside the pixel and blur the refracted checkerboard), only glass-mask pixels (primary hit on glass, 2 px silhouette
erosion), max depth 64, no Russian roulette (unbounded crossings), 1024 spp. Solids: `dielectric` / `roughdielectric` GGX (alpha = max(clamp(r, .04, 1)^2, .002)),
int_ior = ior, specular transmittance = baseColor, interior absorption exp(-sigma_t * length) by a BSDF wrapper (volpath's medium kills paths at random: 5-25 %
noise; `--medium stochastic` kept as oracle: it agrees with the wrapper within 2 % in the mean). Thin-walled: glTF single Fresnel sample per surface (Schlick
at N.V, reflect or straight transmission x baseColor) as a small Python BSDF, NOT `thindielectric` (slab inter-reflection: T = (1-F)/(1+F), 4 % darker per
surface). Rough thin-walled glass is refused (no ground truth; open).
Validation of the reference itself: an independent analytic sphere tracer (exact expectation of the Fresnel tree) agrees with Mitsuba to 0.2 % per channel mean
in a uniform environment, median per-pixel error < 3 % with the checkerboard (a few pixels on sharp edges differ by sub-pixel tessellation shifts).
Measured reference noise (relative standard error of the mean, max over RGB, 1024 spp): roughness-ior 1.7 % mean / 3.1 % p95 (the rough rows dominate),
attenuation 1.5 / 2.4 %, factor-tint, thin-volume, glass-behind-glass, emissive-panel 0.6-0.7 / 1.4-1.7 %. Rendering cost 10-20 min per variant at 4 threads.
Model differences glTF vs Mitsuba (`model-differences`, analytic, smooth spheres, vs exact unbounded Fresnel): Schlick Fresnel 0.3-0.7 % mean / 1.4-3.7 %
p95 on clear spheres (ior 1.33 / 1.5 / 2); the 4-interface bound <= 0.2 % (both ends: env or black); an extreme attenuation (distance .3, blue .15) raised
the Fresnel difference to 7.9 % / 18.9 %, so the attenuation fixture was moderated (<= 1.2 % / 3.1 %). Rough solids (Walter vs Schlick + separable Smith):
not measurable analytically, absorbed by the bound. Tint applies at every transmitting interface (baseColor^2 through a closed body), so the gated volumes have
a white base and the tinted fixtures are single-crossing panes.

Accumulation capture: `--capture-linear-accumulate N` (CPU float64 mean of the N frames ending at `--capture-linear-frame`, window validated). The runner uses 32
(frames 96..127). Residual noise measured on today's engine (opaque glass, near deterministic): 0.04 % between two seeds at N = 1, 8 and 32 alike, i.e. nothing
to average yet; the noise of the real refraction transport must be measured when the engine side lands (OPEN).

Runner keys and controls (`tools/f15_check.py`, all "pending": "ticket 13"; `pending_matches` = `transmission-ignored`, `declared-transmission-ignored` for rtoff).
Controls: transmission-ignored (opaque Appendix B dielectric: exact environment quadrature + traced mirror reflection of the scene for smooth glass: without the
reflection an opaque cube sees no checkerboard and failed the twin at p95 21 %), transmission-not-tinted, volume-ignored, ior-ignored (no bend: gltf_thin, media kept),
thin-as-volume, backface-rejected (back faces of closed meshes removed: today's `reflections.metal:21`), emitter-double-counted (radiance x 2), back-glass-black.
Dropped/not applicable: a control identical to the reference on a fixture (e.g. not-tinted on white bases) is not listed there; `thin-as-volume` only where thin
objects exist. In a uniform environment opaque white = non-attenuating thin glass, hence the rtoff gate fails only on the attenuating spheres.

Red state on the M5 Max (`build/f15-check/t13` seven keys, `t13b` single-sided; `f15_check.py --run/--evaluate --expect green`: GREEN, exit 0; 1280x720; captures
with 0 archive misses after the archive update). Gate worst object mean / p95 (every gate FAILS, as asserted), twin (engine vs transmission-ignored) worst mean / p95
(PASSES), controls (all FAIL their gate):
- roughness-ior 0.538 / 0.743, twin 0.034 / 0.124; attenuation 1.002 / 1.940, twin 0.019 / 0.087; factor-tint 1.255 / 4.415, twin 0.015 / 0.098;
  thin-volume 0.942 / 1.368, twin 0.014 / 0.080; single-sided 0.949 / 1.383, twin 0.022 / 0.102; glass-behind-glass 0.958 / 1.383, twin 0.032 / 0.095;
  emissive-panel 0.432 / 0.829, twin 0.0005 / 0.001; rtoff 1.315 / 1.359 (attenuating spheres only), twin 5e-7.
- Pooled glass-mask statistic (information): 0.35 / 0.74, 0.63 / 1.30, 0.48 / 1.21, 0.56 / 1.28, 0.49 / 1.26, 0.43 / 1.15, 0.43 / 0.83, 0.21 / 1.36.
Note the glass-behind-glass ior 2 sphere sits at 0.054 / 0.135 against the opaque-glass engine (a mirror-like sphere behind a pane: weak discriminator alone, the
key fails through the other objects); `back-glass-black` discriminates it.

Not done / open: the engine side and every other criterion of this ticket (crossings <= 4 counter, command buffer time, Sponza D2, temporal clip, forced overflow,
side-by-side Khronos pages, `tools/archive_check.sh` / `hitch_check.sh` of the future pipelines, support report text for `--rt off`); the engine accumulation noise
of the real transport; rough thin-walled glass; the exit-interface tint (orchestrator rule: factor 1 there, tint per transmitting event is my reading); ctest 18/18.

2026-11-14, orchestrator decision on the harness's open question:
- The baseColor tint follows the same rule as the transmission factor: it is
  applied at the camera-visible ENTRY surface only, and the exit and interior
  crossings are pure dielectric interfaces.
- The reference tints at every transmitting event. That differs only on
  tinted closed bodies; every gated volume uses a white base, so no gate
  depends on it. Align the reference before a tinted closed body becomes a
  gated fixture.

2026-11-14 - engine side (agent), verified on M5 Max; NOT fully green, see "Open".

Implemented: `src/renderer/transmission_math.h` (Schlick with the transmitted cosine and TIR, branch selection floor 1/32, Beer-Lambert, Walter weight, thin BTDF;
doctests incl. Monte Carlo estimator checks), `bsdf.h` thin BTDF + declared weights, resolve/forward/DI diffuse x (1 - t) and thin-walled delta-light BTDF,
composite, lobe guide (factor, world thickness via `SurfaceInput::volumeScale`, medium, ior; CPU expectation; corrupt control), transport
(`reflectionRefractSample` / `reflectionRefractWalk` in `reflections.metal`: one Fresnel-chosen sample, refraction always traced, 4 crossings / 8 events,
back faces accepted, Beer-Lambert on the real inside segment, tint at the visible entry only, emitters owned by the transport, VNDF sampling, smooth interface
at the roughness floor), declared `--rt off` term in the SSR and probe-only kernels, denoiser (refract key/axis, thin axis -V), MetalFX reactive, support table,
targeted archive update, runner keys no longer pending (denoise off, accumulate 112: see the commit message for the evidence).

Measured (reference = Mitsuba 3.9.1, `tools/f15_check.py --evaluate`; gate = per object mean <= 5 %, p95 <= 15 %):
- thin-volume (1024 spp): pass, worst mean 2.3 %, p95 6.9 %; single-sided (1024 spp): pass, worst mean 2.6 %, p95 9.9 % (the cube needed 8 events); all controls fail.
- emissive-panel (256 spp, denoiser on): pass, worst mean 1.7 %, p95 5.2 %; glass-behind-glass: pass, worst mean 4.2 %, p95 14.8 %; rtoff: pass, worst mean 0.0002.
- factor-tint (1024 spp, denoise off, accumulate 112): 7 of 8 objects pass; the slab with attenuation 0.2 and t = 1 has mean 3.1 %, p95 15.8 % (limit 15 %).
- attenuation (256 spp ref, denoise off, accumulate 112): worst mean 3.8 %, p95 10.4 %: pass; roughness-ior (256 spp ref): worst mean 6.7 %, p95 16.8 %, noise-limited on
  both sides (the Mitsuba 256-spp reference itself has 5.8 % per-pixel noise on roughness 0.6); per-object mean ratio engine/reference 0.995-1.005 (unbiased).
- Accumulation noise (two seeds, N = 32 frames x 8 samples, glass mask, relative difference): thin-volume 0.55 % mean / 1.9 % p95, single-sided 0.50 / 1.85,
  roughness-ior 0.74 / 2.7 with the denoiser, 4.0 / 12.3 pooled and up to 7.6 / 18 on roughness 0.6 without it.
- `--debug-lobe-guide 8` PASS (thin-volume 29836 extended), corrupt control FAIL; `tools/f15_lobe_guide_overflow_check.sh` PASS on thin-volume and single-sided (dumps identical);
  forced-overflow image vs meshlet path in the glass mask: mean |d| 0.05 % (thin-volume) / 0.10 % (single-sided); shader + API validation 1500 frames glass-behind-glass with ReSTIR + RT: clean;
  eight captures: 0 archive misses, 0 render-thread compiles, 0 GPU allocations after warm-up (four scenes checked).
- Base scenes: base-grid-sun, specular-grid-env@rt, iridescence-grid-point@restir are bit-identical to 65dd54f; clearcoat/sheen/aniso/subsurface env fixtures differ in 0.2-2.7 % of the pixels by <= 6 ulp (compiler contraction).

Open: the full runner with 1024-spp references of roughness-ior / attenuation / glass-behind-glass / emissive-panel (hours of Mitsuba CPU time); factor-tint slab p95 15.8 %;
thin-walled delta-light BTDF is unvalidated against a reference (no analytic-light fixture); temporal clip, Khronos side by side (22), Sponza D2 (24), cost measurement,
`archive_check.sh` / `hitch_check.sh` not run.

2026-11-14 — orchestrator: battery `battery13e` (full runner, 1024-spp references) and the factor-tint diagnosis.
- Full runner (`build/f15-check/t13e`, saved as `results-battery13e.json`): every transmission key passes except
  `transmission-factor-tint`, whose slab 0 (attenuation 0.2, t = 1) has mean 3.1 % and p95 15.78 % (limit 15 %), the same
  value as the engine agent's run: systematic, not noise.
- Diagnosis: the high-error pixels are a ~9 px strip on the slab's left side face, seen at 65 deg incidence. Its mirror
  reflection sees an emissive checkerboard cell (radiance 2, four times the environment). The engine is about 0.05
  darker in all three channels. An independent analytic Fresnel-tree tracer of the box agrees with the Mitsuba
  reference to 0.69 % mean on the slab. Exact Fresnel there is 0.120 and Schlick 0.101.
  - With the analytic Schlick-minus-exact correction added to the Mitsuba reference, the engine's p95 on that object
    falls to 12.05 %.
  - Against the noise-free Schlick tracer it is 9.9 %.
- Decision (orchestrator): the reference must model the glTF Fresnel. The engine follows glTF (Schlick, the
  transmitted cosine on the leaving side, TIR = 1). The reference used Mitsuba's exact Fresnel for smooth solids while
  it already used Schlick for thin walls. The bound is unchanged.
  - `gltf_dielectric` (Schlick) now renders smooth solids and the thin-as-volume control (475d05f, Sonnet agent,
    reviewed). Its exact mode is event-for-event equal to Mitsuba's `dielectric`. Schlick mode matches the analytic
    sphere tracer.
  - On slab 0 the new render agrees with the Schlick tracer at 0.59 % mean, as the old render did with the exact one
    (0.61 %).
  - Rough solids keep Mitsuba's Walter model (declared difference).
  - The shadow reference now records the base module's sha (d77ce96), so its cache re-renders when that module changes.
- The 1024-spp transmission references are being re-rendered with the corrected reference (`build/f15-exact/tref13.log`).
- Still open in this ticket:
  - the temporal clip with transmission, the refract-axis clip control and the MetalFX reactive proof (Sonnet agent
    running);
  - O7 on the transmission scenes;
  - the forced overflow on a glass fixture, rerun by the orchestrator;
  - the crossings bound (structural: `kTransmissionMaxEvents` bounds the loop) and the command-buffer time on
    glass-behind-glass;
  - cost on a quiet machine (ticket 24), the Khronos pages (ticket 22) and Sponza D2 (ticket 24).

2026-11-14 — orchestrator checks on the integrated tree (tickets 13, 14 and 17 in; `build/f15-exact/battery13g.log`, `build/f15-t13-checks`).
- Forced overflow, RT transport with accumulate 112, normal vs `--debug-meshlets 1 --debug-meshlets-corrupt count`, glass mask of the runner (limit 2e-3):
  - thin-volume: mean |d| 6.2e-4 (relative 0.06 %);
  - single-sided: mean |d| 9.9e-4 (relative 0.10 %);
  - frame mean 5e-5;
  - result: PASS.
- O7 (warmup 64, 64 frames), every transmission scene on the RT path (`--rt on --lighting restir --reflections rt`) and on the declared path (`--rt off --reflections probes`):
  - 0 GPU allocations and 0 render-thread compiles everywhere, with one exception: the first RT run of `transmission-rtoff-combinations` had 3 GPU allocations;
  - reproduced and explained. Configurations outside the runner set miss the archive (16-19 misses), so their RT-consumer kernels start on the fallback and the final variant compiles in the background. When it arrives, `RtConsumer::prepare` and `updateIft` create a new intersection function table, one per frame slot (3), because an IFT belongs to its pipeline object;
  - cold Metal compiler cache: compiles take up to 608 ms, the final variant arrives after the warmup, and the run shows 3 allocations. Warm cache: compiles take at most 45 ms and the run shows 0, in two repeats (`o7g-cold`, `o7g-warm1`/`warm2`, a never-compiled clustered + DDGI configuration);
  - load alone does not reproduce it: 0 in 5 runs at load average 10;
  - archived configurations have no misses, so there is no IFT churn;
  - the mechanism predates F15 (F9 consumers, F3 async variants); F15 only adds variants;
  - recorded for ticket 24 as an O7 caveat of non-archived RT configurations, not as an F15 defect.
- Command buffers: glass-behind-glass at 1920x1080 with RT shadows, ReSTIR, DDGI, RT reflections and refraction gives GPU frame max 14.0 ms (p50 13.7). That is far below 1 s (loaded machine, indicative only).
- Crossings bound: structural. `reflectionRefractWalk` loops at most kTransmissionMaxEvents = 8 traces, and a crossing beyond kTransmissionMaxCrossings = 4 ends in the probes/environment (status 2). The unit tests of transmission_math.h cover the estimator. No GPU counter was added.

2026-11-14 — orchestrator: the transmission keys against the corrected (glTF Schlick) reference.
- 1024-spp references re-rendered with `gltf_dielectric`, then `--evaluate` on the battery13e captures (`build/f15-exact/tref13.log`).
- Every key passes. Every control fails, so every key discriminates.

| key | worst object mean | worst p95 |
|---|---|---|
| roughness-ior | 4.97 % | 13.5 % |
| attenuation | 2.8 % | 8.8 % |
| factor-tint | 2.7 % | **11.7 %** (was 15.78 % against exact Fresnel) |
| thin-volume | 1.9 % | 5.4 % |
| single-sided | 2.2 % | 5.6 % |
| glass-behind-glass | 2.8 % | 9.6 % |
| emissive-panel | 1.2 % | 3.7 % |
| rtoff | 0.02 % | 0.02 % |

- Roughness-ior's worst object is the roughness 0.6, ior 2 sphere. Its reference is Mitsuba's exact-Fresnel Walter model (declared difference). It is unchanged by the Schlick fix and sits 0.03 points under the bound.
- Criterion 1 (transmission fixtures vs Mitsuba within the bound) is met on M5.
- Temporal clip: the stable denoiser key is integrated (dab4380). The converged control and the per-branch history round are still running in an agent.

2026-11-14 — temporal clip record (agent; decisions marked "orchestrator" are the orchestrator's). Criterion "Temporal clip (orbit, moving light) within F13.6;
transmissive pixels reactive for MetalFX temporal": reactive part MET, temporal clip NOT MET (stays unticked, partial).

Built (all in `tools/f15_clip_check.py`, `tools/f15_reactive_check.py`, `--transmission-motion`, `--debug-transmission-both-branches`):
- Motion: `--transmission-motion` on `transmission-roughness-ior`: scripted camera orbit 12 deg * sin(t) about the target (<= 0.2 deg/frame, no cut) and one emissive
  panel (radiance 3) sliding in x behind the glass; deterministic under `--fixed-timestep`; the static fixtures are unchanged without it.
- Reactive proof (`tools/f15_reactive_check.py`, `--capture-linear-signal reactive` = the R8Unorm MetalFX mask): 9 cases ALL MET. In the visibility resolve AND the
  indexed-overflow fallback (forced by `--debug-meshlets 1 --debug-meshlets-corrupt count`), plain and lit, every pixel of the eroded glass discs is 0.75 (4632/4632 on
  thin-volume, 12404/12404 with motion) and none outside the dilated discs; the opaque `base-grid-sun` control marks 0 pixels in all four path/lighting combinations.
- Stable denoiser key of transmissive pixels (factor >= 0.5: the refract key whichever Fresnel branch was drawn): `denoiseStableLobeKey`, dab4380.
- Every clip capture passes `--pipeline-sync` and is rejected on fallback draws (9f669f1). A seed run with ~100 fallback frames once biased the control by 8 %.
- Converged control: the per-frame mean of N independent 8-spp runs (seeds 1000 + k, halves A/B alternating), each with the verification-only
  `--debug-transmission-both-branches` (both Fresnel branches at the visible interface). The estimator is proven equal to the product one: accumulated 112 frames, mean
  |both - standard| / mean on the eroded glass discs 0.35 % (thin-volume) and 0.53 % (roughness-ior), mean ratio 1.0000 (bound declared before: 2 %, 1 +- 0.01).
  Self-test: half A against half B with p95, max and flicker at most 1/3 of the gates; the runner refuses to judge (exit 5) when it fails.
  Seeds needed (flicker falls as 1/sqrt(N); 16 -> 96 measured 0.0713 -> 0.0292 on roughness 0.3): smooth ior 1.33 / 1.5 pass at 96, ior 2.0 passes at 128
  (p95 0.0092, max 0.0145, flicker 0.0057 against 0.02 / 0.05 / 0.0067); roughness 0.1 and 0.3 sit at flicker 0.0128 / 0.0253 at 128 and need about 460 / 1,800 seeds
  (4 h / 17 h of GPU).
- Bit-identity (product path vs the phase head b886141, 24 frames, same commands): transmission-custom, transmission-raw, clip-custom, coat-custom, edit-custom and
  negative-lobe-flip are bit-identical with the verification switch compiled in.

Orchestrator decisions: roughness 0.1 and 0.3 are NOT judgeable (control cost): reported as information, not gated (a measurability statement, not a gate change). The
smooth ior 1.33 / 1.5 / 2.0 ROIs have a converged control and stay gated.

Gate table (F13 thresholds unchanged: p95 <= 0.06, max <= 0.15, flicker <= 0.02; 128-seed control, 640x360, 160 frames, 1 spp; `build/f15-clip-t13b`):

| ROI (gated) | custom p95 / max / flicker | raw 1 spp | negative-refract-axis |
|---|---|---|---|
| smooth ior 1.33 | 0.1745 / 0.1995 / 0.0252 | 0.155 / 0.171 / 0.045 | 0.1745 / 0.1995 / 0.0252 |
| smooth ior 1.5 | 0.1690 / 0.1875 / 0.0234 | 0.149 / 0.154 / 0.047 | 0.1690 / 0.1875 / 0.0234 |
| smooth ior 2.0 | 0.1448 / 0.1843 / 0.0267 | 0.185 / 0.357 / 0.083 | 0.1448 / 0.1843 / 0.0267 |

Information only (control not judgeable): roughness 0.1 custom 0.145 / 0.150 / 0.026 (raw 0.247 / 0.256 / 0.146); roughness 0.3 custom 0.086 / 0.141 / 0.024 (raw 0.402 / 0.414 /
0.374; negative 0.088 / 0.210 / 0.034); roughness 0.6 custom 0.049 / 0.068 / 0.017; whole grid 0.077 / 0.082 / 0.013. The single-8-spp control (old) is kept in the report: its
own seed-vs-seed noise is p95 0.07 to 0.20, which is why it was withdrawn as a judge.

Verdict: the gated smooth ROIs FAIL on all three checks; `negative-refract-axis` cannot show "fails while custom passes" (same numbers on the smooth ROIs: the axis term is
~4e-6 for smooth lobes and no spatial filter acts; on roughness 0.3 it acts, flicker +39 %, max 0.210 against 0.141). The runner states the failure (`TRANSMISSION CLIP
FAILED`, exit 1); `--group coat` and the other groups are unaffected.

Cause: the error is at the edges of the refracted checkerboard (88 % of the error energy; custom is WORSE than raw on smooth glass: 0.17 against 0.15). The surface motion vectors
do not move the refracted image, so the temporal history drags the previous image along under the orbit and the moving panel. Through lens spheres the previous-frame pixel of a
refracted point is not a reprojection of the surface pixel; it needs the virtual-image motion. Orchestrator: not attempted in F15, an EDGE/F16 item.

Rejected spike A (orchestrator decision: not adopted; numbers in `docs/opt-log.md`): per-branch denoiser history (a second reflect-branch history, the transport handing the
denoiser the unweighted contribution of the drawn branch). 16 B per pixel per history buffer (66 MiB at 1080p), flicker 0.025 -> 0.021 on the smooth ROIs, p95 and max unchanged
(0.17 / 0.19), so smooth-glass p95 stays far above 0.06. Also tried and reverted: endpoint-strict history for smooth refract samples (p95 0.17 -> 0.11 on ior 1.33 / 1.5, flicker worse on
ior 2.0: 0.031 -> 0.058, stochastic endpoints through internal Fresnel branching).

Not done / open: criterion "temporal clip within F13.6" (unticked); virtual-image reprojection (EDGE/F16); judging roughness 0.1 and 0.3 (about 460 / 1,800 seeds).

### Ticket 22 closure (2026-11-14)

- The four side-by-side models rendered (TransmissionTest also exercises transmissionTexture, AttenuationTest thicknessTexture, qualitatively). Open: C22-D4 (dark grain through glass over single-sided labels, disproved as missing GI by a --gi ddgi diagnostic), C22-D7 (grain at high IOR in one settled frame).
