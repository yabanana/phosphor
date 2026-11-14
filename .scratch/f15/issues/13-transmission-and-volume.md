# 13: Specular transmission and volume attenuation (ray-traced)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_transmission` + `KHR_materials_volume` per ADR-0004: Fresnel-chosen reflect/refract sample, rough GGX BTDF, always traced with RT whatever the reflection band, interface traversal that accepts the inside back faces of the transmissive medium (single-sided closed meshes included), ≤ 4 crossings, Beer–Lambert along the real inside path (a medium entered at a front face ends at any hit), thin-walled without offset; probes/environment along the refracted ray without RT; refract lobe id/axis/IOR-scaled roughness to the denoiser; emitters hit by refracted rays owned by the transport, delta lights by the thin-walled DI BTDF.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Transmission fixtures (factor, roughness × ior, thin vs volume, attenuation) vs Mitsuba 3.9.1 rough dielectric (same roughness) + homogeneous absorbing medium on the glass mask, with nothing behind the glass lit through glass: decision-11 transmission bound
- [ ] Single-sided closed glass mesh renders the exit correctly; negative control keeping today's back-face rejection (`reflections.metal:21`) fails
- [ ] Glass behind glass is not black; crossings never exceed the bound; command buffers far below 1 s
- [ ] `--rt off`: image compared with the declared probes/environment approximation (image check + negative control), and the support report says so
- [ ] Emissive panel behind thin glass: no double count
- [ ] Ray budget per pixel unchanged on non-transmissive scenes (Sponza D2 within band); extra cost on the glass fixture measured
- [ ] Temporal clip (orbit, moving light) within F13.6; transmissive pixels reactive for MetalFX temporal
- [ ] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] TransmissionTest, TransmissionRoughnessTest, TransmissionThinwallTestGrid, AttenuationTest side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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
