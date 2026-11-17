# F15: advanced materials and shader variants — handoff

Branch `cl/f15-plan-c38b51`, base 75924b2 (plan approved by the owner on 2026-11-13), M5 Max 128 GB, macOS 27.3.
Plan: [plans/F15.md](plans/F15.md); tickets `.scratch/f15/issues/01..24`; decisions `.scratch/f15/decisions/`; ADR-0003 (permutations by
function constants), ADR-0004 (RT transmission, one Fresnel-chosen sample per pixel), ADR-0005 (OpenPBR subsurface + draft KHR), ADR-0006
(base model = glTF Appendix B), ADR-0007 (specular multiple-scattering compensation, on by default). The ticket 24 record
(`.scratch/f15/issues/24-phase-closure.md`) is the detailed ledger behind this page.

**Status (round 5, 2026-11-17).** Round 1 (2026-11-15, tip 720ee63) delivered every ID with partial items. The owner then asked to finish every
partial before the PR, to fix earlier-phase defects found on the way and to implement six items the plan had declared out of scope (round 2,
114 commits). Round 3 (2026-11-16) closed the round-2 partials that writers could close inside the 14:30 timebox and ran a final battery that
found 21 f15_check failures; the owner decided each of them. Round 4 (2026-11-17) applied those decisions (layer test flag, multiple-scattering
fixes, combo regression) on tip 817b080; its battery became a pre-check when the owner reversed the combo fix and asked to fix the non-lobe
multiple-scattering kernel now. Round 5 reverted the combo fix (588f711), integrated and GPU-verified the non-lobe MS fix (47dcefe, targeted
archive af1a2b5) and re-ran ONE complete final battery on the new binary (phosphor 991b2f95, metallib 4518d6ea, archive 26742e85; section
"Closure verification"). **No roadmap box F15.1-F15.4 is ticked** (each ID keeps an open item, see below). Physical Apple9/T0 certification
stays `EXTERNAL_VALIDATION_PENDING`. Sponza D2 on the final binary: band met as an interleaved A/B (t03 -> final5 +0.051 ms, 0 allocations)
in a timing window that quietgate flagged disturbed (absolute GPU p95 15.35 ms, not a quiet-machine value).
Nothing is pushed; no PR is opened without the owner.

## Owner decisions of round 2 (2026-11-15/16)

- Implemented instead of declared: specular multiple-scattering compensation, ON by default (ADR-0007, `--specular-ms on|off`; references that
  evaluate the base lobe model it); multi-bounce reflections on every smooth surface (`--reflection-bounces 0..3`, default 3, stop at cumulative
  Fresnel throughput < 0.01); non-uniform (per-slot) `KHR_texture_transform`; back-lit thick subsurface through an optional RT random-walk tier
  (`--subsurface-mode rt-walk`, default stays the screen-space diffusion); `COLOR_0` (zero cost when absent); class pipelines bit-identical to
  the all-lobes pipeline by an exact-maths second resolve module for scenes with material features (F15.2 option (a)).
- Scope additions kept: `KHR_materials_unlit` and `KHR_lights_punctual` (plan line 22 and decision 02 updated, 4e89658).
- Exclusions (plan "Rischi", 264b552): (a) AO, froxel fog, ReSTIR reconnection and sky-visibility rays see glass as opaque (named in the support
  table, dcfb3ea); (b) secondary RT hits are shaded with the base reduction.
- Rough-glass temporal ROIs (roughness <= 0.3) declared (option (b), b97aa49): measured 0.087 / 0.091 against the 0.06 gate, kept as information.
- Transmission cost: optimised by measurement only (adopt a measured benefit, record the spike either way).

## Owner decisions of rounds 3-5 (2026-11-16/17)

- Chess king/queen subsurface bound on the RT walk: declared OPEN (option A), follow-up D.
- `aniso-narrow-sun`: declared OPEN on the exact module (option A). Pixel (765,76), sphere 23: 3.77 % exact against 2.21 % fast; 16 pixels
  above 1 % against 6. The spiral-rule hypothesis was tested and rejected (diff kept in `build/aniso3`). Follow-up E: per-term dump N, T', H, D,
  V, F of the pixel. No fast-module gate, no threshold or pixel change. **On narrow anisotropic lobes the exact resolve module, which is the
  default for every scene with material features, is less close to the reference than the fast module.**
- F15.2 archive build time: declared NOT demonstrably met (option A): +25.7 % median and +27.4 % on per-round minima against the +25 % budget
  in round 3, with a ±10 % run-to-run spread. Cause: 226 `phx_exact` visibility_(lit_)resolve class pipelines, 124 of them constant-26
  variants. The box stays unticked; follow-up: class pipelines only for the constant-26 values in use, or a plan that names them.
- 16 layer-equal keys: a test flag forces the exact module on featureless scenes (option B, `--debug-material-module exact-all`); the
  f15_check layer captures use it. The test changed after the measurement that found the failure; the A/B numbers are in ticket 23.
- MS mismatches on `sheen-probe-env` and `texture-coat-sun`: diagnosis with a 1.5 h timebox and a fix on the wrong side (option B), the
  reference fix motivated by the spec and the consistent base path, with a unit test that fails without it (N != Nc).
- `combo-all-lobes-env` sphere 2 (round 4, then reversed): the round-4 fix 817b080 (transported anisotropic pixels take the exact,
  lobe-independent diffuse weight) brought sphere 2 to 0.92 % but cost white-furnace energy on transported dielectric anisotropic pixels.
  The owner first accepted it, then decided **"Annullare il fix"** on the corrected furnace numbers (`furnace-aniso` spheres 4-11 about
  1.000 -> 0.964-0.992, `furnace-combinations` spheres 1 / 2 / 6 / 9 0.9976 / 1.0000 / 0.9714 / 0.9790 -> 0.9694 / 0.9808 / 0.9478 /
  0.9764). Only the composite correction is reverted (588f711); the 4-D anisotropic table and the resolve stay. `combo-all-lobes-env`
  sphere 2 is OPEN at 5.12 % against 5 % (bisection 4.07 % on 7704b21; cause: on transported pixels the table's diffuse complement is
  0.910 where the exact weight is 0.830). Follow-up: an exact diffuse complement for transported pixels, first trying the lobe-independent
  weight 0.830.
- Non-lobe MS under probe reflections: **fix now** (owner, 2026-11-17). The key `ms-probe-env` had to fail on 817b080 and pass with the
  fix, controls included; integrate, targeted harvest of the new kernel, re-evaluate the probe / MS / composite / furnace keys, exactness
  and visual, then one complete final battery with the quiet window and a fresh-freeze corpus. Done (below; the timing window was flagged
  disturbed, see "Measured cost"). The test's metals-only limit is
  declared. Exactness: `manylights-restir` is an expected red against the pre-fix reference, proven by the `--specular-ms off` A/B; no
  reference is silently updated.

## Delivered scope per roadmap ID

### F15.1 — extended material model (tickets 01, 03–20)

Round 1 (unchanged, see the tickets): shared BSDF module (01), Appendix B base model (03), versioned description and `KHR_materials_*` parsing
(04), GPU extension records and feature-union specialisation (05), lobe guide (06–08), the lobes clearcoat (09), sheen (10), anisotropy (11),
iridescence (12), transmission + volume with RT refraction transport (13), coloured RT shadows through glass (14), the CSM transmittance map
(15), thin subsurface (16), volumetric subsurface (17/18), combinations and furnace (19), the chess fixture (20).

Rounds 2-5:
- Multiple-scattering compensation (03, 19; 3812871, ec56d4a): Kulla–Conty style energy tables (`src/renderer/ms_table.h`), furnace
  `ms-furnace@resolve` 0.00016 against 0.60 for the control; off path bit-identical to round 1. Anisotropic and transmitted lobes carry no
  compensation (ADR-0007). Round 4 (tickets 03, 09, 10): under probe reflections the lobe-guided probe kernel lights the MS term with the
  probe radiance and the composite skips its environment-lit copy (`sheen-probe-env` worst sphere mean 7.02e-2 -> 1.8e-4, limit 5e-3;
  a3743798); the reference evaluates the coat's own MS lobe on the coat normal Nc, as the base coat path, the engine and
  KHR_materials_clearcoat do (`texture-coat-sun` worst sphere p99 14.8 % (max 28.3 %) -> 0.34 % (max 1.46 %), `texture-coat-env-resolve`
  7.36e-4 -> 4.76e-4, every other coat key unchanged to float noise; unit test with N != Nc fails on the old reference; e176e18e).
  Round 5 (47dcefe, the cherry-picked 66243cb): the non-lobe probe path (featureless scenes under `--reflections probes`) now lights F_ms L
  with the probe radiance at N in a `reflection_probe_only_ms` kernel, and the `_ms` composite skips its environment copy. Key
  `ms-probe-env` (metal spheres, probe 0.2 against a declared environment 0.5): on 817b080 RED, mean error 0.03 / 0.49 / 4.39 / 14.4 / 36.5 /
  56.1 / 74.4 % at roughness 0.1 / 0.2 / 0.35 / 0.5 / 0.7 / 0.85 / 1.0; on the integrated binary GREEN, worst mean 0.008 %, p99 0.038 %,
  0 archive misses; both reference-side controls fail (49.6 %). **Declared limit:** only metal spheres are gated, because the reference's
  `--sky-override 0.2` lights the diffuse term with 0.2 while the engine lights it with the declared 0.5; the key does not cover the
  diffuse / MS split of dielectrics under probes.
- Texture coverage: numeric texture fixtures against the independent reference for coat normal/factor/roughness, sheen colour/roughness,
  anisotropy direction/strength, iridescence thickness/factor, thin-subsurface weight/colour (422cd30, 452580a; worst p99 0.34–0.70 %); glTF
  thin-subsurface encodings equal the description file in images (0 px); per-slot `KHR_texture_transform` at the primary and every secondary site
  (245a1e0..4bb83e9, e4dccba, 78d173d) and inside layer stacks (7fcbb3d), compiled out without transforms (b6e3a55).
- Transmission (13–16): endpoint-matched specular history (smooth-glass clip p95 0.0230 / 0.0234 / 0.0307 against 0.06; the refract-axis control
  fails at 0.17); thin-walled delta-light BTDF fixture, glTF tint rule, rough thin-walled glass (0878ed1, 841fd49); glass receivers behind glass,
  CSM glass with the static cache, half-resolution edge gate (857ef81); closed-volume interior and transmissive alpha-masked casters in the RT
  transmittance (049f570: `volume` 0.6446 / 0.8254 / 0.9570 against 0.6429 / 0.8243 / 0.9565); bounded Fresnel-branch weights (aa3b19b: 0 pixels
  above the analytic bound against 1275–1303); GI at -N for the thin transmitted lobe in every GI mode (500749e); back-lit thin receiver lit
  through RT/DI visibility (`thin-backlit` 3.83x unlit, occluder control 0.000). Round 3: the CSM PCSS filter applies the receiver-plane bias in feature scenes, so a back-lit thin receiver no longer
  self-shadows (`csm-thin` csm/off 0.870 -> 0.99995, RT 0.99995, occluder control fails; 205d50b); sun shadow rays of GI (DDGI, cache, ReSTIR GI)
  and RT-reflection secondary hits through glass equal the analytic transmittance (0.3840 / 0.8640 / 0.4320, error 0.0000; 781467b); the RT walk's exit
  connection casts its shadow ray through glass (pane/open 0.0154 -> 0.3840; glass-free scenes keep the Lite walk; cf21330).
- Volumetric subsurface (17/18): optional RT random-walk tier (f017131..0fb4490): side-lit 0.0635 / 0.1708 against 0.10 / 0.25, back-lit
  0.0919 / 0.2288 against 0.20 / 0.50, product output 0.0621 / 0.1681, moving-camera clip 0.0703 / 0.1902 (control 0.1400 / 0.4482 fails);
  Apple9 and `--rt off` fall back to the diffusion with the reason reported. Chess king/queen bound on the walk: RED on the final tip, equal to the pre-check, declared OPEN by the owner (option A, see "Open and partial").
- Loader: `KHR_materials_unlit`, `KHR_lights_punctual` (spot falloff = engine smoothstep, declared Approximated, measured on every DI path),
  tangent-frame repair (9e54fe2); `COLOR_0` on indexed, mesh, forward and every secondary site with fixtures (a740f2d..4e7ccfc). Round 3: `COLOR_0` on the visibility tile path (662f87f) and the cut-out alpha of DI,
  clustered and fog emitters (31d3bb2; fog emitter fixture: old -0.0176 FAIL, new -0.0674 against baked -0.0666), +34 `_vc` twin pipelines archived
  (e53aaf7). Spot falloff measured on RT reflection, probe capture and GI arms: declared OPEN by the owner with per-path bounds, support table stays
  Approximated. The owner rejected two post-hoc fixture changes (95e184e: fog sign gate removed, RT-reflection patch back to 0.05 m).
- Reflections: the mirror chain on every smooth surface (5e4ea7d..41be99b), the F13 reference models it (`f13_reference.py --mirror-bounces`);
  dielectric corridor class 2 4.40 % against 5 %.
- Anisotropy: the declared environment reads a 4-D anisotropic albedo table (0.08 % against the exact albedo where the rms fit left 6.5 %;
  be6ea68, 141655c); the denoiser lobe-id criterion is measured on the GPU (specular history 40 / 40, flip control 0.0; b6c9b3d); `aniso-narrow-sun` is RED on
  the exact resolve module (one pixel 3.77 % against the 3 % max; the fast module gives 2.21 %), declared OPEN by the owner, follow-up E: on
  narrow anisotropic lobes the default (exact) module is less close to the reference than the fast one. The table makes the diffuse
  complement of TRANSPORTED anisotropic pixels about 0.08 too large: `combo-all-lobes-env` sphere 2 4.07 % (7704b21) -> 5.12 %, OPEN after
  the owner reverted the round-4 correction (see "Owner decisions of rounds 3-5"); on the final binary `aniso-grid-env` 1.93 %,
  `combo-aniso-panel-env` 1.57 %, `furnace-aniso` and `furnace-combinations` identical sphere by sphere to c2514d6.

### F15.2 — permutation generator (ticket 21)

Round 1: feature-class manifests, `variant_gen classes`, targeted archive updates, <= 64 permutations, binning opt-in (never >= 5 % faster).
Round 2: an exact-maths second copy of the visibility resolve kernels (`phx_exact::*`, safe math mode + scoped `contract(off)`, fa6786d) is
selected by every scene with material features: `f15_permutations.py equivalence`, tolerance 0.0, 200 cases (162 scenes): 0 px / 0 ULP; the
controls leave the envelope in 125 of 185, every feature bit 0..8 discriminated. Featureless scenes keep the fast module (images unchanged).
Price (quiet A/B, opt-log 2026-11-16): +0.11–0.22 ms of resolve at 1080p on the layered terrain (+48..78 % of the resolve); scenes without
features pay nothing. Round 3: the old ToyCar / ClearCoatCarPaint variation was reproduced on the old branch and localised to aligned 32-pixel groups of reduced
hit-metadata records with radiance but flags 0 (old exact module + its archive + RT reflections only); the tip never produces it (0 / 480 frame checks,
60 / 60 identical images); `ReflectionPasses::check` fails on such a record under `--debug-lighting` (e6dbb03). Declared localised by the owner; follow-up:
a marker-word debug kernel. Round 4: `--debug-material-module exact-all` (test only) forces the exact module on featureless scenes too
(dc59635e, +5 archived compute pipelines 4c69a39b). Round 5: +1 archived compute pipeline (`reflection_probe_only_ms`, af1a2b5, targeted
harvest). Final equivalence on 991b2f95: 238 cases / 168 scenes 0 px / 0 ULP, controls leave the envelope in 151 of 216, feature bits 0-8
discriminated, overflow 12 / 12 0 ULP, repeats 7 / 7 bit identical. Archive build time: not demonstrably met (owner decision above), the box
stays unticked.

### F15.3 — material testbench against the Khronos models (tickets 02, 03, 22 and every lobe fixture)

`tools/f15_check.py`: 287 scenarios (210 evaluated), decision-11 thresholds, a negative control per gated fixture; references re-rendered with
MS on where they evaluate the base lobe. Final run on 991b2f95: RED only by `aniso-narrow-sun` and `combo-all-lobes-env` (both declared OPEN
by the owner); 14 FAIL rows: those two, the two declared default-tier subsurface keys and the ten `@rt-walk-*` negative controls, which must
fail. `ms-probe-env` (new in round 5) passes. Khronos glTF-Sample-Assets at 8f49239: 24 of 24 render (DiffuseTransmissionTest,
GlamVelvetSofa, TextureTransformMultiTest render: unlit, punctual lights, per-slot transforms). Side-by-side page qualitative by design;
Render Fidelity goldens not used (their download is not approved).

### F15.4 — layered materials (ticket 23)

Round 2: layers blend feature blocks (coat, sheen, anisotropy, iridescence, specular, emissive, thin subsurface) and emission, a zero-weight
coat is exact by construction (e074074); clamp-to-edge mask and edge gate (23b0875); reflection/GI hits and the probe capture see the blended
stack (7375a34); coverage alpha and per-slot transforms in stacks (7fcbb3d). Resolve at 1080p (CC0, exact module, final): single 0.095 ms,
layers-1 0.279, layers-4 0.383, full 0.492 ms. Triplanar mask deferred to F19.2 (plan). The round-1 textured-terrain envelope overshoot (655 ULP) predates the exact module: on the tip all 19 terrain
cases are 0 px / 0 ULP (e6dbb03). Round 4: the 16 layer-equal keys compared a stack on the exact module with its featureless pair on the
fast module; with one maths mode for both sides (`exact-all`, 78 captures) all 16 are 0 px on the final binary. The layer blending itself
was already exact (0 px with both sides on the fast module, round 3); identity is stated within one resolve module.

## Defects found during F15 outside F15 code (owner: fixed inside F15)

Round 1: see [the F13/F14 closure document](implementation/F13-F14-CLOSURE-2026-11-12.md), section "Defects found during F15".
Round 2: transient Metal compiler-service interruptions (`XPC_ERROR_CONNECTION_INTERRUPTED`) now retried by PipelineCache with a fixture
(`--debug-pipeline-xpc-fail N`: N=2 recovers, N=4 fails as required; F3, da0f64e); fused render encoders no longer repeat
setDepthStencilState / setViewport (6691756); `visual_check` captures with `--pipeline-sync` (f21f4c4) and `archive_check` compares pixels with
final pipelines only and compiles the stale scenario with the exact module (4c642fb, 06dd3a7); integration collisions of material / reflection
flags found and pinned by static asserts (24563ce, 980782b); the RtPayload layout pinned at 128 bytes (132/136 bytes crash the Apple GPU compiler,
4462854, f7d1f22).

## Measured cost (perf-log entries of 2026-11-16 and 2026-11-17)

**Timing window of round 5: NOT a quiet machine.** The final timing window on the final binary 991b2f95 (perf-log "F15 closure round 5:
final window", 10:38-11:31) had no Mitsuba and no other GPU job of ours, but quietgate (`build/f15-costs-r5/quiet.json`) flagged it
**disturbed**. Processes seen busy (samples above the busy threshold): `applegpu-nt` 119, `air-pack` 7 and `metal-tt` 3 (metal-tt's own
compilers, archive rounds only), Google Chrome renderers 36, Codex renderers 24, WindowServer 15, git 13, python 11 (not attributed),
Claude app renderers 8 and Claude processes 4,
PerfPowerServices 6, duetexpertd 5, fseventsd 4, runningboardd 2, `llama-server` (a local LLM server) 2. Load average: 4.4 at the start
(gate 6), 2.2-3.5 during the timed GPU steps, 17-19 during the metal-tt rounds (maximum 23.8). Consequences: every comparison below is
valid only as an interleaved A/B (or A/B/A) inside this same window — the D2 band is judged that way, as decision 11 defines it; the
absolute values (D2 GPU p95 15.35 ms, the replica-1 outlier at 16.56 ms, the rig frame times) are NOT quiet-machine values. Items not
re-measured come from the round-2/3/4 windows:

| item | cost |
|---|---|
| Sponza D2 (decision 11 band) | t03 -> final5 +0.051 ms (pairs +0.70 / +0.05 / -0.08), 0 allocations: band MET as an interleaved A/B (median <= 0.10 ms); absolute GPU p95 15.35 ms (<= 16.67 ms; replica-1 outlier 16.56 ms) measured in the disturbed window, not a quiet-machine value |
| transmission, chess back-light rig | B - A frame p50 +12.24 ms (+7.84..+15.56, A-A spread 2.27): `Local clustered RT shading` +9.24, reflection samples +2.88 ms; the 128-byte payload recovered 1.84 ms (round 2) |
| RT walk with glass in the union (optional tier, 720p, 64 walks) | walk pass 95.76 ms against 54.04 ms without glass (+41.7 ms, +77 %): full payload + transmitted exit shadow ray (cf21330); glass-free scenes keep the Lite walk |
| exact resolve module (F15.2 option (a)) | terrain 1080p resolve layers-1 0.282 vs 0.158 ms (+78 %), layers-4 0.353 vs 0.239 (+48 %), full 0.494 vs 0.277 (+78 %); featureless scenes 0 (round 2) |
| layer stack (exact module, union) | single 0.095-0.098, layers-1..4 0.279 / 0.310 / 0.329 / 0.383, full 0.492 ms; feature blocks 1-4 1.260 / 1.501 / 1.689 / 2.530 ms |
| material binning | never adopted: terrain bin arm never 5 % lower; chess side-light +3.9 % (noise 7.7 %), shadow +4.0 % (2.0 %), back-light +6.5 % (3.0 %) |
| RT walk tier (optional) | 4 / 16 / 64 walks 3.92 / 15.15 / 60.11 ms at 720p; screen-space diffusion 0.32 ms |
| iridescence per extra light | +0.033 ms (legacy), +0.036 ms (clustered) (round 2) |
| lobe guide memory | 3 x 158.2 MiB (RayTracing) at 1080p when the feature union is non-empty, none otherwise |
| archive | 2663 pipeline descriptors (1664 compute, 251 mesh, 654 render, 94 tile; +1 compute against round 4), 495.9 MB; round 5 metal-tt per round +23.0 % / +19.0 %, pooled median 56.97 vs 47.48 s (+20.0 %), minima +20.6 %, without the 226 class pipelines; round 3 +25.7 % median: not demonstrably met (owner decision) |
| bench switching | hitch_check 0 hitches / 15 switches, 21 / 21 archive hits, worst post-switch frame 1.28 ms |

Transmission costs against their declared gates (frame budget at 60 fps: 16.67 ms):

| measure | round 5 | round 4 | round 3 | declared gate / budget | status |
|---|---|---|---|---|---|
| glass transport, chess back-light rig 1280x720, frame GPU p50 B - A | +12.24 ms (A 37.13, B 51.86) | +12.51 ms | +12.06 ms | none for the glass itself: ADR-0004 and decision 09 fix the ray budget per pixel ("unchanged; extra traces only for interior crossings"), the plan ("Rischi") says the cost is measured on the fixtures and the Sponza preset is protected by specialisation, tickets 13 / 14 record it as a cost, not a gate | informative. The related gate (non-transmissive scenes unchanged: Sponza D2 band) is MET |
| RT walk with glass in the union (optional tier, 64 walks, 1280x720), walk pass p50 | +41.7 ms (95.76 against 54.04) | +41.4 ms | not measured (the round-3 rig had no subsurface material) | none: ticket 14 records it; ticket 18 states the RT walk is an optional tier whose cost is recorded | informative |

What this means for a 60 fps frame: the glass transport alone (+12.24 ms) is 73 % of a 16.67 ms frame; the back-light rig is a stress
fixture (RT clustered shading, 8 reflection samples) at 37 ms without glass and 52 ms with it, so neither arm is a 60 fps configuration.
The walk with glass at 64 walks (95.8 ms pass) is far outside a 60 fps frame; the tier's product default is 16 walks (15.15 ms without
glass = 91 % of the frame on its own; the glass variant at 16 walks was not measured). The default subsurface tier (screen-space
diffusion, 0.32 ms) and scenes without glass do not pay either cost.

## Open and partial

Every roadmap box F15.1-F15.4 stays unticked. Items below are either OPEN by an owner decision, DECLARED limits, PARTIAL, or external.

**Open by owner decision (2026-11-16/17)**
- `combo-all-lobes-env` sphere 2 (ticket 11): 5.12 % against 5 % after the owner reverted the round-4 correction (588f711). Cause: the
  anisotropic table's diffuse complement on transported pixels (0.910 against the exact 0.830). Follow-up: an exact diffuse complement for
  transported pixels, first trying the lobe-independent weight 0.830 (`1 - max_rgb ∫ F(V·H) cos θ / π dL`, a 1-D Schlick table in N·V;
  iridescence needs its own integral). Expected to differ from the isotropic lookup by about 0.002 at sphere 2, so it may not recover the
  furnace energy on its own (a prediction, not measured).
- Chess king/queen subsurface bound on the RT walk (ticket 20, option A): RED on the final tip, equal to the pre-check (pooled white King
  0.197 / 0.636, white Queen 0.543 / 0.921, black King 0.756 / 2.409, black Queen 1.431 / 3.072 against 0.10 / 0.25; reference noise
  0.101-0.137 / 0.337-0.525; piece luminance -6.7 / +6.3 / +6.3 / +30.1 %). The RT-walk bounds hold on the OpenPBR fixtures. Follow-up D
  (homogeneous medium per piece, identical in engine and reference) not started.
- `aniso-narrow-sun` (ticket 11): one glint pixel 3.77 % against the 3 % max on the exact resolve module (fast module 2.21 %, passes);
  16 pixels above 1 % against 6 on the fast module: on narrow anisotropic lobes the default (exact) module is less close to the reference
  than the fast one. Spiral-rule hypothesis rejected (`build/aniso3`); follow-up E (per-term dump N, T', H, D, V, F of the pixel, exact vs
  fast vs snapshot).
- F15.2 archive build time (ticket 21): round 3 +25.7 % median / +27.4 % minima against +25 % (±10 % spread); round 4 and round 5 windows
  +20-23 %: not demonstrably met; 226 `phx_exact` class pipelines, 124 of them constant-26 variants.
- Spot falloff on secondary paths (ticket 04): DI paths pass (legacy / ReSTIR / clustered worst 2.17 / 2.16 / 2.16 % against 7 %); RT
  reflection fails at tilt 0.55 with the pre-registered 0.05 m patch (7.44 %), probe capture 17.59 %, DDGI not separable, ReSTIR GI
  19.60 %: declared open with these bounds, the support table stays Approximated.
- Old ToyCar / ClearCoatCarPaint run-to-run variation (ticket 21): localised to reduced hit-metadata records with radiance but flags 0
  (old branch only; the tip 0 / 480 frame checks); `--debug-lighting` guard in place; follow-up: a marker-word debug kernel.
- Post-hoc fixture changes rejected (tickets 04, 22): fog COLOR_0 sign gate removed (equivalence gate of record), RT-reflection spot patch
  back to 0.05 m.
- Rough-glass temporal ROIs, roughness <= 0.3 (ticket 13, option b of 2026-11-15): information only.

**Declared limits** (support table and tickets): the default screen-space subsurface tier misses the side-lit volumetric bound (mean
0.25-0.36, p95 0.74-0.92), the optional RT walk tier meets it (side-lit 0.0635 / 0.1708, back-lit 0.0919 / 0.2288); RT walk medium
assumptions (homogeneous, boundary ior 1.5, albedo table ranges); one Fresnel-chosen transmission sample per pixel, <= 4 crossings;
exclusions (a) AO / fog / ReSTIR reconnection / sky rays see glass as opaque and (b) secondary RT hits use the base reduction;
transmittance map limits (single layer, fog with CSM opaque); GI at -N for thin transmission is a DDGI probe lookup; iridescence + specular
bias 4.92 %; dielectric corridor class 2 4.40 % against 5 %; mirror chain stops at 3 bounces / throughput 0.01; anisotropy probe fit and
reflection weights keep the rms fit; a bench primitive without tangents shades anisotropy as strength 0; transmission / volume / volumetric
subsurface are not blendable in layer stacks; triplanar mask deferred to F19.2; caustics and dispersion out of scope; `ms-probe-env` gates
metal spheres only (the reference's sky override also lights the diffuse term, the engine's declared environment does not).

**Partial**
- Ticket 06 criterion 3: the guide allocation is logged per category (3 x 158.2 MiB RayTracing at 1080p, none for an empty union), the
  report JSON carries no memory per category.
- Ticket 02: no recorded test of the runner's lock refusal, manifest fields not quoted.
- Ticket 08 / 24: denoised-DI energy residual (0.977-0.988) stays declared; the energy-preserving a-trous candidate costs +13.27 ms on D2.
- Transmission cost (owner: optimise by measurement): after the 128-byte payload the chess back-light glass effect is +12.24 ms (round 5;
  +12.51 ms in round 4); next candidate `Local clustered RT shading` (+9.24 ms of it). The optional RT walk pays +41.7 ms (+77 %) at 64
  walks when glass is in the feature union.
- Ticket 19: the radiometric accuracy of DDGI, Cache and ReSTIR GI themselves is not verified by F15 fixtures.
- Ticket 22: COLOR_0 is not modelled by the CPU RT reference (`rt_check`); the Khronos side-by-side page stays qualitative (no IBL,
  different tonemap); Render Fidelity goldens unused (download not approved).

**Limits of the verification itself**
- The round-5 timing window was flagged disturbed (Chrome / Codex renderers, WindowServer, `llama-server`, metal-tt compilers): its
  comparisons hold only as interleaved A/B inside the window, its absolute times are not quiet-machine values.
- The battery's forced-Apple9 step runs without `--pipeline-sync`, so its pixel results depend on compile timing; the forced-Apple9 path
  of its six scenarios is verified only by the separate `--pipeline-sync` re-run (6 / 6 at 0 px).

**External / CI**: physical Apple9 / T0 certification `EXTERNAL_VALIDATION_PENDING` (forced-Apple9 runs only); the real
`metal_syntax_check` runs on non-Apple hosts (macOS emulation 48 / 49, the failure is the `__APPLE__` block of metalfx_lifetime.cpp,
unchanged since main). The AGX compiler crash seen twice in round 1 was not reproduced (~200 attempts).

**Found by the round-3/4 batteries and decided by the owner**
- 16 layer-equal keys: one maths mode per comparison (`exact-all`), 0 px on the final binary.
- `sheen-probe-env` (engine) and `texture-coat-sun` (reference): fixed, pass on the final binary.
- `combo-all-lobes-env` sphere 2: the round-4 fix was reverted by the owner; OPEN at 5.12 % (above).
- F15.2 archive build time: declared not demonstrably met, box unticked, follow-up named above.
- Non-lobe MS under probe reflections: fixed and verified in round 5 (`ms-probe-env` RED on 817b080, GREEN on the final binary).

## Closure verification (final binary phosphor 991b2f95…, metallib 4518d6ea…, archive 26742e85…)

Full sha256: phosphor `991b2f9573fed8d5f3b76578126fa70525c566becf97c9a265773a0ce9abd25e`, metallib
`4518d6eacd1232ba7af5d15e6b5fc5a28ffcd4cb4b0f2d38a5726e3da8c3e311`, archive `26742e85da350e007afe1b3f8c2aecd313ac31de4d7490663225a2acc15bb767`,
built from af1a2b5 (later commits are documentation only). Every row below ran on this one binary and this one archive: battery
`build/battery-r5` (2026-11-17 08:14-10:37, no Mitsuba running), the timing window on the frozen clone `build/d2-final/final5` (same
hashes, 10:38-11:31, flagged disturbed by quietgate: not a quiet machine, see "Measured cost"), F13/F14 corpus `build/f15-corpus-r5`
(fresh freeze on the same binary, 11:31-14:16), and the forced-Apple9 `--pipeline-sync` re-run (`build/a9-sync`, 14:16-14:17). The
round-4 run (e01bb209) is a pre-check and the round-3 results (c2514d6f) are superseded; both are kept only in ticket 24.

| check | result |
|---|---|
| ctest | 46 / 46 |
| macOS emulation of metal_syntax_check | 48 / 49 (the known `__APPLE__` block of metalfx_lifetime.cpp) |
| visual_check, 8 benches, API + shader validation | 0 px |
| variant_check (indexed, mesh) | 284 compatible (worst 26 px, delta <= 1), 52 incompatible differ |
| archive_check | all scenarios pass |
| hitch_check | 0 hitches / 15 switches, worst post-switch frame 1.28 ms |
| F13 exactness (`build/exact-final-ms`) | 4 / 5 identical; `manylights-restir` differs, expected: the only probe-reflection featureless scene, it runs the corrected non-lobe MS path (all 439 306 differing pixels brighter, max 1.6e-4); with `--specular-ms off` all 5 scenes are identical between e01bb209 and 991b2f95. Reference not updated |
| subsurface overflow, JIT, XPC N=2 / N=4 | pass; N=4 fails as required |
| forced Apple9 path, 6 scenarios | battery step (no `--pipeline-sync`): exit 0, 0 validation lines, but its pixel results are compile-timing artefacts (limit of the battery, below). Verified only by the `--pipeline-sync` re-run on this binary (`build/a9-sync`): native vs effective apple9 0 px on all 6 (`combo-all-lobes-sun`, `subsurface-volumetric-sun`, `layers-patch`, glass-behind-glass, pane@rt, `texture-coat-sun`), 0 fallback draws, 0 validation lines |
| round-2/3 fixtures (transport, glass receivers, texture transforms, COLOR_0, glass shadow rays, unlit / punctual, residue, shadowed hits, GI environment, walk clip, aniso lobe-id) | pass |
| chess rigs, O7, validation | GREEN |
| spot falloff | DI paths pass; secondary arms RED as declared |
| Khronos corpus + validation | 24 / 24 render, 0 archive misses |
| F15.2 permutation equivalence | 238 cases / 168 scenes 0 px / 0 ULP, every feature bit discriminated |
| transmission temporal clip (converged control rebuilt for this binary) | smooth p95 0.0230 / 0.0234 / 0.0307, max 0.0409 / 0.0480 / 0.0620 against 0.06; refract-axis control fails |
| chess king / queen bound | RED as declared |
| f15_check (287 captures, 210 evaluated) | RED only by `aniso-narrow-sun` and `combo-all-lobes-env` (declared open); `ms-probe-env` GREEN; 14 FAIL rows, the rest declared or negative controls |
| D2 and costs (timing window 10:38-11:31, quietgate: disturbed, not a quiet machine) | band met as an interleaved A/B in that window: t03 -> final5 +0.051 ms (pairs +0.70 / +0.05 / -0.08), 0 allocations; absolute GPU p95 15.35 ms (replica-1 outlier 16.56 ms) is not a quiet-machine value; costs above |
| archive build time (same window) | per round +23.0 % and +19.0 %, pooled median +20.0 %, minima +20.6 % against +25 %; with round 3 at +25.7 % the owner decision stands: not demonstrably met |
| F13/F14 corpus (MS references, fixed F14 oracle) | exit 0: control 30 / 30, cross-check 2 / 2, full-rate-pair 17 / 17, functional 13 / 13, independent-clip 12 / 12, numerical 14 / 14, repeatability-pair 1 / 1; every case's state and verdict equal to the last complete corpus (720ee63); 0 cases with errors (a sibling session's fork storm at 14:02 left no trace in the log or results) |

**Limit of the battery: the forced-Apple9 step depends on compile timing.** The step drops `--pipeline-sync`; under shader validation the
archive is unavailable, so the feature pipelines compile at run time while the scene's base pipeline (no extension lobes) serves those
frames (`FeaturePipelines::serve`, counted as fallback draws), and the captured frame depends on how many frames that took. The native runs
of the battery drew 1 152 / 837 / 485-557 / 81-144 fallback draws (`combo-all-lobes-sun` / `layers-patch` / `subsurface-volumetric-sun` /
`texture-coat-sun`) against 17-18 on the apple9 side, in round 4 and round 5 alike. Hence:
- the 17 343 / 53 236 / 197 318 px differences recorded since round 3 and their explanation ("Apple9 selects the reduced lighting preset")
  were wrong, and the round-4 0 px on `texture-coat-sun` proved nothing; round 5's 51 714 px on that row is the same artefact (re-runs on
  991b2f95 and e01bb209 give the apple9 image whenever the fallback draws are few);
- the forced-Apple9 path of these six scenarios is verified ONLY by the `--pipeline-sync` re-run on the final binary (row above, 6 / 6 at
  0 px, effective capabilities apple9, Hi-Z compute, probe storage not reduced);
- the step was not changed after the fact; follow-up: `--pipeline-sync` in the step.
Not a binary difference either: the f15_check capture of `texture-coat-sun` (with `--pipeline-sync`) is byte-identical between e01bb209
and 991b2f95.

## Next steps

1. Owner review of the branch; push and PR only on the owner's word (`cl/f15-plan-c38b51`).
2. Follow-ups named in "Open and partial": the exact diffuse complement for transported anisotropic pixels (`combo-all-lobes-env`), D (chess
   medium per piece), E (narrow-glint per-term dump), the ToyCar marker kernel, the transmission cost candidate (`Local clustered RT
   shading`), the archive class-pipeline reduction (constant-26 values in use), `--pipeline-sync` in the forced-Apple9 battery step.
3. F16 starts only after F15 is on main (planning checkpoint kept on `phase/f16`).
