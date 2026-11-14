# 17: Subsurface feature, volumetric mode (screen-space diffusion)

Roadmap: F15.1, F15.3

**What to build:** Volumetric subsurface per ADR-0005: resolve and DI write a diffuse-irradiance target for subsurface pixels, a Burley normalised-diffusion pass in screen space (21/55 samples, metres→pixels radius, sub-pixel skip, same-material and depth test) runs after diffuse denoising, recombined with the specular terms; draft `KHR_materials_scatter` volumetric mode mapped through its Kulla–Conty inversion. Back-lit translucency of thick volumes is out of reach of screen-space diffusion and declared.

**Blocked by:** 16

**Status:** resolved PARTIAL (2026-11-14): side-lit bound not met on valid OpenPBR ground truth; ticket 18 triggered and closed by its fallback (screen-space diffusion stays); diffusion cost → ticket 24

- [ ] Side-lit volumetric fixtures vs Mitsuba 3.9.1 `volpath` random walk (homogeneous medium inside a boundary with the engine's roughness) on the piece mask: decision-11 subsurface bound — NOT met (gated mean 0.25-0.36, p95 0.74-0.92; confinement, see ticket 18)
- [x] Back-lit thick fixture measured and reported, not gated (declared limit of screen-space diffusion) — reported: mean error 0.98-1.0 (engine near black)
- [ ] Radius sweep monotonic; sub-pixel radius equals subsurface-off within FP noise; negative control (diffusion off) fails — engine bleed monotonic, sub-pixel equal (0.000000), diffusion-off fails; the sweep KEY fails against the truth (skin saturates): partial
- [x] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture — engine agent run: mean |overflow - normal| 1.3e-5 (limit 2e-3); rerun in the ticket-24 battery
- [ ] Scenes without subsurface unchanged (visual_check 0 px); diffusion pass cost measured — visual_check 0 px and F13 exact captures 0 px (orchestrator mid-phase battery); diffusion pass cost → ticket 24
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames — archive_check PASS, hitch_check OK (mid-phase battery); O7 0/0 at warmup 64 (agent)

## Comments

2026-11-14 — test-first harness (ground truth, fixtures, runner keys, targeted archive update); no engine or shader change.
Commits: ground truth `57925a6`, fixtures `b38f0d0`, runner keys and gate kinds `79eb380`, archive (next commit), this note.

Ground truth (`tools/f15_subsurface_reference.py`, tests `tools/test_f15_subsurface_reference.py`, 13 OK): Mitsuba 3.9.1 `volpath`
(llvm_ad_rgb, 4 threads), each piece ALONE in its scene (engine captures run `--shadows off`; no inter-reflection), snapshot PLY
mesh (winding checked by signed volume, `flip_normals` when inward), boundary `roughdielectric` GGX with alpha = the engine's clamped
roughness^2 and ior 1.5, interior `homogeneous` medium, sun = directional emitter of the snapshot irradiance (a delta source reaches the
medium through the rough BTDF at the boundary vertices of the walk). Medium mapping per channel: mfp = radius x radiusScale (record 2
words 5, 6-8), sigma_t = 1/mfp, single-scatter albedo rho_ss = (1 - s^2)/(1 - g s^2) with s = 4.09712 + 4.20863 rho - sqrt(9.59217 +
41.6808 rho + 17.7126 rho^2) (rho = the colour = multiple-scatter albedo; the draft's Kulla-Conty inversion), HG phase with g (word 9).
The volume block (record 1) is only the "solid" marker the description validation demands (thickness = the sphere diameter, neutral
attenuation, anything else refused). Weight: image = (1 - w) base model + w random walk (linear mix; one walk serves every weight).
Independent checks in the tests: the inversion against a numpy semi-infinite random walk (the walk escapes 0.01-0.07 BELOW the target
albedo at rho 0.3-0.95, g in {0, +-0.3}: the fit is not exact, the walk is the truth), Lambert under the snapshot camera/sun
(conventions, mean error < 2 %), a non-scattering medium leaves the boundary's specular reflection (0.8-1.25 of Appendix B's).
Convergence: a medium is rendered per channel as a GREY medium (the chromatic estimator has fireflies), 2048 spp per grey render split in
two independent halves, then a 5x5 mean over the piece INTERIOR pixels (no averaging across the silhouette). Residual noise from the
halves (expected mean relative luminance error the noise alone would add to the gate): 0.9-2.2 % mean on the pieces of the gated scene,
up to 5.1 % mean (x0.25 radius pieces, weight 1) in the sweep scene, 1.6-3.0 % back-lit; p95 below 7 %. Pooling bias (pool 5 vs 9
measured on four pieces): about 1 % mean. Cost on 4 threads: 250-700 s per scene, per-piece renders cached by content in
`build/f15-reference-cache` (`F15_REFERENCE_CACHE`), shared by the gated and sweep scenes (columns 2-4 are identical pieces).
Roughness 0.7 (alpha 0.49) was chosen for the fixtures because the per-path noise falls with it (alpha 0.25: heavy tails).

Fixtures (`src/testbench/material_validation.*`, 45 scenarios, doctest OK): spheres r 0.4 m, base 0.8, roughness 0.7, sun 3 W/m2 toward
(-0.9, 0.25, 0.35) (terminator crosses the disc), environment 0. Rows: skin (colour 0.85/0.55/0.45, radiusScale 1/0.5/0.25, g 0) and
jade (0.30/0.70/0.45, 0.5/1/0.7, g 0.3: exercises the g term of the mapping). `subsurface-volumetric-sun`: columns (weight, radius m)
= (0, -), (0.5, 0.08), (1, 0.04), (1, 0.08), (1, 0.16) = weight {0, 0.5, 1} at the nominal radius and the sweep x0.5/x1/x2.
`-radius`: weight 1, radius 0.02/0.24/0.04/0.08/0.16 (x0.25 ... x3). `-subpixel`: weight {1, 1, 0.5, 0.5} x radius {0.0005, 0.002} m
(mfp <= 0.23 px), colour = base colour; `-subpixel-off`: the identical scene without a block (capture only). `-backlit`: sun behind
(V.L about -0.96), weight 1, radius 0.08/0.16, camera 4 m. Every carrier holds a volume block (mask 192 = subsurface | volume).

Runner (`tools/f15_check.py`, 82 tests OK): `subsurface-volumetric-sun` (kind `subsurface`: per piece mean relative LUMINANCE error <= 10 %,
p95 <= 25 %, floor 5 % of the piece's mean reference luminance), `-radius` (kind `radius-monotonic`: declared statistic = terminator
bleed = mean Y of the geometrically unlit ROI pixels / mean Y of the lit ones; per sweep series ordered by radius no reversal beyond 0.01
and a rise of at least 0.01 wherever the ground truth rises by that much; the twin is `radius-statistic-match`), both `"pending":
"ticket 17"`, `pending_matches` = `subsurface-off`; `-subpixel` (kind `equal`, engine capture vs the capture of its pair, max relative
luminance error <= 0.002, not an empty image; NOT pending: equal today; `--no-ground-truth`); `-backlit` (kind `subsurface-report`,
never gated). Controls: `subsurface-off`, `diffusion-off`, `radius-ignored` (the median radius 0.08 everywhere), `subsurface-leak`
(sub-pixel pieces rendered at 0.08 m).

Measured red state (M5 Max, `build/f15-check/t17-b`, `--run --expect green`: GREEN exit 0; every capture 0 archive misses, 0 render-thread
compiles, 0 fallbacks after the archive update; the first run failed with archive misses of the mask 192 pipelines):
- sun: FAIL, worst piece mean 2.12 / p95 4.06 (weight 0 pieces 0.0000: base model); twin vs subsurface-off passes (worst mean 1.2e-5).
  Controls all fail: subsurface-off 2.12, diffusion-off 1.46, radius-ignored 0.57 (worst piece mean).
- radius: FAIL (engine bleed 0.0000 everywhere); ground truth bleed skin 0.015, 0.041, 0.091, 0.190, 0.305 and jade 0.030, 0.104, 0.230,
  0.394, 0.562 for radius 0.02 ... 0.24 (rises at every step by >= 0.025); twin passes; all three controls fail.
- subpixel: pass, max relative luminance error 0.000000 on the four pieces (bit-identical today); leak control fails (7.0).
- backlit: reported, worst piece mean 0.99-1.00: the engine image is black to 0.0002-0.0046 mean Y against 0.013-0.106 for the truth.
Dropped controls: none. Interpretations: volumetric weight is the linear mix with the base model; luminance Rec.709 for the bound;
the sun is a delta emitter in the reference (the engine's disk of 0.00465 rad is immaterial for diffuse transport); the bound is
applied to every piece including the neutral weight-0 ones; the x2 / x3 pieces (red mfp 0.16-0.24 m on a 0.4 m sphere) are far
outside the diffusion regime, so the bound may be unreachable there (the decision rule of ticket 10 then applies: ticket 18).
Open: `archive_check`, `hitch_check`, O7 counts, shader validation not run (no engine change); the forced-overflow capture and the
"scenes without subsurface unchanged / diffusion pass cost" criteria belong to the engine ticket.

2026-11-14 — orchestrator decisions (taken before any engine result), applied to the harness:
1. The subsurface bound (mean <= 10 %, p95 <= 25 %) gates only pieces in the DIFFUSION regime: max over RGB of radius x radiusScale <=
   10 % of the piece size (the volume thickness = the sphere diameter, 0.8 m, so 0.08 m; screen-space diffusion is a diffusion
   approximation, the target case is the chess pieces: radius = base colour x 0.003 m on pieces a few cm tall). Pieces outside it stay in
   the radius SWEEP (monotonicity still required) and are measured and REPORTED against the bound, not gated, like the back-lit thick
   fixture. Implemented as `diffusion_regime` in the reference's sphere table (`DIFFUSION_REGIME_FRACTION = 0.1`, tolerance 1e-4) and
   per-piece `gated` / `bound_met` in `subsurface-volumetric-sun`; a scene with no gated piece is not a gate. The fixture needed no
   change: the gated SSS pieces are weight 0.5 and 1 at 0.08 m and weight 1 at 0.04 m for both materials (6 pieces, mfp max 0.04-0.08 m)
   plus the two neutral ones; only the x2 pieces (0.16 m, pieces 4 and 9) are reported. Re-evaluated on the same captures with
   `--evaluate --expect green` (`build/f15-check/t17-b`): GREEN exit 0; sun FAIL with pieces 1,2,3,6,7,8 failing (worst mean 2.12),
   twin passes; controls still fail on gated pieces (subsurface-off and diffusion-off on 1,2,3,6,7,8; radius-ignored on the x0.5 pieces 2
   and 7, worst mean 0.57). Tests: 83 runner and 14 reference tests OK.
2. Jade g = 0.3 stays. Declared mapping the engine is expected to use: HG anisotropy through the similarity relation (reduced scattering
   sigma_s' = sigma_s (1 - g)) to scale the diffusion radius (the multiple-scatter colour already goes through the draft's inversion
   with g); the ground truth uses the real HG walk, so a missing correction shows in the jade pieces.

2026-11-14 — engine side (F15.1, commits 7f8ab18 engine, 3ba6ba7 archive, d5c24dc runner).  The bound is MISSED on every gated SSS piece: the
decision rule of ticket 10 points at ticket 18; not applied here.
- Pipeline: resolve/fallback write the diffuse-irradiance target (constant SSS_OUTPUT 23, RGBA32F, texture 10 / colour attachment 6), the
  colour keeps (1 - w) diffuse + specular, `subsurface_diffuse` (shaders/subsurface.metal) spreads the premultiplied irradiance with the Burley
  profile (21/55 samples, mixture-quantile radii, 3D distance, area factor, same material, sub-pixel skip < 0.5 px) into a new colour version
  before reflections/atmosphere/post.  Static cost per volumetric pixel: 21 or 55 sample gathers (sss, depth, normal reads) + 16-step bisection
  per sample on the mixture CDF (about 100 exp per sample: the radii depend only on the material and could be hoisted into threadgroup memory
  if the measured cost asks for it); one RGBA32F target (33 MB at 1080p) and one extra colour texture, only when the scene has volumetric subsurface.
- Runner (`build/f15-check/t17-b`, GREEN exit 0): sun key FAIL, now asserted red as pending "ticket 18"; radius key pass; sub-pixel pass
  (max relative luminance error 0.000000 on the four pieces); back-lit reported; controls all fail their gates; 0 archive misses, 0 render-thread
  compiles on the five captures.  Per piece (mean / p95 relative luminance error, candidate Y vs ground-truth Y): skin 0.5@0.08 0.240 / 0.376
  (0.192 vs 0.153); skin 1@0.04 0.863 / 1.317 (0.206 vs 0.099); skin 1@0.08 0.821 / 1.417 (0.239 vs 0.117); skin 1@0.16 [reported] 0.985 / 1.689
  (0.268 vs 0.127); jade 0.5@0.08 0.292 / 0.750 (0.182 vs 0.152); jade 1@0.04 0.910 / 1.644 (0.196 vs 0.104); jade 1@0.08 0.736 / 1.604
  (0.222 vs 0.116); jade 1@0.16 [reported] 1.172 / 2.468 (0.250 vs 0.108); the weight-0 pieces 0.0000.  Sweep (terminator bleed engine vs truth):
  skin 0.0063 0.0262 0.0509 0.1377 0.1487 vs 0.0152 0.0407 0.0913 0.1898 0.3047; jade 0.0213 0.0910 0.1221 0.2220 0.2467 vs 0.0304 0.1042 0.2302
  0.3945 0.5617 (monotonic, under-estimated).  Back-lit (reported): mean error 0.956 / 0.986 / 0.995 / 1.000, candidate Y 0.0050 0.0018 0.0004
  0.0000 vs truth 0.0128 0.0332 0.0338 0.1063.
- Why the bound is missed (offline emulation of the shader on the t17-b snapshot reproduced the GPU per-piece Y to 4 digits, so its variants are
  trustworthy): (1) energy: the truth at fully lit pixels reflects 0.47-0.5 of colour x (1 - F) x E (skin 0.416/0.266/0.21 against colour
  0.85/0.55/0.45), the Mitsuba rough dielectric boundary loses energy on every internal reflection; the engine contract (colour = diffuse
  albedo, also the sub-pixel gate: colour = base colour must equal the base model) cannot reproduce it.  A boundary-aware effective albedo
  A (1 - F_i) / (1 - A F_i), F_i = 0.596 for ior 1.5, still leaves the engine 1.4x above the truth, and breaks the sub-pixel gate.  (2) shape:
  even with an ORACLE energy factor (0.72) and a tuned profile width (x0.5 ... x2) the best per-piece mean errors on the gated pieces are
  0.11-0.23 (worst 0.31-0.42 on the jade pieces), p95 0.31-0.9: a screen-space profile cannot reach the 10 % / 25 % bound here.  So ticket 18 is
  warranted by the shape error alone; its mapping must also fix the boundary energy convention (colour = multiple-scattering albedo of the
  medium under the engine's boundary) or the same 2x offset appears.
- Verification (M5 Max): `--debug-lobe-guide 8` on subsurface-volumetric-sun PASS (extended 13314 pixels), `--debug-lobe-guide-corrupt` FAIL
  (13314 mismatches, exit 1); `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` 200 frames exit 0 without a validation message (the first attempt
  found a real defect: the base fallback pipeline had no attachment 6; fixed with the SSS_OUTPUT base fallback); warmup 64: GPU allocations 0;
  forced overflow (`tools/f15_subsurface_overflow_check.sh`): mean |overflow - normal| 1.3e-5 over the covered pixels (limit 2e-3) PASS;
  `tools/f15_lobe_guide_overflow_check.sh` on the fixture: guide dumps identical; a base scene capture (base-grid-sun) is bit-identical to the
  t16 capture; ctest 19/19, runner tests 83 OK.
- Carried: DI/ReSTIR/clustered and GI composite diffuse is not diffused (the DI kernels keep the base model for volumetric pixels, the thin
  consumers ignore the VOLUMETRIC flag; no diffuse-irradiance write from DI); the ambient diffuse of an external reflection composite and
  DI local lights are not diffused; combinations with coat/sheen/anisotropy/iridescence/transmission keep the base diffuse; the on-tile
  resolve experiment has no diffusion; weight/colour/radius-scale textures modulate the irradiance but not the profile shape; visual_check,
  archive_check, hitch_check and the diffusion pass cost are the orchestrator's.

2026-11-14 — orchestrator decision on the engine record (integrated as 7c96691…b618511; the archive conflict was resolved by
semantic filtering of the commit's own additions, +22 compute and +6 render).
- Measured gap: the engine misses the bound by a lot. The engine agent's emulation shows the ground truth returns about
  half the energy of colour x (1 - F).
- Cause: the reference maps colour to a single-scattering albedo with the draft's Kulla-Conty inversion. That inversion,
  like van de Hulst, ignores the dielectric boundary. Under an ior-1.5 boundary, internal reflection traps light and
  the observed albedo falls well below colour.
- OpenPBR, the semantics of ADR-0005, defines subsurface_color as the OBSERVED albedo: "the fraction of the energy
  transmitted through the interface that is transmitted back out and observed", E_ms = (1 - E_spec) C. It names the
  boundary-ignoring inversions as approximations and prescribes no formula.
- The engine's contract follows that definition, and so does the sub-pixel key (colour = base colour equals no block).
- Decision: the ground truth's medium albedo is chosen so that a semi-infinite slab under the SAME boundary satisfies
  the OpenPBR relation. The boundary-aware inversion is tabulated with Mitsuba (Sonnet agent running).
- Unchanged:
  - gates, thresholds, fixtures and controls;
  - the draft's Kulla-Conty formula stays documented as the draft's approximation (the loader maps multiscatterColor to
    C, flagged `draft`).
- Ticket 18 (decision 10's rule): it is decided on the re-measured gap against the corrected reference, not on the
  current numbers.
||||||| parent of 8e097a6 (F15.3 (ticket 17): record the boundary-aware reference and its re-measurement)

2026-11-14 — reference correction (boundary-aware inversion, commit `a32ee1a`; no engine or shader change).
- Ground truth version 3: the medium albedo alpha(C) solves the OpenPBR relation measured with Mitsuba under the SAME ior 1.5 roughdielectric
  boundary (alpha_b 0.49): A = (R_total - R_spec) / (1 - R_spec) of a 1000-mfp sphere under a constant environment (disc mean = cosine-weighted
  diffuse incidence; R_spec from an absorbing medium).  Table `tools/f15_subsurface_albedo.json` (generator `f15_subsurface_reference.py
  albedo-table`, 64 spp per half, 128 px, seed 20261114; noise <= 0.004 below alpha 0.99, heavy tails to 0.04 on the plateau: weighted isotonic
  fit).  Angle dependence of A (0 / 45 / 70 degrees vs diffuse): within 6 % up to alpha 0.99.  Index-matched check: |A - rho| <= 0.043 against
  the Kulla-Conty relation.  Negative control: the draft mapping behind the boundary returns 0.39 (C 0.85), 0.21 (0.55), 0.16 (0.45), 0.09 (0.30).
- FINDING: Mitsuba's single-scattering GGX `roughdielectric` loses energy at every internal reflection, so even alpha = 1 (no absorption) returns
  only A_max = 0.44-0.47 at alpha_b 0.49 (0.34 at 0.7, 0.25 at 1.0).  Colours above A_max (every fixture colour except 0.30 and 0.45) are not
  reachable by any medium under this boundary: they saturate at alpha = 1 (`albedo_saturated` in reference.json).  The OpenPBR relation can
  only be honoured for C <= A_max; above it the truth is the most translucent medium the boundary allows, not an observed albedo C.
- Re-measure (`build/f15-check/t17r`, GREEN exit 0; first attempt `t17r-cold-first` discarded: the first process after a build served the
  fallback pipelines (141/153 archive swaps, 774 fallback draws), the capture had no diffusion; the rerun has 153/153 and 17 fallback draws):
  sun FAIL (pending ticket 18 kept), worst gated piece mean 0.417 / p95 0.979.  Per piece mean/p95, old truth -> new truth: skin 0.5@0.08 0.240/0.376
  -> 0.297/0.806; skin 1@0.04 0.863/1.317 -> 0.283/0.611; skin 1@0.08 0.821/1.417 -> 0.332/0.677; skin 1@0.16 [reported] 0.985/1.689 -> 0.435/0.932;
  jade 0.5@0.08 0.292/0.750 -> 0.285/0.797; jade 1@0.04 0.910/1.644 -> 0.360/0.842; jade 1@0.08 0.736/1.604 -> 0.417/0.979; jade 1@0.16 [reported]
  1.172/2.468 -> 0.740/1.765.  Residual reference noise (halves): mean 1.2-5.6 %, p95 3.5-23 % (piece 2, alpha -> 1 heavy tails).  Radius sweep
  passes (bleed truth skin 0.021 0.059 0.132 0.230 0.343, jade 0.061 0.141 0.249 0.404 0.570 vs engine unchanged).  Back-lit reported: mean error
  0.972 0.989 0.996 1.000.  Controls all fail: subsurface-off 1.57 (gated 1,2,3,6,7,8), diffusion-off 1.06, radius-ignored 0.49 (pieces 2, 7);
  sub-pixel pass (0.000000), leak control fails.

2026-11-14 — fixtures move to roughness 0.35 (ORCHESTRATOR DECISION, recorded before any measurement on the new fixture; commit `1d1b642`).
Decision: at roughness 0.7 (alpha_b 0.49) the Mitsuba rough dielectric cannot represent the OpenPBR material (single-scattering GGX loses
energy at every internal reflection, A_max < most fixture colours); the volumetric fixtures move to a roughness at which every fixture colour
is reachable; roughness 0.7 subsurface is declared "no ground truth" (like rough thin-walled glass), its numbers kept as information.
Gates, thresholds, controls and the inversion definition unchanged.
- CORRECTION of the first record above: the A_max = 0.45 / 0.447 of the first table was biased low by Russian roulette from depth 64 (the long
  walks of a non-absorbing medium were under-sampled; also a sigma_t 1000 sphere).  With roulette from depth 5000 (20000 on a 30-mfp sphere)
  the plateau is stable across sphere size (10-100 mfp) and ior 1 gives exactly A = 1: A at alpha = 1 under the ior 1.5 boundary =
  0.980 (roughness 0.15), 0.948 (0.25), 0.876 (0.35), 0.717 (0.5), 0.503 (0.7), 0.986 (0.04); noise 0.0001-0.0004 (20000-roulette check
  on 30 mfp: 0.993, 0.960, 0.887, 0.724, 0.507).  The qualitative finding stands at 0.7 (0.50), smaller loss than first stated.
- Rule (stated before the change): the LARGEST roughness of {0.15, 0.25, 0.35, 0.5} with A_max >= 0.85 + 3 sigma.  0.5 fails (0.72), 0.35 passes
  (0.876 / 0.887, sigma 0.0003) -> 0.35.  src/testbench: volRoughness, subRoughness, backRoughness and the doctests only; roughness is data: 0
  archive misses, 0 render-thread compiles, 0 fallbacks on the five captures.
- Table regenerated (`tools/f15_subsurface_albedo.json`, schema 2, sigma_t 100, 64 spp per half, roulette 5000, max noise 0.0013): curves for
  alpha_b 0.1225 (g 0 and 0.3) and the index-matched control (|A - rho| <= 0.004 up to rho 0.7); angle dependence <= 3 % (0 / 45 / 70 deg vs
  diffuse); the draft Kulla-Conty mapping returns 0.76 x C at 0.85, 0.63 at 0.7, 0.53 at 0.55, 0.48 at 0.45, 0.41 at 0.30 behind this boundary.
  GROUND_TRUTH_VERSION 4, pieces rendered with roulette from depth 1000.  The 0.7 curves are not kept (the old table was biased); A_max(0.7) is in the table.
- Re-measurement (`build/f15-check/t17s`, GREEN exit 0, `albedo_saturated` false everywhere):
  sun FAIL (pending "ticket 18" kept), worst gated piece mean 0.364 / p95 0.924.  Per piece mean/p95 (candidate Y vs reference Y): skin 0.5@0.08
  0.355/0.924 (0.1945/0.2063); skin 1@0.04 0.252/0.794 (0.2076/0.2124); skin 1@0.08 0.257/0.789 (0.2399/0.2484); skin 1@0.16 [reported]
  0.231/0.607 (0.2678/0.2553); jade 0.5@0.08 0.364/0.904 (0.1841/0.2004); jade 1@0.04 0.299/0.870 (0.1971/0.2104); jade 1@0.08 0.309/0.737
  (0.2230/0.2202); jade 1@0.16 [reported] 0.400/0.972 (0.2499/0.1989).  The mean luminance now agrees within 1-8 % on the gated pieces: the energy
  convention is right and the remaining error is spatial.  Reference residual noise 3-5 % mean, 9-13 % p95 (sweep max 6.5 % mean, back-lit 4-6 / 10-15).
  Radius sweep now FAILS (key marked pending "ticket 18", it was passing against the biased truth): bleed truth skin 0.033 0.073 0.157 0.302
  0.427, jade 0.088 0.191 0.338 0.548 0.705 against engine skin 0.006 0.026 0.051 0.138 0.147 (flat above 0.16 m), jade 0.021 0.090 0.122 0.222
  0.243 (jade passes, skin fails its rise rule at 0.16 -> 0.24).  Back-lit reported: mean error 0.979 0.994 1.001 1.000 (candidate Y 0.032 0.009
  0.001 0.000 vs truth 0.055 0.090 0.085 0.172).  Controls all fail: subsurface-off 1.00 (gated 1,2,3,6,7,8), diffusion-off 0.67, radius-ignored
  0.48 (pieces 2, 7); sub-pixel pass 0.000000, leak control fails.
- Reading: with equal energy the miss is the profile shape (the truth's lateral spread is 2-4x the engine's Burley width at the sweep's large
  radii, the bleed numbers; the error stays 25-36 % mean with the luminance means within 8 %), so the decision-10 rule triggers ticket 18 on VALID
  ground truth.  The residual reference noise (3-5 % mean) is a minor part of it.

2026-11-14 — orchestrator: re-measure on valid ground truth and the profile-mapping experiment. Integrated as 340376f…935b8f1 and
d32de4b (sub-pixel test follows roughness 0.35); ctest 20/20.
- Fixtures moved to boundary roughness 0.35 by the rule fixed before any engine measurement: the largest roughness with
  A_max >= 0.85 + 3 sigma. A_max is 0.876 there; at 0.7 it is 0.50, after the roulette bias was removed.
- With the boundary-aware inversion the energy agrees: candidate mean luminance is within 1-8 % of the truth.
- The gated pieces still miss: mean 0.25-0.36, p95 0.74-0.92. The radius sweep now fails too (skin), so it is pending ticket 18.
- Profile-mapping experiment (shader emulation that matches the GPU to 4 digits, against the t17s references):

| variant | gated mean | gated p95 |
|---|---|---|
| V0, current CB15 mfp fit | 0.25-0.36 | 0.74-0.92 |
| V1, CB15 dmfp fit with the boundary-aware alpha_ss (best principled) | 0.18-0.28 | 0.50-0.77 |
| V2, dmfp with the Kulla-Conty alpha_ss | at least as bad as V0 | at least as bad as V0 |
| V3, plain width scale k = 1.5-4 (diagnostic, unprincipled) | best piece 0.147 | best piece 0.35 |

  - No variant meets the bound on any gated piece.
  - The bleed saturates below the truth at large radius: a two-exponential Burley profile cannot carry the long tail of a near-lossless
    medium behind the boundary.
- Decision 10's rule therefore triggers ticket 18 on a correctly parameterised diffusion.
- The reference noise floor per piece (mean 0.03-0.05, p95 0.09-0.13) leaves the bound reachable.
