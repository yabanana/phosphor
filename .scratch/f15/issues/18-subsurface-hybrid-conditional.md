# 18: Conditional: hybrid ray-traced single scattering for volumetric subsurface

Roadmap: F15.1, F15.3

**What to build:** Only if ticket 17 misses its side-lit bound: add ray-traced zero/single scattering from the exit point to the screen-space diffusion (profile minus single-scatter, no double count). Own bound, timebox and fallback fixed now.

**Blocked by:** 17 (and only if its bound is missed)

**Status:** resolved by the FALLBACK for the default diffusion tier (2026-11-14: bound not met, screen-space diffusion stays); (2026-11-16, round 2: by owner decision the hybrid is implemented as the optional RT random walk (`--subsurface-mode rt-walk`, `--rt on`; see the ticket 17 comment of the same date) and meets the side-lit bound (0.0635 / 0.1708 vs 0.10 / 0.25) and the 2x back-lit bound (0.0919 / 0.2288 vs 0.20 / 0.50) when enabled; the LUT is generalised over the roughness (5 x 2 x 65 table, roughness 0.15..0.7, g 0..0.3; `subsurface-volumetric-rough@rt-walk` worst gated piece 0.0645 / 0.1756, controls fail; the roughness 0.15 knot has no fixture); cost per subsurface pixel is structural only here, the quiet-machine A/B is pending the orchestrator's final battery (ticket 17))

- [ ] Bound: the ticket-17 side-lit subsurface bound on the same fixtures; additionally the back-lit fixture within 2× the side-lit bound — NOT met (measured gap below)
- [x] Timebox 2 weeks; if not met, fallback = keep screen-space diffusion, F15.1 subsurface stays explicitly partial (unticked for the volumetric mode) with the measured gap reported
- [ ] Cost per subsurface pixel measured; scenes without subsurface unchanged — no engine change adopted (nothing to measure); the screen-space pass cost goes to ticket 24
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-14 — orchestrator: TRIGGERED by decision 10's rule. Ticket 17 misses its side-lit bound on valid OpenPBR ground truth
(roughness 0.35, boundary-aware inversion), with the energy right and the profile shape wrong; see ticket 17's last comment.
- Variant choice within the authorized phase (spike results choose variants). The measured cause is the multiple-scattering profile
  shape, and the RT single scattering named in this ticket mostly carries low orders and back-lit light. Two candidates, in order:
  1. a radial diffusion profile TABULATED from the same Mitsuba transport and boundary as the ground truth (semi-infinite slab,
     searchlight), replacing the Burley fit in the screen-space pass;
  2. RT zero/single scattering from the exit point (Zhang and d'Eon 2025; research/realtime-transmission-sss.md S6), profile minus
     single scatter, for the back-lit bound.
- Unchanged: the bound, the timebox and the fallback stated in this ticket.

2026-11-14 — spike 1 (tabulated profile), emulation only; the bound is NOT met in emulation, so nothing was implemented in the engine.
- Generator `tools/f15_subsurface_profile.py` (+ `tools/test_f15_subsurface_profile.py`, 5 tests OK; data `tools/f15_subsurface_profile.json`, 22 profiles,
  about 6 KB as float CDFs): searchlight at normal incidence on a semi-infinite medium behind the ground truth's boundary (ior 1.5, GGX alpha_b 0.1225,
  VNDF microfacet walk), HG g in {0, 0.3}, colour C in 11 values (0.05 ... 0.87) through the same boundary-aware inversion of
  `f15_subsurface_albedo.json`; exit flux binned in 64 log-spaced r/mfp bins (0.02 ... 64), stored as the cumulative flux per edge.  A numpy walk
  (not Mitsuba: a Mitsuba searchlight needs light tracing through the refractive boundary and costs hours); it reproduces the Mitsuba albedo table
  (exit flux 0.7278 at alpha_ss 0.999 against (1 - R_spec) A(0 deg) = 0.7278 x 0.93 -> within 0.02 at 0.5 / 0.95 / 0.99 in the test).  Generation 134 s.
- Emulation (`tools/f15_subsurface_emulation.py`, same gather: 55 samples, area factor, same material, self-normalised, amplitude = C, sub-pixel skip;
  radii from the mixture CDF of the three tabulated profiles, weights P_c / P_mix with the tabulated density), t17s references, gated pieces mean / p95
  (bound 0.10 / 0.25): skin 0.5@0.08 0.226/0.743; skin 1@0.04 0.144/0.524; skin 1@0.08 0.155/0.424; jade 0.5@0.08 0.274/0.766; jade 1@0.04
  0.234/0.614; jade 1@0.08 0.208/0.524; reported: skin 1@0.16 0.166/0.427, jade 1@0.16 0.324/0.849.  Against V0 (0.25-0.36) and V1 (0.18-0.28) it is the
  best variant (about 35 % lower mean than V0) and still 1.4-2.7x over the bound.  Sweep bleed (engine vs truth, skin 0.02/0.04/0.08/0.16/0.24):
  0.019 0.065 0.114 0.206 0.210 vs 0.033 0.073 0.157 0.302 0.427; jade 0.071 0.136 0.238 0.327 0.358 vs 0.088 0.191 0.338 0.548 0.705.
- Cause of the remaining gap (rows of the t17s capture): the tail on the dark side and the lit limb.  The truth's pieces are small compared with the
  diffusion length of a near-lossless medium (alpha_ss 0.9998 -> l_d about 60 mfp, the sphere radius is 10 mfp): light is confined and re-emerges
  over the whole object, the lit limb is dimmer (red 0.42-0.53 vs 0.69-0.72) and the terminator tail brighter than any semi-infinite profile can give;
  dropping the self-normalisation (letting the hidden surface take its share) is worse (means 0.17-0.42).  A screen-space gather of the visible
  surface has no access to the interior or the back; the ray-traced single scattering of the plan (S6) carries low orders and back-lit light, not
  this multiple-scattering confinement.  Noise floor of the references: mean 0.03-0.046, p95 0.085-0.13 (reachable in principle).
- Decision under the ticket's rule: step 2 (RT single scattering) is not started; the screen-space diffusion stays, the volumetric mode is reported
  partial.  The tabulated profile is an optional measured improvement (sweep and gated pieces) whose price is this LUT and a gather that samples by
  its CDF; it is not adopted here because it does not change the verdict.

2026-11-14 — orchestrator: conclusion. The spike is integrated as 4fc42af (generator, LUT, emulation, test); ctest 21/21.
- What was measured:
  - A radial profile tabulated from the same transport and boundary, used in the screen-space gather, gives gated mean 0.14-0.27
    and p95 0.42-0.77. That is the best measured variant, but still 1.4-2.7x over the bound on every gated piece.
  - The remaining error is CONFINEMENT. To return the observed colour behind the boundary (OpenPBR), the medium needs alpha_ss
    close to 1, which gives a diffusion length of about 60 mfp against piece radii of about 10 mfp. Light spreads over the whole
    object: the lit limb is dimmer and the terminator tail is brighter than any semi-infinite profile.
  - A screen-space gather cannot model that. RT zero/single scattering (S6) adds low orders and back-lit light, not this
    multiple-scattering confinement, so step 2 was not started: by this analysis it cannot close the side-lit gap.
- Decision (orchestrator):
  - The ticket's FALLBACK applies. The screen-space diffusion with the CB15 mfp fit stays. The F15.1 volumetric subsurface mode is
    explicitly PARTIAL, and the volumetric keys keep their pending marker naming this ticket.
  - Measured gap: gated mean 0.25-0.36 and p95 0.74-0.92 against 0.10 / 0.25. Back-lit mean error about 1 (the engine is nearly
    black there).
  - The tabulated profile and the V1 dmfp mapping are recorded as measured candidates and NOT adopted. Neither meets the bound;
    the LUT exists only for roughness 0.35 and would need a roughness axis. They stay selectable.
  - The approach that could close the gap is a real-time volumetric random walk (or path-traced SSS) with denoising. That is a
    later-phase EDGE candidate, outside this ticket's scope.
- The thin-walled subsurface mode (ticket 16) is unaffected and stays verified.

2026-11-15 — closure audit (ticket 24):

- 1 bound: NOT MET (see 17 #1, back-lit error ~1). Open.
- 2 timebox / fallback: MET (fallback executed; spike 1 tabulated profile emulation 4fc42af; step 2 RT single scattering not started by analysis, not by timebox expiry). Ticked.
- 3 cost per pixel + unchanged scenes: NOT MET / n.a. (no engine change adopted; screen-space pass cost deferred to 24, unmeasured). Open.
- 4 archive/hitch/O7: MET vacuously (no engine or pipeline change). Ticked.


## 2026-11-15 -- RT tier (agent sss): the hybrid is implemented as the optional RT random walk; the LUT is generalised over the roughness

The ticket-18 analysis (confinement, step 2 "cannot close the side-lit gap") concerned RT SINGLE scattering added to a screen-space diffusion.  The owner's decision makes the
whole transport RT (`--subsurface-mode rt-walk`, ticket 17 comment of the same date): the multiple-scattering confinement is carried by the walk through the real mesh, which the
analysis named as the approach that could close the gap.  Bounds as fixed here: side-lit = the ticket-17 bound on the same fixtures (met: 0.0635 / 0.1708 vs 0.10 / 0.25), back-lit within
2x (met: 0.0919 / 0.2288 vs 0.20 / 0.50).
- [x] Bound / back-lit bound: met on the declared RT mode (the diffusion tier keeps its measured gap, marked in ticket 17).
- [x] Timebox / fallback: the fallback (screen-space diffusion) is the default and the automatic fallback without `--rt on` (logged, in the report, asserted by a runner key).
- [~] Cost per subsurface pixel: structural only here (6 ms at 4 walks, 67-75 ms at 64 walks per frame for the 1280x720 fixtures under load: not a claim); quiet-machine A/B in ticket 17.
- [x] Archive, scenes without subsurface unchanged: sss-1.json, sss-2.json; exactness PFMs identical, visual_check 0 px.

LUT over the roughness (boundary-aware single-scatter albedo, valid only at 0.35 before): `f15_subsurface_reference.py albedo-curves` measured the observed-albedo curves of
roughness 0.15 / 0.25 / 0.5 / 0.7 at g 0 and 0.3 (tools/f15_subsurface_albedo_curves.json, Mitsuba 3.9.1, 64 spp per half, 100-mfp sphere; the plateaus reproduce the independent
7-point A_max table of the first measurement: 0.980 / 0.9485 / 0.8766 / 0.7174 / 0.5034 at g 0); `tools/f15_subsurface_lut.py` inverts the ten curves into the 5 x 2 x 65 table
(`renderer/subsurface_walk_table.h`, generated, staleness test); the walk interpolates linearly in roughness and g, clamped to 0.15..0.7 and 0..0.3 (declared).  Evidence:
`subsurface-volumetric-rough@rt-walk` (roughness {0.25, 0.5, 0.7}, side-lit, Mitsuba 8192 spp per grey render, engine 256 walks x 96 frames): worst gated piece (roughness <= 0.5,
declared before measuring) mean 0.0645 / p95 0.1756; roughness 0.5 pieces 0.5-1.9 %, the reported roughness 0.7 pieces 0.5-1.1 %; controls subsurface-off, diffusion-off, walk-lights-off FAIL.
The roughness 0.15 knot has no fixture (its Mitsuba reference is the noisiest case, alpha 0.0225); it is covered by its curve and the table test only.  An intermediate capture at
2048 spp / 64 walks read 0.1275 / 0.3365: reference residual noise of its own 7.5-8.8 % and the engine's, both reduced by sampling, the bound untouched.
Colours above the largest albedo the boundary lets through (A_max) saturate (alpha = 1) in the engine exactly as in the reference (both read the same plateau): the observed albedo
is then below the requested colour (declared).

## 2026-11-15 product output
The albedo table (roughness 0.15..0.7, g 0..0.3) feeds the product output unchanged; the `-product` keys and the moving-camera clip are recorded in ticket 17 (same date).
