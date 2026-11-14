# 17: Subsurface feature, volumetric mode (screen-space diffusion)

Roadmap: F15.1, F15.3

**What to build:** Volumetric subsurface per ADR-0005: resolve and DI write a diffuse-irradiance target for subsurface pixels, a Burley normalised-diffusion pass in screen space (21/55 samples, metres→pixels radius, sub-pixel skip, same-material and depth test) runs after diffuse denoising, recombined with the specular terms; draft `KHR_materials_scatter` volumetric mode mapped through its Kulla–Conty inversion. Back-lit translucency of thick volumes is out of reach of screen-space diffusion and declared.

**Blocked by:** 16

**Status:** ready-for-agent

- [ ] Side-lit volumetric fixtures vs Mitsuba 3.9.1 `volpath` random walk (homogeneous medium inside a boundary with the engine's roughness) on the piece mask: decision-11 subsurface bound
- [ ] Back-lit thick fixture measured and reported, not gated (declared limit of screen-space diffusion)
- [ ] Radius sweep monotonic; sub-pixel radius equals subsurface-off within FP noise; negative control (diffusion off) fails
- [ ] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Scenes without subsurface unchanged (visual_check 0 px); diffusion pass cost measured
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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
