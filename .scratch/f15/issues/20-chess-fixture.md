# 20: Chess fixture: the roadmap's Track Studio case

Roadmap: F15.1, F15.3

**What to build:** The six Open Chess Set materials render from public assets: ABeautifulGame (pinned) unchanged for the pawn tops, a material-description override file for the four kings/queens with the MaterialX values (weight map, colour = base colour, radius = base colour × scale 0.003 / Queen_W 0.001); back-light and side-light rigs and parameter sweeps; coloured shadows of the pawn tops; conversion note committed with the fixture.

**Blocked by:** 13, 14, 17

**Status:** resolved PARTIAL (2026-11-14): criteria 1, 3 and 4 met on M5; criterion 2 PARTIAL (2026-11-16, round 2, owner decision A: the king/queen subsurface bound on the RT walk is built and evaluated, RED, declared OPEN; see the round-2 section at the end): monotone gate met, the ticket-17 bound on the piece masks cannot be built (reference texture-free and homogeneous, scattering weight map not in the pinned subset); (2026-11-16, round 2: the redundant `setDepthStencilState` / `setViewport` notices are removed (65f2f10) and the `setComputePipelineState` notices are gone after the eight reflection samples became one graph pass (0 notices, 0 validation messages on both back-light rigs under API + shader validation); the back-light rig under shader validation is explained (the engine's 5 s frame wait, not a validation message) and validated at 640x360 with `tools/f15_chess_validate.py` (transmission 0 and 1 exit 0); chess binning A/B/A is recorded in ticket 24 (side-light -1.8 %, shadow rig +6.7 %, back-light +9.8 %, the 5 % rule is not met, union stays default); the RT walk subsurface tier is optional and its chess sssbound gate (Mitsuba references rendering, gate not yet evaluated) is pending the orchestrator's final battery)

- [x] Back-light rig: the scene is visible through the pawn tops; transmission 0 → 1 increases transmitted luminance on the glass mask monotonically; tint changes chroma as specified
- [ ] Side-light rig: subsurface weight 0 → 1 and scale × 0.5 / × 2 change the terminator softness monotonically (MET, gated); ticket-17 bound on the piece masks (NOT BUILDABLE: the ground truth is texture-free and homogeneous, see the record)
- [x] Pawn-top shadows attenuated and tinted (qualitative: curved glass) (RT at the default settings; CSM with a map of 8192: the default 2048 map loses the 2 cm casters)
- [x] Support report lists every feature as implemented or a named approximation; conversion note committed

## Verification record

2026-11-14 -- agent (Opus 5.5), M5 Max, build/release at the commits below, ABeautifulGame from `build/khronos-corpus` (glTF-Sample-Assets 8f49239, CC-BY-4.0, untouched, never committed).
Machine shared with the ticket 21/22 agents: every capture ran under `/tmp/phosphor-gpu-verification.lock`; no timing claim (frame times quoted are indicative).

Commits: `a9c96e5` (bench, override file, runner, tests), `fb45ade` (targeted archive update), `5d5babd` (lit-white masks, pipeline health, conversion note), `920eeb8` (shadow region, CSM map size),
and the commit with this record (support string, ctest registration).

### What was built

* `ChessFixture` (`src/testbench/chess_fixture.*`): bench 2 `--material-scenario chess-backlight | chess-sidelight | chess-shadow` with `/key=value` options (`transmission`, `tint`, `panel`, `weight`,
  `scale`, `view`, `sss`, `tops`, `mask`), `--scene <ABeautifulGame.gltf>` and `--material-overrides <file>` (accepted for chess scenarios; its texture paths are relative to the glTF's directory).
  Fixed cameras, no motion.  Doctests (`tests/test_chess_fixture.cpp`, 5 cases): names and option errors, override values, rig geometry (the tops project into the frame and the straight line through each
  reaches the panel), the pinned asset (pawn tops untouched: transmission 1, tint, thickness 0.22, infinite attenuation; the override textures decode to the same texels as the base colour; hidden
  instances; mask materials); they skip cleanly without the corpus.
* `fixtures/chess/chess-subsurface.overrides.json` and `fixtures/chess/README.md` (the conversion note: every MaterialX Standard Surface input, choice and loss).
* `tools/f15_chess_check.py` (+ `tools/test_f15_chess_check.py`, 24 cases, ctest `f15_chess_check_python`): captures, masks, gates, controls; `--o7` mode.
* Full tree: `phosphor_tests` 892 cases / 2,204,916 assertions pass; ctest 22/22 (the chess doctest skips when the corpus is absent).

### Rig results (captures `build/f15-chess/run3`, 1920x1080, RT glass transport accumulated 112 frames; runner GREEN, every control fails)

Back-light (mask 5684 px, centroid within 2 px of the projection of the eight black tops):

| gate | result | bound | controls (must fail) |
|---|---|---|---|
| monotone | transmitted luminance on the mask 0.1019, 0.2285, 0.3551, 0.4818, 0.6084 for transmission 0, 0.25, 0.5, 0.75, 1 (steps 0.1266, 0.1266, 0.1267, 0.1266) | each step >= 1 % of the final value (0.0061) | transmission ignored, order reversed: both fail |
| visible | 0.6084 with the emissive backdrop, 0.0524 without it: ratio 11.6 | >= 1.5 | backdrop removed: fails |
| tint | asset / white-tint per channel (0.307, 0.505, 0.456); chroma R/G 0.6077 against 0.599 (1.4 %), B/G 0.9021 against 0.900 (0.2 %) | <= 15 % | tint ignored: fails |

Side-light (black view: King_Black 35,840 px, Queen_Black 31,946 px; terminator bleed = mean luminance of the geometrically unlit pixels over the lit ones, regions from the weight 0 capture):

| piece | weight 0 / 0.5 / 1 | scale x0.5 / x1 / x2 (weight 1) | bound |
|---|---|---|---|
| King_Black (0.003 m) | 0.0011 / 0.0399 / 0.0856 | 0.0400 / 0.0856 / 0.1328 | every step >= 0.002 (steps 0.039-0.047) |
| Queen_Black (0.003 m) | 0.0003 / 0.0429 / 0.0921 | 0.0445 / 0.0921 / 0.1346 | same |

White view (weight 0 -> 1): King_White (0.003 m) 0.0008 -> 0.1472, Queen_White (0.001 m) 0.0011 -> 0.0481 (the bleed scales with the radius: 3.06 for a radius ratio of 3).  Controls (subsurface ignored,
scale ignored, order reversed) all fail.  Scale x0.5 and weight 0.5 give nearly the same bleed (0.0400 and 0.0399): the weight is a linear mix and the profile is linear in the radius at this size.

Shadows (receiver region = floor pixels the opaque top shadows and that are lit with the tops hidden, 11,072 px RT / 12,629 px CSM, black row only, the tops' own pixels excluded):

| mode | T (RGB) | T (luma) | chroma R/G, B/G (expected 0.599, 0.900) |
|---|---|---|---|
| RT (`--shadows rt`) | 0.259, 0.433, 0.390 | 0.391 | 0.5988 (0.04 %), 0.9001 (0.01 %) |
| CSM (`--shadows csm`, map 8192) | 0.356, 0.511, 0.474 | 0.474 | 0.697 (16.3 %), 0.929 (3.2 %) |

Gate: 0.05 <= T_luma <= 0.95 and chroma within 20 %.  Controls: transmission ignored (A := the opaque control) fails the attenuation gate, tint ignored (A := the white-tint capture) fails the chroma gate.
Finding: with the default CSM map (2048) T = 1.03 and chroma 1.0 (no shadow at all from the glass tops while the opaque control shadows): cascade 0 spans about 12 m (shadow distance 120 m), 1 cm per
texel against 2 cm tops.  An 8192 map resolves them (probe `build/f15-chess/csm8k`).  The CSM captures of the fixture therefore carry `--shadow-map-size 8192`, a declared setting (a 1 m scene), not
a looser bound; the default-map result stays in this record.  The behaviour for small casters under the default CSM map is a candidate for ticket 24 (cascade fit to the scene).

### Ticket-17 bound on the piece masks: NOT BUILDABLE (partial, reported, nothing invented)

`tools/f15_subsurface_reference.py` (Mitsuba volpath) cannot represent the chess pieces, for these reasons found in its code and in the asset:
1. `material_volumetric` raises "a subsurface texture is set; the reference is texture-free" when the weight, colour or radius-scale texture word is set: the override binds colour and radius to the base colour texture.
2. One homogeneous medium per instance (single colour, single mean free path, `render_grey`): the pieces' colour spans luma 0.04 (black king) to 0.7 and the MaterialX radius (texel x 3 mm) 0.09 to 2.3 mm
   (measured on the four base colour JPEGs); a surface texture is not a volume albedo.
3. One boundary roughness (alpha of the factor), with the measured albedo table valid only for roughness <= 0.35 and colours <= 0.85 (`A_max`); the pieces' roughness texture spans 0.15-0.60 and the
   metallic texture reaches 1 (crowns), and there is no metal in the model.
4. The base-model OFF image (`tools/f15_reference.py`, Appendix B) refuses textures too: the whole stack is texture-free by design.
5. Even where it applies (spheres in the diffusion regime) the volumetric bound is NOT met (ticket 17: gated mean 0.25-0.36, p95 0.74-0.92; ticket 18 fallback applied).  The pieces would be in the
   diffusion regime (mean free path 3 mm against the 0.04 m marker thickness, 10 % rule), so the bound would gate them if a ground truth existed.

Instead the side-light key gates the engine side (monotone bleed with its controls) and records the king/queen consistency ratio above.  The criterion stays unticked.
Engine caveat found while doing this: the subsurface support entry declares `profile_shape_ignores_textures`: the diffusion profile uses the scalar `radius x radiusScale`, so the base-colour radius
texture (0.09-2.3 mm) acts as a uniform 3 mm; the measured bleed is that of 3 mm everywhere.  Listed in the support list below.

### Support report (read from the captures' `materials` section, schema `phosphor.materials.v2`; gate: every row implemented or an approximation that names itself)

| material | feature | support | named approximations (abridged) |
|---|---|---|---|
| Pawn_Top_Black / _White | transmission | approximated | rt_refraction_one_fresnel_chosen_sample; rt_off_probes_environment_along_refracted_ray; factor_at_visible_surface_only; at_most_4_interface_events; schlick_fresnel; caustics_excluded; secondary_gi_and_probe_hits_ignore_transmission; rt_shadow_rays_attenuate_through_glass_straight_line; csm_shadows_one_transmittance_layer_per_texel; csm_static_cache_keeps_casters_opaque (the last two replace the stale `csm_shadows_opaque`, fixed with this record) |
| Pawn_Top_Black / _White | volume | approximated | beer_lambert_along_real_inside_segment_with_rt; rt_off_uses_thickness_hint; thickness_texture_ignored_with_rt; no_scattering (thickness 0.22, infinite attenuation: no absorption) |
| King / Queen, Black / White | subsurface (volumetric) | approximated, PARTIAL (tickets 17/18) | screen_space_burley_diffusion; resolve_lights_only_di_gi_composite_terms_not_diffused; no_back_lit_translucency; profile_shape_ignores_textures; anisotropy_by_similarity_relation; not_composed_with_other_diffuse_lobes |
| King / Queen, Black / White | volume (solid marker) | approximated | as above; the marker is not read by the diffusion |
| all six | base (metal-rough, ORM, normal) | implemented | -- |
| conversion losses (README) | | | subsurface weight map `*_shared_scattering.jpg` not in the pinned subset: constant 1; thin_walled and anisotropy fixed; transmission_depth 0; the Khronos ORM re-pack replaces the MaterialX roughness/metallic images |

### Pipelines, O7, validation, regression

* Archive: the first capture missed 11 pipelines (misses fail a capture); targeted update from all 25 capture configurations and 4 O7 runs (`tools/filter_pipeline_harvest.py --constant 22 --skip-present`,
  `tools/merge_pipeline_harvests.py`): +28 compute, +7 render, +2 tile descriptors (+14 library, +37 specialised function descriptors).  The 28 capture logs show 0 archive misses and 0 fallback draws;
  render-thread compiles: 0 after the first frame everywhere; the `--reflections rt` and `--shadows rt` configurations have 2 and 1 BEFORE the first frame (0.3 ms, pipelines outside the archive such as the
  capture readback), identical to the ticket 13/19 captures of `transmission-thin-volume`; the runner reports both numbers and gates the second.
* O7 (`--o7`, warmup 64, 64 frames, 1920x1080, `build/f15-chess/o7`): 0 GPU allocations on all four rigs (backlight, sidelight, shadow RT, shadow CSM), 0 misses, 0 fallbacks, 0 compiles after the first frame.
* API + shader validation (one run per rig, 40 frames, `build/f15-chess/validation*`): no shader-validation message and no error on sidelight, shadow RT and shadow CSM at 1920x1080 and on all four rigs at
  640x360, except that the back-light rig under shader validation times out on frame 1 ("GPU timeout waiting for frame event", 5 s) at 1080p and at 640x360 with 8 reflection samples; it completes
  cleanly at 320x180 with 1 sample (30 frames, two transmission values, 0 errors): validation cost on a 574k-triangle RT glass scene, not a validation message.  The API layer prints the "Redundant call to
  setDepthStencilState / setViewport / setComputePipelineState / setRenderPipelineState" family (nslog warnings) on the chess rigs; the existing `subsurface-volumetric-sun` fixture prints the same
  family (640, 128 and 512 lines in the same budget), so it is not chess specific (carried to ticket 24).
* `tools/visual_check.sh build/release build/reference-f15-appB` with `BENCH8_ARGS="--instances 100000"`: 8 of 8 benches, 0 differing pixels (max delta 0).
* Indicative cost (not a claim): the back-light rig at 1080p with RT glass transport costs about 1 s per frame (8 reflection samples, 574k triangles); the sidelight and shadow rigs about 8 ms.

### Findings and open items

* **Emissive meshes of the kings/queens hang the GPU.**  The first mask design (pure emissive white on King_Black and Queen_Black, about 75k triangles) timed out on frame 0 ("GPU timeout waiting for
  frame event value 1 (current 0)", reproduced in two runs, a harvest and a debug run); the emissive pawn tops (2208 triangles each) are fine.  The mask now uses a lit white.
  **Fixed afterwards (F15 GPU-safety commit):** cause = the clustered local-DI fallback (`--lighting legacy` with RT lighting passes): emissive triangles are unbounded lights (range 0), so every cluster
  cell overflows and `light_cluster_shade(_rt)` shaded ALL lights per pixel, one visibility ray each (`light_cluster_build` also looped over every light per cell).  Measured with the old shader
  (Pawn_Top_Black emissive, 17,664 lights, 128x72..384x216): "Local clustered RT shading" 58.5 / 56.1 / 49.8 / 70.9 ms at 128x72 / 192x108 / 256x144 / 384x216 (latency bound: linear in the light
  count, 113.7 ms for 35,328 lights, flat in resolution until the GPU fills), "3D light clusters" 5.4-5.6 ms (10.3 ms for 35k); 75k lights is pixels x 75k rays.  Fix: at most 128 lights per pixel
  (`src/renderer/light_budget.h`: exact below, stratified unbiased above), cluster build stops at the first overflow.  After: 0.97 ms at 192x108 for the 17.7k case; `emit=1` kings + queens
  (75k lights) 13.8 ms shading, 14-26 ms GPU frame at 1280x720 (`tools/f15_emissive_bound_check.py`).  Not a bound: the loop of `reflection_probe_capture_fs` over all sampled lights x 16
  (opt-in `--reflection-capture-probe`) showed no measurable cost with 75k lights (frame 0 GPU max 20 vs 13.5 ms without emitters) and is unchanged.
* The first receiver definition of the shadow gate was dominated by the tops' own pixels (hiding a top uncovers the floor) and by a glare percentile (68-393 pixels); the definition was corrected before any
  gate verdict (the bounds are unchanged: 0.05-0.95, 20 %).
* Not run: `tools/archive_check.sh`, `tools/hitch_check.sh` (no pipeline-descriptor change outside the targeted update), ticket 22's side-by-side screenshots, Sponza D2.
* Non-mask captures of `run3` come from the build before the mask change (it touches mask scenes only); the O7 runs and the mask captures are from the final build.
* The scattering weight map (`*_shared_scattering.jpg`) is not in the pinned subset: if the owner approves fetching it, `weightTexture` binds it with one more line per material.

### Denoised energy of the budgeted clustered DI (proof, 2026-11-14)

Question: budget image / exact image was 1.016 +- 0.008 undenoised but 0.973 with `--lighting-denoise custom`.  Method: temporary env patches (reverted, never committed: stride over the light list, budget override, denoiser knobs), scene `chess-sidelight` + Queen_Black emissive,
128x72, `--capture-linear-accumulate 32`, per-seed ratios (8 seeds), "raw" = denoiser pass-through (atrous 0, alpha 1) in the same resolve path (the off/custom resolve paths differ by a constant structure, so only same-mode ratios are compared).

| condition | ratio | sem |
|---|---|---|
| estimator: raw 128 strata / raw 8192 strata | 0.9996 | 0.0059 |
| denoised / raw, 128 strata | 0.977 | 0.004 |
| denoised / raw, 8192 strata | 0.988 | 0.002 |
| denoised / raw, list <= 128 (exact path, stride 1000) | 1.0000 | 0.0000 |
| denoised / raw, 1.2k lights (stride 30) | 0.9995 | ~0 |
| temporal EMA only / raw (atrous 0) | 1.006 | 0.009 |
| luminance weight off (phi 1e6) / raw | 1.004 | 0.007 |
| spatial filter only (alpha 1) / raw | 0.970 | 0.003 |
| history clamp on (opt-in, off by default) / raw | 0.956 | 0.003 |

Cause: the a-trous luminance edge-stopping weight (row-normalised, variance-guided) loses energy on heavy-tailed noise; the loss is concentrated on the emitter's own surface pixels (close-range 1/r^2 terms), grows with the tail weight and is absent for exact lists.  Not a clamp
(clampHistory, soft-change are off by default; the denoise-pack clamp belongs to MetalFX) and not the estimator.  Reference: `tests/test_denoise.cpp` "F15 variance-guided atrous filter loses energy..." (CPU mirror weights: 0.79 raw-level, ~0.985 after temporal accumulation, ~1 with the weight off).
Declared behaviour, no engine change (an energy-conserving filter would change every denoised signal and the F13 corpus; not selected).

2026-11-15 — closure audit (ticket 24):

- 1 back-light rig: MET (steps 0.1266, visible ratio 11.6, chroma within 1.4 % / 0.2 %; controls fail). Ticked.
- 2 side-light rig: PARTIAL (monotone gate met, steps 0.039-0.047; ticket-17 bound on piece masks NOT BUILDABLE, five reasons in the record; Mitsuba comparison on the pieces therefore NOT MET). Open.
- 3 pawn-top shadows: MET qualitative (RT T_luma 0.391; CSM needs --shadow-map-size 8192). Ticked.
- 4 support report + conversion note: MET (fixtures/chess/README.md). Ticked.
Open: scattering weight map not in the pinned subset (weight 1); profile_shape_ignores_textures; API-layer "Redundant call to setDepthStencilState/setViewport" notices on chess and subsurface fixtures carried to 24 and not recorded there; CSM cascade fit to the scene for small casters (candidate); chess binning A/B/A not run.

## F15 closure, loader-misc (2026-11-15): redundant API-layer notices removed; the back-light rig under shader validation

**Redundant `setDepthStencilState` / `setViewport` notices (96 + 96 per capture): FIXED (commit 65f2f10).**  Graph dump of a ReSTIR + RT capture: render pass 2 fuses "Visibility opaque B" and "Visibility alpha B" in one
encoder and both call `MeshRenderer::encodeRaster`, which set the same depth-stencil state and viewport again: one pair per frame (98 lines in a 49-frame capture, 192 in the 96-frame corpus captures, the "known baseline notices" of
tickets 15, 20 and 22).  Fix: `src/platform/metal/render_state_tracker.h` (the executor registers the encoder of every plain render group, the passes set the two states through `metal_state::setDepthStencil / setViewport`, which skip
a value the registered encoder holds; unregistered encoders always get the call).  State stays at Metal's defaults where it was (ImGui, scene passes untouched).  Evidence: notices 98 -> 0 on every capture of
tools/f15_unlit_lights_check.py (12 captures, now a failure criterion of that tool); `MTL_DEBUG_LAYER_WARNING_MODE=assert` under lldb exits 0; images unchanged: visual_check 8/8 0 px (BENCH8_ARGS 100000), the five F13 exactness
PFMs byte-identical, DiffuseTransmissionTest validation clean.  Still printed, a DIFFERENT family: `Redundant call to setComputePipelineState`, 7 per frame with `--reflection-samples 8` (84 in a 12-frame chess capture; 0 with 1 sample):
every reflection sample is its own graph pass fused in one compute encoder and each dispatch sets the same pipeline; ~50 direct `setComputePipelineState` sites make a tracker unsafe there (a stale entry would run the wrong
kernel).  Proposal: collapse the eight sample passes into one pass with eight dispatches (changes the per-pass timing units and the graph key: its own ticket), or give every compute encoder the same registered tracker once all sites go through it.

**Back-light rig under shader validation: EXPLAINED, runnable configuration delivered.**  The "GPU timeout waiting for frame event" is the engine's 5 s frame wait (`kWaitTimeoutMs`, metal_context.cpp) giving up on a slow frame, not a
validation message: the rig costs 620 ms per frame at 1920x1080 without validation (`build/f15-chess-valid2`, 4 frames) and shader validation makes a frame several times slower (640x360, the exact capture flags of
tools/f15_chess_check.py: 220 ms plain, 1126 ms validated with `--debug-lighting 1` (5.1x), 383 ms validated without it), so the 1080p frame needs more than 5 s (reproduced: `GPU timeout True` at 1920x1080 with validation,
exit 1, 0 validation messages).  A longer wait would put multi-second command buffers in front of the WindowServer watchdog (CLAUDE.md GPU safety), so the wait stays.  The rig is validated at 640x360 with the exact flags:
`tools/f15_chess_validate.py` (12 frames, `--debug-lighting 1`, archive off, no accumulation): transmission 0 and 1 both exit 0, 0 validation messages, 0 timeouts, 1.03 s (transmission 0) / 1.02 s (transmission 1) per frame (1.13 / 1.21 s in the diagnostic run); the 640x360 / 8-sample configuration that
the first record listed as timing out did NOT time out here (frame 1.1 s, far below the 5 s wait); the earlier timeout at that size is not reproduced (machine contention at the time is a guess, not verified).

### Follow-up (2026-11-15, orchestrator review): setComputePipelineState notices fixed
The N reflection sample passes (`--reflection-samples` > 1) are now ONE graph pass "F13 reflection samples" with N dispatches that set the pipeline once, a Dispatch->Dispatch device barrier between them (they reuse the scratch
textures; each sample keeps its table, seed and output slice).  Evidence: tools/f15_chess_validate.py (any `Redundant call to set...` counts) on both back-light rigs under API + shader validation: 0 notices, 0 validation messages (was 84);
subsurface-volumetric-sun with 8 samples under the API layer: 0 notices; bit-identical images: cornell RT-reflection capture and an SSR grid capture at `--reflection-samples 8` `cmp` identical to the orchestrator tree's binary, the 5 F13 exactness PFMs `cmp`
identical, visual_check 8/8 0 px.  No pipeline descriptor changed (no archive update).  Per-pass timing units: the eight "F13 reflection sample k" units became one.

### 2026-11-16 — round 2: king/queen subsurface bound on the RT walk (orchestrator), RED, declared OPEN by the owner (option A)

**Gate.** `tools/f15_chess_check.py --group sssbound` (5d4e982; weight-0 guard fixed in a58df60): views white (King_White, Queen_White) and black (King_Black,
Queen_Black), `chess-sidelight/view=<w|b>/zoom=3/weight=1`, `--subsurface-mode rt-walk --subsurface-samples 64 --subsurface-steps 512`, 96 accumulated
frames; the noisy term is pooled 5x5 with the walk-lights-off twin (`f15_check.pool_noisy_term`).  Metric (ticket 17, decision 11): per piece the
relative luminance error over the piece ROI with a 5 % floor; bound mean <= 0.10 and p95 <= 0.25 for every piece the weight-0 control does not pass.
Controls: weight 0 (base model) and walk lights off must fail (both fail).
**Reference.** `tools/f15_chess_subsurface_reference.py` (Mitsuba 3.9.1 volpath, locally homogeneous k-means classes, non-finite repair 00b68a1) at
**6 classes per piece and 512 spp per grey render** (reduced from the tool default 8 / 2048 by the sss writer for time); snapshot sha256 white
641068ed…, black 5304ceeb…, identical to the engine captures.  Its own noise in the metric's form (|Y_a - Y_b| / 2 over Y, same floor), mean / p95:
white King 0.128 / 0.482, white Queen 0.137 / 0.525; black King 0.101 / 0.337, black Queen 0.127 / 0.473: the reference alone exceeds the bound.
**Engine (pre-check on a09f738, phosphor 4507dc22…; `build/final-b/chess-sss-ms-on`).** Raw (unpooled) worst: white 0.606 / 1.180, black 1.467 / 3.582.
Pooled per piece: white King 0.197 / 0.636, white Queen 0.543 / 0.921, black King 0.756 / 2.409, black Queen 1.431 / 3.072.  Piece-mean luminance
engine / reference: white King 0.0830 / 0.0889 (-6.7 %), white Queen 0.1334 / 0.1255 (+6.3 %), black King 0.0621 / 0.0585 (+6.3 %), black Queen
0.0250 / 0.0192 (+30 %).  MS off (`--specular-ms off`, Mitsuba does not model the compensation; `build/final-b/chess-sss-ms-off`): piece means -0.05 /
-0.08 / -0.19 / -0.16 %, gate numbers 0.541 / 0.919 and 1.423 / 3.047: the compensation is not the cause.
**Reading.** The engine errors are 2-11 times the reference noise, so noise explains only part of them.  Two causes stay unresolved: (1) the reference
approximates the textured medium by k-means homogeneous classes while the engine's walk takes the medium of each entry pixel from the textures;
(2) a possible low-albedo gap on the black Queen (single-scattering albedo about 0.3-0.4, mean free path about 0.1 mm).  Re-rendering the reference
with its noise below a third of the gate would need about 17 times the spp (about 50 h at 8 threads) and would very likely stay RED.
**Owner decision (2026-11-16, option A):** declared OPEN; criterion 2 stays PARTIAL; thresholds unchanged; a noise-compensated judge (option C) is not
adopted.  The RT walk bounds stay proven on the homogeneous OpenPBR fixtures (ticket 17/18): side-lit 0.0635 / 0.1708 against 0.10 / 0.25, back-lit
0.0919 / 0.2288 against the 2x bound 0.20 / 0.50, product output 0.0621 / 0.1681, moving-camera clip 0.0703 / 0.1902 (control 0.1400 / 0.4482 fails).
**Follow-up D (named, not started: owner approval needed):** a chess variant with one homogeneous medium per piece, identical in engine and reference,
to separate the reference's class approximation from an engine error.  The final battery re-takes the captures on the final tip (numbers added below).

