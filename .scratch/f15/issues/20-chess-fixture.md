# 20: Chess fixture: the roadmap's Track Studio case

Roadmap: F15.1, F15.3

**What to build:** The six Open Chess Set materials render from public assets: ABeautifulGame (pinned) unchanged for the pawn tops, a material-description override file for the four kings/queens with the MaterialX values (weight map, colour = base colour, radius = base colour × scale 0.003 / Queen_W 0.001); back-light and side-light rigs and parameter sweeps; coloured shadows of the pawn tops; conversion note committed with the fixture.

**Blocked by:** 13, 14, 17

**Status:** resolved PARTIAL (2026-11-14): criteria 1, 3 and 4 met on M5; criterion 2's ticket-17 bound on the piece masks cannot be built (the volumetric ground truth cannot represent the textured chess meshes), and the volumetric mode itself is partial (tickets 17/18)

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
  frame event value 1 (current 0)", reproduced in two runs, a harvest and a debug run); the emissive pawn tops (2208 triangles each) are fine.  The mask now uses a lit white; the engine limit for large
  emissive meshes is not investigated here (candidate for ticket 24).
* The first receiver definition of the shadow gate was dominated by the tops' own pixels (hiding a top uncovers the floor) and by a glare percentile (68-393 pixels); the definition was corrected before any
  gate verdict (the bounds are unchanged: 0.05-0.95, 20 %).
* Not run: `tools/archive_check.sh`, `tools/hitch_check.sh` (no pipeline-descriptor change outside the targeted update), ticket 22's side-by-side screenshots, Sponza D2.
* Non-mask captures of `run3` come from the build before the mask change (it touches mask scenes only); the O7 runs and the mask captures are from the final build.
* The scattering weight map (`*_shared_scattering.jpg`) is not in the pinned subset: if the owner approves fetching it, `weightTexture` binds it with one more line per material.
