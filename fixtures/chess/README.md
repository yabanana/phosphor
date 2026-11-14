# Chess fixture (F15, ticket 20)

The six Open Chess Set materials of the roadmap's Track Studio case, rebuilt in Phosphor from public assets.  Per ADR-0001 no
Track Studio type or file format is involved; per ADR-0005 the subsurface values enter Phosphor's own OpenPBR-based description.

| file | what |
|---|---|
| `chess-subsurface.overrides.json` | `phosphor-material-overrides` v1: subsurface (volumetric) + solid marker for `King_Black`, `King_White`, `Queen_Black`, `Queen_White` |
| `src/testbench/chess_fixture.*` | bench 2 `--material-scenario chess-backlight`, `chess-sidelight`, `chess-shadow` (+ `/key=value` options) |
| `tools/f15_chess_check.py` | serial runner: captures, masks, gates, controls that must fail (`tools/test_f15_chess_check.py`) |
| `tests/test_chess_fixture.cpp` | doctest: scenario names, override values, rig geometry, the pinned asset (skips without it) |

## Assets (never committed)

* **ABeautifulGame**, KhronosGroup/glTF-Sample-Assets at `8f492390ddc7a2ae221a1369ef560589cf8c70fb`, `Models/ABeautifulGame/glTF`,
  CC-BY-4.0 ((c) 2020 ASWF MaterialX Project for the original model, (c) 2022 Ed Mackey for the glTF conversion; original artwork by
  Moeen Sayed and Mujtaba Sayed for SideFX).  Fetched by `tools/fetch_khronos_corpus.py` into `build/khronos-corpus/`
  (git-ignored); hashes in `tools/khronos_corpus.json`.  The runner takes `--gltf` or `$PHOSPHOR_CHESS_GLTF`; the doctests skip when it is
  absent.
* The asset is **loaded unchanged**: nothing is written back, no texture is edited.  The only things the rigs add are procedural
  (the emissive backdrop of the back-light rig, one sun) or declared visibility (see Rigs).

## Pawn tops: loaded unchanged

The official MaterialX -> glTF translation (`standard_surface_to_gltf_pbr`) maps a Standard Surface material with `transmission_depth` 0
exactly to what ABeautifulGame carries, and Phosphor's glTF loader reads it directly:

| MaterialX (Pawn_Top_B / _W) | glTF in ABeautifulGame | Phosphor description |
|---|---|---|
| `transmission` 1 | `KHR_materials_transmission.transmissionFactor` 1 | `transmission.factor` 1 |
| `transmission_color` (0.2995, 0.5, 0.450276) / (1, 1, 0.828), `transmission_depth` 0 | tint in `baseColorFactor` (0.2995, 0.5, 0.45) / (1, 1, 0.828) | `base.baseColor`: the tint multiplies the transmitted light at the surface |
| `thin_walled` false | `KHR_materials_volume.thicknessFactor` 0.22, `attenuationDistance` absent (= +inf), `attenuationColor` 0.8 | `volume`: thickness 0.22 (mesh units), attenuation distance +inf: no absorption (the colour is inert) |
| `specular_IOR` 1.5 | no `KHR_materials_ior` (default 1.5) | `specular.ior` 1.5 (default) |
| roughness / normal images | `Pawn_ORM.jpg` G, `Pawn_normal.jpg` | same slots |

Losses and choices: the tint is a single colour at the surface (Standard Surface semantics at depth 0), it is not absorption along the
path; `thicknessFactor` 0.22 is about ten times the 2.2 cm mesh and has no effect with ray-traced transport (the real inside segment
is traced, `volume` is reported approximated: `thickness_texture_ignored_with_rt`), it only feeds the `--rt off` thickness hint; the
authored `attenuationColor` 0.8 has no effect without an attenuation distance.  The ORM roughness/metallic image of the Khronos re-pack
replaces the MaterialX `pawn_shared_roughness.jpg` (the same artwork, re-packed with a baked occlusion channel in R).

## Kings and queens: the override file

ABeautifulGame deliberately drops subsurface (its README: the glTF extensions are still draft, and the official translation drops every
`subsurface*` input).  The override file adds it by material name, with Phosphor's OpenPBR parameters (ADR-0005):

| MaterialX Standard Surface input | override field | note |
|---|---|---|
| `subsurface` (weight, a float image `*_shared_scattering.jpg`) | `subsurface.weight` = **1**, no `weightTexture` | **LOSS**: the scattering map is a MaterialX asset image that is not in the pinned Khronos subset (not downloaded; its licence in the MaterialX repository is unverified).  Constant 1 is also the fallback of the Track Studio importer.  When the image is provided, `weightTexture` (channel A of the slot, linear) takes it: one line in the file. The bench `weight=F` option replaces the constant for the sweeps |
| `subsurface_color` = the base colour image | `subsurface.colorTexture` = the same JPEG (`King_black_base_color.jpg`, ...), `color` (1, 1, 1) | sRGB texture x constant; the texture manager uploads the JPEG a second time for the override path (same texels, a test compares them) |
| `subsurface_radius` = the base colour image (colour3) | `subsurface.radiusScaleTexture` = the same JPEG, `radiusScale` (1, 1, 1) | the OpenPBR mean free path per channel is `radius x radiusScale x texel`; the texel of the sRGB image is read as linear colour like MaterialX's `lin_rec709` conversion |
| `subsurface_scale` 0.003 (Queen_W 0.001) | `subsurface.radius` 0.003 m (Queen_White 0.001 m) | `mean free path = radius x scale`: `radius` is the OpenPBR radius in metres, the glTF scene is in metres (a king is 0.15 m tall), so the scale is taken as metres (the units of MaterialX are a convention; same reading as Track Studio) |
| `subsurface_anisotropy` 0 | `subsurface.anisotropy` 0 | Henyey-Greenstein g |
| `thin_walled` false | `subsurface.mode` `volumetric` | thin would be diffuse transmission |
| (none) | `volume.thickness` 0.04, neutral attenuation | a pure **marker**: the description's validation requires a volume with thickness > 0 for volumetric subsurface; the volumetric pass (screen-space diffusion) does not read it |
| base colour, metalness, roughness, normal | the asset's own `King_*_base_color.jpg`, `*_ORM.jpg`, normal maps | the Khronos re-pack of the MaterialX images (`king_shared_metallic.jpg` etc. are packed into ORM) |

Source of the values: `standard_surface_chess_set.mtlx` at MaterialX `9176de82` (`.scratch/f15/research/open-chess-set.md` section 2).
Subsurface weight 0 renders the base model; weight 1 replaces the diffuse term by the diffused irradiance.

What the engine does with them (support report, `MATERIALS` line, schema `phosphor.materials.v2`): volumetric subsurface is
**approximated** (screen-space diffusion; named approximations include thick-volume back-lit translucency out of reach, diffusion
regime only) and, per ticket 17/18, volumetric subsurface stays PARTIAL: its side-lit bound against the Mitsuba random-walk ground truth
was not met on valid OpenPBR ground truth.

## Rigs (all deterministic: fixed camera, no scripted motion, `--fixed-timestep`)

* `chess-backlight`: the black pawn row seen from the white side (camera (0, 0.16, -0.40) towards (0, 0.07, 0.156)), a dark plate with
  an emissive checkerboard (radiance 3, 6 mm cells) at z = 0.182 between the black pawn row and the back rank (the back light), constant
  environment 0.2, no sun.  Kings, queens, castles, knights and bishops are **hidden** by the rig (they would block the line of sight
  over the pawns); the pawns and the board stay as loaded.  Options: `transmission=F`, `tint=asset|white`, `panel=on|off`.
* `chess-sidelight`: `King_Black` and `Queen_Black` (or the white pair, `view=white`) from behind their end of the board (camera
  (0, 0.10, 0.53) towards (0, 0.095, 0.22), 0.31 m), one sun of 3 W/m2 from the right and slightly towards the camera
  (towards-sun (0.9, 0.25, 0.35) normalised; mirrored for white), environment 0, shadows off (the pieces would shade each other), all
  pieces present.  Options: `weight=F`, `scale=F` (multiplies `radius`), `view=black|white`, `sss=on|off`.
* `chess-shadow`: pawns and board under one sun 60 degrees from the vertical behind the black pawn row (towards-sun (0, 0.5, 0.866)), the
  camera above the empty rank in front of the row looking down.  The other pieces are hidden (their shadows would fall on the row).
  Options: `tops=on|opaque|hidden`, `tint=asset|white`.  Instances are hidden with both the visible and the shadow role off (the engine
  keeps the two roles equal).
* `mask=1` on any rig: every material untextured and black with specular off, the rig's target materials white diffuse, the only light a
  sun of pi W/m2 along the view direction, environment 0: a target pixel reads cos(N, V), everything else is black, so the capture is the
  mask of the same geometry and camera.  (Pure emissive targets were tried first: the kings' and queens' emissive triangle count hangs the
  GPU, `GPU timeout waiting for frame event` on frame 0.)

## Support list (what every chess feature is)

Read from the report of the captures (`materials` section) by `tools/f15_chess_check.py`; the verification record in
`.scratch/f15/issues/20-chess-fixture.md` carries the table from the run.  Declared approximations come from `src/renderer/material_support.cpp`.
