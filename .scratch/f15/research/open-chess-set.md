# Open Chess Set: MaterialX source vs. glTF "A Beautiful Game" (F15 research)

Research date 2026-11-13. Primary sources only. All SHAs below were resolved with `gh api`; files were downloaded and inspected locally (JSON/XML only, no binary buffers).

Pinned references used throughout:

| Tag | Repo | SHA |
|---|---|---|
| MX-A | AcademySoftwareFoundation/MaterialX (commit used by the Track Studio fixture; its `provenance.json` records it) | `9176de82cd993bace533b734e17f9ca738061714` (also `main` HEAD on 2026-11-13) |
| MX-F | same repo, last commit touching the chess .mtlx file | `537aa920d7e312cbb6b44ae6132abf39c3aa30e6` ("Upgrade resources to 1.39", 2024-06-22); file created in `1e9824c973a9c513276ce7f5ccb2600d4f0863cb` ("Add Chess Set example (#982)", 2022-06-07) |
| KH-S | KhronosGroup/glTF-Sample-Assets | `8f492390ddc7a2ae221a1369ef560589cf8c70fb` (main HEAD 2026-11-13) |
| KH-G | KhronosGroup/glTF (extension specs) | `f93f8afd4dff9252a4191e3e0538d4efe0e181b4` (main HEAD 2026-11-13) |
| KH-PR | KhronosGroup/glTF PR #2453 head (volume_scatter, OPEN) | `134d07661f461ee26d02e0e11b7ebe3e26f28330` |

## 1. Where the official files are

- MaterialX file: https://raw.githubusercontent.com/AcademySoftwareFoundation/MaterialX/9176de82cd993bace533b734e17f9ca738061714/resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx (MaterialX 1.39 document, `colorspace="lin_rec709"`, 555 lines, 15 `standard_surface` + 15 `surfacematerial`, one `look` `L_ChessSet`).
- Textures: `resources/Materials/Examples/StandardSurface/chess_set/*.jpg` in the same commit (43 files per the local `provenance.json`; local sha256 of the .mtlx is `feb15090...272f`, taken from `/Users/danielsan/Documents/REA_RTB/work/design-l1-20261106/.checks/material-library/fixtures/provenance.json`).
- Geometry: `resources/Geometry/chess_set.glb` (+ `resources/Geometry/README.md`) at the same commit.
- Attribution comment inside the .mtlx (line 3): "Chess Set geometry and material contributed by Side Effects, artwork by Moeen Sayed and Mujtaba Sayed."
- MaterialX README section "Open Chess Set" (README.md at MX-A, line 62): "open reference asset ... MaterialX file in the Standard Surface shading model and a geometry file in the glTF format. It was authored by Moeen Sayed and Mujtaba Sayed, and was contributed to the MaterialX project by Side Effects."
- License:
  - MaterialX repository code: Apache-2.0 (`LICENSE` at MX-A; GitHub reports SPDX `Apache-2.0`).
  - Asset (meshes/textures) license in the MaterialX repo: NOT stated in any file I could find (no LICENSE/NOTICE next to the asset; the README gives only attribution). UNVERIFIED for the MaterialX copy.
  - The Khronos re-release of the same asset is explicitly CC-BY-4.0 (see section 4): "(c) 2020, ASWF ... MaterialX Project for Original model" and "(c) 2022, Ed Mackey ... Conversion to glTF". Source: https://github.com/KhronosGroup/glTF-Sample-Assets/blob/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/ABeautifulGame/README.md and `LICENSE.md` in the same folder (all model files CC-BY-4.0).
- Original lineage (Khronos README): made for a SideFX Karma tutorial by Moeen Sayed and Mujtaba Sayed; re-released by ASWF at SIGGRAPH 2022 Open Source Days; geometry converted to GLB by Chris Rydalch and Jonathan Stone for MaterialX PR #982; glTF conversion by Ed Mackey.

## 2. The six materials in MaterialX (Standard Surface)

Source: MX-A .mtlx. Only inputs explicitly authored are listed; everything else is the Standard Surface default from `libraries/bxdf/standard_surface.mtlx` at MX-A (base 1, specular 1, specular_IOR 1.5, specular_color 1, metalness 0, transmission_depth 0, transmission_scatter 0, transmission_scatter_anisotropy 0, thin_walled false, subsurface_anisotropy 0, coat 0, sheen 0, emission 0, opacity 1, tangent default geomprop).

### Pawn tops (transmission)

| Input | Pawn_Top_B (`M_Pawn_Top_B`) | Pawn_Top_W (`M_Pawn_Top_W`) |
|---|---|---|
| base | default 1 | default 1 |
| base_color | constant `1, 1, 1` (no texture) | constant `1, 1, 1` |
| transmission | 1 | 1 |
| transmission_color | `0.2995, 0.5, 0.450276` | `1, 1, 0.828` |
| transmission_depth | default 0 | default 0 |
| transmission_scatter / _anisotropy | default 0 / 0 | default 0 / 0 |
| thin_walled | default false (solid) | default false |
| specular, specular_IOR | defaults 1, 1.5 | defaults 1, 1.5 |
| specular_roughness | image `chess_set/pawn_shared_roughness.jpg` (float) | same shared image |
| metalness | default 0 (no map) | default 0 |
| normal | `normalmap` of `chess_set/pawn_shared_normal.jpg` | same shared image |
| subsurface | 0 | 0 |
| subsurface_color / radius / scale | `1,1,1` / `1,1,1` / 0.003 (inert because subsurface=0) | same |
| coat, sheen, emission, opacity | defaults (off) | defaults |

### Kings and queens (subsurface)

| Input | King_B | King_W | Queen_B | Queen_W |
|---|---|---|---|---|
| base_color | `king_black_base_color.jpg` (sRGB) | `king_white_base_color.jpg` | `queen_black_base_color.jpg` | `queen_white_base_color.jpg` |
| metalness | `king_shared_metallic.jpg` | same | `queen_shared_metallic.jpg` | same |
| specular_roughness | `king_black_roughness.jpg` | `king_white_roughness.jpg` | `queen_black_roughness.jpg` | `queen_white_roughness.jpg` |
| normal | normalmap `king_black_normal.jpg` | `king_white_normal.jpg` | `queen_black_normal.jpg` | `queen_white_normal.jpg` |
| subsurface (weight) | float image `king_shared_scattering.jpg` | same image | float image `queen_shared_scattering.jpg` | same image |
| subsurface_color | = the base_color image (same node output) | same | same | same |
| subsurface_radius | = the base_color image (color3, per-channel) | same | same | same |
| subsurface_scale | 0.003 | 0.003 | 0.003 | **0.001** |
| subsurface_anisotropy | default 0 | default 0 | default 0 | default 0 |
| transmission | default 0 | default 0 | default 0 | default 0 |
| thin_walled | default false | default false | default false | default false |
| coat / sheen / specular | defaults | defaults | defaults | defaults |

Findings relevant to the importer: the subsurface weight is a map (so "subsurface=1" in Track Studio is the map-driven weight, with constant 1 only as the engine's fallback); subsurface_color and subsurface_radius are both bound to the base-color texture; the Queen_W scale is 0.001, not 0.003. Note also a defect in the file: the `M_King_W` surfacematerial (MX-A file, around line 261-263) lists two `surfaceshader` inputs (`King_W` and a leftover `Pawn_Body_W`); strict parsers may pick either. UNVERIFIED how Track Studio resolves this (its `plan_materialx` reads standard_surface nodes by name, see local code below).

### Standard Surface semantics (Autodesk spec, https://autodesk.github.io/standard-surface/, spec version 1.0.1.1; the published page is the `gh-pages` site of github.com/Autodesk/standard-surface, I could not resolve a commit SHA for the pages branch, so no SHA is claimed)

- `transmission` (0-1, default 0): weight of the specular BTDF closure.
- `transmission_color` (default 1,1,1): tint for transmitted light. At `transmission_depth` = 0 it is a plain multiplicative tint applied on the closure at the surface (no distance dependence). At `transmission_depth` > 0 the colour is instead produced by absorption in the interior medium (Beer's law) and the closure tint becomes 1.
- `transmission_depth` (default 0, [0, inf)): distance white light travels in the medium before it becomes exactly `transmission_color`; defines the medium extinction. Units not stated by the spec (scene units).
- `transmission_scatter`, `transmission_scatter_anisotropy`: scattering coefficient and Henyey-Greenstein g of the interior medium. Not used by the chess set.
- `thin_walled`: false = solid, refraction on entry/exit; true = infinitely thin double-sided shell, no refraction, and subsurface becomes diffuse transmission.
- `subsurface` (weight), `subsurface_color`, `subsurface_radius` (per-channel, spec equates it with mean free path), `subsurface_scale` (scalar multiplier of the radius), `subsurface_anisotropy` (HG g). Substrate = `(1 - subsurface) * diffuse + subsurface * subsurface closure`.
- Mean free path in the MaterialX implementation (MX-A `libraries/bxdf/standard_surface.mtlx`, nodes `subsurface_radius_scaled`, `subsurface_bsdf`): `radius_scaled = subsurface_radius * subsurface_scale` is fed to `subsurface_bsdf.radius`, which MaterialX defines as the per-channel mean free path in scene units. So for King_B: MFP_rgb = base_color_texture_rgb * 0.003 (scene units, which Track Studio treats as metres; the units are a convention, not specified by the spec or by the .mtlx, so "metres" is UNVERIFIED but consistent with glb scene scale of about 0.15 m tall pieces, see section 4).
- Important implementation fact (MX-A standard_surface.mtlx): in the MaterialX nodegraph, `transmission_color` is passed only as `tint` of `dielectric_bsdf` (`scatter_mode T`); `transmission_depth`, `transmission_scatter*` are declared but not consumed in the graph (volume is left to the renderer). With depth 0 (the chess set) this is exactly the spec semantics.

## 3. MaterialX translation standard_surface -> glTF PBR

Source: https://raw.githubusercontent.com/AcademySoftwareFoundation/MaterialX/9176de82cd993bace533b734e17f9ca738061714/libraries/bxdf/translation/standard_surface_to_gltf_pbr.mtlx (nodedef `ND_standard_surface_to_gltf_pbr`, nodegraph `NG_standard_surface_to_gltf_pbr`).

| Standard Surface input | glTF output (node output) | Rule in the graph |
|---|---|---|
| base, base_color | baseColor (`base_color_out`) | `base*base_color` (coat colour folded in); then mixed toward a transmission term, see next row |
| transmission_color + transmission_depth + transmission | baseColor (`base_color_out`) | `transmissionBaseColor = (depth > 0) ? (1,1,1) : transmission_color`; `base_color_out = mix(baseColor, transmissionBaseColor, transmission^2)`. So with depth = 0 the tint lands in glTF baseColor |
| transmission | transmissionFactor (`transmission_out`) | pass-through |
| transmission_color (depth > 0 only) | KHR_materials_volume.attenuationColor (`attenuation_color_out`) | `(transmission > 0 and depth > 0) ? transmission_color : (1,1,1)` |
| transmission_depth | KHR_materials_volume.attenuationDistance (`attenuation_distance_out`) | pass-through (0 means "no attenuation distance" and must be mapped to +Infinity by the consumer, since glTF range is (0, inf); the graph itself outputs 0; UNVERIFIED how exporters treat it) |
| thin_walled | KHR_materials_volume.thicknessFactor (`thickness_out`) | `1 - thin_walled` (a 0/1 flag, not a physical thickness) |
| metalness | metallicFactor | pass-through |
| specular_roughness (+anisotropy) | roughnessFactor | `sqrt(alpha_y)` of roughness_anisotropy; equals specular_roughness when anisotropy = 0 |
| specular, specular_color, specular_IOR | KHR_materials_specular, KHR_materials_ior | pass-through (`specular_out`, `specular_color_out`, `ior_out`) |
| specular_anisotropy, specular_rotation | KHR_materials_anisotropy | strength/rotation |
| sheen, sheen_color, sheen_roughness | KHR_materials_sheen | colour = sheen_color*sheen, roughness sqrt |
| coat, coat_color, coat_roughness | KHR_materials_clearcoat | coat colour folded into baseColor / clearcoat weight |
| thin_film_* | KHR_materials_iridescence | |
| transmission_dispersion | KHR_materials_dispersion | `20 / dispersion` (Abbe) |
| emission, emission_color | emissiveFactor / KHR_materials_emissive_strength | |
| normal, tangent, opacity | normalTexture frame / alpha | opacity luminance -> alpha, alphaMode 0/2 |

Dropped / not even declared in the translation nodedef: `subsurface`, `subsurface_color`, `subsurface_radius`, `subsurface_scale`, `subsurface_anisotropy`, `transmission_scatter`, `transmission_scatter_anisotropy`, `transmission_extra_roughness`, `diffuse_roughness`, `coat_IOR` etc. (the nodedef inputs list ends with thin_film_IOR). So the official MaterialX->glTF mapping drops all subsurface.

## 4. Khronos glTF sample "ABeautifulGame"

- Folder: https://github.com/KhronosGroup/glTF-Sample-Assets/tree/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/ABeautifulGame (glTF at `glTF/ABeautifulGame.gltf`, 46,214 bytes + `.bin` 10,829,440 bytes; also `glTF-Binary/ABeautifulGame.glb`).
- README: Extensions Used = `KHR_materials_transmission`, `KHR_materials_volume`; Summary "Chess set using transmission and volume"; license CC-BY-4.0 (see section 1). Key README statement: "A small amount of subsurface scattering on the King and Queen have been skipped for this iteration as the relevant glTF extensions are still in draft form." AO maps were recomputed and merged into ORM textures.
- .gltf header: `asset.generator` "Khronos glTF Blender I/O v3.3.17", `asset.copyright` "Original assets by Moeen Sayed and Mujtaba Sayed for SideFX.", `extensionsUsed` = [KHR_materials_transmission, KHR_materials_volume], no `extensionsRequired`. Sampler: mag LINEAR(9729), min LINEAR_MIPMAP_LINEAR(9987).
- Material names (index): 0 King_Black, 1 King_White, 2 Queen_Black, 3 Queen_White, 4 Chessboard, 5 Pawn_Top_White, 6 Pawn_Body_White, 7 Pawn_Top_Black, 8 Pawn_Body_Black, 9 Castle_Black, 10 Castle_White, 11 Knight_Black, 12 Knight_White, 13 Bishop_Black, 14 Bishop_White. No doubleSided, no alphaMode, no emissive on any material.

Exact material JSON for the six (from `jq` on the .gltf at KH-S):

| glTF material | pbrMetallicRoughness | other fields | extensions |
|---|---|---|---|
| `Pawn_Top_White` (5) | `baseColorFactor` [1, 1, 0.828, 1]; `metallicRoughnessTexture` -> texture 16 (`Pawn_ORM.jpg`); no baseColorTexture | `normalTexture` -> tex 15 (`Pawn_normal.jpg`) | `KHR_materials_transmission` {transmissionFactor 1}; `KHR_materials_volume` {attenuationColor [0.8,0.8,0.8], thicknessFactor 0.22}; attenuationDistance absent = +Infinity |
| `Pawn_Top_Black` (7) | `baseColorFactor` [0.2995, 0.5, 0.45, 1]; `metallicRoughnessTexture` -> tex 21 (same `Pawn_ORM.jpg`) | `normalTexture` -> tex 20 (`Pawn_normal.jpg`) | identical transmission/volume block (factor 1, attenuationColor 0.8, thickness 0.22) |
| `King_Black` (0) | `baseColorTexture` -> tex 2 (`King_black_base_color.jpg`); `metallicRoughnessTexture` -> tex 1 (`King_black_ORM.jpg`) | `normalTexture` tex 0; `occlusionTexture` tex 1 (ORM) | none |
| `King_White` (1) | baseColorTexture tex 5 (`King_white_base_color.jpg`); MR tex 4 (`King_white_ORM.jpg`) | normal tex 3; occlusion tex 4 | none |
| `Queen_Black` (2) | baseColorTexture tex 8; MR tex 7 (`Queen_black_ORM.jpg`) | normal tex 6; occlusion tex 7 | none |
| `Queen_White` (3) | baseColorTexture tex 11; MR tex 10 (`Queen_white_ORM.jpg`) | normal tex 9; occlusion tex 10 | none |

Observations (computed, not quoted):
- The pawn-top tint is carried by `baseColorFactor`, exactly as the MaterialX translation graph prescribes for transmission_depth = 0 (0.2995,0.5,0.45 vs mtlx 0.2995,0.5,0.450276; 1,1,0.828 equal). The `attenuationColor` 0.8 has no effect without `attenuationDistance` (Infinity); thicknessFactor 0.22 is only a refraction thickness. No KHR_materials_ior (default 1.5, equals Standard Surface 1.5), no sheen, no diffuse_transmission, no volume_scatter, no specular extension anywhere in the file.
- Scale check from accessors: the shared pawn-top mesh (accessor 12) spans about 0.0217 x 0.0221 x 0.0217 (units of the mesh), so `thicknessFactor` 0.22 is about 10x the geometric extent. Looks like an authoring/export quirk; it only matters for implementations that use thicknessFactor for refraction offset (Phosphor with a ray-traced thickness would not use it). UNVERIFIED whether intended.
- King/Queen glTF materials are plain metal-rough PBR with no trace of subsurface.
- I did not decode the ORM/`pawn_shared_roughness.jpg` pixels: whether glTF ORM metallic (B) is 0 for pawn tops (as in the .mtlx, metalness default 0) is UNVERIFIED.

glTF extension status (KH-G `extensions/2.0/Khronos/...README.md`): `KHR_materials_transmission` and `KHR_materials_volume` are "Complete, Ratified". `KHR_materials_diffuse_transmission` is "Release Candidate". `KHR_materials_volume_scatter` is not in the Khronos directory at KH-G; it exists only in OPEN PR https://github.com/KhronosGroup/glTF/pull/2453 (head KH-PR; properties `multiscatterColorFactor/Texture` (sRGB texture, RGB) and `scatterAnisotropy` in (-1,1), requires KHR_materials_volume). Earlier `KHR_materials_subsurface` PR #1928 and branch PR #1766 were closed. `thicknessFactor`: "given in the coordinate space of the mesh", 0 = thin-walled; `attenuationDistance`: world space, range (0, +inf), default +Infinity.

## 5. How Track Studio imports it (local, read-only)

Files: `/Users/danielsan/Documents/REA_RTB/work/design-l1-20261106/crates/editor-core/src/terrain_v1/library.rs` (`plan_materialx` at line 830, test `plan_chess_fixture` at line 1802; other `work/*` copies exist, I read the design-l1-20261106 one; "newest terrain3d-* folder" has no distinct logic that I diffed, UNVERIFIED equal) and `.checks/material-library/fixtures/provenance.json` in the same worktree.

- `Shading` fields: `transmission` 0-1, `transmission_color` rgb 0-1, `subsurface` 0-1, `subsurface_scale_m` limited to 0..0.01 m; maps include `subsurface_u8_base64` (single-channel).
- `plan_materialx`: if `transmission > 0` reads `transmission_color` (must be a literal value, a connection is an error); `transmission_depth` is not read. If `subsurface` is a nodegraph connection (float) it becomes the subsurface map and the weight is 1; scale read from `subsurface_scale`, default 0.01 if absent; `subsurface_color` and `subsurface_radius` must be bound to the same source as `base_color` or the import errors. Both effects are also pushed to `unsupported` ("transmission", "subsurface"), i.e. the importer records them as not natively rendered by the source model.
- The test asserts, for the six materials: transmission = 1 with a `transmission_color` array for the pawn tops; subsurface = 1, a `subsurface` map and `subsurface_scale_m` > 0 for kings/queens.
- GPU packing (line ~1524): effects texel = [transmission, subsurface*map weight, subsurface_scale/0.01 m, 0] as bytes.

## 6. Parameter mapping Track Studio -> glTF

| Track Studio parameter | MaterialX source | glTF equivalent | Notes |
|---|---|---|---|
| transmission intensity (`transmission`) | `transmission` (1 for pawn tops) | `KHR_materials_transmission.transmissionFactor` (ABeautifulGame: 1) | Direct 1:1. (Optionally a `transmissionTexture`; not used in the sample.) |
| transmission tint (`transmission_color`) | `transmission_color` with `transmission_depth` = 0 | `pbrMetallicRoughness.baseColorFactor` (ABeautifulGame: tint stored there; base_color is white in mtlx) | Per the MaterialX translation graph. In glTF, transmission is tinted by base colour. If depth > 0 instead: `KHR_materials_volume.attenuationColor` + `attenuationDistance` (not the case here; Track Studio ignores depth). |
| thick/solid flag | `thin_walled` = false | `KHR_materials_volume.thicknessFactor` > 0 (sample: 0.22) | Mapping graph writes 1/0; the sample has an actual (possibly oversized) mesh-space value; no attenuation distance, so no absorption. |
| subsurface weight map (`subsurface` map, `subsurface_u8`) | `subsurface` bound to `king/queen_shared_scattering.jpg` | **no glTF equivalent** in ratified/RC extensions. Candidates: `KHR_materials_volume_scatter` (draft, PR #2453) `multiscatterColorTexture`; thin-walled approximation `KHR_materials_diffuse_transmission` (RC) | ABeautifulGame ships no SSS (README says skipped); weight map has no slot (diffuse_transmission has `diffuseTransmissionTexture`, but that is a different physical model). |
| subsurface scale (`subsurface_scale_m` 0.003 / 0.001 m) | `subsurface_scale` times `subsurface_radius` (= base colour) = MFP | **no glTF equivalent** (volume_scatter draft would need `attenuationDistance` + `multiscatterColor`; a MFP-to-extension conversion is not defined by Khronos or MaterialX) | MaterialX translation drops it. |
| subsurface colour/radius | both = base_color texture | none; (draft: `multiscatterColorTexture`) | Track Studio requires them to equal base_color, which is consistent with the file. |
| base colour, roughness, metalness, normal | image nodes | `baseColorTexture`, ORM `metallicRoughnessTexture` (G/B), `normalTexture` | Same artwork re-packed; AO added in R of ORM. |

## 7. Conclusion

- Pawn tops: Phosphor can reproduce Pawn_Top_B/W by loading ABeautifulGame as-is, provided it implements KHR_materials_transmission + KHR_materials_volume (thickness; attenuationDistance infinite means no absorption) with the tint in baseColorFactor. This matches the MaterialX definition (depth = 0 -> tint on the closure) and the official translation. Open: transmission rendering of a solid with ior 1.5 is the same in both (no ior extension, default 1.5).
- Kings/queens: the glTF sample intentionally omits subsurface, so loading ABeautifulGame will render the four pieces as opaque metal-rough PBR, NOT as the Track Studio (MaterialX) look. The four materials need a Phosphor-defined, documented extension of the glTF mapping: carry (subsurface weight map, subsurface_scale in metres, colour = base colour) either in a Phosphor vendor extension / `extras`, or in KHR_materials_volume_scatter if the owner accepts a draft extension. The official MaterialX->glTF translation drops these inputs, so no standard mapping exists to follow.
- The material correspondences and the units caveat (scale "metres" is a convention) must be written in the F15 plan as explicit decisions; Queen_W differs (0.001).

## 8. Open points (unverified)

1. Asset license of the MaterialX copy of the chess set (only the CC-BY-4.0 notice in the Khronos copy was found).
2. Whether Track Studio resolves the duplicate `surfaceshader` input on `M_King_W` (file defect) to King_W; and whether the newest terrain3d-* worktree code differs from the one read.
3. Metallic channel content of Pawn_ORM.jpg and pawn_shared_roughness.jpg parity (pixels not inspected).
4. Whether thicknessFactor 0.22 on pawn tops is intentional (about 10x mesh extent).
5. Pixel content/range of `king_shared_scattering.jpg` / `queen_shared_scattering.jpg` (not inspected) and whether 0.003 is intended in metres.
6. KHR_materials_volume_scatter status after 2026-02-09 (last PR update); Autodesk spec page commit SHA not obtained.
