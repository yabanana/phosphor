# Khronos reference corpus for the F15.3 material testbench

Research date 2026-11-13. All facts below come from primary sources fetched through the GitHub API / raw files at pinned commits. "Unverified" marks anything not read directly.

## Pinned sources

| Repo | Commit inspected | Date | Notes |
|---|---|---|---|
| KhronosGroup/glTF-Sample-Assets | `8f492390ddc7a2ae221a1369ef560589cf8c70fb` | 2026-10-23 | models, model-index, per-model README/metadata |
| KhronosGroup/glTF-Render-Fidelity | `9d35e142c05a661c9531cb6aaee73d887b8c79ca` | pushed 2026-10-02 | website + thumbnails (webp) |
| KhronosGroup/glTF-Render-Fidelity-Generator | `5eb9669fe83d33c9cd13fbcd61a6b775776b3163` | pushed 2026-11-10 | scenario config, full PNG goldens (Git LFS), environments |
| KhronosGroup/glTF-Sample-Renderer | `cc3d5ff52217a63161c5c876677292546c01564a` | 2026-11-12 | renderer library used by the Viewer |
| KhronosGroup/glTF-Sample-Viewer | `7306f9ce1b43e784ec4cd36215d55ee1f7db2fc0` | main at time of query | web app |

URL pattern for any file: `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/<Name>/glTF/<Name>.gltf` (the `.gltf` variant is plain JSON with external `.bin` and PNG/JPG files). Model index: `.../8f49239.../Models/model-index.json` (150 models; tags only, no extension list - the extensions below were read from each model's `extensionsUsed`/`extensionsRequired` and from `Models/<Name>/README.md`, which lists "Extensions Used").

Key facts:
- In the default `glTF` variant of every model listed below, no `KHR_draco_mesh_compression`, `KHR_texture_basisu` or `EXT_meshopt_compression` appears in `extensionsUsed`, and images are PNG/JPEG only (checked: ToyCar, AnisotropyBarnLamp, ClearCoatTest, SheenTestGrid, IridescenceDielectricSpheres, TextureTransformTest, ScatteringSkull; the others were checked for extensionsUsed only, image formats unverified). Draco / BasisU / Meshopt / WebP appear only as separate variants (`glTF-Draco`, `glTF-KTX-BasisU`, `glTF-Meshopt`, `glTF-WEBP`) per model-index.json. Exception: `SheenWoodLeatherSofa` requires `EXT_texture_webp` in the default variant; `MeshoptCubeTest` uses meshopt.
- `KHR_texture_transform` is carried as a texture-info extension (`offset`, `rotation`, `scale`, optional `texCoord`). tinygltf stores unknown texture-info extensions as generic JSON values, so the loader must read it from `TextureInfo.extensions` (not verified against the Phosphor tinygltf version).
- Sizes below are the sum of files in `Models/<Name>/glTF/` (git tree sizes), excluding screenshots.
- Models marked "Required: ..." list the extension in `extensionsRequired`, so a loader that ignores it renders the model wrongly rather than degrading.

## Models per extension

Class: T = controlled test (grid / labelled / negative-control design), S = showcase. Licenses from each `README.md` "Legal" block and `metadata.json` (SPDX). "CC-BY" = CC-BY-4.0, attribution to the holder shown.

### KHR_materials_clearcoat
| Model | Other extensions | Class | License / attribution | Size | TT/Draco/Basis |
|---|---|---|---|---|---|
| ClearCoatTest | - | T (labels, partial coat, ribs normal) | CC-BY, (c) 2020 Analytical Graphics, Inc., Ed Mackey | 0.27 MB | none |
| ClearCoatCarPaint | texture_transform (required) | T (sphere over car-paint base + normal bump) | CC0, Eric Chadwick | 0.12 MB | TT |
| CompareClearcoat | ior | T (metal spheres rough 0 / 0.5 / 0.5+coat; from the glTF PBR site) | CC0 (logo Khronos legal mark) | 0.20 MB | none |
| ClearcoatWicker | - | S | CC0, Eric Chadwick | 1.3 MB | none |
| TextureTransformMultiTest | clearcoat x9 materials, unlit, TT (required) | T | CC-BY, Analytical Graphics, Ed Mackey | 0.41 MB | TT |
| PotOfCoals, CarConcept, ToyCar, StainedGlassLamp, CommercialRefrigerator, AnisotropyBarnLamp | various | S | mixed CC-BY / CC0 | 1.3-34 MB | see below |

### KHR_materials_sheen
| Model | Other | Class | License | Size | Notes |
|---|---|---|---|---|---|
| SheenTestGrid | - | T (16 spheres, sheenColor x sheenRoughness over checker, baseColor 0.5 blue) | CC0, Eric Chadwick | 1.9 MB | none |
| CompareSheen | - | T | CC0 | 0.89 MB | none |
| SheenCloth | TT (required) | S (Microsoft donation) | CC0, Microsoft | 4.2 MB (no GLB variant) | TT |
| SheenChair | TT (required), variants | S | CC0, Wayfair LLC, Eric Chadwick | 4.1 MB | TT |
| SpecularSilkPouf | specular+sheen on the same material | S | CC-BY, Wayfair 2023 | 4.6 MB | none |
| GlamVelvetSofa | specular+sheen (5 materials), TT required, lights_punctual | S | CC-BY, Wayfair 2021 | 3.2 MB | TT |
| ChairDamaskPurplegold, SheenWoodLeatherSofa (webp, avoid) | | S | CC-BY | 2.0 / 10 MB | TT |

### KHR_materials_anisotropy
| Model | Class | License | Size | Notes |
|---|---|---|---|---|
| AnisotropyStrengthTest | T (strength parameter) | CC0, Ed Mackey | 0.11 MB | none |
| AnisotropyRotationTest | T (rotation offsets; README says full IBL should show horizontal stretching) | CC-BY, (c) 2023 Analytical Graphics, Ed Mackey | 0.84 MB | none |
| AnisotropyDiscTest | T (anisotropy texture map) | CC0, Ed Mackey | 0.25 MB | none |
| CompareAnisotropy | T | CC0 | 1.1 MB | TT |
| CarbonFibre | S/T (anisotropy only) | CC-BY, Wayfair | 0.43 MB | none |
| AnisotropyBarnLamp | S; anisotropy+clearcoat on one material; separate emissive_strength and transmission+volume materials | CC-BY, Wayfair LLC 2023, Eric Chadwick | 7.8 MB | `glTF-KTX-BasisU` is only an alternative variant; the default is plain |

### KHR_materials_transmission / KHR_materials_volume / KHR_materials_ior
| Model | Extensions | Class | License | Size |
|---|---|---|---|---|
| TransmissionTest | transmission (+xmp) | T (grid of uniform/textured transmission) | CC0, Adobe | 1.5 MB |
| TransmissionRoughnessTest | transmission, ior, volume | T (roughness x IOR tiles) | CC-BY, Analytical Graphics 2021 | 0.40 MB |
| TransmissionThinwallTestGrid | transmission, volume, ior (top row no volume = thin wall) | T | CC0 | 1.2 MB |
| TransmissionOrderTest | transmission, volume | T (sorting) | CC0 | 2.3 MB |
| AttenuationTest | transmission, volume | T (attenuation, thickness, scale) | CC-BY, Analytical Graphics 2021 | 0.08 MB |
| IORTestGrid | transmission, volume, specular, ior | T | CC0 | 2.7 MB |
| CompareTransmission / CompareVolume / CompareIor | | T | CC0 | 0.36 / 0.41 / 0.22 MB |
| DragonAttenuation | transmission, volume, variants | S | Stanford Graphics Library license (not CC; check LICENSES/LicenseRef-Stanford-Graphics.txt) | 6.4 MB (Meshopt variants exist) |
| GlassVaseFlowers | transmission, volume | S | CC0 | 1.8 MB |
| GlassHurricaneCandleHolder | transmission, volume | S | CC-BY, Wayfair 2021 | 2.7 MB |
| MosquitoInAmber | transmission, ior, volume | S | CC-BY, Sketchfab/Loic Norgeot | 24 MB |
| ABeautifulGame | transmission, volume | S | CC-BY, ASWF / MaterialX | 30 MB (+ KTX-ETC1S-Draco variant) |
| ToyCar | clearcoat, transmission, sheen (each on a different material), TT | S | CC0, Guido Odendahl | 5.4 MB |
| StainedGlassLamp | transmission, clearcoat (same material), ior, volume, variants | S | CC-BY, Wayfair | 34 MB (huge) |

### KHR_materials_specular
SpecularTest (T, CC-BY Analytical Graphics 2021 Ed Mackey, 0.24 MB), CompareSpecular (T, CC0, 1.4 MB), SpecularSilkPouf (S), GlamVelvetSofa (S), ChairDamaskPurplegold (S), TrafficCone (S; also retroreflection).

### KHR_materials_iridescence
| Model | Other | Class | License | Size |
|---|---|---|---|---|
| IridescenceDielectricSpheres | ior | T (non-metal) | CC0, Khronos | 0.37 MB (glTF only, no GLB) |
| IridescenceMetallicSpheres | ior | T (metal) | CC0, UX3D | 0.35 MB (glTF only) |
| IridescenceSuzanne | ior, transmission, volume, lights_punctual | T/S | CC0, UX3D | 0.51 MB |
| CompareIridescence | - | T | CC0 | 0.22 MB |
| IridescenceLamp | ior, transmission, volume | S | CC-BY, Wayfair 2022 | 4.1 MB |
| IridescentDishWithOlives | ior, transmission, volume | S | CC-BY, Wayfair 2020 | 5.8 MB |
| IridescenceAbalone | - | S | CC-BY, Abby Crawford + Eric Chadwick | 9.3 MB |
| SunglassesKhronos | transmission, volume, ior, iridescence | S | CC-BY, DGG | 0.38 MB (Draco variant exists; default plain) |
| CarConcept | clearcoat+iridescence on one material, emissive_strength, transmission, TT, variants | S | CC-BY, DGG | 11 MB |

### KHR_materials_diffuse_transmission
| Model | Other | Class | License | Size |
|---|---|---|---|---|
| DiffuseTransmissionTest | unlit (required), lights_punctual (required) | T (follows the extension README layout) | CC-BY, DGG 2025 | 0.27 MB |
| DiffuseTransmissionTeacup | - | S | CC0 (Polyhaven, Eric Chadwick) | 4.8 MB |
| DiffuseTransmissionPlant | lights_punctual | S | CC-BY DGG 2024 (+CC0 plant) | 5.8 MB |
| MandarinOrange | - | T | CC-BY, zamdreamer 2015 / Wayfair 2023 | 4.1 MB (glTF only) |

### Volume scatter
The only sample is **ScatteringSkull** (index tag `testing`; CC0, Vladimir Petkovic; glTF dir 7.0 MB incl. a 2 MB `aoThickness.png`; plain PNG). Its `extensionsUsed` is `KHR_xmp_json_ld, KHR_materials_diffuse_transmission, KHR_materials_volume, KHR_materials_dispersion, KHR_materials_volume_scatter, KHR_materials_ior`; `extensionsRequired` absent. Screenshot is a Substance 3D Stager render (README). Naming caveat: the Sample Viewer README (commit 7306f9c) lists `KHR_materials_scatter` (PR #2579) and `gltf_state.js` enables `KHR_materials_scatter`; the skull uses the name `KHR_materials_volume_scatter`, and `KhronosGroup/glTF/extensions/2.0/Khronos` at main contains no scatter directory (only `KHR_materials_volume`). So the extension is still draft/renamed; whether the Sample Renderer reads the `volume_scatter` spelling is unverified. Treat as an optional/experimental case.

### KHR_materials_emissive_strength
EmissiveStrengthTest (T, CC-BY AGI 2022 Ed Mackey, 18 KB, has a `test_fail.jpg` negative screenshot), CompareEmissiveStrength (T, CC0, 114 KB), AnisotropyBarnLamp, CarConcept (S).

### KHR_texture_transform
TextureTransformTest (T, base color only, CC0 Microsoft 2018, 30 KB, glTF only; uses offset, rotation and scale), TextureTransformMultiTest (T, several inputs incl. texCoord override, CC-BY, 0.41 MB; the README expects rows of checkmarks that look identical across the three tiles), plus ClearCoatCarPaint, SheenChair, SheenCloth, ToyCar, CompareAnisotropy, GlamVelvetSofa (all TT use).

Not covered by any listed model on one material: clearcoat+sheen together (ToyCar has them on separate materials; verified per material), anisotropy+sheen. Verified same-material combinations: anisotropy+clearcoat (BarnLamp), sheen+specular (SilkPouf, GlamVelvetSofa), clearcoat+transmission (StainedGlassLamp), iridescence+transmission+volume+ior (IridescenceSuzanne/Lamp), clearcoat+iridescence (CarConcept), clearcoat+ior (CompareClearcoat).

## Recommended minimal corpus

All paths relative to `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/`; fetch the whole `<Name>/glTF/` folder plus `<Name>/README.md` and `metadata.json` (license record). Use the git tree API at the pinned SHA to list the folder (`gh api repos/KhronosGroup/glTF-Sample-Assets/git/trees/8f49239...?recursive=1`).

Tier A, controlled tests (about 20.4 MB total, all plain glTF + PNG/JPG, no Draco/Basis/Meshopt):

| Lobe / purpose | Model | Why |
|---|---|---|
| clearcoat | `ClearCoatTest/glTF/` | factor, roughness, normal, partial coat |
| clearcoat + base-layer + TT | `ClearCoatCarPaint/glTF/` | coat over normal-mapped base, TT required (CC0) |
| sheen | `SheenTestGrid/glTF/` | sheenColor x sheenRoughness grid (CC0) |
| anisotropy | `AnisotropyStrengthTest`, `AnisotropyRotationTest`, `AnisotropyDiscTest` | strength, rotation (rotation = the critical case), texture map |
| transmission | `TransmissionTest`, `TransmissionRoughnessTest` | uniform/textured, roughness x IOR |
| volume / attenuation | `AttenuationTest` (79 KB), `IORTestGrid` | attenuation x thickness; IOR |
| thin-wall vs volumetric | `TransmissionThinwallTestGrid` | same IOR with and without volume |
| specular | `SpecularTest` | specularFactor/color + textures |
| iridescence | `IridescenceDielectricSpheres`, `IridescenceMetallicSpheres` (glTF only), `IridescenceSuzanne` | dielectric vs metal; iridescence+transmission+volume |
| diffuse transmission | `DiffuseTransmissionTest` | needs lights_punctual + unlit support (both required) |
| volume scatter (optional) | `ScatteringSkull` | only sample; experimental naming |
| emissive strength | `EmissiveStrengthTest` | 18 KB |
| texture transform | `TextureTransformTest`, `TextureTransformMultiTest` | TT offset/rotation/scale + texCoord override (the Multi test also carries clearcoat and unlit) |

Tier B, critical combinations on a single material (about 59.5 MB for the full list; take only what the lobe set needs):
- `AnisotropyBarnLamp` (7.8 MB): anisotropy+clearcoat, plus transmission+volume and emissive_strength materials.
- `SpecularSilkPouf` (4.6 MB) and `GlamVelvetSofa` (3.2 MB, TT): sheen+specular.
- `ToyCar` (5.4 MB, CC0): clearcoat, sheen, transmission and TT across separate materials, the Khronos poster for clearcoat quality; eight cameras are included.
- `SheenCloth` (4.2 MB) or `SheenChair` (4.1 MB): sheen with TT.
- `StainedGlassLamp` (34 MB, avoid unless the clearcoat+transmission case is needed; use `CommercialRefrigerator`, 10 MB, as the alternative only if its materials suit).
- `CarConcept` (11 MB): clearcoat+iridescence on one material (verify default-variant images are PNG/JPG before use).
Compare models (`CompareClearcoat`, `CompareSheen`, `CompareIridescence`, ...) are small (0.2-1.4 MB) A/B scenes of "feature off vs on"; useful as a cheap sanity set, CC0.

Avoid: `DragonAttenuation` (Stanford license, not CC), `SheenWoodLeatherSofa` (WebP required), `ABeautifulGame` (30 MB, default variant has no compression but is large), `MosquitoInAmber` (24 MB), `StainedGlassLamp` (34 MB). Any model's `glTF-Draco`, `glTF-KTX-BasisU`, `glTF-Meshopt*`, `glTF-WEBP` variant must not be fetched.

Licensing for redistribution: CC-BY models need the attribution text kept (store each `metadata.json` / README Legal block next to the model). CC0 models need nothing. Models with the Khronos legal mark (`Compare*`, DiffuseTransmissionTest, ToyCar-style READMEs list "Khronos Trademark or Logo" for README images) carry a trademark note for the logo only. The whole Sample-Assets repo README states "Licensed as CC-BY-4.0 International" for the repository itself; per-model README controls.

## Reference images

### In the Sample-Assets repository (per model)
- `Models/<Name>/screenshot/screenshot.jpg` (small thumbnail, e.g. ClearCoatTest 32 KB, ScatteringSkull 15 KB) and often `screenshot_large.jpg` / `screenshot-large.jpg` (ClearCoatTest 174 KB, ToyCar 672 KB), plus ad-hoc images referenced by the README (e.g. `FirstRow.jpg`, `PartialCoat.jpg`, `ibl-example.png`, `dassault-stellar.jpg`, `test_fail.jpg`).
- Generation is not uniform and not specified in a machine-readable way. READMEs state it case by case: SheenTestGrid screenshots come from glTF Sample Viewer with environment "Studio Neutral" (debug views) and a "Punctual Lighting" run (two opposing directional lights); ScatteringSkull from Adobe Substance 3D Stager; DiffuseTransmissionTest includes a Dassault Stellar render; AnisotropyRotationTest `ibl-example.png` is an IBL example. `ReviewProcess.md` (same commit) only mentions an intended "screenshot chain" rule (screenshot is an unmodified capture of the submitted asset) as a to-do, not an enforced property. Conclusion: these screenshots are qualitative only, with unknown, differing lighting and tone mapping per model.
- `metadata.json` carries license, artist, summary and `screenshot` path; `model-index.json` carries label, tags and variants only.

### Khronos Render Fidelity project
- Website repo `glTF-Render-Fidelity` @9d35e14 contains only 1k-class thumbnails (`public/thumbnails/goldens/<scenario>/<renderer>-golden.thumb.webp`, 786 files, 17-40 KB each, built by `src/scripts/create_thumbnails.mjs`) and the HDRs `public/env_maps/chinese_garden_1k.hdr`, `qwantani_afternoon_puresky_1k.hdr` (used by the site's 3D compare pages; their role in the goldens is not stated).
- The full-size goldens are in `glTF-Render-Fidelity-Generator` @5eb9669: `test/goldens/<scenario>/<renderer>-golden.png`, tracked by Git LFS (`.gitattributes: *.png filter=lfs`; the raw files are 131-byte LFS pointers - confirmed by tree sizes). Fetching them needs `git lfs` or the GitHub LFS media URL. They are not in the Sample-Assets repo.
- Renderers: three.js via model-viewer, Filament, Babylon, glTF-Sample-Viewer, three-gpu-pathtracer (real-time or web path tracer); Dassault STELLAR, Chaos V-Ray, Blender Cycles (path tracers), plus Rhodonite and dspbr-pt for some. Counts in the tree: filament 91, model-viewer 90, sample-viewer 90, blender-cycles 90, babylon 89, three-gpu-pathtracer 67, vray 66, rhodonite 56, stellar 32, dspbr-pt 24 images; 86 `khronos-*` scenarios in `test/config.json` plus `DirectionalLightTest`. Not every renderer is run on every model (`exclude` lists in the config, e.g. TransmissionTest excludes three-gpu-pathtracer and vray).
- Models with goldens that overlap the recommended corpus: ClearCoatTest, ClearCoatCarPaint, SpecularTest, TransmissionTest, TransmissionRoughnessTest, AttenuationTest, AnisotropyRotationTest, AnisotropyStrengthTest, AnisotropyBarnLamp, EmissiveStrengthTest, IridescenceDielectricSpheres, IridescenceMetallicSpheres, IridescenceSuzanne, TextureTransformTest, TextureTransformMultiTest, ToyCar, SheenChair, SheenCloth, GlamVelvetSofa, GlassVaseFlowers, MosquitoInAmber, ABeautifulGame, IridescenceLamp, DragonAttenuation, IridescentDishWithOlives. **No golden** exists for SheenTestGrid, IORTestGrid, DiffuseTransmissionTest, AnisotropyDiscTest, TransmissionThinwallTestGrid, ScatteringSkull or the `Compare*` models (checked against `test/config.json`).
- Default scenario (`src/config-reader.ts`): environment `environments/lightroom_14b.hdr` (2.8 MB, in the Generator repo), dimensions 768x768 (overridden per model, e.g. EmissiveStrengthTest 1000x600), vertical FoV 45 degrees, orbit theta 0 / phi 90 / radius 1 with per-model radius/target, skybox not rendered unless `renderSkybox: true` (set for ABeautifulGame, IridescenceLamp, MosquitoInAmber, IridescenceSuzanne and a few more). `common.ts` defines `DEVICE_PIXEL_RATIO = 2`; whether the PNG pixel size is 2x the CSS size could not be checked without LFS (unverified). Some scenarios override the environment (IridescentDishWithOlives and MetalRoughSpheres-HDR use `spruit_sunrise_1k_HDR.hdr`; DamagedHelmet uses `aircraft_workshop_01_1k.hdr`). The Generator also ships `environments/white_furnace.hdr` (143 KB) - see below.
- Tone mapping: the sample-viewer golden renderer sets `GltfState.ToneMaps.ACES_HILL_EXPOSURE_BOOST` explicitly, environment rotation 0, clear colour transparent (0,0,0,0), no environment blur (`src/components/renderers/gltf-sample-viewer.ts`). The other renderers' tone mapping per golden is not documented in the files read (unverified); the point of the project is that each renderer uses its own defaults under the same HDR.
- Comparison tooling: `src/common.ts` `ImageComparator` with the pixelmatch YIQ color delta (`MAX_COLOR_DISTANCE = 35215`), RMS distance ratio in dB, `FIDELITY_TEST_THRESHOLD = -22` dB (for model-viewer regression), black/white and delta images; CLI `npm run test`, `npm run render-goldens` (needs ImageMagick, Puppeteer); results viewer `test/results-viewer.html`. The website's comparison pages use a slider, side-by-side and a difference view (`src/components/ImageComparison/*`).

### Can they be numeric references?
No, not for Phosphor as it lights today. The goldens are produced under a Lightroom-style HDR environment with IBL as the dominant light and an ACES-class tonemap, from renderers that differ from each other (the project itself exists to show their divergence; path tracers and real-time ones differ visibly on clearcoat, sheen, transmission). Phosphor has a constant isotropic environment, a sun disk, punctual/area lights and no HDR environment maps, so even spatially varying specular reflections and anisotropic stretching (the AnisotropyRotationTest README says full IBL is needed) cannot match. Use them as qualitative side-by-side references only (pass criteria like the test READMEs' "rows of checkmarks" for TextureTransformMultiTest, no blur in the smooth column of TransmissionRoughnessTest, sheen color/roughness monotonic along the grid).

For numeric comparison one would need an identical, fully specified light: (a) the same HDR (lightroom_14b.hdr and the sample-viewer settings above, or any HDR the project adds, with the importance-sampled specular prefilter and diffuse SH/irradiance), same camera (FoV 45, radius/target from `test/config.json`), same tonemap, and a reference renderer run locally (Sample-Renderer or Blender Cycles / a path tracer) on the identical inputs; or (b) analytic cases that need no HDR: **white furnace** (`environments/white_furnace.hdr` in the Generator repo is a uniform environment, i.e. exactly Phosphor's constant radiance) which gives closed-form energy conservation checks (albedo <= 1, sheen/clearcoat/transmission energy compensation), and a directional light on test grids compared with the Sample-Renderer using `useIBL=false` / punctual only (the `SheenTestGrid` README shows a "Punctual Lighting" run with two opposing directional lights). A path-tracer render with the constant environment of Phosphor as reference is the cleanest numeric target. The white furnace file content (constant value) was not inspected; whether its radiance is exactly 1.0 is unverified.

## Sample Viewer defaults that matter for a qualitative side-by-side

From `glTF-Sample-Renderer` @cc3d5ff `source/GltfState/gltf_state.js` and `glTF-Sample-Viewer` @7306f9c `src/main.js`, `src/logic/viewer_store.js`:
- Tone mapping: the Viewer app forces `KHR_PBR_NEUTRAL` ("Khronos PBR Neutral"); the library default is also PBR Neutral. Alternatives in the UI/enum: ACES Hill (+ "exposure boost" = exposure 1/0.6), ACES Narkowicz, ACES Hill, linear clamped. Exposure default 1.0 (library) / 0 EV (viewer store). The render-fidelity goldens use ACES Hill exposure boost instead; so Viewer-app screenshots and fidelity goldens differ in tonemapping. Compare Phosphor's tonemap against the one used by the picture you diff against.
- IBL: `useIBL: true`, `iblIntensity` 1.0 (viewer store stores log10, default 0.0 -> 1.0), `environmentRotation` 90 (front face +Z). Environment list in `main.js` (loaded from `KhronosGroup/glTF-Sample-Environments`, branch `low_resolution_hdrs`): Cannon Exterior, Footprint Court, Pisa, Doge's palace, Dining room, Field, Helipad Goldenhour, Papermill Ruins, Studio Neutral, Colorful Studio, Wide Street; the store default `selectedEnvironment: 0` is presumably the first (Cannon Exterior) - inferred, not confirmed in a UI default.
- Clear colour `[0.22, 0.25, 0.29, 1]` in the library; skybox shown in the viewer unless disabled. `useDirectionalLightsWithDisabledIBL = true`: with IBL off the viewer generates directional lights (this is the "Punctual Lighting" mode used for SheenTestGrid screenshots). The UI exposes debug outputs (e.g. sheen, sheen color, sheen roughness) usable as per-lobe visual references.
- All material extensions in this task, including `KHR_materials_volume`, `KHR_materials_diffuse_transmission`, `KHR_materials_dispersion` and `KHR_materials_scatter`, are enabled by default in the renderer (`enabledExtensions`), so a Viewer screenshot of any recommended model shows the extension; each can be toggled off to isolate a lobe.
- The Viewer README additionally lists Draco, Meshopt and quantization support, so Viewer screenshots of compressed variants look identical to plain variants.

## Open points
1. Real size of the PNG goldens and whether DPR 2 is applied: not checked (LFS pointers only). Fetch one with `git lfs` if the diff tool needs it.
2. Per-renderer tone mapping in the goldens other than sample-viewer: not read.
3. Which volume-scatter spelling the Sample Renderer implements (`KHR_materials_scatter` vs `KHR_materials_volume_scatter`): unverified; extension still draft.
4. Image formats for models other than the seven spot-checked ones: unverified (extensionsUsed shows no basisu/webp/draco for them except where noted).
5. Whether tinygltf (the version Phosphor pins) exposes `KHR_texture_transform`, `KHR_materials_*` (it does not parse them natively; they arrive as `Value` trees in `extensions`): not checked here.
6. No model puts clearcoat and sheen on one material; an F15 combination test for that pair will need a hand-authored glTF (e.g. a small variation of `SheenTestGrid` plus a clearcoat block) rather than a Khronos sample.
7. `DiffuseTransmissionTest` requires `KHR_materials_unlit` and `KHR_lights_punctual`; the corpus loader must support both or the model renders wrong.
