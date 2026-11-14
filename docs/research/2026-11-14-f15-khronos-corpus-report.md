# F15.3 Khronos corpus run and defect report (ticket 22)

Run `2026-11-14T10:00:59Z` of `tools/f15_corpus.py` on binary `a7958e5c7a84`; corpus glTF-Sample-Assets `8f49239` (default glTF variant, owner-approved subset, never committed). Text only: the side-by-side page is generated under `build/` by `tools/f15_corpus_page.py` because the Khronos screenshots are third-party content.

## Declared rig and what this report is not

No global parity is claimed and none is ticked. The Khronos images were made with image-based lighting (an HDR environment) and the tonemap of the tool that produced them (Sample Viewer: Khronos PBR Neutral; Render Fidelity goldens: ACES Hill; some screenshots come from path tracers such as dspbr-pt, Stellar or Substance Stager, per each model's README). Phosphor has no image-based lighting: the declared rig is one white sun (3 W/m2) from the upper left behind the camera over a constant isotropic environment (radiance 0.5), exposure 1.0, ACES tonemap, a 60 degree field of view framed from the model bounds, one settled frame. Background, exposure, reflections of the environment and the look of every smooth or anisotropic surface therefore differ by construction. The comparison is qualitative and per feature: the numeric gates of F15 are the independent per-lobe references (tools/f15_check.py), not these images.

Rig constants: 1024x768 (1600x640 for IridescenceSuzanne), 48 warm-up + 48 measured frames, one settled frame captured, `--rt on --shadows rt --lighting restir --reflections rt --lighting-denoise custom --post --tonemap aces`, `--corpus-rig YAW,PITCH` (`src/testbench/corpus_rig.h`).

## Sample Renderer deviations cited

- **Clearcoat F0.** Phosphor follows the ratified text: coat F0 fixed at IOR 1.5 (0.04). The Sample Renderer derives the coat F0 from the material IOR; where a model sets KHR_materials_ior the coat highlight strength differs (negative control `coat-f0-from-ior`, ticket 09).
- **Base under coat / sheen.** Phosphor weights the base by (1 - weight * max(F)) as the extension texts specify; the Sample Renderer composes the layers with its own albedo scaling.
- **Iridescence.** Phosphor evaluates the thin film per light sample at V.H and the environment / albedo terms at N.V (ticket 12); the Sample Renderer uses N.V everywhere, so the hue shift under punctual lights differs near grazing half-vectors.
- **Anisotropy in the environment.** The Sample Renderer fetches the image-based light along a bent normal; Phosphor's declared environment model is a split sum at an effective roughness, which is invisible in a constant environment (ticket 11). Anisotropic stretching of environment reflections, which the Khronos AnisotropyRotationTest README says needs image-based lighting, cannot appear here.
- **Sheen table.** Phosphor generates its own Charlie E(mu, roughness) table (tools/sheen_table.py); the Sample Renderer ships a pre-baked LUT image. Both clamp at the table edge.
- **Lighting and tonemap.** Image-based lighting, the HDR environment itself, its orientation, the clear colour / skybox and the tonemap are different in every Khronos image; see DISCLAIMER.
- **Unsupported by design.** KHR_materials_unlit and KHR_lights_punctual are reported unsupported (decision 02): a model that requires them is refused, a model that merely uses unlit materials renders them lit.

## Run summary

- Models: 21 rendered, 3 refused with a stated loader reason, 0 failed unexpectedly.
- Archive: 0 archive misses and 0 fallback draws over the whole final run (before the targeted update 110 misses over 10 models).
- O7: 0 GPU allocations in the measured frames, summed over the rendered models (warm-up 48 frames, measured 48).
- Validation (API + shader, `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`, archive off): 21 of 21 rendered models clean (exit 0, EXIT marker 0, no validation failure, no unclassified line); the only messages are the known baseline notices of the validation layer (redundant setDepthStencilState / setViewport calls, 96 of each per capture, identical on the phase-head binary per the ticket 15 record) and loader warnings.

## Load / refusal table

| Model | Tier | Required extensions | Result | Archive misses (before -> after update) | Fallback draws | O7 allocs | Validation |
|---|---|---|---|---|---|---|---|
| ClearCoatTest | A | - | rendered | 19 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| ClearCoatCarPaint | A | KHR_texture_transform, KHR_materials_clearcoat | rendered | 19 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| SheenTestGrid | A | KHR_materials_sheen | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| AnisotropyStrengthTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| AnisotropyRotationTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| AnisotropyDiscTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| TransmissionTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| TransmissionRoughnessTest | A | - | rendered | 3 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| TransmissionThinwallTestGrid | A | - | rendered | 3 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| AttenuationTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| IORTestGrid | A | - | rendered | 3 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| SpecularTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| IridescenceDielectricSpheres | A | - | rendered | 3 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| IridescenceMetallicSpheres | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| IridescenceSuzanne | A | - | rendered | 19 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| DiffuseTransmissionTest | A | KHR_materials_unlit, KHR_lights_punctual | REFUSED: glTF /Users/danielsan/Documents/phosphor/.claude/worktrees/f15-plan-c38b51/build/khronos-corpus/DiffuseTransmissionTest/glTF/DiffuseTransmissionTest.gltf requires extension(s) that cannot be honoured: KHR_materials_unlit (not understood by this loader), KHR_lights_punctual (not understood by this loader); refusing to load | 0 -> 0 | 0 | - | refused before rendering (loader reason in the table) |
| EmissiveStrengthTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| TextureTransformTest | A | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| TextureTransformMultiTest | A | KHR_texture_transform | REFUSED: glTF /Users/danielsan/Documents/phosphor/.claude/worktrees/f15-plan-c38b51/build/khronos-corpus/TextureTransformMultiTest/glTF/TextureTransformMultiTest.gltf requires extension(s) that cannot be honoured: KHR_texture_transform (material 1 ('BaseColorTest1Mat') has a non-uniform transform); refusing to load | 0 -> 0 | 0 | - | refused before rendering (loader reason in the table) |
| AnisotropyBarnLamp | B | - | rendered | 19 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| ToyCar | B | - | rendered | 19 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| SpecularSilkPouf | B | KHR_materials_specular, KHR_materials_sheen | rendered | 3 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |
| GlamVelvetSofa | B | KHR_texture_transform | REFUSED: glTF /Users/danielsan/Documents/phosphor/.claude/worktrees/f15-plan-c38b51/build/khronos-corpus/GlamVelvetSofa/glTF/GlamVelvetSofa.gltf requires extension(s) that cannot be honoured: KHR_texture_transform (material 2 ('GlamVelvetSofa_fabric_champagne') has a non-uniform transform); refusing to load | 0 -> 0 | 0 | - | refused before rendering (loader reason in the table) |
| ABeautifulGame | chess | - | rendered | 0 -> 0 | 0 | 0 | clean (exit 0; only the known baseline notices: 192 redundant state calls) |

## Per model

### ClearCoatTest

Shows: clearcoat factor, roughness, coat normal, partial coat (labelled grid; base / coated / coat-only columns).

Engine support report: clearcoat: approximated (secondary_rt_hits_ignore_coat;di_emitters_uncoated) x6.

Framing: 27 instances, bounds [-7.3578, -6.25, -0.06] to [3.1, 6.8545, 1.05], camera distance 13.2657.

Renders correctly:

- All 19 materials load; the three columns (base, coated, coat only) and the six rows (simple, partial, roughness, base normal map, shared normal map, coat normal map) are laid out as in the Khronos image.
- Partial coating (coat factor texture), coat roughness texture and the coat normal texture rows visibly change the coated column against the base column; the base normal map ribs appear on the base tiles.
- Coat-only column: the coat lobe is visible as the sun highlight and the rim reflection of the constant environment.

Named approximations:

- No image-based lighting: the coat-only tiles read near black (coat F0 0.04 x environment 0.5) where the Khronos tiles reflect an HDR environment; the coat highlight is the sun disk only.
- The engine support report names clearcoat as approximated: secondary_rt_hits_ignore_coat, di_emitters_uncoated.
- alphaMode BLEND (label planes) is rendered opaque until F16 (loader warning).

### ClearCoatCarPaint

Shows: clear coat over a normal-mapped car-paint base; KHR_texture_transform required.

Engine support report: clearcoat: approximated (secondary_rt_hits_ignore_coat;di_emitters_uncoated) x1; texture_transform: implemented x1.

Framing: 1 instances, bounds [-0.5001, 0.0098, -0.5001] to [0.5001, 0.9902, 0.5001], camera distance 1.4509.

Renders correctly:

- KHR_texture_transform (required) loads; the car-paint flake normal map shows as sparkle on the red sphere and a coat highlight sits on top.
- The sphere shows the broad Fresnel rim of a smooth coat over a constant environment.

Named approximations:

- Coat F0 fixed at IOR 1.5 (ratified text); the Sample Renderer derives it from the material IOR.
- No HDR environment: the Khronos image shows a street reflected in the coat, here the reflection is the constant environment.

Defects:

- **C22-D3** [open] Isolated dark pixels on the outermost pixel row of the sphere silhouette (also on the SpecularTest, IORTestGrid and Sheen grids). Root-cause hypothesis: Diagnostic runs (build/f15-corpus-diag, same model and rig): the dark pixels are present with --reflections rt (corpus default) and with --reflections ssr and ABSENT with --reflections off, so they come from the reflection signal/composite at geometry edges, not from RT ray origins. Most likely the lobe filter or composite reads the sky/miss pixels next to the silhouette as zero-radiance reflection samples and the outermost pixel row weights them in. Not fixed here: it needs the reflection denoiser/composite shaders (F13), which this ticket does not own.

### SheenTestGrid

Shows: sheenColor x sheenRoughness grid of 16 spheres over a checker; KHR_materials_sheen required.

Engine support report: sheen: approximated (lobe_not_sampled_environment_term_in_composition;sun_disk_point_evaluated;secondary_rt_hits_ignore_sheen) x12.

Framing: 20 instances, bounds [-0.6281, -0.324, -0.045] to [0.2762, 0.3282, 0.0675], camera distance 0.714.

Renders correctly:

- Sheen colour (columns) and sheen roughness (rows) are monotonic: the cyan rim grows with both, the zero-colour column carries no sheen (README criterion).
- KHR_materials_sheen (required) loads; the 16 spheres cast sun shadows on the checker backdrop.

Named approximations:

- Sheen is named approximated: lobe not sampled, environment term in the composition, sun disk point-evaluated, secondary RT hits ignore sheen.
- The constant environment gives every sphere a uniform sheen rim; the Khronos image under 'Footprint Court' has a directional one. Our sun adds a one-sided shadow on the checker, the Khronos grid is lit by the environment only.

### AnisotropyStrengthTest

Shows: anisotropyStrength sweep.

Engine support report: anisotropy: approximated (environment_split_sum_at_effective_roughness;secondary_rt_hits_ignore_anisotropy;no_tangents_means_isotropic) x42.

Framing: 50 instances, bounds [-4.1701, -0.9873, -0.4] to [3.4, 6.4, 0.4], camera distance 7.5653.

Renders correctly:

- Strength rows x roughness columns: the sun highlight stretches along the tangent as strength rises and collapses to the isotropic highlight at strength 0; at roughness 1.0 the anisotropy has no effect (README criterion).

Named approximations:

- Environment split sum at an effective roughness (named): the stretching of environment reflections that the Khronos image shows cannot appear in a constant environment, only the sun highlight is anisotropic.

Defects:

- **C22-D5** [open] Near-mirror anisotropic spheres (strength >= 0.5, roughness <= 0.17) are speckled in the single settled frame; the neighbouring-sphere reflections at strength 0 show black blotches. Root-cause hypothesis: Stochastic VNDF samples of a very narrow anisotropic lobe (8 reflection samples) that the custom denoiser's lobe edge-stopping does not converge in 48 frames; the black blotches are reflections of neighbouring dark spheres (single-bounce RT reflections, see C22-A1).

### AnisotropyRotationTest

Shows: anisotropyRotation offsets (README: needs image-based lighting to show the stretching).

Engine support report: anisotropy: approximated (environment_split_sum_at_effective_roughness;secondary_rt_hits_ignore_anisotropy;no_tangents_means_isotropic) x4.

Framing: 11 instances, bounds [-5.556, -3.5364, -1.0] to [6.6753, 2.4714, 1.0], camera distance 9.8978.

Renders correctly:

- The stacked rings load with tangents + texture + rotation variants; the sun highlight orientation changes with the rotation offsets and the groove normal map ring shows its pattern.

Named approximations:

- The README states the horizontal stretching needs image-based lighting: it is not expected here (constant environment, bent-normal fetch invisible, ticket 11).

### AnisotropyDiscTest

Shows: anisotropy direction texture (disc), strength and rotation.

Engine support report: anisotropy: approximated (environment_split_sum_at_effective_roughness;secondary_rt_hits_ignore_anisotropy;no_tangents_means_isotropic) x11.

Framing: 12 instances, bounds [-3.6801, 0.0858, -0.0001] to [2.6202, 2.6799, 0.001], camera distance 4.5837.

Renders correctly:

- The anisotropy direction texture (disc) loads: in every tile the lower discs show the radial highlight pattern; strength falls off towards roughness 1.0 as in the Khronos image.

Named approximations:

- The 3 % and 10 % strength discs (upper row of each tile) are nearly invisible: with a constant environment only the sun highlight carries the pattern.

### TransmissionTest

Shows: transmission factor / texture / roughness over a backdrop (thin-walled panes).

Engine support report: transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x12.

Framing: 22 instances, bounds [-0.9764, -0.4, -0.521] to [0.8262, 0.6205, 1.368], camera distance 2.2558.

Renders correctly:

- Uniform and textured transmission, opacity (alpha) and metallic transmission spheres are all present; the refracted checker, the tint, the textured stripes and the metallic variant read as in the Khronos image.
- Rough transmission blurs the refracted checker.

Named approximations:

- Thin-walled transmission (no volume) is traced by RT; the spheres refract no environment beyond the constant 0.5.

### TransmissionRoughnessTest

Shows: roughness x IOR tiles of volume slabs over a pattern.

Engine support report: specular_ior: approximated (secondary_rt_hits_use_base_reduction) x4; transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x5; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x5.

Framing: 9 instances, bounds [-1.0, -0.45, -0.15] to [1.0, 0.45, 0.35], camera distance 1.7049.

Renders correctly:

- Roughness x IOR tiles: the smooth column shows the refracted pattern and the pattern blurs monotonically with roughness (README criterion); IOR 1.0 shows no refraction and the higher IORs show the rim reflections.

Named approximations:

- No HDR environment: the bright refracted/reflected highlights of the Khronos tiles come from the environment, here from the sun.

Defects:

- **C22-D4** [open] Near-black grainy squares where a glass tile overlaps the opaque label planes (rows 2 to 5 of the smooth column); the Khronos image shows the label through the glass. Root-cause hypothesis: Disproved: missing ambient/GI at secondary hits (the dark squares are unchanged with --gi ddgi, build/f15-corpus-diag). Still open: the squares sit at the part of the glass tile that overlaps the label panels; the label meshes are single-sided (LabelMat is not doubleSided), so the likely cause is the transmitted ray hitting the BACK of a label plane, which the rasteriser culls but the RT hit shading lights as a back face (no front lighting, noisy transmissive shadow), giving near-black grain. Check by making the RT hit test skip back faces of single-sided opaque surfaces.

### TransmissionThinwallTestGrid

Shows: the same IOR with and without volume (thin wall vs volumetric).

Engine support report: specular_ior: approximated (secondary_rt_hits_use_base_reduction) x8; transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x10; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x5.

Framing: 14 instances, bounds [-11.0493, -0.0088, -0.8991] to [7.7361, 8.6957, 1.3505], camera distance 14.7904.

Renders correctly:

- Thin-wall (top row) against volume (bottom row) spheres over the checker at five IORs: the volume row refracts and inverts the checker (IOR >= 1.33), the thin-wall row does not shift it; IOR 1.0 shows no refraction in either row (the intended difference).

Named approximations:

- Volume thickness is the authored thicknessFactor (1 m); no environment image to reflect.

### AttenuationTest

Shows: attenuationColor x attenuationDistance x thickness.

Engine support report: transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x16; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x11.

Framing: 23 instances, bounds [-10.0, -10.0, -2.0] to [10.0, 10.0, 2.0], camera distance 21.399.

Renders correctly:

- Attenuation colour x distance x thickness: columns read as one blue each and deepen with size and thickness (README pass criterion: a uniform shade per column), thickness texture and node-scale rows included.

Named approximations:

- Transmissive sun shadows (coloured, attenuated) are visible as blue pools on the wall, as in the Khronos path-traced image.

### IORTestGrid

Shows: KHR_materials_ior and specular on transmissive spheres.

Engine support report: specular_ior: approximated (secondary_rt_hits_use_base_reduction) x18; transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x10; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x20.

Framing: 23 instances, bounds [-0.4116, -0.3291, -0.045] to [0.2762, 0.4225, 0.0675], camera distance 0.7852.

Renders correctly:

- Black / white base, specular 0.25 / 1, transmission 0 / 1 at five IORs: the black spheres show the IOR-dependent Fresnel reflection, the transmissive spheres refract the checker; IOR 1.0 with transmission shows no refraction.

Named approximations:

- Specular and IOR use the ratified F0 / F90 model; no HDR environment.

Defects:

- **C22-D7** [open] White transmissive spheres at IOR >= 1.76 are visibly grainy in the single settled frame; the Khronos image is smooth. Root-cause hypothesis: Stochastic refraction (8 samples) of a high-IOR sphere, denoised by the custom denoiser in 48 frames only; the ticket 13 measurements converge with 112 accumulated frames (tools/f15_check.py ACCUMULATE).

### SpecularTest

Shows: specularFactor and specularColorFactor, with textures.

Engine support report: specular_ior: approximated (secondary_rt_hits_use_base_reduction) x21.

Framing: 24 instances, bounds [-0.7508, -0.38, -0.05] to [0.375, 0.38, 0.05], camera distance 0.869.

Renders correctly:

- Specular factor row = specular texture row, the leftmost sphere is black, colour rows darken the centre but not the edge, yellow rows keep a white rim (README criteria); colour factor > 1.0 brightens the reflection.

Named approximations:

- Reflections are the constant environment and neighbouring spheres, not an HDR image.

### IridescenceDielectricSpheres

Shows: iridescence over dielectrics, thickness and film IOR sweep.

Engine support report: iridescence: approximated (environment_film_at_nv_constant_over_lobe;thickness_zero_disables_film;secondary_rt_hits_ignore_iridescence) x343; specular_ior: approximated (secondary_rt_hits_use_base_reduction) x294.

Framing: 346 instances, bounds [-14.5, -12.0, -12.0] to [14.5, 14.5, 12.0], camera distance 47.9055.

Renders correctly:

- 344 materials load (thickness x IOR lattice); the thin-film hue shift is visible on the lower rows where the environment/sun reflect at grazing angles.

Named approximations:

- The film is evaluated per light sample at V.H and at N.V in the environment (ticket 12); the Sample Renderer uses N.V everywhere. White dielectric spheres under a bright constant environment read mostly white: the colour shows only in the reflection.

### IridescenceMetallicSpheres

Shows: iridescence over metals, thickness and film IOR sweep.

Engine support report: iridescence: approximated (environment_film_at_nv_constant_over_lobe;thickness_zero_disables_film;secondary_rt_hits_ignore_iridescence) x343.

Framing: 346 instances, bounds [-14.5, -12.0, -12.0] to [14.5, 14.5, 12.0], camera distance 47.9055.

Renders correctly:

- The metal lattice shows the thin-film colour sweep along thickness and base colour, close to the Khronos image in hue order (blue-magenta-gold-green).

Named approximations:

- Single-bounce RT reflections (see C22-A1).

Defects:

- **C22-A1** [named-approximation] Dense metal lattices show black holes inside each sphere: a sphere sees its neighbours as dark. Root-cause hypothesis: RT reflections are one bounce: the neighbour hit is shaded without its own environment reflection (metals have no diffuse), so a mirror seen in a mirror is dark. Named in the support report as secondary_rt_hits_ignore_<lobe>; the Khronos image renders it with image-based lighting.

### IridescenceSuzanne

Shows: iridescence + transmission + volume + IOR on one material; KHR_lights_punctual used.

Engine support report: iridescence: approximated (environment_film_at_nv_constant_over_lobe;thickness_zero_disables_film;secondary_rt_hits_ignore_iridescence) x3; specular_ior: approximated (secondary_rt_hits_use_base_reduction) x1; transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x1; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x3.

Framing: 3 instances, bounds [-4.3259, -0.9705, -0.7783] to [4.3259, 0.9392, 0.8223], camera distance 4.157.

Renders correctly:

- Three Suzannes: iridescent dielectric, iridescent metal (thickness texture), iridescent + transmission + volume + IOR; the metal one shows the green-magenta thickness variation of the Khronos image.
- KHR_lights_punctual is used but not required: its lights are ignored and the rig light applies.

Named approximations:

- Environment film at N.V constant over the lobe (named).
- The loader warns 'volume requires transmission or subsurface to have an effect' for the opaque materials (volume ignored there).

### DiffuseTransmissionTest

Shows: diffuse transmission; REQUIRES KHR_materials_unlit and KHR_lights_punctual.

Defects:

- **C22-R1** [by-design] Refused: the model REQUIRES KHR_materials_unlit and KHR_lights_punctual. Root-cause hypothesis: Decision 02: both extensions are reported unsupported, so the loader refuses the asset (glTF 3.2) instead of rendering it wrongly. The thin diffuse transmission lobe itself is covered by the independent reference (ticket 16), not by this model.

### EmissiveStrengthTest

Shows: emissiveStrength above 1.

Engine support report: emissive_strength: implemented x4.

Framing: 6 instances, bounds [-8.0026, -6.0011, -2.0] to [8.0011, 4.0094, 1.9989], camera distance 13.6416.

Renders correctly:

- emissiveStrength 1 to 100 folded into the emissive factor: the cubes brighten monotonically from blue to white, the walls and floor stay lit as in the Khronos image.

Named approximations:

- No bloom: the Khronos image shows a glow around the bright cubes.

### TextureTransformTest

Shows: KHR_texture_transform offset / rotation / scale on the base colour.

Engine support report: texture_transform: implemented x6.

Framing: 12 instances, bounds [-1.6, -1.1413, 0.0] to [1.6, 1.05, 0.01], camera distance 2.3329.

Renders correctly:

- Offset, rotation and scale on the base colour: the U, V and UV tiles and the arrow tiles with their check marks match the Khronos image tile for tile (qualitative).
- The README's pass criteria (check mark on every tile, marker squares) hold in the capture.

### TextureTransformMultiTest

Shows: transform on several textures incl. texCoord override, clearcoat and unlit; README: rows of identical checkmarks.

Defects:

- **C22-R2** [open] Refused: material 'BaseColorTest1Mat' has a non-uniform KHR_texture_transform (a transform on one texture slot only). Root-cause hypothesis: The loader bakes a transform into the mesh UV0 and therefore accepts only a UNIFORM one (analyzeTextureTransforms: every textured slot of a material carries the same transform); BaseColorTest1Mat transforms the base colour only (offset, rotation 90 degrees, scale 0.35, texCoord 1) while its other slots are untransformed or on UV0. Supporting it needs per-slot transforms in MaterialDescription, the GPU material record and the shader texture fetches (and UV1 for texCoord overrides): an F15.1 feature, not a corpus-run fix. The model also uses KHR_materials_unlit, which would render lit.

### AnisotropyBarnLamp

Shows: anisotropy + clearcoat on one material, plus transmission/volume and emissive strength.

Engine support report: anisotropy: approximated (environment_split_sum_at_effective_roughness;secondary_rt_hits_ignore_anisotropy;no_tangents_means_isotropic) x1; clearcoat: approximated (secondary_rt_hits_ignore_coat;di_emitters_uncoated) x1; emissive_strength: implemented x1; transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x1; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x1.

Framing: 3 instances, bounds [-0.1046, -0.1983, -0.0] to [0.0862, 0.0569, 0.2265], camera distance 0.4027.

Renders correctly:

- Brushed-copper anisotropy + clear coat on one material (anisotropy texture + coat normal texture) loads; the anisotropic streak follows the spun metal, the bulb material (transmission + volume) is visible.

Named approximations:

- Anisotropy and clearcoat are combined in the resolve (ticket 19); no HDR environment, so the long environment streaks of the Khronos image are absent.

### ToyCar

Shows: clearcoat, sheen, transmission on separate materials, KHR_texture_transform.

Engine support report: clearcoat: approximated (secondary_rt_hits_ignore_coat;di_emitters_uncoated) x1; sheen: approximated (lobe_not_sampled_environment_term_in_composition;sun_disk_point_evaluated;secondary_rt_hits_ignore_sheen) x1; texture_transform: unsupported x1; transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x1.

Framing: 3 instances, bounds [-0.035, -0.0162, -0.037] to [0.0384, 0.0126, 0.037], camera distance 0.0953.

Renders correctly:

- Clear coat on the paint (texture + coat), sheen on the velvet cloth, transmission on the glass; the cloth shows its sheen rim and the car body its flame decals.

Named approximations:

- Sheen and coat on separate materials as authored; secondary RT hits ignore both.

### SpecularSilkPouf

Shows: specular + sheen on one material (required).

Engine support report: sheen: approximated (lobe_not_sampled_environment_term_in_composition;sun_disk_point_evaluated;secondary_rt_hits_ignore_sheen) x1; specular_ior: approximated (secondary_rt_hits_use_base_reduction) x1.

Framing: 1 instances, bounds [-0.305, 0.0, -0.305] to [0.305, 0.1978, 0.305], camera distance 0.7639.

Renders correctly:

- Specular + sheen on one required material: the folds read as satin with the sun highlight and a soft sheen rim.

Named approximations:

- Sheen approximations as in SheenTestGrid.

### GlamVelvetSofa

Shows: specular + sheen, KHR_texture_transform required, KHR_materials_variants / lights_punctual used.

Defects:

- **C22-R2b** [open] Refused: material 'GlamVelvetSofa_fabric_champagne' has a non-uniform KHR_texture_transform (scale 5x5 on the normal texture only). Root-cause hypothesis: Same root cause as C22-R2: the loader bakes only a uniform transform into UV0, the fabric normal texture alone is scaled 5x (the base colour texture is not).

### ABeautifulGame

Shows: transmission + volume on a showcase scene (Track Studio case, ticket 20).

Engine support report: transmission: approximated (rt_refraction_one_fresnel_chosen_sample;rt_off_probes_environment_along_refracted_ray;factor_at_visible_surface_only;at_most_4_interface_events;schlick_fresnel;caustics_excluded;secondary_gi_and_probe_hits_ignore_transmission;rt_shadow_rays_attenuate_through_glass_straight_line;csm_shadows_opaque) x2; volume: approximated (beer_lambert_along_real_inside_segment_with_rt;rt_off_uses_thickness_hint;thickness_texture_ignored_with_rt;no_scattering) x2.

Framing: 49 instances, bounds [-0.3527, -0.0, -0.3527] to [0.3527, 0.169, 0.3527], camera distance 0.8571.

Renders correctly:

- Not a Tier A model (Track Studio case, ticket 20): the board and the pieces (transmission + volume materials) load and render under the rig.

## Defect index

| Id | Model | Status | Summary |
|---|---|---|---|
| C22-D3 | ClearCoatCarPaint | open | Isolated dark pixels on the outermost pixel row of the sphere silhouette (also on the SpecularTest, IORTestGrid and Sheen grids). |
| C22-D5 | AnisotropyStrengthTest | open | Near-mirror anisotropic spheres (strength >= 0.5, roughness <= 0.17) are speckled in the single settled frame; the neighbouring-sphere reflections at strength 0 show black blotches. |
| C22-D4 | TransmissionRoughnessTest | open | Near-black grainy squares where a glass tile overlaps the opaque label planes (rows 2 to 5 of the smooth column); the Khronos image shows the label through the glass. |
| C22-D7 | IORTestGrid | open | White transmissive spheres at IOR >= 1.76 are visibly grainy in the single settled frame; the Khronos image is smooth. |
| C22-A1 | IridescenceMetallicSpheres | named-approximation | Dense metal lattices show black holes inside each sphere: a sphere sees its neighbours as dark. |
| C22-R1 | DiffuseTransmissionTest | by-design | Refused: the model REQUIRES KHR_materials_unlit and KHR_lights_punctual. |
| C22-R2 | TextureTransformMultiTest | open | Refused: material 'BaseColorTest1Mat' has a non-uniform KHR_texture_transform (a transform on one texture slot only). |
| C22-R2b | GlamVelvetSofa | open | Refused: material 'GlamVelvetSofa_fabric_champagne' has a non-uniform KHR_texture_transform (scale 5x5 on the normal texture only). |

