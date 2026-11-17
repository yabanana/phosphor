# KHR_materials_* research for Phosphor F15 (primary sources only)

Research date: 2026-11-13. All claims were read from raw files at the pinned commits below; nothing is from blogs.

## Pinned sources

| Repo | Ref | SHA | Commit date |
|---|---|---|---|
| KhronosGroup/glTF (specs, READMEs, schemas) | `main` | `f93f8afd4dff9252a4191e3e0538d4efe0e181b4` | 2026-11-12 |
| KhronosGroup/glTF-Sample-Viewer (UI shell only; shaders moved out) | `main` | `7306f9ce1b43e784ec4cd36215d55ee1f7db2fc0` | 2026-11-12 |
| KhronosGroup/glTF-Sample-Renderer (the actual shaders; Viewer submodule `glTF-Sample-Renderer`) | `main` | `cc3d5ff52217a63161c5c876677292546c01564a` | 2026-11-12 |
| syoyo/tinygltf, tag `v2.9.7` (what Phosphor pins in `cmake/Dependencies.cmake`) | tag | `488a70a3df62a4df1a736e9e56fb8836580c4888` | 2025-11-02 |
| syoyo/tinygltf, branch `release` = tag `v3.1.0` | tag | `97789b346ed792f413b9520c80bb479f3b63368f` | 2026-11-07 |

URL pattern for every glTF-repo file below: `https://github.com/KhronosGroup/glTF/blob/f93f8afd4dff9252a4191e3e0538d4efe0e181b4/extensions/2.0/Khronos/<EXT>/README.md` (schema: `.../<EXT>/schema/material.<EXT>.schema.json`). Core spec: `https://github.com/KhronosGroup/glTF/blob/f93f8afd4dff9252a4191e3e0538d4efe0e181b4/specification/2.0/Specification.adoc` (Appendix B at lines 3878-4255).

## 0. Status summary (read from each README "## Status" line at f93f8af)

| Extension | Status string in README | Textures (channel, colour space) |
|---|---|---|
| KHR_materials_clearcoat | "Complete, Ratified by the Khronos Group" | clearcoatTexture.r, clearcoatRoughnessTexture.g, clearcoatNormalTexture (normal map); all linear |
| KHR_materials_sheen | Complete, Ratified | sheenColorTexture.rgb (sRGB), sheenRoughnessTexture.a (linear) |
| KHR_materials_anisotropy | Complete, Ratified | anisotropyTexture rg = direction, b = strength; linear |
| KHR_materials_transmission | Complete, Ratified | transmissionTexture.r; linear |
| KHR_materials_volume | Complete, Ratified | thicknessTexture.g; linear |
| KHR_materials_ior | Complete, Ratified | none |
| KHR_materials_specular | Complete, Ratified | specularTexture.a (linear), specularColorTexture.rgb (sRGB) |
| KHR_materials_iridescence | Complete, Ratified | iridescenceTexture.r, iridescenceThicknessTexture.g; linear |
| KHR_materials_diffuse_transmission | **"Release Candidate"** (not ratified) | diffuseTransmissionTexture.a (linear), diffuseTransmissionColorTexture.rgb (sRGB) |
| KHR_materials_emissive_strength | Complete, Ratified | none |
| KHR_materials_dispersion | Complete, Ratified | none |
| KHR_texture_transform | Complete, Ratified | n/a (textureInfo extension) |
| KHR_materials_retroreflection (extra) | "Release Candidate (RC)" | not read in detail |
| KHR_materials_volume_scatter | **Draft, never merged** (PR #2453, open, head `134d07661f461ee26d02e0e11b7ebe3e26f28330`); superseded by `KHR_materials_scatter` | see section 10 |
| KHR_materials_scatter | **Draft, never merged** (PR #2579, open, head `a6cd5b46fb4bd2c9ef83f0bb4a123f45a97a6d9f`, created 2026-05-11, updated 2026-10-20) | see section 10 |

Not verified: ratification dates; the Khronos registry listing (only the repo README status lines were read). There is no `KHR_materials_volume_scatter` or `KHR_materials_scatter` directory on `main` (directory listing of `extensions/2.0/Khronos` at f93f8af).

All of these extensions list the same Exclusions: must not be combined with `KHR_materials_pbrSpecularGlossiness` or `KHR_materials_unlit` (emissive_strength excludes only unlit). `KHR_materials_pbrSpecularGlossiness` lives in `Archived/` and is superseded by KHR_materials_specular (+ ior = 0).

Other open Khronos material drafts seen (only PR titles/first paragraphs read, NOT the specs): #2534 `KHR_materials_coat` ("Advanced Coat", intended to replace KHR_materials_clearcoat plus clearcoat_ior/color/anisotropy/darkening sub-proposals, OpenPBR coat), #2538 `KHR_materials_fuzz` (OpenPBR fuzz; sits on top of the coat, unlike sheen which sits below), #2481 `KHR_materials_diffuse_roughness`, #2530/#2528/#2520/#2518 (clearcoat ior/anisotropy/color/darkening). Risk: the ratified clearcoat/sheen schemas may be superseded in glTF 2.1 timeframe; Phosphor should keep the schema extension-agnostic internally.

## 1. Core BSDF (glTF 2.0 Appendix B, Specification.adoc @ f93f8af lines 3878-4255)

Material = linear blend of dielectric and metal:

```
material = mix(dielectric_brdf, metal_brdf, metallic)
metal_brdf      = conductor_fresnel(f0 = baseColor, bsdf = specular_brdf(α = roughness^2))
dielectric_brdf = fresnel_mix(ior = 1.5, base = diffuse_brdf(color = baseColor), layer = specular_brdf(α = roughness^2))
```

Specular microfacet (Appendix B, "Specular BRDF"): GGX/Trowbridge-Reitz D, height-correlated Smith G:

```
D = α² χ+(N·H) / ( π ((N·H)² (α²-1) + 1)² )
Vis = χ+(H·L) χ+(H·V) / ( 2 ( |N·V| sqrt(α² + (1-α²)(N·L)²) + |N·L| sqrt(α² + (1-α²)(N·V)²) ) )
specular_brdf(α) = Vis * D          // = G*D / (4|N·L||N·V|)
diffuse_brdf(color) = (1/π) * color  // Lambert
```

Fresnel (Schlick) uses the half vector:

```
F = f0 + (1 - f0) (1 - |V·H|)^5
fresnel_mix(ior, base, layer):  f0 = ((1-ior)/(1+ior))^2 ; fr = f0 + (1-f0)(1-abs(VdotH))^5 ; return mix(base, layer, fr)
conductor_fresnel(f0, bsdf) = bsdf * (f0 + (1 - f0) * (1 - abs(VdotH))^5)
```

Normative statements: BRDFs MUST be positive, reciprocal and energy conserving for a "physically accurate light simulation"; "any implementation that adheres to the rules for mixing BRDFs is conformant"; implementations MAY vary BRDF (Appendix B "Complete Model"). Roughness note (added in current main): alpha = 0 must be clamped or handled as a mirror. The spec explicitly says the `fresnel_mix` diffuse/specular coupling "is not very accurate and breaks ... energy conservation" and suggests Kulla-Conty 2017 multiple scattering compensation (informative).

## 2. KHR_materials_clearcoat

README: `.../KHR_materials_clearcoat/README.md`. Schema ranges from `schema/material.KHR_materials_clearcoat.schema.json`.

| Parameter | Type | Default | Range (schema) | Texture |
|---|---|---|---|---|
| clearcoatFactor | number | 0.0 | [0, 1] | clearcoatTexture `.r`, linear |
| clearcoatRoughnessFactor | number | 0.0 | [0, 1] | clearcoatRoughnessTexture `.g`, linear |
| clearcoatNormalTexture | normalTextureInfo (has `scale`) | none | n/a | clearcoat normal map; if absent NO normal mapping on the coat even if base has one |

```
clearcoat = clearcoatFactor * clearcoatTexture.r
clearcoatRoughness = clearcoatRoughnessFactor * clearcoatRoughnessTexture.g
```
"If `clearcoatFactor` is zero, the whole clearcoat layer is disabled." Missing texture = 1.0. A primitive with a clearcoat normal texture MUST have a defined tangent space (NORMAL+TANGENT, or base normalTexture). If both normalTexture and clearcoatNormalTexture exist they SHOULD share texCoord.

BSDF (normative layering, then non-normative simplification):

```
clearcoat_brdf = specular_brdf(normal = clearcoatNormal, α = clearcoatRoughness^2)
coated_material = fresnel_coat(normal = clearcoatNormal, ior = 1.5, weight = clearcoat, base = material, layer = clearcoat_brdf)
function fresnel_coat(normal, ior, weight, base, layer) {
  f0 = ((1-ior)/(1+ior))^2
  fr = f0 + (1 - f0)*(1 - abs(dot(V, normal)))^5      // NOTE: uses V·Nc, not V·H
  return mix(base, layer, weight * fr)
}
clearcoat_fresnel = 0.04 + (1 - 0.04) * (1 - abs(VdotNc))^5
coated_material   = mix(material, clearcoat_brdf, clearcoat * clearcoat_fresnel)
coated_emission   = emission * (1 - clearcoat * clearcoat_fresnel)     // coat is ABOVE emission
```
Rationale quoted: "we compute the microfacet Fresnel term with NdotV instead of VdotH. That means that we ignore the orientation of the microsurface" (valid because coat roughness is low). Listed ignored effects: infinitely thin coat, no refraction, base IOR and coat IOR independent, no scattering between layers, no diffraction. "The material" in the base includes emission and all other extensions (sheen, iridescence, transmission...). Clearcoat IOR is fixed at 1.5 (F0 = 0.04) by the ratified spec.

## 3. KHR_materials_sheen

README: `.../KHR_materials_sheen/README.md`.

| Parameter | Type | Default | Range | Texture |
|---|---|---|---|---|
| sheenColorFactor | number[3] | [0,0,0] | each [0,1] (schema) | sheenColorTexture `.rgb`, **sRGB-encoded** ("sheenColor = sheenColorFactor * sampleLinear(sheenColorTexture).rgb") |
| sheenRoughnessFactor | number | 0.0 | [0,1] (schema) | sheenRoughnessTexture `.a`, linear ("sheenRoughness = sheenRoughnessFactor * sample(sheenRoughnessTexture).a") |

"If `sheenColorFactor` is zero, the whole sheen layer is disabled." Sheen roughness is independent of material roughness. Layering statement (normative prose): "The sheen BRDF is layered on top of the glTF 2.0 Metallic-Roughness material. If clearcoat is active at the same time, clearcoat is layered on top of sheen."

Non-normative implementation notes (Estevez-Kulla "Charlie"):

```
SheenBRDF = nu_s * D_s    // nu_s = G_s / (4|N·L||N·V|)
alpha_g = sheenRoughness * sheenRoughness
inv_r = 1 / alpha_g
sin2h = 1 - NdotH*NdotH
sheen_distribution = (2 + inv_r) * pow(sin2h, inv_r * 0.5) / (2 * PI)

float l(float x, float alpha_g) {
  float one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);
  float a = mix(21.5473, 25.3245, one_minus_alpha_sq);
  float b = mix(3.82987, 3.32435, one_minus_alpha_sq);
  float c = mix(0.19823, 0.16801, one_minus_alpha_sq);
  float d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
  float e = mix(-4.32054, -4.85967, one_minus_alpha_sq);
  return a / (1.0 + b * pow(x, c)) + d * x + e; }
float lambda_sheen(float cos_theta, float alpha_g) {
  return abs(cos_theta) < 0.5 ? exp(l(cos_theta, alpha_g)) : exp(2.0 * l(0.5, alpha_g) - l(1.0 - cos_theta, alpha_g)); }
sheen_visibility = 1.0 / ((1.0 + lambda_sheen(NdotV, alpha_g) + lambda_sheen(NdotL, alpha_g)) * (4.0 * NdotV * NdotL));
// cheaper alternative (Ashikhmin-Premoze), NOT energy conserving with the albedo-scaling below:
sheen_visibility = 1 / (4 * (NdotL + NdotV - NdotL * NdotV))
```
Albedo scaling (Conty-Kulla), "to avoid energy gain":

```
sheen_albedo_scaling = min(1.0 - max3(sheenColor) * E(VdotN), 1.0 - max3(sheenColor) * E(LdotN))
sheen_material = sheenColor * sheen_brdf + material * sheen_albedo_scaling     // material = mix(dielectric_brdf, metal_brdf, metallic)
// cheap form, usable for direct and indirect light:
sheen_albedo_scaling = 1.0 - max3(sheenColor) * E(VdotN)
specular_direct *= s; diffuse_direct *= s; environment_irradiance_indirect *= s; specular_environment_reflectance_indirect *= s
```
E(x) comes from a LUT indexed `(VdotN, sheenRoughness)`: for Charlie visibility, "section 6.2.3 of Enterprise PBR Shading Model"; for Ashikhmin, Filament cmgen `--ibl-dfg --ibl-dfg-cloth` (blue channel). Phosphor must bake its own E LUT (the spec ships none; the Sample Renderer ships `assets/images/lut_sheen_E.png`, see section 12).

## 4. KHR_materials_anisotropy

README: `.../KHR_materials_anisotropy/README.md`.

| Parameter | Type | Default | Range | Notes |
|---|---|---|---|---|
| anisotropyStrength | number | 0.0 | [0,1] (schema) | multiplied by texture blue |
| anisotropyRotation | number | 0.0 | unbounded (radians) | "measured in radians counter-clockwise from the tangent", in tangent/bitangent space |
| anisotropyTexture | textureInfo | none | | RG = direction in [-1,1] (R,G in [0,1] -> X,Y in [-1,1]); B = strength [0,1]; **linear**. Default texel when absent = (1.0, 0.5, 1.0) = +X, full strength |

Direction semantic: the XY vector is "the direction in which highlights will be stretched"; micro-grooves run perpendicular. Direction is rotated by `anisotropyRotation` after dequantisation. Default direction = mesh tangent. A primitive MUST have a defined tangent space (NORMAL+TANGENT or base normalTexture); SHOULD supply TANGENT; normalTexture and anisotropyTexture SHOULD share texCoord.

Normative roughness rule:
```
directionAlphaRoughness = mix(materialAlphaRoughness, 1.0, strength^2)     // along anisotropy direction (tangent by default)
materialAlphaRoughness  = materialRoughness^2
perpendicular direction (bitangent by default) alpha = materialAlphaRoughness
D(h) = 1 / ( π αt αb ( (h·t)²/αt² + (h·b)²/αb² + (h·n)² )² )       // Burley anisotropic GGX
```
Non-normative reference (GLSL in README, "Individual lights"):
```glsl
float anisotropy = u_AnisotropyStrength;  vec2 direction = u_AnisotropyRotation;   // = [cos(rot), sin(rot)]
#if HAS_ANISOTROPY_MAP
  vec3 tex = texture(u_AnisotropyTextureSampler, uv).rgb;
  direction = tex.rg * 2.0 - vec2(1.0);
  direction = mat2(rot.x, rot.y, -rot.y, rot.x) * normalize(direction);
  anisotropy *= tex.b;
#endif
vec3 anisotropicT = normalize(TBN * vec3(direction, 0.0));
vec3 anisotropicB = normalize(cross(normal_geometric, anisotropicT));
float at = mix(alphaRoughness, 1.0, anisotropy * anisotropy);  float ab = alphaRoughness;
float D_GGX_anisotropic(NdotH, TdotH, BdotH, at, ab) { a2 = at*ab; f = vec3(ab*TdotH, at*BdotH, a2*NdotH); w2 = a2/dot(f,f); return a2*w2*w2/M_PI; }
float V_GGX_anisotropic(NdotL, NdotV, BdotV, TdotV, TdotL, BdotL, at, ab) {
  GGXV = NdotL * length(vec3(at*TdotV, ab*BdotV, NdotV));
  GGXL = NdotV * length(vec3(at*TdotL, ab*BdotL, NdotL));
  return clamp(0.5/(GGXV+GGXL), 0.0, 1.0); }
f_specular = F_Schlick(f0,f90,VdotH) * V * D        // BRDF_specularAnisotropicGGX
```
IBL guidance (informative): sample the PMREM with material roughness (the min of the two), bent normal:
```glsl
vec3 bentNormal = normalize(cross(cross(anisotropicB, viewDir), anisotropicB));
float a = pow2(pow2(1.0 - anisotropy * (1.0 - roughness)));
bentNormal = normalize(mix(bentNormal, normal, a));  vec3 reflectVec = reflect(-viewDir, bentNormal);
```
Note: the README's normative text says "alpha roughness" for the stretched direction, the sample code applies the same mix on `alphaRoughness` (consistent). Note the Sample Renderer's `getIBLRadianceAnisotropy` uses `bendFactor^4` with the same shape (section 12).

## 5. KHR_materials_transmission

README: `.../KHR_materials_transmission/README.md`.

| Parameter | Type | Default | Range | Texture |
|---|---|---|---|---|
| transmissionFactor | number | 0.0 | [0,1] | transmissionTexture `.r`, linear, multiplied |

Meaning: fraction of light that penetrates the surface (i.e. is not specularly reflected) and is transmitted instead of diffusely re-emitted. Thin-walled by default: "this transmissive surface is really two surfaces back-to-back", no macroscopic refraction, microfacet-level blur only. baseColor tints the transmitted light (absorption constant = `1 - baseColor`); metals absorb all refracted light ("for a material with metallicFactor=1.0 the value of transmissionFactor doesn't matter"). `alphaMode` is for coverage, not transmission: leave OPAQUE; alpha coverage and transmission can combine. The extension "does not dictate rendering algorithms"; correct ordering "is not an absolute requirement"; minimum expectation = show opaque objects through the surface (IBL-only transmission "will often fall short"); fallback suggestion: alpha blending with 0 < alpha < 1.

Normative material structure:
```
dielectric_brdf = fresnel_mix(ior = 1.5,
    base  = mix(diffuse_brdf(baseColor), specular_btdf(α = roughness^2) * baseColor, transmission),
    layer = specular_brdf(α = roughness^2))
```
"Optical transparency does not require any changes whatsoever to the specular term ... a modification of the base layer only."

Non-normative microfacet BTDF (thin-walled, no refraction), half vector `H_T = normalize(V - 2 (N·L) N + L)` (i.e. light mirrored to the other side):
```
D_T  = α² χ+(N·H_T) / ( π ((N·H_T)²(α²-1)+1)² )
nu_T = χ+(H_T·L / N·L) / ( |N·L| + sqrt(α² + (1-α²)(N·L)²) )  *  χ+(H_T·V / N·V) / ( |N·V| + sqrt(α² + (1-α²)(N·V)²) )
specular_btdf(α) = nu_T * D_T      // separable Smith G_T, NOT height-correlated
```
Volume case (README of KHR_materials_volume, "Implementation"): refractive half vector `H_TR = -normalize(η_i V + η_o L)`;
```
MicrofacetBTDF = (|H_TR·L| |H_TR·V|) / (|N·L| |N·V|) * ( η_o² G_T D_TR ) / ( η_i (H_TR·V) + η_o (H_TR·L) )²
F_TR+ (η_o >= η_i)      = f0 + (1-f0)(1 - |V·H_TR|)^5
F_TR- (η_o <  η_i, no TIR): cosθo = sqrt(1 - sin²θo), sin²θo = (η_i/η_o)² (1 - (V·H_TR)²);  F = f0 + (1-f0)(1-|cosθo|)^5
F_TIR = 1
```
Interaction with ior: the `ior` sets f0 of the Fresnel that splits reflection vs transmission (ior README). Interaction with specular: `dielectric_f0/f90` (section 9) also modulate that split.

## 6. KHR_materials_volume

README: `.../KHR_materials_volume/README.md`. Requires a transmission-like extension to have effect ("Dependencies": must be combined with e.g. KHR_materials_transmission).

| Parameter | Type | Default | Range (README / schema) | Units | Texture |
|---|---|---|---|---|---|
| thicknessFactor | number | 0 | [0, +inf) | mesh local coordinate space (node transform scales it) | thicknessTexture `.g`, linear, [0,1], multiplied |
| attenuationDistance | number | +Infinity (omitted) | (0, +inf), exclusiveMinimum 0 | world space length; "average distance that light travels in the medium before interacting with a particle" | none |
| attenuationColor | number[3] | [1,1,1] | each [0,1] | "color that white light turns into due to absorption when reaching the attenuation distance" (linear) | none |

`thicknessFactor == 0` => thin-walled (volume absent); otherwise the mesh is a closed manifold volume boundary and `doubleSided` has no effect. Medium is homogeneous, not texturable. Ray tracers "should ignore the thickness texture and use the actual, ray-traced distance" but must still check thicknessFactor to choose thin vs volume. Scattering is explicitly out of scope ("σ_s = 0, σ_t = σ_a").

Beer-Lambert (normative definitions, quoted):
```
σ_t = -log(c) / d                 (c = attenuationColor, d = attenuationDistance)
T(x) = e^(-σ_t x)   =>   T(x) = c^(x / d)
```
Refraction uses IOR from KHR_materials_ior (default 1.5). Interaction with specular: "specular color and specular modify F0 and F90 ... but ... In case of total internal reflection, F90 is not scaled by the specular parameter".

## 7. KHR_materials_ior

README: `.../KHR_materials_ior/README.md`; schema: `"ior": {"default": 1.5, "oneOf": [{"minimum":0,"maximum":0},{"minimum":1}]}`. So: ior is 0 (special) or >= 1; default 1.5; typical 1..2, rare >2. Table in README: Water 1.33, Window glass 1.52, Sapphire 1.76, Diamond 2.42.

```
dielectric_brdf = fresnel_mix(ior = ior, base = diffuse_brdf(baseColor), layer = specular_brdf(α = roughness^2))
dielectric_f0 = ((ior - 1)/(ior + 1))^2        // 0.04 at 1.5
```
**ior = 0 is a permanent special mode** (normative): "The effective IOR becomes positive infinity and the Fresnel term MUST evaluate to 1.0 independently of the view or light direction"; all features treat IOR as +inf (e.g. dispersion has no effect); cannot be toggled dynamically (KHR_animation_pointer / interactivity updates MUST be ignored); values <1 other than 0 must not be animated. This is the spec's specular-glossiness compatibility route (see section 9).

Note: ior does NOT change the clearcoat IOR in the ratified spec (clearcoat fixed 1.5); the Sample Renderer deviates (section 12).

## 8. KHR_materials_iridescence

README: `.../KHR_materials_iridescence/README.md`.

| Parameter | Type | Default | Range (schema) | Texture |
|---|---|---|---|---|
| iridescenceFactor | number | 0.0 | [0,1] | iridescenceTexture `.r`, linear |
| iridescenceIor | number | 1.3 | [1, +inf) | thin-film IOR |
| iridescenceThicknessMinimum | number | 100.0 | [0, +inf), nm | maps thickness texture value 0 |
| iridescenceThicknessMaximum | number | 400.0 | [0, +inf), nm | maps value 1; if no thickness texture, thickness = Maximum |
| iridescenceThicknessTexture | textureInfo | none | | `.g`, linear |

```
iridescence = iridescenceFactor * iridescenceTexture.r
thickness   = mix(iridescenceThicknessMinimum, iridescenceThicknessMaximum, iridescenceThicknessTexture.g)
```
Minimum MAY exceed Maximum. Factor 0 => no effect. "All textures in this extension use a single channel in linear space."

Normative structure (replaces Appendix B metal and dielectric):
```
metal_brdf = iridescent_conductor_layer(iridescence_strength = iridescence, iridescence_thickness, iridescence_ior = iridescenceIor,
                outside_ior = 1.0, base_f0 = baseColor, specular_brdf = specular_brdf(α = roughness^2))
dielectric_brdf = iridescent_dielectric_layer(..., outside_ior = 1.0, base_ior = 1.5, base = diffuse_brdf(baseColor), specular_brdf = ...)
//  transmissive base: diffuse_brdf replaced by mix(diffuse_brdf, specular_btdf, transmission); base_ior from KHR_materials_ior;
//  with KHR_materials_specular: base_f0_color = specularColor.rgb, specular_weight = specular
```
Clearcoat: "an additional interface between the clearcoat and thin-film layer would be created. Due to simplicity and artistical friendlyness, this interface is ignored."

Implementation (non-normative):
```
iridescent_conductor_layer = mix( conductor_fresnel(base_f0, specular_brdf),
                                  specular_brdf * iridescent_fresnel(outside_ior, iridescence_ior, base_f0, thickness),
                                  iridescence_strength)
iridescent_dielectric_layer: base_f0 = ((1-base_ior)/(1+base_ior))^2
   iridescent_f = iridescent_fresnel(outside_ior, iridescence_ior, base_f0, thickness)
   return mix( fresnel_mix(base_ior, base, specular_brdf), rgb_mix(base, specular_brdf, iridescent_f), iridescence_strength)
rgb_mix(base, spec, a) = (1 - max(a.r,a.g,a.b)) * base + a * spec
// with specular ext: base_f0 = min(((1-base_ior)/(1+base_ior))^2 * base_f0_color, 1); iridescent_fresnel(..., specular_weight * base_f0, ...);
//   return rgb_mix(base, spec, mix(specular_weight * fr, iridescent_f, iridescence_strength))
```
Thin-film Fresnel (Belcour-Barla 2017 analytic approximation, GLSL in README, section "Iridescence Fresnel"):
```glsl
R0 = IorToFresnel0(iridescenceIor, outsideIor);  R12 = F_Schlick(R0, cosTheta1);  R21 = R12;  T121 = 1.0 - R12;
baseIor = Fresnel0ToIor(baseF0 + 0.0001);        // (1+sqrt(F0))/(1-sqrt(F0)) per channel; used for metals too, extinction assumed 0
R1 = IorToFresnel0(baseIor, iridescenceIor);     R23 = F_Schlick(R1, cosTheta2);
sinTheta2Sq = (outsideIor/iridescenceIor)^2 * (1 - cosTheta1^2); if (1 - sinTheta2Sq < 0) return vec3(1.0);   // TIR
OPD = 2.0 * iridescenceIor * iridescenceThickness * cosTheta2;
phi12 = (iridescenceIor < outsideIor) ? PI : 0;  phi21 = PI - phi12;  phi23[c] = (baseIor[c] < iridescenceIor) ? PI : 0;  phi = phi21 + phi23;
R123 = clamp(R12*R23, 1e-5, 0.9999);  r123 = sqrt(R123);  Rs = sq(T121)*R23/(1 - R123);
C0 = R12 + Rs;  I = C0;  Cm = Rs - T121;
for m in 1..2: Cm *= r123; Sm = 2.0 * evalSensitivity(m*OPD, m*phi); I += Cm*Sm;
F_iridescence = max(I, 0)
evalSensitivity(OPD, shift): phase = 2π OPD 1e-9; val=(5.4856e-13,4.4201e-13,5.2481e-13); pos=(1.6810e+06,1.7953e+06,2.2084e+06); var=(4.3278e+09,9.3046e+09,6.6121e+09);
   xyz = val*sqrt(2π var)*cos(pos*phase+shift)*exp(-sq(phase)*var); xyz.x += 9.7470e-14*sqrt(2π*4.5282e+09)*cos(2.2399e+06*phase+shift[0])*exp(-4.5282e+09*sq(phase)); xyz /= 1.0685e-7; return XYZ_TO_REC709*xyz
XYZ_TO_REC709 (column-major as written) = mat3( 3.2404542,-0.9692660, 0.0556434,  -1.5371385, 1.8760108,-0.2040259,  -0.4985314, 0.0415560, 1.0572252 )
```
Note: the README's `cosTheta2` for R23 is Snell-refracted into the film with `outsideIor/iridescenceIor`; the Sample Renderer's code differs only by clamping `baseF0` to `[0, 0.9999]` instead of `+0.0001`.

## 9. KHR_materials_specular (+ ior interplay)

README: `.../KHR_materials_specular/README.md`; "Supersedes KHR_materials_pbrSpecularGlossiness".

| Parameter | Type | Default | Range (schema) | Texture |
|---|---|---|---|---|
| specularFactor | number | 1.0 | [0,1] | specularTexture `.a`, linear |
| specularColorFactor | number[3] | [1,1,1] | each >= 0 (may exceed 1) | specularColorTexture `.rgb`, **sRGB** |

```
specularColor = specularColorFactor * sampleLinear(specularColorTexture).rgb
specular      = specularFactor * sample(specularTexture).a
dielectric_brdf = fresnel_mix(f0_color = specularColor.rgb, ior = 1.5, weight = specular, base = diffuse_brdf(baseColor), layer = specular_brdf(α = roughness^2))
function fresnel_mix(f0_color, ior, weight, base, layer) {
  f0 = ((1-ior)/(1+ior))^2 * f0_color ;  f0 = min(f0, float3(1.0))
  fr = f0 + (1 - f0)*(1 - abs(VdotH))^5
  return (1 - weight * max_value(fr)) * base + weight * fr * layer }
// flattened form
dielectric_f0  = min(0.04 * specularColor, float3(1.0)) * specular        // clamp BEFORE multiplying by specular
dielectric_f90 = specular                                                   // f90 is NOT 1 any more
dielectric_fresnel = mix(dielectric_f0, dielectric_f90, fresnel_w)
dielectric_brdf = dielectric_fresnel * specular_brdf + (1 - max_value(dielectric_fresnel)) * diffuse_brdf
// with KHR_materials_ior:
dielectric_f0 = min(((ior - outside_ior)/(ior + outside_ior))^2 * specularColor, float3(1.0)) * specular     // outside_ior = 1.0
```
The metal BRDF is not affected. `max(r,g,b)` is used to avoid inverse colours and gain energy ("To avoid inverse colors and ensure energy conservation"). Conversion helpers in README: `dielectric_f0 = 0.08 * reflectance` => `specularColorFactor = 2*reflectance`; spec-gloss => `specularColorFactor = specularFactor` and `KHR_materials_ior.ior = 0`. With transmission+volume, specular does NOT change the refraction angle (only the reflect/transmit ratio).

## 10. KHR_materials_diffuse_transmission (Release Candidate) and subsurface proposals

### 10.1 KHR_materials_diffuse_transmission (RC)
README: `.../KHR_materials_diffuse_transmission/README.md`. Last README commit: `603619ef9d242d10d7f3c83d0b768e3487f04aa3` (2025-10-13, "Fix schema and cleanup language (#2537)"). Status: **Release Candidate**.

| Parameter | Type | Default | Range (schema) | Texture |
|---|---|---|---|---|
| diffuseTransmissionFactor | number | 0 | [0,1] | diffuseTransmissionTexture `.a`, linear |
| diffuseTransmissionColorFactor | number[3] | [1,1,1] | each [0,1] | diffuseTransmissionColorTexture `.rgb`, sRGB |

Definition: fraction of non-specularly reflected light that is diffusely transmitted; attenuation of an infinitely thin surface is constant = `1 - diffuseTransmissionColor`. README inconsistency: the text says the colour texture values "are multiplied by the diffuseTransmissionFactor to determine the final diffuse transmission color" while the formula block is `diffuseTransmissionColor = diffuseTransmissionColorFactor * diffuseTransmissionColorTexture.rgb` (no factor); the normative structure below only multiplies by `diffuseTransmission` in the `mix`, so the formula is the self-consistent reading (flag as spec ambiguity).

Normative (section "Material Structure Updates", marked "This section is normative"):
```
dielectric_brdf = fresnel_mix(ior = 1.5,
   base  = mix(diffuse_brdf(color = baseColor), diffuse_btdf(color = diffuseTransmissionColor), diffuseTransmission),
   layer = specular_brdf(α = roughness^2))
```
"takes away energy from the diffuse reflection BSDF and passes it to the diffuse transmission BSDF. The specular reflection BSDF and Fresnel weighting are not affected."
Informative Lambert lobes: `diffuse_brdf(c) = (1/π) c` if V and L on the same hemisphere else 0; `diffuse_btdf(c) = (1/π) c` if on opposite hemispheres else 0.
Combos: with KHR_materials_transmission, "the transmission effect overrides the diffuse transmission effect":
```
base = mix(diffuse_bsdf, specular_btdf(α = roughness^2) * baseColor, transmission)
diffuse_bsdf = mix(diffuse_brdf(baseColor), diffuse_btdf(diffuseTransmissionColor), diffuseTransmission)
```
With KHR_materials_volume: diffuse BTDF is the transmission through the volume boundary; interior transport is handled solely by volume (attenuation only).

### 10.2 KHR_materials_volume_scatter (Draft, PR #2453)
`https://github.com/KhronosGroup/glTF/pull/2453` head `134d07661f461ee26d02e0e11b7ebe3e26f28330`, created 2024-10-31, last update 2026-02-09, open. README status "Draft", contributors "TODO". Must be combined with KHR_materials_volume. Properties: `multiscatterColorFactor` number[3] (default [0,0,0]), `multiscatterColorTexture` (RGB, sRGB), `scatterAnisotropy` (-1,1) default 0. Replaces the older `KHR_materials_subsurface` PR #1928 (closed 2024-11-21). Discussion in the PR (Dec 2024-Feb 2025) shows contention over naming it "subsurface scattering" vs volume scattering (surface BSSRDF in other models vs a volumetric medium in glTF).

### 10.3 KHR_materials_scatter (Draft, PR #2579) - the current Khronos SSS proposal
`https://github.com/KhronosGroup/glTF/pull/2579` ("Add KHR_materials_scatter (replaces #2453)"), head `a6cd5b46fb4bd2c9ef83f0bb4a123f45a97a6d9f`, branch `mbond/KHR_materials_scatter`, created 2026-05-11, updated 2026-10-20, open, not draft-flagged on GitHub, README Status "Draft". Authors: Mike Bond (Adobe), Tobias Haeussler and Bastian Sdorra (Dassault). PR body: replaces BOTH `KHR_materials_volume_scatter` and `KHR_materials_diffuse_transmission`; mirrors OpenPBR/Standard Surface where volumes and diffusely transmitting surfaces are not simultaneous. It has **no effect without KHR_materials_transmission**; optionally combined with KHR_materials_volume; **excludes KHR_materials_diffuse_transmission**.

| Parameter | Type | Default | Range | Texture |
|---|---|---|---|---|
| scatterStrengthFactor | number | 0 | [0,1] | scatterStrengthTexture `.a` |
| multiscatterColorFactor | number[3] | [1,1,1] | not constrained in schema | multiscatterColorTexture `.rgb`, sRGB |
| scatterAnisotropy | number | 0 | (-1, 1) exclusive | none |

Modes: thin-walled when volume absent or `thicknessFactor = 0`; volumetric when `thicknessFactor > 0`.
Thin-walled structure (normative section):
```
dielectric_brdf = fresnel_mix(ior = 1.5,
  base = mix( diffuse_brdf(baseColor),
              mix( specular_btdf(α = roughness^2) * baseColor, scatter_bsdf(color = multiscatterColor, g = scatterAnisotropy), scatterStrength),
              transmission),
  layer = specular_brdf(α = roughness^2))
scatter_bsdf(color, g): forward = (1+g)/2, backward = (1-g)/2 ;
   opposite hemispheres -> (1/π) * forward * color ;  same hemisphere -> (1/π) * backward * color
```
Volumetric: surface BSDF unchanged; adds scattering coefficient to the medium.
```
ρ_ms* = scatterStrength * multiscatterColor                (per channel)
s   = 4.09712 + 4.20863 ρ_ms* - sqrt(9.59217 + 41.6808 ρ_ms* + 17.7126 ρ_ms*²)     // clamp to [0,1]; if ρ_ms* == 0 then s = 1
ρ_ss = (1 - s²) / (1 - g s²)                                // single-scatter albedo (Kulla-Conty 2017 mapping)
σ_t = -log(c)/d (from KHR_materials_volume);  σ_s = σ_t ρ_ss;  σ_a = σ_t (1 - ρ_ss)
HG phase: p(v,l;g) = (1/4π) (1-g²) / (1 + g² + 2g(v·l))^(3/2)       // README's sign convention as written
```
README allows approximating volumetric mode with thin-walled behaviour for dense, short-distance scattering; ray tracers should use real path lengths. Caveat quoted: textured multiscatterColor in volumetric mode "violates reciprocity in path tracing". The scatterStrength/anisotropy formulas are draft and may change.

Not verified: whether other Khronos subsurface proposals exist outside the glTF repo (searched issues/PRs in KhronosGroup/glTF only for `volume_scatter`, `subsurface`, `sss`, `scatter`: results above).

## 11. KHR_materials_emissive_strength, KHR_materials_dispersion, KHR_texture_transform

### emissive_strength (README `.../KHR_materials_emissive_strength/README.md`)
`emissiveStrength` number, default 1.0, schema minimum 0.0. Unitless multiplier of core emissive; pseudocode `color += emissiveFactor.rgb * sRGB_to_Linear(emissiveTexture.rgb) * emissiveStrength;`. Core emissive factor is clamped [0,1] (the extension lifts the HDR limit). Does not alter the core physical-unit statement. Note for layering: clearcoat README says emissive is darkened by `(1 - clearcoat * clearcoat_fresnel)`.

### dispersion (README `.../KHR_materials_dispersion/README.md`)
`dispersion` number, default 0 (= off), schema minimum 0, "range between [0,1] is the range of realistic values". Stored as `20/Abbe`. Depends on KHR_materials_volume (it is a volume-transmission effect).
```
V_d = (n_d - 1)/(n_F - n_C);  λ_F=486.13nm, λ_d=587.56nm, λ_C=656.27nm
n(λ) = max(n_d + (n_d-1)/V_d * (523655/λ² - 1.5168), 1)       // Cauchy, 2 terms, λ in nm
halfSpread = (ior - 1.0) * 0.025 * dispersion;  iors = vec3(ior - halfSpread, ior, ior + halfSpread)   // R,G,B; clamp red >= 1
```
README table: Diamond/Water 55 (0.36), Crown glass 59 (0.33), Polycarbonate 32 (0.625), Rutile 9.8 (2.04). IOR = 0 mode disables dispersion (ior README).

### KHR_texture_transform (README `.../KHR_texture_transform/README.md`, schema `schema/textureInfo.KHR_texture_transform.schema.json`)
Extension on any textureInfo (including normalTextureInfo/occlusionTextureInfo, and the textureInfo inside each KHR_materials_* extension): `offset` [0,0], `rotation` 0 (radians, counter-clockwise on UVs), `scale` [1,1], `texCoord` (integer, minimum 0, overrides textureInfo.texCoord when supported).
```glsl
mat3 translation = mat3(1,0,0, 0,1,0, Offset.x, Offset.y, 1);
mat3 rotation = mat3( cos(R), sin(R), 0,  -sin(R), cos(R), 0,  0, 0, 1 );
mat3 scale = mat3(Scale.x,0,0, 0,Scale.y,0, 0,0,1);
uvTransformed = ( translation * rotation * scale * vec3(Uv.xy, 1) ).xy;      // order: scale, then rotate, then translate
```
Values unbounded; sane only with REPEAT wrap or result within [0,1]. UV origin is the top-left of the image. Exporter note: produce a fallback UV set for viewers lacking the extension. Consequence for Phosphor: with the F15 extension textures there can be up to ~9-10 independent transforms per material; decide whether to support per-texture transform or a shared one.

## 12. Layering and energy combination order

Derived from the READMEs above (no single normative "stack" document exists; each README states its own splice point):

1. **Base** (core): `mix(dielectric_brdf, metal_brdf, metallic)`.
2. **Dielectric internals** (all inside `fresnel_mix`'s `base`/`layer`): `base = mix(diffuse_bsdf, specular_btdf*baseColor, transmission)` where `diffuse_bsdf = mix(diffuse_brdf(baseColor), diffuse_btdf(diffuseTransmissionColor), diffuseTransmission)` (diffuse_transmission README, explicit that transmission overrides it); the Fresnel weights are modified by `KHR_materials_ior` (f0 from ior), `KHR_materials_specular` (f0 colour, f90 = weight, base scaled by `1 - weight*max(fr)`), and `KHR_materials_iridescence` (f replaced by thin-film Fresnel via `rgb_mix`, strength-mixed). Volume sits behind the transmitted lobe (attenuation + refraction); scatter (draft) either replaces the BTDF (thin) or adds σ_s to the medium. Metals: iridescence replaces `conductor_fresnel`; anisotropy changes D/Vis of both specular lobes (the README samples apply to the specular BRDF, base spec_brdf of both metal and dielectric in the Sample Renderer).
3. **Sheen** over the base: `sheen_material = sheenColor*sheen_brdf + material*sheen_albedo_scaling` (sheen README: "layered on top of the glTF 2.0 Metallic-Roughness material").
4. **Clearcoat** over sheen: `mix(sheen_material, clearcoat_brdf, clearcoat*clearcoat_fresnel)` (sheen README: "clearcoat is layered on top of sheen"; clearcoat README: layered on top of the material "including emission and all extensions").
5. **Emission** is under the coat: `coated_emission = emission*(1 - clearcoat*clearcoat_fresnel)`; emission is not darkened by sheen in the ratified text.

Order, top to bottom: clearcoat > sheen > (iridescence/specular/ior/transmission/diffuse_transmission/volume inside the dielectric-vs-metal base) with emissive under the coat. Normative energy statements found: (a) core: BRDFs MUST be positive, reciprocal, energy conserving for physically accurate simulation (Appendix B); (b) clearcoat: base weighted `1 - clearcoat*fresnel` via `mix` (energy conserving "with a simple layering function"); (c) specular and iridescence: base weighted by `1 - max(F)` to avoid gain and inverse colours; (d) sheen: albedo scaling `1 - max3(sheenColor)*E(...)`; with the cheap Ashikhmin visibility the BRDF "will not be energy conserving"; (e) diffuse_transmission: energy moves from diffuse reflection to transmission. Explicitly NOT modelled: inter-layer refraction, inter-reflection between coat and base, coat IOR coupling, iridescence-under-coat interface.

## 13. How the Khronos glTF Sample Viewer / Sample Renderer implements it

Repo layout: the Viewer repo (`7306f9c`) is a UI shell with the renderer as submodule `glTF-Sample-Renderer` (`cc3d5ff`). Shaders: `source/Renderer/shaders/{pbr.frag, brdf.glsl, ibl.glsl, punctual.glsl, material_info.glsl, iridescence.glsl, functions.glsl, textures.glsl, scatter.frag, scatter_back_depth.frag, specular_glossiness.frag}`; LUT generation `source/ibl_sampler.js` + `source/shaders/ibl_filtering.frag`; sheen E LUT = pre-baked `assets/images/lut_sheen_E.png` (`u_SheenELUT`). Viewer README lists supported: anisotropy, clearcoat, diffuse_transmission, dispersion, emissive_strength, ior, iridescence, pbrSpecularGlossiness, retroreflection, sheen, specular, transmission, unlit, variants, volume, and `KHR_materials_scatter` (linked to PR #2579); it does NOT list volume_scatter.

Layering in `pbr.frag` (both IBL and punctual, identical structure):
```glsl
color = mix(f_dielectric_brdf_ibl, f_metal_brdf_ibl, materialInfo.metallic);
color = f_sheen + color * albedoSheenScaling;
color = mix(color, clearcoat_brdf, clearcoatFactor * clearcoatFresnel);
... color = f_emissive * (1.0 - clearcoatFactor * clearcoatFresnel) + color;
```
- Dielectric: `f_dielectric_brdf = mix(f_diffuse, f_specular_dielectric, F)` with `F = F_Schlick(f0_dielectric*specularWeight, f90_dielectric(=specularWeight), |VdotH|)`; `f_diffuse` is `mix(f_diffuse, f_specular_transmission, transmissionFactor)` after the diffuse-transmission mix (so transmission overrides diffuse transmission, as in the spec). Note: the Sample Renderer uses `mix(base, layer, F)` with a vec3 F, i.e. NOT the spec's `(1 - weight*max(fr))*base` form (comment in source: "Do we need to handle vec3 fresnel here?"). For iridescence it does call `rgb_mix` (spec form).
- IBL: split-sum GGX LUT (`u_GGXLUT`, generated at 1024x1024, 512 samples, `ibl_sampler.js` `sampleGGXLut`), `getIBLGGXFresnel` = Fdez-Aguera multiple-scattering compensation (`FssEss + FmsEms`, `F_avg = F0 + (1-F0)/21`) - this goes beyond the spec. Prefiltered radiance cubemap 256 px, 1024 GGX samples, lod = `roughness*(mipCount-1)`. Sheen IBL: Charlie-filtered cubemap + `u_CharlieLUT` (`.b` channel = directional albedo) -> `getIBLRadianceCharlie = sheenLight * sheenColor * brdf`; albedo scaling LUT `u_SheenELUT` (`.r`) sampled at `(NdotV, sheenRoughness)`; punctual uses `min(scaling(NdotV), scaling(NdotL))` exactly like the README.
- Clearcoat: coat F0 = `((ior-1)/(ior+1))^2` of the **material's** `ior` (`getClearCoatInfo` in `material_info.glsl`), deviating from the README's fixed 1.5; F90 = 1; coat Fresnel evaluated with `clampedDot(clearcoatNormal, v)` (README-consistent); coat lobe = plain GGX (`BRDF_specularGGX`) with coat normal and `clearcoatRoughness²`; roughness clamped to [0,1].
- Anisotropy: README formulas verbatim (`D_GGX_anisotropic`, `V_GGX_anisotropic` in `brdf.glsl`); `ab = clamp(alpha, 0.001, 1)`; anisotropic T from the (rotated) texture direction through TBN, `anisotropicB = cross(ng, T)`; IBL uses a bent-normal reflection with `bendFactor^4` (README suggests `pow2(pow2(...))`, same) while using only the base-roughness LOD.
- Iridescence: README's Belcour-Barla analytic code (`iridescence.glsl` `evalIridescence`), evaluated once per fragment at `NdotV` (for IBL AND punctual lights, i.e. NOT per light), for both dielectric (`f0_dielectric`) and metal (`baseColor`); if thickness == 0, iridescenceFactor is forced to 0.
- Transmission: **screen-space, opaque framebuffer**. `renderer.js` (~l.1135): when transmissive drawables exist, all opaque and non-transmissive-blend drawables plus the environment are first rendered (MSAA, linear) into `opaqueFramebuffer`, which is mip-chain-filtered; transmissive materials are then drawn sorted back-to-front and sample that texture (`u_TransmissionFramebufferSampler`) at `lod = log2(width) * applyIorToRoughness(roughness, ior)`, `applyIorToRoughness = roughness * clamp(ior*2 - 2, 0, 1)`. So transmissive objects do NOT see each other (only opaque scene + environment). Refraction: `getVolumeTransmissionRay = normalize(refract(-v, n, 1/ior)) * thickness * modelScale`; exit point projected to the framebuffer; attenuation by `pow(attenuationColor, length/attenuationDistance)`; result multiplied by `baseColor`. Punctual-light transmission (`getPunctualRadianceTransmission`): mirrors the light vector through the surface (`l_mirror`), `h = normalize(l_mirror + v)`, `baseColor * D_GGX * V_GGX` (NOT the separable-G BTDF of the README; it reuses the reflection D/Vis with an IOR-scaled roughness).
- Dispersion: 3 refraction rays with `iors = (ior - halfSpread, ior, ior + halfSpread)` (README formula), one framebuffer tap per channel.
- Diffuse transmission: lambertian BTDF with `getDiffuseLight(-n)` (irradiance on the back side) in IBL, `lightIntensity * clampedDot(-n, l) / π * color` for punctual lights on the back hemisphere; volume attenuation applied with `thickness * average model scale`; diffuse BRDF scaled by `1 - factor`.
- KHR_materials_scatter (draft) in the Renderer: thin-walled branch implemented as in the draft (`forwardFraction/backwardFraction` lobes, IBL and punctual; mixes into the transmission term by `scatterStrength`); volumetric branch implemented with extra passes (`scatter.frag`, `scatter_back_depth.frag`): back-face depth buffer gives thickness, a 16-step single-scattering ray march (Henyey-Greenstein + Beer-Lambert, `getVolumeSingleScattering`, planar-slab chord approximation, "avoids depth peeling") plus a screen-space Burley diffusion profile blur (`getSubsurfaceScattering`, Christensen-Burley 2015, `u_ScatterSamples`) for multi-scatter. Added to `color` scaled by `(1 - metallic) * (1 - clearcoatFactor*clearcoatFresnel) * (1 - iridescenceFactor) * transmissionFactor`. So the Sample Renderer implements SSS as post-lit screen-space blur plus single-scatter march, not by integrating the BSDF; it is a pragmatic approximation of a draft spec.
- volume_scatter (the older draft): not implemented as a separate extension.

## 14. tinygltf and KHR extension handling

**Phosphor pins `v2.9.7`** (`cmake/Dependencies.cmake`: `GIT_TAG v2.9.7`, SHA `488a70a3...`, header `tiny_gltf.h`). The `release` branch HEAD (`v3.1.0`, `97789b3...`) no longer has `tiny_gltf.h`: it ships a C11 rewrite (`tiny_gltf_v3.h/.c`, `tinygltf_json_c.h`; `attic/` holds old files). Upgrading to current `release` is an API rewrite, not a version bump; keep pinning `v2.9.x` unless a migration is planned.

`tiny_gltf.h` @ `v2.9.7` (https://github.com/syoyo/tinygltf/blob/v2.9.7/tiny_gltf.h):
- `typedef std::map<std::string, Value> ExtensionMap;` (l.516). Every glTF object, including `Material` (l.753), `TextureInfo` (l.673), `NormalTextureInfo` (l.690), `OcclusionTextureInfo` (l.710), `PbrMetallicRoughness` (l.730) and `Texture` (l.655), has `Value extras; ExtensionMap extensions;`.
- `ParseExtensionsProperty` (l.4220) reads `"extensions"` into `ExtensionMap`: each extension value is parsed generically to `tinygltf::Value` via `ParseJsonAsValue`; **extension entries that are not JSON objects are skipped** (`if (!IsObject(itObj)) continue;`). `ParseTextureInfo` (l.4461) parses `index`, `texCoord`, then `ParseExtrasAndExtensions`, so `KHR_texture_transform` lands in `TextureInfo::extensions["KHR_texture_transform"]` as a `Value` object (unparsed).
- `ParseMaterial` (l.5348) fills core fields, then `material->extensions.clear(); ParseExtrasAndExtensions(...)`. So `material.extensions["KHR_materials_clearcoat"]` etc. are `Value` objects; **no KHR_materials_* extension and no KHR_texture_transform is parsed natively** (grep: only `KHR_lights_punctual` l.5216/6616, `KHR_draco_mesh_compression` l.5137 and `MSFT_lod` get special handling; `KHR_materials_emissive_strength`, `KHR_texture_basisu` etc. are absent).
- `Value` API (l.256+): `IsNumber/IsReal/IsInt/IsArray/IsObject/IsString`, `GetNumberAsDouble()`, `GetNumberAsInt()`, `Has(key)`, `Get(key)` (returns a static null Value if missing; `assert(IsObject())`), `Get(idx)`, `ArrayLen()`, `Keys()`. Texture refs inside an extension (`clearcoatTexture: {index, texCoord}`) must be converted by hand: they are plain `Value` objects, NOT `TextureInfo`; their nested `extensions` (texture_transform) are also plain Values. Material `values`/`additionalValues` (legacy `ParameterMap`) do not contain extension data (the loop skips `extensions` and `extras`).
- Optional `SetStoreOriginalJSONForExtrasAndExtensions` fills `extensions_json_string` per object.
- Because tinygltf decodes all images by index and `Texture::source` is only the core field, textures whose image comes from an extension (`KHR_texture_basisu` etc.) have `source = -1` (not fully verified; relevant to F15 only if compressed textures are used).
- Writer: `SerializeExtensionMap` (l.7274) round-trips unknown extensions.

`tiny_gltf_v3.h` @ `v3.1.0`: extensions are `tg3_extras_ext { const tg3_extension *extensions; uint32_t extensions_count; tg3_str extensions_json; }` with `tg3_extension { tg3_str name; tg3_value value; }`; the only material-extension-related symbol is the name macro `TG3_EXT_MATERIALS_EMISSIVE_STRENGTH "KHR_materials_emissive_strength"` (l.252, inside the `TG3_GLTF21_UNSTABLE` "promoted extension names"), i.e. still not parsed into typed fields (grep for `KHR_materials`, `clearcoat`, `KHR_texture_transform`: no other hits). The README states glTF 2.1 is a draft and not ratified.

## 15. Implications for the F15 schema/shader (derived, not from sources)

- Schema per material (factors + texture refs with channel+colour-space): clearcoat{factor, roughness, normal tex}, sheen{colour sRGB, roughness .a}, anisotropy{strength, rotation, tex rg/b}, transmission{factor .r}, volume{thickness .g, attenuationColor/Distance}, ior (0 special), specular{factor .a, colour sRGB}, iridescence{factor .r, ior, thickMin/Max, thickness .g}, diffuse_transmission (RC){factor .a, colour sRGB}, emissive_strength, dispersion; texture transform per textureInfo.
- Reserve for baked LUTs: GGX DFG (split-sum), sheen E(NdotV, sheenRoughness) (Charlie; spec gives no table), plus Charlie directional-albedo for IBL.
- Needs tangent space for anisotropy and clearcoat normal map (MikkTSpace already vendored).
- Transmission with visibility buffer: Sample Renderer approach = opaque colour mip chain + back-to-front sort; the spec only requires opaque objects visible.

## 16. Open points / uncertainties

1. Ratification dates and the Khronos registry (khronos.org) listing were not read; "Ratified" comes from the README Status line at f93f8af only.
2. `KHR_materials_diffuse_transmission` is a **Release Candidate** (README), and the proposed `KHR_materials_scatter` (draft PR #2579) says it would replace both it and volume_scatter and be mutually exclusive with it. The final shape of diffuse transmission / SSS in glTF is unstable; the ratified path today is volume (absorption only) + transmission.
3. diffuse_transmission README text/formula mismatch on whether `diffuseTransmissionColor` is multiplied by the factor (section 10.1); the normative `mix` formula is self-consistent without it.
4. (resolved) sheenRoughnessFactor schema range is [0,1], verified.
5. KHR_materials_clearcoat coat F0 is fixed 1.5 in the spec but follows the material IOR in the Sample Renderer (`material_info.glsl getClearCoatInfo`); `KHR_materials_clearcoat_ior` is an open draft (#2530), `KHR_materials_coat` (#2534) would replace clearcoat entirely. Which to mirror in Phosphor is a decision for F15 (fixed 1.5 is the ratified behaviour).
6. The Sample Renderer's dielectric Fresnel mix is `mix(diffuse, specular, F_vec3)`, not the spec's `(1 - weight*max(fr))*base + weight*fr*layer`; they differ for coloured F (specular colour, iridescence uses rgb_mix correctly).
7. No normative text defines the transmission-through-multiple-transparent-surfaces behaviour; only "at least opaque objects". The Sample Renderer does not let transmissive objects see each other.
8. The Sample Renderer's SSS (scatter) is a screen-space approximation of a draft; not a reference for correctness. The Henyey-Greenstein sign in the draft README (`+2g(v·l)`, with `v` the outgoing-view convention) was copied as written; the Renderer's code uses `-2g cosTheta`; convention to be fixed when implementing.
9. Sheen E LUT provenance: `lut_sheen_E.png` is a pre-baked asset in the Renderer; the README only says the table is in Enterprise PBR section 6.2.3 (not read) or must be generated with Filament cmgen. Phosphor should generate it itself with a negative control (not verified how the Sample Renderer produced the PNG).
10. tinygltf: Phosphor pins `v2.9.7`; `release`/`v3.1.0` is a different C API with the same lack of native KHR material parsing (only an emissive_strength name macro). Whether v2.9.7 correctly handles image sources for extension-only textures (basisu/webp) was not tested.
11. Open OPT/EDGE-style catalogue note: `KHR_materials_retroreflection` (RC) and `KHR_materials_fuzz`/`diffuse_roughness`/`coat` drafts exist but were intentionally not researched beyond status.
12. All Sample Renderer behaviour described comes from reading source at `cc3d5ff`, not from running it; iridescence/volume code paths were read at a skim level for formulas only.
