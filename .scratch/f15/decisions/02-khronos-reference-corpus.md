# 02: Khronos reference corpus for F15.3

Type: research
Status: resolved
Blocked by: none

## Question

Which Khronos glTF-Sample-Assets models exercise each extension of ticket 01
(one per lobe plus critical combinations), with license, size, extensions and
KHR_texture_transform usage, pinned commit and hashes; what reference images
exist (Sample Viewer screenshots, Khronos glTF Render Fidelity golden images:
renderers, environment and camera used) and whether they can serve as numeric
or only qualitative references.
Findings: `.scratch/f15/research/khronos-corpus.md`.

## Answer

Findings: [khronos-corpus.md](../research/khronos-corpus.md)
(glTF-Sample-Assets `8f49239`, Render-Fidelity `9d35e14`, Generator `5eb9669`,
Sample-Renderer `cc3d5ff`, Sample-Viewer `7306f9c`).

- Corpus Tier A (~20 MB, plain glTF + PNG/JPG, CC0/CC-BY-4.0): ClearCoatTest,
  ClearCoatCarPaint, SheenTestGrid, AnisotropyStrength/Rotation/DiscTest,
  TransmissionTest, TransmissionRoughnessTest, TransmissionThinwallTestGrid,
  AttenuationTest, IORTestGrid, SpecularTest, IridescenceDielectricSpheres,
  IridescenceMetallicSpheres, IridescenceSuzanne, DiffuseTransmissionTest,
  EmissiveStrengthTest, TextureTransformTest, TextureTransformMultiTest.
  Tier B (combinations): AnisotropyBarnLamp, ToyCar, SpecularSilkPouf,
  GlamVelvetSofa; ABeautifulGame for the Track Studio case. Avoid
  DragonAttenuation (non-CC), WebP/Draco/BasisU/Meshopt variants.
- No sample puts clearcoat and sheen on one material: a hand-authored fixture is
  needed. The only volume-scatter sample (ScatteringSkull) uses a draft
  spelling; treat as experimental.
- Reference images (per-model screenshots, Render Fidelity goldens under an HDR
  environment with ACES) are qualitative only: Phosphor has no image-based
  lighting and a different tonemap.
- Decision: F15.3 numeric gates come from Phosphor-side independent
  references under frozen Phosphor lighting (per-lobe CPU reference written from
  the specification, Mitsuba for ray casting/ground-truth transport); the
  Khronos models are rendered under a declared Phosphor light rig and compared
  side by side with the Khronos images, with per-feature defects documented,
  never as global parity. KHR_texture_transform (offset/rotation/scale on UV0)
  is required by several corpus models and joins F15.1's loader scope;
  KHR_materials_unlit and KHR_lights_punctual are reported as unsupported
  (DiffuseTransmissionTest is then qualitative only).

Amendment (owner, 2026-11-15, F15 closure round 2): KHR_materials_unlit and KHR_lights_punctual are now an approved scope addition, no
longer unsupported by design. The loader lowers unlit onto the emission path; the surface is never a sampled emitter or GI emission.
Punctual lights are imported as ECS lights (lux and candela taken as W/m2 and W/sr). The spot falloff stays the engine smoothstep, a
declared deviation from the specification's informative t^2. DiffuseTransmissionTest renders. Per-slot (non-uniform)
KHR_texture_transform and COLOR_0 vertex colours were also approved by the owner on the same day. A second UV set per material
stays out of scope.
