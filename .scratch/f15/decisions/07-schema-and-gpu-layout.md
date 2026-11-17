# 07: Material description schema and GPU data layout

Type: grilling
Status: resolved
Blocked by: 01

## Question

What does the versioned material description contain (base, features, texture
references with transforms, layer stack), how is it versioned, how does it
reach the GPU (frozen 80-byte base record, extension records, per-pixel lobe
parameters for DI/reflections) and how does the support report look?

## Answer

Evidence: `gpu_types.h` (`GPUMaterial` 80 B = `GPUDeltaRecord` payload),
`scene_store.h` (material deltas), `gltf_loader.cpp` (core fields only),
research 01/02, ADR-0001.

- **Material description** (portable, `phosphor_core`): schema version
  constant; base metallic-roughness parameters; one optional parameter block per
  material feature (specular+ior, clearcoat, sheen, anisotropy, iridescence,
  transmission, volume, subsurface); texture references carry index, colour
  space, channel and texCoord; a bounded layer stack (F15.4). Unit-tested
  validation (ranges, ior = 0 special mode, required companions such as
  volume needing transmission or subsurface).
- **Texture transforms**: when every texture of a material shares one
  transform on UV0, the loader bakes it into that primitive's UV0 (affine
  transforms commute with interpolation: exact up to rounding); otherwise the
  support report marks texture_transform unsupported for that material. No GPU
  layout change for transforms.
- **GPU**: `GPUMaterial` stays 80 bytes and its consumers unchanged; feature
  bits go into `flags`, `pad[0]` holds the first extension record of the
  material. Extension parameters live in a separate GPU buffer of 80-byte
  records (one or more per material, `GPUDeltaRecord`-compatible), updated by
  `SceneStore` deltas exactly like base materials, so a parameter edit changes
  the next frame. The store exposes the scene feature union (like
  `hasEmissive`). DI and reflections receive the resolved per-pixel lobe
  parameters through an additional guide buffer allocated and read only when
  the scene feature union is non-empty; Sponza allocates nothing new.
- **Support report**: per material and feature, implemented / approximated
  (declared approximation id) / unsupported; unknown extensions listed;
  assets whose `extensionsRequired` lists an unsupported extension are refused
  with an error (glTF 2.0 rule; wording verified at implementation). Printed as
  a `MATERIALS` log line and written into the benchmark report (schema bump).
- ADR-0002 records the frozen base record and the separate extension buffer.
