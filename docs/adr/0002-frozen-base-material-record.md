# Material features live outside the frozen 80-byte base material record

`GPUMaterial` stays 80 bytes because it equals the payload of `GPUDeltaRecord`,
the only path materials take to the GPU, and because RT, GI, light sampling and
culling read it on every scene. F15 material features (clearcoat, sheen,
anisotropy, iridescence, specular/ior, transmission/volume, subsurface) store
their parameters in a separate buffer of 80-byte extension records updated by
the same `SceneStore` delta mechanism; the base record only gains feature bits
in `flags` (bit 8 + `MaterialFeature`) and the index of its first extension
record in its padding (`extensionIndex`, `MATERIAL_EXTENSION_INVALID` = none).
Scenes without features (Sponza) keep identical buffers, pipelines and images.

Implemented in F15.1 (ticket 05): an extended material owns a fixed group of
`MATERIAL_EXT_RECORDS` = 4 consecutive records, allocated when it first becomes
extended (a counted structure change) and kept for the life of the material, so
toggling a feature rewrites records and never reallocates. The layouts of every
feature are frozen in `gpu_types.h` (record 0 specular+ior, clearcoat, sheen;
record 1 anisotropy, iridescence, transmission, volume; record 2 subsurface;
record 3 layer stack), each 80 bytes with a `static_assert`.

## Considered Options

- Growing `GPUMaterial`: rejected, it grows every material/instance delta and
  every secondary-hit read for all scenes.
- A per-feature texture atlas of parameters: rejected, parameters must follow
  per-entity edits through the existing delta path.
