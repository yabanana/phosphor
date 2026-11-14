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

## F15.4 (ticket 23): record 3 in use

The layer stack (record 3) is rendered. Its frozen 80 bytes keep `maskTex`, `layerCount`, `layerMaterial[4]`,
`packedMaskChannels` and `heightContrast[4]`; the nine padding words now hold `uvScale[4]` (per-layer texture
tiling, 0 for a layer that does not exist), `heightTex[4]` (per-layer height texture, `INVALID_TEXTURE_INDEX` when the layer has none) and one
padding word. Size and offsets of the existing fields are unchanged, so no existing scene, delta or archive entry
changes; scenes without layers never read the record, and a material without a stack packs the same bytes into it as before F15.4 (the
unused words are 0), so snapshots and reference caches of every other scene stay valid. The layer materials are ordinary entries of the material
library (`GPUMaterial` records, base parameters only); the stack's own `GPUMaterial` keeps its base fields as the
appearance for secondary hits and probes (declared limit: ray-traced reflections, GI and probes see that single
record, not the blend).
