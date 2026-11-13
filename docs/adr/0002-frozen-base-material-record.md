# Material features live outside the frozen 80-byte base material record

`GPUMaterial` stays 80 bytes because it equals the payload of `GPUDeltaRecord`,
the only path materials take to the GPU, and because RT, GI, light sampling and
culling read it on every scene. F15 material features (clearcoat, sheen,
anisotropy, iridescence, specular/ior, transmission/volume, subsurface) store
their parameters in a separate buffer of 80-byte extension records updated by
the same `SceneStore` delta mechanism; the base record only gains feature bits
in `flags` and the index of its first extension record in its padding. Scenes
without features (Sponza) keep identical buffers, pipelines and images.

## Considered Options

- Growing `GPUMaterial`: rejected, it grows every material/instance delta and
  every secondary-hit read for all scenes.
- A per-feature texture atlas of parameters: rejected, parameters must follow
  per-entity edits through the existing delta path.
