# Specular transmission is ray-traced in F15; raster transparency stays in F16

Owner decision, 2026-11-13. F15 renders `KHR_materials_transmission`/`volume`
for surfaces of the opaque visibility buffer by tracing refracted rays inside
the existing specular transport (one Fresnel-chosen sample per pixel, at most
four interface crossings, Beer–Lambert along the real inside path). Glass keeps
DI, RT reflections and the denoiser on its reflective lobes and can see other
glass. Without `--rt on` the transmitted radiance comes from probes or the
environment (declared approximation). Transmissive casters shadow opaquely.
Blended transparency, OIT, the opaque colour pyramid used by the Khronos viewer,
liquids and dispersion belong to F16.

## Considered Options

- Khronos-style opaque colour pyramid in F15: rejected, it needs glass outside
  the visibility pass and a lit transparent pass (F16 infrastructure), and loses
  glass-through-glass and the RT lighting of the glass surface.
