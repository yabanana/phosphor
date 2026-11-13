# 04: Real-time techniques for transmission and subsurface scattering

Type: research
Status: resolved
Blocked by: none

## Question

Which real-time techniques implement KHR transmission/volume and subsurface
scattering (thin diffuse transmission and volumetric SSS) in a deferred or
visibility-buffer renderer: screen-space refraction over an opaque colour
pyramid (Khronos Sample Viewer, Filament), ray-traced refraction, separable
screen-space SSS (Jimenez), Burley normalized diffusion in screen space
(Unity HDRP, Unreal), ray-traced or random-walk SSS. For each: inputs it
needs from the pipeline, how it maps to the glTF parameters, published cost
and quality limits, and what the Khronos Sample Viewer does.
Findings: `.scratch/f15/research/realtime-transmission-sss.md`.

## Answer

Findings: [realtime-transmission-sss.md](../research/realtime-transmission-sss.md)
(Sample-Renderer `cc3d5ff`, Viewer `7306f9c`, papers/talks cited there).

- Transmission: opaque colour pyramid (Khronos, ~0.07 ms image-space on an RTX
  2070S at 1080p), back-face depth, ray-traced refraction (0.67/1.27/1.58 ms for
  1/2/3 rays, same GPU), hybrid. The pyramid needs transmissive geometry out of
  the opaque visibility pass plus a separate lit transmissive pass; ray-traced
  refraction keeps glass in the visibility buffer and reuses the RT reflection
  machinery, sees other glass and the real back faces.
- Subsurface: separable Gaussian (Jimenez 2015, 0.30–0.49 ms 1080p), Burley
  normalised diffusion in screen space (Golubev/HDRP, Khronos Sample Renderer:
  diffuse-lighting pre-pass, 21/55 samples, same-material test, metres→pixels
  radius, sub-pixel skip), hybrid RT single scattering + diffusion (NVIDIA
  2025), random walk (validation only).
- Phosphor fit: diffuse stays separable (`GI·diffuseAlbedo` composition), so
  the resolve/DI can write a diffuse-irradiance target for subsurface pixels.
- Decisions taken in tickets 09 (transmission) and 10 (subsurface).
