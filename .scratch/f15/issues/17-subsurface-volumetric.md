# 17: Subsurface feature, volumetric mode (screen-space diffusion)

Roadmap: F15.1, F15.3

**What to build:** Volumetric subsurface per ADR-0005: resolve and DI write a diffuse-irradiance target for subsurface pixels, a Burley normalised-diffusion pass in screen space (21/55 samples, metres→pixels radius, sub-pixel skip, same-material and depth test) runs after diffuse denoising, recombined with the specular terms; draft `KHR_materials_scatter` volumetric mode mapped through its Kulla–Conty inversion. Back-lit translucency of thick volumes is out of reach of screen-space diffusion and declared.

**Blocked by:** 16

**Status:** ready-for-agent

- [ ] Side-lit volumetric fixtures vs Mitsuba 3.9.1 `volpath` random walk (homogeneous medium inside a boundary with the engine's roughness) on the piece mask: decision-11 subsurface bound
- [ ] Back-lit thick fixture measured and reported, not gated (declared limit of screen-space diffusion)
- [ ] Radius sweep monotonic; sub-pixel radius equals subsurface-off within FP noise; negative control (diffusion off) fails
- [ ] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Scenes without subsurface unchanged (visual_check 0 px); diffusion pass cost measured
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
