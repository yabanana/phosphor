# 05: Lobe guide: DI, reflections, composition and denoiser see material features

Roadmap: F15.1, F15.3

**What to build:** A per-pixel lobe guide with the resolved feature parameters is written by the resolve and read by DI (ReSTIR and clustered), the specular transport, the composition (effective diffuse weight instead of albedo × (1 − metallic)) and the denoiser (edge-stopping roughness = effective transport roughness of the significant lobes; the F13 denoiser filters BRDF-weighted radiance and does not demodulate). It is allocated and read only when the scene feature union is non-empty. Proven with ior and specular.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] ior/specular grids pass decision-11 thresholds with `--lighting restir` and `clustered` and with RT reflections; the support report marks every light path of decision 06 for ior/specular
- [ ] Guide allocated only for a non-empty union (memory per category in the report); Sponza unchanged (visual_check 0 px, D2 within noise)
- [ ] Negative control: DI reading the base BRDF instead of the guide fails the DI grid bound
- [ ] F13 temporal metrics on a specular-extension clip within the F13.6 contract
