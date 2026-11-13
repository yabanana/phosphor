# 08: Specular transport, composition and denoiser contract for multi-lobe materials

Roadmap: F15.1, F15.3

**What to build:** The specular transport samples from the shared BSDF with the guide's lobes (lobe choice with a mixture pdf), the composition uses the effective diffuse weight and gains the environment/probe terms that lobes need (e.g. sheen), and the denoiser receives through a new specialised binding the sampled-lobe id, axis (reflect/refract) and roughness: neighbours and history are rejected across lobe ids; `GPUDISurface` stays unchanged. Proven with ior/specular, ready for coat and transmission.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] ior/specular grids pass with RT reflections, SSR and probes; the support report marks every light path of decision 06 for ior/specular
- [ ] Denoiser negative control: using the reflect axis for a refract-axis lobe fails the clip; history across lobe ids rejected (unit-tested on the CPU mirror of the rule)
- [ ] F13 temporal metrics on a specular-extension clip within the F13.6 contract
- [ ] Scenes without features unchanged (visual_check 0 px, D2 within band)
- [ ] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
- [ ] The composite's environment/GI diffuse uses the effective diffuse weight of extended materials: `tools/f15_reference.py` DECLARED `extension_diffuse` switches from `composite-base` to `extension` and `specular-grid-env-resolve` stays green against it (until then the support report names `di_transport_and_composite_diffuse_use_base_model`)
- [ ] Carried from ticket 05: a uniform-environment furnace over EXTENDED materials (specular grid at albedo 1) stays ≤ 1 per pixel once the composite uses the extended diffuse weight; indirect/denoised signals recover within the F13.6 recovery criterion after an extension edit; adaptive 2×2 shading reuse (`--adaptive-shading`) leaves feature-bit pixels identical to the run without reuse
