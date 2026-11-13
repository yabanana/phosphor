# 06: Clearcoat lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_clearcoat` (factor, roughness, coat normal map) renders per decision 06: coat GGX with F0 0.04 over the base and emission (sun-disk quadrature covers the coat lobe), coat term in the environment, coat-or-base lobe choice in the specular transport, effective diffuse weight in the composition, coat roughness in the denoiser edge-stopping where the coat dominates.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Clearcoat grids (factor × roughness, coat normal) pass decision-11 thresholds against the independent reference; furnace bound holds
- [ ] Rough base + smooth coat fixture: the coat reflection survives the denoiser and matches the reference within the environment/transport bound
- [ ] Negative control (coat ignored, or coat F0 from the material ior as in the Sample Renderer) fails the bound
- [ ] ClearCoatTest and ClearCoatCarPaint side by side under the declared rig; deviations from the Sample Renderer listed
- [ ] Temporal clip of the coat fixture within the F13.6 contract
- [ ] Emitter ownership: the DI/RT share of emitter specular for the coat lobe uses the coat roughness (F13 partition extended); an emissive-panel fixture shows no double count against the reference
