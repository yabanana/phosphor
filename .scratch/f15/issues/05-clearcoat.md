# 05: Clearcoat lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_clearcoat` (factor, roughness, coat normal map) renders per decision 06: coat GGX with F0 0.04 over the base and emission, coat term in the environment, coat-or-base lobe choice in the specular transport, effective diffuse weight in GI, combined albedo for the denoiser.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Clearcoat grids (factor × roughness, coat normal) pass decision-11 thresholds against the independent reference; furnace bound holds
- [ ] Specular transport chooses the coat lobe with the documented mixture pdf; reflections fixture with a rough base + smooth coat shows the coat reflection (reference comparison)
- [ ] Negative control (coat ignored, or coat F0 from material ior as in the Sample Renderer) fails the bound
- [ ] ClearCoatTest and ClearCoatCarPaint side by side under the declared rig; deviations from the Sample Renderer listed
- [ ] Denoiser stability: F13 temporal metrics on a clearcoat clip within the F13.6 contract
