# 09: Clearcoat lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_clearcoat` (factor, roughness, coat normal) per decision 06: coat GGX with F0 0.04 over the base and emission (sun-disk quadrature covers the coat lobe), coat term in the environment, coat-or-base lobe choice in the transport, coat lobe id/roughness to the denoiser, emitter ownership by coat roughness.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Clearcoat grids (factor × roughness, coat normal) pass decision-11 thresholds against the independent reference; furnace bound holds
- [ ] Rough base + smooth coat: the coat reflection survives the denoiser and matches the reference within the transport bound
- [ ] Emitter ownership: an emissive-panel fixture shows no double count between DI and the coat lobe's transport
- [ ] Negative control (coat ignored, or coat F0 from the material ior as in the Sample Renderer) fails the bound
- [ ] ClearCoatTest and ClearCoatCarPaint side by side under the declared rig; Sample Renderer deviations listed; temporal clip within F13.6
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
