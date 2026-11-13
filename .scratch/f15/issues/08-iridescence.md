# 08: Iridescence lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_iridescence` renders per decision 06 with the Belcour–Barla thin-film Fresnel evaluated per light sample at V·H and at N·V for environment/albedo terms, on metals and dielectrics.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Iridescence fixtures (thickness sweep on dielectric and metal, thickness texture, factor) pass decision-11 thresholds against a reference implementing the specification's code independently
- [ ] Thickness 0 disables the effect as specified; furnace bound holds
- [ ] Negative control (film ior ignored) fails the bound
- [ ] Measured per-light cost on the fixture recorded; IridescenceDielectricSpheres/MetallicSpheres side by side
