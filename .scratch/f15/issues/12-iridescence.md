# 12: Iridescence lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_iridescence` per decision 06: Belcour–Barla thin-film Fresnel per light sample at V·H and at N·V for environment/weight terms, on metals and dielectrics.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Iridescence fixtures (thickness sweep on dielectric and metal, thickness texture, factor) pass decision-11 thresholds against an independent implementation of the specification's code
- [ ] Thickness 0 disables the effect; furnace bound holds; negative control (film ior ignored) fails
- [ ] Per-light cost measured; IridescenceDielectricSpheres/MetallicSpheres side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
