# 07: DI reads the lobe guide (ReSTIR and clustered)

Roadmap: F15.1, F15.3

**What to build:** ReSTIR and clustered DI evaluate the shared BSDF with the lobe guide's parameters for extended pixels (target and shading), proven with ior/specular.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] ior/specular grids pass decision-11 thresholds with `--lighting restir` and `clustered`
- [ ] Negative control: DI reading the base BRDF instead of the guide fails the DI grid bound
- [ ] Scenes without features unchanged (visual_check 0 px)
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
