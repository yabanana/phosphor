# 16: Subsurface feature, thin mode (diffuse transmission)

Roadmap: F15.1, F15.3

**What to build:** The subsurface feature (OpenPBR parameters, ADR-0005) in its thin mode: a diffuse-transmission lobe for analytic lights on the back hemisphere, back-side environment and GI irradiance at −N; fed by `KHR_materials_diffuse_transmission` (RC), the draft `KHR_materials_scatter` thin mode and the material-description file.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Thin fixtures (backlit plane, factor and colour sweeps) pass decision-11 thresholds; furnace bound holds (energy moved, not added)
- [ ] Both glTF encodings and the description file produce the same description (doctest) and image
- [ ] Negative control (lobe on the front hemisphere) fails; support report labels RC/draft inputs
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
