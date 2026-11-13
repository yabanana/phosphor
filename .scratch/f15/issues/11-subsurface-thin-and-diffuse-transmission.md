# 11: Subsurface feature, thin mode (diffuse transmission)

Roadmap: F15.1, F15.3

**What to build:** The subsurface feature (OpenPBR parameters, ADR-0005) exists end to end in its thin mode: a diffuse-transmission lobe for analytic lights on the back hemisphere, back-side environment and GI irradiance at −N; fed by `KHR_materials_diffuse_transmission` (RC), the draft `KHR_materials_scatter` thin mode and the material-description file.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Thin fixtures (backlit plane, factor and colour sweeps) pass decision-11 thresholds against the independent reference; furnace bound holds (energy moved from diffuse reflection, not added)
- [ ] Both glTF encodings and the description file produce the same description for equivalent inputs (doctest) and the same image
- [ ] Negative control (lobe evaluated on the front hemisphere) fails the bound
- [ ] Support report labels RC/draft inputs
