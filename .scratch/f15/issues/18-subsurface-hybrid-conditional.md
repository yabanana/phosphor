# 18: Conditional: hybrid ray-traced single scattering for volumetric subsurface

Roadmap: F15.1, F15.3

**What to build:** Only if ticket 17 misses its side-lit bound: add ray-traced zero/single scattering from the exit point to the screen-space diffusion (profile minus single-scatter, no double count). Own bound, timebox and fallback fixed now.

**Blocked by:** 17 (and only if its bound is missed)

**Status:** ready-for-agent

- [ ] Bound: the ticket-17 side-lit subsurface bound on the same fixtures; additionally the back-lit fixture within 2× the side-lit bound
- [ ] Timebox 2 weeks; if not met, fallback = keep screen-space diffusion, F15.1 subsurface stays explicitly partial (unticked for the volumetric mode) with the measured gap reported
- [ ] Cost per subsurface pixel measured; scenes without subsurface unchanged
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
