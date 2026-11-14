# 18: Conditional: hybrid ray-traced single scattering for volumetric subsurface

Roadmap: F15.1, F15.3

**What to build:** Only if ticket 17 misses its side-lit bound: add ray-traced zero/single scattering from the exit point to the screen-space diffusion (profile minus single-scatter, no double count). Own bound, timebox and fallback fixed now.

**Blocked by:** 17 (and only if its bound is missed)

**Status:** ready-for-agent

- [ ] Bound: the ticket-17 side-lit subsurface bound on the same fixtures; additionally the back-lit fixture within 2× the side-lit bound
- [ ] Timebox 2 weeks; if not met, fallback = keep screen-space diffusion, F15.1 subsurface stays explicitly partial (unticked for the volumetric mode) with the measured gap reported
- [ ] Cost per subsurface pixel measured; scenes without subsurface unchanged
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-14 — orchestrator: TRIGGERED by decision 10's rule. Ticket 17 misses its side-lit bound on valid OpenPBR ground truth
(roughness 0.35, boundary-aware inversion), with the energy right and the profile shape wrong; see ticket 17's last comment.
- Variant choice within the authorized phase (spike results choose variants). The measured cause is the multiple-scattering profile
  shape, and the RT single scattering named in this ticket mostly carries low orders and back-lit light. Two candidates, in order:
  1. a radial diffusion profile TABULATED from the same Mitsuba transport and boundary as the ground truth (semi-infinite slab,
     searchlight), replacing the Burley fit in the screen-space pass;
  2. RT zero/single scattering from the exit point (Zhang and d'Eon 2025; research/realtime-transmission-sss.md S6), profile minus
     single scatter, for the back-lit bound.
- Unchanged: the bound, the timebox and the fallback stated in this ticket.
