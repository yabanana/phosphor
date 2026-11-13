# 24: F15 phase closure: residue audit, regressions, measurements and handoff

Roadmap: F15.1–F15.4

**What to build:** Severe residue audit (roadmap lines, plan, exit criteria, TODOs), full regression battery, Sponza budget re-measured, Apple9 forced path, documentation and per-ID handoff; roadmap boxes ticked only for verified scope.

**Blocked by:** 15, 18 (if triggered), 20, 21, 22, 23

**Status:** ready-for-agent

- [ ] `ctest`, `metal_syntax_check`, `tools/visual_check.sh` (0 px vs the ticket-03 baseline), F13/F14 corpus, `variant_check`, `archive_check`, `hitch_check` green; API validation clean (shader validation where the F14.4 defect does not apply)
- [ ] Sponza D2 GPU p95 A/B/A against the ticket-03 baseline within the decision-11 band; 0 GPU allocations
- [ ] `--force-family apple9` run of the material bench with features
- [ ] ROADMAP F15.1–F15.4 ticked only where verified; approximations and deferred items listed (caustics, dispersion, multiple-scattering compensation, back-lit thick subsurface if the hybrid is not triggered or not met, non-uniform texture transforms, coated emission of sampled emitters, GI/probe rays hitting glass as opaque)
- [ ] perf-log, opt-log, RENDERER.md and per-ID handoff written; no push or PR without the owner
