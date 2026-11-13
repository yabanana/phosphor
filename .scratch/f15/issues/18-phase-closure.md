# 18: F15 phase closure: residue audit, regressions, measurements and handoff

Roadmap: F15.1–F15.4

**What to build:** Severe residue audit of F15 (roadmap lines, plan, exit criteria, TODOs), full regression battery, Sponza budget re-measured, Apple9 forced path, documentation and the per-ID handoff; roadmap boxes ticked only for verified scope.

**Blocked by:** 14, 15, 16, 17

**Status:** ready-for-agent

- [ ] `ctest`, `metal_syntax_check`, `tools/visual_check.sh` (0 px), F13/F14 corpus runner, `variant_check`, `archive_check`, `hitch_check` green; API validation clean (shader validation where the F14.4 defect does not apply)
- [ ] Sponza D2 GPU p95 A/B/A against the F15 base commit on a quiet machine, within noise; 0 GPU allocations
- [ ] `--force-family apple9` run of the material bench with features
- [ ] ROADMAP F15.1–F15.4 ticked only where verified; approximations and deferred items listed (coloured shadows, dispersion, multiple-scattering compensation, hybrid subsurface if not triggered, non-uniform texture transforms, coated emission of sampled emitters)
- [ ] perf-log, opt-log, RENDERER.md and per-ID handoff (what, how verified, numbers, defects, open items, next steps) written; no push or PR without the owner
