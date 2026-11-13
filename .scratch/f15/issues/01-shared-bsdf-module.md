# 01: One shared BSDF module for resolve, DI and reflections

Roadmap: F15.1 (prefactor)

**What to build:** The primary-surface BSDF that is duplicated today in the resolve shading, the DI BRDF/target and the reflection sampling/weight becomes one MSL module that all three call, so a lobe added later is implemented once. Every image stays as it is.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] The three GPU copies call the shared module; no second GGX/Fresnel implementation remains in resolve, DI or reflections (secondary-hit base reduction may keep its own reduced form, documented)
- [ ] `tools/visual_check.sh` 0 pixels against references taken before the change on every bench, API + shader validation clean
- [ ] F13/F14 corpus runner subset touching DI/reflections green with unchanged numbers; any FP-contraction difference is explained and removed, not tolerated
- [ ] Sponza D2 preset GPU p95 within noise of the parent commit (A/B/A, quiet machine)
- [ ] `ctest` and `metal_syntax_check` green
