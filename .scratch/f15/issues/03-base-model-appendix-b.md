# 03: Base model conforms to glTF Appendix B (owner decision) and baselines are re-taken

Roadmap: F15.1, F15.3 (prefactor)

**What to build:** Owner decision 2026-11-13 (ADR-0006): the shared BSDF module implements glTF 2.0 Appendix B — dielectric `mix(diffuse, specular, F(0.04))`, metal `specular · F(baseColor)`, `mix(dielectric, metal, metallic)` — and the environment/GI diffuse is weighted by the dielectric directional Fresnel, in resolve, forward, DI, reflection weights and composition. The F12 diagnostic signal `gi_reference_diffuse` stays pure Lambert so the Mitsuba Cornell oracle remains valid. Every baseline that depends on the model is re-taken deliberately with before/after evidence.

**Blocked by:** 01, 02

**Status:** ready-for-agent

- [ ] Ticket-02 base grids (all metallic values) and furnace pass decision-11 bounds; negative control (old weighting) fails
- [ ] visual_check references re-taken; before/after diff per bench documented in the closure log (expected change: intermediate metallic and ambient/GI diffuse)
- [ ] Declared material model of the independent references updated where they model the engine BSDF (`tools/f13_reference.py` and the clip references); affected Mitsuba clip references re-rendered with battery pacing; F13/F14 corpus green on the new baseline; F12 oracle unchanged
- [ ] Sponza D2 GPU p95 re-measured A/B/A (within the decision-11 band; cost change expected ≈ 0); 0 GPU allocations
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
