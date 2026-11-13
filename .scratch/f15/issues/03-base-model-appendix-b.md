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

## Comments

- 2026-11-13 (orchestrator): shared BSDF now Appendix B (diffuse weighted by
  the dielectric Fresnel F(0.04)); environment/GI diffuse weighted by
  1 − E(0.04) and environment specular from the engine's DFG table instead of
  the Karis fit (the fit under-estimated E at grazing views and by up to 14 %
  at roughness 0.3–0.6, which made the transport furnace gain energy:
  dielectric max 1.076 with Karis, 1.009 with the table). `tools/f13_reference.py`
  declares the Appendix B diffuse weighting (numerically inert for the F13
  fixtures, whose metallic is 0 or 1: no reference clip re-render needed). The
  F12 diagnostic signals stay pure Lambert.
- Shader-validation defect found and worked around: fragment functions read
  the DFG table through `FrameConstants::dfgTable` (evidence in
  `docs/research/2026-11-13-f15-shader-validation-constant-arrays.md`).
- Remaining shared-module duplicates removed (review finding on ticket 01):
  reflection-probe F0/DFG and the probe-prefilter and RT-regularisation alpha
  floors call `bsdf.h`; verified bit-identical on the probes and RT reflection
  captures before the model change.
- GPU: `tools/f15_check.py --run --expect green` → GREEN on M5 (direct grid all
  metallic rows p99 ≤ 0.09 %; environment grid within 5 %; furnace-resolve max
  1.00000; furnace-base max sphere mean 1.00001; every negative control fails).
  Captures `build/f15-check/t03e`.
- visual_check: before/after vs `build/reference-f15base` = intended change
  (PSNR 29.9–55.1 dB, max delta 2–64; metals brighter with the DFG table);
  new references `build/reference-f15-appB` (BENCH8_ARGS="--instances 100000"),
  verify run 8/8 at 0 px, no validation message.
- Pending: F13/F14 corpus on the new baseline (≈ 2.5 h, serial GPU) and the
  Sponza D2 A/B/A on a quiet machine (the first attempt was contaminated by an
  external 1600 % CPU job, sky sentinel 0.74–1.12 ms; discarded).
