# 07: DI reads the lobe guide (ReSTIR and clustered)

Roadmap: F15.1, F15.3

**What to build:** ReSTIR and clustered DI evaluate the shared BSDF with the lobe guide's parameters for extended pixels (target and shading), proven with ior/specular.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] ior/specular grids pass decision-11 thresholds with `--lighting restir` and `clustered`
- [ ] Negative control: DI reading the base BRDF instead of the guide fails the DI grid bound
- [ ] Scenes without features unchanged (visual_check 0 px)
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13: test-first harness in place (scenario `specular-grid-point`,
runner keys `specular-grid-point`, `@restir`, `@clustered`).
- Reference: a white point light with the engine's declared range window, and
  Mitsuba shadow rays for the DI variants (`--visibility rays`).
- M5, `build/f15-check/t07pre3`, `--expect green`: GREEN.
  - legacy (resolve local lights) passes, and every control fails.
  - @restir and @clustered FAIL the extended reference. They pass against the
    `specular-ignored` control, i.e. DI is exactly the base BRDF: red for the
    right reason, asserted by `pending_matches`.
- When this ticket lands, remove `pending` and `pending_matches` from the two DI
  keys; the gate must then pass.
- The DI gate runs unfiltered, with a 2-pixel terminator band (decision 11 DI
  amendment). The denoised DI deviates from the base model by up to 18 % per
  pixel on metals; that is ticket 08.
