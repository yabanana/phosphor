# 07: DI reads the lobe guide (ReSTIR and clustered)

Roadmap: F15.1, F15.3

**What to build:** ReSTIR and clustered DI evaluate the shared BSDF with the lobe guide's parameters for extended pixels (target and shading), proven with ior/specular.

**Blocked by:** 06

**Status:** resolved (2026-11-13)

- [x] ior/specular grids pass decision-11 thresholds with `--lighting restir` and `clustered`
- [x] Negative control: DI reading the base BRDF instead of the guide fails the DI grid bound
- [x] Scenes without features unchanged (visual_check 0 px)
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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

2026-11-13 — implementation and M5 verification. Code: 4f9e2ed (Sonnet agent,
reviewed), archive 3d3ed78, consistency fix de9483d, archive_check fix (this
batch). Batch log: `build/f15-exact/battery07.log`.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t07`): GREEN on
  all 12 keys.
  - `specular-grid-point@restir`: worst p99 0.058 %, worst max 0.127 %.
  - `@clustered`: the same values.
  - Every reference control fails, on all keys.
- Negative controls:
  - Before this ticket, `t07pre3`: the base-BRDF DI fails the bound and passes
    against the `specular-ignored` control.
  - Live, `build/f15-check/t07neg`: with `--debug-lobe-guide-corrupt` (+0.25 on
    the guide's ior) the same gate fails, worst p99 517 %, RED. DI really
    reads the guide.
- Scenes without features:
  - F13 exact captures (5 cases) 0 px against base03;
  - visual_check 8/8 at 0 px.
- Archive: +14 compute (the 7 `_lobe` kernels at 1 and 511).
  - archive_check: every local scenario PASS after this batch's script fix
    (the stale scenario compiled the `_lobe` sources twice).
  - hitch_check: 0 hitches, 0 render-thread compiles.
  - RT shade pipelines are compiled and never archived (pre-existing policy):
    a ReSTIR run makes 8 compiler calls, 6 base RT plus 2 lobe RT, on
    workers.
- Deviations:
  - Guide-aware DI kernels are separate `_lobe` entry points included from
    `material_passes.metal`: metal-tt cannot lower constant-free descriptors
    of a kernel that references constant 22.
  - The ReSTIR passes switch together (one BSDF per frame) and only the
    active estimator's variants are requested (de9483d).
  - The DI gate is unfiltered (decision 11 DI amendment); the denoised DI
    is ticket 08.
