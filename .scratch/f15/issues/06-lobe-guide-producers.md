# 06: Lobe guide producers: resolve, guide kernel and overflow fallback

Roadmap: F15.1

**What to build:** A per-pixel lobe guide (layout for all features frozen with static_asserts) is written by the visibility resolve, the guide kernel and the indexed overflow fallback for pixels of extended materials, allocated and written only when the scene feature union is non-empty. Consumers come in tickets 07 and 08.

**Blocked by:** 05

**Status:** resolved (2026-11-13; D2 band carried to ticket 24)

- [x] Guide content equals a CPU reconstruction of the resolved feature parameters on the ior/specular fixture (readback check with a corruption control that must FAIL)
- [x] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Guide allocated only for a non-empty union (memory per category in the report); Sponza unchanged (visual_check 0 px, D2 within band) — allocation and visual_check done; D2 band → ticket 24
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — implementation and M5 verification. Contract 44551f6 (layout and
packing); producers 6dd55e4/6a64b0e (Sonnet agent, reviewed); filter tool
ddbd504; archive b7cfea0. Batch log: `build/f15-exact/battery06.log`.
- Readback check `--debug-lobe-guide 8` on specular-grid-sun:
  `pixels 43116 extended 41364 base 1752 unverified 0 | PASS` on every
  checked frame once the producers are specialised. Frames before that print
  SKIP: the guide is not marked valid while the base pipelines serve.
- Negative control `--debug-lobe-guide-corrupt`: FAIL (41364 mismatches),
  exit 1.
- Forced overflow (`tools/f15_lobe_guide_overflow_check.sh`):
  - normal and overflow runs both PASS on 63 lines each;
  - the guide dumps are byte-identical (18 432 000 bytes).
- Scenes without features:
  - F13 exact captures (sponza-d2, cornell-restir, cornell-clustered-ssr,
    manylights-restir, grid-sky) 0 px against base03;
  - visual_check 8/8 at 0 px;
  - Sponza engine resources +32 768 B against 75924b2, which is exactly the
    DFG table of ticket 03: no guide allocated, 0 GPU allocations.
- Archive: +2 compute (`material_lighting_guides` 1/511) and +2 render
  (`forward_di_receiver_guide_fs` 1/511), filtered with `--skip-present`.
  Function labels are metallib hashes, so a plain merge would have
  re-added the 16 ticket-05 pipelines.
  - archive_check: every local scenario PASS.
  - hitch_check: 15 switches, 0 hitches, 0 render-thread compiles.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t06`) GREEN on
  all 12 keys. The two DI keys fail as asserted (pending ticket 07).
- Deviations, recorded in the agent's commit messages:
  - the visibility resolve does not write the guide: the pre-resolve guide
    kernel covers every meshlet pixel, and both call `shadeSurface`, so the
    parameters have one source;
  - a separate receiver fragment `forward_di_receiver_guide_fs` keeps the
    base receiver byte-identical; it uses `[[early_fragment_tests]]` and a
    `raster_order_group` device write;
  - `di_receiver_pack` clears the record of receivers it invalidates.
- Open: Sponza D2 band (ticket 24). There is no coplanar-overlap fixture yet
  to prove the raster-order-group need.
