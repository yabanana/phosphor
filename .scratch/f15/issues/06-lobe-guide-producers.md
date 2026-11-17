# 06: Lobe guide producers: resolve, guide kernel and overflow fallback

Roadmap: F15.1

**What to build:** A per-pixel lobe guide (layout for all features frozen with static_asserts) is written by the visibility resolve, the guide kernel and the indexed overflow fallback for pixels of extended materials, allocated and written only when the scene feature union is non-empty. Consumers come in tickets 07 and 08.

**Blocked by:** 05

**Status:** resolved (2026-11-16, round 2: the coplanar-overlap fixture for the raster order group is built (`lobe-guide-coplanar`, tools/f15_lobe_guide_coplanar_check.py, 2026-11-15 loader-misc section): with the group 20/20 frames pass for equal and skewed per-fragment cost, the control without the group passes 20/20 at equal cost and FAILS 20/20 with skewed cost (11 148-12 948 of 62 500 px), so the need shows whenever overlapping fragments complete out of submission order; Sponza D2 band recorded as met in ticket 24 under owner option A (720ee63), the checkbox line here still carries the older figure; still partial: per-category report memory for the guide is not quoted (criterion 3 PARTIAL per the closure audit), and the visibility resolve still does not write the guide (the pre-resolve kernel covers every meshlet pixel; deviation recorded in commits and this ticket only))

- [x] Guide content equals a CPU reconstruction of the resolved feature parameters on the ior/specular fixture (readback check with a corruption control that must FAIL)
- [x] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Guide allocated only for a non-empty union (memory per category in the report); Sponza unchanged (visual_check 0 px, D2 within band) — allocation and visual_check done; D2 band → ticket 24 — D2 band MET under the owner's option A (ticket 24: 2026-11-15 median -0.038 ms against ticket 03 + 26fe229, 0 GPU allocations; round-2 quiet rotation t03 -> final -0.059 ms; the final binary's rotation is pending the final battery (ticket 24)); still PARTIAL: the report JSON carries no memory per category (the allocation is logged per category since 8d54b5f, 3 x 158.2 MiB RayTracing at 1080p, none for an empty union)
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
- 2026-11-13 later: regression found by ticket 08b and fixed in 039e22b. In scenes
  without features `di_receiver_pack` left its guide binding (buffer 3) unset,
  and shader validation aborted every DI scene. This battery ran with API
  validation only, so it missed the abort. Lesson: F15 batteries include
  shader-validation runs of an F13 DI fixture.

2026-11-15 — closure audit (ticket 24):

- 1 guide equals CPU reconstruction + control: MET (--debug-lobe-guide 8: 43116 px, unverified 0, PASS; corrupt FAIL). Ticked.
- 2 forced overflow: MET (tools/f15_lobe_guide_overflow_check.sh, 18 432 000 bytes identical). Ticked.
- 3 allocation only for non-empty union, Sponza unchanged: PARTIAL (allocation and visual_check 8/8 0 px, F13 exact 0 px, Sponza +32 768 B = DFG table; per-category report memory not quoted; D2 band not met). Open.
- 4 archive/hitch: MET (b7cfea0, ddbd504; hitch 0), under API validation only; the shader-validation abort (buffer 3 unbound, 6a64b0e) was found in 08b and fixed in 039e22b. Ticked.
Open: no coplanar-overlap fixture proving the raster_order_group need (no owner in 07/08); the visibility resolve does not write the guide (pre-resolve kernel covers every meshlet pixel), deviation recorded in commits only.

## F15 closure, loader-misc (2026-11-15): coplanar-overlap fixture for the raster order group

Open item: "no coplanar-overlap fixture proving the raster_order_group need of `forward_di_receiver_guide_fs`".  DONE (with a finding about what the proof needs).

- Fixture `lobe-guide-coplanar` (bench scenario, src/testbench/material_validation.cpp): ONE draw of 16 exactly coplanar, fully overlapping quads (identical positions, so one depth per pixel; a culled back quad
  gives the scene bounds a z extent, the reflection probe needs it), the vertex normals of layer i tilted by (i - 7.5) x 4 degrees about +X, one coat-only material, sun.  Forced through the overflow receivers
  with `--debug-meshlets 1 --debug-meshlets-corrupt count` (the meshlet self-check FAILs by design, exit 1, like tools/f15_lobe_guide_overflow_check.sh).  All 16 fragments of a pixel pass the equal-depth test,
  the colour attachments keep the last, the guide must keep the record of the same fragment: for a coat-only material the coat normal of the guide IS the shading normal of the fragment, and
  `--debug-lobe-guide 1` compares it with the shading normal of the colour attachments, so it fails when the guide kept another layer.  Runner: `tools/f15_lobe_guide_coplanar_check.py` (4 captures x 20 checked
  frames, 1280x720, 62 500 pixels per frame covered by the stack); new debug options `--debug-lobe-guide-no-rog` (entry point `forward_di_receiver_guide_norog_fs`: the same body, shared function
  `diReceiverGuideBody`, without the attribute) and `--debug-lobe-guide-delay` (`DI_LOBE_GUIDE_DELAY`, gpu_types.h bit 12: a bounded data-dependent delay before the write, up to 16 384 dependent fma, longest for
  the first layer).  Neither is used by the engine.
- **Result**, `build/f15-coplanar` on the M5 Max: with the raster order group 20 of 20 frames pass with equal cost per fragment and 20 of 20 with the skewed cost; the control WITHOUT the group passes 20 of 20 frames when every
  fragment costs the same (a first version of this fixture, 4 coplanar instances in four draws, also passed: the hardware completes equal-cost fragments in submission order, so that control proves nothing) and FAILS 20 of 20 frames
  with the skewed cost, 11 148 to 12 948 mismatching pixels out of 62 500 per frame (the guide of a pixel holds an earlier layer than the colour attachments).  The raster order group is therefore needed whenever overlapping
  fragments complete out of submission order (different materials, textures, cache misses), and this fixture shows it; it is not needed, and not observable, while their cost is equal.  Fixture failed before: the control
  (no group) is the "before" state.
- Criteria were fixed before the delay captures (docstring of the tool): with the group every checked frame passes and at least 32 000 pixels are extended; the control must fail on at least one frame.
- Tests: tests/test_material_validation.cpp "lobe-guide-coplanar is one mesh of 16 coplanar layers ..." (identical positions, distinct normals, union = coat only), tests/test_launch_options.cpp (the two options need `--debug-lobe-guide`).
- The visibility resolve still does not write the guide (deviation of ticket 06): the pre-resolve guide kernel covers every meshlet pixel; unchanged.

### 2026-11-16 — round 3: criterion 3, guide memory per category (orchestrator)

`DirectLighting::reserveGuide` logs the allocation (`Lobe guide: N frame slots x S MiB (P pixels x 80 B), category RayTracing, feature union M`) and the
release for an empty union.  Measured (terrain, 1920x1080, `--lighting restir`, build/int14/guide-mem): the single material (empty feature union) allocates
no guide (no log line); `layers-1` (union 0x100) allocates 3 frame slots x 158.2 MiB = 474.6 MiB in the RayTracing category.  `--lighting legacy` has no DI
passes and allocates none.  The report JSON has no per-category memory section and the startup "GPU memory" line is printed before the first frame graph
(identical 598.8 MiB in both arms), hence the log line.  Candidate, not activated (no measured need): one frame-lifetime guide instead of one per frame
slot would save two thirds.  The visibility resolve still does not write the guide (the pre-resolve kernel covers every meshlet pixel), recorded here and
in the handoff.

2026-11-16 — status reconciliation (closure): the closure-audit carry 'no coplanar-overlap fixture' (ticket 24 list, 2026-11-15) is CLOSED by `lobe-guide-coplanar` (see Status; 20 / 20 with the group, control FAILS 20 / 20 at skewed cost). Criterion 3 stays PARTIAL by owner-recorded decision: the report JSON has no memory per category, the allocation is logged per category since 8d54b5f (3 x 158.2 MiB RayTracing at 1080p, none for an empty union). The D2 wording in criterion 3 now reads the final binary's rotation as pending the final battery (ticket 24). The visibility resolve still does not write the guide (declared deviation).
