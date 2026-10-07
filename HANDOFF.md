# F13/F14 final SOURCE delivery — NON VERIFICATA

Writer worktree: `/Users/danielsan/.codex/worktrees/f10-f12-development/phosphor`.
Development branch: `codex/f13-f14-development`.
Final immutable delivery ref: `codex/f13-f14-delivery` (created after this handoff).
Initial F10/F12 frozen source ref remains `codex/f10-f12-delivery` at
`4b80e36f7399f264e4c8cc4c4e5f0607d5793128`; its original handoff is preserved
in `docs/implementation/F10-F12-DELIVERY.md`.

The owner authorized SOURCE WRITING + READ ONLY. This writer and its agents
performed NO configure/build, C++/MSL compilation, CPU/GPU tests, renderer,
benchmark, profiler, simulator/container, reference generation, dependency
installation or OS operation. Mechanical source check: `git diff --check` only.
Authored tests/runners are NOT EXECUTED. No roadmap acceptance, push, PR or main
merge is performed. **HARD STOP AFTER F14**. F14.5 remains an unactivated candidate
because its particle/wet-material consumers are absent; no F15/OPT is started.

## Review base and incremental integration

The tester selected source snapshot `2918f78`, integrated it as `4b3303d`, and
subsequently integrated through `411b66b`. These are source boundaries, not
writer-side validation. A second immutable review ref preserves
`4e9773f49225f6c1593b6c2614e92f8f07b1cb6e` as `codex/f13-f14-review-base`.
All additional hooks after those snapshots are revisionable incremental commits.
Source candidate before this documentation commit: `272750b6275ef26121c7ba173cf274b88ec7909d`.

Preserve the tester's owned F9 PipelineCache compiler/archive and
`drainFrameReadbacks`, caster buckets, ICB buffer10 textures, receiver orientation,
shadow capture3/4/5, LIGHTING flushing, DI/GI proposal-chain age and exact primitive
receiver reconstruction. This writer did not redo those diagnoses. Relevant
owner commits include `f4629c8`, `970d7ce`, `41171d0`, `9d49e21` and `838f446`.
Code already carried from the tester must be reconciled once, not applied twice.
Source-origin mappings and earlier corrections are in
`docs/implementation/F13-F14-INCREMENTAL-HANDOFF.md` and
`HANDOFF-REVIEW-CORRECTIONS.md`.

PipelineCache and `docs/ROADMAP.md` are untouched in the writer's full range.
The optional typed denoised gateway is a concrete review artifact:
`docs/patches/f13-denoised-factory.patch`. The owner/tester has independently
applied its gateway; this branch never applies it or creates a second compiler.
Factory injection uses the owned typed request/retire callbacks. Without them,
production selects actual custom fallback and native fixtures report
NOT_EXECUTED_NATIVE. Production nonunit output policy remains Unverified.

## Connected F13 source

| ID | Source and boundary |
|---|---|
| F13.1 | `reflection_settings.*`, `reflections.metal`, `reflection_passes.*`: GGX weighted Lo, RT/SSR/cache/probe paths, own PSO-relative IFT and hit metadata |
| F13.2 | `ao.metal`: world-radius RTAO/GTAO; malformed rays are bounded before hardware trace while independent input diagnostics survive |
| F13.3 | `metalfx_denoise.*`, `denoise_pack.metal`: exact active channels/crop, typed async factory, real SDK encode, per-view/slot ownership, explicit fallback and output-unit experiment |
| F13.4 | `denoise_passes.*`, `denoise.metal`: exact live content epochs, per-view/signal histories, rejection/moments/atrous, always-on numerical checks after actual completion |
| F13.5 | Actual scene RT or unshadowed static indexed capture, including static hierarchy descendants, world placement, roughness mip filtering and sticky validity |
| F13.6 | Independent CPU equations, procedural mirror/roughness/cavity/motion scenes, actual negative controls, PFM ROI/flicker/ghost/recovery metrics and frozen serial corpus |

No full-float signal is subtracted from summed half HDR. Bit32 makes surface
resolve/indexed fallback produce a residual of sun, emission, legacy direct and
optional old specular ambient. A positive RAW Float32 assembly adds raw DI/GI or
ambient for the complete pre-reflection SSR input. Final Float32 composition
adds selected DI/GI/ambient/specular once. Post keeps physical HDR Float32 before
exposure. Relevant connected fix: `f7d837b` + `bd1b8d9`; source diagnostic format
readback fix: `9e9ba9d`. No image tolerance or failure predicate is widened.
The remaining primary residual retains the established surface storage precision.

Actual controls `--debug-reflection-corrupt history|motion|normal` poison produced
foreign-view history or F13-owned copies of consumed inputs, then check actual
state before sanitization. `017acce` + `fc9eb20` connect these paths. The safe old
previous-history rejection control remains separate. Default raw history clipping
is disabled by the tester's expectation-preserving baseline `1b052e6` (carried as
`5910351`); optional biased clipping remains explicitly experimental.

Capture namespace: HDR0, raw indirect diffuse1, direct2; tester shadow mask3,
world position4, world normal5; raw specular6, scalar AO7, actual custom-filtered
indirect diffuse8. Name8 is `indirect-diffuse-filtered`, requires GI + custom.
`3ae662e` + `272750b` export actual giSelectedE multiplied by current guide
albedo*(1-metallic)/pi exactly once. Raw1 is unchanged. Scalar3/7 extract R and
replicate it to RGB from declared R or RGBA storage.

## Connected F14 source and real numerical hooks

| ID | Source and boundary |
|---|---|
| F14.1 | SI metre atmosphere, physical transmittance/multiple/sky LUTs, exact revisions, independent Simpson/Gauss CPU references |
| F14.2 | RT or bounded CSM froxel injection, current sampled world lights, DDGI isotropic adapter, exponential prefix, per-view history and snapshots |
| F14.3 | Original bounded procedural clouds, shell intervals, self-shadow, full-rate control, low-rate temporal/depth reconstruction, declared air/cloud centroid approximation |
| F14.4 | Shared deterministic sun/moon/stars clock before scene lighting, jump resets and display-only exposure hint |

`volume_diagnostics.*`, `volume_oracle.*`, `volume_diagnostics.metal` are real
same-frame hooks, connected by `411b66b` + `c3690d5`, with cache-safe uniform and
current imported-resource fixes `c6381a8` + `db6d159`.
`--volume-oracle DIR` emits immutable per-frame/view JSON, complete expected and
submitted scalar ABI words, actual producer stamps, numerical RGB+T and device
counters. `--fog-homogeneous` sets heightFalloff0 with sigma.01/source(.01,.02,.03)
and exports actual prefix results against metre-distance Beer-Lambert equations.
`--debug-volume-corrupt units|history|light|lut` changes actual source/history or
omits a needed LUT after warmup; an unarmed control is never accepted as exercised.
The solar probe uses the actual shared disk producer for toward/away/tangent
RGBA32, compared to an independent CPU radiance equation above half range.
Sparse numerical tolerances are predeclared exploratory checks, not certification.

`tools/volume_snapshot_check.py` consumes these actual native records, recomputes
solar/fog/epoch/device-counter gates and checks frozen source/binary/manifest
provenance. The serial corpus writes provenance before each case; no manual
legacy samples adapter or fake PASS flag substitutes for GPU data.
Detailed ABI, bindings and tolerances: `docs/implementation/F14-ORACLE-HANDOFF.md`.
Tester-owned aerial endpoint, sky wrapping and unbiased fog-history fixes were
carried as `7bf5d52`, `2f1787c`, `9807ad9` without writer execution.

## Genuine native SDK fixtures and tester entry points

`--denoised-fixture constant|impulse|channels|lifecycle|wide-hdr` with
`--denoised-fixture-output DIR` instantiates the actual owned-gateway adapter.
`--denoised-fixture-pre-exposed` selects an explicit experimental restore policy.
`b7d8d21` + `dd3cbf6` + `2a97a24` connect generated real channels, raster Depth32,
actual packed guides/unit exposure, actual SDK half output and physical Float32
readbacks. `7eeb7b6` requests one owned-cache reload after actual native encodes;
lifecycle requires multiple captured views/extents/generations and cuts/retirements.
Final summary is also in the renderer's schema10 `denoised_fixture` object.

The source runner independently rechecks PFM equations/packed channels, records
both output-unit hypotheses and paired impulse scaling/support. No native filter
identity or unit-energy assumption is made. Submitted retirement is not final
SDK destruction proof. The optional existing macOS leaks-at-exit wrapper records
real process evidence; unavailable/missing/failed checks cannot pass. It does not
certify all private SDK ownership automatically. See
`docs/implementation/F13-SDK-FIXTURE-HANDOFF.md`.

Written tester entry points (NONE EXECUTED BY THIS WRITER):

- `tools/f13_f14_check.py`: default PLAN ONLY; --run consumes an immutable manifest,
  serial GPU lock, actual process exits/validation/readbacks, external independent
  image references, exact PFM matching and unchanged F8 caps.
- `tools/metalfx_denoise_fixture_check.py`: default PLAN ONLY; explicit missing/native
  gateway, source/binary/metallib hashes, constant/impulse/channels/lifecycle/wide-HDR
  native records and optional process-at-exit checks.
- `tools/harvest_f13_f14_pipelines.sh`: serial compiler-recorded engine descriptors;
  no archive was harvested by this writer.
- CMake registers all authored host/core files, shader shared dependencies, CPU
  equation/negative tests and Python independent metrics/fixture/snapshot checks.

Source hooks are concrete. Actual compiler/CPU/GPU, energetic/reference quality,
negative-control exercise, SDK units/lifetime, archive/CI and hardware acceptance
remain the tester's work. Independent reference images are not generated here;
missing evidence stays pending. No stage is marked accepted merely because a
source path or smoke result exists. This writer stops at this frozen delivery.
