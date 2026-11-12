# Dynamic Sponza follow-up — 2026-11-05

This is a verified increment toward F13/F14, **not phase closure**. The native
wide-HDR 1% gate and the complete 1080p 16.67 ms frame budget remain unmet.
No resolution, sampling, rejection or image-quality threshold is reduced.
F15 and the unselected OPT/weather candidates remain stopped.

## Correctness and selected cost changes

F13.4/F13.6 previously compared the current receiver depth with a depth stored
in the previous camera space. A camera translation or rotation could therefore
reject the same world receiver. Historical world positions are now transformed
with the current view's depth row before applying the existing relative-depth
and plane thresholds. The scalar shared ABI grows from 96 to 112 bytes; CPU
tests cover translation, rotation, foreign geometry and nonfinite transforms.

The selected optimizations retain Float32, the original 25 taps, tap order,
variance/color weights, temporal feedback and independent GPU state checks:

1. Specialize the existing kernels for DI/GI/specular/AO with function constant
   40. An explicit generic specialization remains a diagnostic control.
2. Cache only the stride-4 geometric guides in an 8×8 workgroup, with a 24×24
   halo. The 13.5 KiB shared allocation fits shader validation. Specular keeps
   its original shading normal, roughness and endpoint predicates.
3. When DI, GI and AO are all selected and the existing preset uses three
   iterations, share their final stride-4 geometry work in one dispatch. Each
   signal retains separate colors, variance, output, history and checker.
   Reduced/Apple9 presets and incomplete signal sets use the original graph.

The single group-size experiment (8 to 32 threads for generic 1D dispatch)
regressed the full frame by 2.61% in a valid pair and was removed. An earlier
guide-cache timing pair drifted 6.14%, above the frozen 5% cap; it is retained
as inconclusive. Neither result is counted as an improvement.

## Dynamic evidence and its boundary

`bench/f13_budget/dynamic.py` freezes the full workload, external glTF asset
hashes, binary/metallib/source identity, controls and unchanged caps before
running. It defaults to planning only and refuses an existing destination.
All GPU work is serial. DwarfStar was allowed to become inactive; it was never
suspended or terminated by this task.

Six API/shader-validated clips contain 384 complete linear frames each at
322×182, including incomplete workgroups. They compare the same binary's
generic/unbatched/reference-stride-4 control with the selected kernels, for
camera-only motion, camera plus a 64-second solar cycle, and forced Apple9
capabilities. Worst peak-normalized errors are 6.13513e-7, 1.36190e-6 and
7.28457e-7 respectively (cap 5e-5). Both locally available official compiler
versions, 32023.883 and 32023.921, produce the same comparison results.

Immutable sidecars contain the actual camera matrices, frame/view/extents,
scene/signal epochs and completed DI/GI/specular/AO GPU checkpoints. Cuts at
frames 119/239/359 reset all four histories, with zero reuse at the cut. The
camera-only sequence actually reuses all signals and has four reset frames
including startup. **The dynamic-sun sequence resets on all 384 frames** under
the existing conservative light-revision policy. These checks establish
identity/rejection behavior and optimization equivalence; they are not a
converged independent dynamic GGX/GI reference and do not close F13.6.

## Timing, native admission and build provenance

The preset remains full Sponza, mesh visibility, ReSTIR DI, solar RT, dense
DDGI 16×8×16 with 64 rays, RT reflections, RTAO, custom denoising,
atmosphere/fog/clouds, native 1920×1080. Each run uses 128 warmup and 256 measured
frames, fixed timestep, no UI/vsync/capture/validation, serial GPU timestamps.
The device is M5 Max 128 GB, macOS 27.2 build 26B123, on AC power, with no thermal
warning reported by `pmset`. Physical lower-tier devices remain untested.

The first three complete A/B/A replicas use runtime compilation: valid baseline
drifts 3.23%, 3.07%, 1.50%; full-frame p50 reductions 11.95%, 12.96%, 13.47%.
Candidate p95 values 30.4090, 31.5311, 29.2943 ms all fail the 16.67 ms target.
All nine runs have zero steady GPU allocations. These are not AOT measurements.
Exact reports, later compatible-archive measurements and regression outcomes
are preserved in the [machine-readable ledger](../results/F13-F14-dynamic-Sponza-M5Max-2026-11-05.json).

With the compatible archive actually loaded, three more A/B/A replicas have
baseline drifts 4.40%, 1.48%, 0.34% and p50 reductions 10.59%, 13.05%, 10.53%.
Candidate GPU p95 values are 29.3456, 30.8008, 30.9616 ms. All nine runs have
zero steady GPU allocations and no archive misses/unavailable requests.
A separate 32-frame API-validated runtime/AOT image pair is bit-identical.
The later native-lifetime whitelist extension is a cold diagnostic-only change:
these custom timing runs request/encode zero native denoised scalers, and the
shader library remains identical. A post-extension Sponza smoke checks that path.

The older compiler's archive compiles but the current OS rejects it; image
equality after that fallback does not establish AOT correctness. The second
official toolchain was already cached locally and is mounted read-only.
Its compiler/translator 32023.921 produces an archive accepted by this driver;
the initial API-validated smoke records 18 hits, zero misses/unavailable and
zero compiler calls. No OS/Xcode installation or modification is performed.
The new optional `PHOSPHOR_METAL_TOOLCHAIN_DIR` routes build, archive and shader
watcher through the same official toolchain; the default still uses `xcrun`.
The mount must remain available for watcher rebuilds.

New pipeline records come from actual Metal serializer runs, including all
four signals and the generic constant. Only superseded denoise compute records
are replaced; unrelated full pipeline identities and RT linkage are preserved.
The replacement operation rejects names without fresh harvested descriptors.
`PHOSPHOR_BUILD_PIPELINE_ARCHIVE=OFF` is only a bootstrap option; the verified
Release build enables archive generation.

The original 96-frame native wide-HDR fixture retains physical target
[368640,128,64], q=1/64, exact manual exposure and the 1% output cap. On the
current runtime all 96 frames encode natively and pass channel checks, but the
80 steady frames reach 5.17578125% restored error. The fixture fails.
Production continues to select custom Float32 before graph creation.

## Native lifetime requalification on the current image

The first current-image lifecycle run rendered 300 native frames successfully
but leaked 20 allocations / 12,800 bytes. The existing guard correctly reported
`signature_matched=0`. Read-only debugger inspection identifies MetalFX 40.9
image `8A5DFA85-CD09-37D8-82BB-7D886AC4F294`, exact class
`_M4FXTemporalDenoisingScalingEffect`, instance size 1248, `_timingRecord` offset
1072 and type `^{CHistoryRecord=fIIff[120f]ff}`. The live allocation is 512 bytes;
the constructor requests 508 bytes at +1496 and stores the zeroed scalar record
through its ivar offset. Neither the inspected `dealloc` nor `.cxx_destruct`
deletes that field. The ordinary-release leak stacks identify that constructor
allocation at +1500. This is fresh image/layout/allocation evidence, not an
assumption based on the unchanged marketing version.

The exact new UUID is added alongside the previously verified UUID; class,
layout, encoding, allocation-size, unchanged-pointer and weak-destruction
guards remain mandatory. Unknown images and capture wrappers still receive
ordinary release. No SDK ivar or method is changed. It remains a narrowly
guarded private-ABI workaround, not a framework fix or production-HDR admission.

The same-binary lifecycle pair renders 300 native frames, four views, three
extents and two generations, with 20 scaler requests. Protected release yields
20 actual weak-nil/reclaimed/matching observations and **0 leaks / 0 bytes**.
`PHOSPHOR_METALFX_DENOISED_PLAIN_RELEASE=1` preserves the numerical results but
reproduces **20 leaks / 12,800 bytes**. A separate direct-spawn Guard Malloc run
actually loads libgmalloc, passes all 300 numerical/channel/guide checks,
reclaims all 20 records and reports no detected memory error. An earlier
attempt through the leaks wrapper did not load libgmalloc and is explicitly
rejected as Guard Malloc evidence. The fixture's internal retirement counters
are never relabeled as destruction proof; process-exit and actual weak-reference
observations remain separate evidence.

## Regression results and retained failures

- Release C++/MSL/AOT and Debug builds pass; all nine CTest groups pass.
- F7/F8: 33/33 functional/readback/negative/image checks pass.
- Independent physical AO: all retained gates pass; steady RMSE p95
  0.0056365923, worst 0.0066107586, flicker 0.000259648, ghost/recovery zero.
- F13/F14 functional controls: 13/13 pass, including two/four views, resize,
  cuts, requested native fallback, and all history/motion/normal/PDF/light/
  overflow/volume negatives. The runner exits 1 for pending broader reference
  evidence; it is not reported as full phase acceptance.
- F14: nine 300-frame runs pass functional and sparse physical checks.
  Reconstructed/full-rate clouds pass retained gates (p95 RMSE 0.0010773261,
  flicker 0.000151810, support ghost fraction 0.06391824, cut recovery zero).
  The full-rate control remains a same-renderer control.
- Shader reload self-test, unchanged-library negative and real watcher
  broken/valid edits pass. RT reload passes before/after the generation change;
  the intentional IFT poison is detected on all three frame slots with exit 1.
  The runner's stale poison anchor was corrected from `rt_common.h` to the
  actual exported intersection body in `rt_intersections.metal`.
- The entire F9 matrix executes: **56/60 pass**, with the original checks and
  zero failures required. `native/apple9-bench8` and `native/apple9-probe-diffuse`
  fail. Both failure types reproduce on the frozen unmodified a83fcf7 binary
  on this runtime, at the same frame/ray with identical diagnostic values.

The bench8 witness is frame48/ray64: the independent reference finds slot9358,
mesh3, primitive22 at t=14.7642250350 with u=1.01818834e-6, while the GPU reports
a miss. The diffuse witness is frame37/ray162: slot3136/mesh51/primitive57 has
world residual 3.52927909e-5 versus the unchanged bound 3.14801330e-5. Neither
failure is waived, converted into ambiguity or fixed by widening a tolerance.
Read-only debugger witnesses and original failed runs are retained. These
current-runtime failures require further investigation; the complete matrix is
not green and the requested closure/merge is not claimed.

## Per-task disposition and remaining work

| Task | Disposition |
|---|---|
| F13.1/F13.2/F13.5 | Existing reflection/AO/probe behavior retained; broad physical roughness, grazing and disocclusion references remain open. |
| F13.3 | Native wide-HDR fails unchanged admission; the current-image timing-record retirement has scoped 300-frame/zero-leak/Guard Malloc evidence. |
| F13.4 | Camera-depth bug corrected and exact-math cost changes exercised on M5; physical T0 certification remains external. |
| F13.6 | Camera/cut/completed-history controls and independent AO regression are separate from the still-missing converged dynamic lighting corpus. Solar changes still conservatively invalidate history. |
| F14.1–F14.4 | Existing atmosphere/fog/cloud/clock paths are regression-tested; sparse physical snapshots remain narrower than general image-quality certification. |
| F14.5 | Nonactivated weather candidate; no later-phase implementation. |

Next work is to qualify the existing dynamic-light history policy against a
frozen independent reference, reduce the remaining measured full-frame cost,
and obtain a native SDK radiometric profile that actually meets the original
cap. None is replaced by a green CPU suite, image equivalence or custom fallback.

Raw data remain in the writer worktree under
`build/f13-f14-dynamic-20261105/`; commands, return codes/EXIT markers and
artifact/patch hashes are retained, including failed and inconclusive trials.

Reproduce the dynamic protocol with fresh output directories:

```sh
python bench/f13_budget/dynamic.py --binary build/closure-release/phosphor --scene assets/sponza/Sponza.gltf --out build/new-dynamic-quality --mode quality --run
python bench/f13_budget/dynamic.py --binary build/closure-release/phosphor --scene assets/sponza/Sponza.gltf --out build/new-dynamic-perf --mode perf --run
```

The budget evaluator uses exactly 1000/60 ms; historical manifests print the
rounded 16.67 ms. Every candidate fails both values, so no result is admitted
by rounding. Stop after F14; no later-phase work or automatic OPT catalog is
included in this increment.
