# F13/F14 native lifetime and full-frame budget follow-up — 2026-10-08

The owner requires native MetalFX correctness and the full Sponza1080p budget
before advancing. No later phase is started. **This change does not close the
native HDR gate or the 60fps budget.** It fixes a precisely identified native
allocation leak on the verified runtime and reduces measured custom-frame work
without changing resolution, GI/reflection ray counts or spatial filter taps.

## Native allocation root cause and bounded retirement

Device: physical M5 Max128GB, macOS27.2 (26B5091g), MetalFX40.9. The previous
public create-only reductions really destroy their Objective-C scalers: weak
references become nil, yet each leaves an allocation reported as640B by `leaks`.

A read-only debugger inspection traced the exact constructor call at offset
+1496: it requests508bytes with `operator new` and stores them in `_timingRecord`.
The runtime type encoding is `^{CHistoryRecord=fIIff[120f]ff}`: scalar timing
values, including120float samples; no owned pointers or C++ destructor. The
branch obtains the HUD properties singleton. Neither the verified `dealloc`
nor `.cxx_destruct` deletes this field. `MTL_HUD_ENABLED=0` together with
`MTL_HUD_LOGGING_ENABLED=0` still reproduces the leak. These findings identify
the allocation; they do not rely on its size alone.

The bridge now records this pointer at adoption, verifies it has not changed
before release, performs ordinary SDK release after GPU completion on the
existing cache workers, and deletes the scalar record **only after a weak
reference proves the scaler dead**. It does not alter SDK ivars, swizzle methods,
retain scalers forever or restart a helper process.

This is a private-ABI workaround, not an Apple framework fix or a public API.
It is deliberately restricted to all of:

- loaded MetalFX Mach-O UUID `EAB68CBC-376A-3791-BD74-B094FA13AD4E`;
- exact runtime class `_M4FXTemporalDenoisingScalingEffect` and1248-byte layout;
- `_timingRecord` at offset1072 with the complete measured type encoding;
- a matching live allocation and identical pointer at retirement;
- actual scaler deallocation, after the adapter's existing GPU retirement.

Unknown builds, changed layouts, capture wrappers and surviving owners receive
ordinary release. Their lifetime is unqualified until measured; no speculative
free is attempted. `PHOSPHOR_METALFX_DENOISED_PLAIN_RELEASE=1` forces the leaking
negative control. The old F8 temporal self-reference logic remains separate.
There is no per-frame reclamation work or new GPU dispatch.

A standalone four-object pair reports2560B with ordinary release and0B after
verified record retirement. The actual engine fixture renders300native frames,
four views, three extents and two pipeline generations, creates/retires20scalers,
passes its channel/numerical checks and reports **0 leaks / 0 bytes**. Raw
allocation/disassembly/control evidence stays under `build/native-budget-closure`.
The same300-frame lifecycle also passes with Guard Malloc plus API/shader validation, with all20records actually reclaimed and no process error. Four final constant/scaled/impulse/channel cases pass; SDK preExposure property observations are saved per completed frame, including scale transitions.
The ordinary public reduction in `bench/metalfx_denoised/sdk_create_only.mm`
remains unchanged, so the original SDK defect can still be reproduced.

## HDR remains a separate failure

The original physical target[368640,128,64], q=1/64 and1% per-channel error cap
are preserved. Earlier public mask/Float32 alternatives still failed at5.1758%.
The new control removes the remaining HALF input restriction entirely:
RGBA32Float input/output, physical input with q=1, exact R16 exposure23/2^24,
real packed-channel readback and96actual native encodes. The SDK output still
saturates red at65504 and badly perturbs green/blue. The test fails; Float32
resource declarations do not establish Float32 internal dynamic range.

Only the explicit wide-HDR fixture can enable
`PHOSPHOR_DIAGNOSTIC_METALFX_FLOAT32=1` plus
`PHOSPHOR_DIAGNOSTIC_METALFX_PHYSICAL=1`; it also requires manual exposure.
The packer selects a Float32 limit only for that actual format, preserving
HALF overflow detection elsewhere. No tone-map/inverse compensation, channel
remapping, residual substitution or tolerance relaxation is applied.

Production native admission therefore remains disabled before graph creation;
`--lighting-denoise metalfx` selects the already declared custom Float32 fallback.
Fixing lifetime is not permission to render out-of-contract native HDR.
A second explicit unit control keeps q=1/64 in packing/restoration and the
same exact manual exposure, but sets the SDK preExposure property to1. Its
getter is recorded after assignment. That hypothesis also fails: relative
maximum1.8046875 (180.47%), compared with0.0517578125 (5.18%) on the preserved
original path. `PHOSPHOR_DIAGNOSTIC_METALFX_PACKED_UNITS=1` is fixture-only;
it is not adopted as an exposure mapping or product path.
Apple's [pre-exposure](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure)
and [exposure texture](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/exposuretexture)
contracts are retained as the unit reference. No claim that Apple has acknowledged
or fixed either issue is made.

## Selected performance changes

1. Cache the unchanged5x5 atrous stencil in threadgroup memory for strides1/2.
   Values remain Float32; tap order, weights, temporal history and rejection
   rules are unchanged. Full128-thread groups cooperatively load20x12 or24x16
   tiles, then bounds-check output after the barrier. The largest tile is15360B,
   which also fits when shader validation instruments its memory. Validity uses
   a distinct metadata bit; a compile-time assertion prevents flag collisions.
2. Adapt **short surface aerial quadrature** to distance and the shortest
   atmospheric scale height, with at least8samples and128samples per scale
   height, capped by the original budget. LUT/sky/ground-boundary integrals and
   long rays retain their previous budgets. This is checked against the original
   quadrature, not assumed exact. The frozen 1% image cap remains unchanged.
3. In DDGI mode, retain only one safe unused element in each of the13 per-pixel
   reservoir bindings. DDGI never reads them. Cache/ReSTIR modes keep their full
   reservoirs. Skip the unused cache update in non-diagnostic DDGI frames;
   diagnostic cache invariants/corruption cases remain active. This saves exactly
   3,450,468,736 logical resource bytes at1080p, with zero steady GPU allocations.

Rejected experiments remain visible: widening generic1D dispatch groups and a
power64 multiplication chain had no useful measured gain; large stride4 shared
tiles exceeded the validation limit; SIMD row exchange and sparse sublattice
variants were correct after fixes but did not improve the wider filter. The
selected code keeps the original stride4+ kernel. An initial experimental
validity bit collided with the specular error bit; the image oracle rejected it,
and the fixed code has an explicit disjoint-bit assertion. No failed experiment
is an acceptance result.

## Verification and remaining budget

`bench/f13_budget/quality.py` freezes and reproduces the same-binary reference,
stencil-only and stencil+quadrature comparisons. Nine API/shader-validated
runs cover32complete frames each at322x182, including incomplete workgroups,
at noon/sunrise/night. The selected stencil's peak-normalized worst error is
5.34e-7; the full adaptive case's worst relative image error is0.000624721 with
its declared peak-based floor. Both pass the predeclared caps. A separate full
1920x1080 pair has relative maximum1.78527e-5; forced Apple9 has1.90325e-5 (capability simulation on M5 only). A separate production/AOT old-binary-versus-selected pair, with debug cache maintenance disabled, passes at2.26106e-5. These comparisons complement,
rather than replace, independent physical references.

Nine CTest groups pass. The independent physical AO sequence passes. Requested
native fallback, four-view/resize lifecycle, F13 history/motion/normal negatives
and F14 units/history/light/LUT negatives pass their functional gates. The
existing runner's broader reference corpus remains explicitly pending.

Timing uses the original complete Sponza command: mesh visibility, ReSTIR DI,
solar RT, dense DDGI16x8x16/64rays, RT reflections, RTAO, custom denoising,
atmosphere/fog/clouds, native1920x1080,128warmup+256measured frames, fixed timestep,
no UI/vsync/validation/captures and serial GPU timing. Baseline executable,
metallib and archive were copied together from the PR20 build before changes.
Repeated A/B/A series and a conditioned-device repeat preserve every report.
There is meaningful baseline drift, so the ledger distinguishes valid pairs
from rejected timing pairs. The selected workload is cheaper, but **16.67ms is
not reached**. The selected runs span p50=29.4446–35.7503ms in the first
series, and31.0503–33.4688ms after conditioning. Of six A/B/A comparisons,
three satisfy the frozen5% baseline-drift cap; their reductions are26.90%,
31.40% and25.73%. The other three timing pairs remain rejected. This supports
adopting the cheaper implementation, not a stable60fps claim. Neither an
isolated fastest run nor the custom fallback closes this remaining budget.

See [machine-readable evidence](../results/F13-F14-budget-followup-M5Max-2026-10-08.json)
for exact times, drift decisions, checksums and raw paths. The remaining work is
native wide-HDR qualification, reliable sustained frame-budget closure and the
already declared broader dynamic GGX/reference coverage. F15 and subsequent
phases stay stopped; no OPT catalogue is marked completed by this patch.
