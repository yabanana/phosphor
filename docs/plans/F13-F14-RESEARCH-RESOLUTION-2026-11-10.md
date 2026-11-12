# F13/F14: researched resolution plan, 2026-11-10

Status: execution plan, not phase acceptance. Baseline source d8a56bf.
Owner sequence: close all retained gates, test, merge and push, then F15/F16.
No permission request is needed to execute the engine work. No source found
guarantees that all four gates can be closed on the current MetalFX runtime.

## 1. Establish the reference domain before tuning image quality

Verified mismatch: visibility_resolve.metal builds primary material guides
using UV derivatives and material_shading.h samples trilinear mips with
anisotropy. tools/f12_reference.py uses bilinear base-level texture samples.
Mitsuba 3's standard bitmap plugin documents bilinear/nearest interpolation,
not the required mipmapped anisotropic footprint:
[official texture documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html).
Increasing SPP alone cannot remove this model discrepancy.

Implementation sequence:
1. Extend snapshot/readback evidence with primary primitive/material identity,
   UV and UV derivatives, mip levels, sampler state, and effective material
   factors, tied to the existing frame/scene hash. Keep GPU material values as
   diagnostic witnesses, not independent material ground truth.
2. Reconstruct primary intersections/UV derivatives independently on the CPU.
   Export actual mip chains and reproduce transfer functions, wrap, factors,
   and filtered-sample half rounding. Use synthetic texture/triangle tests
   before Sponza. PBRT describes footprint-aware mip and EWA filtering:
   [Image Texture](https://www.pbr-book.org/4ed/Textures_and_Materials/Image_Texture).
   EWA is a reference method, not proof of equivalence to Apple's sampler.
3. If exact hardware filtering cannot be reproduced, use a clearly labelled
   conditional transport oracle with frozen material inputs to isolate GI,
   while retaining a separate material-parity gate. This oracle alone cannot
   certify end-to-end F13/F14 or substitute for that gate.
4. Replace scalar Python per-path overhead with a vectorized LLVM CPU adapter
   or a compiled native plugin, only after matched deterministic fixture tests.
   [Mitsuba variants](https://mitsuba.readthedocs.io/en/latest/src/key_topics/variants.html)
   describes the backends; local LLVM availability and 3.9.1 compatibility
   must be verified. A backend switch is not itself a variance reduction.
5. Use direct indirect-path accumulation, NEE/MIS and independent seeds, with
   exact direct/emissive accounting and fixture tests. Preserve the full
   frozen crop, temporal events, nonzero-support requirement and 2% seed cap.
   Do not qualify from the 4x4 diagnostic or from two identically biased runs.

Exit: receiver and material correspondence, convergent independent diffuse
and GGX references for the declared dynamic corpus, then unchanged temporal,
bias, leak and recovery checks. Fix any production errors revealed by these
references before moving to final performance qualification.

## 2. Fix the two F9 numerical failure mechanisms

The missing edge hit and the hit's interpolation residual are distinct.
Camera rebasing has already failed the isolated Metal edge witness.

References:
- [Woop, Benthin, Wald, 2013](https://jcgt.org/published/0002/01/05/):
  robust ray-space edge tests, conservative bounds and difficult-case fallback.
- [PBRT rounding-error analysis](https://www.pbr-book.org/4ed/Shapes/Managing_Rounding_Error):
  bounded reconstruction and ray origins; large arbitrary epsilon is unsafe.
- [Apple acceleration structures](https://developer.apple.com/documentation/metal/ray-tracing-with-acceleration-structures):
  custom intersections and bounding-box geometry are supported.
- [NVIDIA self-intersection analysis](https://developer.nvidia.com/blog/solving-self-intersection-artifacts-in-directx-raytracing/):
  reconstruction ideas are useful; NVIDIA hardware constants are not Metal guarantees.

Implementation sequence:
1. Freeze the recorded edge and residual witnesses, add adjacent/shared-edge,
   grazing, mirrored, scaled and moving instances. Compare to the existing
   double CPU oracle without changing its tolerances.
2. For returned hits, test stable object-space barycentric reconstruction,
   affine accumulation order/FMA and, if needed, robust refinement against the
   same named triangle. Apply a successful fix in all production consumers.
3. For missed triangles, prototype conservative bounding-box candidate
   traversal plus a robust custom triangle intersection on the isolated corpus.
   A triangle callback cannot recover a triangle hardware never reports.
   The paper's double fallback needs an explicitly validated Metal-compatible
   alternative; do not assume FP64 shader support or that watertightness alone
   guarantees an exact hit for the isolated boundary witness.
4. Benchmark the prototype before adoption. It must preserve face/alpha/mask,
   closest-hit/any-hit, instance identity, t bounds and dynamic updates.
   No scene-name or failing-ray-ID special cases.

Exit: full F9 matrix, including all four currently failing native/Apple9 cases,
passes unchanged; production Sponza outputs and frame cost remain measured.

## 3. Resolve the MetalFX contract or prove its external limit

The API says preExposure undoes input premultiplication:
[preExposure](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure).
That statement alone does not establish output units or arbitrary HDR accuracy.
The API defines mask1 as skip denoising:
[mask property](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/denoisestrengthmasktexture).
A [WWDC26 transcript](https://developer.apple.com/videos/play/wwdc2026/359/)
describes the opposite direction. Prefer the property contract and actual
readback; this discrepancy is already recorded in the repository.

Already rejected: skip mask1, Float32 resource declarations, physical q=1,
and packed-color SDK-unit variants. Do not repeat those as new solutions.
Their failures and the original 368640/128/64 fixture remain preserved.

Next action: reduce the existing failing 96-frame case to a public-SDK-only
numeric reproducer with authored textures, verified guide/motion/depth inputs,
SDK output and restoration read separately, plus ordinary-HDR positive controls.
Compare wrapper and minimal sample on the same SDK/image/device. If the minimal
sample passes, port the identified integration fix and rerun the full fixture.
If it fails too, prepare a reproducible Apple feedback package; sending it
requires separate authorization. Do not claim Apple has acknowledged a defect.

Exit: actual native output meets the original 1% per-channel HDR gate plus
channel/impulse/temporal/lifetime controls. Custom fallback is not native
qualification. If the current framework cannot pass, this gate remains open;
a roadmap waiver would change acceptance and is outside the present instruction.
No OS changes, exposure fitting, residual replacement or RGB rescaling to hide failure.

## 4. Recover the full 1080p budget using measured bottlenecks

Current retained p95: 22.69–23.17 ms; target 16.67 ms requires roughly another
27–28% reduction. R32 variance and integer normal-power trials were rejected.

Use an Xcode Metal capture with shader source and per-line costs. Collect
occupancy, launch stalls, register/threadgroup pressure and memory limiters.
Apple distinguishes isolated-pass counters from overlapping workload timing:
[GPU performance workflow](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance),
[counter statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics).
Headless timestamps alone do not identify an ALU or bandwidth bottleneck.
If captures/counters are unavailable, record that gap instead of inventing
hardware causes from pass timings.

Prioritize the recorded reflection pass and aggregate temporal/atrous work.
Choose a targeted change from the measured limiter: repeated geometry fetch,
live-state/register pressure, traversal data layout or necessary pass traffic.
Do not default to more arithmetic substitutions or new scheduling infrastructure.
Retain the selected preset, resolution, rays, histories and Float32 domain.

Exit: three valid A/B/A comparisons under the 5% drift rule, full 1080p GPU p95
at or below 16.67 ms, zero steady allocations and qualified AOT operation,
plus all unchanged image/reference gates. DwarfStar must be inactive; do not suspend it.

## Final integration

Run the complete required CPU, GPU, F7/F8, F9, F13/F14, native lifecycle and
negative-control suites on the final candidate. Re-read roadmap task lines,
exit criteria and outstanding TODOs. Publish exact source/test evidence.
Only then merge and push and start F15, followed by its dependent F16 work.
A research plan, an isolated diagnostic pass or an external bug reproducer
does not close a phase.
