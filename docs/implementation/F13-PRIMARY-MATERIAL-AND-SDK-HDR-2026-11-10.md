# F13 primary material audit and standalone SDK HDR reduction

This increment implements reference-domain diagnostics and the public-SDK HDR
reduction from the research plan. It does not close F13/F14.

## Actual primary materials and complete texture mips

The snapshot previously exported only mip0, despite the texture manager
already retaining the device-generated mip chain for diagnostic use.
It now exports every retained level, decoded to linear RGB with independent
linear alpha, and records source sRGB status and chain completeness.
Base-level texels remain unchanged. The importer validates level extents,
resource paths and completeness; the digest includes the primary witness.

The snapshot also copies the actual completed-frame GPUDISurface buffer after
its producers. Primary position, instance/generation/material identity,
albedo, metallic and roughness are exported as a GPU witness, explicitly
marked non-independent. Capacity, generation, finiteness and completion are
checked. This does not substitute for independent material ground truth.
UV/derivative reconstruction and a matched filtering model remain outstanding.

Five Sponza captures at frames 118/119/120/239/240 passed API and shader
validation. Each contains 73 textures and 755 levels including the bases.
Every old base RGB/alpha file is unchanged. The captured filtered GI images
are bit-identical to the previous corresponding captures.
All nine CTest groups passed after restoration of the production shader,
including mip-shape/resource-path, stale-generation and exact mip-value checks.

At frame 119, all 768 primary positions in the fixed 32x24 crop agree with
the independently traced CPU positions. Effective diffuse albedo
(base RGB times one-minus-metallic) differs between the GPU primary material
and the CPU base-level material: maximum absolute RGB error 0.1480767,
p95 0.0741768; every pixel exceeds 0.001. These are diagnostic statistics,
not new acceptance thresholds.

An isolated shader control changed only the primary material sampler to
explicit level0, preserving all other arithmetic. On the same frame, the
maximum difference falls to 0.000312924 and p95 to 0.000179175; no pixel
exceeds 0.001. Production filtering was restored immediately afterwards.
Residual sampling/UV/half-precision differences still need analysis.
The control supports texture footprint mismatch as the dominant cause of
this material discrepancy, but does not quantify the GI transport error.

The audit tool is bench/f13_budget/materials.py. It does not admit references,
change the frozen crop or exclude mismatched positions from phase acceptance.
It reports geometry mismatch counts separately. Raw captures, original/control
shader copies, protocols and per-pixel diagnostics are in
build/f13-f14-material-witness.

The Mitsuba PLY loader documents unflipped texture coordinates by default;
no UV-flip workaround was introduced:
[official shape documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_shapes.html).
The diffuse BSDF exposes its actual reflectance through
[eval_diffuse_reflectance](https://github.com/mitsuba-renderer/mitsuba3/blob/v3.9.1/src/bsdfs/diffuse.cpp).

## HDR failure reproduced without engine code

bench/metalfx_denoised/sdk_hdr.mm is a standalone public Metal/MetalFX program.
It supplies the same fixed analytic plane, packed RGB, material factors,
reverse-Z depth, projection, manual exposure and input scaling.
There are no engine shaders, packing/restoration kernels, graph, or private
framework changes. Restoration is a host division by the authored scale.
The actual input color is copied back and verified before SDK execution.

On the current M5 Max/macOS 27.3 runtime:
- Ordinary constant case: 96 encodes, 80 steady frames, maximum relative error
  0.00390625; numerical PASS and raw exit0.
- Original wide-colored-HDR case: 96 encodes, 80 steady frames, maximum
  relative error 0.0517578125; numerical FAIL and raw exit1.

The wide result exactly reproduces the engine fixture's maximum error outside
the engine. It narrows the failure to the supplied public SDK contract/runtime
combination; it does not establish that Apple has acknowledged an SDK defect or
promised the application's arbitrary-HDR 1% accuracy requirement.
It does not measure framework lifetime.

Validation logged framework performance warnings, including redundant pipeline
binding; no GPU/API/shader errors matched the existing error detector.
An initial combined-output log interleaved SDK stderr with buffered stdout.
The final runs retain separate complete JSONL and stderr streams; all 96
frame records and their summary reductions are validated. Raw evidence:
build/f13-f14-sdk-hdr-reduction/separate-streams.
A source-and-log reproduction package is prepared locally and has not been sent.

## Remaining acceptance

Native HDR remains failed. Material correspondence and independent dynamic
transport remain unqualified. The retained production renderer is still
the shared GI/AO final-pass implementation; no new timing gain is claimed.
F9 and the 16.67 ms budget remain open. Merge/push and F15/F16 remain gated.
