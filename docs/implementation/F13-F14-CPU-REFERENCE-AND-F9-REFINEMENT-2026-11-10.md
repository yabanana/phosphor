# CPU reference acceleration and F9 hit refinement, 2026-11-10

F13/F14 remains unaccepted. The owner sequence is unchanged: all retained
gates, merge/push, then F15/F16. This increment addresses reference throughput
and two of the four failing F9 cases.

## Stable reconstruction of returned ray hits

The frozen Sponza diffuse witness has t=6.90264368 from hardware versus
6.902596999387488 from the CPU geometric intersection. Its barycentric
reconstruction misses the actual ray by 3.5292790874e-5 m, exceeding the
unchanged 3.14801329545e-5 m bound. The large, thin triangle makes separately
rounded barycentric formulas ill-conditioned.

A ray-relative plane intersection followed by a pivoted 2D solve now refines
returned surface hits in rt_common.h. Closest-hit consumers in DDGI,
ReSTIR GI, reflections and the diagnostic ray path use the same helper.
Hardware still selects the instance/primitive and performs face/alpha tests.
Misses and visibility-only shadow rays retain the original path. Invalid
generation/mesh/primitive and invalid refinement results retain the raw hit.
The preliminary unpivoted Cramer/Moller variant produced new residual failures
and was rejected; its artifact remains in the trial directory.

The standalone frozen-ray fixture invokes the production helper with the
original ray and hardware hit. Its residual is 4.49953218682e-7 m. The bound
is unchanged; shadow/miss controls are bit-identical. The full F9 matrix
improves from 56/60 to 58/60. Native and forced-Apple9 diffuse probes pass.
Both Bench8 cases still miss frame48/ray64. Disabling culling in the existing
one-triangle edge control does not recover the hit; its interior positive
controls pass. Refining a returned hit cannot recover a missed one.

Nine CTest groups, the independent physical AO fixture, 13 selected F13/F14
functional controls and all nine 300-frame F14 functional/numerical cases pass.
The first F14 run was invalidated by the runner's source-change guard when
an unrelated reference script changed during execution. It is not counted;
the complete second run kept sources fixed.

Full-preset 1080p A/B/A pairs retain 128 warmup + 256 timed frames.
Baseline drifts are 0.88%, 1.27%, 3.58%; all are valid under 5%.
GPU p50 changes are -0.68%, -0.74%, +0.78% (positive means faster).
Candidate GPU p95 values are 22.6879, 23.1582 and 23.4935 ms.
There is no demonstrated performance improvement; the 16.67 ms gate remains
open. The CPU oracle was paused only during these timing runs and resumed
afterwards. DwarfStar was not suspended.

## CPU reference computation

The existing scalar adapter is preserved as default. The optional LLVM CPU
variant evaluates the same texels, wrapping, interpolation, half conversion
and material factors in batches. A separate-process comparison tests seven
semantics on 25 materials at 1,029 UVs, including negative/boundary coordinates:
540,225 scalar values are bit-identical. This verifies texture computation,
not equality of independent Monte Carlo images or GPU material filtering.

LLVM20.1.8 was fetched from Homebrew, its bottle checksum verified, and only
the shared library extracted under build/reference-llvm. Its local zstd
dependency was resolved to the already installed library and the local copy
re-signed. Nothing was installed globally. The receipt records both original
archive and final library hashes.

Primary rays are independently selected and position-validated by the scalar
reference, then cached for vector transport. The full-minus-direct estimator,
seeds 1234/5678, nonzero-support test and 2% convergence cap are preserved.
The frozen run plan records increased checkpoints up to 4,194,304 spp.
The renderer's radiance is not read to choose rays or convergence.
Exact source copies matching the initial runner/adapter hashes are retained
beside the run. A later unused UV helper addition does not change that frozen
source evidence.

## Primary texture footprint

Independent perspective UV derivatives are implemented and tested against
finite differences and vertex permutations. The material audit disables
Mitsuba object merging, so it also checks source instance identity. At frame119,
all 768 primary positions/identities match, with maximum independent UV
difference 1.91034381114e-6.

An exploratory anisotropic box-tap model using the actual exported mip chain
reduces material max absolute RGB discrepancy from 0.14808 to 0.01607 and p95
to 0.00572. This model is NOT adopted or qualified as Apple's hardware filter.
The production sampler is unchanged; material-domain admission remains false.
A diagnostic using shared primary material factors and independent transport
is labelled conditional and cannot certify the full material pipeline.

## Profiling tool limitation

A full 1080p Metal trace with shader sources was captured at frame128.
After the owner opened Xcode, both the original capture and a separate
no-alias capture opened but crashed Xcode on Replay. The reports identify
SIGSEGV at null in GPUToolsPlatformSupport's GTDependencyTracker_processDataSource.
The tool framework reports version2027.0.9. These attempts produced no valid
shader profiler counters. The no-alias configuration was diagnostic only.
No OS/installer change or production aliasing change was made.

## Artifacts and boundaries

- build/f13-f14-f9-refinement: frozen witness, rejected trial, 60-case matrix,
  CPU/AO/functional/volume checks and new Sponza captures.
- build/f13-f14-refinement-perf: all nine measured reports and drift decisions.
- build/f13-f14-llvm-validation: scalar/LLVM checks and early transport trials.
- build/f13-f14-converged-cpu/frame-000119: frozen high-SPP run and exact source copies.
- build/f13-f14-filter-footprint: independent UV and experimental filter audits.
- build/f13-f14-profile-capture and build/f13-f14-profile-noalias: trace captures.
- bench/f9_spike/SURFACE_REFINEMENT.md: standalone regression instructions.

Native HDR is still unqualified; the public-SDK reduction from the preceding
increment reproduces the same 5.17578125% error. No native HDR fix is claimed here.
The remaining edge miss, material/reference acceptance and full frame budget
still block merge and advancement.

## Final reference result and image boundary

The frame119 32x24 crop reaches seed NRMSE 0.00988302253 at 4,194,304 spp
(0.9883%, under 2%). At 1,048,576 spp it was 0.02054896766 and correctly
remained failed. Nonzero signal support is 0.00017621945 RMS.
This is a converged CPU transport estimate for the current material model,
not a qualified end-to-end reference: primary_material_parity_verified and
reference_accepted remain false. Other dynamic frames are not promoted.

As a separate conditional diagnostic, applying the known primary diffuse
material factors to the two estimates yields 1.0058% seed disagreement.
Against that conditional estimate, the new GPU frame119 relative RMSE is
28.136 (approximately 2814%). This shared-input diagnostic does not certify
primary material processing or by itself establish which transport
approximation causes the large discrepancy. No exposure fitting is used.

All five new Sponza frames pass API/shader validation. Their normalized
maximum filtered-GI changes relative to the pre-refinement binary are
5.239e-5, 2.011e-4, 1.879e-4, 3.458e-4 and 5.134e-4 for frames
118/119/120/239/240. These exceed the 5e-5 optimization-equivalence cap.
The hit correction is therefore NOT claimed as an image-equivalent
optimization: it intentionally changes intersection reconstruction, and
independent physical-quality qualification is still required before merge.
The cap has not been changed.
