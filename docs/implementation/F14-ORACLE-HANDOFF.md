# F14 sparse numeric snapshots — SOURCE ONLY / NON VERIFIED

This package authors real GPU producer/readback hooks and independent CPU
comparisons. No compiler, test, shader, GPU, renderer, benchmark or profiler was
run by the writer. There are no measured numeric results in this document and no
acceptance claim. F14.5 remains NOT ACTIVATED; work stops after F14.

## Parent integration API

`AtmospherePasses::consumeDiagnostics(u32 completedSlot) -> bool` is the only
new Engine-facing call. Invoke it after the frame completion event reaches
`frameIndex + 1`, before slot reuse and before shader reload. It returns true
when no snapshot is pending; false means an actual numeric, epoch or GPU counter
check failed. In-flight reads and unconsumed slot reuse throw. The parent must
propagate false to the diagnostic run's failure/exit status.

Configuration is read from parent-owned `LaunchOptions`: `volumeOracle`
(immutable snapshot directory), `fogHomogeneous` and `debugVolumeCorrupt`
(0 none, 1 units, 2 history, 3 light, 4 omitted/stale LUT). `debugLighting=N`
selects frames satisfying `(frameIndex + 1) % N == 0`; N must be positive when
diagnostics are enabled. Homogeneous fog requires fog and an output directory.
History corruption requires fog or clouds with temporal reconstruction. Root
owns CLI validation, Engine calls, CMake and the external tester/runner.

Register `src/renderer/volume_oracle.cpp` in the portable core,
`src/platform/metal/volume_diagnostics.cpp` in the app and
`tests/test_volume_oracle.cpp` with the portable tests. The existing shader glob
picks `shaders/volume_diagnostics.metal`. Its transitive shader dependencies are
`atmosphere_common.h`, `renderer/gpu_types.h`, `renderer/volume_noise.h` and the
parent's `renderer/volume_math.h`. Retain the tester's `volumeIntegralFactor`
changes in existing production shaders; this hook does not reintroduce MSL
`expm1`.

Existing `AtmospherePasses::readResources()` exposes borrowed current LUT
textures/graph refs, composed output ref and complete current atmosphere/fog/
cloud parameter values. The caller must declare graph reads and obey completion;
this getter alone is not a CPU readback.

## Actual resources and bindings

All buffers use GpuMemory; all six compute pipelines use PipelineCache. Render
graph imports, reads, writes and dispatch stages govern ordering. No raw barrier
is inserted. The actual global transmittance/multiple producer stamp buffer and
per-view sky stamp buffers persist. A stamp pass is declared only after its
corresponding real LUT producer; merely requesting new parameters never stamps
an old LUT with the expected epoch.

| Kernel | Buffer bindings | Texture bindings |
|---|---|---|
| `volume_lut_stamp` | atmosphere 0, diagnostic 17, actual produced stamps 18 | none |
| `volume_homogeneous_fog` | fog parameters 0, fresh cells 1, diagnostic 17 | none |
| `volume_foreign_fog_history` | diagnostic 17, actual previous fog cells 18 | none |
| `volume_foreign_cloud_history` | diagnostic 17, actual previous cloud history 18 | none |
| `volume_solar_wide_probe` | atmosphere 0 | float32 output 5 |
| `volume_numeric_collect` | atmosphere 0, diagnostic 17, actual global stamps 18, actual view stamps 19, shared samples 20, integrated fog 21, filtered fog cells 22 | transmittance 0, multiple 1, sky 2, solar probe 3 |

`GPUVolumeDiagnosticParams` is 96 bytes; `GPUVolumeNumericSample` is 32 bytes.
Each in-flight slot owns 11 sample records and distinct argument tables. Prepare
refreshes all uniform addresses for that frame, including all three stamp
addresses. Foreign history parameters upload during the current execute call.
Cached graph callbacks resolve slot-dependent fog/history/counter buffers from
PassContext imports; they never capture a frame upload address or first-slot
physical source buffer.

The collector reads two transmittance texels, one multiple-scattering texel, two
sky texels and three actual solar probe texels. Homogeneous mode adds integrated
fog samples at slices 1, half depth and last depth in the central column. There
are no sampled GPU values in the CPU reference path.

The 3x1 solar target is RGBA32Float. The actual production `atmoSolarDisk` helper
is evaluated toward the submitted sun, away from it and along a perpendicular
tangent. The independent reference is `E / (pi * sin(angularRadius)^2)` toward
the sun and zero for the other two rays. It checks physical top-of-atmosphere
irradiance before exposure, tone mapping or half storage. Each solar sample
carries the actual submitted producer's sky revision, independently of the sky
LUT's stored producer revision. The authored solar tests include half saturation,
wrong orientation, foreign epoch and a bright away ray; they were not run.

## Independent references and fixed exploratory tolerances

Transmittance uses portable adaptive-Simpson optical-depth integration. Multiple
scattering uses independent Gauss-Legendre angular quadrature (4 polar nodes,
8 azimuth directions). Sky uses independent single scattering plus Gauss
integration of medium/source transport; needed multiple-scattering grid nodes
are memoized from CPU quadrature, never read from the GPU LUT. These are sparse
exploratory comparisons, not a claim that the approximate closure equals a full
multiple-scattering radiative-transfer solver.

The homogeneous GPU fixture replaces actual fresh froxel injection, using
extinction 0.01 m^-1, RGB source (0.01, 0.02, 0.03) per metre and height falloff 0.
The existing production temporal/integration passes then consume those cells.
Normal fixture mode disables temporal reuse. The CPU reference is analytic
`T=exp(-sigma*distance)`, `L=source*(1-T)/sigma`; distance comes from the same-frame
inverse camera/view matrices and actual logarithmic slice endpoints, in metres.
History corruption deliberately re-enables old-cell consumption when it exists.

| Case | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| transmittance | 0.02 | 0.05 |
| multiple scattering | 0.02 | 0.25 |
| sky view | 0.02 | 0.25 |
| homogeneous fog | 2e-5 | 2e-4 |
| solar disk | 0.01 | 2e-4 |

The predeclared gate for each RGB/T component is
`abs(actual-expected) <= absolute + relative*abs(expected)` with exact expected
epoch equality. Nonfinite values fail. These values are frozen in
`VolumeOracleSettings` before execution, are recorded per case, and do not replace
the tester's wider image/performance/quality acceptance requirements.

## Immutable JSON and provenance

The output is `frame-%06llu-view-%u.json` in `volumeOracle`, schema
`phosphor.volume-oracle.v1`, `kind: f14-volume`. Existing paths are rejected;
snapshots are not overwritten. It contains frame/view, requested/armed fault,
full expected and submitted atmosphere/fog ABI words, physical settings, actual
GPU counters, per-case expected/actual RGB/T and epochs, absolute/relative errors,
fixed tolerances and pass flags. Aggregate maximum errors, epoch mismatch count
and numeric failure count derive from the actual comparisons. `certification`
is always false.

Optional `volumeOracle/provenance.json` is copied as `supplied_manifest`. Its
`source_sha`, `binary_sha` and `manifest` values are explicitly supplied evidence;
they are not asserted to have been independently verified by the writer.
Otherwise these fields are null. GPU name and PipelineCache shader generation
are captured from the active process. `source_hash_at_prepare` is FNV-1a64 of the
listed local producer/helper/reference source files in fixed order at prepare;
it is not a binary or cryptographic hash. Missing source files produce
`unavailable`, never an invented hash. The hash includes the parent's volume_math
helper and portable atmosphere/fog/reference code. The tester must attach its
actual immutable build and binary manifest for acceptance-grade provenance.

## Negative controls and limits

Units corrupt actual extinction/source scaling and force actual LUT producers;
light corrupts actual submitted irradiance/source. Omitted-LUT control waits for
existing LUTs, changes consumer physics/revisions and deliberately omits their
producers. Stored actual producer stamps then disagree with expected revisions.
History control changes actual old fog/cloud view tags and permits the existing
negative path to consume those foreign cells. `invalid_history` increments only
when the shader actually accepts a foreign identity; it is not incremented just
because the control was requested.

`corruption_armed` reports an attempted control after prerequisites existed; it
does not claim that reprojection or cloud visibility exercised that control.
A requested negative run that never produces a failed case or actual mismatch
counter supplies no detection evidence. The tester must warm the cache/history,
use visible overlapping receivers, and reject unexercised negative controls.
Sparse probes do not certify an entire LUT, cloud image, clock sequence or SDK
path. GPU numerical results, positive controls, all negative controls, resize/
lifetime behavior and source/binary provenance remain NON VERIFIED by this writer.
