# F14 writer handoff — NON VERIFIED

Worktree `/Users/danielsan/.codex/worktrees/f14-atmosphere/phosphor`, branch
`codex/f14-atmosphere`, exact base `4b80e36f7399f264e4c8cc4c4e5f0607d5793128`.
No build/configuration, MSL compilation, tests, renderer, GPU probe, benchmark,
profiler, simulator/container or asset download was executed. `git diff
--check` is the only mechanical source check. No ROADMAP acceptance was ticked.
Hard stop after F14. F14.5 rain/snow/wetness/lightning is NOT ACTIVATED because
the F15/F16/F19 consumer capabilities are absent.

## Commits

- c5a8f57: F14 shared scalar ABI and initial physical/resource contract.
- bcc3b7e: portable SI physics, exact revisions, single celestial clock,
  analytic fog and original procedural fields.
- 471c7f9: transmittance/multiscattering/sky-view/physical HDR shaders and
  froxel CSM/RT injection, temporal/prefix integration/composition.
- 0c164d5: procedural cloud shell ray marching, self-shadow, temporal/depth
  reconstruction and configurable bounded history count.
- 9106ead: independent CPU numerical quadrature and authored negative/physics
  tests; reference/low-rate termination separation.
- Final documentation commit completes this source handoff. All are UNVERIFIED.

Exact APIs and kernel argument tables are in F14-CONTRACT.md and headers
atmosphere.h, fog_settings.h, cloud_settings.h. Shared structs: atmosphere384,
fog448/cell48/integrated16, clouds272/history48, volume counters32 bytes.

## Integration ownership

Root owns Engine/CLI/CMake/post, Metal host, pipeline registration/harvesting
and all compilation/runtime evidence. Create GPU resources through GpuMemory,
PSOs through PipelineCache, declare every access/stage in the graph. The fog
RT kernel owns an RtConsumer/IFT from its own resolved PSO and refreshes it on
hot reload; typed TLAS+BLAS refs remain Dispatch dependencies.

Root replaces or consistently updates the scene sun/moon BEFORE SceneRenderer
and F10/F11/F12 preparation from the SAME DayNightClock state. Source linear
RGB values are never exposed/gamma corrected. Solar disk exceeds half range;
F14 physical composed HDR must be RGBA32Float until the root applies exposure
and any permitted scaler conversion. Raw cloud guide is RG32Float. Full-rate
cloud reference disables history and uses native post for raw capture.

AtmosphereVersions compares complete bit tuples. Physics changes rebuild
transmittance+multiple+sky; sun/moon/camera changes rebuild only sky. Clock
jumps reset every lighting/volume history. The shader multiscattering closure
uses an isotropic geometric-series approximation with denominator floor1e-3
for extreme density, not a claimed exact multi-bounce solution.

Fog is world-point scattering. CSM fallback must have declared bounded range
(root selects120m); local/moon visibility requires RT for actual scene shadows.
RT volume rays use no invented surface-normal offset. The complete F11 light
list is sampled through PMF/area-PDF with unbiased weighting. DDGI injection
is six-orientation E/pi isotropic radiance approximation, explicitly documented.

Clouds use an original periodic value-fBm/Worley field with fixed bounded
neighborhoods; no imported generator or cloud asset. Noise/schema/seed/wind
and clock are versioned. Two cloud-shell intervals retain the far sheet on
a tangent ray. Opacity stops at current scene geometry, low-rate ratios1..4
use nearest-depth block coverage, temporal history is wind-advected/reprojected
and clamped to compatible current neighborhoods. Disocclusion upsample fallback
is transparent rather than old cloud content. Cloud self-shadow includes the
complete bounded shell path. Cloud shadows on terrain are not added here.

## Tester steps — NOT EXECUTED

Register portable sources/test files and three MSL sources plus helper headers,
then compile app/MSL, CPU tests and Linux metal_syntax_check. Harvest new PSOs.
Run M5 native and effective Apple9 software paths with API/shader validation;
no physical M3 certification can be claimed. Preserve exact source/SDK/OS,
scene, units, clock, seed, resolution, preset, frame-ring/view manifest.

Use independent CPU quadrature for transmittance/sky/multiple LUT readback:
zenith/horizon/space and vacuum/zero coefficients, declared before comparison.
Written tolerances include analytic vertical optical depth1e-7, midpoint
reference2e-4, phase normalization1e-6/1e-4. GPU-vs-reference tolerance must
be frozen by the tester for LUT interpolation/sample-count differences; it
must not be invented after a failing image.

For fog, homogeneous density verifies T=exp(-sigma*WORLDdistance) and source
integral=(1-T)*source/sigma. Check prefix/readback on froxel columns before
bilinear output interpolation. Place a light behind a wall/alpha caster and
verify actual RT visibility. Compare CSM only within declared coverage. Move
lights/emitters/GI and cut/reset/switch all views; no old signal may survive a
revision. Inspect screen-edge depth/camera slicing and look for banding/halo.

For clouds, compare deterministic full rate against each low-rate/history
preset at frozen clock/camera; preserve linear PFMs through LinearCapture.
Include camera below/inside/above cloud layers, planetary tangent, sun disk,
thin foreground depth and screen borders. Freeze/wind-advect/reverse/jump time,
change density/noise/seed and switch views. Observe actual recovery/coverage,
not only a static metric. Verify cloud extinction remains correct after
low-rate lighting termination so bright solar radiance does not leak.

Negative controls: corruption units multiplies optical depth/extinction1000;
history deliberately consumes mismatched generation and increments independent
invalidHistory diagnostic; light/shadow corruption must be injected and seen
by tester reference/readback rather than merely recognizing a CLI flag. Counter
clear runs before consumers. The parent must make nonzero counters fail its
runner and use a scene where the intended history/light case is active.

No requested F14 capacity is certified or measured by this writer. Presets,
history quality, LUT error, perf/memory/allocations, lifecycle/leaks and final
integration remain tester obligations. No F15/OPT work was started.
