# F13 surface-signal package — SOURCE ONLY / NON VERIFIED

Base `4b80e36f7399f264e4c8cc4c4e5f0607d5793128`, managed worktree
`/Users/danielsan/.codex/worktrees/f13-surface-signals/phosphor`, branch
`codex/f13-surface-signals`. No configure/build/C++ or MSL compilation/test/
renderer/simulator/GPU reference/profiler/benchmark has run. The only check
performed by the writer is `git diff --check`. No ROADMAP checkbox, physical
device acceptance, PR, push or merge is delivered. Stop after this F13 package;
F14 belongs to another writer and no F15/OPT work is authorized here.

## Scope, dependencies and integration owner

F13.1: `reflection_settings.*`, `reflection_common.h`, `reflections.metal`:
GGX solid-angle estimator, RT low-roughness selection, integer-pixel SSR,
directional outgoing-radiance cache lookup at an actual secondary surface,
explicit probe/environment fallback, hit/path/secondary incarnation metadata.
F13.2: WORLD-radius cosine RTAO and numerical GTAO horizon baseline in `ao.metal`.
F13.4: `denoise_settings.*`/`denoise.metal`: per-signal temporal accumulation,
identity/reprojection rejection, raw moments, short-history spatial variance,
bounded à-trous and an independent state check/negative history control.
F13.5: `reflection_probe.*`/`reflection_probes.metal`: static scene capture,
box parallax, normalized GGX mip filtering and deterministic probe fallback.
F13.6: CPU reference and negative controls in `test_reflections.cpp` and
`test_denoise.cpp`; the root owns integrated image/temporal runners and metrics.
F13.3 MetalFX denoised adapter is written separately by the F12 agent.

The root owns Metal host/Engine/CMake/CLI/report/composition and all existing
material shader changes. New kernels are complete source paths, but this
writer worktree alone does not register/build them or create a reachable F13
frame. The root must apply its F9 `rtMaterialFrontFacing(hit,instances,count)`
shader helper before compiling `reflections.metal`: WORLD geometric facing
must be converted with the instance mirrored flag before single-sided
material rejection. No old diagnostic PSO/IFT is reused.

Read-only F12 access needed: cache buffer + graph reference, current
GPUProbeGridParams upload address, probe states buffer/reference, irradiance
and distance-moment textures/references, and GPUProbeTraceExtra. Disabled GI
uses safe dummy resources and REFLECTION_ENABLE_GI unset; current irradiance
is accessed only when the flag is set AND all grid counts are at least2.

## Signal units and composition

All distances/radii/positions are metres. Surface normals are signed WORLD
vectors: geometric normal for W&B ray origin and DI/GI/AO history, actual
material shading normal for SPECULAR history/BRDF. Roughness is perceptual
roughness, not squared roughness. Frame motion is the existing F8 current→
previous INPUT PIXELS, +Y down, unjittered. Depth is reverse-Z; positive guide
view distance is separate from device depth. No normalized texture sampling
uses a larger F8 backing extent: SSR/depth accesses use logical integer pixels.

DI/GI/SPECULAR RGB represent linear outgoing Lo, already carrying the receiving
material factors. Reflection RGB is GGX/Fresnel weighted exactly once; it is
not bare incident radiance. Root removes the old hemisphere SPECULAR term when
F13 reflections are enabled, then adds the reflection Lo once. SSR reads a
stable PRE-REFLECTION radiance image (DI/GI/sun/emissive/residual diffuse),
never the final image containing its own reflection. Guide reconstruction
precedes lighting; no resolve→guide or specular-output→SSR-input cycle.

AO is independent scalar VISIBILITY in[0,1]. It affects only the residual
ambient diffuse when transported GI is disabled. It never multiplies DI,
GI, specular, emission or the whole HDR. The root stores actual material
occlusion in GPUDISurface.pad[0] as a float bitcast when needed for its exact
ambient-only correction; the F13 shader/CPU layouts do not grow that surface.

Hit distance is WORLD primary→secondary distance, strictly positive for a
reached RT/SSR/cache surface. Probe/environment miss and a rejected NDF
hemisphere use distance0 and secondarySlot~0. A miss is not a fake far hit.
MetalFX may disable its optional hit-distance channel for unknown/unsupported
semantics; custom history uses path/secondary identity independently.

## Reflection estimator and fallbacks

Sample GGX NDF half-vector with alpha=max(roughness²,0.002), then reflect-V.
This is explicitly NDF sampling, not the variance-reducing VNDF routine of
[Heitz,JCGT2018](https://jcgt.org/published/0007/04/01/). Solid-angle proposal
is `q=D(H)*NdotH/(4*VdotH)` and receiver weight is `specBRDF*NdotL/q`.
Rejected reflected hemispheres contribute BLACK proposals; they are never
resampled until accepted, which would introduce energy bias. The independent
CPU reference integrates the BRDF over uniform solid angle, not the sampler.

RT probability decreases smoothly between experimental roughness thresholds
0.2/0.4. The chosen RT/SSR route is a stochastic mixture, not summed energy.
RT uses RT_MASK_INDIRECT, closest intersector, WORLD W&B origin and alphaLOD0.
Every hit evaluates current textured material (LOD0), emission, visible local
and physical sun samples, plus indirect DDGI irradiance once when enabled.
Single-sided rejection uses the root material-facing helper; transformed
normals/tangents retain mirrored handedness. A true miss uses probe/environment.

SSR quadratically spaces bounded world-distance steps, brackets a positive
view-distance crossing and refines it before thickness validation. Background,
invalid/offscreen depth and failed brackets become explicit misses. A reached
SSR pixel may use the F12 cache ONLY with its secondary world point, geometric
normal and outgoing direction toward the receiver. Lookup checks full key,
age/generation/revisions through gi_cache_common.h. Cache RGB is outgoing
RADIANCE, never DDGI irradiance masquerading as it. A cache miss uses the
pre-reflection screen image; missing geometry goes to probe/environment.

RT/SSR sampled-ray fallback reads RAW probe mip0 along the GGX sample to avoid
filtering roughness twice. Dedicated `reflection_probe_only` uses prefiltered
roughness mip and the existing analytic split-sum BRDF fit: a declared
approximation, not the exact GGX estimator. It has SPECULAR_SAMPLE_PREFILTERED
and proposalPDF0. These two paths are alternatives. Probe coverage/blend
weights normalize overlapping captures and fill incomplete coverage with the
declared environment. The environment is a parameter, not measured sky data.

The split-sum fit's finite BRDF-integral weight is projected onto [0,1] before
multiplying incident radiance. This is a domain correction for that analytic
approximation only: roughness1 gives B=-.0024, which otherwise makes black
or strongly saturated metals emit negative reflection channels. NaN/Inf fit
weights and invalid radiance remain errors recorded before reduction; there
is no final-radiance clamp hiding them. Independent uniform-solid-angle GGX
quadrature checks black/white metals at roughness .5/1 and NdotV .05/.5/1.
Projection onto the physical interval cannot worsen absolute integral error.
It does not make the fit exact: at roughness1, NdotV1, F0=1 the fit is .45
versus exact 1-ln(2)=.3068528. A CPU-only 256x512 quadrature at roughness1,
NdotV .05 gives approximately .847777 for F0=1 and .032090 for F0=0 versus
bounded-fit .45 and0 respectively. These limitations need image-level quality
decisions separately; they are not passed GPU reference gates.

## AO model and limits

RTAO samples cosine-weighted geometric hemisphere visibility over radiusR,
with any-hit RT_MASK_INDIRECT, alphaLOD0 and WORLD W&B + explicit origin bias.
The CPU radius callback oracle uses the same metre domain; open/half-space
controls and scale changes distinguish radius semantics from pixel radius.

GTAO uses screen-depth height-field horizons, bounded steps within projected
WORLD radius and numerical cosine-weighted angular slice integration. Unknown
screen/depth samples remain unoccluded; coplanar/below-geometric-hemisphere
samples are ignored and radius-edge attenuation is explicit. The formulation
follows the horizon approach in
[Jimenez et al.,Activision](https://research.activision.com/publications/2020-03/practical-real-time-strategies-for-accurate-indirect-occlusion).
It is not a claim of exact arbitrary-scene AO. The `thickness` field controls
a conservative height-field horizon erosion: samples beyond that WORLD
interval after the maximum horizon may relax its cosine. This is an explicit
thin-feature heuristic, not measured back-layer thickness. It needs the thin
wall/halo sweep against RTAO, the visibility oracle/fallback where available.

All radius/ray/slice/step/preset choices are unmeasured. A screen-only height
field cannot know offscreen/back-layer geometry. Cavities, thin walls,
silhouettes, world-scale sweeps and halo ROI metrics remain mandatory tester
work before adopting the GTAO preset.

## Immutable GPU sizes and entrypoint bindings

gpu_types.h is the single scalar-only definition. GPUReflectionParams288 B,
GPUSpecularSample48 B, GPUAOParams176 B, GPUReflectionProbe64 B,
GPUProbeCaptureParams112 B, GPUProbeFilterParams32 B, GPUDenoiseParams96 B,
GPUDenoiseHistory112 B. Signal IDs: DI0,GI1,SPECULAR2,AO3. Reflection paths:
NONE0,RT1,SSR2,CACHE3,PROBE4,ENVIRONMENT5.

| Kernel | Buffers | Textures | Grid |
| --- | --- | --- | --- |
| reflection_ssr | params0,surfaces1,cache13,GIparams14,metadataOUT17,probes18 | preRadiance0,depth1,cubeArray2,rawLoOUT5,distanceOUT6 | width*height1D |
| reflection_rt | above +TLAS2,ownIFT3,instances4,RTmeshes5,vertices6,RTindices7,materials8,DItextures9,GPULights10,sampled11,emitters12,GIstates15,traceExtra16 | above +GIirradiance3,GImoments4 | width*height1D |
| reflection_capture_rt | reflection_rt slots0,2..12,14..16 (NO surfaces/cache/SSR) | GIirradiance3,GImoments4,faceOUT5 | side*side1D |
| reflection_probe_only | params0,surfaces1,metadataOUT17,probes18 | cubeArray2,rawLoOUT5,distanceOUT6 | width*height1D |
| ao_gtao | AOparams0,surfaces1 | depth0,visibilityOUT1 | width*height1D |
| ao_rtao | above +TLAS2,ownIFT3,instances4 | visibilityOUT1 | width*height1D |
| denoise_temporal | denoiseParams0,surfaces1,previousHistory2,optionalSpecMetadata3,nextHistoryOUT4 | raw0,motion1,temporalOUT2,momentsOUT3 | width*height1D |
| denoise_atrous | params0,surfaces1,optionalSpecMetadata3 | previousFilter0,moments1,nextFilterOUT2 | width*height1D |
| denoise_corrupt_history_view | params0,historyRW1 | none | width*height1D |
| denoise_check | params0,nextHistory1,atomicCounts2 | output0 | width*height1D |
| reflection_probe_prefilter | filterParams0 | rawCubeArray0,destinationMip2DArrayViewOUT1 | side*side*6 |

Raster capture `reflection_probe_capture_vs/fs`: captureParams0,vertices1,
instances2,materials3,GPULights4,DITextureHandles5,fullStaticSlots6,sampled7,
emitterRecords8; color0 plus reverse-Z Depth32. Host draws FULL selected
static geometry per face, not the main camera's cull list. This no-RT capture
is an explicitly UNSHADOWED direct/material/environment baseline. Shadow-correct
actual-scene capture uses reflection_capture_rt and its OWN IFT. A cooked or
analytic environment input, when selected by the root, must be labelled as
such rather than actual scene capture.

`reflectionProbeViewProjection` supplies six reverse-Z face matrices with
the Metal viewport Y convention matching `reflectionCubeDirection`; the root
sets appropriate raster winding for that projection. `reflection_probe_prefilter`
reads a DIFFERENT raw mip0 cube-array resource, writes each requested mip
through a 2D-array view (slice6*cubeIndex+face), and normalizes NdotL weights.
Constant input radiance remains constant at every roughness. Metadata.enabled
staysfalse while captures/prefilter are missing/stale; host enables it only
after the matching source generation is complete. Capture/resize/generation,
geometry/material/light/hot-reload invalidation belong to the root, never a
per-frame hidden upload or allocation inside these shaders.

## Histories, variance and graph lifetime

Each active view AND signal owns previous/next history, dimensions, epoch and
signalRevision. Temporal reads previous at the reprojected pixel and writes a
physically DIFFERENT next buffer. Rejection covers background, nonfinite
values, slot/incarnation, material, signal/view/epoch/revision, depth, normal
and world normal-plane separation. SPECULAR additionally rejects roughness,
path, secondary slot/incarnation and world hit-distance disagreement. A camera
cut/resize/DRS/scene change must invalidate every affected view/signal; pool
growth replacing all view buffers invalidates ALL those histories.

Temporal mean/moments accumulate with bounded alpha/history length. The
baseline has `clampHistory=false`: a noisy raw neighborhood is not a bound on
expected radiance or visibility. Exact enumeration of independent Bernoulli
3x3 samples verifies temporal expectation preservation for DI/GI/SPECULAR/AO;
the old raw-extrema clamp alone biases a steady p=0.1 signal downward by more
than 0.03 per update. Identity, geometry, light/material revisions and epochs
still reject stale history. The optional RGB clamp is an explicitly biased
experiment requiring its own energy/quality evidence, not a baseline repair
for motion. Short history uses a compatible3x3 local variance estimate; a
requested clamp acts only on filtered history, never the raw light/reference
buffer. Raw first/
second moments remain separate. à-trous uses5x5 B3 weights and independent
stride1/2/4 outputs (maximum16), normal/depth/luminance/roughness/path edges;
its final output is not feedback history. AO is clamped[0,1]. This is a
lightweight SVGF-inspired baseline, not a claim to reproduce all features of
[Schied et al.,HPG2017](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/).

The root declares every guide/depth/material/texture/cache/probe/history
read/write at its exact stage, plus BLAS AND typed TLAS reads at Dispatch for
each RT consumer. Resource/IFT retirement waits for the last reader, all
allocations use GpuMemory and pipeline requests use PipelineCache. There is
no raw device allocation or manual pass/barrier in this package. Per-signal
history storage is112*w*h bytes PER history buffer and active view/signal;
the host's allocation/active-view policy and reporting must preserve its
memory budget. That formula is layout arithmetic, not measured memory.

denoise_check counts[pixels,valid,invalidStateOrDomain,invalidOutput]; host
clears fourwords and fails on counts2/3. History flags=1 diagnoses nonfinite
source/normal/moments rather than treating black output as accepted. The
negative view-corruption kernel must reset reused history length to1; root
runner must observe that effect, not merely accept its command-line flag.

## Written references and TESTER commands — NOT EXECUTED

CPU checks: rough-metal independent integral1-ln2, GGX sample distribution
including black proposals/missing-support negative; mirrored frame energy;
known SSR plane/background/offscreen; AO world radius/half-space/horizon and
no-GI-double-count; cube direction/raster orientation, box parallax, constant
prefilter energy and bounded split-sum; independent signal/view/epoch/geometry
history negatives, raw moment variance, camera-cut reset, secondary hit/path/
generation and roughness rejection, AO clamps, bounded history and edge weights.
No test has run. Sampling/generation source is project-authored, using local
helpers and published formulas; no external masks/SDK code/assets were copied.
F13 random helpers are explicitly the existing white hash fallback, not
unverified white noise labelled STBN.

Root adds the three new renderer.cpp files, two test files, four.metal files
and reflection_common.h dependency to its build, applies F9 material-facing
helper and its host/grafo integration. TESTER then executes in that checkout:

```sh
cmake -S . -B build/f13-check -G Ninja
cmake --build build/f13-check
./build/f13-check/tests/phosphor_tests --test-case="F13*"
ctest --test-dir build/f13-check --output-on-failure
```

MSL compilation/Metal syntax, consumer IFT/archive/hot-reload/lifetime and
zero measured-frame GPU allocations remain tester gates. GPU mirror/rough
surfaces, offscreen/environment fallbacks, probe volume/mip borders, cavities/
thin walls/open planes, noisy constant/impulse, moving lights/emissive/GI,
motion/cut/resize/two views, new disoccluded pixels, bad history and reference
comparison require the root's final CLI/runner. Freeze ROI/tolerances/preset
before execution; retain raw/reference outputs separately. No performance,
ghosting, quality, physical M3 or phase-completion result is implied. Apple9
is the feature floor; M5 forced paths are functional tests only when run.


Integration correction: static raster probe membership includes hierarchical
children only when their entire parent chain is valid, visible, explicitly
static and free of procedural motion. Membership/cull records are refreshed
on instance/node/material deltas, not only slot-layout changes. Identical
slot lists retain their GPU buffer and argument tables. Probe placement uses
resolved WORLD matrices from the capture frame's exact motion phases; CPU
identity placeholders never define volume or capture position. Static raster
bounds cover its eligible draw list; actual RT capture bounds cover current
visible geometry including descendants of animated roots. The CPU gate compares
a translated/mirrored hierarchy against independent flattened world geometry
and verifies moving-ancestor exclusion plus same-frame bounds changes.
