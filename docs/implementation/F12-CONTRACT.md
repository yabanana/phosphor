# F12 implementation contract — NON VERIFIED

Written against base 7997f12713f07895bdf27d7ab27f7947f3da7bd7. No compiler,
MSL compiler, test, renderer, reference generation, benchmark or profiler was run
by the writing agent. No acceptance, performance, quality or hardware claim is
made. ROADMAP remains untouched. Root owns GpuMemory/PipelineCache/render-graph,
view lifetime, Engine/CLI/report integration and all execution.

## Signal and integration boundary

Guides precede the material resolve and come directly from depth/V-buffer:
RGBA32Float WORLD point and WORLD GEOMETRIC normal, w > 0 valid. They cannot
depend on an already lit HDR result. Output is linear RGB INDIRECT IRRADIANCE
in RGBA32Float (RGBA16Float after a measured precision comparison only).
The resolve applies diffuseAlbedo / pi once to this signal. Direct light,
analytic ambient/specular environment, primary visible emissive and total HDR
are independent signals and are never scaled by GI.

DDGI ray payload irradiance integrates secondary REFLECTED radiance: secondary
Le is removed at the first segment because F11 already samples direct emissive.
Emissive energy reaches indirect transport via shadow-tested NEE at secondary
surfaces and their diffuse bounce. Cache entries retain FULL outgoing radiance
(Le + reflected) so a future reflection consumer can use their radiometric
meaning. The GI candidate removes secondary Le before forming its target.
Direct environment misses likewise contribute zero to the indirect signal;
configured sky lights SECONDARY surfaces through a visibility-tested cosine ray.
The engine's analytic hemisphere is not a physical exported sky; root must
explicitly choose/configure the same physical sky for a reference experiment.

The secondary material reduction is deliberately diffuse:
baseColorFactor * half(filtered base RGB) * (1 - clamp(metallicFactor *
half(filtered MR.b))), Lambertian / pi; same emissive factor and half(filtered
emissive texel), LOD0, MASK alpha via F9. Normal maps/specular/caustics are
outside this diffuse GI estimator. The original full material is still exported.

## Resource and shader ABI

Shared layouts appended to gpu_types.h:

| Type | Size | Meaning |
|---|---:|---|
| GPUProbeGridParams | 160 B | 10 scalar blocks, grid/preset/scene/view/cache epochs |
| GPUProbeState | 32 B | bounded WORLD offset, state, generation, active age |
| GPUProbeRay | 32 B | unit direction, signed distance, reflected linear radiance |
| GPUProbeTraceExtra | 32 B | sampledLightCount, cacheUpdateCount, cacheCandidateCount, frameSeed, sunAngularRadius+pad |
| GPURadianceCacheEntry | 64 B | full spatial/normal/direction key, RGB radiance, age/revisions |
| GPUGiReservoir | 128 B | secondary area sample, source guides, sum/M/W, revisions |

All textures and buffers originate in GpuMemory. No shader allocates storage.
Use original/full geometry for quality validation; F9 proxy manifests are
corpus measurements and cannot certify GI. TLAS/BLAS are graph reads at Dispatch,
not unchecked pointers to an F9 diagnostic dispatch. Every RT PSO links the
single rt_alpha_generic intersection function and owns its own IFT, rebuilt
when that PSO changes. New consumers define PHOSPHOR_RT_NO_ALPHA_FUNCTION before
rt_common.h, relying on root's guard around its function definition.

Each RT IFT slot 0 binds the existing F9 material0/texture1/vertex2/index3/
instance4/RTMesh5/RtParams6 table. Readonly per-slot AS/instance/mesh/range views
must remain valid until the graph consumer completes. Preserve typed tlasRef,
BLAS dependency and deferred lifetime; do not substitute diagnostic IFTs.

| Kernel | Buffers | Textures |
|---|---|---|
| ddgi_trace | AS0 params1 states2 rays3 IFT4 instances5 RTmeshes6 vertices7 RTindices8 materials9 textureTable10 analyticLights11 cacheCandidates12 sampledLights13 extra14 emissiveRecords16 | previousIrradiance0 previousMoments1 |
| ddgi_classify | params0 states1 rays2 | none |
| ddgi_blend | params0 states1 rays2 | previousIrr0 previousMoments1 nextIrr2 nextMoments3 |
| ddgi_resolve | params0 states1 | WORLDpoint0 geometricNormal1 currentIrr2 currentMoments3 outputIndirectE4 |
| radiance_cache_update | params0 cache1 candidates2 extra3 | none |
| gi_candidates | AS0 params1 states2 output3 IFT4 instances5 RTmeshes6 vertices7 RTindices8 materials9 textureTable10 analyticLights11 sampledLights13 extra14 cache15 emissiveRecords16 | WORLDpoint0 geometricNormal1 currentIrr2 currentMoments3 |
| gi_temporal | AS0 params1 fresh2 previous3 output4 IFT5 instances6 | WORLDpoint0 geometricNormal1 currentToPreviousMotionPixels2 |
| gi_spatial | AS0 params1 input2 output3 IFT4 instances5 | WORLDpoint0 geometricNormal1 |
| gi_shade | params0 reservoirs1 states2 | WORLDpoint0 geometricNormal1 currentIrr2 currentMoments3 outputIndirectE4 |

F11 dependencies are restir_common.h, GPUSampledLight 80 B, GPUEmissiveSurface
80 B, DITextureHandle 8 B and diSampleTexturedLight. Emissive records are indexed
by light index, actual current material UV/half-filtered emissive/MASK sampling
matches F11. When sampledLightCount > 0, analyticLights supplies directional
suns and the helper skips analytic point/spot entries to prevent duplicates.
The sampled list is complete: area/punctual/emissive. Its simple proposal is
uniform 1/N, divided out along with area PDF. F11 alias selection can replace
uniform selection only while keeping the actual proposal PMF in the estimator.

TraceExtra.sunAngularRadius is the same F10 physical solar disk radius, radians,
0 for an exact directional delta. Secondary sun NEE samples uniform solid angle
in that cone, with normalization2/(1+cos(radius)) preserving GPULight's
perpendicular irradiance. Offline snapshot retains the radius; the reference
uses a distant physical disk at1e6m with Le=E_perp/(pi*sin(radius)^2). That
finite-distance approximation and penumbra/unit agreement need tester evidence.

## DDGI estimator, atlas and relocation

All lengths are metres. Baseline experimental preset: 8x4x8 probes, spacing2m,
64 Fibonacci sphere rays/probe, irradiance6 and distance14 interior texels,
distance cap100m, hysteresis0.95, normal bias0.02m, backface threshold0.25,
min front distance0.1m, relocation step0.1m, maximum offset0.45 spacing.
These are NOT measured/adopted settings. Portable config checks counts,
positive dimensions/finite scalars, total probes <=65536, ray count8..4096,
atlas dimension <=16384 and bounded allocation arithmetic.

Each probe is a tile: x=probe%countX, y=probe/countX. Irradiance atlas size is
countX*(irrTexels+2) by countY*countZ*(irrTexels+2); distance atlas analogous.
The one-texel border is seam-folded octahedral topology. Kernel blend dispatch
covers the maximum dimensions of both atlases and writes each border directly.
Use distinct previous/next resources, initialized/reset state buffers, and
graph ordering trace -> classify -> blend -> resolve. Trace sees stable OLD
offsets/states; classify may relocate; blend discards relocated/inactive probe
history. Re-trace inactive probes in subsequent frames until they recover.

For uniform sphere quadrature:
E(n)=4*pi/N * sum_i L_reflected(i)*max(dot(n,omega_i),0).
Moments use weights max(dot(n,omega_i),0)^50 normalized by sum of weights.
Backface distances are signed, misses use maxDistance. Visibility is cubic
Chebyshev variance bound outside the mean, exact1 inside the mean. This
variance bound/wrap weighting/trilinear spatial interpolation are explicit
DDGI approximation biases, not exact occlusion. Atlas history is not read at
all when reset/age<=1, avoiding 0*NaN from uninitialized prior textures.
Within-cell eight probes use relocation-adjusted positions, moments toward
the WORLD point, geometric-normal wrap, and trilinear weights. Outside grid
or with no valid active probes returns zero.

More than the backface threshold marks a probe inside geometry. It moves
toward the closest backface exit; probes close to a frontface move away.
Offset normalized by spacing is clamped to maxRelocation. A relocated probe
is inactive/age0 for this update, not allowed to reuse rays from its old
position. It must be traced again at the new location. Scene/light/material
changes reset generation and atlases; root increments these revisions for
texture edits, intensity toggles, transforms and structure changes.

## Bounded radiance cache

Key: floor(WORLD point / cellSize), oct16 normal bin, oct16 outgoing direction
bin, plus exact geometry/light/material revisions and cache epoch. The shader
uses cellSize=min(spacing)/4 (0.5m in the baseline); CPU config exposes the same
value. Compare full keys after hashing. Open addressing examines at most
probeLimit entries and chooses stale/empty lanes first, otherwise oldest
entry in that window. Epoch reset physically clears storage so wrap/reused
generation cannot resurrect old values. Modular ages are valid only while
maxAge <2^31. RGB is finite/nonnegative, sample count bounded (shader64).

Correctness-first GPU update dispatches ONE lane, at most256 rotating samples
per frame, bounded probing. This intentionally serial baseline has no speed
claim and must be included in GI budgets. Graph writer/reader ordering gives
complete values without RGB atomics, lock publication races or spin loops.
Performance experiments can parallelize it later, after functional evidence.

Cache values are outgoing radiance at the SECONDARY hit toward the primary
receiver, never primary radiance added as GI. A miss computes secondary
Le+shadowed direct Lambertian+DDGI indirect diffuse. A lookup hit subtracts
current Le before GI shading. Quantized spatial/normal/direction filtering is
an approximation: nearby materials/thin surfaces can share a key and require
leak/error validation. DDGI remains selectable even if cache is rejected.

## GI RIS and reconnection

Candidate ray is cosine hemisphere: q_omega=cos_x/pi. After a hit at y,
q_area=q_omega*cos_y/r², with cos_y=max(dot(n_y,-wi),0).
The shader proposal is measured from the ACTUAL W&B-offset ray origin/direction;
target evaluates the true reconstructed geometric receiver. This accounts for
the changed endpoint sampling density instead of silently replacing it with
the receiver geometry's density.
Area integrand f_area=L_reflected(y)*cos_x*cos_y/(pi*r²).
Target is its RGB luminance. Fresh reservoir weight is target/q_area;
W=sumWeights/(M*selectedTarget); output E_indirect=pi*f_area*W. Primary
diffuseAlbedo/pi is applied later, once. Fresh one-ray result is pi*L_reflected.
Zero-contribution/backface/occluded paths count in M; they must not become a
conditional positive-path estimator. A candidate with M>0 but no weight shades
zero; it does not fall back to a nonzero DDGI result to hide the rejected path.

Reconnection preserves a fixed WORLD-space secondary endpoint and AREA measure,
so area->area Jacobian J=1. Dynamic geometry/light/material/global revisions
and secondary instance generations invalidate reused paths; no unsupported
deformation Jacobian is invented. Each shift re-traces RT_MASK_INDIRECT with
any-hit alphaLOD0 and WORLD W&B endpoint bias. Temporal reprojection uses the
existing current->previous PIXEL motion; stored source WORLD point/normal
reject disocclusion. Four adjacent spatial neighbors use distinct IO buffers.
History age <32, history M contribution <=32, normal cosine0.95 and spacing-
relative position tolerances are declared experimental. ViewRevision is the
per-view signal epoch; root must not share previous reservoirs across views.

Basic reuse weight = target_receiver(y)*W_source*M_source;
final W=sum(w)/(M_total*selectedTarget). This M normalization has support/
correlation bias when source visibility domains differ. It is explicitly an
experimental BIASED ReSTIR GI baseline, not the unbiased GRIS/MIS estimator.
Fresh-only (cache path with temporal/spatial omitted), and DDGI, are available
comparison/fallbacks. No cache/GI quality closure is claimed without reference.
F13 owns complete denoised/temporal output acceptance; F12 exports the noisy
indirect signal and exact history dependencies.

## Independent reference before F32

exportOfflineReference consumes full raster mesh/index/UV/normal streams,
current materials/lights, explicit same-frame GPU WORLD instance matrices,
camera per view including jitter, and exact LINEAR RGBA texture texels. Missing
texture texels fail export; substituting white/default or local/base transform
is forbidden. Root captures/retains texture texels before upload or reads them
after completed GPU work. This portable component does not read GPU resources.
ReferenceAreaLight copies sampled light plus optional emissive material/UV data;
root converts GPUEmissiveSurface fields, never binary-reinterprets different ABI.

Export writes full PLYs, unexposed Float32 linear PFM texture maps and scene.json,
validates all ranges/scalars, refuses overwrite and publishes by sibling staging
rename. The snapshot retains original PBR/normal/occlusion/alpha/emissive values,
full WORLD transforms, IDs/generations and revisions. No reference GPU image is
generated by the writing agent. F32 does not yet exist as a validated oracle.

tools/f12_reference.py uses externally installed Mitsuba3 scalar_rgb, never
installs packages. A custom snapshot texture evaluates bilinear repeat LOD0 and
half arithmetic before factors, including an alpha threshold AFTER filtering.
Diffuse mode is the F12 material reduction; principled mode is explicitly
different from engine GGX and returns a nonaccepted reference. Infinite far
projection, fovY, current +Xright/+Ydown jitter feature displacement and WORLD
camera match the snapshot; UV orientation, plugin ABI, camera jitter signs,
units, integrator depth decomposition and mapped models need tester validation.

Punctual quartic range attenuation, spot interpolation and two-sided emitter
differences refuse a strict reference unless --allow-model-differences is
explicit; an override is always NON_ACCEPTED_REFERENCE. Mesh emissive
triangles in the F11 sampling list are not emitted twice: actual full mesh
emission is retained. The current script's independent path tracer can export
total radiance or indirect diffuse reflected radiance (depth8 minus depth2,
signed Monte Carlo values, never clamped). Linear Float32 PFM/EXR only.

Convergence checkpoints64/256/1024/4096, fixed seed and scene SHA256 are saved.
The default2% relative difference threshold is an experimental predeclared
criterion, not an observed result. Named regions compare signed bias, leak
and RMSE; global PSNR alone is insufficient. Compare the same SIGNAL, not raw
irradiance against a radiance reference. F13 will add temporal denoise quality.

## Tester commands — NOT EXECUTED

Root adds probe_grid.cpp/radiance_cache.cpp/offline_reference.cpp to core,
test_gi.cpp to tests, shader modules/helpers to shader lists and harvest. Then:

- Configure/build portable targets and run ctest (Debug/Release).
- Build Metal host + MSL against pinned local metal-cpp, metal_syntax_check,
  Metal4 linked alpha PSOs/own IFT hot reload, API/shader validation.
- Read back probe directions/distances/RGB/atlas borders/moments/states, cache
  keys/RGB and RIS q_area/M/W against portable oracles for declared FP tolerance.
- Cornell diffuse/emissive, empty scene, thin walls, probe inside wall, moved sun,
  emissive on/off/motion, disocclusion, camera cut, resize, view2/4, scene switches,
  cache full/collision/eviction/repeated epoch/reset, lifecycle/leaks.
- Each negative must FAIL its acceptance predicate: omit visibility moments,
  omit classification/relocation, wrong bias, stale light/material/geometry
  revision, foreign view history, ignore full cache key, mix q_omega with q_area,
  remove area-PDF/selection-PMF, retain first-segment Le in GI.
- On exact exported snapshots with already installed reference dependencies:
  python3 tools/f12_reference.py plan --output <corpus.json>
  python3 tools/f12_reference.py render <snapshot> --output <reference-dir>
  python3 tools/f12_reference.py compare <ref.pfm> <engine-indirect-radiance.pfm>
  --regions <frozen-regions.json> --report <quality.json>
  --max-relative-rmse <frozen-threshold> --max-relative-bias <frozen-threshold>
- Measure actual integrated frame incl DDGI trace/classify/blend/cache update/
  RIS reuse/shade, memory and required denoise, warmup/steady state, 3 replicas.
  Native and effective Apple9 on M5 Max128GB; no physical M3 certification.

## Primary sources consulted, not code copied

- Majercik et al., production DDGI, JCGT2021:
  https://jcgt.org/published/0010/02/01/
- Ouyang et al., ReSTIR GI2021:
  https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
- NVIDIA RTXDI integration/model limitations:
  https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirGI.md
- Mitsuba official material/texture/sensor contracts:
  https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
  https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_textures.html
  https://raw.githubusercontent.com/mitsuba-renderer/mitsuba3/master/src/sensors/perspective.cpp

Algorithms/shader/reference code are original implementation. Local legacy
Vulkan is reference only; no Vulkan backend or build is introduced.
