# F11 sampling package — NON VERIFIED

Written from `7997f12713f07895bdf27d7ab27f7947f3da7bd7` in the managed
`f11-light-sampling` worktree. This is a source handoff, not phase acceptance.
No configuration, C++/MSL compilation, test, renderer, image, timing, GPU
reference generation or noise generation has been executed. Only source reads
and `git diff --check` are allowed for this package. ROADMAP stays unchanged.

## Delivery and integration ownership

F11.1 supplies `reservoir.h`, candidate/temporal/spatial/shade kernels in
`restir_di.metal`, and a small-light independent CPU integration reference.
F11.2 supplies point/spot, parallelogram, disk/ellipse, cylindrical tube side
and emissive triangle sampling, including LOD0 emission and MASK textures.
F11.3 uses a Vose alias table with a full-support mixture; no speculative BVH.
F11.4 supplies an original scalar STBN generator and a separately named white
fallback. F11.5 supplies conservative logarithmic 3D clusters, correct full
iteration on overflow, and unmeasured full/reduced Apple9 presets.

The root integrates CMake, pipeline descriptors, buffers via GpuMemory,
render-graph dependencies, materials/geometry extraction, CLI/report and the
final resolve. These new source files are not reachable from the base build
until that integration is applied. They deliberately do not edit PipelineCache,
the active main checkout, F9 work or the roadmap.

## Integral, proposal and normalization

The domain is the disjoint union of local-light endpoints: one discrete
endpoint for each punctual light, and world-space area for each emitter.
Directional sun illumination is separate F10. Integrand `f(x,y)` is the same
GGX/Smith/Schlick textured PBR direct contribution used by material resolve,
including the receiving cosine and light geometry, before visibility.

For area endpoints, `q(y)=pLight(light)*pArea(endpoint|light)` with `pArea=1/A`.
For a punctual endpoint the conditional discrete density is 1 (the field is
named `pdfArea` for the shared ABI, but is not m^-2 in this case). The exposed
solid-angle Jacobian is `pOmega=pArea*r^2/absCosEmitter`; area geometry is
`absCosEmitter/r^2`. One-sided emitters use the positive facing cosine.
Reservoir weights always use the AREA/discrete proposal, never `pOmega`.

The scalar target is `pHat(x,y)=max(luminance(f(x,y)), targetFloor)` for every
valid light endpoint, with a finite STRICTLY POSITIVE floor. This is an
importance heuristic, not an added radiance term. A black backside, MASK
cutout, receiver hemisphere or out-of-range endpoint therefore remains in
the target support. This full-support choice makes the simple `sum M`
normalization valid in ideal arithmetic, avoiding zero-support energy loss
when receiver BRDFs/normals differ. Degenerate zero-area shapes integrate
black and count as zero-weight proposals. Finite precision, correlations
and convergence still require the independent reference tests.

Candidates stream `w=pHat/q`, including zero final-contribution proposals in
`M`. With selected endpoint `y`, `W=sum(w)/(M*pHat(x,y))`. Final local output
is `f(x,y)*W*visibility(x,y)`. A source reservoir with normalized `W_s` and
effective count `m_s=min(M_s,maxHistoryM)` streams
`w_s=pHat_current(y_s)*W_s*m_s`. Finalization divides by total represented
`M` and the selected CURRENT target. Capping multiplicity does not alter
`W_s`. There is no visibility cached in candidate/reused reservoirs and no
firefly radiance clamp. FP32 weight overflow and malformed PDF/random values
mark `pad[0]` with `DI_ERROR_*`, invalidate final output and MUST fail a debug
readback; they cannot silently produce an accepted biased frame.

Temporal output is history; spatial output is shading input, never feedback
history. Spatial input/output must be different buffers. Every merge
reevaluates the selected light endpoint and target at the destination.

## Light units and shapes

Point/spot `emission` is intensity (W/sr, using existing renderer numeric
units); area `emission` is radiance (W/(m² sr)), world distances are metres.
Punctual range window and smoothstep spot cones match material resolve.
Area range zero is physically unbounded; a positive range enables the same
quartic range window as the punctual path.

| Type | Shape parameters | Area | Endpoint mapping |
| --- | --- | --- | --- |
| Rectangle | position=center; U/V=half extent vectors, shear allowed | `4*length(cross(U,V))` | `center+(2u-1)U+(2v-1)V` |
| Disk/ellipse | position=center; U/V=plane basis; radius>0 | `pi*r²*length(cross(U,V))` | `center+r*sqrt(u)*(cos(2pi v)U+sin(2pi v)V)` |
| Tube | position=center; U=half axis; V=radial reference; radius>0 | `4*pi*r*length(U)` | Uniform cylindrical SIDE, no endcaps |
| Triangle | position=a; U=b-a; V=c-a | `length(cross(U,V))/2` | sqrt barycentrics `(1-sqrt(u),sqrt(u)*(1-v),sqrt(u)*v)` |

Tube endcaps are not silently approximated: author them as disks when needed.
Power alias weights are only heuristics; area emits `pi*A*Le` per side.
`DI_LIGHT_TWO_SIDED` controls physical emitter facing. Mirrored mesh emitter
normals use `DI_LIGHT_MIRRORED`, preserving object outward-facing orientation
without swapping UVs or barycentrics. RGB emission is linear.

Textured triangles have a per-light `GPUEmissiveSurface` (96 B): local triangle
positions/UVs, instance slot/incarnation, material and validity. The scene
must extract FULL geometry, independent of camera culling and RT proxies.
`light_emissive_update` transforms endpoints from the actual GPU scene after
`Scene transforms` into a per-slot light buffer. Material/instance mismatch
turns the entry into a black zero-area source until the host revises it.
Emissive texture and MASK base alpha use explicit LOD0 with raster half texel
arithmetic. `diSampleTexturedLight` and the CPU callback reference share the
UV mapping; the GPU sample takes the current material emission factors.
Changing transforms, emission/alpha material or membership requires a light
revision reset (no area-domain shift/Jacobian reuse across moving emitters).

## Immutable GPU ABI

`gpu_types.h` remains the single scalar-only layout definition. Sizes are:
GPUSampledLight 80 B, GPUEmissiveSurface 96 B, GPUEmissiveUpdateParams 16 B,
GPUAliasEntry 16 B, GPUDIReservoir 64 B, GPUDISurface 96 B, GPUDIParams 96 B,
GPULightCluster 16 B, GPULightClusterParams 112 B.

| Pass | Buffers | Textures |
| --- | --- | --- |
| `light_emissive_update` | update params0, emitter records1, instances2, materials3, source lights4, current lights5 | none |
| `restir_di_candidates` | DI params0, surfaces1, lights2, alias3, output7, STBN ranks8 | none |
| `restir_di_temporal` | params0, surfaces1, lights2, candidates4, history5, previous surfaces6, output7, STBN8 | motion0 |
| `restir_di_spatial` | params0, surfaces1, lights2, temporal4, output7, STBN8 | none |
| `restir_di_shade` | params0, surfaces1, lights2, spatial3 | LOCAL direct output0 |
| `restir_di_shade_rt` | above + instances4, TLAS5, consumer PSO IFT6 | same |
| `light_cluster_build` | cluster params0, lights1, cells2, indices3 | none |
| `light_cluster_shade` | DI params0, surfaces1, lights2, cluster params3, cells4, indices5, STBN6 | LOCAL direct output0 |
| `light_cluster_shade_rt` | above + instances7, TLAS8, consumer PSO IFT9 | same |

ALL ReSTIR kernels and cluster shade kernels additionally bind emitter
records12, GPUMaterial table13, DITextureHandle table14. Analytic/constant
lights never read those records; valid fallback/dummy resources still satisfy
the argument-table contract. The common full emissive helper signature is:

```cpp
DISample diSampleTexturedLight(GPUSampledLight light, uint lightIndex,
    float2 uv, float3 receiver, const device GPUEmissiveSurface* emitters,
    const device GPUMaterial* materials, const device DITextureHandle* textures);
```

`diSampleLight` without texture buffers supports analytic/constant sources.
F12 must call the full helper to preserve textured-emissive reference parity.
`diIncident` returns emitted radiance times emitter geometry, BEFORE the
receiving cosine, BRDF and inverse proposal density.

`GPUDISurface` must come from depth/V-buffer reconstruction BEFORE resolve:
world position, geometric WORLD normal, actual textured material shading
normal/albedo/roughness/metallic, normalized toward-camera view direction,
positive view depth and instance/material revisions. No dependency from the
resolved HDR back to the guide/lighting passes is permitted. Resolve replaces
only the old LOCAL direct-light loop with this output. Sun, ambient, emissive
and later indirect contributions are added separately; never multiply HDR.

The graph declares scene/geometry/material/texture reads, per-view history
reads/writes, and BLAS AND typed TLAS reads at Dispatch for both RT shaders.
This preserves F9 BLAS maintenance and AS→Dispatch ordering. Every RT shade
PSO owns its own IFT, recreated when the resolved PSO changes. Generic alpha
is linked from rt_scene.metal; the root guards its definition using
PHOSPHOR_RT_NO_ALPHA_FUNCTION for other consumers.

## History, rejection and fallbacks

Motion is F8 current→previous in INPUT PIXELS, +Y down, unjittered. Temporal
reprojects `floor(currentPixel+0.5+motion)` and rejects nonfinite/offscreen
coordinates, background, depth difference, geometric-normal disagreement,
normal-plane separation, instance slot/incarnation and material revision.
Camera cut, resize/internal scale, world/view/signal replacement and explicit
reset advance historyEpoch and/or set DI_RESET_HISTORY. Light mutation
advances lightRevision. Light ID/generation and finite positive W/target are
checked on EVERY reuse; age bounds stop indefinite old sample persistence.

Cluster cells use positive near/far, rigid view transform, logarithmic z and
conservative sphere/plane intersections. Shape extent enlarges light range.
Per-cell `count` retains the total light count. `overflow=1` or an outside
frustum receiver means shade EVERY local light, with one uniform endpoint
per area source and inverse area PDF; no truncated-energy fallback.
GPUDIParams.pad1 forces this full iteration for the root's independent
`Local brute force` GPU pass, even without cluster overflow.
The RT cluster variant traces each contributing endpoint, using its own IFT.
The non-RT variant is explicitly unshadowed, not a visibility-correct mode.

Full preset is 8 candidates/4 neighbors, reduced is 2/1; all target/guide
thresholds, memory/ray costs and numeric presets are EXPERIMENTAL UNMEASURED.
They require only effective Apple9 features, but do not certify physical M3
or 30 fps on T0. No additional BVH, reservoir half packing or optimization
is adopted without a measured bottleneck.

## STBN definition and provenance

`stochastic_sampling.cpp` is newly authored source; no SDK code, masks or
licensed texture corpus is imported. The algorithm follows the scalar STBN
same-spatial-slice OR same-pixel-through-time Gaussian energy described by
[Wolfe et al., EGSR 2022](https://research.nvidia.com/publication/2022-07_spatiotemporal-blue-noise-masks).
It includes toroidal distances, binary density relaxation, descending cluster
ranking, ascending void ranking and complementary upper-half ranking.
Generator version1, seed, size, sigma and relaxation limit are explicit data.
Default loading preset is 8×8×16 with 64 independent generated dimensions;
this has NOT been generated or spectrally inspected in this handoff.

Runtime uses pixel/frame/dimension rank indexing and a seed-defined common
Cranley-Patterson rotation per dimension. Candidate i owns dimensions
5i..5i+4 (maximum8); cluster endpoints use40/41; temporal merge uses48;
neighbor i owns49+3i..51+3i (maximum4). The generated masks tile in x/y/frame;
the white fallback is explicitly labelled white. Never label the hash as
blue noise. Exact-rank permutation tests are necessary but not spectral or
visual proof. Relaxation convergence, XY/Z low-frequency spectra, correlation
between dimensions, temporal periodicity and A/B image sequences remain
tester work before STBN is selected as a measured-quality preset.

The reservoir equations follow the basic full-support RIS/merge form from
[Bitterli et al., SIGGRAPH 2020](https://research.nvidia.com/index.php/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct).
This package does not claim support correction, pairwise MIS, visibility reuse
or newer GRIS shifts that it does not implement.

## Verification contract — ALL NON EXECUTED

Root first adds light_sampling.cpp/stochastic_sampling.cpp to phosphor_core,
test_light_sampling.cpp to phosphor_tests, and the new MSL files/dependencies
to the shader build. After its host/graph integration, the TESTER runs:

```sh
cmake -S . -B build/f11-check -G Ninja
cmake --build build/f11-check
./build/f11-check/tests/phosphor_tests --test-case="F11*"
ctest --test-dir build/f11-check --output-on-failure
```

Those commands are instructions, not output or a claim that the base tree
already links these sources. MSL compilation and Metal host syntax checks
are root integration gates. Full GPU pipeline/archive/hot reload tests need
the tester's final CLI/report and execution manifest.

CPU tests cover independent small-light BRDF/visibility reference, quantized
alias distribution and dominant/invalid weights, area moments/Jacobians,
texture/MASK UVs, mirrored/dynamic emissive oracle, exact finite discrete RIS
expectation including dark proposals, missing-M and source-target negatives,
weight errors/overflow, generation/view/epoch/material/depth/normal rejection,
pixel-space reprojection, bounded cluster overflow and STBN permutation/
repeatability/invalid configs. None has run. GPU expectation/reference,
checkerboard disocclusion, behind-wall alpha/area samples, moving/deleted
lights, resize/cut/two views, hotspot overflow, inactive-dimension controls,
NaN/PDF/M corruption, hot reload IFT identity and zero measured-frame GPU
allocations must be verified independently. STBN spectra are not replaced
by successful unit tests.

F11 output is raw stochastic LOCAL direct illumination. The denoised Many
Lights exit depends on F13; this package does not close that boundary or
assert temporal quality/performance. M5 development proof and physical T0
certification remain separate records.

Parent integration amendment (NON VERIFIED): emissive records now include three live raster vertex indices and a geometryValid flag (96 B total); light_emissive_update additionally reads current GPUVertex buffer6, while updateParams.pad names its vertex count. Local positions/UV remain the portable oracle/export metadata. Geometry maintenance therefore changes world emitter sampling in the same frame. DI identity/domain revision is separate from full radiance content revision; GI invalidates on the latter.
