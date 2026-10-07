# F10 shadow contract — NON VERIFIED

Written from `7997f12713f07895bdf27d7ab27f7947f3da7bd7`. No build, MSL
compilation, test, GPU execution or measurement has been performed by this
writer. Integration and verification belong to the tester.

`GPUShadowCascade` is 112 bytes, `GPUShadowParams` 864, `GPUShadowSurface` 48,
`GPUShadowHistory` 64, `GPUShadowCounters` 32. Scalar layouts are shared in
`gpu_types.h`; `shadow_layout.h` gives exact buffer/texture slots.

The input V-buffer is the F7 meshlet format: biased cluster ID, phase B starts
at `candidateCapacity`, and the triangle is local to its meshlet. Guides fetch
meshlets, meshlet vertices, packed triangles and phase A/B candidates, never
the F9 proxy index stream. The world point comes from reverse-Z depth and the
inverse jittered view projection. The normal is the normalized world-space
triangle cross product, with its hemisphere selected toward the outgoing ray
before the W&B offset. World-position texture output requires RGBA32Float;
the geometric-normal output uses RGBA16Float. Both have validity in alpha.

Each mask is visibility of ONE identified light (1 visible, 0 occluded).
Only that light's direct BRDF contribution is multiplied in resolve. Ambient,
emissive and other lights retain their values. Guides → shadows → resolve is
an acyclic graph; post-resolve normal/motion buffers cannot feed shadows.

All GPU buffers/textures are allocated through GpuMemory; every PSO through
PipelineCache; every access/barrier through the render graph. Every RT consumer
links `rt_alpha_generic` and owns an IFT from ITS resolved compute PSO. Recreate
the IFT after hot reload/PSO publication; never borrow the F9 diagnostic IFT.
Its alpha buffers follow `rt_common.h` slots 0–6, alpha LOD is zero, the trace
mask is `RT_MASK_SHADOW`, and traversal is any-hit. Import the typed slot TLAS
ref and BLAS dependencies at Dispatch, preserving F9 lifetime/synchronization.

CSM rasterization uses reverse-Z clear 0/Greater and four independent Depth32Float maps. Each cascade is its own graph resource/render pass; PCSS binds the four maps at texture slots 2/7/8/9. Draw
ALL live shadow-casting scene slots, independently of camera visibility;
GPU caster flags only reject spheres outside each LIGHT orthographic volume.
Indexed fallback uses the mesh's full index range; mesh draws use all meshlets
of that instance, with a 2D grid. Neither camera visible lists nor Hi-Z lists
are caster lists. Alpha uses the raster material rule at explicit LOD zero.

Solar PCSS is orthographic: penumbra world radius is receiver-to-blocker
WORLD distance × tan(solar angular radius), with no perspective denominator.
CSM split selection uses positive camera view depth. Defaults are experimental
opt-in presets, with CSM available when RT is absent. Nominal RT sampling is
ONE uniform solar-solid-angle ray per valid pixel, then temporal moments and
bounded spatial filtering. Histories are per view AND signal/light; ring
buffer slots do not identify a history. Without previous object transforms,
world-point reprojection rejects moved surfaces instead of claiming motion
reuse. Light/caster/material changes invalidate history through revisions.

Static cache reuse must match light, caster, alpha-material and projection
revisions. Updates are bounded; stale/missing entries use current dynamic
CSM, never their old texels. Graph ordering waits for the entry's last writer
and reader before reuse/overwrite. F10.3 visibility joins F11.1; F13 provides
the later complete lighting denoiser.

## Entry points and bindings

All shader definitions are in `shaders/shadows.metal`. Empty or invalid guide
pixels have `valid=0`; mask sky/background is one. Counters are eight atomic
u32 words in `GPUShadowCounters` order, cleared once before consumers.

| Entry | Buffers | Textures | Dispatch/draw |
|---|---|---|---|
| `shadow_clear_counters` | 15 counters | none | 8 threads |
| `shadow_surface_guides` | 1 params, 2 OUT surface, 3 instances, 5 mesh info, 6 vertices, 7 meshlets, 8 meshlet vertex indices, 9 packed triangles, 10/11 candidate A/B, 15 counters | 0 depth, 1 V-buffer, 5 OUT world point, 6 OUT geometric normal | width × height |
| `shadow_caster_flags` | 1 params, 3 instances, 5 mesh info, 14 OUT flags, 15 counters | none | slotCount threads |
| `shadow_depth_vertex` | 1 params, 3 instances, 6 vertices, 14 caster flags | none | full raster index stream, full scene slot instances |
| `shadow_depth_mesh` | 1 params, 3 instances, 5 mesh info, 6 vertices, 7 meshlets, 8 vertex indices, 9 packed triangles, 14 caster flags | none | 128 threads per mesh group |
| `shadow_depth_fragment` | 16 materials, 17 bindless texture handles | bindless through 17 | alpha at LOD0, no color output |
| `shadow_csm_pcss` | 1 params, 2 surface | 2/7/8/9 depth maps, 4 OUT mask | width × height |
| `shadow_sun_rt` | 0 TLAS, 1 params, 2 surface, 3 instances, 4 own IFT, 15 counters | 4 OUT raw mask | width × height, exactly one ray per valid lit pixel |
| `shadow_temporal` | 1 params, 2 surface, 12 previous history, 13 OUT next history, 15 counters | 3 raw mask, 4 OUT temporal mask | width × height |
| `shadow_filter` | 1 params, 2 surface | 3 temporal mask, 4 OUT filtered mask | width × height |
| `shadow_contact` | 1 params, 2 surface, 15 counters | 0 depth, 3 main mask, 4 OUT final mask | width × height |

Each depth pass sets `cascadeIndex=0..3`, clear zero and depth compare Greater.
For indexed draws, `casterSlot=~0u` selects `instance_id` as the full-scene slot;
`pad` is the draw's mesh ID. The host supplies its mesh's full raster index
range and vertexOffset as baseVertex, and draws `instanceCount=slotCount`.
Other meshes/invalid/noncasting slots produce a clipped degenerate triangle.
Explicit casterSlot mode selects exactly that slot. No index buffer is read
by the vertex shader; the raster encoder consumes it at Vertex.

For the mesh path, `casterSlot=~0u` means grid X is GLOBAL meshlet ID and grid Y
is full-scene slot ID. Launch `(globalMeshletCount,slotCount,1)` mesh groups;
the shader keeps only the meshlet range of that slot's own mesh. Explicit
casterSlot mode uses `meshletFirst + group.y*meshletGridWidth + group.x`, with
the true mesh range checked in the shader. Both modes bound per-group output
by the existing MESHLET_MESH_GROUP=128 ABI. Very large all-pairs scenes may
need batched grids or the indexed fallback after tester validation; this is
a simple correctness baseline, with no performance claim.

Solar samples are uniform over solid angle of the cone subtended by the
physical disk: `pdf=1/[2*pi*(1-cos(angularRadius))]`. The visibility estimator
is the average of binary visibility under this distribution, so the PDF
cancels and the nominal sample count is one. It is not an irradiance estimator.
The zero angular-radius case produces a deterministic direction. The hash
seed includes pixel, frame, view and light ID; F11's defined stochastic source
can replace this stream only with corresponding quality evidence.

The temporal pass reprojects the CURRENT world point using the PREVIOUS
jittered VP, then validates slot/generation, view, light ID/revision, combined
caster/material scene revision, normal and world position. It stores visibility
and second moment with bounded sample count. It clips previous visibility
to the current 3×3 neighborhood of the same slot/incarnation. The 3×3 spatial
filter reads temporal output and does not write its result into temporal
history. This avoids recursively expanding blur. Camera cuts, resize, light
change, alpha revision and scene switch clear HISTORY_VALID through a separate
HistoryRegistry for the shadow signal; histories cannot alias while GPU
readers remain active. Moving receivers without previous object poses reject
reuse through world-position mismatch; full motion reuse is a later extension.

Contact marching uses current jittered VP/depth, a finite world distance and
thickness, at most 64 steps, and pixel-center world reconstruction. Offscreen,
sky or invalid depth leaves the main signal unchanged. Contact composes as
`min(mainVisibility,contactVisibility)` and never darkens an already occluded
main signal a second time.

The normal stored in the guide is the world triangle normal oriented toward
the camera-visible hemisphere. A solar/secondary ray separately chooses its
outgoing hemisphere before W&B; materials and normal maps never control the
offset. GI consumers using the texture must retain the validity test.

## Portable APIs and cache ownership

`makeShadowCascades(ShadowCamera,towardLight,ShadowSettings,span<ShadowBounds>)`
uses the UNJITTERED inverse VP for stable receiver volumes, 10% near overlap
for cascade blending, texel-snapped light-space XY and full-scene caster
bounds for light-space depth. Actual per-pixel guide matrices remain jittered.
`ShadowTechnique` intentionally differs from CLI `ShadowMode`. The host maps
the selected CLI technique and owns all GPU allocations/pipelines/pass access.

`shadowDirtyTiles(cascade,bounds)` returns an 8×8 u64 dirty mask. Union OLD and
NEW caster bounds. Material changes invalidate every affected caster region;
light changes invalidate its entries. The caller expands dirty bounds by the
maximum PCSS filter footprint, in addition to the function's one-texel guard.
Each key is `(view,light,cascade,tile)`; each revision is
`(light,caster,material,projection)`. Projection revision includes the actual
snapped cascade transform and bias/preset changes, not just camera ID.

`ShadowStaticCache` allocates its fixed vector once. `beginFrame(frame,completed,
budget)` resets update quota; `request(key,revision)` returns Cached/Update/
DynamicFallback and `requiredCompletion`. For Update, render fresh static
region depth then `publish(entry,revision,submission)` once its write is
submitted; graph ordering still enforces requiredCompletion. For Cached,
declare its previous writer dependency before reading. `read(entry,submission)`
records the latest reader. Miss/stale/budget starvation uses a current dynamic
CSM region, never stale cached content. Pending entries and in-flight readers
cannot be evicted. Obsolete revision publications and unordered overwrites
throw. This is cache admission/lifetime policy; the host owns static/dynamic
depth composition and region-render commands.

## Source-only review and verification still required

Only `git diff --check` has been used. No compilation or test success is
claimed. The tester must register the portable source/test and metallib source
in CMake, request all PSOs through PipelineCache, harvest descriptors, integrate
the graph, and execute CPU tests, Apple9-compatible paths on M5, API/shader
validation, readbacks, golden images, temporal clips and budget/lifecycle tests.
The shader source includes the NO_ALPHA macro for compatibility; the aggregator
extracts the only `rt_alpha_generic` definition into `rt_intersections.metal`.
Do not compile the original base's unguarded generic body into multiple TUs.

CPU fixtures cover independently projected frustum corners, subtexel camera
motion, off-camera caster retention, conservative border culling, reverse-Z
bias and orthographic penumbra units, solar sample distribution, every history
identity/revision/reset rejection, cache stale/update-budget/in-flight handling,
and changed-region dirty masks. Written controls must fail if the comparison
is inverted, the perspective denominator is introduced, camera culling is
reused, light/caster/material revisions are ignored or a stale writer overtakes
a reader. GPU controls are `corruption` bias/caster/history; cache corruption
is injected by the host's revision checker. Every negative control needs a
tester-established failing readback, not an assumed result from source.

No ROADMAP tasks were ticked. F10.3 joins F11.1 and the complete lighting
denoised exit remains bounded by F13. M5 development acceptance and physical
Apple9/M3 certification are separate; neither has been performed for this
package. The planned indexed layered-output path was replaced by four ordinary
depth passes in agreement with the aggregator, preserving distinct graph refs.
Apple's [layer-selection documentation](https://developer.apple.com/documentation/metal/rendering-to-multiple-texture-slices-in-a-draw-command)
was consulted for the earlier array option; no device behavior was inferred.
