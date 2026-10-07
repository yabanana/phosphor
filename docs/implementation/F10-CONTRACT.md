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

CSM rasterization uses reverse-Z clear 0/Greater and four array layers. Draw
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
