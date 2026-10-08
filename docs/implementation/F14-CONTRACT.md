# F14 physical atmosphere and volumes — NON VERIFIED

Base `4b80e36f7399f264e4c8cc4c4e5f0607d5793128`. SOURCE WRITING ONLY.
No compilation, tests, rendering, profiling or measurement has been executed.
Stop after F14; F14.5 weather remains NOT ACTIVATED pending F15/F16/F19.

ALL WORLD units are metres. Beta/extinction/density coefficients are m^-1.
RGB is linear before exposure. Planet centre defaults to (0,-6360000,0), so
world origin is local ground. LUT keys compare exact parameter tuples, not
age or hash. Transmittance/multiscattering physics key excludes sun motion;
sky-view key includes camera, sun/moon directions/irradiances and physics.

ABI: GPUAtmosphereParams384, GPUFogParams448, GPUFogCell48,
GPUFogIntegrated16, GPUCloudParams272, GPUCloudHistory48,
GPUVolumeCounters32 bytes. Definitions are scalar-only in gpu_types.h.

Atmosphere kernels: atmosphere_transmittance, atmosphere_multiscattering,
atmosphere_sky_view, atmosphere_apply. Buffer0=GPUAtmosphereParams.
Texture0=transmittance LUT,1=multiscattering LUT,2=sky-view LUT,
3=linear scene HDR,4=reverse-Z depth,5=OUTPUT (LUT or HDR as appropriate).
Buffer15=GPUVolumeCounters when the entry declares counters.

Fog kernels: fog_inject (sun CSM fallback), fog_inject_rt (own RT consumer),
fog_temporal, fog_integrate, fog_apply. Buffer0=GPUFogParams,1=current OUT cell
or integrated buffer,2=previous/current cell input,3=readonly DDGI params,
4=DDGI probe states,5=F11 GPUSampledLight buffer,6=alias table,
7=emissive surface records,8=materials,9=bindless texture handles,
10=GPU instances,11=TLAS,12=OWN PSO's IFT,13=GPUShadowParams,
14=GPUAtmosphereParams,15=GPUVolumeCounters. Textures0/1=DDGI irradiance/distance atlases,
2=linear HDR,3=depth,4=output HDR; 8/9/10/11=FOUR CSM depth maps,
12=atmosphere transmittance LUT. maxLocalLights is bounded alias-sample count
per froxel (default1, max16), never a truncation of the entire F11 light list.
Fog shadows query the froxel WORLD point, never a screen shadow mask.
Froxel temporal source uses a linear mean after identity/position/extinction
rejection. It does not clip the importance-weighted mean to current raw
neighborhood extrema: with one contributing light of proposal probability
.1 and history weight .9, that old clip loses more than30% of expected source
energy in one steady update. An independent512-outcome CPU test enumerates
the3x3 light proposals for several probabilities and intensities; the rejected
formula is retained only as a negative control in that test.
RT links the single rt_alpha_generic TU and retains typed TLAS/BLAS graph
dependencies at Dispatch; each RT consumer owns/reloads its own IFT.

Cloud kernels: clouds_march, clouds_temporal, clouds_apply.
Buffer0=GPUCloudParams,1=GPUAtmosphereParams,2=current OUT history,
3=previous history,15=counters. Texture0=scene reverse-Z depth,
1=transmittance LUT,2=multiscattering LUT,3=raw cloud radiance+T,
4=raw guide (cloud centroid distance, opaque distance),5=OUTPUT cloud/mapped
HDR,6=linear scene input HDR,7=OUTPUT raw guide when marching.
Full-rate reference sets cloud width/height equal output extent; low resolution
changes only width/height. History and reconstruction use independent per-view
state, scene-depth validity and neighborhood clamping; clock jumps reset.

CloudParams.maxHistorySamples is u32 at the old final-pad location (ABI size
remains272). Cloud raw radiance+T, composed HDR and sky/reference LUTs use
RGBA32Float; raw guide is RG32Float, since cloud distances exceed half range.
Full-rate reference uses equal cloud/output extent, midpoint marching, no
history reuse and no light-step early termination. Low-rate ratios are1..4.
Below terminationTransmittance, low-rate skips expensive lighting traces but
continues extinction so direct solar-disk transmission is not artificially
retained. Primary marching stops only when transmittance underflows to zero.

Exact per-entry fog bindings: inject uses0/1/3..15 as above; temporal output
cell1, fresh input2, PREVIOUS view-history16 and counters15. Integrate output
GPUFogIntegrated1 and filtered cells2, one thread per XY column loops bounded
gridZ<=128. Apply integrated1/cells2 plus scene2/depth3/output4 textures; prefix
interpolation is bilinear XY and stops within the actual opaque depth slice.
Full-rate cloud march writes raw tex5/guide7 and fresh buffer2. Temporal reads
raw tex3/guide4/previous buffer3 and writes tex5/next buffer2. Apply reads
depth0/cloud3/guide4/HDR6 and writes HDR5. Fresh, temporal and final graph refs
and argument tables are distinct; no encoded table is repurposed for a later
pass in that frame.

All volume histories belong to a VIEW/SIGNAL, not a frame-ring slot. The root
sets epochs from scene/geometry/material, physics/noise/coverage/wind, light
revisions and clock jump. Camera cuts, extent replacements and view mismatches
clear HISTORY_VALID. Frozen-frame old buffers remain GPU-owned through their
last reader; persistent graph imports provide first-access previous-frame
ordering. Clock changes never read a post-resolve screen shadow/normal buffer.

The CSM fog fallback has declared finite coverage. Its far volume must stay
within the configured shadow receiver distance or the root must extend CSM
coverage; outside the maps the shader explicitly uses visible. Root currently
chooses the bounded120m CSM fog preset rather than claiming universal500m
coverage. CSM fallback local/moon lights are unshadowed; RT mode performs
actual froxel-point visibility for sun/moon and each sampled local emitter
with its own PSO-specific IFT. Froxel volume origins do not invent a geometric
normal for W&B; world origin and bounded metre ray segments are used.

F11 alias sampling includes the complete light list: local sample contribution
is incident radiance ×phase / (selectionPMF ×areaPDF), with punctual endpoints
discrete. No receiver-surface cosine is applied to volume scattering. DDGI
ambient uses six diffuse irradiance orientations divided by6*pi, an explicit
isotropic incident-radiance approximation, with probe epoch/active/age guards.

Numerical presets (steps, LUT extents, history weights, phase, solar/stellar
intensity, fog/cloud density) have no measured quality/cost acceptance. The
clock is a deterministic diurnal/lunar model with configurable latitude and
declination, not a precision ephemeris. Scene GPULights receive atmospheric
attenuation at a declared reference position; volumetric/sky shaders receive
extraterrestrial irradiance and evaluate atmospheric transmittance locally.
ExposureEv100 is a target from the same clock; it is applied once by the root
post-processing path, never in the physical linear kernels.

Written CPU references are independent adaptive-Simpson optical-depth/radiance
quadrature and Gauss-Legendre-polar/trapezoid-azimuth multiple-scattering
quadrature. GPU uses midpoint segment integration and Fibonacci angular rays.
Tests also use analytic vertical Rayleigh+ozone depth, vacuum and homogeneous
fog limits, phase normalization, exact version tuples, clock resets, periodic
noise bounds/advection and history identity/depth negatives. NO tests were run.

One portable clock supplies sun, moon, star rotation, cloud/fog time and
exposure target to root BEFORE light/scene preparation. It also names a jump
revision for F10/F11/F12/F13 histories. Atmosphere/volume composition occurs
AFTER F13 scene color/reflections and BEFORE exposure/upscaling/UI.

Primary sources consulted: [Hillaire EGSR 2020](https://doi.org/10.1111/cgf.14050),
[author reference implementation](https://github.com/sebh/UnrealEngineSkyAtmosphere)
and its [MIT license](https://raw.githubusercontent.com/sebh/UnrealEngineSkyAtmosphere/master/LICENSE),
[Nubis author presentation](https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine).
Physics equations are independently implemented; procedural noise/star catalogue
are newly defined here, with no external asset or noise-generator download.
