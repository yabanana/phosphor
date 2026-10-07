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
15=GPUVolumeCounters. Textures0/1=DDGI irradiance/distance atlases,
2=linear HDR,3=depth,4=output HDR; 8/9/10/11=FOUR CSM depth maps.
Fog shadows query the froxel WORLD point, never a screen shadow mask.
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
