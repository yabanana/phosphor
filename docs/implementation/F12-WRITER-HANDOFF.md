# F12 writing-agent handoff — NON VERIFIED

Base: 7997f12713f07895bdf27d7ab27f7947f3da7bd7.
Branch: codex/f12-gi-contracts.
Worktree: /Users/danielsan/.codex/worktrees/f12-gi-contracts/phosphor.

All implementation/test/runner code is WRITTEN, NOT EXECUTED. Only source reads,
primary-source research, source edits, Git commits and git diff --check were
performed. No configure/build/MSL/ctest/GPU/render/reference/benchmark/profiler/
container/download ran here. No roadmap task is ticked and no phase is declared
complete. Main and phase/f9 checkouts/build/assets were not modified.

## Commit boundaries

- a7e33e8 — GPUProbeGridParams160B, state32B, ray32B, traceExtra32B,
  cache64B, reservoir128B ABI. Root reconciles append-only gpu_types.h with
  independent F10/F11 additions.
- dd69003 — F12.5 portable exact same-frame full scene/texture/camera export,
  linear Float32 PFM and independent external Mitsuba scalar_rgb reference runner.
- 1844db6 — F12.1/F12.4 portable probe grid + shader trace/classify/blend/resolve,
  atlases/moments/bounded relocation, full actual textured secondary material,
  emissive direct sampling and reflected-only indirect signal.
- e378bf4 — F12.2 bounded full-key directional radiance cache, serial rotating
  GPU update baseline; miss obtains secondary radiance using DDGI.
- f5757cc — F12.3 diffuse area-measure candidate RIS, temporal reprojection/
  disocclusion, visibility-tested spatial reconnection, indirect irradiance
  shading, explicit experimental basic-reuse bias and fresh/DDGI comparisons.
- The final follow-up adds CPU oracle/negative tests, full implementation
  contract, explicit branch avoiding raw previous atlas reads during reset,
  finite/bounded cache sample cap, exact camera jitter and infinite-far mapping.
  Its ID is available in git log after this file is committed.

Files: src/renderer/{probe_grid,radiance_cache,offline_reference}.{h,cpp},
src/renderer/gi_reservoir.h, appended gpu_types.h, shaders/{ddgi,radiance_cache,
restir_gi}.metal, shaders/{gi_common,gi_cache_common}.h, tests/test_gi.cpp,
tools/f12_reference.py, docs/implementation/F12-CONTRACT.md.

## Integration responsibilities

Root owns core/test/shader CMake wiring, PipelineCache PSOs + own IFT per reload,
GpuMemory resources/retirement/budgets, per-slot AccelerationStructures readonly
API and typed graph Dispatch dependencies, Engine/CLI/report/view history
integration, same-frame WORLD/material/texture export hooks, corpus scenarios and
all acceptance execution. Shader bindings/layouts and exact expected pass order
are in F12-CONTRACT.md. Do not reuse the diagnostic F9 IFT.

F11 dependency: GPUSampledLight80B + GPUEmissiveSurface80B, actual textured
diSampleTexturedLight and DITextureHandle8B. Both F12 RT shader kernels bind
emissiveRecords16; material9/textureTable10/sampled13/extra14 are stable. F9
dependency: root guards rt_alpha_generic definition with
PHOSPHOR_RT_NO_ALPHA_FUNCTION so separate consumers link the single definition.

Geometry guides are direct WORLD point/geometric normal before material resolve,
RGBA32Float valid-w textures. F12 outputs INDIRECT IRRADIANCE. Resolve applies
diffuseAlbedo/pi once and must retain F11 direct/emissive/HDR independent.
Atlas GI excludes first-hit Le; cache contains FULL outgoing radiance and GI
queries it only at SECONDARY hits, subtracting current Le before GI target.

## Required tester execution, not performed

Use the complete NOT EXECUTED command/scene/negative matrix in F12-CONTRACT.md:
portable compile+ctest, Metal/MSL and pinned SDK checks; API/shader validation,
RT PSO reload/IFT lifetime and AS Dispatch barriers; readback against oracles;
Cornell/empty/thin wall/probe inside wall; moving sun/emissive on/off/transforms;
disocclusion/cut/resize/view count/switch/leaks; bounded cache full/collision/
eviction/epoch wrap; q-area and emitter-energy negative controls; independent
linear reference export/render/convergence/region comparison; integrated frame
cost including cache update and denoising; native/effective Apple9 on M5 Max.
The reference renderer must already be installed; this agent installed nothing.

No GPU reference exists yet. Reference export must use completed SAME-FRAME GPU
WORLD instances, current full materials, full raster geometry and exact linear
texture texels. Missing textures deliberately fail, never silently approximate.
Camera jitterPixels is feature displacement +Xright/+Ydown. For Camera's clip
jitter, derive feature displacement = (-projection[2][0]*width/2,
+projection[2][1]*height/2). Mitsuba principal-point offsets negate it divided by
dimensions. UV/jitter/texture plugin ABI and unit mapping require tester proof.

## Assumptions and open questions

- Numeric presets and update budgets are experimental/unmeasured. Serial cache
  writer prioritizes bounded correctness; no performance benefit is claimed.
- DDGI Chebyshev/wrap/grid interpolation and cache quantization have documented
  approximation bias. Thin-wall/material-boundary tests are mandatory.
- ReSTIR GI basic bounded-M reuse is explicitly BIASED for differing source
  visibility support/correlation. No generalized MIS proof is claimed. Fresh
  independent RIS and DDGI remain available; root must report the selected mode.
- F13 owns denoised temporal output acceptance. F12 quality/cost cannot be
  declared closed from noisy-signal tests alone.
- External diffuse transport matches the F12 secondary Lambertian reduction.
  Principled full PBR, finite range quartic windows, spot interpolation and
  double-sided emission currently have explicit external model differences and
  cannot become an accepted reference by using an override.
- Emissive source geometry/UV/alpha must come from the full current scene. Root
  owns the dynamic extraction buffer and revisions; no camera-visible-only list.
- Only M5 Max128GB is available. Effective Apple9 software path is not physical
  M3 certification. No MetalFX/macOS/installer investigation was reopened.
