> Stato corrente: il seguito conserva la consegna iniziale dell'adapter. Gateway e fixture native sono ora disponibili, ma F13.3 resta parziale: la produzione scene-linear/HDR usa custom Float32 per radiometria non qualificata. Vedi [F13-RADIOMETRY-POLICY.md](F13-RADIOMETRY-POLICY.md); le vecchie indicazioni di gateway non integrato non descrivono lo stato corrente.

# F13.3 MetalFX denoised adapter — WRITTEN / NON VERIFIED

Base4b80e36f7399f264e4c8cc4c4e5f0607d5793128. New managed worktree
/Users/danielsan/.codex/worktrees/f13-metalfx-adapter/phosphor. No configure,
build, compiler/MSL, test, renderer, profiler, benchmark, simulator, dependency
installation or OS operation was run. ROADMAP is unchanged.

## Actual factory boundary

Read the configured local metal-cpp SDK:

- build/_deps/metal_cpp-src/MetalFX/MTL4FXTemporalDenoisedScaler.hpp
- build/_deps/metal_cpp-src/MetalFX/MTLFXTemporalDenoisedScaler.hpp
- Xcode macOS SDK MetalFX.framework/Headers/MTLFXTemporalDenoisedScaler.h.

The denoised factory is a different typed API:
MTLFX::TemporalDenoisedScalerDescriptor::newTemporalDenoisedScaler(device,
const MTL4::Compiler*) -> MTL4FX::TemporalDenoisedScaler*.
Its encode call accepts MTL4::CommandBuffer*. The existing PipelineCache only
exports requestTemporalScaler(TemporalScalerDescriptor*) for TemporalScaler and
keeps compiler_ private. That API cannot create this denoised type.

PipelineCache has NOT been edited. Its F9 archival/lifetime/readback corrections
remain aggregator/tester owned. docs/patches/f13-denoised-factory.patch is an
unapplied, concrete gateway proposal using its EXISTING compiler/utility queue,
guarded by PHOSPHOR_METALFX_DENOISED_FACTORY and SDK availability. No compiler
or compute/render pipeline is created outside PipelineCache.

The adapter accepts injected Factory.request/retire callbacks. The request
copies its descriptor before returning a queue future; retirement schedules
plain release on utility workers AFTER the adapter's GPU-completion deferral.
Without both callbacks the requested native path reports MissingFactory and
returns the caller's real custom-denoised composite. No ordinary scaler cast,
fake native pass or lifetime acceptance is implied.

After the tester reconciles the gateway with F9, root can wire:

    MetalfxDenoise::Factory gateway;
    #ifdef PHOSPHOR_METALFX_DENOISED_GATEWAY_AVAILABLE
    gateway.request=[&pipelines](auto* d){return pipelines.requestTemporalDenoisedScaler(d);};
    gateway.retire=[&pipelines](auto s){pipelines.retireTemporalDenoisedScaler(std::move(s));};
    #endif

The callback captures must outlive the adapter and deferred retirements.
Default gate stays absent/custom. This is an explicit user ownership boundary;
there is no additional routine owner approval request.

## Channel contract

Apple describes world normals, diffuse/specular auxiliaries (including Fresnel
for specular), linear roughness and primary-to-secondary hit length in the
denoised workflow. The current SDK separately exposes motion scale, reversed
depth, exposure, optional reactive/skip-denoise masks and per-channel usages.
Sources: [WWDC25](https://developer.apple.com/videos/play/wwdc2025/211/),
[denoised SDK](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase).

The portable contract rejects view/tangent or unorm normals, isolated-signal
color, squared/gamma roughness, reversed motion direction, UV/NDC motion and
jitter included in motion. Its descriptor formats are CANDIDATE choices, not
a promised format-support matrix: factory success and exact scaler getters/
texture usage checks are mandatory on the actual device.

| Channel | Packed format | Data |
|---|---|---|
| Color | RGBA16Float | complete noisy linear radiance * declared preExposure, composed ONCE |
| Depth | Depth32Float | reverse-Z, far0; exact active rectangle copied by Blit |
| Motion | RG32Float | current to previous INPUT pixels, +Y down, unjittered |
| Diffuse albedo | RGBA16Float | material diffuse factor; dark for metallic surfaces |
| Specular albedo | RGBA16Float | deterministic primary Fresnel/specular approximation |
| Normal | RGBA16Float | signed unit WORLD shading normal xyz, unused w0 |
| Roughness | R16Float | original perceptual roughness scalar in linear storage |
| Specular distance | R32Float | optional WORLD primary-to-secondary ray length |
| Reactive | R8Unorm | optional1 suppresses temporal history |
| Strength | R8Unorm | optional1 skips denoising; differs from reactive |
| Exposure | R16Float1x1 | fixed1, keeping the before-Post exposure HDR signal |
| SDK output | RGBA16Float/private | denoised, reconstructed; output scale requires scalar fixture |
| Returned output | RGBA32Float/private | physical linear radiance restored by1/preExposure |

No gamma/tonemap, normal0.5 remapping, roughness square, albedo/radiance mixing
or input-pixel motion rescaling occurs. Motion scales are1/1. F8's current
validated raster-jitter mapping is preserved. Frame supplies current WORLD-
to-view and UNJITTERED reverse-Z view-to-clip matrices plus separate jitter.
Matrix/jitter equivalence still needs an independent moving-pixel/camera fixture;
the SDK's names alone do not certify this mapping.

The denoised base does NOT expose standard TemporalScaler's inputContentWidth/
Height setters. Packed textures therefore have EXACT active input dimensions.
No backing-sized DRS texture is assigned as if it had a smaller active region.
The adapter does not call the guessed denoised descriptor dynamic-content
setters present in metal-cpp but absent from the configured Objective-C header.

Colour outside finite nonnegative half range AFTER declared preExposure scaling is reported, not accepted. Pack
repairs protect finite SDK inputs but any repair fails the readback predicate.
Exposure/preExposure must match actual source multiplication; default1 consumes
unexposed raw radiance. Optional hit distance defaults DISABLED until real
RT/SSR/miss conventions are checked; unavailable/unknown hit distances must not
be fabricated from camera depth.

Wide-HDR normalization is explicit: pack multiplies physical noisy color by
Frame.preExposure BEFORE half-range validation, unit exposure texture stays1,
the SDK receives the same preExposure scalar, and a separate PipelineCache
compute pass copies SDK half output into physical RGBA32Float by1/preExposure.
The public [preExposure documentation](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure)
specifies input division by the declared multiplier; it does not expressly
guarantee output scale. Options.sdkOutputScale therefore defaults Unverified.
Nonunit scaling under that default selects custom fallback with an explicit
output-unit diagnostic. PreExposed is an experimental declaration requiring
the tester's independent constant/impulse SDK roundtrip before adoption.
F14's1/64 normalization is a declared experiment, not an observed result.

## API and graph integration

MetalfxDenoise(c,PipelineCache,Options,Factory={}) accepts:

- Frame: slot/view/global index, active/output extents, signalEpoch, current
  matrices, jitter pixels, preExposure, cut/reset, portable channel semantics.
- Inputs: raw composite color/depth/motion/diffuse/specular/WORLD-normal/
  SEPARATE roughness, optional distance/reactive/strength, actual custom fallback.
- prepareFrame polls the typed queue future and records requested/effective
  diagnostics. version() must participate in root's graph cache key.
- addToGraph returns one SDK output OR the caller's custom output; bindFrame
  binds current view/slot resources each frame.
- Stats report requested/sdk/device/factory states, real factory requests,
  encoded native frames, fallback frames, resets and discarded obsolete requests.
- readPackChecks inspects a completed slot; drainPackChecks accounts completed
  current AND retired resize records. Drain after waitIdle at final verification.

Root composes DI/GI/reflection/sun/emissive once upstream. SDK combined color
and custom independent signal filters are alternative paths, not additive
outputs. AO affects its defined residual ambient signal upstream; it is not
fed as a separate SDK color or multiplied into total HDR here.

When SDK output is selected it is already output-sized/reconstructed and
dejittered. Downstream Post must use output dimensions as its input and skip
the standard F8 temporal upscaler. Otherwise it would upscale twice or crop
the reconstructed image as a DRS backing rectangle. Tone mapping/exposure/
presentation remain root-owned downstream steps.

All engine-owned targets/neutral guides/history anchors/readback buffers are
created through GpuMemory, in RenderTargets or Other categories. Every guide
usage from the REAL scaler is added to its texture descriptor and checked;
output is private. Framework-internal allocations are opaque and must be
measured separately, not invented as an engine memory accounting value.

Pass order and declared accesses:

1. denoise_pack_clear Compute writes eight counters.
2. active depth crop Blit reads source depth, writes exact packedDepth32.
3. guide pack Compute reads all enabled source guides and neutral optional
   resource, RMW counters, writes all packed scalar/vector/color/exposure guides.
4. Temporal denoised HDR External reads packed guides, RMW per-view opaque
   history anchor and writes output. PassContext supplies borrowed MTL4 command
   buffer/fence; the framework waits/updates that fence. No unmanaged encoder
   boundary or async queue shortcut is introduced.
5. denoise_restore_radiance Compute reads SDK output and writes physical
   RGBA32Float; root Post consumes this returned signal without another
   pre-exposure undo or standard temporal pass.

The token is a graph dependency anchor for OPAQUE framework state, not a fake
GPU data representation of its history. StageExternal provides the conservative
boundary including ML/compute/blit/raster/AS. Source depth must be actual valid
raster reverse-Z depth; an invalid float depth is detected by the pack check and
must fail the request's verification.

Packing kernel ABI: params0/counters1, source texture color0/normal1/roughness2/
diffuse3/specular4/motion5/distance6/reactive7/strength16/depth19; destination
color8/normal9/roughness10/diffuse11/specular12/motion13/distance14/reactive15/
strength17/exposure18. Inactive optional sources bind a defined neutral texture;
their disabled branches do not read active image coordinates from that1x1 data.
Counter fields: pixels/color/normal/albedo/roughness/motion/distance/mask errors.
Pixel counts reduce per SIMD group; error counters are sparse object errors.
Restore ABI is GPUMetalfxRestoreParams buffer0, SDK output texture0, physical
RGBA32Float destination1. It has its own per-view/per-slot argument table.

## View, resize, hot reload and lifetime

Each view owns one scaler and separate per-frame-slot packed/output targets.
Descriptor identity includes input AND output extents, optional channel flags
and PipelineCache generation. Extent changes settle for four observed frames
(configurable) before creating a replacement. A late null/error/result from an
obsolete descriptor is discarded, never poisoning the current descriptor.
Hot reload invalidates instance/history and reconstructs through the gateway.
Camera/content epoch/cut/reset invalidate history without assuming other views
share the same camera or temporal signal.

GPU-used scalers retire only after all in-flight frames complete, then standard
release happens on workers. Never-encoded results can retire directly on workers.
Pending futures are drained before destruction while PipelineCache still lives.
GpuMemory handles target retirement; unread per-slot counters survive resize in
a separate retirement queue until their GPU completion, then are copied and
released. Final verification must drain late records before destroying the adapter.

DENOISED lifetime is UNVERIFIED. The F8 TemporalScaler self-cycle workaround is
NOT applied, not generalized, and no private-reference inspection or OS change
is introduced. Standard ownership may reveal framework-specific behavior; the
tester must establish creation/resize/view-count/reload/destruction/capture/leak
behavior for this distinct type before any lifetime acceptance.

## Tester work, NOT EXECUTED

Root adds new adapter source, portable test/layout/contract, denoise_pack.metal,
shader dependencies and pipeline harvest. Root wires actual reflection and
color composer, Post selection, CLI/report and the optional gateway feature gate.

Required tests after reconciliation:

- Portable unit tests for wrong spaces/units/direction/roughness/color,
  half overflow, normal validity, obsolete extent/generation futures.
- macOS host/MSL build and pinned local SDK syntax coverage, including
  PHOSPHOR_DISABLE_METALFX_DENOISED and missing gateway/unsupported SDK/device.
- Ensure missing gateway really uses custom denoiser and reports its reason.
- With real gateway: descriptor rejection, scale boundaries, required usage
  flags, exact DRS crop, independent channel impulse/constant/ramp readbacks.
- Pixel motion/jitter/matrices, camera rotation/cut, reflection roughness,
  emissive/sun/light movement, disocclusion, dynamic objects,4 views, resize,
  repeated input/output extent change, hot reload and scene switching.
- Wide-HDR constant/impulse inputs at preExposure1 and1/64, actual SDK output
  scale, physical post-restore identity and overflow rejection AFTER scaling.
- Bad normal encoding, motion sign/scale, roughness squaring, isolated-signal
  color, corrupted guide, missing valid specular distance MUST fail controls.
- API/shader validation, final readback drain, leak/lifetime/capture paths and
  no steady-state engine allocation claim without actual measurements.
- Compare custom versus denoised SDK with identical raw signal, scene/camera/
  seed/units/resolution and reference; include pack/crop/ML/Post memory and time,
  warmup/steady state and three replicas. Only M5 Max128GB is physically available.

No F13 acceptance or roadmap tick follows from this source package.
