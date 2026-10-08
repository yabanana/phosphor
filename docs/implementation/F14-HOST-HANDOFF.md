# F14 Metal host handoff — NON VERIFIED

Managed worktree `/Users/danielsan/.codex/worktrees/f14-atmosphere-host/phosphor`,
branch `codex/f14-atmosphere-host`, exact base
`be08e3f8d79c9c300ca577266cde2edd42c5ee6d`.

SOURCE WRITING ONLY. No configure/build, MSL compilation, CPU/GPU test,
renderer, benchmark/profiler, simulator/container or asset download was run.
Only reading, writing, commits and `git diff --check`. No acceptance/ROADMAP
checkbox was changed. Stop after F14; F14.5 weather and every F15/OPT remain
outside this task.

## Commit boundaries

- `fc1dd2e`: new AtmospherePasses header/interface.
- `5d8f4b8`: complete LUT/fog/cloud graph host, physical clock/environment,
  per-view histories and output RGBA32Float composition.
- `c5236a9`: clock/readonly readback getters, physical-source validation,
  exception-safe constructor cleanup.
- `3258e5d`: front-air/cloud-centroid composition correction and independent
  analytic CPU fixture/negative old-order test.
- `5552140`: explicit CLOUD_SCENE_HAS_ATMOSPHERE flag, preserving independent
  cloud-only selector semantics without double-adding visible air.
- `e8eeba4`: recorded-write LUT readiness and final cloud composition numerical
  validation; no readiness is claimed before a dispatch is recorded.
- Final handoff commit includes this document and completion overflow guard.

The portable/shader package preceding this host is `c5a8f57..9106ead`;
`fbeb5cd` supplies its final F14-CONTRACT/F14-HANDOFF docs. This host base already
includes that source package through the tester's merged integration. Apply
the host commits in sequence; do not reapply copied root source.

## Exact root-facing API

```cpp
AtmospherePasses(MetalContext&, PipelineCache&, SceneRenderer&,
                DirectLightingPasses*, AccelerationStructures*, GiPasses*,
                ShadowPasses*, const LaunchOptions&);
std::array<GPULight,2> prepareLighting(double clockSeconds, glm::dvec3 camera);
void prepareFrame(const ShadowPasses::Frame&, u64 geometryEpoch, u64 materialEpoch);
rg::TextureRef addToGraph(rg::RenderGraph&, rg::TextureRef linearHdr, rg::TextureRef depth);
void bindFrame(MetalGraphExecutor&);
u64 version() const;
const GiEnvironment& environment() const;
u64 clockEpoch() const;
bool clockReset() const;
float exposureEv100() const;
ReadResources readResources() const;
bool check(u32 completedFrameSlot) const;
```

LaunchOptions dependencies supplied by root: bool atmosphere/fog/clouds/
cloudFullRate; float atmoDayLength(default1200)/atmoStartHour(default12);
u32 timeJumpEveryN(default0). planetCameraHeight is root camera policy and is
not read by this adapter. Existing forceApple9 selects the explicitly reduced
sky/froxel/low-cloud presets, not physical M3 emulation or certification.

Root calls prepareLighting exactly once BEFORE changing scene lights,
SceneRenderer/AS/DI/GI preparation. Replace/update the sun then moon returned
by it; set GiPasses::setEnvironment(environment()) before GI preparation.
clockEpoch/clockReset let root propagate deliberate jumps to F10/F11/F12/F13
history domains without XOR/cancelling revision arithmetic. ExposureEv100 is
a hint from the same clock: actual exposure occurs once in root Post, never
in physical shaders or capture.

Root then calls prepareFrame AFTER every current-frame producer has prepared,
adds the producer graphs (Shadow/Direct/GI/F13), and calls addToGraph after F13
physical scene/reflection color but BEFORE exposure/upscale/UI. Pass the LAST
current receiver-depth version, not a stale prepass handle. Bind after graph
compile. Include version() in the graph key: dimensions/view, LUT-needed mask,
resource growth and RT-vs-CSM graph shape change it. Unchanged physical LUTs
have no update dispatch; camera/sun changes affect sky only. Hot reload forces
physical LUT rebuild and invalidates all per-view sky readiness.

F14 input must be active logical extent in a matching linear RGBA16Float or
RGBA32Float source; display/tone-mapped formats are rejected. Outputs and
LUTs are RGBA32Float. Cloud centroid/opaque guide is RG32Float. Root's physical
Post/upscaler conversion must preserve dynamic range until the single exposure
operation; ordinary temporal upscaler is rejected by the parent contract.

## Resource ownership and histories

New buffers/textures use GpuMemory/RenderTargets accounting. Parameters use
frameUploads. Every PSO uses PipelineCache; the fog RT path owns an RtConsumer
from fog_inject_rt's own resolved PSO and recreates IFT on hot reload. Typed
TLAS and ALL BLAS dependencies are declared at Dispatch through F9's readonly
declareTraceReads; no diagnostic IFT is borrowed. Argument tables are distinct
per pass/frame slot and never repurposed after another draw/dispatch encoded.

Physical transmittance/multiple LUTs are persistent; sky LUT is per view.
Readiness marks recorded writes, not GPU completion; persistent graph imports
order actual writes before consumers and subsequent-frame reuse. Check/readback
must wait for frameEvent>=recordedIndex+1. ReadResources provides borrowed
LUT pointers/refs and exact CPU params for tester graph readbacks only.

Each fog/cloud history is ONE persistent buffer per view and signal. Current,
filtered and integration buffers are per frame slot. A graph snapshot copies
the filtered state into history AFTER its readers; the same persistent virtual
resource receives a write, so previous-frame first-access ordering follows
the actual physical buffer. There is no double-buffer binding swap that hides
cross-resource hazards. Allocation replacement invalidates ALL views, including
dormant views. Frame-slot reuse while its old GPU frame is pending is rejected.

Volume semantic tuple is exact: scene, geometry, material, clock, environment,
local-light/pipeline generations, near/far bits and RT method. No XOR/hash/age
is used as equality. Strict lighting changes reset history; this baseline can
reject temporal reuse continuously under moving sun. Its cost/quality is not
accepted or measured. Static/frozen lighting and wind-advection paths still
use the written history/reconstruction implementation.

## Lighting and finite coverage

GI environment uses fixed ground reference (planet centre +up*(radius+2m))
and bounded CPU hemispherical quadrature. Exact physics/solar/lunar/stellar/
pipeline/jump tuple changes its monotonic epoch; current camera and frame age
do not. This is an explicit isotropic GI environment approximation, not a
directional environment-map integration or precision ephemeris.

Fog reads current DDGI probe states and irradiance/distance atlas through
readonly Gi resources. Without GI, the flag is disabled and safe bound resources
prevent invalid argument tables. Local emission uses the COMPLETE F11 world
light/emitter list, with an owned uniform PMF1/count proposal table; it never
truncates the light list. Root must instantiate the DirectLightingPasses light
producer for fog even when visible shading stays Legacy. If it is missing while
the scene has local lights, preparation fails explicitly rather than dropping
illumination. Producers must declare their graph refs before F14.

With active AS, fog performs actual froxel-point sun/moon/local RT visibility.
Without AS and with Shadow, it uses world CSM queries with declared far120m
(or smaller actual CSM coverage). No screen sun-mask sampling is permitted.
Without both, shadow flags are disabled and the runtime reports unshadowed.
All local/GI parameters remain world metres/m^-1, linear RGB before exposure.

## Cloud/air composition

The root-approved correction treats the opacity-weighted cloud scattering
centroid as a thin layer:

`out = Tcloud*sceneAtmos + TairFront*Lcloud + LairFront*(1-Tcloud)`.

This preserves air scattering IN FRONT and attenuates cloud radiance on its
way to the camera. It is explicitly a CENTROID APPROXIMATION, not a full mixed
air/cloud multiple-scattering integral. clouds_apply now reads atmosphere
params buffer1, transmittance texture1 and multiple texture2. Their readonly
graph accesses are cold consumers of already-built LUTs; no LUT dependency
cycle is introduced. Existing opaque-depth rejection and full-rate reference
remain unchanged. CLOUD_SCENE_HAS_ATMOSPHERE distinguishes visible atmosphere
from optical LUT availability, so cloud-only rendering does not add front air
twice. Composition and all intermediate color outputs remain distinct RGBA32.

test_volumes includes an independent homogeneous front-air/cloud/back-air
analytical case and a negative control for old `Lcloud+Tcloud*sceneAtmos`.
These tests are WRITTEN ONLY. Full-rate clouds use equal input/output extents,
midpoint marching, no history and no lighting termination; low rate uses
ratio2 (or4 in reduced Apple9) and bounded history/depth reconstruction. The
near-ground fog preset composes in front of cloud+scene afterward. Arbitrary
overlapping-media exact integration and cloud shadows cast on terrain are not
claimed by this baseline.

## Tester work still NOT EXECUTED

Register the new cpp/header/helper, shader changes and test fixture; compile
C++/MSL and appropriate Linux metal_syntax_check, CPU tests and PSO harvesting.
Exercise the exact options/hooks above, then inspect API/shader validation and
all source readbacks: persistent LUT key changes/no-change/hot reload; all view
histories through growth/reset/jump/switch; current local-light/alpha/emissive
and DDGI inputs; RT own-IFT reload; CSM finite coverage; cloud full-vs-low,
opaque/depth borders and front-air analytic transport. Preserve linear PFMs
before exposure with LinearCapture and freeze thresholds before comparisons.

Read counters only after completion; nonfinite/invalidUnits/invalidHistory
make check return false. Startup unrecorded slots have no work/check evidence.
Root still owns independent numerical LUT/fog/cloud reference comparison,
negative-control runner failure proof, allocation/leak/lifecycle validation,
quality clips, budgets and hardware/report status. No such result exists from
this writer. F14.5 remains NOT ACTIVATED and no F15/OPT work is present.
